"""Shared trainer: same optimizer, scheduler, loss, early stopping for all U-Nets."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.optim as optim

from .config import CONFIG
from .losses import CombinedLoss
from .metrics import calculate_metrics, confusion_counts, metrics_from_counts

logger = logging.getLogger(__name__)


def split_encoder_decoder_params(
    model: torch.nn.Module,
    encoder_prefixes: Sequence[str],
) -> Tuple[list, list]:
    """Assign each unique Parameter to encoder (low LR) or decoder (higher LR)."""
    prefixes = tuple(encoder_prefixes)
    encoder, decoder = [], []
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        root = name.split('.', 1)[0]
        if root in prefixes or name in prefixes:
            encoder.append(param)
        else:
            decoder.append(param)
    return encoder, decoder


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        encoder_prefixes: Sequence[str],
        checkpoint_path: str,
        epochs: Optional[int] = None,
        log_every: int = 10,
        early_stopping: Optional[bool] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs if epochs is not None else CONFIG['training']['epochs']
        self.log_every = log_every
        # Final retrain: val_loader=None → no ES, no test peek. CV folds keep default ES.
        if early_stopping is None:
            early_stopping = val_loader is not None
        self.early_stopping = bool(early_stopping) and val_loader is not None
        self.criterion = CombinedLoss()

        enc_params, dec_params = split_encoder_decoder_params(model, encoder_prefixes)
        groups = []
        if enc_params:
            groups.append({'params': enc_params, 'lr': CONFIG['training']['lr_backbone']})
        if dec_params:
            groups.append({'params': dec_params, 'lr': CONFIG['training']['lr_decoder']})
        if not groups:
            groups = [{'params': list(model.parameters()), 'lr': CONFIG['training']['lr_decoder']}]

        self.optimizer = optim.AdamW(groups, weight_decay=CONFIG['training']['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=CONFIG['training']['scheduler_factor'],
            patience=CONFIG['training']['scheduler_patience'],
        )
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [],
            'train_precision': [], 'val_precision': [],
            'lr': [],
        }
        self.best_iou = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0}
        n = max(len(self.train_loader), 1)
        for i, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                m = calculate_metrics(outputs, masks)
                total_loss += loss.item()
                total_metrics['iou'] += m['iou']
                total_metrics['dice'] += m['dice']
                total_metrics['precision'] += m['precision']
            if i % self.log_every == 0:
                logger.info(
                    'Batch %d/%d, Loss: %.4f, IoU: %.4f',
                    i, len(self.train_loader), loss.item(), m['iou'],
                )
        return (
            total_loss / n,
            total_metrics['iou'] / n,
            total_metrics['dice'] / n,
            total_metrics['precision'] / n,
        )

    def validate(self) -> Dict[str, float]:
        """Micro-averaged val metrics (sum TP/FP/FN over the whole held-out fold)."""
        if self.val_loader is None:
            raise RuntimeError('validate() called with val_loader=None')
        self.model.eval()
        total_loss = 0.0
        tp = fp = fn = tn = 0.0
        n_batches = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                ctp, cfp, cfn, ctn = confusion_counts(outputs, masks)
                tp += ctp
                fp += cfp
                fn += cfn
                tn += ctn
                n_batches += 1
        metrics = metrics_from_counts(tp, fp, fn, tn)
        metrics['loss'] = total_loss / max(n_batches, 1)
        return metrics

    def train(self) -> Dict:
        has_val = self.val_loader is not None
        if has_val:
            logger.info('Training for up to %d epochs (val early stopping)', self.epochs)
        else:
            # Fixed-epoch final retrain on 100% of train-pool. No val loader, no ES,
            # test set is not touched here (evaluated once after this loop).
            logger.info(
                'Training for a fixed %d epochs (no val, no early stopping, no test peek)',
                self.epochs,
            )
        for epoch in range(self.epochs):
            start = time.time()
            tr_loss, tr_iou, tr_dice, tr_prec = self.train_epoch()
            if has_val:
                val = self.validate()
                self.scheduler.step(val['iou'])
                current_lr = self.optimizer.param_groups[0]['lr']
                val_loss, val_iou, val_dice, val_prec = (
                    val['loss'], val['iou'], val['dice'], val['precision'],
                )
                if val['iou'] > self.best_iou:
                    self.best_iou = val['iou']
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    logger.info(
                        'New best model saved (IoU=%.4f) → %s',
                        self.best_iou, self.checkpoint_path,
                    )
                else:
                    self.patience_counter += 1
                logger.info(
                    'Epoch %d/%d (%.1fs) | Train loss=%.4f IoU=%.4f | '
                    'Val loss=%.4f IoU=%.4f Dice=%.4f Prec=%.4f | LR=%.2e',
                    epoch + 1, self.epochs, time.time() - start,
                    tr_loss, tr_iou, val_loss, val_iou, val_dice, val_prec, current_lr,
                )
            else:
                # Step LR from train IoU so ReduceLROnPlateau still works without val/test.
                self.scheduler.step(tr_iou)
                current_lr = self.optimizer.param_groups[0]['lr']
                val_loss = val_iou = val_dice = val_prec = float('nan')
                self.best_iou = tr_iou
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.info(
                    'Epoch %d/%d (%.1fs) | Train loss=%.4f IoU=%.4f Dice=%.4f Prec=%.4f | LR=%.2e',
                    epoch + 1, self.epochs, time.time() - start,
                    tr_loss, tr_iou, tr_dice, tr_prec, current_lr,
                )
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(tr_iou)
            self.history['val_iou'].append(val_iou)
            self.history['train_dice'].append(tr_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_precision'].append(tr_prec)
            self.history['val_precision'].append(val_prec)
            self.history['lr'].append(current_lr)
            if (
                has_val
                and self.early_stopping
                and self.patience_counter >= CONFIG['training']['early_stopping_patience']
            ):
                logger.info('Early stopping. Best IoU: %.4f', self.best_iou)
                break
        return self.history
