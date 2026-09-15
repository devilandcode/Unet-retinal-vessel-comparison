"""
Patient-level 5-fold CV runner used by U-Net, U-Net++ and Attention U-Net.

5-fold CV runs on the train-pool only. After all folds, each architecture is
retrained once on 100% of the train-pool for median(epochs_trained) epochs
(no val early stopping) and evaluated once on the held-out test set.

Usage (from project root):
    python u-net.py
    python u-net.py --max-epochs 1 --fold 0          # smoke one fold
    python attention-u-net-stare.py --splits-only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import statistics
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader

from .config import CONFIG, N_FOLDS, PROJECT_ROOT, SPLIT_SEED
from .cv_results import build_cv_document, cv_results_path, save_cv_results
from .data_io import load_samples
from .dataset import RetinaDataset, get_transforms
from .metrics import measure_inference_speed
from .seed import fold_train_seed, set_seed
from .splits import (
    default_folds_path,
    load_or_create_folds,
    samples_for_fold,
    samples_for_test,
    train_pool_samples,
)
from .trainer import Trainer
from .viz import plot_training_history, save_prediction_collage

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
    )
    return logging.getLogger('cv')


def parse_cv_args(dataset_name: str, architecture_name: str, argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f'{architecture_name} on {dataset_name} — patient-level 5-fold CV + held-out test',
    )
    p.add_argument('--dataset', default=dataset_name)
    p.add_argument('--folds-file', default=None, help='Reuse a shared folds JSON')
    p.add_argument('--fold', type=int, default=None, help='Run a single fold (0-4)')
    p.add_argument('--max-epochs', type=int, default=None, help='Override CONFIG epochs')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--splits-only', action='store_true', help='Write folds JSON and exit')
    p.add_argument('--num-workers', type=int, default=None)
    return p.parse_args(argv)


def _make_loader(dataset, batch_size, shuffle, device, num_workers, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init(worker_id):
        set_seed(seed + worker_id)

    pin = getattr(device, 'type', None) == 'cuda'
    workers = num_workers if num_workers is not None else (2 if pin else 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        generator=generator if shuffle else None,
        worker_init_fn=_worker_init if workers > 0 else None,
    )


def _median_cv_epochs(fold_records: Sequence[dict]) -> int:
    epochs = [int(r['epochs_trained']) for r in fold_records if r.get('epochs_trained')]
    if not epochs:
        return CONFIG['training']['epochs']
    return max(1, int(statistics.median(epochs)))


def _final_retrain_and_test(
    architecture_name: str,
    dataset_name: str,
    model_factory: Callable,
    encoder_prefixes: Sequence[str],
    samples,
    payload: dict,
    fold_records: Sequence[dict],
    device,
    batch_size: int,
    num_workers,
    results_dir: Path,
):
    """Retrain once on 100% of train-pool, then evaluate once on held-out test.

    Epoch budget = median(epochs_trained) across the 5 CV folds. Trainer gets
    val_loader=None so there is no val early stopping and the test set is never
    used until after training finishes.
    """
    final_epochs = _median_cv_epochs(fold_records)
    train_s = train_pool_samples(samples, payload)
    test_s = samples_for_test(samples, payload)
    if not test_s:
        raise ValueError('Held-out test is empty; cannot run final evaluation')
    final_seed = fold_train_seed(SPLIT_SEED, N_FOLDS)
    set_seed(final_seed)
    logger.info(
        '--- Final retrain on ALL train-pool (%d images / %d patients) '
        'for %d epochs (median CV epochs_trained); then one test eval (%d images) ---',
        len(train_s), len(payload.get('train_pool_patients') or []),
        final_epochs, len(test_s),
    )
    train_ds = RetinaDataset(train_s, transform=get_transforms('train'), enhance_vessels=True)
    train_loader = _make_loader(train_ds, batch_size, True, device, num_workers, final_seed)
    model = model_factory()
    ckpt = PROJECT_ROOT / 'checkpoints' / dataset_name / architecture_name / 'final.pth'
    trainer = Trainer(
        model, train_loader, None, device,
        encoder_prefixes=encoder_prefixes,
        checkpoint_path=str(ckpt),
        epochs=final_epochs,
        log_every=5 if dataset_name in {'stare', 'chase', 'hrf'} else 10,
        early_stopping=False,
    )
    trainer.train()
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    test_ds = RetinaDataset(test_s, transform=get_transforms('test'), enhance_vessels=True)
    test_loader = _make_loader(test_ds, batch_size, False, device, num_workers, final_seed)
    trainer.val_loader = test_loader
    raw = trainer.validate()
    test_metrics = {
        'iou': raw['iou'],
        'dice': raw['dice'],
        'precision': raw['precision'],
        'recall': raw['recall'],
        'f1': raw['f1'],
        'loss': raw['loss'],
    }
    test_collage = results_dir / f'{architecture_name}-cv-test-predictions.png'
    save_prediction_collage(
        model, test_loader, device,
        save_path=str(test_collage),
        n_samples=min(4, len(test_ds)),
        title=f'{architecture_name} {dataset_name} held-out test',
    )
    logger.info('Saved test collage: %s', test_collage)
    logger.info(
        'TEST IoU=%.4f Dice=%.4f Prec=%.4f Rec=%.4f F1=%.4f',
        test_metrics['iou'], test_metrics['dice'], test_metrics['precision'],
        test_metrics['recall'], test_metrics['f1'],
    )
    final_train = {
        'epochs': final_epochs,
        'n_train': len(train_s),
        'n_test': len(test_s),
        'train_seed': final_seed,
        'checkpoint': str(ckpt),
        'note': (
            'Trained from scratch on 100% of train-pool for median(epochs_trained) '
            'from the 5 CV folds. val_loader=None: no validation split, no early '
            'stopping, ReduceLROnPlateau stepped on train IoU. Held-out test was '
            'evaluated once after training.'
        ),
    }
    return test_metrics, final_train


def run_cv(
    architecture_name: str,
    dataset_name: str,
    model_factory: Callable,
    encoder_prefixes: Sequence[str],
    argv=None,
):
    args = parse_cv_args(dataset_name, architecture_name, argv)
    setup_logging()
    logger.info('=' * 80)
    logger.info('%s | %s | patient-level %d-fold CV + held-out test', architecture_name, args.dataset, N_FOLDS)
    logger.info('=' * 80)

    samples = load_samples(args.dataset)
    folds_path = Path(args.folds_file) if args.folds_file else default_folds_path(args.dataset)
    payload, folds_path = load_or_create_folds(
        args.dataset, samples, folds_path=folds_path, n_folds=N_FOLDS, seed=SPLIT_SEED,
    )
    logger.info(
        'Folds file: %s (%d patients, %d images, test=%d images / %d patients)',
        folds_path, payload['n_patients'], payload['n_images'],
        payload.get('n_test_images', 0), len(payload.get('test_patients') or []),
    )
    if args.splits_only:
        logger.info('Splits written. Exiting.')
        return payload

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)
    batch_size = args.batch_size or CONFIG['training']['batch_size']
    epochs = args.max_epochs if args.max_epochs is not None else CONFIG['training']['epochs']

    dummy = model_factory()
    total_params = sum(p.numel() for p in dummy.parameters())
    trainable_params = sum(p.numel() for p in dummy.parameters() if p.requires_grad)
    del dummy

    fold_records = []
    fold_indices = [args.fold] if args.fold is not None else list(range(len(payload['folds'])))
    results_dir = PROJECT_ROOT / 'results' / args.dataset
    summary_collage = results_dir / f'{architecture_name}-cv-predictions.png'
    summary_from_fold0 = None

    for fold_i in fold_indices:
        fold = payload['folds'][fold_i]
        train_seed = fold_train_seed(SPLIT_SEED, fold_i)
        set_seed(train_seed)
        train_s, val_s = samples_for_fold(samples, fold)
        logger.info(
            '--- Fold %d  train=%d images / %d patients  val=%d / %d  seed=%d ---',
            fold_i, len(train_s), len(fold['train_patients']),
            len(val_s), len(fold['val_patients']), train_seed,
        )
        train_ds = RetinaDataset(train_s, transform=get_transforms('train'), enhance_vessels=True)
        val_ds = RetinaDataset(val_s, transform=get_transforms('val'), enhance_vessels=True)
        train_loader = _make_loader(train_ds, batch_size, True, device, args.num_workers, train_seed)
        val_loader = _make_loader(val_ds, batch_size, False, device, args.num_workers, train_seed)

        model = model_factory()
        ckpt = PROJECT_ROOT / 'checkpoints' / args.dataset / architecture_name / f'fold_{fold_i}.pth'
        trainer = Trainer(
            model, train_loader, val_loader, device,
            encoder_prefixes=encoder_prefixes,
            checkpoint_path=str(ckpt),
            epochs=epochs,
            log_every=5 if args.dataset in {'stare', 'chase', 'hrf'} else 10,
        )
        history = trainer.train()
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        val_metrics = trainer.validate()
        speed = measure_inference_speed(model, val_loader, device, num_samples=min(20, len(val_ds)))
        record = {
            'fold': fold_i,
            'train_seed': train_seed,
            'n_train': len(train_s),
            'n_val': len(val_s),
            'train_patients': fold['train_patients'],
            'val_patients': fold['val_patients'],
            'epochs_trained': len(history['train_loss']),
            'metrics': {
                'iou': val_metrics['iou'],
                'dice': val_metrics['dice'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1'],
                'loss': val_metrics['loss'],
                'best_iou': trainer.best_iou,
                'inference_fps': speed['fps'],
                'inference_time_ms': speed['avg_time_per_image'] * 1000.0,
            },
        }
        fold_records.append(record)
        plot_training_history(
            history,
            save_path=str(results_dir / f'{architecture_name}-cv-fold{fold_i}-history.png'),
            title=f'{architecture_name} {args.dataset} fold {fold_i}',
        )
        fold_collage = results_dir / f'{architecture_name}-cv-fold{fold_i}-predictions.png'
        save_prediction_collage(
            model, val_loader, device,
            save_path=str(fold_collage),
            n_samples=4,
        )
        logger.info('Saved predictions collage: %s', fold_collage)
        if fold_i == 0:
            shutil.copy2(fold_collage, summary_collage)
            summary_from_fold0 = fold_collage
            logger.info('Saved summary collage: %s', summary_collage)
        logger.info(
            'Fold %d done: IoU=%.4f Dice=%.4f Prec=%.4f',
            fold_i, val_metrics['iou'], val_metrics['dice'], val_metrics['precision'],
        )

    if summary_from_fold0 is None and fold_records:
        last_i = fold_records[-1]['fold']
        last_collage = results_dir / f'{architecture_name}-cv-fold{last_i}-predictions.png'
        if last_collage.exists():
            shutil.copy2(last_collage, summary_collage)
            logger.info('Saved summary collage: %s', summary_collage)

    out_path = cv_results_path(args.dataset, architecture_name)
    if args.fold is not None and out_path.exists():
        import json
        with out_path.open('r', encoding='utf-8') as f:
            existing = json.load(f)
        by_fold = {rec['fold']: rec for rec in existing.get('per_fold', [])}
        for rec in fold_records:
            by_fold[rec['fold']] = rec
        fold_records = [by_fold[i] for i in sorted(by_fold)]

    test_metrics = None
    final_train = None
    n_folds_done = len(fold_records)
    run_final = args.fold is None and n_folds_done == len(payload['folds'])
    if run_final:
        test_metrics, final_train = _final_retrain_and_test(
            architecture_name=architecture_name,
            dataset_name=args.dataset,
            model_factory=model_factory,
            encoder_prefixes=encoder_prefixes,
            samples=samples,
            payload=payload,
            fold_records=fold_records,
            device=device,
            batch_size=batch_size,
            num_workers=args.num_workers,
            results_dir=results_dir,
        )

    extra_info = {
        'device': str(device),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'n_patients': payload['n_patients'],
        'n_images': payload['n_images'],
        'n_test_images': payload.get('n_test_images'),
        'test_patients': payload.get('test_patients'),
        'split_seed': SPLIT_SEED,
    }
    doc = build_cv_document(
        args.dataset,
        architecture_name,
        str(folds_path),
        fold_records,
        extra_info=extra_info,
        test_metrics=test_metrics,
        final_train=final_train,
    )
    save_cv_results(doc, out_path)
    logger.info('Saved CV results: %s', out_path)
    agg = doc.get('aggregate_metrics', {})
    for key in ('iou', 'dice', 'precision'):
        if key in agg:
            s = agg[key]
            logger.info(
                '  val %s: %.4f ± %.4f  (95%% CI [%.4f, %.4f])',
                key, s['mean'], s['std'], s['ci95_low'], s['ci95_high'],
            )
    if test_metrics:
        logger.info(
            '  TEST IoU=%.4f Dice=%.4f Prec=%.4f Rec=%.4f F1=%.4f',
            test_metrics['iou'], test_metrics['dice'], test_metrics['precision'],
            test_metrics['recall'], test_metrics['f1'],
        )
    logger.info('=' * 80)
    return doc
