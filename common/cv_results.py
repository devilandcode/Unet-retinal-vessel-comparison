"""CV result JSON (mean ± std and 95% CI). Old single-run files are left untouched."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG, PROJECT_ROOT
from .metrics import aggregate_fold_metrics
from .splits import PROTOCOL


METRIC_KEYS = (
    'iou', 'dice', 'precision', 'recall', 'f1', 'loss',
    'best_iou', 'inference_fps', 'inference_time_ms',
)


def cv_results_path(dataset: str, architecture: str) -> Path:
    return PROJECT_ROOT / 'results' / dataset.lower() / f'{architecture}-cv.json'


def build_cv_document(
    dataset: str,
    architecture: str,
    folds_path: str,
    fold_records: List[Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    final_train: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    per_fold_metrics = [rec['metrics'] for rec in fold_records]
    keys = [k for k in METRIC_KEYS if k in per_fold_metrics[0]] if per_fold_metrics else []
    doc = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset,
            'architecture': architecture,
            'protocol': PROTOCOL,
            'n_folds': len(fold_records),
            'folds_file': str(folds_path),
            'config': CONFIG,
            **(extra_info or {}),
        },
        'aggregate_metrics': aggregate_fold_metrics(per_fold_metrics, keys=keys),
        'per_fold': fold_records,
    }
    if test_metrics is not None:
        doc['test_metrics'] = test_metrics
    if final_train is not None:
        doc['final_train'] = final_train
    return doc


def save_cv_results(doc: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return path
