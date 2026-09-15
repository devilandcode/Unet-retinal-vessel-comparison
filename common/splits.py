"""
Patient-level 5-fold CV on the train-pool, plus a held-out test set.

Left/right eyes of the same person stay in the same split (train-pool vs test
and across CV folds). The fold file is written once and reused by U-Net,
U-Net++, Attention U-Net and nnU-Net v2.

DRIVE
-----
Original DRIVE: 40 fundus photos, one per patient.
  training images 21_training … 40_training  → patients 21–40
  test images     01_test … 20_test          → patients 01–20
  01_training and 01_test are NOT the same person (those IDs never coexist).

Kaggle copy `abdallahwagih/retina-blood-vessel`:
  Held-out TEST = all labeled images under Data/test (20 files). Never in CV.
  5-fold patient-level CV ONLY on Data/train (80 files).

Grouping rules, in order:
  1. Official DRIVE stem `NN_training` / `NN_test` / `NN_manual*` → patient NN
  2. Suffix _left/_right/_L/_R → same prefix = same patient
  3. Otherwise each unique image is its own patient (DRIVE has 1 eye / person).
     Train/test folders with the same numeric stem (train/0.png vs test/0.png)
     are treated as different patients — they are different photos.

STARE
-----
20 annotated images (Hoover vessel subset), 16 patients. No official test split.
A FIXED patient-level test set (~20% of patients, 3–4 people) is held out with
SPLIT_SEED; fellow-eye pairs stay entirely in test or entirely in the train-pool.
Remaining patients: 5-fold CV (need ≥5 train-pool patients).

CHASE_DB1
---------
28 images, 14 children, both eyes. Filenames like Image_01L.jpg / Image_01R.jpg.
Patient id = the numeric stem (01), not L/R. Fellow eyes never split across
train-pool / val / held-out test. No official test split: ~20% of patients
held out with SPLIT_SEED; remaining: 5-fold CV. Vessel GT: prefer *_1stHO.png.

HRF
---
Typical vessel set: 45 high-res fundus images (15 healthy / 15 glaucoma / 15 DR).
Filenames like 01_h.jpg, 07_g.jpg, 15_dr.jpg. Patient id = full case stem
including the disease letter (01_h, 01_g and 01_dr are three people). If a
left/right suffix is present (01_h_L / 01_h_R), it is stripped so fellow eyes
stay together. FOV masks (*_mask) are not vessel GT. No official test split:
~20% of patients held out with SPLIT_SEED; remaining: 5-fold CV.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import N_FOLDS, PROJECT_ROOT, SPLIT_SEED

logger = logging.getLogger(__name__)

PROTOCOL = 'patient-level-5-fold-cv-held-out-test'
STARE_TEST_PATIENT_FRACTION = 0.2
HELD_OUT_TEST_PATIENT_FRACTION = STARE_TEST_PATIENT_FRACTION

# Hoover vessel-subset fellow eyes (OD/OS of the same person).
STARE_FELLOW_EYES: Dict[str, str] = {
    'im0081': 'im0081',
    'im0082': 'im0081',
    'im0162': 'im0162',
    'im0163': 'im0162',
    'im0235': 'im0235',
    'im0236': 'im0235',
    'im0239': 'im0239',
    'im0240': 'im0239',
}

_DRIVE_OFFICIAL = re.compile(
    r'^(\d{1,2})_(training|test|manual)',
    re.IGNORECASE,
)
_LEFT_RIGHT = re.compile(
    r'^(.*?)(?:_|-)(left|right|lefteye|righteye|[lr])$',
    re.IGNORECASE,
)
_CHASE_EYE = re.compile(
    r'^(?:image[_-]?)?(\d{1,2})[lr]$',
    re.IGNORECASE,
)
_HRF_CASE = re.compile(
    r'^(\d{1,2})_(h|g|dr)(?:[_-]?(?:left|right|[lr]))?$',
    re.IGNORECASE,
)
_HRF_CASE_COMPACT = re.compile(
    r'^(\d{1,2})(h|g|dr)[_-]?[lr]$',
    re.IGNORECASE,
)
_OBSERVER_SUFFIX = re.compile(
    r'_(?:1st|2nd)\s*ho$',
    re.IGNORECASE,
)
_SAFE_CASE = re.compile(r'[^A-Za-z0-9_-]+')
_DRIVE_SPLIT_CHILD = frozenset({
    'image', 'images', 'mask', 'masks', '1st_manual',
})


@dataclass
class Sample:
    image_path: str
    mask_path: str
    patient_id: str
    case_id: str


def _stem(path: str) -> str:
    return Path(path).stem


def sanitize_case_id(raw: str) -> str:
    cleaned = _SAFE_CASE.sub('_', raw).strip('_')
    return cleaned or 'case'


def drive_patient_id(image_path: str) -> str:
    """See module docstring for grouping rules."""
    path = Path(image_path)
    stem = path.stem
    official = _DRIVE_OFFICIAL.match(stem)
    if official:
        return f"{int(official.group(1)):02d}"
    lr = _LEFT_RIGHT.match(stem)
    if lr:
        return lr.group(1)
    parent = path.parent.name.lower()
    if parent in {'image', 'images', 'mask', 'masks'}:
        parent = path.parent.parent.name.lower()
    return f"{parent}_{stem}"


def stare_patient_id(image_path: str) -> str:
    """See module docstring for fellow-eye pairing."""
    stem = _stem(image_path).lower()
    stem = re.sub(r'\.(ah|vk)$', '', stem)
    return STARE_FELLOW_EYES.get(stem, stem)


def chase_patient_id(image_path: str) -> str:
    """Numeric child id: Image_01L / Image_01R → 01. See module docstring."""
    stem = _OBSERVER_SUFFIX.sub('', _stem(image_path))
    m = _CHASE_EYE.match(stem)
    if m:
        return f"{int(m.group(1)):02d}"
    lr = _LEFT_RIGHT.match(stem)
    if lr:
        nested = _CHASE_EYE.match(lr.group(1))
        if nested:
            return f"{int(nested.group(1)):02d}"
        return lr.group(1)
    return stem


def hrf_patient_id(image_path: str) -> str:
    """Case stem with disease letter; L/R stripped. See module docstring."""
    stem = _stem(image_path)
    stem = re.sub(r'_(manual1|mask|fov)$', '', stem, flags=re.IGNORECASE)
    stem = _OBSERVER_SUFFIX.sub('', stem)
    m = _HRF_CASE.match(stem)
    if m:
        return f"{int(m.group(1)):02d}_{m.group(2).lower()}"
    compact = _HRF_CASE_COMPACT.match(stem)
    if compact:
        return f"{int(compact.group(1)):02d}_{compact.group(2).lower()}"
    chase = _CHASE_EYE.match(stem)
    if chase:
        return f"{int(chase.group(1)):02d}"
    lr = _LEFT_RIGHT.match(stem)
    if lr:
        nested = _HRF_CASE.match(lr.group(1))
        if nested:
            return f"{int(nested.group(1)):02d}_{nested.group(2).lower()}"
        return lr.group(1)
    return stem


def patient_id_for(dataset: str, image_path: str) -> str:
    name = dataset.lower()
    if name == 'drive':
        return drive_patient_id(image_path)
    if name == 'stare':
        return stare_patient_id(image_path)
    if name == 'chase':
        return chase_patient_id(image_path)
    if name == 'hrf':
        return hrf_patient_id(image_path)
    raise ValueError(f"Unknown dataset: {dataset}")


def classify_drive_split(image_path: str) -> str:
    """Return 'train' or 'test' from folder layout (Kaggle Data/test vs Data/train)."""
    parts = [p.lower() for p in Path(image_path).parts]
    for i in range(len(parts) - 1):
        if parts[i + 1] not in _DRIVE_SPLIT_CHILD:
            continue
        if parts[i] == 'test':
            return 'test'
        if parts[i] in {'train', 'training'}:
            return 'train'
    stem = Path(image_path).stem.lower()
    if re.search(r'(^|_)test($|_)', stem) or stem.endswith('_test'):
        return 'test'
    if 'training' in stem:
        return 'train'
    raise ValueError(
        f'Cannot classify DRIVE train/test for {image_path}. '
        'Expected Data/train or Data/test (Kaggle) or training/ vs test/ (official).'
    )


def assign_unique_case_ids(image_paths: Sequence[str]) -> List[str]:
    """Stable unique nnU-Net case identifiers (no collisions across train/test)."""
    used = set()
    ids: List[str] = []
    for path in image_paths:
        p = Path(path)
        base = sanitize_case_id(p.stem)
        if base in used:
            base = sanitize_case_id(f"{p.parent.name}_{p.stem}")
        suffix = 2
        candidate = base
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        ids.append(candidate)
    return ids


def make_samples(
    dataset: str,
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
) -> List[Sample]:
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}"
        )
    case_ids = assign_unique_case_ids(image_paths)
    return [
        Sample(
            image_path=str(img),
            mask_path=str(msk),
            patient_id=patient_id_for(dataset, str(img)),
            case_id=cid,
        )
        for img, msk, cid in zip(image_paths, mask_paths, case_ids)
    ]


def assert_no_patient_leak(train: Sequence[Sample], val: Sequence[Sample]) -> None:
    train_p = {s.patient_id for s in train}
    val_p = {s.patient_id for s in val}
    leak = train_p & val_p
    if leak:
        raise ValueError(f"Patient leak across train/val: {sorted(leak)}")


def assert_test_held_out(payload: dict) -> None:
    """Test patients/cases must not appear in any CV fold."""
    test_p = set(payload.get('test_patients') or [])
    test_c = set(payload.get('test_cases') or [])
    if not test_p or not test_c:
        raise ValueError('Held-out test is empty (test_patients / test_cases).')
    for fold in payload['folds']:
        fold_p = set(fold['train_patients']) | set(fold['val_patients'])
        fold_c = set(fold['train_cases']) | set(fold['val_cases'])
        leak_p = test_p & fold_p
        leak_c = test_c & fold_c
        if leak_p or leak_c:
            raise ValueError(
                f"Test leak into fold {fold['fold']}: "
                f"patients={sorted(leak_p)} cases={sorted(leak_c)}"
            )


def held_out_test_patient_count(n_patients: int, n_folds: int = N_FOLDS) -> int:
    n_test = max(1, int(round(n_patients * HELD_OUT_TEST_PATIENT_FRACTION)))
    n_test = min(n_test, n_patients - n_folds)
    if n_test < 1 or n_patients - n_test < n_folds:
        raise ValueError(
            f'Need a held-out test AND >= {n_folds} train-pool patients; '
            f'got {n_patients} patients total'
        )
    return n_test


def stare_test_patient_count(n_patients: int, n_folds: int = N_FOLDS) -> int:
    return held_out_test_patient_count(n_patients, n_folds=n_folds)


def _random_patient_holdout(
    samples: Sequence[Sample],
    n_folds: int,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """Hold out ~20% of patients; fellow-eye groups share patient_id so they stay together."""
    patients = sorted({s.patient_id for s in samples})
    n_test = held_out_test_patient_count(len(patients), n_folds=n_folds)
    rng = np.random.RandomState(seed)
    order = patients[:]
    rng.shuffle(order)
    test_ids = set(order[:n_test])
    train_pool = [s for s in samples if s.patient_id not in test_ids]
    test = [s for s in samples if s.patient_id in test_ids]
    assert_no_patient_leak(train_pool, test)
    return train_pool, test


def split_train_pool_and_test(
    dataset: str,
    samples: Sequence[Sample],
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
) -> Tuple[List[Sample], List[Sample]]:
    """Train-pool (CV) vs held-out test. Fellow-eye groups stay together."""
    name = dataset.lower()
    if name == 'drive':
        train_pool = [s for s in samples if classify_drive_split(s.image_path) == 'train']
        test = [s for s in samples if classify_drive_split(s.image_path) == 'test']
        if not test:
            raise ValueError(
                'DRIVE held-out test is empty. Expected labeled images in Data/test '
                '(Kaggle abdallahwagih/retina-blood-vessel) or official test/.'
            )
        if not train_pool:
            raise ValueError('DRIVE train-pool is empty. Expected Data/train images.')
        if len({s.patient_id for s in train_pool}) < n_folds:
            raise ValueError(
                f'DRIVE train-pool has fewer than {n_folds} patients for {n_folds}-fold CV'
            )
        assert_no_patient_leak(train_pool, test)
        return train_pool, test
    if name in {'stare', 'chase', 'hrf'}:
        return _random_patient_holdout(samples, n_folds=n_folds, seed=seed)
    raise ValueError(f'Unknown dataset: {dataset}')


def make_patient_folds(
    samples: Sequence[Sample],
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
) -> List[dict]:
    """Shuffle patients with `seed`, then round-robin into n folds. Same folds for all models.

    Call this on the train-pool only — held-out test patients must not be in `samples`.
    """
    if n_folds < 2:
        raise ValueError('n_folds must be >= 2')
    patients = sorted({s.patient_id for s in samples})
    if len(patients) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} patients for {n_folds}-fold CV, got {len(patients)}"
        )
    rng = np.random.RandomState(seed)
    order = patients[:]
    rng.shuffle(order)
    patient_fold = {pid: i % n_folds for i, pid in enumerate(order)}

    folds = []
    for fold_i in range(n_folds):
        val_patients = sorted(p for p, f in patient_fold.items() if f == fold_i)
        train_patients = sorted(p for p, f in patient_fold.items() if f != fold_i)
        val_set = {p for p in val_patients}
        train_samples = [s for s in samples if s.patient_id not in val_set]
        val_samples = [s for s in samples if s.patient_id in val_set]
        assert_no_patient_leak(train_samples, val_samples)
        folds.append({
            'fold': fold_i,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'train_cases': [s.case_id for s in train_samples],
            'val_cases': [s.case_id for s in val_samples],
            'n_train_images': len(train_samples),
            'n_val_images': len(val_samples),
        })
    return folds


def _grouping_notes() -> Dict[str, str]:
    return {
        'drive': (
            'DRIVE held-out TEST = Data/test (Kaggle) or official test/; never in CV. '
            '5-fold patient-level CV on Data/train only. '
            'Official IDs 01–20 (test) and 21–40 (train) are distinct people. '
            'Filename rules: NN_training/NN_test → patient NN; _left/_right share a prefix; '
            'otherwise unique file = unique patient (train vs test numeric stems stay separate).'
        ),
        'stare': (
            'STARE Hoover vessel subset: no official test split. '
            f'~{int(STARE_TEST_PATIENT_FRACTION * 100)}% of patients held out as a fixed '
            'test set (SPLIT_SEED); fellow-eye pairs im0081/82, im0162/63, im0235/36, '
            'im0239/40 share a patient id and stay entirely in test or train-pool. '
            'Remaining patients: 5-fold CV.'
        ),
        'chase': (
            'CHASE_DB1: 28 images from 14 children (both eyes). '
            'Patient id = numeric stem (Image_01L / Image_01R → 01); fellow eyes stay '
            'in the same split. No official test split. '
            f'~{int(HELD_OUT_TEST_PATIENT_FRACTION * 100)}% of patients held out '
            '(SPLIT_SEED); remaining patients: 5-fold CV. Masks: prefer 1stHO.'
        ),
        'hrf': (
            'HRF vessel set: typically 45 images (15 healthy / 15 glaucoma / 15 DR). '
            'Patient id = case stem including disease letter (01_h ≠ 01_g ≠ 01_dr); '
            'optional L/R suffix is stripped so fellow eyes stay together. '
            'FOV masks (*_mask) are not vessel GT. No official test split. '
            f'~{int(HELD_OUT_TEST_PATIENT_FRACTION * 100)}% of patients held out '
            '(SPLIT_SEED); remaining patients: 5-fold CV.'
        ),
    }


def folds_payload(
    dataset: str,
    samples: Sequence[Sample],
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
    grouping_note: str = '',
) -> dict:
    train_pool, test = split_train_pool_and_test(
        dataset, samples, n_folds=n_folds, seed=seed,
    )
    folds = make_patient_folds(train_pool, n_folds=n_folds, seed=seed)
    payload = {
        'dataset': dataset,
        'protocol': PROTOCOL,
        'n_folds': n_folds,
        'split_seed': seed,
        'n_patients': len({s.patient_id for s in samples}),
        'n_images': len(samples),
        'n_train_pool_patients': len({s.patient_id for s in train_pool}),
        'n_train_pool_images': len(train_pool),
        'n_test_images': len(test),
        'grouping': grouping_note or _grouping_notes().get(dataset.lower(), ''),
        'patients': sorted({s.patient_id for s in samples}),
        'train_pool_patients': sorted({s.patient_id for s in train_pool}),
        'test_patients': sorted({s.patient_id for s in test}),
        'test_cases': [s.case_id for s in test],
        'cases': [
            {
                'case_id': s.case_id,
                'patient_id': s.patient_id,
                'image': Path(s.image_path).name,
                'mask': Path(s.mask_path).name,
            }
            for s in samples
        ],
        'folds': folds,
    }
    assert_test_held_out(payload)
    return payload


def default_folds_path(dataset: str) -> Path:
    return PROJECT_ROOT / 'splits' / f'{dataset.lower()}_folds.json'


def save_folds(payload: dict, path: Optional[Path] = None) -> Path:
    path = Path(path) if path is not None else default_folds_path(payload['dataset'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_folds(path: Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def load_or_create_folds(
    dataset: str,
    samples: Sequence[Sample],
    folds_path: Optional[Path] = None,
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
    grouping_note: str = '',
) -> Tuple[dict, Path]:
    path = Path(folds_path) if folds_path is not None else default_folds_path(dataset)
    if path.exists():
        payload = load_folds(path)
        if payload.get('protocol') == PROTOCOL:
            _validate_folds_against_samples(payload, samples, dataset=dataset, seed=seed)
            return payload, path
        logger.warning(
            'Obsolete folds protocol %r in %s; regenerating %s',
            payload.get('protocol'), path, PROTOCOL,
        )
    payload = folds_payload(
        dataset,
        samples,
        n_folds=n_folds,
        seed=seed,
        grouping_note=grouping_note or _grouping_notes().get(dataset.lower(), ''),
    )
    save_folds(payload, path)
    return payload, path


def _validate_folds_against_samples(
    payload: dict,
    samples: Sequence[Sample],
    dataset: Optional[str] = None,
    seed: int = SPLIT_SEED,
) -> None:
    known = {s.case_id for s in samples}
    for fold in payload['folds']:
        missing_train = set(fold['train_cases']) - known
        missing_val = set(fold['val_cases']) - known
        if missing_train or missing_val:
            raise ValueError(
                'Saved folds reference unknown case ids. Delete the folds JSON '
                f'to regenerate. Missing train={sorted(missing_train)[:5]} '
                f'val={sorted(missing_val)[:5]}'
            )
        train_p = {s.patient_id for s in samples if s.case_id in set(fold['train_cases'])}
        val_p = {s.patient_id for s in samples if s.case_id in set(fold['val_cases'])}
        leak = train_p & val_p
        if leak:
            raise ValueError(f"Saved folds leak patients: {sorted(leak)}")
    missing_test = set(payload.get('test_cases') or []) - known
    if missing_test:
        raise ValueError(
            'Saved test_cases reference unknown ids. Delete the folds JSON '
            f'to regenerate. Missing={sorted(missing_test)[:5]}'
        )
    if payload.get('split_seed') != seed:
        raise ValueError(
            f"Saved split_seed={payload.get('split_seed')} != {seed}. "
            'Delete the folds JSON to regenerate.'
        )
    assert_test_held_out(payload)
    ds = dataset or payload.get('dataset')
    if ds:
        train_pool, test = split_train_pool_and_test(
            ds, samples, n_folds=len(payload['folds']), seed=seed,
        )
        expected_test = [s.case_id for s in test]
        if list(payload['test_cases']) != expected_test and set(payload['test_cases']) != set(expected_test):
            raise ValueError(
                'Saved held-out test does not match current data/seed. '
                'Delete the folds JSON to regenerate.'
            )
        expected_pool = {s.case_id for s in train_pool}
        fold_cases = set()
        for fold in payload['folds']:
            fold_cases.update(fold['train_cases'])
            fold_cases.update(fold['val_cases'])
        if fold_cases != expected_pool:
            raise ValueError(
                'Saved CV folds do not cover the current train-pool. '
                'Delete the folds JSON to regenerate.'
            )


def samples_for_fold(
    samples: Sequence[Sample],
    fold: dict,
) -> Tuple[List[Sample], List[Sample]]:
    val_ids = set(fold['val_cases'])
    train_ids = set(fold['train_cases'])
    train = [s for s in samples if s.case_id in train_ids]
    val = [s for s in samples if s.case_id in val_ids]
    assert_no_patient_leak(train, val)
    return train, val


def samples_for_test(samples: Sequence[Sample], payload: dict) -> List[Sample]:
    test_ids = set(payload['test_cases'])
    return [s for s in samples if s.case_id in test_ids]


def train_pool_samples(samples: Sequence[Sample], payload: dict) -> List[Sample]:
    test_ids = set(payload['test_cases'])
    pool = [s for s in samples if s.case_id not in test_ids]
    test = [s for s in samples if s.case_id in test_ids]
    assert_no_patient_leak(pool, test)
    return pool


def to_nnunet_splits(payload: dict) -> List[dict]:
    """nnU-Net v2 splits_final.json: list of {train, val} case-id lists (train-pool only)."""
    return [
        {'train': fold['train_cases'], 'val': fold['val_cases']}
        for fold in payload['folds']
    ]
