"""Load DRIVE, STARE, CHASE_DB1 and HRF image/mask pairs."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CONFIG, PROJECT_ROOT
from .splits import Sample, make_samples

logger = logging.getLogger(__name__)

IMAGE_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.ppm')
MASK_EXTS = ('*.png', '*.gif', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.ppm')


def _list_files(directory: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    if not directory.exists():
        return files
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def _list_files_recursive(directory: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    if not directory.exists():
        return files
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def _kaggle_download(kaggle_id: str, label: str) -> Optional[Path]:
    try:
        import kagglehub
    except ImportError:
        logger.info('kagglehub not installed; trying local %s paths', label)
        return None
    try:
        path = kagglehub.dataset_download(kaggle_id)
        logger.info('%s downloaded to %s', label, path)
        return Path(path)
    except Exception as exc:
        logger.warning('kagglehub download of %s failed: %s', kaggle_id, exc)
        return None


def pair_by_stem(
    image_paths: Sequence[Path],
    mask_paths: Sequence[Path],
    numeric_fallback: bool = True,
) -> Tuple[List[str], List[str]]:
    """Pair images to masks by stem, then optionally by leading numeric id."""
    masks_by_stem: Dict[str, Path] = {}
    for m in mask_paths:
        masks_by_stem.setdefault(m.stem.lower(), m)

    numeric: Dict[str, Path] = {}
    if numeric_fallback:
        for m in mask_paths:
            digits = ''.join(ch for ch in m.stem if ch.isdigit())
            if digits:
                numeric.setdefault(digits, m)

    images, masks = [], []
    for img in image_paths:
        mask = masks_by_stem.get(img.stem.lower())
        if mask is None and numeric_fallback:
            digits = ''.join(ch for ch in img.stem if ch.isdigit())
            mask = numeric.get(digits)
        if mask is None:
            logger.warning('No mask for image %s', img.name)
            continue
        images.append(str(img))
        masks.append(str(mask))
    return images, masks


def _download_drive_kaggle() -> Optional[Path]:
    try:
        import kagglehub
    except ImportError:
        logger.info('kagglehub not installed; trying local DRIVE paths')
        return None
    try:
        path = kagglehub.dataset_download(CONFIG['dataset']['kaggle_id'])
        logger.info('DRIVE downloaded to %s', path)
        return Path(path)
    except Exception as exc:
        logger.warning('kagglehub download failed: %s', exc)
        return None


def resolve_drive_root(explicit: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(os.path.expanduser(explicit)))
    env = os.environ.get('DRIVE_ROOT')
    if env:
        candidates.append(Path(os.path.expanduser(env)))
    candidates.extend([
        PROJECT_ROOT / 'Data',
        PROJECT_ROOT / 'retina-blood-vessel',
        PROJECT_ROOT / 'retina-blood-vessels',
        Path.home() / 'Downloads' / 'retina-blood-vessel',
    ])
    downloaded = _download_drive_kaggle()
    if downloaded is not None:
        candidates.insert(0, Path(downloaded))

    for root in candidates:
        if (root / 'Data' / 'train' / 'image').exists():
            return root
        if (root / 'train' / 'image').exists():
            return root
        if (root / 'training' / 'images').exists():
            return root
    return None


def _collect_split_dirs(root: Path) -> List[Tuple[Path, Path]]:
    """Return (image_dir, mask_dir) pairs for every labeled split we can find."""
    options = [
        (root / 'Data' / 'train' / 'image', root / 'Data' / 'train' / 'mask'),
        (root / 'Data' / 'test' / 'image', root / 'Data' / 'test' / 'mask'),
        (root / 'train' / 'image', root / 'train' / 'mask'),
        (root / 'test' / 'image', root / 'test' / 'mask'),
        (root / 'training' / 'images', root / 'training' / '1st_manual'),
        (root / 'test' / 'images', root / 'test' / '1st_manual'),
    ]
    found = []
    seen = set()
    for img_dir, mask_dir in options:
        key = (img_dir.resolve() if img_dir.exists() else img_dir,
               mask_dir.resolve() if mask_dir.exists() else mask_dir)
        if img_dir.exists() and mask_dir.exists() and key not in seen:
            seen.add(key)
            found.append((img_dir, mask_dir))
    return found


def load_drive_samples(root: Optional[str] = None) -> List[Sample]:
    drive_root = resolve_drive_root(root)
    if drive_root is None:
        raise FileNotFoundError(
            'DRIVE dataset not found. Download '
            'https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel '
            'or set DRIVE_ROOT.'
        )
    logger.info('DRIVE root: %s', drive_root)
    all_images: List[str] = []
    all_masks: List[str] = []
    for img_dir, mask_dir in _collect_split_dirs(drive_root):
        imgs = _list_files(img_dir, IMAGE_EXTS)
        msks = _list_files(mask_dir, MASK_EXTS)
        paired_i, paired_m = pair_by_stem(imgs, msks)
        logger.info('  %s: %d pairs', img_dir, len(paired_i))
        all_images.extend(paired_i)
        all_masks.extend(paired_m)
    if not all_images:
        raise FileNotFoundError(f'No DRIVE image/mask pairs under {drive_root}')
    samples = make_samples('drive', all_images, all_masks)
    logger.info('DRIVE total: %d images, %d patients',
                len(samples), len({s.patient_id for s in samples}))
    return samples


def resolve_stare_dirs(
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
) -> Tuple[Path, Path]:
    img_candidates = []
    mask_candidates = []
    if images_dir:
        img_candidates.append(Path(os.path.expanduser(images_dir)))
    if masks_dir:
        mask_candidates.append(Path(os.path.expanduser(masks_dir)))
    if os.environ.get('STARE_IMAGES_DIR'):
        img_candidates.append(Path(os.environ['STARE_IMAGES_DIR']))
    if os.environ.get('STARE_MASKS_DIR'):
        mask_candidates.append(Path(os.environ['STARE_MASKS_DIR']))
    img_candidates.extend([
        PROJECT_ROOT / 'stare-images',
        Path('/kaggle/input/datasets/devilandcode/stare-vessels/stare-images'),
        Path('/kaggle/input/stare-vessels/stare-images'),
    ])
    mask_candidates.extend([
        PROJECT_ROOT / 'stare-masks',
        Path('/kaggle/input/datasets/devilandcode/stare-vessels/stare-masks'),
        Path('/kaggle/input/stare-vessels/stare-masks'),
    ])
    img_path = next((p for p in img_candidates if p.exists()), None)
    mask_path = next((p for p in mask_candidates if p.exists()), None)
    if img_path is None or mask_path is None:
        raise FileNotFoundError(
            'STARE folders not found. Expected ./stare-images and ./stare-masks '
            'or STARE_IMAGES_DIR / STARE_MASKS_DIR.'
        )
    return img_path, mask_path


def load_stare_samples(
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
) -> List[Sample]:
    img_dir, mask_dir = resolve_stare_dirs(images_dir, masks_dir)
    logger.info('STARE images: %s', img_dir)
    logger.info('STARE masks: %s', mask_dir)
    images = _list_files(img_dir, IMAGE_EXTS)
    masks = _list_files(mask_dir, MASK_EXTS)
    paired_i, paired_m = pair_by_stem(images, masks)
    if not paired_i:
        raise FileNotFoundError('No STARE image/mask pairs')
    samples = make_samples('stare', paired_i, paired_m)
    logger.info('STARE total: %d images, %d patients',
                len(samples), len({s.patient_id for s in samples}))
    return samples


_HO_MASK = re.compile(r'_(?:1st|2nd)\s*ho$', re.IGNORECASE)
_HRF_IMAGE_STEM = re.compile(r'^\d{1,2}_(h|g|dr)$', re.IGNORECASE)
_CHASE_IMAGE_HINT = re.compile(r'^(?:image[_-]?)?\d{1,2}[lr]$', re.IGNORECASE)
_MASK_DIR_NAMES = frozenset({
    'manual1', 'manual', 'manualsegm', '1st_manual', 'masks', 'mask',
    'gt', 'labels', 'annotations',
})


def _is_observer_mask(path: Path) -> bool:
    return bool(_HO_MASK.search(path.stem))


def _is_fov_mask(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith('_mask') or stem.endswith('_fov') or 'fov' in stem


def pair_chase_images_masks(
    image_paths: Sequence[Path],
    mask_paths: Sequence[Path],
) -> Tuple[List[str], List[str]]:
    """Pair CHASE fundus photos to observer masks; prefer 1stHO over 2ndHO."""
    first: Dict[str, Path] = {}
    second: Dict[str, Path] = {}
    plain: Dict[str, Path] = {}
    for mask in mask_paths:
        key = _HO_MASK.sub('', mask.stem).lower()
        sl = mask.stem.lower().replace(' ', '')
        if sl.endswith('1stho'):
            first[key] = mask
        elif sl.endswith('2ndho'):
            second[key] = mask
        else:
            plain.setdefault(key, mask)
    images, masks = [], []
    for img in image_paths:
        key = img.stem.lower()
        mask = first.get(key) or second.get(key) or plain.get(key)
        if mask is None:
            logger.warning('No CHASE mask for image %s', img.name)
            continue
        images.append(str(img))
        masks.append(str(mask))
    return images, masks


def _dedupe_by_stem(images: Sequence[str], masks: Sequence[str]) -> Tuple[List[str], List[str]]:
    seen = set()
    out_i, out_m = [], []
    for img, mask in zip(images, masks):
        key = Path(img).stem.lower()
        if key in seen:
            continue
        seen.add(key)
        out_i.append(img)
        out_m.append(mask)
    return out_i, out_m


def resolve_chase_root(explicit: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(os.path.expanduser(explicit)))
    env = os.environ.get('CHASE_ROOT')
    if env:
        candidates.append(Path(os.path.expanduser(env)))
    downloaded = _kaggle_download(CONFIG['dataset']['chase_kaggle_id'], 'CHASE_DB1')
    if downloaded is not None:
        candidates.insert(0, downloaded)
    candidates.extend([
        PROJECT_ROOT / 'chase-db1',
        PROJECT_ROOT / 'CHASEDB1',
        PROJECT_ROOT / 'CHASE_DB1',
        Path('/kaggle/input/chase-db1'),
        Path('/kaggle/input/datasets/hoverstrange/chase-db1'),
        Path('/kaggle/input/hoverstrange/chase-db1'),
    ])
    for root in candidates:
        if not root.exists():
            continue
        files = _list_files_recursive(root, IMAGE_EXTS)
        if any(_is_observer_mask(p) or _CHASE_IMAGE_HINT.match(p.stem) for p in files):
            return root
        if files:
            return root
    return None


def load_chase_samples(root: Optional[str] = None) -> List[Sample]:
    env_img = os.environ.get('CHASE_IMAGES_DIR')
    env_mask = os.environ.get('CHASE_MASKS_DIR')
    if env_img and env_mask:
        img_dir = Path(os.path.expanduser(env_img))
        mask_dir = Path(os.path.expanduser(env_mask))
        images = [p for p in _list_files(img_dir, IMAGE_EXTS) if not _is_observer_mask(p)]
        masks = _list_files(mask_dir, MASK_EXTS)
        paired_i, paired_m = pair_chase_images_masks(images, masks)
        chase_root = img_dir
    else:
        chase_root = resolve_chase_root(root)
        if chase_root is None:
            raise FileNotFoundError(
                'CHASE_DB1 not found. Download Kaggle hoverstrange/chase-db1 '
                '(Internet on / kagglehub) or Add Input, or set CHASE_ROOT.'
            )
        files = _list_files_recursive(chase_root, IMAGE_EXTS + MASK_EXTS)
        images = [p for p in files if not _is_observer_mask(p) and not _is_fov_mask(p)]
        images = [p for p in images if _CHASE_IMAGE_HINT.match(p.stem)] or images
        masks = [p for p in files if _is_observer_mask(p) or p.parent.name.lower() in _MASK_DIR_NAMES]
        if not masks:
            masks = [p for p in files if p.suffix.lower() in {'.png', '.gif'} and p not in images]
        paired_i, paired_m = pair_chase_images_masks(images, masks)
    paired_i, paired_m = _dedupe_by_stem(paired_i, paired_m)
    if not paired_i:
        raise FileNotFoundError(f'No CHASE_DB1 image/mask pairs under {chase_root}')
    logger.info('CHASE_DB1 root: %s (%d pairs)', chase_root, len(paired_i))
    samples = make_samples('chase', paired_i, paired_m)
    logger.info('CHASE_DB1 total: %d images, %d patients',
                len(samples), len({s.patient_id for s in samples}))
    return samples


def resolve_hrf_root(explicit: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(os.path.expanduser(explicit)))
    env = os.environ.get('HRF_ROOT')
    if env:
        candidates.append(Path(os.path.expanduser(env)))
    downloaded = _kaggle_download(CONFIG['dataset']['hrf_kaggle_id'], 'HRF')
    if downloaded is not None:
        candidates.insert(0, downloaded)
    candidates.extend([
        PROJECT_ROOT / 'hrf',
        PROJECT_ROOT / 'HRF',
        Path('/kaggle/input/hrf-images-for-dr'),
        Path('/kaggle/input/datasets/lavanya456/hrf-images-for-dr'),
        Path('/kaggle/input/lavanya456/hrf-images-for-dr'),
        Path('/kaggle/input/hrf'),
    ])
    for root in candidates:
        if root.exists():
            return root
    return None


def _hrf_dir_pairs(root: Path) -> List[Tuple[Path, Path]]:
    options = [
        (root / 'images', root / 'manual1'),
        (root / 'images', root / 'manual'),
        (root / 'images', root / 'manualsegm'),
        (root / 'healthy', root / 'healthy_manualsegm'),
        (root / 'glaucoma', root / 'glaucoma_manualsegm'),
        (root / 'diabetic_retinopathy', root / 'diabetic_retinopathy_manualsegm'),
        (root / 'diabetic retinopathy', root / 'diabetic retinopathy_manualsegm'),
    ]
    dirs = [p for p in root.rglob('*') if p.is_dir()] + [root]
    by_name: Dict[str, List[Path]] = {}
    for d in dirs:
        by_name.setdefault(d.name.lower(), []).append(d)
    for img_name, mask_name in (
        ('images', 'manual1'),
        ('images', 'manual'),
        ('image', 'manual1'),
        ('healthy', 'healthy_manualsegm'),
        ('glaucoma', 'glaucoma_manualsegm'),
        ('diabetic_retinopathy', 'diabetic_retinopathy_manualsegm'),
    ):
        for img_dir in by_name.get(img_name, []):
            parent = img_dir.parent
            mask_dir = parent / mask_name
            if mask_dir.exists() and (img_dir, mask_dir) not in options:
                options.append((img_dir, mask_dir))
    found = []
    seen = set()
    for img_dir, mask_dir in options:
        if img_dir.exists() and mask_dir.exists():
            key = (img_dir.resolve(), mask_dir.resolve())
            if key not in seen:
                seen.add(key)
                found.append((img_dir, mask_dir))
    return found


def _looks_like_hrf_image(path: Path) -> bool:
    return bool(_HRF_IMAGE_STEM.match(path.stem))


def load_hrf_samples(
    root: Optional[str] = None,
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
) -> List[Sample]:
    env_img = images_dir or os.environ.get('HRF_IMAGES_DIR')
    env_mask = masks_dir or os.environ.get('HRF_MASKS_DIR')
    all_images: List[str] = []
    all_masks: List[str] = []
    hrf_root: Optional[Path] = None

    if env_img and env_mask:
        img_dir = Path(os.path.expanduser(env_img))
        mask_dir = Path(os.path.expanduser(env_mask))
        hrf_root = img_dir
        imgs = [p for p in _list_files(img_dir, IMAGE_EXTS) if not _is_fov_mask(p)]
        msks = [p for p in _list_files(mask_dir, MASK_EXTS) if not _is_fov_mask(p)]
        paired_i, paired_m = pair_by_stem(imgs, msks, numeric_fallback=False)
        all_images.extend(paired_i)
        all_masks.extend(paired_m)
    else:
        hrf_root = resolve_hrf_root(root)
        if hrf_root is None:
            raise FileNotFoundError(
                'HRF dataset not found. kagglehub id lavanya456/hrf-images-for-dr '
                '(Internet on) or Add Input / set HRF_ROOT. Need vessel GT (manual1), '
                'not only IQA good/bad JPGs.'
            )
        for img_dir, mask_dir in _hrf_dir_pairs(hrf_root):
            imgs = [p for p in _list_files(img_dir, IMAGE_EXTS) if not _is_fov_mask(p)]
            msks = [p for p in _list_files(mask_dir, MASK_EXTS) if not _is_fov_mask(p)]
            paired_i, paired_m = pair_by_stem(imgs, msks, numeric_fallback=False)
            logger.info('  %s + %s: %d pairs', img_dir, mask_dir, len(paired_i))
            all_images.extend(paired_i)
            all_masks.extend(paired_m)
        if not all_images:
            files = _list_files_recursive(hrf_root, IMAGE_EXTS + MASK_EXTS)
            images = [
                p for p in files
                if _looks_like_hrf_image(p) and not _is_fov_mask(p) and not _is_observer_mask(p)
            ]
            masks = [
                p for p in files
                if not _is_fov_mask(p)
                and (
                    p.parent.name.lower() in _MASK_DIR_NAMES
                    or p.suffix.lower() in {'.tif', '.tiff', '.gif', '.png'}
                )
                and not _looks_like_hrf_image(p)
            ]
            if not masks:
                masks = [
                    p for p in files
                    if p not in images and not _is_fov_mask(p)
                    and p.suffix.lower() in {'.tif', '.tiff', '.gif', '.png'}
                ]
            if not images:
                images = [
                    p for p in files
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                    and not _is_observer_mask(p) and not _is_fov_mask(p)
                ]
            paired_i, paired_m = pair_by_stem(images, masks, numeric_fallback=False)
            all_images.extend(paired_i)
            all_masks.extend(paired_m)

    all_images, all_masks = _dedupe_by_stem(all_images, all_masks)
    if not all_images:
        raise FileNotFoundError(
            f'No HRF image/mask pairs under {hrf_root}. '
            'The Kaggle dump lavanya456/hrf-images-for-dr is often IQA-only '
            '(1_good.JPG / 1_bad.JPG) with no vessel masks. Add a vessel-segmentation '
            'HRF copy (images + manual1, names like 01_h.jpg) via Add Input or HRF_ROOT / '
            'HRF_IMAGES_DIR + HRF_MASKS_DIR.'
        )
    logger.info('HRF root: %s', hrf_root)
    samples = make_samples('hrf', all_images, all_masks)
    logger.info('HRF total: %d images, %d patients',
                len(samples), len({s.patient_id for s in samples}))
    return samples


def load_samples(dataset: str, **kwargs) -> List[Sample]:
    name = dataset.lower()
    if name == 'drive':
        return load_drive_samples(kwargs.get('root'))
    if name == 'stare':
        return load_stare_samples(kwargs.get('images_dir'), kwargs.get('masks_dir'))
    if name == 'chase':
        return load_chase_samples(kwargs.get('root'))
    if name == 'hrf':
        return load_hrf_samples(
            kwargs.get('root'), kwargs.get('images_dir'), kwargs.get('masks_dir'),
        )
    raise ValueError(f'Unknown dataset: {dataset}')
