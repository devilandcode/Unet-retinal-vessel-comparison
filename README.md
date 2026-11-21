# Comparing U-Net, U-Net++, and Attention U-Net on Retinal Vessels

Open experiment code accompanying the report: a single comparison of three U-Net-family architectures on four public collections of color fundus photographs. The repository lets you check the split protocol, rerun training, and match the published figures in JSON. The image collections **are not included**.

## Architectures and datasets

| Architecture | Script (DRIVE) | STARE | CHASE_DB1 | HRF |
|--------------|----------------|-------|-----------|-----|
| U-Net | `u-net.py` | `u-net-stare.py` | `u-net-chase.py` | `u-net-hrf.py` |
| U-Net++ | `u-net-plus-plus.py` | `u-net-plus-plus-stare.py` | `u-net-plus-plus-chase.py` | `u-net-plus-plus-hrf.py` |
| Attention U-Net | `attention-u-net.py` | `attention-u-net-stare.py` | `attention-u-net-chase.py` | `attention-u-net-hrf.py` |

All three models use a ResNet-34 encoder pretrained on ImageNet; only the decoder paths differ. Inputs are 768×768 with three color channels. The loss, augmentations, and hyperparameters are shared (`common/config.py`).

## Evaluation protocol

1. **The split unit is the patient.** Both eyes of the same person stay together and are never placed in different train, validation, and test subsets.
2. **The held-out test set** is formed once and is not used for training, early stopping, or hyperparameter selection.
3. **Five-fold cross-validation** runs only on the training pool. Each of the five folds is trained **from scratch** with its own parameter initialization (`SPLIT_SEED + 1000 + fold index`). Weights are not carried over between folds.
4. Across the five folds we report the mean, the sample standard deviation, and a two-sided 95% confidence interval (Student’s t distribution, four degrees of freedom).
5. **The final evaluation on the held-out set is a single run.** After the five folds, the model is trained once on the full training pool (epoch budget equal to the median across folds, with no early stopping) and evaluated once on the test set. There is no repeated final run, so the test metrics have no mean ± standard deviation.

## Setup

You need Python 3.10+ and, for full training, a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `kagglehub` package is needed only if you want to download public copies through Kaggle. Local directories work just as well (see below). Secrets are not stored in the repository; if you use the Kaggle API, configure it yourself following the Kaggle documentation.

Check the split protocol (no data and no training):

```bash
python -m unittest tests.test_protocol
```

## Data: where to get it and where to put it

The datasets **are not distributed** with the code. Obtain them from the rights holders and point the scripts at the path.

### DRIVE

How to point at the data:

- the `DRIVE_ROOT` environment variable — the root of a directory that contains `train/image` + `train/mask` and `test/image` + `test/mask` (or `Data/train/...`, `Data/test/...`);
- or the official layout `training/images` + `training/1st_manual` and `test/images` + `test/1st_manual` — the code reads that too, but then the file count is 40 and the saved `drive_folds.json` for 100 frames **will not match**: delete it and recreate it (`python u-net.py --splits-only`).

### STARE

Hoover’s vessel subset: 20 annotated frames. Put the images in `./stare-images` and the masks in `./stare-masks`, or set `STARE_IMAGES_DIR` and `STARE_MASKS_DIR`.

### CHASE_DB1

28 images from 14 children. Use the `CHASE_ROOT` variable or the `./chase-db1` directory. Masks from the first observer (`*1stHO*`) are preferred.

### HRF

45 frames (healthy / glaucoma / diabetic retinopathy) with **vessel** masks (`manual1` and similar), not field-of-view masks. Use `HRF_ROOT`, or `HRF_IMAGES_DIR` together with `HRF_MASKS_DIR`.

## How the three architectures share the same split

The files in `splits/*.json` fix the training pool, the held-out test set, and the five folds. All three architectures read the same file for a given dataset. As long as that file is present, the comparison uses the same patients, not different random partitions.

- `split_seed` is `20260830` in every file.
- For STARE, CHASE_DB1, and HRF the held-out test set is about 20% of patients at the same split seed; paired eyes stay together.
- For DRIVE the test set comes from the `test` directory (20 files in the copy used here) and the training pool from `train` (80 files).

If you delete a split JSON, the script rebuilds it from the files currently on disk. To reproduce the **published** tables, keep the files in `splits/`.

## How to run training

Examples (from the repository root, once the data is in place):

```bash
python u-net.py                 # DRIVE, all five folds plus one final test
python u-net.py --fold 0        # a single fold
python u-net.py --splits-only   # only write or check the split
python attention-u-net-stare.py
python u-net-plus-plus-chase.py
python u-net-hrf.py --max-epochs 1 --fold 0
```

Results are written to `results/<dataset>/<architecture>-cv.json` and, locally, to PNG collages. This package ships only the JSON from the published experiment in advance — see `results/README.md`.

## Data licenses

The code may be used to check the protocol. Images from DRIVE, STARE, CHASE_DB1, and HRF belong to their rights holders; download them from the official collection pages and follow their terms. They are not part of this repository.
