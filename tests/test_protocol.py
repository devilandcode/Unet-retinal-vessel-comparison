"""Smoke tests for patient grouping, fold leak, and metric aggregation."""

from __future__ import annotations

import ast
import inspect
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import CONFIG, SPLIT_SEED
from common.metrics import aggregate_fold_metrics, mean_std_ci, metrics_from_counts
from common.nnunet_epochs import (
    DEFAULT_EPOCHS,
    epoch_trainer_class_name,
    epoch_trainer_source,
    epochs_from_trainer_name,
    trainer_name_from_results_dir,
)
from common.splits import (
    PROTOCOL,
    STARE_FELLOW_EYES,
    assert_no_patient_leak,
    assert_test_held_out,
    chase_patient_id,
    classify_drive_split,
    drive_patient_id,
    folds_payload,
    held_out_test_patient_count,
    hrf_patient_id,
    load_or_create_folds,
    make_patient_folds,
    make_samples,
    samples_for_fold,
    samples_for_test,
    stare_patient_id,
    stare_test_patient_count,
    to_nnunet_splits,
    train_pool_samples,
)


class DrivePatientIdTests(unittest.TestCase):
    def test_official_train_test_are_different_patients(self):
        self.assertEqual(drive_patient_id('/data/training/images/21_training.tif'), '21')
        self.assertEqual(drive_patient_id('/data/test/images/01_test.tif'), '01')
        # Original DRIVE never has both 21_training and 21_test; 01–20 vs 21–40 are distinct people.
        self.assertNotEqual(
            drive_patient_id('21_training.tif'),
            drive_patient_id('01_test.tif'),
        )

    def test_manual_mask_stem_maps_to_same_patient(self):
        self.assertEqual(drive_patient_id('21_manual1.gif'), '21')

    def test_left_right_share_patient(self):
        self.assertEqual(drive_patient_id('p07_left.png'), drive_patient_id('p07_right.png'))

    def test_kaggle_train_test_numeric_stems_do_not_collide(self):
        train = drive_patient_id('/ds/Data/train/image/0.png')
        test = drive_patient_id('/ds/Data/test/image/0.png')
        self.assertNotEqual(train, test)


class StarePatientIdTests(unittest.TestCase):
    def test_fellow_eyes_share_id(self):
        self.assertEqual(stare_patient_id('im0081.ppm'), stare_patient_id('im0082.ppm'))
        self.assertEqual(stare_patient_id('im0162.ppm'), stare_patient_id('im0163.ppm'))
        self.assertEqual(stare_patient_id('im0235.ppm'), stare_patient_id('im0236.ppm'))
        self.assertEqual(stare_patient_id('im0239.ppm'), stare_patient_id('im0240.ppm'))

    def test_singletons_are_unique(self):
        self.assertNotEqual(stare_patient_id('im0001.ppm'), stare_patient_id('im0002.ppm'))
        self.assertEqual(stare_patient_id('im0044.ppm'), 'im0044')

    def test_local_stare_set_has_sixteen_patients(self):
        stems = [
            'im0001', 'im0002', 'im0003', 'im0004', 'im0005', 'im0044',
            'im0077', 'im0081', 'im0082', 'im0139', 'im0162', 'im0163',
            'im0235', 'im0236', 'im0239', 'im0240', 'im0255', 'im0291',
            'im0319', 'im0324',
        ]
        patients = {stare_patient_id(s + '.ppm') for s in stems}
        self.assertEqual(len(stems), 20)
        self.assertEqual(len(patients), 16)
        self.assertTrue(set(STARE_FELLOW_EYES.values()).issubset(patients))


class ChasePatientIdTests(unittest.TestCase):
    def test_left_right_share_numeric_patient(self):
        self.assertEqual(chase_patient_id('Image_01L.jpg'), '01')
        self.assertEqual(chase_patient_id('Image_01R.jpg'), '01')
        self.assertEqual(
            chase_patient_id('/ds/Image_01L.jpg'),
            chase_patient_id('/ds/Image_01R.jpg'),
        )

    def test_different_children_are_different_patients(self):
        self.assertNotEqual(chase_patient_id('Image_01L.jpg'), chase_patient_id('Image_02L.jpg'))

    def test_observer_suffix_stripped(self):
        self.assertEqual(chase_patient_id('Image_07L_1stHO.png'), '07')

    def test_pair_prefers_first_observer(self):
        from common.data_io import pair_chase_images_masks
        images = [Path('/c/Image_01L.jpg')]
        masks = [Path('/c/Image_01L_2ndHO.png'), Path('/c/Image_01L_1stHO.png')]
        imgs, msks = pair_chase_images_masks(images, masks)
        self.assertEqual(imgs, ['/c/Image_01L.jpg'])
        self.assertTrue(msks[0].endswith('1stHO.png'))


class HrfPatientIdTests(unittest.TestCase):
    def test_disease_letter_keeps_patients_distinct(self):
        self.assertEqual(hrf_patient_id('01_h.jpg'), '01_h')
        self.assertEqual(hrf_patient_id('01_g.jpg'), '01_g')
        self.assertEqual(hrf_patient_id('01_dr.jpg'), '01_dr')
        self.assertNotEqual(hrf_patient_id('01_h.jpg'), hrf_patient_id('01_g.jpg'))
        self.assertNotEqual(hrf_patient_id('01_h.jpg'), hrf_patient_id('01_dr.jpg'))

    def test_fellow_eyes_share_patient(self):
        self.assertEqual(hrf_patient_id('01_h_L.jpg'), hrf_patient_id('01_h_R.jpg'))
        self.assertEqual(hrf_patient_id('01_h_L.jpg'), '01_h')
        self.assertEqual(hrf_patient_id('07_gL.jpg'), hrf_patient_id('07_gR.jpg'))


class FoldLeakTests(unittest.TestCase):
    def _synthetic_drive(self, n=40):
        images = [f'/drive/images/{i:02d}_training.png' for i in range(1, n + 1)]
        masks = [f'/drive/masks/{i:02d}_training.png' for i in range(1, n + 1)]
        return make_samples('drive', images, masks)

    def test_no_patient_in_both_sides(self):
        samples = self._synthetic_drive(40)
        folds = make_patient_folds(samples, n_folds=5, seed=20260830)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            train, val = samples_for_fold(samples, fold)
            assert_no_patient_leak(train, val)
            self.assertTrue(val)
            self.assertTrue(train)

    def test_all_images_of_a_patient_stay_together(self):
        images = ['/x/im0081.ppm', '/x/im0082.ppm', '/x/im0001.ppm', '/x/im0002.ppm',
                  '/x/im0003.ppm', '/x/im0004.ppm']
        masks = [p.replace('/x/', '/m/') for p in images]
        samples = make_samples('stare', images, masks)
        folds = make_patient_folds(samples, n_folds=5, seed=1)
        for fold in folds:
            val_cases = set(fold['val_cases'])
            pair = {'im0081', 'im0082'}
            in_val = pair & val_cases
            self.assertIn(len(in_val), (0, 2), 'fellow eyes split across folds')

    def test_identical_folds_for_all_architectures(self):
        samples = self._synthetic_drive(40)
        a = make_patient_folds(samples, n_folds=5, seed=20260830)
        b = make_patient_folds(samples, n_folds=5, seed=20260830)
        self.assertEqual(a, b)

    def test_saved_folds_roundtrip(self):
        samples = synthetic_drive_kaggle(80, 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'drive_folds.json'
            payload, written = load_or_create_folds('drive', samples, folds_path=path)
            again, _ = load_or_create_folds('drive', samples, folds_path=written)
            self.assertEqual(payload['folds'], again['folds'])
            self.assertEqual(payload['test_cases'], again['test_cases'])
            nn = to_nnunet_splits(payload)
            self.assertEqual(len(nn), 5)
            self.assertIn('train', nn[0])
            self.assertIn('val', nn[0])


STARE_STEMS = [
    'im0001', 'im0002', 'im0003', 'im0004', 'im0005', 'im0044',
    'im0077', 'im0081', 'im0082', 'im0139', 'im0162', 'im0163',
    'im0235', 'im0236', 'im0239', 'im0240', 'im0255', 'im0291',
    'im0319', 'im0324',
]


def synthetic_drive_kaggle(n_train=80, n_test=20):
    """Kaggle layout: Data/train (80) + Data/test (20)."""
    train_imgs = [f'/ds/Data/train/image/{i}.png' for i in range(n_train)]
    train_msks = [f'/ds/Data/train/mask/{i}.png' for i in range(n_train)]
    test_imgs = [f'/ds/Data/test/image/{i}.png' for i in range(n_test)]
    test_msks = [f'/ds/Data/test/mask/{i}.png' for i in range(n_test)]
    return make_samples('drive', train_imgs + test_imgs, train_msks + test_msks)


def synthetic_stare_twenty():
    images = [f'/stare-images/{s}.ppm' for s in STARE_STEMS]
    masks = [f'/stare-masks/{s}.ppm' for s in STARE_STEMS]
    return make_samples('stare', images, masks)


CHASE_STEMS = [f'Image_{i:02d}{eye}' for i in range(1, 15) for eye in ('L', 'R')]
HRF_STEMS = (
    [f'{i:02d}_h' for i in range(1, 16)]
    + [f'{i:02d}_g' for i in range(1, 16)]
    + [f'{i:02d}_dr' for i in range(1, 16)]
)


def synthetic_chase_twentyeight():
    images = [f'/chase/{s}.jpg' for s in CHASE_STEMS]
    masks = [f'/chase/{s}_1stHO.png' for s in CHASE_STEMS]
    return make_samples('chase', images, masks)


def synthetic_hrf_fortyfive():
    images = [f'/hrf/images/{s}.jpg' for s in HRF_STEMS]
    masks = [f'/hrf/manual1/{s}.tif' for s in HRF_STEMS]
    return make_samples('hrf', images, masks)


def _patient_cases(samples):
    by_pid = {}
    for s in samples:
        by_pid.setdefault(s.patient_id, []).append(s.case_id)
    return by_pid


class DriveHeldOutTestTests(unittest.TestCase):
    def test_kaggle_80_train_20_test_no_leak(self):
        samples = synthetic_drive_kaggle(80, 20)
        payload = folds_payload('drive', samples, n_folds=5, seed=SPLIT_SEED)
        self.assertEqual(payload['protocol'], PROTOCOL)
        self.assertEqual(payload['n_test_images'], 20)
        self.assertEqual(len(payload['test_cases']), 20)
        self.assertEqual(payload['n_train_pool_images'], 80)
        self.assertEqual(len(payload['folds']), 5)
        assert_test_held_out(payload)
        fold_cases = set()
        for fold in payload['folds']:
            fold_cases.update(fold['train_cases'])
            fold_cases.update(fold['val_cases'])
            train, val = samples_for_fold(samples, fold)
            assert_no_patient_leak(train, val)
        test_cases = set(payload['test_cases'])
        self.assertTrue(test_cases.isdisjoint(fold_cases))
        self.assertEqual(fold_cases | test_cases, {s.case_id for s in samples})
        test_s = samples_for_test(samples, payload)
        pool = train_pool_samples(samples, payload)
        self.assertEqual(len(test_s), 20)
        self.assertEqual(len(pool), 80)
        for s in test_s:
            self.assertEqual(classify_drive_split(s.image_path), 'test')
        for s in pool:
            self.assertEqual(classify_drive_split(s.image_path), 'train')

    def test_cv_never_sees_data_test_folder(self):
        samples = synthetic_drive_kaggle(80, 20)
        payload = folds_payload('drive', samples)
        for fold in payload['folds']:
            for s in samples:
                if s.case_id in fold['train_cases'] or s.case_id in fold['val_cases']:
                    self.assertNotIn('/Data/test/', s.image_path.replace('\\', '/'))

    def test_nnunet_splits_exclude_test_cases(self):
        samples = synthetic_drive_kaggle(80, 20)
        payload = folds_payload('drive', samples)
        test = set(payload['test_cases'])
        for fold in to_nnunet_splits(payload):
            self.assertTrue(test.isdisjoint(fold['train']))
            self.assertTrue(test.isdisjoint(fold['val']))

    def test_obsolete_protocol_is_regenerated(self):
        samples = synthetic_drive_kaggle(80, 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'drive_folds.json'
            path.write_text(json.dumps({
                'dataset': 'drive',
                'protocol': 'patient-level-5-fold-cv',
                'folds': [],
            }))
            payload, _ = load_or_create_folds('drive', samples, folds_path=path)
            self.assertEqual(payload['protocol'], PROTOCOL)
            self.assertEqual(payload['n_test_images'], 20)


class StareHeldOutTestTests(unittest.TestCase):
    def test_fellow_eyes_not_split_across_train_pool_and_test(self):
        samples = synthetic_stare_twenty()
        payload = folds_payload('stare', samples, n_folds=5, seed=SPLIT_SEED)
        self.assertEqual(payload['protocol'], PROTOCOL)
        n_patients = 16
        n_test_p = stare_test_patient_count(n_patients, n_folds=5)
        self.assertGreaterEqual(n_test_p, 3)
        self.assertLessEqual(n_test_p, 4)
        self.assertEqual(len(payload['test_patients']), n_test_p)
        self.assertGreaterEqual(payload['n_train_pool_patients'], 5)
        self.assertEqual(len(payload['folds']), 5)
        assert_test_held_out(payload)
        by_pid = _patient_cases(samples)
        test_cases = set(payload['test_cases'])
        pool_cases = set()
        for fold in payload['folds']:
            pool_cases.update(fold['train_cases'])
            pool_cases.update(fold['val_cases'])
        self.assertTrue(test_cases.isdisjoint(pool_cases))
        for pid, cases in by_pid.items():
            n_test = sum(c in test_cases for c in cases)
            n_pool = sum(c in pool_cases for c in cases)
            self.assertFalse(
                n_test and n_pool,
                f'fellow-eye patient {pid} split across train-pool and test',
            )
            for fold in payload['folds']:
                val = set(fold['val_cases'])
                train = set(fold['train_cases'])
                in_val = [c for c in cases if c in val]
                in_train = [c for c in cases if c in train]
                self.assertFalse(
                    in_val and in_train,
                    f'fellow-eye patient {pid} split across fold {fold["fold"]}',
                )

    def test_five_folds_cover_train_pool_only(self):
        samples = synthetic_stare_twenty()
        payload = folds_payload('stare', samples, n_folds=5, seed=SPLIT_SEED)
        pool = {s.case_id for s in train_pool_samples(samples, payload)}
        covered = set()
        for fold in payload['folds']:
            covered.update(fold['val_cases'])
        self.assertEqual(covered, pool)
        self.assertEqual(len(payload['folds']), 5)

    def test_repo_stare_folds_json_is_new_protocol(self):
        path = ROOT / 'splits' / 'stare_folds.json'
        if not path.exists():
            self.skipTest('splits/stare_folds.json missing')
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(payload['protocol'], PROTOCOL)
        self.assertIn('test_patients', payload)
        self.assertIn('test_cases', payload)
        self.assertGreaterEqual(payload['n_train_pool_patients'], 5)
        assert_test_held_out(payload)
        pair = {'im0235', 'im0236'}
        test = set(payload['test_cases'])
        pool = set()
        for fold in payload['folds']:
            pool.update(fold['train_cases'])
            pool.update(fold['val_cases'])
        in_test = pair & test
        in_pool = pair & pool
        self.assertTrue(len(in_test) in (0, 2) and not (in_test and in_pool))


class ChaseHeldOutTestTests(unittest.TestCase):
    def test_fellow_eyes_not_split_and_no_patient_leak(self):
        samples = synthetic_chase_twentyeight()
        self.assertEqual(len(samples), 28)
        self.assertEqual(len({s.patient_id for s in samples}), 14)
        payload = folds_payload('chase', samples, n_folds=5, seed=SPLIT_SEED)
        self.assertEqual(payload['protocol'], PROTOCOL)
        n_test_p = held_out_test_patient_count(14, n_folds=5)
        self.assertEqual(n_test_p, 3)
        self.assertEqual(len(payload['test_patients']), 3)
        self.assertEqual(payload['n_train_pool_patients'], 11)
        self.assertEqual(payload['n_test_images'], 6)
        self.assertEqual(payload['n_train_pool_images'], 22)
        assert_test_held_out(payload)
        by_pid = _patient_cases(samples)
        test_cases = set(payload['test_cases'])
        pool_cases = set()
        for fold in payload['folds']:
            pool_cases.update(fold['train_cases'])
            pool_cases.update(fold['val_cases'])
            train, val = samples_for_fold(samples, fold)
            assert_no_patient_leak(train, val)
        self.assertTrue(test_cases.isdisjoint(pool_cases))
        for pid, cases in by_pid.items():
            n_test = sum(c in test_cases for c in cases)
            n_pool = sum(c in pool_cases for c in cases)
            self.assertFalse(
                n_test and n_pool,
                f'fellow-eye patient {pid} split across train-pool and test',
            )
            self.assertIn(len(cases), (2,), msg=pid)
            for fold in payload['folds']:
                val = set(fold['val_cases'])
                train = set(fold['train_cases'])
                in_val = [c for c in cases if c in val]
                in_train = [c for c in cases if c in train]
                self.assertFalse(
                    in_val and in_train,
                    f'fellow-eye patient {pid} split across fold {fold["fold"]}',
                )

    def test_repo_chase_folds_json_if_present(self):
        path = ROOT / 'splits' / 'chase_folds.json'
        if not path.exists():
            self.skipTest('splits/chase_folds.json missing')
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(payload['protocol'], PROTOCOL)
        self.assertEqual(payload['n_patients'], 14)
        self.assertEqual(payload['n_images'], 28)
        assert_test_held_out(payload)
        pair = {'Image_01L', 'Image_01R'}
        test = set(payload['test_cases'])
        pool = set()
        for fold in payload['folds']:
            pool.update(fold['train_cases'])
            pool.update(fold['val_cases'])
        in_test = pair & test
        in_pool = pair & pool
        self.assertTrue(len(in_test) in (0, 2) and not (in_test and in_pool))


class HrfHeldOutTestTests(unittest.TestCase):
    def test_no_patient_leak_and_disease_letters_stay_distinct(self):
        samples = synthetic_hrf_fortyfive()
        self.assertEqual(len(samples), 45)
        self.assertEqual(len({s.patient_id for s in samples}), 45)
        payload = folds_payload('hrf', samples, n_folds=5, seed=SPLIT_SEED)
        self.assertEqual(payload['protocol'], PROTOCOL)
        n_test_p = held_out_test_patient_count(45, n_folds=5)
        self.assertEqual(n_test_p, 9)
        self.assertEqual(len(payload['test_patients']), 9)
        self.assertEqual(payload['n_train_pool_patients'], 36)
        assert_test_held_out(payload)
        for fold in payload['folds']:
            train, val = samples_for_fold(samples, fold)
            assert_no_patient_leak(train, val)
        test_s = samples_for_test(samples, payload)
        pool = train_pool_samples(samples, payload)
        self.assertEqual(len(test_s), 9)
        self.assertEqual(len(pool), 36)

    def test_synthetic_fellow_eyes_stay_together(self):
        images = [
            '/hrf/01_h_L.jpg', '/hrf/01_h_R.jpg',
            '/hrf/02_h.jpg', '/hrf/03_h.jpg', '/hrf/04_h.jpg',
            '/hrf/05_h.jpg', '/hrf/06_h.jpg',
        ]
        masks = [p.replace('/hrf/', '/m/') for p in images]
        samples = make_samples('hrf', images, masks)
        self.assertEqual(hrf_patient_id(images[0]), hrf_patient_id(images[1]))
        folds = make_patient_folds(samples, n_folds=5, seed=1)
        pair = {Path(images[0]).stem, Path(images[1]).stem}
        for fold in folds:
            val = set(fold['val_cases'])
            in_val = pair & val
            self.assertIn(len(in_val), (0, 2), 'HRF fellow eyes split across folds')

    def test_repo_hrf_folds_json_if_present(self):
        path = ROOT / 'splits' / 'hrf_folds.json'
        if not path.exists():
            self.skipTest('splits/hrf_folds.json missing')
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(payload['protocol'], PROTOCOL)
        self.assertEqual(payload['n_patients'], 45)
        self.assertEqual(payload['n_images'], 45)
        assert_test_held_out(payload)
        self.assertNotEqual(
            next(c['patient_id'] for c in payload['cases'] if c['case_id'] == '01_h'),
            next(c['patient_id'] for c in payload['cases'] if c['case_id'] == '01_g'),
        )


def _child_init_ast(src: str) -> ast.FunctionDef:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    return item
    raise AssertionError('generated source has no class __init__')


class NnUNetEpochFlagTests(unittest.TestCase):
    def test_trainer_class_name_and_source(self):
        self.assertEqual(epoch_trainer_class_name(100), 'nnUNetTrainer_100epochs')
        self.assertEqual(epoch_trainer_class_name(1), 'nnUNetTrainer_1epochs')
        src = epoch_trainer_source(100)
        self.assertIn('class nnUNetTrainer_100epochs(nnUNetTrainer):', src)
        self.assertIn('self.num_epochs = 100', src)
        self.assertNotIn('*args', src)
        self.assertNotIn('**kwargs', src)
        init = _child_init_ast(src)
        self.assertIsNone(init.args.vararg)
        self.assertIsNone(init.args.kwarg)
        self.assertEqual(epochs_from_trainer_name('nnUNetTrainer_100epochs'), 100)
        self.assertIsNone(epochs_from_trainer_name('nnUNetTrainer'))
        self.assertEqual(
            trainer_name_from_results_dir(Path('nnUNetTrainer_100epochs__nnUNetPlans__2d')),
            'nnUNetTrainer_100epochs',
        )
        with self.assertRaises(ValueError):
            epoch_trainer_class_name(0)

    def test_generated_init_copies_parent_signature_not_varargs(self):
        def parent_init(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device=None,
        ):
            pass

        src = epoch_trainer_source(100, inspect.signature(parent_init))
        self.assertNotIn('*args', src)
        self.assertNotIn('**kwargs', src)
        self.assertIn('self.num_epochs = 100', src)
        self.assertIn('unpack_dataset', src)
        self.assertIn('super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)', src)

        init = _child_init_ast(src)
        self.assertIsNone(init.args.vararg)
        self.assertIsNone(init.args.kwarg)
        names = [a.arg for a in init.args.args]
        self.assertEqual(
            names,
            ['self', 'plans', 'configuration', 'fold', 'dataset_json', 'unpack_dataset', 'device'],
        )

        class nnUNetTrainer:
            def __init__(self, plans, configuration, fold, dataset_json,
                         unpack_dataset=True, device=None):
                self.my_init_kwargs = {}
                for k in inspect.signature(self.__init__).parameters.keys():
                    self.my_init_kwargs[k] = locals()[k]
                self.num_epochs = 1000

        ns = {
            'nnUNetTrainer': nnUNetTrainer,
            'torch': types.SimpleNamespace(device=lambda kind='cuda': kind),
        }
        exec(src[src.index('class '):], ns)
        cls = ns['nnUNetTrainer_100epochs']
        params = list(inspect.signature(cls.__init__).parameters)
        self.assertNotIn('args', params)
        self.assertNotIn('kwargs', params)
        obj = cls({}, '2d', 0, {})
        self.assertEqual(obj.num_epochs, 100)
        self.assertEqual(obj.my_init_kwargs['plans'], {})
        self.assertEqual(obj.my_init_kwargs['configuration'], '2d')
        self.assertEqual(obj.my_init_kwargs['fold'], 0)

    def test_generated_init_strips_parent_varargs(self):
        def parent_init(self, plans, *args, **kwargs):
            pass

        src = epoch_trainer_source(5, inspect.signature(parent_init))
        self.assertNotIn('*args', src)
        self.assertNotIn('**kwargs', src)
        init = _child_init_ast(src)
        self.assertIsNone(init.args.vararg)
        self.assertIsNone(init.args.kwarg)
        self.assertEqual([a.arg for a in init.args.args], ['self', 'plans'])
        self.assertIn('self.num_epochs = 5', src)

    def test_parse_args_epochs_default_and_dataset_choices(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('nnunet_v2_cli', ROOT / 'nnunet-v2.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = mod.parse_args(['--dataset', 'chase', '--train', '--epochs', '50'])
        self.assertEqual(args.dataset, 'chase')
        self.assertEqual(args.epochs, 50)
        self.assertTrue(args.train)
        defaulted = mod.parse_args(['--dataset', 'hrf', '--all'])
        self.assertEqual(defaulted.epochs, DEFAULT_EPOCHS)
        self.assertEqual(DEFAULT_EPOCHS, 100)
        self.assertEqual(
            set(mod.DATASET_IDS),
            {'drive', 'stare', 'chase', 'hrf'},
        )
        self.assertEqual(mod.DATASET_IDS['chase'][0], 503)
        self.assertEqual(mod.DATASET_IDS['hrf'][0], 504)


class AugmentationTests(unittest.TestCase):
    def test_config_has_shared_magnitudes(self):
        aug = CONFIG['augmentation']
        self.assertEqual(aug['rotation'], 15)
        self.assertEqual(aug['brightness'], 0.1)
        self.assertEqual(aug['contrast'], 0.1)
        self.assertEqual(aug['shift_limit'], 0.05)
        self.assertEqual(aug['scale_limit'], 0.1)
        self.assertIn('horizontal_flip_p', aug)
        self.assertIn('grid_distort_limit', aug)

    def _albu(self):
        try:
            import albumentations as A  # noqa: F401
            from common.dataset import get_transforms
            return get_transforms
        except ImportError:
            self.skipTest('albumentations not installed')

    def test_train_pipeline_has_flips_shift_scale_and_mild_distortion(self):
        get_transforms = self._albu()
        t = get_transforms('train', image_size=64)
        names = [type(x).__name__ for x in t.transforms]
        joined = ' '.join(names)
        self.assertIn('HorizontalFlip', joined)
        self.assertTrue('ShiftScaleRotate' in joined or 'Affine' in joined)
        self.assertTrue('GridDistortion' in joined or 'Elastic' in joined)
        self.assertIn('RandomBrightnessContrast', joined)

    def test_val_and_test_are_resize_only(self):
        get_transforms = self._albu()
        for phase in ('val', 'test'):
            t = get_transforms(phase, image_size=64)
            names = [type(x).__name__ for x in t.transforms]
            self.assertEqual(names, ['Resize'], msg=phase)

    def test_val_is_deterministic(self):
        import numpy as np
        get_transforms = self._albu()
        rng = np.random.RandomState(0)
        img = rng.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        mask = np.zeros((48, 48), dtype=np.float32)
        t = get_transforms('val', image_size=32)
        a = t(image=img, mask=mask)
        b = t(image=img, mask=mask)
        np.testing.assert_array_equal(a['image'], b['image'])
        np.testing.assert_array_equal(a['mask'], b['mask'])


class MetricAggregationTests(unittest.TestCase):
    def test_mean_std_ci_known_values(self):
        values = [0.70, 0.72, 0.68, 0.71, 0.69]
        stats = mean_std_ci(values)
        self.assertAlmostEqual(stats['mean'], sum(values) / 5, places=10)
        self.assertEqual(stats['n'], 5)
        self.assertGreater(stats['std'], 0)
        self.assertLess(stats['ci95_low'], stats['mean'])
        self.assertGreater(stats['ci95_high'], stats['mean'])
        half = (stats['ci95_high'] - stats['ci95_low']) / 2
        expected_half = 2.7764451051977987 * stats['std'] / math.sqrt(5)
        self.assertAlmostEqual(half, expected_half, places=6)

    def test_single_value_ci_collapses(self):
        stats = mean_std_ci([0.5])
        self.assertEqual(stats['std'], 0.0)
        self.assertEqual(stats['ci95_low'], 0.5)
        self.assertEqual(stats['ci95_high'], 0.5)

    def test_perfect_overlap_metrics(self):
        m = metrics_from_counts(tp=10, fp=0, fn=0, tn=90)
        self.assertEqual(m['iou'], 1.0)
        self.assertEqual(m['dice'], 1.0)
        self.assertEqual(m['precision'], 1.0)

    def test_aggregate_fold_metrics_keys(self):
        folds = [
            {'iou': 0.6, 'dice': 0.7, 'precision': 0.8},
            {'iou': 0.62, 'dice': 0.72, 'precision': 0.81},
            {'iou': 0.58, 'dice': 0.68, 'precision': 0.79},
            {'iou': 0.61, 'dice': 0.71, 'precision': 0.80},
            {'iou': 0.59, 'dice': 0.69, 'precision': 0.78},
        ]
        agg = aggregate_fold_metrics(folds)
        self.assertIn('iou', agg)
        self.assertIn('mean', agg['iou'])
        self.assertIn('ci95_low', agg['precision'])


class RealStareGroupingTests(unittest.TestCase):
    def test_repo_stare_files_if_present(self):
        img_dir = ROOT / 'stare-images'
        if not img_dir.exists():
            self.skipTest('stare-images not in repo')
        paths = sorted(p for p in img_dir.glob('*.ppm'))
        patients = [stare_patient_id(str(p)) for p in paths]
        self.assertGreaterEqual(len(paths), 20)
        self.assertEqual(len(set(patients)), 16)
        pair = [stare_patient_id(str(img_dir / 'im0081.ppm')),
                stare_patient_id(str(img_dir / 'im0082.ppm'))]
        self.assertEqual(pair[0], pair[1])


class CombinedLossSanityTests(unittest.TestCase):
    def test_forward_accepts_out_of_range_and_nan(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed')
        from common.losses import CombinedLoss

        loss_fn = CombinedLoss()
        pred = torch.tensor(
            [[[[1.5, -0.2, float('nan'), float('inf'), float('-inf'), 0.5]]]],
            dtype=torch.float32,
        )
        target = torch.tensor([[[[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]]], dtype=torch.float32)
        out = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(out).item())
        self.assertEqual(out.ndim, 0)

    def test_forward_healthy_probabilities_still_finite(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed')
        from common.losses import CombinedLoss

        loss_fn = CombinedLoss()
        pred = torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float32)
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        out = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(out).item())


class EncoderParamGroupTests(unittest.TestCase):
    def test_unique_params_not_duplicated_across_groups(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest('torch not installed')
        from common.trainer import split_encoder_decoder_params

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = nn.Linear(4, 4)
                self.decoder = nn.Linear(4, 1)

        model = Tiny()
        enc, dec = split_encoder_decoder_params(model, ('enc1',))
        enc_ids = {id(p) for p in enc}
        dec_ids = {id(p) for p in dec}
        self.assertTrue(enc_ids)
        self.assertTrue(dec_ids)
        self.assertFalse(enc_ids & dec_ids)
        self.assertEqual(len(enc_ids) + len(dec_ids), sum(1 for _ in model.parameters()))


class VisualizationTests(unittest.TestCase):
    def test_plot_training_history_creates_parent_dirs(self):
        from common.viz import plot_training_history

        history = {
            'train_loss': [0.5, 0.4],
            'val_loss': [0.6, 0.5],
            'train_iou': [0.1, 0.2],
            'val_iou': [0.1, 0.15],
            'train_dice': [0.2, 0.3],
            'val_dice': [0.2, 0.25],
            'lr': [1e-4, 5e-5],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results' / 'drive' / 'history.png'
            plot_training_history(history, save_path=str(path))
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_save_prediction_collage_writes_png(self):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            self.skipTest('torch not installed')
        from common.viz import save_prediction_collage

        class DummySeg(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x.mean(dim=1, keepdim=True))

        images = torch.randn(5, 3, 32, 32)
        masks = (torch.rand(5, 1, 32, 32) > 0.7).float()
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)
        model = DummySeg()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results' / 'drive' / 'u-net-cv-fold0-predictions.png'
            save_prediction_collage(model, loader, torch.device('cpu'), str(path), n_samples=4)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_numpy_collage_green_overlay(self):
        import numpy as np
        from common.viz import save_numpy_prediction_collage

        images = [np.full((16, 16, 3), 40, dtype=np.uint8) for _ in range(2)]
        masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(2)]
        preds = [np.zeros((16, 16), dtype=np.uint8) for _ in range(2)]
        masks[0][4:8, 4:8] = 1
        preds[0][4:8, 4:8] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'out.png'
            save_numpy_prediction_collage(images, masks, preds, save_path=str(path))
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()


