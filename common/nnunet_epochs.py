"""Install a tiny nnU-Net v2 trainer subclass with a fixed epoch budget.

nnUNetv2_train has no --max-epochs. Trainers are discovered by class name under
nnunetv2.training.nnUNetTrainer. We write nnUNetTrainer_{N}epochs.py next to
nnUNetTrainer.py and pass -tr nnUNetTrainer_{N}epochs.

The generated __init__ must copy nnUNetTrainer's explicit parameters. The parent
does `self.my_init_kwargs[k] = locals()[k]` for each name in the *child*
signature; `*args, **kwargs` therefore KeyError because parent locals() has no
`args`. Official variants (nnUNetTrainer_5epochs, …) use the same pattern.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG

DEFAULT_EPOCHS = int(CONFIG['training']['epochs'])
OFFICIAL_NNUNET_EPOCHS = 1000
_TRAINER_EPOCHS = re.compile(r'^nnUNetTrainer_(\d+)epochs$')

# Common pip/Kaggle nnunetv2 signature when inspect is unavailable.
_FALLBACK_INIT_PARAMS = (
    'self, plans: dict, configuration: str, fold: int, dataset_json: dict, '
    "unpack_dataset: bool = True, device: torch.device = torch.device('cuda')"
)
_FALLBACK_SUPER_ARGS = (
    'plans, configuration, fold, dataset_json, unpack_dataset, device'
)


def epoch_trainer_class_name(epochs: int) -> str:
    n = int(epochs)
    if n < 1:
        raise ValueError(f'epochs must be >= 1, got {epochs}')
    return f'nnUNetTrainer_{n}epochs'


def _annotation_source(annotation: Any) -> Optional[str]:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _default_source(default: Any) -> tuple[Optional[str], bool]:
    if default is inspect.Parameter.empty:
        return None, False
    cls = type(default)
    if cls.__name__ == 'device' and str(getattr(cls, '__module__', '')).startswith('torch'):
        index = getattr(default, 'index', None)
        spec = default.type if index is None else f'{default.type}:{index}'
        return f'torch.device({spec!r})', True
    if default is True or default is False or default is None:
        return repr(default), False
    if isinstance(default, (int, float, str)):
        return repr(default), False
    return repr(default), 'torch' in str(getattr(cls, '__module__', ''))


def _render_parameter(param: inspect.Parameter) -> tuple[str, bool]:
    needs_torch = False
    piece = param.name
    annotation = _annotation_source(param.annotation)
    if annotation:
        piece = f'{param.name}: {annotation}'
        if 'torch' in annotation:
            needs_torch = True
    default_src, torch_default = _default_source(param.default)
    if torch_default:
        needs_torch = True
    if default_src is not None:
        piece = f'{piece} = {default_src}'
    return piece, needs_torch


def _render_init_from_signature(sig: inspect.Signature) -> tuple[str, str, bool]:
    """Build `def __init__(...)` params (incl. self) and `super().__init__(...)` args.

    Skips *args / **kwargs so the child signature names exist in the parent locals().
    """
    needs_torch = False
    def_parts = ['self']
    call_names = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        rendered, torch_needed = _render_parameter(param)
        needs_torch = needs_torch or torch_needed
        def_parts.append(rendered)
        call_names.append(name)
    return ', '.join(def_parts), ', '.join(call_names), needs_torch


def _nnunet_trainer_init_signature() -> Optional[inspect.Signature]:
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        return None
    return inspect.signature(nnUNetTrainer.__init__)


def epoch_trainer_source(
    epochs: int,
    init_signature: Optional[inspect.Signature] = None,
) -> str:
    class_name = epoch_trainer_class_name(epochs)
    sig = init_signature if init_signature is not None else _nnunet_trainer_init_signature()
    if sig is not None:
        param_list, super_args, needs_torch = _render_init_from_signature(sig)
    else:
        param_list, super_args, needs_torch = (
            _FALLBACK_INIT_PARAMS,
            _FALLBACK_SUPER_ARGS,
            True,
        )
    lines = ['from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer']
    if needs_torch:
        lines.append('import torch')
    lines.extend([
        '',
        f'class {class_name}(nnUNetTrainer):',
        f'    def __init__({param_list}):',
        f'        super().__init__({super_args})',
        f'        self.num_epochs = {int(epochs)}',
        '',
    ])
    return '\n'.join(lines)


def trainer_name_from_results_dir(trainer_dir: Path) -> str:
    return Path(trainer_dir).name.split('__')[0]


def epochs_from_trainer_name(trainer_name: str):
    m = _TRAINER_EPOCHS.match(trainer_name)
    return int(m.group(1)) if m else None


def install_epoch_trainer(epochs: int) -> str:
    """Write the trainer into the installed nnunetv2 package. Returns class name."""
    class_name = epoch_trainer_class_name(epochs)
    try:
        import nnunetv2.training.nnUNetTrainer as pkg
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    except ImportError as exc:
        raise RuntimeError(
            'nnunetv2 is not installed. pip install nnunetv2 SimpleITK'
        ) from exc
    dest = Path(pkg.__file__).resolve().parent / f'{class_name}.py'
    dest.write_text(
        epoch_trainer_source(epochs, inspect.signature(nnUNetTrainer.__init__)),
        encoding='utf-8',
    )
    return class_name
