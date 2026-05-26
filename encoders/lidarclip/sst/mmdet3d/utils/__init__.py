from mmcv.utils import Registry, build_from_cfg, print_log

from .collect_env import collect_env
from .logger import get_root_logger

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env', 'print_log',
    'register_all_modules',
]


def register_all_modules(init_default_scope=False):
    """No-op for mmengine compatibility. Old mmdet3d registries are populated
    via module imports, not through this mechanism."""
    pass
