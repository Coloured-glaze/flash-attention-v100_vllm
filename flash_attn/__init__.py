__version__ = "2.7.2.post1"

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    fa_version_unsupported_reason,
    get_scheduler_metadata,
    is_fa_version_supported,
    _wrapped_flash_attn_backward,
)

# Lazy import to avoid circular import
def _get_vllm_fa2_C():
    """Lazy import of vllm_fa2_C to avoid circular import."""
    import importlib
    return importlib.import_module("vllm_flash_attn._vllm_fa2_C")

_vllm_fa2_C = None

from . import flash_attn_interface

