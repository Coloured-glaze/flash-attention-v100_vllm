__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in vLLM
# Use lazy loading to avoid circular import
import importlib

def _get_vllm_fa2_C():
    """Lazy import of _vllm_fa2_C to avoid circular import."""
    return importlib.import_module("._vllm_fa2_C", __package__)

_vllm_fa2_C = None

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    fa_version_unsupported_reason,
    is_fa_version_supported,
    _wrapped_flash_attn_backward,
)