__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in vLLM
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    fa_version_unsupported_reason,
    is_fa_version_supported,
    _wrapped_flash_attn_backward,
    _vllm_fa2_C,
)