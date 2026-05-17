__version__ = "2.7.2.post1"

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    fa_version_unsupported_reason,
    get_scheduler_metadata,
    is_fa_version_supported,
    _wrapped_flash_attn_backward,
)

from . import flash_attn_interface

