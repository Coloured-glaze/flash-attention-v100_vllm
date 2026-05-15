__version__ = "2.7.2.post1"

from .flash_attn_interface import (
    flash_attn_func,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

__all__ = [
    "flash_attn_func",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
