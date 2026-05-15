from vllm_flash_attn.flash_attn_interface import (
    flash_attn_func,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

import vllm_flash_attn.flash_attn_interface

flash_attn_cuda = vllm_flash_attn.flash_attn_interface._vllm_fa2_C
