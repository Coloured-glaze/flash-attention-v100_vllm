cmake_minimum_required(VERSION 3.26)

project(vllm_flash_attn LANGUAGES CXX CUDA)

# C++ standard can be overridden via -DCMAKE_CXX_STANDARD=20
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

set(FA2_ENABLED ON)

# ============================================================================
# 编译优化选项：减少模板实例化，大幅缩短编译时间
# ============================================================================
# Flash Attention 使用大量 C++ 模板和 BOOL_SWITCH 宏，每个 run_flash_fwd 调用会
# 实例化 2^6 = 64 个不同的 CUDA kernel 变体。禁用不需要的特性可以显著减少编译时间。
#
# 使用方法（环境变量）：
#   export FA_DISABLE_ALIBI=1      # 禁用 ALIBI 位置编码
#   export FA_DISABLE_LOCAL=1      # 禁用 local/sliding window attention
#   export FA_DISABLE_SOFTCAP=1    # 禁用 softcap 特性
#   export FA_DISABLE_UNEVEN_K=1   # 禁用非对齐 K 维度优化
#
# 效果估算：
#   - 禁用 1 个特性：编译时间减少约 50%
#   - 禁用 2 个特性：编译时间减少约 75%
#   - 禁用 3 个特性：编译时间减少约 87.5%
#   - 禁用全部 4 个：编译时间减少约 93.75%
#
# 注意：禁用特性后，运行时调用该特性会报错或回退到慢路径
# ============================================================================

# Option to selectively build only specific head dimension kernels.
# Set to a comma-separated list of head dimensions, e.g. "128" or "64,128".
# If empty (default), all head dimension kernels will be built.
set(FA_HDIM "" CACHE STRING "Comma-separated list of head dimensions to build (e.g. '128' or '64,128'). Empty = build all.")

# Feature disable options - each reduces kernel template instantiations by ~50%
set(FA_DISABLE_ALIBI OFF CACHE BOOL "Disable ALIBI attention bias support (reduces compile time ~50%)")
set(FA_DISABLE_LOCAL OFF CACHE BOOL "Disable local/sliding window attention (reduces compile time ~50%)")
set(FA_DISABLE_SOFTCAP OFF CACHE BOOL "Disable softcap attention scaling (reduces compile time ~50%)")
set(FA_DISABLE_UNEVEN_K OFF CACHE BOOL "Disable non-aligned K dimension optimization (reduces compile time ~50%)")

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Supported python versions. These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12" "3.13")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: these should be kept in sync with the torch version in setup.py.
# Likely should also be in sync with the vLLM version.
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.7.0")

find_python_constrained_versions(${PYTHON_SUPPORTED_VERSIONS})

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
message(DEBUG "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
if (NOT Torch_FOUND)
    # conda install -c conda-forge kineto
    
    # Suppress kineto library not found warning (kineto is optional for profiling)
    set(TORCH_KINETO_LIBRARY "" CACHE STRING "Kineto library path (optional)" FORCE)
    find_package(Torch REQUIRED)
endif()

set(VLLM_GPU_LANG "CUDA")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_FA_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_FA_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if (NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_FA_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif ()


# Other flags
# list(APPEND VLLM_FA_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xptxas=-v)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
list(APPEND VLLM_FA_GPU_FLAGS -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

# Replace instead of appending, nvcc doesn't like duplicate -O flags.
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

# MSVC specific flags
if (MSVC)
    string(REPLACE "/O2" "/O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

    # Add /EHsc for exception handling
    add_compile_options(/EHsc)

    # Disable specific warnings:
    # C4101: unreferenced local variable
    # C177: variable was declared but never referenced (CUDA)

    # add_compile_options(/wd4101)
    # list(APPEND VLLM_FA_GPU_FLAGS -Xcompiler=/wd4101)

    # Disable CUDA warning #177 (variable declared but never referenced)
    # list(APPEND VLLM_FA_GPU_FLAGS --diag-suppress=177)
endif()

#
# _C extension
#

if (FA2_ENABLED)
    file(GLOB _FA2_ALL_SRCS 
        "csrc/flash_attn/src/flash_fwd_split_hdim*.cu"
        "csrc/flash_attn/src/flash_fwd_hdim*.cu")

    if (FA_HDIM STREQUAL "")
        set(FA2_GEN_SRCS "${_FA2_ALL_SRCS}")
        message(STATUS "FA2: Building ALL head dimension kernels")
    else ()
        string(REPLACE "," ";" _FA_HDIM_LIST "${FA_HDIM}")
        set(_FA2_HDIM_MAX 0)
        foreach(_HDIM ${_FA_HDIM_LIST})
            if (_HDIM GREATER _FA2_HDIM_MAX)
                set(_FA2_HDIM_MAX ${_HDIM})
            endif()
        endforeach()
        string(FIND "${FA_HDIM}" "," _FA_HDIM_COMMA_POS)
        if (_FA_HDIM_COMMA_POS EQUAL -1)
            set(FA2_GEN_SRCS "")
            foreach(_SRC ${_FA2_ALL_SRCS})
                get_filename_component(_SRC_NAME "${_SRC}" NAME_WE)
                string(FIND "${_SRC_NAME}" "hdim${_FA2_HDIM_MAX}_" _FOUND)
                if (NOT _FOUND EQUAL -1)
                    list(APPEND FA2_GEN_SRCS "${_SRC}")
                endif()
            endforeach()
            list(LENGTH FA2_GEN_SRCS _N_SRCS)
            message(STATUS "***** FA2: Building only hdim=${_FA2_HDIM_MAX} (${_N_SRCS} files) ${FA2_GEN_SRCS} *****")
        else ()
            set(FA2_GEN_SRCS "")
            set(_FA2_HDIM_DEFS "")
            foreach(_HDIM ${_FA_HDIM_LIST})
                list(APPEND _FA2_HDIM_DEFS "FA2_HDIM_${_HDIM}")
                foreach(_SRC ${_FA2_ALL_SRCS})
                    get_filename_component(_SRC_NAME "${_SRC}" NAME_WE)
                    string(FIND "${_SRC_NAME}" "hdim${_HDIM}_" _FOUND)
                    if (NOT _FOUND EQUAL -1)
                        list(APPEND FA2_GEN_SRCS "${_SRC}")
                    endif()
                endforeach()
            endforeach()
            list(REMOVE_DUPLICATES FA2_GEN_SRCS)
            list(LENGTH FA2_GEN_SRCS _N_SRCS)
            message(STATUS "***** FA2: Building SELECTIVE hdims: ${FA_HDIM} (${_N_SRCS} files) ${FA2_GEN_SRCS} *****")
            set(_FA2_SELECTIVE_DEFS "FA2_HDIM_SELECTIVE")
            foreach(_DEF ${_FA2_HDIM_DEFS})
                list(APPEND _FA2_SELECTIVE_DEFS "${_DEF}")
            endforeach()
        endif ()
    endif ()

    # For CUDA we set the architectures on a per file basis
    if (VLLM_GPU_LANG STREQUAL "CUDA")
        set(FA2_ARCHS "7.0")
        message(STATUS "FA2_ARCHS: ${FA2_ARCHS}")

        set_gencode_flags_for_srcs(
            SRCS "${FA2_GEN_SRCS}"
            CUDA_ARCHS "${FA2_ARCHS}")
    endif()

    define_gpu_extension_target(
        _vllm_fa2_C
        DESTINATION vllm_flash_attn
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES
            csrc/flash_attn/flash_api.cpp
            csrc/flash_attn/flash_api_torch_lib.cpp
            ${FA2_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_FA_GPU_FLAGS}
        USE_SABI 3
        WITH_SOABI)

    target_include_directories(_vllm_fa2_C PRIVATE
        csrc/flash_attn
        csrc/flash_attn/src
        csrc/common
        csrc/cutlass/include)

    # custom definitions
    if (FA_HDIM STREQUAL "")
        target_compile_definitions(_vllm_fa2_C PRIVATE
            FLASHATTENTION_DISABLE_BACKWARD
            FLASHATTENTION_DISABLE_DROPOUT
            FLASHATTENTION_DISABLE_PYBIND
        )
    else ()
        string(FIND "${FA_HDIM}" "," _FA_HDIM_COMMA_POS)
        if (_FA_HDIM_COMMA_POS EQUAL -1)
            target_compile_definitions(_vllm_fa2_C PRIVATE
                FLASHATTENTION_DISABLE_BACKWARD
                FLASHATTENTION_DISABLE_DROPOUT
                FLASHATTENTION_DISABLE_PYBIND
                FA2_HDIM_MAX=${_FA2_HDIM_MAX}
                FA2_HDIM_EXACT=${_FA2_HDIM_MAX}
            )
        else ()
            target_compile_definitions(_vllm_fa2_C PRIVATE
                FLASHATTENTION_DISABLE_BACKWARD
                FLASHATTENTION_DISABLE_DROPOUT
                FLASHATTENTION_DISABLE_PYBIND
                FA2_HDIM_MAX=${_FA2_HDIM_MAX}
                ${_FA2_SELECTIVE_DEFS}
            )
        endif ()
    endif ()

    # Feature disable definitions - reduces template instantiations
    if (FA_DISABLE_ALIBI)
        target_compile_definitions(_vllm_fa2_C PRIVATE FLASHATTENTION_DISABLE_ALIBI)
        message(STATUS "FA2: ALIBI support DISABLED")
    endif()
    if (FA_DISABLE_LOCAL)
        target_compile_definitions(_vllm_fa2_C PRIVATE FLASHATTENTION_DISABLE_LOCAL)
        message(STATUS "FA2: Local attention support DISABLED")
    endif()
    if (FA_DISABLE_SOFTCAP)
        target_compile_definitions(_vllm_fa2_C PRIVATE FLASHATTENTION_DISABLE_SOFTCAP)
        message(STATUS "FA2: Softcap support DISABLED")
    endif()
    if (FA_DISABLE_UNEVEN_K)
        target_compile_definitions(_vllm_fa2_C PRIVATE FLASHATTENTION_DISABLE_UNEVEN_K)
        message(STATUS "FA2: Uneven K optimization DISABLED")
    endif()
endif ()