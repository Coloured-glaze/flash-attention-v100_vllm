// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#include <c10/util/Exception.h>

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifndef FA2_HDIM_MAX
#define FA2_HDIM_MAX 256
#endif

#if FA2_HDIM_MAX >= 512
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    if (HEADDIM <= 32) { \
      constexpr static int kHeadDim = 32; \
      return __VA_ARGS__(); \
    } else if (HEADDIM <= 64) { \
      constexpr static int kHeadDim = 64; \
      return __VA_ARGS__(); \
    } else if (HEADDIM <= 96) { \
      constexpr static int kHeadDim = 96; \
      return __VA_ARGS__(); \
    } else if (HEADDIM <= 128) { \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__(); \
    } else if (HEADDIM <= 192) { \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__(); \
    } else if (HEADDIM <= 256) { \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__(); \
    } else { \
      constexpr static int kHeadDim = 512; \
      return __VA_ARGS__(); \
    } \
  }()
#elif FA2_HDIM_MAX >= 256
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); } \
    else if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); } \
    else if (HEADDIM <= 96) { constexpr static int kHeadDim = 96; return __VA_ARGS__(); } \
    else if (HEADDIM <= 128) { constexpr static int kHeadDim = 128; return __VA_ARGS__(); } \
    else if (HEADDIM <= 192) { constexpr static int kHeadDim = 192; return __VA_ARGS__(); } \
    else { constexpr static int kHeadDim = 256; return __VA_ARGS__(); } \
  }()
#elif FA2_HDIM_MAX >= 192
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); } \
    else if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); } \
    else if (HEADDIM <= 96) { constexpr static int kHeadDim = 96; return __VA_ARGS__(); } \
    else if (HEADDIM <= 128) { constexpr static int kHeadDim = 128; return __VA_ARGS__(); } \
    else { constexpr static int kHeadDim = 192; return __VA_ARGS__(); } \
  }()
#elif FA2_HDIM_MAX >= 128
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); } \
    else if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); } \
    else if (HEADDIM <= 96) { constexpr static int kHeadDim = 96; return __VA_ARGS__(); } \
    else { constexpr static int kHeadDim = 128; return __VA_ARGS__(); } \
  }()
#elif FA2_HDIM_MAX >= 96
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); } \
    else if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); } \
    else { constexpr static int kHeadDim = 96; return __VA_ARGS__(); } \
  }()
#elif FA2_HDIM_MAX >= 64
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); } \
    else { constexpr static int kHeadDim = 64; return __VA_ARGS__(); } \
  }()
#else
#define HEADDIM_SWITCH(HEADDIM, ...) \
  [&] { \
    constexpr static int kHeadDim = 32; \
    return __VA_ARGS__(); \
  }()
#endif


#ifdef FA2_HDIM_SELECTIVE

#ifdef FA2_HDIM_32
#define _HDIM_ELSEIF_32(HEADDIM, ...) else if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_32(HEADDIM, ...) 
#endif

#ifdef FA2_HDIM_64
#define _HDIM_ELSEIF_64(HEADDIM, ...) else if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_64(HEADDIM, ...) 
#endif

#ifdef FA2_HDIM_96
#define _HDIM_ELSEIF_96(HEADDIM, ...) else if (HEADDIM <= 96) { constexpr static int kHeadDim = 96; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_96(HEADDIM, ...) 
#endif

#ifdef FA2_HDIM_128
#define _HDIM_ELSEIF_128(HEADDIM, ...) else if (HEADDIM <= 128) { constexpr static int kHeadDim = 128; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_128(HEADDIM, ...) 
#endif

#ifdef FA2_HDIM_192
#define _HDIM_ELSEIF_192(HEADDIM, ...) else if (HEADDIM <= 192) { constexpr static int kHeadDim = 192; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_192(HEADDIM, ...) 
#endif

#ifdef FA2_HDIM_256
#define _HDIM_ELSEIF_256(HEADDIM, ...) else if (HEADDIM <= 256) { constexpr static int kHeadDim = 256; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_256(HEADDIM, ...) 
#endif

#ifdef FA2_HDIM_512
#define _HDIM_ELSEIF_512(HEADDIM, ...) else if (HEADDIM <= 512) { constexpr static int kHeadDim = 512; return __VA_ARGS__(); }
#else
#define _HDIM_ELSEIF_512(HEADDIM, ...) 
#endif

#undef HEADDIM_SWITCH
#define HEADDIM_SWITCH(HEADDIM, ...) [&] { if (false) {} _HDIM_ELSEIF_32(HEADDIM, __VA_ARGS__) _HDIM_ELSEIF_64(HEADDIM, __VA_ARGS__) _HDIM_ELSEIF_96(HEADDIM, __VA_ARGS__) _HDIM_ELSEIF_128(HEADDIM, __VA_ARGS__) _HDIM_ELSEIF_192(HEADDIM, __VA_ARGS__) _HDIM_ELSEIF_256(HEADDIM, __VA_ARGS__) _HDIM_ELSEIF_512(HEADDIM, __VA_ARGS__) else { TORCH_CHECK(false, "Unsupported head dimension: ", HEADDIM); } }()

#endif
