/**
 * @file common_base.h
 *
 * Some basic definitions shared over the library
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_COMMON_BASE_H_
#define LSIMD_COMMON_BASE_H_

#include <light_simd/arch.h>

#include <cstddef>
#include <stdint.h>

namespace lsimd
{
	// primitive types

	typedef  int8_t  i8;
	typedef uint8_t  u8;

	typedef  int16_t i16;
	typedef uint16_t u16;

	typedef  int32_t i32;
	typedef uint32_t u32;

	typedef float  f32;
	typedef double f64;

	using std::size_t;
	using std::ptrdiff_t;

	typedef i32 index_t;

	// tag types

	struct aligned_t { };
	struct unaligned_t { };

	struct zero_t { };

	// forward declaration

	template<typename T, typename Kind> struct simd;

	template<typename T, typename Kind> struct simd_pack;

	template<typename T, unsigned int N, typename Kind>
	struct simd_kernel;

}

// useful macros

#if (LSIMD_COMPILER == LSIMD_GCC || LSIMD_COMPILER == LSIMD_CLANG )

#define LSIMD_ALIGN(n) __attribute__((aligned(n)))
#define LSIMD_ENSURE_INLINE __attribute__((always_inline))

#elif (LSIMD_COMPILER == LSIMD_MSVC)

#define LSIMD_ALIGN(n) __declspec(align(n))
#define LSIMD_ENSURE_INLINE __forceinline

#endif




#endif /* COMMON_BASE_H_ */
