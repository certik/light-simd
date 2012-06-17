/**
 * @file common_base.h
 *
 * This file defines a set of basic types that will be used by the entire library.
 * Particularly, it contains the definitions of
 *
 * - Basic numeric types (e.g. f32, f64, i32, etc)
 * - Basic tag types (e.g. aligned_t, unaligned_t and zero_t)
 * - Forward declaration of basic template classes (e.g. simd_pack, simd_vec, etc)
 * - A set of useful macros
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
#include <cstdio>

/**
 * The main namespace of the Light SIMD library.
 *
 * Most names, including classes, functions, constants
 * are declared in this namespace.
 */
namespace lsimd
{
	// primitive types

	/**
	 * 8-bit signed integer.
	 */
	typedef  int8_t  i8;

	/**
	 * 8-bit unsigned integer.
	 */
	typedef uint8_t  u8;

	/**
	 * 16-bit signed integer.
	 */
	typedef  int16_t i16;

	/**
	 * 16-bit unsigned integer.
	 */
	typedef uint16_t u16;

	/**
	 * 32-bit signed integer.
	 */
	typedef  int32_t i32;

	/**
	 * 32-bit unsigned integer.
	 */
	typedef uint32_t u32;

	/**
	 * single-precision (32-bit) floating-point real number.
	 */
	typedef float  f32;

	/**
	 * double-precision (64-bit) floating-point real number.
	 */
	typedef double f64;

	/**
	 * The unsigned integer type to represent sizes.
	 *
	 * @remark This is simply using std::size_t.
	 */
	using std::size_t;

	/**
	 * The unsigned integer type to represent the offset between two pointers.
	 *
	 * @remark This is simply using std::ptrdiff_t.
	 */
	using std::ptrdiff_t;

	typedef i32 index_t;

	// tag types

	/**
	 * A tag type that indicates the provided address is properly aligned.
	 *
	 * @see unaligned_t.
	 */
	struct aligned_t { };

	/**
	 * A tag type that indicates that the provided address is not necessarily
	 * aligned to the proper boundary.
	 *
	 * @see aligned_t
	 */
	struct unaligned_t { };

	/**
	 * A tag type that indicates to initialize all elements to zero values.
	 */
	struct zero_t { };


	// forward declaration


	/**
	 * A tag type that indicates to use SSE 128-bit data types for SIMD processing.
	 */
	struct sse_kind { };


	/**
	 * The default kind of data types for SIMD processing.
	 *
	 * @remark Currently, only SSE has been supported by the library,
	 *         and thus this is set to be the same as \ref sse_kind.
	 *         It will be set depending on platform, when more
	 *         SIMD kinds (e.g. AVX and NEON) are supported.
	 *
	 * @see sse_kind.
	 */
	typedef sse_kind default_simd_kind;


	template<typename T, typename Kind> struct simd;

	template<typename T, typename Kind=default_simd_kind> struct simd_pack;

	template<typename T, int N, typename Kind=default_simd_kind> struct simd_vec;

	template<typename T, int M, int N, typename Kind=default_simd_kind> struct simd_mat;

}

// useful macros

#if (LSIMD_COMPILER == LSIMD_GCC || LSIMD_COMPILER == LSIMD_CLANG )

/**
 * Specifies the minimum alignment of the ensuring variable/array.
 *
 * For example, the following code
 *
 * \code{.cpp}
 * LSIMD_ALIGN(16) a[4];
 * \endcode
 *
 * ensures that the array a is aligned to 16-byte boundary
 */
#define LSIMD_ALIGN(n) __attribute__((aligned(n)))

/**
 * Forces the ensuing function to be inlined.
 */
#define LSIMD_ENSURE_INLINE __attribute__((always_inline))

#elif (LSIMD_COMPILER == LSIMD_MSVC)

#define LSIMD_ALIGN(n) __declspec(align(n))
#define LSIMD_ENSURE_INLINE __forceinline

#endif




#endif /* COMMON_BASE_H_ */
