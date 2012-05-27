/**
 * @file sse_const.h
 *
 * Useful SSE constants
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_CONST_H_
#define LSIMD_SSE_CONST_H_

#include "sse_base.h"

namespace lsimd
{

	template<typename T> struct sse_const;

	template<> struct sse_const<float>
	{
		LSIMD_ENSURE_INLINE static __m128 zeros()
		{
			return _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE static __m128 ones()
		{
			return _mm_set1_ps(1.f);
		}

		LSIMD_ENSURE_INLINE static __m128 twos()
		{
			return _mm_set1_ps(2.f);
		}

		LSIMD_ENSURE_INLINE static __m128 halfs()
		{
			return _mm_set1_ps(0.5f);
		}

		LSIMD_ENSURE_INLINE static __m128 sign_mask()
		{
			return _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		}
	};


	template<> struct sse_const<double>
	{
		LSIMD_ENSURE_INLINE static __m128d zeros()
		{
			return _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE static __m128d ones()
		{
			return _mm_set1_pd(1.0);
		}

		LSIMD_ENSURE_INLINE static __m128d twos()
		{
			return _mm_set1_pd(2.0);
		}

		LSIMD_ENSURE_INLINE static __m128 halfs()
		{
			return _mm_set1_pd(0.5);
		}

		LSIMD_ENSURE_INLINE static __m128d sign_mask()
		{
			return _mm_castsi128_pd(_mm_set_epi32(0x80000000, 0, 0x80000000, 0));
		}
	};


}

#endif /* SSE_CONST_H_ */
