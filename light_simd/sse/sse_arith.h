/**
 * @file sse_arith.h
 *
 * SSE Arithmetic computation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_ARITH_H_
#define LSIMD_SSE_ARITH_H_

#include "sse_pack.h"

namespace lsimd
{
	/********************************************
	 *
	 *  Basic arithmetic functions
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator + (LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_add_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator + (LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_add_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator - (LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_sub_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator - (LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_sub_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator * (LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_mul_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator * (LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_mul_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator / (LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_div_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator / (LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_div_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk operator - (LSIMD_VT(sse_f32pk) a)
	{
		return _mm_xor_ps(_mm_set1_ps(-0.f), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk operator - (const LSIMD_VT(sse_f64pk) a)
	{
		return _mm_xor_pd(_mm_set1_pd(-0.0), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk abs(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_andnot_ps(_mm_set1_ps(-0.f), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk abs(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_andnot_pd(_mm_set1_pd(-0.0), a.v);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32pk vmin(LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_min_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk vmin(LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_min_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk vmax(LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_max_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk vmax(LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_max_pd(a.v, b.v);
	}


	/********************************************
	 *
	 *  Other arithmetic functions
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk sqrt(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_sqrt_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk sqrt(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_sqrt_pd(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk rcp(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_div_ps(_mm_set1_ps(1.0f), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk approx_rcp(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_rcp_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk rcp(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_div_pd(_mm_set1_pd(1.0), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk rsqrt(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(a.v));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk approx_rsqrt(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_rsqrt_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk rsqrt(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a.v));
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32pk sqr(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_mul_ps(a.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk sqr(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_mul_pd(a.v, a.v);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32pk cube(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_mul_ps(_mm_mul_ps(a.v, a.v), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk cube(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_mul_pd(_mm_mul_pd(a.v, a.v), a.v);
	}


	/********************************************
	 *
	 *  Single-scalar arithmetic
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk add_s(LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_add_ss(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk add_s(LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_add_sd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk sub_s(LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_sub_ss(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk sub_s(LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_sub_sd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk mul_s(LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_mul_ss(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk mul_s(LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_mul_sd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk div_s(LSIMD_VT(sse_f32pk) a, LSIMD_VT(sse_f32pk) b)
	{
		return _mm_div_ss(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk div_s(LSIMD_VT(sse_f64pk) a, LSIMD_VT(sse_f64pk) b)
	{
		return _mm_div_sd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk rcp_s(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_div_ss(_mm_set_ss(1.0f), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk rcp_s(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_div_sd(_mm_set_sd(1.0), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk sqrt_s(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_sqrt_ss(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk sqrt_s(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_sqrt_sd(a.v, a.v);
	}



	/********************************************
	 *
	 *  Rounding functions
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk floor_sse2(LSIMD_VT(sse_f32pk) a)
	{
		__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.v));
		__m128 b = _mm_and_ps(_mm_cmpgt_ps(t, a.v), _mm_set1_ps(1.0f));

		return _mm_sub_ps(t, b);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk floor_sse2(LSIMD_VT(sse_f64pk) a)
	{
		__m128d t = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.v));
		__m128d b = _mm_and_pd(_mm_cmpgt_pd(t, a.v), _mm_set1_pd(1.0));

		return _mm_sub_pd(t, b);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32pk ceil_sse2(LSIMD_VT(sse_f32pk) a)
	{
		__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.v));
		__m128 b = _mm_and_ps(_mm_cmplt_ps(t, a.v), _mm_set1_ps(1.0f));

		return _mm_add_ps(t, b);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f64pk ceil_sse2(LSIMD_VT(sse_f64pk) a)
	{
		__m128d t = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.v));
		__m128d b = _mm_and_pd(_mm_cmplt_pd(t, a.v), _mm_set1_pd(1.0));

		return _mm_add_pd(t, b);
	}


#ifdef LSIMD_HAS_SSE4_1
	LSIMD_ENSURE_INLINE
	inline sse_f32pk floor_sse4(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_floor_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk floor_sse4(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_floor_pd(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk ceil_sse4(LSIMD_VT(sse_f32pk) a)
	{
		return _mm_ceil_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk ceil_sse4(LSIMD_VT(sse_f64pk) a)
	{
		return _mm_ceil_pd(a.v);
	}
#endif

	LSIMD_ENSURE_INLINE
	inline sse_f32pk floor(LSIMD_VT(sse_f32pk) a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return floor_sse4(a);
#else
		return floor_sse2(a);
#endif
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk floor(LSIMD_VT(sse_f64pk) a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return floor_sse4(a);
#else
		return floor_sse2(a);
#endif
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk ceil(LSIMD_VT(sse_f32pk) a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return ceil_sse4(a);
#else
		return ceil_sse2(a);
#endif
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk ceil(LSIMD_VT(sse_f64pk) a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return ceil_sse4(a);
#else
		return ceil_sse2(a);
#endif
	}


}

#endif /* SSE_ARITH_H_ */
