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

#include "sse_base.h"
#include "sse_const.h"

namespace lsimd
{
	/********************************************
	 *
	 *  Basic arithmetic functions
	 *
	 ********************************************/


	// basic arithmetic

	LSIMD_ENSURE_INLINE
	inline sse_f32p add(const sse_f32p a, const sse_f32p b)
	{
		return _mm_add_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p add(const sse_f64p a, const sse_f64p b)
	{
		return _mm_add_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p sub(const sse_f32p a, const sse_f32p b)
	{
		return _mm_sub_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p sub(const sse_f64p a, const sse_f64p b)
	{
		return _mm_sub_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p mul(const sse_f32p a, const sse_f32p b)
	{
		return _mm_mul_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p mul(const sse_f64p a, const sse_f64p b)
	{
		return _mm_mul_pd(a.v, b.v);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32p neg(const sse_f32p a)
	{
		return _mm_xor_ps(sse_const<f32>::sign_mask(), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p neg(const sse_f64p a)
	{
		return _mm_xor_pd(sse_const<f64>::sign_mask(), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p abs(const sse_f32p a)
	{
		return _mm_andnot_ps(sse_const<f32>::sign_mask(), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p abs(const sse_f64p a)
	{
		return _mm_andnot_pd(sse_const<f64>::sign_mask(), a.v);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32p vmin(const sse_f32p a, const sse_f32p b)
	{
		return _mm_min_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p vmin(const sse_f64p a, const sse_f64p b)
	{
		return _mm_min_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p vmax(const sse_f32p a, const sse_f32p b)
	{
		return _mm_max_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p vmax(const sse_f64p a, const sse_f64p b)
	{
		return _mm_max_pd(a.v, b.v);
	}



	/********************************************
	 *
	 *  Floating-point only arithmetic functions
	 *
	 ********************************************/


	LSIMD_ENSURE_INLINE
	inline sse_f32p div(const sse_f32p a, const sse_f32p b)
	{
		return _mm_div_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p div(const sse_f64p a, const sse_f64p b)
	{
		return _mm_div_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p sqrt(const sse_f32p a)
	{
		return _mm_sqrt_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p sqrt(const sse_f64p a)
	{
		return _mm_sqrt_pd(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p rcp(const sse_f32p a)
	{
		return _mm_div_ps(sse_const<f32>::ones(), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p approx_rcp(const sse_f32p a)
	{
		return _mm_rcp_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p rcp(const sse_f64p a)
	{
		return _mm_div_pd(sse_const<f64>::ones(), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p rsqrt(const sse_f32p a)
	{
		return _mm_div_ps(sse_const<f32>::ones(), _mm_sqrt_ps(a.v));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p approx_rsqrt(const sse_f32p a)
	{
		return _mm_rsqrt_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p rsqrt(const sse_f64p a)
	{
		return _mm_div_pd(sse_const<f64>::ones(), _mm_sqrt_pd(a.v));
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32p sqr(const sse_f32p a)
	{
		return _mm_mul_ps(a.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p sqr(const sse_f64p a)
	{
		return _mm_mul_pd(a.v, a.v);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32p cube(const sse_f32p a)
	{
		return _mm_mul_ps(_mm_mul_ps(a.v, a.v), a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p cube(const sse_f64p a)
	{
		return _mm_mul_pd(_mm_mul_pd(a.v, a.v), a.v);
	}




	/********************************************
	 *
	 *  Rounding functions
	 *
	 ********************************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32p floor_sse2(const sse_f32p a)
	{
		__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.v));
		__m128 b = _mm_and_ps(_mm_cmpgt_ps(t, a.v), sse_const<f32>::ones());

		return _mm_sub_ps(t, b);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p floor_sse2(const sse_f64p a)
	{
		__m128d t = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.v));
		__m128d b = _mm_and_pd(_mm_cmpgt_pd(t, a.v), sse_const<f64>::ones());

		return _mm_sub_pd(t, b);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f32p ceil_sse2(const sse_f32p a)
	{
		__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.v));
		__m128 b = _mm_and_ps(_mm_cmplt_ps(t, a.v), sse_const<f32>::ones());

		return _mm_add_ps(t, b);
	}


	LSIMD_ENSURE_INLINE
	inline sse_f64p ceil_sse2(const sse_f64p a)
	{
		__m128d t = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.v));
		__m128d b = _mm_and_pd(_mm_cmplt_pd(t, a.v), sse_const<f64>::ones());

		return _mm_add_pd(t, b);
	}


#ifdef LSIMD_HAS_SSE4_1
	LSIMD_ENSURE_INLINE
	inline sse_f32p floor_sse4(const sse_f32p a)
	{
		return _mm_floor_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p floor_sse4(const sse_f64p a)
	{
		return _mm_floor_pd(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p ceil_sse4(const sse_f32p a)
	{
		return _mm_ceil_ps(a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p ceil_sse4(const sse_f64p a)
	{
		return _mm_ceil_pd(a.v);
	}
#endif

	LSIMD_ENSURE_INLINE
	inline sse_f32p floor(const sse_f32p a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return floor_sse4(a);
#else
		return floor_sse2(a);
#endif
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p floor(const sse_f64p a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return floor_sse4(a);
#else
		return floor_sse2(a);
#endif
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32p ceil(const sse_f32p a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return ceil_sse4(a);
#else
		return ceil_sse2(a);
#endif
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64p ceil(const sse_f64p a)
	{
#ifdef LSIMD_HAS_SSE4_1
		return ceil_sse4(a);
#else
		return ceil_sse2(a);
#endif
	}


}

#endif /* SSE_ARITH_H_ */