/**
 * @file sse_blas_ker.h
 *
 * SSE kernel for Small matrix Linear algebra
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_BLAS_KER_H_
#define LSIMD_SSE_BLAS_KER_H_

#include "sse_arith.h"

namespace lsimd
{

	LSIMD_ENSURE_INLINE
	sse_f32v4 add_prod(sse_f32v4 y, sse_f32v4 a, sse_f32v4 x)  // y + a * x
	{
		return add(y, mul(a, x));
	}

#ifdef LSIMD_HAS_SSE3

	LSIMD_ENSURE_INLINE
	sse_f32v4 hadd(sse_f32v4 x, sse_f32v4 y)
	{
		return _mm_hadd_ps(x.v, y.v);
	}

	LSIMD_ENSURE_INLINE
	sse_f64v2 hadd(sse_f64v2 x, sse_f64v2 y)
	{
		return _mm_hadd_pd(x.v, y.v);
	}

#endif


#ifdef LSIMD_HAS_SSE4_1

	LSIMD_ENSURE_INLINE
	f32 dot(sse_f32v4 x, sse_f32v4 y)
	{
		__m128 v = _mm_dp_ps(x.v, y.v, 0xF1);
		return _mm_cvtss_f32(v);
	}

	LSIMD_ENSURE_INLINE
	f64 dot(sse_f64v2 x, sse_f64v2 y)
	{
		__m128d v = _mm_dp_pd(x.v, y.v, 0x31);
		return _mm_cvtsd_f64(v);
	}

#endif

}

#endif /* SSE_BLAS_H_ */
