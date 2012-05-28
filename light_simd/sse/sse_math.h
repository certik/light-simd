/**
 * @file sse_math.h
 *
 * Elementary mathematical functions for SSE
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MATH_H_
#define LSIMD_SSE_MATH_H_

#include "sse_base.h"

#ifdef LSIMD_USE_SVML

// External function prototypes

#define SVML_SSE_F( name ) __svml_##name##f4
#define SVML_SSE_D( name ) __svml_##name##2

#define DECLARE_SVML_SSE_EXTERN1( name ) \
	__m128  SVML_SSE_F(name)( __m128 ); \
	__m128d SVML_SSE_D(name)( __m128d );

#define DECLARE_SVML_SSE_EXTERN2( name ) \
	__m128  SVML_SSE_F(name)( __m128,  __m128  ); \
	__m128d SVML_SSE_D(name)( __m128d, __m128d );

extern "C"
{
	DECLARE_SVML_SSE_EXTERN1( exp )
	DECLARE_SVML_SSE_EXTERN1( log )
}

namespace lsimd
{
	LSIMD_ENSURE_INLINE sse_f32v4 exp( sse_f32v4 x )
	{
		return SVML_SSE_F(exp)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64v2 exp( sse_f64v2 x )
	{
		return SVML_SSE_D(exp)(x.v);
	}


	LSIMD_ENSURE_INLINE sse_f32v4 log( sse_f32v4 x )
	{
		return SVML_SSE_F(log)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64v2 log( sse_f64v2 x )
	{
		return SVML_SSE_D(exp)(x.v);
	}
}

#undef DECLARE_SVML_SSE_EXTERN1

#undef SVML_SSE_F
#undef SVML_SSE_D

#endif  /* LSIMD_USE_SMVL */


#endif
