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

#include "sse_pack.h"

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
	DECLARE_SVML_SSE_EXTERN1( cbrt )
	DECLARE_SVML_SSE_EXTERN2( pow )
	DECLARE_SVML_SSE_EXTERN2( hypot )

	DECLARE_SVML_SSE_EXTERN1( exp )
	DECLARE_SVML_SSE_EXTERN1( exp2 )
	DECLARE_SVML_SSE_EXTERN1( exp10 )
	DECLARE_SVML_SSE_EXTERN1( expm1 )

	DECLARE_SVML_SSE_EXTERN1( log )
	DECLARE_SVML_SSE_EXTERN1( log2 )
	DECLARE_SVML_SSE_EXTERN1( log10 )
	DECLARE_SVML_SSE_EXTERN1( log1p )

	DECLARE_SVML_SSE_EXTERN1( sin )
	DECLARE_SVML_SSE_EXTERN1( cos )
	DECLARE_SVML_SSE_EXTERN1( tan )

	DECLARE_SVML_SSE_EXTERN1( asin )
	DECLARE_SVML_SSE_EXTERN1( acos )
	DECLARE_SVML_SSE_EXTERN1( atan )
	DECLARE_SVML_SSE_EXTERN2( atan2 )

	DECLARE_SVML_SSE_EXTERN1( sinh )
	DECLARE_SVML_SSE_EXTERN1( cosh )
	DECLARE_SVML_SSE_EXTERN1( tanh )

	DECLARE_SVML_SSE_EXTERN1( asinh )
	DECLARE_SVML_SSE_EXTERN1( acosh )
	DECLARE_SVML_SSE_EXTERN1( atanh )

	DECLARE_SVML_SSE_EXTERN1( erf )
	DECLARE_SVML_SSE_EXTERN1( erfc )
}

namespace lsimd
{
	LSIMD_ENSURE_INLINE sse_f32pk cbrt( sse_f32pk x )
	{
		return SVML_SSE_F(cbrt)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk cbrt( sse_f64pk x )
	{
		return SVML_SSE_D(cbrt)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk pow( sse_f32pk x, sse_f32pk e )
	{
		return SVML_SSE_F(pow)(x.v, e.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk pow( sse_f64pk x, sse_f64pk e )
	{
		return SVML_SSE_D(pow)(x.v, e.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk hypot( sse_f32pk x, sse_f32pk y )
	{
		return SVML_SSE_F(hypot)(x.v, y.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk hypot( sse_f64pk x, sse_f64pk y )
	{
		return SVML_SSE_D(hypot)(x.v, y.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk exp( sse_f32pk x )
	{
		return SVML_SSE_F(exp)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk exp( sse_f64pk x )
	{
		return SVML_SSE_D(exp)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk exp2( sse_f32pk x )
	{
		return SVML_SSE_F(exp2)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk exp2( sse_f64pk x )
	{
		return SVML_SSE_D(exp2)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk exp10( sse_f32pk x )
	{
		return SVML_SSE_F(exp10)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk exp10( sse_f64pk x )
	{
		return SVML_SSE_D(exp10)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk expm1( sse_f32pk x )
	{
		return SVML_SSE_F(expm1)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk expm1( sse_f64pk x )
	{
		return SVML_SSE_D(expm1)(x.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk log( sse_f32pk x )
	{
		return SVML_SSE_F(log)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk log( sse_f64pk x )
	{
		return SVML_SSE_D(log)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk log2( sse_f32pk x )
	{
		return SVML_SSE_F(log2)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk log2( sse_f64pk x )
	{
		return SVML_SSE_D(log2)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk log10( sse_f32pk x )
	{
		return SVML_SSE_F(log10)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk log10( sse_f64pk x )
	{
		return SVML_SSE_D(log10)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk log1p( sse_f32pk x )
	{
		return SVML_SSE_F(log1p)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk log1p( sse_f64pk x )
	{
		return SVML_SSE_D(log1p)(x.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk sin( sse_f32pk x )
	{
		return SVML_SSE_F(sin)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk sin( sse_f64pk x )
	{
		return SVML_SSE_D(sin)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk cos( sse_f32pk x )
	{
		return SVML_SSE_F(cos)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk cos( sse_f64pk x )
	{
		return SVML_SSE_D(cos)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk tan( sse_f32pk x )
	{
		return SVML_SSE_F(tan)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk tan( sse_f64pk x )
	{
		return SVML_SSE_D(tan)(x.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk asin( sse_f32pk x )
	{
		return SVML_SSE_F(asin)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk asin( sse_f64pk x )
	{
		return SVML_SSE_D(asin)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk acos( sse_f32pk x )
	{
		return SVML_SSE_F(acos)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk acos( sse_f64pk x )
	{
		return SVML_SSE_D(acos)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk atan( sse_f32pk x )
	{
		return SVML_SSE_F(atan)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk atan( sse_f64pk x )
	{
		return SVML_SSE_D(atan)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk atan2( sse_f32pk x, sse_f32pk y )
	{
		return SVML_SSE_F(atan2)(x.v, y.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk atan2( sse_f64pk x, sse_f64pk y )
	{
		return SVML_SSE_D(atan2)(x.v, y.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk sinh( sse_f32pk x )
	{
		return SVML_SSE_F(sinh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk sinh( sse_f64pk x )
	{
		return SVML_SSE_D(sinh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk cosh( sse_f32pk x )
	{
		return SVML_SSE_F(cosh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk cosh( sse_f64pk x )
	{
		return SVML_SSE_D(cosh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk tanh( sse_f32pk x )
	{
		return SVML_SSE_F(tanh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk tanh( sse_f64pk x )
	{
		return SVML_SSE_D(tanh)(x.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk asinh( sse_f32pk x )
	{
		return SVML_SSE_F(asinh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk asinh( sse_f64pk x )
	{
		return SVML_SSE_D(asinh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk acosh( sse_f32pk x )
	{
		return SVML_SSE_F(acosh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk acosh( sse_f64pk x )
	{
		return SVML_SSE_D(acosh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk atanh( sse_f32pk x )
	{
		return SVML_SSE_F(atanh)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk atanh( sse_f64pk x )
	{
		return SVML_SSE_D(atanh)(x.v);
	}



	LSIMD_ENSURE_INLINE sse_f32pk erf( sse_f32pk x )
	{
		return SVML_SSE_F(erf)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk erf( sse_f64pk x )
	{
		return SVML_SSE_D(erf)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f32pk erfc( sse_f32pk x )
	{
		return SVML_SSE_F(erfc)(x.v);
	}

	LSIMD_ENSURE_INLINE sse_f64pk erfc( sse_f64pk x )
	{
		return SVML_SSE_D(erfc)(x.v);
	}
}

#undef DECLARE_SVML_SSE_EXTERN1

#undef SVML_SSE_F
#undef SVML_SSE_D

#endif  /* LSIMD_USE_SMVL */


#endif
