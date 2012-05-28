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

extern "C"
{
	__m128 vmlsExp4(__m128);
	__m128 vmlsLn4 (__m128);

	__m128d vmldExp2(__m128d);
	__m128d vmldLn2 (__m128d);
}


namespace lsimd
{
	sse_f32p exp(sse_f32p x)
	{
		return vmlsExp4(x.v);
	}

	sse_f64p exp(sse_f64p x)
	{
		return vmldExp2(x.v);
	}

	sse_f32p log(sse_f32p x)
	{
		return vmlsLn4(x.v);
	}

	sse_f64p log(sse_f64p x)
	{
		return vmldLn2(x.v);
	}
}


#endif  /* LSIMD_USE_SMVL */


#endif
