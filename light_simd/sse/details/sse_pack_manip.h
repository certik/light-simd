/**
 * @file sse_pack_manip.h
 *
 * Internal implementation of entry manipulation for SSE packs
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_PACK_MANIP_H_
#define LSIMD_SSE_PACK_MANIP_H_

#include "../sse_base.h"

namespace lsimd {  namespace sse {


	/********************************************
	 *
	 *  Entry extraction
	 *
	 ********************************************/

	template<int I> inline f32 f32p_extract(__m128 a);

	template<int I> inline f64 f64p_extract(__m128d a);


#if defined(LSIMD_HAS_SSE4_1)

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<0>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 0);
		return r;
	}

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<1>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 1);
		return r;
	}

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<2>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 2);
		return r;
	}

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<3>(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 3);
		return r;
	}

#else

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<0>(__m128 a)
	{
		return _mm_cvtss_f32(a);
	}

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<1>(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4)));
	}

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<2>(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 8)));
	}

	template<>
	LSIMD_ENSURE_INLINE inline f32 f32p_extract<3>(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 12)));
	}


#endif

	template<>
	LSIMD_ENSURE_INLINE inline f64 f64p_extract<0>(__m128d a)
	{
		return _mm_cvtsd_f64(a);
	}

	template<>
	LSIMD_ENSURE_INLINE inline f64 f64p_extract<1>(__m128d a)
	{
		return _mm_cvtsd_f64(_mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(a), 8)));
	}


} }

#endif /* SSE_PACK_MANIP_H_ */
