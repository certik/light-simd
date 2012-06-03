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

namespace lsimd {  namespace detail {


	/********************************************
	 *
	 *  Entry extraction
	 *
	 ********************************************/

#if defined(LSIMD_HAS_SSE4_1)

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e0(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 0);
		return r;
	}

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e1(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 1);
		return r;
	}

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e2(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 2);
		return r;
	}

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e3(__m128 a)
	{
		f32 r;
		_MM_EXTRACT_FLOAT(r, a, 3);
		return r;
	}

#else

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e0(__m128 a)
	{
		return _mm_cvtss_f32(a);
	}

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e1(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4)));
	}

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e2(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 8)));
	}

	LSIMD_ENSURE_INLINE inline f32 extract_f32p_e3(__m128 a)
	{
		return _mm_cvtss_f32(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 12)));
	}


#endif

	LSIMD_ENSURE_INLINE inline f64 extract_f64p_e0(__m128d a)
	{
		return _mm_cvtsd_f64(a);
	}

	LSIMD_ENSURE_INLINE inline f64 extract_f64p_e1(__m128d a)
	{
		return _mm_cvtsd_f64(_mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(a), 8)));
	}


	template<typename T, int I> struct entry_extractor;

	template<> struct entry_extractor<f32, 0>
	{
		LSIMD_ENSURE_INLINE static f32 get(__m128 a) { return extract_f32p_e0(a); }
	};

	template<> struct entry_extractor<f32, 1>
	{
		LSIMD_ENSURE_INLINE static f32 get(__m128 a) { return extract_f32p_e1(a); }
	};

	template<> struct entry_extractor<f32, 2>
	{
		LSIMD_ENSURE_INLINE static f32 get(__m128 a) { return extract_f32p_e2(a); }
	};

	template<> struct entry_extractor<f32, 3>
	{
		LSIMD_ENSURE_INLINE static f32 get(__m128 a) { return extract_f32p_e3(a); }
	};

	template<> struct entry_extractor<f64, 0>
	{
		LSIMD_ENSURE_INLINE static f64 get(__m128d a) { return extract_f64p_e0(a); }
	};

	template<> struct entry_extractor<f64, 1>
	{
		LSIMD_ENSURE_INLINE static f64 get(__m128d a) { return extract_f64p_e1(a); }
	};

} }

#endif /* SSE_PACK_MANIP_H_ */
