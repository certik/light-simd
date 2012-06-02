/**
 * @file sse_base.h
 *
 * The basic types for SSE packs
 *
 * All intrinsic headers are also included here.
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_BASE_H_
#define LSIMD_SSE_BASE_H_

#include <light_simd/common/common_base.h>

#include <xmmintrin.h> 	// for SSE
#include <emmintrin.h> 	// for SSE2

#ifdef LSIMD_HAS_SSE3
#include <pmmintrin.h> 	// for SSE3
#endif

#ifdef LSIMD_HAS_SSSE3
#include <tmmintrin.h> 	// for SSSE3
#endif

#ifdef LSIMD_HAS_SSE4_1
#include <smmintrin.h> 	// for SSE4 (include 4.1 & 4.2)
#endif

#define LSIMD_ALIGN_SSE LSIMD_ALIGN(16)


namespace lsimd {


	/********************************************
	 *
	 *  Useful constants
	 *
	 ********************************************/

	template<typename T> struct sse_const;

	template<> struct sse_const<f32>
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
			return _mm_castsi128_ps(_mm_set1_epi32(int(0x80000000)));
		}
	};


	template<> struct sse_const<f64>
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

		LSIMD_ENSURE_INLINE static __m128d halfs()
		{
			return _mm_set1_pd(0.5);
		}

		LSIMD_ENSURE_INLINE static __m128d sign_mask()
		{
			return _mm_castsi128_pd(_mm_set_epi32(int(0x80000000), 0, int(0x80000000), 0));
		}
	};


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


	/********************************************
	 *
	 *  SSE pack classes
	 *
	 ********************************************/

	template<typename T> struct sse_pack;

	template<>
	struct sse_pack<f32>
	{
		// types

		typedef f32 value_type;
		typedef __m128 intern_type;
		static const unsigned int pack_width = 4;

		union
		{
			__m128 v;
			LSIMD_ALIGN_SSE f32 e[4];
		};

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}


		// constructors

		LSIMD_ENSURE_INLINE sse_pack() { }

		LSIMD_ENSURE_INLINE sse_pack(const __m128 v_)
		: v(v_) { }

		LSIMD_ENSURE_INLINE sse_pack( zero_t )
		{
			v = _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE explicit sse_pack(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE sse_pack(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE sse_pack(const f32* a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		LSIMD_ENSURE_INLINE sse_pack(const f32* a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}


		// set, load, store

		LSIMD_ENSURE_INLINE void set_zero()
		{
			v = _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE void set(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE void set(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE void load(const f32* a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		LSIMD_ENSURE_INLINE void load(const f32* a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}

		LSIMD_ENSURE_INLINE void store(f32* a, aligned_t) const
		{
			_mm_store_ps(a, v);
		}

		LSIMD_ENSURE_INLINE void store(f32* a, unaligned_t) const
		{
			_mm_storeu_ps(a, v);
		}

		// entry access

		LSIMD_ENSURE_INLINE __m128 intern() const
		{
			return v;
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 extract() const
		{
			return entry_extractor<f32, I>::get(v);
		}


		// constants

		LSIMD_ENSURE_INLINE static sse_pack zeros()
		{
			return sse_const<f32>::zeros();
		}

		LSIMD_ENSURE_INLINE static sse_pack ones()
		{
			return sse_const<f32>::ones();
		}

		LSIMD_ENSURE_INLINE static sse_pack twos()
		{
			return sse_const<f32>::twos();
		}

		LSIMD_ENSURE_INLINE static sse_pack halfs()
		{
			return sse_const<f32>::halfs();
		}

	};


	template<>
	struct sse_pack<f64>
	{
		// types

		typedef f64 value_type;
		typedef __m128d intern_type;
		static const unsigned int pack_width = 2;

		union
		{
			__m128d v;
			LSIMD_ALIGN_SSE f64 e[2];
		};

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		// constructors

		LSIMD_ENSURE_INLINE sse_pack() { }

		LSIMD_ENSURE_INLINE sse_pack(const intern_type v_)
		: v(v_) { }

		LSIMD_ENSURE_INLINE sse_pack( zero_t )
		{
			v = _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE explicit sse_pack(const f64 x)
		{
			v = _mm_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE sse_pack(const f64 e0, const f64 e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		LSIMD_ENSURE_INLINE sse_pack(const f64* a, aligned_t)
		{
			v = _mm_load_pd(a);
		}

		LSIMD_ENSURE_INLINE sse_pack(const f64* a, unaligned_t)
		{
			v = _mm_loadu_pd(a);
		}


		// set, load, store

		LSIMD_ENSURE_INLINE void set_zero()
		{
			v = _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE void set(const f64 x)
		{
			v = _mm_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE void set(const f64 e0, const f64 e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		LSIMD_ENSURE_INLINE void load(const f64* a, aligned_t)
		{
			v = _mm_load_pd(a);
		}

		LSIMD_ENSURE_INLINE void load(const f64* a, unaligned_t)
		{
			v = _mm_loadu_pd(a);
		}

		LSIMD_ENSURE_INLINE void store(f64* a, aligned_t) const
		{
			_mm_store_pd(a, v);
		}

		LSIMD_ENSURE_INLINE void store(f64* a, unaligned_t) const
		{
			_mm_storeu_pd(a, v);
		}

		// entry access

		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return v;
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 extract() const
		{
			return entry_extractor<f64, I>::get(v);
		}

		// constants

		LSIMD_ENSURE_INLINE static sse_pack zeros()
		{
			return sse_const<f64>::zeros();
		}

		LSIMD_ENSURE_INLINE static sse_pack ones()
		{
			return sse_const<f64>::ones();
		}

		LSIMD_ENSURE_INLINE static sse_pack twos()
		{
			return sse_const<f64>::twos();
		}

		LSIMD_ENSURE_INLINE static sse_pack halfs()
		{
			return sse_const<f64>::halfs();
		}

	};


	// typedefs

	typedef sse_pack<f32> sse_f32pk;
	typedef sse_pack<f64> sse_f64pk;

}

#endif /* SSE_BASE_H_ */
