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
	 *  SSE pack classes
	 *
	 ********************************************/

	template<typename T> struct sse_pack;

	template<>
	struct sse_pack<f32>
	{
		typedef __m128 intern_type;
		static const int pack_width = 4;

		union
		{
			__m128 v;
			LSIMD_ALIGN_SSE f32 e[4];
		};


		LSIMD_ENSURE_INLINE
		sse_pack() { }

		LSIMD_ENSURE_INLINE
		sse_pack(const __m128 v_)
		: v(v_) { }

		LSIMD_ENSURE_INLINE
		sse_pack( zero_t )
		{
			v = _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f32 *__restrict__ a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f32 *__restrict__ a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}

		LSIMD_ENSURE_INLINE
		void set_zero()
		{
			v = _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE
		void set(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		LSIMD_ENSURE_INLINE
		void set(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *__restrict__ a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *__restrict__ a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *__restrict__ a, aligned_t) const
		{
			_mm_store_ps(a, v);
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *__restrict__ a, unaligned_t) const
		{
			_mm_storeu_ps(a, v);
		}

		LSIMD_ENSURE_INLINE
		intern_type intern() const
		{
			return v;
		}

		LSIMD_ENSURE_INLINE
		f32 get_e(const int i) const
		{
			return e[i];
		}

		LSIMD_ENSURE_INLINE
		void set_e(const int i, const f32 x)
		{
			e[i] = x;
		}

	};


	template<>
	struct sse_pack<f64>
	{
		typedef __m128d intern_type;
		static const int pack_width = 2;

		union
		{
			__m128d v;
			LSIMD_ALIGN_SSE f64 e[2];
		};


		LSIMD_ENSURE_INLINE
		sse_pack() { }

		LSIMD_ENSURE_INLINE
		sse_pack(const intern_type v_)
		: v(v_) { }

		LSIMD_ENSURE_INLINE
		sse_pack( zero_t )
		{
			v = _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f64 x)
		{
			v = _mm_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f64 e0, const f64 e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f64 *__restrict__ a, aligned_t)
		{
			v = _mm_load_pd(a);
		}

		LSIMD_ENSURE_INLINE
		sse_pack(const f64 *__restrict__ a, unaligned_t)
		{
			v = _mm_loadu_pd(a);
		}

		LSIMD_ENSURE_INLINE
		void set_zero()
		{
			v = _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE
		void set(const f64 x)
		{
			v = _mm_set1_pd(x);
		}

		LSIMD_ENSURE_INLINE
		void set(const f64 e0, const f64 e1)
		{
			v = _mm_set_pd(e1, e0);
		}

		LSIMD_ENSURE_INLINE
		void load(const f64 *__restrict__ a, aligned_t)
		{
			v = _mm_load_pd(a);
		}

		LSIMD_ENSURE_INLINE
		void load(const f64 *__restrict__ a, unaligned_t)
		{
			v = _mm_loadu_pd(a);
		}

		LSIMD_ENSURE_INLINE
		void store(f64 *__restrict__ a, aligned_t) const
		{
			_mm_store_pd(a, v);
		}

		LSIMD_ENSURE_INLINE
		void store(f64 *__restrict__ a, unaligned_t) const
		{
			_mm_storeu_pd(a, v);
		}

		LSIMD_ENSURE_INLINE
		intern_type intern() const
		{
			return v;
		}

		LSIMD_ENSURE_INLINE
		f64 get_e(const int i) const
		{
			return e[i];
		}

		LSIMD_ENSURE_INLINE
		void set_e(const int i, const f64 x)
		{
			e[i] = x;
		}
	};


	// typedefs

	typedef sse_pack<f32> sse_f32p;
	typedef sse_pack<f64> sse_f64p;

}

#endif /* SSE_BASE_H_ */
