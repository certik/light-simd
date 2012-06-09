/**
 * @file sse_pack.h
 *
 * The SSE pack class
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_PACK_H_
#define LSIMD_SSE_PACK_H_

#include "details/sse_pack_bits.h"

namespace lsimd
{

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

		LSIMD_ENSURE_INLINE __m128 intern() const
		{
			return v;
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

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f32 *a)
		{
			v = sse::partial_load<I>(a);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f32 *a) const
		{
			sse::partial_store<I>(a, v);
		}


		// entry manipulation

		LSIMD_ENSURE_INLINE f32 to_scalar() const
		{
			return _mm_cvtss_f32(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 extract() const
		{
			return sse::f32p_extract<I>(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_pack broadcast() const
		{
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(I, I, I, I));
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_front() const
		{
			return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), (I << 2)));
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_back() const
		{
			return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), (I << 2)));
		}


		// special entry manipulation

		template<int I0, int I1, int I2, int I3>
		LSIMD_ENSURE_INLINE sse_pack swizzle() const
		{
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(I3, I2, I1, I0));
		}

		LSIMD_ENSURE_INLINE sse_pack dup_low() const // [0, 1, 0, 1]
		{
			return sse::f32_dup_low(v);
		}

		LSIMD_ENSURE_INLINE sse_pack dup_high() const // [2, 3, 2, 3]
		{
			return sse::f32_dup_high(v);
		}

		LSIMD_ENSURE_INLINE sse_pack dup2_low() const // [0, 0, 2, 2]
		{
			return sse::f32_dup2_low(v);
		}

		LSIMD_ENSURE_INLINE sse_pack dup2_high() const // [1, 1, 3, 3]
		{
			return sse::f32_dup2_high(v);
		}


		// statistics

		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return sse::f32_sum(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_sum() const
		{
			return sse::f32_partial_sum<I>(v);
		}

		LSIMD_ENSURE_INLINE f32 max() const
		{
			return sse::f32_max(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_max() const
		{
			return sse::f32_partial_max<I>(v);
		}

		LSIMD_ENSURE_INLINE f32 min() const
		{
			return sse::f32_min(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_min() const
		{
			return sse::f32_partial_min<I>(v);
		}


		// constants

		LSIMD_ENSURE_INLINE static sse_pack zeros()
		{
			return _mm_setzero_ps();
		}

		LSIMD_ENSURE_INLINE static sse_pack ones()
		{
			return _mm_set1_ps(1.f);
		}

		LSIMD_ENSURE_INLINE static sse_pack twos()
		{
			return _mm_set1_ps(2.f);
		}

		LSIMD_ENSURE_INLINE static sse_pack halfs()
		{
			return _mm_set1_ps(0.5f);
		}

		// debug

		LSIMD_ENSURE_INLINE bool test_equal(f32 e0, f32 e1, f32 e2, f32 e3) const
		{
			return e[0] == e0 && e[1] == e1 && e[2] == e2 && e[3] == e3;
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]); std::printf(", ");
			std::printf(fmt, e[2]); std::printf(", ");
			std::printf(fmt, e[3]);
			std::printf(")");
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

		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return v;
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

		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f64 *a)
		{
			v = sse::partial_load<I>(a);
		}

		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f64 *a) const
		{
			sse::partial_store<I>(a, v);
		}

		// entry manipulation

		LSIMD_ENSURE_INLINE f64 to_scalar() const
		{
			return _mm_cvtsd_f64(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 extract() const
		{
			return sse::f64p_extract<I>(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_pack broadcast() const
		{
			return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(I, I));
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_front() const
		{
			return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), (I << 3)));
		}

		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_back() const
		{
			return _mm_castsi128_pd(_mm_slli_si128(_mm_castpd_si128(v), (I << 3)));
		}


		// special entry manipulation

		template<int I0, int I1>
		LSIMD_ENSURE_INLINE sse_pack swizzle() const
		{
			return _mm_shuffle_pd(v, v, _MM_SHUFFLE2(I1, I0));
		}

		LSIMD_ENSURE_INLINE sse_pack dup_low() const // [0, 0]
		{
			return sse::f64_dup_low(v);
		}

		LSIMD_ENSURE_INLINE sse_pack dup_high() const // [1, 1]
		{
			return sse::f64_dup_high(v);
		}


		// statistics

		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return sse::f64_sum(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_sum() const
		{
			return sse::f64_partial_sum<I>(v);
		}

		LSIMD_ENSURE_INLINE f64 max() const
		{
			return sse::f64_max(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_max() const
		{
			return sse::f64_partial_max<I>(v);
		}

		LSIMD_ENSURE_INLINE f64 min() const
		{
			return sse::f64_min(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_min() const
		{
			return sse::f64_partial_min<I>(v);
		}


		// constants

		LSIMD_ENSURE_INLINE static sse_pack zeros()
		{
			return _mm_setzero_pd();
		}

		LSIMD_ENSURE_INLINE static sse_pack ones()
		{
			return _mm_set1_pd(1.0);
		}

		LSIMD_ENSURE_INLINE static sse_pack twos()
		{
			return _mm_set1_pd(2.0);
		}

		LSIMD_ENSURE_INLINE static sse_pack halfs()
		{
			return _mm_set1_pd(0.5);
		}

		// debug

		LSIMD_ENSURE_INLINE bool test_equal(f64 e0, f64 e1) const
		{
			return e[0] == e0 && e[1] == e1;
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("(");
			std::printf(fmt, e[0]); std::printf(", ");
			std::printf(fmt, e[1]);
			std::printf(")");
		}

	};


	// typedefs

	typedef sse_pack<f32> sse_f32pk;
	typedef sse_pack<f64> sse_f64pk;


	// Some auxiliary routines

	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk shuffle(sse_f32pk a, sse_f32pk b)
	{
		return _mm_shuffle_ps(a.v, b.v, _MM_SHUFFLE(I3, I2, I1, I0));
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline sse_f64pk shuffle(sse_f64pk a, sse_f64pk b)
	{
		return _mm_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(I1, I0));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk merge_low(sse_f32pk a, sse_f32pk b)
	{
		return _mm_movelh_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk merge_high(sse_f32pk a, sse_f32pk b)
	{
		return _mm_movehl_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk merge_low(sse_f64pk a, sse_f64pk b)
	{
		return _mm_castps_pd(
				_mm_movelh_ps(_mm_castpd_ps(a.v), _mm_castpd_ps(b.v)));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk merge_high(sse_f64pk a, sse_f64pk b)
	{
		return _mm_castps_pd(
				_mm_movehl_ps(_mm_castpd_ps(a.v), _mm_castpd_ps(b.v)));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk unpack_low(sse_f32pk a, sse_f32pk b)
	{
		return _mm_unpacklo_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk unpack_high(sse_f32pk a, sse_f32pk b)
	{
		return _mm_unpackhi_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk unpack_low(sse_f64pk a, sse_f64pk b)
	{
		return _mm_unpacklo_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk unpack_high(sse_f64pk a, sse_f64pk b)
	{
		return _mm_unpackhi_pd(a.v, b.v);
	}


}

#endif /* SSE_PACK_H_ */
