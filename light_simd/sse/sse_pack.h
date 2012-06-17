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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#endif

namespace lsimd
{

	template<typename T> struct sse_pack;

	/**
	 * The class to represent an SSE pack comprised of
	 * 4 single-precision floating-point numbers (f32)
	 */
	template<>
	struct sse_pack<f32>
	{

		/**
		 * The scalar value type
		 */
		typedef f32 value_type;

		/**
		 * The builtin representation type
		 */
		typedef __m128 intern_type;

		/**
		 * The number of scalars in a pack
		 */
		static const unsigned int pack_width = 4;

		union
		{
			__m128 v;  /**< The builtin representation  */
			LSIMD_ALIGN_SSE f32 e[4];  /**< The representation in an array of scalars */
		};


		// constructors

		/**
		 * Default constructor
		 *
		 * The entries in this pack are left uninitialized
		 */
		LSIMD_ENSURE_INLINE sse_pack() { }

		/**
		 * Constructs a pack using builtin representation
		 *
		 * @param v_ the builtin representation of a pack
		 */
		LSIMD_ENSURE_INLINE sse_pack(const __m128 v_)
		: v(v_) { }

		/**
		 * Constructs a pack with all entries initialized to zeros
		 */
		LSIMD_ENSURE_INLINE sse_pack( zero_t )
		{
			v = _mm_setzero_ps();
		}

		/**
		 * Constructs a pack with all entries initialized
		 * to a given value
		 *
		 * @param x the value used to initialize the pack
		 */
		LSIMD_ENSURE_INLINE explicit sse_pack(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		/**
		 * Constructs a pack with given values
		 *
		 * @param e0 the 0-th entry value (the lowest-end)
		 * @param e1 the 1st entry value
		 * @param e2 the 2nd entry value
		 * @param e3 the 3rd entry value (the highest-end)
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		/**
		 * Constructs a pack by loading the entry values from
		 * a properly aligned memory address
		 *
		 * @param a the memory address from which values are
		 *          loaded
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f32* a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		/**
		 * Constructs a pack by loading the entry values from
		 * an memory address that is not necessarily aligned
		 *
		 * @param a the memory address from which values are
		 *          loaded
		 */
		LSIMD_ENSURE_INLINE sse_pack(const f32* a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}


		/**
		 * @name Basic Information Retrieval Methods
		 *
		 * The member functions to get basic information about the SIMD pack.
		 */
		///@{


		/**
		 * Get the pack width (the number of scalars in a pack)
		 *
		 * @return the value of \ref pack_width
		 */
		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		/**
		 * Get the builtin representation
		 *
		 * @return a copy of the builtin representation variable
		 */
		LSIMD_ENSURE_INLINE __m128 intern() const
		{
			return v;
		}

		///@}


		/**
		 * @name Import and Export Methods
		 *
		 * The member functions to set, load and store entry values.
		 */
		///@{

		/**
		 * Set all scalar entries to zeros
		 */
		LSIMD_ENSURE_INLINE void set_zero()
		{
			v = _mm_setzero_ps();
		}

		/**
		 * Set all scalar entries to a given value
		 *
		 * @param x the value to be set to all entries
		 */
		LSIMD_ENSURE_INLINE void set(const f32 x)
		{
			v = _mm_set1_ps(x);
		}

		/**
		 * Set given values to the entries
		 *
		 * @param e0 the value to be set to the 0-th entry (lowest end)
		 * @param e1 the value to be set to the 1st entry
		 * @param e2 the value to be set to the 2nd entry
		 * @param e3 the value to be set to the 3rd entry (highest end)
		 */
		LSIMD_ENSURE_INLINE void set(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			v = _mm_set_ps(e3, e2, e1, e0);
		}

		/**
		 * Load all entries from an aligned memory address
		 *
		 * @param a the memory address from which the values are loaded
		 */
		LSIMD_ENSURE_INLINE void load(const f32* a, aligned_t)
		{
			v = _mm_load_ps(a);
		}

		/**
		 * Load all entries from an memory address that is not
		 * necessarily aligned
		 *
		 * @param a the memory address from which the values
		 *          are loaded
		 */
		LSIMD_ENSURE_INLINE void load(const f32* a, unaligned_t)
		{
			v = _mm_loadu_ps(a);
		}

		/**
		 * Store all entries to a properly aligned memory
		 * address
		 *
		 * @param a the memory address from which the values
		 *          are stored
		 */
		LSIMD_ENSURE_INLINE void store(f32* a, aligned_t) const
		{
			_mm_store_ps(a, v);
		}

		/**
		 * Store all entries to the memory address that is not
		 * necessarily aligned
		 *
		 * @param a the memory address from which the values
		 *          are stored
		 */
		LSIMD_ENSURE_INLINE void store(f32* a, unaligned_t) const
		{
			_mm_storeu_ps(a, v);
		}

		/**
		 * Load a subset of entries from a given memory address
		 *
		 * @tparam I the number of entries to be loaded.
		 *           The value of I must be either 1, 2, or 3
		 *
		 * @param a the memory address from which the values
		 *          are loaded
		 *
		 * @remark the loaded values are set to the lower-end of
		 *         the pack, while the entries at higher-end are
		 *         set to zeros
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const f32 *a)
		{
			v = sse::partial_load<I>(a);
		}

		/**
		 * Store a subset of entries to a given memory address
		 *
		 * @tparam I the number of entries to be stored.
		 *           The value of I must be either 1, 2, or 3.
		 *
		 * @param a the memory address
		 *
		 * @remark This method stores the first I values at
		 *         the lower end of the pack
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(f32 *a) const
		{
			sse::partial_store<I>(a, v);
		}

		///@}

		/**
		 * @name Entry Manipulation Methods
		 *
		 * The member functions to extract entries or switch their positions
		 */
		///@{


		/**
		 * Extract the entry at lowest end
		 *
		 * @return the scalar value of the entry at lowest
		 *         end (i.e. \ref e[0])
		 *
		 * @remark To extract the scalar at arbitrary position,
		 *         one may use another member function \ref extract.
		 */
		LSIMD_ENSURE_INLINE f32 to_scalar() const
		{
			return _mm_cvtss_f32(v);
		}

		/**
		 * Extract the entry at given position
		 *
		 * @tparam I the entry position.
		 *           The value of I must be within [0, 3].
		 *
		 * @return the I-th entry of this pack.
		 *
		 * @remark extract<0>() is equivalent to to_scalar().
		 *
		 * @see to_scalar
		 */
		template<int I>
		LSIMD_ENSURE_INLINE f32 extract() const
		{
			return sse::f32p_extract<I>(v);
		}

		/**
		 * Broadcast the entry at a given position
		 *
		 * @tparam I the position of the entry to be broadcasted.
		 *           The value of I must be within [0, 3].
		 *
		 * @return a pack whose entries are all equal to
		 *         the I-th entry of this pack
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack bsx() const
		{
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(I, I, I, I));
		}

		/**
		 * Shift entries towards the low end
		 * (with zeros shift-in from the high end)
		 *
		 * @tparam I the distance to shift (in terms of the number
		 *           of scalars).
		 *           The value of I must be within [0, 4].
		 *
		 * @return The shifted pack, of which the k-th
		 *         entry equals the (k+I)-th entry of this pack,
		 *         when k < 4 - I, or zero otherwise.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_front() const
		{
			return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), (I << 2)));
		}

		/**
		 * Shift entries towards the high end
		 * (with zeros shift-in from the low end)
		 *
		 * @tparam I the distance to shift (in terms of the number
		 *           of scalars).
		 *           The value of I must be within [0, 4].
		 *
		 * @return The shifted pack, of which the k-th
		 *         entry equals the (k-I)-th entry of this pack,
		 *         when k >= I, or zero otherwise.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE sse_pack shift_back() const
		{
			return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), (I << 2)));
		}

		///@}


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

		LSIMD_ENSURE_INLINE f32 (max)() const
		{
			return sse::f32_max(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f32 partial_max() const
		{
			return sse::f32_partial_max<I>(v);
		}

		LSIMD_ENSURE_INLINE f32 (min)() const
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

		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return test_equal(r[0], r[1], r[2], r[3]);
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


	/**
	 * The class to represent an SSE pack comprised of
	 * 4 single-precision floating-point numbers (f32)
	 */
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
		LSIMD_ENSURE_INLINE sse_pack bsx() const
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

		LSIMD_ENSURE_INLINE f64 (max)() const
		{
			return sse::f64_max(v);
		}

		template<int I>
		LSIMD_ENSURE_INLINE f64 partial_max() const
		{
			return sse::f64_partial_max<I>(v);
		}

		LSIMD_ENSURE_INLINE f64 (min)() const
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

		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return test_equal(r[0], r[1]);
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

	/**
	 * A short name for sse_pack<f32>
	 */
	typedef sse_pack<f32> sse_f32pk;

	/**
	 * A short name for sse_pack<f64>
	 */
	typedef sse_pack<f64> sse_f64pk;


	// Some auxiliary routines

	template<int I0, int I1, int I2, int I3>
	LSIMD_ENSURE_INLINE
	inline sse_f32pk shuffle(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_shuffle_ps(a.v, b.v, _MM_SHUFFLE(I3, I2, I1, I0));
	}

	template<int I0, int I1>
	LSIMD_ENSURE_INLINE
	inline sse_f64pk shuffle(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(I1, I0));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk merge_low(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_movelh_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk merge_high(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_movehl_ps(b.v, a.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk unpack_low(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_unpacklo_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk unpack_high(const sse_f32pk& a, const sse_f32pk& b)
	{
		return _mm_unpackhi_ps(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk unpack_low(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_unpacklo_pd(a.v, b.v);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk unpack_high(const sse_f64pk& a, const sse_f64pk& b)
	{
		return _mm_unpackhi_pd(a.v, b.v);
	}


}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif /* SSE_PACK_H_ */
