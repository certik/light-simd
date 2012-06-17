/**
 * @file simd_pack.h
 *
 * The common SIMD vector class
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_PACK_H_
#define LSIMD_SIMD_PACK_H_

#include <light_simd/common/common_base.h>
#include <light_simd/sse/sse_pack.h>

namespace lsimd
{

	/**
	 * SIMD type traits for SSE Kind
	 *
	 * @tparam T The scalar type
	 *
	 */
	template<typename T>
	struct simd<T, sse_kind>
	{
		/**
		 * The SIMD-pack implementation type
		 */
		typedef sse_pack<T> impl_type;

		/**
		 * The builtin representation type
		 *
		 * Here, intern_type is
		 * - __m128 (when T is f32)
		 * - __m128d (when T is f64)
		 * - __m128i (when T is an integer type)
		 *
		 */
		typedef typename impl_type::intern_type intern_type;

		/**
		 * The number of scalars in each pack
		 *
		 * @remark
		 *  pack_width = 4 (when T is f32)
		 *  pack_width = 2 (when T is f64)
		 */
		static const unsigned int pack_width = impl_type::pack_width;

		/**
		 * The number of SSE 128-bit registers in a processing core
		 */
		static const unsigned int max_registers = 8;
	};


	/**
	 * The class to represent an SIMD pack, a small vector of
	 * numbers that can fit in an SIMD register
	 *
	 * @tparam T the scalar type
	 * @tparam Kind the kind of SIMD instructions (e.g. \ref sse_kind)
	 *
	 * @remark
	 * 	This class is just a wrapper of an implementing class that
	 * 	depends on both T and Kind.
	 * 	For example, when Kind is \ref sse_kind, it wraps the
	 * 	template class sse_pack
	 */
	template<typename T, typename Kind>
	struct simd_pack
	{
		/**
		 * The corresponding scalar type
		 */
		typedef T value_type;

		/**
		 * The class that actually implements the SIMD pack
		 */
		typedef typename simd<T, Kind>::impl_type impl_type;

		/**
		 * The compiler builtin representation type
		 */
		typedef typename simd<T, Kind>::intern_type intern_type;

		/**
		 * The number of scalars in each pack
		 */
		static const unsigned int pack_width = simd<T, Kind>::pack_width;


		/**
		 * The embedded object of the implementing class
		 */
		impl_type impl;


		/**
		 * Default constructor
		 *
		 * The entries in the pack are left uninitialized
		 */
		LSIMD_ENSURE_INLINE simd_pack() { }

		/**
		 * Constructs a pack using the actual implementation object
		 *
		 * @param imp the object that actually implements the pack
		 *            representation
		 */
		LSIMD_ENSURE_INLINE simd_pack(const impl_type& imp)
		: impl(imp) { }

		/**
		 * Constructs a pack using builtin representation
		 *
		 * @param v the builtin representation of a pack
		 */
		LSIMD_ENSURE_INLINE simd_pack(intern_type v)
		: impl(v) { }

		/**
		 * Constructs a pack initialized as all zeros
		 */
		LSIMD_ENSURE_INLINE simd_pack( zero_t )
		: impl(zero_t()) { }

		/**
		 * Constructs a pack with all scalar entries initialized
		 * to a given value
		 *
		 * @param x the value used to initialize the pack
		 */
		LSIMD_ENSURE_INLINE explicit simd_pack(const T x)
		: impl(x) { }

		/**
		 * Constructs a pack by loading the entry values from
		 * a properly aligned memory address
		 *
		 * @param a the memory address from which values are 
		 *          loaded
		 */
		LSIMD_ENSURE_INLINE simd_pack(const T* a, aligned_t)
		: impl(a, aligned_t()) { }

		/**
		 * Constructs a pack by loading the entry values from
		 * an memory address that is not necessarily aligned
		 *
		 * @param a the memory address from which values are 
		 *          loaded
		 */
		LSIMD_ENSURE_INLINE simd_pack(const T* a, unaligned_t)
		: impl(a, unaligned_t()) { }


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
		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return impl.intern();
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
			impl.set_zero();
		}

		/**
		 * Set all scalar entries to a given value
		 *
		 * @param x the value to be set to all entries
		 */
		LSIMD_ENSURE_INLINE void set(const T x)
		{
			impl.set(x);
		}

		/**
		 * Load all entries from an aligned memory address
		 *
		 * @param a the memory address from which the values are loaded
		 */
		LSIMD_ENSURE_INLINE void load(const T* a, aligned_t)
		{
			impl.load(a, aligned_t());
		}

		/**
		 * Load all entries from an memory address that is not
		 * necessarily aligned
		 *
		 * @param a the memory address from which the values
		 *          are loaded
		 */
		LSIMD_ENSURE_INLINE void load(const T* a, unaligned_t)
		{
			impl.load(a, unaligned_t());
		}

		/**
		 * Store all entries to a properly aligned memory
		 * address
		 *
		 * @param a the memory address from which the values
		 *          are stored
		 */
		LSIMD_ENSURE_INLINE void store(T* a, aligned_t) const
		{
			impl.store(a, aligned_t());
		}

		/**
		 * Store all entries to the memory address that is not
		 * necessarily aligned
		 *
		 * @param a the memory address from which the values
		 *          are stored
		 */
		LSIMD_ENSURE_INLINE void store(T* a, unaligned_t) const
		{
			impl.store(a, unaligned_t());
		}

		/**
		 * Load a subset of entries from a given memory address
		 *
		 * @tparam I the number of entries to be loaded.
		 *           The value of I must be within
		 *           [1, \ref pack_width - 1]
		 *
		 * @param a the memory address from which the values
		 *          are loaded
		 *
		 * @remark the loaded values are set to the lower-end of
		 *         the pack, while the entries at higher-end are
		 *         set to zeros
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_load(const T *a)
		{
			impl.partial_load<I>(a);
		}

		/**
		 * Store a subset of entries to a given memory address
		 *
		 * @tparam I the number of entries to be stored.
		 *           The value of I must be within
		 *           [1, \ref pack_width - 1].
		 *
		 * @param a the memory address
		 *
		 * @remark This method stores the first I values at
		 *         the lower end of the pack
		 */
		template<int I>
		LSIMD_ENSURE_INLINE void partial_store(T *a) const
		{
			impl.partial_store<I>(a);
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
		 *         end
		 *
		 * @remark To extract the scalar at arbitrary position,
		 *         one may use another member function \ref extract.
		 */
		LSIMD_ENSURE_INLINE T to_scalar() const
		{
			return impl.to_scalar();
		}

		/**
		 * Extract the entry at given position
		 *
		 * @tparam I the entry position.
		 *           The value of I must be within
		 *           [0, \ref pack_width - 1].
		 *
		 * @return the I-th entry of this pack.
		 *
		 * @remark extract<0>() is equivalent to to_scalar().
		 *
		 * @see to_scalar
		 */
		template<int I>
		LSIMD_ENSURE_INLINE T extract() const
		{
			return impl.extract<I>();
		}

		/**
		 * Broadcast the entry at a given position
		 *
		 * @tparam I the position of the entry to be broadcasted.
		 *           The value of I must be within
		 *           [0, \ref pack_width - 1].
		 *
		 * @return a pack whose entries are all equal to
		 *         the I-th entry of this pack
		 */
		template<int I>
		LSIMD_ENSURE_INLINE simd_pack bsx() const
		{
			return impl.bsx<I>();
		}

		/**
		 * Shift entries towards the low end
		 * (with zeros shift-in from the high end)
		 *
		 * @tparam I the distance to shift (in terms of the number
		 *           of scalars).
		 *           The value of I must be within
		 *           [0, \ref pack_width].
		 *
		 * @return The shifted pack, of which the k-th
		 *         entry equals the (k+I)-th entry of this pack,
		 *         when k < \ref pack_width - I, or zero
		 *         otherwise.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE simd_pack shift_front() const
		{
			return impl.shift_front<I>();
		}

		/**
		 * Shift entries towards the high end
		 * (with zeros shift-in from the low end)
		 *
		 * @tparam I the distance to shift (in terms of the number
		 *           of scalars).
		 *           The value of I must be within
		 *           [0, \ref pack_width].
		 *
		 * @return The shifted pack, of which the k-th
		 *         entry equals the (k-I)-th entry of this pack,
		 *         when k >= I, or zero otherwise.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE simd_pack shift_back() const
		{
			return impl.shift_back<I>();
		}

		///@}


		/**
		 * @name Statistics Methods
		 *
		 * The member functions to evaluate statistics over entries
		 */
		///@{

		/**
		 * Evaluate the sum of all entries
		 *
		 * @return the sum of all entries
		 */
		LSIMD_ENSURE_INLINE T sum() const
		{
			return impl.sum();
		}

		/**
		 * Evaluate the sum of a subset of entries
		 *
		 * @tparam I the number of entries to be used
		 *
		 * @return the sum of first I entries from the lowest end.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE T partial_sum() const
		{
			return impl.partial_sum<I>();
		}

		/**
		 * Evaluate the maximum of a subset of entries
		 *
		 * @tparam I the number of entries to be used
		 *
		 * @return the maximum of first I entries from the lowest end.
		 */
		LSIMD_ENSURE_INLINE T (max)() const
		{
			return (impl.max)();
		}

		/**
		 * Evaluate the maximum of a subset of entries
		 *
		 * @tparam I the number of entries to be used
		 *
		 * @return the maximum of first I entries from the lowest end.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE T partial_max() const
		{
			return impl.partial_max<I>();
		}

		/**
		 * Evaluate the minimum of a subset of entries
		 *
		 * @tparam I the number of entries to be used
		 *
		 * @return the minimum of first I entries from the lowest end.
		 */
		LSIMD_ENSURE_INLINE T (min)() const
		{
			return (impl.min)();
		}

		/**
		 * Evaluate the minimum of a subset of entries
		 *
		 * @tparam I the number of entries to be used
		 *
		 * @return the minimum of first I entries from the lowest end.
		 */
		template<int I>
		LSIMD_ENSURE_INLINE T partial_min() const
		{
			return impl.partial_min<I>();
		}

		///@}


		/**
		 * Constant Generating Methods
		 *
		 * The static member functions to generate packs comprised
		 * of some common useful values
		 */
		///@{

		/**
		 * Get an all-zero pack
		 *
		 * @returns a pack with all entries being zeros
		 */
		LSIMD_ENSURE_INLINE static simd_pack zeros()
		{
			return impl_type::zeros();
		}

		/**
		 * Get an all-one pack
		 *
		 * @returns a pack with all entries being ones
		 */
		LSIMD_ENSURE_INLINE static simd_pack ones()
		{
			return impl_type::ones();
		}

		/**
		 * Get an all-two pack
		 *
		 * @returns a pack with all entries being twos
		 */
		LSIMD_ENSURE_INLINE static simd_pack twos()
		{
			return impl_type::twos();
		}

		/**
		 * Get an all-half pack
		 *
		 * @returns a pack with all entries being 0.5
		 */
		LSIMD_ENSURE_INLINE static simd_pack halfs()
		{
			return impl_type::halfs();
		}

		///@}

	};

}

#endif /* SIMD_BASE_H_ */









