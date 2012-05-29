/**
 * @file simd_vec.h
 *
 * The common SIMD vector class
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_VEC_H_
#define LSIMD_SIMD_VEC_H_

#include <light_simd/common/common_base.h>
#include <light_simd/sse/sse_base.h>

namespace lsimd
{

	// forward declarations

	template<typename T, typename Kind> struct simd;

	template<typename T, typename Kind> struct simd_vec;


	/******************************************************
	 *
	 *  Kind-based type dispatch
	 *
	 *  Currently, only SSE is supported. Supports for
	 *  other kinds (e.g. AVX) will be added in future.
	 *
	 ******************************************************/

	struct sse_kind { };

	typedef sse_kind default_simd_kind;

	template<typename T>
	struct simd<T, sse_kind>
	{
		typedef sse_vec<T> impl_type;
		typedef typename impl_type::intern_type intern_type;
		static const unsigned int pack_width = impl_type::pack_width;
	};


	/******************************************************
	 *
	 *  The SIMD vector class
	 *
	 *  This is a light-weight zero-overhead wrapper
	 *  of a specific SIMD implementation
	 *
	 ******************************************************/

	template<typename T, typename Kind=default_simd_kind>
	struct simd_vec
	{
		typedef T value_type;

		typedef typename simd<T, Kind>::impl_type impl_type;
		typedef typename simd<T, Kind>::intern_type intern_type;
		static const unsigned int pack_width = simd<T, Kind>::pack_width;

		// types

		impl_type impl;

		LSIMD_ENSURE_INLINE unsigned int width() const
		{
			return pack_width;
		}

		LSIMD_ENSURE_INLINE intern_type intern() const
		{
			return impl.intern();
		}

		// constructors

		LSIMD_ENSURE_INLINE simd_vec() { }

		LSIMD_ENSURE_INLINE simd_vec(const impl_type imp)
		: impl(imp) { }

		LSIMD_ENSURE_INLINE simd_vec(const intern_type v)
		: impl(v) { }

		LSIMD_ENSURE_INLINE explicit simd_vec( zero_t )
		: impl(zero_t()) { }

		LSIMD_ENSURE_INLINE explicit simd_vec(const T x)
		: impl(x) { }

		LSIMD_ENSURE_INLINE simd_vec(const T* a, aligned_t)
		: impl(a, aligned_t()) { }

		LSIMD_ENSURE_INLINE simd_vec(const T* a, unaligned_t)
		: impl(a, unaligned_t()) { }


		// set, load, store

		LSIMD_ENSURE_INLINE void set_zero()
		{
			impl.set_zero();
		}

		LSIMD_ENSURE_INLINE void set(const T x)
		{
			impl.set(x);
		}

		LSIMD_ENSURE_INLINE void load(const T* a, aligned_t)
		{
			impl.load(a, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const T* a, unaligned_t)
		{
			impl.load(a, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(T* a, aligned_t) const
		{
			impl.store(a, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(T* a, unaligned_t) const
		{
			impl.store(a, unaligned_t());
		}


		// constants

		LSIMD_ENSURE_INLINE static simd_vec zeros()
		{
			return impl_type::zeros();
		}

		LSIMD_ENSURE_INLINE static simd_vec ones()
		{
			return impl_type::ones();
		}

		LSIMD_ENSURE_INLINE static simd_vec twos()
		{
			return impl_type::twos();
		}

		LSIMD_ENSURE_INLINE static simd_vec halfs()
		{
			return impl_type::halfs();
		}

	};

}

#endif /* SIMD_BASE_H_ */









