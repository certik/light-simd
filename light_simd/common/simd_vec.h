/**
 * @file simd_vec.h
 *
 * SIMD-based fixed-length vector classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_VEC_H_
#define LSIMD_SIMD_VEC_H_

#include "simd_pack.h"
#include <light_simd/sse/sse_vec.h>

namespace lsimd
{
	template<typename T, int N, typename Kind>
	struct simd_vec_traits;

	template<typename T, int N>
	struct simd_vec_traits<T, N, sse_kind>
	{
		typedef sse_vec<T, N> impl_type;
	};


	template<typename T, int N, typename Kind>
	struct simd_vec
	{
		typedef T value_type;
		typedef typename simd_vec_traits<T, N, Kind>::impl_type impl_type;

		impl_type impl;

		// constructors

		LSIMD_ENSURE_INLINE simd_vec() { }

		LSIMD_ENSURE_INLINE simd_vec( zero_t ) : impl( zero_t() ) { }

		LSIMD_ENSURE_INLINE simd_vec( impl_type imp ) : impl(imp) { }

		LSIMD_ENSURE_INLINE simd_vec( const T *x, aligned_t )
		: impl(x, aligned_t()) { }

		LSIMD_ENSURE_INLINE simd_vec( const T *x, unaligned_t )
		: impl(x, unaligned_t()) { }

		// load / store

		LSIMD_ENSURE_INLINE void load( const T *x, aligned_t )
		{
			impl.load(x, aligned_t() );
		}

		LSIMD_ENSURE_INLINE void load( const T *x, unaligned_t )
		{
			impl.load(x, unaligned_t() );
		}

		LSIMD_ENSURE_INLINE void store( T *x, aligned_t ) const
		{
			impl.store(x, aligned_t() );
		}

		LSIMD_ENSURE_INLINE void store( T *x, unaligned_t ) const
		{
			impl.store(x, unaligned_t() );
		}

		// arithmetic

		LSIMD_ENSURE_INLINE simd_vec operator + (simd_vec rhs) const
		{
			return impl + rhs.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec operator - (simd_vec rhs) const
		{
			return impl - rhs.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec operator * (simd_vec rhs) const
		{
			return impl * rhs.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator += (simd_vec rhs)
		{
			impl += rhs.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator -= (simd_vec rhs)
		{
			impl -= rhs.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator *= (simd_vec rhs)
		{
			impl *= rhs.impl;
			return *this;
		}

		// stats

		LSIMD_ENSURE_INLINE T sum() const
		{
			return impl.sum();
		}

		LSIMD_ENSURE_INLINE T dot(simd_vec rhs) const
		{
			return impl.dot(rhs.impl);
		}

	};

}

#endif /* SIMD_MATMUL_H_ */
