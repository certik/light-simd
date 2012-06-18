/**
 * @file simd_vec.h
 *
 * @brief SIMD-based fixed-size vector classes.
 *
 * @author Dahua Lin
 *
 * @copyright
 *
 * Copyright (C) 2012 Dahua Lin
 * 
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without restriction, 
 * including without limitation the rights to use, copy, modify, merge, 
 * publish, distribute, sublicense, and/or sell copies of the Software, 
 * and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
	/**
	 * @defgroup linalg_module Linear Algebra Module
	 *
	 * @brief Fixed-size vector and matrix classes and linear algebraic computation.
	 *
	 * This module comprises several classes to represent fixed size vectors 
	 * and matrices, and a set of functions to support linear algebraic computation.
	 *
	 */


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
		typedef simd_pack<T, Kind> pack_type;

		impl_type impl;

		// constructors

		LSIMD_ENSURE_INLINE simd_vec() { }

		LSIMD_ENSURE_INLINE simd_vec( zero_t ) : impl( zero_t() ) { }

		LSIMD_ENSURE_INLINE simd_vec( const impl_type& imp ) : impl(imp) { }

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

		LSIMD_ENSURE_INLINE simd_vec operator + (const simd_vec& rhs) const
		{
			return impl + rhs.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec operator - (const simd_vec& rhs) const
		{
			return impl - rhs.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec operator % (const simd_vec& rhs) const
		{
			return impl % rhs.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator += (const simd_vec& rhs)
		{
			impl += rhs.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator -= (const simd_vec& rhs)
		{
			impl -= rhs.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator %= (const simd_vec& rhs)
		{
			impl %= rhs.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE simd_vec operator * (const pack_type& s) const
		{
			return impl * s.impl;
		}

		LSIMD_ENSURE_INLINE simd_vec& operator *= (const pack_type& s)
		{
			impl *= s.impl;
			return *this;
		}

		// stats

		LSIMD_ENSURE_INLINE T sum() const
		{
			return impl.sum();
		}

		LSIMD_ENSURE_INLINE T dot(const simd_vec& rhs) const
		{
			return impl.dot(rhs.impl);
		}

	};

}

#endif /* SIMD_MATMUL_H_ */
