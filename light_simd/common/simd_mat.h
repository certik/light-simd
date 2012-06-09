/**
 * @file simd_mat.h
 *
 * SIMD-based fixed-size matrix classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_MAT_H_
#define LSIMD_SIMD_MAT_H_

#include "simd_vec.h"
#include <light_simd/sse/sse_mat.h>

namespace lsimd
{
	template<typename T, int M, int N, typename Kind>
	struct simd_mat_traits;

	template<typename T, int M, int N>
	struct simd_mat_traits<T, M, N, sse_kind>
	{
		typedef sse_mat<T, M, N> impl_type;
	};


	template<typename T, int M, int N, typename Kind>
	struct simd_mat
	{
		typedef T value_type;
		typedef typename simd_mat_traits<T, M, N, Kind>::impl_type impl_type;
		impl_type impl;

		// constructors

		LSIMD_ENSURE_INLINE
		simd_mat( impl_type imp ) : impl(imp) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const f32 *x, aligned_t)
		: impl(x, aligned_t()) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const f32 *x, unaligned_t)
		: impl(x, unaligned_t()) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const f32 *x, int ldim, aligned_t)
		: impl(x, ldim, aligned_t()) { }

		LSIMD_ENSURE_INLINE
		simd_mat(const f32 *x, int ldim, unaligned_t)
		: impl(x, ldim, unaligned_t()) { }


		// load / store

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, aligned_t)
		{
			impl.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, unaligned_t)
		{
			impl.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, int ldim, aligned_t)
		{
			impl.load(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, int ldim, unaligned_t)
		{
			impl.load(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, aligned_t)
		{
			impl.load_trans(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, unaligned_t)
		{
			impl.load_trans(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, int ldim, aligned_t)
		{
			impl.load_trans(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, int ldim, unaligned_t)
		{
			impl.load_trans(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, aligned_t) const
		{
			impl.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, unaligned_t) const
		{
			impl.store(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, int ldim, aligned_t) const
		{
			impl.store(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, int ldim, unaligned_t) const
		{
			impl.store(x, ldim, unaligned_t());
		}


		// arithmetic

		LSIMD_ENSURE_INLINE
		simd_mat operator + (simd_mat r) const
		{
			return impl + r.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat operator - (simd_mat r) const
		{
			return impl - r.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat operator % (simd_mat r) const
		{
			return impl % r.impl;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator += (simd_mat r)
		{
			impl += r.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator -= (simd_mat r)
		{
			impl -= r.impl;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		simd_mat& operator %= (simd_mat r)
		{
			impl %= r.impl;
			return *this;
		}


		// linear algebra

		LSIMD_ENSURE_INLINE
		simd_vec<f32, M> operator * (simd_vec<f32, N> v) const
		{
			return impl * v.impl;
		}

	};

}

#endif /* SIMD_MAT_H_ */
