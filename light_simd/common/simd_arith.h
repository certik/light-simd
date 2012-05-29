/**
 * @file simd_arith.h
 *
 * Arithmetic calculations for SIMD vector
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SIMD_ARITH_H_
#define LSIMD_SIMD_ARITH_H_

#include "simd_vec.h"

#include <light_simd/sse/sse_arith.h>

namespace lsimd
{
	// Arithmetic functions

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> add(const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return add(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> sub(const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return sub(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> mul(const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return mul(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> div(const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return div(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> neg(const simd_vec<T, Kind> a)
	{
		return neg(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> abs(const simd_vec<T, Kind> a)
	{
		return abs(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> vmin(const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return vmin(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> vmax(const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return vmax(a.impl, b.impl);
	}


	// Operators overloading

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator + (const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return add(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator + (const simd_vec<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return add(a.impl, impl_t(b));
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator + (const T a, const simd_vec<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return add(impl_t(a), b.impl);
	}


	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator - (const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return sub(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator - (const simd_vec<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return sub(a.impl, impl_t(b));
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator - (const T a, const simd_vec<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return sub(impl_t(a), b.impl);
	}



	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator * (const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return mul(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator * (const simd_vec<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return mul(a.impl, impl_t(b));
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator * (const T a, const simd_vec<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return mul(impl_t(a), b.impl);
	}


	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator / (const simd_vec<T, Kind> a, const simd_vec<T, Kind> b)
	{
		return div(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator / (const simd_vec<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return div(a.impl, impl_t(b));
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator / (const T a, const simd_vec<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return div(impl_t(a), b.impl);
	}


	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> operator - (const simd_vec<T, Kind> a)
	{
		return neg(a.impl);
	}


	// Derived arithmetic functions

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> sqrt(const simd_vec<T, Kind> a)
	{
		return sqrt(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> rcp(const simd_vec<T, Kind> a)
	{
		return rcp(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> rsqrt(const simd_vec<T, Kind> a)
	{
		return rsqrt(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> sqr(const simd_vec<T, Kind> a)
	{
		return sqr(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> cube(const simd_vec<T, Kind> a)
	{
		return cube(a.impl);
	}


	// Rounding

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> floor(const simd_vec<T, Kind> a)
	{
		return floor(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_vec<T, Kind> ceil(const simd_vec<T, Kind> a)
	{
		return ceil(a.impl);
	}

}

#endif /* SIMD_ARITH_H_ */
