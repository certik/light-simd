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

#include "simd_pack.h"

#include <light_simd/sse/sse_arith.h>

namespace lsimd
{
	// Arithmetic functions


	// Operators overloading

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator + (const simd_pack<T, Kind> a, const simd_pack<T, Kind> b)
	{
		return a.impl + b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator + (const simd_pack<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl + impl_t(b);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator + (const T a, const simd_pack<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) + b.impl;
	}


	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const simd_pack<T, Kind> a, const simd_pack<T, Kind> b)
	{
		return a.impl - b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const simd_pack<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl - impl_t(b);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const T a, const simd_pack<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) - b.impl;
	}



	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator * (const simd_pack<T, Kind> a, const simd_pack<T, Kind> b)
	{
		return a.impl * b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator * (const simd_pack<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl * impl_t(b);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator * (const T a, const simd_pack<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) * b.impl;
	}


	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator / (const simd_pack<T, Kind> a, const simd_pack<T, Kind> b)
	{
		return a.impl / b.impl;
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator / (const simd_pack<T, Kind> a, const T b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return a.impl / impl_t(b);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator / (const T a, const simd_pack<T, Kind> b)
	{
		typedef typename simd<T, Kind>::impl_type impl_t;
		return impl_t(a) / b.impl;
	}


	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> operator - (const simd_pack<T, Kind> a)
	{
		return - a.impl;
	}


	// Other arithmetic functions

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> abs(const simd_pack<T, Kind> a)
	{
		return abs(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> vmin(const simd_pack<T, Kind> a, const simd_pack<T, Kind> b)
	{
		return vmin(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> vmax(const simd_pack<T, Kind> a, const simd_pack<T, Kind> b)
	{
		return vmax(a.impl, b.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> sqrt(const simd_pack<T, Kind> a)
	{
		return sqrt(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> rcp(const simd_pack<T, Kind> a)
	{
		return rcp(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> rsqrt(const simd_pack<T, Kind> a)
	{
		return rsqrt(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> sqr(const simd_pack<T, Kind> a)
	{
		return sqr(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> cube(const simd_pack<T, Kind> a)
	{
		return cube(a.impl);
	}


	// Rounding

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> floor(const simd_pack<T, Kind> a)
	{
		return floor(a.impl);
	}

	template<typename T, typename Kind>
	LSIMD_ENSURE_INLINE
	inline simd_pack<T, Kind> ceil(const simd_pack<T, Kind> a)
	{
		return ceil(a.impl);
	}

}

#endif /* SIMD_ARITH_H_ */
