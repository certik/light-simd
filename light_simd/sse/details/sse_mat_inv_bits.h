/*
 * @file sse_mat_inv_bits.h
 *
 * The internal implementation of matrix determinant & inverse
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_INV_BITS_H_
#define LSIMD_SSE_MAT_INV_BITS_H_

#include "sse_mat_bits_f32.h"
#include "sse_mat_bits_f64.h"

namespace lsimd { namespace sse {

	/**********************************
	 *
	 *  Matrix 2 x 2
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk det_p(const smat<f32, 2, 2>& a)
	{
		sse_f32pk p = a.m_pk.swizzle<3,2,1,0>();
		p = mul(a.m_pk, p);
		return _mm_sub_ss(p.v, sse::f32_dup2_high(p.v));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk det_p(const smat<f64, 2, 2>& a)
	{
		sse_f64pk p = a.m_pk1.swizzle<1,0>();
		p = mul(a.m_pk0, p);
		return _mm_sub_sd(p.v, sse::f64_dup_high(p.v));
	}

	template<typename T>
	LSIMD_ENSURE_INLINE
	inline T det(const smat<T, 2, 2>& a)
	{
		return det_p(a).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline smat<f32, 2, 2> inv(const smat<f32, 2, 2>& a)
	{
		smat<f32, 2, 2> r;

		sse_f32pk dv = det_p(a).broadcast<0>();
		sse_f32pk c = div( sse_f32pk(1.f, -1.f, -1.f, 1.f), dv );

		r.m_pk = mul(a.m_pk.swizzle<3,1,2,0>(), c);
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat<f64, 2, 2> inv(const smat<f64, 2, 2>& a)
	{
		smat<f64, 2, 2> r;

		sse_f64pk dv = det_p(a).broadcast<0>();

		sse_f64pk c0 = div( sse_f64pk(1.f, -1.f), dv );
		sse_f64pk c1 = c0.swizzle<1,0>();

		r.m_pk0 = unpack_high(a.m_pk1, a.m_pk0);
		r.m_pk1 = unpack_low (a.m_pk1, a.m_pk0);

		r.m_pk0 = mul(r.m_pk0, c0);
		r.m_pk1 = mul(r.m_pk1, c1);

		return r;
	}



	/**********************************
	 *
	 *  Matrix 3 x 3
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk det_p(const smat<f32, 3, 3>& a)
	{
		sse_f32pk u = shuffle<1,0,0,1>(a.m_pk0, a.m_pk1);
		u = mul(u, a.m_pk2.swizzle<0,1,1,0>());
		u = mul(u, shuffle<2,2,2,2>(a.m_pk1, a.m_pk0));

		sse_f32pk v = a.m_pk0;
		v = mul(v, a.m_pk1.swizzle<1,0,1,0>());
		v = mul(v, a.m_pk2.broadcast<2>());

		u = add(u, u.dup_high());
		u = add(u, v);

		return _mm_sub_ss(u.v, sse::f32_dup2_high(u.v));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk det_p(const smat<f64, 3, 3>& a)
	{
		sse_f64pk u1 = a.m_pk0l.swizzle<1,0>();
		u1 = mul(u1, a.m_pk2l);
		u1 = mul(u1, a.m_pk1h.dup_low());

		sse_f64pk u2 = a.m_pk1l;
		u2 = mul(u2, a.m_pk2l.swizzle<1,0>());
		u2 = mul(u2, a.m_pk0h.dup_low());

		sse_f64pk u3 = a.m_pk0l;
		u3 = mul(u3, a.m_pk1l.swizzle<1,0>());
		u3 = mul(u3, a.m_pk2h.dup_low());

		u1 = add(u1, u2);
		u1 = add(u1, u3);

		return _mm_sub_sd(u1.v, sse::f64_dup_high(u1.v));
	}

	template<typename T>
	LSIMD_ENSURE_INLINE
	inline T det(const smat<T, 3, 3>& a)
	{
		return det_p(a).to_scalar();
	}



} }

#endif 
