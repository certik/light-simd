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
	 *  Auxiliary functions
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline sse_f32pk hdiff(sse_f32pk p)
	{
		return _mm_sub_ss(p.v, sse::f32_dup2_high(p.v));
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk hdiff(sse_f64pk p)
	{
		return _mm_sub_sd(p.v, sse::f64_dup_high(p.v));
	}


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
		return hdiff(p);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk det_p(const smat<f64, 2, 2>& a)
	{
		sse_f64pk p = a.m_pk1.swizzle<1,0>();
		p = mul(a.m_pk0, p);
		return hdiff(p);
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
		// generate product terms

		sse_f32pk b = shuffle<1,0,1,0>(a.m_pk2, a.m_pk1);

		sse_f32pk u1 = mul(a.m_pk0.dup_low(), b);
		sse_f32pk u2 = mul(a.m_pk1, b);

		u1 = mul(u1, shuffle<2,2,2,2>(a.m_pk1, a.m_pk2));
		u2 = mul(u2, shuffle<2,2,2,2>(a.m_pk0, a.m_pk0));

		// aggregate the terms

		u1 = sub(u1.dup_high(), u1);
		u1 = add(u1, u2);

		return hdiff(u1);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk det_p(const smat<f64, 3, 3>& a)
	{
		// generate product terms

		sse_f64pk c = a.m_pk2l.swizzle<1,0>();

		sse_f64pk u1 = mul(a.m_pk0l, a.m_pk1l.swizzle<1,0>());
		sse_f64pk u2 = mul(a.m_pk0l, c);
		sse_f64pk u3 = mul(a.m_pk1l, c);

		u1 = mul(u1, a.m_pk2h.broadcast<0>());
		u2 = mul(u2, a.m_pk1h.broadcast<0>());
		u3 = mul(u3, a.m_pk0h.broadcast<0>());

		// aggregate the terms

		u1 = sub(u1, u2);
		u1 = add(u1, u3);

		return hdiff(u1);
	}

	template<typename T>
	LSIMD_ENSURE_INLINE
	inline T det(const smat<T, 3, 3>& a)
	{
		return det_p(a).to_scalar();
	}



} }

#endif 
