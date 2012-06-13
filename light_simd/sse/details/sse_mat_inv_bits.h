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

	LSIMD_ENSURE_INLINE
	inline sse_f64pk rev_hdiff(sse_f64pk p)
	{
		return _mm_sub_sd(sse::f64_dup_high(p.v), p.v);
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


	inline smat<f32, 3, 3> inv(const smat<f32, 3, 3>& a)
	{
		// calculate co-factor matrix

		sse_f32pk p0 = a.m_pk0.swizzle<1,2,0,2>();
		sse_f32pk p1 = a.m_pk1.swizzle<1,2,0,2>();

		sse_f32pk q1 = a.m_pk1.swizzle<2,1,2,0>();
		sse_f32pk q2 = a.m_pk2.swizzle<2,1,2,0>();

		sse_f32pk c0 = mul(p1, q2);
		sse_f32pk c1 = mul(p0, q2);
		sse_f32pk c2 = mul(p0, q1);

		c0 = sub(shuffle<0,3,1,2>(c0, c1), shuffle<1,2,0,3>(c0, c1));

		p0 = merge_low(a.m_pk1, a.m_pk0);
		q2 = a.m_pk2.swizzle<1,0,1,0>();

		sse_f32pk d0 = mul(p0, q2);

		c1 = sub(shuffle<0,3,0,3>(c2, d0), shuffle<1,2,1,2>(c2, d0));

		c2 = mul(p0, p0.swizzle<3,2,1,0>());
		c2.v = _mm_sub_ss(c2.shift_front<1>().v, c2.v);

		// calculate determinant

		sse_f32pk detv = mul(c0, a.m_pk0);
		__m128 v2 = _mm_mul_ss(c1.dup_high().v, a.m_pk0.dup_high().v);

		detv.v = _mm_add_ss(_mm_add_ss(detv.v, detv.dup2_high().v), v2);

		// joint into adjoint matrix

		__m128 msk = _mm_castsi128_ps(
				_mm_setr_epi32((int)(0xffffffff), (int)(0xffffffff), (int)(0xffffffff), 0));

		smat<f32, 3, 3> r;
		r.m_pk0.v = _mm_and_ps(shuffle<0, 2, 0, 0>(c0, c1).v, msk);
		r.m_pk1.v = _mm_and_ps(shuffle<1, 3, 1, 1>(c0, c1).v, msk);
		r.m_pk2.v = _mm_and_ps(shuffle<2, 3, 0, 0>(c1, c2).v, msk);

		// multiply with the rcp(detv)

		sse_f32pk sca = _mm_div_ss( _mm_set1_ps(1.0f), detv.v);
		sca = sca.broadcast<0>();

		r.m_pk0 = mul(r.m_pk0, sca);
		r.m_pk1 = mul(r.m_pk1, sca);
		r.m_pk2 = mul(r.m_pk2, sca);

		return r;
	}


	inline smat<f64, 3, 3> inv(const smat<f64, 3, 3>& a)
	{
		smat<f64, 3, 3> r;

		// calculate co-factors

		// row 0

		sse_f64pk a0 = shuffle<1, 0>(a.m_pk0l, a.m_pk0h);
		sse_f64pk a1 = shuffle<1, 0>(a.m_pk1l, a.m_pk1h);
		sse_f64pk b1 = shuffle<0, 1>(a.m_pk1h, a.m_pk1l);
		sse_f64pk b2 = shuffle<0, 1>(a.m_pk2h, a.m_pk2l);

		sse_f64pk c00 = mul( a1, b2 );
		sse_f64pk c01 = mul( a0, b2 );
		sse_f64pk c02 = mul( a0, b1 );

		r.m_pk0l = sub(shuffle<0,1>(c00, c01), shuffle<1,0>(c00, c01));
		r.m_pk0h = sub(c02, c02.dup_high());

		// row 1

		a0 = shuffle<0, 0>(a.m_pk0l, a.m_pk0h);
		a1 = shuffle<0, 0>(a.m_pk1l, a.m_pk1h);
		b1 = shuffle<0, 0>(a.m_pk1h, a.m_pk1l);
		b2 = shuffle<0, 0>(a.m_pk2h, a.m_pk2l);

		sse_f64pk c10 = mul( a1, b2 );
		sse_f64pk c11 = mul( a0, b2 );
		sse_f64pk c12 = mul( a0, b1 );

		r.m_pk1l = sub(shuffle<1,0>(c10, c11), shuffle<0,1>(c10, c11));
		r.m_pk1h = sub(c12.dup_high(), c12);

		// row 3

		a0 = shuffle<0, 1>(a.m_pk0l, a.m_pk0l);
		a1 = shuffle<0, 1>(a.m_pk1l, a.m_pk1l);
		b1 = shuffle<1, 0>(a.m_pk1l, a.m_pk1l);
		b2 = shuffle<1, 0>(a.m_pk2l, a.m_pk2l);

		sse_f64pk c20 = mul( a1, b2 );
		sse_f64pk c21 = mul( a0, b2 );
		sse_f64pk c22 = mul( a0, b1 );

		r.m_pk2l = sub(shuffle<0,1>(c20, c21), shuffle<1,0>(c20, c21));
		r.m_pk2h = sub(c22, c22.dup_high());

		// calculate determinant

		sse_f64pk detv;
		detv.v = _mm_mul_sd( r.m_pk0l.v, a.m_pk0l.v );
		detv.v = _mm_add_sd(detv.v, _mm_mul_sd(r.m_pk1l.v, a.m_pk0l.dup_high().v));
		detv.v = _mm_add_sd(detv.v, _mm_mul_sd(r.m_pk2l.v, a.m_pk0h.v));

		// multiply with rcp(det)

		sse_f64pk sca = _mm_div_sd( _mm_set1_pd(1.0), detv.v );
		sca = sca.broadcast<0>();

		r.m_pk0l = mul(r.m_pk0l, sca);
		r.m_pk0h = mul(r.m_pk0h, sca);
		r.m_pk1l = mul(r.m_pk1l, sca);
		r.m_pk1h = mul(r.m_pk1h, sca);
		r.m_pk2l = mul(r.m_pk2l, sca);
		r.m_pk2h = mul(r.m_pk2h, sca);

		return r;
	}



} }

#endif 
