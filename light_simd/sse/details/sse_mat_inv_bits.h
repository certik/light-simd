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
		return sub_s(p, p.dup2_high());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk hdiff(sse_f64pk p)
	{
		return sub_s(p, p.dup_high());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk rev_hdiff(sse_f64pk p)
	{
		return sub_s(p.dup_high(), p);
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk pre_det(smat<f32,2,2> a)
	{
		return mul(a.m_pk, a.m_pk.swizzle<3,2,1,0>());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk pre_det(smat<f64,2,2> a)
	{
		return mul(a.m_pk0, a.m_pk1.swizzle<1,0>());
	}

	LSIMD_ENSURE_INLINE
	inline smat<f32,2,2> mm2x2(smat<f32,2,2> a, smat<f32,2,2> b)
	{
		return add(
				mul(a.m_pk.dup_low(),  b.m_pk.dup2_low()),
				mul(a.m_pk.dup_high(), b.m_pk.dup2_high()) );
	}

	LSIMD_ENSURE_INLINE
	inline smat<f64,2,2> mm2x2(smat<f64,2,2> a, smat<f64,2,2> b)
	{
		smat<f64, 2, 2> r;

		r.m_pk0 = add(
				mul(a.m_pk0, b.m_pk0.broadcast<0>()),
				mul(a.m_pk1, b.m_pk0.broadcast<1>()));

		r.m_pk1 = add(
				mul(a.m_pk0, b.m_pk1.broadcast<0>()),
				mul(a.m_pk1, b.m_pk1.broadcast<1>()));

		return r;
	}

	LSIMD_ENSURE_INLINE
	inline sse_f32pk mm2x2_trace_p(smat<f32,2,2> a, smat<f32,2,2> b)
	{
		sse_f32pk p = mul(a.m_pk, b.m_pk.swizzle<0,2,1,3>());
		p = add(p, p.dup_high());
		return add_s(p, p.dup2_high());
	}

	LSIMD_ENSURE_INLINE
	inline sse_f64pk mm2x2_trace_p(smat<f64,2,2> a, smat<f64,2,2> b)
	{
		sse_f64pk p = mul(a.m_pk0, unpack_low(b.m_pk0, b.m_pk1));
		p = add(p, mul(a.m_pk1, unpack_high(b.m_pk0, b.m_pk1)));
		return add_s(p, p.dup_high());
	}

	LSIMD_ENSURE_INLINE
	inline smat<f32,2,2> adjoint_signmask_f32()
	{
		sse_f32pk p = _mm_castsi128_ps(
				_mm_setr_epi32(0, (int)0x80000000, (int)0x80000000, 0));
		return p;
	}

	LSIMD_ENSURE_INLINE
	inline smat<f32,2,2> adjoint_mat(smat<f32,2,2> a, smat<f32,2,2> mask)
	{
		sse_f32pk p = _mm_xor_ps(a.m_pk.swizzle<3,1,2,0>().v, mask.m_pk.v);
		return p;
	}

	LSIMD_ENSURE_INLINE
	inline smat<f64,2,2> adjoint_signmask_f64()
	{
		smat<f64,2,2> m;
		m.m_pk0.v = _mm_setr_pd(0.0, -0.0);
		m.m_pk1.v = _mm_setr_pd(-0.0, 0.0);
		return m;
	}

	LSIMD_ENSURE_INLINE
	inline smat<f64,2,2> adjoint_mat(smat<f64,2,2> a, smat<f64,2,2> mask)
	{
		smat<f64,2,2> r;
		r.m_pk0.v = _mm_xor_pd(unpack_high(a.m_pk1, a.m_pk0).v, mask.m_pk0.v);
		r.m_pk1.v = _mm_xor_pd(unpack_low (a.m_pk1, a.m_pk0).v, mask.m_pk1.v);
		return r;
	}


	/**********************************
	 *
	 *  Matrix 2 x 2
	 *
	 **********************************/

	LSIMD_ENSURE_INLINE
	inline f32 det(const smat<f32, 2, 2>& a)
	{
		sse_f32pk p = pre_det(a);
		return hdiff(p).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline f64 det(const smat<f64, 2, 2>& a)
	{
		sse_f64pk p = pre_det(a);
		return hdiff(p).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline smat<f32, 2, 2> inv(const smat<f32, 2, 2>& a)
	{
		smat<f32, 2, 2> r;

		sse_f32pk dv = hdiff(pre_det(a)).broadcast<0>();
		sse_f32pk c = div( sse_f32pk(1.f, -1.f, -1.f, 1.f), dv );

		r.m_pk = mul(a.m_pk.swizzle<3,1,2,0>(), c);
		return r;
	}

	LSIMD_ENSURE_INLINE
	inline smat<f64, 2, 2> inv(const smat<f64, 2, 2>& a)
	{
		smat<f64, 2, 2> r;

		sse_f64pk dv = hdiff(pre_det(a)).broadcast<0>();

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
	inline f32 det(const smat<f32, 3, 3>& a)
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

		return hdiff(u1).to_scalar();
	}

	LSIMD_ENSURE_INLINE
	inline f64 det(const smat<f64, 3, 3>& a)
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

		return hdiff(u1).to_scalar();
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
		c2 = sub_s(c2.shift_front<1>(), c2);

		// calculate determinant

		sse_f32pk detv = mul(c0, a.m_pk0);
		sse_f32pk v2 = mul_s(c1.dup_high(), a.m_pk0.dup_high());

		detv = add_s(add_s(detv, detv.dup2_high()), v2);

		// joint into adjoint matrix

		__m128 msk = _mm_castsi128_ps(
				_mm_setr_epi32((int)(0xffffffff), (int)(0xffffffff), (int)(0xffffffff), 0));

		smat<f32, 3, 3> r;
		r.m_pk0.v = _mm_and_ps(shuffle<0, 2, 0, 0>(c0, c1).v, msk);
		r.m_pk1.v = _mm_and_ps(shuffle<1, 3, 1, 1>(c0, c1).v, msk);
		r.m_pk2.v = _mm_and_ps(shuffle<2, 3, 0, 0>(c1, c2).v, msk);

		// multiply with the rcp(detv)

		sse_f32pk sca = rcp_s(detv).broadcast<0>();

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

		sse_f64pk detv = mul_s( r.m_pk0l, a.m_pk0l );
		detv = add_s(detv, mul_s(r.m_pk1l, a.m_pk0l.dup_high()));
		detv = add_s(detv, mul_s(r.m_pk2l, a.m_pk0h));

		// multiply with rcp(det)

		sse_f64pk sca = rcp_s(detv).broadcast<0>();

		r.m_pk0l = mul(r.m_pk0l, sca);
		r.m_pk0h = mul(r.m_pk0h, sca);
		r.m_pk1l = mul(r.m_pk1l, sca);
		r.m_pk1h = mul(r.m_pk1h, sca);
		r.m_pk2l = mul(r.m_pk2l, sca);
		r.m_pk2h = mul(r.m_pk2h, sca);

		return r;
	}



	/******************************************************
	 *
	 *  Matrix 4 x 4
	 *
	 *  Let X = [A B; C D], then
	 *
	 *  |X| = |A| |D| + |B| |C| - tr(A# * B * D# * C)
	 *  Here, A# and D# are adjoints of A and D
	 *
	 ******************************************************/

	inline f32 det(const smat<f32, 4, 4>& X)
	{
		// pre-determinant for 2x2 blocks

		smat<f32,2,2> A = merge_low (X.m_pk0, X.m_pk1);
		smat<f32,2,2> C = merge_high(X.m_pk0, X.m_pk1);

		sse_f32pk dA = pre_det(A);
		sse_f32pk dC = pre_det(C);

		smat<f32,2,2> B = merge_low (X.m_pk2, X.m_pk3);
		smat<f32,2,2> D = merge_high(X.m_pk2, X.m_pk3);

		sse_f32pk dB = pre_det(B);
		sse_f32pk dD = pre_det(D);

		// combine terms

		sse_f32pk u1 = merge_low( dA, dB );
		sse_f32pk u2 = merge_low( dD, dC );
		sse_f32pk comb = sub(shuffle<0,2,0,2>(u1, u2), shuffle<1,3,1,3>(u1, u2));
		comb = mul(comb, comb.dup_high());
		comb = add_s(comb, comb.dup2_high());

		// adjoint matrices for a and d

		smat<f32,2,2> adj_mask = adjoint_signmask_f32();

		smat<f32,2,2> Aa = adjoint_mat(A, adj_mask);
		smat<f32,2,2> Da = adjoint_mat(D, adj_mask);

		// Q = A# * B * D# * C

		smat<f32,2,2> AaB = mm2x2(Aa, B);
		smat<f32,2,2> DaC = mm2x2(Da, C);
		sse_f32pk qtr = mm2x2_trace_p(AaB, DaC);

		return sub_s(comb, qtr).to_scalar();
	}


	inline f64 det(const smat<f64, 4, 4>& X)
	{
		// pre-determinant for 2 x 2 blocks

		smat<f64,2,2> A( X.m_pk0l, X.m_pk1l );
		smat<f64,2,2> C( X.m_pk0h, X.m_pk1h );

		sse_f64pk dA = pre_det(A);
		sse_f64pk dC = pre_det(C);

		smat<f64,2,2> B( X.m_pk2l, X.m_pk3l );
		smat<f64,2,2> D( X.m_pk2h, X.m_pk3h );

		sse_f64pk dB = pre_det(B);
		sse_f64pk dD = pre_det(D);

		// combine terms

		sse_f64pk ab_p = unpack_low (dA, dB);
		sse_f64pk ab_n = unpack_high(dA, dB);
		sse_f64pk dc_p = unpack_low (dD, dC);
		sse_f64pk dc_n = unpack_high(dD, dC);

		ab_p = sub(ab_p, ab_n);
		dc_p = sub(dc_p, dc_n);

		sse_f64pk comb = mul(ab_p, dc_p);
		comb = add_s(comb, comb.dup_high());

		// adjoint matrices for a and d

		smat<f64,2,2> adj_mask = adjoint_signmask_f64();

		smat<f64,2,2> Aa = adjoint_mat(A, adj_mask);
		smat<f64,2,2> Da = adjoint_mat(D, adj_mask);

		// Q = A# * B * D# * C

		smat<f64,2,2> AaB = mm2x2(Aa, B);
		smat<f64,2,2> DaC = mm2x2(Da, C);
		sse_f64pk qtr = mm2x2_trace_p(AaB, DaC);

		return sub_s(comb, qtr).to_scalar();
	}


	inline smat<f32,4,4> inv(const smat<f32,4,4>& X)
	{
		// blocking and evaluate pre-determinant

		smat<f32,2,2> A = merge_low (X.m_pk0, X.m_pk1);
		smat<f32,2,2> C = merge_high(X.m_pk0, X.m_pk1);

		sse_f32pk dA = hdiff(pre_det(A));
		sse_f32pk dC = hdiff(pre_det(C));

		smat<f32,2,2> B = merge_low (X.m_pk2, X.m_pk3);
		smat<f32,2,2> D = merge_high(X.m_pk2, X.m_pk3);

		sse_f32pk dB = hdiff(pre_det(B));
		sse_f32pk dD = hdiff(pre_det(D));

		// adjoint matrices

		smat<f32,2,2> adj_mask = adjoint_signmask_f32();

		smat<f32,2,2> Aa = adjoint_mat(A, adj_mask);
		smat<f32,2,2> Ba = adjoint_mat(B, adj_mask);
		smat<f32,2,2> Ca = adjoint_mat(C, adj_mask);
		smat<f32,2,2> Da = adjoint_mat(D, adj_mask);

		// incomplete partial inverses

		smat<f32,2,2> IA = mm2x2(B, mm2x2(Da, C));
		smat<f32,2,2> IB = mm2x2(D, mm2x2(Ba, A));
		smat<f32,2,2> IC = mm2x2(A, mm2x2(Ca, D));
		smat<f32,2,2> ID = mm2x2(C, mm2x2(Aa, B));

		// determinant

		sse_f32pk comb = add_s(mul_s(dA, dD), mul_s(dB, dC));
		sse_f32pk qtr = mm2x2_trace_p(Aa, IA);
		sse_f32pk detv = sub_s(comb, qtr);
		sse_f32pk rdetv = rcp_s(detv).broadcast<0>();

		// complete partial inverses

		A.m_pk = mul(A.m_pk, dD.broadcast<0>());
		B.m_pk = mul(B.m_pk, dC.broadcast<0>());
		C.m_pk = mul(C.m_pk, dB.broadcast<0>());
		D.m_pk = mul(D.m_pk, dA.broadcast<0>());

		IA = adjoint_mat(A.msub(IA), adj_mask);
		IB = adjoint_mat(C.msub(IB), adj_mask);
		IC = adjoint_mat(B.msub(IC), adj_mask);
		ID = adjoint_mat(D.msub(ID), adj_mask);

		// assemble into the inverse matrix

		smat<f32,4,4> Y;
		Y.m_pk0 = mul(merge_low (IA.m_pk, IC.m_pk), rdetv);
		Y.m_pk1 = mul(merge_high(IA.m_pk, IC.m_pk), rdetv);
		Y.m_pk2 = mul(merge_low (IB.m_pk, ID.m_pk), rdetv);
		Y.m_pk3 = mul(merge_high(IB.m_pk, ID.m_pk), rdetv);

		return Y;
	}


	inline smat<f64,4,4> inv(const smat<f64,4,4>& X)
	{
		// blocking and evaluate pre-determinant

		smat<f64,2,2> A( X.m_pk0l, X.m_pk1l );
		smat<f64,2,2> C( X.m_pk0h, X.m_pk1h );

		sse_f64pk dA = hdiff(pre_det(A)).broadcast<0>();
		sse_f64pk dC = hdiff(pre_det(C)).broadcast<0>();

		smat<f64,2,2> B( X.m_pk2l, X.m_pk3l );
		smat<f64,2,2> D( X.m_pk2h, X.m_pk3h );

		sse_f64pk dB = hdiff(pre_det(B)).broadcast<0>();
		sse_f64pk dD = hdiff(pre_det(D)).broadcast<0>();

		// adjoint matrices

		smat<f64,2,2> adj_mask = adjoint_signmask_f64();

		smat<f64,2,2> Aa = adjoint_mat(A, adj_mask);
		smat<f64,2,2> Ba = adjoint_mat(B, adj_mask);
		smat<f64,2,2> Ca = adjoint_mat(C, adj_mask);
		smat<f64,2,2> Da = adjoint_mat(D, adj_mask);

		// incomplete partial inverses

		smat<f64,2,2> IA = mm2x2(B, mm2x2(Da, C));
		smat<f64,2,2> IB = mm2x2(D, mm2x2(Ba, A));
		smat<f64,2,2> IC = mm2x2(A, mm2x2(Ca, D));
		smat<f64,2,2> ID = mm2x2(C, mm2x2(Aa, B));

		// determinant

		sse_f64pk comb = add_s(mul_s(dA, dD), mul_s(dB, dC));
		sse_f64pk qtr = mm2x2_trace_p(Aa, IA);
		sse_f64pk detv = sub_s(comb, qtr);
		sse_f64pk rdetv = rcp_s(detv).broadcast<0>();

		// complete partial inverses

		A.m_pk0 = mul(A.m_pk0, dD);
		A.m_pk1 = mul(A.m_pk1, dD);
		B.m_pk0 = mul(B.m_pk0, dC);
		B.m_pk1 = mul(B.m_pk1, dC);
		C.m_pk0 = mul(C.m_pk0, dB);
		C.m_pk1 = mul(C.m_pk1, dB);
		D.m_pk0 = mul(D.m_pk0, dA);
		D.m_pk1 = mul(D.m_pk1, dA);

		IA = adjoint_mat(A.msub(IA), adj_mask);
		IB = adjoint_mat(C.msub(IB), adj_mask);
		IC = adjoint_mat(B.msub(IC), adj_mask);
		ID = adjoint_mat(D.msub(ID), adj_mask);

		// assemble into the inverse matrix

		smat<f64,4,4> Y;
		Y.m_pk0l = mul(IA.m_pk0, rdetv);
		Y.m_pk0h = mul(IC.m_pk0, rdetv);
		Y.m_pk1l = mul(IA.m_pk1, rdetv);
		Y.m_pk1h = mul(IC.m_pk1, rdetv);
		Y.m_pk2l = mul(IB.m_pk0, rdetv);
		Y.m_pk2h = mul(ID.m_pk0, rdetv);
		Y.m_pk3l = mul(IB.m_pk1, rdetv);
		Y.m_pk3h = mul(ID.m_pk1, rdetv);

		return Y;
	}

} }

#endif 
