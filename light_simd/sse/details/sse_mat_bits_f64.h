/**
 * @file sse_mat_bits_f64.h
 *
 *  The internal implementation of SSE matrix (for f64)
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_BITS_F64_H_
#define LSIMD_SSE_MAT_BITS_F64_H_

#include "../sse_vec.h"

namespace lsimd { namespace sse {

	template<typename T, int M, int N> class smat;

	/********************************************
	 *
	 *  2 x 2
	 *
	 ********************************************/

	template<> class smat<f64, 2, 2>
	{
	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0, sse_f64pk pk1) : m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + 2, AlignT());

			m_pk0 = unpack_low(pr0, pr1);
			m_pk1 = unpack_high(pr0, pr1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + ldim, AlignT());

			m_pk0 = unpack_low(pr0, pr1);
			m_pk1 = unpack_high(pr0, pr1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + ldim, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0 = add(m_pk0, r.m_pk0);
			o.m_pk1 = add(m_pk1, r.m_pk1);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0 = sub(m_pk0, r.m_pk0);
			o.m_pk1 = sub(m_pk1, r.m_pk1);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0 = mul(m_pk0, r.m_pk0);
			o.m_pk1 = mul(m_pk1, r.m_pk1);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0 = add(m_pk0, r.m_pk0);
			m_pk1 = add(m_pk1, r.m_pk1);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0 = sub(m_pk0, r.m_pk0);
			m_pk1 = sub(m_pk1, r.m_pk1);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0 = mul(m_pk0, r.m_pk0);
			m_pk1 = mul(m_pk1, r.m_pk1);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 2> transform(sse_vec<f64, 2> v) const
		{
			sse_f64pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f64pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			return sse_vec<f64, 2>(add(p0, p1));
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0.e[0] + m_pk1.e[1];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1]) &&
					m_pk1.test_equal(r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2 x 2]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	};


	/********************************************
	 *
	 *  2 x 3
	 *
	 ********************************************/

	template<> class smat<f64, 2, 3>
	{
	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;
		sse_f64pk m_pk2;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0, sse_f64pk pk1, sse_f64pk pk2)
		: m_pk0(pk0), m_pk1(pk1), m_pk2(pk2) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0( zero_t() ), m_pk1( zero_t() ), m_pk2( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 2, AlignT());
			m_pk2.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + ldim, AlignT());
			m_pk2.load(x + ldim * 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pa(x, AlignT());
			sse_f64pk pb(x + 2, AlignT());
			sse_f64pk pc(x + 4, AlignT());

			m_pk0 = shuffle<0, 1>(pa, pb);
			m_pk1 = shuffle<1, 0>(pa, pc);
			m_pk2 = shuffle<0, 1>(pb, pc);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk p0l, p0h, p1l, p1h;

			p0l.load(x, AlignT());
			p0h.partial_load<1>(x + 2);

			p1l.load(x + ldim, AlignT());
			p1h.partial_load<1>(x + ldim + 2);

			m_pk0 = unpack_low(p0l, p1l);
			m_pk1 = unpack_high(p0l, p1l);
			m_pk2 = unpack_low(p0h, p1h);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 2, AlignT());
			m_pk2.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + ldim, AlignT());
			m_pk2.store(x + ldim * 2, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0 = add(m_pk0, r.m_pk0);
			o.m_pk1 = add(m_pk1, r.m_pk1);
			o.m_pk2 = add(m_pk2, r.m_pk2);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0 = sub(m_pk0, r.m_pk0);
			o.m_pk1 = sub(m_pk1, r.m_pk1);
			o.m_pk2 = sub(m_pk2, r.m_pk2);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0 = mul(m_pk0, r.m_pk0);
			o.m_pk1 = mul(m_pk1, r.m_pk1);
			o.m_pk2 = mul(m_pk2, r.m_pk2);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0 = add(m_pk0, r.m_pk0);
			m_pk1 = add(m_pk1, r.m_pk1);
			m_pk2 = add(m_pk2, r.m_pk2);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0 = sub(m_pk0, r.m_pk0);
			m_pk1 = sub(m_pk1, r.m_pk1);
			m_pk2 = sub(m_pk2, r.m_pk2);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0 = mul(m_pk0, r.m_pk0);
			m_pk1 = mul(m_pk1, r.m_pk1);
			m_pk2 = mul(m_pk2, r.m_pk2);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 2> transform(sse_vec<f64, 3> v) const
		{
			sse_f64pk p0 = mul(m_pk0, v.m_pk0.broadcast<0>());
			sse_f64pk p1 = mul(m_pk1, v.m_pk0.broadcast<1>());
			sse_f64pk p2 = mul(m_pk2, v.m_pk1.broadcast<0>());
			return sse_vec<f64, 2>(add(add(p0, p1), p2));
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0.e[0] + m_pk1.e[1];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1]) &&
					m_pk1.test_equal(r[2], r[3]) &&
					m_pk2.test_equal(r[4], r[5]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2 x 3]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
			std::printf("    m_pk2 = "); m_pk2.dump(fmt); std::printf("\n");
		}

	};



	/********************************************
	 *
	 *  2 x 4
	 *
	 ********************************************/

	template<> class smat<f64, 2, 4>
	{
	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;
		sse_f64pk m_pk2;
		sse_f64pk m_pk3;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0, sse_f64pk pk1, sse_f64pk pk2, sse_f64pk pk3)
		: m_pk0(pk0), m_pk1(pk1), m_pk2(pk2), m_pk3(pk3) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0( zero_t() ), m_pk1( zero_t() ), m_pk2( zero_t() ), m_pk3( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 2, AlignT());
			m_pk2.load(x + 4, AlignT());
			m_pk3.load(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + ldim, AlignT());
			m_pk2.load(x + ldim * 2, AlignT());
			m_pk3.load(x + ldim * 3, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk p0l(x, AlignT());
			sse_f64pk p0h(x + 2, AlignT());
			sse_f64pk p1l(x + 4, AlignT());
			sse_f64pk p1h(x + 6, AlignT());

			m_pk0 = unpack_low (p0l, p1l);
			m_pk1 = unpack_high(p0l, p1l);
			m_pk2 = unpack_low (p0h, p1h);
			m_pk3 = unpack_high(p0h, p1h);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;

			sse_f64pk p0l(x, AlignT());
			sse_f64pk p0h(x + 2, AlignT());
			sse_f64pk p1l(x1, AlignT());
			sse_f64pk p1h(x1 + 2, AlignT());

			m_pk0 = unpack_low (p0l, p1l);
			m_pk1 = unpack_high(p0l, p1l);
			m_pk2 = unpack_low (p0h, p1h);
			m_pk3 = unpack_high(p0h, p1h);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 2, AlignT());
			m_pk2.store(x + 4, AlignT());
			m_pk3.store(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x2 = x + 2 * ldim;

			m_pk0.store(x, AlignT());
			m_pk1.store(x + ldim, AlignT());
			m_pk2.store(x2, AlignT());
			m_pk3.store(x2 + ldim, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0 = add(m_pk0, r.m_pk0);
			o.m_pk1 = add(m_pk1, r.m_pk1);
			o.m_pk2 = add(m_pk2, r.m_pk2);
			o.m_pk3 = add(m_pk3, r.m_pk3);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0 = sub(m_pk0, r.m_pk0);
			o.m_pk1 = sub(m_pk1, r.m_pk1);
			o.m_pk2 = sub(m_pk2, r.m_pk2);
			o.m_pk3 = sub(m_pk3, r.m_pk3);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0 = mul(m_pk0, r.m_pk0);
			o.m_pk1 = mul(m_pk1, r.m_pk1);
			o.m_pk2 = mul(m_pk2, r.m_pk2);
			o.m_pk3 = mul(m_pk3, r.m_pk3);
			return o;
		}


		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0 = add(m_pk0, r.m_pk0);
			m_pk1 = add(m_pk1, r.m_pk1);
			m_pk2 = add(m_pk2, r.m_pk2);
			m_pk3 = add(m_pk3, r.m_pk3);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0 = sub(m_pk0, r.m_pk0);
			m_pk1 = sub(m_pk1, r.m_pk1);
			m_pk2 = sub(m_pk2, r.m_pk2);
			m_pk3 = sub(m_pk3, r.m_pk3);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0 = mul(m_pk0, r.m_pk0);
			m_pk1 = mul(m_pk1, r.m_pk1);
			m_pk2 = mul(m_pk2, r.m_pk2);
			m_pk3 = mul(m_pk3, r.m_pk3);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 2> transform(sse_vec<f64, 4> v) const
		{
			sse_f64pk p0 = mul(m_pk0, v.m_pk0.broadcast<0>());
			sse_f64pk p1 = mul(m_pk1, v.m_pk0.broadcast<1>());
			sse_f64pk p2 = mul(m_pk2, v.m_pk1.broadcast<0>());
			sse_f64pk p3 = mul(m_pk3, v.m_pk1.broadcast<1>());

			p0 = add(p0, p1);
			p2 = add(p2, p3);
			return sse_vec<f64, 2>(add(p0, p2));
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0.e[0] + m_pk1.e[1];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1]) &&
					m_pk1.test_equal(r[2], r[3]) &&
					m_pk2.test_equal(r[4], r[5]) &&
					m_pk3.test_equal(r[6], r[7]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2 x 4]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
			std::printf("    m_pk2 = "); m_pk2.dump(fmt); std::printf("\n");
			std::printf("    m_pk3 = "); m_pk3.dump(fmt); std::printf("\n");
		}

	};


	/********************************************
	 *
	 *  3 x 2
	 *
	 ********************************************/

	template<> class smat<f64, 3, 2>
	{
	public:
		sse_f64pk m_pk0l;
		sse_f64pk m_pk0h;
		sse_f64pk m_pk1l;
		sse_f64pk m_pk1h;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0l, sse_f64pk pk0h, sse_f64pk pk1l, sse_f64pk pk1h)
		: m_pk0l(pk0l), m_pk0h(pk0h), m_pk1l(pk1l), m_pk1h(pk1h) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0l( zero_t() ), m_pk0h( zero_t() ), m_pk1l( zero_t() ), m_pk1h( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0l.load(x, AlignT());
			m_pk0h.partial_load<1>(x + 2);
			m_pk1l.load(x + 3, unaligned_t());
			m_pk1h.partial_load<1>(x + 5);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;

			m_pk0l.load(x, AlignT());
			m_pk0h.partial_load<1>(x + 2);

			m_pk1l.load(x1, AlignT());
			m_pk1h.partial_load<1>(x1 + 2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + 2, AlignT());
			sse_f64pk pr2(x + 4, AlignT());

			_load_trans(pr0, pr1, pr2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + ldim, AlignT());
			sse_f64pk pr2(x + ldim * 2, AlignT());

			_load_trans(pr0, pr1, pr2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0l.store(x, AlignT());
			m_pk0h.partial_store<1>(x + 2);
			m_pk1l.store(x + 3, unaligned_t());
			m_pk1h.partial_store<1>(x + 5);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x1 = x + ldim;

			m_pk0l.store(x, AlignT());
			m_pk0h.partial_store<1>(x + 2);
			m_pk1l.store(x1, AlignT());
			m_pk1h.partial_store<1>(x1 + 2);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0l = add(m_pk0l, r.m_pk0l);
			o.m_pk0h = add(m_pk0h, r.m_pk0h);
			o.m_pk1l = add(m_pk1l, r.m_pk1l);
			o.m_pk1h = add(m_pk1h, r.m_pk1h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0l = sub(m_pk0l, r.m_pk0l);
			o.m_pk0h = sub(m_pk0h, r.m_pk0h);
			o.m_pk1l = sub(m_pk1l, r.m_pk1l);
			o.m_pk1h = sub(m_pk1h, r.m_pk1h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0l = mul(m_pk0l, r.m_pk0l);
			o.m_pk0h = mul(m_pk0h, r.m_pk0h);
			o.m_pk1l = mul(m_pk1l, r.m_pk1l);
			o.m_pk1h = mul(m_pk1h, r.m_pk1h);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0l = add(m_pk0l, r.m_pk0l);
			m_pk0h = add(m_pk0h, r.m_pk0h);
			m_pk1l = add(m_pk1l, r.m_pk1l);
			m_pk1h = add(m_pk1h, r.m_pk1h);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0l = sub(m_pk0l, r.m_pk0l);
			m_pk0h = sub(m_pk0h, r.m_pk0h);
			m_pk1l = sub(m_pk1l, r.m_pk1l);
			m_pk1h = sub(m_pk1h, r.m_pk1h);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0l = mul(m_pk0l, r.m_pk0l);
			m_pk0h = mul(m_pk0h, r.m_pk0h);
			m_pk1l = mul(m_pk1l, r.m_pk1l);
			m_pk1h = mul(m_pk1h, r.m_pk1h);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 3> transform(sse_vec<f64, 2> v) const
		{
			sse_f64pk v0 = v.m_pk.broadcast<0>();
			sse_f64pk p0l = mul(m_pk0l, v0);
			sse_f64pk p0h = mul(m_pk0h, v0);

			sse_f64pk v1 = v.m_pk.broadcast<1>();
			sse_f64pk p1l = mul(m_pk1l, v1);
			sse_f64pk p1h = mul(m_pk1h, v1);

			sse_f64pk pl = add(p0l, p1l);
			sse_f64pk ph = add(p0h, p1h);

			return sse_vec<f64, 3>(pl, ph);
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0l.e[0] + m_pk1l.e[1];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0l.test_equal(r[0], r[1]) &&
					m_pk0h.test_equal(r[2], 0.0) &&
					m_pk1l.test_equal(r[3], r[4]) &&
					m_pk1h.test_equal(r[5], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3 x 2]:\n");

			std::printf("    m_pk0 = ");
			m_pk0l.dump(fmt);
			std::printf(" ");
			m_pk0h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk1 = ");
			m_pk1l.dump(fmt);
			std::printf(" ");
			m_pk1h.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(sse_f64pk pr0, sse_f64pk pr1, sse_f64pk pr2)
		{
			sse_f64pk z = zero_t();

			m_pk0l = unpack_low (pr0, pr1);
			m_pk0h = unpack_low (pr2, z);
			m_pk1l = unpack_high(pr0, pr1);
			m_pk1h = unpack_high(pr2, z);
		}

	};



	/********************************************
	 *
	 *  3 x 3
	 *
	 ********************************************/

	template<> class smat<f64, 3, 3>
	{
	public:
		sse_f64pk m_pk0l;
		sse_f64pk m_pk0h;
		sse_f64pk m_pk1l;
		sse_f64pk m_pk1h;
		sse_f64pk m_pk2l;
		sse_f64pk m_pk2h;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0l, sse_f64pk pk0h, sse_f64pk pk1l, sse_f64pk pk1h,
				sse_f64pk pk2l, sse_f64pk pk2h)
		: m_pk0l(pk0l), m_pk0h(pk0h), m_pk1l(pk1l), m_pk1h(pk1h)
		, m_pk2l(pk2l), m_pk2h(pk2h){ }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0l( zero_t() ), m_pk0h( zero_t() ), m_pk1l( zero_t() ), m_pk1h( zero_t() )
		, m_pk2l( zero_t() ), m_pk2h( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0l.load(x, AlignT());
			m_pk0h.partial_load<1>(x + 2);
			m_pk1l.load(x + 3, unaligned_t());
			m_pk1h.partial_load<1>(x + 5);
			m_pk2l.load(x + 6, AlignT());
			m_pk2h.partial_load<1>(x + 8);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;

			m_pk0l.load(x, AlignT());
			m_pk0h.partial_load<1>(x + 2);

			m_pk1l.load(x1, AlignT());
			m_pk1h.partial_load<1>(x1 + 2);

			m_pk2l.load(x2, AlignT());
			m_pk2h.partial_load<1>(x2 + 2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk l0, l1, l2;
			sse_f64pk h0, h1, h2;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x + 3, unaligned_t());
			h1.partial_load<1>(x + 5);

			l2.load(x + 6, AlignT());
			h2.partial_load<1>(x + 8);

			sse_f64pk z = zero_t();

			m_pk0l = unpack_low(l0, l1);
			m_pk0h = unpack_low(l2, z);

			m_pk1l = unpack_high(l0, l1);
			m_pk1h = unpack_high(l2, z);

			m_pk2l = unpack_low(h0, h1);
			m_pk2h = unpack_low(h2, z);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk l0, l1, l2;
			sse_f64pk h0, h1, h2;

			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x1, AlignT());
			h1.partial_load<1>(x1 + 2);

			l2.load(x2, AlignT());
			h2.partial_load<1>(x2 + 2);

			sse_f64pk z = zero_t();

			m_pk0l = unpack_low(l0, l1);
			m_pk0h = unpack_low(l2, z);

			m_pk1l = unpack_high(l0, l1);
			m_pk1h = unpack_high(l2, z);

			m_pk2l = unpack_low(h0, h1);
			m_pk2h = unpack_low(h2, z);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0l.store(x, AlignT());
			m_pk0h.partial_store<1>(x + 2);
			m_pk1l.store(x + 3, unaligned_t());
			m_pk1h.partial_store<1>(x + 5);
			m_pk2l.store(x + 6, AlignT());
			m_pk2h.partial_store<1>(x + 8);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x1 = x + ldim;
			f64 *x2 = x1 + ldim;

			m_pk0l.store(x, AlignT());
			m_pk0h.partial_store<1>(x + 2);
			m_pk1l.store(x1, AlignT());
			m_pk1h.partial_store<1>(x1 + 2);
			m_pk2l.store(x2, AlignT());
			m_pk2h.partial_store<1>(x2 + 2);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0l = add(m_pk0l, r.m_pk0l);
			o.m_pk0h = add(m_pk0h, r.m_pk0h);
			o.m_pk1l = add(m_pk1l, r.m_pk1l);
			o.m_pk1h = add(m_pk1h, r.m_pk1h);
			o.m_pk2l = add(m_pk2l, r.m_pk2l);
			o.m_pk2h = add(m_pk2h, r.m_pk2h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0l = sub(m_pk0l, r.m_pk0l);
			o.m_pk0h = sub(m_pk0h, r.m_pk0h);
			o.m_pk1l = sub(m_pk1l, r.m_pk1l);
			o.m_pk1h = sub(m_pk1h, r.m_pk1h);
			o.m_pk2l = sub(m_pk2l, r.m_pk2l);
			o.m_pk2h = sub(m_pk2h, r.m_pk2h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0l = mul(m_pk0l, r.m_pk0l);
			o.m_pk0h = mul(m_pk0h, r.m_pk0h);
			o.m_pk1l = mul(m_pk1l, r.m_pk1l);
			o.m_pk1h = mul(m_pk1h, r.m_pk1h);
			o.m_pk2l = mul(m_pk2l, r.m_pk2l);
			o.m_pk2h = mul(m_pk2h, r.m_pk2h);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0l = add(m_pk0l, r.m_pk0l);
			m_pk0h = add(m_pk0h, r.m_pk0h);
			m_pk1l = add(m_pk1l, r.m_pk1l);
			m_pk1h = add(m_pk1h, r.m_pk1h);
			m_pk2l = add(m_pk2l, r.m_pk2l);
			m_pk2h = add(m_pk2h, r.m_pk2h);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0l = sub(m_pk0l, r.m_pk0l);
			m_pk0h = sub(m_pk0h, r.m_pk0h);
			m_pk1l = sub(m_pk1l, r.m_pk1l);
			m_pk1h = sub(m_pk1h, r.m_pk1h);
			m_pk2l = sub(m_pk2l, r.m_pk2l);
			m_pk2h = sub(m_pk2h, r.m_pk2h);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0l = mul(m_pk0l, r.m_pk0l);
			m_pk0h = mul(m_pk0h, r.m_pk0h);
			m_pk1l = mul(m_pk1l, r.m_pk1l);
			m_pk1h = mul(m_pk1h, r.m_pk1h);
			m_pk2l = mul(m_pk2l, r.m_pk2l);
			m_pk2h = mul(m_pk2h, r.m_pk2h);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 3> transform(sse_vec<f64, 3> v) const
		{
			sse_f64pk v0 = v.m_pk0.broadcast<0>();
			sse_f64pk p0l = mul(m_pk0l, v0);
			sse_f64pk p0h = mul(m_pk0h, v0);

			sse_f64pk v1 = v.m_pk0.broadcast<1>();
			sse_f64pk p1l = mul(m_pk1l, v1);
			sse_f64pk p1h = mul(m_pk1h, v1);

			sse_f64pk v2 = v.m_pk1.broadcast<0>();
			sse_f64pk p2l = mul(m_pk2l, v2);
			sse_f64pk p2h = mul(m_pk2h, v2);

			sse_f64pk pl = add(add(p0l, p1l), p2l);
			sse_f64pk ph = add(add(p0h, p1h), p2h);

			return sse_vec<f64, 3>(pl, ph);
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0l.e[0] + m_pk1l.e[1] + m_pk2h.e[0];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0l.test_equal(r[0], r[1]) &&
					m_pk0h.test_equal(r[2], 0.0) &&
					m_pk1l.test_equal(r[3], r[4]) &&
					m_pk1h.test_equal(r[5], 0.0) &&
					m_pk2l.test_equal(r[6], r[7]) &&
					m_pk2h.test_equal(r[8], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3 x 3]:\n");

			std::printf("    m_pk0 = ");
			m_pk0l.dump(fmt);
			std::printf(" ");
			m_pk0h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk1 = ");
			m_pk1l.dump(fmt);
			std::printf(" ");
			m_pk1h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk2 = ");
			m_pk2l.dump(fmt);
			std::printf(" ");
			m_pk2h.dump(fmt);
			std::printf("\n");
		}

	};



	/********************************************
	 *
	 *  3 x 4
	 *
	 ********************************************/

	template<> class smat<f64, 3, 4>
	{
	public:
		sse_f64pk m_pk0l;
		sse_f64pk m_pk0h;
		sse_f64pk m_pk1l;
		sse_f64pk m_pk1h;
		sse_f64pk m_pk2l;
		sse_f64pk m_pk2h;
		sse_f64pk m_pk3l;
		sse_f64pk m_pk3h;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0l, sse_f64pk pk0h, sse_f64pk pk1l, sse_f64pk pk1h,
				sse_f64pk pk2l, sse_f64pk pk2h, sse_f64pk pk3l, sse_f64pk pk3h)
		: m_pk0l(pk0l), m_pk0h(pk0h), m_pk1l(pk1l), m_pk1h(pk1h)
		, m_pk2l(pk2l), m_pk2h(pk2h), m_pk3l(pk3l), m_pk3h(pk3h) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0l( zero_t() ), m_pk0h( zero_t() ), m_pk1l( zero_t() ), m_pk1h( zero_t() )
		, m_pk2l( zero_t() ), m_pk2h( zero_t() ), m_pk3l( zero_t() ), m_pk3h( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0l.load(x, AlignT());
			m_pk0h.partial_load<1>(x + 2);
			m_pk1l.load(x + 3, unaligned_t());
			m_pk1h.partial_load<1>(x + 5);
			m_pk2l.load(x + 6, AlignT());
			m_pk2h.partial_load<1>(x + 8);
			m_pk3l.load(x + 9, unaligned_t());
			m_pk3h.partial_load<1>(x + 11);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;
			const f64 *x3 = x2 + ldim;

			m_pk0l.load(x, AlignT());
			m_pk0h.partial_load<1>(x + 2);

			m_pk1l.load(x1, AlignT());
			m_pk1h.partial_load<1>(x1 + 2);

			m_pk2l.load(x2, AlignT());
			m_pk2h.partial_load<1>(x2 + 2);

			m_pk3l.load(x3, AlignT());
			m_pk3h.partial_load<1>(x3 + 2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			_load_trans(x, x + 4, x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, x + ldim * 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0l.store(x, AlignT());
			m_pk0h.partial_store<1>(x + 2);
			m_pk1l.store(x + 3, unaligned_t());
			m_pk1h.partial_store<1>(x + 5);
			m_pk2l.store(x + 6, AlignT());
			m_pk2h.partial_store<1>(x + 8);
			m_pk3l.store(x + 9, unaligned_t());
			m_pk3h.partial_store<1>(x + 11);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x1 = x + ldim;
			f64 *x2 = x1 + ldim;
			f64 *x3 = x2 + ldim;

			m_pk0l.store(x, AlignT());
			m_pk0h.partial_store<1>(x + 2);
			m_pk1l.store(x1, AlignT());
			m_pk1h.partial_store<1>(x1 + 2);
			m_pk2l.store(x2, AlignT());
			m_pk2h.partial_store<1>(x2 + 2);
			m_pk3l.store(x3, AlignT());
			m_pk3h.partial_store<1>(x3 + 2);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0l = add(m_pk0l, r.m_pk0l);
			o.m_pk0h = add(m_pk0h, r.m_pk0h);
			o.m_pk1l = add(m_pk1l, r.m_pk1l);
			o.m_pk1h = add(m_pk1h, r.m_pk1h);
			o.m_pk2l = add(m_pk2l, r.m_pk2l);
			o.m_pk2h = add(m_pk2h, r.m_pk2h);
			o.m_pk3l = add(m_pk3l, r.m_pk3l);
			o.m_pk3h = add(m_pk3h, r.m_pk3h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0l = sub(m_pk0l, r.m_pk0l);
			o.m_pk0h = sub(m_pk0h, r.m_pk0h);
			o.m_pk1l = sub(m_pk1l, r.m_pk1l);
			o.m_pk1h = sub(m_pk1h, r.m_pk1h);
			o.m_pk2l = sub(m_pk2l, r.m_pk2l);
			o.m_pk2h = sub(m_pk2h, r.m_pk2h);
			o.m_pk3l = sub(m_pk3l, r.m_pk3l);
			o.m_pk3h = sub(m_pk3h, r.m_pk3h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0l = mul(m_pk0l, r.m_pk0l);
			o.m_pk0h = mul(m_pk0h, r.m_pk0h);
			o.m_pk1l = mul(m_pk1l, r.m_pk1l);
			o.m_pk1h = mul(m_pk1h, r.m_pk1h);
			o.m_pk2l = mul(m_pk2l, r.m_pk2l);
			o.m_pk2h = mul(m_pk2h, r.m_pk2h);
			o.m_pk3l = mul(m_pk3l, r.m_pk3l);
			o.m_pk3h = mul(m_pk3h, r.m_pk3h);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0l = add(m_pk0l, r.m_pk0l);
			m_pk0h = add(m_pk0h, r.m_pk0h);
			m_pk1l = add(m_pk1l, r.m_pk1l);
			m_pk1h = add(m_pk1h, r.m_pk1h);
			m_pk2l = add(m_pk2l, r.m_pk2l);
			m_pk2h = add(m_pk2h, r.m_pk2h);
			m_pk3l = add(m_pk3l, r.m_pk3l);
			m_pk3h = add(m_pk3h, r.m_pk3h);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0l = sub(m_pk0l, r.m_pk0l);
			m_pk0h = sub(m_pk0h, r.m_pk0h);
			m_pk1l = sub(m_pk1l, r.m_pk1l);
			m_pk1h = sub(m_pk1h, r.m_pk1h);
			m_pk2l = sub(m_pk2l, r.m_pk2l);
			m_pk2h = sub(m_pk2h, r.m_pk2h);
			m_pk3l = sub(m_pk3l, r.m_pk3l);
			m_pk3h = sub(m_pk3h, r.m_pk3h);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0l = mul(m_pk0l, r.m_pk0l);
			m_pk0h = mul(m_pk0h, r.m_pk0h);
			m_pk1l = mul(m_pk1l, r.m_pk1l);
			m_pk1h = mul(m_pk1h, r.m_pk1h);
			m_pk2l = mul(m_pk2l, r.m_pk2l);
			m_pk2h = mul(m_pk2h, r.m_pk2h);
			m_pk3l = mul(m_pk3l, r.m_pk3l);
			m_pk3h = mul(m_pk3h, r.m_pk3h);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 3> transform(sse_vec<f64, 4> v) const
		{
			sse_f64pk v0 = v.m_pk0.broadcast<0>();
			sse_f64pk p0l = mul(m_pk0l, v0);
			sse_f64pk p0h = mul(m_pk0h, v0);

			sse_f64pk v1 = v.m_pk0.broadcast<1>();
			sse_f64pk p1l = mul(m_pk1l, v1);
			sse_f64pk p1h = mul(m_pk1h, v1);

			sse_f64pk v2 = v.m_pk1.broadcast<0>();
			sse_f64pk p2l = mul(m_pk2l, v2);
			sse_f64pk p2h = mul(m_pk2h, v2);

			sse_f64pk v3 = v.m_pk1.broadcast<1>();
			sse_f64pk p3l = mul(m_pk3l, v3);
			sse_f64pk p3h = mul(m_pk3h, v3);

			p0l = add(p0l, p1l);
			p0h = add(p0h, p1h);
			p2l = add(p2l, p3l);
			p2h = add(p2h, p3h);

			sse_f64pk pl = add(p0l, p2l);
			sse_f64pk ph = add(p0h, p2h);

			return sse_vec<f64, 3>(pl, ph);
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0l.e[0] + m_pk1l.e[1] + m_pk2h.e[0];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0l.test_equal(r[0], r[1]) &&
					m_pk0h.test_equal(r[2], 0.0) &&
					m_pk1l.test_equal(r[3], r[4]) &&
					m_pk1h.test_equal(r[5], 0.0) &&
					m_pk2l.test_equal(r[6], r[7]) &&
					m_pk2h.test_equal(r[8], 0.0) &&
					m_pk3l.test_equal(r[9], r[10]) &&
					m_pk3h.test_equal(r[11], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3 x 4]:\n");

			std::printf("    m_pk0 = ");
			m_pk0l.dump(fmt);
			std::printf(" ");
			m_pk0h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk1 = ");
			m_pk1l.dump(fmt);
			std::printf(" ");
			m_pk1h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk2 = ");
			m_pk2l.dump(fmt);
			std::printf(" ");
			m_pk2h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk3 = ");
			m_pk3l.dump(fmt);
			std::printf(" ");
			m_pk3h.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f64 *r0, const f64 *r1, const f64 *r2, AlignT)
		{
			sse_f64pk l0(r0,     AlignT());
			sse_f64pk h0(r0 + 2, AlignT());
			sse_f64pk l1(r1,     AlignT());
			sse_f64pk h1(r1 + 2, AlignT());
			sse_f64pk l2(r2,     AlignT());
			sse_f64pk h2(r2 + 2, AlignT());

			sse_f64pk z = zero_t();

			m_pk0l = unpack_low(l0, l1);
			m_pk0h = unpack_low(l2, z);

			m_pk1l = unpack_high(l0, l1);
			m_pk1h = unpack_high(l2, z);

			m_pk2l = unpack_low(h0, h1);
			m_pk2h = unpack_low(h2, z);

			m_pk3l = unpack_high(h0, h1);
			m_pk3h = unpack_high(h2, z);
		}

	};



	/********************************************
	 *
	 *  4 x 2
	 *
	 ********************************************/

	template<> class smat<f64, 4, 2>
	{
	public:
		sse_f64pk m_pk0l;
		sse_f64pk m_pk0h;
		sse_f64pk m_pk1l;
		sse_f64pk m_pk1h;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0l, sse_f64pk pk0h, sse_f64pk pk1l, sse_f64pk pk1h)
		: m_pk0l(pk0l), m_pk0h(pk0h), m_pk1l(pk1l), m_pk1h(pk1h) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0l( zero_t() ), m_pk0h( zero_t() ), m_pk1l( zero_t() ), m_pk1h( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0l.load(x,     AlignT());
			m_pk0h.load(x + 2, AlignT());
			m_pk1l.load(x + 4, AlignT());
			m_pk1h.load(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;

			m_pk0l.load(x,     AlignT());
			m_pk0h.load(x + 2, AlignT());

			m_pk1l.load(x1,     AlignT());
			m_pk1h.load(x1 + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + 2, AlignT());
			sse_f64pk pr2(x + 4, AlignT());
			sse_f64pk pr3(x + 6, AlignT());

			_load_trans(pr0, pr1, pr2, pr3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			const f64 *x2 = x + ldim * 2;

			sse_f64pk pr0(x, AlignT());
			sse_f64pk pr1(x + ldim, AlignT());
			sse_f64pk pr2(x2, AlignT());
			sse_f64pk pr3(x2 + ldim, AlignT());

			_load_trans(pr0, pr1, pr2, pr3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0l.store(x, AlignT());
			m_pk0h.store(x + 2, AlignT());
			m_pk1l.store(x + 4, AlignT());
			m_pk1h.store(x + 6, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x1 = x + ldim;

			m_pk0l.store(x, AlignT());
			m_pk0h.store(x + 2, AlignT());
			m_pk1l.store(x1, AlignT());
			m_pk1h.store(x1 + 2, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0l = add(m_pk0l, r.m_pk0l);
			o.m_pk0h = add(m_pk0h, r.m_pk0h);
			o.m_pk1l = add(m_pk1l, r.m_pk1l);
			o.m_pk1h = add(m_pk1h, r.m_pk1h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0l = sub(m_pk0l, r.m_pk0l);
			o.m_pk0h = sub(m_pk0h, r.m_pk0h);
			o.m_pk1l = sub(m_pk1l, r.m_pk1l);
			o.m_pk1h = sub(m_pk1h, r.m_pk1h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0l = mul(m_pk0l, r.m_pk0l);
			o.m_pk0h = mul(m_pk0h, r.m_pk0h);
			o.m_pk1l = mul(m_pk1l, r.m_pk1l);
			o.m_pk1h = mul(m_pk1h, r.m_pk1h);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0l = add(m_pk0l, r.m_pk0l);
			m_pk0h = add(m_pk0h, r.m_pk0h);
			m_pk1l = add(m_pk1l, r.m_pk1l);
			m_pk1h = add(m_pk1h, r.m_pk1h);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0l = sub(m_pk0l, r.m_pk0l);
			m_pk0h = sub(m_pk0h, r.m_pk0h);
			m_pk1l = sub(m_pk1l, r.m_pk1l);
			m_pk1h = sub(m_pk1h, r.m_pk1h);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0l = mul(m_pk0l, r.m_pk0l);
			m_pk0h = mul(m_pk0h, r.m_pk0h);
			m_pk1l = mul(m_pk1l, r.m_pk1l);
			m_pk1h = mul(m_pk1h, r.m_pk1h);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 4> transform(sse_vec<f64, 2> v) const
		{
			sse_f64pk v0 = v.m_pk.broadcast<0>();
			sse_f64pk p0l = mul(m_pk0l, v0);
			sse_f64pk p0h = mul(m_pk0h, v0);

			sse_f64pk v1 = v.m_pk.broadcast<1>();
			sse_f64pk p1l = mul(m_pk1l, v1);
			sse_f64pk p1h = mul(m_pk1h, v1);

			sse_f64pk pl = add(p0l, p1l);
			sse_f64pk ph = add(p0h, p1h);

			return sse_vec<f64, 4>(pl, ph);
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0l.e[0] + m_pk1l.e[1];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0l.test_equal(r[0], r[1]) &&
					m_pk0h.test_equal(r[2], r[3]) &&
					m_pk1l.test_equal(r[4], r[5]) &&
					m_pk1h.test_equal(r[6], r[7]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4 x 2]:\n");

			std::printf("    m_pk0 = ");
			m_pk0l.dump(fmt);
			std::printf(" ");
			m_pk0h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk1 = ");
			m_pk1l.dump(fmt);
			std::printf(" ");
			m_pk1h.dump(fmt);
			std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(sse_f64pk pr0, sse_f64pk pr1, sse_f64pk pr2, sse_f64pk pr3)
		{
			m_pk0l = unpack_low (pr0, pr1);
			m_pk0h = unpack_low (pr2, pr3);
			m_pk1l = unpack_high(pr0, pr1);
			m_pk1h = unpack_high(pr2, pr3);
		}

	};



	/********************************************
	 *
	 *  4 x 3
	 *
	 ********************************************/

	template<> class smat<f64, 4, 3>
	{
	public:
		sse_f64pk m_pk0l;
		sse_f64pk m_pk0h;
		sse_f64pk m_pk1l;
		sse_f64pk m_pk1h;
		sse_f64pk m_pk2l;
		sse_f64pk m_pk2h;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0l, sse_f64pk pk0h, sse_f64pk pk1l, sse_f64pk pk1h,
				sse_f64pk pk2l, sse_f64pk pk2h)
		: m_pk0l(pk0l), m_pk0h(pk0h), m_pk1l(pk1l), m_pk1h(pk1h)
		, m_pk2l(pk2l), m_pk2h(pk2h){ }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0l( zero_t() ), m_pk0h( zero_t() ), m_pk1l( zero_t() ), m_pk1h( zero_t() )
		, m_pk2l( zero_t() ), m_pk2h( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0l.load(x,     AlignT());
			m_pk0h.load(x + 2, AlignT());
			m_pk1l.load(x + 4, AlignT());
			m_pk1h.load(x + 6, AlignT());
			m_pk2l.load(x + 8, AlignT());
			m_pk2h.load(x + 10, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;

			m_pk0l.load(x, AlignT());
			m_pk0h.load(x + 2, AlignT());

			m_pk1l.load(x1, AlignT());
			m_pk1h.load(x1 + 2, AlignT());

			m_pk2l.load(x2, AlignT());
			m_pk2h.load(x2 + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			sse_f64pk l0, l1, l2, l3;
			sse_f64pk h0, h1, h2, h3;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x + 3, unaligned_t());
			h1.partial_load<1>(x + 5);

			l2.load(x + 6, AlignT());
			h2.partial_load<1>(x + 8);

			l3.load(x + 9, unaligned_t());
			h3.partial_load<1>(x + 11);

			m_pk0l = unpack_low(l0, l1);
			m_pk0h = unpack_low(l2, l3);

			m_pk1l = unpack_high(l0, l1);
			m_pk1h = unpack_high(l2, l3);

			m_pk2l = unpack_low(h0, h1);
			m_pk2h = unpack_low(h2, h3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			sse_f64pk l0, l1, l2, l3;
			sse_f64pk h0, h1, h2, h3;

			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;
			const f64 *x3 = x2 + ldim;

			l0.load(x, AlignT());
			h0.partial_load<1>(x + 2);

			l1.load(x1, AlignT());
			h1.partial_load<1>(x1 + 2);

			l2.load(x2, AlignT());
			h2.partial_load<1>(x2 + 2);

			l3.load(x3, AlignT());
			h3.partial_load<1>(x3 + 2);

			m_pk0l = unpack_low(l0, l1);
			m_pk0h = unpack_low(l2, l3);

			m_pk1l = unpack_high(l0, l1);
			m_pk1h = unpack_high(l2, l3);

			m_pk2l = unpack_low(h0, h1);
			m_pk2h = unpack_low(h2, h3);
		}


		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0l.store(x,     AlignT());
			m_pk0h.store(x + 2, AlignT());
			m_pk1l.store(x + 4, AlignT());
			m_pk1h.store(x + 6, AlignT());
			m_pk2l.store(x + 8, AlignT());
			m_pk2h.store(x + 10, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x1 = x + ldim;
			f64 *x2 = x1 + ldim;

			m_pk0l.store(x,      AlignT());
			m_pk0h.store(x + 2,  AlignT());
			m_pk1l.store(x1,     AlignT());
			m_pk1h.store(x1 + 2, AlignT());
			m_pk2l.store(x2,     AlignT());
			m_pk2h.store(x2 + 2, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0l = add(m_pk0l, r.m_pk0l);
			o.m_pk0h = add(m_pk0h, r.m_pk0h);
			o.m_pk1l = add(m_pk1l, r.m_pk1l);
			o.m_pk1h = add(m_pk1h, r.m_pk1h);
			o.m_pk2l = add(m_pk2l, r.m_pk2l);
			o.m_pk2h = add(m_pk2h, r.m_pk2h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0l = sub(m_pk0l, r.m_pk0l);
			o.m_pk0h = sub(m_pk0h, r.m_pk0h);
			o.m_pk1l = sub(m_pk1l, r.m_pk1l);
			o.m_pk1h = sub(m_pk1h, r.m_pk1h);
			o.m_pk2l = sub(m_pk2l, r.m_pk2l);
			o.m_pk2h = sub(m_pk2h, r.m_pk2h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0l = mul(m_pk0l, r.m_pk0l);
			o.m_pk0h = mul(m_pk0h, r.m_pk0h);
			o.m_pk1l = mul(m_pk1l, r.m_pk1l);
			o.m_pk1h = mul(m_pk1h, r.m_pk1h);
			o.m_pk2l = mul(m_pk2l, r.m_pk2l);
			o.m_pk2h = mul(m_pk2h, r.m_pk2h);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0l = add(m_pk0l, r.m_pk0l);
			m_pk0h = add(m_pk0h, r.m_pk0h);
			m_pk1l = add(m_pk1l, r.m_pk1l);
			m_pk1h = add(m_pk1h, r.m_pk1h);
			m_pk2l = add(m_pk2l, r.m_pk2l);
			m_pk2h = add(m_pk2h, r.m_pk2h);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0l = sub(m_pk0l, r.m_pk0l);
			m_pk0h = sub(m_pk0h, r.m_pk0h);
			m_pk1l = sub(m_pk1l, r.m_pk1l);
			m_pk1h = sub(m_pk1h, r.m_pk1h);
			m_pk2l = sub(m_pk2l, r.m_pk2l);
			m_pk2h = sub(m_pk2h, r.m_pk2h);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0l = mul(m_pk0l, r.m_pk0l);
			m_pk0h = mul(m_pk0h, r.m_pk0h);
			m_pk1l = mul(m_pk1l, r.m_pk1l);
			m_pk1h = mul(m_pk1h, r.m_pk1h);
			m_pk2l = mul(m_pk2l, r.m_pk2l);
			m_pk2h = mul(m_pk2h, r.m_pk2h);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 4> transform(sse_vec<f64, 3> v) const
		{
			sse_f64pk v0 = v.m_pk0.broadcast<0>();
			sse_f64pk p0l = mul(m_pk0l, v0);
			sse_f64pk p0h = mul(m_pk0h, v0);

			sse_f64pk v1 = v.m_pk0.broadcast<1>();
			sse_f64pk p1l = mul(m_pk1l, v1);
			sse_f64pk p1h = mul(m_pk1h, v1);

			sse_f64pk v2 = v.m_pk1.broadcast<0>();
			sse_f64pk p2l = mul(m_pk2l, v2);
			sse_f64pk p2h = mul(m_pk2h, v2);

			sse_f64pk pl = add(add(p0l, p1l), p2l);
			sse_f64pk ph = add(add(p0h, p1h), p2h);

			return sse_vec<f64, 4>(pl, ph);
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0l.e[0] + m_pk1l.e[1] + m_pk2h.e[0];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0l.test_equal(r[0], r[1]) &&
					m_pk0h.test_equal(r[2], r[3]) &&
					m_pk1l.test_equal(r[4], r[5]) &&
					m_pk1h.test_equal(r[6], r[7]) &&
					m_pk2l.test_equal(r[8], r[9]) &&
					m_pk2h.test_equal(r[10], r[11]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4 x 3]:\n");

			std::printf("    m_pk0 = ");
			m_pk0l.dump(fmt);
			std::printf(" ");
			m_pk0h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk1 = ");
			m_pk1l.dump(fmt);
			std::printf(" ");
			m_pk1h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk2 = ");
			m_pk2l.dump(fmt);
			std::printf(" ");
			m_pk2h.dump(fmt);
			std::printf("\n");
		}

	};



	/********************************************
	 *
	 *  4 x 4
	 *
	 ********************************************/

	template<> class smat<f64, 4, 4>
	{
	public:
		sse_f64pk m_pk0l;
		sse_f64pk m_pk0h;
		sse_f64pk m_pk1l;
		sse_f64pk m_pk1h;
		sse_f64pk m_pk2l;
		sse_f64pk m_pk2h;
		sse_f64pk m_pk3l;
		sse_f64pk m_pk3h;

		LSIMD_ENSURE_INLINE
		smat(sse_f64pk pk0l, sse_f64pk pk0h, sse_f64pk pk1l, sse_f64pk pk1h,
				sse_f64pk pk2l, sse_f64pk pk2h, sse_f64pk pk3l, sse_f64pk pk3h)
		: m_pk0l(pk0l), m_pk0h(pk0h), m_pk1l(pk1l), m_pk1h(pk1h)
		, m_pk2l(pk2l), m_pk2h(pk2h), m_pk3l(pk3l), m_pk3h(pk3h) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t )
		: m_pk0l( zero_t() ), m_pk0h( zero_t() ), m_pk1l( zero_t() ), m_pk1h( zero_t() )
		, m_pk2l( zero_t() ), m_pk2h( zero_t() ), m_pk3l( zero_t() ), m_pk3h( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, AlignT)
		{
			m_pk0l.load(x,      AlignT());
			m_pk0h.load(x + 2,  AlignT());
			m_pk1l.load(x + 4,  AlignT());
			m_pk1h.load(x + 6,  AlignT());
			m_pk2l.load(x + 8,  AlignT());
			m_pk2h.load(x + 10, AlignT());
			m_pk3l.load(x + 12, AlignT());
			m_pk3h.load(x + 14, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f64 *x, int ldim, AlignT)
		{
			const f64 *x1 = x + ldim;
			const f64 *x2 = x1 + ldim;
			const f64 *x3 = x2 + ldim;

			m_pk0l.load(x,     AlignT());
			m_pk0h.load(x + 2, AlignT());

			m_pk1l.load(x1,     AlignT());
			m_pk1h.load(x1 + 2, AlignT());

			m_pk2l.load(x2,     AlignT());
			m_pk2h.load(x2 + 2, AlignT());

			m_pk3l.load(x3,     AlignT());
			m_pk3h.load(x3 + 2, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, AlignT)
		{
			_load_trans(x, x + 4, x + 8, x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f64 *x, int ldim, AlignT)
		{
			const f64 *x2 = x + ldim * 2;
			_load_trans(x, x + ldim, x2, x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, AlignT) const
		{
			m_pk0l.store(x,     AlignT());
			m_pk0h.store(x + 2, AlignT());
			m_pk1l.store(x + 4, AlignT());
			m_pk1h.store(x + 6, AlignT());
			m_pk2l.store(x + 8, AlignT());
			m_pk2h.store(x + 10, AlignT());
			m_pk3l.store(x + 12, AlignT());
			m_pk3h.store(x + 14, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f64 *x, int ldim, AlignT) const
		{
			f64 *x1 = x + ldim;
			f64 *x2 = x1 + ldim;
			f64 *x3 = x2 + ldim;

			m_pk0l.store(x,      AlignT());
			m_pk0h.store(x  + 2, AlignT());
			m_pk1l.store(x1,     AlignT());
			m_pk1h.store(x1 + 2, AlignT());
			m_pk2l.store(x2,     AlignT());
			m_pk2h.store(x2 + 2, AlignT());
			m_pk3l.store(x3,     AlignT());
			m_pk3h.store(x3 + 2, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			smat o;
			o.m_pk0l = add(m_pk0l, r.m_pk0l);
			o.m_pk0h = add(m_pk0h, r.m_pk0h);
			o.m_pk1l = add(m_pk1l, r.m_pk1l);
			o.m_pk1h = add(m_pk1h, r.m_pk1h);
			o.m_pk2l = add(m_pk2l, r.m_pk2l);
			o.m_pk2h = add(m_pk2h, r.m_pk2h);
			o.m_pk3l = add(m_pk3l, r.m_pk3l);
			o.m_pk3h = add(m_pk3h, r.m_pk3h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			smat o;
			o.m_pk0l = sub(m_pk0l, r.m_pk0l);
			o.m_pk0h = sub(m_pk0h, r.m_pk0h);
			o.m_pk1l = sub(m_pk1l, r.m_pk1l);
			o.m_pk1h = sub(m_pk1h, r.m_pk1h);
			o.m_pk2l = sub(m_pk2l, r.m_pk2l);
			o.m_pk2h = sub(m_pk2h, r.m_pk2h);
			o.m_pk3l = sub(m_pk3l, r.m_pk3l);
			o.m_pk3h = sub(m_pk3h, r.m_pk3h);
			return o;
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			smat o;
			o.m_pk0l = mul(m_pk0l, r.m_pk0l);
			o.m_pk0h = mul(m_pk0h, r.m_pk0h);
			o.m_pk1l = mul(m_pk1l, r.m_pk1l);
			o.m_pk1h = mul(m_pk1h, r.m_pk1h);
			o.m_pk2l = mul(m_pk2l, r.m_pk2l);
			o.m_pk2h = mul(m_pk2h, r.m_pk2h);
			o.m_pk3l = mul(m_pk3l, r.m_pk3l);
			o.m_pk3h = mul(m_pk3h, r.m_pk3h);
			return o;
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk0l = add(m_pk0l, r.m_pk0l);
			m_pk0h = add(m_pk0h, r.m_pk0h);
			m_pk1l = add(m_pk1l, r.m_pk1l);
			m_pk1h = add(m_pk1h, r.m_pk1h);
			m_pk2l = add(m_pk2l, r.m_pk2l);
			m_pk2h = add(m_pk2h, r.m_pk2h);
			m_pk3l = add(m_pk3l, r.m_pk3l);
			m_pk3h = add(m_pk3h, r.m_pk3h);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk0l = sub(m_pk0l, r.m_pk0l);
			m_pk0h = sub(m_pk0h, r.m_pk0h);
			m_pk1l = sub(m_pk1l, r.m_pk1l);
			m_pk1h = sub(m_pk1h, r.m_pk1h);
			m_pk2l = sub(m_pk2l, r.m_pk2l);
			m_pk2h = sub(m_pk2h, r.m_pk2h);
			m_pk3l = sub(m_pk3l, r.m_pk3l);
			m_pk3h = sub(m_pk3h, r.m_pk3h);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk0l = mul(m_pk0l, r.m_pk0l);
			m_pk0h = mul(m_pk0h, r.m_pk0h);
			m_pk1l = mul(m_pk1l, r.m_pk1l);
			m_pk1h = mul(m_pk1h, r.m_pk1h);
			m_pk2l = mul(m_pk2l, r.m_pk2l);
			m_pk2h = mul(m_pk2h, r.m_pk2h);
			m_pk3l = mul(m_pk3l, r.m_pk3l);
			m_pk3h = mul(m_pk3h, r.m_pk3h);
		}

		LSIMD_ENSURE_INLINE sse_vec<f64, 4> transform(sse_vec<f64, 4> v) const
		{
			sse_f64pk v0 = v.m_pk0.broadcast<0>();
			sse_f64pk p0l = mul(m_pk0l, v0);
			sse_f64pk p0h = mul(m_pk0h, v0);

			sse_f64pk v1 = v.m_pk0.broadcast<1>();
			sse_f64pk p1l = mul(m_pk1l, v1);
			sse_f64pk p1h = mul(m_pk1h, v1);

			sse_f64pk v2 = v.m_pk1.broadcast<0>();
			sse_f64pk p2l = mul(m_pk2l, v2);
			sse_f64pk p2h = mul(m_pk2h, v2);

			sse_f64pk v3 = v.m_pk1.broadcast<1>();
			sse_f64pk p3l = mul(m_pk3l, v3);
			sse_f64pk p3h = mul(m_pk3h, v3);

			p0l = add(p0l, p1l);
			p0h = add(p0h, p1h);
			p2l = add(p2l, p3l);
			p2h = add(p2h, p3h);

			sse_f64pk pl = add(p0l, p2l);
			sse_f64pk ph = add(p0h, p2h);

			return sse_vec<f64, 4>(pl, ph);
 		}

	public:
		LSIMD_ENSURE_INLINE f64 trace() const
		{
			return m_pk0l.e[0] + m_pk1l.e[1] + m_pk2h.e[0] + m_pk3h.e[1];
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return  m_pk0l.test_equal(r[0], r[1]) &&
					m_pk0h.test_equal(r[2], r[3]) &&
					m_pk1l.test_equal(r[4], r[5]) &&
					m_pk1h.test_equal(r[6], r[7]) &&
					m_pk2l.test_equal(r[8], r[9]) &&
					m_pk2h.test_equal(r[10], r[11]) &&
					m_pk3l.test_equal(r[12], r[13]) &&
					m_pk3h.test_equal(r[14], r[15]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4 x 4]:\n");

			std::printf("    m_pk0 = ");
			m_pk0l.dump(fmt);
			std::printf(" ");
			m_pk0h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk1 = ");
			m_pk1l.dump(fmt);
			std::printf(" ");
			m_pk1h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk2 = ");
			m_pk2l.dump(fmt);
			std::printf(" ");
			m_pk2h.dump(fmt);
			std::printf("\n");

			std::printf("    m_pk3 = ");
			m_pk3l.dump(fmt);
			std::printf(" ");
			m_pk3h.dump(fmt);
			std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f64 *r0, const f64 *r1, const f64 *r2, const f64 *r3, AlignT)
		{
			sse_f64pk l0(r0,     AlignT());
			sse_f64pk h0(r0 + 2, AlignT());
			sse_f64pk l1(r1,     AlignT());
			sse_f64pk h1(r1 + 2, AlignT());
			sse_f64pk l2(r2,     AlignT());
			sse_f64pk h2(r2 + 2, AlignT());
			sse_f64pk l3(r3,     AlignT());
			sse_f64pk h3(r3 + 2, AlignT());

			m_pk0l = unpack_low(l0, l1);
			m_pk0h = unpack_low(l2, l3);

			m_pk1l = unpack_high(l0, l1);
			m_pk1h = unpack_high(l2, l3);

			m_pk2l = unpack_low(h0, h1);
			m_pk2h = unpack_low(h2, h3);

			m_pk3l = unpack_high(h0, h1);
			m_pk3h = unpack_high(h2, h3);
		}

	};



} }


#endif /* SSE_MAT_BITS_F64_H_ */
