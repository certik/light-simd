/**
 * @file sse_mat_bits_f32.h
 *
 * The internal implementation of SSE matrix (for f32)
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SSE_MAT_BITS_F32_H_
#define SSE_MAT_BITS_F32_H_

#include "../sse_vec.h"

namespace lsimd { namespace sse {

	template<typename T, int M, int N> class smat;


	/********************************************
	 *
	 *  2 x 2
	 *
	 ********************************************/

	template<> class smat<f32, 2, 2>
	{
	private:
		sse_f32pk m_pk;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk) : m_pk(pk) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk.load(x, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);

			m_pk = merge_low(p0, p1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			sse_f32pk t(x, AlignT());
			m_pk = t.swizzle<0,2,1,3>();
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);

			m_pk = unpack_low(p0, p1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk.store(x, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			sse::partial_store<2>(x, m_pk.v);
			sse::partial_store<2>(x + ldim, m_pk.dup_high().v);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return add(m_pk, r.m_pk);
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return sub(m_pk, r.m_pk);
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return mul(m_pk, r.m_pk);
		}

		LSIMD_ENSURE_INLINE void inplace_madd(smat r)
		{
			m_pk = add(m_pk, r.m_pk);
		}

		LSIMD_ENSURE_INLINE void inplace_msub(smat r)
		{
			m_pk = sub(m_pk, r.m_pk);
		}

		LSIMD_ENSURE_INLINE void inplace_mmul(smat r)
		{
			m_pk = mul(m_pk, r.m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec<f32, 2> transform(sse_vec<f32, 2> v) const
		{
			sse_f32pk p = v.m_pk.swizzle<0, 0, 1, 1>();
			p = mul(p, m_pk);
			p = add(p, p.dup_low());
			return p.shift_front<2>();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2 x 2]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	};



	/********************************************
	 *
	 *  2 x 3
	 *
	 ********************************************/

	template<> class smat<f32, 2, 3>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1) : m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.partial_load<2>(x + 4);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);

			m_pk0 = merge_low(p0, p1);
			m_pk1.partial_load<2>(x + 2 * ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.partial_store<2>(x + 4);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			m_pk0.partial_store<2>(x);
			m_pk0.dup_high().partial_store<2>(x + ldim);
			m_pk1.partial_store<2>(x + 2 * ldim);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 2> transform(sse_vec<f32, 3> v) const
		{
			sse_f32pk p1 = unpack_low(v.m_pk, v.m_pk);
			sse_f32pk p2 = unpack_high(v.m_pk, v.m_pk);

			p1 = mul(p1, m_pk0);
			p2 = mul(p2, m_pk1);

			p1 = add(p1, p1.dup_low());
			return p1.shift_front<2>();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], r[3]) &&
					m_pk1.test_equal(r[4], r[5], 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2 x 3]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE void _load_trans(const f32 *r0, const f32 *r1)
		{
			sse_f32pk pr0a, pr0b, pr1a, pr1b;

			pr0a.partial_load<2>(r0);
			pr0b.partial_load<1>(r0 + 2);
			pr1a.partial_load<2>(r1);
			pr1b.partial_load<1>(r1 + 2);

			m_pk0 = unpack_low(pr0a, pr1a);
			m_pk1 = unpack_low(pr0b, pr1b);
		}
	};



	/********************************************
	 *
	 *  2 x 4
	 *
	 ********************************************/

	template<> class smat<f32, 2, 4>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1) : m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);
			p2.partial_load<2>(x2);
			p3.partial_load<2>(x2 + ldim);

			m_pk0 = merge_low(p0, p1);
			m_pk1 = merge_low(p2, p3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			f32 *x2 = x + 2 * ldim;

			m_pk0.partial_store<2>(x);
			m_pk0.dup_high().partial_store<2>(x + ldim);

			m_pk1.partial_store<2>(x2);
			m_pk1.dup_high().partial_store<2>(x2 + ldim);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 2> transform(sse_vec<f32, 4> v) const
		{
			sse_f32pk p1 = unpack_low(v.m_pk, v.m_pk);
			sse_f32pk p2 = unpack_high(v.m_pk, v.m_pk);

			p1 = mul(p1, m_pk0);
			p2 = mul(p2, m_pk1);

			p1 = add(p1, p1.dup_low());
			return p1.shift_front<2>();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], r[3]) &&
					m_pk1.test_equal(r[4], r[5], r[6], r[7]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2 x 4]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	private:

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void _load_trans(const f32 *r0, const f32 *r1, AlignT)
		{
			sse_f32pk pr0(r0, AlignT());
			sse_f32pk pr1(r1, AlignT());

			m_pk0 = unpack_low(pr0, pr1);
			m_pk1 = unpack_high(pr0, pr1);
		}
	};



	/********************************************
	 *
	 *  3 x 2
	 *
	 ********************************************/

	template<> class smat<f32, 3, 2>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1) : m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.partial_load<3>(x);
			m_pk1.partial_load<3>(x + 3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			m_pk0.partial_load<3>(x);
			m_pk1.partial_load<3>(x + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			sse_f32pk p01, p2;

			p01.load(x, AlignT());
			p2.partial_load<2>(x + 4);

			p01 = p01.swizzle<0,2,1,3>();
			p2 = p2.swizzle<0,2,1,3>();

			m_pk0 = merge_low (p01, p2);
			m_pk1 = merge_high(p01, p2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			sse_f32pk p0, p1, p2;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);
			p2.partial_load<2>(x + 2 * ldim);

			p0 = unpack_low(p0, p1);
			p2 = p2.swizzle<0,2,1,3>();

			m_pk0 = merge_low (p0, p2);
			m_pk1 = merge_high(p0, p2);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.partial_store<3>(x);
			m_pk1.partial_store<3>(x + 3);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			m_pk0.partial_store<3>(x);
			m_pk1.partial_store<3>(x + ldim);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 3> transform(sse_vec<f32, 2> v) const
		{
			sse_f32pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f32pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			return add(p0, p1);
 		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], 0.f) &&
					m_pk1.test_equal(r[3], r[4], r[5], 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3 x 2]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}
	};



	/********************************************
	 *
	 *  3 x 3
	 *
	 ********************************************/

	template<> class smat<f32, 3, 3>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;
		sse_f32pk m_pk2;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1, sse_f32pk pk2)
		: m_pk0(pk0), m_pk1(pk1), m_pk2(pk2) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ), m_pk2( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.partial_load<3>(x);
			m_pk1.partial_load<3>(x + 3);
			m_pk2.partial_load<3>(x + 6);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			m_pk0.partial_load<3>(x);
			m_pk1.partial_load<3>(x + ldim);
			m_pk2.partial_load<3>(x + 2 * ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3, x + 6);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			_load_trans(x, x + ldim, x + 2 * ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.partial_store<3>(x);
			m_pk1.partial_store<3>(x + 3);
			m_pk2.partial_store<3>(x + 6);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			m_pk0.partial_store<3>(x);
			m_pk1.partial_store<3>(x + ldim);
			m_pk2.partial_store<3>(x + 2 * ldim);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1), add(m_pk2, r.m_pk2));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1), sub(m_pk2, r.m_pk2));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1), mul(m_pk2, r.m_pk2));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 3> transform(sse_vec<f32, 3> v) const
		{
			sse_f32pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f32pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			sse_f32pk p2 = mul(m_pk2, v.m_pk.broadcast<2>());
			return add(add(p0, p1), p2);
 		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], 0.f) &&
					m_pk1.test_equal(r[3], r[4], r[5], 0.f) &&
					m_pk2.test_equal(r[6], r[7], r[8], 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3 x 3]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
			std::printf("    m_pk2 = "); m_pk2.dump(fmt); std::printf("\n");
		}

	private:
		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2)
		{
			sse_f32pk p0, p1, p2;

			p0.partial_load<2>(r0);
			p1.partial_load<2>(r1);
			p2.partial_load<2>(r2);

			sse_f32pk u0 = unpack_low(p0, p1);
			sse_f32pk u1 = p2.swizzle<0,2,1,3>();

			m_pk0 = merge_low (u0, u1);
			m_pk1 = merge_high(u0, u1);

			m_pk2.set(r0[2], r1[2], r2[2], 0.f);
		}
	};


	/********************************************
	 *
	 *  3 x 4
	 *
	 ********************************************/

	template<> class smat<f32, 3, 4>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;
		sse_f32pk m_pk2;
		sse_f32pk m_pk3;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1, sse_f32pk pk2, sse_f32pk pk3)
		: m_pk0(pk0), m_pk1(pk1), m_pk2(pk2), m_pk3(pk3) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ), m_pk2( zero_t() ), m_pk3( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.partial_load<3>(x);
			m_pk1.partial_load<3>(x + 3);
			m_pk2.partial_load<3>(x + 6);
			m_pk3.partial_load<3>(x + 9);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			m_pk0.partial_load<3>(x);
			m_pk1.partial_load<3>(x + ldim);
			m_pk2.partial_load<3>(x2);
			m_pk3.partial_load<3>(x2 + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3, x + 6, x + 9);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;
			_load_trans(x, x + ldim, x2, x2 + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.partial_store<3>(x);
			m_pk1.partial_store<3>(x + 3);
			m_pk2.partial_store<3>(x + 6);
			m_pk3.partial_store<3>(x + 9);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			f32 *x2 = x + 2 * ldim;

			m_pk0.partial_store<3>(x);
			m_pk1.partial_store<3>(x + ldim);
			m_pk2.partial_store<3>(x2);
			m_pk3.partial_store<3>(x2 + ldim);
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(
					add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1),
					add(m_pk2, r.m_pk2), add(m_pk3, r.m_pk3));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(
					sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1),
					sub(m_pk2, r.m_pk2), sub(m_pk3, r.m_pk3));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(
					mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1),
					mul(m_pk2, r.m_pk2), mul(m_pk3, r.m_pk3));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 3> transform(sse_vec<f32, 4> v) const
		{
			sse_f32pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f32pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			sse_f32pk p2 = mul(m_pk2, v.m_pk.broadcast<2>());
			sse_f32pk p3 = mul(m_pk3, v.m_pk.broadcast<3>());

			return add(add(p0, p1), add(p2, p3));
 		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], r[3]) &&
					m_pk1.test_equal(r[4], r[5], r[6], r[7]) &&
					m_pk2.test_equal(r[8], r[9], r[10], r[11]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3 x 4]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
			std::printf("    m_pk2 = "); m_pk2.dump(fmt); std::printf("\n");
		}

	private:

		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2, const f32 *r3)
		{
			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(r0);
			p1.partial_load<2>(r1);
			p2.partial_load<2>(r2);
			p3.partial_load<2>(r3);

			p0 = unpack_low(p0, p1);
			p2 = unpack_low(p2, p3);

			m_pk0 = merge_low(p0, p2);
			m_pk1 = merge_high(p0, p2);

			m_pk2.set(r0[2], r1[2], r2[2], r3[2]);
		}
	};



	/********************************************
	 *
	 *  4 x 2
	 *
	 ********************************************/

	template<> class smat<f32, 4, 2>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1) : m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			sse_f32pk p01, p23;

			p01.load(x, AlignT());
			p23.load(x + 4, AlignT());

			sse_f32pk u0 = unpack_low(p01, p23);
			sse_f32pk u1 = unpack_high(p01, p23);

			m_pk0 = unpack_low(u0, u1);
			m_pk1 = unpack_high(u0, u1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(x);
			p1.partial_load<2>(x + ldim);
			p2.partial_load<2>(x2);
			p3.partial_load<2>(x2 + ldim);

			sse_f32pk u0 = unpack_low(p0, p2);
			sse_f32pk u1 = unpack_low(p1, p3);

			m_pk0 = unpack_low(u0, u1);
			m_pk1 = unpack_high(u0, u1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 4, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + ldim, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 4> transform(sse_vec<f32, 2> v) const
		{
			sse_f32pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f32pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			return add(p0, p1);
 		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], r[3]) &&
					m_pk1.test_equal(r[4], r[5], r[6], r[7]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4 x 2]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}
	};



	/********************************************
	 *
	 *  4 x 3
	 *
	 ********************************************/

	template<> class smat<f32, 4, 3>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;
		sse_f32pk m_pk2;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1, sse_f32pk pk2)
		: m_pk0(pk0), m_pk1(pk1), m_pk2(pk2) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ), m_pk2( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 4, AlignT());
			m_pk2.load(x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + ldim, AlignT());
			m_pk2.load(x + 2 * ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3, x + 6, x + 9);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;
			_load_trans(x, x + ldim, x2, x2 + ldim);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 4, AlignT());
			m_pk2.store(x + 8, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + ldim, AlignT());
			m_pk2.store(x + 2 * ldim, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1), add(m_pk2, r.m_pk2));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1), sub(m_pk2, r.m_pk2));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1), mul(m_pk2, r.m_pk2));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 4> transform(sse_vec<f32, 3> v) const
		{
			sse_f32pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f32pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			sse_f32pk p2 = mul(m_pk2, v.m_pk.broadcast<2>());
			return add(add(p0, p1), p2);
 		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], r[3]) &&
					m_pk1.test_equal(r[4], r[5], r[6], r[7]) &&
					m_pk2.test_equal(r[8], r[9], r[10], r[11]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4 x 3]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
			std::printf("    m_pk2 = "); m_pk2.dump(fmt); std::printf("\n");
		}

	private:

		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2, const f32 * r3)
		{
			sse_f32pk p0, p1, p2, p3;

			p0.partial_load<2>(r0);
			p1.partial_load<2>(r1);
			p2.partial_load<2>(r2);
			p3.partial_load<3>(r3);

			p0 = unpack_low(p0, p1);
			p2 = unpack_low(p2, p3);

			m_pk0 = merge_low (p0, p2);
			m_pk1 = merge_high(p0, p2);

			m_pk2.set(r0[2], r1[2], r2[2], r3[2]);
		}
	};


	/********************************************
	 *
	 *  4 x 4
	 *
	 ********************************************/

	template<> class smat<f32, 4, 4>
	{
	private:
		sse_f32pk m_pk0;
		sse_f32pk m_pk1;
		sse_f32pk m_pk2;
		sse_f32pk m_pk3;

		LSIMD_ENSURE_INLINE
		smat(sse_f32pk pk0, sse_f32pk pk1, sse_f32pk pk2, sse_f32pk pk3)
		: m_pk0(pk0), m_pk1(pk1), m_pk2(pk2), m_pk3(pk3) { }

	public:
		LSIMD_ENSURE_INLINE
		smat() { }

		LSIMD_ENSURE_INLINE
		smat( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ), m_pk2( zero_t() ) { }

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 4, AlignT());
			m_pk2.load(x + 8, AlignT());
			m_pk3.load(x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;

			m_pk0.load(x, AlignT());
			m_pk1.load(x + ldim, AlignT());
			m_pk2.load(x2, AlignT());
			m_pk3.load(x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, AlignT)
		{
			_load_trans(x, x + 3, x + 6, x + 9, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load_trans(const f32 *x, int ldim, AlignT)
		{
			const f32 *x2 = x + 2 * ldim;
			_load_trans(x, x + ldim, x2, x2 + ldim, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 4, AlignT());
			m_pk2.store(x + 8, AlignT());
			m_pk3.store(x + 12, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			f32 *x2 = x + 2 * ldim;

			m_pk0.store(x, AlignT());
			m_pk1.store(x + ldim, AlignT());
			m_pk2.store(x2, AlignT());
			m_pk3.store(x2 + ldim, AlignT());
		}

	public:
		LSIMD_ENSURE_INLINE smat madd(smat r) const
		{
			return smat(
					add(m_pk0, r.m_pk0), add(m_pk1, r.m_pk1),
					add(m_pk2, r.m_pk2), add(m_pk3, r.m_pk3));
		}

		LSIMD_ENSURE_INLINE smat msub(smat r) const
		{
			return smat(
					sub(m_pk0, r.m_pk0), sub(m_pk1, r.m_pk1),
					sub(m_pk2, r.m_pk2), sub(m_pk3, r.m_pk3));
		}

		LSIMD_ENSURE_INLINE smat mmul(smat r) const
		{
			return smat(
					mul(m_pk0, r.m_pk0), mul(m_pk1, r.m_pk1),
					mul(m_pk2, r.m_pk2), mul(m_pk3, r.m_pk3));
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

		LSIMD_ENSURE_INLINE sse_vec<f32, 4> transform(sse_vec<f32, 4> v) const
		{
			sse_f32pk p0 = mul(m_pk0, v.m_pk.broadcast<0>());
			sse_f32pk p1 = mul(m_pk1, v.m_pk.broadcast<1>());
			sse_f32pk p2 = mul(m_pk2, v.m_pk.broadcast<2>());
			sse_f32pk p3 = mul(m_pk3, v.m_pk.broadcast<3>());

			return add(add(p0, p1), add(p2, p3));
 		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return  m_pk0.test_equal(r[0], r[1], r[2], r[3]) &&
					m_pk1.test_equal(r[4], r[5], r[6], r[7]) &&
					m_pk2.test_equal(r[8], r[9], r[10], r[11]) &&
					m_pk3.test_equal(r[12], r[13], r[14], r[15]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4 x 4]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
			std::printf("    m_pk2 = "); m_pk2.dump(fmt); std::printf("\n");
			std::printf("    m_pk3 = "); m_pk3.dump(fmt); std::printf("\n");
		}

	private:

		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load_trans(const f32 *r0, const f32 *r1, const f32 *r2, const f32 *r3, AlignT)
		{
			sse_f32pk p0, p1, p2, p3;

			p0.load(r0, AlignT());
			p1.load(r1, AlignT());
			p2.load(r2, AlignT());
			p3.load(r3, AlignT());

			sse_f32pk u0l = unpack_low (p0, p1);
			sse_f32pk u0h = unpack_high(p0, p1);
			sse_f32pk u1l = unpack_low (p2, p3);
			sse_f32pk u1h = unpack_high(p2, p3);

			m_pk0 = merge_low (u0l, u1l);
			m_pk1 = merge_high(u0l, u1l);
			m_pk2 = merge_low (u0h, u1h);
			m_pk3 = merge_high(u0h, u1h);
		}
	};

} }

#endif /* SSE_MAT_BITS_F32_H_ */



