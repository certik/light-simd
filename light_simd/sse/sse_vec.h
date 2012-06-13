/**
 * @file sse_vec.h
 *
 * SSE vectors of fixed length
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_VEC_H_
#define LSIMD_SSE_VEC_H_

#include "sse_arith.h"

namespace lsimd
{


	template<typename T, int N> class sse_vec;


	/********************************************
	 *
	 *  sse_vec class for f32
	 *
	 ********************************************/

	template<> class sse_vec<f32, 1>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f32pk p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0)
		{
			m_pk.v = _mm_set_ss(e0);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(_mm_add_ss(m_pk.v, rhs.m_pk.v));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(_mm_sub_ss(m_pk.v, rhs.m_pk.v));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(_mm_mul_ss(m_pk.v, rhs.m_pk.v));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk.v = _mm_add_ss(m_pk.v, rhs.m_pk.v);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk.v = _mm_sub_ss(m_pk.v, rhs.m_pk.v);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk.v = _mm_mul_ss(m_pk.v, rhs.m_pk.v);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.to_scalar();
		}

		LSIMD_ENSURE_INLINE f32 dot(sse_vec rhs) const
		{
			return _mm_cvtss_f32(_mm_mul_ss(m_pk.v, rhs.m_pk.v));
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], 0.f, 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [1]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};


	template<> class sse_vec<f32, 2>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f32pk p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0, const f32 e1)
		{
			m_pk.v = _mm_setr_ps(e0, e1, 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<2>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.partial_store<2>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.partial_store<2>(x);
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(add(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(sub(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(mul(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk = add(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk = sub(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk = mul(m_pk, rhs.m_pk);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.partial_sum<2>();
		}

		LSIMD_ENSURE_INLINE f32 dot(sse_vec rhs) const
		{
			return mul(m_pk, rhs.m_pk).partial_sum<2>();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], 0.f, 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [2]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};


	template<> class sse_vec<f32, 3>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f32pk p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0, const f32 e1, const f32 e2)
		{
			m_pk.v = _mm_setr_ps(e0, e1, e2, 0.f);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.partial_load<3>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.partial_store<3>(x);
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.partial_store<3>(x);
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(add(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(sub(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(mul(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk = add(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk = sub(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk = mul(m_pk, rhs.m_pk);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.partial_sum<3>();
		}

		LSIMD_ENSURE_INLINE f32 dot(sse_vec rhs) const
		{
			return mul(m_pk, rhs.m_pk).partial_sum<3>();
		}


	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], r[2], 0.f);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [3]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};


	template<> class sse_vec<f32, 4>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128 p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f32pk p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f32 e0, const f32 e1, const f32 e2, const f32 e3)
		{
			m_pk.v = _mm_setr_ps(e0, e1, e2, e3);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f32 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f32 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, aligned_t) const
		{
			m_pk.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f32 *x, unaligned_t) const
		{
			m_pk.store(x, unaligned_t());
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(add(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(sub(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(mul(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk = add(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk = sub(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk = mul(m_pk, rhs.m_pk);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f32 sum() const
		{
			return m_pk.sum();
		}

		LSIMD_ENSURE_INLINE f32 dot(sse_vec rhs) const
		{
			return mul(m_pk, rhs.m_pk).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f32 *r) const
		{
			return m_pk.test_equal(r[0], r[1], r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f32 [4]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f32pk m_pk;
	};



	/********************************************
	 *
	 *  sse_vec class for f64
	 *
	 ********************************************/

	template<> class sse_vec<f64, 1>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128d p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f64pk p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0)
		{
			m_pk.v = _mm_set_sd(e0);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			m_pk.partial_load<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			m_pk.partial_store<1>(x);
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(_mm_add_sd(m_pk.v, rhs.m_pk.v));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(_mm_sub_sd(m_pk.v, rhs.m_pk.v));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(_mm_mul_sd(m_pk.v, rhs.m_pk.v));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk.v = _mm_add_sd(m_pk.v, rhs.m_pk.v);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk.v = _mm_sub_sd(m_pk.v, rhs.m_pk.v);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk.v = _mm_mul_sd(m_pk.v, rhs.m_pk.v);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return m_pk.to_scalar();
		}

		LSIMD_ENSURE_INLINE f64 dot(sse_vec rhs) const
		{
			return _mm_cvtsd_f64(_mm_mul_sd(m_pk.v, rhs.m_pk.v));
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk.test_equal(r[0], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [1]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f64pk m_pk;
	};


	template<> class sse_vec<f64, 2>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(__m128d p) : m_pk(p) { }
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f64pk p) : m_pk(p) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0, const f64 e1)
		{
			m_pk.v = _mm_setr_pd(e0, e1);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			m_pk.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			m_pk.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			m_pk.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			m_pk.store(x, unaligned_t());
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(add(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(sub(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(mul(m_pk, rhs.m_pk));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk = add(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk = sub(m_pk, rhs.m_pk);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk = mul(m_pk, rhs.m_pk);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return m_pk.sum();
		}

		LSIMD_ENSURE_INLINE f64 dot(sse_vec rhs) const
		{
			return mul(m_pk, rhs.m_pk).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk.test_equal(r[0], r[1]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [2]:\n");
			std::printf("    m_pk = "); m_pk.dump(fmt); std::printf("\n");
		}

	public:
		sse_f64pk m_pk;
	};


	template<> class sse_vec<f64, 3>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f64pk pk0, sse_f64pk pk1)
		: m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0, const f64 e1, const f64 e2)
		{
			m_pk0.v = _mm_setr_pd(e0, e1);
			m_pk1.v = _mm_set_sd(e2);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			_store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			_store(x, unaligned_t());
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(add(m_pk0, rhs.m_pk0), _mm_add_sd(m_pk1.v, rhs.m_pk1.v));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(sub(m_pk0, rhs.m_pk0), _mm_sub_sd(m_pk1.v, rhs.m_pk1.v));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(mul(m_pk0, rhs.m_pk0), _mm_mul_sd(m_pk1.v, rhs.m_pk1.v));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk0 = add(m_pk0, rhs.m_pk0);
			m_pk1 = _mm_add_sd(m_pk1.v, rhs.m_pk1.v);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk0 = sub(m_pk0, rhs.m_pk0);
			m_pk1 = _mm_sub_sd(m_pk1.v, rhs.m_pk1.v);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk0 = mul(m_pk0, rhs.m_pk0);
			m_pk1 = _mm_mul_sd(m_pk1.v, rhs.m_pk1.v);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return add(m_pk0, m_pk1).sum();
		}

		LSIMD_ENSURE_INLINE f64 dot(sse_vec rhs) const
		{
			return (operator %(rhs)).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk0.test_equal(r[0], r[1]) && m_pk1.test_equal(r[2], 0.0);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [3]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.partial_load<1>(x + 2);
		}


		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.partial_store<1>(x + 2);
		}

	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;
	};


	template<> class sse_vec<f64, 4>
	{
	public:
		LSIMD_ENSURE_INLINE explicit sse_vec(sse_f64pk pk0, sse_f64pk pk1)
		: m_pk0(pk0), m_pk1(pk1) { }

	public:
		LSIMD_ENSURE_INLINE sse_vec() { }

		LSIMD_ENSURE_INLINE sse_vec( zero_t ) : m_pk0( zero_t() ), m_pk1( zero_t() ) { }

		LSIMD_ENSURE_INLINE sse_vec(const f64 e0, const f64 e1, const f64 e2, const f64 e3)
		{
			m_pk0.v = _mm_setr_pd(e0, e1);
			m_pk1.v = _mm_setr_pd(e2, e3);
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE sse_vec(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, aligned_t)
		{
			_load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void load(const f64 *x, unaligned_t)
		{
			_load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, aligned_t) const
		{
			_store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE void store(f64 *x, unaligned_t) const
		{
			_store(x, unaligned_t());
		}

	public:

		LSIMD_ENSURE_INLINE sse_vec operator + (sse_vec rhs) const
		{
			return sse_vec(add(m_pk0, rhs.m_pk0), add(m_pk1, rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec operator - (sse_vec rhs) const
		{
			return sse_vec(sub(m_pk0, rhs.m_pk0), sub(m_pk1, rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec operator % (sse_vec rhs) const
		{
			return sse_vec(mul(m_pk0, rhs.m_pk0), mul(m_pk1, rhs.m_pk1));
		}

		LSIMD_ENSURE_INLINE sse_vec& operator += (sse_vec rhs)
		{
			m_pk0 = add(m_pk0, rhs.m_pk0);
			m_pk1 = add(m_pk1, rhs.m_pk1);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator -= (sse_vec rhs)
		{
			m_pk0 = sub(m_pk0, rhs.m_pk0);
			m_pk1 = sub(m_pk1, rhs.m_pk1);
			return *this;
		}

		LSIMD_ENSURE_INLINE sse_vec& operator %= (sse_vec rhs)
		{
			m_pk0 = mul(m_pk0, rhs.m_pk0);
			m_pk1 = mul(m_pk1, rhs.m_pk1);
			return *this;
		}

	public:
		LSIMD_ENSURE_INLINE f64 sum() const
		{
			return add(m_pk0, m_pk1).sum();
		}

		LSIMD_ENSURE_INLINE f64 dot(sse_vec rhs) const
		{
			return (operator %(rhs)).sum();
		}

	public:
		LSIMD_ENSURE_INLINE bool test_equal(const f64 *r) const
		{
			return m_pk0.test_equal(r[0], r[1]) && m_pk1.test_equal(r[2], r[3]);
		}

		LSIMD_ENSURE_INLINE void dump(const char *fmt) const
		{
			std::printf("f64 [4]:\n");
			std::printf("    m_pk0 = "); m_pk0.dump(fmt); std::printf("\n");
			std::printf("    m_pk1 = "); m_pk1.dump(fmt); std::printf("\n");
		}

	private:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _load(const f64 *x, AlignT)
		{
			m_pk0.load(x, AlignT());
			m_pk1.load(x + 2, AlignT());
		}


		template<typename AlignT>
		LSIMD_ENSURE_INLINE
		void _store(f64 *x, AlignT) const
		{
			m_pk0.store(x, AlignT());
			m_pk1.store(x + 2, AlignT());
		}

	public:
		sse_f64pk m_pk0;
		sse_f64pk m_pk1;
	};

}

#endif /* SSE_VEC_H_ */
