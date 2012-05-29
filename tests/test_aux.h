/**
 * @file test_aux.h
 *
 * Auxiliary facilities for testing
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_TEST_AUX_H_
#define LSIMD_TEST_AUX_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <light_simd/simd.h>

namespace lsimd
{
	template<typename T>
	inline void clear_zeros(int n, T *a)
	{
		for (int i = 0; i < n; ++i) a[i] = T(0);
	}

	template<typename T>
	inline T rand_val(const T lb, const T ub)
	{
		double r = double(std::rand()) / RAND_MAX;
		r = double(lb) + r * double(ub - lb);
		return T(r);
	}

	template<typename T>
	inline void fill_rand(int n, T *a, T lb, T ub)
	{
		for (int i = 0; i < n; ++i)
		{
			a[i] = rand_val(lb, ub);
		}
	}

	template<typename T>
	inline bool test_equal(int n, const T *a, const T *b)
	{
		for (int i = 0; i < n; ++i)
		{
			if (a[i] != b[i]) return false;
		}
		return true;
	}



	template<typename T, typename Kind>
	struct simd_array
	{
	public:
		typedef simd_vec<T, Kind> vec_t;
		static const unsigned pack_width = simd<T, Kind>::pack_width;

		simd_array(int n)
		: m_n(n), m_nv(m_n / pack_width), m_data(alloc(n))
		{ }

		simd_array(const simd_array& a)
		: m_n(a.m_n), m_nv(m_n / pack_width), m_data(alloc(a.m_n))
		{
			::memcpy(m_data, a.m_data, sizeof(T) * size_t(m_n));
		}

		~simd_array()
		{
			dealloc(m_data);
		}

		int nelems() const { return m_n; }

		const T *data() const { return m_data; }

		T *data() { return m_data; }

		void set_zeros()
		{
			::memset(m_data, 0, sizeof(T) * size_t(m_n));
		}

		void set_rand(T lb, T ub)
		{
			fill_rand(m_n, m_data, lb, ub);
		}

		LSIMD_ENSURE_INLINE
		T operator[] (int i) const { return m_data[i]; }

		LSIMD_ENSURE_INLINE
		T& operator[] (int i) { return m_data[i]; }

		LSIMD_ENSURE_INLINE
		vec_t get_pack(int i)
		{
			return vec_t(m_data + i * pack_width, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void set_pack(int i, vec_t p)
		{
			p.store(m_data + i * pack_width, aligned_t());
		}

	private:
		int m_n;
		int m_nv;
		T *m_data;

	private:
		simd_array& operator = (const simd_array& );

		static T *alloc(int n)
		{
#if (LSIMD_COMPILER == LSIMD_MSVC)
			return (T*)::_aligned_malloc(size_t(n) * sizeof(T), 64));

#else
			char* p = 0;
			::posix_memalign((void**)(&p), 64, size_t(n) * sizeof(T));
			return (T*)p;
#endif
		}

		static void dealloc(T *p)
		{
#if (LSIMD_COMPILER == LSIMD_MSVC)
			::_aligned_free(p);
#else
			::free(p);
#endif
		}

	};


	template<typename T, typename Kind, class Op>
	double eval_approx_accuracy(unsigned n, const T lb_a, const T ub_a)
	{
		double max_dev = 0.0;
		const unsigned w = simd<T, Kind>::pack_width;
		LSIMD_ALIGN_SSE T src[w];
		LSIMD_ALIGN_SSE T dst[w];

		for (unsigned k = 0; k < n; ++k)
		{
			simd_vec<T, Kind> a;

			for (unsigned i = 0; i < w; ++i)
			{
				src[i] = rand_val(lb_a, ub_a);
			}

			a.load(src, aligned_t());

			LSIMD_ALIGN_SSE T r0[w];

			for (unsigned i = 0; i < w; ++i)
			{
				r0[i] = Op::eval_scalar(src[i]);
			}

			simd_vec<T, Kind> r = Op::eval_vector(a);
			r.store(dst, aligned_t());

			for (unsigned i = 0; i < w; ++i)
			{
				double cdev = std::fabs(double(dst[i]) - double(r0[i])) / double(r0[i]);
				if (cdev > max_dev) max_dev = cdev;
			}
		}

		return max_dev;
	}

	template<typename T, typename Kind, class Op>
	double eval_approx_accuracy(unsigned n,
			const T lb_a, const T ub_a,
			const T lb_b, const T ub_b)
	{
		double max_dev = 0.0;
		const unsigned w = simd<T, Kind>::pack_width;
		LSIMD_ALIGN_SSE T sa[w];
		LSIMD_ALIGN_SSE T sb[w];
		LSIMD_ALIGN_SSE T dst[w];

		for (unsigned k = 0; k < n; ++k)
		{
			simd_vec<T, Kind> a;
			simd_vec<T, Kind> b;

			for (unsigned i = 0; i < w; ++i)
			{
				sa[i] = rand_val(lb_a, ub_a);
				sb[i] = rand_val(lb_b, ub_b);
			}

			a.load(sa, aligned_t());
			b.load(sb, aligned_t());

			LSIMD_ALIGN_SSE T r0[w];

			for (unsigned i = 0; i < w; ++i)
			{
				r0[i] = Op::eval_scalar(sa[i], sb[i]);
			}

			simd_vec<T, Kind> r = Op::eval_vector(a, b);
			r.store(dst, aligned_t());

			for (unsigned i = 0; i < w; ++i)
			{
				double cdev = std::fabs(double(dst[i]) - double(r0[i])) / double(r0[i]);
				if (cdev > max_dev) max_dev = cdev;
			}
		}

		return max_dev;
	}




	LSIMD_ENSURE_INLINE
	inline uint64_t read_tsc(void) {
	    uint32_t lo, hi;
	    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx" );
	    return (uint64_t)hi << 32 | lo;
	}

	template<typename T>
	LSIMD_ENSURE_INLINE
	void force_to_reg(const simd_vec<T, sse_kind> x)
	{
		asm volatile("" : : "x"(x.impl.v));
	}

	template<class Op>
	uint64_t tsc_bench(Op op, unsigned warming_times, unsigned repeat_times)
	{
		for (unsigned i = 0; i < warming_times; ++i) op.run();

		uint64_t tic = read_tsc();
		for (unsigned i = 0; i < repeat_times; ++i) op.run();
		uint64_t toc = read_tsc();

		return toc - tic;  // total cycles
	}


	template<typename T, typename Kind, class Op, unsigned Len>
	struct wrap_op
	{
		static const unsigned w = simd<T, Kind>::pack_width;
		const T *a;

		wrap_op(const T *a_)
		: a(a_)
		{
		}

		LSIMD_ENSURE_INLINE
		void run()
		{
			for (unsigned i = 0; i < Len; ++i)
			{
				simd_vec<T, Kind> x0(a + i * w, aligned_t());
				force_to_reg(x0);

				simd_vec<T, Kind> r = Op::run(x0);
				force_to_reg(r);
			}
		}
	};

	template<typename T, typename Kind, class Op, unsigned Len>
	struct wrap_op2
	{
		static const unsigned w = simd<T, Kind>::pack_width;
		const T *a;
		const T *b;

		wrap_op2(const T *a_, const T *b_)
		: a(a_), b(b_)
		{
		}

		LSIMD_ENSURE_INLINE
		void run()
		{
			for (unsigned i = 0; i < Len; ++i)
			{
				simd_vec<T, Kind> x0(a + i * w, b + i * w, aligned_t());
				force_to_reg(x0);

				simd_vec<T, Kind> y0(a + i * w, b + i * w, aligned_t());
				force_to_reg(y0);

				simd_vec<T, Kind> r = Op::run(x0);
				force_to_reg(r);
			}
		}
	};
}

#endif /* TEST_AUX_H_ */
