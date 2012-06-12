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
#include <light_test/tests.h>


namespace lsimd
{

	/********************************************
	 *
	 *  Test case bases
	 *
	 ********************************************/


	extern ::ltest::test_suite lsimd_main_suite;
	extern void add_test_packs();

	template<typename T>
	class tcase_base : public ltest::test_case
	{
		char m_name[128];

	public:
		tcase_base(const char *nam)
		{
			std::sprintf(m_name, "%s [f%d]", nam, int(8 * sizeof(T)));
		}

		const char *name() const
		{
			return m_name;
		}
	};


	template<typename T, int N>
	class tcase1_base : public ltest::test_case
	{
		char m_name[128];

	public:
		tcase1_base(const char *nam)
		{
			std::sprintf(m_name, "%s [f%d x %d]", nam, int(8 * sizeof(T)), N);
		}

		const char *name() const
		{
			return m_name;
		}
	};

	template<typename T, int M, int N>
	class tcase2_base : public ltest::test_case
	{
		char m_name[128];

	public:
		tcase2_base(const char *nam)
		{
			std::sprintf(m_name, "%s [f%d x %d x %d]", nam, int(8 * sizeof(T)), M, N);
		}

		const char *name() const
		{
			return m_name;
		}
	};


	/********************************************
	 *
	 *  Array functions
	 *
	 ********************************************/


	template<typename T>
	inline void clear_zeros(int n, T *a)
	{
		for (int i = 0; i < n; ++i) a[i] = T(0);
	}

	template<typename T>
	inline void fill_const(int n, T *a, T v)
	{
		for (int i = 0; i < n; ++i) a[i] = v;
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


	template<typename T, typename Kind, class Op>
	double eval_approx_accuracy(unsigned n, const T lb_a, const T ub_a)
	{
		double max_dev = 0.0;
		const unsigned w = simd<T, Kind>::pack_width;
		LSIMD_ALIGN_SSE T src[w];
		LSIMD_ALIGN_SSE T dst[w];

		for (unsigned k = 0; k < n; ++k)
		{
			simd_pack<T, Kind> a;

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

			simd_pack<T, Kind> r = Op::eval_vector(a);
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
			simd_pack<T, Kind> a;
			simd_pack<T, Kind> b;

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

			simd_pack<T, Kind> r = Op::eval_vector(a, b);
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
	inline void force_to_reg(const simd_pack<T, sse_kind> x)
	{
		asm volatile("" : : "x"(x.impl.v));
	}

	LSIMD_ENSURE_INLINE
	inline void force_to_reg(const f32& x)
	{
		asm volatile("" : : "x"(x));
	}

	LSIMD_ENSURE_INLINE
	inline void force_to_reg(const f64& x)
	{
		asm volatile("" : : "x"(x));
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
				simd_pack<T, Kind> x0(a + i * w, aligned_t());
				force_to_reg(x0);

				Op::run(x0);
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
				simd_pack<T, Kind> x0(a + i * w, b + i * w, aligned_t());
				force_to_reg(x0);

				simd_pack<T, Kind> y0(a + i * w, b + i * w, aligned_t());
				force_to_reg(y0);

				Op::run(x0, y0);
			}
		}
	};
}


/********************************************
 *
 *  TUseful macros for testing
 *
 ********************************************/


#define ASSERT_SIMD_EQ( v, r ) \
	if ( !(v).impl.test_equal(r) ) throw ::ltest::assertion_failure(__FILE__, __LINE__, #v " == " #r)


#define GCASE( tname ) \
	template<typename T> \
	class tname##_tests : public tcase_base<T> { \
	public: \
		tname##_tests() : tcase_base<T>( #tname ) { } \
		void run(); \
	}; \
	template<typename T> \
	void tname##_tests<T>::run()

#define SCASE( tname, ty ) \
	template<> \
	class tname##_tests<ty> : public tcase_base<ty> { \
	public: \
		tname##_tests() : tcase_base<ty>( #tname ) { } \
		void run(); \
	}; \
	void tname##_tests<ty>::run()

#define GCASE1( tname ) \
	template<typename T, int N> \
	class tname##_tests : public tcase1_base<T, N> { \
	public: \
		tname##_tests() : tcase1_base<T, N>( #tname ) { } \
		void run(); \
	}; \
	template<typename T, int N> \
	void tname##_tests<T, N>::run()

#define SCASE1( tname, n ) \
	template<typename T> \
	class tname##_tests<T, n> : public tcase1_base<T, n> { \
	public: \
		tname##_tests() : tcase1_base<T, n>( #tname ) { } \
		void run(); \
	}; \
	template<typename T> \
	void tname##_tests<T, n>::run()

#define GCASE2( tname ) \
	template<typename T, int M, int N> \
	class tname##_tests : public tcase2_base<T, M, N> { \
	public: \
		tname##_tests() : tcase2_base<T, M, N>( #tname ) { } \
		void run(); \
	}; \
	template<typename T, int M, int N> \
	void tname##_tests<T, M, N>::run()



#endif /* TEST_AUX_H_ */



