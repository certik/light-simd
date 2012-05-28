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

#include <light_simd/sse.h>

namespace lsimd
{
	template<typename T>
	inline void clear_zeros(int n, T *a)
	{
		for (int i = 0; i < n; ++i) a[i] = T(0);
	}


	template<typename T>
	inline void fill_rand(int n, T *a, T lb, T ub)
	{
		for (int i = 0; i < n; ++i)
			a[i] = lb + (T(std::rand()) / RAND_MAX) * (ub - lb);
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



	template<typename T>
	struct sse_array
	{
	public:
		sse_array(int n)
		: m_n(n), m_nv(m_n / sse_pack<T>::pack_width), m_data(alloc(n))
		{ }

		sse_array(const sse_array& a)
		: m_n(a.m_n), m_nv(m_n / sse_pack<T>::pack_width), m_data(alloc(a.m_n))
		{
			::memcpy(m_data, a.m_data, sizeof(T) * size_t(m_n));
		}

		~sse_array()
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
		sse_pack<T> get_pack(int i)
		{
			return sse_pack<T>(m_data + i * sse_pack<T>::pack_width, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void set_pack(int i, sse_pack<T> p)
		{
			p.store(m_data + i * sse_pack<T>::pack_width, aligned_t());
		}

	private:
		int m_n;
		int m_nv;
		T *m_data;

	private:
		sse_array& operator = (const sse_array& );

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


	template<typename T>
	T max_dev(const sse_array<T>& a, const sse_array<T>& b)
	{
		T s(0);
		for (int i = 0; i < a.nelems(); ++i)
		{
			T v = std::fabs(a[i] - b[i]);
			if (v > s) s = v;
		}
		return s;
	}



	LSIMD_ENSURE_INLINE
	inline uint64_t read_tsc(void) {
	    uint32_t lo, hi;
	    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx" );
	    return (uint64_t)hi << 32 | lo;
	}


	template<typename T, class Op>
	uint64_t bench_op(Op op, unsigned warming_times, unsigned repeat_times,
			int arr_len, T *a, const T lb_a, const T ub_a)
	{
		fill_rand(arr_len, a, lb_a, ub_a);

		for (unsigned i = 0; i < warming_times; ++i) op.run(a);

		uint64_t tic = read_tsc();
		for (unsigned i = 0; i < repeat_times; ++i) op.run(a);
		uint64_t toc = read_tsc();

		return toc - tic;  // total cycles
	}


	template<typename T, class Op>
	uint64_t bench_op(Op op, unsigned warming_times, unsigned repeat_times,
			int arr_len,
			T *a, const T lb_a, const T ub_a,
			T *b, const T lb_b, const T ub_b)
	{
		fill_rand(arr_len, a, lb_a, ub_b);
		fill_rand(arr_len, b, lb_b, ub_b);

		for (unsigned i = 0; i < warming_times; ++i) op.run(a, b);

		uint64_t tic = read_tsc();
		for (unsigned i = 0; i < repeat_times; ++i) op.run(a, b);
		uint64_t toc = read_tsc();

		return toc - tic;  // total cycles
	}

}

#endif /* TEST_AUX_H_ */