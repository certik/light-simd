/**
 * @file bench_sse_vecs.cpp
 *
 * Benchmark of SSE vector operations
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;

const unsigned arr_len = 256;
const unsigned num_vecs = arr_len / 4;
const unsigned warming_times = 10;

LSIMD_ALIGN(128) f32 af[arr_len];
LSIMD_ALIGN(128) f64 ad[arr_len];

LSIMD_ALIGN(128) f32 bf[arr_len];
LSIMD_ALIGN(128) f64 bd[arr_len];

template<typename T> struct data_s;

template<> struct data_s<f32>
{
	LSIMD_ENSURE_INLINE
	static const f32 *src() { return af; }

	LSIMD_ENSURE_INLINE
	static f32 *dst() { return bf; }
};

template<> struct data_s<f64>
{
	LSIMD_ENSURE_INLINE
	static const f64 *src() { return ad; }

	LSIMD_ENSURE_INLINE
	static f64 *dst() { return bd; }
};


template<typename T, int N, template<typename U, int M> class OpT>
inline void bench(unsigned repeat_times)
{
	OpT<T, N> op1;
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	double cpv = double(cs1) / (double(repeat_times) * double(num_vecs));

	std::printf("\t%-10s:  %.1f cycles / vec\n", op1.name(), cpv);
}


template<typename T, int N>
struct addcp_op
{
	static const unsigned int Len = arr_len / 4;

	const char *name() const { return "add-copy"; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();
		T *dst = data_s<T>::dst();

		for (unsigned i = 0; i < Len; ++i)
		{
			simd_vec<T, N, sse_kind> v(src + i * 4, aligned_t());
			simd_vec<T, N, sse_kind> v2(dst + i * 4, aligned_t());

			(v + v2).store(dst + i * 4, aligned_t());
		}
	}
};


template<typename T, int N>
struct ldsum_op
{
	static const unsigned int Len = arr_len / 4;

	const char *name() const { return "load-sum"; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();

		for (unsigned i = 0; i < Len; ++i)
		{
			simd_vec<T, N, sse_kind> v(src + i * 4, aligned_t());

			T s = v.sum();
			force_to_reg(s);
		}
	}
};


template<typename T, int N>
struct lddot_op
{
	static const unsigned int Len = arr_len / 4;

	const char *name() const { return "load-dot"; }

	LSIMD_ENSURE_INLINE
	void run()
	{
		const T *src = data_s<T>::src();

		for (unsigned i = 0; i < Len; ++i)
		{
			simd_vec<T, N, sse_kind> v(src + i * 4, aligned_t());

			T s = v.dot(v);
			force_to_reg(s);
		}
	}
};



template<typename T, int N>
void do_bench()
{
	const unsigned int rtimes = 4000000 / sizeof(T);

	std::printf("Benchmarks on f%lu x %d\n", sizeof(T) * 8, N);
	std::printf("================================\n");

	bench<T, N, addcp_op>(rtimes);
	bench<T, N, ldsum_op>(rtimes);
	bench<T, N, lddot_op>(rtimes);

	std::printf("\n");
}


int main(int argc, char *argv[])
{
	fill_rand(arr_len, af, 0.f, 1.f);
	fill_rand(arr_len, ad, 0.0, 1.0);

	do_bench<f32, 1>();
	do_bench<f32, 2>();
	do_bench<f32, 3>();
	do_bench<f32, 4>();

	do_bench<f64, 1>();
	do_bench<f64, 2>();
	do_bench<f64, 3>();
	do_bench<f64, 4>();

}




