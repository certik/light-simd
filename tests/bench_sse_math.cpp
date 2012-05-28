/**
 * @file bench_sse_math.cpp
 *
 * Benchmarking of SSE math functions
 *
 * @author Dahua Lin
 */


#include "test_aux.h"


using namespace lsimd;

const int arr_len = 64;
const unsigned warming_times = 10;

#define FORCE_CALC(var) asm volatile("" : : "x"(var.v));

template<typename T>
struct exp_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "exp"; }

	int folds() const { return 2; }

	LSIMD_ENSURE_INLINE
	void run(T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = exp(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = exp(tmp1); FORCE_CALC(tmp2)
		}
	}
};

template<typename T>
struct log_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "log"; }

	int folds() const { return 2; }

	LSIMD_ENSURE_INLINE
	void run(T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = log(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = log(tmp1); FORCE_CALC(tmp2)
		}
	}
};




inline void report_bench(const char *name, unsigned rtimes, uint64_t cycles,
		int nops, int folds, int pack_w)
{
	double cpo_f = double(cycles) / (double(rtimes) * double(arr_len));

	int cpoi = int(cpo_f);
	cpoi = (cpoi - nops) / folds;  // re-calibrated

	std::printf("\t%-10s:   %4d cycles / %d op\n", name, cpoi, pack_w);
}


template<typename T, template<typename U> class OpT>
inline void bench(unsigned repeat_times,
		T *a, const T la, const T ua)
{
	OpT<T> op;
	uint64_t cycles = bench_op(op, warming_times, repeat_times, arr_len, a, la, ua);
	report_bench(op.name(), repeat_times, cycles, 1, op.folds(), (int)sse_pack<T>::pack_width);
}


template<typename T, template<typename U> class OpT>
inline void bench(unsigned repeat_times,
		T *a, const T la, const T ua,
		T *b, const T lb, const T ub)
{
	OpT<T> op;
	uint64_t cycles = bench_op(op, warming_times, repeat_times, arr_len, a, la, ua, b, lb, ub);
	report_bench(op.name(), repeat_times, cycles, 2, op.folds(), (int)sse_pack<T>::pack_width);
}


LSIMD_ALIGN(256) f32 af[arr_len];
LSIMD_ALIGN(256) f32 bf[arr_len];
LSIMD_ALIGN(256) f64 ad[arr_len];
LSIMD_ALIGN(256) f64 bd[arr_len];

int main(int argc, char *argv[])
{
	std::printf("Benchmarks on f32\n");
	std::printf("============================\n");

	const unsigned rt_f1 =  2000000;

	bench<f32, exp_op>(rt_f1, af, -2.f, 2.f);
	bench<f32, log_op>(rt_f1, af, 10.f, 1000.f);

	std::printf("\n");


	std::printf("Benchmarks on f64\n");
	std::printf("============================\n");

	const unsigned rt_d1 =  500000;

	bench<f64, exp_op>(rt_d1, ad, -2.0, 2.0);
	bench<f64, log_op>(rt_d1, ad, 10.0, 1000.0);

	std::printf("\n");

}

