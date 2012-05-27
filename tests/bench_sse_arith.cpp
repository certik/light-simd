/**
 * @file bench_sse_arith.cpp
 *
 * Benchmarking of SSE arithmetic calculation
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;

const int arr_len = 64;
const unsigned warming_times = 10;

#define FORCE_CALC(var) asm volatile("" : : "x"(var.v));

template<typename T>
struct add_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "add"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(T *a, const T *b)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());
			sse_pack<T> bi(b + i * width, aligned_t());

			sse_pack<T> tmp1 = add(ai, bi);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = add(tmp1, bi); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = add(tmp2, bi); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = add(tmp3, bi); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct sub_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "sub"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a, const T *b)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());
			sse_pack<T> bi(b + i * width, aligned_t());

			sse_pack<T> tmp1 = sub(ai, bi);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = sub(tmp1, bi); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = sub(tmp2, bi); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = sub(tmp3, bi); FORCE_CALC(tmp4)
		}
	}
};



template<typename T>
struct mul_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "mul"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a, const T *b)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());
			sse_pack<T> bi(b + i * width, aligned_t());

			sse_pack<T> tmp1 = mul(ai, bi);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = mul(tmp1, bi); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = mul(tmp2, bi); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = mul(tmp3, bi); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct div_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "div"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a, const T *b)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());
			sse_pack<T> bi(b + i * width, aligned_t());

			sse_pack<T> tmp1 = div(ai, bi);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = div(tmp1, bi); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = div(tmp2, bi); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = div(tmp3, bi); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct neg_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "neg"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = neg(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = neg(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = neg(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = neg(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct abs_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "abs"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = abs(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = abs(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = abs(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = abs(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct min_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "min"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a, const T *b)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());
			sse_pack<T> bi(b + i * width, aligned_t());

			sse_pack<T> tmp1 = vmin(ai, bi);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = vmin(tmp1, bi); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = vmin(tmp2, bi); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = vmin(tmp3, bi); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct max_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "max"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a, const T *b)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());
			sse_pack<T> bi(b + i * width, aligned_t());

			sse_pack<T> tmp1 = vmax(ai, bi);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = vmax(tmp1, bi); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = vmax(tmp2, bi); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = vmax(tmp3, bi); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct sqr_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "sqr"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = sqr(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = sqr(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = sqr(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = sqr(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct cube_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "cube"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = cube(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = cube(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = cube(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = cube(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct sqrt_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "sqrt"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = sqrt(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = sqrt(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = sqrt(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = sqrt(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct rcp_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "rcp"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = rcp(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = rcp(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = rcp(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = rcp(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct rsqrt_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "rsqrt"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = rsqrt(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = rsqrt(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = rsqrt(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = rsqrt(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct approx_rcp_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "rcp(a)"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = approx_rcp(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = approx_rcp(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = approx_rcp(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = approx_rcp(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct approx_rsqrt_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "rsqrt(a)"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = approx_rsqrt(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = approx_rsqrt(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = approx_rsqrt(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = approx_rsqrt(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct floor_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "floor"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = floor(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = floor(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = floor(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = floor(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct ceil_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "ceil"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = ceil(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = ceil(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = ceil(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = ceil(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct floor2_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "floor(2)"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = floor_sse2(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = floor_sse2(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = floor_sse2(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = floor_sse2(tmp3); FORCE_CALC(tmp4)
		}
	}
};


template<typename T>
struct ceil2_op
{
	static const int width = sse_pack<T>::pack_width;
	static const int nvecs = arr_len / width;

	const char *name() const { return "ceil(2)"; }

	int folds() const { return 4; }

	LSIMD_ENSURE_INLINE
	void run(const T *a)
	{
		for (int i = 0; i < nvecs; ++i)
		{
			sse_pack<T> ai(a + i * width, aligned_t());

			sse_pack<T> tmp1 = ceil_sse2(ai);   FORCE_CALC(tmp1)
			sse_pack<T> tmp2 = ceil_sse2(tmp1); FORCE_CALC(tmp2)
			sse_pack<T> tmp3 = ceil_sse2(tmp2); FORCE_CALC(tmp3)
			sse_pack<T> tmp4 = ceil_sse2(tmp3); FORCE_CALC(tmp4)
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

	const unsigned rt_f1 = 2000000;
	const unsigned rt_f2 =  500000;

	bench<f32, add_op>(rt_f1, af, -10.f, 10.f, bf, -10.f, 10.f);
	bench<f32, sub_op>(rt_f1, af, -10.f, 10.f, bf, -10.f, 10.f);
	bench<f32, mul_op>(rt_f1, af, -10.f, 10.f, bf,  -3.f,  3.f);
	bench<f32, div_op>(rt_f1, af, -10.f, 10.f, bf,   1.f,  3.f);
	bench<f32, min_op>(rt_f1, af, -10.f, 10.f, bf, -10.f, 10.f);
	bench<f32, max_op>(rt_f1, af, -10.f, 10.f, bf, -10.f, 10.f);

	bench<f32, neg_op>(rt_f1, af, -10.f, 10.f);
	bench<f32, abs_op>(rt_f1, af, -10.f, 10.f);
	bench<f32, sqr_op>(rt_f1, af, -10.f, 10.f);
	bench<f32, cube_op>(rt_f2, af, -2.f,  2.f);
	bench<f32, sqrt_op>(rt_f1, af,  1.f, 10.f);
	bench<f32, rcp_op>(rt_f1, af,   2.f,  5.f);
	bench<f32, rsqrt_op>(rt_f2, af, 2.f,  5.f);

	bench<f32, approx_rcp_op>(rt_f1, af,   2.f,  5.f);
	bench<f32, approx_rsqrt_op>(rt_f1, af, 2.f,  5.f);

	bench<f32, floor_op>(rt_f1, af, -10.f, 10.f);
	bench<f32, ceil_op> (rt_f1, af, -10.f, 10.f);
	bench<f32, floor2_op>(rt_f1, af, -10.f, 10.f);
	bench<f32, ceil2_op> (rt_f1, af, -10.f, 10.f);

	std::printf("\n");


	std::printf("Benchmarks on f64\n");
	std::printf("============================\n");

	const unsigned rt_d1 = 1000000;
	const unsigned rt_d2 =  200000;

	bench<f64, add_op>(rt_d1, ad, -10.0, 10.0, bd, -10.0, 10.0);
	bench<f64, sub_op>(rt_d1, ad, -10.0, 10.0, bd, -10.0, 10.0);
	bench<f64, mul_op>(rt_d1, ad, -10.0, 10.0, bd,  -3.0,  3.0);
	bench<f64, div_op>(rt_d1, ad, -10.0, 10.0, bd,   1.0,  3.0);
	bench<f64, min_op>(rt_f1, ad, -10.0, 10.0, bd, -10.0, 10.0);
	bench<f64, max_op>(rt_f1, ad, -10.0, 10.0, bd, -10.0, 10.0);

	bench<f64, neg_op>(rt_d1, ad, -10.0, 10.0);
	bench<f64, abs_op>(rt_d1, ad, -10.0, 10.0);
	bench<f64, sqr_op>(rt_d1, ad, -10.f, 10.f);
	bench<f64, cube_op>(rt_d1, ad, -3.0,  3.0);
	bench<f64, sqrt_op>(rt_d2, ad,  1.0, 10.0);
	bench<f64, rcp_op>(rt_d1, ad,   2.0,  5.0);
	bench<f64, rsqrt_op>(rt_d2, ad, 2.0,  5.0);

	bench<f64, floor_op>(rt_f1, ad, -10.0, 10.0);
	bench<f64, ceil_op> (rt_f1, ad, -10.0, 10.0);
	bench<f64, floor2_op>(rt_f1, ad, -10.0, 10.0);
	bench<f64, ceil2_op> (rt_f1, ad, -10.0, 10.0);

	std::printf("\n");

}


