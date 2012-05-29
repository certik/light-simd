/**
 * @file bench_sse_math.cpp
 *
 * Benchmarking of SSE math functions
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;

const unsigned arr_len = 64;
const unsigned warming_times = 10;

inline void report_bench(const char *name, unsigned rtimes, uint64_t cycles,
		unsigned pack_w, int nops)
{
	double cpo_f = double(cycles) / (double(rtimes) * double(arr_len));

	int cpoi = int(cpo_f);
	cpoi = (cpoi - nops);  // re-calibrated

	std::printf("\t%-5s :   %4d cycles / %u op\n", name, cpoi, pack_w);
}


template<typename T, template<typename U> class OpT>
inline void bench(unsigned repeat_times, T *pa)
{
	const T lb = OpT<T>::lbound();
	const T ub = OpT<T>::ubound();

	fill_rand(arr_len, pa, lb, ub);

	wrap_op<T, sse_kind, OpT<T>, arr_len> op1(pa);
	uint64_t cs1 = tsc_bench(op1, warming_times, repeat_times);

	report_bench(OpT<T>::name(), repeat_times, cs1, simd<T, sse_kind>::pack_width, 1);
}




template<typename T>
struct pow_op
{
	static const char *name() { return "pow"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(3); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return pow(x, simd_vec<T, sse_kind>(2.5));
	}
};

template<typename T>
struct cbrt_op
{
	static const char *name() { return "cbrt"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(3); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return cbrt(x);
	}
};

template<typename T>
struct hypot_op
{
	static const char *name() { return "hypot"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(3); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return hypot(x, simd_vec<T, sse_kind>(2.0));
	}
};


template<typename T>
struct exp_op
{
	static const char *name() { return "exp"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(3); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return exp(x);
	}
};

template<typename T>
struct exp2_op
{
	static const char *name() { return "exp2"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(3); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return exp2(x);
	}
};

template<typename T>
struct exp10_op
{
	static const char *name() { return "exp10"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(3); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return exp2(x);
	}
};

template<typename T>
struct log_op
{
	static const char *name() { return "log"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return exp(x);
	}
};

template<typename T>
struct log2_op
{
	static const char *name() { return "log2"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return exp(x);
	}
};

template<typename T>
struct log10_op
{
	static const char *name() { return "log10"; }

	static T lbound() { return T(1); }
	static T ubound() { return T(100); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return exp(x);
	}
};

template<typename T>
struct expm1_op
{
	static const char *name() { return "expm1"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(0.1); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return expm1(x);
	}
};

template<typename T>
struct log1p_op
{
	static const char *name() { return "log1p"; }

	static T lbound() { return T(0); }
	static T ubound() { return T(0.1); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return log1p(x);
	}
};


template<typename T>
struct sin_op
{
	static const char *name() { return "sin"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return sin(x);
	}
};

template<typename T>
struct cos_op
{
	static const char *name() { return "cos"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return cos(x);
	}
};

template<typename T>
struct tan_op
{
	static const char *name() { return "tan"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return tan(x);
	}
};

template<typename T>
struct asin_op
{
	static const char *name() { return "asin"; }

	static T lbound() { return T(-1); }
	static T ubound() { return T(1); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return asin(x);
	}
};

template<typename T>
struct acos_op
{
	static const char *name() { return "acos"; }

	static T lbound() { return T(-1); }
	static T ubound() { return T(1); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return acos(x);
	}
};

template<typename T>
struct atan_op
{
	static const char *name() { return "atan"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return atan(x);
	}
};

template<typename T>
struct atan2_op
{
	static const char *name() { return "atan2"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return atan2(x, simd_vec<T, sse_kind>::ones());
	}
};


template<typename T>
struct sinh_op
{
	static const char *name() { return "sinh"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return sinh(x);
	}
};

template<typename T>
struct cosh_op
{
	static const char *name() { return "cosh"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return cosh(x);
	}
};

template<typename T>
struct tanh_op
{
	static const char *name() { return "tanh"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return tanh(x);
	}
};

template<typename T>
struct asinh_op
{
	static const char *name() { return "asinh"; }

	static T lbound() { return T(-1); }
	static T ubound() { return T(1); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return asinh(x);
	}
};

template<typename T>
struct acosh_op
{
	static const char *name() { return "acosh"; }

	static T lbound() { return T(-1); }
	static T ubound() { return T(1); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return acosh(x);
	}
};

template<typename T>
struct atanh_op
{
	static const char *name() { return "atanh"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return atanh(x);
	}
};


template<typename T>
struct erf_op
{
	static const char *name() { return "erf"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return erf(x);
	}
};

template<typename T>
struct erfc_op
{
	static const char *name() { return "erfc"; }

	static T lbound() { return T(-10); }
	static T ubound() { return T(10); }

	LSIMD_ENSURE_INLINE
	static simd_vec<T, sse_kind> run(simd_vec<T, sse_kind> x)
	{
		return erfc(x);
	}
};




LSIMD_ALIGN(256) f32 af[arr_len];
LSIMD_ALIGN(256) f32 bf[arr_len];
LSIMD_ALIGN(256) f64 ad[arr_len];
LSIMD_ALIGN(256) f64 bd[arr_len];

int main(int argc, char *argv[])
{
	std::printf("Benchmarks on f32\n");
	std::printf("==========================================\n");

	const unsigned rt_f1 = 1000000;

	bench<f32, pow_op>   (rt_f1 / 10, af);
	bench<f32, cbrt_op>  (rt_f1, af);
	bench<f32, hypot_op> (rt_f1, af);

	bench<f32, exp_op>   (rt_f1, af);
	bench<f32, exp2_op>  (rt_f1, af);
	bench<f32, exp10_op> (rt_f1, af);

	bench<f32, log_op>   (rt_f1, af);
	bench<f32, log2_op>  (rt_f1, af);
	bench<f32, log10_op> (rt_f1, af);

	bench<f32, expm1_op> (rt_f1, af);
	bench<f32, log1p_op> (rt_f1, af);

	bench<f32, sin_op>   (rt_f1, af);
	bench<f32, cos_op>   (rt_f1, af);
	bench<f32, tan_op>   (rt_f1, af);

	bench<f32, asin_op>  (rt_f1, af);
	bench<f32, acos_op>  (rt_f1, af);
	bench<f32, atan_op>  (rt_f1, af);
	bench<f32, atan2_op> (rt_f1, af);

	bench<f32, sinh_op>  (rt_f1, af);
	bench<f32, cosh_op>  (rt_f1, af);
	bench<f32, tanh_op>  (rt_f1, af);

	bench<f32, asinh_op> (rt_f1 / 2, af);
	bench<f32, acosh_op> (rt_f1, af);
	bench<f32, atanh_op> (rt_f1, af);

	bench<f32, erf_op>   (rt_f1 / 2, af);
	bench<f32, erfc_op>  (rt_f1 / 2, af);

	std::printf("\n");


	std::printf("Benchmarks on f64\n");
	std::printf("==========================================\n");

	const unsigned rt_d1 = 500000;

	bench<f64, pow_op>   (rt_d1 / 5, ad);
	bench<f64, cbrt_op>  (rt_d1, ad);
	bench<f64, hypot_op> (rt_d1, ad);

	bench<f64, exp_op>   (rt_d1, ad);
	bench<f64, exp2_op>  (rt_d1, ad);
	bench<f64, exp10_op> (rt_d1, ad);

	bench<f64, log_op>   (rt_d1, ad);
	bench<f64, log2_op>  (rt_d1, ad);
	bench<f64, log10_op> (rt_d1, ad);

	bench<f64, expm1_op> (rt_d1, ad);
	bench<f64, log1p_op> (rt_d1, ad);

	bench<f64, sin_op>   (rt_d1, ad);
	bench<f64, cos_op>   (rt_d1, ad);
	bench<f64, tan_op>   (rt_d1, ad);

	bench<f64, asin_op>  (rt_d1, ad);
	bench<f64, acos_op>  (rt_d1, ad);
	bench<f64, atan_op>  (rt_d1, ad);
	bench<f64, atan2_op> (rt_d1, ad);

	bench<f64, sinh_op>  (rt_d1, ad);
	bench<f64, cosh_op>  (rt_d1, ad);
	bench<f64, tanh_op>  (rt_d1, ad);

	bench<f64, asinh_op> (rt_d1 / 2, ad);
	bench<f64, acosh_op> (rt_d1, ad);
	bench<f64, atanh_op> (rt_d1, ad);

	bench<f64, erf_op>   (rt_d1 / 2, ad);
	bench<f64, erfc_op>  (rt_d1 / 5, ad);

	std::printf("\n");

}

