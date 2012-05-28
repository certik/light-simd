/*
 * @file test_sse_math.cpp
 *
 * Test the accuracy of SSE math functions
 *
 * @author Dahua Lin
 */

#include "test_aux.h"

#include <cmath>

using namespace lsimd;

const int N = 50 * 1024;

template<typename T, template<typename U> class OpT>
void test_accuracy_u()
{
	T lb_x = OpT<T>::lb_x();
	T ub_x = OpT<T>::ub_x();

	double max_rdev = eval_sse_approx_accuracy<T, OpT<T> >(N, lb_x, ub_x);
	std::printf("  %-6s [%4g:%4g]             ==> max-rdev = %8.3g\n",
			OpT<T>::name(), OpT<T>::lb_x(), OpT<T>::ub_x(), max_rdev);
}

template<typename T, template<typename U> class OpT>
void test_accuracy_b()
{
	T lb_x = OpT<T>::lb_x();
	T ub_x = OpT<T>::ub_x();

	T lb_y = OpT<T>::lb_y();
	T ub_y = OpT<T>::ub_y();

	double max_rdev = eval_sse_approx_accuracy<T, OpT<T> >(N, lb_x, ub_x, lb_y, ub_y);
	std::printf("  %-6s [%4g:%4g] [%4g:%4g] ==> max-rdev = %8.3g\n",
			OpT<T>::name(),
			OpT<T>::lb_x(), OpT<T>::ub_x(),
			OpT<T>::lb_y(), OpT<T>::ub_y(),
			max_rdev);
}



template<typename T>
struct pow_s
{
	static const char *name() { return "pow"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(2); }

	static T eval_scalar(const T x, const T y) { return std::pow(x, y); }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return pow(x, y);
	}
};


template<typename T>
struct exp_s
{
	static const char *name() { return "exp"; }

	static T lb_x() { return T(-3); }
	static T ub_x() { return T(3); }

	static T eval_scalar(const T x) { return std::exp(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return exp(x);
	}
};

template<typename T>
struct log_s
{
	static const char *name() { return "log"; }

	static T lb_x() { return T(0.1); }
	static T ub_x() { return T(100); }

	static T eval_scalar(const T x) { return std::log(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return log(x);
	}
};

template<typename T>
struct log10_s
{
	static const char *name() { return "log10"; }

	static T lb_x() { return T(0.1); }
	static T ub_x() { return T(100); }

	static T eval_scalar(const T x) { return std::log10(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return log10(x);
	}
};


template<typename T>
struct sin_s
{
	static const char *name() { return "sin"; }

	static T lb_x() { return T(-3); }
	static T ub_x() { return T(3); }

	static T eval_scalar(const T x) { return std::sin(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return sin(x);
	}
};


template<typename T>
struct cos_s
{
	static const char *name() { return "cos"; }

	static T lb_x() { return T(-3); }
	static T ub_x() { return T(3); }

	static T eval_scalar(const T x) { return std::cos(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return cos(x);
	}
};


template<typename T>
struct tan_s
{
	static const char *name() { return "tan"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1.5); }

	static T eval_scalar(const T x) { return std::tan(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return tan(x);
	}
};



template<typename T>
struct asin_s
{
	static const char *name() { return "asin"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1); }

	static T eval_scalar(const T x) { return std::asin(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return asin(x);
	}
};


template<typename T>
struct acos_s
{
	static const char *name() { return "acos"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1); }

	static T eval_scalar(const T x) { return std::acos(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return acos(x);
	}
};


template<typename T>
struct atan_s
{
	static const char *name() { return "atan"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::atan(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return atan(x);
	}
};


template<typename T>
struct atan2_s
{
	static const char *name() { return "atan2"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }
	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return std::atan2(x, y); }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return atan2(x, y);
	}
};


template<typename T>
struct sinh_s
{
	static const char *name() { return "sinh"; }

	static T lb_x() { return T(-3); }
	static T ub_x() { return T(3); }

	static T eval_scalar(const T x) { return std::sinh(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return sinh(x);
	}
};


template<typename T>
struct cosh_s
{
	static const char *name() { return "cosh"; }

	static T lb_x() { return T(-3); }
	static T ub_x() { return T(3); }

	static T eval_scalar(const T x) { return std::cosh(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return cosh(x);
	}
};


template<typename T>
struct tanh_s
{
	static const char *name() { return "tanh"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1.5); }

	static T eval_scalar(const T x) { return std::tanh(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return tanh(x);
	}
};




#ifdef __GXX_EXPERIMENTAL_CXX0X__

template<typename T>
struct cbrt_s
{
	static const char *name() { return "cbrt"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::cbrt(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return cbrt(x);
	}
};

template<typename T>
struct hypot_s
{
	static const char *name() { return "hypot"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return std::hypot(x, y); }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return hypot(x, y);
	}
};


template<typename T>
struct exp10_s
{
	static const char *name() { return "exp10"; }

	static T lb_x() { return T(-2); }
	static T ub_x() { return T(2); }

	static T eval_scalar(const T x) { return std::pow(T(10), x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return exp10(x);
	}
};

template<typename T>
struct exp2_s
{
	static const char *name() { return "exp2"; }

	static T lb_x() { return T(-4); }
	static T ub_x() { return T(5); }

	static T eval_scalar(const T x) { return std::exp2(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return exp2(x);
	}
};

template<typename T>
struct expm1_s
{
	static const char *name() { return "expm1"; }

	static T lb_x() { return T(0); }
	static T ub_x() { return T(0.1); }

	static T eval_scalar(const T x) { return std::expm1(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return expm1(x);
	}
};

template<typename T>
struct log2_s
{
	static const char *name() { return "log2"; }

	static T lb_x() { return T(0.01); }
	static T ub_x() { return T(100); }

	static T eval_scalar(const T x) { return std::log2(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return log2(x);
	}
};


template<typename T>
struct log1p_s
{
	static const char *name() { return "log1p"; }

	static T lb_x() { return T(0); }
	static T ub_x() { return T(0.1); }

	static T eval_scalar(const T x) { return std::log1p(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return log1p(x);
	}
};


template<typename T>
struct asinh_s
{
	static const char *name() { return "asinh"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1); }

	static T eval_scalar(const T x) { return std::asinh(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return asinh(x);
	}
};


template<typename T>
struct acosh_s
{
	static const char *name() { return "acosh"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1); }

	static T eval_scalar(const T x) { return std::acosh(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return acosh(x);
	}
};


template<typename T>
struct atanh_s
{
	static const char *name() { return "atanh"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::atanh(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return atanh(x);
	}
};


template<typename T>
struct erf_s
{
	static const char *name() { return "erf"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1); }

	static T eval_scalar(const T x) { return std::erf(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return erf(x);
	}
};

template<typename T>
struct erfc_s
{
	static const char *name() { return "erfc"; }

	static T lb_x() { return T(-1); }
	static T ub_x() { return T(1); }

	static T eval_scalar(const T x) { return std::erfc(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return erfc(x);
	}
};


#endif



template<typename T>
void test_all()
{
	std::printf("power, exp, log:\n");
	std::printf("----------------------------\n");

	test_accuracy_b<T, pow_s>();
	test_accuracy_u<T, exp_s>();
	test_accuracy_u<T, log_s>();
	test_accuracy_u<T, log10_s>();

#ifdef __GXX_EXPERIMENTAL_CXX0X__

	test_accuracy_u<T, cbrt_s>();
	test_accuracy_b<T, hypot_s>();

	test_accuracy_u<T, exp2_s>();
	test_accuracy_u<T, exp10_s>();
	test_accuracy_u<T, expm1_s>();

	test_accuracy_u<T, log2_s>();
	test_accuracy_u<T, log1p_s>();

#endif

	std::printf("\n");

	std::printf("trigonometric:\n");
	std::printf("----------------------------\n");

	test_accuracy_u<T, sin_s>();
	test_accuracy_u<T, cos_s>();
	test_accuracy_u<T, tan_s>();

	test_accuracy_u<T, asin_s>();
	test_accuracy_u<T, acos_s>();
	test_accuracy_u<T, atan_s>();
	test_accuracy_b<T, atan2_s>();

	std::printf("\n");

	std::printf("hyperbolic:\n");
	std::printf("----------------------------\n");

#ifdef __GXX_EXPERIMENTAL_CXX0X__

	test_accuracy_u<T, sinh_s>();
	test_accuracy_u<T, cosh_s>();
	test_accuracy_u<T, tanh_s>();

	test_accuracy_u<T, asinh_s>();
	test_accuracy_u<T, acosh_s>();
	test_accuracy_u<T, atanh_s>();

#endif

	std::printf("\n");

	std::printf("error functions:\n");
	std::printf("----------------------------\n");

#ifdef __GXX_EXPERIMENTAL_CXX0X__

	test_accuracy_u<T, erf_s>();
	test_accuracy_u<T, erfc_s>();

#endif

}


int main(int argc, char *argv[])
{
	std::printf("Tests on f32\n");
	std::printf("===========================================================\n");
	test_all<f32>();

	std::printf("\n");

	std::printf("Tests on f64\n");
	std::printf("===========================================================\n");
	test_all<f64>();

	std::printf("\n");

	return 0;
}





