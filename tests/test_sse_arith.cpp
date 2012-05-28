/**
 * @file test_sse_arith.cpp
 *
 * Test the accuracy of Arithmetic functions
 *
 * @author Dahua Lin
 */

#include "test_aux.h"

using namespace lsimd;

const int N = 100 * 1024;


template<typename T, template<typename U> class OpT>
void test_accuracy_u()
{
	T lb_x = OpT<T>::lb_x();
	T ub_x = OpT<T>::ub_x();

	double maxdev = eval_sse_approx_accuracy<T, OpT<T> >(N, lb_x, ub_x);
	std::printf("\t%-9s:    max-rdev = %8.3g\n",
			OpT<T>::name(), maxdev);
}

template<typename T, template<typename U> class OpT>
void test_accuracy_b()
{
	T lb_x = OpT<T>::lb_x();
	T ub_x = OpT<T>::ub_x();

	T lb_y = OpT<T>::lb_y();
	T ub_y = OpT<T>::ub_y();

	double maxdev = eval_sse_approx_accuracy<T, OpT<T> >(N, lb_x, ub_x, lb_y, ub_y);
	std::printf("\t%-9s:    max-rdev = %8.3g\n",
			OpT<T>::name(), maxdev);
}




template<typename T>
struct add_s
{
	static const char *name() { return "add"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x + y; }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return add(x, y);
	}
};


template<typename T>
struct sub_s
{
	static const char *name() { return "sub"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x - y; }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return sub(x, y);
	}
};


template<typename T>
struct mul_s
{
	static const char *name() { return "mul"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(-10); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x * y; }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return mul(x, y);
	}
};


template<typename T>
struct div_s
{
	static const char *name() { return "div"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x / y; }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return div(x, y);
	}
};


template<typename T>
struct neg_s
{
	static const char *name() { return "neg"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return -x; }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return neg(x);
	}
};


template<typename T>
struct abs_s
{
	static const char *name() { return "abs"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::fabs(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return abs(x);
	}
};


template<typename T>
struct min_s
{
	static const char *name() { return "min"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x < y ? x : y; }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return vmin(x, y);
	}
};


template<typename T>
struct max_s
{
	static const char *name() { return "max"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T lb_y() { return T(1); }
	static T ub_y() { return T(10); }

	static T eval_scalar(const T x, const T y) { return x > y ? x : y; }

	static sse_vec<T> eval_vector(const sse_vec<T> x, const sse_vec<T> y)
	{
		return vmax(x, y);
	}
};


template<typename T>
struct sqr_s
{
	static const char *name() { return "sqr"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return x * x; }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return sqr(x);
	}
};


template<typename T>
struct sqrt_s
{
	static const char *name() { return "sqrt"; }

	static T lb_x() { return T(0.1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::sqrt(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return sqrt(x);
	}
};


template<typename T>
struct rcp_s
{
	static const char *name() { return "rcp"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / x; }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return rcp(x);
	}
};


template<typename T>
struct rsqrt_s
{
	static const char *name() { return "rsqrt"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / std::sqrt(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return rsqrt(x);
	}
};


template<typename T>
struct rcp_a_s
{
	static const char *name() { return "rcp(a)"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / x; }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return approx_rcp(x);
	}
};


template<typename T>
struct rsqrt_a_s
{
	static const char *name() { return "rsqrt(a)"; }

	static T lb_x() { return T(1); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return T(1) / std::sqrt(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return approx_rsqrt(x);
	}
};


template<typename T>
struct cube_s
{
	static const char *name() { return "cube"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return x * x * x; }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return cube(x);
	}
};


template<typename T>
struct floor_s
{
	static const char *name() { return "floor"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::floor(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return floor(x);
	}
};

template<typename T>
struct ceil_s
{
	static const char *name() { return "ceil"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::ceil(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return ceil(x);
	}
};

template<typename T>
struct floor2_s
{
	static const char *name() { return "floor(2)"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::floor(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return floor_sse2(x);
	}
};

template<typename T>
struct ceil2_s
{
	static const char *name() { return "ceil(2)"; }

	static T lb_x() { return T(-10); }
	static T ub_x() { return T(10); }

	static T eval_scalar(const T x) { return std::ceil(x); }

	static sse_vec<T> eval_vector(const sse_vec<T> x)
	{
		return ceil_sse2(x);
	}
};


template<typename T>
void test_all()
{
	test_accuracy_b<T, add_s>();
	test_accuracy_b<T, sub_s>();
	test_accuracy_b<T, mul_s>();
	test_accuracy_b<T, div_s>();

	test_accuracy_u<T, neg_s>();
	test_accuracy_u<T, abs_s>();
	test_accuracy_b<T, min_s>();
	test_accuracy_b<T, max_s>();

	test_accuracy_u<T, sqr_s>();
	test_accuracy_u<T, sqrt_s>();
	test_accuracy_u<T, rcp_s>();
	test_accuracy_u<T, rsqrt_s>();
	test_accuracy_u<T, cube_s>();

	test_accuracy_u<T, floor_s>();
	test_accuracy_u<T, ceil_s>();
	test_accuracy_u<T, floor2_s>();
	test_accuracy_u<T, ceil2_s>();
}


int main(int argc, char *argv[])
{
	std::printf("Tests on f32\n");
	std::printf("================================\n");
	test_all<f32>();

	test_accuracy_u<f32, rcp_a_s>();
	test_accuracy_u<f32, rsqrt_a_s>();

	std::printf("\n");

	std::printf("Tests on f64\n");
	std::printf("================================\n");
	test_all<f64>();

	std::printf("\n");

	return 0;
}


