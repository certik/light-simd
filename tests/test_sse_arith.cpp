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


template<typename T>
void compare_results(const char *name, T lb, T ub, const sse_array<T>& a, const sse_array<T>& b)
{
	T dev = max_dev(a, b);
	std::printf("\t%-6s:  [%4g, %4g] \t-->\t max-dev = %8.3g\n", name, lb, ub, dev);
}


template<typename T>
void test_add()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);
	sse_array<T> b(ns);

	a.set_rand(lb, ub);
	b.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = a[i] + b[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, add(a.get_pack(i), b.get_pack(i)) );

	compare_results("add", lb, ub, c, c0);
}

template<typename T>
void test_sub()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);
	sse_array<T> b(ns);

	a.set_rand(lb, ub);
	b.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = a[i] - b[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, sub(a.get_pack(i), b.get_pack(i)) );

	compare_results("sub", lb, ub, c, c0);
}


template<typename T>
void test_mul()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);
	sse_array<T> b(ns);

	a.set_rand(lb, ub);
	b.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = a[i] * b[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, mul(a.get_pack(i), b.get_pack(i)) );

	compare_results("mul", lb, ub, c, c0);
}


template<typename T>
void test_div()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);
	sse_array<T> b(ns);

	a.set_rand(lb, ub);
	b.set_rand(T(1), ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = a[i] / b[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, div(a.get_pack(i), b.get_pack(i)) );

	compare_results("div", lb, ub, c, c0);
}


template<typename T>
void test_min()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);
	sse_array<T> b(ns);

	a.set_rand(lb, ub);
	b.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = (a[i] < b[i] ? a[i] : b[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, vmin(a.get_pack(i), b.get_pack(i)) );

	compare_results("min", lb, ub, c, c0);
}


template<typename T>
void test_max()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);
	sse_array<T> b(ns);

	a.set_rand(lb, ub);
	b.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = (a[i] > b[i] ? a[i] : b[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, vmax(a.get_pack(i), b.get_pack(i)) );

	compare_results("max", lb, ub, c, c0);
}

template<typename T>
void test_neg()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = -a[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, neg(a.get_pack(i)) );

	compare_results("neg", lb, ub, c, c0);
}


template<typename T>
void test_abs()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = std::fabs(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, abs(a.get_pack(i)) );

	compare_results("abs", lb, ub, c, c0);
}

template<typename T>
void test_sqrt()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = std::sqrt(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, sqrt(a.get_pack(i)) );

	compare_results("sqrt", lb, ub, c, c0);
}


template<typename T>
void test_rcp()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(1);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = T(1) / (a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, rcp(a.get_pack(i)) );

	compare_results("rcp", lb, ub, c, c0);
}


template<typename T>
void test_rsqrt()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(1);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = T(1) / std::sqrt(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, rsqrt(a.get_pack(i)) );

	compare_results("rsqrt", lb, ub, c, c0);
}


template<typename T>
void test_sqr()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = a[i] * a[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, sqr(a.get_pack(i)) );

	compare_results("sqr", lb, ub, c, c0);
}

template<typename T>
void test_cube()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = a[i] * a[i] * a[i];
	for (int i = 0; i < nv; ++i) c.set_pack( i, cube(a.get_pack(i)) );

	compare_results("cube", lb, ub, c, c0);
}


template<typename T>
void test_floor()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);
	for (int i = 0; i <= 20; ++i) a[3 * i] = T(i - 10);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = std::floor(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, floor_sse4(a.get_pack(i)) );

	compare_results("floor", lb, ub, c, c0);
}


template<typename T>
void test_floor2()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);
	for (int i = 0; i <= 20; ++i) a[3 * i] = T(i - 10);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = std::floor(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, floor_sse2(a.get_pack(i)) );

	compare_results("floor2", lb, ub, c, c0);
}


template<typename T>
void test_ceil()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);
	for (int i = 0; i <= 20; ++i) a[3 * i] = T(i - 10);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = std::ceil(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, ceil_sse4(a.get_pack(i)) );

	compare_results("ceil", lb, ub, c, c0);
}


template<typename T>
void test_ceil2()
{
	const int ns = N;
	const int nv = ns / sse_vec<T>::pack_width;

	const T lb = T(-10);
	const T ub = T(10);

	sse_array<T> a(ns);

	a.set_rand(lb, ub);
	for (int i = 0; i <= 20; ++i) a[3 * i] = T(i - 10);

	sse_array<T> c0(ns);
	sse_array<T> c(ns);

	c0.set_zeros();
	c.set_zeros();

	for (int i = 0; i < ns; ++i) c0[i] = std::ceil(a[i]);
	for (int i = 0; i < nv; ++i) c.set_pack( i, ceil_sse2(a.get_pack(i)) );

	compare_results("ceil2", lb, ub, c, c0);
}


template<typename T>
void test_all()
{
	test_add<T>();
	test_sub<T>();
	test_mul<T>();
	test_div<T>();
	test_min<T>();
	test_max<T>();
	test_neg<T>();
	test_abs<T>();

	test_sqrt<T>();
	test_rcp<T>();
	test_rsqrt<T>();
	test_sqr<T>();
	test_cube<T>();

	test_floor<T>();
	test_floor2<T>();
	test_ceil<T>();
	test_ceil2<T>();

	std::printf("\n");
}


int main(int argc, char *argv[])
{
	std::printf("Tests on f32\n");
	std::printf("================================\n");
	test_all<f32>();

	std::printf("Tests on f64\n");
	std::printf("================================\n");
	test_all<f64>();

	return 0;
}


