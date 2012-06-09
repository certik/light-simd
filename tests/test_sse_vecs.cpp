/**
 * @file test_sse_vecs.cpp
 *
 * Testing the correctness of sse_vec classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;

// explicit instantiation for thorough syntax check

template class lsimd::sse_vec<f32, 1>;
template class lsimd::sse_vec<f32, 2>;
template class lsimd::sse_vec<f32, 3>;
template class lsimd::sse_vec<f32, 4>;

template class lsimd::sse_vec<f64, 1>;
template class lsimd::sse_vec<f64, 2>;
template class lsimd::sse_vec<f64, 3>;
template class lsimd::sse_vec<f64, 4>;

template struct lsimd::simd_vec<f32, 1, sse_kind>;
template struct lsimd::simd_vec<f32, 2, sse_kind>;
template struct lsimd::simd_vec<f32, 3, sse_kind>;
template struct lsimd::simd_vec<f32, 4, sse_kind>;

template struct lsimd::simd_vec<f64, 1, sse_kind>;
template struct lsimd::simd_vec<f64, 2, sse_kind>;
template struct lsimd::simd_vec<f64, 3, sse_kind>;
template struct lsimd::simd_vec<f64, 4, sse_kind>;


// Test cases

#define TEST_ITEM( name ) \
	if ( test_##name<T, N>() ) { std::printf("Tests on %-16s: passed\n", #name);  } \
	else { std::printf("Tests on %-16s: failed\n", #name); return false; }


template<typename T, int N>
bool test_set();

template<>
bool test_set<f32, 1>()
{
	f32 a[1] = { 1.1f };
	f32 b[4] = {0.f, 0.f, 0.f, 0.f};

	sse_vec<f32, 1> v(a[0]);
	v.store(b, unaligned_t());

	f32 r[4] = {1.1f, 0.f, 0.f, 0.f};
	return test_equal(4, b, r);
}

template<>
bool test_set<f32, 2>()
{
	f32 a[2] = { 1.1f, 2.2f };
	f32 b[4] = {0.f, 0.f, 0.f, 0.f};

	sse_vec<f32, 2> v(a[0], a[1]);
	v.store(b, unaligned_t());

	f32 r[4] = {1.1f, 2.2f, 0.f, 0.f};
	return test_equal(4, b, r);
}

template<>
bool test_set<f32, 3>()
{
	f32 a[3] = { 1.1f, 2.2f, 3.3f };
	f32 b[4] = {0.f, 0.f, 0.f, 0.f};

	sse_vec<f32, 3> v(a[0], a[1], a[2]);
	v.store(b, unaligned_t());

	f32 r[4] = {1.1f, 2.2f, 3.3f, 0.f};
	return test_equal(4, b, r);
}

template<>
bool test_set<f32, 4>()
{
	f32 a[4] = { 1.1f, 2.2f, 3.3f, 4.4f };
	f32 b[4] = {0.f, 0.f, 0.f, 0.f};

	sse_vec<f32, 4> v(a[0], a[1], a[2], a[3]);
	v.store(b, unaligned_t());

	f32 r[4] = {1.1f, 2.2f, 3.3f, 4.4f};
	return test_equal(4, b, r);
}


template<>
bool test_set<f64, 1>()
{
	f64 a[1] = { 1.1 };
	f64 b[4] = {0.0, 0.0, 0.0, 0.0};

	sse_vec<f64, 1> v(a[0]);
	v.store(b, unaligned_t());

	f64 r[4] = {1.1, 0.0, 0.0, 0.0};
	return test_equal(4, b, r);
}

template<>
bool test_set<f64, 2>()
{
	f64 a[2] = { 1.1, 2.2 };
	f64 b[4] = {0.0, 0.0, 0.0, 0.0};

	sse_vec<f64, 2> v(a[0], a[1]);
	v.store(b, unaligned_t());

	f64 r[4] = {1.1, 2.2, 0.0, 0.0};
	return test_equal(4, b, r);
}

template<>
bool test_set<f64, 3>()
{
	f64 a[3] = { 1.1, 2.2, 3.3 };
	f64 b[4] = {0.0, 0.0, 0.0, 0.0};

	sse_vec<f64, 3> v(a[0], a[1], a[2]);
	v.store(b, unaligned_t());

	f64 r[4] = {1.1, 2.2, 3.3, 0.0};
	return test_equal(4, b, r);
}

template<>
bool test_set<f64, 4>()
{
	f64 a[4] = { 1.1, 2.2, 3.3, 4.4 };
	f64 b[4] = {0.0, 0.0, 0.0, 0.0};

	sse_vec<f64, 4> v(a[0], a[1], a[2], a[3]);
	v.store(b, unaligned_t());

	f64 r[4] = {1.1, 2.2, 3.3, 4.4};
	return test_equal(4, b, r);
}


template<typename T, int N>
bool test_load_store()
{
	const int L = 4;

	T src_u[L] = { T(1.1), T(2.2), T(3.3), T(4.4) };
	T dst_u[L] = { T(-1), T(-1), T(-1), T(-1) };

	LSIMD_ALIGN_SSE T src_a[L] = { T(1.1), T(2.2), T(3.3), T(4.4) };
	LSIMD_ALIGN_SSE T dst_a[L] = { T(-1), T(-1), T(-1), T(-1) };

	T r[L] = { T(-1), T(-1), T(-1), T(-1) };
	for (int i = 0; i < N; ++i) r[i] = src_a[i];

	simd_vec<T, N, sse_kind> v1(src_a, aligned_t());
	v1.store(dst_a, aligned_t());

	if ( !test_equal(L, dst_a, r) ) return false;

	simd_vec<T, N, sse_kind> v2(src_u, unaligned_t());
	v2.store(dst_u, unaligned_t());

	if ( !test_equal(L, dst_u, r) ) return false;

	for (int i = 0; i < L; ++i) dst_u[i] = T(-1);
	for (int i = 0; i < L; ++i) dst_a[i] = T(-1);

	simd_vec<T, N, sse_kind> v3;
	v3.load(src_a, aligned_t());
	v3.store(dst_a, aligned_t());

	if ( !test_equal(L, dst_a, r) ) return false;

	simd_vec<T, N, sse_kind> v4;
	v4.load(src_u, unaligned_t());
	v4.store(dst_u, unaligned_t());

	if ( !test_equal(L, dst_u, r) ) return false;

	return true;
}


template<typename T, int N>
bool test_zero()
{
	const int L = 4;

	LSIMD_ALIGN_SSE T dst[L] = { T(-1), T(-1), T(-1), T(-1) };

	T r[L] = { T(-1), T(-1), T(-1), T(-1) };
	for (int i = 0; i < N; ++i) r[i] = T(0);

	simd_vec<T, N, sse_kind> v = zero_t();

	v.store(dst, aligned_t());

	return test_equal(L, dst, r);
}


template<typename T, int N>
bool test_add()
{
	const int L = 4;

	LSIMD_ALIGN_SSE T a[L] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[L] = { T(3), T(7), T(2), T(6) };

	T r[L];
	for (int i = 0; i < L; ++i) r[i] = T(-1);
	for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	if ( !(va + vb).impl.test_equal(r) ) return false;

	va += vb;
	if ( !va.impl.test_equal(r) ) return false;

	return true;
}


template<typename T, int N>
bool test_sub()
{
	const int L = 4;

	LSIMD_ALIGN_SSE T a[L] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[L] = { T(3), T(7), T(2), T(6) };

	T r[L];
	for (int i = 0; i < L; ++i) r[i] = T(-1);
	for (int i = 0; i < N; ++i) r[i] = a[i] - b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	if ( !(va - vb).impl.test_equal(r) ) return false;

	va -= vb;
	if ( !va.impl.test_equal(r) ) return false;

	return true;
}


template<typename T, int N>
bool test_mul()
{
	const int L = 4;

	LSIMD_ALIGN_SSE T a[L] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[L] = { T(3), T(7), T(2), T(6) };


	T r[L];
	for (int i = 0; i < L; ++i) r[i] = T(-1);
	for (int i = 0; i < N; ++i) r[i] = a[i] * b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	if ( !(va % vb).impl.test_equal(r) ) return false;

	va %= vb;
	if ( !va.impl.test_equal(r) ) return false;

	return true;
}


template<typename T, int N>
bool test_sum()
{
	const int L = 4;

	LSIMD_ALIGN_SSE T a[L] = { T(1), T(2), T(3), T(4) };

	T s(0);
	for (int i = 0; i < N; ++i) s += a[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());

	return va.sum() == s;
}


template<typename T, int N>
bool test_dot()
{
	const int L = 4;

	LSIMD_ALIGN_SSE T a[L] = { T(1), T(2), T(3), T(4) };
	LSIMD_ALIGN_SSE T b[L] = { T(3), T(7), T(2), T(6) };

	T s(0);
	for (int i = 0; i < N; ++i) s += a[i] * b[i];

	simd_vec<T, N, sse_kind> va(a, aligned_t());
	simd_vec<T, N, sse_kind> vb(b, aligned_t());

	return va.dot(vb) == s;
}



template<typename T, int N>
bool do_tests()
{
	TEST_ITEM( set )
	TEST_ITEM( zero )
	TEST_ITEM( load_store )

	TEST_ITEM( add )
	TEST_ITEM( sub )
	TEST_ITEM( mul )

	TEST_ITEM( sum )
	TEST_ITEM( dot )

	return true;
}





bool do_all_tests()
{
	bool passed = true;

	std::printf("Tests of f32 x 1 \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 1>();
	std::printf("\n");

	std::printf("Tests of f32 x 2 \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 2>();
	std::printf("\n");

	std::printf("Tests of f32 x 3 \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 3>();
	std::printf("\n");

	std::printf("Tests of f32 x 4 \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 4>();
	std::printf("\n");

	std::printf("Tests of f64 x 1 \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 1>();
	std::printf("\n");

	std::printf("Tests of f64 x 2 \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 2>();
	std::printf("\n");

	std::printf("Tests of f64 x 3 \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 3>();
	std::printf("\n");

	std::printf("Tests of f64 x 4 \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 4>();
	std::printf("\n");

	return passed;
}


int main(int argc, char *argv[])
{
	if (do_all_tests())
	{
		std::printf("All tests passed!\n\n");
		return 0;
	}
	else
	{
		std::printf("Some tests failed!\n\n");
		return -1;
	}
}


