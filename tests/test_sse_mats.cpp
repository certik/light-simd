/**
 * @file test_sse_mats.cpp
 *
 * Test the correctness of sse_mat classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;

// explicit instantiation for thorough syntax check
/*
template class lsimd::sse_mat<f32, 2, 2>;
template class lsimd::sse_mat<f32, 2, 3>;
template class lsimd::sse_mat<f32, 2, 4>;
template class lsimd::sse_mat<f32, 3, 2>;
template class lsimd::sse_mat<f32, 3, 3>;
template class lsimd::sse_mat<f32, 3, 4>;
template class lsimd::sse_mat<f32, 4, 2>;
template class lsimd::sse_mat<f32, 4, 3>;
template class lsimd::sse_mat<f32, 4, 4>;

template class lsimd::sse_mat<f64, 2, 2>;
template class lsimd::sse_mat<f64, 2, 3>;
template class lsimd::sse_mat<f64, 2, 4>;
template class lsimd::sse_mat<f64, 3, 2>;
template class lsimd::sse_mat<f64, 3, 3>;
template class lsimd::sse_mat<f64, 3, 4>;
template class lsimd::sse_mat<f64, 4, 2>;
template class lsimd::sse_mat<f64, 4, 3>;
template class lsimd::sse_mat<f64, 4, 4>;


template struct lsimd::simd_mat<f32, 2, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 2, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 2, 4, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 4, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 4, sse_kind>;

template struct lsimd::simd_mat<f64, 2, 2, sse_kind>;
template struct lsimd::simd_mat<f64, 2, 3, sse_kind>;
template struct lsimd::simd_mat<f64, 2, 4, sse_kind>;
template struct lsimd::simd_mat<f64, 3, 2, sse_kind>;
template struct lsimd::simd_mat<f64, 3, 3, sse_kind>;
template struct lsimd::simd_mat<f64, 3, 4, sse_kind>;
template struct lsimd::simd_mat<f64, 4, 2, sse_kind>;
template struct lsimd::simd_mat<f64, 4, 3, sse_kind>;
template struct lsimd::simd_mat<f64, 4, 4, sse_kind>;
*/

#define TEST_ITEM( name ) \
	if ( test_##name<T, M, N>() ) { std::printf("Tests on %-16s: passed\n", #name);  } \
	else { std::printf("Tests on %-16s: failed\n", #name); return false; }


const int MaxArrLen = 36;
const int LDa = 8;
const int LDu = 5;


template<typename T, int M, int N>
bool test_zero()
{
	T r[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(0);

	simd_mat<T, M, N, sse_kind> a = zero_t();

	if (!a.impl.test_equal(r)) return false;

	return true;
}



template<typename T, int M, int N>
bool test_load()
{
	LSIMD_ALIGN_SSE T src[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) src[i] = T(i+1);

	simd_mat<T, M, N, sse_kind> aa(src, aligned_t());
	if (!aa.impl.test_equal(src)) return false;

	simd_mat<T, M, N, sse_kind> au(src + 1, unaligned_t());
	if (!au.impl.test_equal(src+1)) return false;


	T br[M * N];

	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) br[i + j * M] = src[i + j * LDa];

	simd_mat<T, M, N, sse_kind> ba(src, LDa, aligned_t());
	if (!ba.impl.test_equal(br))
	{
		ba.impl.dump("%4g");
		return false;
	}

	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) br[i + j * M] = src[1 + i + j * LDu];

	simd_mat<T, M, N, sse_kind> bu(src+1, LDu, unaligned_t());
	if (!bu.impl.test_equal(br)) return false;

	return true;
}


template<typename T, int M, int N>
bool test_store()
{
	LSIMD_ALIGN_SSE T src[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) src[i] = T(i+1);

	LSIMD_ALIGN_SSE T da[MaxArrLen];
	T dd[MaxArrLen];
	T r[MaxArrLen];

	simd_mat<T, M, N, sse_kind> a(src, aligned_t());
	if (!a.impl.test_equal(src)) return false;

	// store continuous align

	for (int i = 0; i < MaxArrLen; ++i) da[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int i = 0; i < M * N; ++i) r[i] = src[i];

	a.store(da, aligned_t());

	if (!test_equal(MaxArrLen, da, r)) return false;

	// store continuous non-align

	for (int i = 0; i < MaxArrLen; ++i) dd[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int i = 0; i < M * N; ++i) r[i+1] = src[i];

	a.store(dd + 1, unaligned_t());

	if (!test_equal(MaxArrLen, dd, r)) return false;

	// store non-continuous align

	for (int i = 0; i < MaxArrLen; ++i) da[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * LDa] = src[i + j * M];

	a.store(da, LDa, aligned_t());

	if (!test_equal(MaxArrLen, da, r)) return false;

	// store non-continuous non-align

	for (int i = 0; i < MaxArrLen; ++i) dd[i] = T(-1);
	for (int i = 0; i < MaxArrLen; ++i) r[i] = T(-1);
	for (int j = 0; j < N; ++j)
		for (int i = 0; i < M; ++i) r[1 + i + j * LDu] = src[i + j * M];

	a.store(dd + 1, LDu, unaligned_t());

	if (!test_equal(MaxArrLen, dd, r)) return false;

	return true;
}


template<typename T, int M, int N>
bool test_load_trans()
{
	LSIMD_ALIGN_SSE T src[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) src[i] = T(i+1);
	T r[M * N];

	simd_mat<T, M, N, sse_kind> a;

	a.load_trans(src, aligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[i * N + j];

	if (!a.impl.test_equal(r)) return false;

	a.load_trans(src + 1, unaligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[1 + i * N + j];

	if (!a.impl.test_equal(r)) return false;

	a.load_trans(src, LDa, aligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[i * LDa + j];

	if (!a.impl.test_equal(r)) return false;

	a.load_trans(src + 1, LDu, unaligned_t());

	for (int j = 0; j  < N; ++j)
		for (int i = 0; i < M; ++i) r[i + j * M] = src[1 + i * LDu + j];

	if (!a.impl.test_equal(r)) return false;

	return true;
}


template<typename T, int M, int N>
bool test_arith()
{
	LSIMD_ALIGN_SSE T sa[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sa[i] = T(i+1);

	LSIMD_ALIGN_SSE T sb[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sb[i] = T(MaxArrLen - 2 * i);

	T r[M * N];

	simd_mat<T, M, N, sse_kind> a(sa, aligned_t());
	simd_mat<T, M, N, sse_kind> b(sb, aligned_t());

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] + sb[i];
	if (!(a + b).impl.test_equal(r)) return false;

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] - sb[i];
	if (!(a - b).impl.test_equal(r)) return false;

	for (int i = 0; i < M * N; ++i) r[i] = sa[i] * sb[i];
	if (!(a % b).impl.test_equal(r)) return false;

	return true;
}


template<typename T, int M, int N>
bool test_mtimes()
{
	LSIMD_ALIGN_SSE T sa[MaxArrLen];
	for (int i = 0; i < MaxArrLen; ++i) sa[i] = T(i+1);

	LSIMD_ALIGN_SSE T sx[N];
	for (int j = 0; j < N; ++j) sx[j] = T(j+1);

	LSIMD_ALIGN_SSE T r[M];
	for (int i = 0; i < M; ++i)
	{
		T s(0);
		for (int j = 0; j < N; ++j) s += sa[i + j * M] * sx[j];
		r[i] = s;
	}

	simd_mat<T, M, N, sse_kind> a(sa, aligned_t());
	simd_vec<T, N, sse_kind> x(sx, aligned_t());

	simd_vec<T, M, sse_kind> y = a * x;

	if (!y.impl.test_equal(r))
	{
		sse_vec<T, M> yr(r, aligned_t());
		std::printf("yr = "); yr.dump("%4g"); std::printf("\n");
		std::printf("y  = "); y.impl.dump("%4g"); std::printf("\n");
		return false;
	}

	return true;
}


template<typename T, int M, int N>
bool do_tests()
{
	TEST_ITEM( zero )
	TEST_ITEM( load )
	// TEST_ITEM( store )
	// TEST_ITEM( load_trans )
	// TEST_ITEM( arith )
	// TEST_ITEM( mtimes )

	return true;
}


bool do_all_tests()
{
	bool passed = true;

	/*
	std::printf("Tests of f32 [2 x 2] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 2, 2>();
	std::printf("\n");

	std::printf("Tests of f32 [2 x 3] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 2, 3>();
	std::printf("\n");

	std::printf("Tests of f32 [2 x 4] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 2, 4>();
	std::printf("\n");

	std::printf("Tests of f32 [3 x 2] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 3, 2>();
	std::printf("\n");

	std::printf("Tests of f32 [3 x 3] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 3, 3>();
	std::printf("\n");

	std::printf("Tests of f32 [3 x 4] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 3, 4>();
	std::printf("\n");

	std::printf("Tests of f32 [4 x 2] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 4, 2>();
	std::printf("\n");

	std::printf("Tests of f32 [4 x 3] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 4, 3>();
	std::printf("\n");

	std::printf("Tests of f32 [4 x 4] \n");
	std::printf("==============================\n");
	passed &= do_tests<f32, 4, 4>();
	std::printf("\n");
*/


	std::printf("Tests of f64 [2 x 2] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 2, 2>();
	std::printf("\n");

	std::printf("Tests of f64 [2 x 3] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 2, 3>();
	std::printf("\n");

	std::printf("Tests of f64 [2 x 4] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 2, 4>();
	std::printf("\n");

	std::printf("Tests of f64 [3 x 2] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 3, 2>();
	std::printf("\n");

	std::printf("Tests of f64 [3 x 3] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 3, 3>();
	std::printf("\n");

	std::printf("Tests of f64 [3 x 4] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 3, 4>();
	std::printf("\n");

	std::printf("Tests of f64 [4 x 2] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 4, 2>();
	std::printf("\n");

	std::printf("Tests of f64 [4 x 3] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 4, 3>();
	std::printf("\n");

	std::printf("Tests of f64 [4 x 4] \n");
	std::printf("==============================\n");
	passed &= do_tests<f64, 4, 4>();
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



