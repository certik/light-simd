/*
 * @file test_sse_inv.cpp
 *
 * Unit testing of matrix inverse
 *
 * @author Dahua Lin
 */


#include "test_aux.h"
#include "linalg_ref.h"

using namespace lsimd;
using namespace ltest;

template<typename T>
void special_fill_mat(int n, T *x)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			T b = T(4 - j);

			T v = b;
			for (int k = 0; k < i; ++k) v *= b;

			x[i + j * n] = v;
		}
	}
}

template<typename T, int N>
T special_det()
{
	switch (N)
	{
	case 2: return T(-12);
	case 3: return T(-48);
	case 4: return T(288);
	}

	return T(0);
}


GCASE1( det )
{
	LSIMD_ALIGN_SSE T src[N * N];
	special_fill_mat(N, src);

	sse_mat<T, N, N> a(src, aligned_t());

	simple_mat<T, N, N> a0(src);

	T v0 = special_det<T, N>();
	ASSERT_EQ( ref_determinant(a0), v0 );

	// std::printf("det(a) --> %g\n", det(a));
	ASSERT_EQ( det(a), v0 );
}


GCASE1( inv )
{
	LSIMD_ALIGN_SSE T av[N * N];
	LSIMD_ALIGN_SSE T bv[N * N];
	special_fill_mat(N, av);

	sse_mat<T,N,N> a(av, aligned_t());
	sse_mat<T,N,N> inv_a = inv(a);
	inv_a.store(bv, aligned_t());

	T E[N * N];
	T E0[N * N];

	for (int i = 0; i < N; ++i) E0[i + i * N] = T(1);

	simple_mat<T,N,N> am(av);
	simple_mat<T,N,N> bm(bv);
	simple_mat<T,N,N> cm(E);
	ref_mm(am, bm, cm);

	T tol = sizeof(T) == 4 ? T(1.0e-6) : T(1.0e-12);

	ASSERT_VEC_APPROX(N*N, E, E0, tol);
}


test_pack* det_tpack()
{
	test_pack *tp = new test_pack( "det" );

	tp->add( new det_tests<f32, 2>() );
	tp->add( new det_tests<f64, 2>() );

	tp->add( new det_tests<f32, 3>() );
	tp->add( new det_tests<f64, 3>() );

	return tp;
}

test_pack* inv_tpack()
{
	test_pack *tp = new test_pack( "inv" );

	tp->add( new inv_tests<f32, 2>() );
	tp->add( new inv_tests<f64, 2>() );

	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( det_tpack() );
	lsimd_main_suite.add( inv_tpack() );
}








