/**
 * @file test_sse_mm.cpp
 *
 * Unit testing for SSE matrix-multiplication
 *
 * @author Dahua Lin
 */


#include "test_aux.h"
#include "linalg_ref.h"

using namespace lsimd;
using namespace ltest;


template<typename T, int M, int K, int N>
class matmul_tests : public test_case
{
	char m_name[128];

	LSIMD_ALIGN_SSE T arr_a[M * K];
	LSIMD_ALIGN_SSE T arr_b[K * N];
	LSIMD_ALIGN_SSE T arr_c0[M * N];
	LSIMD_ALIGN_SSE T arr_cr[M * N];

public:
	matmul_tests()
	{
		std::sprintf(m_name, "mm (%d x %d) * (%d x %d)", M, K, K, N);
	}

	const char *name() const
	{
		return m_name;
	}

	void run()
	{
		// compute ground-truth

		simple_mat<T, M, K> a0(arr_a);
		simple_mat<T, K, N> b0(arr_b);
		simple_mat<T, M, N> c0(arr_c0);

		for (int i = 0; i < M * K; ++i) a0[i] = T(i + 1);
		for (int i = 0; i < K * N; ++i) b0[i] = T(i + 2);
		for (int i = 0; i < M * N; ++i) c0[i] = T(-1);

		ref_mm(a0, b0, c0);

		// use SIMD

		simd_mat<T, M, K, sse_kind> a( arr_a, aligned_t() );
		simd_mat<T, K, N, sse_kind> b( arr_b, aligned_t() );
		simd_mat<T, M, N, sse_kind> c = a * b;

		c.store( arr_cr, aligned_t() );

		if (!test_vector_equal(M * N, arr_c0, arr_cr))
		{
			std::printf("\n");
			simple_mat<T, M, N> cr(arr_cr);

			a0.print("%5g ");
			std::printf("*\n");
			b0.print("%5g ");
			std::printf("==>\n");
			c0.print("%5g ");
			std::printf("actual result = \n");
			cr.print("%5g ");
			std::printf("\n");
		}

		ASSERT_VEC_EQ( M * N, arr_c0, arr_cr );
	}
};


template<typename T>
void add_cases_to_matmul_tpack(test_pack* tp)
{
	tp->add( new matmul_tests<T, 2, 2, 2>() );
	tp->add( new matmul_tests<T, 2, 2, 3>() );
	tp->add( new matmul_tests<T, 2, 2, 4>() );

	tp->add( new matmul_tests<T, 2, 3, 2>() );
	tp->add( new matmul_tests<T, 2, 3, 3>() );
	tp->add( new matmul_tests<T, 2, 3, 4>() );

	tp->add( new matmul_tests<T, 2, 4, 2>() );
	tp->add( new matmul_tests<T, 2, 4, 3>() );
	tp->add( new matmul_tests<T, 2, 4, 4>() );

	tp->add( new matmul_tests<T, 3, 2, 2>() );
	tp->add( new matmul_tests<T, 3, 2, 3>() );
	tp->add( new matmul_tests<T, 3, 2, 4>() );

	tp->add( new matmul_tests<T, 3, 3, 2>() );
	tp->add( new matmul_tests<T, 3, 3, 3>() );
	tp->add( new matmul_tests<T, 3, 3, 4>() );

	tp->add( new matmul_tests<T, 3, 4, 2>() );
	tp->add( new matmul_tests<T, 3, 4, 3>() );
	tp->add( new matmul_tests<T, 3, 4, 4>() );

	tp->add( new matmul_tests<T, 4, 2, 2>() );
	tp->add( new matmul_tests<T, 4, 2, 3>() );
	tp->add( new matmul_tests<T, 4, 2, 4>() );

	tp->add( new matmul_tests<T, 4, 3, 2>() );
	tp->add( new matmul_tests<T, 4, 3, 3>() );
	tp->add( new matmul_tests<T, 4, 3, 4>() );

	tp->add( new matmul_tests<T, 4, 4, 2>() );
	tp->add( new matmul_tests<T, 4, 4, 3>() );
	tp->add( new matmul_tests<T, 4, 4, 4>() );
}


test_pack* matmul_tpack_f32()
{
	test_pack *tp = new test_pack( "matmul_f32" );
	add_cases_to_matmul_tpack<f32>(tp);
	return tp;
}

test_pack* matmul_tpack_f64()
{
	test_pack *tp = new test_pack( "matmul_f64" );
	add_cases_to_matmul_tpack<f64>(tp);
	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( matmul_tpack_f32() );
	lsimd_main_suite.add( matmul_tpack_f64() );
}







