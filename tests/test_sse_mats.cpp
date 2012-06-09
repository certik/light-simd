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

template class lsimd::sse_mat<f32, 2, 2>;
template class lsimd::sse_mat<f32, 2, 3>;
template class lsimd::sse_mat<f32, 2, 4>;
template class lsimd::sse_mat<f32, 3, 2>;
template class lsimd::sse_mat<f32, 3, 3>;
template class lsimd::sse_mat<f32, 3, 4>;
template class lsimd::sse_mat<f32, 4, 2>;
template class lsimd::sse_mat<f32, 4, 3>;
template class lsimd::sse_mat<f32, 4, 4>;

template struct lsimd::simd_mat<f32, 2, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 2, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 2, 4, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 3, 4, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 2, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 3, sse_kind>;
template struct lsimd::simd_mat<f32, 4, 4, sse_kind>;


#define TEST_ITEM( name ) \
	if ( test_##name<T, M, N>() ) { std::printf("Tests on %-16s: passed\n", #name);  } \
	else { std::printf("Tests on %-16s: failed\n", #name); return false; }


template<typename T, int M, int N>
bool do_tests()
{

	return true;
}

bool do_all_tests()
{
	bool passed = true;

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



