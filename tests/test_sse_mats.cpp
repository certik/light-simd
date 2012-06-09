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


bool do_all_tests()
{
	return true;
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



