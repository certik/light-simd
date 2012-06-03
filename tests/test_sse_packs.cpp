/**
 * @file test_sse_packs.cpp
 *
 * Testing the correctness of sse_pack classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;

// explicit instantiation for thorough syntax check


template struct lsimd::sse_pack<f32>;
template struct lsimd::sse_pack<f64>;
template struct lsimd::simd_pack<f32, sse_kind>;
template struct lsimd::simd_pack<f64, sse_kind>;

static_assert( simd<f32, sse_kind>::pack_width == 4, "Incorrect simd pack_width" );
static_assert( simd<f64, sse_kind>::pack_width == 2, "Incorrect simd pack_width" );


#define TEST_ITEM( name ) \
	if ( test_##name<T>() ) { std::printf("Tests on %-16s: passed\n", #name);  } \
	else { std::printf("Tests on %-16s: failed\n", #name); return false; }


template<typename T>
bool test_load_store()
{
	const int w = simd<T, sse_kind>::pack_width;

	LSIMD_ALIGN_SSE T a[4] = {1.f, 2.f, 3.f, 5.f};
	LSIMD_ALIGN_SSE T b[4];

	simd_pack<T, sse_kind> v;
	v.load(a, aligned_t());

	clear_zeros(w, b);
	v.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	v.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;

	v.load(a, unaligned_t());
	clear_zeros(w, b);
	v.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	v.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;


	simd_pack<T, sse_kind> v2(a, aligned_t());

	clear_zeros(w, b);
	v2.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	v2.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;

	simd_pack<T, sse_kind> v3(a, unaligned_t());

	clear_zeros(w, b);
	v3.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	v3.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;

	return true;
}


template<typename T>
bool test_set();

template<>
bool test_set<f32>()
{
	const int w = sse_f32pk::pack_width;

	const f32 v0(0);
	const f32 v1(1.23f);
	const f32 v2(-3.42f);
	const f32 v3(4.57f);
	const f32 v4(-5.26f);

	LSIMD_ALIGN_SSE f32 r0[w] = {v0, v0, v0, v0};
	LSIMD_ALIGN_SSE f32 r1[w] = {v1, v1, v1, v1};
	LSIMD_ALIGN_SSE f32 r2[w] = {v1, v2, v3, v4};

	LSIMD_ALIGN_SSE f32 b[w];

	simd_pack<f32, sse_kind> p(v1);

	clear_zeros(w, b);
	p.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;

	simd_pack<f32, sse_kind> p2;
	p2.set(v1);

	clear_zeros(w, b);
	p2.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;

	sse_f32pk q(v1, v2, v3, v4);

	clear_zeros(w, b);
	q.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;

	sse_f32pk q2;
	q2.set(v1, v2, v3, v4);

	clear_zeros(w, b);
	q2.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;


	simd_pack<f32, sse_kind> z = zero_t();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	simd_pack<f32, sse_kind> z2;
	z2.set_zero();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z2.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	return true;
}


template<>
bool test_set<f64>()
{
	const int w = sse_f64pk::pack_width;

	const f64 v0(0);
	const f64 v1(1.23);
	const f64 v2(-3.42);

	LSIMD_ALIGN_SSE f64 r0[w] = {v0, v0};
	LSIMD_ALIGN_SSE f64 r1[w] = {v1, v1};
	LSIMD_ALIGN_SSE f64 r2[w] = {v1, v2};

	LSIMD_ALIGN_SSE f64 b[w];

	simd_pack<f64, sse_kind> p(v1);

	clear_zeros(w, b);
	p.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;

	simd_pack<f64, sse_kind> p2;
	p2.set(v1);

	clear_zeros(w, b);
	p2.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;


	sse_f64pk q(v1, v2);

	clear_zeros(w, b);
	q.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;

	sse_f64pk q2;
	q2.set(v1, v2);

	clear_zeros(w, b);
	q2.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;


	simd_pack<f64, sse_kind> z = zero_t();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	simd_pack<f64, sse_kind> z2;
	z2.set_zero();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z2.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	return true;
}


template<typename T>
bool test_to_scalar()
{
	T sv = T(1.25);
	LSIMD_ALIGN_SSE T src[4] = {sv, T(1), T(2), T(3)};

	simd_pack<T> a(src, aligned_t());

	return a.to_scalar() == sv;
}


template<typename T>
bool test_extract();

template<>
bool test_extract<f32>()
{
	LSIMD_ALIGN_SSE f32 src[4] = {1.11f, 2.22f, 3.33f, 4.44f};

	sse_f32pk a(src, aligned_t());

	f32 e0 = a.extract<0>();
	f32 e1 = a.extract<1>();
	f32 e2 = a.extract<2>();
	f32 e3 = a.extract<3>();

	if ( e0 == src[0] && e1 == src[1] && e2 == src[2] && e3 == src[3] )
	{
		return true;
	}
	else
	{
		return false;
	}
}


template<>
bool test_extract<f64>()
{
	LSIMD_ALIGN_SSE f64 src[4] = {1.11, 2.22, 3.33, 4.44};

	sse_f64pk a(src, aligned_t());

	f64 e0 = a.extract<0>();
	f64 e1 = a.extract<1>();

	if ( e0 == src[0] && e1 == src[1] )
	{
		return true;
	}
	else
	{
		return false;
	}
}


template<typename T>
bool test_broadcast();

template<>
bool test_broadcast<f32>()
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	sse_f32pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f32 r0[4] = {s[0], s[0], s[0], s[0]};
	a.broadcast<0>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r0) ) return false;

	LSIMD_ALIGN_SSE f32 r1[4] = {s[1], s[1], s[1], s[1]};
	a.broadcast<1>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r1) ) return false;

	LSIMD_ALIGN_SSE f32 r2[4] = {s[2], s[2], s[2], s[2]};
	a.broadcast<2>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r2) ) return false;

	LSIMD_ALIGN_SSE f32 r3[4] = {s[3], s[3], s[3], s[3]};
	a.broadcast<3>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r3) ) return false;

	return true;
}


template<>
bool test_broadcast<f64>()
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[2] = {0.0, 0.0};

	sse_f64pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f64 r0[4] = {s[0], s[0]};
	a.broadcast<0>().store(dst, aligned_t());
	if ( !test_equal(2, dst, r0) ) return false;

	LSIMD_ALIGN_SSE f64 r1[4] = {s[1], s[1]};
	a.broadcast<1>().store(dst, aligned_t());
	if ( !test_equal(2, dst, r1) ) return false;

	return true;
}


template<typename T>
bool test_swizzle();

template<>
bool test_swizzle<f32>()
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	sse_f32pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f32 r1[4] = {1.f, 2.f, 3.f, 4.f};
	a.swizzle<0,1,2,3>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r1) ) return false;

	LSIMD_ALIGN_SSE f32 r2[4] = {4.f, 3.f, 2.f, 1.f};
	a.swizzle<3,2,1,0>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r2) ) return false;

	LSIMD_ALIGN_SSE f32 r3[4] = {2.f, 1.f, 4.f, 3.f};
	a.swizzle<1,0,3,2>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r3) ) return false;

	LSIMD_ALIGN_SSE f32 r4[4] = {3.f, 4.f, 1.f, 2.f};
	a.swizzle<2,3,0,1>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r4) ) return false;

	LSIMD_ALIGN_SSE f32 r5[4] = {3.f, 2.f, 1.f, 4.f};
	a.swizzle<2,1,0,3>().store(dst, aligned_t());
	if ( !test_equal(4, dst, r5) ) return false;

	return true;
}


template<>
bool test_swizzle<f64>()
{
	LSIMD_ALIGN_SSE f64 s[4] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[4] = {0.0, 0.0};

	sse_f64pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f64 r0[2] = {1.0, 1.0};
	a.swizzle<0,0>().store(dst, aligned_t());
	if ( !test_equal(2, dst, r0) ) return false;

	LSIMD_ALIGN_SSE f64 r1[2] = {1.0, 2.0};
	a.swizzle<0,1>().store(dst, aligned_t());
	if ( !test_equal(2, dst, r1) ) return false;

	LSIMD_ALIGN_SSE f64 r2[2] = {2.0, 1.0};
	a.swizzle<1,0>().store(dst, aligned_t());
	if ( !test_equal(2, dst, r2) ) return false;

	LSIMD_ALIGN_SSE f64 r3[2] = {2.0, 2.0};
	a.swizzle<1,1>().store(dst, aligned_t());
	if ( !test_equal(2, dst, r3) ) return false;

	return true;
}



template<typename T>
bool test_shift();

template<>
bool test_shift<f32>()
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	sse_f32pk a(s, aligned_t());

	// shift front

	LSIMD_ALIGN_SSE f32 rf0[4] = {1.f, 2.f, 3.f, 4.f};
	a.shift_front<0>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rf0) ) return false;

	LSIMD_ALIGN_SSE f32 rf1[4] = {2.f, 3.f, 4.f, 0.f};
	a.shift_front<1>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rf1) ) return false;

	LSIMD_ALIGN_SSE f32 rf2[4] = {3.f, 4.f, 0.f, 0.f};
	a.shift_front<2>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rf2) ) return false;

	LSIMD_ALIGN_SSE f32 rf3[4] = {4.f, 0.f, 0.f, 0.f};
	a.shift_front<3>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rf3) ) return false;

	LSIMD_ALIGN_SSE f32 rf4[4] = {0.f, 0.f, 0.f, 0.f};
	a.shift_front<4>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rf4) ) return false;

	// shift back

	LSIMD_ALIGN_SSE f32 rb0[4] = {1.f, 2.f, 3.f, 4.f};
	a.shift_back<0>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rb0) ) return false;

	LSIMD_ALIGN_SSE f32 rb1[4] = {0.0f, 1.f, 2.f, 3.f};
	a.shift_back<1>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rb1) ) return false;

	LSIMD_ALIGN_SSE f32 rb2[4] = {0.0f, 0.0f, 1.f, 2.f};
	a.shift_back<2>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rb2) ) return false;

	LSIMD_ALIGN_SSE f32 rb3[4] = {0.0f, 0.0f, 0.0f, 1.f};
	a.shift_back<3>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rb3) ) return false;

	LSIMD_ALIGN_SSE f32 rb4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	a.shift_back<4>().store(dst, aligned_t());
	if ( !test_equal(4, dst, rb4) ) return false;

	return true;
}



template<>
bool test_shift<f64>()
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[2] = {0.0, 0.0};

	sse_f64pk a(s, aligned_t());

	// shift front

	LSIMD_ALIGN_SSE f64 rf0[2] = {1.0, 2.0};
	a.shift_front<0>().store(dst, aligned_t());
	if ( !test_equal(2, dst, rf0) ) return false;

	LSIMD_ALIGN_SSE f64 rf1[2] = {2.0, 0.0};
	a.shift_front<1>().store(dst, aligned_t());
	if ( !test_equal(2, dst, rf1) ) return false;

	LSIMD_ALIGN_SSE f64 rf2[2] = {0.0, 0.0};
	a.shift_front<2>().store(dst, aligned_t());
	if ( !test_equal(2, dst, rf2) ) return false;

	// shift back

	LSIMD_ALIGN_SSE f64 rb0[2] = {1.0, 2.0};
	a.shift_back<0>().store(dst, aligned_t());
	if ( !test_equal(2, dst, rb0) ) return false;

	LSIMD_ALIGN_SSE f64 rb1[2] = {0.0, 1.0};
	a.shift_back<1>().store(dst, aligned_t());
	if ( !test_equal(2, dst, rb1) ) return false;

	LSIMD_ALIGN_SSE f64 rb2[2] = {0.0, 0.0};
	a.shift_back<2>().store(dst, aligned_t());
	if ( !test_equal(2, dst, rb2) ) return false;

	return true;
}


template<typename T>
bool do_tests()
{
	TEST_ITEM( load_store )
	TEST_ITEM( set )
	TEST_ITEM( to_scalar )
	TEST_ITEM( extract )
	TEST_ITEM( broadcast )
	TEST_ITEM( swizzle )
	TEST_ITEM( shift )

	return true;
}



bool do_all_tests()
{
	bool passed = true;

	std::printf("Tests of sse_pack<f32>\n");
	std::printf("==============================\n");
	if (!do_tests<f32>()) passed = false;

	std::printf("\n");

	std::printf("Tests of sse_pack<f64>\n");
	std::printf("==============================\n");
	if (!do_tests<f64>()) passed = false;

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


