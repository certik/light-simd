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
bool do_tests()
{
	TEST_ITEM( load_store )
	TEST_ITEM( set )
	TEST_ITEM( extract )

	return true;
}



bool do_all_tests()
{
	std::printf("Tests of sse_pack<f32>\n");
	std::printf("==============================\n");
	if (!do_tests<f32>()) return false;

	std::printf("\n");

	std::printf("Tests of sse_pack<f64>\n");
	std::printf("==============================\n");
	if (!do_tests<f64>()) return false;

	std::printf("\n");

	return true;
}


int main(int argc, char *argv[])
{
	if (do_all_tests())
	{
		return 0;
	}
	else
	{
		return -1;
	}
}


