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


#define TEST_ITEM( name ) \
	if ( test_##name<T>() ) { std::printf("Tests on %-16s: passed\n", #name);  } \
	else { std::printf("Tests on %-16s: failed\n", #name); return false; }


template<typename T>
bool test_load_store()
{
	const int w = sse_pack<T>::pack_width;

	LSIMD_ALIGN_SSE T a[4] = {1.f, 2.f, 3.f, 5.f};
	LSIMD_ALIGN_SSE T b[4];

	sse_pack<T> p;
	p.load(a, aligned_t());

	clear_zeros(w, b);
	p.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	p.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;

	p.load(a, unaligned_t());

	clear_zeros(w, b);
	p.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	p.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;


	sse_pack<T> qa(a, aligned_t());

	clear_zeros(w, b);
	qa.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	qa.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;

	sse_pack<T> qu(a, unaligned_t());

	clear_zeros(w, b);
	qu.store(b, aligned_t());
	if (!test_equal(w, a, b)) return false;

	clear_zeros(w, b);
	qu.store(b, unaligned_t());
	if (!test_equal(w, a, b)) return false;

	return true;
}


template<typename T>
bool test_set();

template<>
bool test_set<float>()
{
	const int w = sse_f32p::pack_width;

	const f32 v0(0);
	const f32 v1(1.23f);
	const f32 v2(-3.42f);
	const f32 v3(4.57f);
	const f32 v4(-5.26f);

	LSIMD_ALIGN_SSE f32 r0[w] = {v0, v0, v0, v0};
	LSIMD_ALIGN_SSE f32 r1[w] = {v1, v1, v1, v1};
	LSIMD_ALIGN_SSE f32 r2[w] = {v1, v2, v3, v4};

	LSIMD_ALIGN_SSE f32 b[w];

	sse_f32p p(v1);

	clear_zeros(w, b);
	p.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;

	sse_f32p p2;
	p2.set(v1);

	clear_zeros(w, b);
	p2.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;


	sse_f32p q(v1, v2, v3, v4);

	clear_zeros(w, b);
	q.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;

	sse_f32p q2;
	q2.set(v1, v2, v3, v4);

	clear_zeros(w, b);
	q2.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;


	sse_f32p z =  zero_t();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	sse_f32p z2;
	z2.set_zero();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z2.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	return true;
}


template<>
bool test_set<double>()
{
	const int w = sse_f64p::pack_width;

	const f64 v0(0);
	const f64 v1(1.23);
	const f64 v2(-3.42);

	LSIMD_ALIGN_SSE f64 r0[w] = {v0, v0};
	LSIMD_ALIGN_SSE f64 r1[w] = {v1, v1};
	LSIMD_ALIGN_SSE f64 r2[w] = {v1, v2};

	LSIMD_ALIGN_SSE f64 b[w];

	sse_f64p p(v1);

	clear_zeros(w, b);
	p.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;

	sse_f64p p2;
	p2.set(v1);

	clear_zeros(w, b);
	p2.store(b, aligned_t());
	if (!test_equal(w, r1, b)) return false;


	sse_f64p q(v1, v2);

	clear_zeros(w, b);
	q.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;

	sse_f64p q2;
	q2.set(v1, v2);

	clear_zeros(w, b);
	q2.store(b, aligned_t());
	if (!test_equal(w, r2, b)) return false;


	sse_f64p z = zero_t();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	sse_f64p z2;
	z2.set_zero();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z2.store(b, aligned_t());
	if (!test_equal(w, r0, b)) return false;

	return true;
}




template<typename T>
bool do_tests()
{
	TEST_ITEM( load_store )
	TEST_ITEM( set )

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


