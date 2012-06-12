/**
 * @file test_sse_packs.cpp
 *
 * Testing the correctness of sse_pack classes
 *
 * @author Dahua Lin
 */


#include "test_aux.h"

using namespace lsimd;
using namespace ltest;

// explicit instantiation for thorough syntax check


template struct lsimd::sse_pack<f32>;
template struct lsimd::sse_pack<f64>;
template struct lsimd::simd_pack<f32, sse_kind>;
template struct lsimd::simd_pack<f64, sse_kind>;

static_assert( simd<f32, sse_kind>::pack_width == 4, "Incorrect simd pack_width" );
static_assert( simd<f64, sse_kind>::pack_width == 2, "Incorrect simd pack_width" );



/************************************************
 *
 *  constructs
 *
 ************************************************/

GCASE( load_store )
{
	const int w = simd<T, sse_kind>::pack_width;

	LSIMD_ALIGN_SSE T a[4] = {T(1), T(2), T(3), T(5)};
	LSIMD_ALIGN_SSE T b[4] = {T(0), T(0), T(0), T(0)};

	simd_pack<T, sse_kind> v;
	v.load(a, aligned_t());

	clear_zeros(w, b);
	v.store(b, aligned_t());
	ASSERT_VEC_EQ(w, a, b);

	clear_zeros(w, b);
	v.store(b, unaligned_t());
	ASSERT_VEC_EQ(w, a, b);

	v.load(a, unaligned_t());
	clear_zeros(w, b);
	v.store(b, aligned_t());
	ASSERT_VEC_EQ(w, a, b);

	clear_zeros(w, b);
	v.store(b, unaligned_t());
	ASSERT_VEC_EQ(w, a, b);

	simd_pack<T, sse_kind> v2(a, aligned_t());

	clear_zeros(w, b);
	v2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, a, b);

	clear_zeros(w, b);
	v2.store(b, unaligned_t());
	ASSERT_VEC_EQ(w, a, b);

	simd_pack<T, sse_kind> v3(a, unaligned_t());

	clear_zeros(w, b);
	v3.store(b, aligned_t());
	ASSERT_VEC_EQ(w, a, b);

	clear_zeros(w, b);
	v3.store(b, unaligned_t());
	ASSERT_VEC_EQ(w, a, b);

}


template<typename T> class partial_load_tests;

SCASE( partial_load, f32 )
{
	f32 a[4] = {1.f, 2.f, 3.f, 4.f};
	f32 b[4];

	simd_pack<f32, sse_kind> v;

	v.partial_load<1>(a);
	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	v.store(b, unaligned_t());

	f32 r1[4] = {1.f, 0.f, 0.f, 0.f};
	ASSERT_VEC_EQ(4, b, r1);

	v.partial_load<2>(a);
	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	v.store(b, unaligned_t());

	f32 r2[4] = {1.f, 2.f, 0.f, 0.f};
	ASSERT_VEC_EQ(4, b, r2);

	v.partial_load<3>(a);
	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	v.store(b, unaligned_t());

	f32 r3[4] = {1.f, 2.f, 3.f, 0.f};
	ASSERT_VEC_EQ(4, b, r3);
}


SCASE( partial_load, f64 )
{
	f64 a[2] = {1.0, 2.0};
	f64 b[2] = {-1.0, -1.0};

	simd_pack<f64, sse_kind> v;

	v.partial_load<1>(a);
	v.store(b, unaligned_t());

	f64 r1[2] = {1.0, 0.0};
	ASSERT_VEC_EQ(2, b, r1);
}


template<typename T> class partial_store_tests;

SCASE( partial_store, f32 )
{
	f32 a[4] = {1.f, 2.f, 3.f, 4.f};
	f32 b[4];

	simd_pack<f32, sse_kind> p(a, unaligned_t());

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store<1>(b);
	f32 r1[4] = {1.f, -1.f, -1.f, -1.f};
	ASSERT_VEC_EQ(4, b, r1);

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store<2>(b);
	f32 r2[4] = {1.f, 2.f, -1.f, -1.f};
	ASSERT_VEC_EQ(4, b, r2);

	for (int i = 0; i < 4; ++i) b[i] = -1.f;
	p.partial_store<3>(b);
	f32 r3[4] = {1.f, 2.f, 3.f, -1.f};
	ASSERT_VEC_EQ(4, b, r3);
}

SCASE( partial_store, f64 )
{
	f64 a[2] = {1.0, 2.0};
	f64 b[2] = {-1.0, -1.0};

	simd_pack<f64, sse_kind> p(a, unaligned_t());

	p.partial_store<1>(b);
	f64 r1[2] = {1.0, -1.0};
	ASSERT_VEC_EQ(2, b, r1);
}




template<typename T> class set_tests;

SCASE( set, f32 )
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
	ASSERT_VEC_EQ( w, r1, b );

	simd_pack<f32, sse_kind> p2;
	p2.set(v1);

	clear_zeros(w, b);
	p2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r1, b);

	sse_f32pk q(v1, v2, v3, v4);

	clear_zeros(w, b);
	q.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r2, b);

	sse_f32pk q2;
	q2.set(v1, v2, v3, v4);

	clear_zeros(w, b);
	q2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r2, b);

	simd_pack<f32, sse_kind> z = zero_t();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r0, b);

	simd_pack<f32, sse_kind> z2;
	z2.set_zero();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r0, b);

}


SCASE( set, f64 )
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
	ASSERT_VEC_EQ(w, r1, b);

	simd_pack<f64, sse_kind> p2;
	p2.set(v1);

	clear_zeros(w, b);
	p2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r1, b);


	sse_f64pk q(v1, v2);

	clear_zeros(w, b);
	q.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r2, b);

	sse_f64pk q2;
	q2.set(v1, v2);

	clear_zeros(w, b);
	q2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r2, b);


	simd_pack<f64, sse_kind> z = zero_t();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r0, b);

	simd_pack<f64, sse_kind> z2;
	z2.set_zero();

	for (int i = 0; i < w; ++i) b[i] = v1;
	z2.store(b, aligned_t());
	ASSERT_VEC_EQ(w, r0, b);
}


test_pack* tpack_constructs()
{
	ltest::test_pack* tp = new test_pack("constructs");

	tp->add( new load_store_tests<f32>() );
	tp->add( new load_store_tests<f64>() );

	tp->add( new partial_load_tests<f32>() );
	tp->add( new partial_load_tests<f64>() );

	tp->add( new partial_store_tests<f32>() );
	tp->add( new partial_store_tests<f64>() );

	tp->add( new set_tests<f32>() );
	tp->add( new set_tests<f64>() );

	return tp;
}


/************************************************
 *
 *  entry manipulation
 *
 ************************************************/


GCASE( to_scalar )
{
	T sv = T(1.25);
	LSIMD_ALIGN_SSE T src[4] = {sv, T(1), T(2), T(3)};

	simd_pack<T> a(src, aligned_t());

	ASSERT_EQ( a.to_scalar(), sv );
}


template<typename T> class extract_tests;

SCASE( extract, f32 )
{
	LSIMD_ALIGN_SSE f32 src[4] = {1.11f, 2.22f, 3.33f, 4.44f};

	simd_pack<f32, sse_kind> a(src, aligned_t());

	f32 e0 = a.extract<0>();
	f32 e1 = a.extract<1>();
	f32 e2 = a.extract<2>();
	f32 e3 = a.extract<3>();

	ASSERT_EQ( e0, src[0]);
	ASSERT_EQ( e1, src[1]);
	ASSERT_EQ( e2, src[2]);
	ASSERT_EQ( e3, src[3] );
}


SCASE( extract, f64 )
{
	LSIMD_ALIGN_SSE f64 src[4] = {1.11, 2.22, 3.33, 4.44};

	simd_pack<f64, sse_kind> a(src, aligned_t());

	f64 e0 = a.extract<0>();
	f64 e1 = a.extract<1>();

	ASSERT_EQ( e0, src[0] );
	ASSERT_EQ( e1, src[1] );
}


template<typename T> class broadcast_tests;

SCASE( broadcast, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	simd_pack<f32, sse_kind> a(s, aligned_t());

	LSIMD_ALIGN_SSE f32 r0[4] = {s[0], s[0], s[0], s[0]};
	a.broadcast<0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r0);

	LSIMD_ALIGN_SSE f32 r1[4] = {s[1], s[1], s[1], s[1]};
	a.broadcast<1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r1);

	LSIMD_ALIGN_SSE f32 r2[4] = {s[2], s[2], s[2], s[2]};
	a.broadcast<2>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r2);

	LSIMD_ALIGN_SSE f32 r3[4] = {s[3], s[3], s[3], s[3]};
	a.broadcast<3>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r3);
}


SCASE( broadcast, f64 )
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[2] = {0.0, 0.0};

	simd_pack<f64, sse_kind> a(s, aligned_t());

	LSIMD_ALIGN_SSE f64 r0[4] = {s[0], s[0]};
	a.broadcast<0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r0);

	LSIMD_ALIGN_SSE f64 r1[4] = {s[1], s[1]};
	a.broadcast<1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r1);
}


template<typename T> class shift_tests;

SCASE( shift, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	simd_pack<f32, sse_kind> a(s, aligned_t());

	// shift front

	LSIMD_ALIGN_SSE f32 rf0[4] = {1.f, 2.f, 3.f, 4.f};
	a.shift_front<0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rf0);

	LSIMD_ALIGN_SSE f32 rf1[4] = {2.f, 3.f, 4.f, 0.f};
	a.shift_front<1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rf1);

	LSIMD_ALIGN_SSE f32 rf2[4] = {3.f, 4.f, 0.f, 0.f};
	a.shift_front<2>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rf2);

	LSIMD_ALIGN_SSE f32 rf3[4] = {4.f, 0.f, 0.f, 0.f};
	a.shift_front<3>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rf3);

	LSIMD_ALIGN_SSE f32 rf4[4] = {0.f, 0.f, 0.f, 0.f};
	a.shift_front<4>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rf4);

	// shift back

	LSIMD_ALIGN_SSE f32 rb0[4] = {1.f, 2.f, 3.f, 4.f};
	a.shift_back<0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rb0);

	LSIMD_ALIGN_SSE f32 rb1[4] = {0.0f, 1.f, 2.f, 3.f};
	a.shift_back<1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rb1);

	LSIMD_ALIGN_SSE f32 rb2[4] = {0.0f, 0.0f, 1.f, 2.f};
	a.shift_back<2>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rb2);

	LSIMD_ALIGN_SSE f32 rb3[4] = {0.0f, 0.0f, 0.0f, 1.f};
	a.shift_back<3>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rb3);

	LSIMD_ALIGN_SSE f32 rb4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	a.shift_back<4>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, rb4);
}



SCASE( shift, f64 )
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[2] = {0.0, 0.0};

	simd_pack<f64, sse_kind> a(s, aligned_t());

	// shift front

	LSIMD_ALIGN_SSE f64 rf0[2] = {1.0, 2.0};
	a.shift_front<0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, rf0);

	LSIMD_ALIGN_SSE f64 rf1[2] = {2.0, 0.0};
	a.shift_front<1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, rf1);

	LSIMD_ALIGN_SSE f64 rf2[2] = {0.0, 0.0};
	a.shift_front<2>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, rf2);

	// shift back

	LSIMD_ALIGN_SSE f64 rb0[2] = {1.0, 2.0};
	a.shift_back<0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, rb0);

	LSIMD_ALIGN_SSE f64 rb1[2] = {0.0, 1.0};
	a.shift_back<1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, rb1);

	LSIMD_ALIGN_SSE f64 rb2[2] = {0.0, 0.0};
	a.shift_back<2>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, rb2);

}


template<typename T> class duplicate_tests;

SCASE( duplicate, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	sse_f32pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f32 r01[4] = {1.f, 2.f, 1.f, 2.f};
	a.dup_low().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r01);

	LSIMD_ALIGN_SSE f32 r23[4] = {3.f, 4.f, 3.f, 4.f};
	a.dup_high().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r23);

	LSIMD_ALIGN_SSE f32 r02[4] = {1.f, 1.f, 3.f, 3.f};
	a.dup2_low().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r02);

	LSIMD_ALIGN_SSE f32 r13[4] = {2.f, 2.f, 4.f, 4.f};
	a.dup2_high().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r13);

}


SCASE( duplicate, f64 )
{
	LSIMD_ALIGN_SSE f64 s[2] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[2] = {0.0, 0.0};

	sse_f64pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f64 r0[2] = {1.0, 1.0};
	a.dup_low().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r0);

	LSIMD_ALIGN_SSE f64 r1[2] = {2.0, 2.0};
	a.dup_high().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r1);

}


template<typename T> class swizzle_tests;

SCASE( swizzle, f32 )
{
	LSIMD_ALIGN_SSE f32 s[4] = {1.f, 2.f, 3.f, 4.f};
	LSIMD_ALIGN_SSE f32 dst[4] = {0.f, 0.f, 0.f, 0.f};

	sse_f32pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f32 r1[4] = {1.f, 2.f, 3.f, 4.f};
	a.swizzle<0,1,2,3>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r1);

	LSIMD_ALIGN_SSE f32 r2[4] = {4.f, 3.f, 2.f, 1.f};
	a.swizzle<3,2,1,0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r2);

	LSIMD_ALIGN_SSE f32 r3[4] = {2.f, 1.f, 4.f, 3.f};
	a.swizzle<1,0,3,2>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r3);

	LSIMD_ALIGN_SSE f32 r4[4] = {3.f, 4.f, 1.f, 2.f};
	a.swizzle<2,3,0,1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r4);

	LSIMD_ALIGN_SSE f32 r5[4] = {3.f, 2.f, 1.f, 4.f};
	a.swizzle<2,1,0,3>().store(dst, aligned_t());
	ASSERT_VEC_EQ(4, dst, r5);
}


SCASE( swizzle, f64 )
{
	LSIMD_ALIGN_SSE f64 s[4] = {1.0, 2.0};
	LSIMD_ALIGN_SSE f64 dst[4] = {0.0, 0.0};

	sse_f64pk a(s, aligned_t());

	LSIMD_ALIGN_SSE f64 r0[2] = {1.0, 1.0};
	a.swizzle<0,0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r0);

	LSIMD_ALIGN_SSE f64 r1[2] = {1.0, 2.0};
	a.swizzle<0,1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r1);

	LSIMD_ALIGN_SSE f64 r2[2] = {2.0, 1.0};
	a.swizzle<1,0>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r2);

	LSIMD_ALIGN_SSE f64 r3[2] = {2.0, 2.0};
	a.swizzle<1,1>().store(dst, aligned_t());
	ASSERT_VEC_EQ(2, dst, r3);
}

test_pack* tpack_manipulates()
{
	ltest::test_pack* tp = new test_pack("manipulates");

	tp->add( new to_scalar_tests<f32>() );
	tp->add( new to_scalar_tests<f64>() );

	tp->add( new extract_tests<f32>() );
	tp->add( new extract_tests<f64>() );

	tp->add( new broadcast_tests<f32>() );
	tp->add( new broadcast_tests<f64>() );

	tp->add( new shift_tests<f32>() );
	tp->add( new shift_tests<f64>() );

	tp->add( new duplicate_tests<f32>() );
	tp->add( new duplicate_tests<f64>() );

	tp->add( new swizzle_tests<f32>() );
	tp->add( new swizzle_tests<f64>() );

	return tp;
}



/************************************************
 *
 *  entry statistics
 *
 ************************************************/

template<typename T> class sum_tests;

SCASE( sum, f32 )
{
	LSIMD_ALIGN_SSE f32 a[4] = {1.f, 2.f, 3.f, 4.f};
	simd_pack<f32, sse_kind> p(a, aligned_t());

	ASSERT_EQ( p.sum(), 10.f );
	ASSERT_EQ( p.partial_sum<1>(), 1.f );
	ASSERT_EQ( p.partial_sum<2>(), 3.f );
	ASSERT_EQ( p.partial_sum<3>(), 6.f );
}

SCASE( sum, f64 )
{
	LSIMD_ALIGN_SSE f64 a[2] = {2.0, 3.0};
	simd_pack<f64, sse_kind> p(a, aligned_t());

	ASSERT_EQ( p.sum(), 5.0 );
	ASSERT_EQ( p.partial_sum<1>(), 2.0 );
}


template<typename T> class max_tests;

SCASE( max, f32 )
{
	LSIMD_ALIGN_SSE f32 a[4] = {1.f, 3.f, 4.f, 2.f};
	simd_pack<f32, sse_kind> p(a, aligned_t());

	ASSERT_EQ( p.max(), 4.f );
	ASSERT_EQ( p.partial_max<1>(), 1.f );
	ASSERT_EQ( p.partial_max<2>(), 3.f );
	ASSERT_EQ( p.partial_max<3>(), 4.f );
}

SCASE( max, f64 )
{
	LSIMD_ALIGN_SSE f64 a[2] = {2.0, 3.0};
	simd_pack<f64, sse_kind> p(a, aligned_t());

	ASSERT_EQ( p.max(), 3.0 );
	ASSERT_EQ( p.partial_max<1>(), 2.0 );
}


template<typename T> class min_tests;

SCASE( min, f32 )
{
	LSIMD_ALIGN_SSE f32 a[4] = {3.f, 2.f, 4.f, 1.f};
	simd_pack<f32, sse_kind> p(a, aligned_t());

	ASSERT_EQ( p.min(), 1.f );
	ASSERT_EQ( p.partial_min<1>(), 3.f );
	ASSERT_EQ( p.partial_min<2>(), 2.f );
	ASSERT_EQ( p.partial_min<3>(), 2.f );
}

SCASE( min, f64 )
{
	LSIMD_ALIGN_SSE f64 a[2] = {2.0, 3.0};
	simd_pack<f64, sse_kind> p(a, aligned_t());

	ASSERT_EQ( p.min(), 2.0 );
	ASSERT_EQ( p.partial_min<1>(), 2.0 );
}



test_pack* tpack_statistics()
{
	ltest::test_pack* tp = new test_pack("statistics");

	tp->add( new sum_tests<f32>() );
	tp->add( new sum_tests<f64>() );

	tp->add( new max_tests<f32>() );
	tp->add( new max_tests<f64>() );

	tp->add( new min_tests<f32>() );
	tp->add( new min_tests<f64>() );

	return tp;
}


void lsimd::add_test_packs()
{
	lsimd_main_suite.add( tpack_constructs() );
	lsimd_main_suite.add( tpack_manipulates() );
	lsimd_main_suite.add( tpack_statistics() );
}





