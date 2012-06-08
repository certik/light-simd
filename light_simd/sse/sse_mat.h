/*
 * @file sse_mat.h
 *
 * SSE-based fixed-size small matrices
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef LSIMD_SSE_MAT_H_
#define LSIMD_SSE_MAT_H_

#include "sse_vec.h"

namespace lsimd
{

	template<typename T, int M, int N> class _sse_mat;
	template<typename T, int M, int N> class sse_mat;


	/********************************************
	 *
	 *  sse_mat class (generic)
	 *
	 ********************************************/

	template<typename T, int M, int N>
	class sse_mat
	{
	private:
		_sse_mat<T, M, N> intern;

		LSIMD_ENSURE_INLINE
		sse_mat(_sse_mat<T, M, N> a) : intern(a) { }

	public:
		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, aligned_t)
		{
			intern.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, unaligned_t)
		{
			intern.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, int ldim, aligned_t)
		{
			intern.load(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, int ldim, unaligned_t)
		{
			intern.load(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, trans_t, aligned_t)
		{
			intern.load(x, trans_t(), aligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, trans_t, unaligned_t)
		{
			intern.load(x, trans_t(), unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, int ldim, trans_t, aligned_t)
		{
			intern.load(x, ldim, trans_t(), aligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const f32 *x, int ldim, trans_t, unaligned_t)
		{
			intern.load(x, ldim, trans_t(), unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, aligned_t)
		{
			intern.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, unaligned_t)
		{
			intern.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, int ldim, aligned_t)
		{
			intern.load(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, int ldim, unaligned_t)
		{
			intern.load(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, trans_t, aligned_t)
		{
			intern.load(x, trans_t(), aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, trans_t, unaligned_t)
		{
			intern.load(x, trans_t(), unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, int ldim, trans_t, aligned_t)
		{
			intern.load(x, ldim, trans_t(), aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const f32 *x, int ldim, trans_t, unaligned_t)
		{
			intern.load(x, ldim, trans_t(), unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, aligned_t) const
		{
			intern.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, unaligned_t) const
		{
			intern.store(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, int ldim, aligned_t) const
		{
			intern.store(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(f32 *x, int ldim, unaligned_t) const
		{
			intern.store(x, ldim, unaligned_t());
		}

	public:
		LSIMD_ENSURE_INLINE
		sse_mat operator + (sse_mat r) const
		{
			return intern.madd(r.intern);
		}

		LSIMD_ENSURE_INLINE
		sse_mat operator - (sse_mat r) const
		{
			return intern.msub(r.intern);
		}

		LSIMD_ENSURE_INLINE
		sse_vec<f32, 2> operator * (sse_vec<f32, 2> v) const
		{
			return intern.transform(v);
		}

	};

	/********************************************
	 *
	 *  _sse_mat class for f32
	 *
	 ********************************************/

	template<> class _sse_mat<f32, 2, 2>
	{
	private:
		sse_f32pk m_pk;

		LSIMD_ENSURE_INLINE
		_sse_mat(sse_f32pk pk) : m_pk(pk) { }

	public:
		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, AlignT)
		{
			m_pk.load(x, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, AlignT)
		{
			__m128 p0 = sse::partial_load<2>(x);
			__m128 p1 = sse::partial_load<2>(x + ldim);

			m_pk = _mm_movelh_ps(p0, p1);
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, trans_t, AlignT)
		{
			load(x, AlignT());
			_transpose();
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void load(const f32 *x, int ldim, trans_t, AlignT)
		{
			load(x, ldim, AlignT());
			_transpose();
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, AlignT) const
		{
			m_pk.store(x, AlignT());
		}

		template<typename AlignT>
		LSIMD_ENSURE_INLINE void store(f32 *x, int ldim, AlignT) const
		{
			sse::partial_store<2>(x, m_pk.v);
			sse::partial_store<2>(x + ldim, m_pk.dup_high().v);
		}

	public:
		LSIMD_ENSURE_INLINE _sse_mat madd(_sse_mat r) const
		{
			return add(m_pk, m_pk);
		}

		LSIMD_ENSURE_INLINE _sse_mat msub(_sse_mat r) const
		{
			return sub(m_pk, m_pk);
		}

		LSIMD_ENSURE_INLINE sse_vec<f32, 2> transform(sse_vec<f32, 2> v) const
		{
			sse_f32pk p = v.m_pk.swizzle<0, 0, 1, 1>();
			p = mul(p, m_pk);
			p = add(p, p.dup_low());
			return p.shift_front<2>();
		}

	private:
		LSIMD_ENSURE_INLINE void _transpose()
		{
			m_pk = m_pk.swizzle<0,2,1,3>();
		}
	};





}

#endif 
