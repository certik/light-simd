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

#include "details/sse_mat_bits_f32.h"

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
		sse::smat<T, M, N> intern;

		LSIMD_ENSURE_INLINE
		sse_mat(sse::smat<T, M, N> a) : intern(a) { }

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
		void load_trans(const f32 *x, aligned_t)
		{
			intern.load_trans(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, unaligned_t)
		{
			intern.load_trans(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, int ldim, aligned_t)
		{
			intern.load_trans(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const f32 *x, int ldim, unaligned_t)
		{
			intern.load_trans(x, ldim, unaligned_t());
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
		sse_mat operator % (sse_mat r) const
		{
			return intern.mmul(r.intern);
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator += (sse_mat r)
		{
			intern.inplace_madd(r.intern);
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator -= (sse_mat r)
		{
			intern.inplace_msub(r.intern);
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator %= (sse_mat r)
		{
			intern.inplace_mmul(r.intern);
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_vec<f32, 2> operator * (sse_vec<f32, 2> v) const
		{
			return intern.transform(v);
		}

	};

}

#endif 
