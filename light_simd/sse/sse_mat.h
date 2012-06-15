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

#include "details/sse_mat_comp_bits.h"
#include "details/sse_mat_matmul_bits.h"
#include "details/sse_mat_sol_bits.h"

namespace lsimd
{


	template<typename T, int M, int N> class sse_mat;


	/********************************************
	 *
	 *  sse_mat class (generic)
	 *
	 ********************************************/


	template<typename T, int M, int N>
	class sse_mat
	{
	public:
		sse::smat_core<T, M, N> core;

		LSIMD_ENSURE_INLINE
		sse_mat(const sse::smat_core<T, M, N>& a) : core(a) { }

	public:
		LSIMD_ENSURE_INLINE
		sse_mat() { }

		LSIMD_ENSURE_INLINE
		sse_mat( zero_t ) : core( zero_t() ) { }

		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, aligned_t)
		{
			core.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, unaligned_t)
		{
			core.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, int ldim, aligned_t)
		{
			core.load(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		sse_mat(const T *x, int ldim, unaligned_t)
		{
			core.load(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, aligned_t)
		{
			core.load(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, unaligned_t)
		{
			core.load(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, int ldim, aligned_t)
		{
			core.load(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load(const T *x, int ldim, unaligned_t)
		{
			core.load(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, aligned_t)
		{
			core.load_trans(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, unaligned_t)
		{
			core.load_trans(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, int ldim, aligned_t)
		{
			core.load_trans(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void load_trans(const T *x, int ldim, unaligned_t)
		{
			core.load_trans(x, ldim, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, aligned_t) const
		{
			core.store(x, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, unaligned_t) const
		{
			core.store(x, unaligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, int ldim, aligned_t) const
		{
			core.store(x, ldim, aligned_t());
		}

		LSIMD_ENSURE_INLINE
		void store(T *x, int ldim, unaligned_t) const
		{
			core.store(x, ldim, unaligned_t());
		}

	public:
		LSIMD_ENSURE_INLINE
		sse_mat operator + (const sse_mat& r) const
		{
			return core + r.core;
		}

		LSIMD_ENSURE_INLINE
		sse_mat operator - (const sse_mat& r) const
		{
			return core - r.core;
		}

		LSIMD_ENSURE_INLINE
		sse_mat operator % (const sse_mat& r) const
		{
			return core % r.core;
		}

		LSIMD_ENSURE_INLINE
		sse_mat operator * (LSIMD_VT(sse_pack<T>) s) const
		{
			return core * s;
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator += (const sse_mat& r)
		{
			core += r.core;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator -= (const sse_mat& r)
		{
			core -= r.core;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator %= (const sse_mat& r)
		{
			core %= r.core;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_mat& operator *= (LSIMD_VT(sse_pack<T>) s)
		{
			core *= s;
			return *this;
		}

		LSIMD_ENSURE_INLINE
		sse_vec<T, M> operator * (const sse_vec<T, N>& v) const
		{
			return transform(core, v);
		}

		LSIMD_ENSURE_INLINE
		T trace() const
		{
			return core.trace();
		}

	public:
		LSIMD_ENSURE_INLINE
		bool test_equal(const T *r) const
		{
			return core.test_equal(r);
		}

		LSIMD_ENSURE_INLINE
		void dump(const char *fmt) const
		{
			core.dump(fmt);
		}

	};

	template<typename T, int M, int K, int N>
	inline sse_mat<T, M, N> operator * (const sse_mat<T, M, K>& a, const sse_mat<T, K, N>& b)
	{
		sse_mat<T, M, N> c;
		sse::mtimes_op<T, M, K, N>::run(a.core, b.core, c.core);
		return c;
	}


	template<typename T, int N>
	LSIMD_ENSURE_INLINE
	inline T det(const sse_mat<T, N, N>& a)
	{
		return sse::det(a.core);
	}

	template<typename T, int N>
	inline sse_mat<T, N, N> inv(const sse_mat<T, N, N>& a)
	{
		sse_mat<T, N, N> r;
		sse::inv(a.core, r.core);
		return r;
	}

	template<typename T, int N>
	inline T inv_and_det(const sse_mat<T, N, N>& a, sse_mat<T, N, N>& r)
	{
		return sse::inv(a.core, r.core);
	}

	template<typename T, int N>
	inline sse_vec<T, N> solve(const sse_mat<T, N, N>& A, const sse_vec<T, N>& b)
	{
		sse_vec<T, N> x;
		sse::solve(A.core, b, x);
		return x;
	}

	template<typename T, int N, int N2>
	inline sse_mat<T, N, N2> solve(const sse_mat<T, N, N>& A, const sse_mat<T, N, N2>& B)
	{
		sse_mat<T, N, N2> X;
		sse::solve(A.core, B.core, X.core);
		return X;
	}

}

#endif 
