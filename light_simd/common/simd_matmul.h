/**
 * @file simd_matmul.h
 *
 * Small matrix multiplication using SIMD
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SIMD_MATMUL_H_
#define SIMD_MATMUL_H_

#include <light_simd/common/common_base.h>

namespace lsimd
{

	template<typename VT>
	LSIMD_ENSURE_INLINE
	VT simd_mv_cm_n44_p4(VT col0, VT col1, VT col2, VT col3, VT x)  // column-major (4 x 4) * (4 x 1)
	{
		VT y0 = mul(col0, VT(x.e[0]));
		VT y1 = mul(col1, VT(x.e[1]));
		VT y2 = mul(col2, VT(x.e[2]));
		VT y3 = mul(col3, VT(x.e[3]));

		y0 = add(y0, y1);
		y2 = add(y2, y3);

		return add(y0, y2);
	}

	template<typename VT>
	LSIMD_ENSURE_INLINE
	VT simd_mv_cm_t44_p4(VT col0, VT col1, VT col2, VT col3, VT x)  // column-major T(4 x 4) * (4 x 1)
	{
		VT q0 = mul(col0, x);
		VT q1 = mul(col1, x);
		VT q2 = mul(col2, x);
		VT q3 = mul(col3, x);

		q0 = hadd(q0, q1);
		q1 = hadd(q2, q3);

		return hadd(q0, q1);
	}

}

#endif /* SIMD_MATMUL_H_ */
