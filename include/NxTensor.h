#ifndef _NxTENSOR_H_
#define _NxTENSOR_H_

#include <string.h>
#include "NxCore.h"

/// Return the element at position i, j
#define NxTensor_AT(M, i, j) (M)->data[(i)*(M)->n + (j)]
/// Return the position of an element mapped from 2D to 1D
/// Where N is the number of columns of the tensor.
#define NxIDX(N, i, j) ((i) * (N) + (j))
/// Broadcast the Tensor over the Rows (Peform the Operation over the Rows).
#define NxAXIS_ROW 0
/// Broadcast the Tensor over the Columns (Peform the Operation over the Columns).
#define NxAXIS_COL 1

/** 
 * @brief Represents our main data type in our Neural Networks.
 *
 * This struct abstracts all the needed actions and behaviors of the tensor
 * data type in Mathematics. it stores the data and also can be maniuplutated
 * using the below functions from calculating any math function and abstracted
 * using these functions e.g (pow, sqrt, square...).
 * 
 * Consider it as more like NumPy ndarrays where it also abstracted alot of the
 * math functions to be used directly into the data of the tensor which helps us
 * not bather with for loops.
 */
typedef struct NxTensor {
	u64 id; ///< id of the tensor helpful in AutoDiff later. 
	u64 m; ///< number of rows. 
	u64 n; ///< number of columns. 
	NxDTYPE* data; ///< the actual data of the tensor stored as 1D array in the Heap. with size (m*n) 
	bool allocated; ///< whether this tensor is allocated (initialized) or not. 
    u32 refcount; ///< how many times the object used in (useful in memory deallocation later).
}NxTensor;


NxCDEF void NxTensor_alloc            (NxTensor* A, u64 m, u64 n);
NxCDEF void NxTensor_alloc_zeros      (NxTensor* A, u64 m, u64 n);
NxCDEF void NxTensor_alloc_ones       (NxTensor* A, u64 m, u64 n); 
NxCDEF void NxTensor_alloc_rand       (NxTensor* A, u64 m, u64 n); 
NxCDEF void NxTensor_alloc_randn      (NxTensor* A, u64 m, u64 n); 
NxCDEF void NxTensor_alloc_eye        (NxTensor* A, u64 m); 
NxCDEF void NxTensor_alloc_arange     (NxTensor* A, NxDTYPE start, NxDTYPE end, NxDTYPE step); 
NxCDEF void NxTensor_alloc_linspace   (NxTensor* A, NxDTYPE start, NxDTYPE end, u64 size); 
NxCDEF void NxTensor_alloc_ones_like  (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_alloc_full       (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_alloc_zeros_like (NxTensor* C, NxTensor* A);

NxCDEF void NxTensor_set_data         (NxTensor* A, NxDTYPE* data); 
NxCDEF void NxTensor_copy_data        (NxTensor* C, NxTensor* A);

NxCDEF void NxTensor_read             (NxTensor* A, str fname); 
NxCDEF void NxTensor_read_binary      (NxTensor* A, str fname); 
NxCDEF void NxTensor_write            (NxTensor* A, str fname); 
NxCDEF void NxTensor_write_binary     (NxTensor* A, str fname); 

NxCDEF void NxTensor_add_tensor       (NxTensor* C, NxTensor* A, NxTensor* B); 
NxCDEF void NxTensor_sub_tensor       (NxTensor* C, NxTensor* A, NxTensor* B); 
NxCDEF void NxTensor_mul_tensor       (NxTensor* C, NxTensor* A, NxTensor* B); 
NxCDEF void NxTensor_div_tensor       (NxTensor* C, NxTensor* A, NxTensor* B); 

NxCDEF void NxTensor_add_tensor_boradcast  (NxTensor* C, NxTensor* A, NxTensor* B, u8); 
NxCDEF void NxTensor_sub_tensor_boradcast  (NxTensor* C, NxTensor* A, NxTensor* B, u8); 
NxCDEF void NxTensor_mul_tensor_broadcast  (NxTensor* C, NxTensor* A, NxTensor* B, u8); 
NxCDEF void NxTensor_div_tensor_broadcast  (NxTensor* C, NxTensor* A, NxTensor* B, u8); 

NxCDEF void NxTensor_matmul_tensor    (NxTensor* C, NxTensor* A, NxTensor* B); 

NxCDEF void NxTensor_add_scalar       (NxTensor* C, NxTensor* A, NxDTYPE B); 
NxCDEF void NxTensor_sub_scalar       (NxTensor* C, NxTensor* A, NxDTYPE B); 
NxCDEF void NxTensor_mul_scalar       (NxTensor* C, NxTensor* A, NxDTYPE B); 
NxCDEF void NxTensor_div_scalar       (NxTensor* C, NxTensor* A, NxDTYPE B); 
NxCDEF void NxTensor_add_scalar_      (NxTensor* A, NxDTYPE B);
NxCDEF void NxTensor_sub_scalar_      (NxTensor* A, NxDTYPE B);
NxCDEF void NxTensor_mul_scalar_      (NxTensor* A, NxDTYPE B);
NxCDEF void NxTensor_div_scalar_      (NxTensor* A, NxDTYPE B);

NxCDEF void NxTensor_transpose        (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_transpose_       (NxTensor* A);
NxCDEF void NxTensor_reshape          (NxTensor* C, NxTensor* A, u64 m, u64 n);
NxCDEF void NxTensor_reshape_         (NxTensor* A, u64 m, u64 n);

NxCDEF void NxTensor_sum_tensor       (NxTensor* C, NxTensor* A, u32 axis);
NxCDEF void NxTensor_expand           (NxTensor* C, NxTensor* A, u8 axis, u64 n_copies);

NxCDEF void NxTensor_to_string        (NxTensor* A);
NxCDEF void NxTensor_to_string_raw    (NxTensor* A);

NxCDEF void NxTensor_free             (NxTensor* A);

NxCDEF void NxTensor_apply            (NxTensor* C, NxTensor* A, NxDTYPE(*pfunc)(NxDTYPE));
NxCDEF void NxTensor_neg              (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_abs              (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_pow              (NxTensor* C, NxTensor* A, i32 p);
NxCDEF void NxTensor_sign             (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_square           (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_exp              (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_log              (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_log10            (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_cos              (NxTensor* C, NxTensor* A);
NxCDEF void NxTensor_sin              (NxTensor* C, NxTensor* A);

NxCDEF void NxTensor_apply_           (NxTensor* A, NxDTYPE(*pfunc)(NxDTYPE));
NxCDEF void NxTensor_neg_             (NxTensor* A);
NxCDEF void NxTensor_abs_             (NxTensor* A);
NxCDEF void NxTensor_pow_             (NxTensor* A, i32 p);
NxCDEF void NxTensor_sign_            (NxTensor* A);
NxCDEF void NxTensor_square_          (NxTensor* A);
NxCDEF void NxTensor_exp_             (NxTensor* A);
NxCDEF void NxTensor_log_             (NxTensor* A);
NxCDEF void NxTensor_log10_           (NxTensor* A);
NxCDEF void NxTensor_cos_             (NxTensor* A);
NxCDEF void NxTensor_sin_             (NxTensor* A);

NxCDEF u64  NxTensor_size             (NxTensor* A);
NxDTYPE  NxTensor_sum                 (NxTensor* A);

#endif /* _NxTENSOR_H_ */

/****************************************************************************
 * Copyright (C) 2023 by Moaz Mohammed El-Essawey                           *
 *                                                                          *
 * This file is part of Nexum Library.                                      *
 *                                                                          *
 *   Nexum is free software: you can redistribute it and/or modify it       *
 *   under the terms of the GNU Lesser General Public License as published  *
 *   by the Free Software Foundation, either version 3 of the License, or   *
 *   (at your option) any later version.                                    *
 *                                                                          *
 *   Box is distributed in the hope that it will be useful,                 *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 *   GNU Lesser General Public License for more details.                    *
 *                                                                          *
 *   You should have received a copy of the GNU Lesser General Public       *
 *   License along with Box.  If not, see <http://www.gnu.org/licenses/>.   *
 ****************************************************************************/

/**
 * @file NxTensor.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */
