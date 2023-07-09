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
 * @file Nexum_Tensor.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_TENSOR_H_
#define _Nexum_TENSOR_H_

#include "Nexum_Core.h"

/// Broadcast the Tensor over the Rows (Peform the Operation over the Rows).
#define BROADCAST_AXIS_ROW 0
/// Broadcast the Tensor over the Columns (Peform the Operation over the Columns).
#define BROADCAST_AXIS_COL 1

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
typedef struct Nexum_Tesnor {
	u64 id; ///< id of the tensor helpful in AutoDiff later. 
	u64 m; ///< number of rows. 
	u64 n; ///< number of columns. 
	f64* data; ///< the actual data of the tensor stored as 1D array in the Heap. with size (m*n) 
	bool allocated; ///< whether this tensor is allocated (initialized) or not. 
}Nexum_Tensor;


void Nexum_Tensor_init                  (Nexum_Tensor* A, u64 m, u64 n);
void Nexum_Tensor_init_zeros            (Nexum_Tensor* A, u64 m, u64 n);
void Nexum_Tensor_init_ones             (Nexum_Tensor* A, u64 m, u64 n); 
void Nexum_Tensor_init_rand             (Nexum_Tensor* A, u64 m, u64 n); 
void Nexum_Tensor_init_randn            (Nexum_Tensor* A, u64 m, u64 n); 
void Nexum_Tensor_init_eye              (Nexum_Tensor* A, u64 m); 
void Nexum_Tensor_init_arange           (Nexum_Tensor* A, f64 start, f64 end, f64 step); 
void Nexum_Tensor_init_linspace         (Nexum_Tensor* A, f64 start, f64 end, u64 size); 
void Nexum_Tensor_init_ones_like        (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_init_fill             (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_init_zeros_like       (Nexum_Tensor* C, Nexum_Tensor* A);

void Nexum_Tensor_set_data              (Nexum_Tensor* A, f64* data); 

void Nexum_Tensor_read                  (Nexum_Tensor* A, str fname); 
void Nexum_Tensor_read_binary           (Nexum_Tensor* A, str fname); 
void Nexum_Tensor_write                 (Nexum_Tensor* A, str fname); 
void Nexum_Tensor_write_binary          (Nexum_Tensor* A, str fname); 

void Nexum_Tensor_add_tensor            (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B); 
void Nexum_Tensor_sub_tensor            (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B); 
void Nexum_Tensor_mul_tensor            (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B); 
void Nexum_Tensor_div_tensor            (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B); 

void Nexum_Tensor_add_tensor_boradcast  (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8); 
void Nexum_Tensor_sub_tensor_boradcast  (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8); 
void Nexum_Tensor_mul_tensor_broadcast  (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8); 
void Nexum_Tensor_div_tensor_broadcast  (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8); 

void Nexum_Tensor_matmul_tensor         (Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B); 

void Nexum_Tensor_add_scalar            (Nexum_Tensor* C, Nexum_Tensor* A, f64 B); 
void Nexum_Tensor_sub_scalar            (Nexum_Tensor* C, Nexum_Tensor* A, f64 B); 
void Nexum_Tensor_mul_scalar            (Nexum_Tensor* C, Nexum_Tensor* A, f64 B); 
void Nexum_Tensor_div_scalar            (Nexum_Tensor* C, Nexum_Tensor* A, f64 B); 
void Nexum_Tensor_add_scalar_           (Nexum_Tensor* A, f64 B);
void Nexum_Tensor_sub_scalar_           (Nexum_Tensor* A, f64 B);
void Nexum_Tensor_mul_scalar_           (Nexum_Tensor* A, f64 B);
void Nexum_Tensor_div_scalar_           (Nexum_Tensor* A, f64 B);

void Nexum_Tensor_transpose             (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_transpose_            (Nexum_Tensor* A);
void Nexum_Tensor_reshape               (Nexum_Tensor* C, Nexum_Tensor* A, u64 m, u64 n);
void Nexum_Tensor_reshape_              (Nexum_Tensor* A, u64 m, u64 n);

void Nexum_Tensor_expand                (Nexum_Tensor* C, Nexum_Tensor* A, u8 axis, u64 n_copies);

void Nexum_Tensor_print                 (Nexum_Tensor* A);
void Nexum_Tensor_print_raw             (Nexum_Tensor* A);

void Nexum_Tensor_free                  (Nexum_Tensor* A);

void Nexum_Tensor_apply                 (Nexum_Tensor* C, Nexum_Tensor* A, f64(*pfunc)(f64));
void Nexum_Tensor_apply_                (Nexum_Tensor* A, f64(*pfunc)(f64));
void Nexum_Tensor_neg                   (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_neg_                  (Nexum_Tensor* A);
void Nexum_Tensor_abs                   (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_abs_                  (Nexum_Tensor* A);
void Nexum_Tensor_pow                   (Nexum_Tensor* C, Nexum_Tensor* A, i32 p);
void Nexum_Tensor_pow_                  (Nexum_Tensor* A, i32 p);
void Nexum_Tensor_sign                  (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_sign_                 (Nexum_Tensor* A);
void Nexum_Tensor_square                (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_square_               (Nexum_Tensor* A);
void Nexum_Tensor_exp                   (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_exp_                  (Nexum_Tensor* A);
void Nexum_Tensor_log                   (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_log_                  (Nexum_Tensor* A);
void Nexum_Tensor_log10                 (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_log10_                (Nexum_Tensor* A);
void Nexum_Tensor_cos                   (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_cos_                  (Nexum_Tensor* A);
void Nexum_Tensor_sin                   (Nexum_Tensor* C, Nexum_Tensor* A);
void Nexum_Tensor_sin_                  (Nexum_Tensor* A);

u64  Nexum_Tensor_size                  (Nexum_Tensor* A);
f64  Nexum_Tensor_sum                   (Nexum_Tensor* A);

#endif /* Nexum_TENSOR_H_ */
