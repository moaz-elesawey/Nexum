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
 * @file Nexum_Dense.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_DENSE_H_
#define _Nexum_DENSE_H_

#include "Nexum_Core.h"
#include "Nexum_Tensor.h"
#include "Nexum_Activation.h"

/** 
 * @brief Represent more abstraction over `Nexum_Tensor`.
 *
 * This Structure hold two main components the `weights`, and `bias` term
 * which are two tensors that do all the work from forward to backward pathes.
 * Also it holds the type of Activation to apply to the output of the forward path
 */
typedef struct Nexum_Dense {
	u64 in_features; ///< Number of input features.
	u64 out_features; ///< Number of output features.
	Nexum_Activation act; ///< What activation to apply to the output.
	bool initialized; ///< whether the layer is initilized or not.
	Nexum_Tensor weights; ///< Tensor to hold the weights of the layer.
	Nexum_Tensor bias; ///< Tensor to hold the bias term of the layer.
}Nexum_Dense;


void Nexum_Dense_init                      (Nexum_Dense*, u64, u64, Nexum_Activation);
void Nexum_Dense_forward                   (Nexum_Dense*, Nexum_Dense*, Nexum_Dense*);
void Nexum_Dense_print                     (Nexum_Dense*);
void Nexum_Dense_read                      (Nexum_Dense*, str);
void Nexum_Dense_read_binary               (Nexum_Dense*, str);
void Nexum_Dense_write                     (Nexum_Dense*, str);
void Nexum_Dense_write_binary              (Nexum_Dense*, str);
void Nexum_Dense_free                      (Nexum_Dense*);

#endif /* _Nexum_DENSE_H_ */
