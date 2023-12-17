#ifndef _NxLAYERS_H_
#define _NxLAYERS_H_

#include "NxCore.h"
#include "NxTensor.h"
#include "NxActivations.h"

/** 
 * @brief Represent more abstraction over `NxTensor`.
 *
 * This Structure hold two main components the `weights`, and `bias` term
 * which are two tensors that do all the work from forward to backward pathes.
 * Also it holds the type of Activation to apply to the output of the forward path
 */
typedef struct NxDense {
	u64 in_features; ///< Number of input features.
	u64 out_features; ///< Number of output features.
	NxActivation act; ///< What activation to apply to the output.
	bool initialized; ///< whether the layer is initilized or not.
	NxTensor weights; ///< Tensor to hold the weights of the layer.
	NxTensor bias; ///< Tensor to hold the bias term of the layer.
}NxDense;


void NxDense_alloc                     (NxDense*, u64, u64, NxActivation);
void NxDense_forward                   (NxDense*, NxDense*, NxDense*);
void NxDense_to_string                 (NxDense*);
void NxDense_read                      (NxDense*, str);
void NxDense_read_binary               (NxDense*, str);
void NxDense_write                     (NxDense*, str);
void NxDense_write_binary              (NxDense*, str);
void NxDense_free                      (NxDense*);

/** 
 * @brief Represent more abstraction over `NxTensor`.
 *
 * This Structure hold two main components the `weights`, and `bias` term
 * which are two tensors that do all the work from forward to backward pathes.
 * Also it holds the type of Activation to apply to the output of the forward path
 */
typedef struct NxConv1D {
	u64 n_filters; ///< Number of input features.
	u64 kernel_size; ///< Number of output features.
	u64 padding; ///<
	u64 stride; ///< 
	NxActivation act; ///< What activation to apply to the output.
	bool initialized; ///< whether the layer is initilized or not.
	NxTensor weights; ///< Tensor to hold the weights of the layer.
	NxTensor bias; ///< Tensor to hold the bias term of the layer.
}NxConv1D;


void NxConv1D_alloc                     (NxConv1D*, u64, u64, NxActivation);
void NxConv1D_forward                   (NxConv1D*, NxConv1D*, NxConv1D*);
void NxConv1D_to_string                 (NxConv1D*);
void NxConv1D_read                      (NxConv1D*, str);
void NxConv1D_read_binary               (NxConv1D*, str);
void NxConv1D_write                     (NxConv1D*, str);
void NxConv1D_write_binary              (NxConv1D*, str);
void NxConv1D_free                      (NxConv1D*);

/** 
 * @brief Represent more abstraction over `NxTensor`.
 *
 * This Structure hold two main components the `weights`, and `bias` term
 * which are two tensors that do all the work from forward to backward pathes.
 * Also it holds the type of Activation to apply to the output of the forward path
 */
typedef struct NxConv2D {
	u64 n_filters; ///< Number of input features.
	u64 kernel_size; ///< Number of output features.
	u64 padding; ///<
	u64 stride; ///< 
	NxActivation act; ///< What activation to apply to the output.
	bool initialized; ///< whether the layer is initilized or not.
	NxTensor weights; ///< Tensor to hold the weights of the layer.
	NxTensor bias; ///< Tensor to hold the bias term of the layer.
}NxConv2D;

void NxConv2D_alloc                     (NxConv2D*, u64, u64, NxActivation);
void NxConv2D_forward                   (NxConv2D*, NxConv2D*, NxConv2D*);
void NxConv2D_to_string                 (NxConv2D*);
void NxConv2D_read                      (NxConv2D*, str);
void NxConv2D_read_binary               (NxConv2D*, str);
void NxConv2D_write                     (NxConv2D*, str);
void NxConv2D_write_binary              (NxConv2D*, str);
void NxConv2D_free                      (NxConv2D*);


#endif /* _NxDENSE_H_ */

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
 * @file NxLayers.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

