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
 * @file Nexum_Model.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_MODEL_H_
#define _Nexum_MODEL_H_

#include "Nexum_Core.h"
#include "Nexum_Activations.h"


/// Define a Sequential Model like the one in Keras Tensorflow models.
typedef struct Nexum_ModelSequential {
	str name; ///< The name of the model.
}Nexum_ModelSequential;

void Nexum_ModelSequential_append_Dense     (u64 in_feature, u64 out_features, Nexum_Activation act);
void Nexum_ModelSequential_append_Conv1D    (u64 n_filters, u64 kernel_size, u8 stride, u8 padding);
void Nexum_ModelSequential_append_Conv2D    (u64 n_filters, u64 kernel_size, u8 stride, u8 padding);
void Nexum_ModelSequential_append_MaxPool1D (u64 pool_size);
void Nexum_ModelSequential_append_MaxPool2D (u64 pool_size);

void Nexum_ModelSequential_train            (Nexum_ModelSequential* model, Nexum_Tensor* x_train, Nexum_Tensor* y_train, u64 batch_size);
void Nexum_ModelSequential_evaluate         (Nexum_ModelSequential* model);
void Nexum_ModelSequential_predict          (Nexum_ModelSequential* model);

#endif /* _Nexum_MODEL_H_ */
