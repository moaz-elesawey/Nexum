#ifndef _NxMODEL_H_
#define _NxMODEL_H_

#include "NxCore.h"
#include "NxActivations.h"


/// Define a Sequential Model like the one in Keras Tensorflow models.
typedef struct NxModelSequential {
	str name; ///< The name of the model.
}NxModelSequential;

void NxModelSequential_append_Dense     (u64 in_feature, u64 out_features, NxActivation act);
void NxModelSequential_append_Conv1D    (u64 n_filters, u64 kernel_size, u8 stride, u8 padding);
void NxModelSequential_append_Conv2D    (u64 n_filters, u64 kernel_size, u8 stride, u8 padding);
void NxModelSequential_append_MaxPool1D (u64 pool_size);
void NxModelSequential_append_MaxPool2D (u64 pool_size);

void NxModelSequential_train            (NxModelSequential* model, NxTensor* x_train, NxTensor* y_train, u64 batch_size);
void NxModelSequential_evaluate         (NxModelSequential* model);
void NxModelSequential_predict          (NxModelSequential* model);

#endif /* _NxMODEL_H_ */

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
 * @file NxModel.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

