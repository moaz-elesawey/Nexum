#ifndef _NxLOSS_H_
#define _NxLOSS_H_

#include "NxCore.h"
#include "NxTensor.h"

/* NxLoss.c */
f64  NxLoss_mean_squared_error         (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_mse                        (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_mean_absolute_error        (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_mae                        (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_root_mean_squared_error    (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_rmse                       (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_categorical_crossentropy   (NxTensor* y_true, NxTensor* y_pred);
f64  NxLoss_binary_crossentropy        (NxTensor* y_true, NxTensor* y_pred);

#endif /* _NxLOSS_H_ */

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
 * @file NxLoss.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

