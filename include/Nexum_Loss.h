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
 * @file Nexum_Loss.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_LOSS_H_
#define _Nexum_LOSS_H_

#include "Nexum_Core.h"
#include "Nexum_Tensor.h"

/* Nexum_Loss.c */
f64  Nexum_Loss_mean_squared_error         (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_mse                        (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_mean_absolute_error        (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_mae                        (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_root_mean_squared_error    (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_rmse                       (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_categorical_crossentropy   (Nexum_Tensor*, Nexum_Tensor*);
f64  Nexum_Loss_binary_crossentropy        (Nexum_Tensor*, Nexum_Tensor*);


#endif /* _Nexum_LOSS_H_ */
