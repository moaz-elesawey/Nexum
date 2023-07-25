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
 * @file Nexum_Activation.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_ACTIVATION_H_
#define _Nexum_ACTIVATION_H_

#include "Nexum_Core.h"

/// Simple Enum to store the type of Activations can be used to the Neural Network Layers.
typedef enum Nexum_Activation {
	Nexum_Activation_None, ///< Applies Not Activation
	Nexum_Activation_ReLU, ///< Applies Rectified Linear Unit Activation.
	Nexum_Activation_Sigmoid, ///< Applies Sigmoid Activation.
	Nexum_Activation_Tanh, ///< Applies Tanh Activation.
	Nexum_Activation_ELU, ///< Applies Exponential Linear Unit Activation.
	Nexum_Activation_PReLU, ///< Applies Leaky Rectified Linear Unit Activation.
} Nexum_Activation;

#endif /* _Nexum_ACTIVATION_H_ */

