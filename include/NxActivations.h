#ifndef _NxACTIVATION_H_
#define _NxACTIVATION_H_

#include "NxCore.h"

/// Simple Enum to store the type of Activations can be used to the Neural Network Layers.
typedef enum NxActivation {
	NxActivation_None, ///< Applies Not Activation
	NxActivation_ReLU, ///< Applies Rectified Linear Unit Activation.
	NxActivation_Sigmoid, ///< Applies Sigmoid Activation.
	NxActivation_Tanh, ///< Applies Tanh Activation.
	NxActivation_ELU, ///< Applies Exponential Linear Unit Activation.
	NxActivation_PReLU, ///< Applies Leaky Rectified Linear Unit Activation.
} NxActivation;

#endif /* _NxACTIVATION_H_ */

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
 * @file NxActivation.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

