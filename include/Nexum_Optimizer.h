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
 * @file Nexum_Optimizer.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#ifndef _Nexum_OPTIMIZER_H_
#define _Nexum_OPTIMIZER_H_

#include "Nexum_Core.h"

/// Model Parameters to Optimize.
typedef struct Nexum_OptimizerParameters{

}Nexum_OptimizerParameters;


/// Stachistic Gradient Descent
typedef struct Nexum_OptimizerSGD{
	f64 lr; ///< Learning rate or step size.
}Nexum_OptimizerSGD;

void Nexum_OptimizerSGD_zero_gradients         (void);
void Nexum_OptimizerSGD_update_parameters      (void);

/// Applying Simple Momentum to SGD
typedef struct Nexum_OptimizerMomentum {
	f64 lr; ///< Learning rate or step size.
}Nexum_OptimizerMomentum;

void Nexum_OptimizerMomentum_zero_gradients    (void);
void Nexum_OptimizerMomentum_update_parameters (void);

/// Applying More Complex Momentum to SGD.
typedef struct Nexum_OptimizerAdam {
	f64 lr; ///< Learning rate or step size.
}Nexum_OptimizerAdam;

void Nexum_OptimizerAdam_zero_gradients        (void);
void Nexum_OptimizerAdam_update_parameters     (void);

#endif /* _Nexum_OPTIMIZER_H_ */

