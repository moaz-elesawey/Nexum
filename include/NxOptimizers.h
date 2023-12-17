#ifndef _NxOPTIMIZER_H_
#define _NxOPTIMIZER_H_

#include "NxCore.h"

/// Model Parameters to Optimize.
typedef struct NxOptimizerParameters{

}NxOptimizerParameters;


/// Stachistic Gradient Descent
typedef struct NxOptimizerSGD{
	f64 lr; ///< Learning rate or step size.
}NxOptimizerSGD;

void NxOptimizerSGD_zero_gradients         (void);
void NxOptimizerSGD_update_parameters      (void);

/// Applying Simple Momentum to SGD
typedef struct NxOptimizerMomentum {
	f64 lr; ///< Learning rate or step size.
}NxOptimizerMomentum;

void NxOptimizerMomentum_zero_gradients    (void);
void NxOptimizerMomentum_update_parameters (void);

/// Applying More Complex Momentum to SGD.
typedef struct NxOptimizerAdam {
	f64 lr; ///< Learning rate or step size.
}NxOptimizerAdam;

void NxOptimizerAdam_zero_gradients        (void);
void NxOptimizerAdam_update_parameters     (void);

#endif /* _NxOPTIMIZER_H_ */

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
 * @file NxOptimizer.h
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */
