#include "NxLosses.h"


#include <stdio.h>
#include <math.h>


f64 NxLoss_mean_squared_error(NxTensor* y_true, NxTensor* y_pred) {
    f64 loss = 0.0;
    NxTensor D;
    NxTensor_sub_tensor(&D, y_true, y_pred);
    NxTensor_pow_(&D, 2);
    NxTensor_mul_scalar_(&D, 0.5f/NxTensor_size(&D));
    loss = NxTensor_sum(&D);
    NxTensor_free(&D);
    return loss;
}

f64 NxLoss_mse(NxTensor* y_true, NxTensor* y_pred) {
    return NxLoss_mean_squared_error(y_true, y_pred);
}

f64 NxLoss_mean_absolute_error(NxTensor* y_true, NxTensor* y_pred) {
    f64 loss = 0.0;
    NxTensor D;
    NxTensor_sub_tensor(&D, y_true, y_pred);
    NxTensor_abs_(&D);
    NxTensor_mul_scalar_(&D, 0.5f/NxTensor_size(&D));
    loss = NxTensor_sum(&D);
    NxTensor_free(&D);
    return loss;
}

f64 NxLoss_mae(NxTensor* y_true, NxTensor* y_pred) {
    return NxLoss_mean_absolute_error(y_true, y_pred);
}

f64 NxLoss_root_mean_squared_error(NxTensor* y_true, NxTensor* y_pred) {
    return sqrt(NxLoss_mse(y_true, y_pred));
}

f64 NxLoss_rmse(NxTensor* y_true, NxTensor* y_pred) {
    return NxLoss_root_mean_squared_error(y_true, y_pred);
}

f64 NxLoss_categorical_crossentropy(NxTensor* y_true, NxTensor* y_pred) {
    f64 loss = 0;

    (void) y_true;
    (void) y_pred;

    return loss;
}

f64 NxLoss_binary_crossentropy(NxTensor* y_true, NxTensor* y_pred) {
    f64 loss = 0;

    (void) y_true;
    (void) y_pred;
    
    return loss;
}
