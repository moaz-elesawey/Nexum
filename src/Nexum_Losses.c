#include "Nexum_Losses.h"


#include <stdio.h>
#include <math.h>


f64 Nexum_Loss_mean_squared_error(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    f64 loss = 0.0;
    Nexum_Tensor D;
    Nexum_Tensor_sub_tensor(&D, y_true, y_pred);
    Nexum_Tensor_pow_(&D, 2);
    Nexum_Tensor_mul_scalar_(&D, 0.5f/Nexum_Tensor_size(&D));
    loss = Nexum_Tensor_sum(&D);
    Nexum_Tensor_free(&D);
    return loss;
}

f64 Nexum_Loss_mse(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    return Nexum_Loss_mean_squared_error(y_true, y_pred);
}

f64 Nexum_Loss_mean_absolute_error(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    f64 loss = 0.0;
    Nexum_Tensor D;
    Nexum_Tensor_sub_tensor(&D, y_true, y_pred);
    Nexum_Tensor_abs_(&D);
    Nexum_Tensor_mul_scalar_(&D, 0.5f/Nexum_Tensor_size(&D));
    loss = Nexum_Tensor_sum(&D);
    Nexum_Tensor_free(&D);
    return loss;
}

f64 Nexum_Loss_mae(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    return Nexum_Loss_mean_absolute_error(y_true, y_pred);
}

f64 Nexum_Loss_root_mean_squared_error(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    return sqrt(Nexum_Loss_mse(y_true, y_pred));
}

f64 Nexum_Loss_rmse(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    return Nexum_Loss_root_mean_squared_error(y_true, y_pred);
}

f64 Nexum_Loss_categorical_crossentropy(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    f64 loss = 0;

    (void) y_true;
    (void) y_pred;

    return loss;
}

f64 Nexum_Loss_binary_crossentropy(Nexum_Tensor* y_true, Nexum_Tensor* y_pred) {
    f64 loss = 0;

    (void) y_true;
    (void) y_pred;
    
    return loss;
}
