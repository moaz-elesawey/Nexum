#include <stdio.h>
#include "Nexum.h"


int main(int argc, char** argv) {
	Nexum_Tensor X, Y;
	Nexum_Tensor W1, W2, b1, b2;

	printf("here 1\n");
	f64* x_data = (f64*) malloc(4*2* sizeof (f64));
	f64* y_data = (f64*) malloc(4*1* sizeof (f64));
	x_data[0] = 0.0f; x_data[1] = 0.0f;
	x_data[2] = 1.0f; x_data[3] = 0.0f;
	x_data[4] = 0.0f; x_data[5] = 1.0f;
	x_data[6] = 1.0f; x_data[7] = 1.0f;

	y_data[0] = 0.0f; y_data[1] = 1.0f;
	y_data[2] = 1.0f; y_data[3] = 0.0f;
	printf("here 2\n");

	Nexum_Tensor_init(&X, 4, 2);
	printf("here 3\n");
	/* Nexum_Tensor_init(&Y, 4, 1); */
	printf("here 4\n");

	/* Nexum_Tensor_set_data(&X, x_data); */
	/* Nexum_Tensor_set_data(&Y, y_data); */

	/* Nexum_Tensor_print(&X); */
	/* Nexum_Tensor_print(&Y); */

	Nexum_Tensor_init_randn(&W1, 2, 16);
	Nexum_Tensor_init_zeros(&b1, 16, 1);
	Nexum_Tensor_init_randn(&W2, 16, 1);
	Nexum_Tensor_init_zeros(&b1, 1 , 1);

	printf("Press enter to continue...");getchar();
	Nexum_Tensor_free(&X); Nexum_Tensor_free(&Y);
	Nexum_Tensor_free(&W1); Nexum_Tensor_free(&W2);
	Nexum_Tensor_free(&b1); Nexum_Tensor_free(&b2);
	printf("Press enter to continue...");getchar();
}

