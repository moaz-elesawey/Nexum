#include <stdio.h>
#include "Nexum.h"

#undef Nexum_DEBUG

int main(void) {
	const u64 N = 2, M = 4, P = 1;

	Nexum_Tensor X, Y;
	Nexum_Tensor W, b;

	Nexum_Tensor T1, T2, T3, T4;

	Nexum_Tensor_alloc_zeros(&X, N, M);
	Nexum_Tensor_alloc_zeros(&Y, P, M);
	Nexum_Tensor_alloc_randn(&W, P, N);
	Nexum_Tensor_alloc_zeros(&b, P, N);

	X.data[0] = 0.0f; X.data[1] = 1.0f;
	X.data[2] = 0.0f; X.data[3] = 1.0f;
	X.data[4] = 0.0f; X.data[5] = 0.0f;
	X.data[6] = 1.0f; X.data[7] = 1.0f;

	Y.data[0] = 0.0f; Y.data[1] = 1.0f;
	Y.data[2] = 1.0f; Y.data[3] = 0.0f;

	Nexum_Tensor_to_string(&X);
	Nexum_Tensor_to_string(&Y);

	Nexum_Tensor_matmul_tensor(&T1, &W, &X);
	// Nexum_Tensor_add_tensor(&T2, &T1, &b);
	
	// Nexum_Tensor_to_string(&T2);

	printf("Press enter to continue...");getchar();
	Nexum_Tensor_free(&X); Nexum_Tensor_free(&Y);
	Nexum_Tensor_free(&W); Nexum_Tensor_free(&b);
	printf("Press enter to continue...");getchar();
}
