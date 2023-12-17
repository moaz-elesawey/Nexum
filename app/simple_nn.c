#include <stdio.h>
#include "Nexum.h"

#undef NxDEBUG

int main(void) {
	const u64 N = 2, M = 4, P = 1;

	NxTensor X, Y;
	NxTensor W, b;

	NxTensor T1, T2, T3, T4;

	NxTensor_alloc_zeros(&X, N, M);
	NxTensor_alloc_zeros(&Y, P, M);
	NxTensor_alloc_randn(&W, P, N);
	NxTensor_alloc_zeros(&b, P, N);

	X.data[0] = 0.0f; X.data[1] = 1.0f;
	X.data[2] = 0.0f; X.data[3] = 1.0f;
	X.data[4] = 0.0f; X.data[5] = 0.0f;
	X.data[6] = 1.0f; X.data[7] = 1.0f;

	Y.data[0] = 0.0f; Y.data[1] = 1.0f;
	Y.data[2] = 1.0f; Y.data[3] = 0.0f;

	NxTensor_to_string(&X);
	NxTensor_to_string(&Y);

	NxTensor_matmul_tensor(&T1, &W, &X);
	NxTensor_add_tensor(&T2, &T1, &b);
	
	NxTensor_to_string(&T2);

	printf("Press enter to continue...");getchar();
	NxTensor_free(&X); NxTensor_free(&Y);
	NxTensor_free(&W); NxTensor_free(&b);
	printf("Press enter to continue...");getchar();
}
