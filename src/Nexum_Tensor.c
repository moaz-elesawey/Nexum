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
 * @file Nexum_Tensor.c
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

#include "Nexum.h"

#include <time.h>
#include <math.h>
#include <cblas.h>
#include <lapack.h>

/**
 * @brief Initialize a Tensor in memory filled with Garbage.
 *
 * Used to inialize a Tensor in Memory with (m*n) size filled
 * with Garbage from Memory which is different from random initialization.
 * for random initialization Nexum_Tensor_init_rand(), Nexum_Tensor_init_randn().
 * and it works like that: it checks first that the tensor is not already allocated
 * or had been allocated before using `allocated` param.
 * and if not it allocates new memory to it which is avery hacky way to avoid Memory Leak.
 *
 * @param A: pointer to the Tensor object that will be allocated.
 * @param m: number of rows to be allocated.
 * @param n: number of columnsto be allocated.
 *
 * @return void.
 *
 * @todo Rewrite code to handle Already allocated Tensors.
 */
void Nexum_Tensor_init(Nexum_Tensor* A, u64 m, u64 n){

	if(A->allocated) {
		if(m*n != A->m*A->n) {
			Nexum_Tensor_free(A);
			A->allocated = true;
			A->data = (f64*)malloc(sizeof(f64)*m*n);
		}
	} else {
		A->allocated = true;
		A->data = (f64*)malloc(sizeof(f64)*m*n);
	}
	A->m = m; A->n = n;

	/* if(!A->allocated) { */
	/* 	A->m = m; A->n = n; */
	/* 	A->allocated = true; */
	/* 	A->data = (f64*) malloc(sizeof (f64) * m*n); */
	/* } */
}

/**
 * @brief Allocate and initialize with zeros
 *
 * Initialize a tensor data to zeros.
 *
 * @param A pointer to the tensor object to inialize.
 * @param m number of rows.
 * @param n number of columns
 *
 * @return void
 */
void Nexum_Tensor_init_zeros(Nexum_Tensor* A, u64 m, u64 n){
	Nexum_Tensor_init(A, m, n);
	for(u64 i=0; i<m; i++) {
		for(u64 j=0; j<n; j++) {
			A->data[n*i + j] = 0.0f;
		}
	}
}

/**
 * @brief Allocate and initialize with ones
 *
 * Initialize a tensor data to one.
 *
 * @param A pointer to the tensor object to inialize.
 * @param m number of rows.
 * @param n number of columns
 *
 * @return void
 */
void Nexum_Tensor_init_ones(Nexum_Tensor* A, u64 m, u64 n){
	Nexum_Tensor_init(A, m, n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = 1.0f;
		}
	}
}

/**
 * @brief Allocate and initialize with uniform random numbers.
 *
 * Initialize a tensor data to random numbers drawn from a uniform distribution
 * and it has value between [0, 1].
 *
 * @param A pointer to the tensor object to inialize.
 * @param m number of rows.
 * @param n number of columns
 *
 * @return void
 * 
 * @todo Reimplement it using the BLAS library.
 */
void Nexum_Tensor_init_rand(Nexum_Tensor* A, u64 m, u64 n){
	srand(time(NULL));
	Nexum_Tensor_init(A, m, n);
	for(u64 i=0; i<m; i++) {
		for(u64 j=0; j<n; j++) {
			A->data[n*i + j] = (f64)rand()/(f64)RAND_MAX;
		}
	}
}

/**
 * @brief Allocate and initialize with normal random
 *
 * Initialize a tensor data to random numbers drawn from a normal distibution
 * with 0 as the mean and 1 for the standard deviation.
 *
 * @param A pointer to the tensor object to initialize.
 * @param m number of rows.
 * @param n number of columns.
 *
 * @return void
 *
 * @todo Reimplement it using the BLAS library.
 */
void Nexum_Tensor_init_randn(Nexum_Tensor* A, u64 m, u64 n){
	srand(time(NULL));
	Nexum_Tensor_init(A, m, n);
	for(u64 i=0; i<m; i++) {
		for(u64 j=0; j<n; j++) {
			A->data[n*i + j] = (f64)rand()/(f64)RAND_MAX;
		}
	}
}

/**
 * @brief Allocate and initialize I matrix
 *
 * Initializes a tensor data to the well-know I matrix data with 1 on the diagonal
 * and 0 everywhere else.
 *
 * @param A pointer to the tensor obj to initialize.
 * @param m number of rows.
 *
 * @note we only use the number of rows here as the I matrix must be square matrix.
 *
 * @return void
 */
void Nexum_Tensor_init_eye(Nexum_Tensor* A, u64 m){
	Nexum_Tensor_init(A, m, m);
	for(u64 i=0; i<m; i++) {
		A->data[m*i + i] = 1.0f;
	}
}

/**
 * @brief Allocate and initialize a range of numbers
 *
 * Initializes the tensor data with values between bounds which are (start, end)
 * which is exclusive in this case with a step size (step).
 *
 * The size of the tensor will be calculated using the step size and incrementing 
 * value unitl we reach to the (end).
 *
 * The size (length) of the tensor is calculated using (end - start) / step.
 *
 * @param A pointer to the tensor to initialize.
 * @param start starting value of the tensor data.
 * @param end ending value (exclusive).
 * @param step The step size to increment with.
 *
 * @return void
 */
void Nexum_Tensor_init_arange(Nexum_Tensor* A, f64 start, f64 end, f64 step) {
	u64 n = (u64)ceil((end - start) / step);
	Nexum_Tensor_init(A, 1, n);
	f64 curr = start;
	for(u64 j=0; j<A->n; j++) {
		A->data[j] = curr;
		curr += step;
	}
}

/**
 * @brief Allocate and initialize step spaced array
 *
 * The Same as Nexum_Tensor_init_arange() but instead you provide the step size.
 * It get calculated using the size of the tensor which passed as an argument.
 *
 * The step is calculated using (end - start) / end;
 *
 * @param A pointer to the tensor to initialzie.
 * @param start starting value of the tensor data.
 * @param end ending value of the tensor data (exclusive).
 * @param size the length of the tensor.
 *
 * @return void
 */
void Nexum_Tensor_init_linspace(Nexum_Tensor* A, f64 start, f64 end, u64 size) {
	f64 step = (end - start) / (f64)size;
	Nexum_Tensor_init(A, 1, size);
	Nexum_Tensor_init_arange(A, start, end, step);
}

/**
 * @brief Change the data of the tensor to another
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_set_data(Nexum_Tensor* A, f64* data) {
	if(A->allocated) {
		printf("here\n");
		A->data = data;
	}
}

/**
 * @brief Perform element wize addition operation
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_add_tensor(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B){
	if (!(A->m == B->m && A->n == B->n)) {
		fprintf(stderr, "Cannot add two matrices with shape (%lu, %lu), (%lu, %lu).\n",
				A->m, A->n, B->m, B->n);
		exit(EXIT_FAILURE);
	}

	Nexum_Tensor_init(C, A->m, A->n);
	cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);

	/* for(u64 i=0; i<C->m; i++) { */
	/* 	for(u64 j=0; j<C->n; j++) { */
	/* 		C->data[C->n*i + j] = A->data[A->n*i + j] + B->data[B->n*i + j]; */
	/* 	} */
	/* } */
}

/**
 * @brief Perform element wize substraction operation
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_sub_tensor(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B){
	if (!(A->m == B->m && A->n == B->n)) {
		fprintf(stderr, "Cannot add two matrices with shape (%lu, %lu), (%lu, %lu).\n",
				A->m, A->n, B->m, B->n);
		exit(EXIT_FAILURE);
	}

	Nexum_Tensor_init(C, A->m, A->n);
	cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize multiplication operation
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_mul_tensor(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B){
	if (!(A->m == B->m && A->n == B->n)) {
		fprintf(stderr, "Cannot add two matrices with shape (%lu, %lu), (%lu, %lu).\n",
				A->m, A->n, B->m, B->n);
		exit(EXIT_FAILURE);
	}

	Nexum_Tensor_init(C, A->m, A->n);
	cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize division operation
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_div_tensor(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B){
	if (!(A->m == B->m && A->n == B->n)) {
		fprintf(stderr, "Cannot add two matrices with shape (%lu, %lu), (%lu, %lu).\n",
				A->m, A->n, B->m, B->n);
		exit(EXIT_FAILURE);
	}

	Nexum_Tensor_init(C, A->m, A->n);
	cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize addition with broadcast
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_add_tensor_boradcast(Nexum_Tensor* C, Nexum_Tensor* B, Nexum_Tensor* A, u8 axis) {
	if (axis == BROADCAST_AXIS_ROW) {

	} else if (axis == BROADCAST_AXIS_COL) {

	}
}

/**
 * @brief Perform element wize substraction with broadcast
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_sub_tensor_boradcast(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8 axis) {
	if (axis == BROADCAST_AXIS_ROW) {

	} else if (axis == BROADCAST_AXIS_COL) {

	}
}

/**
 * @brief Perform element wize multiplication with broadcast
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_mul_tensor_broadcast(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8 axis) {
	if (axis == BROADCAST_AXIS_ROW) {

	} else if (axis == BROADCAST_AXIS_COL) {

	}
}

/**
 * @brief Perform element wize division with broadcast
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_div_tensor_broadcast(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B, u8 axis) {
	if (axis == BROADCAST_AXIS_ROW) {

	} else if (axis == BROADCAST_AXIS_COL) {

	}
}

/**
 * @brief Perform element wize addition with scalar value.
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_add_scalar(Nexum_Tensor* C, Nexum_Tensor* A, f64 B){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<C->m; i++) {
		for(u64 j=0; j<C->n; j++) {
			C->data[C->n*i + j] = A->data[A->n*i + j] + B;
		}
	}
}

/**
 * @brief Perform element wize substraction with scalar value.
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_sub_scalar(Nexum_Tensor* C, Nexum_Tensor* A, f64 B){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<C->m; i++) {
		for(u64 j=0; j<C->n; j++) {
			C->data[C->n*i + j] = A->data[A->n*i + j] - B;
		}
	}
}

/**
 * @brief Perform element wize multiplication with scalar value.
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_mul_scalar(Nexum_Tensor* C, Nexum_Tensor* A, f64 B){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<C->m; i++) {
		for(u64 j=0; j<C->n; j++) {
			C->data[C->n*i + j] = A->data[A->n*i + j] * B;
		}
	}
}

/**
 * @brief Perform element wize division with scalar value.
 *
 *
 * @param
 * @param
 *
 * @return void
 */
void Nexum_Tensor_div_scalar(Nexum_Tensor* C, Nexum_Tensor* A, f64 B){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<C->m; i++) {
		for(u64 j=0; j<C->n; j++) {
			C->data[C->n*i + j] = A->data[A->n*i + j] / B;
		}
	}
}

void Nexum_Tensor_add_scalar_(Nexum_Tensor* A, f64 B){
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = A->data[A->n*i + j] + B;
		}
	}
}

void Nexum_Tensor_sub_scalar_(Nexum_Tensor* A, f64 B){
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = A->data[A->n*i + j] - B;
		}
	}
}

void Nexum_Tensor_mul_scalar_(Nexum_Tensor* A, f64 B){
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = A->data[A->n*i + j] * B;
		}
	}
}

void Nexum_Tensor_div_scalar_(Nexum_Tensor* A, f64 B){
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = A->data[A->n*i + j] / B;
		}
	}
}

/**
 * @brief Perform Tensor Multiplication.
 *
 * This function is used to perfor the matrix multiplication operation. This function
 * uses the BLAS Subroutine `cblas_dgemm` to perform the multiplication and that for two resoans.
 * one it's very fast how this routine is implemented and second it's quite stable and don't
 * have unsual behaviors.
 *
 * First it check if the multiplication operation is valid for the supplied tensors where the
 * number of columns of the first tensor must equals the number of rows of the second tensor.
 * as `A->n == B->m`.
 *
 * @param C The resulting tensor of the product.
 * @param A The first tensor with shape (m, n).
 * @param B The second tensor with shape (n, k).
 *
 * @see Nexum_Tensor_mul_tensor()
 *
 * @return void
 */
void Nexum_Tensor_matmul_tensor(Nexum_Tensor* C, Nexum_Tensor* A, Nexum_Tensor* B){
	if(A->n != B->m){
		fprintf(stderr, "Cannot multiply matrix with shape (%lu, %lu) with (%lu, %lu).\n",
				A->m, A->n, B->m, B->n);
		exit(EXIT_FAILURE);
	}

	Nexum_Tensor_init(C, A->m, B->n);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				C->m, C->n, A->n, 1.0, A->data, A->n, B->data, B->n, 1.0, C->data, C->n);
}

void Nexum_Tensor_reshape(Nexum_Tensor* B, Nexum_Tensor* A, u64 m, u64 n) {
	if(m*n != A->m*A->n) {
		fprintf(stderr, "cannot reshape tensor with shape (%lu, %lu) to (%lu, %lu)\n",
				A->m, A->n, m, n);
		exit(EXIT_FAILURE);
	}
	Nexum_Tensor_init(B, m, n);
	Nexum_Tensor_set_data(B, A->data);
}

/**
 * @brief Inplace reshaping of a tensor from shape to another.
 *
 * Inplace Reshaping from the original tensor shape to the specified shape in the
 * argument and it first check if the new shape match the new shape.
 *
 * @param A pointer to the tensor object to initialize.
 * @param m new number of rows.
 * @param n new number of columns
 *
 * @return void
 */

void Nexum_Tensor_reshape_(Nexum_Tensor* A, u64 m, u64 n) {
	if(m*n != A->m*A->n) {
		fprintf(stderr, "cannot reshape tensor with shape (%lu, %lu) to (%lu, %lu).\n",
				A->m, A->n, m, n);
		exit(EXIT_FAILURE);
	}
	A->m = m; A->n = n;
}

/**
 * @brief Transpose the Tensor object.
 *
 * Perform transpose operation on the tensor A and store the resulting in the B tensor.
 *
 * @param B The transposed tensor.
 * @param A The tensor to calculate the transpose for.
 *
 * @return void
 */
void Nexum_Tensor_transpose(Nexum_Tensor* B, Nexum_Tensor* A) {
	Nexum_Tensor_free(B);
	Nexum_Tensor_init(B, A->n, A->m);
	Nexum_Tensor_set_data(B, A->data);
}

/**
 * @brief Transpose the Tensor object inplace.
 *
 * Perform transpose operation on the tensor A and store the resulting in the same tensor. 
 *
 * @param A The tensor to calculate the transpose for.
 *
 * @return void
 */
void Nexum_Tensor_transpose_(Nexum_Tensor* A) {
	u64 temp;

	temp = A->n;
	A->n = A->m;
	A->m = temp;
}

/**
 * @brief Print a tensor to the console.
 *
 * Print a tensor data to the screen in very nice way.
 *
 * @param A pointer to the tensor to print.
 *
 * @return void
 */
void Nexum_Tensor_print(Nexum_Tensor* A){
	printf("Tensor(%ld, %ld)\n", A->m, A->n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			printf("%5.2f, ", A->data[A->n*i + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * @brief Print tensor data to the screen in scientific way.
 *
 * Prints tensor data to the console in the scientific way using the `.3e` formatting
 * specifier.
 *
 * @param A pointer to the tensor to print.
 *
 * @return void.
 * */
void Nexum_Tensor_print_raw(Nexum_Tensor* A){
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			printf("%.3e ", A->data[A->n*i + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * @brief Read a tensor from utf-8 text file.
 *
 * Read a tensor data and size from the UTF-8 text file the is writen using scientific
 * notation using `.3e` formating specifier.
 *
 * The file have the following design
 * ```txt
 * 2 2
 * 1.00e+00 0.00e+00
 * 0.00e+00 1.00e+00
 * ```
 *
 * The tensor only get populated if the file does exists
 *
 * @param A pointer to the  tensor to read from file.
 * @param fname path of the file.
 *
 * @return void.
 */
void Nexum_Tensor_read(Nexum_Tensor* A, str fname) {
	FILE* fptr = fopen(fname, READ_MODE);
	u64 m, n;
	if(fptr == NULL) {
		fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
		exit(EXIT_FAILURE);
	}
	
	fscanf(fptr, "%lu %lu", &m, &n);
	Nexum_Tensor_init(A, m, n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			fscanf(fptr, "%lf", &(A->data[A->n*i + j]));
		}
	}
	fclose(fptr);
}

/**
 * Read a tensor from binary file (.bin).
 */
void Nexum_Tensor_read_binary(Nexum_Tensor* A, str fname){
	FILE* fptr = fopen(fname, READ_BINARY_MODE);
	u64 m, n;
	if(fptr == NULL) {
		fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
		exit(EXIT_FAILURE);
	}
	fread(&m, sizeof (u64), 1, fptr);
	fread(&n, sizeof (u64), 1, fptr);
	Nexum_Tensor_init(A, m, n);
	fread(&(A->data[0]), sizeof (f64), m*n, fptr);
	fclose(fptr);
}

/**
 * Write a tensor to utf-8 text file.
 */
void Nexum_Tensor_write(Nexum_Tensor* A, str fname) {
	FILE* fptr = fopen(fname, WRITE_MODE);

	if(fptr == NULL) {
		fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
		exit(EXIT_FAILURE);
	}
	fprintf(fptr, "%lu %lu\n", A->m, A->n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			fprintf(fptr, "%.3e ", A->data[A->n*i + j]);
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n");
	fclose(fptr);
}

/**
 * Write a tensor to binary file (.bin).
 */
void Nexum_Tensor_write_binary(Nexum_Tensor* A, str fname){
	FILE* fptr = fopen(fname, WRITE_BINARY_MODE);

	if(fptr == NULL) {
		fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
		exit(EXIT_FAILURE);
	}
	fwrite(&(A->m), sizeof (u64), 1, fptr);
	fwrite(&(A->n), sizeof (u64), 1, fptr);
	fwrite(&(A->data[0]), sizeof (f64), A->m*A->n, fptr);
	fclose(fptr);
}

/**
 * @brief Free the tensor date from the Headp memory.
 *
 * This function is used to clean and free memory from the tensor data that is might
 * be sometime very big so cleaning memory from unused tensor will be very helpful.
 *
 * Also this function tries to reduce error due to memory overflow or double free of
 * the same data by checking that the specified tensor is already allocated or not
 * refer to Nexum_Tensor::allocated.
 *
 * @param A pointer to the tensor to free.
 *
 * @return void
 *
 * @todo Reimplement the function to check any uncleaned memory.
 */
void Nexum_Tensor_free(Nexum_Tensor* A){
	if(A->allocated) {
		free(A->data);
	}
	A->m = 0; A->n = 0;
	A->allocated = false;
}

f64 Nexum_Tensor_sum(Nexum_Tensor* A) {
    f64 sum = 0.0;
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			sum +=  A->data[A->n*i + j];
		}
	}
    return sum;
}

u64 Nexum_Tensor_size(Nexum_Tensor* A) {
    if(A->allocated) {
        return A->m*A->n;
    }
    return 0;
}

void Nexum_Tensor_neg(Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<C->m; i++) {
		for(u64 j=0; j<C->n; j++) {
			C->data[C->n*i + j] = -A->data[A->n*i + j];
		}
	}
}

void Nexum_Tensor_neg_(Nexum_Tensor* A) {
	Nexum_Tensor_neg(A, A);
}

void Nexum_Tensor_pow(Nexum_Tensor* C, Nexum_Tensor* A, i32 p) {
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			C->data[A->n*i + j] = pow(A->data[A->n*i + j], p);
		}
	}
}

void Nexum_Tensor_pow_(Nexum_Tensor* A, i32 p) {
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = pow(A->data[A->n*i + j], p);
		}
	}
}

void Nexum_Tensor_apply  (Nexum_Tensor* C, Nexum_Tensor* A, f64(*pfunc)(f64)) {
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			C->data[A->n*i + j] = pfunc(A->data[A->n*i + j]);
		}
	}
}

void Nexum_Tensor_apply_ (Nexum_Tensor* A, f64(*pfunc)(f64)) {
	Nexum_Tensor_apply(A, A, pfunc);
}

void Nexum_Tensor_abs(Nexum_Tensor* C, Nexum_Tensor* A) {
	Nexum_Tensor_apply(C, A, fabs);
}

void Nexum_Tensor_abs_(Nexum_Tensor* A) {
	Nexum_Tensor_apply_(A, fabs);
}

void Nexum_Tensor_sign   (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			A->data[A->n*i + j] = A->data[A->n*i + j] >= 0 ? 1.f : -1.0f;
		}
	}
}

void Nexum_Tensor_sign_  (Nexum_Tensor* A){
	Nexum_Tensor_sign(A, A);
}

void Nexum_Tensor_square (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_init(C, A->m, A->n);
	for(u64 i=0; i<A->m; i++) {
		for(u64 j=0; j<A->n; j++) {
			C->data[A->n*i + j] = pow(A->data[A->n*i + j], 2);
		}
	}
}

void Nexum_Tensor_square_(Nexum_Tensor* A){
	Nexum_Tensor_square(A, A);
}

void Nexum_Tensor_exp    (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_apply(C, A, exp);
}

void Nexum_Tensor_exp_   (Nexum_Tensor* A){
	Nexum_Tensor_apply_(A, exp);
}

void Nexum_Tensor_log    (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_apply(C, A, log);
}

void Nexum_Tensor_log_   (Nexum_Tensor* A){
	Nexum_Tensor_apply_(A, log);
}

void Nexum_Tensor_log10  (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_apply(C, A, log10);
}

void Nexum_Tensor_log10_ (Nexum_Tensor* A){
	Nexum_Tensor_apply_(A, log10);
}

void Nexum_Tensor_cos    (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_apply(C, A, cos);
}

void Nexum_Tensor_cos_   (Nexum_Tensor* A){
	Nexum_Tensor_apply_(A, cos);
}

void Nexum_Tensor_sin    (Nexum_Tensor* C, Nexum_Tensor* A){
	Nexum_Tensor_apply(C, A, sin);
}

void Nexum_Tensor_sin_   (Nexum_Tensor* A){
	Nexum_Tensor_apply_(A, sin);
}

