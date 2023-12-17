#include "NxCore.h"
#include "NxTensor.h"

#include <time.h>
#include <math.h>
// #include <cblas.h>
// #include <lapack.h>

/**
 * @brief Initialize a Tensor in memory filled with Garbage.
 *
 * Used to inialize a Tensor in Memory with (m*n) size filled
 * with Garbage from Memory which is different from random initialization.
 * for random initialization NxTensor_alloc_rand(), NxTensor_alloc_randn().
 * and it works like that: it checks first that the tensor is not already allocated
 * or had been allocated before using `allocated` param.
 * and if not it allocates new memory to it which is avery hacky way to avoid Memory Leak.
 *
 * @param A: pointer to the Tensor object that will be allocated.
 * @param m: number of rows to be allocated.
 * @param n: number of columnsto be allocated..
 *
 * @todo Rewrite code to handle Already allocated Tensors.
 */
NxCDEF void NxTensor_alloc(NxTensor* A, u64 m, u64 n){

    if(!A->allocated) {
        A->data = NxMALLOC(sizeof(NxDTYPE)*m*n);
        NxASSERT(A->data != NULL);
        A->m = m; A->n = n;
        A->allocated = true;
        return ;
    }
    if(m*n != A->m*A->n) {
        NxTensor_free(A);
        NxTensor_alloc(A, m, n);
    }
}

/**
 * @brief Allocate and initialize with zeros
 *
 * Initialize a tensor data to zeros.
 *
 * @param A pointer to the tensor object to inialize.
 * @param m number of rows.
 * @param n number of columns
 */
NxCDEF void NxTensor_alloc_zeros(NxTensor* A, u64 m, u64 n){
    NxTensor_alloc(A, m, n);
    u64 i, j;
    NxLOOP(i, m) {
        NxLOOP(j, n) {
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
 */
NxCDEF void NxTensor_alloc_ones(NxTensor* A, u64 m, u64 n){
    NxTensor_alloc(A, m, n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = 1.0f;
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
 * @todo Reimplement it using the BLAS library.
 */
NxCDEF void NxTensor_alloc_rand(NxTensor* A, u64 m, u64 n){
    srand(time(NULL));
    NxTensor_alloc(A, m, n);
    u64 i, j;
    NxLOOP(i, m) {
        NxLOOP(j, n) {
            A->data[n*i + j] = (NxDTYPE)rand()/(NxDTYPE)RAND_MAX;
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
 * @todo Reimplement it using the BLAS library.
 */
NxCDEF void NxTensor_alloc_randn(NxTensor* A, u64 m, u64 n){
    srand(time(NULL));
    NxTensor_alloc(A, m, n);
    u64 i, j;
    NxLOOP(i, m) {
        NxLOOP(j, n) {
            A->data[n*i + j] = (NxDTYPE)rand()/(NxDTYPE)RAND_MAX;
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
 */
NxCDEF void NxTensor_alloc_eye(NxTensor* A, u64 m){
    NxTensor_alloc(A, m, m);
    u64 i;
    NxLOOP(i, m) {
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
 */
NxCDEF void NxTensor_alloc_arange(NxTensor* A, NxDTYPE start, NxDTYPE end, NxDTYPE step) {
    u64 n = (u64)ceil((end - start) / step);
    NxTensor_alloc(A, 1, n);
    NxDTYPE curr = start;
    u64 j;
    NxLOOP(j, A->n) {
        A->data[j] = curr;
        curr += step;
    }
}

/**
 * @brief Allocate and initialize step spaced array
 *
 * The Same as NxTensor_alloc_arange() but instead you provide the step size.
 * It get calculated using the size of the tensor which passed as an argument.
 *
 * The step is calculated using (end - start) / end;
 *
 * @param A pointer to the tensor to initialzie.
 * @param start starting value of the tensor data.
 * @param end ending value of the tensor data (exclusive).
 * @param size the length of the tensor.
 */
NxCDEF void NxTensor_alloc_linspace(NxTensor* A, NxDTYPE start, NxDTYPE end, u64 size) {
    NxDTYPE step = (end - start) / (NxDTYPE)size;
    NxTensor_alloc(A, 1, size);
    NxTensor_alloc_arange(A, start, end, step);
}

/**
 * @brief Change the data of the tensor to another
 *for
 */
NxCDEF void NxTensor_set_data(NxTensor* A, NxDTYPE* data) {
    NxASSERT(A->allocated);
    A->data = data;
}


/**
 * @brief Copy the data of tensor to another
 * 
 * @param C pointer to the reciever tensor.
 * @param A pointer to the doner tensor.
 */
NxCDEF void NxTensor_copy_data(NxTensor* C, NxTensor* A) {
    NxASSERT(A->allocated);
    NxTensor_alloc(C, A->m, A->n);
    memcpy(C->data, A->data, NxTensor_size(A));
}


/**
 * @brief Perform element wize addition operation
 *
 */
NxCDEF void NxTensor_add_tensor(NxTensor* C, NxTensor* A, NxTensor* B){
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    if (!(A->m == B->m && A->n == B->n)) {
        fprintf(stderr, "Cannot add two matrices with shape (%I64u, %I64u), (%I64u, %I64u).\n",
                A->m, A->n, B->m, B->n);
        exit(EXIT_FAILURE);
    }

    NxTensor_alloc(C, A->m, A->n);
    // cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize substraction operation
 *
 */
NxCDEF void NxTensor_sub_tensor(NxTensor* C, NxTensor* A, NxTensor* B){
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    if (!(A->m == B->m && A->n == B->n)) {
        fprintf(stderr, "Cannot add two matrices with shape (%I64u, %I64u), (%I64u, %I64u).\n",
                A->m, A->n, B->m, B->n);
        exit(EXIT_FAILURE);
    }

    NxTensor_neg_(B);
    NxTensor_alloc(C, A->m, A->n);
    // cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize multiplication operation
 *
 */
NxCDEF void NxTensor_mul_tensor(NxTensor* C, NxTensor* A, NxTensor* B){
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    if (!(A->m == B->m && A->n == B->n)) {
        fprintf(stderr, "Cannot add two matrices with shape (%I64u, %I64u), (%I64u, %I64u).\n",
                A->m, A->n, B->m, B->n);
        exit(EXIT_FAILURE);
    }

    NxTensor_alloc(C, A->m, A->n);
    // cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize division operation
 *
 */
NxCDEF void NxTensor_div_tensor(NxTensor* C, NxTensor* A, NxTensor* B){
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    if (!(A->m == B->m && A->n == B->n)) {
        fprintf(stderr, "Cannot add two matrices with shape (%I64u, %I64u), (%I64u, %I64u).\n",
                A->m, A->n, B->m, B->n);
        exit(EXIT_FAILURE);
    }

    NxTensor_alloc(C, A->m, A->n);
    // cblas_dgeadd(CblasRowMajor, C->m, C->n, 1.0, A->data, A->n, 1.0, C->data, C->n);
}

/**
 * @brief Perform element wize addition with broadcast
 *
 */
NxCDEF void NxTensor_add_tensor_boradcast(NxTensor* C, NxTensor* A, NxTensor* B, u8 axis) {
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    (void) C;
    (void) A;
    (void) B;
    (void) axis;

    if (axis == NxAXIS_ROW) {

    } else if (axis == NxAXIS_COL) {

    }
}

/**
 * @brief Perform element wize substraction with broadcast
 *
 */
NxCDEF void NxTensor_sub_tensor_boradcast(NxTensor* C, NxTensor* A, NxTensor* B, u8 axis) {
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    (void) C;
    (void) A;
    (void) B;
    (void) axis;

    if (axis == NxAXIS_ROW) {

    } else if (axis == NxAXIS_COL) {

    }
}

/**
 * @brief Perform element wize multiplication with broadcast
 *
 */
NxCDEF void NxTensor_mul_tensor_broadcast(NxTensor* C, NxTensor* A, NxTensor* B, u8 axis) {
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    (void) C;
    (void) A;
    (void) B;
    (void) axis;

    if (axis == NxAXIS_ROW) {

    } else if (axis == NxAXIS_COL) {

    }
}

/**
 * @brief Perform element wize division with broadcast
 *
 */
NxCDEF void NxTensor_div_tensor_broadcast(NxTensor* C, NxTensor* A, NxTensor* B, u8 axis) {
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    NxNOTIMPLEMENTED();

    (void) C;
    (void) A;
    (void) B;
    (void) axis;

    if (axis == NxAXIS_ROW) {

    } else if (axis == NxAXIS_COL) {

    }
}

/**
 * @brief Perform element wize addition with scalar value.
 *
 */
NxCDEF void NxTensor_add_scalar(NxTensor* C, NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[C->n*i + j] = A->data[NxIDX(A->n, i, j)] + B;
        }
    }
}

/**
 * @brief Perform element wize substraction with scalar value.
 *
 */
NxCDEF void NxTensor_sub_scalar(NxTensor* C, NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[C->n*i + j] = A->data[NxIDX(A->n, i, j)] - B;
        }
    }
}

/**
 * @brief Perform element wize multiplication with scalar value.
 *
 */
NxCDEF void NxTensor_mul_scalar(NxTensor* C, NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[C->n*i + j] = A->data[NxIDX(A->n, i, j)] * B;
        }
    }
}

/**
 * @brief Perform element wize division with scalar value.
 *
 */
NxCDEF void NxTensor_div_scalar(NxTensor* C, NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[C->n*i + j] = A->data[NxIDX(A->n, i, j)] / B;
        }
    }
}

/**
 * @brief Perform element-wize addition of tensor element to scalar value.
 * 
 * Same as the NxTensor_add_scalar() but performs the operation inplace.
 * 
 * @param A pointer to the tensor object and the output.
 * @param B the scalar value.
 */
NxCDEF void NxTensor_add_scalar_(NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = A->data[NxIDX(A->n, i, j)] + B;
        }
    }
}

/**
 * @brief Perform element-wize substraction of tensor element to scalar value.
 * 
 * Same as the NxTensor_sub_scalar() but performs the operation inplace.
 * 
 * @param A pointer to the tensor object and the output.
 * @param B the scalar value.
 */
NxCDEF void NxTensor_sub_scalar_(NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = A->data[NxIDX(A->n, i, j)] - B;
        }
    }
}

/**
 * @brief Perform element-wize multiplication of tensor element to scalar value.
 * 
 * Same as the NxTensor_mul_scalar() but performs the operation inplace.
 * 
 * @param A pointer to the tensor object and the output.
 * @param B the scalar value.
 */
NxCDEF void NxTensor_mul_scalar_(NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);

    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = A->data[NxIDX(A->n, i, j)] * B;
        }
    }
}

/**
 * @brief Perform element-wize division of tensor element to scalar value.
 * 
 * Same as the NxTensor_div_scalar() but performs the operation inplace.
 * 
 * @param A pointer to the tensor object and the output.
 * @param B the scalar value.
 */
NxCDEF void NxTensor_div_scalar_(NxTensor* A, NxDTYPE B){
    NxASSERT(A->allocated);
    NxASSERT(B != 0);

    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = A->data[NxIDX(A->n, i, j)] / B;
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
 * @see NxTensor_mul_tensor()
 */
NxCDEF void NxTensor_matmul_tensor(NxTensor* C, NxTensor* A, NxTensor* B){
    NxASSERT(A->allocated);
    NxASSERT(B->allocated);

    if(A->n != B->m){
        fprintf(stderr, "Cannot multiply matrix with shape (%I64u, %I64u) with (%I64u, %I64u).\n",
                A->m, A->n, B->m, B->n);
        exit(EXIT_FAILURE);
    }

    NxTensor_alloc(C, A->m, B->n);
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             C->m, C->n, A->n, 1.0, A->data, A->n, B->data, B->n, 1.0, C->data, C->n);
}

/**
 * @brief Reshape tensor to a given shape.
 *
 * Reshapes a tensor from it's shape to a new shape but first it
 * checks that the new shape is the same as older shape.
 *
 * @param B pointer to the output tensor.
 * @param A pointer to the input tensor.
 * @param m new number of rows.
 * @param n new number of columns.
 *
 * @see NxTensor_reshape_() for inplace reshaping.
 */
NxCDEF void NxTensor_reshape(NxTensor* B, NxTensor* A, u64 m, u64 n) {
    NxASSERT(A->allocated);

    if(m*n != A->m*A->n) {
        fprintf(stderr, "cannot reshape tensor with shape (%I64u, %I64u) to (%I64u, %I64u)\n",
                A->m, A->n, m, n);
        exit(EXIT_FAILURE);
    }
    NxTensor_alloc(B, m, n);
    NxTensor_set_data(B, A->data);
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
 */

NxCDEF void NxTensor_reshape_(NxTensor* A, u64 m, u64 n) {
    NxASSERT(A->allocated);

    if(m*n != A->m*A->n) {
        fprintf(stderr, "cannot reshape tensor with shape (%I64u, %I64u) to (%I64u, %I64u).\n",
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
 */
NxCDEF void NxTensor_transpose(NxTensor* B, NxTensor* A) {
    NxASSERT(A->allocated);

    NxTensor_free(B);
    NxTensor_alloc(B, A->n, A->m);
    NxTensor_set_data(B, A->data);
}

/**
 * @brief Transpose the Tensor object inplace.
 *
 * Perform transpose operation on the tensor A and store the resulting in the same tensor. 
 *
 * @param A The tensor to calculate the transpose for.
 */
NxCDEF void NxTensor_transpose_(NxTensor* A) {
    NxASSERT(A->allocated);

    u64 temp;

    temp = A->n;
    A->n = A->m;
    A->m = temp;
}

/**
 * @brief Sums Tensor elements along a given axis.
 * 
 * Sums a tensor alongside given axis that have only two value `NxAXIS_ROW`,
 * and `NxAXIS_COL`.
 * 
 * @param C pointer to the output tensor.
 * @param A pointer to the input tensor.
 * @param axis the axis to sum over.
 */
NxCDEF void NxTensor_sum_tensor(NxTensor* C, NxTensor* A, u32 axis) {
    NxASSERT(A->allocated);

    u64 i, j;

    if(axis == NxAXIS_ROW) {
        NxTensor_alloc_zeros(C, 1, A->n);
        NxLOOP(i, A->m) {
            NxLOOP(j, A->n) {
                C->data[j] += NxTensor_AT(A, i, j);
            }
        }
    } else if (axis == NxAXIS_COL) {
        NxTensor_alloc_zeros(C, A->m, 1);
        NxLOOP(i, A->m) {
            NxLOOP(j, A->n) {
                C->data[i] += NxTensor_AT(A, i, j);
            }
        }
    } else {
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Print a tensor to the console.
 *
 * Print a tensor data to the screen in very nice way.
 *
 * @param A pointer to the tensor to print.
 */
NxCDEF void NxTensor_to_string(NxTensor* A){
    NxASSERT(A->allocated);

    printf("Tensor(%I64d, %I64d)\n", A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            printf("%5.2f, ", A->data[NxIDX(A->n, i, j)]);
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
 * @param A pointer to the tensor to print..
 * */
NxCDEF void NxTensor_to_string_raw(NxTensor* A){
    NxASSERT(A->allocated);

    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            printf("%.3e ", A->data[NxIDX(A->n, i, j)]);
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
 * @param fname path of the file..
 */
NxCDEF void NxTensor_read(NxTensor* A, str fname) {

    FILE* fptr = fopen(fname, READ_MODE);
    u64 m, n;
    if(fptr == NULL) {
        fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
        exit(EXIT_FAILURE);
    }
    
    fscanf(fptr, "%I64u %I64u", &m, &n);
    NxTensor_alloc(A, m, n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            fscanf(fptr, "%lf", &(A->data[NxIDX(A->n, i, j)]));
        }
    }
    fclose(fptr);
}

/**
 * @brief Read a tensor from binary file (.bin).
 *
 * @param A pointer to the tensor object.
 * @param fname filename.
 */
NxCDEF void NxTensor_read_binary(NxTensor* A, str fname){

    FILE* fptr = fopen(fname, READ_BINARY_MODE);
    u64 m, n;
    if(fptr == NULL) {
        fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
        exit(EXIT_FAILURE);
    }
    fread(&m, sizeof (u64), 1, fptr);
    fread(&n, sizeof (u64), 1, fptr);
    NxTensor_alloc(A, m, n);
    fread(&(A->data[0]), sizeof (NxDTYPE), m*n, fptr);
    fclose(fptr);
}

/**
 * @brief Write a tensor to utf-8 text file.
 *
 * @param A pointer to the tensor object.
 * @param fname filename.
 */
NxCDEF void NxTensor_write(NxTensor* A, str fname) {
    NxASSERT(A->allocated);

    FILE* fptr = fopen(fname, WRITE_MODE);

    if(fptr == NULL) {
        fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(fptr, "%I64u %I64u\n", A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            fprintf(fptr, "%.3e ", A->data[NxIDX(A->n, i, j)]);
        }
        fprintf(fptr, "\n");
    }
    fprintf(fptr, "\n");
    fclose(fptr);
}

/**
 * @brief Write a tensor to binary file (.bin).
 * 
 * @param A pointer to the tensor object.
 * @param fname filename.
 */
NxCDEF void NxTensor_write_binary(NxTensor* A, str fname){
    NxASSERT(A->allocated);

    FILE* fptr = fopen(fname, WRITE_BINARY_MODE);

    if(fptr == NULL) {
        fprintf(stderr, "Cannot open file %s file does not exists.\n", fname);
        exit(EXIT_FAILURE);
    }
    fwrite(&(A->m), sizeof (u64), 1, fptr);
    fwrite(&(A->n), sizeof (u64), 1, fptr);
    fwrite(&(A->data[0]), sizeof (NxDTYPE), A->m*A->n, fptr);
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
 * refer to NxTensor::allocated.
 *
 * @param A pointer to the tensor to free.
 *
 * @todo Reimplement the function to check any uncleaned memory.
 */
NxCDEF void NxTensor_free(NxTensor* A){
    if (A->allocated) {
        // NxMESSAGE("INFO", "here");
        free(A->data);
        // NxMESSAGE("DEBUG", "here");
        A->m = 0; A->n = 0;
        A->allocated = false;
    }
}

/**
 * @brief Sums all the tensor values and return it.
 * 
 * @param A pointer to the tensor object.
 * 
 * @return (NxDTYPE)
 */
NxDTYPE NxTensor_sum(NxTensor* A) {
    NxASSERT(A->allocated);

    NxDTYPE sum = 0.0;
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            sum +=  A->data[NxIDX(A->n, i, j)];
        }
    }
    return sum;
}

/**
 * @brief Return the size of the tensor object.
 * 
 * @param A pointer to the tensor object.
 * 
 * @return (u64)
 */
u64 NxTensor_size(NxTensor* A) {
    NxASSERT(A->allocated);

    return A->m*A->n;
}

/**
 * @brief Perform a Negation operation on a tensor.
 * 
 * @param C pointer to the output tensor object.
 * @param A pointer to the input tensor.
 */
NxCDEF void NxTensor_neg(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[C->n*i + j] = -A->data[NxIDX(A->n, i, j)];
        }
    }
}

/**
 * @brief Perform the Negation operation inplace.
 * 
 * @param A pointer to the tensor object.
 */
NxCDEF void NxTensor_neg_(NxTensor* A) {
    NxASSERT(A->allocated);

    NxTensor_neg(A, A);
}

/**
 * @brief Perform the Power operation on a tensor.
 * 
 * @param C pointer to the output tensor object.
 * @param A pointer to the input tensor object.
 * @param p the exponant of the power.
 */
NxCDEF void NxTensor_pow(NxTensor* C, NxTensor* A, i32 p) {
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[NxIDX(A->n, i, j)] = pow(A->data[NxIDX(A->n, i, j)], p);
        }
    }
}

/**
 * @brief Perform the Power operation inplace.
 * 
 * @param A pointer to the tensor object.
 * @param p the exponant of the power.
 */
NxCDEF void NxTensor_pow_(NxTensor* A, i32 p) {
    NxASSERT(A->allocated);

    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = pow(A->data[NxIDX(A->n, i, j)], p);
        }
    }
}

/**
 * @brief Apply a function to every element in the tensor.
 * 
 * Applies a function (pfunc) on all element in the tensor. 
 * this function must have only one input of type (NxDTYPE)
 * and ouput of type (NxDTYPE).
 * 
 * This function will become very handy later as it will be easy to
 * define functions like (`sin`, `cos`, `exp`, ...) in just one line.
 * 
 * @param C pointer to the output tensor.
 * @param A pointer to the input tensor.
 * @param pfunc pointer to the function to apply.
 */
NxCDEF void NxTensor_apply(NxTensor* C, NxTensor* A, NxDTYPE(*pfunc)(NxDTYPE)) {
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            C->data[NxIDX(A->n, i, j)] = pfunc(A->data[NxIDX(A->n, i, j)]);
        }
    }
}

/**
 * @brief Apply a function to every element in the tensor inplace.
 * 
 * @param A pointer to the tensor object.
 * @param pfunc pointer to the function to apply.
 */
NxCDEF void NxTensor_apply_ (NxTensor* A, NxDTYPE(*pfunc)(NxDTYPE)) {
    NxASSERT(A->allocated);

    NxTensor_apply(A, A, pfunc);
}

/**
 * @brief Compute the absolute value of the tensor.
 */
NxCDEF void NxTensor_abs(NxTensor* C, NxTensor* A) {
    NxASSERT(A->allocated);
    NxTensor_apply(C, A, fabs);
}

/**
 * @brief Compute the absolute value of the tensor inplace.
 */
NxCDEF void NxTensor_abs_(NxTensor* A) {
    NxASSERT(A->allocated);
    NxTensor_apply_(A, fabs);
}

/**
 * @brief Compute the sign of the tensor.
 */
NxCDEF void NxTensor_sign(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);

    NxTensor_alloc(C, A->m, A->n);
    u64 i, j;
    NxLOOP(i, A->m) {
        NxLOOP(j, A->n) {
            A->data[NxIDX(A->n, i, j)] = A->data[NxIDX(A->n, i, j)] >= 0 ? 1.f : -1.0f;
        }
    }
}

/**
 * @brief Compute the sign of the tensor inplace.
 */
NxCDEF void NxTensor_sign_(NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_sign(A, A);
}

/**
 * @brief Compute the square of the tensor.
 */
NxCDEF void NxTensor_square(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_pow(C, A, 2);
}

/**
 * @brief Compute the square of the tensor inplace.
 */
NxCDEF void NxTensor_square_(NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_pow_(A, 2);
}

/**
 * @brief Compute the exponential value of the tensor.
 */
NxCDEF void NxTensor_exp(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply(C, A, exp);
}

/**
 * @brief Compute the exponential value of the tensor inplace.
 */
NxCDEF void NxTensor_exp_(NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply_(A, exp);
}

/**
 * @brief Compute the logrithms value of the tensor.
 */
NxCDEF void NxTensor_log(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply(C, A, log);
}

/**
 * @brief Compute the logrithms value of the tensor inplace.
 */
NxCDEF void NxTensor_log_(NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply_(A, log);
}

/**
 * @brief Compute the logrithms with base 10 value of the tensor.
 */
NxCDEF void NxTensor_log10  (NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply(C, A, log10);
}

/**
 * @brief Compute the logrithms with base 10 value of the tensor inplace.
 */
NxCDEF void NxTensor_log10_ (NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply_(A, log10);
}

/**
 * @brief Compute the `cos` of the tensor.
 */
NxCDEF void NxTensor_cos(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply(C, A, cos);
}

/**
 * @brief Compute the `cos` of the tensor inplace.
 */
NxCDEF void NxTensor_cos_(NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply_(A, cos);
}

/**
 * @brief Compute the `sin` of the tensor.
 */
NxCDEF void NxTensor_sin(NxTensor* C, NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply(C, A, sin);
}

/**
 * @brief Compute the `sin` of the tensor inplace.
 */
NxCDEF void NxTensor_sin_(NxTensor* A){
    NxASSERT(A->allocated);
    NxTensor_apply_(A, sin);
}

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
 * @file NxTensor.c
 * @author Moaz El-Essawey.
 * @date 11 July 2023.
 */

