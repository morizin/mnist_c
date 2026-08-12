#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
	do                                                                             \
	{                                                                              \
		cudaError_t err__ = (call);                                               \
		if (err__ != cudaSuccess)                                                 \
		{                                                                          \
			fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
					cudaGetErrorString(err__));                                   \
			exit(EXIT_FAILURE);                                                   \
		}                                                                          \
	} while (0)

/* All matrices are row-major float arrays on the device.
 * Labels are single bytes (0-9) on the device; classes fit comfortably in a byte. */

/* C[M,N] = op(A)[M,K] @ op(B)[K,N].
 * transA: A is physically stored as [K,M] (so op(A) = A^T).
 * transB: B is physically stored as [N,K] (so op(B) = B^T). */
void matmul(const float *A, const float *B, float *C, int M, int K, int N,
			bool transA, bool transB, cudaStream_t stream = 0);

/* dst[i,:] = src[indices[i],:] for i in [0, n_indices), rows of width `dim`. */
void gather_rows(const float *src, const int *indices, float *dst, int n_indices, int dim,
				  cudaStream_t stream = 0);

/* dst[i] = src[indices[i]] */
void gather_labels(const unsigned char *src, const int *indices, unsigned char *dst, int n_indices,
					cudaStream_t stream = 0);

/* In place: Z[i,j] = relu(Z[i,j] + b[j]) */
void add_bias_relu(float *Z, const float *b, int n_rows, int dim, cudaStream_t stream = 0);

/* In place: Z[i,j] = Z[i,j] + b[j] */
void add_bias(float *Z, const float *b, int n_rows, int dim, cudaStream_t stream = 0);

/* Row-wise numerically-stable softmax of `logits` into `probs`, plus per-sample
 * cross-entropy loss against integer labels into `loss` (length n_rows). */
void softmax_cross_entropy(const float *logits, const unsigned char *labels, float *probs, float *loss,
							int n_rows, int n_classes, cudaStream_t stream = 0);

/* dZ[i,c] = (probs[i,c] - onehot(label[i])[c]) / n_rows  -- combined softmax+CE gradient */
void softmax_cross_entropy_backward(const float *probs, const unsigned char *labels, float *dZ,
									 int n_rows, int n_classes, cudaStream_t stream = 0);

/* dZ[i,j] = dA[i,j] * (A[i,j] > 0)   (A is the post-activation output, ReLU is zero-preserving) */
void relu_backward(const float *A, const float *dA, float *dZ, int n_rows, int dim, cudaStream_t stream = 0);

/* db[j] = sum_i dZ[i,j] */
void bias_grad(const float *dZ, float *db, int n_rows, int dim, cudaStream_t stream = 0);

/* SGD with momentum: v = momentum*v - lr*grad; w += v */
void sgd_momentum_update(float *w, const float *grad, float *v, int size, float lr, float momentum,
						  cudaStream_t stream = 0);

/* Zeroes *correct on the host side is the caller's job (cudaMemset); this kernel
 * atomically adds 1 for every row whose argmax matches its label. */
void count_correct(const float *logits, const unsigned char *labels, int *correct, int n_rows, int n_classes,
					cudaStream_t stream = 0);

#endif
