#include "kernels.cuh"

namespace
{

constexpr int TILE_DIM = 16;
constexpr int THREADS_1D = 256;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* C[M,N] = op(A)[M,K] @ op(B)[K,N], tiled shared-memory matmul. */
__global__ void matmul_kernel(const float *A, const float *B, float *C,
							   int M, int K, int N, bool transA, bool transB)
{
	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	int row = blockIdx.y * TILE_DIM + threadIdx.y; /* index into M */
	int col = blockIdx.x * TILE_DIM + threadIdx.x; /* index into N */

	float acc = 0.0f;
	int numTiles = ceil_div(K, TILE_DIM);

	for (int t = 0; t < numTiles; ++t)
	{
		int a_k = t * TILE_DIM + threadIdx.x;
		int b_k = t * TILE_DIM + threadIdx.y;

		As[threadIdx.y][threadIdx.x] = (row < M && a_k < K)
			? (transA ? A[a_k * M + row] : A[row * K + a_k])
			: 0.0f;

		Bs[threadIdx.y][threadIdx.x] = (col < N && b_k < K)
			? (transB ? B[col * K + b_k] : B[b_k * N + col])
			: 0.0f;

		__syncthreads();

#pragma unroll
		for (int k = 0; k < TILE_DIM; ++k)
		{
			acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}
		__syncthreads();
	}

	if (row < M && col < N)
	{
		C[row * N + col] = acc;
	}
}

__global__ void gather_rows_kernel(const float *src, const int *indices, float *dst, int n_indices, int dim)
{
	int row = blockIdx.x;
	if (row >= n_indices)
		return;
	int src_row = indices[row];
	const float *src_ptr = src + (size_t)src_row * dim;
	float *dst_ptr = dst + (size_t)row * dim;
	for (int j = threadIdx.x; j < dim; j += blockDim.x)
	{
		dst_ptr[j] = src_ptr[j];
	}
}

__global__ void gather_labels_kernel(const unsigned char *src, const int *indices, unsigned char *dst, int n_indices)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_indices)
	{
		dst[i] = src[indices[i]];
	}
}

__global__ void add_bias_relu_kernel(float *Z, const float *b, int n_rows, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n_rows * dim)
	{
		int j = idx % dim;
		float v = Z[idx] + b[j];
		Z[idx] = v > 0.0f ? v : 0.0f;
	}
}

__global__ void add_bias_kernel(float *Z, const float *b, int n_rows, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n_rows * dim)
	{
		int j = idx % dim;
		Z[idx] += b[j];
	}
}

__global__ void softmax_ce_kernel(const float *logits, const unsigned char *labels, float *probs, float *loss,
								   int n_rows, int n_classes)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_rows)
		return;

	const float *row = logits + (size_t)i * n_classes;
	float *prow = probs + (size_t)i * n_classes;

	float maxv = row[0];
	for (int c = 1; c < n_classes; ++c)
	{
		maxv = fmaxf(maxv, row[c]);
	}

	float sum = 0.0f;
	for (int c = 0; c < n_classes; ++c)
	{
		float e = expf(row[c] - maxv);
		prow[c] = e;
		sum += e;
	}

	float inv_sum = 1.0f / (sum + 1e-8f);
	for (int c = 0; c < n_classes; ++c)
	{
		prow[c] *= inv_sum;
	}

	loss[i] = -logf(prow[labels[i]] + 1e-8f);
}

__global__ void softmax_ce_backward_kernel(const float *probs, const unsigned char *labels, float *dZ,
											int n_rows, int n_classes)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_rows * n_classes)
		return;
	int i = idx / n_classes;
	int c = idx % n_classes;
	float target = (labels[i] == c) ? 1.0f : 0.0f;
	dZ[idx] = (probs[idx] - target) / (float)n_rows;
}

__global__ void relu_backward_kernel(const float *A, const float *dA, float *dZ, int n_rows, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n_rows * dim)
	{
		dZ[idx] = (A[idx] > 0.0f) ? dA[idx] : 0.0f;
	}
}

__global__ void bias_grad_kernel(const float *dZ, float *db, int n_rows, int dim)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= dim)
		return;
	float sum = 0.0f;
	for (int i = 0; i < n_rows; ++i)
	{
		sum += dZ[(size_t)i * dim + j];
	}
	db[j] = sum;
}

__global__ void sgd_momentum_kernel(float *w, const float *grad, float *v, int size, float lr, float momentum)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		float v_new = momentum * v[idx] - lr * grad[idx];
		v[idx] = v_new;
		w[idx] += v_new;
	}
}

__global__ void count_correct_kernel(const float *logits, const unsigned char *labels, int *correct,
									  int n_rows, int n_classes)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_rows)
		return;

	const float *row = logits + (size_t)i * n_classes;
	int best = 0;
	float best_v = row[0];
	for (int c = 1; c < n_classes; ++c)
	{
		if (row[c] > best_v)
		{
			best_v = row[c];
			best = c;
		}
	}
	if (best == labels[i])
	{
		atomicAdd(correct, 1);
	}
}

} /* namespace */

void matmul(const float *A, const float *B, float *C, int M, int K, int N,
			bool transA, bool transB, cudaStream_t stream)
{
	dim3 block(TILE_DIM, TILE_DIM);
	dim3 grid(ceil_div(N, TILE_DIM), ceil_div(M, TILE_DIM));
	matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N, transA, transB);
}

void gather_rows(const float *src, const int *indices, float *dst, int n_indices, int dim, cudaStream_t stream)
{
	int threads = dim < THREADS_1D ? dim : THREADS_1D;
	if (threads < 1)
		threads = 1;
	gather_rows_kernel<<<n_indices, threads, 0, stream>>>(src, indices, dst, n_indices, dim);
}

void gather_labels(const unsigned char *src, const int *indices, unsigned char *dst, int n_indices,
					cudaStream_t stream)
{
	int blocks = ceil_div(n_indices, THREADS_1D);
	gather_labels_kernel<<<blocks, THREADS_1D, 0, stream>>>(src, indices, dst, n_indices);
}

void add_bias_relu(float *Z, const float *b, int n_rows, int dim, cudaStream_t stream)
{
	int total = n_rows * dim;
	add_bias_relu_kernel<<<ceil_div(total, THREADS_1D), THREADS_1D, 0, stream>>>(Z, b, n_rows, dim);
}

void add_bias(float *Z, const float *b, int n_rows, int dim, cudaStream_t stream)
{
	int total = n_rows * dim;
	add_bias_kernel<<<ceil_div(total, THREADS_1D), THREADS_1D, 0, stream>>>(Z, b, n_rows, dim);
}

void softmax_cross_entropy(const float *logits, const unsigned char *labels, float *probs, float *loss,
							int n_rows, int n_classes, cudaStream_t stream)
{
	softmax_ce_kernel<<<ceil_div(n_rows, THREADS_1D), THREADS_1D, 0, stream>>>(
		logits, labels, probs, loss, n_rows, n_classes);
}

void softmax_cross_entropy_backward(const float *probs, const unsigned char *labels, float *dZ,
									 int n_rows, int n_classes, cudaStream_t stream)
{
	int total = n_rows * n_classes;
	softmax_ce_backward_kernel<<<ceil_div(total, THREADS_1D), THREADS_1D, 0, stream>>>(
		probs, labels, dZ, n_rows, n_classes);
}

void relu_backward(const float *A, const float *dA, float *dZ, int n_rows, int dim, cudaStream_t stream)
{
	int total = n_rows * dim;
	relu_backward_kernel<<<ceil_div(total, THREADS_1D), THREADS_1D, 0, stream>>>(A, dA, dZ, n_rows, dim);
}

void bias_grad(const float *dZ, float *db, int n_rows, int dim, cudaStream_t stream)
{
	bias_grad_kernel<<<ceil_div(dim, THREADS_1D), THREADS_1D, 0, stream>>>(dZ, db, n_rows, dim);
}

void sgd_momentum_update(float *w, const float *grad, float *v, int size, float lr, float momentum,
						  cudaStream_t stream)
{
	sgd_momentum_kernel<<<ceil_div(size, THREADS_1D), THREADS_1D, 0, stream>>>(w, grad, v, size, lr, momentum);
}

void count_correct(const float *logits, const unsigned char *labels, int *correct, int n_rows, int n_classes,
					cudaStream_t stream)
{
	count_correct_kernel<<<ceil_div(n_rows, THREADS_1D), THREADS_1D, 0, stream>>>(
		logits, labels, correct, n_rows, n_classes);
}
