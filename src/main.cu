#include "kernels.cuh"
#include "mnist_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace
{

constexpr int INPUT_DIM = 28 * 28;
constexpr int HIDDEN_DIM = 256;
constexpr int OUTPUT_DIM = 10;
constexpr int BATCH_SIZE = 128;
constexpr int DEFAULT_EPOCHS = 15;
constexpr float LEARNING_RATE = 0.1f;
constexpr float MOMENTUM = 0.9f;
constexpr unsigned int SEED = 343524u;
constexpr float PI_F = 3.14159265358979323846f;

const char *TRAIN_IMAGES = "data/train-images.idx3-ubyte";
const char *TRAIN_LABELS = "data/train-labels.idx1-ubyte";
const char *TEST_IMAGES = "data/t10k-images.idx3-ubyte";
const char *TEST_LABELS = "data/t10k-labels.idx1-ubyte";

/* Standard normal via Box-Muller, driven by rand(). Good enough for weight init. */
float randn(void)
{
	float u1 = (float)((rand() + 1.0) / ((double)RAND_MAX + 1.0));
	float u2 = (float)(rand() / (double)RAND_MAX);
	return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
}

void fill_he_init(float *host_buf, int fan_in, int size)
{
	float std_dev = sqrtf(2.0f / (float)fan_in);
	for (int i = 0; i < size; ++i)
	{
		host_buf[i] = randn() * std_dev;
	}
}

void fisher_yates_shuffle(int *indices, int n)
{
	for (int i = n - 1; i > 0; --i)
	{
		int j = rand() % (i + 1);
		int tmp = indices[i];
		indices[i] = indices[j];
		indices[j] = tmp;
	}
}

struct Params
{
	float *W1, *b1, *W2, *b2; /* device */
};

struct Grads
{
	float *dW1, *db1, *dW2, *db2; /* device */
};

struct Velocity
{
	float *vW1, *vb1, *vW2, *vb2; /* device */
};

/* Scratch buffers sized for up to BATCH_SIZE rows, reused across every mini-batch
 * and (for the forward-only fields) reused again during evaluation. */
struct Scratch
{
	float *batch_x;			  /* [BATCH_SIZE, INPUT_DIM] */
	unsigned char *batch_y;	  /* [BATCH_SIZE] */
	int *indices;				  /* [BATCH_SIZE] */
	float *Z1;					  /* [BATCH_SIZE, HIDDEN_DIM] -- holds A1 (post-ReLU) after forward */
	float *Z2;					  /* [BATCH_SIZE, OUTPUT_DIM] */
	float *probs;				  /* [BATCH_SIZE, OUTPUT_DIM] */
	float *loss;				  /* [BATCH_SIZE] */
	float *dZ2;					  /* [BATCH_SIZE, OUTPUT_DIM] */
	float *dA1;					  /* [BATCH_SIZE, HIDDEN_DIM] */
	float *dZ1;					  /* [BATCH_SIZE, HIDDEN_DIM] */
};

float *device_alloc_f(size_t n)
{
	float *ptr;
	CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
	return ptr;
}

void upload_he(float **dst, int fan_in, int size)
{
	float *host_buf = (float *)malloc(size * sizeof(float));
	fill_he_init(host_buf, fan_in, size);
	*dst = device_alloc_f(size);
	CUDA_CHECK(cudaMemcpy(*dst, host_buf, size * sizeof(float), cudaMemcpyHostToDevice));
	free(host_buf);
}

float *device_zeros_f(size_t n)
{
	float *ptr = device_alloc_f(n);
	CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
	return ptr;
}

void forward(const float *x, int n_rows, const Params &p, Scratch &s)
{
	matmul(x, p.W1, s.Z1, n_rows, INPUT_DIM, HIDDEN_DIM, false, false);
	add_bias_relu(s.Z1, p.b1, n_rows, HIDDEN_DIM);

	matmul(s.Z1, p.W2, s.Z2, n_rows, HIDDEN_DIM, OUTPUT_DIM, false, false);
	add_bias(s.Z2, p.b2, n_rows, OUTPUT_DIM);
}

/* Runs forward + backward + SGD-momentum update for one mini-batch, returns
 * the sum of per-sample losses (caller divides by n_rows for the mean). */
float train_step(const float *x, const unsigned char *y, int n_rows,
				  Params &p, Grads &g, Velocity &v, Scratch &s, float *host_loss_buf)
{
	forward(x, n_rows, p, s);
	softmax_cross_entropy(s.Z2, y, s.probs, s.loss, n_rows, OUTPUT_DIM);

	softmax_cross_entropy_backward(s.probs, y, s.dZ2, n_rows, OUTPUT_DIM);

	/* dW2 = A1^T @ dZ2 : [HIDDEN, OUTPUT] */
	matmul(s.Z1, s.dZ2, g.dW2, HIDDEN_DIM, n_rows, OUTPUT_DIM, true, false);
	bias_grad(s.dZ2, g.db2, n_rows, OUTPUT_DIM);

	/* dA1 = dZ2 @ W2^T : [n_rows, HIDDEN] */
	matmul(s.dZ2, p.W2, s.dA1, n_rows, OUTPUT_DIM, HIDDEN_DIM, false, true);
	relu_backward(s.Z1, s.dA1, s.dZ1, n_rows, HIDDEN_DIM);

	/* dW1 = X^T @ dZ1 : [INPUT, HIDDEN] */
	matmul(x, s.dZ1, g.dW1, INPUT_DIM, n_rows, HIDDEN_DIM, true, false);
	bias_grad(s.dZ1, g.db1, n_rows, HIDDEN_DIM);

	sgd_momentum_update(p.W1, g.dW1, v.vW1, INPUT_DIM * HIDDEN_DIM, LEARNING_RATE, MOMENTUM);
	sgd_momentum_update(p.b1, g.db1, v.vb1, HIDDEN_DIM, LEARNING_RATE, MOMENTUM);
	sgd_momentum_update(p.W2, g.dW2, v.vW2, HIDDEN_DIM * OUTPUT_DIM, LEARNING_RATE, MOMENTUM);
	sgd_momentum_update(p.b2, g.db2, v.vb2, OUTPUT_DIM, LEARNING_RATE, MOMENTUM);

	CUDA_CHECK(cudaMemcpy(host_loss_buf, s.loss, n_rows * sizeof(float), cudaMemcpyDeviceToHost));
	float sum = 0.0f;
	for (int i = 0; i < n_rows; ++i)
	{
		sum += host_loss_buf[i];
	}
	return sum;
}

/* Evaluates accuracy over the full dataset pointed to by d_x/d_y (already on
 * device, contiguous, normalized), processing it in BATCH_SIZE chunks. */
float evaluate(const float *d_x, const unsigned char *d_y, int n_total, const Params &p, Scratch &s)
{
	int *d_correct;
	CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));
	CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));

	for (int offset = 0; offset < n_total; offset += BATCH_SIZE)
	{
		int cur_bs = n_total - offset < BATCH_SIZE ? n_total - offset : BATCH_SIZE;
		const float *x_ptr = d_x + (size_t)offset * INPUT_DIM;
		const unsigned char *y_ptr = d_y + offset;

		forward(x_ptr, cur_bs, p, s);
		count_correct(s.Z2, y_ptr, d_correct, cur_bs, OUTPUT_DIM);
	}

	int host_correct = 0;
	CUDA_CHECK(cudaMemcpy(&host_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_correct));

	return (float)host_correct / (float)n_total;
}

} /* namespace */

int main(int argc, char *argv[])
{
	int epochs = DEFAULT_EPOCHS;
	if (argc > 1)
	{
		epochs = atoi(argv[1]);
		if (epochs <= 0)
		{
			fprintf(stderr, "usage: %s [epochs]\n", argv[0]);
			return EXIT_FAILURE;
		}
	}

	srand(SEED);

	printf("Loading MNIST...\n");
	MnistDataset train_ds, test_ds;
	mnist_load(TRAIN_IMAGES, TRAIN_LABELS, &train_ds);
	mnist_load(TEST_IMAGES, TEST_LABELS, &test_ds);
	printf("  train: %d samples (%dx%d)\n", train_ds.num_samples, train_ds.rows, train_ds.cols);
	printf("  test:  %d samples (%dx%d)\n", test_ds.num_samples, test_ds.rows, test_ds.cols);

	int n_train = train_ds.num_samples;
	int n_test = test_ds.num_samples;

	/* Normalize pixels to [0,1] floats on the host, once. */
	float *h_train_x = (float *)malloc((size_t)n_train * INPUT_DIM * sizeof(float));
	float *h_test_x = (float *)malloc((size_t)n_test * INPUT_DIM * sizeof(float));
	for (size_t i = 0; i < (size_t)n_train * INPUT_DIM; ++i)
	{
		h_train_x[i] = train_ds.pixels[i] / 255.0f;
	}
	for (size_t i = 0; i < (size_t)n_test * INPUT_DIM; ++i)
	{
		h_test_x[i] = test_ds.pixels[i] / 255.0f;
	}

	float *d_train_x = device_alloc_f((size_t)n_train * INPUT_DIM);
	float *d_test_x = device_alloc_f((size_t)n_test * INPUT_DIM);
	CUDA_CHECK(cudaMemcpy(d_train_x, h_train_x, (size_t)n_train * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_test_x, h_test_x, (size_t)n_test * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
	free(h_train_x);
	free(h_test_x);

	unsigned char *d_train_y, *d_test_y;
	CUDA_CHECK(cudaMalloc(&d_train_y, n_train));
	CUDA_CHECK(cudaMalloc(&d_test_y, n_test));
	CUDA_CHECK(cudaMemcpy(d_train_y, train_ds.labels, n_train, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_test_y, test_ds.labels, n_test, cudaMemcpyHostToDevice));

	mnist_free(&train_ds);
	mnist_free(&test_ds);

	/* Parameters: He init for both layers (ReLU feeds layer 1's output; layer 2
	 * has no activation, but the same scaling is a reasonable default). */
	Params p{};
	upload_he(&p.W1, INPUT_DIM, INPUT_DIM * HIDDEN_DIM);
	p.b1 = device_zeros_f(HIDDEN_DIM);
	upload_he(&p.W2, HIDDEN_DIM, HIDDEN_DIM * OUTPUT_DIM);
	p.b2 = device_zeros_f(OUTPUT_DIM);

	Grads g{};
	g.dW1 = device_alloc_f(INPUT_DIM * HIDDEN_DIM);
	g.db1 = device_alloc_f(HIDDEN_DIM);
	g.dW2 = device_alloc_f(HIDDEN_DIM * OUTPUT_DIM);
	g.db2 = device_alloc_f(OUTPUT_DIM);

	Velocity v{};
	v.vW1 = device_zeros_f(INPUT_DIM * HIDDEN_DIM);
	v.vb1 = device_zeros_f(HIDDEN_DIM);
	v.vW2 = device_zeros_f(HIDDEN_DIM * OUTPUT_DIM);
	v.vb2 = device_zeros_f(OUTPUT_DIM);

	Scratch s{};
	s.batch_x = device_alloc_f((size_t)BATCH_SIZE * INPUT_DIM);
	CUDA_CHECK(cudaMalloc(&s.batch_y, BATCH_SIZE));
	CUDA_CHECK(cudaMalloc(&s.indices, BATCH_SIZE * sizeof(int)));
	s.Z1 = device_alloc_f((size_t)BATCH_SIZE * HIDDEN_DIM);
	s.Z2 = device_alloc_f((size_t)BATCH_SIZE * OUTPUT_DIM);
	s.probs = device_alloc_f((size_t)BATCH_SIZE * OUTPUT_DIM);
	s.loss = device_alloc_f(BATCH_SIZE);
	s.dZ2 = device_alloc_f((size_t)BATCH_SIZE * OUTPUT_DIM);
	s.dA1 = device_alloc_f((size_t)BATCH_SIZE * HIDDEN_DIM);
	s.dZ1 = device_alloc_f((size_t)BATCH_SIZE * HIDDEN_DIM);

	int *h_shuffle = (int *)malloc(n_train * sizeof(int));
	for (int i = 0; i < n_train; ++i)
	{
		h_shuffle[i] = i;
	}
	float *h_loss_buf = (float *)malloc(BATCH_SIZE * sizeof(float));

	printf("Training: %d epochs, batch_size=%d, lr=%g, momentum=%g, hidden=%d\n",
		   epochs, BATCH_SIZE, LEARNING_RATE, MOMENTUM, HIDDEN_DIM);

	int num_batches = n_train / BATCH_SIZE;

	for (int epoch = 1; epoch <= epochs; ++epoch)
	{
		clock_t epoch_start = clock();
		fisher_yates_shuffle(h_shuffle, n_train);

		double epoch_loss = 0.0;
		int samples_seen = 0;

		for (int b = 0; b < num_batches; ++b)
		{
			int cur_bs = BATCH_SIZE;
			CUDA_CHECK(cudaMemcpy(s.indices, h_shuffle + (size_t)b * BATCH_SIZE,
								   cur_bs * sizeof(int), cudaMemcpyHostToDevice));

			gather_rows(d_train_x, s.indices, s.batch_x, cur_bs, INPUT_DIM);
			gather_labels(d_train_y, s.indices, s.batch_y, cur_bs);

			epoch_loss += train_step(s.batch_x, s.batch_y, cur_bs, p, g, v, s, h_loss_buf);
			samples_seen += cur_bs;
		}

		CUDA_CHECK(cudaDeviceSynchronize());
		float test_acc = evaluate(d_test_x, d_test_y, n_test, p, s);
		double elapsed = (double)(clock() - epoch_start) / CLOCKS_PER_SEC;

		printf("Epoch %2d/%d | train_loss=%.4f | test_acc=%.2f%% | %.1fs\n",
			   epoch, epochs, epoch_loss / samples_seen, test_acc * 100.0f, elapsed);
	}

	free(h_shuffle);
	free(h_loss_buf);

	CUDA_CHECK(cudaFree(d_train_x));
	CUDA_CHECK(cudaFree(d_test_x));
	CUDA_CHECK(cudaFree(d_train_y));
	CUDA_CHECK(cudaFree(d_test_y));

	CUDA_CHECK(cudaFree(p.W1));
	CUDA_CHECK(cudaFree(p.b1));
	CUDA_CHECK(cudaFree(p.W2));
	CUDA_CHECK(cudaFree(p.b2));
	CUDA_CHECK(cudaFree(g.dW1));
	CUDA_CHECK(cudaFree(g.db1));
	CUDA_CHECK(cudaFree(g.dW2));
	CUDA_CHECK(cudaFree(g.db2));
	CUDA_CHECK(cudaFree(v.vW1));
	CUDA_CHECK(cudaFree(v.vb1));
	CUDA_CHECK(cudaFree(v.vW2));
	CUDA_CHECK(cudaFree(v.vb2));

	CUDA_CHECK(cudaFree(s.batch_x));
	CUDA_CHECK(cudaFree(s.batch_y));
	CUDA_CHECK(cudaFree(s.indices));
	CUDA_CHECK(cudaFree(s.Z1));
	CUDA_CHECK(cudaFree(s.Z2));
	CUDA_CHECK(cudaFree(s.probs));
	CUDA_CHECK(cudaFree(s.loss));
	CUDA_CHECK(cudaFree(s.dZ2));
	CUDA_CHECK(cudaFree(s.dA1));
	CUDA_CHECK(cudaFree(s.dZ1));

	return 0;
}
