#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <stdbool.h>
#include <math.h>

typedef struct
{
	double *weight;
	double *grad;
	int *shape;
	int size;
	int dim;
	bool requires_grad;
} Tensor;

struct Layer;

typedef struct Layer
{
	Tensor *weights;
	Tensor *biases;
	bool requires_grad;
	struct Layer *next_layer;
	struct Layer *prev_layer;
} Layer;

void init_tensor(Tensor *tensor, int *shape, int dim);

void init_layer(Layer *layer, int *weight_shape, int weight_dim, int *bias_shape, int bias_dim)
{
	layer->weights = malloc(sizeof(Tensor));
	layer->biases = malloc(sizeof(Tensor));
	init_tensor(layer->weights, weight_shape, weight_dim);
	init_tensor(layer->biases, bias_shape, bias_dim);
	layer->requires_grad = true;
	layer->next_layer = NULL;
	layer->prev_layer = NULL;
}

double cross_entropy_loss(Tensor *predictions, double *labels)
{
	double loss = 0.0;
	for (int i = 0; i < predictions->size; i++)
	{
		loss -= labels[i] * log(predictions->weight[i] + 1e-8);
		printf("Label: %f, Prediction: %f, Loss Contribution: %f\n", labels[i], predictions->weight[i], -labels[i] * log(predictions->weight[i] + 1e-8));
	}
	return loss;
}

void one_hot_encode(unsigned char *label, Tensor *encoded)
{
	for (int j = 0; j < encoded->shape[encoded->dim - 2]; j++)
	{
		for (int i = 0; i < encoded->shape[encoded->dim - 1]; i++)
		{
			encoded->weight[j * encoded->shape[encoded->dim - 1] + i] = (i == label[j]) ? 1.0 : 0.0;
		}
	}
}

void init_tensor(Tensor *tensor, int *shape, int dim)
{
	tensor->dim = dim;
	tensor->shape = malloc(dim * sizeof(int));
	tensor->size = 1;
	tensor->requires_grad = true;
	for (int i = 0; i < dim; i++)
	{
		tensor->shape[i] = shape[i];
		tensor->size *= shape[i];
	}
	tensor->weight = malloc(tensor->size * sizeof(double));
	tensor->grad = malloc(tensor->size * sizeof(double));
}

void matmul(Tensor *A, Tensor *B, Tensor *C)
{
	if (A->dim != 2 || B->dim != 2 || C->dim != 2)
	{
		fprintf(stderr, "matmul only supports 2D tensors\n");
		exit(EXIT_FAILURE);
	}

	if (A->shape[A->dim - 1] != B->shape[B->dim - 2])
	{
		perror("Matrix dimensions do not match for multiplication");
		exit(EXIT_FAILURE);
	}
	if (C->shape[C->dim - 2] != A->shape[A->dim - 2] || C->shape[C->dim - 1] != B->shape[B->dim - 1])
	{
		perror("Output matrix dimensions do not match the expected size");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < C->shape[C->dim - 2]; i++)
	{
		for (int j = 0; j < C->shape[C->dim - 1]; j++)
		{
			double sum = 0.0;
			for (int l = 0; l < A->shape[A->dim - 1]; l++)
			{
				sum += A->weight[i * A->shape[A->dim - 1] + l] * B->weight[j + B->shape[B->dim - 1] * l];
			}
			C->weight[i * C->shape[C->dim - 1] + j] = sum;
		}
	}
}

void softmax(Tensor *input)
{
	double max = input->weight[0];
	for (int i = 1; i < input->size; i++)
	{
		if (input->weight[i] > max)
		{
			max = input->weight[i];
		}
	}

	double sum = 0.0;
	for (int i = 0; i < input->size; i++)
	{
		input->weight[i] = exp(input->weight[i] - max);
		sum += input->weight[i];
	}
	for (int i = 0; i < input->size; i++)
	{
		input->weight[i] /= (sum + 1e-8);
	}
}

double relu(double value);

void download_and_extract(
	char *images_arr,
	char *labels_arr,
	char *image_file,
	char *label_file,
	int *total)
{
	FILE *images_ptr, *labels_ptr;
	uint32_t rows, cols, magic, size;

	images_ptr = fopen(image_file, "rb");
	if (images_ptr == NULL)
	{
		fprintf(stderr, "Error: Could not open the file\n");
		exit(EXIT_FAILURE);
	}

	fread(&magic, 4, 1, images_ptr);

	magic = ntohl(magic);
	if (magic != 2051)
	{
		fprintf(stderr, "Magic number mismatch, expected 2051, got %u\n", magic);
		exit(EXIT_FAILURE);
	}
	fread(&size, 4, 1, images_ptr);
	fread(&rows, 4, 1, images_ptr);
	fread(&cols, 4, 1, images_ptr);

	size = ntohl(size);
	*total = size;
	rows = ntohl(rows);
	cols = ntohl(cols);

	fread(images_arr, 1, size * rows * cols, images_ptr);

	labels_ptr = fopen(label_file, "rb");
	if (labels_ptr == NULL)
	{
		fprintf(stderr, "Error: Could not open the file\n");
		exit(EXIT_FAILURE);
	}

	fread(&magic, 4, 1, labels_ptr);
	if (ntohl(magic) != 2049)
	{
		fprintf(stderr, "Magic number mismatch, expected 2049, got %u\n", ntohl(magic));
		exit(EXIT_FAILURE);
	}
	fread(&size, 4, 1, labels_ptr);
	size = ntohl(size);
	if (size != *total)
	{
		fprintf(stderr, "Size mismatch between images and labels, expected %u, got %u\n", *total, size);
		exit(EXIT_FAILURE);
	}
	fread(labels_arr, 1, size, labels_ptr);

	fclose(images_ptr);
	fclose(labels_ptr);
}

void print_matrix(Tensor *matrix)
{
	for (int i = 0; i < matrix->shape[0]; i++)
	{
		for (int j = 0; j < matrix->shape[1]; j++)
		{
			printf("%f ", matrix->weight[i * matrix->shape[1] + j]);
		}
		printf("\n");
	}
}

void init_weights(Tensor *weights)
{
	for (int i = 0; i < weights->size; i++)
	{
		weights->weight[i] = (double)rand() / RAND_MAX;
		weights->grad[i] = 0.0;
	}
}

void layer_norm(Tensor *input, int size)
{
	double mean = 0.0;
	for (int i = 0; i < size; i++)
	{
		mean += input->weight[i];
	}
	mean /= size;

	double variance = 0.0;
	for (int i = 0; i < size; i++)
	{
		variance += (input->weight[i] - mean) * (input->weight[i] - mean);
	}
	variance /= size;
	double stddev = sqrt(variance + 1e-8);
	for (int i = 0; i < size; i++)
	{
		input->weight[i] = (input->weight[i] - mean) / stddev;
	}
}

void forward_pass(
	Tensor *input,	// 784
	Tensor *output, // 10  ← separate buffer
	Tensor *weights,
	Tensor *biases)
{
	matmul(weights, input, output);
	layer_norm(output, output->shape[0]);
	for (int i = 0; i < output->shape[0]; i++)
	{
		output->weight[i] = relu(output->weight[i] + biases->weight[i]);
	}
}

void normalize(unsigned char *data, Tensor *normalized_data)
{
	normalized_data->requires_grad = false;
	for (int i = 0; i < normalized_data->size; i++)
	{
		normalized_data->weight[i] = (double)data[i] / 255.0;
		normalized_data->grad[i] = 0.0;
	}
}

double relu(double value)
{
	if (value < 0)
	{
		value = 0;
	}
	return value;
}

int main(int argc, char *argv[])
{
	srand(343524);
	uint32_t total = 0;
	unsigned char *train_images = malloc(70000 * 28 * 28);
	unsigned char *train_labels = malloc(70000);
	double *result_matrix = malloc(28 * 28 * sizeof(double));

	char *train_images_files = "data/train-images-idx3-ubyte/train-images-idx3-ubyte";
	char *train_labels_files = "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

	printf("Loading MNIST dataset...\n");

	Tensor *input_tensor = malloc(sizeof(Tensor));
	Tensor *labels = malloc(sizeof(Tensor));
	init_tensor(labels, (int[]){total, 10}, 2);
	labels->requires_grad = false;

	one_hot_encode(train_labels, labels);
	init_tensor(input_tensor, (int[]){28 * 28, 1}, 2);

	Tensor *weights1_tensor = malloc(sizeof(Tensor));
	Tensor *weights2_tensor = malloc(sizeof(Tensor));

	Tensor *biases1_tensor = malloc(sizeof(Tensor));
	Tensor *biases2_tensor = malloc(sizeof(Tensor));
	init_tensor(weights1_tensor, (int[]){512, 28 * 28}, 2);
	init_tensor(weights2_tensor, (int[]){10, 512}, 2);
	init_tensor(biases1_tensor, (int[]){512, 1}, 2);
	init_tensor(biases2_tensor, (int[]){10, 1}, 2);

	Tensor *hidden = malloc(sizeof(Tensor));
	Tensor *output = malloc(sizeof(Tensor));
	init_tensor(hidden, (int[]){512, 1}, 2);
	init_tensor(output, (int[]){10, 1}, 2);

	init_weights(weights1_tensor);
	init_weights(weights2_tensor);
	init_weights(biases1_tensor);
	init_weights(biases2_tensor);

	download_and_extract(train_images, train_labels, train_images_files, train_labels_files, &total);

	printf("Total images: %d\n", total);

	if (train_images == NULL || train_labels == NULL)
	{
		fprintf(stderr, "Error: Memory allocation failed\n");
		return 1;
	}

	double loss = 0.0;
	for (int i = 0; i < 10; i++)
	{
		normalize(&train_images[i * 28 * 28], input_tensor);
		forward_pass(input_tensor, hidden, weights1_tensor, biases1_tensor);
		forward_pass(hidden, output, weights2_tensor, biases2_tensor);
		softmax(output);
		print_matrix(output);
		loss = cross_entropy_loss(output, labels->weight + i * labels->shape[1]);
		printf("Sample %d: Loss = %f\n", i, loss);
	}

	free(train_images);
	free(train_labels);
	free(input_tensor);
	free(weights1_tensor);
	free(weights2_tensor);
	free(biases1_tensor);
	free(biases2_tensor);
	free(result_matrix);
	free(output);
	free(hidden);
	return 0;
}
