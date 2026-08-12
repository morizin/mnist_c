#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

/* Loads MNIST IDX3 (images) / IDX1 (labels) file pairs into host memory. */

typedef struct
{
	int num_samples;
	int rows;
	int cols;
	unsigned char *pixels; /* num_samples * rows * cols, row-major, 0-255 */
	unsigned char *labels; /* num_samples, values 0-9 */
} MnistDataset;

/* Reads both files fully into freshly malloc'd buffers. Exits the process
 * on any I/O error, magic-number mismatch, or image/label count mismatch. */
void mnist_load(const char *images_path, const char *labels_path, MnistDataset *out);

void mnist_free(MnistDataset *ds);

#endif
