#include "mnist_loader.h"

#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t read_u32_be(FILE *fp, const char *field_name, const char *path)
{
	uint32_t v;
	if (fread(&v, sizeof(v), 1, fp) != 1)
	{
		fprintf(stderr, "mnist_load: failed to read %s from %s\n", field_name, path);
		exit(EXIT_FAILURE);
	}
	return ntohl(v);
}

void mnist_load(const char *images_path, const char *labels_path, MnistDataset *out)
{
	FILE *img_fp = fopen(images_path, "rb");
	if (!img_fp)
	{
		fprintf(stderr, "mnist_load: could not open %s\n", images_path);
		exit(EXIT_FAILURE);
	}

	uint32_t magic = read_u32_be(img_fp, "image magic", images_path);
	if (magic != 2051)
	{
		fprintf(stderr, "mnist_load: bad image magic in %s (expected 2051, got %u)\n", images_path, magic);
		exit(EXIT_FAILURE);
	}
	uint32_t num_images = read_u32_be(img_fp, "image count", images_path);
	uint32_t rows = read_u32_be(img_fp, "rows", images_path);
	uint32_t cols = read_u32_be(img_fp, "cols", images_path);

	size_t image_bytes = (size_t)num_images * rows * cols;
	unsigned char *pixels = malloc(image_bytes);
	if (!pixels)
	{
		fprintf(stderr, "mnist_load: out of memory allocating %zu bytes for pixels\n", image_bytes);
		exit(EXIT_FAILURE);
	}
	if (fread(pixels, 1, image_bytes, img_fp) != image_bytes)
	{
		fprintf(stderr, "mnist_load: truncated image file %s\n", images_path);
		exit(EXIT_FAILURE);
	}
	fclose(img_fp);

	FILE *lbl_fp = fopen(labels_path, "rb");
	if (!lbl_fp)
	{
		fprintf(stderr, "mnist_load: could not open %s\n", labels_path);
		exit(EXIT_FAILURE);
	}

	magic = read_u32_be(lbl_fp, "label magic", labels_path);
	if (magic != 2049)
	{
		fprintf(stderr, "mnist_load: bad label magic in %s (expected 2049, got %u)\n", labels_path, magic);
		exit(EXIT_FAILURE);
	}
	uint32_t num_labels = read_u32_be(lbl_fp, "label count", labels_path);
	if (num_labels != num_images)
	{
		fprintf(stderr, "mnist_load: image/label count mismatch (%u vs %u)\n", num_images, num_labels);
		exit(EXIT_FAILURE);
	}

	unsigned char *labels = malloc(num_labels);
	if (!labels)
	{
		fprintf(stderr, "mnist_load: out of memory allocating labels\n");
		exit(EXIT_FAILURE);
	}
	if (fread(labels, 1, num_labels, lbl_fp) != num_labels)
	{
		fprintf(stderr, "mnist_load: truncated label file %s\n", labels_path);
		exit(EXIT_FAILURE);
	}
	fclose(lbl_fp);

	out->num_samples = (int)num_images;
	out->rows = (int)rows;
	out->cols = (int)cols;
	out->pixels = pixels;
	out->labels = labels;
}

void mnist_free(MnistDataset *ds)
{
	free(ds->pixels);
	free(ds->labels);
	ds->pixels = NULL;
	ds->labels = NULL;
}
