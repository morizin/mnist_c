NVCC ?= nvcc
CC ?= cc

# Covers Volta through Ada/Hopper. Trim this list to your card to speed up compiles.
ARCH := \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90

CFLAGS := -O2 -Wall -Iinclude
NVCCFLAGS := -O2 -std=c++14 -Iinclude $(ARCH)

BUILD_DIR := build
TARGET := $(BUILD_DIR)/mnist_train

.PHONY: all run clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/mnist_loader.o: src/mnist_loader.c include/mnist_loader.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/kernels.o: src/kernels.cu include/kernels.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cu include/kernels.cuh include/mnist_loader.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(BUILD_DIR)/mnist_loader.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/main.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
