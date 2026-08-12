# mnist_c — MNIST training in hand-written CUDA

A from-scratch 2-layer MLP (784 → 256 ReLU → 10) trained on MNIST with
mini-batch SGD + momentum. Every kernel — tiled matmul, bias+ReLU fusion,
softmax/cross-entropy (forward and backward), the batch-shuffle gather, and
the SGD-momentum update — is hand-written CUDA C++; no cuBLAS/cuDNN.

> Built and reviewed on a machine with no NVIDIA GPU / CUDA toolkit, so it
> has not been compiled or run locally. Build and run it on a CUDA-capable
> machine before trusting the numbers; see "Known risk areas" below for
> what to check first if something's off.

## Layout

```
include/mnist_loader.h   IDX file format loader (declarations)
include/kernels.cuh      CUDA kernel launcher declarations
src/mnist_loader.c       IDX loader implementation (plain C, host-only)
src/kernels.cu           Kernel implementations + host launch wrappers
src/main.cu              Training loop: data upload, epochs, eval
Makefile                 nvcc/cc build
data/                    MNIST IDX files (train/test images + labels)
```

## Build

Requires the CUDA toolkit (`nvcc`) and an NVIDIA GPU.

```sh
make            # builds build/mnist_train
make run        # builds and runs with default settings
./build/mnist_train [epochs]   # override epoch count (default 15)
```

`Makefile`'s `ARCH` variable targets sm_70 through sm_90 (Volta–Hopper);
trim it to your card to speed up compilation.

## Data

`data/*.idx3-ubyte` / `data/*.idx1-ubyte` are already present. If you need
to re-fetch them, `scripts/download.sh` pulls the original Kaggle zip.

## Design notes

- Row-major float32 throughout; labels are single bytes (0-9) on the device.
- One generic tiled matmul kernel (`matmul`, 16x16 shared-memory tiles)
  handles all forward/backward matrix products via `transA`/`transB` flags,
  rather than separate kernels per op.
- Softmax + cross-entropy forward and backward are fused (backward writes
  `probs - one_hot(label)` directly), avoiding a separate Jacobian.
- Each epoch reshuffles a host-side index array (Fisher-Yates) and uploads
  just the batch's indices; a `gather_rows`/`gather_labels` kernel pulls
  the actual batch data from the full training set that lives on the GPU
  for the whole run — no per-epoch re-upload of the dataset.
- Plain SGD with momentum (`v = momentum*v - lr*grad; w += v`), no Adam/etc.

## Known risk areas (check these first if training misbehaves)

Written without the ability to compile or run against real hardware, so
these are the spots most worth double-checking against actual output:

- Matmul transpose-flag index math in `src/kernels.cu` (`matmul_kernel`)
  and the corresponding call sites in `src/main.cu::train_step` — the
  transA/transB argument for each of the 5 matmul calls encodes the
  forward/backward shapes by hand.
- Numerical stability of `softmax_ce_kernel` for degenerate inputs.
- Off-by-one/remainder handling: batches drop `n_train % BATCH_SIZE`
  samples per epoch (shuffled away, not systematically excluded); the
  final partial batch is *not* dropped during evaluation.
