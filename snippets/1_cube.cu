#include <stdio.h>

// NOTE: This is automatically added by nvcc
#include "cuda_runtime.h"  // for cudaFree, cudaMalloc, cudaMe...

#define ARRAY_SIZE (96)

// __global__ is a declaration specifier (a C language construct)
// This is the way that CUDA knows that it's a kernel as opposed to CPU code
// The pointers must be allocated on the device for this to work
__global__ void cube(float* d_out, float* d_in) {
  // threadIdx is a c struct with .x, .y and .z as members
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f * f;
}

int main() {
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // Generate the input array on the host
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];

  // Declare GPU memory pointers
  float* d_in;
  float* d_out;

  // Allocate GPU memory
  cudaMalloc((void**)&d_in, ARRAY_BYTES);
  cudaMalloc((void**)&d_out, ARRAY_BYTES);

  // Transfer the data to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // Launch the kernel on one block, using ARRAY_SIZE threads per block
  // This is called the launch operator
  // It's possible to have dim3(x,y,z) as a parameter in both `grid of blocks`
  // (first parameters) and for `threads per block` (second parameters)
  // The below is equivalent to dim3(1,1,1), dim3(ARRAY_SIZE,1,1)
  // One could also have a third parameter to the launch operator which
  // specifies shared memory per block in bytes
  // See
  // https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
  // for definitions on blocks and threads
  cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

  // Copy back the result array to the CPU
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // Print the resulting array
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }

  // Free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);
}
