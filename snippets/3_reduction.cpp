#include <device_launch_parameters.h>  // for blockIdx, threadIdx

#include <cstddef>  // for size_t

// NOTE: This is automatically added by nvcc
#include "cuda_runtime.h"  // for cudaFree, cudaMalloc, cudaMe...

__global__ void reduce_wo_shared_memory(float* d_out, float* d_in) {
  // Let's illustrate this kernel by setting blocks=4 and threads=4 on an array
  // with 16 elements:
  // The setup will look like this:
  //               Block 0 | Block 1 | Block 2   | Block 3
  // ----------------------------------------------------------
  // Iteration 1:  s = 2. Note that only 0 and 1 is accessed
  //               0 1 2 3 | 4   5  6 7 | 8  9  10 11 | 12 13 14 15
  //               2<--' : | 10<----' : | 18<---'  :  | 26<---'  :
  //                 4<--' |    12<---' |    20<---'  |     28<--'
  // Synch threads
  // ----------------------------------------------------------
  // Iteration 2:  s = 1. Note that only 0 is accessed
  //               2 4 2 3 | 10 12  6 7 | 18 20 10 11 | 26 28 14 15
  //               6<'     | 22<'       | 38<-'       | 54<-'
  // Synch threads
  // ----------------------------------------------------------
  // Result:       6 4 2 3 | 22 12  6 7 | 38 20 10 11 | 54 28 14 15
  //
  // The first

  std::size_t global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  std::size_t local_idx = threadIdx.x;

  // NOTE: blockDim - is the total block size
  //       If we start with 1024, then this will start at 512
  //       We then right bitshift (move all bits one place to the right)
  //       this is effectively an integer division by 2
  //       In other words, for each iteration s is halved until it reaches 0
  for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    // If local_idx is larger or equal to s, then we will get out of bounds in
    // the first iteration
    if (local_idx < s) {
      // If we are in the active half of the reduce, we will add
      d_in[global_idx] += d_in[global_idx + s];
    }
    __syncthreads();
  }

  // Write to output array, only for the first local index
  if (local_idx == 0) {
    d_out[blockIdx.x] = d_in[global_idx];
  }
}

__global__ void reduce_w_shared_memory(float* d_out, float* d_in) {
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f * f * f;
}

int main() {
  constexpr std::size_t threads = 1024;
  constexpr std::size_t blocks = 1024;
  constexpr std::size_t array_size = threads * blocks;
  constexpr std::size_t array_bytes = array_size * sizeof(float);

  // Generate the input array on the host
  // NOTE: It would be faster to do this on the accelerator
  float h_in[array_size];
  for (std::size_t i = 0; i < array_size; ++i) {
    h_in[i] = float(i);
  }
  float h_out[array_size];

  // Declare GPU memory pointers
  float* d_in;
  float* d_out;

  // Allocate GPU memory
  cudaMalloc((void**)&d_in, array_bytes);
  cudaMalloc((void**)&d_out, array_bytes);

  // Transfer the data to the GPU
  cudaMemcpy(d_in, h_in, array_bytes, cudaMemcpyHostToDevice);

  reduce_wo_shared_memory<<<blocks, threads>>>(d_out, d_in);
  reduce_w_shared_memory<<<blocks, threads, >>>(d_out, d_in);

  // Copy back the result array to the CPU
  cudaMemcpy(h_out, d_out, array_bytes, cudaMemcpyDeviceToHost);

  // Free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);
}
