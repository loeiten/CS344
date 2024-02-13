#include <device_launch_parameters.h>  // for blockIdx, threadIdx

#include <cstddef>  // for size_t
#include <iostream>
#include <sstream>
#include <stdexcept>

// NOTE: This is automatically added by nvcc
#include "cuda_runtime.h"  // for cudaFree, cudaMalloc, cudaMe...

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

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
  // The shared data is allocated in the kernel call: 3rd arg of <<<b, t, shmem>>>
  extern __shared__ float shared_data[];

  int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int local_idx = threadIdx.x;

  // Load the shared memory from the global memory
  shared_data[local_idx] = d_in[global_idx];
  // We must make sure that entire block is loaded
  __syncthreads();  

  // Now we will do the reduction in the shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_idx < s) {
      shared_data[local_idx] += shared_data[local_idx + s];
    }
    // We must make sure all adds at one stage are done
    __syncthreads();  
  }

  // Write to output array, only thread 0 writes
  if (local_idx == 0) {
    d_out[blockIdx.x] = shared_data[0];
  }
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
  float* d_intermediate;
  float* d_out;

  // Allocate GPU memory
  checkCudaErrors(cudaMalloc((void**)&d_in, array_bytes));
  checkCudaErrors(cudaMemset(d_in, 0, array_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_intermediate, array_bytes));
  checkCudaErrors(cudaMemset(d_intermediate, 0, array_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float)));
  checkCudaErrors(cudaMemset(d_out, 0, array_bytes));

  // Transfer the data to the GPU
  cudaMemcpy(d_in, h_in, array_bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsed_time;
  std::size_t expected_result = array_size * (array_size + 1) / 2;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Run the non-shared memory kernel
  std::cout << "Reducing without using shared memory..." << std::endl;
  cudaEventRecord(start);
  reduce_wo_shared_memory<<<blocks, threads>>>(d_intermediate, d_in);
  // In the final reduce, we only need one block
  reduce_wo_shared_memory<<<1, threads>>>(d_out, d_intermediate);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  cudaMemcpy(h_out, d_out, array_bytes, cudaMemcpyDeviceToHost);
  std::cout << "...it took " << elapsed_time << " ms" << std::endl;
  // Copy back the result array to the CPU
  std::cout << "...it took " << elapsed_time << " ms" << std::endl;

  // Check the answer against the analytical result
  if (expected_result != static_cast<std::size_t>(*h_out)) {
    std::stringstream ss;
    ss << "Expected " << expected_result << ", got " << *h_out << std::endl;
    throw std::runtime_error(ss.str());
  }

  // Reset the values
  checkCudaErrors(cudaMemset(d_intermediate, 0, array_bytes));
  checkCudaErrors(cudaMemset(d_out, 0, array_bytes));
  *h_out = 0.0;
  elapsed_time = 0.0;

  // Run the shared memory kernel
  std::cout << "Reducing using shared memory" << std::endl;
  cudaEventRecord(start);
  reduce_w_shared_memory<<<blocks, threads, threads * sizeof(float)>>>(
      d_intermediate, d_in);
  // In the final reduce, we only need one block
  reduce_w_shared_memory<<<1, threads, threads * sizeof(float)>>>(
      d_out, d_intermediate);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "...it took " << elapsed_time << " ms" << std::endl;
  // Copy back the result array to the CPU
  cudaMemcpy(h_out, d_out, array_bytes, cudaMemcpyDeviceToHost);

  // Check for errors
  if (expected_result != static_cast<std::size_t>(*h_out)) {
    std::stringstream ss;
    ss << "Expected " << expected_result << ", got " << *h_out << std::endl;
    throw std::runtime_error(ss.str());
  }

  // Free GPU memory allocation
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
}
