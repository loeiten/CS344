#include <device_launch_parameters.h>  // for blockIdx, threadIdx

#include <cmath>      // for fabs, max
#include <cstddef>    // for size_t
#include <iostream>   // for cout, endl
#include <random>     // for random_device, mt19937, uniform_real_distribution
#include <sstream>    // for stringstream
#include <stdexcept>  // for runtime_error

// NOTE: This is automatically added by nvcc
#include "cuda_runtime.h"  // for cudaFree, cudaMalloc, cudaMe...

#define CheckCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__ void ReduceWoSharedMemory(double* d_out, double* d_in) {
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

__global__ void ReduceWSharedMemory(double* d_out, const double* d_in) {
  // The shared data is allocated in the kernel call: 3rd arg of <<<b, t,
  // shmem>>>
  extern __shared__ double shared_data[];

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

double GenerateRandomNumber() {
  // NOTE: The static variables will only be called once
  static std::random_device rd;
  static std::mt19937 gen(rd());
  // We need should keep the numbers around the same order of
  // magnitude in order to prevent precision loss
  static std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return dis(gen);
}

bool IsClose(double a, double b, double rel_tol = 1.0e-5,
             double abs_tol = 1.0e-8) {
  // Check if absolute difference is within absolute tolerance
  if (std::fabs(a - b) <= abs_tol) {
    return true;
  }

  // Check if relative difference is within relative tolerance
  if (std::fabs(a - b) <= rel_tol * std::max(std::fabs(a), std::fabs(b))) {
    return true;
  }

  return false;
}

int main() {
  constexpr std::size_t kThreads = 1024;
  constexpr std::size_t kBlocks = 1024;
  constexpr std::size_t kArraySize = kThreads * kBlocks;
  constexpr std::size_t kArrayBytes = kArraySize * sizeof(double);

  // Generate the input array on the host
  // NOTE: We use double in order to avoid loss of precision
  double h_in[kArraySize];
  double expected_result = 0.0;
  for (std::size_t i = 0; i < kArraySize; ++i) {
    h_in[i] = GenerateRandomNumber();
    expected_result += h_in[i];
  }
  double h_out;

  // Declare GPU memory pointers
  double* d_in;
  double* d_intermediate;
  double* d_out;

  // Allocate GPU memory
  CheckCudaErrors(cudaMalloc((void**)&d_in, kArrayBytes));
  CheckCudaErrors(cudaMalloc((void**)&d_intermediate, kArrayBytes));
  CheckCudaErrors(cudaMalloc((void**)&d_out, sizeof(double)));
  // Zero out the memory
  CheckCudaErrors(cudaMemset(d_in, 0, kArraySize));
  CheckCudaErrors(cudaMemset(d_intermediate, 0, kArraySize));
  CheckCudaErrors(cudaMemset(d_out, 0, 1));

  // Transfer the data to the GPU
  CheckCudaErrors(cudaMemcpy(d_in, h_in, kArrayBytes, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsed_time;

  CheckCudaErrors(cudaEventCreate(&start));
  CheckCudaErrors(cudaEventCreate(&stop));

  // Run the non-shared memory kernel
  std::cout << "Reducing without using shared memory..." << std::endl;
  CheckCudaErrors(cudaEventRecord(start));
  ReduceWoSharedMemory<<<kBlocks, kThreads>>>(d_intermediate, d_in);
  // In the final reduce, we only need one block
  ReduceWoSharedMemory<<<1, kThreads>>>(d_out, d_intermediate);
  CheckCudaErrors(cudaEventRecord(stop));
  CheckCudaErrors(cudaEventSynchronize(stop));
  CheckCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
  std::cout << "...it took " << elapsed_time << " ms" << std::endl;
  // Copy back the result array to the CPU
  CheckCudaErrors(
      cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));

  // Check the answer against the analytical result
  if (!IsClose(expected_result, h_out, 1e-2, 1e-1)) {
    std::stringstream ss;
    ss << "Expected " << expected_result << ", got " << h_out << std::endl;
    throw std::runtime_error(ss.str());
  }

  // Reset the values
  CheckCudaErrors(cudaMemset(d_intermediate, 0, kArraySize));
  CheckCudaErrors(cudaMemset(d_out, 0, 1));
  h_out = 0.0;
  elapsed_time = 0.0;

  // Run the shared memory kernel
  std::cout << "Reducing using shared memory" << std::endl;
  CheckCudaErrors(cudaEventRecord(start));
  ReduceWSharedMemory<<<kBlocks, kThreads, kThreads * sizeof(double)>>>(
      d_intermediate, d_in);
  // In the final reduce, we only need one block
  ReduceWSharedMemory<<<1, kThreads, kThreads * sizeof(double)>>>(
      d_out, d_intermediate);
  CheckCudaErrors(cudaEventRecord(stop));
  CheckCudaErrors(cudaEventSynchronize(stop));
  CheckCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
  std::cout << "...it took " << elapsed_time << " ms" << std::endl;
  // Copy back the result array to the CPU
  CheckCudaErrors(
      cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));

  // Check for errors
  if (!IsClose(expected_result, h_out, 1e-2, 1e-1)) {
    std::stringstream ss;
    ss << "Expected " << expected_result << ", got " << h_out << std::endl;
    throw std::runtime_error(ss.str());
  }

  // Free GPU memory allocation
  CheckCudaErrors(cudaEventDestroy(start));
  CheckCudaErrors(cudaEventDestroy(stop));
  CheckCudaErrors(cudaFree(d_in));
  CheckCudaErrors(cudaFree(d_out));
}
