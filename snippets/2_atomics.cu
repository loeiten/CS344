#include <cuda_runtime.h>              // for cudaEventCreate, cudaEventDestroy
#include <device_launch_parameters.h>  // for blockDim, blockIdx, threadIdx
#include <driver_types.h>              // for CUevent_st, cudaEvent_t, cudaM...

#include <cstring>     // for strcmp
#include <filesystem>  // for operator<<, path
#include <iostream>    // for operator<<, basic_ostream, bas...
#include <string>      // for char_traits, stoi, allocator

#include "vector_types.h"  // for uint3, dim3

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

void PrintArray(const int* array, const int num_elements) {
  std::cout << "[";
  std::string delimeter = "";
  for (int i = 0; i < 10; ++i) {
    std::cout << delimeter << array[i];
    delimeter = " ,";
  }
  std::cout << ",..., " << array[num_elements - 1] << "]" << std::endl;
}

__global__ void IncrementNaive(int* g, const int num_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i = i % num_elements;
  g[i] = g[i] + 1;
}

__global__ void IncrementAtomic(int* g, const int num_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i = i % num_elements;
  // atomicAdd(&g[i], 1);
  g[i] = g[i] + 1;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage:" << std::filesystem::path(argv[0]).stem()
              << " naive/atomic threads elements" << std::endl;
    return -1;
  }
  constexpr int block_width = 1000;
  const int num_threads = std::stoi(argv[2]);
  const int num_elements = std::stoi(argv[3]);
  std::cout << num_threads << " total threads in " << num_threads / block_width
            << " writing to " << num_elements << " array elements" << std::endl;

  // Declare and allocate host memory
  int* h_array = new int[num_elements]{};
  const int array_bytes = num_elements * sizeof(int);

  // Declare, allocated and zero out GPU memory
  int* d_array;
  cudaMalloc(reinterpret_cast<void**>(&d_array), array_bytes);
  cudaMemset(reinterpret_cast<void*>(d_array), 0, array_bytes);

  // Launch kernel
  GpuTimer timer;
  timer.Start();
  if (std::strcmp(argv[1], "naive") == 0) {
    IncrementNaive<<<num_threads / block_width, block_width>>>(d_array,
                                                               num_elements);
  } else if (std::strcmp(argv[1], "atomic") == 0) {
    IncrementAtomic<<<num_threads / block_width, block_width>>>(d_array,
                                                                num_elements);
  } else {
    std::cout << "Usage:" << std::filesystem::path(argv[0]).stem()
              << " naive/atomic threads elements" << std::endl;
    std::cout << argv[1] << " not understood" << std::endl;

    delete[] h_array;
    cudaFree(d_array);
    return -1;
  }
  timer.Stop();

  // Copy back the array of sums from GPU and print
  cudaMemcpy(h_array, d_array, array_bytes, cudaMemcpyDeviceToHost);
  PrintArray(h_array, num_elements);
  std::cout << "Time elapsed: " << timer.Elapsed() << "ms" << std::endl;

  // Free host memory allocation
  delete[] h_array;

  // Free GPU memory allocation
  cudaFree(d_array);

  return 0;
}
