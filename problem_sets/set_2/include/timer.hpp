#ifndef PROBLEM_SETS_SET_2_INCLUDE_TIMER_HPP_
#define PROBLEM_SETS_SET_2_INCLUDE_TIMER_HPP_

#include <cuda_runtime.h>  // for cudaEventCreate, cudaEventDestroy, cudaEve...
#include <driver_types.h>  // for CUevent_st, cudaEvent_t

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

#endif  // PROBLEM_SETS_SET_2_INCLUDE_TIMER_HPP_
