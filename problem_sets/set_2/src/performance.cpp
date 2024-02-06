#include "../include/performance.hpp"

#include <cuda_runtime.h>  // for cudaFree, cudaDeviceSynchro...
#include <stdio.h>         // for printf
#include <stdlib.h>        // for atof, exit

#include <cstddef>     // for size_t
#include <filesystem>  // for absolute, path, create_dire...
#include <iostream>    // for operator<<, endl, basic_ost...
#include <string>      // for allocator, operator+, string
#include <unordered_map>

#include "../include/compare.hpp"         // for compareImages
#include "../include/image.hpp"           // for Image
#include "../include/reference_calc.hpp"  // for referenceCalculation
#include "../include/timer.hpp"           // for GpuTimer
#include "../include/utils.hpp"           // for check, checkCudaErrors
#include "driver_types.h"                 // for cudaMemcpyDeviceToHost
#include "vector_types.h"                 // for uchar4

void PrintPerformance(std::string kernel_name, std::size_t bytes_processed,
                      float time_in_ms) {
  // Calculate performance compared to ideal
  // We can see how well we are performing by checking achieved
  // throughput/bandwidth
  //
  // WARNING: The cinque_terre image is likely too small to measure the
  //          performance as we're also measuring the kernel launch overheads
  //          etc.
  //          To get a more realistic number either pick a larger image, or make
  //          the launch so that it loops over more data

  // Memory bandwidth for the device
  // From
  // https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
  std::unordered_map<std::string, float> device_bw{
      {"NVIDIA A100-PG509-200", 1.555e9}};
  int devCount;
  bool calculate_performance = true;
  cudaGetDeviceCount(&devCount);
  std::string prev_device_name = "";
  std::string device_name;
  for (int i = 0; i < devCount; ++i) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    device_name = props.name;
    if (prev_device_name != "") {
      if (prev_device_name != device_name) {
        std::cerr << "Warning: There is a mix of devices, could not calculate "
                     "the performance number"
                  << std::endl;
        calculate_performance = false;
        break;
      }
    }
    prev_device_name = device_name;
  }
  if (calculate_performance) {
    if (device_bw.find(device_name) == device_bw.end()) {
      std::cerr << "Warning: Do not know the memory bandwidth of "
                << device_name << ", please look it up, and add it to device_bw"
                << std::endl;
    } else {
      float performance = 100 * (bytes_processed / (time_in_ms / 1.0e3)) /
                          (device_bw[device_name]);
      std::cout << "Processed the kernel" << kernel_name << " which processed "
                << bytes_processed << " bytes in " << time_in_ms << " ms"
                << std::endl;
      std::cout << "As the max bandwidth of the system is"
                << device_bw[device_name] << "bytes/sec for " << device_name
                << ", this is " << performance << "% of theoretical max"
                << std::endl;
    }
  }
}
