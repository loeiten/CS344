#include "../include/performance.hpp"

#include <cuda_runtime.h>  // for cudaGetDeviceCount, cudaGetDeviceProperties

#include <cstddef>   // for size_t
#include <iostream>  // for operator<<, basic_ostream, endl, basic_ost...
#include <string>    // for string, basic_string, operator<<, hash

#include "driver_types.h"  // for cudaDeviceProp

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
  int devCount;
  bool calculate_performance = true;
  cudaGetDeviceCount(&devCount);
  std::string prev_device_name = "";
  std::string device_name;
  cudaDeviceProp props;
  for (int i = 0; i < devCount; ++i) {
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
    // NOTE: Multiply with 2 for bidirectional, divide by 8 to get bytes from
    // bits
    float max_bidirectional_bandwidth =
        static_cast<float>(props.memoryBusWidth) *
        static_cast<float>(props.memoryClockRate) * 2.0 / 8.0;
    float throughput =
        (static_cast<float>(bytes_processed) / (time_in_ms / 1.0e3));
    float performance = 100 * throughput / max_bidirectional_bandwidth;
    std::cout << "Ran the kernel " << kernel_name << std::endl;
    std::cout << "Processed " << bytes_processed << " bytes in " << time_in_ms
              << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " bytes/s" << std::endl;
    std::cout << "Max bidirectional bandwidth: " << max_bidirectional_bandwidth
              << " bytes/s (" << device_name << ")" << std::endl;
    std::cout << "Performance: " << performance << " % of theoretical max"
              << std::endl;
  }
}
