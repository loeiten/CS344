#ifndef PROBLEM_SETS_SET_2_INCLUDE_UTILS_HPP_
#define PROBLEM_SETS_SET_2_INCLUDE_UTILS_HPP_

#include <cuda_runtime.h>  // for cudaGetErrorString
#include <driver_types.h>  // for cudaSuccess
#include <stdlib.h>        // for exit, size_t

#include <algorithm>  // for max, min
#include <cassert>    // for assert
#include <cstdint>    // for uint64_t
#include <iomanip>    // for operator<<, setprecision
#include <iostream>   // for operator<<, endl, basic_ostream, cerr, bas...

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

template <typename T>
void checkResultsExact(const T* const ref, const T* const gpu, size_t numElem) {
  // check that the GPU result matches the CPU result
  for (size_t i = 0; i < numElem; ++i) {
    if (ref[i] != gpu[i]) {
      std::cerr << "Difference at pos " << i << std::endl;
      // the + is magic to convert char to int without messing
      // with other types
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i]
                << "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
  }
}

template <typename T>
void checkResultsEps(const T* const ref, const T* const gpu, size_t numElem,
                     double eps1, double eps2) {
  assert(eps1 >= 0 && eps2 >= 0);
  std::uint64_t totalDiff = 0;
  unsigned numSmallDifferences = 0;
  for (size_t i = 0; i < numElem; ++i) {
    // subtract smaller from larger in case of unsigned types
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > 0 && diff <= eps1) {
      numSmallDifferences++;
    } else if (diff > eps1) {
      std::cerr << "Difference at pos " << +i << " exceeds tolerance of "
                << eps1 << std::endl;
      std::cerr << "Reference: " << std::setprecision(17) << +ref[i]
                << "\nGPU      : " << +gpu[i] << std::endl;
      exit(1);
    }
    totalDiff += diff * diff;
  }
  double percentSmallDifferences =
      (double)numSmallDifferences / (double)numElem;
  if (percentSmallDifferences > eps2) {
    std::cerr << "Total percentage of non-zero pixel difference between the "
                 "two images exceeds "
              << 100.0 * eps2 << "%" << std::endl;
    std::cerr << "Percentage of non-zero pixel differences: "
              << 100.0 * percentSmallDifferences << "%" << std::endl;
    exit(1);
  }
}

// Uses the autodesk method of image comparison
// Note the the tolerance here is in PIXELS not a percentage of input pixels
template <typename T>
void checkResultsAutodesk(const T* const ref, const T* const gpu,
                          size_t numElem, double variance, size_t tolerance) {
  size_t numBadPixels = 0;
  for (size_t i = 0; i < numElem; ++i) {
    T smaller = std::min(ref[i], gpu[i]);
    T larger = std::max(ref[i], gpu[i]);
    T diff = larger - smaller;
    if (diff > variance) ++numBadPixels;
  }

  if (numBadPixels > tolerance) {
    std::cerr << "Too many bad pixels in the image." << numBadPixels << "/"
              << tolerance << std::endl;
    exit(1);
  }
}

#endif  // PROBLEM_SETS_SET_2_INCLUDE_UTILS_HPP_
