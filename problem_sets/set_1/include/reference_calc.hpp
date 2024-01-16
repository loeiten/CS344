#ifndef PROBLEM_SETS_SET_1_INCLUDE_REFERENCE_CALC_HPP_
#define PROBLEM_SETS_SET_1_INCLUDE_REFERENCE_CALC_HPP_

#include <cuda_runtime.h>

#include <cstddef>

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char* const greyImage, std::size_t numRows,
                          std::size_t numCols);

#endif  // PROBLEM_SETS_SET_1_INCLUDE_REFERENCE_CALC_HPP_