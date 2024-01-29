#ifndef PROBLEM_SETS_SET_2_INCLUDE_REFERENCE_CALC_HPP_
#define PROBLEM_SETS_SET_2_INCLUDE_REFERENCE_CALC_HPP_

#include <vector_types.h>  // for uchar4

#include <cstddef>  // for size_t

void referenceCalculation(const uchar4* const rgbaImage,
                          uchar4* const outputImage, std::size_t numRows,
                          std::size_t numCols, const float* const filter,
                          const int filterWidth);

#endif  // PROBLEM_SETS_SET_2_INCLUDE_REFERENCE_CALC_HPP_
