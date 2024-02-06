#ifndef PROBLEM_SETS_SET_1_INCLUDE_PERFORMANCE_HPP_
#define PROBLEM_SETS_SET_1_INCLUDE_PERFORMANCE_HPP_

#include <cstddef>  // for size_t
#include <string>   // for string

void PrintPerformance(const std::string &kernel_name,
                      std::size_t bytes_processed, float time_in_ms);

#endif  // PROBLEM_SETS_SET_1_INCLUDE_PERFORMANCE_HPP_
