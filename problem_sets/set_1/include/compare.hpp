#ifndef PROBLEM_SETS_SET_1_INCLUDE_COMPARE_HPP_
#define PROBLEM_SETS_SET_1_INCLUDE_COMPARE_HPP_

#include <string>  // for string

void compareImages(std::string reference_filename, std::string test_filename,
                   bool useEpsCheck, double perPixelError, double globalError);

#endif  // PROBLEM_SETS_SET_1_INCLUDE_COMPARE_HPP_
