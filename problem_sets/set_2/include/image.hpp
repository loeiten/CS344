#ifndef PROBLEM_SETS_SET_2_INCLUDE_IMAGE_HPP_
#define PROBLEM_SETS_SET_2_INCLUDE_IMAGE_HPP_

#include <vector_types.h>  // for uchar4

#include <cstddef>  // for size_t
#include <string>   // for string

#include "opencv2/core/mat.hpp"  // for Mat

class Image {
 public:
  Image() = default;
  ~Image();
  std::size_t numRows();
  std::size_t numCols();

  // return types are void since any internal error will be handled by quitting
  // no point in returning error codes...
  void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                  uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                  unsigned char **d_redBlurred, unsigned char **d_greenBlurred,
                  unsigned char **d_blueBlurred, float **h_filter,
                  int *filterWidth, const std::string &filename);

  void postProcess(const std::string &output_file, uchar4 *data_ptr);

  void generateReferenceImage(std::string input_filename,
                              std::string output_filename, int kernel_size);

 private:
  cv::Mat imageInputRGBA;
  cv::Mat imageOutputRGBA;

  uchar4 *d_inputImageRGBA__;
  uchar4 *d_outputImageRGBA__;

  float *h_filter__;
};

#endif  // PROBLEM_SETS_SET_2_INCLUDE_IMAGE_HPP_
