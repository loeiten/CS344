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
  std::size_t num_rows();
  std::size_t num_cols();

  // return types are void since any internal error will be handled by quitting
  // no point in returning error codes...
  void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                  uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                  unsigned char **d_red_blurred,
                  unsigned char **d_green_blurred,
                  unsigned char **d_blue_blurred, float **h_filter,
                  int *filter_width, const std::string &filename);

  void postProcess(const std::string &output_file, uchar4 *data_ptr);

  void generateReferenceImage(std::string input_filename,
                              std::string output_filename, int kernel_size);

  uchar4 *d_outputImageRGBA__;

 private:
  cv::Mat imageInputRGBA;
  cv::Mat imageOutputRGBA;

  uchar4 *d_inputImageRGBA__;

  float *h_filter__;

  uchar4 *h_inputImageRGBA_ = nullptr;
  uchar4 *h_outputImageRGBA_ = nullptr;
};

#endif  // PROBLEM_SETS_SET_2_INCLUDE_IMAGE_HPP_
