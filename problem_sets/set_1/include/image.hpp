#ifndef PROBLEM_SETS_SET_1_INCLUDE_IMAGE_HPP_
#define PROBLEM_SETS_SET_1_INCLUDE_IMAGE_HPP_

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
  // returns a pointer to an RGBA version of the input image
  // and a pointer to the single channel grey-scale output
  // on both the host and device
  void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                  uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                  const std::string &filename);

  void postProcess(const std::string &output_file, unsigned char *data_ptr);

  void generateReferenceImage(std::string input_filename,
                              std::string output_filename);

 private:
  cv::Mat imageRGBA;
  cv::Mat imageGrey;

  uchar4 *d_rgbaImage__;
  unsigned char *d_greyImage__;

  uchar4 *h_rgbaImage_ = nullptr;
};

#endif  // PROBLEM_SETS_SET_1_INCLUDE_IMAGE_HPP_
