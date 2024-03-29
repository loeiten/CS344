#include "../include/image.hpp"

#include <cuda_runtime.h>  // for cudaFree, cudaMalloc, cudaMe...
#include <driver_types.h>  // for cudaMemcpyHostToDevice
#include <stdlib.h>        // for exit

#include <cstddef>   // for size_t
#include <cstring>   // for memcpy
#include <iostream>  // for operator<<, endl, basic_ostream
#include <string>    // for operator<<, string, allocator

#include "../include/utils.hpp"          // for check, checkCudaErrors
#include "opencv2/core/hal/interface.h"  // for CV_8UC1
#include "opencv2/core/mat.inl.hpp"      // for _InputArray::_InputArray
#include "opencv2/imgcodecs.hpp"         // for imread, imwrite, IMREAD_COLOR
#include "opencv2/imgproc.hpp"           // for cvtColor, COLOR_BGR2RGBA
#include "vector_types.h"                // for uchar4

Image::~Image() {
  if (h_rgbaImage_ == nullptr) {
    delete[] h_rgbaImage_;
  }
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

std::size_t Image::num_rows() { return imageRGBA.rows; }
std::size_t Image::num_cols() { return imageRGBA.cols; }

void Image::preProcess(uchar4 **inputImage, unsigned char **greyImage,
                       uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                       const std::string &filename) {
  // make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
  std::cout << "File read from: " << filename << std::endl;
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

  // allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  // This shouldn't ever happen given the way the images are created
  // at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  // About padding and alignment:
  // https://blog.quarkslab.com/unaligned-accesses-in-cc-what-why-and-solutions-to-do-it-properly.html
  // https://www.youtube.com/watch?v=E0QhZ6tNoRg&ab_channel=C%2B%2BWeeklyWithJasonTurner
  // The following line gives padding issues
  // *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  // Hence we replace it with
  auto *imageRGBAPtr = imageRGBA.ptr<unsigned char>(0);
  h_rgbaImage_ = new uchar4[imageRGBA.total()];
  std::memcpy(h_rgbaImage_, imageRGBAPtr,
              imageRGBA.total() * imageRGBA.elemSize());
  *inputImage = h_rgbaImage_;

  *greyImage = imageGrey.ptr<unsigned char>(0);

  const std::size_t numPixels = num_rows() * num_cols();
  // allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(
      *d_greyImage, 0,
      numPixels *
          sizeof(unsigned char)));  // make sure no memory is left laying around

  // copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void Image::postProcess(const std::string &output_file,
                        unsigned char *data_ptr) {
  cv::Mat output(num_rows(), num_cols(), CV_8UC1, (void *)data_ptr);

  // output the image
  cv::imwrite(output_file.c_str(), output);
  std::cout << "File written to: " << output_file << std::endl;
}

void Image::generateReferenceImage(std::string input_filename,
                                   std::string output_filename) {
  cv::Mat reference = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

  cv::imwrite(output_filename, reference);
  std::cout << "File written to: " << output_filename << std::endl;
}
