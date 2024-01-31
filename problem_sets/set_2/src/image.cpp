#include "../include/image.hpp"

#include <cuda_runtime.h>  // for cudaMalloc, cudaMemset, cuda...
#include <math.h>          // for expf
#include <stdlib.h>        // for exit

#include <iostream>  // for operator<<, endl, basic_ostream
#include <string>    // for string, operator<<

#include "../include/utils.hpp"          // for check, checkCudaErrors
#include "driver_types.h"                // for cudaMemcpyHostToDevice
#include "opencv2/core/hal/interface.h"  // for CV_8UC4
#include "opencv2/core/mat.inl.hpp"      // for _InputArray::_InputArray
#include "opencv2/core/types.hpp"        // for Size2i
#include "opencv2/imgcodecs.hpp"         // for imread, imwrite, IMREAD_COLOR
#include "opencv2/imgproc.hpp"           // for cvtColor, GaussianBlur, COLO...
#include "vector_types.h"                // for uchar4

Image::~Image() {
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);

  delete[] h_inputImageRGBA_;
  delete[] h_outputImageRGBA_;
  delete[] h_filter__;
}

std::size_t Image::numRows() { return imageInputRGBA.rows; }
std::size_t Image::numCols() { return imageInputRGBA.cols; }

void Image::preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                       uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                       unsigned char **d_redBlurred,
                       unsigned char **d_greenBlurred,
                       unsigned char **d_blueBlurred, float **h_filter,
                       int *filterWidth, const std::string &filename) {
  // make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, cv::COLOR_BGR2RGBA);

  // allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  // This shouldn't ever happen given the way the images are created
  // at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  // About padding and alignment:
  // https://blog.quarkslab.com/unaligned-accesses-in-cc-what-why-and-solutions-to-do-it-properly.html
  // https://www.youtube.com/watch?v=E0QhZ6tNoRg&ab_channel=C%2B%2BWeeklyWithJasonTurner
  // The following line gives padding issues
  // *h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  // *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
  // Hence we replace it with
  auto *inputImageRGBAPtr = imageInputRGBA.ptr<unsigned char>(0);
  auto *outputImageRGBAPtr = imageOutputRGBA.ptr<unsigned char>(0);
  h_inputImageRGBA_ = new uchar4[imageInputRGBA.total()];
  h_outputImageRGBA_ = new uchar4[imageOutputRGBA.total()];
  std::memcpy(h_inputImageRGBA_, inputImageRGBAPtr,
              imageInputRGBA.total() * imageInputRGBA.elemSize());
  std::memcpy(h_outputImageRGBA_, outputImageRGBAPtr,
              imageOutputRGBA.total() * imageOutputRGBA.elemSize());
  *h_inputImageRGBA = h_inputImageRGBA_;
  *h_outputImageRGBA = h_outputImageRGBA_;

  const std::size_t numPixels = numRows() * numCols();
  // allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(
      *d_outputImageRGBA, 0,
      numPixels *
          sizeof(uchar4)));  // make sure no memory is left laying around

  // copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyHostToDevice));

  d_inputImageRGBA__ = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;

  // now create the filter that they will use
  const int blurKernelWidth = 9;
  const float blurKernelSigma = 2.;

  *filterWidth = blurKernelWidth;

  // create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f;  // for normalization

  for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
    for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
      float filterValue = expf(-(float)(c * c + r * r) /
                               (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c +
                  blurKernelWidth / 2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
    for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
      (*h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c +
                  blurKernelWidth / 2] *= normalizationFactor;
    }
  }

  // blurred
  checkCudaErrors(cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels));
  checkCudaErrors(
      cudaMalloc(d_greenBlurred, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred, sizeof(unsigned char) * numPixels));
  checkCudaErrors(
      cudaMemset(*d_redBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(
      cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(
      cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char) * numPixels));
}

void Image::postProcess(const std::string &output_file, uchar4 *data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC4, (void *)data_ptr);

  cv::Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, cv::COLOR_RGBA2BGR);
  // output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}

// An unused bit of code showing how to accomplish this assignment using OpenCV.
// It is much faster
//    than the naive implementation in reference_calc.cpp.
void Image::generateReferenceImage(std::string input_file,
                                   std::string reference_file,
                                   int kernel_size) {
  cv::Mat input = cv::imread(input_file);
  // Create an identical image for the output as a placeholder
  cv::Mat reference = cv::imread(input_file);
  cv::GaussianBlur(input, reference, cv::Size2i(kernel_size, kernel_size), 0);
  cv::imwrite(reference_file, reference);
}
