#include <cuda_runtime.h>  // for cudaFree, cudaDeviceSynchro...
#include <stdio.h>         // for printf
#include <stdlib.h>        // for atof, exit

#include <cstddef>     // for size_t
#include <filesystem>  // for absolute, path, create_dire...
#include <iostream>    // for operator<<, endl, basic_ost...
#include <string>      // for allocator, operator+, string

#include "../include/compare.hpp"         // for compareImages
#include "../include/image.hpp"           // for Image
#include "../include/reference_calc.hpp"  // for referenceCalculation
#include "../include/timer.hpp"           // for GpuTimer
#include "../include/utils.hpp"           // for check, checkCudaErrors
#include "driver_types.h"                 // for cudaMemcpyDeviceToHost
#include "vector_types.h"                 // for uchar4

// Declare function found in student_func.cu
// We cannot include this as an header as it contains device code
void your_gaussian_blur(const uchar4 *const h_inputImageRGBA,
                        uchar4 *const d_inputImageRGBA,
                        uchar4 *const d_outputImageRGBA,
                        const std::size_t num_rows, const std::size_t num_cols,
                        float const *const d_filter, unsigned char *const d_red,
                        unsigned char *const d_green,
                        unsigned char *const d_blue,
                        unsigned char *d_red_blurred,
                        unsigned char *d_green_blurred,
                        unsigned char *d_blue_blurred, const int filter_width);

void allocateMemoryAndCopyToGPU(const std::size_t num_rowsImage,
                                const std::size_t num_colsImage,
                                const float *const h_filter,
                                const std::size_t filter_width,
                                float **d_filter, unsigned char **d_red,
                                unsigned char **d_green,
                                unsigned char **d_blue);

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_red_blurred, *d_green_blurred, *d_blue_blurred;

  float *h_filter;
  int filter_width;

  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path reference_path;
  std::string base_name = "cinque_terre";
  std::string extension = "gold";
  std::string output_dir_name = "output";
  std::string file_name;
  double per_pixel_error = 0.0;
  double global_error = 0.0;
  bool useEpsCheck = false;
  switch (argc) {
    case 1:
      file_name = "./data/" + base_name + "." + extension;
      input_path = std::filesystem::absolute(file_name);
      file_name = "./" + output_dir_name + "/" + base_name + "_gpu.png";
      output_path = std::filesystem::absolute(file_name);
      file_name = "./" + output_dir_name + "/" + base_name + "_cpu.png";
      reference_path = std::filesystem::absolute(file_name);
      break;
    case 2:
      input_path = std::filesystem::absolute(argv[1]);
      base_name = input_path.stem().string();
      file_name = "./" + output_dir_name + "/" + base_name + "_gpu.png";
      output_path = std::filesystem::absolute(file_name);
      file_name = "./" + output_dir_name + "/" + base_name + "_cpu.png";
      reference_path = std::filesystem::absolute(file_name);
      break;
    case 3:
      input_path = std::filesystem::absolute(argv[1]);
      output_path = std::filesystem::absolute(argv[2]);
      base_name = input_path.stem().string();
      file_name = base_name + "_cpu.png";
      reference_path = output_path.parent_path() / file_name;
      break;
    case 4:
      input_path = std::filesystem::absolute(argv[1]);
      output_path = std::filesystem::absolute(argv[2]);
      reference_path = std::filesystem::absolute(argv[3]);
      break;
    case 6:
      useEpsCheck = true;
      input_path = std::filesystem::absolute(argv[1]);
      output_path = std::filesystem::absolute(argv[2]);
      reference_path = std::filesystem::absolute(argv[3]);
      per_pixel_error = atof(argv[4]);
      global_error = atof(argv[5]);
      break;
    default:
      std::cerr
          << "Usage: ./problem_set_2 [input_path] [output_path] "
             "[reference_path] [per_pixel_error] [global_error]\n"
             "The output_path and reference_path will be generated by the "
             "code.\n"
             "The per_pixel_error and global_error are epsilon tolerances."
          << std::endl;
      exit(1);
  }
  // Create the directories
  if (!std::filesystem::exists((output_path.parent_path()))) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  if (!std::filesystem::exists((reference_path.parent_path()))) {
    std::filesystem::create_directories(reference_path.parent_path());
  }

  Image image;

  // load the image and give us our input and output pointers
  image.preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA,
                   &d_outputImageRGBA, &d_red_blurred, &d_green_blurred,
                   &d_blue_blurred, &h_filter, &filter_width,
                   input_path.string());

  float *d_filter = nullptr;
  unsigned char *d_red = nullptr;
  unsigned char *d_green = nullptr;
  unsigned char *d_blue = nullptr;
  // WARNING: If we do not pass a pointer here, the function will allocate to
  //          the local variables inside the function.
  //          In other words: As we pass by value then changes to the variables
  //          inside the function will not affect the variables in the callee
  allocateMemoryAndCopyToGPU(image.num_rows(), image.num_cols(), h_filter,
                             filter_width, &d_filter, &d_red, &d_green,
                             &d_blue);
  GpuTimer timer;
  timer.Start();
  // call the students' code
  your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,
                     image.num_rows(), image.num_cols(), d_filter, d_red,
                     d_green, d_blue, d_red_blurred, d_green_blurred,
                     d_blue_blurred, filter_width);
  timer.Stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  float elapsed_ms = timer.Elapsed();
  int err = printf("Your code ran in: %f msecs.\n", elapsed_ms);

  if (err < 0) {
    // Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!"
              << std::endl;
    exit(1);
  }

  std::size_t numPixels = image.num_rows() * image.num_cols();

  // check results and output the blurred image
  // copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, image.d_outputImageRGBA__,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyDeviceToHost));

  image.postProcess(output_path.string(), h_outputImageRGBA);

  referenceCalculation(h_inputImageRGBA, h_outputImageRGBA, image.num_rows(),
                       image.num_cols(), h_filter, filter_width);

  image.postProcess(reference_path.string(), h_outputImageRGBA);

  // Cheater easy way with OpenCV
  // image.generateReferenceImage(input_path.string(), reference_path.string(),
  // filter_width);

  compareImages(reference_path.string(), output_path.string(), useEpsCheck,
                per_pixel_error, global_error);

  // Free allocated memory
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_red_blurred));
  checkCudaErrors(cudaFree(d_green_blurred));
  checkCudaErrors(cudaFree(d_blue_blurred));

  return 0;
}
