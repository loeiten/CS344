// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words,
// we multiply each weight with the pixel underneath it. Finally, we add up all
// of the multiplied numbers and assign that value to our output for the current
// pixel. We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// input_channel, using the array of weights, and put the result in the
// output_channel.

// Here is an example of computing a blur, using a weighted average, for a
// single pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its
// width. We refer to the array of weights as a filter, and we refer to its
// width with the variable filter_width.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called
// checkCudaErrors. You should wrap your allocation and copying statements like
// we've done in the code we're supplying you. Here is an example of the unsafe
// way to allocate memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * num_rows * num_cols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * num_rows *
// num_cols));
//
// Writing code the safe way requires slightly more typing, but is very helpful
// for catching mistakes. If you write code the unsafe way and you make a
// mistake, then any subsequent kernels won't compute anything, and it will be
// hard to figure out why. Writing code the safe way will inform you as soon as
// you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include <cuda_runtime.h>              // for cudaDeviceSynchronize
#include <cuda_runtime_api.h>          // for cudaMemcpy, cudaGetLastError
#include <device_launch_parameters.h>  // for blockIdx, threadIdx
#include <driver_types.h>              // for cudaMemcpyDeviceToHost
#include <vector_functions.h>          // for make_int2
#include <vector_types.h>              // for uchar4, dim3, int2

#include <cstddef>  // for size_t

#include "../include/utils.hpp"  // for checkCudaErrors

__global__ void gaussian_blur(const unsigned char* const input_channel,
                              unsigned char* const output_channel, int num_rows,
                              int num_cols, const float* const filter,
                              const int filter_width) {
  // TODO:

  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before
  // accessing GPU memory:
  //
  // if ( absolute_image_position_x >= num_cols ||
  //      absolute_image_position_y >= num_rows )
  // {
  //     return;
  // }

  // NOTE: If a thread's absolute position 2D position is within the image, but
  // some of its neighbors are outside the image, then you will need to be extra
  // careful. Instead of trying to read such a neighbor value from GPU memory
  // (which won't work because the value is out of bounds), you should
  // explicitly clamp the neighbor values you read to be within the bounds of
  // the image. If this is not clear to you, then please refer to sequential
  // reference solution for the exact clamping semantics you should follow.

  // Solution:
  int absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
  int absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (absolute_image_position_x >= num_cols ||
      absolute_image_position_y >= num_rows) {
    return;
  }
  int filter_center_in_image_abs_idx =
      num_cols * absolute_image_position_y + absolute_image_position_x;

  // Loop through the filter
  // Optimiziation: If we have more pixels than threads, we can calculate the
  //                optimal work one thread should do.
  //                Essentially one could
  //                1. Decompose (the image in smaller parts)
  //                2. Do the task (calculate the blur)
  //                3. Aggregate (combine the smaller parts)
  //
  //                The size of the parts should take into account the hardware
  //                i.e. that a warp consist of 32 threads (A100)
  //                Within a block of threads, the threads are executed in
  //                groups of one warp
  //                2 warps can run simultaneously on a SM
  //                There are 6912 (A100) CUDA cores - how many threads that can
  //                simltaneously run on the GPU
  //                A CUDA core being what runs a thread
  //                We should activate more threads than available CUDA cores as
  //                threads can be swapped in order to hide latency
  //
  //                We can see how well we are performing by checking
  //                achieved throughput/bandwidth
  //
  //                Where the acheived throughput (measured before and after
  //                the kernel) = 2*Bytes in image/time it took
  //                We multiply with 2 as there will be at least be one read and
  //                one write operation
  //
  //                For cinque_terre.gold we have 557x313 pixels per channel
  //                if each pixel is a uchar, then it's one byte, so
  //                2*Bytes in image (per channel) = 3.48682*1e5 bytes
  //                WARNING: This is likely too little to measure
  //                         the performance as we're also measuring the
  //                         kernel launch overheads etc.
  //                         To get a more realistic number either pick a larger
  //                         image, or make the launch so that it loops over
  //                         more data
  //
  //                Memory bandwidth for A100 40 GB
  //                = 1555GB/sec = 1.555*1e12 bytes/s
  //
  //                We then get:
  //                Performance
  //                = (3.48682*1e5 bytes/time[s])/(1.555*1e12 bytes/s)
  //                = 2.24232*1e-7/time

  // Image coordinates
  // We want to go from filter_abs_idx to image_abs_idx
  // image_abs_idx =
  //   filter_center_in_image_abs_idx +
  //   filter_rel_row*image_width +
  //   filter_rel_col
  //
  // where
  //
  // filter_mid = (filter_width*filter_width - 1)/2
  // filter_rel_col = filter_abs_idx%filter_width - (filter_width-1)/2
  // filter_rel_row =
  //   (filter_abs_idx - filter_mid - filter_rel_col)/filter_width

  int last_filter_idx = filter_width * filter_width - 1;
  // NOTE: The filter_width must be odd for there to be a one-celled mid-point
  int filter_mid = last_filter_idx / 2;
  float center_value = 0;
  for (int filter_idx = 0; filter_idx <= last_filter_idx; ++filter_idx) {
    int filter_rel_col = filter_idx % filter_width - (filter_width - 1) / 2;
    int filter_rel_row =
        (filter_abs_idx - filter_mid - filter_rel_col) / filter_width;
    int image_idx = filter_center_in_image_abs_idx + filter_rel_row * num_cols +
                    filter_rel_col;
    // NOTE: We have no contention when reading (as the values do not change)
    // and we have no contention when writing (as there is only one place
    // writing to the image). However, it is inefficient as threads are
    // re-reading the values of the input_channel and filter
    // filter_width*filter_width times
    output_channel[image_idx] = input_channel[image_idx] + filter[filter_idx];
  }
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                                 int num_rows, int num_cols,
                                 unsigned char* const redChannel,
                                 unsigned char* const greenChannel,
                                 unsigned char* const blueChannel) {
  // TODO:
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before
  // accessing GPU memory:
  //
  // if ( absolute_image_position_x >= num_cols ||
  //      absolute_image_position_y >= num_rows )
  // {
  //     return;
  // }

  // Solution:
  // Calculate the absolute positions
  int absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
  int absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (absolute_image_position_x >= num_cols ||
      absolute_image_position_y >= num_rows) {
    return;
  }
  int sequential_image_position =
      num_cols * absolute_image_position_y + absolute_image_position_x;

  redChannel[sequential_image_position] =
      inputImageRGBA[sequential_image_position].x;
  greenChannel[sequential_image_position] =
      inputImageRGBA[sequential_image_position].y;
  blueChannel[sequential_image_position] =
      inputImageRGBA[sequential_image_position].z;
}

// This kernel takes in three color channels and recombines them
// into one image.  The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char* const redChannel,
                                  const unsigned char* const greenChannel,
                                  const unsigned char* const blueChannel,
                                  uchar4* const outputImageRGBA, int num_rows,
                                  int num_cols) {
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * num_cols + thread_2D_pos.x;

  // make sure we don't try and access memory outside the image
  // by having any threads mapped there return early
  if (thread_2D_pos.x >= num_cols || thread_2D_pos.y >= num_rows) {
    return;
  }

  unsigned char red = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue = blueChannel[thread_1D_pos];

  // Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

void allocateMemoryAndCopyToGPU(const std::size_t num_rowsImage,
                                const std::size_t num_colsImage,
                                const float* const h_filter,
                                const std::size_t filter_width, float* d_filter,
                                unsigned char* d_red, unsigned char* d_green,
                                unsigned char* d_blue) {
  // NOTE: This is the first cuda function that will be called from main
  // allocate memory for the three different channels
  // original
  checkCudaErrors(cudaMalloc(
      &d_red, sizeof(unsigned char) * num_rowsImage * num_colsImage));
  checkCudaErrors(cudaMalloc(
      &d_green, sizeof(unsigned char) * num_rowsImage * num_colsImage));
  checkCudaErrors(cudaMalloc(
      &d_blue, sizeof(unsigned char) * num_rowsImage * num_colsImage));

  // TODO:
  // Allocate memory for the filter on the GPU
  // Use the pointer d_filter that we have already declared for you
  // You need to allocate memory for the filter with cudaMalloc
  // be sure to use checkCudaErrors like the above examples to
  // be able to tell if anything goes wrong
  // IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  // Solution:
  // NOTE: We've sanitized the code and changed d_filter to an input parameter
  checkCudaErrors(
      cudaMalloc(&d_filter, sizeof(float) * filter_width * filter_width));

  // TODO:
  // Copy the filter on the host (h_filter) to the memory you just allocated
  // on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  // Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter,
                             sizeof(float) * filter_width * filter_width,
                             cudaMemcpyHostToDevice));
}

void your_gaussian_blur(
    const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const std::size_t num_rows,
    const std::size_t num_cols, unsigned char const* const d_red,
    unsigned char const* const d_green, unsigned char const* const d_blue,
    unsigned char* d_red_blurred, unsigned char* d_green_blurred,
    unsigned char* d_blue_blurred, const int filter_width) {
  // TODO: Set reasonable block size (i.e., number of threads per block)
  // Solution:
  // We set one thread per pixel, see ../../set_1/src/student_func.cu for
  // explanation
  constexpr int max_threads_per_dim = 32;
  const dim3 blockSize(max_threads_per_dim, max_threads_per_dim, 1);

  // TODO:
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  // Solution:
  const int x_blocks =
      (num_cols + (max_threads_per_dim - 1)) / max_threads_per_dim;
  const int y_blocks =
      (num_rows + (max_threads_per_dim - 1)) / max_threads_per_dim;
  const dim3 gridSize(x_blocks, y_blocks, 1);

  // TODO: Launch a kernel for separating the RGBA image into different color
  // channels
  // Solution:
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, num_rows,
                                            num_cols, d_red_blurred,
                                            d_green_blurred, d_blue_blurred);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // TODO: Call your convolution kernel here 3 times, once for each color
  // channel.
  gaussian_blur < < < gridSize,
      blockSize >>>>
          (d_red, d_red_blurred, num_rows, num_cols, d_filter, filter_width);
  gaussian_blur < < < gridSize,
      blockSize >>>> (d_green, d_green_blurred, num_rows, num_cols, d_filter,
                      filter_width);
  gaussian_blur < < < gridSize,
      blockSize >>>>
          (d_blue, d_blue_blurred, num_rows, num_cols, d_filter, filter_width);

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors()
  // immediately after launching your kernel to make sure that you didn't make
  // any mistakes.
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for
  // you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_red_blurred, d_green_blurred,
                                             d_blue_blurred, d_outputImageRGBA,
                                             num_rows, num_cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

// Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
// Solution: Moved to main.cpp
