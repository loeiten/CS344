// Homework 1
// Color to Greyscale Conversion

// A common way to represent color images is known as RGBA - the color
// is specified by how much Red, Grean and Blue is in it.
// The 'A' stands for Alpha and is used for transparency, it will be
// ignored in this homework.

// Each channel Red, Blue, Green and Alpha is represented by one byte.
// Since we are using one byte for each color there are 256 different
// possible values for each color. This means we use 4 bytes per pixel.

// Greyscale images are represented by a single intensity value per pixel
// which is one byte in size.

// To convert an image from color to grayscale one simple method is to
// set the intensity to the average of the RGB channels.  But we will
// use a more sophisticated method that takes into account how the eye
// perceives color and weights the channels unequally.

// The eye responds most strongly to green followed by red and then blue.
// The NTSC (National Television System Committee) recommends the following
// formula for color to greyscale conversion:

// I = .299f * R + .587f * G + .114f * B

// Notice the trailing f's on the numbers which indicate that they are
// single precision floating point constants and not double precision
// constants.

// You should fill in the kernel as well as set the block and grid sizes
// so that the entire image is processed.

#include <device_launch_parameters.h>  // for blockIdx, threadIdx
#include <vector_types.h>              // for uchar4, dim3

#include <cstddef>  // for size_t

#include "../include/utils.hpp"  // for checkCudaErrors

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage,
                                  unsigned char* const greyImage, int numRows,
                                  int numCols) {
  // TODO
  // Fill in the kernel to convert from color to greyscale
  // the mapping from components of a uchar4 to RGBA is:
  //  .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  // The output (greyImage) at each pixel should be the result of
  // applying the formula: output = .299f * R + .587f * G + .114f * B;
  // Note: We will be ignoring the alpha channel for this conversion

  // First create a mapping from the 2D block and grid locations
  // to an absolute 2D location in the image, then use that to
  // calculate a 1D offset

  // Solution:
  // NOTE: This calculation could've been global
  const int max_idx = numRows * numCols;

  // In our example with numCol=5 and numRow=3
  // We have blockDim.x = blockDim.y = 2
  // If we were to edit pixel with x=4, y=1 we be at blockIdx.x=1 as we are at
  // the last x block of 3 (index 2 of 2) and the first thread
  // Since we have row-major indexing, we get
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int global_idx = numCols * y + x;

  if (global_idx < max_idx) {
    greyImage[global_idx] = .299f * rgbaImage[global_idx].x +
                            .587f * rgbaImage[global_idx].y +
                            .114f * rgbaImage[global_idx].z;
  }
}

void your_rgba_to_greyscale(uchar4* const d_rgbaImage,
                            unsigned char* const d_greyImage,
                            std::size_t numRows, std::size_t numCols) {
  // You must fill in the correct sizes for the blockSize and gridSize
  // currently only one block with one thread is being launched
  // const dim3 blockSize(1, 1, 1);  // TODO
  // const dim3 gridSize(1, 1, 1);   // TODO

  // Solution:
  // Parameters are x, y, z
  // x denotes the column of the image
  // y denotes the row of the image

  // At the current point I don't know whether it's better to launch more
  // threads or more blocks
  // Following this disucssion:
  // https://forums.developer.nvidia.com/t/blocks-vs-threads/259/9
  // it seems like:
  // 1. Launching more blocks than multiprocessors* will result in blocks being
  //    stalled waiting for resources
  // 2. Registers and shared memory are shared within a multiprocessor, so if
  //    one block uses more than half of available resources on a multiprocessor
  //    only one block will run on that multiprocessor
  // 3. Due to resource contraints it could make sense to minimize the threads
  //    per block
  //
  // * these are called streaming multiprocessors or (sm)
  //
  // Assuming we have relatively small images we can cap the threads per block
  // to 64
  constexpr int max_threads_per_block = 64;
  // To find the number of blocks we could therefore roof divide n_dim with
  // max_threads_per_block
  // We need then to take care that we are not writing out of bounds
  // Example: Assume we have
  // numCols=5
  // numRows=3
  // max_threads_per_block=2
  // If we use one thread per pixel, we would need 3*5=15 threads
  // x_blocks=RoofDivide(5, 2) = 3
  // y_blocks=RoofDivide(3, 2) = 2
  // RoofDivide(nominator, denomiator) = (nominator+(denomiator-1))/(denomiator)
  const int x_blocks =
      (numCols + (max_threads_per_block - 1)) / max_threads_per_block;
  const int y_blocks =
      (numRows + (max_threads_per_block - 1)) / max_threads_per_block;
  // We can then have
  // dim3 gridSize(x_blocks, y_blocks, 1)
  // dim3 blockSize(max_threads_per_block, max_threads_per_block, 1)
  // This will result in
  // x_blocks*max_threads_per_block*y_blocks*max_threads_per_block = 3*2*2*2=24
  // threads in total, but only 15 have to do actual work
  const dim3 gridSize(x_blocks, y_blocks, 1);
  const dim3 blockSize(max_threads_per_block, max_threads_per_block, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows,
                                             numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
