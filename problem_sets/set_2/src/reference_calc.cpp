#include <stddef.h>  // for size_t

#include <algorithm>  // for max, min
#include <cassert>    // for assert

#include "vector_functions.hpp"  // for make_uchar4
#include "vector_types.h"        // for uchar4

void channelConvolution(const unsigned char *const channel,
                        unsigned char *const channelBlurred,
                        const size_t num_rows, const size_t num_cols,
                        const float *filter, const int filter_width) {
  // Dealing with an even width filter is trickier
  assert(filter_width % 2 == 1);

  // For every pixel in the image
  for (int r = 0; r < (int)num_rows; ++r) {
    for (int c = 0; c < (int)num_cols; ++c) {
      float result = 0.f;
      // For every value in the filter around the pixel (c, r)
      for (int filter_r = -filter_width / 2; filter_r <= filter_width / 2;
           ++filter_r) {
        for (int filter_c = -filter_width / 2; filter_c <= filter_width / 2;
             ++filter_c) {
          // Find the global image position for this filter position
          // clamp to boundary of the image
          int image_r = std::min(std::max(r + filter_r, 0),
                                 static_cast<int>(num_rows - 1));
          int image_c = std::min(std::max(c + filter_c, 0),
                                 static_cast<int>(num_cols - 1));

          float image_value =
              static_cast<float>(channel[image_r * num_cols + image_c]);
          float filter_value =
              filter[(filter_r + filter_width / 2) * filter_width + filter_c +
                     filter_width / 2];

          result += image_value * filter_value;
        }
      }

      channelBlurred[r * num_cols + c] = result;
    }
  }
}

void referenceCalculation(const uchar4 *const rgbaImage,
                          uchar4 *const outputImage, size_t num_rows,
                          size_t num_cols, const float *const filter,
                          const int filter_width) {
  unsigned char *red = new unsigned char[num_rows * num_cols];
  unsigned char *blue = new unsigned char[num_rows * num_cols];
  unsigned char *green = new unsigned char[num_rows * num_cols];

  unsigned char *redBlurred = new unsigned char[num_rows * num_cols];
  unsigned char *blueBlurred = new unsigned char[num_rows * num_cols];
  unsigned char *greenBlurred = new unsigned char[num_rows * num_cols];

  // First we separate the incoming RGBA image into three separate channels
  // for Red, Green and Blue
  for (size_t i = 0; i < num_rows * num_cols; ++i) {
    uchar4 rgba = rgbaImage[i];
    red[i] = rgba.x;
    green[i] = rgba.y;
    blue[i] = rgba.z;
  }

  // Now we can do the convolution for each of the color channels
  channelConvolution(red, redBlurred, num_rows, num_cols, filter, filter_width);
  channelConvolution(green, greenBlurred, num_rows, num_cols, filter,
                     filter_width);
  channelConvolution(blue, blueBlurred, num_rows, num_cols, filter,
                     filter_width);

  // now recombine into the output image - Alpha is 255 for no transparency
  for (size_t i = 0; i < num_rows * num_cols; ++i) {
    uchar4 rgba =
        make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
    outputImage[i] = rgba;
  }

  delete[] red;
  delete[] green;
  delete[] blue;

  delete[] redBlurred;
  delete[] greenBlurred;
  delete[] blueBlurred;
}
