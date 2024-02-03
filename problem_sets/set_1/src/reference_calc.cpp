#include <vector_types.h>  // for uchar4

#include <cstddef>  // for size_t

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char* const greyImage, std::size_t num_rows,
                          std::size_t num_cols) {
  for (std::size_t r = 0; r < num_rows; ++r) {
    for (std::size_t c = 0; c < num_cols; ++c) {
      uchar4 rgba = rgbaImage[r * num_cols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * num_cols + c] = channelSum;
    }
  }
}
