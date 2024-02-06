#include <stddef.h>  // for NULL

#include <filesystem>  // for path, absolute, operator/, opera...
#include <iostream>    // for operator<<, endl, basic_ostream
#include <string>      // for string, char_traits, basic_string

#include "../include/utils.hpp"      // for checkResultsEps, checkResultsExact
#include "opencv2/core.hpp"          // for minMaxLoc
#include "opencv2/core/mat.hpp"      // for Mat, operator-, MatExpr, abs
#include "opencv2/core/mat.inl.hpp"  // for Mat::channels, Mat::operator=
#include "opencv2/imgcodecs.hpp"     // for imread, imwrite

void compareImages(std::string reference_filename, std::string test_filename,
                   bool useEpsCheck, double per_pixel_error,
                   double global_error) {
  cv::Mat reference = cv::imread(reference_filename, -1);
  cv::Mat test = cv::imread(test_filename, -1);

  cv::Mat diff = abs(reference - test);

  cv::Mat diffSingleChannel =
      diff.reshape(1, 0);  // convert to 1 channel, same # rows

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL,
                NULL);  // NULL because we don't care about location

  // now perform transform so that we bump values to the full range

  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(reference.channels(), 0);

  std::filesystem::path diff_path =
      std::filesystem::absolute(reference_filename).parent_path() /
      "cpu_gpu_difference.png";
  cv::imwrite(diff_path.string(), diff);
  std::cout << "Image written to: " << diff_path << std::endl;
  // OK, now we can start comparing values...
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (useEpsCheck) {
    checkResultsEps(referencePtr, testPtr,
                    reference.rows * reference.cols * reference.channels(),
                    per_pixel_error, global_error);
  } else {
    checkResultsExact(referencePtr, testPtr,
                      reference.rows * reference.cols * reference.channels());
  }

  std::cout << "PASS" << std::endl;
  return;
}
