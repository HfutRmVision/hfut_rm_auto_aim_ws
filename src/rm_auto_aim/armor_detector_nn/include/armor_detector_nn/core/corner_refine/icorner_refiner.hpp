#ifndef ARMOR_DETECTOR_NN_ICORNER_REFINER_HPP_
#define ARMOR_DETECTOR_NN_ICORNER_REFINER_HPP_

#include <array>

#include <opencv2/core.hpp>

#include "armor_detector_nn/core/detection_types.hpp"

namespace fyt::auto_aim {

enum class RefineFailReason {
  NONE,
  ROI_TOO_SMALL,
  ROI_TOO_LARGE,
  TOO_FEW_BRIGHT_POINTS,
  PCA_UNSTABLE,
  GEOMETRY_INVALID,
  TIMEOUT,
  REFINE_WORSE,
  NO_LIGHTBAR_PAIR
};

struct RefineResult {
  bool ok{false};
  std::array<cv::Point2f, 4> refined_keypoints;
  double refine_quality{0.0};
  double elapsed_ms{0.0};
  RefineFailReason reason{RefineFailReason::NONE};
};

class ICornerRefiner {
public:
  virtual ~ICornerRefiner() = default;

  // Refine the 4 keypoints of a detection using local image processing.
  // On failure, refined_keypoints should be set to the original keypoints
  // and ok should be false.
  virtual RefineResult refine(
    const cv::Mat& frame,
    const ArmorDetection& detection) = 0;
};

}  // namespace fyt::auto_aim

#endif
