#ifndef ARMOR_DETECTOR_NN_DETECTION_QUALITY_FILTER_HPP_
#define ARMOR_DETECTOR_NN_DETECTION_QUALITY_FILTER_HPP_

#include <array>
#include <vector>

#include "armor_detector_nn/core/detection_types.hpp"
#include "armor_detector_nn/core/detector_config.hpp"

namespace fyt::auto_aim {

struct ArmorGeometryMetrics {
  bool valid{false};
  double armor_ratio{0.0};
  double side_ratio{0.0};
  double rectangular_error_deg{0.0};
  double min_lightbar_length_px{0.0};
  double area_px{0.0};
};

// Canonical keypoint order used by the NN detector PnP path:
// [0]=left_bottom, [1]=left_top, [2]=right_top, [3]=right_bottom.
std::array<cv::Point2f, 4> canonicalArmorKeypoints(
  const std::array<cv::Point2f, 4>& keypoints);

RawDetection canonicalizeRawDetectionGeometry(const RawDetection& detection);

ArmorDetection canonicalizeArmorDetectionGeometry(const ArmorDetection& detection);

ArmorGeometryMetrics computeArmorGeometryMetrics(const ArmorDetection& detection);

bool passesArmorGeometryQuality(
  const ArmorDetection& detection,
  const QualityFilterConfig& config);

std::vector<ArmorDetection> filterByGeometryQuality(
  const std::vector<ArmorDetection>& detections,
  const QualityFilterConfig& config);

std::vector<ArmorDetection> suppressDuplicateDetections(
  const std::vector<ArmorDetection>& detections,
  const QualityFilterConfig& config);

}  // namespace fyt::auto_aim

#endif
