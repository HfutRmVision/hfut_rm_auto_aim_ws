#include "armor_detector_nn/postprocess/detection_quality_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace fyt::auto_aim {

namespace {

constexpr double kEps = 1e-6;
constexpr double kPi = 3.14159265358979323846;

double pointDistance(const cv::Point2f& a, const cv::Point2f& b) {
  return cv::norm(a - b);
}

bool isFinitePoint(const cv::Point2f& p) {
  return std::isfinite(p.x) && std::isfinite(p.y);
}

double normalizeAngle(double angle) {
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

double angularDistance(double a, double b) {
  return std::abs(normalizeAngle(a - b));
}

std::array<cv::Point2f, 4> orderedByImageGeometry(
    const std::array<cv::Point2f, 4>& keypoints) {
  std::array<cv::Point2f, 4> sorted = keypoints;
  std::sort(sorted.begin(), sorted.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) {
              if (std::abs(a.y - b.y) > 1e-3F) return a.y < b.y;
              return a.x < b.x;
            });

  std::array<cv::Point2f, 2> top{sorted[0], sorted[1]};
  std::array<cv::Point2f, 2> bottom{sorted[2], sorted[3]};
  std::sort(top.begin(), top.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
  std::sort(bottom.begin(), bottom.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

  return {bottom[0], top[0], top[1], bottom[1]};
}

double polygonArea(const std::array<cv::Point2f, 4>& pts) {
  double area = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const auto& a = pts[i];
    const auto& b = pts[(i + 1) % pts.size()];
    area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
  }
  return std::abs(area) * 0.5;
}

float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);

  const float inter_w = std::max(0.0F, x2 - x1);
  const float inter_h = std::max(0.0F, y2 - y1);
  const float inter = inter_w * inter_h;
  const float area_a = std::max(0.0F, a.width) * std::max(0.0F, a.height);
  const float area_b = std::max(0.0F, b.width) * std::max(0.0F, b.height);
  return inter / (area_a + area_b - inter + 1e-6F);
}

double meanKeypointDistance(
    const ArmorDetection& a,
    const ArmorDetection& b) {
  const auto ak = canonicalArmorKeypoints(a.keypoints);
  const auto bk = canonicalArmorKeypoints(b.keypoints);

  double sum = 0.0;
  for (size_t i = 0; i < ak.size(); ++i) {
    sum += pointDistance(ak[i], bk[i]);
  }
  return sum / static_cast<double>(ak.size());
}

bool isArmorTypeConsistent(const ArmorDetection& detection) {
  const auto& number = detection.publish_number;
  const auto& type = detection.publish_type;

  if (number == "1") {
    return type == "large";
  }
  if (number == "2" || number == "outpost" || number == "sentry") {
    return type == "small";
  }
  return type == "small" || type == "large";
}

}  // namespace

std::array<cv::Point2f, 4> canonicalArmorKeypoints(
    const std::array<cv::Point2f, 4>& keypoints) {
  return orderedByImageGeometry(keypoints);
}

RawDetection canonicalizeRawDetectionGeometry(const RawDetection& detection) {
  RawDetection normalized = detection;
  normalized.keypoints = canonicalArmorKeypoints(detection.keypoints);
  return normalized;
}

ArmorDetection canonicalizeArmorDetectionGeometry(const ArmorDetection& detection) {
  ArmorDetection normalized = detection;
  normalized.keypoints = canonicalArmorKeypoints(detection.keypoints);
  normalized.center =
    (normalized.keypoints[0] + normalized.keypoints[1] +
     normalized.keypoints[2] + normalized.keypoints[3]) * 0.25F;
  return normalized;
}

ArmorGeometryMetrics computeArmorGeometryMetrics(const ArmorDetection& detection) {
  ArmorGeometryMetrics metrics;

  if (detection.bbox.width <= 0.0F || detection.bbox.height <= 0.0F) {
    return metrics;
  }

  for (const auto& kp : detection.keypoints) {
    if (!isFinitePoint(kp)) {
      return metrics;
    }
  }

  const auto pts = canonicalArmorKeypoints(detection.keypoints);
  const double left_length = pointDistance(pts[1], pts[0]);
  const double right_length = pointDistance(pts[2], pts[3]);
  const double top_length = pointDistance(pts[2], pts[1]);
  const double bottom_length = pointDistance(pts[3], pts[0]);
  const double max_side_length = std::max(left_length, right_length);
  const double min_side_length = std::min(left_length, right_length);
  const double max_plate_width = std::max(top_length, bottom_length);

  if (max_side_length <= kEps || min_side_length <= kEps) {
    return metrics;
  }

  const cv::Point2f left_center = (pts[0] + pts[1]) * 0.5F;
  const cv::Point2f right_center = (pts[2] + pts[3]) * 0.5F;
  const cv::Point2f left_to_right = right_center - left_center;
  if (pointDistance(left_center, right_center) <= kEps) {
    return metrics;
  }

  const double roll = std::atan2(left_to_right.y, left_to_right.x);
  const double left_angle = std::atan2((pts[0] - pts[1]).y, (pts[0] - pts[1]).x);
  const double right_angle = std::atan2((pts[3] - pts[2]).y, (pts[3] - pts[2]).x);
  const double left_rect_error = angularDistance(left_angle, roll + kPi * 0.5);
  const double right_rect_error = angularDistance(right_angle, roll + kPi * 0.5);

  metrics.valid = true;
  metrics.armor_ratio = max_plate_width / max_side_length;
  metrics.side_ratio = max_side_length / min_side_length;
  metrics.rectangular_error_deg =
    std::max(left_rect_error, right_rect_error) * 180.0 / kPi;
  metrics.min_lightbar_length_px = min_side_length;
  metrics.area_px = polygonArea(pts);
  return metrics;
}

bool passesArmorGeometryQuality(
    const ArmorDetection& detection,
    const QualityFilterConfig& config) {
  if (!config.enabled) {
    return true;
  }

  if (!isArmorTypeConsistent(detection)) {
    return false;
  }

  const auto metrics = computeArmorGeometryMetrics(detection);
  if (!metrics.valid) {
    return false;
  }

  if (metrics.area_px < config.min_area_px) {
    return false;
  }
  if (metrics.min_lightbar_length_px < config.min_lightbar_length_px) {
    return false;
  }
  if (metrics.armor_ratio < config.min_armor_ratio ||
      metrics.armor_ratio > config.max_armor_ratio) {
    return false;
  }
  if (metrics.side_ratio > config.max_side_ratio) {
    return false;
  }
  if (metrics.rectangular_error_deg > config.max_rectangular_error_deg) {
    return false;
  }
  return true;
}

std::vector<ArmorDetection> filterByGeometryQuality(
    const std::vector<ArmorDetection>& detections,
    const QualityFilterConfig& config) {
  if (!config.enabled) {
    return detections;
  }

  std::vector<ArmorDetection> filtered;
  filtered.reserve(detections.size());
  for (const auto& det : detections) {
    auto normalized = canonicalizeArmorDetectionGeometry(det);
    if (passesArmorGeometryQuality(normalized, config)) {
      filtered.push_back(std::move(normalized));
    }
  }
  return filtered;
}

std::vector<ArmorDetection> suppressDuplicateDetections(
    const std::vector<ArmorDetection>& detections,
    const QualityFilterConfig& config) {
  if (!config.enabled || !config.deduplicate_enabled || detections.size() < 2) {
    return detections;
  }

  std::vector<ArmorDetection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(),
            [](const ArmorDetection& a, const ArmorDetection& b) {
              return a.confidence > b.confidence;
            });

  std::vector<ArmorDetection> kept;
  kept.reserve(sorted.size());
  for (const auto& candidate : sorted) {
    bool duplicated = false;
    for (const auto& selected : kept) {
      if (candidate.color != selected.color) {
        continue;
      }
      const bool bbox_duplicate =
        config.duplicate_iou_threshold > 0.0 &&
        computeIoU(candidate.bbox, selected.bbox) > config.duplicate_iou_threshold;
      const bool keypoint_duplicate =
        config.duplicate_keypoint_mean_dist_px > 0.0 &&
        meanKeypointDistance(candidate, selected) < config.duplicate_keypoint_mean_dist_px;
      if (bbox_duplicate || keypoint_duplicate) {
        duplicated = true;
        break;
      }
    }
    if (!duplicated) {
      kept.push_back(candidate);
    }
  }
  return kept;
}

}  // namespace fyt::auto_aim
