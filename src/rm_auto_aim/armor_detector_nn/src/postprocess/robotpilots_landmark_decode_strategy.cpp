#include "armor_detector_nn/postprocess/robotpilots_landmark_decode_strategy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "armor_detector_nn/postprocess/detection_quality_filter.hpp"

namespace fyt::auto_aim {

namespace {

constexpr int kFeatureCount = 22;
constexpr int kKeypointDims = 8;
constexpr int kObjectnessIndex = 8;
constexpr int kColorStart = 9;
constexpr int kColorCount = 4;
constexpr int kNumberStart = 13;
constexpr int kNumberCount = 9;

void applyKeypointRemap(std::array<cv::Point2f, 4>& keypoints,
                        const PostprocessConfig& config) {
  if (config.keypoint_remap.size() != keypoints.size()) {
    return;
  }

  std::array<cv::Point2f, 4> remapped = keypoints;
  for (size_t k = 0; k < keypoints.size(); ++k) {
    const int src = config.keypoint_remap[k];
    if (src < 0 || src >= static_cast<int>(keypoints.size())) {
      return;
    }
    remapped[k] = keypoints[src];
  }
  keypoints = remapped;
}

}  // namespace

std::vector<RawDetection> RobotPilotsLandmarkDecodeStrategy::decode(
    const std::vector<TensorOutput>& outputs,
    const ImageMeta& image_meta,
    const PostprocessConfig& config)
{
  if (outputs.empty()) {
    return {};
  }

  const auto& out = outputs[0];
  const auto& shape = out.info.shape;
  if (shape.size() != 3) {
    throw std::runtime_error(
      "RobotPilotsLandmarkDecodeStrategy: expected [1, N, 22], rank=" +
      std::to_string(shape.size()));
  }

  if (shape[0] != 1 || shape[2] != kFeatureCount) {
    throw std::runtime_error(
      "RobotPilotsLandmarkDecodeStrategy: expected shape [1, N, 22], got [" +
      std::to_string(shape[0]) + "," + std::to_string(shape[1]) + "," + std::to_string(shape[2]) + "]");
  }

  const int num_candidates = static_cast<int>(shape[1]);
  if (out.host_data.size() != static_cast<size_t>(num_candidates * kFeatureCount)) {
    throw std::runtime_error("RobotPilotsLandmarkDecodeStrategy: output buffer size mismatch");
  }

  std::vector<RawDetection> detections;
  detections.reserve(std::min(num_candidates, config.max_detections * 10));

  const float* data = out.host_data.data();
  for (int i = 0; i < num_candidates; ++i) {
    const float* row = data + i * kFeatureCount;

    const float confidence = sigmoid(row[kObjectnessIndex]);
    if (!std::isfinite(confidence) || confidence < config.conf_threshold) {
      continue;
    }

    int color_idx = -1;
    float color_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < kColorCount; ++c) {
      if (row[kColorStart + c] > color_score) {
        color_score = row[kColorStart + c];
        color_idx = c;
      }
    }

    if (color_idx < 0 || color_idx >= kColorCount) continue;
    if (color_idx == 2 || color_idx == 3) continue;  // gray/purple

    int number_idx = -1;
    float number_score = -std::numeric_limits<float>::infinity();
    for (int n = 0; n < kNumberCount; ++n) {
      if (row[kNumberStart + n] > number_score) {
        number_score = row[kNumberStart + n];
        number_idx = n;
      }
    }

    const int mapped_class_id = mapToClassId(color_idx, number_idx);
    if (mapped_class_id < 0) {
      continue;
    }

    RawDetection det;
    det.class_id = mapped_class_id;
    det.class_score = number_score;
    det.object_score = confidence;
    det.confidence = confidence;

    for (int k = 0; k < kKeypointDims / 2; ++k) {
      det.keypoints[k] = cv::Point2f(row[k * 2], row[k * 2 + 1]);
    }

    applyKeypointRemap(det.keypoints, config);
    restoreKeypoints(det.keypoints, image_meta);
    if (config.keypoint_auto_reorder) {
      det.keypoints = canonicalArmorKeypoints(det.keypoints);
    }
    det.bbox = bboxFromKeypoints(det.keypoints);
    detections.push_back(std::move(det));
  }

  return detections;
}

float RobotPilotsLandmarkDecodeStrategy::sigmoid(float x) {
  if (x >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-x));
  }
  const float e = std::exp(x);
  return e / (1.0F + e);
}

int RobotPilotsLandmarkDecodeStrategy::mapToClassId(int color_idx, int number_idx) {
  // number_idx:
  // 0:G, 1:1, 2:2, 3:3, 4:4, 5:5, 6:O, 7:Bs, 8:Bb
  int offset = -1;
  switch (number_idx) {
    case 1: offset = 0; break;  // 1
    case 2: offset = 1; break;  // 2
    case 3: offset = 2; break;  // 3
    case 4: offset = 3; break;  // 4
    case 5: offset = 4; break;  // 5
    case 6: offset = 5; break;  // outpost
    case 0: offset = 6; break;  // G -> sentry
    case 7: offset = 6; break;  // Bs -> sentry
    default: return -1;         // Bb / unknown
  }

  if (color_idx == 1) {       // red
    return 7 + offset;
  }
  if (color_idx == 0) {       // blue
    return offset;
  }
  return -1;
}

cv::Rect2f RobotPilotsLandmarkDecodeStrategy::bboxFromKeypoints(
    const std::array<cv::Point2f, 4>& keypoints) {
  float min_x = keypoints[0].x;
  float max_x = keypoints[0].x;
  float min_y = keypoints[0].y;
  float max_y = keypoints[0].y;

  for (int i = 1; i < 4; ++i) {
    min_x = std::min(min_x, keypoints[i].x);
    max_x = std::max(max_x, keypoints[i].x);
    min_y = std::min(min_y, keypoints[i].y);
    max_y = std::max(max_y, keypoints[i].y);
  }

  return {min_x, min_y, std::max(0.0F, max_x - min_x), std::max(0.0F, max_y - min_y)};
}

void RobotPilotsLandmarkDecodeStrategy::restoreKeypoints(
    std::array<cv::Point2f, 4>& keypoints,
    const ImageMeta& meta) {
  for (auto& kp : keypoints) {
    kp.x = (kp.x - meta.pad_left) / meta.scale_x;
    kp.y = (kp.y - meta.pad_top) / meta.scale_y;
    kp.x = std::max(0.0F, std::min(kp.x, static_cast<float>(meta.original_width)));
    kp.y = std::max(0.0F, std::min(kp.y, static_cast<float>(meta.original_height)));
  }
}

}  // namespace fyt::auto_aim
