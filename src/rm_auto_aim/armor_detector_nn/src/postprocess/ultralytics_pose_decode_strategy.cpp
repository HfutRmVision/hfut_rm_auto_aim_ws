#include "armor_detector_nn/postprocess/ultralytics_pose_decode_strategy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "armor_detector_nn/postprocess/detection_quality_filter.hpp"

namespace fyt::auto_aim {

std::vector<RawDetection> UltralyticsPoseDecodeStrategy::decode(
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
      "UltralyticsPoseDecodeStrategy: expected 3D output [1, C, N], got rank " +
      std::to_string(shape.size()));
  }

  int batch   = static_cast<int>(shape[0]);
  int channels = static_cast<int>(shape[1]);
  int num_candidates = static_cast<int>(shape[2]);

  if (batch != 1) {
    throw std::runtime_error("UltralyticsPoseDecodeStrategy: batch != 1 not supported");
  }
  if (channels < config.class_offset + config.num_classes) {
    throw std::runtime_error(
      "UltralyticsPoseDecodeStrategy: output channels (" +
      std::to_string(channels) + ") too few for class_offset + num_classes");
  }

  const float* data = out.host_data.data();
  std::vector<RawDetection> detections;
  detections.reserve(std::min(num_candidates, config.max_detections * 10));

  for (int i = 0; i < num_candidates; ++i) {
    RawDetection det;
    if (decodeCandidate(data, i, num_candidates, config, det)) {
      restoreBbox(det, image_meta, config);
      restoreKeypoints(det, image_meta, config);
      detections.push_back(std::move(det));
    }
  }

  return detections;
}

bool UltralyticsPoseDecodeStrategy::decodeCandidate(
    const float* data,
    int index,
    int num_candidates,
    const PostprocessConfig& config,
    RawDetection& detection) const
{
  int C = config.class_offset + config.num_classes +
          config.num_keypoints * config.keypoint_dims;
  if (config.bbox_offset + 4 > C) C = config.bbox_offset + 4;

  // Read class scores and find max
  float max_score = 0.0F;
  int max_class = -1;
  for (int c = 0; c < config.num_classes; ++c) {
    float score = data[(config.class_offset + c) * num_candidates + index];
    if (score > max_score) {
      max_score = score;
      max_class = c;
    }
  }

  if (!std::isfinite(max_score) || max_score < config.conf_threshold) {
    return false;
  }

  detection.class_id = max_class;
  detection.class_score = max_score;
  detection.object_score = 1.0F;
  detection.confidence = max_score;

  // Read bbox: cx, cy, w, h
  float cx = data[(config.bbox_offset + 0) * num_candidates + index];
  float cy = data[(config.bbox_offset + 1) * num_candidates + index];
  float w  = data[(config.bbox_offset + 2) * num_candidates + index];
  float h  = data[(config.bbox_offset + 3) * num_candidates + index];

  detection.bbox = cv::Rect2f(cx - w / 2, cy - h / 2, w, h);

  // Read keypoints
  for (int k = 0; k < config.num_keypoints; ++k) {
    float kx = data[(config.keypoint_offset + k * config.keypoint_dims + 0) * num_candidates + index];
    float ky = data[(config.keypoint_offset + k * config.keypoint_dims + 1) * num_candidates + index];
    detection.keypoints[k] = cv::Point2f(kx, ky);
  }

  // Apply keypoint remap to canonical order
  if (!config.keypoint_remap.empty() &&
      config.keypoint_remap.size() == static_cast<size_t>(config.num_keypoints)) {
    std::array<cv::Point2f, 4> remapped = detection.keypoints;
    bool valid_remap = true;
    for (int k = 0; k < config.num_keypoints; ++k) {
      int src = config.keypoint_remap[k];
      if (src >= 0 && src < config.num_keypoints) {
        remapped[k] = detection.keypoints[src];
      } else {
        valid_remap = false;
        break;
      }
    }
    if (valid_remap) {
      detection.keypoints = remapped;
    }
  }

  return true;
}

void UltralyticsPoseDecodeStrategy::restoreBbox(
    RawDetection& detection,
    const ImageMeta& meta,
    const PostprocessConfig& /*config*/) const
{
  // Convert from model-space (letterbox) to original image coordinates
  float x = detection.bbox.x;
  float y = detection.bbox.y;
  float w = detection.bbox.width;
  float h = detection.bbox.height;

  // Undo letterbox padding and scaling
  x = (x - meta.pad_left) / meta.scale_x;
  y = (y - meta.pad_top)  / meta.scale_y;
  w = w / meta.scale_x;
  h = h / meta.scale_y;

  // Clamp to image bounds
  x = std::max(0.0F, x);
  y = std::max(0.0F, y);
  w = std::min(w, meta.original_width - x);
  h = std::min(h, meta.original_height - y);

  detection.bbox = cv::Rect2f(x, y, w, h);
}

void UltralyticsPoseDecodeStrategy::restoreKeypoints(
    RawDetection& detection,
    const ImageMeta& meta,
    const PostprocessConfig& config) const
{
  for (auto& kp : detection.keypoints) {
    kp.x = (kp.x - meta.pad_left) / meta.scale_x;
    kp.y = (kp.y - meta.pad_top)  / meta.scale_y;

    kp.x = std::max(0.0F, std::min(kp.x, static_cast<float>(meta.original_width)));
    kp.y = std::max(0.0F, std::min(kp.y, static_cast<float>(meta.original_height)));
  }

  if (config.keypoint_auto_reorder) {
    detection.keypoints = canonicalArmorKeypoints(detection.keypoints);
  }
}

}  // namespace fyt::auto_aim
