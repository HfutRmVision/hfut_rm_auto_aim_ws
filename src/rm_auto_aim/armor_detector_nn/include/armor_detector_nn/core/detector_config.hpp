#ifndef ARMOR_DETECTOR_NN_DETECTOR_CONFIG_HPP_
#define ARMOR_DETECTOR_NN_DETECTOR_CONFIG_HPP_

#include <cstddef>
#include <string>
#include <vector>

namespace fyt::auto_aim {

enum class BackendType { ONNX_RUNTIME, OPENVINO, TENSORRT };
enum class Precision { FP32, FP16, INT8 };
enum class ColorFilterSource { MODEL, DISABLED };
enum class DetectMode { RED, BLUE, DISABLED };
enum class CopyPolicy { NEVER_COPY, COPY_ON_WRITE_DEBUG, ALWAYS_COPY };

inline std::string backendTypeToString(BackendType type) {
  switch (type) {
    case BackendType::ONNX_RUNTIME: return "onnxruntime";
    case BackendType::OPENVINO:     return "openvino";
    case BackendType::TENSORRT:     return "tensorrt";
    default:                        return "unknown";
  }
}

struct BackendConfig {
  BackendType type{BackendType::ONNX_RUNTIME};
  std::string device{"cpu"};
  Precision precision{Precision::FP32};
  std::string model_path;
  std::string engine_path;
  std::string openvino_xml_path;
  std::string openvino_bin_path;
  std::string input_name{"images"};
  std::vector<std::string> output_names{"output0"};
  int warmup_iterations{10};
  int num_threads{2};
  bool use_pinned_memory{true};

  bool allow_fallback{false};
  BackendType fallback_type{BackendType::ONNX_RUNTIME};
};

struct PreprocessConfig {
  int input_width{640};
  int input_height{640};
  std::string input_color{"rgb"};
  bool normalize{true};
  std::vector<double> mean{0.0, 0.0, 0.0};
  std::vector<double> std{255.0, 255.0, 255.0};
  float pad_value{114.0F};
};

struct PostprocessConfig {
  std::string strategy{"ultralytics_pose"};
  int num_classes{14};
  int num_keypoints{4};
  int keypoint_dims{2};
  int bbox_offset{0};
  int class_offset{4};
  int keypoint_offset{18};
  float conf_threshold{0.35F};
  float nms_threshold{0.45F};
  int max_detections{32};
  bool class_agnostic_nms{false};
  std::vector<int> keypoint_remap{1, 0, 3, 2};
  bool keypoint_auto_reorder{false};
};

struct QualityFilterConfig {
  bool enabled{false};
  double min_armor_ratio{1.0};
  double max_armor_ratio{5.0};
  double max_side_ratio{1.5};
  double max_rectangular_error_deg{25.0};
  double min_lightbar_length_px{2.0};
  double min_area_px{20.0};

  bool deduplicate_enabled{true};
  double duplicate_iou_threshold{0.60};
  double duplicate_keypoint_mean_dist_px{8.0};
};

struct LabelMapConfig {
  std::string path;
};

struct SingleYawConfig {
  int max_iterations{15};
  double huber_delta{3.0};
  double pitch_deg_default{15.0};
  double roll_deg_default{0.0};
  bool outpost_pitch_sign{true};
};

struct SlidingWindowConfig {
  int window_size{8};
  int min_frames{4};
  double max_time_span_ms{300};
  double max_solver_time_ms{2.0};
  int max_opt_iters{20};
  double sigma_prior_xy{0.08};
  double sigma_prior_z{0.15};
  double sigma_prior_yaw{0.35};
  double sigma_smooth_xy{0.05};
  double sigma_smooth_z{0.10};
  double sigma_smooth_yaw{0.10};
  double sigma_kp_min{1.0};
  double sigma_kp_scale{5.0};
  double huber_delta{3.0};
};

struct RefinerConfig {
  std::string mode{"single_yaw"};  // none | single_yaw | sliding_window
};

struct GateConfig {
  double max_raw_reproj_error{0.0};
  double max_reproj_error{3.0};
  double max_pose_delta_m{0.20};
  double max_yaw_delta_deg{20.0};
  bool require_finite{true};
};

struct DepthCorrectionConfig {
  bool enabled{false};
  double min_depth_delta_m{0.08};
  double blend_alpha{0.85};
  double max_correction_m{0.60};
  double max_scale{1.35};
  double min_lightbar_length_px{4.0};
};

struct TrackerConfig {
  std::string strategy{"internal_iou"};
  double iou_threshold{0.30};
  int max_missed{15};
  int min_hits{2};
  int max_center_dist_px{120};
};

struct CornerRefineConfig {
  bool enabled{false};
  std::string method{"sp25_lightbar"};
  bool apply_on_confirmed_only{true};
  int max_targets_per_frame{1};
  double time_budget_ms{2.0};
  double roi_expand_ratio{1.1};
  int min_bright_points{50};
  double pca_stability_threshold{0.85};
  double max_aspect_ratio{5.0};
  double min_aspect_ratio{1.5};
  double max_corner_shift_px{5.0};
  double max_mean_corner_shift_px{2.5};
  double max_refined_center_shift_px{2.0};
  double min_refine_quality{0.88};
  bool preserve_perspective{true};
  double max_edge_angle_delta_deg{12.0};
  double max_area_ratio_delta{0.12};
  double max_length_ratio_delta{0.20};
  double full_roi_expand_ratio{1.45};
  double binary_threshold{150.0};
  double min_contour_area_px{6.0};
  double min_lightbar_length_px{6.0};
  double min_lightbar_ratio{1.4};
  double max_lightbar_ratio{20.0};
  double max_lightbar_angle_error_deg{45.0};
  double max_rectangular_error_deg{0.0};
  double max_side_ratio{2.2};
  double max_lightbar_match_error_px{26.0};
  double max_pair_center_shift_px{45.0};
};

struct TraditionalFusionConfig {
  bool enabled{false};
  std::string strategy{"fallback"};  // fallback | always
  int min_nn_detections{1};
  float confidence_scale{0.92F};

  int binary_thres{160};
  double light_min_ratio{0.08};
  double light_max_ratio{0.4};
  double light_max_angle{40.0};
  int light_color_diff_thresh{25};

  double armor_min_light_ratio{0.6};
  double armor_min_small_center_distance{0.8};
  double armor_max_small_center_distance{3.2};
  double armor_min_large_center_distance{3.2};
  double armor_max_large_center_distance{5.0};
  double armor_max_angle{35.0};

  bool use_classifier{true};
  double classifier_threshold{0.7};
  std::vector<std::string> ignore_classes{"negative"};
  bool use_pca{true};
};

struct PoseConfig {
  std::string pnp_method{"ippe"};
  double small_armor_width{0.133};
  double small_armor_height{0.050};
  double large_armor_width{0.225};
  double large_armor_height{0.050};

  RefinerConfig refiner;
  SingleYawConfig single_yaw;
  SlidingWindowConfig sliding;
  GateConfig gate;
  DepthCorrectionConfig depth_correction;
  bool force_pnp_rotate_180{false};
};

struct RuntimeConfig {
  ColorFilterSource color_filter_source{ColorFilterSource::MODEL};
  bool publish_empty{true};
  CopyPolicy copy_policy{CopyPolicy::COPY_ON_WRITE_DEBUG};
  bool profile{true};
};

struct DetectorConfig {
  std::string target_frame{"odom"};

  BackendConfig backend;
  PreprocessConfig preprocess;
  PostprocessConfig postprocess;
  LabelMapConfig label_map;
  PoseConfig pose;
  RuntimeConfig runtime;
  QualityFilterConfig quality_filter;

  TrackerConfig tracker;
  CornerRefineConfig corner_refine;
  TraditionalFusionConfig traditional_fusion;
};

struct BackendInfo {
  std::string backend_name;
  std::string precision;
  int min_batch_size{1};
  int max_batch_size{1};
  bool dynamic_batch{false};
};

}  // namespace fyt::auto_aim

#endif
