#include "armor_detector_nn/armor_detector_nn_node.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/create_timer_ros.h>

#include "rm_utils/assert.hpp"
#include "rm_utils/logger/log.hpp"
#include "rm_utils/url_resolver.hpp"

#include "armor_detector/light_corner_corrector.hpp"
#include "armor_detector/number_classifier.hpp"
#include "armor_detector_nn/core/pose_refine/pose_refiner.hpp"
#include "armor_detector_nn/core/tracker/internal_iou_tracker_strategy.hpp"
#include "armor_detector_nn/core/corner_refine/roi_pca_corner_refiner.hpp"
#include "armor_detector_nn/postprocess/detection_quality_filter.hpp"

namespace fyt::auto_aim {

namespace {

float rectIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float w = std::max(0.0F, x2 - x1);
  const float h = std::max(0.0F, y2 - y1);
  const float inter = w * h;
  const float area_a = std::max(0.0F, a.width) * std::max(0.0F, a.height);
  const float area_b = std::max(0.0F, b.width) * std::max(0.0F, b.height);
  return inter / (area_a + area_b - inter + 1e-6F);
}

float meanKeypointDistance(
    const std::array<cv::Point2f, 4>& a,
    const std::array<cv::Point2f, 4>& b) {
  float sum = 0.0F;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<float>(cv::norm(a[i] - b[i]));
  }
  return sum / static_cast<float>(a.size());
}

cv::Rect2f bboxFromKeypoints(const std::array<cv::Point2f, 4>& keypoints) {
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

std::string traditionalModelLabel(const ArmorDetection& detection) {
  const char color = detection.color == fyt::EnemyColor::RED ? 'R' : 'B';
  if (detection.publish_number == "outpost") return std::string("T") + color + "O";
  if (detection.publish_number == "sentry") return std::string("T") + color + "S";
  if (detection.publish_number == "base") return std::string("T") + color + "B";
  return std::string("T") + color + detection.publish_number;
}

}  // namespace

ArmorDetectorNNNode::ArmorDetectorNNNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("armor_detector", options)
{
  FYT_REGISTER_LOGGER("armor_detector", "~/fyt2024-log", INFO);
  FYT_REGISTER_LOGGER("armor_detector_nn", "~/fyt2024-log", INFO);
  FYT_INFO("armor_detector", "Starting ArmorDetectorNNNode (neural-network detector)");

  initializeParameters();
  validateParameters();

  // --- detector ---
  detector_ = std::make_unique<ArmorDetectorNN>(config_);
  if (!detector_->initialize()) {
    FYT_ERROR("armor_detector", "Failed to initialize detector. "
              "Node will start but detection is disabled.");
  }

  if (config_.traditional_fusion.enabled) {
    initializeTraditionalDetector();
  }

  // --- pose estimator ---
  pose_estimator_adapter_ = std::make_unique<ArmorPoseEstimatorAdapter>(config_.pose);
  // Reference estimator: aligned with armor_detector-like baseline
  // (PnP + IPPE disambiguation, no pose refiner).
  {
    auto ref_pose_cfg = config_.pose;
    ref_pose_cfg.refiner.mode = "none";
    ref_pose_cfg.depth_correction.enabled = false;
    pose_estimator_reference_adapter_ =
      std::make_unique<ArmorPoseEstimatorAdapter>(ref_pose_cfg);
  }

  // --- Phase 1 / Phase 4: pose refiner ---
  if (config_.pose.refiner.mode == "sliding_window") {
    auto refiner = std::make_shared<SlidingWindowRefiner>(
      config_.pose.sliding, config_.pose.single_yaw, config_.pose.gate);
    pose_estimator_adapter_->setRefiner(refiner);
    FYT_INFO("armor_detector", "Pose refiner initialized: mode=sliding_window");
  } else if (config_.pose.refiner.mode == "single_yaw") {
    auto refiner = std::make_shared<SingleYawRefiner>(
      config_.pose.single_yaw, config_.pose.gate);
    pose_estimator_adapter_->setRefiner(refiner);
    FYT_INFO("armor_detector", "Pose refiner initialized: mode=single_yaw");
  }

  // --- Phase 2: tracker ---
  if (config_.tracker.strategy == "internal_iou") {
    tracker_ = std::make_shared<InternalIoUTrackerStrategy>(config_.tracker);
    FYT_INFO("armor_detector", "Tracker initialized: strategy=internal_iou");
  }

  // --- Phase 3: corner refiner ---
  if (config_.corner_refine.enabled) {
    corner_refiner_ = std::make_shared<RoiPcaCornerRefiner>(config_.corner_refine);
    FYT_INFO("armor_detector", "Corner refiner initialized (enabled)");
  }

  // --- debug drawer ---
  debug_drawer_ = std::make_unique<DebugDrawer>();

  // --- tf ---
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // --- subscriptions ---
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "image_raw", rclcpp::SensorDataQoS(),
    std::bind(&ArmorDetectorNNNode::imageCallback, this, std::placeholders::_1));

  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info", rclcpp::SensorDataQoS(),
    std::bind(&ArmorDetectorNNNode::cameraInfoCallback, this, std::placeholders::_1));

  // --- publishers ---
  armors_pub_ = this->create_publisher<rm_interfaces::msg::Armors>(
    "armor_detector/armors", rclcpp::SensorDataQoS());

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "armor_detector/marker", rclcpp::SensorDataQoS());

  // --- service ---
  set_mode_srv_ = this->create_service<rm_interfaces::srv::SetMode>(
    "armor_detector/set_mode",
    std::bind(&ArmorDetectorNNNode::setModeCallback, this,
              std::placeholders::_1, std::placeholders::_2));

  // --- debug ---
  if (debug_) {
    createDebugPublishers();
  }

  // --- profiler ---
  if (config_.runtime.profile) {
    profiler_ = std::make_unique<Profiler>();
  }

  // --- parameter callback ---
  on_set_parameters_callback_handle_ =
    this->add_on_set_parameters_callback(
      std::bind(&ArmorDetectorNNNode::onSetParameters, this, std::placeholders::_1));

  // --- heartbeat ---
  heartbeat_ = HeartBeatPublisher::create(this);

  FYT_INFO("armor_detector", "ArmorDetectorNNNode initialized. Mode: DISABLED.");
}

void ArmorDetectorNNNode::initializeParameters() {
  debug_ = this->declare_parameter("debug", true);
  debug_pose_compare_ = this->declare_parameter("debug_pose_compare", false);
  publish_in_target_frame_ =
    this->declare_parameter("publish_in_target_frame", false);
  config_.target_frame = this->declare_parameter("target_frame", "odom");

  // backend
  {
    std::string type_str = this->declare_parameter("backend.type", "onnxruntime");
    config_.backend.device     = this->declare_parameter("backend.device", "cpu");
    std::string prec_str       = this->declare_parameter("backend.precision", "fp32");
    config_.backend.model_path      = this->declare_parameter("backend.model_path", "");
    config_.backend.engine_path     = this->declare_parameter("backend.engine_path", "");
    config_.backend.openvino_xml_path = this->declare_parameter("backend.openvino_model_xml", "");
    config_.backend.openvino_bin_path = this->declare_parameter("backend.openvino_model_bin", "");
    config_.backend.input_name    = this->declare_parameter("backend.input_name", "images");
    config_.backend.output_names  = this->declare_parameter("backend.output_names",
                                        std::vector<std::string>{"output0"});
    config_.backend.warmup_iterations = this->declare_parameter("backend.warmup_iterations", 10);
    config_.backend.num_threads    = this->declare_parameter("backend.num_threads", 2);
    config_.backend.use_pinned_memory = this->declare_parameter("backend.use_pinned_memory", true);
    config_.backend.allow_fallback = this->declare_parameter("backend.allow_fallback", false);
    std::string fallback_str       = this->declare_parameter("backend.fallback_type", "onnxruntime");

    if (type_str == "openvino") config_.backend.type = BackendType::OPENVINO;
    else if (type_str == "tensorrt") config_.backend.type = BackendType::TENSORRT;
    else config_.backend.type = BackendType::ONNX_RUNTIME;

    if (prec_str == "fp16") config_.backend.precision = Precision::FP16;
    else if (prec_str == "int8") config_.backend.precision = Precision::INT8;
    else config_.backend.precision = Precision::FP32;

    if (fallback_str == "openvino") config_.backend.fallback_type = BackendType::OPENVINO;
    else if (fallback_str == "tensorrt") config_.backend.fallback_type = BackendType::TENSORRT;
    else config_.backend.fallback_type = BackendType::ONNX_RUNTIME;
  }

  // preprocess
  {
    config_.preprocess.input_width  = this->declare_parameter("preprocess.input_width", 640);
    config_.preprocess.input_height = this->declare_parameter("preprocess.input_height", 640);
    config_.preprocess.input_layout = this->declare_parameter("preprocess.input_layout", "nchw");
    config_.preprocess.input_color  = this->declare_parameter("preprocess.input_color", "rgb");
    config_.preprocess.resize_mode  = this->declare_parameter("preprocess.resize_mode", "letterbox");
    config_.preprocess.normalize    = this->declare_parameter("preprocess.normalize", true);
    config_.preprocess.mean = this->declare_parameter("preprocess.mean",
                                  std::vector<double>{0.0, 0.0, 0.0});
    config_.preprocess.std  = this->declare_parameter("preprocess.std",
                                  std::vector<double>{255.0, 255.0, 255.0});
    config_.preprocess.pad_value = static_cast<float>(
      this->declare_parameter("preprocess.pad_value", 114.0));

    // mean/std already stored with correct type
  }

  // postprocess
  {
    config_.postprocess.strategy = this->declare_parameter("postprocess.strategy", "ultralytics_pose");
    config_.postprocess.output_layout = this->declare_parameter("postprocess.output_layout", "channels_first");
    config_.postprocess.num_classes     = this->declare_parameter("postprocess.num_classes", 14);
    config_.postprocess.num_keypoints   = this->declare_parameter("postprocess.num_keypoints", 4);
    config_.postprocess.keypoint_dims   = this->declare_parameter("postprocess.keypoint_dims", 2);
    config_.postprocess.bbox_offset     = this->declare_parameter("postprocess.bbox_offset", 0);
    config_.postprocess.class_offset    = this->declare_parameter("postprocess.class_offset", 4);
    config_.postprocess.keypoint_offset = this->declare_parameter("postprocess.keypoint_offset", 18);
    config_.postprocess.box_format      = this->declare_parameter("postprocess.box_format", "cxcywh");
    config_.postprocess.conf_threshold  = this->declare_parameter("postprocess.conf_threshold", 0.35);
    config_.postprocess.nms_threshold   = this->declare_parameter("postprocess.nms_threshold", 0.45);
    config_.postprocess.max_detections   = this->declare_parameter("postprocess.max_detections", 32);
    config_.postprocess.class_agnostic_nms = this->declare_parameter("postprocess.class_agnostic_nms", false);
    {
      auto remap = this->declare_parameter("postprocess.keypoint_remap",
                    std::vector<int64_t>{1, 0, 3, 2});
      config_.postprocess.keypoint_remap.assign(remap.begin(), remap.end());
    }
    config_.postprocess.head_already_applied = this->declare_parameter("postprocess.head_already_applied", true);
    config_.postprocess.keypoint_auto_reorder = this->declare_parameter("postprocess.keypoint_auto_reorder", false);
  }

  // label_map
  {
    config_.label_map.path = this->declare_parameter("label_map.path", "");
  }

  // quality_filter: geometry gate and same-target duplicate suppression
  {
    config_.quality_filter.enabled =
      this->declare_parameter("quality_filter.enabled", false);
    config_.quality_filter.min_armor_ratio =
      this->declare_parameter("quality_filter.min_armor_ratio", 1.0);
    config_.quality_filter.max_armor_ratio =
      this->declare_parameter("quality_filter.max_armor_ratio", 5.0);
    config_.quality_filter.max_side_ratio =
      this->declare_parameter("quality_filter.max_side_ratio", 1.5);
    config_.quality_filter.max_rectangular_error_deg =
      this->declare_parameter("quality_filter.max_rectangular_error_deg", 25.0);
    config_.quality_filter.min_lightbar_length_px =
      this->declare_parameter("quality_filter.min_lightbar_length_px", 2.0);
    config_.quality_filter.min_area_px =
      this->declare_parameter("quality_filter.min_area_px", 20.0);
    config_.quality_filter.deduplicate_enabled =
      this->declare_parameter("quality_filter.deduplicate_enabled", true);
    config_.quality_filter.duplicate_iou_threshold =
      this->declare_parameter("quality_filter.duplicate_iou_threshold", 0.60);
    config_.quality_filter.duplicate_keypoint_mean_dist_px =
      this->declare_parameter("quality_filter.duplicate_keypoint_mean_dist_px", 8.0);
  }

  // pose
  {
    config_.pose.pnp_method = this->declare_parameter("pose.pnp_method", "ippe");
    config_.pose.small_armor_width  = this->declare_parameter("pose.small_armor_width", 0.133);
    config_.pose.small_armor_height = this->declare_parameter("pose.small_armor_height", 0.050);
    config_.pose.large_armor_width  = this->declare_parameter("pose.large_armor_width", 0.225);
    config_.pose.large_armor_height = this->declare_parameter("pose.large_armor_height", 0.050);

    // Phase 1 — refiner
    config_.pose.refiner.mode = this->declare_parameter("pose.refiner.mode", "single_yaw");
    config_.pose.single_yaw.max_iterations = this->declare_parameter("pose.single_yaw.max_iterations", 15);
    config_.pose.single_yaw.huber_delta = this->declare_parameter("pose.single_yaw.huber_delta", 3.0);
    config_.pose.single_yaw.pitch_deg_default = this->declare_parameter("pose.single_yaw.pitch_deg_default", 15.0);
    config_.pose.single_yaw.roll_deg_default = this->declare_parameter("pose.single_yaw.roll_deg_default", 0.0);
    config_.pose.single_yaw.outpost_pitch_sign = this->declare_parameter("pose.single_yaw.outpost_pitch_sign", true);

    // Optional: force PnP result rotate 180 degrees (workaround for select-solution ambiguity)
    config_.pose.force_pnp_rotate_180 = this->declare_parameter("pose.force_pnp_rotate_180", false);

    // Phase 4 — sliding-window refiner
    config_.pose.sliding.window_size = this->declare_parameter("pose.sliding.window_size", 8);
    config_.pose.sliding.min_frames = this->declare_parameter("pose.sliding.min_frames", 4);
    config_.pose.sliding.max_time_span_ms = this->declare_parameter("pose.sliding.max_time_span_ms", 300.0);
    config_.pose.sliding.max_solver_time_ms = this->declare_parameter("pose.sliding.max_solver_time_ms", 2.0);
    config_.pose.sliding.max_opt_iters = this->declare_parameter("pose.sliding.max_opt_iters", 20);
    config_.pose.sliding.sigma_prior_xy = this->declare_parameter("pose.sliding.sigma_prior_xy", 0.08);
    config_.pose.sliding.sigma_prior_z = this->declare_parameter("pose.sliding.sigma_prior_z", 0.15);
    config_.pose.sliding.sigma_prior_yaw = this->declare_parameter("pose.sliding.sigma_prior_yaw", 0.35);
    config_.pose.sliding.sigma_smooth_xy = this->declare_parameter("pose.sliding.sigma_smooth_xy", 0.05);
    config_.pose.sliding.sigma_smooth_z = this->declare_parameter("pose.sliding.sigma_smooth_z", 0.10);
    config_.pose.sliding.sigma_smooth_yaw = this->declare_parameter("pose.sliding.sigma_smooth_yaw", 0.10);
    config_.pose.sliding.sigma_kp_min = this->declare_parameter("pose.sliding.sigma_kp_min", 1.0);
    config_.pose.sliding.sigma_kp_scale = this->declare_parameter("pose.sliding.sigma_kp_scale", 5.0);
    config_.pose.sliding.huber_delta = this->declare_parameter("pose.sliding.huber_delta", 3.0);

    config_.pose.gate.max_raw_reproj_error = this->declare_parameter("pose.gate.max_raw_reproj_error", 0.0);
    config_.pose.gate.max_reproj_error = this->declare_parameter("pose.gate.max_reproj_error", 3.0);
    config_.pose.gate.max_pose_delta_m = this->declare_parameter("pose.gate.max_pose_delta_m", 0.20);
    config_.pose.gate.max_yaw_delta_deg = this->declare_parameter("pose.gate.max_yaw_delta_deg", 20.0);
    config_.pose.gate.require_finite = this->declare_parameter("pose.gate.require_finite", true);

    config_.pose.depth_correction.enabled =
      this->declare_parameter("pose.depth_correction.enabled", false);
    config_.pose.depth_correction.min_depth_delta_m =
      this->declare_parameter("pose.depth_correction.min_depth_delta_m", 0.08);
    config_.pose.depth_correction.blend_alpha =
      this->declare_parameter("pose.depth_correction.blend_alpha", 0.85);
    config_.pose.depth_correction.max_correction_m =
      this->declare_parameter("pose.depth_correction.max_correction_m", 0.60);
    config_.pose.depth_correction.max_scale =
      this->declare_parameter("pose.depth_correction.max_scale", 1.35);
    config_.pose.depth_correction.min_lightbar_length_px =
      this->declare_parameter("pose.depth_correction.min_lightbar_length_px", 4.0);
  }

  // tracker (Phase 2)
  {
    config_.tracker.strategy = this->declare_parameter("tracker.strategy", "internal_iou");
    config_.tracker.iou_threshold = this->declare_parameter("tracker.iou_threshold", 0.30);
    config_.tracker.max_missed = this->declare_parameter("tracker.max_missed", 15);
    config_.tracker.min_hits = this->declare_parameter("tracker.min_hits", 2);
    config_.tracker.max_center_dist_px = this->declare_parameter("tracker.max_center_dist_px", 120);
  }

  // corner_refine (Phase 3)
  {
    config_.corner_refine.enabled = this->declare_parameter("corner_refine.enabled", false);
    config_.corner_refine.method = this->declare_parameter("corner_refine.method", "sp25_lightbar");
    config_.corner_refine.apply_on_confirmed_only = this->declare_parameter("corner_refine.apply_on_confirmed_only", true);
    config_.corner_refine.max_targets_per_frame = this->declare_parameter("corner_refine.max_targets_per_frame", 1);
    config_.corner_refine.time_budget_ms = this->declare_parameter("corner_refine.time_budget_ms", 2.0);
    config_.corner_refine.roi_expand_ratio = this->declare_parameter("corner_refine.roi_expand_ratio", 1.1);
    config_.corner_refine.min_bright_points = this->declare_parameter("corner_refine.min_bright_points", 50);
    config_.corner_refine.pca_stability_threshold = this->declare_parameter("corner_refine.pca_stability_threshold", 0.85);
    config_.corner_refine.max_aspect_ratio = this->declare_parameter("corner_refine.max_aspect_ratio", 5.0);
    config_.corner_refine.min_aspect_ratio = this->declare_parameter("corner_refine.min_aspect_ratio", 1.5);
    config_.corner_refine.max_corner_shift_px = this->declare_parameter("corner_refine.max_corner_shift_px", 5.0);
    config_.corner_refine.max_mean_corner_shift_px = this->declare_parameter("corner_refine.max_mean_corner_shift_px", 2.5);
    config_.corner_refine.max_refined_center_shift_px = this->declare_parameter("corner_refine.max_refined_center_shift_px", 2.0);
    config_.corner_refine.min_refine_quality = this->declare_parameter("corner_refine.min_refine_quality", 0.88);
    config_.corner_refine.preserve_perspective = this->declare_parameter("corner_refine.preserve_perspective", true);
    config_.corner_refine.max_edge_angle_delta_deg = this->declare_parameter("corner_refine.max_edge_angle_delta_deg", 12.0);
    config_.corner_refine.max_area_ratio_delta = this->declare_parameter("corner_refine.max_area_ratio_delta", 0.12);
    config_.corner_refine.max_length_ratio_delta = this->declare_parameter("corner_refine.max_length_ratio_delta", 0.20);
    config_.corner_refine.full_roi_expand_ratio = this->declare_parameter("corner_refine.full_roi_expand_ratio", 1.45);
    config_.corner_refine.binary_threshold = this->declare_parameter("corner_refine.binary_threshold", 150.0);
    config_.corner_refine.min_contour_area_px = this->declare_parameter("corner_refine.min_contour_area_px", 6.0);
    config_.corner_refine.min_lightbar_length_px = this->declare_parameter("corner_refine.min_lightbar_length_px", 6.0);
    config_.corner_refine.min_lightbar_ratio = this->declare_parameter("corner_refine.min_lightbar_ratio", 1.4);
    config_.corner_refine.max_lightbar_ratio = this->declare_parameter("corner_refine.max_lightbar_ratio", 20.0);
    config_.corner_refine.max_lightbar_angle_error_deg = this->declare_parameter("corner_refine.max_lightbar_angle_error_deg", 45.0);
    config_.corner_refine.max_rectangular_error_deg = this->declare_parameter("corner_refine.max_rectangular_error_deg", 0.0);
    config_.corner_refine.max_side_ratio = this->declare_parameter("corner_refine.max_side_ratio", 2.2);
    config_.corner_refine.max_lightbar_match_error_px = this->declare_parameter("corner_refine.max_lightbar_match_error_px", 26.0);
    config_.corner_refine.max_pair_center_shift_px = this->declare_parameter("corner_refine.max_pair_center_shift_px", 45.0);
  }

  // traditional_fusion: reuse armor_detector traditional lightbar pipeline as
  // fallback/augmentation candidates before tracker and PnP.
  {
    auto& tf = config_.traditional_fusion;
    tf.enabled = this->declare_parameter("traditional_fusion.enabled", false);
    tf.strategy = this->declare_parameter("traditional_fusion.strategy", "fallback");
    tf.min_nn_detections =
      this->declare_parameter("traditional_fusion.min_nn_detections", 1);
    tf.confidence_scale =
      static_cast<float>(this->declare_parameter(
        "traditional_fusion.confidence_scale", 0.92));

    tf.binary_thres =
      this->declare_parameter("traditional_fusion.binary_thres", 160);
    tf.light_min_ratio =
      this->declare_parameter("traditional_fusion.light.min_ratio", 0.08);
    tf.light_max_ratio =
      this->declare_parameter("traditional_fusion.light.max_ratio", 0.4);
    tf.light_max_angle =
      this->declare_parameter("traditional_fusion.light.max_angle", 40.0);
    tf.light_color_diff_thresh =
      this->declare_parameter("traditional_fusion.light.color_diff_thresh", 25);

    tf.armor_min_light_ratio =
      this->declare_parameter("traditional_fusion.armor.min_light_ratio", 0.6);
    tf.armor_min_small_center_distance =
      this->declare_parameter("traditional_fusion.armor.min_small_center_distance", 0.8);
    tf.armor_max_small_center_distance =
      this->declare_parameter("traditional_fusion.armor.max_small_center_distance", 3.2);
    tf.armor_min_large_center_distance =
      this->declare_parameter("traditional_fusion.armor.min_large_center_distance", 3.2);
    tf.armor_max_large_center_distance =
      this->declare_parameter("traditional_fusion.armor.max_large_center_distance", 5.0);
    tf.armor_max_angle =
      this->declare_parameter("traditional_fusion.armor.max_angle", 35.0);

    tf.use_classifier =
      this->declare_parameter("traditional_fusion.use_classifier", true);
    tf.classifier_threshold =
      this->declare_parameter("traditional_fusion.classifier_threshold", 0.7);
    tf.ignore_classes =
      this->declare_parameter("traditional_fusion.ignore_classes",
                              std::vector<std::string>{"negative"});
    tf.use_pca =
      this->declare_parameter("traditional_fusion.use_pca", true);
  }

  // runtime
  {
    std::string cfs = this->declare_parameter("runtime.color_filter_source", "model");
    if (cfs == "disabled") config_.runtime.color_filter_source = ColorFilterSource::DISABLED;
    else config_.runtime.color_filter_source = ColorFilterSource::MODEL;
    config_.runtime.publish_empty = this->declare_parameter("runtime.publish_empty", true);
    std::string copy_policy_str = this->declare_parameter("runtime.copy_policy", "copy_on_write_debug");
    config_.runtime.profile = this->declare_parameter("runtime.profile", true);

    if (copy_policy_str == "never_copy") config_.runtime.copy_policy = CopyPolicy::NEVER_COPY;
    else if (copy_policy_str == "always_copy") config_.runtime.copy_policy = CopyPolicy::ALWAYS_COPY;
    else config_.runtime.copy_policy = CopyPolicy::COPY_ON_WRITE_DEBUG;
  }
}

void ArmorDetectorNNNode::validateParameters() {
  if (config_.postprocess.conf_threshold < 0.0 || config_.postprocess.conf_threshold > 1.0) {
    FYT_ERROR("armor_detector", "conf_threshold out of range, using default 0.35");
    config_.postprocess.conf_threshold = 0.35F;
  }
  if (config_.postprocess.nms_threshold < 0.0 || config_.postprocess.nms_threshold > 1.0) {
    FYT_ERROR("armor_detector", "nms_threshold out of range, using default 0.45");
    config_.postprocess.nms_threshold = 0.45F;
  }
  if (config_.quality_filter.min_armor_ratio <= 0.0) {
    config_.quality_filter.min_armor_ratio = 1.0;
  }
  if (config_.quality_filter.max_armor_ratio < config_.quality_filter.min_armor_ratio) {
    config_.quality_filter.max_armor_ratio = config_.quality_filter.min_armor_ratio;
  }
  if (config_.quality_filter.max_side_ratio < 1.0) {
    config_.quality_filter.max_side_ratio = 1.0;
  }
  if (config_.quality_filter.max_rectangular_error_deg < 0.0) {
    config_.quality_filter.max_rectangular_error_deg = 25.0;
  }
  if (config_.quality_filter.min_lightbar_length_px < 0.0) {
    config_.quality_filter.min_lightbar_length_px = 0.0;
  }
  if (config_.quality_filter.min_area_px < 0.0) {
    config_.quality_filter.min_area_px = 0.0;
  }
  if (config_.quality_filter.duplicate_iou_threshold < 0.0 ||
      config_.quality_filter.duplicate_iou_threshold > 1.0) {
    config_.quality_filter.duplicate_iou_threshold = 0.60;
  }
  if (config_.quality_filter.duplicate_keypoint_mean_dist_px < 0.0) {
    config_.quality_filter.duplicate_keypoint_mean_dist_px = 0.0;
  }
  if (config_.pose.depth_correction.min_depth_delta_m < 0.0) {
    config_.pose.depth_correction.min_depth_delta_m = 0.0;
  }
  config_.pose.depth_correction.blend_alpha =
    std::clamp(config_.pose.depth_correction.blend_alpha, 0.0, 1.0);
  if (config_.pose.depth_correction.max_correction_m < 0.0) {
    config_.pose.depth_correction.max_correction_m = 0.0;
  }
  if (config_.pose.depth_correction.max_scale < 1.0) {
    config_.pose.depth_correction.max_scale = 1.0;
  }
  if (config_.pose.depth_correction.min_lightbar_length_px < 0.0) {
    config_.pose.depth_correction.min_lightbar_length_px = 0.0;
  }
  auto& tf = config_.traditional_fusion;
  if (tf.strategy != "fallback" && tf.strategy != "always") {
    FYT_ERROR("armor_detector",
              "Unknown traditional_fusion.strategy '{}', using fallback",
              tf.strategy.c_str());
    tf.strategy = "fallback";
  }
  if (tf.min_nn_detections < 0) {
    tf.min_nn_detections = 0;
  }
  tf.confidence_scale = std::clamp(tf.confidence_scale, 0.0F, 1.0F);
  tf.binary_thres = std::clamp(tf.binary_thres, 0, 255);
}

void ArmorDetectorNNNode::initializeTraditionalDetector() {
  const auto& tf = config_.traditional_fusion;
  Detector::LightParams light_params{
    .min_ratio = tf.light_min_ratio,
    .max_ratio = tf.light_max_ratio,
    .max_angle = tf.light_max_angle,
    .color_diff_thresh = tf.light_color_diff_thresh};

  Detector::ArmorParams armor_params{
    .min_light_ratio = tf.armor_min_light_ratio,
    .min_small_center_distance = tf.armor_min_small_center_distance,
    .max_small_center_distance = tf.armor_max_small_center_distance,
    .min_large_center_distance = tf.armor_min_large_center_distance,
    .max_large_center_distance = tf.armor_max_large_center_distance,
    .max_angle = tf.armor_max_angle};

  const auto color = current_mode_ == DetectMode::BLUE
                       ? fyt::EnemyColor::BLUE
                       : fyt::EnemyColor::RED;
  traditional_detector_ = std::make_unique<Detector>(
    tf.binary_thres, color, light_params, armor_params);

  if (tf.use_classifier) {
    namespace fs = std::filesystem;
    fs::path model_path = utils::URLResolver::getResolvedPath(
      "package://armor_detector/model/lenet.onnx");
    fs::path label_path = utils::URLResolver::getResolvedPath(
      "package://armor_detector/model/label.txt");
    if (fs::exists(model_path) && fs::exists(label_path)) {
      traditional_detector_->classifier = std::make_unique<NumberClassifier>(
        model_path.string(), label_path.string(),
        tf.classifier_threshold, tf.ignore_classes);
    } else {
      FYT_ERROR("armor_detector",
                "Traditional fusion classifier model not found: {}",
                model_path.string().c_str());
    }
  }

  if (tf.use_pca) {
    traditional_detector_->corner_corrector =
      std::make_unique<LightCornerCorrector>();
  }

  FYT_INFO("armor_detector",
           "Traditional fusion initialized: strategy={}, classifier={}, pca={}",
           tf.strategy.c_str(), static_cast<int>(tf.use_classifier),
           static_cast<int>(tf.use_pca));
}

void ArmorDetectorNNNode::updateTraditionalDetectorColor() {
  if (!traditional_detector_) {
    return;
  }
  if (current_mode_ == DetectMode::RED) {
    traditional_detector_->detect_color = fyt::EnemyColor::RED;
  } else if (current_mode_ == DetectMode::BLUE) {
    traditional_detector_->detect_color = fyt::EnemyColor::BLUE;
  }
}

std::vector<ArmorDetection> ArmorDetectorNNNode::detectTraditional(
    const cv::Mat& bgr_frame) {
  std::vector<ArmorDetection> detections;
  if (!traditional_detector_) {
    return detections;
  }

  cv::Mat rgb_frame;
  cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);
  auto armors = traditional_detector_->detect(rgb_frame);
  detections.reserve(armors.size());

  for (const auto& armor : armors) {
    if (armor.type == ArmorType::INVALID || armor.number.empty()) {
      continue;
    }
    if (armor.number == "negative") {
      continue;
    }

    ArmorDetection det;
    det.publish_number = armor.number;
    det.publish_type =
      armor.type == ArmorType::LARGE ? "large" : "small";
    det.color = traditional_detector_->detect_color;
    det.confidence = std::clamp(
      armor.confidence * config_.traditional_fusion.confidence_scale,
      0.0F, 1.0F);
    det.keypoints = {{
      armor.left_light.bottom,
      armor.left_light.top,
      armor.right_light.top,
      armor.right_light.bottom
    }};
    det = canonicalizeArmorDetectionGeometry(det);
    det.bbox = bboxFromKeypoints(det.keypoints);
    det.center =
      (det.keypoints[0] + det.keypoints[1] +
       det.keypoints[2] + det.keypoints[3]) * 0.25F;
    det.model_label = traditionalModelLabel(det);
    detections.push_back(std::move(det));
  }

  return detections;
}

std::vector<ArmorDetection> ArmorDetectorNNNode::mergeDetections(
    const std::vector<ArmorDetection>& nn_detections,
    const std::vector<ArmorDetection>& traditional_detections) const {
  std::vector<ArmorDetection> merged = nn_detections;
  merged.reserve(nn_detections.size() + traditional_detections.size());

  constexpr float kDuplicateIoU = 0.35F;
  constexpr float kDuplicateMeanKptDistPx = 12.0F;

  for (const auto& candidate : traditional_detections) {
    bool consumed = false;
    for (auto& existing : merged) {
      const bool same_color = existing.color == candidate.color;
      if (!same_color) {
        continue;
      }
      const bool bbox_duplicate =
        rectIoU(existing.bbox, candidate.bbox) > kDuplicateIoU;
      const bool keypoint_duplicate =
        meanKeypointDistance(existing.keypoints, candidate.keypoints) <
        kDuplicateMeanKptDistPx;
      if (bbox_duplicate || keypoint_duplicate) {
        if (candidate.confidence > existing.confidence) {
          auto replacement = candidate;
          replacement.track_id = existing.track_id;
          replacement.track_age = existing.track_age;
          replacement.track_hits = existing.track_hits;
          existing = std::move(replacement);
        }
        consumed = true;
        break;
      }
    }
    if (!consumed) {
      merged.push_back(candidate);
    }
  }

  if (config_.quality_filter.enabled) {
    merged = filterByGeometryQuality(merged, config_.quality_filter);
    merged = suppressDuplicateDetections(merged, config_.quality_filter);
  }
  return merged;
}

void ArmorDetectorNNNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img_msg)
{
  if (current_mode_ == DetectMode::DISABLED) {
    FYT_DEBUG("armor_detector", "Received image frame but detection is DISABLED. Ignoring.");
    return;
  }
  const bool nn_ready = detector_ && detector_->isInitialized();
  if (!nn_ready && !traditional_detector_) {
    FYT_ERROR("armor_detector",
              "No initialized detector available. Cannot process image.");
    return;
  }
  if (!nn_ready) {
    static bool warned_traditional_only = false;
    if (!warned_traditional_only) {
      FYT_WARN("armor_detector",
               "NN detector is unavailable; using traditional fusion detector only.");
      warned_traditional_only = true;
    }
  }

  FYT_DEBUG("armor_detector", "Received image frame (timestamp: {}.{})",
            img_msg->header.stamp.sec, img_msg->header.stamp.nanosec);

  auto t_start = std::chrono::steady_clock::now();

  // cv_bridge — apply copy policy
  cv::Mat frame;
  try {
    bool need_copy = (config_.runtime.copy_policy == CopyPolicy::ALWAYS_COPY) ||
                     (config_.runtime.copy_policy == CopyPolicy::COPY_ON_WRITE_DEBUG && debug_);
    if (need_copy) {
      frame = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
    } else {
      frame = cv_bridge::toCvShare(img_msg, "bgr8")->image;
    }
  } catch (const cv_bridge::Exception& e) {
    FYT_ERROR("armor_detector", "cv_bridge error: {}", e.what());
    return;
  }

  auto t_preprocess_end = std::chrono::steady_clock::now();

  Eigen::Matrix3d R_imu_camera = Eigen::Matrix3d::Identity();
  bool have_target_to_camera_tf = false;
  auto extract_rotation = [&](const geometry_msgs::msg::TransformStamped &t) {
    tf2::Quaternion tf_q;
    tf2::fromMsg(t.transform.rotation, tf_q);
    tf2::Matrix3x3 tf2_matrix(tf_q);
    R_imu_camera << tf2_matrix.getRow(0)[0], tf2_matrix.getRow(0)[1], tf2_matrix.getRow(0)[2],
                    tf2_matrix.getRow(1)[0], tf2_matrix.getRow(1)[1], tf2_matrix.getRow(1)[2],
                    tf2_matrix.getRow(2)[0], tf2_matrix.getRow(2)[1], tf2_matrix.getRow(2)[2];
  };
  try {
    const rclcpp::Time target_time = img_msg->header.stamp;
    auto target_to_camera = tf2_buffer_->lookupTransform(
        config_.target_frame, img_msg->header.frame_id, target_time,
        tf2::durationFromSec(0.01));
    have_target_to_camera_tf = true;
    extract_rotation(target_to_camera);
  } catch (tf2::ExtrapolationException &ex) {
    FYT_WARN("armor_detector",
             "TF at image stamp not cached, fallback to latest: {}", ex.what());
    try {
      auto target_to_camera = tf2_buffer_->lookupTransform(
          config_.target_frame, img_msg->header.frame_id, tf2::TimePointZero);
      have_target_to_camera_tf = true;
      extract_rotation(target_to_camera);
    } catch (tf2::TransformException &ex2) {
      FYT_ERROR("armor_detector", "Fallback transform error: {}", ex2.what());
      return;
    }
  } catch (tf2::TransformException &ex) {
    FYT_ERROR("armor_detector", "Transform error: {}", ex.what());
    return;
  }

  FrameDetections fd;
  fd.header = img_msg->header;
  if (nn_ready) {
    auto results = detector_->detectBatch({frame}, {img_msg->header});
    if (!results.empty()) {
      fd = std::move(results[0]);
    }
  }

  if (!nn_ready && !traditional_detector_) {
    if (config_.runtime.publish_empty) {
      publishEmptyArmors(img_msg->header);
    }
    return;
  }

  for (auto& det : fd.detections) {
    det.stamp = img_msg->header.stamp;
  }

  if (traditional_detector_ && config_.traditional_fusion.enabled) {
    const bool run_traditional =
      config_.traditional_fusion.strategy == "always" ||
      static_cast<int>(fd.detections.size()) <
        config_.traditional_fusion.min_nn_detections;
    if (run_traditional) {
      updateTraditionalDetectorColor();
      auto traditional = detectTraditional(frame);
      for (auto& det : traditional) {
        det.stamp = img_msg->header.stamp;
      }
      const auto nn_count = fd.detections.size();
      const auto traditional_count = traditional.size();
      fd.detections = mergeDetections(fd.detections, traditional);
      FYT_DEBUG("armor_detector",
                "Traditional fusion: nn={} traditional={} merged={}",
                nn_count, traditional_count, fd.detections.size());
    }
  }

  auto t_detect_end = std::chrono::steady_clock::now();

  // --- Phase 2: Tracker association ---
  if (tracker_) {
    auto tracked = tracker_->associate(fd.detections, img_msg->header.stamp);
    fd.detections.clear();
    fd.detections.reserve(tracked.size());
    for (auto& td : tracked) {
      fd.detections.push_back(std::move(td.det));
    }
  }

  // --- Phase 3: Corner refinement (optional, only confirmed and limited) ---
  if (corner_refiner_ && !fd.detections.empty()) {
    int refined_count = 0;
    for (auto& det : fd.detections) {
      if (config_.corner_refine.apply_on_confirmed_only &&
          det.track_hits < config_.tracker.min_hits) continue;
      if (refined_count >= config_.corner_refine.max_targets_per_frame) break;

      auto refine_res = corner_refiner_->refine(frame, det);
      if (refine_res.ok) {
        det.keypoints = refine_res.refined_keypoints;
        det.center =
          (det.keypoints[0] + det.keypoints[1] +
           det.keypoints[2] + det.keypoints[3]) * 0.25F;
        float min_x = det.keypoints[0].x;
        float max_x = det.keypoints[0].x;
        float min_y = det.keypoints[0].y;
        float max_y = det.keypoints[0].y;
        for (int k = 1; k < 4; ++k) {
          min_x = std::min(min_x, det.keypoints[k].x);
          max_x = std::max(max_x, det.keypoints[k].x);
          min_y = std::min(min_y, det.keypoints[k].y);
          max_y = std::max(max_y, det.keypoints[k].y);
        }
        det.bbox = cv::Rect2f(
          min_x, min_y,
          std::max(0.0F, max_x - min_x),
          std::max(0.0F, max_y - min_y));
        refined_count++;
      }
    }
  }

  // PnP pose estimation
  std::vector<PoseEstimate> poses;
  std::vector<PoseEstimate> poses_ref;
  if (cam_info_ && pose_estimator_adapter_) {
    poses = pose_estimator_adapter_->estimateBatch(
      fd.detections, *cam_info_, R_imu_camera);
    if (debug_pose_compare_ && pose_estimator_reference_adapter_) {
      poses_ref =
        pose_estimator_reference_adapter_->estimateBatch(
          fd.detections, *cam_info_, R_imu_camera);
    }
  } else {
    poses.resize(fd.detections.size());
    if (debug_pose_compare_) {
      poses_ref.resize(fd.detections.size());
    }
  }

  if (debug_pose_compare_ && poses_ref.size() == poses.size()) {
    auto rad2deg = [](double r) { return r * 180.0 / M_PI; };
    auto wrapDeg = [&](double deg) {
      while (deg > 180.0) deg -= 360.0;
      while (deg < -180.0) deg += 360.0;
      return deg;
    };

    for (size_t i = 0; i < poses.size(); ++i) {
      const auto& d = fd.detections[i];
      const auto& cur = poses[i];
      const auto& ref = poses_ref[i];

      if (!cur.valid || !ref.valid) {
        FYT_DEBUG(
          "armor_detector",
          "[PoseCmp] id={} num={} type={} cur_valid={} ref_valid={}",
          d.track_id, d.publish_number.c_str(), d.publish_type.c_str(),
          static_cast<int>(cur.valid), static_cast<int>(ref.valid));
        continue;
      }

      double cy = rad2deg(cur.yaw), cp = rad2deg(cur.pitch), cr = rad2deg(cur.roll);
      double ry = rad2deg(ref.yaw), rp = rad2deg(ref.pitch), rr = rad2deg(ref.roll);
      double dy = wrapDeg(cy - ry), dp = wrapDeg(cp - rp), dr = wrapDeg(cr - rr);

      FYT_DEBUG(
        "armor_detector",
        "[PoseCmp] id={} num={} type={} mode={} ref_mode={}; cur(ypr)={:.2f}/{:.2f}/{:.2f} ref(ypr)={:.2f}/{:.2f}/{:.2f} d(ypr)={:.2f}/{:.2f}/{:.2f}; err(cur/ref)={:.3f}/{:.3f}",
        d.track_id, d.publish_number.c_str(), d.publish_type.c_str(),
        static_cast<int>(cur.mode), static_cast<int>(ref.mode),
        cy, cp, cr, ry, rp, rr, dy, dp, dr,
        cur.reproj_error_refined, ref.reproj_error_refined);
    }
  }

  auto t_pose_end = std::chrono::steady_clock::now();

  // Build and publish Armors message
  {
    rm_interfaces::msg::Armors armors_msg;
    armors_msg.header = img_msg->header;
    if (publish_in_target_frame_) {
      armors_msg.header.frame_id = config_.target_frame;
    }

    for (size_t i = 0; i < fd.detections.size(); ++i) {
      rm_interfaces::msg::Armor armor;
      armor.number = fd.detections[i].publish_number;
      armor.type   = fd.detections[i].publish_type;
      armor.distance_to_image_center =
        ArmorPoseEstimatorAdapter::distanceToImageCenter(
          fd.detections[i].center, cam_center_);
      armor.detection_confidence = fd.detections[i].confidence;
      armor.has_image_geometry = true;
      armor.bbox_xywh[0] = fd.detections[i].bbox.x;
      armor.bbox_xywh[1] = fd.detections[i].bbox.y;
      armor.bbox_xywh[2] = fd.detections[i].bbox.width;
      armor.bbox_xywh[3] = fd.detections[i].bbox.height;
      for (int k = 0; k < 4; ++k) {
        geometry_msgs::msg::Point32 p;
        p.x = fd.detections[i].keypoints[k].x;
        p.y = fd.detections[i].keypoints[k].y;
        p.z = 0.0f;
        armor.image_corners[k] = p;
      }
      // 0: keypoint order as provided by detector postprocess.
      armor.corners_ordering = 0;

      if (poses[i].valid) {
        geometry_msgs::msg::Pose pose_camera;
        pose_camera.position.x = poses[i].translation.x();
        pose_camera.position.y = poses[i].translation.y();
        pose_camera.position.z = poses[i].translation.z();
        pose_camera.orientation.x = poses[i].rotation.x();
        pose_camera.orientation.y = poses[i].rotation.y();
        pose_camera.orientation.z = poses[i].rotation.z();
        pose_camera.orientation.w = poses[i].rotation.w();

        if (publish_in_target_frame_) {
          if (have_target_to_camera_tf) {
            geometry_msgs::msg::PoseStamped in_pose;
            geometry_msgs::msg::PoseStamped out_pose;
            in_pose.header = img_msg->header;
            in_pose.pose = pose_camera;
            try {
              out_pose = tf2_buffer_->transform(
                in_pose, config_.target_frame, tf2::durationFromSec(0.005));
              armor.pose = out_pose.pose;
            } catch (const tf2::TransformException& ex) {
              FYT_WARN("armor_detector",
                       "Pose transform to target frame failed: {}", ex.what());
              armor.pose = pose_camera;
            }
          } else {
            armor.pose = pose_camera;
          }
        } else {
          armor.pose = pose_camera;
        }
      }

      // Fill refiner quality metadata (Phase 1: covariance always invalid)
      armor.pose_estimate_mode = static_cast<uint8_t>(poses[i].mode);
      armor.pose_quality_score = static_cast<float>(poses[i].quality_score);
      armor.reproj_error_raw = static_cast<float>(poses[i].reproj_error_raw);
      armor.reproj_error_refined = static_cast<float>(poses[i].reproj_error_refined);
      armor.pose_condition_number = static_cast<float>(poses[i].condition_number);
      armor.pose_num_points = static_cast<uint16_t>(poses[i].num_points);
      armor.pose_num_inliers = static_cast<uint16_t>(poses[i].num_inliers);
      armor.pose_covariance_valid = poses[i].covariance_valid;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          armor.pose_covariance_xyz_yaw[r * 4 + c] =
              poses[i].covariance_xyz_yaw(r, c);
        }
      }

      armors_msg.armors.push_back(armor);
    }

    armors_pub_->publish(armors_msg);

    // Markers follow the same frame as published armors.
    if (debug_ && marker_pub_) {
      publishMarkers(fd.detections, poses, armors_msg.header);
    }
  }

  // Debug image
  if (debug_) {
    publishDebugImage(frame, fd, poses);
  }

  // Profiler
  if (profiler_) {
    auto t_total = std::chrono::steady_clock::now();
    ProfilerEntry entry{};
    if (detector_ && detector_->isInitialized()) {
      entry = detector_->lastProfiler();
    }
    entry.pose_ms  = std::chrono::duration<double, std::milli>(t_pose_end - t_start).count();
    entry.total_ms = std::chrono::duration<double, std::milli>(t_total - t_start).count();
    profiler_->record(entry);
  }

  FYT_DEBUG("armor_detector",
            "Frame processed. Preprocess: {:.2f} ms, Detect: {:.2f} ms, Pose: {:.2f} ms",
            std::chrono::duration<double, std::milli>(t_preprocess_end - t_start).count(),
            std::chrono::duration<double, std::milli>(t_detect_end - t_preprocess_end).count(),
            std::chrono::duration<double, std::milli>(t_pose_end - t_detect_end).count());

}

void ArmorDetectorNNNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& ci_msg)
{
  cam_center_ = cv::Point2f(ci_msg->k[2], ci_msg->k[5]);
  cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*ci_msg);
  FYT_INFO("armor_detector", "Camera info received: {}x{}",
           ci_msg->width, ci_msg->height);
  cam_info_sub_.reset();
}

void ArmorDetectorNNNode::setModeCallback(
    const std::shared_ptr<rm_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<rm_interfaces::srv::SetMode::Response> response)
{
  response->success = true;
  response->message = "0";

  VisionMode mode = static_cast<VisionMode>(request->mode);

  switch (mode) {
    case VisionMode::AUTO_AIM_RED:
      current_mode_ = DetectMode::RED;
      if (detector_) detector_->setTargetColor(fyt::EnemyColor::RED);
      updateTraditionalDetectorColor();
      FYT_INFO("armor_detector", "Mode set to RED");
      break;
    case VisionMode::AUTO_AIM_BLUE:
      current_mode_ = DetectMode::BLUE;
      if (detector_) detector_->setTargetColor(fyt::EnemyColor::BLUE);
      updateTraditionalDetectorColor();
      FYT_INFO("armor_detector", "Mode set to BLUE");
      break;
    default:
      current_mode_ = DetectMode::DISABLED;
      FYT_INFO("armor_detector", "Mode set to DISABLED");
      break;
  }
}

void ArmorDetectorNNNode::publishEmptyArmors(const std_msgs::msg::Header& header) {
  rm_interfaces::msg::Armors msg;
  msg.header = header;
  armors_pub_->publish(msg);
}

void ArmorDetectorNNNode::publishMarkers(
    const std::vector<ArmorDetection>& detections,
    const std::vector<PoseEstimate>& poses,
    const std_msgs::msg::Header& header)
{
  visualization_msgs::msg::MarkerArray marker_array;

  for (size_t i = 0; i < detections.size(); ++i) {
    // Armor cube marker
    visualization_msgs::msg::Marker m;
    m.header = header;
    m.ns = "armors_nn";
    m.id = static_cast<int>(i);
    m.action = visualization_msgs::msg::Marker::ADD;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.scale.x = 0.03;
    m.scale.y = 0.15;
    m.scale.z = 0.12;
    m.color.a = 1.0;
    m.lifetime = rclcpp::Duration::from_seconds(0.1);

    if (detections[i].color == fyt::EnemyColor::RED) {
      m.color.r = 1.0;
      m.color.g = 0.0;
      m.color.b = 0.0;
    } else {
      m.color.r = 0.0;
      m.color.g = 0.0;
      m.color.b = 1.0;
    }

    if (poses[i].valid) {
      m.pose.position.x = poses[i].translation.x();
      m.pose.position.y = poses[i].translation.y();
      m.pose.position.z = poses[i].translation.z();
      m.pose.orientation.x = poses[i].rotation.x();
      m.pose.orientation.y = poses[i].rotation.y();
      m.pose.orientation.z = poses[i].rotation.z();
      m.pose.orientation.w = poses[i].rotation.w();
    }

    marker_array.markers.push_back(m);

    // Armor plate outline: 4 line segments connecting keypoints
    if (detections[i].keypoints[0].x > 0 && detections[i].keypoints[3].x > 0) {
      visualization_msgs::msg::Marker outline;
      outline.header = header;
      outline.ns = "armors_nn_outline";
      outline.id = static_cast<int>(i);
      outline.action = visualization_msgs::msg::Marker::ADD;
      outline.type = visualization_msgs::msg::Marker::LINE_STRIP;
      outline.scale.x = 0.005;  // line width
      outline.color.a = 0.8;
      outline.lifetime = rclcpp::Duration::from_seconds(0.1);
      if (detections[i].color == fyt::EnemyColor::RED) {
        outline.color.r = 1.0;
      } else {
        outline.color.b = 1.0;
      }
      // kpt0→kpt1→kpt2→kpt3→kpt0 (xy image coords → 3D at z=1 for RViz)
      for (int k = 0; k <= 4; ++k) {
        const auto& kp = detections[i].keypoints[k % 4];
        geometry_msgs::msg::Point pt;
        pt.x = kp.x;
        pt.y = kp.y;
        pt.z = 0.0;
        outline.points.push_back(pt);
      }
      marker_array.markers.push_back(outline);
    }

    // Text marker
    visualization_msgs::msg::Marker text;
    text.header = header;
    text.ns = "armors_nn_text";
    text.id = static_cast<int>(i);
    text.action = visualization_msgs::msg::Marker::ADD;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.scale.z = 0.1;
    text.color.a = 1.0;
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    text.lifetime = rclcpp::Duration::from_seconds(0.1);
    text.text = detections[i].publish_number;
    if (poses[i].valid) {
      text.pose.position.x = poses[i].translation.x();
      text.pose.position.y = poses[i].translation.y() - 0.1;
      text.pose.position.z = poses[i].translation.z();
    }
    marker_array.markers.push_back(text);
  }

  marker_pub_->publish(marker_array);
}

void ArmorDetectorNNNode::publishDebugImage(
    const cv::Mat& frame,
    const FrameDetections& fd,
    const std::vector<PoseEstimate>& /*poses*/)
{
  cv::Mat debug_img = frame.clone();

  debug_drawer_->drawDetections(debug_img, fd.detections, true);

  if (profiler_) {
    double fps = profiler_->avgFPS();
    double latency = profiler_->avgTotalMs();
    BackendInfo bi;
    if (detector_ && detector_->isInitialized()) {
      bi = detector_->backendInfo();
    } else {
      bi.backend_name = "traditional";
      bi.precision = "opencv";
    }
    debug_drawer_->drawProfiler(debug_img, fps, latency,
                                bi.backend_name, bi.precision);
    debug_drawer_->drawArmorsCount(debug_img, static_cast<int>(fd.detections.size()));
  }

  auto msg = cv_bridge::CvImage(fd.header, "bgr8", debug_img).toImageMsg();
  result_img_pub_.publish(*msg);
}

void ArmorDetectorNNNode::createDebugPublishers() {
  result_img_pub_ = image_transport::create_publisher(this, "armor_detector/result_img");
}

void ArmorDetectorNNNode::destroyDebugPublishers() {
  result_img_pub_.shutdown();
}

rcl_interfaces::msg::SetParametersResult
ArmorDetectorNNNode::onSetParameters(const std::vector<rclcpp::Parameter>& params) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  bool corner_refine_reconfigure = false;

  for (const auto& p : params) {
    const auto& name = p.get_name();
    if (name == "debug") {
      debug_ = p.as_bool();
      debug_ ? createDebugPublishers() : destroyDebugPublishers();
    } else if (name == "debug_pose_compare") {
      debug_pose_compare_ = p.as_bool();
    } else if (name == "publish_in_target_frame") {
      publish_in_target_frame_ = p.as_bool();
    } else if (name == "corner_refine.enabled") {
      config_.corner_refine.enabled = p.as_bool();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.method") {
      config_.corner_refine.method = p.as_string();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.apply_on_confirmed_only") {
      config_.corner_refine.apply_on_confirmed_only = p.as_bool();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_targets_per_frame") {
      config_.corner_refine.max_targets_per_frame = p.as_int();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.time_budget_ms") {
      config_.corner_refine.time_budget_ms = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.roi_expand_ratio") {
      config_.corner_refine.roi_expand_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.min_bright_points") {
      config_.corner_refine.min_bright_points = p.as_int();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.pca_stability_threshold") {
      config_.corner_refine.pca_stability_threshold = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_aspect_ratio") {
      config_.corner_refine.max_aspect_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.min_aspect_ratio") {
      config_.corner_refine.min_aspect_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_corner_shift_px") {
      config_.corner_refine.max_corner_shift_px = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_mean_corner_shift_px") {
      config_.corner_refine.max_mean_corner_shift_px = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_refined_center_shift_px") {
      config_.corner_refine.max_refined_center_shift_px = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.min_refine_quality") {
      config_.corner_refine.min_refine_quality = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.preserve_perspective") {
      config_.corner_refine.preserve_perspective = p.as_bool();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_edge_angle_delta_deg") {
      config_.corner_refine.max_edge_angle_delta_deg = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_area_ratio_delta") {
      config_.corner_refine.max_area_ratio_delta = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_length_ratio_delta") {
      config_.corner_refine.max_length_ratio_delta = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.full_roi_expand_ratio") {
      config_.corner_refine.full_roi_expand_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.binary_threshold") {
      config_.corner_refine.binary_threshold = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.min_contour_area_px") {
      config_.corner_refine.min_contour_area_px = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.min_lightbar_length_px") {
      config_.corner_refine.min_lightbar_length_px = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.min_lightbar_ratio") {
      config_.corner_refine.min_lightbar_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_lightbar_ratio") {
      config_.corner_refine.max_lightbar_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_lightbar_angle_error_deg") {
      config_.corner_refine.max_lightbar_angle_error_deg = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_rectangular_error_deg") {
      config_.corner_refine.max_rectangular_error_deg = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_side_ratio") {
      config_.corner_refine.max_side_ratio = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_lightbar_match_error_px") {
      config_.corner_refine.max_lightbar_match_error_px = p.as_double();
      corner_refine_reconfigure = true;
    } else if (name == "corner_refine.max_pair_center_shift_px") {
      config_.corner_refine.max_pair_center_shift_px = p.as_double();
      corner_refine_reconfigure = true;
    }
  }

  if (corner_refine_reconfigure) {
    if (config_.corner_refine.enabled) {
      corner_refiner_ = std::make_shared<RoiPcaCornerRefiner>(config_.corner_refine);
    } else {
      corner_refiner_.reset();
    }
  }

  return result;
}

}  // namespace fyt::auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(fyt::auto_aim::ArmorDetectorNNNode)
