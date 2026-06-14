// Copyright (C) FYT Vision Group. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
// GimbalPipelineNode — unified node merging MaxEntropyTracker +
// TargetSelector + GimbalController. Inter-node ROS2 topics are replaced
// by direct C++ function calls to eliminate serialization / scheduling
// latency.

#include "gimbal_pipeline/gimbal_pipeline_node.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <rm_utils/heartbeat.hpp>
#include <sstream>
#include <iomanip>
#include <unordered_set>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rm_utils/logger/log.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "max_entropy_tracker/msg_converter.hpp"
#include "max_entropy_tracker/trackers/norm_4armor_tracker.hpp"
#include "max_entropy_tracker/trackers/norm4_v3/tracker/norm4_tracker_v2.hpp"
#include "max_entropy_tracker/visualization.hpp"

// Gimbal strategies
#include "gimbal_controller/strategies/current_position_strategy.hpp"
#include "gimbal_controller/strategies/mpc_control_strategy.hpp"
#include "gimbal_controller/strategies/predicted_position_strategy.hpp"
#include "gimbal_controller/strategies/state_machine_strategy.hpp"
#include "gimbal_controller/fire_advisor.hpp"

namespace
{

bool hasParameterOverride(rclcpp::Node & node, const std::string & key)
{
  const auto params_interface = node.get_node_parameters_interface();
  if (!params_interface) {
    return false;
  }
  const auto & overrides = params_interface->get_parameter_overrides();
  return overrides.find(key) != overrides.end();
}

bool shouldWarnDeprecatedOnce(const std::string & deprecated_key)
{
  static std::unordered_set<std::string> warned_keys;
  return warned_keys.insert(deprecated_key).second;
}

double readCompatDoubleParameter(
  rclcpp::Node & node,
  const std::string & canonical_key,
  const std::string & deprecated_key,
  double conflict_eps = 1e-9)
{
  const double canonical_value = node.get_parameter(canonical_key).as_double();
  const double deprecated_value = node.get_parameter(deprecated_key).as_double();
  const bool canonical_overridden = hasParameterOverride(node, canonical_key);
  const bool deprecated_overridden = hasParameterOverride(node, deprecated_key);

  if (deprecated_overridden && !canonical_overridden) {
    if (shouldWarnDeprecatedOnce(deprecated_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Parameter '%s' is deprecated; please use '%s'. Applying deprecated value: %.6f",
        deprecated_key.c_str(),
        canonical_key.c_str(),
        deprecated_value);
    }
    return deprecated_value;
  }

  if (deprecated_overridden && canonical_overridden &&
    std::abs(canonical_value - deprecated_value) > conflict_eps)
  {
    if (shouldWarnDeprecatedOnce(deprecated_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Both deprecated '%s' and canonical '%s' are set with different values "
        "(deprecated=%.6f, canonical=%.6f). Canonical value will be used.",
        deprecated_key.c_str(),
        canonical_key.c_str(),
        deprecated_value,
        canonical_value);
    }
  }

  return canonical_value;
}

int readCompatIntParameter(
  rclcpp::Node & node,
  const std::string & canonical_key,
  const std::string & deprecated_key)
{
  const int canonical_value = node.get_parameter(canonical_key).as_int();
  const int deprecated_value = node.get_parameter(deprecated_key).as_int();
  const bool canonical_overridden = hasParameterOverride(node, canonical_key);
  const bool deprecated_overridden = hasParameterOverride(node, deprecated_key);

  if (deprecated_overridden && !canonical_overridden) {
    if (shouldWarnDeprecatedOnce(deprecated_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Parameter '%s' is deprecated; please use '%s'. Applying deprecated value: %d",
        deprecated_key.c_str(),
        canonical_key.c_str(),
        deprecated_value);
    }
    return deprecated_value;
  }

  if (deprecated_overridden && canonical_overridden && canonical_value != deprecated_value) {
    if (shouldWarnDeprecatedOnce(deprecated_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Both deprecated '%s' and canonical '%s' are set with different values "
        "(deprecated=%d, canonical=%d). Canonical value will be used.",
        deprecated_key.c_str(),
        canonical_key.c_str(),
        deprecated_value,
        canonical_value);
    }
  }

  return canonical_value;
}

bool readCompatBoolParameter(
  rclcpp::Node & node,
  const std::string & canonical_key,
  const std::string & deprecated_key)
{
  const bool canonical_value = node.get_parameter(canonical_key).as_bool();
  const bool deprecated_value = node.get_parameter(deprecated_key).as_bool();
  const bool canonical_overridden = hasParameterOverride(node, canonical_key);
  const bool deprecated_overridden = hasParameterOverride(node, deprecated_key);

  if (deprecated_overridden && !canonical_overridden) {
    if (shouldWarnDeprecatedOnce(deprecated_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Parameter '%s' is deprecated; please use '%s'. Applying deprecated value: %s",
        deprecated_key.c_str(),
        canonical_key.c_str(),
        deprecated_value ? "true" : "false");
    }
    return deprecated_value;
  }

  if (deprecated_overridden && canonical_overridden && canonical_value != deprecated_value) {
    if (shouldWarnDeprecatedOnce(deprecated_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Both deprecated '%s' and canonical '%s' are set with different values "
        "(deprecated=%s, canonical=%s). Canonical value will be used.",
        deprecated_key.c_str(),
        canonical_key.c_str(),
        deprecated_value ? "true" : "false",
        canonical_value ? "true" : "false");
    }
  }

  return canonical_value;
}

double readUnifiedDoubleParameter(
  rclcpp::Node & node,
  const std::string & canonical_key,
  const std::initializer_list<std::string> & fallback_keys,
  double conflict_eps = 1e-9)
{
  const double canonical_value = node.get_parameter(canonical_key).as_double();
  const bool canonical_overridden = hasParameterOverride(node, canonical_key);

  if (canonical_overridden) {
    for (const auto & fallback_key : fallback_keys) {
      if (!hasParameterOverride(node, fallback_key)) {
        continue;
      }
      const double fallback_value = node.get_parameter(fallback_key).as_double();
      if (std::abs(canonical_value - fallback_value) > conflict_eps &&
        shouldWarnDeprecatedOnce(fallback_key))
      {
        RCLCPP_WARN(
          node.get_logger(),
          "Both '%s' and compatibility key '%s' are set with different values "
          "(canonical=%.6f, compatibility=%.6f). Canonical value will be used.",
          canonical_key.c_str(),
          fallback_key.c_str(),
          canonical_value,
          fallback_value);
      }
    }
    return canonical_value;
  }

  for (const auto & fallback_key : fallback_keys) {
    if (!hasParameterOverride(node, fallback_key)) {
      continue;
    }
    const double fallback_value = node.get_parameter(fallback_key).as_double();
    if (shouldWarnDeprecatedOnce(fallback_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Parameter '%s' is preferred; applying compatibility key '%s' value: %.6f",
        canonical_key.c_str(),
        fallback_key.c_str(),
        fallback_value);
    }
    return fallback_value;
  }

  return canonical_value;
}

int readUnifiedIntParameter(
  rclcpp::Node & node,
  const std::string & canonical_key,
  const std::initializer_list<std::string> & fallback_keys)
{
  const int canonical_value = node.get_parameter(canonical_key).as_int();
  const bool canonical_overridden = hasParameterOverride(node, canonical_key);

  if (canonical_overridden) {
    for (const auto & fallback_key : fallback_keys) {
      if (!hasParameterOverride(node, fallback_key)) {
        continue;
      }
      const int fallback_value = node.get_parameter(fallback_key).as_int();
      if (canonical_value != fallback_value && shouldWarnDeprecatedOnce(fallback_key)) {
        RCLCPP_WARN(
          node.get_logger(),
          "Both '%s' and compatibility key '%s' are set with different values "
          "(canonical=%d, compatibility=%d). Canonical value will be used.",
          canonical_key.c_str(),
          fallback_key.c_str(),
          canonical_value,
          fallback_value);
      }
    }
    return canonical_value;
  }

  for (const auto & fallback_key : fallback_keys) {
    if (!hasParameterOverride(node, fallback_key)) {
      continue;
    }
    const int fallback_value = node.get_parameter(fallback_key).as_int();
    if (shouldWarnDeprecatedOnce(fallback_key)) {
      RCLCPP_WARN(
        node.get_logger(),
        "Parameter '%s' is preferred; applying compatibility key '%s' value: %d",
        canonical_key.c_str(),
        fallback_key.c_str(),
        fallback_value);
    }
    return fallback_value;
  }

  return canonical_value;
}

}  // namespace

namespace fyt::auto_aim {

/* ================================================================ */
/*  Construction                                                     */
/* ================================================================ */

GimbalPipelineNode::GimbalPipelineNode(const rclcpp::NodeOptions &options)
    : Node("gimbal_pipeline", options) {
  RCLCPP_INFO(get_logger(), "Initializing GimbalPipelineNode (unified pipeline)");

  // Register loggers used by inlined target_selector code
  try {
    FYT_REGISTER_LOGGER("target_selector", "logs/gimbal_pipeline", INFO);
  } catch (...) {
    // Logger may already be registered
  }

  // ── 1. Declare all parameters ──
  declareTrackerParameters();
  declareTargetSelectorParameters();
  declareGimbalControllerParameters();

  RCLCPP_INFO(get_logger(), "Parameters declared, now loading...");

  // ── 2. Read common / tracker params ──
  target_frame_ = get_parameter("target_frame").as_string();
  source_frame_ = get_parameter("source_frame").as_string();
  predict_rate_ = get_parameter("predict_rate").as_double();
  debug_mode_ = get_parameter("debug_mode").as_bool();
  visualization_frame_ = get_parameter("visualization_frame").as_string();
  tracker_2d_image_debug_enable_ =
      get_parameter("tracker.debug_2d_viz.enable").as_bool();
  tracker_2d_image_debug_width_ =
      std::max(static_cast<int>(get_parameter("tracker.debug_2d_viz.width").as_int()), 320);
  tracker_2d_image_debug_height_ =
      std::max(static_cast<int>(get_parameter("tracker.debug_2d_viz.height").as_int()), 240);
  tracker_2d_image_debug_jpeg_quality_ =
      std::clamp(static_cast<int>(get_parameter("tracker.debug_2d_viz.jpeg_quality").as_int()), 20, 100);
  tracker_timeout_s_ = std::max(get_parameter("tracker_timeout").as_double(), 1e-3);

  tracker_config_ = UnifiedConfig::create_default();
  applyTrackerParamsToConfig();

  // ── TF2 buffer — shared by TFHandler, MessageFilter, and gimbal state ──
  // Must be created before TFHandler and before initGimbalComponents().
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      get_node_base_interface(), get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  // TF handler (used by tracker for armor → odom transform)
  // Shares tf2_buffer_ so MessageFilter and TFHandler use the same cache.
  tf_handler_ = std::make_unique<TFHandler>(tf2_buffer_, target_frame_);

  // Tracker manager
  double dt = (predict_rate_ > 0) ? (1.0 / predict_rate_) : 0.01;
  tracker_manager_ = std::make_unique<TrackerManager>(
      tracker_config_, dt,
      get_parameter("default_r1").as_double(),
      get_parameter("default_r2").as_double(),
      get_parameter("default_dza").as_double(),
      tracker_timeout_s_,
      get_parameter("enable_oscillation_detection").as_bool());

  robot_description_facade_ = std::make_unique<robot_description::RobotDescriptionFacade>();
  robot_description_facade_->setStrictUnknownReject(
      get_parameter("robot_description.strict_unknown_reject").as_bool());
  {
    const auto mode_raw =
      get_parameter("robot_description.default_projection_mode").as_string();
    std::string mode_lower = mode_raw;
    std::transform(
      mode_lower.begin(), mode_lower.end(), mode_lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    using RU = robot_description::TrackedRobotUsage;
    RU::ProjectionMode default_mode = RU::ProjectionMode::YAW_PLANE;
    if (mode_lower == "full_se3") {
      default_mode = RU::ProjectionMode::FULL_SE3;
    } else if (mode_lower == "yaw_plane") {
      default_mode = RU::ProjectionMode::YAW_PLANE;
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Unknown robot_description.default_projection_mode='%s', fallback to yaw_plane",
        mode_raw.c_str());
    }

    const auto full_se3_ids_vec =
      get_parameter("robot_description.full_se3_ids").as_string_array();
    std::unordered_set<std::string> full_se3_ids(
      full_se3_ids_vec.begin(), full_se3_ids_vec.end());

    const auto full_se3_types_vec =
      get_parameter("robot_description.full_se3_robot_types").as_integer_array();
    std::unordered_set<uint8_t> full_se3_robot_types;
    for (const auto v : full_se3_types_vec) {
      if (v < 0 || v > 255) {
        RCLCPP_WARN(
          get_logger(),
          "robot_description.full_se3_robot_types contains out-of-range value %ld, ignored",
          static_cast<long>(v));
        continue;
      }
      full_se3_robot_types.insert(static_cast<uint8_t>(v));
    }

    RU::setProjectionModePolicy(default_mode, full_se3_ids, full_se3_robot_types);
  }

  external_targets_enable_ = get_parameter("external_targets.enable").as_bool();
  external_targets_buff_enable_ = get_parameter("external_targets.buff.enable").as_bool();
  external_targets_buff_topic_ = get_parameter("external_targets.buff.topic").as_string();
  external_targets_buff_timeout_s_ = get_parameter("external_targets.buff.timeout_s").as_double();
  allowed_ids_by_mode_.clear();
  for (int mode = 0; mode <= 5; ++mode) {
    const auto key =
      "external_targets.allowed_ids_by_mode.mode_" + std::to_string(mode);
    const auto arr = get_parameter(key).as_string_array();
    allowed_ids_by_mode_[mode] = std::unordered_set<std::string>(arr.begin(), arr.end());
  }
  refreshExternalTargetAllowlist(current_mode_);

  if (external_targets_enable_ && external_targets_buff_enable_) {
    adapters::BuffTargetAdapter::Config cfg;
    cfg.enable = true;
    cfg.topic = external_targets_buff_topic_;
    cfg.timeout_s = std::max(0.01, external_targets_buff_timeout_s_);
    cfg.target_frame = target_frame_;
    buff_target_adapter_ = std::make_unique<adapters::BuffTargetAdapter>(*this, cfg);
  }

  {
    std::ostringstream oss;
    const auto supported_ids = robot_description_facade_->supportedRobotIds();
    for (size_t i = 0; i < supported_ids.size(); ++i) {
      if (i != 0) {
        oss << ",";
      }
      oss << supported_ids[i];
    }
    RCLCPP_INFO(
      get_logger(),
      "RobotDescription initialized (strict_unknown_reject=%s, supported_ids=[%s])",
      robot_description_facade_->strictUnknownReject() ? "true" : "false",
      oss.str().c_str());
  }

  RCLCPP_INFO(get_logger(), "Tracker initialized (predict_rate=%.1f Hz)",
              predict_rate_);

  // ── 3. Target selector ──
  selector_strategy_name_ =
      get_parameter("selector.strategy").as_string();
  selection_config_.reference_yaw =
      get_parameter("selector.reference_yaw").as_double();
  selection_config_.max_yaw_deviation =
      get_parameter("selector.max_yaw_deviation").as_double();
  selection_config_.max_distance =
      get_parameter("selector.max_distance").as_double();
  selection_config_.min_confidence =
      get_parameter("selector.min_confidence").as_double();
  selection_config_.hysteresis_threshold =
      get_parameter("selector.hysteresis_threshold").as_double();
    selection_config_.priority_robot_ids =
      get_parameter("selector.priority_robot_ids").as_string_array();
    selection_config_.sticky_lock_frames =
      get_parameter("selector.sticky_lock_frames").as_int();
    selection_config_.sticky_lost_frames =
      get_parameter("selector.sticky_lost_frames").as_int();
  initSelectionStrategy();

  RCLCPP_INFO(get_logger(), "[GimbalPipelineNode] selector_strategy: %s", selector_strategy_name_.c_str());

  // ── 4. Gimbal controller ──
  bullet_speed_ = get_parameter("controller.bullet_speed").as_double();
  control_rate_ = get_parameter("controller.control_rate").as_double();
  current_gimbal_strategy_name_ =
      get_parameter("controller.strategy").as_string();
  ballistic_mode_ = get_parameter("controller.ballistic_mode").as_string();

  // TF2 buffer was already created above (shared with TFHandler & MessageFilter).

  initGimbalComponents();
  initGimbalStrategies();

  // Configure solver parameters
  double shooting_range_w = get_parameter("controller.solver.shooting_range_width").as_double();
  double shooting_range_h = get_parameter("controller.solver.shooting_range_height").as_double();
  double side_angle = get_parameter("controller.solver.side_angle").as_double();
  double min_switching_v_yaw = get_parameter("controller.solver.min_switching_v_yaw").as_double();
  double prediction_delay = readUnifiedDoubleParameter(
    *this,
    "controller.delay.prediction_extra_s",
    {
      "controller.solver.prediction_delay",
      "solver.prediction_delay",
      "controller.state_machine.prediction_delay",
      "state_machine.prediction_delay",
      "controller.mpc.prediction_delay_s",
      "mpc.prediction_delay_s"
    });
  double max_prediction_time = readCompatDoubleParameter(
    *this, "controller.solver.max_prediction_time", "solver.max_prediction_time");
  double max_tracking_v_yaw = get_parameter("controller.solver.max_tracking_v_yaw").as_double();
  int transfer_thresh = get_parameter("controller.solver.transfer_thresh").as_int();
  double gravity = get_parameter("controller.solver.gravity").as_double();
  double resistance = get_parameter("controller.solver.resistance").as_double();
  int iteration_times = get_parameter("controller.solver.iteration_times").as_int();
  double pitch_offset = get_parameter("controller.solver.pitch_offset").as_double();
  double yaw_offset = get_parameter("controller.solver.yaw_offset").as_double();
  double facing_enter_angle = get_parameter("controller.solver.facing_enter_angle").as_double();
  double facing_exit_angle = get_parameter("controller.solver.facing_exit_angle").as_double();
  bool radial_dynamic_enable = get_parameter("controller.solver.radial_dynamic.enable").as_bool();
  double radial_dynamic_v_yaw_ref = get_parameter("controller.solver.radial_dynamic.v_yaw_ref").as_double();
  double radial_dynamic_shrink_ratio = get_parameter("controller.solver.radial_dynamic.shrink_ratio").as_double();
  double radial_dynamic_min_angle_deg = get_parameter("controller.solver.radial_dynamic.min_angle_deg").as_double();
  double radial_dynamic_bias_gain_deg = get_parameter("controller.solver.radial_dynamic.bias_gain_deg").as_double();
  double radial_dynamic_max_bias_deg = get_parameter("controller.solver.radial_dynamic.max_bias_deg").as_double();
  bool virtual_auto_switch_enable = get_parameter("controller.solver.virtual_pose.auto_switch.enable").as_bool();
  double virtual_auto_switch_enter_vyaw =
    get_parameter("controller.solver.virtual_pose.auto_switch.enter_vyaw").as_double();
  double virtual_auto_switch_exit_vyaw =
    get_parameter("controller.solver.virtual_pose.auto_switch.exit_vyaw").as_double();
  std::string virtual_auto_switch_method_str =
    get_parameter("controller.solver.virtual_pose.auto_switch.selection_method").as_string();
  int virtual_auto_switch_fixed_id =
    get_parameter("controller.solver.virtual_pose.auto_switch.fixed_id").as_int();
  int virtual_fixed_id = get_parameter("controller.solver.virtual_pose.fixed_id").as_int();
  double sp_vision_low_speed_vyaw =
    get_parameter("controller.solver.sp_vision.low_speed_vyaw").as_double();
  double sp_vision_shootable_angle_deg =
    get_parameter("controller.solver.sp_vision.shootable_angle_deg").as_double();
  double sp_vision_coming_angle_deg =
    get_parameter("controller.solver.sp_vision.coming_angle_deg").as_double();
  double sp_vision_leaving_angle_deg =
    get_parameter("controller.solver.sp_vision.leaving_angle_deg").as_double();
  double sp_vision_outpost_coming_angle_deg =
    get_parameter("controller.solver.sp_vision.outpost_coming_angle_deg").as_double();
  double sp_vision_outpost_leaving_angle_deg =
    get_parameter("controller.solver.sp_vision.outpost_leaving_angle_deg").as_double();
  bool sp_vision_hold_current_until_jump =
    get_parameter("controller.solver.sp_vision.hold_current_until_jump").as_bool();
  bool sp_vision_zero_speed_fallback =
    get_parameter("controller.solver.sp_vision.zero_speed_fallback").as_bool();
  double controller_delay = readUnifiedDoubleParameter(
    *this,
    "controller.delay.control_latency_s",
    {
      "controller.solver.controller_delay",
      "solver.controller_delay",
      "controller.mpc.control_delay_s",
      "mpc.control_delay_s"
    });
  double trigger_to_muzzle_s = readUnifiedDoubleParameter(
    *this,
    "controller.delay.trigger_to_muzzle_s",
    {
      "controller.fire.trigger_to_muzzle_s",
      "controller.solver.trigger_to_muzzle_s",
      "solver.trigger_to_muzzle_s"
    });
  double max_processing_delay_s = readUnifiedDoubleParameter(
    *this,
    "controller.delay.max_processing_delay_s",
    {
      "controller.mpc.max_processing_delay_s",
      "mpc.max_processing_delay_s"
    });
  std::string selection_method_str = get_parameter("controller.solver.selection_method").as_string();
  std::string fire_policy = get_parameter("controller.fire.decision_policy").as_string();
  int fire_flight_time_iters = readUnifiedIntParameter(
    *this,
    "controller.delay.flight_time_iters",
    {
      "controller.fire.flight_time_iters",
      "controller.mpc.flight_time_iters",
      "mpc.flight_time_iters"
    });
  double fire_facing_filter_opening_angle_deg =
    get_parameter("controller.fire.facing_filter_opening_angle_deg").as_double();
  bool fire_use_gimbal_kinematics =
    get_parameter("controller.fire.use_gimbal_kinematics").as_bool();
  const bool fire_velocity_low_pass_enable =
    get_parameter("controller.fire.velocity_low_pass.enable").as_bool();
  const double fire_velocity_low_pass_alpha =
    get_parameter("controller.fire.velocity_low_pass.alpha").as_double();
  const double fire_velocity_low_pass_reset_timeout_s =
    get_parameter("controller.fire.velocity_low_pass.reset_timeout_s").as_double();
  const bool fire_probability_enable =
    get_parameter("controller.fire.probability.enable").as_bool();
  const double fire_probability_window_ms =
    get_parameter("controller.fire.probability.future_window_ms").as_double();
  const double fire_probability_step_ms =
    get_parameter("controller.fire.probability.future_step_ms").as_double();
  const std::string fire_probability_window_fusion =
    get_parameter("controller.fire.probability.window_fusion").as_string();
  const double fire_probability_softmax_beta =
    get_parameter("controller.fire.probability.softmax_beta").as_double();
  const std::string fire_probability_gate_strategy =
    get_parameter("controller.fire.probability.gate.strategy").as_string();
  const int fire_probability_burst_count =
    get_parameter("controller.fire.probability.burst.burst_bullet_count").as_int();
  const int fire_probability_min_hit_count =
    get_parameter("controller.fire.probability.burst.min_hit_count").as_int();
  const double fire_probability_ref_p0 =
    get_parameter("controller.fire.probability.evidence.reference_probability_p0").as_double();
  const double fire_probability_evidence_window_ms =
    get_parameter("controller.fire.probability.evidence.window_ms").as_double();
  const double fire_probability_evidence_log_clip =
    get_parameter("controller.fire.probability.evidence.log_clip").as_double();
  const double fire_probability_evidence_epsilon =
    get_parameter("controller.fire.probability.evidence.epsilon").as_double();
  const bool fire_probability_neutralize_unshootable_samples =
    get_parameter("controller.fire.probability.evidence.neutralize_unshootable_samples").as_bool();
  const double fire_probability_negative_evidence_scale =
    get_parameter("controller.fire.probability.evidence.negative_evidence_scale").as_double();
  const double fire_probability_negative_clip_scale =
    get_parameter("controller.fire.probability.evidence.negative_clip_scale").as_double();
  const double fire_probability_evidence_deadband =
    get_parameter("controller.fire.probability.evidence.deadband").as_double();
  const double fire_probability_temperature =
    get_parameter("controller.fire.probability.temperature.value").as_double();
  const double fire_probability_theta_on_cold =
    get_parameter("controller.fire.probability.temperature.theta_on_cold").as_double();
  const double fire_probability_theta_on_hot =
    get_parameter("controller.fire.probability.temperature.theta_on_hot").as_double();
  const double fire_probability_theta_hold_cold =
    get_parameter("controller.fire.probability.temperature.theta_hold_cold").as_double();
  const double fire_probability_theta_hold_hot =
    get_parameter("controller.fire.probability.temperature.theta_hold_hot").as_double();
  const double fire_probability_theta_reset_cold =
    get_parameter("controller.fire.probability.temperature.theta_reset_cold").as_double();
  const double fire_probability_theta_reset_hot =
    get_parameter("controller.fire.probability.temperature.theta_reset_hot").as_double();
  const double fire_probability_min_fire_ms =
    get_parameter("controller.fire.probability.commit.min_fire_ms").as_double();
  const double fire_probability_cooldown_ms =
    get_parameter("controller.fire.probability.commit.cooldown_ms").as_double();
  fire_prob_vis_enable_ = get_parameter("controller.fire.visualization.enable").as_bool();
  fire_prob_vis_ellipse_samples_ =
    get_parameter("controller.fire.visualization.ellipse_samples").as_int();
  fire_prob_vis_max_impact_points_ =
    get_parameter("controller.fire.visualization.max_impact_points").as_int();
  fire_prob_image_debug_enable_ =
    get_parameter("controller.fire.visualization.image_debug.enable").as_bool();
  fire_prob_image_debug_publish_rate_hz_ = std::max(
    get_parameter("controller.fire.visualization.image_debug.publish_rate_hz").as_double(), 0.1);
  fire_prob_image_debug_width_ = std::max(
    static_cast<int>(get_parameter("controller.fire.visualization.image_debug.width").as_int()), 320);
  fire_prob_image_debug_height_ = std::max(
    static_cast<int>(get_parameter("controller.fire.visualization.image_debug.height").as_int()), 240);
  fire_prob_image_debug_show_text_ =
    get_parameter("controller.fire.visualization.image_debug.show_text").as_bool();
  fire_prob_image_debug_show_sigma_ellipse_ =
    get_parameter("controller.fire.visualization.image_debug.show_sigma_ellipse").as_bool();
  fire_prob_image_debug_show_velocity_fan_ =
    get_parameter("controller.fire.visualization.image_debug.show_velocity_fan").as_bool();
  const std::string fire_target_visibility_policy =
    get_parameter("controller.fire.target_visibility_policy").as_string();

  if (fire_target_visibility_policy == "facing_only") {
    const bool opening_overridden =
      hasParameterOverride(*this, "controller.fire.facing_filter_opening_angle_deg");
    if (!opening_overridden || fire_facing_filter_opening_angle_deg >= 180.0 - 1e-9) {
      fire_facing_filter_opening_angle_deg = std::clamp(2.0 * facing_exit_angle, 0.0, 180.0);
    }
  } else if (fire_target_visibility_policy == "legacy_all") {
    fire_facing_filter_opening_angle_deg = 180.0;
  } else {
    RCLCPP_WARN(
      get_logger(),
      "Unknown controller.fire.target_visibility_policy='%s', fallback to facing_only.",
      fire_target_visibility_policy.c_str());
    fire_facing_filter_opening_angle_deg = std::clamp(2.0 * facing_exit_angle, 0.0, 180.0);
  }

  facing_enter_angle_deg_ = facing_enter_angle;
  facing_exit_angle_deg_ = facing_exit_angle;
  radial_dynamic_enable_ = radial_dynamic_enable;
  radial_dynamic_v_yaw_ref_ = std::max(radial_dynamic_v_yaw_ref, 1e-6);
  radial_dynamic_shrink_ratio_ = std::clamp(radial_dynamic_shrink_ratio, 0.0, 1.0);
  radial_dynamic_min_angle_deg_ = std::max(radial_dynamic_min_angle_deg, 0.0);
  radial_dynamic_bias_gain_deg_ = std::max(radial_dynamic_bias_gain_deg, 0.0);
  radial_dynamic_max_bias_deg_ = std::max(radial_dynamic_max_bias_deg, 0.0);
  virtual_auto_switch_enable_ = virtual_auto_switch_enable;
  mpc_dt_debug_ = std::max(get_parameter("controller.mpc.dt").as_double(), 1e-4);

  armor_selector_->setParameters(side_angle, min_switching_v_yaw);
  armor_selector_->setFacingParameters(facing_enter_angle, facing_exit_angle);
  armor_selector_->setRadialDynamicParameters(
    radial_dynamic_enable,
    radial_dynamic_v_yaw_ref,
    radial_dynamic_shrink_ratio,
    radial_dynamic_min_angle_deg,
    radial_dynamic_bias_gain_deg,
    radial_dynamic_max_bias_deg);
  armor_selector_->setVirtualPoseParameters(
    virtual_auto_switch_enable,
    virtual_auto_switch_enter_vyaw,
    virtual_auto_switch_exit_vyaw);
  gimbal_controller::ArmorSelector::SelectionMethod auto_switch_method =
    gimbal_controller::ArmorSelector::SelectionMethod::VIRTUAL_POSE;
  if (virtual_auto_switch_method_str == "virtual_fixed_id") {
    auto_switch_method = gimbal_controller::ArmorSelector::SelectionMethod::VIRTUAL_FIXED_ID;
  }
  armor_selector_->setVirtualAutoSwitchMethod(auto_switch_method);
  armor_selector_->setVirtualAutoSwitchFixedId(virtual_auto_switch_fixed_id);
  armor_selector_->setVirtualFixedId(virtual_fixed_id);
  armor_selector_->setSpVisionParameters(
    sp_vision_low_speed_vyaw,
    sp_vision_shootable_angle_deg,
    sp_vision_coming_angle_deg,
    sp_vision_leaving_angle_deg,
    sp_vision_outpost_coming_angle_deg,
    sp_vision_outpost_leaving_angle_deg,
    sp_vision_hold_current_until_jump,
    sp_vision_zero_speed_fallback);

  // 配置选板策略
  gimbal_controller::ArmorSelector::SelectionMethod sel_method =
    gimbal_controller::ArmorSelector::SelectionMethod::MIN_MOVEMENT_WITH_FACING;
  if (selection_method_str == "min_movement") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::MIN_MOVEMENT;
  } else if (selection_method_str == "min_movement_with_radial") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::MIN_MOVEMENT_WITH_RADIAL;
  } else if (selection_method_str == "decision_angle") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::DECISION_ANGLE;
  } else if (selection_method_str == "virtual_pose") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::VIRTUAL_POSE;
  } else if (selection_method_str == "virtual_fixed_id") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::VIRTUAL_FIXED_ID;
  } else if (selection_method_str == "facing_or_virtual_pose") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::FACING_OR_VIRTUAL_POSE;
  } else if (selection_method_str == "facing_or_virtual_fixed_id") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::FACING_OR_VIRTUAL_FIXED_ID;
  } else if (selection_method_str == "sp_vision_25" || selection_method_str == "sp_vision") {
    sel_method = gimbal_controller::ArmorSelector::SelectionMethod::SP_VISION_25;
  }
  armor_selector_->setSelectionMethod(sel_method);
  radial_selection_enabled_ =
    (sel_method == gimbal_controller::ArmorSelector::SelectionMethod::MIN_MOVEMENT_WITH_RADIAL);
  RCLCPP_INFO(get_logger(), "[GimbalController] selection_method: %s", selection_method_str.c_str());
  RCLCPP_INFO(
    get_logger(),
    "[DelayUnified] pred_extra=%.4fs ctrl_latency=%.4fs trig2muzzle=%.4fs max_proc=%.4fs iters=%d",
    prediction_delay,
    controller_delay,
    trigger_to_muzzle_s,
    max_processing_delay_s,
    fire_flight_time_iters);
  RCLCPP_INFO(
    get_logger(),
    "[FireVisibility] policy=%s opening=%.2fdeg use_kinematics=%s vel_lpf=%s alpha=%.2f reset=%.3fs",
    fire_target_visibility_policy.c_str(),
    fire_facing_filter_opening_angle_deg,
    fire_use_gimbal_kinematics ? "true" : "false",
    fire_velocity_low_pass_enable ? "true" : "false",
    std::clamp(fire_velocity_low_pass_alpha, 0.0, 1.0),
    std::max(fire_velocity_low_pass_reset_timeout_s, 0.0));

  fire_advisor_->setParameters(shooting_range_w, shooting_range_h);
  if (fire_policy == "ellipse") {
    fire_advisor_->setDecisionPolicy(
      std::make_shared<gimbal_controller::EllipseFireDecisionPolicy>());
  } else {
    fire_advisor_->setDecisionPolicy(
      std::make_shared<gimbal_controller::AxisThresholdFireDecisionPolicy>());
  }
  if (fire_advice_engine_) {
    fire_advice_engine_->setFlightTimeIterations(fire_flight_time_iters);
    fire_advice_engine_->setFacingFilterOpeningAngleDeg(fire_facing_filter_opening_angle_deg);
    fire_advice_engine_->setUseGimbalKinematics(fire_use_gimbal_kinematics);
    gimbal_controller::FireAdviceVelocityLowPassConfig velocity_filter_cfg;
    velocity_filter_cfg.enable = fire_velocity_low_pass_enable;
    velocity_filter_cfg.alpha = fire_velocity_low_pass_alpha;
    velocity_filter_cfg.reset_timeout_s = fire_velocity_low_pass_reset_timeout_s;
    fire_advice_engine_->setVelocityLowPassConfig(velocity_filter_cfg);
    gimbal_controller::fire_advice::ProbabilityConfig prob_cfg;
    prob_cfg.enable = fire_probability_enable;
    prob_cfg.future_window_ms = std::max(fire_probability_window_ms, 0.0);
    prob_cfg.future_step_ms = std::max(fire_probability_step_ms, 1.0);
    prob_cfg.softmax_fusion = (fire_probability_window_fusion == "softmax");
    prob_cfg.softmax_beta = fire_probability_softmax_beta;
    prob_cfg.use_tracker_covariance =
      get_parameter("controller.fire.probability.use_tracker_covariance").as_bool();
    prob_cfg.strict_covariance =
      get_parameter("controller.fire.probability.strict_covariance").as_bool();
    prob_cfg.fallback_sigma_x =
      get_parameter("controller.fire.probability.fallback_sigma_x").as_double();
    prob_cfg.fallback_sigma_y =
      get_parameter("controller.fire.probability.fallback_sigma_y").as_double();
    prob_cfg.fallback_sigma_z =
      get_parameter("controller.fire.probability.fallback_sigma_z").as_double();
    prob_cfg.sigma_x0 =
      get_parameter("controller.fire.probability.ballistic_sigma_x0").as_double();
    prob_cfg.sigma_y0 =
      get_parameter("controller.fire.probability.ballistic_sigma_y0").as_double();
    prob_cfg.sigma_z0 =
      get_parameter("controller.fire.probability.ballistic_sigma_z0").as_double();
    prob_cfg.growth_x =
      get_parameter("controller.fire.probability.ballistic_growth_x").as_double();
    prob_cfg.growth_y =
      get_parameter("controller.fire.probability.ballistic_growth_y").as_double();
    prob_cfg.growth_z =
      get_parameter("controller.fire.probability.ballistic_growth_z").as_double();
    prob_cfg.enable_normal_velocity_weight =
      get_parameter("controller.fire.probability.normal_velocity_weight.enable").as_bool();
    prob_cfg.normal_v_ref =
      get_parameter("controller.fire.probability.normal_velocity_weight.v_ref").as_double();
    prob_cfg.normal_w_min =
      get_parameter("controller.fire.probability.normal_velocity_weight.w_min").as_double();
    prob_cfg.enable_normal_velocity_gate =
      get_parameter("controller.fire.probability.normal_velocity_gate.enable").as_bool();
    prob_cfg.require_front_face =
      get_parameter("controller.fire.probability.normal_velocity_gate.require_front_face").as_bool();
    prob_cfg.normal_v_activate_min =
      get_parameter("controller.fire.probability.normal_velocity_gate.v_activate_min").as_double();
    prob_cfg.front_face_epsilon =
      get_parameter("controller.fire.probability.normal_velocity_gate.front_epsilon").as_double();
    prob_cfg.max_complement_angle_deg =
      get_parameter("controller.fire.probability.normal_velocity_gate.max_complement_angle_deg").as_double();
    // Reuse solver hitbox size as probability hit rectangle in SI meters.
    prob_cfg.armor_width_m = std::max(shooting_range_w, 1e-6);
    prob_cfg.armor_height_m = std::max(shooting_range_h, 1e-6);

    gimbal_controller::fire_advice::SigmaPointConfig sigma_cfg;
    sigma_cfg.enable = get_parameter("controller.fire.probability.sigma_point.enable").as_bool();
    sigma_cfg.use_unscented =
      get_parameter("controller.fire.probability.sigma_point.method").as_string() == "unscented";
    sigma_cfg.sigma_v0 =
      get_parameter("controller.fire.probability.sigma_point.sigma_v0").as_double();
    sigma_cfg.sigma_delay =
      get_parameter("controller.fire.probability.sigma_point.sigma_delay").as_double();
    sigma_cfg.rho = get_parameter("controller.fire.probability.sigma_point.rho").as_double();
    sigma_cfg.alpha = get_parameter("controller.fire.probability.sigma_point.alpha").as_double();
    sigma_cfg.beta = get_parameter("controller.fire.probability.sigma_point.beta").as_double();
    sigma_cfg.kappa = get_parameter("controller.fire.probability.sigma_point.kappa").as_double();

    gimbal_controller::fire_advice::FireGateConfig gate_cfg;
    if (fire_probability_gate_strategy == "burst_evidence") {
      gate_cfg.strategy = gimbal_controller::fire_advice::FireGateConfig::Strategy::kBurstEvidence;
    } else if (fire_probability_gate_strategy == "legacy") {
      gate_cfg.strategy = gimbal_controller::fire_advice::FireGateConfig::Strategy::kLegacy;
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Unknown controller.fire.probability.gate.strategy='%s', fallback to legacy.",
        fire_probability_gate_strategy.c_str());
      gate_cfg.strategy = gimbal_controller::fire_advice::FireGateConfig::Strategy::kLegacy;
    }
    gate_cfg.integrator_mode =
      get_parameter("controller.fire.probability.gate.mode").as_string() == "integrator";
    gate_cfg.alpha = get_parameter("controller.fire.probability.gate.alpha").as_double();
    gate_cfg.fire_on_th = get_parameter("controller.fire.probability.gate.fire_on_th").as_double();
    gate_cfg.fire_off_th = get_parameter("controller.fire.probability.gate.fire_off_th").as_double();
    gate_cfg.integrator_base_probability =
      get_parameter("controller.fire.probability.gate.integrator_base_probability").as_double();
    gate_cfg.integrator_rise =
      get_parameter("controller.fire.probability.gate.integrator_rise").as_double();
    gate_cfg.integrator_fall =
      get_parameter("controller.fire.probability.gate.integrator_fall").as_double();
    gate_cfg.burst_bullet_count = std::max(fire_probability_burst_count, 1);
    gate_cfg.min_hit_count = std::max(fire_probability_min_hit_count, 1);
    gate_cfg.reference_probability_p0 = fire_probability_ref_p0;
    gate_cfg.evidence_window_ms = std::max(fire_probability_evidence_window_ms, 0.0);
    gate_cfg.log_evidence_clip = fire_probability_evidence_log_clip;
    gate_cfg.evidence_epsilon = fire_probability_evidence_epsilon;
    gate_cfg.neutralize_unshootable_samples = fire_probability_neutralize_unshootable_samples;
    gate_cfg.negative_evidence_scale = fire_probability_negative_evidence_scale;
    gate_cfg.negative_clip_scale = fire_probability_negative_clip_scale;
    gate_cfg.evidence_deadband = fire_probability_evidence_deadband;
    gate_cfg.temperature = fire_probability_temperature;
    gate_cfg.theta_on_cold = fire_probability_theta_on_cold;
    gate_cfg.theta_on_hot = fire_probability_theta_on_hot;
    gate_cfg.theta_hold_cold = fire_probability_theta_hold_cold;
    gate_cfg.theta_hold_hot = fire_probability_theta_hold_hot;
    gate_cfg.theta_reset_cold = fire_probability_theta_reset_cold;
    gate_cfg.theta_reset_hot = fire_probability_theta_reset_hot;
    gate_cfg.min_fire_ms = std::max(fire_probability_min_fire_ms, 0.0);
    gate_cfg.cooldown_ms = std::max(fire_probability_cooldown_ms, 0.0);

    fire_advice_engine_->setProbabilityConfig(prob_cfg, sigma_cfg, gate_cfg);
  }
  if (gimbal_control_core_) {
    gimbal_controller::FireDecisionConfig fire_cfg;
    fire_cfg.prediction_delay_s = std::max(prediction_delay, 0.0);
    fire_cfg.control_latency_s = std::max(controller_delay, 0.0);
    fire_cfg.trigger_to_muzzle_s = std::max(trigger_to_muzzle_s, 0.0);
    fire_cfg.max_processing_delay_s = std::max(max_processing_delay_s, 0.0);
    fire_cfg.yaw_offset_rad = yaw_offset * M_PI / 180.0;
    fire_cfg.pitch_offset_rad = pitch_offset * M_PI / 180.0;
    fire_cfg.include_processing_delay = true;
    fire_cfg.include_control_latency_in_target_prediction = false;
    gimbal_control_core_->setFireDecisionConfig(fire_cfg);
  }
  local_compensator_->setParameters(bullet_speed_, gravity, resistance,
                                    iteration_times);

  // Strategy-specific configuration
  auto predicted_strategy = std::dynamic_pointer_cast<
      gimbal_controller::PredictedPositionStrategy>(
      gimbal_strategies_["predicted"]);
  if (predicted_strategy) {
    predicted_strategy->setPredictionParameters(prediction_delay, max_prediction_time);
    predicted_strategy->setMaxProcessingDelay(max_processing_delay_s);
    predicted_strategy->setManualOffset(pitch_offset, yaw_offset);
    predicted_strategy->setTrackingCenterParams(max_tracking_v_yaw, transfer_thresh);
    predicted_strategy->setControllerDelay(controller_delay);
    predicted_strategy->setTriggerToMuzzleDelay(trigger_to_muzzle_s);
  }
  auto current_strategy = std::dynamic_pointer_cast<
      gimbal_controller::CurrentPositionStrategy>(
      gimbal_strategies_["current"]);
  if (current_strategy) {
    current_strategy->setManualOffset(pitch_offset, yaw_offset);
    current_strategy->setControllerDelay(controller_delay);
    current_strategy->setMaxProcessingDelay(max_processing_delay_s);
    current_strategy->setTriggerToMuzzleDelay(trigger_to_muzzle_s);
  }

  // Configure adaptive controller_delay (AIMD)
  bool   adaptive_enable   = get_parameter("controller.solver.adaptive_delay.enable").as_bool();
  int    adaptive_threshold = get_parameter("controller.solver.adaptive_delay.fire_wait_threshold").as_int();
  double adaptive_mul      = get_parameter("controller.solver.adaptive_delay.mul_factor").as_double();
  double adaptive_step     = get_parameter("controller.solver.adaptive_delay.add_step").as_double();
  double adaptive_max      = get_parameter("controller.solver.adaptive_delay.max_delay").as_double();
  double adaptive_min      = get_parameter("controller.solver.adaptive_delay.min_delay").as_double();
  double adaptive_max_lin  = get_parameter("controller.solver.adaptive_delay.max_linear_speed").as_double();
  double adaptive_max_ang  = get_parameter("controller.solver.adaptive_delay.max_angular_speed").as_double();

  RCLCPP_INFO(get_logger(),
    "[GimbalController] adaptive_delay: %s (init=%.4f s, min=%.4f, max=%.4f, "
    "add_step=%.4f, mul=%.2f, thresh=%d, max_lin=%.1f, max_ang=%.1f)",
    adaptive_enable ? "ENABLED" : "disabled",
    controller_delay, adaptive_min, adaptive_max,
    adaptive_step, adaptive_mul, adaptive_threshold,
    adaptive_max_lin, adaptive_max_ang);

  if (predicted_strategy) {
    predicted_strategy->setAdaptiveDelayParams(
      adaptive_enable, controller_delay,
      adaptive_min, adaptive_max,
      adaptive_step, adaptive_mul, adaptive_threshold,
      adaptive_max_lin, adaptive_max_ang);
  }
  if (current_strategy) {
    current_strategy->setAdaptiveDelayParams(
      adaptive_enable, controller_delay,
      adaptive_min, adaptive_max,
      adaptive_step, adaptive_mul, adaptive_threshold,
      adaptive_max_lin, adaptive_max_ang);
  }

  double sm_facing_enter = get_parameter("controller.state_machine.facing_enter_angle").as_double();
  double sm_facing_exit = get_parameter("controller.state_machine.facing_exit_angle").as_double();
  double sm_spin_thresh = get_parameter("controller.state_machine.spin_v_yaw_thresh").as_double();
  double sm_calm_thresh = get_parameter("controller.state_machine.calm_v_yaw_thresh").as_double();
  int sm_spin_enter = get_parameter("controller.state_machine.spin_enter_count").as_int();
  int sm_spin_exit = get_parameter("controller.state_machine.spin_exit_count").as_int();
  double sm_side_angle = get_parameter("controller.state_machine.side_angle").as_double();
  double sm_prediction_delay = prediction_delay;
  double sm_max_prediction = readCompatDoubleParameter(
    *this,
    "controller.state_machine.max_prediction_time",
    "state_machine.max_prediction_time");

  auto sm_strategy_ptr = std::dynamic_pointer_cast<
      gimbal_controller::StateMachineStrategy>(
      gimbal_strategies_["state_machine"]);
  if (sm_strategy_ptr) {
    sm_strategy_ptr->setFacingParameters(sm_facing_enter, sm_facing_exit);
    sm_strategy_ptr->setSpinParameters(sm_spin_thresh, sm_calm_thresh,
                                       sm_spin_enter, sm_spin_exit,
                                       sm_side_angle);
    sm_strategy_ptr->setPredictionParameters(sm_prediction_delay,
                                             sm_max_prediction);
    sm_strategy_ptr->setMaxProcessingDelay(max_processing_delay_s);
    sm_strategy_ptr->setManualOffset(pitch_offset, yaw_offset);
    sm_strategy_ptr->setTriggerToMuzzleDelay(trigger_to_muzzle_s);
  }

  // ── 配置 GimbalCmd 输出端保护滤波器 ──
  {
    gimbal_controller::GimbalCmdFilterConfig fcfg;
    fcfg.enable_clamping             = get_parameter("controller.output_filter.enable_clamping").as_bool();
    fcfg.max_yaw_diff                = get_parameter("controller.output_filter.max_yaw_diff").as_double();
    fcfg.max_pitch_diff              = get_parameter("controller.output_filter.max_pitch_diff").as_double();
    fcfg.enable_outlier_rejection    = get_parameter("controller.output_filter.enable_outlier_rejection").as_bool();
    fcfg.outlier_threshold_yaw       = get_parameter("controller.output_filter.outlier_threshold_yaw").as_double();
    fcfg.outlier_threshold_pitch     = get_parameter("controller.output_filter.outlier_threshold_pitch").as_double();
    fcfg.max_outlier_count           = get_parameter("controller.output_filter.max_outlier_count").as_int();
    fcfg.enable_rate_limiter         = get_parameter("controller.output_filter.enable_rate_limiter").as_bool();
    fcfg.max_yaw_rate                = get_parameter("controller.output_filter.max_yaw_rate").as_double();
    fcfg.max_pitch_rate              = get_parameter("controller.output_filter.max_pitch_rate").as_double();
    fcfg.enable_moving_average       = get_parameter("controller.output_filter.enable_moving_average").as_bool();
    fcfg.moving_average_window_size  = get_parameter("controller.output_filter.moving_average_window_size").as_int();
    fcfg.enable_ema                  = get_parameter("controller.output_filter.enable_ema").as_bool();
    fcfg.ema_alpha                   = get_parameter("controller.output_filter.ema_alpha").as_double();
    fcfg.enable_one_euro             = get_parameter("controller.output_filter.enable_one_euro").as_bool();
    fcfg.one_euro_freq               = get_parameter("controller.output_filter.one_euro_freq").as_double();
    fcfg.one_euro_min_cutoff         = get_parameter("controller.output_filter.one_euro_min_cutoff").as_double();
    fcfg.one_euro_beta               = get_parameter("controller.output_filter.one_euro_beta").as_double();
    fcfg.one_euro_d_cutoff           = get_parameter("controller.output_filter.one_euro_d_cutoff").as_double();
    if (gimbal_control_core_) {
      gimbal_control_core_->setFilterConfig(fcfg);
    }
    RCLCPP_INFO(get_logger(),
      "[GimbalCmdFilter] clamp=%s(%.1f°,%.1f°) outlier=%s(%.1f°,%.1f°,max%d)"
      " rate=%s(%.1f°,%.1f°) mean=%s(win=%d) ema=%s(a=%.2f) 1euro=%s(f=%.0f,mc=%.2f,b=%.4f)",
      fcfg.enable_clamping ? "ON" : "off", fcfg.max_yaw_diff, fcfg.max_pitch_diff,
      fcfg.enable_outlier_rejection ? "ON" : "off",
      fcfg.outlier_threshold_yaw, fcfg.outlier_threshold_pitch, fcfg.max_outlier_count,
      fcfg.enable_rate_limiter ? "ON" : "off", fcfg.max_yaw_rate, fcfg.max_pitch_rate,
      fcfg.enable_moving_average ? "ON" : "off", fcfg.moving_average_window_size,
      fcfg.enable_ema ? "ON" : "off", fcfg.ema_alpha,
      fcfg.enable_one_euro ? "ON" : "off",
      fcfg.one_euro_freq, fcfg.one_euro_min_cutoff, fcfg.one_euro_beta);
  }

  RCLCPP_INFO(get_logger(),
     "GimbalPipelineNode initialized: target_frame=%s, control_rate=%.0f Hz, "
     "selector_strategy=%s, gimbal_strategy=%s, ballistic_mode=%s",
     target_frame_.c_str(), control_rate_,
     selector_strategy_name_.c_str(),
     current_gimbal_strategy_name_.c_str(),
     ballistic_mode_.c_str());

  // ── 5. ROS2 external interfaces ──
  rclcpp::QoS sensor_qos(10);
  sensor_qos.best_effort();

  // Subscribe: /armor_detector/armors via tf2_ros::MessageFilter
  // This mirrors armor_solver's design: the callback is only invoked once
  // the TF transform at the message's timestamp is available in tf2_buffer_,
  // guaranteeing that TFHandler::transform_pose() uses the correct
  // camera-frame → odom transform (the one that matches the image capture
  // moment) and not a stale/future transform that would cause drift.
  armors_sub_.subscribe(this, "/armor_detector/armors",
                         rmw_qos_profile_sensor_data);
  tf2_filter_ = std::make_shared<tf2_armor_filter>(
      armors_sub_, *tf2_buffer_, target_frame_,
      /*queue_size=*/10,
      get_node_logging_interface(), get_node_clock_interface(),
      std::chrono::duration<int>(1));
  tf2_filter_->registerCallback(&GimbalPipelineNode::armorsCallback, this);

  // Subscribe: /joint_states (input from serial driver)
  joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      std::bind(&GimbalPipelineNode::jointStateCallback, this,
                std::placeholders::_1));

  // Subscribe: camera_info (for FOV soft constraint)
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "camera_info", rclcpp::SensorDataQoS(),
      std::bind(&GimbalPipelineNode::cameraInfoCallback, this,
                std::placeholders::_1));

  // Publish: cmd_gimbal (output to serial driver)
  gimbal_cmd_pub_ = create_publisher<rm_interfaces::msg::GimbalCmd>(
      "cmd_gimbal", rclcpp::SensorDataQoS());

  // Maneuver states publisher (always-on, for chart monitoring)
  maneuver_states_pub_ =
      create_publisher<rm_interfaces::msg::ManeuverStates>(
          "~/maneuver_states", rclcpp::SensorDataQoS());

  // Debug publishers
  if (debug_mode_) {
    debug_tracked_robots_pub_ =
        create_publisher<rm_interfaces::msg::TrackedRobots>(
            "~/tracked_robots", sensor_qos);
    debug_selected_target_pub_ =
        create_publisher<rm_interfaces::msg::SelectedTarget>(
            "~/selected_target", rclcpp::SensorDataQoS());
    debug_target_pub_ = create_publisher<rm_interfaces::msg::Target>(
        "~/target", sensor_qos);
    debug_delay_audit_pub_ = create_publisher<rm_interfaces::msg::DelayAudit>(
      "~/delay_audit", rclcpp::SensorDataQoS());
    debug_fire_advice_pub_ = create_publisher<rm_interfaces::msg::FireAdviceDebug>(
      "~/fire_advice_debug", rclcpp::SensorDataQoS());
    debug_armor_selection_pub_ = create_publisher<std_msgs::msg::String>(
      "~/armor_selection_debug", rclcpp::SensorDataQoS());
    debug_evidence_frame_pub_ = create_publisher<std_msgs::msg::String>(
      "~/evidence_frame_debug", rclcpp::SensorDataQoS());
    debug_tracker_marker_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(
            "~/tracker_markers", 10);
    debug_gimbal_marker_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(
            "~/gimbal_markers", 10);
    debug_maneuver_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(
            "~/maneuver_markers", 10);
    if (fire_prob_image_debug_enable_) {
      debug_fire_plane_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "~/fire_debug/armor_plane", rclcpp::SensorDataQoS());
      debug_fire_normal_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "~/fire_debug/normal_view", rclcpp::SensorDataQoS());
    }
    if (tracker_2d_image_debug_enable_) {
      debug_tracker_2d_image_pub_ =
          create_publisher<sensor_msgs::msg::CompressedImage>(
              "~/tracker_debug/TwoD_tracks/compressed", rclcpp::SensorDataQoS());
    }
  }

  RCLCPP_INFO(get_logger(), "Subscribed to topics: /armor_detector/armors (with TF sync), /joint_states, camera_info");

  // Service: ~/set_mode
  set_mode_srv_ = create_service<rm_interfaces::srv::SetMode>(
      "~/set_mode",
      std::bind(&GimbalPipelineNode::setModeCallback, this,
                std::placeholders::_1, std::placeholders::_2));

  // Timer: 250 Hz control loop
  auto period = std::chrono::duration<double>(1.0 / control_rate_);
  control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&GimbalPipelineNode::timerCallback, this));

  if (debug_mode_) initMarkers();

  RCLCPP_INFO(get_logger(), "Service ~/set_mode ready");

  // ─── Prediction logger ────────────────────────────────────────
  if (get_parameter("logging.enable").as_bool()) {
    prediction_logger_ = std::make_unique<PredictionLogger>(
        get_parameter("logging.output_dir").as_string(),
        get_parameter("logging.robot_id_filter").as_string(),
        get_parameter("logging.flush_every_n").as_int());
    RCLCPP_INFO(get_logger(), "PredictionLogger enabled, output: %s",
                get_parameter("logging.output_dir").as_string().c_str());
  }

  RCLCPP_INFO(get_logger(), "PredictionLogger: %s", prediction_logger_ ? "enabled" : "disabled");

  // ── 6. Heartbeat ──
  heartbeat_ = HeartBeatPublisher::create(this);

  RCLCPP_INFO(get_logger(), "GimbalPipelineNode (unified pipeline) initialized successfully");

  RCLCPP_INFO(get_logger(),
              "GimbalPipelineNode initialized: target_frame=%s, "
              "control_rate=%.0f Hz, strategy=%s, ballistic=%s",
              target_frame_.c_str(), control_rate_,
              current_gimbal_strategy_name_.c_str(),
              ballistic_mode_.c_str());
}

/* ================================================================ */
/*  Parameter declarations                                           */
/* ================================================================ */

void GimbalPipelineNode::declareTrackerParameters() {
  // Basic
  declare_parameter("target_frame", "odom");
  declare_parameter("source_frame", "camera_optical_frame");
  declare_parameter("predict_rate", 100.0);
  declare_parameter("default_r1", 0.15);
  declare_parameter("default_r2", 0.20);
  declare_parameter("default_dza", 0.0);
  declare_parameter("tracker_timeout", 0.5);
  declare_parameter("debug_mode", false);
  declare_parameter("enable_oscillation_detection", false);
  declare_parameter("visualization_frame", "odom");
  declare_parameter("tracker.debug_2d_viz.enable", false);
  declare_parameter("tracker.debug_2d_viz.width", 960);
  declare_parameter("tracker.debug_2d_viz.height", 540);
  declare_parameter("tracker.debug_2d_viz.jpeg_quality", 70);
  declare_parameter("robot_description.strict_unknown_reject", true);
  declare_parameter("robot_description.default_projection_mode", std::string("yaw_plane"));
  declare_parameter(
    "robot_description.full_se3_ids",
    std::vector<std::string>{"big_buff", "small_buff"});
  declare_parameter("robot_description.full_se3_robot_types", std::vector<int64_t>{});
  declare_parameter("external_targets.enable", false);
  declare_parameter("external_targets.buff.enable", false);
  declare_parameter("external_targets.buff.topic", std::string("/auto_buff/tracked_robot"));
  declare_parameter("external_targets.buff.timeout_s", 0.3);
  declare_parameter("external_targets.allowed_ids_by_mode.mode_0", std::vector<std::string>{});
  declare_parameter("external_targets.allowed_ids_by_mode.mode_1", std::vector<std::string>{});
  declare_parameter("external_targets.allowed_ids_by_mode.mode_2", std::vector<std::string>{"small_buff"});
  declare_parameter("external_targets.allowed_ids_by_mode.mode_3", std::vector<std::string>{"small_buff"});
  declare_parameter("external_targets.allowed_ids_by_mode.mode_4", std::vector<std::string>{"big_buff"});
  declare_parameter("external_targets.allowed_ids_by_mode.mode_5", std::vector<std::string>{"big_buff"});

  // UKF
  declare_parameter("ukf.alpha", 0.001);
  declare_parameter("ukf.beta", 2.0);
  declare_parameter("ukf.kappa", 0.0);
  declare_parameter("ukf.obs_noise_pos", 0.05);
  declare_parameter("ukf.obs_noise_yaw", 0.05);
  declare_parameter("ukf.enable_ypd_observation_noise", false);
  declare_parameter("ukf.ypd_sigma_azi", 0.01);
  declare_parameter("ukf.ypd_sigma_ele", 0.01);
  declare_parameter("ukf.ypd_sigma_dist_coeff", 0.08);
  declare_parameter("ukf.dual_obs_noise_pos", 0.01);
  declare_parameter("ukf.dual_obs_noise_yaw", 0.03);
  declare_parameter("ukf.dual_obs_geometry_noise_scale", 0.2);
  declare_parameter("ukf.single_obs_update_weight_pos", 0.05);
  declare_parameter("ukf.enable_innovation_gating", false);
  declare_parameter("ukf.innovation_gate_chi2_threshold", 9.49);

  // Motion
  declare_parameter("motion.translation_model", "CA");
  declare_parameter("motion.cv_process_noise_vel", 0.5);
  declare_parameter("motion.ca_process_noise_acc", 1.0);
  declare_parameter("motion.singer_alpha", 0.5);
  declare_parameter("motion.singer_sigma", 2.0);
  declare_parameter("motion.process_noise_r", 0.02);
  declare_parameter("motion.process_noise_dz", 0.005);

  // Spin
  declare_parameter("spin.spin_process_noise_yaw_rate", 0.3);
  declare_parameter("spin.spin_process_noise_yaw_acc", 1.0);
  declare_parameter("spin.spin_process_noise_delta_rate", 0.3);
  declare_parameter("spin.spin_process_noise_delta_acc", 3.0);

  // Entropy
  declare_parameter("entropy.temperature", 2.0);
  declare_parameter("entropy.use_adaptive", true);
  declare_parameter("entropy.k_prior_weight", 0.7);

  // Tracker
  declare_parameter("tracker.implementation", std::string("adaptive"));
  declare_parameter("tracker.tracking_thres", 2);
  declare_parameter("tracker.lost_thres", 8);
  declare_parameter("tracker.temp_lost_thres", 3);
  declare_parameter("tracker.max_match_distance", 2.0);
  declare_parameter("tracker.max_match_yaw_diff", 1.0);
  declare_parameter("tracker.n_panels", 4);
  declare_parameter("tracker.panel_angle_step", M_PI / 2.0);
  declare_parameter("tracker.periodic_binding_enable", false);
  declare_parameter("tracker.periodic_binding_weight", 0.35);
  declare_parameter("tracker.periodic_binding_spin_rate_gate", 0.8);
  declare_parameter("tracker.jump_binding_enable", true);
  declare_parameter("tracker.jump_binding_confirm_frames", 3);
  declare_parameter("tracker.jump_binding_z_jump_min", 0.015);
  declare_parameter("tracker.jump_binding_dz_match_tolerance", 0.03);
  declare_parameter("tracker.jump_binding_dz_gate", 0.010);
  declare_parameter("tracker.jump_binding_yaw_err_gate", 0.35);
  declare_parameter("tracker.jump_binding_cost_margin_min", 0.08);
  declare_parameter("tracker.jump_binding_switch_cooldown", 2);
  declare_parameter("tracker.jump_binding_dz_ema_alpha", 0.20);
  declare_parameter("tracker.jump_binding_confidence_floor", 0.15);
  declare_parameter("tracker.degraded_single_obs_enable", true);
  declare_parameter("tracker.degraded_single_obs_streak", 8);
  declare_parameter("tracker.degraded_q_scale_r", 4.0);
  declare_parameter("tracker.degraded_q_scale_dza", 4.0);

  // Constraints
  declare_parameter("constraints.min_radius", 0.12);
  declare_parameter("constraints.max_radius", 0.5);
  declare_parameter("constraints.min_dz", -1.0);
  declare_parameter("constraints.max_dz", 1.0);

  // Outpost-specific (known 3-armor geometry + max-entropy mode switch)
  declare_parameter("outpost.translation_model", "CV");
  declare_parameter("outpost.rotation_model", "CV");
  declare_parameter("outpost.use_tracker_v2", false);
  declare_parameter("outpost.use_tracker_v3", false);
  declare_parameter("outpost.tracking_thres", 2);
  declare_parameter("outpost.lost_thres", 40);
  declare_parameter("outpost.temp_lost_thres", 30);
  declare_parameter("outpost.max_match_distance", 2.0);
  declare_parameter("outpost.max_match_yaw_diff", 1.0);
  declare_parameter("outpost.singer_alpha", 0.0);
  declare_parameter("outpost.singer_sigma", 0.0);
  declare_parameter("outpost.spin_process_noise_theta_rate", 0.0);
  declare_parameter("outpost.spin_process_noise_theta_acc", 0.0);
  declare_parameter("outpost.radius", 0.26);
  // Outpost semantic contract:
  //   id0=highest, id1=middle, id2=lowest.
  declare_parameter("outpost.z_offset_0", 0.06);
  declare_parameter("outpost.z_offset_1", 0.0);
  declare_parameter("outpost.z_offset_2", -0.06);
  declare_parameter("outpost.panel_angle_step", 2.0 * M_PI / 3.0);
  declare_parameter("outpost.softmax_temperature", 1.5);
  declare_parameter("outpost.weight_yaw", 1.0);
  declare_parameter("outpost.weight_z_state", 6.0);
  declare_parameter("outpost.weight_z_history", 2.0);
  declare_parameter("outpost.weight_xy_residual", 2.5);
  declare_parameter("outpost.weight_switch_penalty", 0.05);
  declare_parameter("outpost.entropy_enter", 0.75);
  declare_parameter("outpost.entropy_exit", 0.55);
  declare_parameter("outpost.max_prob_enter", 0.60);
  declare_parameter("outpost.max_prob_exit", 0.75);
  declare_parameter("outpost.stable_frames", 4);
  declare_parameter("outpost.z_history_window", 15);
  declare_parameter("outpost.single_mode_confidence_scale", 0.70);
  declare_parameter("outpost.binding_use_new_binder_pipeline", false);
  declare_parameter("outpost.binding_enable_multi_obs", true);
  declare_parameter("outpost.binding_transition_confirm_frames", 3);
  declare_parameter("outpost.binding_same_panel_yaw_gate", 0.35);
  declare_parameter("outpost.binding_same_panel_z_gate", 0.08);
  declare_parameter("outpost.binding_same_panel_xy_gate", 0.18);
  declare_parameter("outpost.binding_min_candidate_prob", 0.40);
  declare_parameter("outpost.binding_min_candidate_margin", 0.12);
  declare_parameter("outpost.binding_switch_strong_score", 0.60);
  declare_parameter("outpost.binding_period_window", 12);
  declare_parameter("outpost.binding_period_weight", 0.60);
  declare_parameter("outpost.binding_topology_prior_weight", 4.0);
  declare_parameter("outpost.binding_period_min_spin_rate", 0.8);
  declare_parameter("outpost.spin_direction_confirm_frames", 3);
  declare_parameter("outpost.binding_period_update_min_confidence", 0.55);
  declare_parameter("outpost.binding_period_update_min_jump", 0.015);
  declare_parameter("outpost.binding_dz_ema_alpha", 0.20);
  declare_parameter("outpost.binding_confidence_floor", 0.15);
  declare_parameter("outpost.z_audit_rebind_enable", true);
  declare_parameter("outpost.z_audit_rebind_confirm_frames", 3);
  declare_parameter("outpost.z_audit_rebind_min_confidence", 0.60);
  declare_parameter("outpost.z_audit_rebind_min_jump", 0.015);
  declare_parameter("outpost.binding_conflict_position_scale", 0.10);
  declare_parameter("outpost.alpha_pos", 0.65);
  declare_parameter("outpost.beta_vel", 0.30);
  declare_parameter("outpost.alpha_yaw", 0.60);
  declare_parameter("outpost.beta_yaw_rate", 0.25);
  declare_parameter("outpost.assume_static_center", true);
  declare_parameter("outpost.linear_velocity_damping", 0.90);
  declare_parameter("outpost.yaw_rate_damping", 0.98);
  declare_parameter("outpost.max_center_speed", 1.00);
  declare_parameter("outpost.max_yaw_rate", 12.0);
  declare_parameter("outpost.max_yaw_rate_step", 3.0);
  declare_parameter("outpost.mode_enter_confirm_frames", 3);
  declare_parameter("outpost.mode_exit_confirm_frames", 4);
  declare_parameter("outpost.mode_min_dwell_frames", 6);
  declare_parameter("outpost.mode_enter_threshold", 0.72);
  declare_parameter("outpost.mode_exit_threshold", 0.45);
  declare_parameter("outpost.mode_weight_jump", 0.30);
  declare_parameter("outpost.mode_weight_dual", 0.20);
  declare_parameter("outpost.mode_weight_margin", 0.20);
  declare_parameter("outpost.mode_weight_health", 0.20);
  declare_parameter("outpost.mode_weight_entropy", 0.10);
  declare_parameter("outpost.ambiguous_publish_single_armor_semantics", true);
  declare_parameter("outpost.ambiguous_single_armor_zero_offset", true);
  declare_parameter("outpost.ambiguous_backend_use_imm_adapter", false);
  declare_parameter("outpost.v2_warmup_enable", true);
  declare_parameter("outpost.v2_warmup_min_groups", 3);
  declare_parameter("outpost.v2_warmup_min_samples_per_group", 2);
  declare_parameter("outpost.v2_warmup_max_frames", 60);
  declare_parameter("outpost.v2_warmup_z_jump_gate", 0.025);
  declare_parameter("outpost.v2_warmup_yaw_jump_gate", 0.75);
  declare_parameter("outpost.v2_warmup_xyz_jump_gate", 0.18);
  declare_parameter("outpost.v2_warmup_ratio_min", 1.55);
  declare_parameter("outpost.v2_warmup_ratio_max", 2.45);
  declare_parameter("outpost.v2_warmup_min_large_diff", 0.06);
  declare_parameter("outpost.v3.topk", 3);
  declare_parameter("outpost.v3.min_top1_confidence", 0.5);
  declare_parameter("outpost.v3.min_top1_top2_margin", 1.0);
  declare_parameter("outpost.v3.max_reconstruction_pos_error", 0.3);
  declare_parameter("outpost.v3.gate_single_total_nis", 11.34);
  declare_parameter("outpost.v3.gate_single_pos_chi2", 9.0);
  declare_parameter("outpost.v3.posterior_max_center_jump", 0.5);
  declare_parameter("outpost.v3.posterior_max_yaw_jump", 0.5);
  declare_parameter("outpost.v3.posterior_max_yaw_rate", 15.0);
  declare_parameter("outpost.v3.posterior_max_yaw_acc", 30.0);
  declare_parameter("outpost.v3.mode_p_enter_structured", 0.7);
  declare_parameter("outpost.v3.mode_m_enter_structured", 1.5);
  declare_parameter("outpost.v3.mode_stable_frames", 5);
  declare_parameter("outpost.v3.mode_p_exit_structured", 0.4);
  declare_parameter("outpost.v3.mode_m_exit_structured", 0.5);
  declare_parameter("outpost.v3.mode_degraded_frames", 10);
  declare_parameter("outpost.v3.prior_panel_switch_penalty", 0.5);
  declare_parameter("outpost.v3.initial_p_pos", 0.01);
  declare_parameter("outpost.v3.initial_p_vel", 1.0);
  declare_parameter("outpost.v3.initial_p_acc", 10.0);
  declare_parameter("outpost.v3.initial_p_yaw", 0.1);
  declare_parameter("outpost.v3.initial_p_yaw_rate", 1.0);
  declare_parameter("outpost.v3.initial_p_yaw_acc", 5.0);
  declare_parameter("outpost.v3.process_noise_acc", 2.0);
  declare_parameter("outpost.v3.process_noise_yaw_acc", 3.0);
  declare_parameter("outpost.v3.observation_sigma_pos_xy", 0.02);
  declare_parameter("outpost.v3.observation_sigma_pos_z", 0.03);
  declare_parameter("outpost.v3.warmup_enable", true);
  declare_parameter("outpost.v3.warmup_frames", 8);
  declare_parameter("outpost.v3.warmup_min_settle_frames", 3);
  declare_parameter("outpost.v3.warmup_min_margin_to_commit", 1.2);
  declare_parameter("outpost.v3.warmup_min_confidence_to_commit", 0.65);
  declare_parameter("outpost.v3.phase_audit_enable", true);
  declare_parameter("outpost.v3.phase_audit_min_jump", 0.015);
  declare_parameter("outpost.v3.phase_audit_dz_gate", 0.035);
  declare_parameter("outpost.v3.phase_audit_confirm_frames", 2);

  // Maneuver detection
  declare_parameter("maneuver.enable", true);
  declare_parameter("maneuver.nis_threshold_single", 238.807);
  declare_parameter("maneuver.nis_threshold_dual", 4132.110);
  declare_parameter("maneuver.innov_norm_threshold_single", 0.1279);
  declare_parameter("maneuver.innov_norm_threshold_dual", 0.0613);
  declare_parameter("maneuver.mad_filter_enable", false);
  declare_parameter("maneuver.mad_window", 10);
  declare_parameter("maneuver.mad_k", 3.0);

  // Common binder config (Norm4/Outpost v2 pipeline)
  declare_parameter("binder.confirm_frames", 3);
  declare_parameter("binder.lock_new_hold_frames", 2);
  declare_parameter("binder.force_rebind_bad_frames", 10);
  declare_parameter("binder.pending_window_frames", 0);
  declare_parameter("binder.post_jump_min_confidence", 0.45);
  declare_parameter("binder.confidence_floor", 0.15);
  declare_parameter("binder.z_jump_min", 0.015);
  declare_parameter("binder.dz_match_tolerance", 0.03);
  declare_parameter("binder.dz_gate", 0.010);
  declare_parameter("binder.yaw_err_gate", 0.35);
  declare_parameter("binder.cost_margin_min", 0.08);
  declare_parameter("binder.dz_ema_alpha", 0.20);
  declare_parameter("binder.periodic_enable", false);
  declare_parameter("binder.periodic_window", 12);
  declare_parameter("binder.periodic_weight", 0.60);
  declare_parameter("binder.periodic_min_spin_rate", 0.8);
  declare_parameter("binder.periodic_update_min_jump", 0.015);
  declare_parameter("binder.periodic_signature_threshold", 0.60);
  declare_parameter("binder.reacquire_gap_dt_gate", 0.12);
  declare_parameter("binder.reacquire_lost_frames_gate", 1);
  declare_parameter("binder.z_cluster_ema_alpha", 0.25);
  declare_parameter("binder.z_cluster_assign_gate", 0.10);
  declare_parameter("binder.min_candidate_prob", 0.40);
  declare_parameter("binder.min_candidate_margin", 0.12);
  declare_parameter("binder.switch_strong_score", 0.60);
  declare_parameter("binder.single_obs_history_window", 8);
  declare_parameter("binder.dual_obs_enable", true);
  declare_parameter("binder.scorer_enable", true);
  declare_parameter("binder.same_panel_yaw_gate", 0.35);
  declare_parameter("binder.same_panel_z_gate", 0.08);
  declare_parameter("binder.same_panel_xy_gate", 0.18);
  declare_parameter("binder.z_audit_rebind_enable", false);
  declare_parameter("binder.z_audit_rebind_confirm_frames", 3);
  declare_parameter("binder.z_audit_rebind_min_confidence", 0.60);
  declare_parameter("binder.z_audit_rebind_min_jump", 0.015);
  declare_parameter("binder.enable_soft_fusion", false);
  declare_parameter("binder.soft_fusion_w_seq", 0.25);
  declare_parameter("binder.soft_fusion_w_geo", 0.40);
  declare_parameter("binder.soft_fusion_w_dyn", 0.20);
  declare_parameter("binder.soft_fusion_w_continuity", 0.15);
  declare_parameter("binder.soft_fusion_w_topology", 0.15);

  // Norm4 v2 common pipeline / anti-pingpong controls
  declare_parameter("norm4_v2.enable_common_pipeline", false);
  declare_parameter("norm4_v2.enable_phase_memory", true);
  declare_parameter("norm4_v2.enable_kinematic_anti_pingpong", true);
  declare_parameter("norm4_v2.enable_2d_tracker", false);
  declare_parameter("norm4_v2.enable_proxy_manager", false);
  declare_parameter("norm4_v2.phase_memory.enable_phase_memory", true);
  declare_parameter("norm4_v2.phase_memory.enable_kinematic_anti_pingpong", true);
  declare_parameter("norm4_v2.phase_memory.sequence_window_size", 10);
  declare_parameter("norm4_v2.phase_memory.ping_pong_pattern_threshold", 0.7);
  declare_parameter("norm4_v2.phase_memory.enable_opposite_jump_detect", true);
  declare_parameter(
      "norm4_v2.phase_memory.anti_pingpong.min_consistent_frames_to_commit", 3);
  declare_parameter("norm4_v2.phase_memory.anti_pingpong.jerk_gate", 1.5);
  declare_parameter("norm4_v2.phase_memory.anti_pingpong.yaw_rate_jump_gate", 2.0);
  declare_parameter("norm4_v2.phase_memory.anti_pingpong.velocity_dir_cos_min", 0.2);
  declare_parameter("norm4_v2.phase_memory.anti_pingpong.pending_timeout_frames", 12);

  // Norm4 V2 UKF Backend V1
  declare_parameter("norm4_v2.ukf_v1.enabled", true);
  declare_parameter("norm4_v2.ukf_v1.force_rotation_ca", false);
  declare_parameter("norm4_v2.ukf_v1.dual_raw_batch", true);
  declare_parameter("norm4_v2.ukf_v1.sigma_pos_xy", 0.06);
  declare_parameter("norm4_v2.ukf_v1.sigma_pos_z", 0.08);
  declare_parameter("norm4_v2.ukf_v1.sigma_yaw", 0.12);
  declare_parameter("norm4_v2.ukf_v1.dual_raw_R_scale", 1.5);
  declare_parameter("norm4_v2.ukf_v1.gate.single_total_nis", 25.0);
  declare_parameter("norm4_v2.ukf_v1.gate.single_pos_chi2", 16.0);
  declare_parameter("norm4_v2.ukf_v1.gate.single_yaw_chi2", 9.0);
  declare_parameter("norm4_v2.ukf_v1.gate.dual_total_nis", 45.0);
  declare_parameter("norm4_v2.ukf_v1.gate.dual_each_pos_chi2", 16.0);
  declare_parameter("norm4_v2.ukf_v1.gate.dual_each_yaw_chi2", 9.0);
  declare_parameter("norm4_v2.ukf_v1.single_update.structural_gain_r", 0.0);
  declare_parameter("norm4_v2.ukf_v1.single_update.structural_gain_dza", 0.0);
  declare_parameter("norm4_v2.ukf_v1.dual_update.structural_gain_r", 0.05);
  declare_parameter("norm4_v2.ukf_v1.dual_update.structural_gain_dza", 0.02);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.max_center_jump", 0.25);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.max_yaw_jump", 0.80);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.min_r", 0.05);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.max_r", 0.50);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.max_r_jump", 0.05);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.min_dza", 0.0);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.max_dza", 0.15);
  declare_parameter("norm4_v2.ukf_v1.posterior_sanity.max_dza_jump", 0.03);

  // Norm4 V2 Hypothesis Selector
  declare_parameter("norm4_v2.hypothesis_selector.topk", 4);
  declare_parameter("norm4_v2.hypothesis_selector.commit_top1_only", true);
  declare_parameter("norm4_v2.hypothesis_selector.min_top1_confidence", 0.55);
  declare_parameter("norm4_v2.hypothesis_selector.min_top1_top2_margin", 0.0);
  declare_parameter("norm4_v2.hypothesis_selector.ambiguous_margin", 1.0);
  declare_parameter("norm4_v2.hypothesis_selector.include_rejected_in_debug", true);
  declare_parameter("norm4_v2.hypothesis_selector.evidence_prior_enable", false);
  declare_parameter("norm4_v2.hypothesis_selector.max_reconstruction_pos_error", 0.30);

  // Norm4 V2 Warmup
  declare_parameter("norm4_v2.warmup.enable_dual_seed_01", true);
  declare_parameter("norm4_v2.warmup.warmup_frames", 8);
  declare_parameter("norm4_v2.warmup.min_settle_frames", 3);
  declare_parameter("norm4_v2.warmup.min_margin_to_commit", 1.5);
  declare_parameter("norm4_v2.warmup.min_confidence_to_commit", 0.70);

  // Norm4 V2 Mode Routing
  declare_parameter("norm4_v2.mode_routing.ambiguous_output", "single_plate_3d");
  declare_parameter("norm4_v2.mode_routing.structured_output", "structured_ukf");
  declare_parameter("norm4_v2.mode_routing.ambiguous_structured_backend_mode", "shallow_or_predict");
  declare_parameter("norm4_v2.mode_routing.structured_single_plate_mode", "shallow");

  // Norm4 V2 Single-Plate Bridge
  declare_parameter("norm4_v2.single_plate_bridge.enable", false);
  declare_parameter("norm4_v2.single_plate_bridge.source_semantic", "track2d_id");
  declare_parameter("norm4_v2.single_plate_bridge.backend_type", "norm4_ambiguous_backend");
  declare_parameter("norm4_v2.single_plate_bridge.require_semantic_stable_frames", 2);

  // Norm4 V2 Fallback
  declare_parameter("norm4_v2.fallback.predict_only_on_reject", true);
  declare_parameter("norm4_v2.fallback.enable_ambiguous_single_fallback", true);

  // Norm4 V3 (dedicated for trackers/norm4_v3/tracker/norm4_tracker_v2.hpp)
  declare_parameter("norm4_v3.enable_common_pipeline", false);
  declare_parameter("norm4_v3.enable_phase_memory", true);
  declare_parameter("norm4_v3.enable_kinematic_anti_pingpong", true);
  declare_parameter("norm4_v3.enable_2d_tracker", false);
  declare_parameter("norm4_v3.enable_proxy_manager", false);
  declare_parameter("norm4_v3.phase_memory.enable_phase_memory", true);
  declare_parameter("norm4_v3.phase_memory.enable_kinematic_anti_pingpong", true);
  declare_parameter("norm4_v3.phase_memory.sequence_window_size", 10);
  declare_parameter("norm4_v3.phase_memory.ping_pong_pattern_threshold", 0.7);
  declare_parameter("norm4_v3.phase_memory.enable_opposite_jump_detect", true);
  declare_parameter(
      "norm4_v3.phase_memory.anti_pingpong.min_consistent_frames_to_commit", 3);
  declare_parameter("norm4_v3.phase_memory.anti_pingpong.jerk_gate", 1.5);
  declare_parameter("norm4_v3.phase_memory.anti_pingpong.yaw_rate_jump_gate", 2.0);
  declare_parameter("norm4_v3.phase_memory.anti_pingpong.velocity_dir_cos_min", 0.2);
  declare_parameter("norm4_v3.phase_memory.anti_pingpong.pending_timeout_frames", 12);

  declare_parameter("norm4_v3.ukf_v1.enabled", true);
  declare_parameter("norm4_v3.ukf_v1.force_rotation_ca", false);
  declare_parameter("norm4_v3.ukf_v1.dual_raw_batch", true);
  declare_parameter("norm4_v3.ukf_v1.sigma_pos_xy", 0.06);
  declare_parameter("norm4_v3.ukf_v1.sigma_pos_z", 0.08);
  declare_parameter("norm4_v3.ukf_v1.sigma_yaw", 0.12);
  declare_parameter("norm4_v3.ukf_v1.dual_raw_R_scale", 1.5);
  declare_parameter("norm4_v3.ukf_v1.gate.single_total_nis", 25.0);
  declare_parameter("norm4_v3.ukf_v1.gate.single_pos_chi2", 16.0);
  declare_parameter("norm4_v3.ukf_v1.gate.single_yaw_chi2", 9.0);
  declare_parameter("norm4_v3.ukf_v1.gate.dual_total_nis", 45.0);
  declare_parameter("norm4_v3.ukf_v1.gate.dual_each_pos_chi2", 16.0);
  declare_parameter("norm4_v3.ukf_v1.gate.dual_each_yaw_chi2", 9.0);
  declare_parameter("norm4_v3.ukf_v1.single_update.structural_gain_r", 0.0);
  declare_parameter("norm4_v3.ukf_v1.single_update.structural_gain_dza", 0.0);
  declare_parameter("norm4_v3.ukf_v1.dual_update.structural_gain_r", 0.05);
  declare_parameter("norm4_v3.ukf_v1.dual_update.structural_gain_dza", 0.02);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.max_center_jump", 0.25);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.max_yaw_jump", 0.80);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.min_r", 0.05);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.max_r", 0.50);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.max_r_jump", 0.05);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.min_dza", 0.0);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.max_dza", 0.15);
  declare_parameter("norm4_v3.ukf_v1.posterior_sanity.max_dza_jump", 0.03);

  declare_parameter("norm4_v3.ukf_v2.enabled", true);
  declare_parameter("norm4_v3.ukf_v2.force_rotation_ca", false);
  declare_parameter("norm4_v3.ukf_v2.dual_raw_batch", true);
  declare_parameter("norm4_v3.ukf_v2.sigma_pos_xy", 0.06);
  declare_parameter("norm4_v3.ukf_v2.sigma_pos_z", 0.08);
  declare_parameter("norm4_v3.ukf_v2.sigma_yaw", 0.12);
  declare_parameter("norm4_v3.ukf_v2.dual_raw_R_scale", 1.5);
  declare_parameter("norm4_v3.ukf_v2.gate.single_total_nis", 25.0);
  declare_parameter("norm4_v3.ukf_v2.gate.single_pos_chi2", 16.0);
  declare_parameter("norm4_v3.ukf_v2.gate.single_yaw_chi2", 9.0);
  declare_parameter("norm4_v3.ukf_v2.gate.dual_total_nis", 45.0);
  declare_parameter("norm4_v3.ukf_v2.gate.dual_each_pos_chi2", 16.0);
  declare_parameter("norm4_v3.ukf_v2.gate.dual_each_yaw_chi2", 9.0);
  declare_parameter("norm4_v3.ukf_v2.single_update.structural_gain_r", 0.0);
  declare_parameter("norm4_v3.ukf_v2.single_update.structural_gain_dza", 0.0);
  declare_parameter("norm4_v3.ukf_v2.dual_update.structural_gain_r", 0.05);
  declare_parameter("norm4_v3.ukf_v2.dual_update.structural_gain_dza", 0.02);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.max_center_jump", 0.25);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.max_yaw_jump", 0.80);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.min_r", 0.05);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.max_r", 0.50);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.max_r_jump", 0.05);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.min_dza", 0.0);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.max_dza", 0.15);
  declare_parameter("norm4_v3.ukf_v2.posterior_sanity.max_dza_jump", 0.03);

  declare_parameter("norm4_v3.inekf.enabled", true);
  declare_parameter("norm4_v3.inekf.force_rotation_ca", false);
  declare_parameter("norm4_v3.inekf.dual_raw_batch", true);
  declare_parameter("norm4_v3.inekf.sigma_pos_xy", 0.06);
  declare_parameter("norm4_v3.inekf.sigma_pos_z", 0.08);
  declare_parameter("norm4_v3.inekf.sigma_yaw", 0.12);
  declare_parameter("norm4_v3.inekf.dual_raw_R_scale", 1.5);
  declare_parameter("norm4_v3.inekf.gate.single_total_nis", 25.0);
  declare_parameter("norm4_v3.inekf.gate.single_pos_chi2", 16.0);
  declare_parameter("norm4_v3.inekf.gate.single_yaw_chi2", 9.0);
  declare_parameter("norm4_v3.inekf.gate.dual_total_nis", 45.0);
  declare_parameter("norm4_v3.inekf.gate.dual_each_pos_chi2", 16.0);
  declare_parameter("norm4_v3.inekf.gate.dual_each_yaw_chi2", 9.0);
  declare_parameter("norm4_v3.inekf.single_update.structural_gain_r", 0.0);
  declare_parameter("norm4_v3.inekf.single_update.structural_gain_dza", 0.0);
  declare_parameter("norm4_v3.inekf.dual_update.structural_gain_r", 0.05);
  declare_parameter("norm4_v3.inekf.dual_update.structural_gain_dza", 0.02);
  declare_parameter("norm4_v3.inekf.posterior_sanity.max_center_jump", 0.25);
  declare_parameter("norm4_v3.inekf.posterior_sanity.max_yaw_jump", 0.80);
  declare_parameter("norm4_v3.inekf.posterior_sanity.min_r", 0.05);
  declare_parameter("norm4_v3.inekf.posterior_sanity.max_r", 0.50);
  declare_parameter("norm4_v3.inekf.posterior_sanity.max_r_jump", 0.05);
  declare_parameter("norm4_v3.inekf.posterior_sanity.min_dza", 0.0);
  declare_parameter("norm4_v3.inekf.posterior_sanity.max_dza", 0.15);
  declare_parameter("norm4_v3.inekf.posterior_sanity.max_dza_jump", 0.03);

  declare_parameter("norm4_v3.slow_structure.enable", true);
  declare_parameter("norm4_v3.slow_structure.q_theta_r1", 1.0e-6);
  declare_parameter("norm4_v3.slow_structure.q_theta_r2", 1.0e-6);
  declare_parameter("norm4_v3.slow_structure.q_theta_dza", 5.0e-7);
  declare_parameter("norm4_v3.slow_structure.prior_r1", 0.15);
  declare_parameter("norm4_v3.slow_structure.prior_r2", 0.20);
  declare_parameter("norm4_v3.slow_structure.prior_dza", 0.0);
  declare_parameter("norm4_v3.slow_structure.prior_sigma_r", 0.06);
  declare_parameter("norm4_v3.slow_structure.prior_sigma_dza", 0.06);
  declare_parameter("norm4_v3.slow_structure.alpha_r1_single", 0.0);
  declare_parameter("norm4_v3.slow_structure.alpha_r2_single", 0.0);
  declare_parameter("norm4_v3.slow_structure.alpha_dza_single", 0.0);
  declare_parameter("norm4_v3.slow_structure.alpha_r1_dual", 0.05);
  declare_parameter("norm4_v3.slow_structure.alpha_r2_dual", 0.05);
  declare_parameter("norm4_v3.slow_structure.alpha_dza_dual", 0.02);
  declare_parameter("norm4_v3.slow_structure.prior_pull_gain", 0.002);
  declare_parameter("norm4_v3.slow_structure.min_r", 0.05);
  declare_parameter("norm4_v3.slow_structure.max_r", 0.50);
  declare_parameter("norm4_v3.slow_structure.min_dza", 0.0);
  declare_parameter("norm4_v3.slow_structure.max_dza", 0.12);

  declare_parameter("norm4_v3.backend_config.backend_type", "ukf_v1");
  declare_parameter("norm4_v3.backend_config.motion_profile", "default");
  declare_parameter("norm4_v3.backend_config.noise_profile", "default");
  declare_parameter("norm4_v3.backend_config.structure_profile", "slow");
  declare_parameter("norm4_v3.inekf_runtime.motion_profile", "default");
  declare_parameter("norm4_v3.inekf_runtime.noise_profile", "default");
  declare_parameter("norm4_v3.inekf_runtime.structure_profile", "slow");
  declare_parameter("norm4_v3.inekf_runtime.translation_model", "");
  declare_parameter("norm4_v3.inekf_runtime.cv_process_noise_vel", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.ca_process_noise_acc", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.singer_alpha", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.singer_sigma", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.process_noise_r", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.process_noise_dz", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.spin_process_noise_delta_rate", -1.0);
  declare_parameter("norm4_v3.inekf_runtime.spin_process_noise_delta_acc", -1.0);

  declare_parameter("norm4_v3.hypothesis_selector.topk", 4);
  declare_parameter("norm4_v3.hypothesis_selector.commit_top1_only", true);
  declare_parameter("norm4_v3.hypothesis_selector.min_top1_confidence", 0.55);
  declare_parameter("norm4_v3.hypothesis_selector.min_top1_top2_margin", 0.0);
  declare_parameter("norm4_v3.hypothesis_selector.ambiguous_margin", 1.0);
  declare_parameter("norm4_v3.hypothesis_selector.include_rejected_in_debug", true);
  declare_parameter("norm4_v3.hypothesis_selector.evidence_prior_enable", false);
  declare_parameter("norm4_v3.hypothesis_selector.max_reconstruction_pos_error", 0.30);

  declare_parameter("norm4_v3.warmup.enable_dual_seed_01", true);
  declare_parameter("norm4_v3.warmup.warmup_frames", 8);
  declare_parameter("norm4_v3.warmup.min_settle_frames", 3);
  declare_parameter("norm4_v3.warmup.min_margin_to_commit", 1.5);
  declare_parameter("norm4_v3.warmup.min_confidence_to_commit", 0.70);

  declare_parameter("norm4_v3.mode_routing.ambiguous_output", "single_plate_3d");
  declare_parameter("norm4_v3.mode_routing.structured_output", "structured_ukf");
  declare_parameter("norm4_v3.mode_routing.ambiguous_structured_backend_mode", "shallow_or_predict");
  declare_parameter("norm4_v3.mode_routing.structured_single_plate_mode", "shallow");

  declare_parameter("norm4_v3.single_plate_bridge.enable", false);
  declare_parameter("norm4_v3.single_plate_bridge.source_semantic", "track2d_id");
  declare_parameter("norm4_v3.single_plate_bridge.backend_type", "norm4_ambiguous_backend");
  declare_parameter("norm4_v3.single_plate_bridge.require_semantic_stable_frames", 2);

  declare_parameter("norm4_v3.fallback.predict_only_on_reject", true);
  declare_parameter("norm4_v3.fallback.enable_ambiguous_single_fallback", true);
  declare_parameter("norm4_v3.debug_log.enable", false);
  declare_parameter("norm4_v3.debug_log.throttle_ms", 500);
  declare_parameter("norm4_v3.debug_log.verbose", false);

  // Panel mismatch detection
  declare_parameter("panel_mismatch.enable", true);
  declare_parameter("panel_mismatch.window_size", 8);
  declare_parameter("panel_mismatch.threshold_t1", 0.0009);
  declare_parameter("panel_mismatch.confirm_count", 3);
  declare_parameter("panel_mismatch.reinit_count", 5);
  declare_parameter("panel_mismatch.apply_correction", false);

  // Output smoother
  declare_parameter("smoother.enable", true);
  declare_parameter("smoother.enable_position_smooth", true);
  declare_parameter("smoother.enable_yaw_smooth", true);
  declare_parameter("smoother.enable_velocity_smooth", true);
  declare_parameter("smoother.enable_structural_convergence", true);
  declare_parameter("smoother.pos_min_cutoff", 1.5);
  declare_parameter("smoother.pos_beta", 0.01);
  declare_parameter("smoother.pos_d_cutoff", 1.0);
  declare_parameter("smoother.yaw_min_cutoff", 1.0);
  declare_parameter("smoother.yaw_beta", 0.005);
  declare_parameter("smoother.yaw_d_cutoff", 1.0);
  declare_parameter("smoother.vel_min_cutoff", 2.0);
  declare_parameter("smoother.vel_beta", 0.01);
  declare_parameter("smoother.vel_d_cutoff", 1.0);
  declare_parameter("smoother.rm_initial_step", 0.5);
  declare_parameter("smoother.rm_gamma", 0.75);
  declare_parameter("smoother.rm_n0", 5);
  declare_parameter("smoother.rm_dual_obs_boost", 3.0);
  declare_parameter("smoother.rm_min_radius", 0.12);
  declare_parameter("smoother.rm_max_radius", 0.5);
  declare_parameter("smoother.rm_min_dz", -1.0);
  declare_parameter("smoother.rm_max_dz", 1.0);
  declare_parameter("smoother.rm_convergence_eps", 1e-4);
  declare_parameter("smoother.default_freq", 30.0);

  // Outlier filter (independent of smoother.enable)
  declare_parameter("smoother.enable_outlier_filter",    false);
  declare_parameter("smoother.outlier_method",           std::string("mad"));
  declare_parameter("smoother.outlier_window_size",      10);
  declare_parameter("smoother.outlier_min_samples",      5);
  declare_parameter("smoother.outlier_mad_k",            3.5);
  declare_parameter("smoother.outlier_iqr_k",            1.5);
  declare_parameter("smoother.outlier_mahal_threshold",  9.21);
}

void GimbalPipelineNode::declareTargetSelectorParameters() {
  declare_parameter("selector.strategy", "min_yaw_deviation");
  declare_parameter("selector.reference_yaw", 0.0);
  declare_parameter("selector.max_yaw_deviation", M_PI);
  declare_parameter("selector.max_distance", 10.0);
  declare_parameter("selector.min_confidence", 0.3);
  declare_parameter("selector.hysteresis_threshold", 0.1);
  declare_parameter("selector.priority_robot_ids", std::vector<std::string>{});
  declare_parameter("selector.sticky_lock_frames", 3);
  declare_parameter("selector.sticky_lost_frames", 3);
}

void GimbalPipelineNode::declareGimbalControllerParameters() {
  declare_parameter("controller.bullet_speed", 20.0);
  declare_parameter("controller.control_rate", 250.0);
  declare_parameter("controller.strategy", "current");
  declare_parameter("controller.ballistic_mode", "service");

  // Solver
  declare_parameter("controller.solver.shooting_range_width", 0.135);
  declare_parameter("controller.solver.shooting_range_height", 0.135);
  declare_parameter("controller.solver.side_angle", 15.0);
  declare_parameter("controller.solver.min_switching_v_yaw", 1.0);
  declare_parameter("controller.solver.prediction_delay", 0.0);
  declare_parameter("controller.solver.max_prediction_time", 0.5);
  declare_parameter("controller.solver.max_tracking_v_yaw", 6.0);
  declare_parameter("controller.solver.transfer_thresh", 5);
  declare_parameter("controller.solver.gravity", 9.8);
  declare_parameter("controller.solver.resistance", 0.001);
  declare_parameter("controller.solver.iteration_times", 20);
  declare_parameter("controller.solver.pitch_offset", 0.0);
  declare_parameter("controller.solver.yaw_offset", 0.0);
  declare_parameter("controller.solver.facing_enter_angle", 40.0);
  declare_parameter("controller.solver.facing_exit_angle", 55.0);
  declare_parameter("controller.solver.radial_dynamic.enable", false);
  declare_parameter("controller.solver.radial_dynamic.v_yaw_ref", 8.0);
  declare_parameter("controller.solver.radial_dynamic.shrink_ratio", 0.6);
  declare_parameter("controller.solver.radial_dynamic.min_angle_deg", 5.0);
  declare_parameter("controller.solver.radial_dynamic.bias_gain_deg", 0.0);
  declare_parameter("controller.solver.radial_dynamic.max_bias_deg", 0.0);
  declare_parameter("controller.solver.virtual_pose.auto_switch.enable", false);
  declare_parameter("controller.solver.virtual_pose.auto_switch.enter_vyaw", 8.0);
  declare_parameter("controller.solver.virtual_pose.auto_switch.exit_vyaw", 6.0);
  declare_parameter("controller.solver.virtual_pose.auto_switch.selection_method",
                    std::string("virtual_pose"));
  declare_parameter("controller.solver.virtual_pose.auto_switch.fixed_id", 0);
  declare_parameter("controller.solver.virtual_pose.fixed_id", 0);
  declare_parameter("controller.solver.sp_vision.low_speed_vyaw", 2.0);
  declare_parameter("controller.solver.sp_vision.shootable_angle_deg", 60.0);
  declare_parameter("controller.solver.sp_vision.coming_angle_deg", 60.0);
  declare_parameter("controller.solver.sp_vision.leaving_angle_deg", 20.0);
  declare_parameter("controller.solver.sp_vision.outpost_coming_angle_deg", 70.0);
  declare_parameter("controller.solver.sp_vision.outpost_leaving_angle_deg", 30.0);
  declare_parameter("controller.solver.sp_vision.hold_current_until_jump", false);
  declare_parameter("controller.solver.sp_vision.zero_speed_fallback", true);
  declare_parameter("controller.solver.controller_delay", 0.0);
  declare_parameter("controller.solver.trigger_to_muzzle_s", 0.0);
  declare_parameter("controller.solver.selection_method", std::string("min_movement_with_facing"));

  // Unified delay parameters (preferred)
  declare_parameter("controller.delay.prediction_extra_s", 0.0);
  declare_parameter("controller.delay.control_latency_s", 0.0);
  declare_parameter("controller.delay.trigger_to_muzzle_s", 0.0);
  declare_parameter("controller.delay.max_processing_delay_s", 0.5);
  declare_parameter("controller.delay.flight_time_iters", 2);

  declare_parameter("controller.fire.trigger_to_muzzle_s", 0.0);
  declare_parameter("controller.fire.decision_policy", std::string("axis_threshold"));
  declare_parameter("controller.fire.flight_time_iters", 2);
  declare_parameter("controller.fire.facing_filter_opening_angle_deg", 180.0);
  declare_parameter("controller.fire.use_gimbal_kinematics", false);
  declare_parameter("controller.fire.target_visibility_policy", std::string("facing_only"));
  declare_parameter("controller.fire.velocity_low_pass.enable", true);
  declare_parameter("controller.fire.velocity_low_pass.alpha", 0.35);
  declare_parameter("controller.fire.velocity_low_pass.reset_timeout_s", 0.25);
  declare_parameter("controller.fire.probability.enable", false);
  declare_parameter("controller.fire.probability.future_window_ms", 50.0);
  declare_parameter("controller.fire.probability.future_step_ms", 10.0);
  declare_parameter("controller.fire.probability.window_fusion", std::string("max"));
  declare_parameter("controller.fire.probability.softmax_beta", 20.0);
  declare_parameter("controller.fire.probability.use_tracker_covariance", true);
  declare_parameter("controller.fire.probability.strict_covariance", false);
  declare_parameter("controller.fire.probability.fallback_sigma_x", 0.02);
  declare_parameter("controller.fire.probability.fallback_sigma_y", 0.02);
  declare_parameter("controller.fire.probability.fallback_sigma_z", 0.03);
  declare_parameter("controller.fire.probability.ballistic_sigma_x0", 0.010);
  declare_parameter("controller.fire.probability.ballistic_sigma_y0", 0.015);
  declare_parameter("controller.fire.probability.ballistic_sigma_z0", 0.015);
  declare_parameter("controller.fire.probability.ballistic_growth_x", 0.03);
  declare_parameter("controller.fire.probability.ballistic_growth_y", 0.06);
  declare_parameter("controller.fire.probability.ballistic_growth_z", 0.08);
  declare_parameter("controller.fire.probability.normal_velocity_weight.enable", false);
  declare_parameter("controller.fire.probability.normal_velocity_weight.v_ref", 28.0);
  declare_parameter("controller.fire.probability.normal_velocity_weight.w_min", 0.5);
  declare_parameter("controller.fire.probability.normal_velocity_gate.enable", true);
  declare_parameter("controller.fire.probability.normal_velocity_gate.require_front_face", true);
  declare_parameter("controller.fire.probability.normal_velocity_gate.v_activate_min", 8.0);
  declare_parameter("controller.fire.probability.normal_velocity_gate.front_epsilon", 1e-4);
  declare_parameter("controller.fire.probability.normal_velocity_gate.max_complement_angle_deg", 90.0);
  declare_parameter("controller.fire.probability.sigma_point.enable", true);
  declare_parameter("controller.fire.probability.sigma_point.method", std::string("unscented"));
  declare_parameter("controller.fire.probability.sigma_point.sigma_v0", 0.3);
  declare_parameter("controller.fire.probability.sigma_point.sigma_delay", 0.005);
  declare_parameter("controller.fire.probability.sigma_point.rho", 0.0);
  declare_parameter("controller.fire.probability.sigma_point.alpha", 0.7);
  declare_parameter("controller.fire.probability.sigma_point.beta", 2.0);
  declare_parameter("controller.fire.probability.sigma_point.kappa", 0.0);
  declare_parameter("controller.fire.probability.gate.strategy", std::string("legacy"));
  declare_parameter("controller.fire.probability.gate.mode", std::string("lowpass"));
  declare_parameter("controller.fire.probability.gate.alpha", 0.85);
  declare_parameter("controller.fire.probability.gate.fire_on_th", 0.65);
  declare_parameter("controller.fire.probability.gate.fire_off_th", 0.35);
  declare_parameter("controller.fire.probability.gate.integrator_base_probability", 0.45);
  declare_parameter("controller.fire.probability.gate.integrator_rise", 8.0);
  declare_parameter("controller.fire.probability.gate.integrator_fall", 6.0);
  declare_parameter("controller.fire.probability.burst.burst_bullet_count", 5);
  declare_parameter("controller.fire.probability.burst.min_hit_count", 1);
  declare_parameter("controller.fire.probability.evidence.reference_probability_p0", 0.60);
  declare_parameter("controller.fire.probability.evidence.window_ms", 50.0);
  declare_parameter("controller.fire.probability.evidence.log_clip", 2.0);
  declare_parameter("controller.fire.probability.evidence.epsilon", 1e-3);
  declare_parameter("controller.fire.probability.evidence.neutralize_unshootable_samples", true);
  declare_parameter("controller.fire.probability.evidence.negative_evidence_scale", 0.35);
  declare_parameter("controller.fire.probability.evidence.negative_clip_scale", 0.35);
  declare_parameter("controller.fire.probability.evidence.deadband", 0.10);
  declare_parameter("controller.fire.probability.temperature.value", 0.5);
  declare_parameter("controller.fire.probability.temperature.theta_on_cold", 0.90);
  declare_parameter("controller.fire.probability.temperature.theta_on_hot", 0.75);
  declare_parameter("controller.fire.probability.temperature.theta_hold_cold", 0.70);
  declare_parameter("controller.fire.probability.temperature.theta_hold_hot", 0.55);
  declare_parameter("controller.fire.probability.temperature.theta_reset_cold", 0.45);
  declare_parameter("controller.fire.probability.temperature.theta_reset_hot", 0.35);
  declare_parameter("controller.fire.probability.commit.min_fire_ms", 20.0);
  declare_parameter("controller.fire.probability.commit.cooldown_ms", 80.0);
  declare_parameter("controller.fire.visualization.enable", true);
  declare_parameter("controller.fire.visualization.ellipse_samples", 64);
  declare_parameter("controller.fire.visualization.max_impact_points", 120);
  declare_parameter("controller.fire.visualization.image_debug.enable", false);
  declare_parameter("controller.fire.visualization.image_debug.publish_rate_hz", 10.0);
  declare_parameter("controller.fire.visualization.image_debug.width", 960);
  declare_parameter("controller.fire.visualization.image_debug.height", 540);
  declare_parameter("controller.fire.visualization.image_debug.show_text", true);
  declare_parameter("controller.fire.visualization.image_debug.show_sigma_ellipse", true);
  declare_parameter("controller.fire.visualization.image_debug.show_velocity_fan", true);

  // Deprecated aliases (for migration from legacy gimbal_controller keys)
  declare_parameter("solver.prediction_delay", 0.0);
  declare_parameter("solver.max_prediction_time", 0.5);
  declare_parameter("solver.controller_delay", 0.0);
  declare_parameter("solver.trigger_to_muzzle_s", 0.0);

  // Adaptive controller_delay (AIMD)
  declare_parameter("controller.solver.adaptive_delay.enable",              false);
  declare_parameter("controller.solver.adaptive_delay.fire_wait_threshold", 10);
  declare_parameter("controller.solver.adaptive_delay.mul_factor",          1.2);
  declare_parameter("controller.solver.adaptive_delay.add_step",            0.005);
  declare_parameter("controller.solver.adaptive_delay.max_delay",           0.10);
  declare_parameter("controller.solver.adaptive_delay.min_delay",           0.0);
  declare_parameter("controller.solver.adaptive_delay.max_linear_speed",    3.0);
  declare_parameter("controller.solver.adaptive_delay.max_angular_speed",   10.0);

  // State machine
  declare_parameter("controller.state_machine.facing_enter_angle", 40.0);
  declare_parameter("controller.state_machine.facing_exit_angle", 55.0);
  declare_parameter("controller.state_machine.spin_v_yaw_thresh", 4.0);
  declare_parameter("controller.state_machine.calm_v_yaw_thresh", 2.0);
  declare_parameter("controller.state_machine.spin_enter_count", 5);
  declare_parameter("controller.state_machine.spin_exit_count", 5);
  declare_parameter("controller.state_machine.side_angle", 15.0);
  declare_parameter("controller.state_machine.prediction_delay", 0.0);
  declare_parameter("controller.state_machine.max_prediction_time", 0.5);

  // Deprecated aliases (for migration from legacy gimbal_controller keys)
  declare_parameter("state_machine.prediction_delay", 0.0);
  declare_parameter("state_machine.max_prediction_time", 0.5);

  // MPC strategy
  declare_parameter("controller.mpc.N", 20);
  declare_parameter("controller.mpc.dt", 0.01);
  declare_parameter("controller.mpc.control_delay_s", 0.0);
  declare_parameter("controller.mpc.max_accel", 30.0);
  declare_parameter("controller.mpc.q_yaw", 100.0);
  declare_parameter("controller.mpc.q_pitch", 100.0);
  declare_parameter("controller.mpc.q_yaw_vel", 10.0);
  declare_parameter("controller.mpc.q_pitch_vel", 10.0);
  declare_parameter("controller.mpc.r_yaw", 0.01);
  declare_parameter("controller.mpc.r_pitch", 0.01);
  declare_parameter("controller.mpc.s_yaw", 5.0);
  declare_parameter("controller.mpc.s_pitch", 5.0);

  // MPC delay compensation
  declare_parameter("controller.mpc.enable_delay_compensation", true);
  declare_parameter("controller.mpc.allow_muzzle_compensation", true);
  declare_parameter("controller.mpc.prediction_delay_s", 0.0);
  declare_parameter("controller.mpc.flight_time_iters", 2);
  declare_parameter("controller.mpc.max_processing_delay_s", 0.5);
  declare_parameter("controller.mpc.yaw_feedforward_k_s", 0.0);

  // Deprecated aliases (for migration from old unscoped mpc delay keys)
  declare_parameter("mpc.control_delay_s", 0.0);
  declare_parameter("mpc.enable_delay_compensation", true);
  declare_parameter("mpc.prediction_delay_s", 0.0);
  declare_parameter("mpc.flight_time_iters", 2);
  declare_parameter("mpc.max_processing_delay_s", 0.5);

  // MPC 机动自适应权重衰减
  declare_parameter("controller.mpc.maneuver_adapt.enable",  false);
  declare_parameter("controller.mpc.maneuver_adapt.a_max",   3.0);
  declare_parameter("controller.mpc.maneuver_adapt.eta",     0.2);
  declare_parameter("controller.mpc.maneuver_adapt.tau",    10.0);
  declare_parameter("controller.mpc.maneuver_adapt.r_scale", 10.0);

  // MPC 命中概率权重加权 (Q 权重缩放)
  declare_parameter("controller.mpc.weighting.enable", false);
  declare_parameter("controller.mpc.weighting.alpha", 3.0);
  declare_parameter("controller.mpc.weighting.k_omega", 0.5);
  declare_parameter("controller.mpc.weighting.sigma_min", 0.05);
  declare_parameter("controller.mpc.weighting.sigma_max", 0.5);
  declare_parameter("controller.mpc.weighting.sigma_sys", 0.02);
  declare_parameter("controller.mpc.weighting.target_size", 0.135);
  declare_parameter("controller.mpc.weighting.delay_s", 0.0);
  declare_parameter("controller.mpc.weighting.max_w", 6.0);
  declare_parameter("controller.mpc.weighting.smooth_alpha", 0.7);
  declare_parameter("controller.mpc.weighting.min_distance", 0.1);
  declare_parameter("controller.mpc.weighting.sigma_beta", 0.3);
  declare_parameter("controller.mpc.weighting.gamma", 0.5);

  // MPC 轨迹生成前速度 clamp
  declare_parameter("controller.mpc.vel_clamp.enable",          false);
  declare_parameter("controller.mpc.vel_clamp.max_linear_speed", 5.0);
  declare_parameter("controller.mpc.vel_clamp.max_v_yaw",        10.0);

  // MPC FOV 软约束
  declare_parameter("controller.mpc.fov_constraint.enable",                  false);
  declare_parameter("controller.mpc.fov_constraint.margin",                  0.05);
  declare_parameter("controller.mpc.fov_constraint.slack_weight",            1000.0);
  declare_parameter("controller.mpc.fov_constraint.constraint_steps",        0);
  declare_parameter("controller.mpc.fov_constraint.dynamic_margin.enable",   false);
  declare_parameter("controller.mpc.fov_constraint.dynamic_margin.vel_scale",0.01);
  declare_parameter("controller.mpc.fov_constraint.fallback_fov_yaw",        0.35);
  declare_parameter("controller.mpc.fov_constraint.fallback_fov_pitch",      0.26);

  // MPC 数值稳健性: 在线 RMS 归一化
  declare_parameter("controller.mpc.normalization.enable", false);
  declare_parameter("controller.mpc.normalization.mode", std::string("rms"));
  declare_parameter("controller.mpc.normalization.window_size", 80);
  declare_parameter("controller.mpc.normalization.min_samples", 10);
  declare_parameter("controller.mpc.normalization.rms_epsilon", 1e-6);
  declare_parameter("controller.mpc.normalization.typical_state.yaw", 1.0);
  declare_parameter("controller.mpc.normalization.typical_state.pitch", 1.0);
  declare_parameter("controller.mpc.normalization.typical_state.yaw_vel", 1.0);
  declare_parameter("controller.mpc.normalization.typical_state.pitch_vel", 1.0);
  declare_parameter("controller.mpc.normalization.typical_control.yaw_acc", 1.0);
  declare_parameter("controller.mpc.normalization.typical_control.pitch_acc", 1.0);
  declare_parameter("controller.mpc.normalization.typical_delta_control.yaw_acc", 1.0);
  declare_parameter("controller.mpc.normalization.typical_delta_control.pitch_acc", 1.0);

  // MPC 数值稳健性: Hessian 自适应对角正则
  declare_parameter("controller.mpc.regularization.enable", false);
  declare_parameter("controller.mpc.regularization.epsilon_abs", 1e-8);
  declare_parameter("controller.mpc.regularization.epsilon_rel", 1e-6);
  declare_parameter("controller.mpc.regularization.epsilon_max", 1e-2);
  declare_parameter("controller.mpc.regularization.retry_on_fail", true);
  declare_parameter("controller.mpc.regularization.retry_scale", 10.0);

  // MPC 数值诊断: 低成本常开 + 高成本抽样
  declare_parameter("controller.mpc.diagnostics.enable", false);
  declare_parameter("controller.mpc.diagnostics.low_cost_always", true);
  declare_parameter("controller.mpc.diagnostics.high_cost_enable", false);
  declare_parameter("controller.mpc.diagnostics.high_cost_sample_every", 20);
  declare_parameter("controller.mpc.diagnostics.log_every", 50);
  declare_parameter("controller.mpc.diagnostics.log_on_failure", true);
  declare_parameter("controller.mpc.diagnostics.active_tol", 1e-4);
  declare_parameter("controller.mpc.diagnostics.rank_tol_rel", 1e-9);

  // ─── GimbalCmd 输出端保护滤波器 ──────────────────────────────
  // 0. Clamping — 绝对限幅
  declare_parameter("controller.output_filter.enable_clamping",         true);
  declare_parameter("controller.output_filter.max_yaw_diff",            15.0);
  declare_parameter("controller.output_filter.max_pitch_diff",          10.0);
  // 1. 外点检测
  declare_parameter("controller.output_filter.enable_outlier_rejection", true);
  declare_parameter("controller.output_filter.outlier_threshold_yaw",   8.0);
  declare_parameter("controller.output_filter.outlier_threshold_pitch",  5.0);
  declare_parameter("controller.output_filter.max_outlier_count",        3);
  // 2. Rate Limiter
  declare_parameter("controller.output_filter.enable_rate_limiter",     true);
  declare_parameter("controller.output_filter.max_yaw_rate",            5.0);
  declare_parameter("controller.output_filter.max_pitch_rate",          3.0);
  // 3. 滑动窗口均值
  declare_parameter("controller.output_filter.enable_moving_average",   false);
  declare_parameter("controller.output_filter.moving_average_window_size", 3);
  // 4. EMA
  declare_parameter("controller.output_filter.enable_ema",              false);
  declare_parameter("controller.output_filter.ema_alpha",               0.7);
  // 5. 1-Euro 自适应滤波
  declare_parameter("controller.output_filter.enable_one_euro",         false);
  declare_parameter("controller.output_filter.one_euro_freq",           250.0);
  declare_parameter("controller.output_filter.one_euro_min_cutoff",     1.0);
  declare_parameter("controller.output_filter.one_euro_beta",           0.007);
  declare_parameter("controller.output_filter.one_euro_d_cutoff",       1.0);

  // ─── Prediction logger ────────────────────────────────────────
  declare_parameter("logging.enable",          false);
  declare_parameter("logging.output_dir",       std::string("/tmp/prediction_logs"));
  declare_parameter("logging.robot_id_filter",  std::string(""));
  declare_parameter("logging.flush_every_n",    50);
}

/* ================================================================ */
/*  Apply tracker parameters to config                               */
/* ================================================================ */

void GimbalPipelineNode::applyTrackerParamsToConfig() {
  auto &c = tracker_config_;

  c.ukf.alpha = get_parameter("ukf.alpha").as_double();
  c.ukf.beta = get_parameter("ukf.beta").as_double();
  c.ukf.kappa = get_parameter("ukf.kappa").as_double();
  c.ukf.obs_noise_pos = get_parameter("ukf.obs_noise_pos").as_double();
  c.ukf.obs_noise_yaw = get_parameter("ukf.obs_noise_yaw").as_double();
  c.ukf.enable_ypd_observation_noise =
      get_parameter("ukf.enable_ypd_observation_noise").as_bool();
  c.ukf.ypd_sigma_azi = get_parameter("ukf.ypd_sigma_azi").as_double();
  c.ukf.ypd_sigma_ele = get_parameter("ukf.ypd_sigma_ele").as_double();
  c.ukf.ypd_sigma_dist_coeff =
      get_parameter("ukf.ypd_sigma_dist_coeff").as_double();
  c.ukf.dual_obs_noise_pos = get_parameter("ukf.dual_obs_noise_pos").as_double();
  c.ukf.dual_obs_noise_yaw = get_parameter("ukf.dual_obs_noise_yaw").as_double();
  c.ukf.dual_obs_geometry_noise_scale =
      get_parameter("ukf.dual_obs_geometry_noise_scale").as_double();
  c.ukf.single_obs_update_weight_pos =
      get_parameter("ukf.single_obs_update_weight_pos").as_double();
  c.ukf.enable_innovation_gating =
      get_parameter("ukf.enable_innovation_gating").as_bool();
  c.ukf.innovation_gate_chi2_threshold =
      get_parameter("ukf.innovation_gate_chi2_threshold").as_double();

  auto tm_str = get_parameter("motion.translation_model").as_string();
  c.motion.translation_model = translation_model_from_string(tm_str);
  c.motion.cv_process_noise_vel =
      get_parameter("motion.cv_process_noise_vel").as_double();
  c.motion.ca_process_noise_acc =
      get_parameter("motion.ca_process_noise_acc").as_double();
  c.motion.singer_alpha = get_parameter("motion.singer_alpha").as_double();
  c.motion.singer_sigma = get_parameter("motion.singer_sigma").as_double();
  c.motion.process_noise_r = get_parameter("motion.process_noise_r").as_double();
  c.motion.process_noise_dz =
      get_parameter("motion.process_noise_dz").as_double();

  c.spin.spin_process_noise_yaw_rate =
      get_parameter("spin.spin_process_noise_yaw_rate").as_double();
  c.spin.spin_process_noise_yaw_acc =
      get_parameter("spin.spin_process_noise_yaw_acc").as_double();
  c.spin.spin_process_noise_delta_rate =
      get_parameter("spin.spin_process_noise_delta_rate").as_double();
  c.spin.spin_process_noise_delta_acc =
      get_parameter("spin.spin_process_noise_delta_acc").as_double();

  c.entropy.temperature = get_parameter("entropy.temperature").as_double();
  c.entropy.use_adaptive = get_parameter("entropy.use_adaptive").as_bool();
  c.entropy.k_prior_weight =
      get_parameter("entropy.k_prior_weight").as_double();

  c.tracker.implementation =
      get_parameter("tracker.implementation").as_string();
  std::transform(c.tracker.implementation.begin(), c.tracker.implementation.end(),
                 c.tracker.implementation.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (c.tracker.implementation != "adaptive" &&
      c.tracker.implementation != "norm4" &&
      c.tracker.implementation != "norm4_v2") {
    RCLCPP_WARN(
        get_logger(),
        "Unknown tracker.implementation='%s', fallback to 'adaptive'. "
        "Supported values: adaptive | norm4 | norm4_v2",
        c.tracker.implementation.c_str());
    c.tracker.implementation = "adaptive";
  }

  c.tracker.tracking_thres = get_parameter("tracker.tracking_thres").as_int();
  c.tracker.lost_thres = get_parameter("tracker.lost_thres").as_int();
  c.tracker.temp_lost_thres =
      get_parameter("tracker.temp_lost_thres").as_int();
  c.tracker.max_match_distance =
      get_parameter("tracker.max_match_distance").as_double();
  c.tracker.max_match_yaw_diff =
      get_parameter("tracker.max_match_yaw_diff").as_double();
  c.tracker.n_panels = get_parameter("tracker.n_panels").as_int();
  c.tracker.panel_angle_step =
      get_parameter("tracker.panel_angle_step").as_double();
    c.tracker.periodic_binding_enable =
      get_parameter("tracker.periodic_binding_enable").as_bool();
    c.tracker.periodic_binding_weight =
      get_parameter("tracker.periodic_binding_weight").as_double();
    c.tracker.periodic_binding_spin_rate_gate =
      get_parameter("tracker.periodic_binding_spin_rate_gate").as_double();
    c.tracker.jump_binding_enable =
      get_parameter("tracker.jump_binding_enable").as_bool();
    c.tracker.jump_binding_confirm_frames =
      get_parameter("tracker.jump_binding_confirm_frames").as_int();
    c.tracker.jump_binding_z_jump_min =
      get_parameter("tracker.jump_binding_z_jump_min").as_double();
    c.tracker.jump_binding_dz_match_tolerance =
      get_parameter("tracker.jump_binding_dz_match_tolerance").as_double();
    c.tracker.jump_binding_dz_gate =
      get_parameter("tracker.jump_binding_dz_gate").as_double();
    c.tracker.jump_binding_yaw_err_gate =
      get_parameter("tracker.jump_binding_yaw_err_gate").as_double();
    c.tracker.jump_binding_cost_margin_min =
      get_parameter("tracker.jump_binding_cost_margin_min").as_double();
    c.tracker.jump_binding_switch_cooldown =
      get_parameter("tracker.jump_binding_switch_cooldown").as_int();
    c.tracker.jump_binding_dz_ema_alpha =
      get_parameter("tracker.jump_binding_dz_ema_alpha").as_double();
    c.tracker.jump_binding_confidence_floor =
      get_parameter("tracker.jump_binding_confidence_floor").as_double();
    c.tracker.degraded_single_obs_enable =
      get_parameter("tracker.degraded_single_obs_enable").as_bool();
    c.tracker.degraded_single_obs_streak =
      get_parameter("tracker.degraded_single_obs_streak").as_int();
    c.tracker.degraded_q_scale_r =
      get_parameter("tracker.degraded_q_scale_r").as_double();
    c.tracker.degraded_q_scale_dza =
      get_parameter("tracker.degraded_q_scale_dza").as_double();

    c.tracker.periodic_binding_weight =
      std::max(0.0, c.tracker.periodic_binding_weight);
    c.tracker.periodic_binding_spin_rate_gate =
      std::max(0.0, c.tracker.periodic_binding_spin_rate_gate);
    c.tracker.jump_binding_confirm_frames =
      std::max(1, c.tracker.jump_binding_confirm_frames);
    c.tracker.jump_binding_z_jump_min =
      std::max(0.0, c.tracker.jump_binding_z_jump_min);
    c.tracker.jump_binding_dz_match_tolerance =
      std::max(0.0, c.tracker.jump_binding_dz_match_tolerance);
    c.tracker.jump_binding_dz_gate =
      std::max(0.0, c.tracker.jump_binding_dz_gate);
    c.tracker.jump_binding_yaw_err_gate =
      std::max(1e-3, c.tracker.jump_binding_yaw_err_gate);
    c.tracker.jump_binding_cost_margin_min =
      std::max(0.0, c.tracker.jump_binding_cost_margin_min);
    c.tracker.jump_binding_switch_cooldown =
      std::max(0, c.tracker.jump_binding_switch_cooldown);
    c.tracker.jump_binding_dz_ema_alpha =
      std::clamp(c.tracker.jump_binding_dz_ema_alpha, 0.01, 1.0);
    c.tracker.jump_binding_confidence_floor =
      std::clamp(c.tracker.jump_binding_confidence_floor, 0.0, 0.95);
    c.tracker.degraded_single_obs_streak =
      std::max(1, c.tracker.degraded_single_obs_streak);
    c.tracker.degraded_q_scale_r =
      std::max(0.1, c.tracker.degraded_q_scale_r);
    c.tracker.degraded_q_scale_dza =
      std::max(0.1, c.tracker.degraded_q_scale_dza);

  c.constraints.min_radius =
      get_parameter("constraints.min_radius").as_double();
  c.constraints.max_radius =
      get_parameter("constraints.max_radius").as_double();
  c.constraints.min_dz = get_parameter("constraints.min_dz").as_double();
  c.constraints.max_dz = get_parameter("constraints.max_dz").as_double();

  c.maneuver.enable = get_parameter("maneuver.enable").as_bool();
  c.maneuver.nis_threshold_single =
      get_parameter("maneuver.nis_threshold_single").as_double();
  c.maneuver.nis_threshold_dual =
      get_parameter("maneuver.nis_threshold_dual").as_double();
  c.maneuver.innov_norm_threshold_single =
      get_parameter("maneuver.innov_norm_threshold_single").as_double();
  c.maneuver.innov_norm_threshold_dual =
      get_parameter("maneuver.innov_norm_threshold_dual").as_double();
  c.maneuver.mad_filter_enable =
      get_parameter("maneuver.mad_filter_enable").as_bool();
  c.maneuver.mad_window =
      get_parameter("maneuver.mad_window").as_int();
  c.maneuver.mad_k =
      get_parameter("maneuver.mad_k").as_double();
  c.maneuver.mad_window = std::max(1, c.maneuver.mad_window);
  c.maneuver.mad_k = std::max(0.1, c.maneuver.mad_k);

  c.binder.confirm_frames = get_parameter("binder.confirm_frames").as_int();
  c.binder.lock_new_hold_frames =
      get_parameter("binder.lock_new_hold_frames").as_int();
  c.binder.force_rebind_bad_frames =
      get_parameter("binder.force_rebind_bad_frames").as_int();
  c.binder.pending_window_frames =
      get_parameter("binder.pending_window_frames").as_int();
  c.binder.post_jump_min_confidence =
      get_parameter("binder.post_jump_min_confidence").as_double();
  c.binder.confidence_floor =
      get_parameter("binder.confidence_floor").as_double();
  c.binder.z_jump_min = get_parameter("binder.z_jump_min").as_double();
  c.binder.dz_match_tolerance =
      get_parameter("binder.dz_match_tolerance").as_double();
  c.binder.dz_gate = get_parameter("binder.dz_gate").as_double();
  c.binder.yaw_err_gate = get_parameter("binder.yaw_err_gate").as_double();
  c.binder.cost_margin_min =
      get_parameter("binder.cost_margin_min").as_double();
  c.binder.dz_ema_alpha = get_parameter("binder.dz_ema_alpha").as_double();
  c.binder.periodic_enable =
      get_parameter("binder.periodic_enable").as_bool();
  c.binder.periodic_window =
      get_parameter("binder.periodic_window").as_int();
  c.binder.periodic_weight =
      get_parameter("binder.periodic_weight").as_double();
  c.binder.periodic_min_spin_rate =
      get_parameter("binder.periodic_min_spin_rate").as_double();
  c.binder.periodic_update_min_jump =
      get_parameter("binder.periodic_update_min_jump").as_double();
  c.binder.periodic_signature_threshold =
      get_parameter("binder.periodic_signature_threshold").as_double();
  c.binder.reacquire_gap_dt_gate =
      get_parameter("binder.reacquire_gap_dt_gate").as_double();
  c.binder.reacquire_lost_frames_gate =
      get_parameter("binder.reacquire_lost_frames_gate").as_int();
  c.binder.z_cluster_ema_alpha =
      get_parameter("binder.z_cluster_ema_alpha").as_double();
  c.binder.z_cluster_assign_gate =
      get_parameter("binder.z_cluster_assign_gate").as_double();
  c.binder.min_candidate_prob =
      get_parameter("binder.min_candidate_prob").as_double();
  c.binder.min_candidate_margin =
      get_parameter("binder.min_candidate_margin").as_double();
  c.binder.switch_strong_score =
      get_parameter("binder.switch_strong_score").as_double();
  c.binder.single_obs_history_window =
      get_parameter("binder.single_obs_history_window").as_int();
  c.binder.dual_obs_enable =
      get_parameter("binder.dual_obs_enable").as_bool();
  c.binder.scorer_enable = get_parameter("binder.scorer_enable").as_bool();
  c.binder.same_panel_yaw_gate =
      get_parameter("binder.same_panel_yaw_gate").as_double();
  c.binder.same_panel_z_gate =
      get_parameter("binder.same_panel_z_gate").as_double();
  c.binder.same_panel_xy_gate =
      get_parameter("binder.same_panel_xy_gate").as_double();
  c.binder.z_audit_rebind_enable =
      get_parameter("binder.z_audit_rebind_enable").as_bool();
  c.binder.z_audit_rebind_confirm_frames =
      get_parameter("binder.z_audit_rebind_confirm_frames").as_int();
  c.binder.z_audit_rebind_min_confidence =
      get_parameter("binder.z_audit_rebind_min_confidence").as_double();
  c.binder.z_audit_rebind_min_jump =
      get_parameter("binder.z_audit_rebind_min_jump").as_double();
  c.binder.enable_soft_fusion =
      get_parameter("binder.enable_soft_fusion").as_bool();
  c.binder.soft_fusion_w_seq =
      get_parameter("binder.soft_fusion_w_seq").as_double();
  c.binder.soft_fusion_w_geo =
      get_parameter("binder.soft_fusion_w_geo").as_double();
  c.binder.soft_fusion_w_dyn =
      get_parameter("binder.soft_fusion_w_dyn").as_double();
  c.binder.soft_fusion_w_continuity =
      get_parameter("binder.soft_fusion_w_continuity").as_double();
  c.binder.soft_fusion_w_topology =
      get_parameter("binder.soft_fusion_w_topology").as_double();

  c.binder.confirm_frames = std::max(1, c.binder.confirm_frames);
  c.binder.lock_new_hold_frames = std::max(0, c.binder.lock_new_hold_frames);
  c.binder.force_rebind_bad_frames = std::max(1, c.binder.force_rebind_bad_frames);
  c.binder.pending_window_frames = std::max(0, c.binder.pending_window_frames);
  c.binder.post_jump_min_confidence =
      std::clamp(c.binder.post_jump_min_confidence, 0.0, 1.0);
  c.binder.confidence_floor = std::clamp(c.binder.confidence_floor, 0.0, 1.0);
  c.binder.z_jump_min = std::max(0.0, c.binder.z_jump_min);
  c.binder.dz_match_tolerance = std::max(0.0, c.binder.dz_match_tolerance);
  c.binder.dz_gate = std::max(0.0, c.binder.dz_gate);
  c.binder.yaw_err_gate = std::max(1e-3, c.binder.yaw_err_gate);
  c.binder.cost_margin_min = std::max(0.0, c.binder.cost_margin_min);
  c.binder.dz_ema_alpha = std::clamp(c.binder.dz_ema_alpha, 0.01, 1.0);
  c.binder.periodic_window = std::max(3, c.binder.periodic_window);
  c.binder.periodic_weight = std::max(0.0, c.binder.periodic_weight);
  c.binder.periodic_min_spin_rate = std::max(0.0, c.binder.periodic_min_spin_rate);
  c.binder.periodic_update_min_jump = std::max(0.0, c.binder.periodic_update_min_jump);
  c.binder.periodic_signature_threshold =
      std::clamp(c.binder.periodic_signature_threshold, 0.0, 1.0);
  c.binder.reacquire_gap_dt_gate = std::max(0.0, c.binder.reacquire_gap_dt_gate);
  c.binder.reacquire_lost_frames_gate = std::max(0, c.binder.reacquire_lost_frames_gate);
  c.binder.z_cluster_ema_alpha = std::clamp(c.binder.z_cluster_ema_alpha, 0.01, 1.0);
  c.binder.z_cluster_assign_gate = std::max(0.0, c.binder.z_cluster_assign_gate);
  c.binder.min_candidate_prob = std::clamp(c.binder.min_candidate_prob, 0.0, 1.0);
  c.binder.min_candidate_margin = std::clamp(c.binder.min_candidate_margin, 0.0, 1.0);
  c.binder.switch_strong_score = std::clamp(c.binder.switch_strong_score, 0.0, 1.0);
  c.binder.single_obs_history_window = std::max(1, c.binder.single_obs_history_window);
  c.binder.same_panel_yaw_gate = std::max(1e-3, c.binder.same_panel_yaw_gate);
  c.binder.same_panel_z_gate = std::max(1e-3, c.binder.same_panel_z_gate);
  c.binder.same_panel_xy_gate = std::max(1e-3, c.binder.same_panel_xy_gate);
  c.binder.z_audit_rebind_confirm_frames =
      std::max(1, c.binder.z_audit_rebind_confirm_frames);
  c.binder.z_audit_rebind_min_confidence =
      std::clamp(c.binder.z_audit_rebind_min_confidence, 0.0, 1.0);
  c.binder.z_audit_rebind_min_jump = std::max(0.0, c.binder.z_audit_rebind_min_jump);
  c.binder.soft_fusion_w_seq = std::max(0.0, c.binder.soft_fusion_w_seq);
  c.binder.soft_fusion_w_geo = std::max(0.0, c.binder.soft_fusion_w_geo);
  c.binder.soft_fusion_w_dyn = std::max(0.0, c.binder.soft_fusion_w_dyn);
  c.binder.soft_fusion_w_continuity = std::max(0.0, c.binder.soft_fusion_w_continuity);
  c.binder.soft_fusion_w_topology = std::max(0.0, c.binder.soft_fusion_w_topology);

  c.norm4_v2.enable_common_pipeline =
      get_parameter("norm4_v2.enable_common_pipeline").as_bool();
  c.norm4_v2.enable_phase_memory =
      get_parameter("norm4_v2.enable_phase_memory").as_bool();
  c.norm4_v2.enable_kinematic_anti_pingpong =
      get_parameter("norm4_v2.enable_kinematic_anti_pingpong").as_bool();
  c.norm4_v2.enable_2d_tracker =
      get_parameter("norm4_v2.enable_2d_tracker").as_bool();
  c.norm4_v2.enable_proxy_manager =
      get_parameter("norm4_v2.enable_proxy_manager").as_bool();
  c.norm4_v2.phase_memory.enable_phase_memory =
      get_parameter("norm4_v2.phase_memory.enable_phase_memory").as_bool();
  c.norm4_v2.phase_memory.enable_kinematic_anti_pingpong =
      get_parameter("norm4_v2.phase_memory.enable_kinematic_anti_pingpong").as_bool();
  c.norm4_v2.phase_memory.sequence_window_size =
      get_parameter("norm4_v2.phase_memory.sequence_window_size").as_int();
  c.norm4_v2.phase_memory.ping_pong_pattern_threshold =
      get_parameter("norm4_v2.phase_memory.ping_pong_pattern_threshold").as_double();
  c.norm4_v2.phase_memory.enable_opposite_jump_detect =
      get_parameter("norm4_v2.phase_memory.enable_opposite_jump_detect").as_bool();
  c.norm4_v2.phase_memory.anti_pingpong.min_consistent_frames_to_commit =
      get_parameter(
          "norm4_v2.phase_memory.anti_pingpong.min_consistent_frames_to_commit")
          .as_int();
  c.norm4_v2.phase_memory.anti_pingpong.jerk_gate =
      get_parameter("norm4_v2.phase_memory.anti_pingpong.jerk_gate").as_double();
  c.norm4_v2.phase_memory.anti_pingpong.yaw_rate_jump_gate =
      get_parameter("norm4_v2.phase_memory.anti_pingpong.yaw_rate_jump_gate").as_double();
  c.norm4_v2.phase_memory.anti_pingpong.velocity_dir_cos_min =
      get_parameter("norm4_v2.phase_memory.anti_pingpong.velocity_dir_cos_min")
          .as_double();
  c.norm4_v2.phase_memory.anti_pingpong.pending_timeout_frames =
      get_parameter("norm4_v2.phase_memory.anti_pingpong.pending_timeout_frames")
          .as_int();

  c.norm4_v2.phase_memory.enable_phase_memory = c.norm4_v2.enable_phase_memory;
  c.norm4_v2.phase_memory.enable_kinematic_anti_pingpong =
      c.norm4_v2.enable_kinematic_anti_pingpong;
  c.norm4_v2.phase_memory.sequence_window_size =
      std::max(3, c.norm4_v2.phase_memory.sequence_window_size);

  // Norm4 V2 UKF Backend V1
  c.norm4_v2.ukf_v1.enabled =
      get_parameter("norm4_v2.ukf_v1.enabled").as_bool();
  c.norm4_v2.ukf_v1.force_rotation_ca =
      get_parameter("norm4_v2.ukf_v1.force_rotation_ca").as_bool();
  c.norm4_v2.ukf_v1.dual_raw_batch =
      get_parameter("norm4_v2.ukf_v1.dual_raw_batch").as_bool();
  c.norm4_v2.ukf_v1.sigma_pos_xy =
      get_parameter("norm4_v2.ukf_v1.sigma_pos_xy").as_double();
  c.norm4_v2.ukf_v1.sigma_pos_z =
      get_parameter("norm4_v2.ukf_v1.sigma_pos_z").as_double();
  c.norm4_v2.ukf_v1.sigma_yaw =
      get_parameter("norm4_v2.ukf_v1.sigma_yaw").as_double();
  c.norm4_v2.ukf_v1.dual_raw_R_scale =
      get_parameter("norm4_v2.ukf_v1.dual_raw_R_scale").as_double();
  c.norm4_v2.ukf_v1.gate.single_total_nis =
      get_parameter("norm4_v2.ukf_v1.gate.single_total_nis").as_double();
  c.norm4_v2.ukf_v1.gate.single_pos_chi2 =
      get_parameter("norm4_v2.ukf_v1.gate.single_pos_chi2").as_double();
  c.norm4_v2.ukf_v1.gate.single_yaw_chi2 =
      get_parameter("norm4_v2.ukf_v1.gate.single_yaw_chi2").as_double();
  c.norm4_v2.ukf_v1.gate.dual_total_nis =
      get_parameter("norm4_v2.ukf_v1.gate.dual_total_nis").as_double();
  c.norm4_v2.ukf_v1.gate.dual_each_pos_chi2 =
      get_parameter("norm4_v2.ukf_v1.gate.dual_each_pos_chi2").as_double();
  c.norm4_v2.ukf_v1.gate.dual_each_yaw_chi2 =
      get_parameter("norm4_v2.ukf_v1.gate.dual_each_yaw_chi2").as_double();
  c.norm4_v2.ukf_v1.single_update.structural_gain_r =
      get_parameter("norm4_v2.ukf_v1.single_update.structural_gain_r").as_double();
  c.norm4_v2.ukf_v1.single_update.structural_gain_dza =
      get_parameter("norm4_v2.ukf_v1.single_update.structural_gain_dza").as_double();
  c.norm4_v2.ukf_v1.dual_update.structural_gain_r =
      get_parameter("norm4_v2.ukf_v1.dual_update.structural_gain_r").as_double();
  c.norm4_v2.ukf_v1.dual_update.structural_gain_dza =
      get_parameter("norm4_v2.ukf_v1.dual_update.structural_gain_dza").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.max_center_jump =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.max_center_jump").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.max_yaw_jump =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.max_yaw_jump").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.min_r =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.min_r").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.max_r =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.max_r").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.max_r_jump =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.max_r_jump").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.min_dza =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.min_dza").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.max_dza =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.max_dza").as_double();
  c.norm4_v2.ukf_v1.posterior_sanity.max_dza_jump =
      get_parameter("norm4_v2.ukf_v1.posterior_sanity.max_dza_jump").as_double();

  // Norm4 V2 Hypothesis Selector
  c.norm4_v2.hypothesis_selector.topk =
      get_parameter("norm4_v2.hypothesis_selector.topk").as_int();
  c.norm4_v2.hypothesis_selector.commit_top1_only =
      get_parameter("norm4_v2.hypothesis_selector.commit_top1_only").as_bool();
  c.norm4_v2.hypothesis_selector.min_top1_confidence =
      get_parameter("norm4_v2.hypothesis_selector.min_top1_confidence").as_double();
  c.norm4_v2.hypothesis_selector.min_top1_top2_margin =
      get_parameter("norm4_v2.hypothesis_selector.min_top1_top2_margin").as_double();
  c.norm4_v2.hypothesis_selector.ambiguous_margin =
      get_parameter("norm4_v2.hypothesis_selector.ambiguous_margin").as_double();
  c.norm4_v2.hypothesis_selector.include_rejected_in_debug =
      get_parameter("norm4_v2.hypothesis_selector.include_rejected_in_debug").as_bool();
  c.norm4_v2.hypothesis_selector.evidence_prior_enable =
      get_parameter("norm4_v2.hypothesis_selector.evidence_prior_enable").as_bool();
  c.norm4_v2.hypothesis_selector.max_reconstruction_pos_error =
      get_parameter("norm4_v2.hypothesis_selector.max_reconstruction_pos_error").as_double();

  // Norm4 V2 Warmup
  c.norm4_v2.warmup.enable_dual_seed_01 =
      get_parameter("norm4_v2.warmup.enable_dual_seed_01").as_bool();
  c.norm4_v2.warmup.warmup_frames =
      get_parameter("norm4_v2.warmup.warmup_frames").as_int();
  c.norm4_v2.warmup.min_settle_frames =
      get_parameter("norm4_v2.warmup.min_settle_frames").as_int();
  c.norm4_v2.warmup.min_margin_to_commit =
      get_parameter("norm4_v2.warmup.min_margin_to_commit").as_double();
  c.norm4_v2.warmup.min_confidence_to_commit =
      get_parameter("norm4_v2.warmup.min_confidence_to_commit").as_double();

  // Norm4 V2 Mode Routing
  c.norm4_v2.mode_routing.ambiguous_output =
      get_parameter("norm4_v2.mode_routing.ambiguous_output").as_string();
  c.norm4_v2.mode_routing.structured_output =
      get_parameter("norm4_v2.mode_routing.structured_output").as_string();
  c.norm4_v2.mode_routing.ambiguous_structured_backend_mode =
      get_parameter("norm4_v2.mode_routing.ambiguous_structured_backend_mode").as_string();
  c.norm4_v2.mode_routing.structured_single_plate_mode =
      get_parameter("norm4_v2.mode_routing.structured_single_plate_mode").as_string();

  // Norm4 V2 Single-Plate Bridge
  c.norm4_v2.single_plate_bridge.enable =
      get_parameter("norm4_v2.single_plate_bridge.enable").as_bool();
  c.norm4_v2.single_plate_bridge.source_semantic =
      get_parameter("norm4_v2.single_plate_bridge.source_semantic").as_string();
  c.norm4_v2.single_plate_bridge.backend_type =
      get_parameter("norm4_v2.single_plate_bridge.backend_type").as_string();
  c.norm4_v2.single_plate_bridge.require_semantic_stable_frames =
      get_parameter("norm4_v2.single_plate_bridge.require_semantic_stable_frames").as_int();

  // Norm4 V2 Fallback
  c.norm4_v2.fallback.predict_only_on_reject =
      get_parameter("norm4_v2.fallback.predict_only_on_reject").as_bool();
  c.norm4_v2.fallback.enable_ambiguous_single_fallback =
      get_parameter("norm4_v2.fallback.enable_ambiguous_single_fallback").as_bool();
  c.norm4_v2.phase_memory.ping_pong_pattern_threshold =
      std::clamp(c.norm4_v2.phase_memory.ping_pong_pattern_threshold, 0.0, 1.0);
  c.norm4_v2.phase_memory.anti_pingpong.min_consistent_frames_to_commit =
      std::max(
          1,
          c.norm4_v2.phase_memory.anti_pingpong.min_consistent_frames_to_commit);
  c.norm4_v2.phase_memory.anti_pingpong.jerk_gate =
      std::max(0.0, c.norm4_v2.phase_memory.anti_pingpong.jerk_gate);
  c.norm4_v2.phase_memory.anti_pingpong.yaw_rate_jump_gate =
      std::max(0.0, c.norm4_v2.phase_memory.anti_pingpong.yaw_rate_jump_gate);
  c.norm4_v2.phase_memory.anti_pingpong.velocity_dir_cos_min =
      std::clamp(
          c.norm4_v2.phase_memory.anti_pingpong.velocity_dir_cos_min, -1.0, 1.0);
  c.norm4_v2.phase_memory.anti_pingpong.pending_timeout_frames =
      std::max(1, c.norm4_v2.phase_memory.anti_pingpong.pending_timeout_frames);

  c.norm4_v3.enable_common_pipeline =
      get_parameter("norm4_v3.enable_common_pipeline").as_bool();
  c.norm4_v3.enable_phase_memory =
      get_parameter("norm4_v3.enable_phase_memory").as_bool();
  c.norm4_v3.enable_kinematic_anti_pingpong =
      get_parameter("norm4_v3.enable_kinematic_anti_pingpong").as_bool();
  c.norm4_v3.enable_2d_tracker =
      get_parameter("norm4_v3.enable_2d_tracker").as_bool();
  c.norm4_v3.enable_proxy_manager =
      get_parameter("norm4_v3.enable_proxy_manager").as_bool();
  c.norm4_v3.phase_memory.enable_phase_memory =
      get_parameter("norm4_v3.phase_memory.enable_phase_memory").as_bool();
  c.norm4_v3.phase_memory.enable_kinematic_anti_pingpong =
      get_parameter("norm4_v3.phase_memory.enable_kinematic_anti_pingpong").as_bool();
  c.norm4_v3.phase_memory.sequence_window_size =
      get_parameter("norm4_v3.phase_memory.sequence_window_size").as_int();
  c.norm4_v3.phase_memory.ping_pong_pattern_threshold =
      get_parameter("norm4_v3.phase_memory.ping_pong_pattern_threshold").as_double();
  c.norm4_v3.phase_memory.enable_opposite_jump_detect =
      get_parameter("norm4_v3.phase_memory.enable_opposite_jump_detect").as_bool();
  c.norm4_v3.phase_memory.anti_pingpong.min_consistent_frames_to_commit =
      get_parameter(
          "norm4_v3.phase_memory.anti_pingpong.min_consistent_frames_to_commit")
          .as_int();
  c.norm4_v3.phase_memory.anti_pingpong.jerk_gate =
      get_parameter("norm4_v3.phase_memory.anti_pingpong.jerk_gate").as_double();
  c.norm4_v3.phase_memory.anti_pingpong.yaw_rate_jump_gate =
      get_parameter("norm4_v3.phase_memory.anti_pingpong.yaw_rate_jump_gate").as_double();
  c.norm4_v3.phase_memory.anti_pingpong.velocity_dir_cos_min =
      get_parameter("norm4_v3.phase_memory.anti_pingpong.velocity_dir_cos_min")
          .as_double();
  c.norm4_v3.phase_memory.anti_pingpong.pending_timeout_frames =
      get_parameter("norm4_v3.phase_memory.anti_pingpong.pending_timeout_frames")
          .as_int();

  c.norm4_v3.phase_memory.enable_phase_memory = c.norm4_v3.enable_phase_memory;
  c.norm4_v3.phase_memory.enable_kinematic_anti_pingpong =
      c.norm4_v3.enable_kinematic_anti_pingpong;
  c.norm4_v3.phase_memory.sequence_window_size =
      std::max(3, c.norm4_v3.phase_memory.sequence_window_size);

  c.norm4_v3.ukf_v1.enabled =
      get_parameter("norm4_v3.ukf_v1.enabled").as_bool();
  c.norm4_v3.ukf_v1.force_rotation_ca =
      get_parameter("norm4_v3.ukf_v1.force_rotation_ca").as_bool();
  c.norm4_v3.ukf_v1.dual_raw_batch =
      get_parameter("norm4_v3.ukf_v1.dual_raw_batch").as_bool();
  c.norm4_v3.ukf_v1.sigma_pos_xy =
      get_parameter("norm4_v3.ukf_v1.sigma_pos_xy").as_double();
  c.norm4_v3.ukf_v1.sigma_pos_z =
      get_parameter("norm4_v3.ukf_v1.sigma_pos_z").as_double();
  c.norm4_v3.ukf_v1.sigma_yaw =
      get_parameter("norm4_v3.ukf_v1.sigma_yaw").as_double();
  c.norm4_v3.ukf_v1.dual_raw_R_scale =
      get_parameter("norm4_v3.ukf_v1.dual_raw_R_scale").as_double();
  c.norm4_v3.ukf_v1.gate.single_total_nis =
      get_parameter("norm4_v3.ukf_v1.gate.single_total_nis").as_double();
  c.norm4_v3.ukf_v1.gate.single_pos_chi2 =
      get_parameter("norm4_v3.ukf_v1.gate.single_pos_chi2").as_double();
  c.norm4_v3.ukf_v1.gate.single_yaw_chi2 =
      get_parameter("norm4_v3.ukf_v1.gate.single_yaw_chi2").as_double();
  c.norm4_v3.ukf_v1.gate.dual_total_nis =
      get_parameter("norm4_v3.ukf_v1.gate.dual_total_nis").as_double();
  c.norm4_v3.ukf_v1.gate.dual_each_pos_chi2 =
      get_parameter("norm4_v3.ukf_v1.gate.dual_each_pos_chi2").as_double();
  c.norm4_v3.ukf_v1.gate.dual_each_yaw_chi2 =
      get_parameter("norm4_v3.ukf_v1.gate.dual_each_yaw_chi2").as_double();
  c.norm4_v3.ukf_v1.single_update.structural_gain_r =
      get_parameter("norm4_v3.ukf_v1.single_update.structural_gain_r").as_double();
  c.norm4_v3.ukf_v1.single_update.structural_gain_dza =
      get_parameter("norm4_v3.ukf_v1.single_update.structural_gain_dza").as_double();
  c.norm4_v3.ukf_v1.dual_update.structural_gain_r =
      get_parameter("norm4_v3.ukf_v1.dual_update.structural_gain_r").as_double();
  c.norm4_v3.ukf_v1.dual_update.structural_gain_dza =
      get_parameter("norm4_v3.ukf_v1.dual_update.structural_gain_dza").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.max_center_jump =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.max_center_jump").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.max_yaw_jump =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.max_yaw_jump").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.min_r =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.min_r").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.max_r =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.max_r").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.max_r_jump =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.max_r_jump").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.min_dza =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.min_dza").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.max_dza =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.max_dza").as_double();
  c.norm4_v3.ukf_v1.posterior_sanity.max_dza_jump =
      get_parameter("norm4_v3.ukf_v1.posterior_sanity.max_dza_jump").as_double();

  auto load_norm4_v3_ukf_cfg = [this](const std::string &prefix,
                                      Norm4V3UkfConfig *out) {
    out->enabled = get_parameter(prefix + ".enabled").as_bool();
    out->force_rotation_ca =
        get_parameter(prefix + ".force_rotation_ca").as_bool();
    out->dual_raw_batch = get_parameter(prefix + ".dual_raw_batch").as_bool();
    out->sigma_pos_xy = get_parameter(prefix + ".sigma_pos_xy").as_double();
    out->sigma_pos_z = get_parameter(prefix + ".sigma_pos_z").as_double();
    out->sigma_yaw = get_parameter(prefix + ".sigma_yaw").as_double();
    out->dual_raw_R_scale =
        get_parameter(prefix + ".dual_raw_R_scale").as_double();
    out->gate.single_total_nis =
        get_parameter(prefix + ".gate.single_total_nis").as_double();
    out->gate.single_pos_chi2 =
        get_parameter(prefix + ".gate.single_pos_chi2").as_double();
    out->gate.single_yaw_chi2 =
        get_parameter(prefix + ".gate.single_yaw_chi2").as_double();
    out->gate.dual_total_nis =
        get_parameter(prefix + ".gate.dual_total_nis").as_double();
    out->gate.dual_each_pos_chi2 =
        get_parameter(prefix + ".gate.dual_each_pos_chi2").as_double();
    out->gate.dual_each_yaw_chi2 =
        get_parameter(prefix + ".gate.dual_each_yaw_chi2").as_double();
    out->single_update.structural_gain_r =
        get_parameter(prefix + ".single_update.structural_gain_r").as_double();
    out->single_update.structural_gain_dza =
        get_parameter(prefix + ".single_update.structural_gain_dza")
            .as_double();
    out->dual_update.structural_gain_r =
        get_parameter(prefix + ".dual_update.structural_gain_r").as_double();
    out->dual_update.structural_gain_dza =
        get_parameter(prefix + ".dual_update.structural_gain_dza").as_double();
    out->posterior_sanity.max_center_jump =
        get_parameter(prefix + ".posterior_sanity.max_center_jump").as_double();
    out->posterior_sanity.max_yaw_jump =
        get_parameter(prefix + ".posterior_sanity.max_yaw_jump").as_double();
    out->posterior_sanity.min_r =
        get_parameter(prefix + ".posterior_sanity.min_r").as_double();
    out->posterior_sanity.max_r =
        get_parameter(prefix + ".posterior_sanity.max_r").as_double();
    out->posterior_sanity.max_r_jump =
        get_parameter(prefix + ".posterior_sanity.max_r_jump").as_double();
    out->posterior_sanity.min_dza =
        get_parameter(prefix + ".posterior_sanity.min_dza").as_double();
    out->posterior_sanity.max_dza =
        get_parameter(prefix + ".posterior_sanity.max_dza").as_double();
    out->posterior_sanity.max_dza_jump =
        get_parameter(prefix + ".posterior_sanity.max_dza_jump").as_double();
  };
  load_norm4_v3_ukf_cfg("norm4_v3.ukf_v2", &c.norm4_v3.ukf_v2);
  load_norm4_v3_ukf_cfg("norm4_v3.inekf", &c.norm4_v3.inekf);

  c.norm4_v3.slow_structure.enable =
      get_parameter("norm4_v3.slow_structure.enable").as_bool();
  c.norm4_v3.slow_structure.q_theta_r1 =
      get_parameter("norm4_v3.slow_structure.q_theta_r1").as_double();
  c.norm4_v3.slow_structure.q_theta_r2 =
      get_parameter("norm4_v3.slow_structure.q_theta_r2").as_double();
  c.norm4_v3.slow_structure.q_theta_dza =
      get_parameter("norm4_v3.slow_structure.q_theta_dza").as_double();
  c.norm4_v3.slow_structure.prior_r1 =
      get_parameter("norm4_v3.slow_structure.prior_r1").as_double();
  c.norm4_v3.slow_structure.prior_r2 =
      get_parameter("norm4_v3.slow_structure.prior_r2").as_double();
  c.norm4_v3.slow_structure.prior_dza =
      get_parameter("norm4_v3.slow_structure.prior_dza").as_double();
  c.norm4_v3.slow_structure.prior_sigma_r =
      get_parameter("norm4_v3.slow_structure.prior_sigma_r").as_double();
  c.norm4_v3.slow_structure.prior_sigma_dza =
      get_parameter("norm4_v3.slow_structure.prior_sigma_dza").as_double();
  c.norm4_v3.slow_structure.alpha_r1_single =
      get_parameter("norm4_v3.slow_structure.alpha_r1_single").as_double();
  c.norm4_v3.slow_structure.alpha_r2_single =
      get_parameter("norm4_v3.slow_structure.alpha_r2_single").as_double();
  c.norm4_v3.slow_structure.alpha_dza_single =
      get_parameter("norm4_v3.slow_structure.alpha_dza_single").as_double();
  c.norm4_v3.slow_structure.alpha_r1_dual =
      get_parameter("norm4_v3.slow_structure.alpha_r1_dual").as_double();
  c.norm4_v3.slow_structure.alpha_r2_dual =
      get_parameter("norm4_v3.slow_structure.alpha_r2_dual").as_double();
  c.norm4_v3.slow_structure.alpha_dza_dual =
      get_parameter("norm4_v3.slow_structure.alpha_dza_dual").as_double();
  c.norm4_v3.slow_structure.prior_pull_gain =
      get_parameter("norm4_v3.slow_structure.prior_pull_gain").as_double();
  c.norm4_v3.slow_structure.min_r =
      get_parameter("norm4_v3.slow_structure.min_r").as_double();
  c.norm4_v3.slow_structure.max_r =
      get_parameter("norm4_v3.slow_structure.max_r").as_double();
  c.norm4_v3.slow_structure.min_dza =
      get_parameter("norm4_v3.slow_structure.min_dza").as_double();
  c.norm4_v3.slow_structure.max_dza =
      get_parameter("norm4_v3.slow_structure.max_dza").as_double();

  c.norm4_v3.backend_config.backend_type =
      get_parameter("norm4_v3.backend_config.backend_type").as_string();
  c.norm4_v3.backend_config.motion_profile =
      get_parameter("norm4_v3.backend_config.motion_profile").as_string();
  c.norm4_v3.backend_config.noise_profile =
      get_parameter("norm4_v3.backend_config.noise_profile").as_string();
  c.norm4_v3.backend_config.structure_profile =
      get_parameter("norm4_v3.backend_config.structure_profile").as_string();
  c.norm4_v3.inekf_runtime.motion_profile =
      get_parameter("norm4_v3.inekf_runtime.motion_profile").as_string();
  c.norm4_v3.inekf_runtime.noise_profile =
      get_parameter("norm4_v3.inekf_runtime.noise_profile").as_string();
  c.norm4_v3.inekf_runtime.structure_profile =
      get_parameter("norm4_v3.inekf_runtime.structure_profile").as_string();
  c.norm4_v3.inekf_runtime.translation_model =
      get_parameter("norm4_v3.inekf_runtime.translation_model").as_string();
  c.norm4_v3.inekf_runtime.cv_process_noise_vel =
      get_parameter("norm4_v3.inekf_runtime.cv_process_noise_vel").as_double();
  c.norm4_v3.inekf_runtime.ca_process_noise_acc =
      get_parameter("norm4_v3.inekf_runtime.ca_process_noise_acc").as_double();
  c.norm4_v3.inekf_runtime.singer_alpha =
      get_parameter("norm4_v3.inekf_runtime.singer_alpha").as_double();
  c.norm4_v3.inekf_runtime.singer_sigma =
      get_parameter("norm4_v3.inekf_runtime.singer_sigma").as_double();
  c.norm4_v3.inekf_runtime.process_noise_r =
      get_parameter("norm4_v3.inekf_runtime.process_noise_r").as_double();
  c.norm4_v3.inekf_runtime.process_noise_dz =
      get_parameter("norm4_v3.inekf_runtime.process_noise_dz").as_double();
  c.norm4_v3.inekf_runtime.spin_process_noise_delta_rate =
      get_parameter("norm4_v3.inekf_runtime.spin_process_noise_delta_rate")
          .as_double();
  c.norm4_v3.inekf_runtime.spin_process_noise_delta_acc =
      get_parameter("norm4_v3.inekf_runtime.spin_process_noise_delta_acc")
          .as_double();

  c.norm4_v3.hypothesis_selector.topk =
      get_parameter("norm4_v3.hypothesis_selector.topk").as_int();
  c.norm4_v3.hypothesis_selector.commit_top1_only =
      get_parameter("norm4_v3.hypothesis_selector.commit_top1_only").as_bool();
  c.norm4_v3.hypothesis_selector.min_top1_confidence =
      get_parameter("norm4_v3.hypothesis_selector.min_top1_confidence").as_double();
  c.norm4_v3.hypothesis_selector.min_top1_top2_margin =
      get_parameter("norm4_v3.hypothesis_selector.min_top1_top2_margin").as_double();
  c.norm4_v3.hypothesis_selector.ambiguous_margin =
      get_parameter("norm4_v3.hypothesis_selector.ambiguous_margin").as_double();
  c.norm4_v3.hypothesis_selector.include_rejected_in_debug =
      get_parameter("norm4_v3.hypothesis_selector.include_rejected_in_debug").as_bool();
  c.norm4_v3.hypothesis_selector.evidence_prior_enable =
      get_parameter("norm4_v3.hypothesis_selector.evidence_prior_enable").as_bool();
  c.norm4_v3.hypothesis_selector.max_reconstruction_pos_error =
      get_parameter("norm4_v3.hypothesis_selector.max_reconstruction_pos_error").as_double();

  c.norm4_v3.warmup.enable_dual_seed_01 =
      get_parameter("norm4_v3.warmup.enable_dual_seed_01").as_bool();
  c.norm4_v3.warmup.warmup_frames =
      get_parameter("norm4_v3.warmup.warmup_frames").as_int();
  c.norm4_v3.warmup.min_settle_frames =
      get_parameter("norm4_v3.warmup.min_settle_frames").as_int();
  c.norm4_v3.warmup.min_margin_to_commit =
      get_parameter("norm4_v3.warmup.min_margin_to_commit").as_double();
  c.norm4_v3.warmup.min_confidence_to_commit =
      get_parameter("norm4_v3.warmup.min_confidence_to_commit").as_double();

  c.norm4_v3.mode_routing.ambiguous_output =
      get_parameter("norm4_v3.mode_routing.ambiguous_output").as_string();
  c.norm4_v3.mode_routing.structured_output =
      get_parameter("norm4_v3.mode_routing.structured_output").as_string();
  c.norm4_v3.mode_routing.ambiguous_structured_backend_mode =
      get_parameter("norm4_v3.mode_routing.ambiguous_structured_backend_mode").as_string();
  c.norm4_v3.mode_routing.structured_single_plate_mode =
      get_parameter("norm4_v3.mode_routing.structured_single_plate_mode").as_string();

  c.norm4_v3.single_plate_bridge.enable =
      get_parameter("norm4_v3.single_plate_bridge.enable").as_bool();
  c.norm4_v3.single_plate_bridge.source_semantic =
      get_parameter("norm4_v3.single_plate_bridge.source_semantic").as_string();
  c.norm4_v3.single_plate_bridge.backend_type =
      get_parameter("norm4_v3.single_plate_bridge.backend_type").as_string();
  c.norm4_v3.single_plate_bridge.require_semantic_stable_frames =
      get_parameter("norm4_v3.single_plate_bridge.require_semantic_stable_frames").as_int();

  c.norm4_v3.fallback.predict_only_on_reject =
      get_parameter("norm4_v3.fallback.predict_only_on_reject").as_bool();
  c.norm4_v3.fallback.enable_ambiguous_single_fallback =
      get_parameter("norm4_v3.fallback.enable_ambiguous_single_fallback").as_bool();
  c.norm4_v3.debug_log.enable =
      get_parameter("norm4_v3.debug_log.enable").as_bool();
  c.norm4_v3.debug_log.throttle_ms =
      std::max<int>(50, static_cast<int>(
                            get_parameter("norm4_v3.debug_log.throttle_ms").as_int()));
  c.norm4_v3.debug_log.verbose =
      get_parameter("norm4_v3.debug_log.verbose").as_bool();
  c.norm4_v3.phase_memory.ping_pong_pattern_threshold =
      std::clamp(c.norm4_v3.phase_memory.ping_pong_pattern_threshold, 0.0, 1.0);
  c.norm4_v3.phase_memory.anti_pingpong.min_consistent_frames_to_commit =
      std::max(
          1,
          c.norm4_v3.phase_memory.anti_pingpong.min_consistent_frames_to_commit);
  c.norm4_v3.phase_memory.anti_pingpong.jerk_gate =
      std::max(0.0, c.norm4_v3.phase_memory.anti_pingpong.jerk_gate);
  c.norm4_v3.phase_memory.anti_pingpong.yaw_rate_jump_gate =
      std::max(0.0, c.norm4_v3.phase_memory.anti_pingpong.yaw_rate_jump_gate);
  c.norm4_v3.phase_memory.anti_pingpong.velocity_dir_cos_min =
      std::clamp(
          c.norm4_v3.phase_memory.anti_pingpong.velocity_dir_cos_min, -1.0, 1.0);
  c.norm4_v3.phase_memory.anti_pingpong.pending_timeout_frames =
      std::max(1, c.norm4_v3.phase_memory.anti_pingpong.pending_timeout_frames);

  c.panel_mismatch.enable =
      get_parameter("panel_mismatch.enable").as_bool();
  c.panel_mismatch.window_size =
      get_parameter("panel_mismatch.window_size").as_int();
  c.panel_mismatch.threshold_t1 =
      get_parameter("panel_mismatch.threshold_t1").as_double();
  c.panel_mismatch.confirm_count =
      get_parameter("panel_mismatch.confirm_count").as_int();
  c.panel_mismatch.reinit_count =
      get_parameter("panel_mismatch.reinit_count").as_int();
    c.panel_mismatch.apply_correction =
      get_parameter("panel_mismatch.apply_correction").as_bool();

    c.outpost.translation_model = translation_model_from_string(
      get_parameter("outpost.translation_model").as_string());
    c.outpost.rotation_model = rotation_model_from_string(
      get_parameter("outpost.rotation_model").as_string());
    c.outpost.use_tracker_v2 =
      get_parameter("outpost.use_tracker_v2").as_bool();
    c.outpost.use_tracker_v3 =
      get_parameter("outpost.use_tracker_v3").as_bool();
    c.outpost.tracking_thres = get_parameter("outpost.tracking_thres").as_int();
    c.outpost.lost_thres = get_parameter("outpost.lost_thres").as_int();
    c.outpost.temp_lost_thres =
      get_parameter("outpost.temp_lost_thres").as_int();
    c.outpost.max_match_distance =
      get_parameter("outpost.max_match_distance").as_double();
    c.outpost.max_match_yaw_diff =
      get_parameter("outpost.max_match_yaw_diff").as_double();
    c.outpost.singer_alpha = get_parameter("outpost.singer_alpha").as_double();
    c.outpost.singer_sigma = get_parameter("outpost.singer_sigma").as_double();
    c.outpost.spin_process_noise_theta_rate =
      get_parameter("outpost.spin_process_noise_theta_rate").as_double();
    c.outpost.spin_process_noise_theta_acc =
      get_parameter("outpost.spin_process_noise_theta_acc").as_double();
    c.outpost.radius = get_parameter("outpost.radius").as_double();
    c.outpost.z_offset_0 = get_parameter("outpost.z_offset_0").as_double();
    c.outpost.z_offset_1 = get_parameter("outpost.z_offset_1").as_double();
    c.outpost.z_offset_2 = get_parameter("outpost.z_offset_2").as_double();
    c.outpost.panel_angle_step =
      get_parameter("outpost.panel_angle_step").as_double();
    c.outpost.softmax_temperature =
      get_parameter("outpost.softmax_temperature").as_double();
    c.outpost.weight_yaw = get_parameter("outpost.weight_yaw").as_double();
    c.outpost.weight_z_state =
      get_parameter("outpost.weight_z_state").as_double();
    c.outpost.weight_z_history =
      get_parameter("outpost.weight_z_history").as_double();
    c.outpost.weight_xy_residual =
      get_parameter("outpost.weight_xy_residual").as_double();
    c.outpost.weight_switch_penalty =
      get_parameter("outpost.weight_switch_penalty").as_double();
    c.outpost.entropy_enter =
      get_parameter("outpost.entropy_enter").as_double();
    c.outpost.entropy_exit =
      get_parameter("outpost.entropy_exit").as_double();
    c.outpost.max_prob_enter =
      get_parameter("outpost.max_prob_enter").as_double();
    c.outpost.max_prob_exit =
      get_parameter("outpost.max_prob_exit").as_double();
    c.outpost.stable_frames =
      get_parameter("outpost.stable_frames").as_int();
    c.outpost.z_history_window =
      get_parameter("outpost.z_history_window").as_int();
    c.outpost.single_mode_confidence_scale =
      get_parameter("outpost.single_mode_confidence_scale").as_double();
    c.outpost.binding_use_new_binder_pipeline =
      get_parameter("outpost.binding_use_new_binder_pipeline").as_bool();
    c.outpost.binding_enable_multi_obs =
      get_parameter("outpost.binding_enable_multi_obs").as_bool();
    c.outpost.binding_transition_confirm_frames =
      get_parameter("outpost.binding_transition_confirm_frames").as_int();
    c.outpost.binding_same_panel_yaw_gate =
      get_parameter("outpost.binding_same_panel_yaw_gate").as_double();
    c.outpost.binding_same_panel_z_gate =
      get_parameter("outpost.binding_same_panel_z_gate").as_double();
    c.outpost.binding_same_panel_xy_gate =
      get_parameter("outpost.binding_same_panel_xy_gate").as_double();
    c.outpost.binding_min_candidate_prob =
      get_parameter("outpost.binding_min_candidate_prob").as_double();
    c.outpost.binding_min_candidate_margin =
      get_parameter("outpost.binding_min_candidate_margin").as_double();
    c.outpost.binding_switch_strong_score =
      get_parameter("outpost.binding_switch_strong_score").as_double();
    c.outpost.binding_period_window =
      get_parameter("outpost.binding_period_window").as_int();
    c.outpost.binding_period_weight =
      get_parameter("outpost.binding_period_weight").as_double();
    c.outpost.binding_topology_prior_weight =
      get_parameter("outpost.binding_topology_prior_weight").as_double();
    c.outpost.binding_period_min_spin_rate =
      get_parameter("outpost.binding_period_min_spin_rate").as_double();
    c.outpost.spin_direction_confirm_frames =
      get_parameter("outpost.spin_direction_confirm_frames").as_int();
    c.outpost.binding_period_update_min_confidence =
      get_parameter("outpost.binding_period_update_min_confidence").as_double();
    c.outpost.binding_period_update_min_jump =
      get_parameter("outpost.binding_period_update_min_jump").as_double();
    c.outpost.binding_dz_ema_alpha =
      get_parameter("outpost.binding_dz_ema_alpha").as_double();
    c.outpost.binding_confidence_floor =
      get_parameter("outpost.binding_confidence_floor").as_double();
    c.outpost.z_audit_rebind_enable =
      get_parameter("outpost.z_audit_rebind_enable").as_bool();
    c.outpost.z_audit_rebind_confirm_frames =
      get_parameter("outpost.z_audit_rebind_confirm_frames").as_int();
    c.outpost.z_audit_rebind_min_confidence =
      get_parameter("outpost.z_audit_rebind_min_confidence").as_double();
    c.outpost.z_audit_rebind_min_jump =
      get_parameter("outpost.z_audit_rebind_min_jump").as_double();
    c.outpost.binding_conflict_position_scale =
      get_parameter("outpost.binding_conflict_position_scale").as_double();
    c.outpost.alpha_pos = get_parameter("outpost.alpha_pos").as_double();
    c.outpost.beta_vel = get_parameter("outpost.beta_vel").as_double();
    c.outpost.alpha_yaw = get_parameter("outpost.alpha_yaw").as_double();
    c.outpost.beta_yaw_rate =
      get_parameter("outpost.beta_yaw_rate").as_double();
    c.outpost.assume_static_center =
      get_parameter("outpost.assume_static_center").as_bool();
    c.outpost.linear_velocity_damping =
      get_parameter("outpost.linear_velocity_damping").as_double();
    c.outpost.yaw_rate_damping =
      get_parameter("outpost.yaw_rate_damping").as_double();
    c.outpost.max_center_speed =
      get_parameter("outpost.max_center_speed").as_double();
    c.outpost.max_yaw_rate =
      get_parameter("outpost.max_yaw_rate").as_double();
    c.outpost.max_yaw_rate_step =
      get_parameter("outpost.max_yaw_rate_step").as_double();
    c.outpost.mode_enter_confirm_frames =
      get_parameter("outpost.mode_enter_confirm_frames").as_int();
    c.outpost.mode_exit_confirm_frames =
      get_parameter("outpost.mode_exit_confirm_frames").as_int();
    c.outpost.mode_min_dwell_frames =
      get_parameter("outpost.mode_min_dwell_frames").as_int();
    c.outpost.mode_enter_threshold =
      get_parameter("outpost.mode_enter_threshold").as_double();
    c.outpost.mode_exit_threshold =
      get_parameter("outpost.mode_exit_threshold").as_double();
    c.outpost.mode_weight_jump =
      get_parameter("outpost.mode_weight_jump").as_double();
    c.outpost.mode_weight_dual =
      get_parameter("outpost.mode_weight_dual").as_double();
    c.outpost.mode_weight_margin =
      get_parameter("outpost.mode_weight_margin").as_double();
    c.outpost.mode_weight_health =
      get_parameter("outpost.mode_weight_health").as_double();
    c.outpost.mode_weight_entropy =
      get_parameter("outpost.mode_weight_entropy").as_double();
    c.outpost.ambiguous_publish_single_armor_semantics =
      get_parameter("outpost.ambiguous_publish_single_armor_semantics").as_bool();
    c.outpost.ambiguous_single_armor_zero_offset =
      get_parameter("outpost.ambiguous_single_armor_zero_offset").as_bool();
    c.outpost.ambiguous_backend_use_imm_adapter =
      get_parameter("outpost.ambiguous_backend_use_imm_adapter").as_bool();
    c.outpost.v2_warmup_enable =
      get_parameter("outpost.v2_warmup_enable").as_bool();
    c.outpost.v2_warmup_min_groups =
      get_parameter("outpost.v2_warmup_min_groups").as_int();
    c.outpost.v2_warmup_min_samples_per_group =
      get_parameter("outpost.v2_warmup_min_samples_per_group").as_int();
    c.outpost.v2_warmup_max_frames =
      get_parameter("outpost.v2_warmup_max_frames").as_int();
    c.outpost.v2_warmup_z_jump_gate =
      get_parameter("outpost.v2_warmup_z_jump_gate").as_double();
    c.outpost.v2_warmup_yaw_jump_gate =
      get_parameter("outpost.v2_warmup_yaw_jump_gate").as_double();
    c.outpost.v2_warmup_xyz_jump_gate =
      get_parameter("outpost.v2_warmup_xyz_jump_gate").as_double();
    c.outpost.v2_warmup_ratio_min =
      get_parameter("outpost.v2_warmup_ratio_min").as_double();
    c.outpost.v2_warmup_ratio_max =
      get_parameter("outpost.v2_warmup_ratio_max").as_double();
    c.outpost.v2_warmup_min_large_diff =
      get_parameter("outpost.v2_warmup_min_large_diff").as_double();
    c.outpost.v3_topk = get_parameter("outpost.v3.topk").as_int();
    c.outpost.v3_min_top1_confidence =
      get_parameter("outpost.v3.min_top1_confidence").as_double();
    c.outpost.v3_min_top1_top2_margin =
      get_parameter("outpost.v3.min_top1_top2_margin").as_double();
    c.outpost.v3_max_reconstruction_pos_error =
      get_parameter("outpost.v3.max_reconstruction_pos_error").as_double();
    c.outpost.v3_gate_single_total_nis =
      get_parameter("outpost.v3.gate_single_total_nis").as_double();
    c.outpost.v3_gate_single_pos_chi2 =
      get_parameter("outpost.v3.gate_single_pos_chi2").as_double();
    c.outpost.v3_posterior_max_center_jump =
      get_parameter("outpost.v3.posterior_max_center_jump").as_double();
    c.outpost.v3_posterior_max_yaw_jump =
      get_parameter("outpost.v3.posterior_max_yaw_jump").as_double();
    c.outpost.v3_posterior_max_yaw_rate =
      get_parameter("outpost.v3.posterior_max_yaw_rate").as_double();
    c.outpost.v3_posterior_max_yaw_acc =
      get_parameter("outpost.v3.posterior_max_yaw_acc").as_double();
    c.outpost.v3_mode_p_enter_structured =
      get_parameter("outpost.v3.mode_p_enter_structured").as_double();
    c.outpost.v3_mode_m_enter_structured =
      get_parameter("outpost.v3.mode_m_enter_structured").as_double();
    c.outpost.v3_mode_stable_frames =
      get_parameter("outpost.v3.mode_stable_frames").as_int();
    c.outpost.v3_mode_p_exit_structured =
      get_parameter("outpost.v3.mode_p_exit_structured").as_double();
    c.outpost.v3_mode_m_exit_structured =
      get_parameter("outpost.v3.mode_m_exit_structured").as_double();
    c.outpost.v3_mode_degraded_frames =
      get_parameter("outpost.v3.mode_degraded_frames").as_int();
    c.outpost.v3_prior_panel_switch_penalty =
      get_parameter("outpost.v3.prior_panel_switch_penalty").as_double();
    c.outpost.v3_initial_p_pos =
      get_parameter("outpost.v3.initial_p_pos").as_double();
    c.outpost.v3_initial_p_vel =
      get_parameter("outpost.v3.initial_p_vel").as_double();
    c.outpost.v3_initial_p_acc =
      get_parameter("outpost.v3.initial_p_acc").as_double();
    c.outpost.v3_initial_p_yaw =
      get_parameter("outpost.v3.initial_p_yaw").as_double();
    c.outpost.v3_initial_p_yaw_rate =
      get_parameter("outpost.v3.initial_p_yaw_rate").as_double();
    c.outpost.v3_initial_p_yaw_acc =
      get_parameter("outpost.v3.initial_p_yaw_acc").as_double();
    c.outpost.v3_process_noise_acc =
      get_parameter("outpost.v3.process_noise_acc").as_double();
    c.outpost.v3_process_noise_yaw_acc =
      get_parameter("outpost.v3.process_noise_yaw_acc").as_double();
    c.outpost.v3_observation_sigma_pos_xy =
      get_parameter("outpost.v3.observation_sigma_pos_xy").as_double();
    c.outpost.v3_observation_sigma_pos_z =
      get_parameter("outpost.v3.observation_sigma_pos_z").as_double();
    c.outpost.v3_warmup_enable =
      get_parameter("outpost.v3.warmup_enable").as_bool();
    c.outpost.v3_warmup_frames =
      get_parameter("outpost.v3.warmup_frames").as_int();
    c.outpost.v3_warmup_min_settle_frames =
      get_parameter("outpost.v3.warmup_min_settle_frames").as_int();
    c.outpost.v3_warmup_min_margin_to_commit =
      get_parameter("outpost.v3.warmup_min_margin_to_commit").as_double();
    c.outpost.v3_warmup_min_confidence_to_commit =
      get_parameter("outpost.v3.warmup_min_confidence_to_commit").as_double();
    c.outpost.v3_phase_audit_enable =
      get_parameter("outpost.v3.phase_audit_enable").as_bool();
    c.outpost.v3_phase_audit_min_jump =
      get_parameter("outpost.v3.phase_audit_min_jump").as_double();
    c.outpost.v3_phase_audit_dz_gate =
      get_parameter("outpost.v3.phase_audit_dz_gate").as_double();
    c.outpost.v3_phase_audit_confirm_frames =
      get_parameter("outpost.v3.phase_audit_confirm_frames").as_int();

    c.outpost.tracking_thres = std::max(1, c.outpost.tracking_thres);
    c.outpost.lost_thres = std::max(1, c.outpost.lost_thres);
    c.outpost.temp_lost_thres = std::max(1, c.outpost.temp_lost_thres);
    c.outpost.max_match_distance = std::max(0.0, c.outpost.max_match_distance);
    c.outpost.max_match_yaw_diff = std::max(0.0, c.outpost.max_match_yaw_diff);

    c.outpost.binding_transition_confirm_frames =
      std::max(1, c.outpost.binding_transition_confirm_frames);
    c.outpost.binding_same_panel_yaw_gate =
      std::max(1e-3, c.outpost.binding_same_panel_yaw_gate);
    c.outpost.binding_same_panel_z_gate =
      std::max(1e-3, c.outpost.binding_same_panel_z_gate);
    c.outpost.binding_same_panel_xy_gate =
      std::max(1e-3, c.outpost.binding_same_panel_xy_gate);
    c.outpost.binding_min_candidate_prob =
      std::clamp(c.outpost.binding_min_candidate_prob, 0.0, 1.0);
    c.outpost.binding_min_candidate_margin =
      std::clamp(c.outpost.binding_min_candidate_margin, 0.0, 1.0);
    c.outpost.binding_switch_strong_score =
      std::clamp(c.outpost.binding_switch_strong_score, 0.0, 1.0);
    c.outpost.binding_period_window = std::max(3, c.outpost.binding_period_window);
    c.outpost.binding_period_weight =
      std::max(0.0, c.outpost.binding_period_weight);
    c.outpost.binding_topology_prior_weight =
      std::max(0.0, c.outpost.binding_topology_prior_weight);
    c.outpost.binding_period_min_spin_rate =
      std::max(0.0, c.outpost.binding_period_min_spin_rate);
    c.outpost.spin_direction_confirm_frames =
      std::max(1, c.outpost.spin_direction_confirm_frames);
    c.outpost.binding_period_update_min_confidence =
      std::clamp(c.outpost.binding_period_update_min_confidence, 0.0, 1.0);
    c.outpost.binding_period_update_min_jump =
      std::max(0.0, c.outpost.binding_period_update_min_jump);
    c.outpost.max_yaw_rate_step =
      std::max(0.0, c.outpost.max_yaw_rate_step);
    c.outpost.binding_dz_ema_alpha =
      std::clamp(c.outpost.binding_dz_ema_alpha, 0.01, 1.0);
    c.outpost.binding_confidence_floor =
      std::clamp(c.outpost.binding_confidence_floor, 0.0, 0.95);
    c.outpost.z_audit_rebind_confirm_frames =
      std::max(1, c.outpost.z_audit_rebind_confirm_frames);
    c.outpost.z_audit_rebind_min_confidence =
      std::clamp(c.outpost.z_audit_rebind_min_confidence, 0.0, 1.0);
    c.outpost.z_audit_rebind_min_jump =
      std::max(0.0, c.outpost.z_audit_rebind_min_jump);
    c.outpost.v2_warmup_min_groups =
      std::clamp(c.outpost.v2_warmup_min_groups, 2, 6);
    c.outpost.v2_warmup_min_samples_per_group =
      std::max(1, c.outpost.v2_warmup_min_samples_per_group);
    c.outpost.v2_warmup_max_frames =
      std::max(1, c.outpost.v2_warmup_max_frames);
    c.outpost.v2_warmup_z_jump_gate =
      std::max(0.0, c.outpost.v2_warmup_z_jump_gate);
    c.outpost.v2_warmup_yaw_jump_gate =
      std::max(0.0, c.outpost.v2_warmup_yaw_jump_gate);
    c.outpost.v2_warmup_xyz_jump_gate =
      std::max(0.0, c.outpost.v2_warmup_xyz_jump_gate);
    c.outpost.v2_warmup_ratio_min =
      std::max(1.0, c.outpost.v2_warmup_ratio_min);
    c.outpost.v2_warmup_ratio_max =
      std::max(c.outpost.v2_warmup_ratio_min,
               c.outpost.v2_warmup_ratio_max);
    c.outpost.v2_warmup_min_large_diff =
      std::max(0.0, c.outpost.v2_warmup_min_large_diff);
    c.outpost.binding_conflict_position_scale =
      std::clamp(c.outpost.binding_conflict_position_scale, 0.0, 1.0);
    c.outpost.weight_xy_residual = std::max(0.0, c.outpost.weight_xy_residual);
    c.outpost.weight_switch_penalty = std::max(0.0, c.outpost.weight_switch_penalty);
    c.outpost.mode_enter_confirm_frames =
      std::max(1, c.outpost.mode_enter_confirm_frames);
    c.outpost.mode_exit_confirm_frames =
      std::max(1, c.outpost.mode_exit_confirm_frames);
    c.outpost.mode_min_dwell_frames =
      std::max(1, c.outpost.mode_min_dwell_frames);
    c.outpost.mode_enter_threshold =
      std::clamp(c.outpost.mode_enter_threshold, 0.0, 1.0);
    c.outpost.mode_exit_threshold =
      std::clamp(c.outpost.mode_exit_threshold, 0.0, 1.0);
    c.outpost.mode_weight_jump = std::max(0.0, c.outpost.mode_weight_jump);
    c.outpost.mode_weight_dual = std::max(0.0, c.outpost.mode_weight_dual);
    c.outpost.mode_weight_margin = std::max(0.0, c.outpost.mode_weight_margin);
    c.outpost.mode_weight_health = std::max(0.0, c.outpost.mode_weight_health);
    c.outpost.mode_weight_entropy = std::max(0.0, c.outpost.mode_weight_entropy);
    c.outpost.v3_topk = std::max(1, c.outpost.v3_topk);
    c.outpost.v3_mode_stable_frames = std::max(1, c.outpost.v3_mode_stable_frames);
    c.outpost.v3_mode_degraded_frames = std::max(1, c.outpost.v3_mode_degraded_frames);
    c.outpost.v3_warmup_frames = std::max(1, c.outpost.v3_warmup_frames);
    c.outpost.v3_warmup_min_settle_frames =
      std::max(1, c.outpost.v3_warmup_min_settle_frames);
    c.outpost.v3_phase_audit_confirm_frames =
      std::max(1, c.outpost.v3_phase_audit_confirm_frames);

  const bool z_descending =
      (c.outpost.z_offset_0 > c.outpost.z_offset_1) &&
      (c.outpost.z_offset_1 > c.outpost.z_offset_2);
  if (!z_descending) {
    RCLCPP_WARN(
        get_logger(),
        "Outpost z-offset semantic mismatch: expected z0>z1>z2 for "
        "[highest,middle,lowest], got [%.4f, %.4f, %.4f]",
        c.outpost.z_offset_0, c.outpost.z_offset_1, c.outpost.z_offset_2);
  }

  const double expected_step = 2.0 * M_PI / 3.0;
  if (std::abs(std::abs(c.outpost.panel_angle_step) - expected_step) > 1e-3) {
    RCLCPP_WARN(
        get_logger(),
        "Outpost panel_angle_step=%.6f differs from 2pi/3; semantic contract "
        "(top-down clockwise: 0->2->1) assumes 120deg spacing.",
        c.outpost.panel_angle_step);
  }

  static bool outpost_semantic_logged = false;
  if (!outpost_semantic_logged) {
    outpost_semantic_logged = true;
    RCLCPP_INFO(
        get_logger(),
        "Outpost semantic contract enabled: id0=highest@0deg, clockwise order "
        "id0->id2->id1.");
  }

  // Output smoother
  smoother_config_.enable = get_parameter("smoother.enable").as_bool();
  smoother_config_.enable_position_smooth =
      get_parameter("smoother.enable_position_smooth").as_bool();
  smoother_config_.enable_yaw_smooth =
      get_parameter("smoother.enable_yaw_smooth").as_bool();
  smoother_config_.enable_velocity_smooth =
      get_parameter("smoother.enable_velocity_smooth").as_bool();
  smoother_config_.enable_structural_convergence =
      get_parameter("smoother.enable_structural_convergence").as_bool();
  smoother_config_.pos_min_cutoff =
      get_parameter("smoother.pos_min_cutoff").as_double();
  smoother_config_.pos_beta = get_parameter("smoother.pos_beta").as_double();
  smoother_config_.pos_d_cutoff =
      get_parameter("smoother.pos_d_cutoff").as_double();
  smoother_config_.yaw_min_cutoff =
      get_parameter("smoother.yaw_min_cutoff").as_double();
  smoother_config_.yaw_beta = get_parameter("smoother.yaw_beta").as_double();
  smoother_config_.yaw_d_cutoff =
      get_parameter("smoother.yaw_d_cutoff").as_double();
  smoother_config_.vel_min_cutoff =
      get_parameter("smoother.vel_min_cutoff").as_double();
  smoother_config_.vel_beta = get_parameter("smoother.vel_beta").as_double();
  smoother_config_.vel_d_cutoff =
      get_parameter("smoother.vel_d_cutoff").as_double();
  smoother_config_.rm_initial_step =
      get_parameter("smoother.rm_initial_step").as_double();
  smoother_config_.rm_gamma = get_parameter("smoother.rm_gamma").as_double();
  smoother_config_.rm_n0 = get_parameter("smoother.rm_n0").as_int();
  smoother_config_.rm_dual_obs_boost =
      get_parameter("smoother.rm_dual_obs_boost").as_double();
  smoother_config_.rm_min_radius =
      get_parameter("smoother.rm_min_radius").as_double();
  smoother_config_.rm_max_radius =
      get_parameter("smoother.rm_max_radius").as_double();
  smoother_config_.rm_min_dz =
      get_parameter("smoother.rm_min_dz").as_double();
  smoother_config_.rm_max_dz =
      get_parameter("smoother.rm_max_dz").as_double();
  smoother_config_.rm_convergence_eps =
      get_parameter("smoother.rm_convergence_eps").as_double();
  smoother_config_.default_freq =
      get_parameter("smoother.default_freq").as_double();

  // Outlier filter parameters
  smoother_config_.enable_outlier_filter =
      get_parameter("smoother.enable_outlier_filter").as_bool();
  smoother_config_.outlier_method =
      get_parameter("smoother.outlier_method").as_string();
  smoother_config_.outlier_window_size =
      get_parameter("smoother.outlier_window_size").as_int();
  smoother_config_.outlier_min_samples =
      get_parameter("smoother.outlier_min_samples").as_int();
  smoother_config_.outlier_mad_k =
      get_parameter("smoother.outlier_mad_k").as_double();
  smoother_config_.outlier_iqr_k =
      get_parameter("smoother.outlier_iqr_k").as_double();
  smoother_config_.outlier_mahal_threshold =
      get_parameter("smoother.outlier_mahal_threshold").as_double();

  RCLCPP_INFO(get_logger(),
              "Tracker parameters applied (smoother %s, outlier_filter %s [%s])",
              smoother_config_.enable ? "ON" : "OFF",
              smoother_config_.enable_outlier_filter ? "ON" : "OFF",
              smoother_config_.outlier_method.c_str());
}

/* ================================================================ */
/*  Armors callback — tracker + selector pipeline                    */
/* ================================================================ */

void GimbalPipelineNode::armorsCallback(
    const rm_interfaces::msg::Armors::SharedPtr msg) {
  rclcpp::Time msg_time(msg->header.stamp);
  double current_time = msg_time.seconds();
  if (msg->armors.empty()) {
    RCLCPP_DEBUG_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Received empty armors message, running missing-target update");
  }

  // ── Step 1: Group observations by robot ID ──
  std::unordered_map<std::string, std::vector<ObservationData>> obs_by_robot;
  std::string sf =
      msg->header.frame_id.empty() ? source_frame_ : msg->header.frame_id;

  const bool strict_unknown_reject =
      robot_description_facade_ && robot_description_facade_->strictUnknownReject();

  for (const auto &armor : msg->armors) {
    const bool supported_robot_id =
        robot_description_facade_ &&
        robot_description_facade_->isSupportedRobotId(armor.number);

    // Strict mode: skip unsupported IDs to prevent false tracker creation.
    if (strict_unknown_reject && !supported_robot_id) {
      if (debug_mode_)
        RCLCPP_WARN(get_logger(), "Ignoring armor with invalid ID: '%s'",
                    armor.number.c_str());
      continue;
    }
    auto obs =
        tf_handler_->transform_armor_to_observation(armor, sf, msg_time);
    if (!obs.has_value()) {
      if (debug_mode_)
        RCLCPP_WARN(get_logger(), "TF failed for armor %s",
                    armor.number.c_str());
      continue;
    }
    obs_by_robot[armor.number].push_back(obs.value());
  }

  // ── Log observations (before update, so we record incoming sensor data) ──
  if (prediction_logger_) {
    int64_t ts_ns = msg_time.nanoseconds();
    for (const auto &[rid, obs_list] : obs_by_robot) {
      bool is_dual = (obs_list.size() >= 2);
      std::vector<LogObservation> log_obs;
      log_obs.reserve(obs_list.size());
      for (const auto &o : obs_list) {
        LogObservation lo;
        lo.x          = o.x;
        lo.y          = o.y;
        lo.z          = o.z;
        lo.yaw        = o.yaw;
        lo.panel_id   = o.panel_id.value_or(-1);
        lo.confidence = o.confidence;
        lo.is_dual_obs = is_dual;
        log_obs.push_back(lo);
      }
      prediction_logger_->logObservations(ts_ns, rid, log_obs);
    }
  }

  // ── Step 2: Run tracker core frame process in manager ──
  auto frame_result = tracker_manager_->process_frame(
      obs_by_robot, current_time, smoother_config_);
  if (!frame_result.removed_stale_ids.empty() && debug_mode_) {
    RCLCPP_INFO(get_logger(), "Removed %zu stale trackers",
                frame_result.removed_stale_ids.size());
  }
  if (!frame_result.removed_lost_ids.empty() && debug_mode_) {
    RCLCPP_INFO(get_logger(), "Removed %zu lost trackers",
                frame_result.removed_lost_ids.size());
  }

  // ── Step 3: Build TrackedRobots message (internal) ──
  auto tracked_msg = buildTrackedRobotsMsg(msg->header);

  // ── Log tracker posterior states (after update, before selection) ──
  if (prediction_logger_) {
    int64_t ts_ns = msg_time.nanoseconds();
    for (const auto &robot : tracked_msg.robots) {
      const auto center_position = robot_description::TrackedRobotUsage::centerPosition(robot);
      const auto linear_velocity = robot_description::TrackedRobotUsage::linearVelocity(robot);
      LogTrackerState st;
      st.center_x           = center_position.x();
      st.center_y           = center_position.y();
      st.center_z           = center_position.z();
      st.vel_x              = linear_velocity.x();
      st.vel_y              = linear_velocity.y();
      st.vel_z              = linear_velocity.z();
      st.yaw                = robot_description::TrackedRobotUsage::yaw(robot);
      st.yaw_velocity       = robot_description::TrackedRobotUsage::yawVelocity(robot);
      st.yaw_acceleration   = robot_description::TrackedRobotUsage::yawAcceleration(robot);
      st.radius_1           = robot.radius;
      st.radius_2           = robot.radius_2;
      st.dza                = robot.d_za;
      st.track_state        = robot.track_state;
      st.num_armors         = robot.num_armors;
      st.visible_armor_count = robot.visible_armor_count;
      st.is_visible         = robot.is_visible;
      st.confidence         = robot.confidence;

      // ── 机动检测指标：从对应 tracker 的 UKF 内部读取 ──
      auto *tracker = tracker_manager_->get(robot.robot_id);
      if (tracker && tracker->is_initialized()) {
        const auto &ukf = tracker->spin_filter();
        const auto &idx = ukf.state_idx();
        const auto &xv  = ukf.x();
        const auto &Pv  = ukf.P();

        // 创新向量
        const auto &iv = ukf.last_innov_xyz();
        if (iv.size() >= 3) {
          st.innov_x = iv(0);
          st.innov_y = iv(1);
          st.innov_z = iv(2);
        }
        st.innov_yaw   = ukf.last_innov_yaw();
        st.nis         = ukf.last_nis();
        st.update_type = ukf.last_update_type();

        // P 对角线 —— 位置与速度
        st.p_var_x  = Pv(idx.X(),  idx.X());
        st.p_var_y  = Pv(idx.Y(),  idx.Y());
        st.p_var_z  = Pv(idx.Z(),  idx.Z());
        st.p_var_vx = Pv(idx.VX(), idx.VX());
        st.p_var_vy = Pv(idx.VY(), idx.VY());
        st.p_var_vz = Pv(idx.VZ(), idx.VZ());

        // 加速度状态（仅 CA / Singer 过程模型存在 AX/AY/AZ）
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (idx.has("AX")) {
          st.p_var_ax = Pv(idx.AX(), idx.AX());
          st.p_var_ay = Pv(idx.AY(), idx.AY());
          st.p_var_az = Pv(idx.AZ(), idx.AZ());
          st.accel_x  = xv(idx.AX());
          st.accel_y  = xv(idx.AY());
          st.accel_z  = xv(idx.AZ());
          st.accel_magnitude = std::sqrt(xv(idx.AX()) * xv(idx.AX()) +
                                         xv(idx.AY()) * xv(idx.AY()) +
                                         xv(idx.AZ()) * xv(idx.AZ()));
        } else {
          st.p_var_ax = kNaN;
          st.p_var_ay = kNaN;
          st.p_var_az = kNaN;
          st.accel_x         = kNaN;
          st.accel_y         = kNaN;
          st.accel_z         = kNaN;
          st.accel_magnitude = kNaN;
        }

        if (robot.robot_id == "outpost") {
          auto apply_outpost_snapshot = [&](const auto &snap) {
            if (!snap.valid) return;
            st.outpost_mode = snap.track_mode;
            st.estimated_id = snap.estimated_id;
            st.runtime_panel_id = snap.runtime_panel_id;
            st.bound_height_label = snap.bound_height_label;
            st.obs_inferred_id = snap.obs_inferred_id;
            st.obs_inferred_id_z = snap.obs_inferred_id_z;
            st.candidate_panel_id = snap.candidate_panel_id;
            st.candidate_prob = snap.candidate_prob;
            st.candidate_margin = snap.candidate_margin;
            st.selected_xy_residual = snap.selected_xy_residual;
            st.outpost_entropy = snap.entropy_norm;
            st.outpost_max_prob = snap.max_prob;
            st.hyp_cost_0 = snap.hyp_costs[0];
            st.hyp_cost_1 = snap.hyp_costs[1];
            st.hyp_cost_2 = snap.hyp_costs[2];
            st.hyp_prob_0 = snap.hyp_probs[0];
            st.hyp_prob_1 = snap.hyp_probs[1];
            st.hyp_prob_2 = snap.hyp_probs[2];
            st.center_yaw_est = snap.center_yaw_est;
            st.has_observation = snap.has_observation ? 1 : 0;
            st.obs_x = snap.obs_x;
            st.obs_y = snap.obs_y;
            st.obs_z = snap.obs_z;
            st.obs_yaw = snap.obs_yaw;
            st.obs_z_jump = snap.obs_z_jump;
            st.obs_dz_from_audit_center = snap.obs_dz_from_audit_center;
            st.obs_z_audit_cost_0 = snap.obs_z_audit_costs[0];
            st.obs_z_audit_cost_1 = snap.obs_z_audit_costs[1];
            st.obs_z_audit_cost_2 = snap.obs_z_audit_costs[2];
            st.binding_confidence = snap.binding_confidence;
            st.switch_event = snap.switch_event;
            st.switch_reason = snap.switch_reason;
            st.transition_state = snap.transition_state;
            st.z_audit_conflict_count = snap.z_audit_conflict_count;
            st.z_audit_confidence = snap.z_audit_confidence;
            st.publish_x = snap.publish_x;
            st.publish_y = snap.publish_y;
            st.publish_z = snap.publish_z;
            st.period_confidence = snap.period_confidence;
            st.period_update_applied = snap.period_update_applied;
            st.period_phase_index = snap.period_phase_index;
            st.spin_direction = snap.spin_direction;
            st.dz_small_est = snap.dz_small_est;
            st.dz_large_est = snap.dz_large_est;
          };

          if (const auto *outpost_tracker =
                  dynamic_cast<const OutpostArmorTracker *>(tracker);
              outpost_tracker != nullptr) {
            apply_outpost_snapshot(outpost_tracker->debug_snapshot());
          } else if (const auto *outpost_v2_tracker =
                         dynamic_cast<const OutpostTrackerV2 *>(tracker);
                     outpost_v2_tracker != nullptr) {
            apply_outpost_snapshot(outpost_v2_tracker->debug_snapshot());
          }
        }
      }

      prediction_logger_->logTrackerState(ts_ns, robot.robot_id, st);
    }
  }

  // ── Step 4: Target selection (direct C++ call, no ROS topic!) ──
  SelectionResult sel_result;
  if (!tracked_msg.robots.empty()) {
    sel_result = selectTargetInternal(tracked_msg);
  }

  // ── Step 5: Store results for timerCallback (thread-safe) ──
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    latest_tracked_robots_ =
        std::make_shared<rm_interfaces::msg::TrackedRobots>(tracked_msg);
    latest_selected_target_id_ = sel_result.robot_id;
    latest_selected_confidence_ = sel_result.confidence;
    latest_update_time_ = now();  // record local clock for processing_delay
  }

  // ── Step 6: Debug publishing ──
  const auto tracker_views = tracker_manager_->initialized_tracker_views();
  logNorm4V3TrackerDebug(tracker_views);
  if (debug_mode_) {
    if (debug_tracked_robots_pub_ && !tracked_msg.robots.empty())
      debug_tracked_robots_pub_->publish(tracked_msg);

    if (debug_selected_target_pub_) {
      rm_interfaces::msg::SelectedTarget sel_msg;
      sel_msg.header.stamp = now();
      sel_msg.header.frame_id = target_frame_;
      sel_msg.robot_id = sel_result.robot_id;
      sel_msg.confidence = sel_result.confidence;
      sel_msg.selection_strategy = selector_strategy_name_;
      debug_selected_target_pub_->publish(sel_msg);
    }

    if (debug_tracker_marker_pub_) {
      rclcpp::Time stamp(msg->header.stamp);
      auto marker_array = build_tracker_markers(
          visualization_frame_, tracker_views, stamp);
      debug_tracker_marker_pub_->publish(marker_array);
    }

    if (debug_maneuver_pub_) {
      publishManeuverMarkers(msg->header);
    }

    if (debug_tracker_2d_image_pub_) {
      publish2DTrackerDebugImage(msg->header, tracker_views);
    }
    if (debug_evidence_frame_pub_) {
      publishEvidenceFrameDebug(msg->header, tracker_views);
    }
  }

  // ── Step 7: Publish maneuver states (always-on, for chart monitoring) ──
  if (maneuver_states_pub_) {
    rm_interfaces::msg::ManeuverStates states_msg;
    states_msg.header.stamp    = msg->header.stamp;
    states_msg.header.frame_id = target_frame_;
    for (const auto &view : tracker_views) {
      if (!view.tracker) continue;
      const auto result = view.tracker->assess_maneuver();
      const auto &ukf   = view.tracker->spin_filter();
      rm_interfaces::msg::ManeuverState s;
      s.robot_id        = view.robot_id;
      s.is_maneuvering  = result.is_maneuvering;
      s.nis             = result.nis;
      s.innov_norm      = result.innov_norm;
      s.innov_yaw_abs   = std::abs(ukf.last_innov_yaw());
      s.update_type     = result.update_type;
      states_msg.states.push_back(s);
    }
    maneuver_states_pub_->publish(states_msg);
  }
}

/* ================================================================ */
/*  Build TrackedRobots message (from tracker state)                 */
/* ================================================================ */

rm_interfaces::msg::TrackedRobots GimbalPipelineNode::buildTrackedRobotsMsg(
    const std_msgs::msg::Header &header) {
  rm_interfaces::msg::TrackedRobots tracked_msg;
  tracked_msg.header = header;
  tracked_msg.header.frame_id = target_frame_;

  auto tracking_ids = tracker_manager_->active_robot_ids();
  for (const auto &rid : tracking_ids) {
    auto *tracker = tracker_manager_->get(rid);
    if (!tracker || (!tracker->is_tracking() && !tracker->is_temp_lost())) continue;

    const rclcpp::Time stamp(header.stamp);
    const double ts = stamp.seconds();
    auto post = tracker_manager_->post_process_output(rid, ts, smoother_config_);
    const SmoothedOutput *smoothed = post.has_smoothed ? &post.smoothed : nullptr;
    const int visible_armor_count =
        tracker_manager_->visible_observation_count(rid);

    // Publish target for debug
    if (debug_mode_ && debug_target_pub_) {
      auto target = buildTargetMessage(header, rid, *tracker, smoothed);
      debug_target_pub_->publish(target);
    }

    auto robot = buildTrackedRobotMessage(
        header, rid, *tracker, smoothed, visible_armor_count);

    if (robot.robot_id.empty()) {
      continue;
    }

    tracked_msg.robots.push_back(robot);
  }

  mergeExternalTargets(tracked_msg, header);

  return tracked_msg;
}

void GimbalPipelineNode::mergeExternalTargets(
    rm_interfaces::msg::TrackedRobots & tracked_msg,
    const std_msgs::msg::Header & header)
{
  if (!external_targets_enable_ || !external_targets_buff_enable_ || !buff_target_adapter_) {
    return;
  }

  const auto now = rclcpp::Time(header.stamp);
  auto buff_robot_opt = buff_target_adapter_->latestValid(now);
  if (!buff_robot_opt.has_value()) {
    return;
  }

  auto buff_robot = buff_robot_opt.value();
  if (!active_external_allowed_ids_.empty() &&
    active_external_allowed_ids_.find(buff_robot.robot_id) == active_external_allowed_ids_.end())
  {
    return;
  }

  buff_robot.header = tracked_msg.header;
  bool replaced = false;
  for (auto & robot : tracked_msg.robots) {
    if (robot.robot_id == buff_robot.robot_id) {
      robot = buff_robot;
      replaced = true;
      break;
    }
  }
  if (!replaced) {
    tracked_msg.robots.push_back(buff_robot);
  }
}

void GimbalPipelineNode::refreshExternalTargetAllowlist(int mode)
{
  current_mode_ = mode;
  auto it = allowed_ids_by_mode_.find(mode);
  if (it == allowed_ids_by_mode_.end()) {
    active_external_allowed_ids_.clear();
    return;
  }
  active_external_allowed_ids_ = it->second;
}

/* ================================================================ */
/*  Target message builders (from MaxEntropyTrackerNode)             */
/* ================================================================ */

rm_interfaces::msg::Target GimbalPipelineNode::buildTargetMessage(
    const std_msgs::msg::Header &header, const std::string &robot_id,
  BaseTracker &tracker, const SmoothedOutput *smoothed) {
  rm_interfaces::msg::Target target;
  target.header = header;
  target.header.frame_id = target_frame_;
  target.tracking = true;
  target.id = robot_id;

  // Keep debug target semantic aligned with tracked robot profile.
  target.armors_num = 4;
  if (robot_id == "outpost" || robot_id == "base") {
    target.armors_num = 3;
  }
  const int runtime_num_armors = tracker.effective_num_armors();
  if (runtime_num_armors > 0) {
    target.armors_num = runtime_num_armors;
  }

  if (smoothed) {
    target.position.x = smoothed->center_position.x();
    target.position.y = smoothed->center_position.y();
    target.position.z = smoothed->center_position.z();
    target.velocity.x = smoothed->velocity.x();
    target.velocity.y = smoothed->velocity.y();
    target.velocity.z = smoothed->velocity.z();
    target.yaw = smoothed->yaw;
    target.v_yaw = smoothed->yaw_velocity;
    target.radius_1 = smoothed->r1;
    target.radius_2 = smoothed->r2;
    target.d_za = smoothed->dza;
  } else {
    auto pos = tracker.get_center_position();
    target.position.x = pos.x();
    target.position.y = pos.y();
    target.position.z = pos.z();
    const auto &filter = tracker.spin_filter();
    auto idx = filter.state_idx();
    const auto &x = filter.x();
    const auto pub_vel = tracker.get_publish_velocity();
    target.velocity.x = pub_vel.x();
    target.velocity.y = pub_vel.y();
    target.velocity.z = pub_vel.z();
    target.yaw = tracker.get_yaw();
    target.v_yaw = x(idx.DELTA_RATE());
    auto [r1, r2] = tracker.get_radii();
    target.radius_1 = r1;
    target.radius_2 = r2;
    target.d_za = filter.get_dza();
  }

  target.d_zc = 0.0;
  target.yaw_diff = 0.0;
  target.position_diff = 0.0;
  return target;
}

rm_interfaces::msg::TrackedRobot GimbalPipelineNode::buildTrackedRobotMessage(
    const std_msgs::msg::Header &header, const std::string &robot_id,
  BaseTracker &tracker, const SmoothedOutput *smoothed,
  int visible_armor_count) {
  rm_interfaces::msg::TrackedRobot empty_msg;

  if (!robot_description_facade_) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "RobotDescriptionFacade is not initialized, skip TrackedRobot build");
    return empty_msg;
  }

  robot_description::TrackedRobotBuildInput input{
    header,
    target_frame_,
    robot_id,
    tracker,
    smoothed,
    visible_armor_count};

  auto build_result = robot_description_facade_->tryBuildTrackedRobot(input);
  if (!build_result.ok()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Rejected TrackedRobot build for id='%s': %s",
      robot_id.c_str(),
      build_result.reason.c_str());
    return empty_msg;
  }

  return build_result.robot;
}

/* ================================================================ */
/*  Target selection (internal, no ROS topic)                        */
/* ================================================================ */

void GimbalPipelineNode::initSelectionStrategy() {
  if (selector_strategy_name_ == "min_yaw_deviation") {
    selection_strategy_ = std::make_unique<MinYawDeviationStrategy>();
  } else if (selector_strategy_name_ == "priority_list") {
    selection_strategy_ = std::make_unique<PriorityListStrategy>();
  } else if (selector_strategy_name_ == "sticky_min_yaw_deviation") {
    selection_strategy_ = std::make_unique<StickyMinYawDeviationStrategy>();
  } else {
    RCLCPP_WARN(get_logger(), "Unknown selector strategy '%s', using min_yaw_deviation",
                selector_strategy_name_.c_str());
    selection_strategy_ = std::make_unique<MinYawDeviationStrategy>();
  }
  RCLCPP_INFO(get_logger(), "Selection strategy: %s",
              selection_strategy_->getName().c_str());
}

SelectionResult GimbalPipelineNode::selectTargetInternal(
    const rm_interfaces::msg::TrackedRobots &robots) {
  selection_config_.current_target_id = current_target_id_;

  auto result = selection_strategy_->selectTarget(robots, selection_config_);

  if (!result.has_value()) {
    current_target_id_ = "";
    return SelectionResult();
  }

  bool target_changed = (result->robot_id != current_target_id_);
  if (target_changed) {
    RCLCPP_INFO(get_logger(), "Target changed: %s -> %s (yaw_dev=%.3f, dist=%.2f)",
                current_target_id_.empty() ? "none" : current_target_id_.c_str(),
                result->robot_id.c_str(), result->yaw_deviation,
                result->distance);
    current_target_id_ = result->robot_id;
  }

  return *result;
}

/* ================================================================ */
/*  Gimbal controller initialization                                 */
/* ================================================================ */

void GimbalPipelineNode::initGimbalComponents() {
  position_calculator_ =
      std::make_shared<gimbal_controller::ArmorPositionCalculator>();
  armor_selector_ = std::make_shared<gimbal_controller::ArmorSelector>();
  ballistic_client_ =
      std::make_shared<gimbal_controller::BallisticSolverClient>(this);
  local_compensator_ =
      std::make_shared<gimbal_controller::LocalTrajectoryCompensator>();
  fire_advisor_ = std::make_shared<gimbal_controller::FireAdvisor>();
  fire_advice_engine_ = std::make_shared<gimbal_controller::FireAdviceEngine>();
  fire_advice_engine_->setComponents(
    position_calculator_, ballistic_client_, local_compensator_, fire_advisor_);
  fire_advice_engine_->setBallisticMode(ballistic_mode_);
  gimbal_control_core_ = std::make_shared<gimbal_controller::GimbalControlCore>();
  gimbal_control_core_->setFireModules(fire_advice_engine_, fire_advisor_);
}

void GimbalPipelineNode::initGimbalStrategies() {
  auto current_s =
      std::make_shared<gimbal_controller::CurrentPositionStrategy>();
  current_s->setComponents(position_calculator_, armor_selector_,
                           ballistic_client_, local_compensator_,
                           fire_advisor_);
  current_s->setBallisticMode(ballistic_mode_);
  gimbal_strategies_["current"] = current_s;

  auto predicted_s =
      std::make_shared<gimbal_controller::PredictedPositionStrategy>();
  predicted_s->setComponents(position_calculator_, armor_selector_,
                             ballistic_client_, local_compensator_,
                             fire_advisor_);
  predicted_s->setBallisticMode(ballistic_mode_);
  gimbal_strategies_["predicted"] = predicted_s;

  auto mpc_s = std::make_shared<gimbal_controller::MpcControlStrategy>();
  mpc_s->setComponents(position_calculator_, armor_selector_,
                       ballistic_client_, local_compensator_, fire_advisor_);
  mpc_s->setBallisticMode(ballistic_mode_);
  mpc_s->initReferenceGenerator();

  const double mpc_control_delay_s = readUnifiedDoubleParameter(
    *this,
    "controller.delay.control_latency_s",
    {
      "controller.mpc.control_delay_s",
      "mpc.control_delay_s",
      "controller.solver.controller_delay",
      "solver.controller_delay"
    });
  const bool mpc_enable_delay_compensation = readCompatBoolParameter(
    *this, "controller.mpc.enable_delay_compensation", "mpc.enable_delay_compensation");
  const bool mpc_allow_muzzle_compensation =
    get_parameter("controller.mpc.allow_muzzle_compensation").as_bool();
  const double mpc_prediction_delay_s = readUnifiedDoubleParameter(
    *this,
    "controller.delay.prediction_extra_s",
    {
      "controller.mpc.prediction_delay_s",
      "mpc.prediction_delay_s",
      "controller.solver.prediction_delay",
      "solver.prediction_delay"
    });
  const double mpc_trigger_to_muzzle_s = readUnifiedDoubleParameter(
    *this,
    "controller.delay.trigger_to_muzzle_s",
    {
      "controller.fire.trigger_to_muzzle_s",
      "controller.solver.trigger_to_muzzle_s",
      "solver.trigger_to_muzzle_s"
    });
  const int mpc_flight_time_iters = readUnifiedIntParameter(
    *this,
    "controller.delay.flight_time_iters",
    {
      "controller.mpc.flight_time_iters",
      "mpc.flight_time_iters",
      "controller.fire.flight_time_iters"
    });
  const double mpc_max_processing_delay_s = readUnifiedDoubleParameter(
    *this,
    "controller.delay.max_processing_delay_s",
    {
      "controller.mpc.max_processing_delay_s",
      "mpc.max_processing_delay_s"
    });

  mpc_s->setMpcParameters(
    get_parameter("controller.mpc.N").as_int(),
    get_parameter("controller.mpc.dt").as_double(),
    mpc_control_delay_s,
    get_parameter("controller.mpc.max_accel").as_double(),
    get_parameter("controller.mpc.q_yaw").as_double(),
    get_parameter("controller.mpc.q_pitch").as_double(),
    get_parameter("controller.mpc.q_yaw_vel").as_double(),
    get_parameter("controller.mpc.q_pitch_vel").as_double(),
    get_parameter("controller.mpc.r_yaw").as_double(),
    get_parameter("controller.mpc.r_pitch").as_double(),
    get_parameter("controller.mpc.s_yaw").as_double(),
    get_parameter("controller.mpc.s_pitch").as_double());
  mpc_s->setDelayCompensation(
    mpc_enable_delay_compensation,
    mpc_prediction_delay_s,
    mpc_trigger_to_muzzle_s,
    mpc_allow_muzzle_compensation,
    mpc_flight_time_iters,
    mpc_max_processing_delay_s);
  mpc_s->setYawFeedforward(
    get_parameter("controller.mpc.yaw_feedforward_k_s").as_double());
  mpc_s->setManualOffset(
    get_parameter("controller.solver.pitch_offset").as_double(),
    get_parameter("controller.solver.yaw_offset").as_double());
  mpc_s->setManeuverAdaptParameters(
    get_parameter("controller.mpc.maneuver_adapt.enable").as_bool(),
    get_parameter("controller.mpc.maneuver_adapt.a_max").as_double(),
    get_parameter("controller.mpc.maneuver_adapt.eta").as_double(),
    get_parameter("controller.mpc.maneuver_adapt.tau").as_double(),
    get_parameter("controller.mpc.maneuver_adapt.r_scale").as_double());
  mpc_s->setWeightingParameters(
    get_parameter("controller.mpc.weighting.enable").as_bool(),
    get_parameter("controller.mpc.weighting.alpha").as_double(),
    get_parameter("controller.mpc.weighting.k_omega").as_double(),
    get_parameter("controller.mpc.weighting.sigma_min").as_double(),
    get_parameter("controller.mpc.weighting.sigma_max").as_double(),
    get_parameter("controller.mpc.weighting.sigma_sys").as_double(),
    get_parameter("controller.mpc.weighting.target_size").as_double(),
    get_parameter("controller.mpc.weighting.delay_s").as_double(),
    get_parameter("controller.mpc.weighting.max_w").as_double(),
    get_parameter("controller.mpc.weighting.smooth_alpha").as_double(),
    get_parameter("controller.mpc.weighting.min_distance").as_double(),
    get_parameter("controller.bullet_speed").as_double(),
    get_parameter("controller.mpc.weighting.sigma_beta").as_double(),
    get_parameter("controller.mpc.weighting.gamma").as_double());
  {
    gimbal_controller::mpc::VelocityClampConfig vel_clamp_cfg;
    vel_clamp_cfg.enable =
      get_parameter("controller.mpc.vel_clamp.enable").as_bool();
    vel_clamp_cfg.max_linear_speed =
      get_parameter("controller.mpc.vel_clamp.max_linear_speed").as_double();
    vel_clamp_cfg.max_v_yaw =
      get_parameter("controller.mpc.vel_clamp.max_v_yaw").as_double();
    mpc_s->setVelocityClamp(vel_clamp_cfg);
  }
  mpc_s->setFovConstraintParameters(
    get_parameter("controller.mpc.fov_constraint.enable").as_bool(),
    get_parameter("controller.mpc.fov_constraint.margin").as_double(),
    get_parameter("controller.mpc.fov_constraint.slack_weight").as_double(),
    get_parameter("controller.mpc.fov_constraint.constraint_steps").as_int(),
    get_parameter("controller.mpc.fov_constraint.dynamic_margin.enable").as_bool(),
    get_parameter("controller.mpc.fov_constraint.dynamic_margin.vel_scale").as_double(),
    get_parameter("controller.mpc.fov_constraint.fallback_fov_yaw").as_double(),
    get_parameter("controller.mpc.fov_constraint.fallback_fov_pitch").as_double());
  mpc_s->setNumericalNormalizationParameters(
    get_parameter("controller.mpc.normalization.enable").as_bool(),
    get_parameter("controller.mpc.normalization.window_size").as_int(),
    get_parameter("controller.mpc.normalization.min_samples").as_int(),
    get_parameter("controller.mpc.normalization.rms_epsilon").as_double(),
    get_parameter("controller.mpc.normalization.mode").as_string(),
    Eigen::Vector4d(
      get_parameter("controller.mpc.normalization.typical_state.yaw").as_double(),
      get_parameter("controller.mpc.normalization.typical_state.pitch").as_double(),
      get_parameter("controller.mpc.normalization.typical_state.yaw_vel").as_double(),
      get_parameter("controller.mpc.normalization.typical_state.pitch_vel").as_double()),
    Eigen::Vector2d(
      get_parameter("controller.mpc.normalization.typical_control.yaw_acc").as_double(),
      get_parameter("controller.mpc.normalization.typical_control.pitch_acc").as_double()),
    Eigen::Vector2d(
      get_parameter("controller.mpc.normalization.typical_delta_control.yaw_acc").as_double(),
      get_parameter("controller.mpc.normalization.typical_delta_control.pitch_acc").as_double()));
  mpc_s->setHessianRegularizationParameters(
    get_parameter("controller.mpc.regularization.enable").as_bool(),
    get_parameter("controller.mpc.regularization.epsilon_abs").as_double(),
    get_parameter("controller.mpc.regularization.epsilon_rel").as_double(),
    get_parameter("controller.mpc.regularization.epsilon_max").as_double(),
    get_parameter("controller.mpc.regularization.retry_on_fail").as_bool(),
    get_parameter("controller.mpc.regularization.retry_scale").as_double());
  mpc_s->setDiagnosticsParameters(
    get_parameter("controller.mpc.diagnostics.enable").as_bool(),
    get_parameter("controller.mpc.diagnostics.low_cost_always").as_bool(),
    get_parameter("controller.mpc.diagnostics.high_cost_enable").as_bool(),
    get_parameter("controller.mpc.diagnostics.high_cost_sample_every").as_int(),
    get_parameter("controller.mpc.diagnostics.log_every").as_int(),
    get_parameter("controller.mpc.diagnostics.log_on_failure").as_bool(),
    get_parameter("controller.mpc.diagnostics.active_tol").as_double(),
    get_parameter("controller.mpc.diagnostics.rank_tol_rel").as_double());
  gimbal_strategies_["mpc"] = mpc_s;

  auto sm_s = std::make_shared<gimbal_controller::StateMachineStrategy>();
  sm_s->setComponents(position_calculator_, armor_selector_,
                      ballistic_client_, local_compensator_, fire_advisor_);
  sm_s->setBallisticMode(ballistic_mode_);
  gimbal_strategies_["state_machine"] = sm_s;

  if (gimbal_control_core_) {
    gimbal_control_core_->setStrategies(&gimbal_strategies_);
  }
}

/* ================================================================ */
/*  Joint state callback                                             */
/* ================================================================ */

void GimbalPipelineNode::jointStateCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  for (size_t i = 0; i < msg->name.size(); ++i) {
    if (msg->name[i] == "yaw_joint")
      current_yaw_ = msg->position[i];
    else if (msg->name[i] == "pitch_joint")
      current_pitch_ = msg->position[i];
  }
}

void GimbalPipelineNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (msg->k[0] < 1e-6 || msg->k[4] < 1e-6 || msg->width == 0 || msg->height == 0) {
    return;  // 无效的相机内参
  }
  double fx = msg->k[0];
  double fy = msg->k[4];
  double fov_half_yaw = std::atan(static_cast<double>(msg->width) / (2.0 * fx));
  double fov_half_pitch = std::atan(static_cast<double>(msg->height) / (2.0 * fy));

  // 通过核心类透传 FOV 更新，避免 node 直接耦合具体策略实现。
  if (gimbal_control_core_) {
    gimbal_control_core_->updateFov(fov_half_yaw, fov_half_pitch);
  }

  RCLCPP_INFO_ONCE(get_logger(),
      "[FOV] camera_info received: fov_yaw=%.1f° fov_pitch=%.1f° (fx=%.1f fy=%.1f %dx%d)",
      fov_half_yaw * 2.0 * 180.0 / M_PI, fov_half_pitch * 2.0 * 180.0 / M_PI,
      fx, fy, msg->width, msg->height);
}

void GimbalPipelineNode::updateGimbalState() {
  try {
    auto gimbal_tf = tf2_buffer_->lookupTransform(target_frame_,
                                                   "gimbal_link",
                                                   tf2::TimePointZero);
    auto msg_q = gimbal_tf.transform.rotation;
    tf2::Quaternion tf_q;
    tf2::fromMsg(msg_q, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    current_yaw_ = yaw;
    current_pitch_ = -pitch;
  } catch (const tf2::TransformException &) {
    // fall through — use joint_states values
  }
}

void GimbalPipelineNode::buildControlContextFromCache(
    gimbal_controller::GimbalControlContext & context,
    std::string & selected_id) {
  context.is_tracking = false;
  context.is_temp_lost = false;
  context.is_maneuvering = false;

  // Read shared state (thread-safe)
  rm_interfaces::msg::TrackedRobots::SharedPtr robots;
  rclcpp::Time data_update_time{0, 0, RCL_ROS_TIME};
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    robots = latest_tracked_robots_;
    selected_id = latest_selected_target_id_;
    data_update_time = latest_update_time_;
  }

  if (!robots || robots->robots.empty()) {
    return;
  }

  // Cache freshness check: stale target cache should not drive control.
  const double data_age = (context.current_time - data_update_time).seconds();
  const double max_data_age = std::max(tracker_timeout_s_, 1e-3);
  if (data_age > max_data_age) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Stale tracking data (age=%.3fs > %.3fs), ignoring",
      data_age, max_data_age);
    return;
  }

  const rm_interfaces::msg::TrackedRobot * selected_robot = nullptr;
  if (!selected_id.empty()) {
    for (const auto & robot : robots->robots) {
      if (robot.robot_id == selected_id) {
        selected_robot = &robot;
        break;
      }
    }
  } else {
    selected_robot = &robots->robots[0];
    selected_id = selected_robot->robot_id;
  }

  if (!selected_robot) {
    return;
  }

  context.target_robot = *selected_robot;
  context.target_stamp = data_update_time;
  context.is_tracking =
      (selected_robot->track_state == rm_interfaces::msg::TrackedRobot::TRACKING);
  context.is_temp_lost =
      (selected_robot->track_state == rm_interfaces::msg::TrackedRobot::TEMP_LOST);

  auto * tracker = tracker_manager_->get(selected_robot->robot_id);
  context.is_maneuvering = (tracker && tracker->is_initialized()) ?
    tracker->assess_maneuver().is_maneuvering : false;
}

void GimbalPipelineNode::publishDelayAuditDebug(
    const gimbal_controller::GimbalControlContext & context,
    const gimbal_controller::DelayAuditSnapshot & audit,
    const std::string & strategy_name) {
  if (!debug_delay_audit_pub_) {
    return;
  }

  rm_interfaces::msg::DelayAudit msg;
  msg.header.stamp = context.current_time;
  msg.header.frame_id = target_frame_;
  msg.strategy_name = audit.strategy_name.empty() ? strategy_name : audit.strategy_name;
  msg.valid = audit.valid;
  msg.tracking = audit.tracking;
  msg.processing_delay_s = audit.processing_delay_s;
  msg.prediction_extra_s = audit.prediction_extra_s;
  msg.flight_time_s = audit.flight_time_s;
  msg.total_prediction_time_s = audit.total_prediction_time_s;
  msg.control_latency_s = audit.control_latency_s;
  msg.fire_control_compensation_s = audit.fire_control_compensation_s;
  msg.control_delay_steps = audit.control_delay_steps;
  msg.uses_delayed_b = audit.uses_delayed_b;
  msg.double_compensation_risk = audit.double_compensation_risk;
  debug_delay_audit_pub_->publish(msg);
}

void GimbalPipelineNode::publishFireAdviceDebug(
    const gimbal_controller::GimbalControlContext & context,
    const rm_interfaces::msg::GimbalCmd & cmd,
    const gimbal_controller::FireAdviceDebugSnapshot & snapshot) {
  if (!debug_fire_advice_pub_) {
    return;
  }

  rm_interfaces::msg::FireAdviceDebug msg;
  msg.header.stamp = context.current_time;
  msg.header.frame_id = target_frame_;
  msg.target_id = snapshot.target_id.empty() ? cmd.target_id : snapshot.target_id;
  msg.mode = snapshot.mode;
  msg.track_state = snapshot.track_state;
  msg.evaluated = snapshot.evaluated;
  msg.valid = snapshot.valid;
  msg.fire_advice = snapshot.fire_advice;
  msg.best_candidate_index = snapshot.best_candidate_index;
  msg.yaw_error = snapshot.yaw_error;
  msg.pitch_error = snapshot.pitch_error;
  msg.best_candidate_facing_ok = snapshot.best_candidate_facing_ok;
  msg.candidate_count_total = snapshot.candidate_count_total;
  msg.candidate_count_facing_eligible = snapshot.candidate_count_facing_eligible;
  msg.candidate_count_facing_rejected = snapshot.candidate_count_facing_rejected;
  msg.probability_enabled = snapshot.probability_enabled;
  msg.p_hit_window = snapshot.p_hit_window;
  msg.fire_score = snapshot.fire_score;
  msg.best_tau_ms = snapshot.best_tau_ms;
  msg.e_u = snapshot.e_u;
  msg.e_v = snapshot.e_v;
  msg.sigma_u = snapshot.sigma_u;
  msg.sigma_v = snapshot.sigma_v;
  msg.armor_width_m = snapshot.armor_width_m;
  msg.armor_height_m = snapshot.armor_height_m;
  msg.burst_probability = snapshot.burst_probability;
  msg.log_evidence = snapshot.log_evidence;
  msg.evidence_sum = snapshot.evidence_sum;
  msg.evidence_strength = snapshot.evidence_strength;
  msg.gate_strategy = snapshot.gate_strategy;
  msg.gate_state = snapshot.gate_state;

  debug_fire_advice_pub_->publish(msg);

  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 500,
    "[FireAdvice] target=%s mode=%d track_state=%u evaluated=%d valid=%d fire=%d "
    "best=%d yaw_err=%.4fdeg pitch_err=%.4fdeg facing_ok=%d rejected=%d/%d eligible=%d",
    msg.target_id.empty() ? "none" : msg.target_id.c_str(),
    static_cast<int>(msg.mode),
    static_cast<unsigned>(msg.track_state),
    msg.evaluated ? 1 : 0,
    msg.valid ? 1 : 0,
    msg.fire_advice ? 1 : 0,
    msg.best_candidate_index,
    msg.yaw_error * 180.0 / M_PI,
    msg.pitch_error * 180.0 / M_PI,
    msg.best_candidate_facing_ok ? 1 : 0,
    msg.candidate_count_facing_rejected,
    msg.candidate_count_total,
    msg.candidate_count_facing_eligible);
}

/* ================================================================ */
/*  Timer callback — 250 Hz control loop                             */
/* ================================================================ */

void GimbalPipelineNode::timerCallback() {
  // Step 0: 核心类可用性检查
  if (!gimbal_control_core_) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "GimbalControlCore is not initialized, skipping control cycle");
    return;
  }

  gimbal_controller::GimbalControlContext context;
  context.current_time = now();

  // Step 1: 控制禁用时发布 idle 命令并早返回
  if (!enable_) {
    const auto idle_result = gimbal_control_core_->compute(
      context, current_gimbal_strategy_name_, std::string(), false);
    gimbal_cmd_pub_->publish(idle_result.cmd);
    return;
  }

  // Step 2: 更新云台姿态并填充控制上下文基础字段
  updateGimbalState();
  context.current_yaw = current_yaw_;
  context.current_pitch = current_pitch_;
  context.bullet_speed = bullet_speed_;

  // Step 3: 从共享缓存构建目标上下文
  std::string selected_id;
  buildControlContextFromCache(context, selected_id);

  // Step 4: 核心类统一生成命令（strategy + finalize + filter + audit）
  const auto control_result = gimbal_control_core_->compute(
    context, current_gimbal_strategy_name_, selected_id, true);
  if (!control_result.strategy_found) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Gimbal strategy '%s' not found, fallback to idle cmd",
      current_gimbal_strategy_name_.c_str());
  }

  // Step 5: 发布控制命令
  gimbal_cmd_pub_->publish(control_result.cmd);

  // Step 6: 发布调试信息（audit + marker）
  if (debug_mode_ && debug_delay_audit_pub_) {
    publishDelayAuditDebug(
      context,
      control_result.delay_audit,
      current_gimbal_strategy_name_);
  }

  if (debug_mode_ && debug_fire_advice_pub_) {
    publishFireAdviceDebug(
      context,
      control_result.cmd,
      control_result.fire_advice_debug);
  }

  if (debug_mode_ && debug_armor_selection_pub_ && control_result.has_tracking) {
    publishArmorSelectionDebug(context, current_gimbal_strategy_name_);
  }

  if (debug_mode_ && control_result.has_tracking) {
    publishGimbalMarkers(context.target_robot, control_result.cmd, control_result.fire_advice_debug);
    publishFireProbabilityDebugImages(context.target_robot.header, control_result.fire_advice_debug);
  }
}

/* ================================================================ */
/*  Service callback                                                 */
/* ================================================================ */

void GimbalPipelineNode::setModeCallback(
    const std::shared_ptr<rm_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<rm_interfaces::srv::SetMode::Response> response) {
  response->success = true;
  const int mode = request->mode;
  if (mode >= 0 && mode <= 5) {
    enable_ = true;
    refreshExternalTargetAllowlist(mode);
    if (buff_target_adapter_) {
      const bool enable_buff =
        external_targets_enable_ && external_targets_buff_enable_;
      buff_target_adapter_->setEnabled(enable_buff);
    }
    RCLCPP_INFO(get_logger(), "GimbalPipeline enabled (mode=%d)", mode);
  } else {
    enable_ = false;
    if (buff_target_adapter_) {
      buff_target_adapter_->setEnabled(false);
    }
    RCLCPP_INFO(get_logger(), "GimbalPipeline disabled (mode=%d)", mode);
  }
}

gimbal_controller::GimbalControlStrategy::SharedPtr
GimbalPipelineNode::getGimbalStrategy(const std::string &name) const {
  auto it = gimbal_strategies_.find(name);
  return (it != gimbal_strategies_.end()) ? it->second : nullptr;
}

void GimbalPipelineNode::publish2DTrackerDebugImage(
    const std_msgs::msg::Header &header,
    const std::vector<TrackerManager::TrackerConstView> &tracker_views) {
  if (!debug_tracker_2d_image_pub_) return;

  cv::Mat canvas(
      tracker_2d_image_debug_height_, tracker_2d_image_debug_width_, CV_8UC3,
      cv::Scalar(20, 20, 20));

  int draw_count = 0;
  for (const auto &view : tracker_views) {
    if (!view.tracker) continue;
    const evidence::ArmorEvidenceFrame *frame = nullptr;

    if (const auto *norm4 = dynamic_cast<const Norm4ArmorTracker *>(view.tracker)) {
      frame = &norm4->last_evidence_frame();
    } else if (const auto *norm4v2 = dynamic_cast<const Norm4ArmorTrackerV2 *>(view.tracker)) {
      frame = &norm4v2->last_evidence_frame();
    }
    if (!frame) continue;

    const auto &f = *frame;
    if (f.observations.empty()) continue;

    for (const auto &obs : f.observations) {
      if (!obs.image.has_value() || !obs.image->valid) continue;
      const auto &img = obs.image.value();

      int track_id = obs.track2d_id.value_or(-1);
      int color_seed = (track_id >= 0 ? track_id : draw_count);
      cv::Scalar color(
          50 + (color_seed * 71) % 205,
          50 + (color_seed * 131) % 205,
          50 + (color_seed * 193) % 205);

      const int x = std::max(0, static_cast<int>(std::lround(img.bbox_x)));
      const int y = std::max(0, static_cast<int>(std::lround(img.bbox_y)));
      const int w = std::max(1, static_cast<int>(std::lround(img.bbox_w)));
      const int h = std::max(1, static_cast<int>(std::lround(img.bbox_h)));
      cv::rectangle(canvas, cv::Rect(x, y, w, h), color, 2);

      bool has_corners = true;
      for (const auto &c : img.corners) {
        if (!std::isfinite(c.x()) || !std::isfinite(c.y())) {
          has_corners = false;
          break;
        }
      }
      if (has_corners) {
        std::vector<cv::Point> poly;
        poly.reserve(4);
        for (const auto &c : img.corners) {
          poly.emplace_back(
              static_cast<int>(std::lround(c.x())),
              static_cast<int>(std::lround(c.y())));
        }
        const cv::Point *pts = poly.data();
        int npts = static_cast<int>(poly.size());
        cv::polylines(canvas, &pts, &npts, 1, true, color, 1, cv::LINE_AA);
      }

      std::ostringstream oss;
      oss << view.robot_id << " t2d=" << track_id;
      cv::putText(canvas, oss.str(),
                  cv::Point(x, std::max(16, y - 5)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
      ++draw_count;
    }
  }

  cv::putText(
      canvas,
      "2DTracker tracks: " + std::to_string(draw_count),
      cv::Point(10, 22),
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 220, 255), 1, cv::LINE_AA);

  std::vector<uchar> encoded;
  std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY,
                                    tracker_2d_image_debug_jpeg_quality_};
  if (!cv::imencode(".jpg", canvas, encoded, encode_params)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Failed to encode 2D tracker debug image");
    return;
  }

  sensor_msgs::msg::CompressedImage out;
  out.header = header;
  out.format = "jpeg";
  out.data = std::move(encoded);
  debug_tracker_2d_image_pub_->publish(out);
}

void GimbalPipelineNode::publishEvidenceFrameDebug(
    const std_msgs::msg::Header &header,
    const std::vector<TrackerManager::TrackerConstView> &tracker_views) {
  if (!debug_evidence_frame_pub_) return;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  oss << "stamp=" << rclcpp::Time(header.stamp).seconds();

  int norm4_count = 0;
  for (const auto &view : tracker_views) {
    if (!view.tracker) continue;
    const evidence::ArmorEvidenceFrame *frame = nullptr;
    std::string rid = view.robot_id;

    if (const auto *norm4 = dynamic_cast<const Norm4ArmorTracker *>(view.tracker)) {
      frame = &norm4->last_evidence_frame();
    } else if (const auto *norm4v2 = dynamic_cast<const Norm4ArmorTrackerV2 *>(view.tracker)) {
      frame = &norm4v2->last_evidence_frame();
    }
    if (!frame) continue;

    ++norm4_count;
    const auto &f = *frame;
    oss << "\nrobot_id=" << view.robot_id
        << " ts=" << f.timestamp
        << " obs=" << f.obs_count
        << " obs_vec=" << f.observations.size()
        << " t2d=" << f.track2d_evidence.size()
        << " proxy=" << f.proxy_evidence.size()
        << " comp={3d:" << (f.completeness.has_3d_obs ? 1 : 0)
        << ",2d:" << (f.completeness.has_2d_tracks ? 1 : 0)
        << ",proxy:" << (f.completeness.has_proxy ? 1 : 0)
        << ",geo:" << (f.completeness.has_geometry ? 1 : 0)
        << ",rel:" << (f.completeness.has_relation ? 1 : 0)
        << ",ratio:" << f.completeness.fraction() << "}"
        << " relation={valid:" << (f.relation.valid ? 1 : 0)
        << ",has_z_jump:" << (f.relation.has_z_jump ? 1 : 0)
        << ",z_jump:" << f.relation.z_jump
        << ",yaw_delta:" << f.relation.yaw_delta
        << ",spatial:" << f.relation.spatial_consistency
        << ",dual:" << (f.relation.has_dual_obs ? 1 : 0)
        << ",p1:" << f.relation.dual_panel_id_1
        << ",p2:" << f.relation.dual_panel_id_2
        << "}";
  }

  if (norm4_count == 0) {
    oss << "\nno_norm4_tracker";
  }

  std_msgs::msg::String out;
  out.data = oss.str();
  debug_evidence_frame_pub_->publish(out);
}

void GimbalPipelineNode::logNorm4V3TrackerDebug(
    const std::vector<TrackerManager::TrackerConstView> &tracker_views) {
  const auto &dbg_cfg = tracker_config_.norm4_v3.debug_log;
  if (!dbg_cfg.enable) return;

  std::ostringstream oss;
  bool has_norm4v3 = false;
  for (const auto &view : tracker_views) {
    if (!view.tracker) continue;
    const auto *norm4v3 = dynamic_cast<const Norm4ArmorTrackerV2 *>(view.tracker);
    if (!norm4v3) continue;

    has_norm4v3 = true;
    const auto &h = norm4v3->last_hypothesis_debug();
    const auto &s = norm4v3->debug_snapshot();

    oss << " [" << view.robot_id
        << " committed=" << (h.committed ? 1 : 0)
        << " panel=" << s.current_panel_id
        << " cand=" << s.candidate_panel_id
        << " conf=" << h.top1_confidence
        << " margin=" << h.top1_top2_margin;
    if (dbg_cfg.verbose) {
      oss << " mode=" << static_cast<int>(norm4v3->current_mode())
          << " top1_nis=" << s.top1_nis
          << " degraded=" << (h.degraded ? 1 : 0);
    }
    oss << " reason=" << h.decision_reason << "]";
  }

  if (!has_norm4v3) return;
  RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), dbg_cfg.throttle_ms,
      "norm4_v3_debug:%s", oss.str().c_str());
}

/* ================================================================ */
/*  Visualization                                                    */
/* ================================================================ */

void GimbalPipelineNode::initMarkers() {
  position_marker_.ns = "target_position";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y =
      position_marker_.scale.z = 0.15;
  position_marker_.color.a = 1.0;
  position_marker_.color.r = 1.0;

  target_velocity_marker_.type = visualization_msgs::msg::Marker::ARROW;
  target_velocity_marker_.ns = "target_velocity";
  target_velocity_marker_.scale.x = 0.03;
  target_velocity_marker_.scale.y = 0.05;
  target_velocity_marker_.color.a = 1.0;
  target_velocity_marker_.color.g = 1.0;
  target_velocity_marker_.color.b = 1.0;

  armors_marker_.ns = "armors";
  armors_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armors_marker_.scale.x = 0.03;
  armors_marker_.scale.y = 0.23;
  armors_marker_.scale.z = 0.125;
  armors_marker_.color.a = 0.7;
  armors_marker_.color.g = 0.5;
  armors_marker_.color.b = 1.0;

  selection_marker_.ns = "selection";
  selection_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  selection_marker_.scale.x = selection_marker_.scale.y =
      selection_marker_.scale.z = 0.12;
  selection_marker_.color.a = 1.0;
  selection_marker_.color.r = 1.0;
  selection_marker_.color.g = 1.0;

  predicted_marker_.ns = "predicted_hit";
  predicted_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  predicted_marker_.scale.x = predicted_marker_.scale.y =
      predicted_marker_.scale.z = 0.1;
  predicted_marker_.color.a = 1.0;
  predicted_marker_.color.g = 1.0;

  trajectory_marker_.ns = "trajectory";
  trajectory_marker_.type = visualization_msgs::msg::Marker::LINE_STRIP;
  trajectory_marker_.scale.x = 0.02;
  trajectory_marker_.color.a = 0.8;
  trajectory_marker_.color.r = 1.0;
  trajectory_marker_.color.g = 0.75;
  trajectory_marker_.color.b = 0.79;

  radial_allowed_arc_marker_.ns = "radial_allowed_arc";
  radial_allowed_arc_marker_.type = visualization_msgs::msg::Marker::LINE_STRIP;
  radial_allowed_arc_marker_.scale.x = 0.018;
  radial_allowed_arc_marker_.color.a = 0.95;
  radial_allowed_arc_marker_.color.r = 1.0;
  radial_allowed_arc_marker_.color.g = 0.6;
  radial_allowed_arc_marker_.color.b = 0.0;

  radial_allowed_bounds_marker_.ns = "radial_allowed_bounds";
  radial_allowed_bounds_marker_.type = visualization_msgs::msg::Marker::LINE_LIST;
  radial_allowed_bounds_marker_.scale.x = 0.012;
  radial_allowed_bounds_marker_.color.a = 0.95;
  radial_allowed_bounds_marker_.color.r = 1.0;
  radial_allowed_bounds_marker_.color.g = 0.85;
  radial_allowed_bounds_marker_.color.b = 0.2;

  virtual_armor_marker_.ns = "virtual_armor";
  virtual_armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  virtual_armor_marker_.scale.x = 0.03;
  virtual_armor_marker_.scale.y = 0.23;
  virtual_armor_marker_.scale.z = 0.125;
  virtual_armor_marker_.color.a = 0.95;
  virtual_armor_marker_.color.r = 0.1;
  virtual_armor_marker_.color.g = 0.95;
  virtual_armor_marker_.color.b = 0.35;

  virtual_armor_text_marker_.ns = "virtual_armor_text";
  virtual_armor_text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  virtual_armor_text_marker_.scale.z = 0.12;
  virtual_armor_text_marker_.color.a = 1.0;
  virtual_armor_text_marker_.color.r = 0.1;
  virtual_armor_text_marker_.color.g = 1.0;
  virtual_armor_text_marker_.color.b = 0.6;

  color_palette_.clear();
  for (int i = 0; i < 10; ++i) {
    float hue = i * 36.0f;
    color_palette_.push_back(hsvToRgb(hue, 1.0f, 1.0f));
  }
}

void GimbalPipelineNode::publishGimbalMarkers(
    const rm_interfaces::msg::TrackedRobot &target_robot,
    const rm_interfaces::msg::GimbalCmd &cmd,
    const gimbal_controller::FireAdviceDebugSnapshot & fire_snapshot) {
  if (!debug_gimbal_marker_pub_) return;

  const auto normalized_target = robot_description::TrackedRobotUsage::normalizeState(target_robot);
  const auto center_position = robot_description::TrackedRobotUsage::centerPosition(normalized_target);
  const auto linear_velocity = robot_description::TrackedRobotUsage::linearVelocity(normalized_target);
  const double target_yaw = robot_description::TrackedRobotUsage::yaw(normalized_target);
  const double target_yaw_velocity =
      robot_description::TrackedRobotUsage::yawVelocity(normalized_target);

  visualization_msgs::msg::MarkerArray marker_array;
  const bool has_valid_measurement =
    cmd.mode == rm_interfaces::msg::GimbalCmd::MODE_NORMAL_MEASUREMENT &&
    cmd.distance > 0.0;

  // Position
  position_marker_.header = target_robot.header;
  position_marker_.id = 0;
  position_marker_.action = visualization_msgs::msg::Marker::ADD;
  position_marker_.pose.position =
    robot_description::TrackedRobotUsage::toPoint(center_position);
  position_marker_.pose.orientation.w = 1.0;
  marker_array.markers.push_back(position_marker_);

  // Velocity arrow
  target_velocity_marker_.header = target_robot.header;
  target_velocity_marker_.id = 0;
  target_velocity_marker_.action = visualization_msgs::msg::Marker::ADD;
  target_velocity_marker_.points.clear();
  geometry_msgs::msg::Point vel_start =
    robot_description::TrackedRobotUsage::toPoint(center_position);
  geometry_msgs::msg::Point vel_end = vel_start;
  vel_end.x += linear_velocity.x() * 0.5;
  vel_end.y += linear_velocity.y() * 0.5;
  vel_end.z += linear_velocity.z() * 0.5;
  target_velocity_marker_.points.push_back(vel_start);
  target_velocity_marker_.points.push_back(vel_end);
  marker_array.markers.push_back(target_velocity_marker_);

  // Armor plates
  if (!normalized_target.armors_offset.empty()) {
    for (size_t i = 0; i < normalized_target.armors_offset.size(); ++i) {
      auto armor_marker = armors_marker_;
      armor_marker.header = target_robot.header;
      armor_marker.id = static_cast<int>(i);
      armor_marker.action = visualization_msgs::msg::Marker::ADD;
      double cos_yaw = std::cos(target_yaw);
      double sin_yaw = std::sin(target_yaw);
      const auto &offset = normalized_target.armors_offset[i];
      armor_marker.pose.position.x =
          center_position.x() +
          offset.position.x * cos_yaw - offset.position.y * sin_yaw;
      armor_marker.pose.position.y =
          center_position.y() +
          offset.position.x * sin_yaw + offset.position.y * cos_yaw;
      armor_marker.pose.position.z =
          center_position.z() + offset.position.z;
      tf2::Quaternion q_offset;
      tf2::fromMsg(offset.orientation, q_offset);
      if (q_offset.length2() <= 1e-12) {
        q_offset.setRPY(0.0, 0.0, 0.0);
      } else {
        q_offset.normalize();
      }
      tf2::Quaternion q_world_yaw;
      q_world_yaw.setRPY(0.0, 0.0, target_yaw);
      const tf2::Quaternion q_world_armor = q_world_yaw * q_offset;
      armor_marker.pose.orientation = tf2::toMsg(q_world_armor);
      marker_array.markers.push_back(armor_marker);
    }
  }

  // Allowed radial selection range marker (for min_movement_with_radial)
  if (radial_selection_enabled_ && armor_selector_ && normalized_target.num_armors > 0) {
    const double center_x = center_position.x();
    const double center_y = center_position.y();
    const double center_z = center_position.z();

    // Direction from robot center to our gimbal origin (world origin approximation).
    double axis_yaw = std::atan2(-center_y, -center_x);

    double enter_deg = facing_enter_angle_deg_;
    double speed_norm = 0.0;
    double bias_deg = 0.0;
    if (radial_dynamic_enable_) {
      speed_norm = std::clamp(
        std::abs(target_yaw_velocity) / radial_dynamic_v_yaw_ref_, 0.0, 1.0);
      const double scale = 1.0 - radial_dynamic_shrink_ratio_ * speed_norm;
      enter_deg = std::max(enter_deg * scale, radial_dynamic_min_angle_deg_);
      const double bias_mag = std::min(
        radial_dynamic_bias_gain_deg_ * speed_norm,
        radial_dynamic_max_bias_deg_);
      bias_deg = (target_yaw_velocity >= 0.0 ? 1.0 : -1.0) * bias_mag;
      axis_yaw += bias_deg * M_PI / 180.0;
    }
    const double enter_rad = enter_deg * M_PI / 180.0;

    double radius = 0.25;
    for (const auto & offset : normalized_target.armors_offset) {
      const double r = std::hypot(offset.position.x, offset.position.y);
      if (r > radius) {
        radius = r;
      }
    }
    radius = std::clamp(radius * 1.2, 0.2, 0.8);

    radial_allowed_arc_marker_.header = target_robot.header;
    radial_allowed_arc_marker_.id = 100;
    radial_allowed_arc_marker_.action = visualization_msgs::msg::Marker::ADD;
    radial_allowed_arc_marker_.points.clear();

    constexpr int kArcSamples = 48;
    for (int i = 0; i <= kArcSamples; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(kArcSamples);
      const double yaw = axis_yaw - enter_rad + 2.0 * enter_rad * t;
      geometry_msgs::msg::Point p;
      p.x = center_x + radius * std::cos(yaw);
      p.y = center_y + radius * std::sin(yaw);
      p.z = center_z + 0.08;
      radial_allowed_arc_marker_.points.push_back(p);
    }
    marker_array.markers.push_back(radial_allowed_arc_marker_);

    radial_allowed_bounds_marker_.header = target_robot.header;
    radial_allowed_bounds_marker_.id = 101;
    radial_allowed_bounds_marker_.action = visualization_msgs::msg::Marker::ADD;
    radial_allowed_bounds_marker_.points.clear();

    geometry_msgs::msg::Point c;
    c.x = center_x;
    c.y = center_y;
    c.z = center_z + 0.08;

    geometry_msgs::msg::Point p_min;
    p_min.x = center_x + radius * std::cos(axis_yaw - enter_rad);
    p_min.y = center_y + radius * std::sin(axis_yaw - enter_rad);
    p_min.z = center_z + 0.08;

    geometry_msgs::msg::Point p_max;
    p_max.x = center_x + radius * std::cos(axis_yaw + enter_rad);
    p_max.y = center_y + radius * std::sin(axis_yaw + enter_rad);
    p_max.z = center_z + 0.08;

    geometry_msgs::msg::Point p_axis;
    p_axis.x = center_x + radius * std::cos(axis_yaw);
    p_axis.y = center_y + radius * std::sin(axis_yaw);
    p_axis.z = center_z + 0.08;

    radial_allowed_bounds_marker_.points.push_back(c);
    radial_allowed_bounds_marker_.points.push_back(p_min);
    radial_allowed_bounds_marker_.points.push_back(c);
    radial_allowed_bounds_marker_.points.push_back(p_max);
    radial_allowed_bounds_marker_.points.push_back(c);
    radial_allowed_bounds_marker_.points.push_back(p_axis);
    marker_array.markers.push_back(radial_allowed_bounds_marker_);
  }

  // Virtual armor marker (only when auto-switch virtual mode is enabled and active)
  {
    virtual_armor_marker_.header = target_robot.header;
    virtual_armor_marker_.id = 0;
    virtual_armor_marker_.action = visualization_msgs::msg::Marker::DELETE;
    virtual_armor_text_marker_.header = target_robot.header;
    virtual_armor_text_marker_.id = 0;
    virtual_armor_text_marker_.action = visualization_msgs::msg::Marker::DELETE;

    if (virtual_auto_switch_enable_ && armor_selector_ && position_calculator_) {
      auto armor_positions = position_calculator_->calculate(normalized_target);
      if (!armor_positions.empty()) {
        // Use a local copy to avoid mutating runtime selector state during debug visualization.
        auto debug_selector = *armor_selector_;
        auto virtual_selection = debug_selector.selectBest(
          armor_positions,
          center_position,
          target_yaw,
          normalized_target.num_armors,
          target_yaw_velocity,
          current_yaw_,
          current_pitch_);

        if (virtual_selection.is_virtual_target && !virtual_selection.is_center_fallback) {
          virtual_armor_marker_.action = visualization_msgs::msg::Marker::ADD;
          virtual_armor_marker_.pose.position =
            robot_description::TrackedRobotUsage::toPoint(virtual_selection.position);

          const double normal_yaw =
            std::atan2(-center_position.y(), -center_position.x());
          tf2::Quaternion q_virtual;
          q_virtual.setRPY(0.0, -0.2618, normal_yaw);
          virtual_armor_marker_.pose.orientation.x = q_virtual.x();
          virtual_armor_marker_.pose.orientation.y = q_virtual.y();
          virtual_armor_marker_.pose.orientation.z = q_virtual.z();
          virtual_armor_marker_.pose.orientation.w = q_virtual.w();

          virtual_armor_text_marker_.action = visualization_msgs::msg::Marker::ADD;
          virtual_armor_text_marker_.pose.position = virtual_armor_marker_.pose.position;
          virtual_armor_text_marker_.pose.position.z += 0.18;
          virtual_armor_text_marker_.pose.orientation.w = 1.0;
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(3)
              << "vidx=" << virtual_selection.real_selected_index
              << " dYaw=" << virtual_selection.virtual_delta_yaw;
          virtual_armor_text_marker_.text = oss.str();
        }
      }
    }
    marker_array.markers.push_back(virtual_armor_marker_);
    marker_array.markers.push_back(virtual_armor_text_marker_);
  }

  // Selection target
  if (has_valid_measurement) {
    selection_marker_.header = target_robot.header;
    selection_marker_.id = 0;
    selection_marker_.action = visualization_msgs::msg::Marker::ADD;
    double yaw_rad = cmd.yaw * M_PI / 180.0;
    double pitch_rad = cmd.pitch * M_PI / 180.0;
    selection_marker_.pose.position.x =
        cmd.distance * std::cos(pitch_rad) * std::cos(yaw_rad);
    selection_marker_.pose.position.y =
        cmd.distance * std::cos(pitch_rad) * std::sin(yaw_rad);
    selection_marker_.pose.position.z =
        cmd.distance * std::sin(pitch_rad);
    selection_marker_.pose.orientation.w = 1.0;
    marker_array.markers.push_back(selection_marker_);
  }

  // Predicted hit (for predicted strategy)
  if (current_gimbal_strategy_name_ == "predicted" && has_valid_measurement) {
    predicted_marker_.header = target_robot.header;
    predicted_marker_.id = 0;
    predicted_marker_.action = visualization_msgs::msg::Marker::ADD;
    predicted_marker_.pose = selection_marker_.pose;
    predicted_marker_.color.a = 0.6;
    marker_array.markers.push_back(predicted_marker_);
  }

  // Trajectory
  if (has_valid_measurement) {
    trajectory_marker_.header = target_robot.header;
    trajectory_marker_.id = 0;
    trajectory_marker_.action = visualization_msgs::msg::Marker::ADD;
    trajectory_marker_.points.clear();
    int num_points = 20;
    double yaw_rad = cmd.yaw * M_PI / 180.0;
    double pitch_rad = cmd.pitch * M_PI / 180.0;
    for (int i = 0; i <= num_points; ++i) {
      double t = static_cast<double>(i) / num_points;
      double distance = cmd.distance * t;
      geometry_msgs::msg::Point p;
      p.x = distance * std::cos(pitch_rad) * std::cos(yaw_rad);
      p.y = distance * std::cos(pitch_rad) * std::sin(yaw_rad);
      double flight_time = cmd.distance / bullet_speed_ * t;
      p.z = distance * std::sin(pitch_rad) -
            0.5 * 9.8 * flight_time * flight_time;
      trajectory_marker_.points.push_back(p);
    }
    if (cmd.fire_advice) {
      trajectory_marker_.color.r = 0.0;
      trajectory_marker_.color.g = 1.0;
      trajectory_marker_.color.b = 0.0;
    } else {
      trajectory_marker_.color.r = 1.0;
      trajectory_marker_.color.g = 0.75;
      trajectory_marker_.color.b = 0.79;
    }
    marker_array.markers.push_back(trajectory_marker_);
  }

  if (fire_prob_vis_enable_) {
    publishFireProbabilityMarkers(target_robot.header, fire_snapshot, marker_array);
  }

  debug_gimbal_marker_pub_->publish(marker_array);
}

void GimbalPipelineNode::publishFireProbabilityMarkers(
  const std_msgs::msg::Header & header,
  const gimbal_controller::FireAdviceDebugSnapshot & fire_snapshot,
  visualization_msgs::msg::MarkerArray & marker_array)
{
  if (!fire_snapshot.probability_enabled || fire_snapshot.tau_samples.empty()) {
    const std::vector<std::pair<std::string, int>> stale_markers = {
      {"fire_prob/tau_candidates", 0},
      {"fire_prob/error_ellipse_1sigma", 0},
      {"fire_prob/error_ellipse_2sigma", 0},
      {"fire_prob/trajectory_mean", 0},
      {"fire_prob/impact_cloud", 0},
      {"fire_prob/armor_plane", 0},
      {"fire_prob/mean_error_point", 0},
      {"fire_prob/text", 0}
    };
    for (const auto & [ns, id] : stale_markers) {
      visualization_msgs::msg::Marker del;
      del.header = header;
      del.ns = ns;
      del.id = id;
      del.action = visualization_msgs::msg::Marker::DELETE;
      marker_array.markers.push_back(del);
    }
    return;
  }

  visualization_msgs::msg::Marker traj;
  traj.header = header;
  traj.ns = "fire_prob/trajectory_mean";
  traj.id = 0;
  traj.type = visualization_msgs::msg::Marker::LINE_STRIP;
  traj.action = visualization_msgs::msg::Marker::ADD;
  traj.scale.x = 0.01;
  traj.color.a = 0.95;
  traj.color.r = 0.1;
  traj.color.g = 0.95;
  traj.color.b = 0.2;
  for (const auto & s : fire_snapshot.tau_samples) {
    geometry_msgs::msg::Point p;
    p.x = s.impact_x;
    p.y = s.impact_y;
    p.z = s.impact_z;
    traj.points.push_back(p);
  }
  marker_array.markers.push_back(traj);

  visualization_msgs::msg::Marker tau_pts;
  tau_pts.header = header;
  tau_pts.ns = "fire_prob/tau_candidates";
  tau_pts.id = 0;
  tau_pts.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  tau_pts.action = visualization_msgs::msg::Marker::ADD;
  tau_pts.scale.x = 0.03;
  tau_pts.scale.y = 0.03;
  tau_pts.scale.z = 0.03;
  tau_pts.color.a = 0.9;
  tau_pts.points.clear();
  tau_pts.colors.clear();
  for (const auto & s : fire_snapshot.tau_samples) {
    geometry_msgs::msg::Point p;
    p.x = s.impact_x;
    p.y = s.impact_y;
    p.z = s.impact_z;
    tau_pts.points.push_back(p);
    std_msgs::msg::ColorRGBA c;
    c.a = 0.9f;
    c.r = static_cast<float>(1.0 - s.p_hit);
    c.g = static_cast<float>(s.p_hit);
    c.b = 0.1f;
    tau_pts.colors.push_back(c);
  }
  marker_array.markers.push_back(tau_pts);

  visualization_msgs::msg::Marker armor_plane;
  armor_plane.header = header;
  armor_plane.ns = "fire_prob/armor_plane";
  armor_plane.id = 0;
  armor_plane.type = visualization_msgs::msg::Marker::LINE_LIST;
  armor_plane.action = visualization_msgs::msg::Marker::ADD;
  armor_plane.scale.x = 0.008;
  armor_plane.color.a = 0.9;
  armor_plane.color.r = 0.8;
  armor_plane.color.g = 0.95;
  armor_plane.color.b = 1.0;
  const double hw = std::max(fire_snapshot.armor_width_m, 1e-6) * 0.5;
  const double hh = std::max(fire_snapshot.armor_height_m, 1e-6) * 0.5;
  const Eigen::Vector3d c = fire_snapshot.armor_center;
  const Eigen::Vector3d r = fire_snapshot.armor_right;
  const Eigen::Vector3d u = fire_snapshot.armor_up;
  const Eigen::Vector3d p0 = c + r * hw + u * hh;
  const Eigen::Vector3d p1 = c - r * hw + u * hh;
  const Eigen::Vector3d p2 = c - r * hw - u * hh;
  const Eigen::Vector3d p3 = c + r * hw - u * hh;
  const std::array<Eigen::Vector3d, 4> ps = {p0, p1, p2, p3};
  for (int i = 0; i < 4; ++i) {
    geometry_msgs::msg::Point a;
    geometry_msgs::msg::Point b;
    a.x = ps[i].x();
    a.y = ps[i].y();
    a.z = ps[i].z();
    b.x = ps[(i + 1) % 4].x();
    b.y = ps[(i + 1) % 4].y();
    b.z = ps[(i + 1) % 4].z();
    armor_plane.points.push_back(a);
    armor_plane.points.push_back(b);
  }
  marker_array.markers.push_back(armor_plane);

  visualization_msgs::msg::Marker ell1;
  ell1.header = header;
  ell1.ns = "fire_prob/error_ellipse_1sigma";
  ell1.id = 0;
  ell1.type = visualization_msgs::msg::Marker::LINE_STRIP;
  ell1.action = visualization_msgs::msg::Marker::ADD;
  ell1.scale.x = 0.01;
  ell1.color.a = 0.95;
  ell1.color.r = 1.0;
  ell1.color.g = 0.9;
  ell1.color.b = 0.1;
  const int samples = std::max(16, fire_prob_vis_ellipse_samples_);
  for (int i = 0; i <= samples; ++i) {
    const double th = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(samples);
    const double du = fire_snapshot.sigma_u * std::cos(th);
    const double dv = fire_snapshot.sigma_v * std::sin(th);
    const Eigen::Vector3d p3 =
      fire_snapshot.armor_center + fire_snapshot.armor_right * du + fire_snapshot.armor_up * dv;
    geometry_msgs::msg::Point p;
    p.x = p3.x();
    p.y = p3.y();
    p.z = p3.z();
    ell1.points.push_back(p);
  }
  marker_array.markers.push_back(ell1);

  visualization_msgs::msg::Marker ell2 = ell1;
  ell2.ns = "fire_prob/error_ellipse_2sigma";
  ell2.id = 0;
  ell2.color.r = 1.0;
  ell2.color.g = 0.5;
  ell2.color.b = 0.1;
  ell2.points.clear();
  for (int i = 0; i <= samples; ++i) {
    const double th = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(samples);
    const double du = 2.0 * fire_snapshot.sigma_u * std::cos(th);
    const double dv = 2.0 * fire_snapshot.sigma_v * std::sin(th);
    const Eigen::Vector3d p3 =
      fire_snapshot.armor_center + fire_snapshot.armor_right * du + fire_snapshot.armor_up * dv;
    geometry_msgs::msg::Point p;
    p.x = p3.x();
    p.y = p3.y();
    p.z = p3.z();
    ell2.points.push_back(p);
  }
  marker_array.markers.push_back(ell2);

  visualization_msgs::msg::Marker mean_pt;
  mean_pt.header = header;
  mean_pt.ns = "fire_prob/mean_error_point";
  mean_pt.id = 0;
  mean_pt.type = visualization_msgs::msg::Marker::SPHERE;
  mean_pt.action = visualization_msgs::msg::Marker::ADD;
  mean_pt.scale.x = 0.04;
  mean_pt.scale.y = 0.04;
  mean_pt.scale.z = 0.04;
  mean_pt.color.a = 0.95;
  mean_pt.color.r = static_cast<float>(1.0 - fire_snapshot.p_hit_window);
  mean_pt.color.g = static_cast<float>(fire_snapshot.p_hit_window);
  mean_pt.color.b = 0.1f;
  const Eigen::Vector3d mean3 =
    fire_snapshot.armor_center +
    fire_snapshot.armor_right * fire_snapshot.e_u +
    fire_snapshot.armor_up * fire_snapshot.e_v;
  mean_pt.pose.position.x = mean3.x();
  mean_pt.pose.position.y = mean3.y();
  mean_pt.pose.position.z = mean3.z();
  mean_pt.pose.orientation.w = 1.0;
  marker_array.markers.push_back(mean_pt);

  visualization_msgs::msg::Marker cloud;
  cloud.header = header;
  cloud.ns = "fire_prob/impact_cloud";
  cloud.id = 0;
  cloud.type = visualization_msgs::msg::Marker::POINTS;
  cloud.action = visualization_msgs::msg::Marker::ADD;
  cloud.scale.x = 0.012;
  cloud.scale.y = 0.012;
  cloud.color.a = 0.25;
  cloud.color.r = 0.2;
  cloud.color.g = 0.9;
  cloud.color.b = 1.0;
  const int cloud_n = std::max(8, std::min(fire_prob_vis_max_impact_points_, 512));
  for (int i = 0; i < cloud_n; ++i) {
    const double t = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(cloud_n);
    const double rn = std::sqrt(static_cast<double>(i) / static_cast<double>(cloud_n));
    const double du = fire_snapshot.e_u + fire_snapshot.sigma_u * rn * std::cos(t);
    const double dv = fire_snapshot.e_v + fire_snapshot.sigma_v * rn * std::sin(t);
    const Eigen::Vector3d q =
      fire_snapshot.armor_center + fire_snapshot.armor_right * du + fire_snapshot.armor_up * dv;
    geometry_msgs::msg::Point p;
    p.x = q.x();
    p.y = q.y();
    p.z = q.z();
    cloud.points.push_back(p);
  }
  marker_array.markers.push_back(cloud);

  visualization_msgs::msg::Marker txt;
  txt.header = header;
  txt.ns = "fire_prob/text";
  txt.id = 0;
  txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  txt.action = visualization_msgs::msg::Marker::ADD;
  txt.scale.z = 0.10;
  txt.color.a = 1.0;
  txt.color.r = 0.95;
  txt.color.g = 0.95;
  txt.color.b = 0.95;
  txt.pose.position.x = fire_snapshot.armor_center.x();
  txt.pose.position.y = fire_snapshot.armor_center.y();
  txt.pose.position.z = fire_snapshot.armor_center.z() + 0.20;
  txt.pose.orientation.w = 1.0;
  std::ostringstream oss;
  oss << "Pwin=" << std::fixed << std::setprecision(2) << fire_snapshot.p_hit_window
      << " Score=" << fire_snapshot.fire_score;
  if (fire_snapshot.gate_strategy == 1) {
    oss << " Pb=" << std::setprecision(2) << fire_snapshot.burst_probability
        << " S=" << std::setprecision(2) << fire_snapshot.evidence_strength
        << " G=" << fire_snapshot.gate_state;
  }
  oss << " tau=" << std::setprecision(1) << fire_snapshot.best_tau_ms << "ms"
      << " eu=" << std::setprecision(3) << fire_snapshot.e_u << "m"
      << " ev=" << fire_snapshot.e_v << "m"
      << " su=" << fire_snapshot.sigma_u << "m"
      << " sv=" << fire_snapshot.sigma_v << "m"
      << " fire=" << (fire_snapshot.fire_advice ? 1 : 0);
  txt.text = oss.str();
  marker_array.markers.push_back(txt);
}

void GimbalPipelineNode::publishFireProbabilityDebugImages(
  const std_msgs::msg::Header & header,
  const gimbal_controller::FireAdviceDebugSnapshot & fire_snapshot)
{
  if (!fire_prob_image_debug_enable_ || !debug_fire_plane_image_pub_ || !debug_fire_normal_image_pub_) {
    return;
  }
  if (!fire_snapshot.probability_enabled || fire_snapshot.tau_samples.empty()) {
    return;
  }

  const rclcpp::Time stamp = header.stamp;
  const double min_period = 1.0 / std::max(fire_prob_image_debug_publish_rate_hz_, 0.1);
  if (last_fire_prob_image_pub_time_.nanoseconds() > 0 &&
    (stamp - last_fire_prob_image_pub_time_).seconds() < min_period)
  {
    return;
  }
  last_fire_prob_image_pub_time_ = stamp;

  const int w = fire_prob_image_debug_width_;
  const int h = fire_prob_image_debug_height_;
  cv::Mat plane(h, w, CV_8UC3, cv::Scalar(18, 18, 18));
  cv::Mat normal(h, w, CV_8UC3, cv::Scalar(18, 18, 18));

  const int margin = 40;
  const cv::Point2d center_plane(w * 0.45, h * 0.55);
  const double half_w = std::max(fire_snapshot.armor_width_m * 0.5, 1e-6);
  const double half_h = std::max(fire_snapshot.armor_height_m * 0.5, 1e-6);
  const double sx = (w * 0.35 - margin) / half_w;
  const double sy = (h * 0.35 - margin) / half_h;
  const double scale = std::min(sx, sy);

  const cv::Rect armor_rect(
    static_cast<int>(center_plane.x - half_w * scale),
    static_cast<int>(center_plane.y - half_h * scale),
    static_cast<int>(2.0 * half_w * scale),
    static_cast<int>(2.0 * half_h * scale));
  cv::rectangle(plane, armor_rect, cv::Scalar(220, 220, 220), 2);
  cv::line(plane, cv::Point(armor_rect.x, static_cast<int>(center_plane.y)),
    cv::Point(armor_rect.x + armor_rect.width, static_cast<int>(center_plane.y)), cv::Scalar(80, 80, 80), 1);
  cv::line(plane, cv::Point(static_cast<int>(center_plane.x), armor_rect.y),
    cv::Point(static_cast<int>(center_plane.x), armor_rect.y + armor_rect.height), cv::Scalar(80, 80, 80), 1);

  for (const auto & s : fire_snapshot.tau_samples) {
    const int px = static_cast<int>(center_plane.x + s.e_u * scale);
    const int py = static_cast<int>(center_plane.y - s.e_v * scale);
    const int g = static_cast<int>(255.0 * std::clamp(s.p_hit, 0.0, 1.0));
    const int r = 255 - g;
    cv::circle(plane, cv::Point(px, py), 3, cv::Scalar(30, g, r), -1);
  }

  if (fire_prob_image_debug_show_sigma_ellipse_) {
    const int a1 = std::max(1, static_cast<int>(std::abs(fire_snapshot.sigma_u) * scale));
    const int b1 = std::max(1, static_cast<int>(std::abs(fire_snapshot.sigma_v) * scale));
    cv::ellipse(plane, center_plane, cv::Size(a1, b1), 0.0, 0.0, 360.0, cv::Scalar(80, 200, 255), 2);
    cv::ellipse(plane, center_plane, cv::Size(2 * a1, 2 * b1), 0.0, 0.0, 360.0, cv::Scalar(80, 130, 255), 1);
  }

  const cv::Point best_pt(
    static_cast<int>(center_plane.x + fire_snapshot.e_u * scale),
    static_cast<int>(center_plane.y - fire_snapshot.e_v * scale));
  cv::circle(plane, best_pt, 6, cv::Scalar(0, 255, 255), 2);

  const gimbal_controller::fire_advice::TauDebugSample * best_s = &fire_snapshot.tau_samples.front();
  double min_tau_diff = std::numeric_limits<double>::max();
  const double best_tau_s = fire_snapshot.best_tau_ms * 1e-3;
  for (const auto & sample : fire_snapshot.tau_samples) {
    const double d = std::abs(sample.tau_s - best_tau_s);
    if (d < min_tau_diff) {
      min_tau_diff = d;
      best_s = &sample;
    }
  }
  const bool best_front_ok = best_s->front_ok;
  const bool best_gate_ok = best_s->normal_gate_pass;

  if (fire_prob_image_debug_show_text_) {
    std::ostringstream oss1;
    oss1 << "Pwin=" << std::fixed << std::setprecision(2) << fire_snapshot.p_hit_window
         << " Score=" << fire_snapshot.fire_score
         << " Tau=" << std::setprecision(1) << fire_snapshot.best_tau_ms << "ms"
         << " Fire=" << (fire_snapshot.fire_advice ? "Y" : "N");
    cv::putText(plane, oss1.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(230, 230, 230), 1);
    std::ostringstream oss2;
    oss2 << "eu=" << std::setprecision(3) << fire_snapshot.e_u
         << " ev=" << fire_snapshot.e_v
         << " su=" << fire_snapshot.sigma_u
         << " sv=" << fire_snapshot.sigma_v;
    cv::putText(plane, oss2.str(), cv::Point(20, 55), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 1);
  }

  const cv::Point2d c2(w * 0.50, h * 0.58);
  const double axis_len = std::min(w, h) * 0.28;
  const cv::Point2d n_tip(c2.x + axis_len, c2.y);
  cv::arrowedLine(normal, c2, n_tip, cv::Scalar(80, 220, 80), 3, cv::LINE_AA, 0, 0.05);
  cv::putText(normal, "armor normal +n", cv::Point(static_cast<int>(n_tip.x) - 30, static_cast<int>(n_tip.y) - 10),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 240, 120), 1);

  const double v_n = std::max(0.0, best_s->normal_velocity);
  const double v_ref = std::max(1e-3, get_parameter("controller.fire.probability.normal_velocity_weight.v_ref").as_double());
  const double ratio = std::clamp(v_n / v_ref, 0.0, 1.0);
  const double spread_deg = (1.0 - ratio) * 35.0;
  const double center_deg = best_front_ok ? 180.0 : 0.0;
  const double theta_deg = center_deg + (best_front_ok ? -spread_deg : spread_deg);
  const double theta = theta_deg * M_PI / 180.0;
  const cv::Point2d v_tip(c2.x + axis_len * std::cos(theta), c2.y - axis_len * std::sin(theta));
  cv::arrowedLine(normal, c2, v_tip, cv::Scalar(80, 180, 255), 3, cv::LINE_AA, 0, 0.05);
  cv::putText(normal, "bullet velocity", cv::Point(static_cast<int>(v_tip.x) - 30, static_cast<int>(v_tip.y) - 8),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 200, 255), 1);

  if (fire_prob_image_debug_show_velocity_fan_) {
    const cv::Scalar fan_color = best_front_ok ? cv::Scalar(60, 200, 60) : cv::Scalar(60, 60, 220);
    const double fan_half = 15.0 + (1.0 - ratio) * 30.0;
    cv::ellipse(normal, c2, cv::Size(static_cast<int>(axis_len * 0.7), static_cast<int>(axis_len * 0.7)),
      0.0, center_deg - fan_half, center_deg + fan_half, fan_color, 2, cv::LINE_AA);
  }

  cv::line(normal,
    cv::Point(static_cast<int>(c2.x), static_cast<int>(c2.y)),
    cv::Point(static_cast<int>(c2.x + axis_len * ratio), static_cast<int>(c2.y)),
    cv::Scalar(0, 255, 255), 4, cv::LINE_AA);
  cv::putText(normal, "v_n along normal", cv::Point(static_cast<int>(c2.x), static_cast<int>(c2.y) + 24),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

  const cv::Scalar status_color = (best_front_ok && best_gate_ok) ? cv::Scalar(80, 240, 80) : cv::Scalar(80, 80, 240);
  const std::string status_text = (best_front_ok && best_gate_ok) ? "HIT GATE: PASS" : "HIT GATE: BLOCK";
  cv::putText(normal, status_text, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2);

  std::ostringstream ns1;
  ns1 << "front_ok=" << (best_front_ok ? "Y" : "N")
      << " gate_ok=" << (best_gate_ok ? "Y" : "N")
      << " normal_v=" << std::fixed << std::setprecision(2) << best_s->normal_velocity << " m/s";
  cv::putText(normal, ns1.str(), cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1);
  std::ostringstream ns2;
  ns2 << "normal_weight=" << std::setprecision(2) << best_s->normal_weight
      << " p_hit=" << best_s->p_hit;
  cv::putText(normal, ns2.str(), cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1);

  auto plane_msg = cv_bridge::CvImage(header, "bgr8", plane).toImageMsg();
  auto normal_msg = cv_bridge::CvImage(header, "bgr8", normal).toImageMsg();
  debug_fire_plane_image_pub_->publish(*plane_msg);
  debug_fire_normal_image_pub_->publish(*normal_msg);
}

void GimbalPipelineNode::publishManeuverMarkers(
    const std_msgs::msg::Header &header) {
  visualization_msgs::msg::MarkerArray arr;

  // Tracker positions are expressed in target_frame_ (== visualization_frame_).
  // Use visualization_frame_ explicitly to avoid the camera source frame mismatch.
  std_msgs::msg::Header viz_header;
  viz_header.stamp    = header.stamp;
  viz_header.frame_id = visualization_frame_;

  int id = 0;

  const auto tracker_views = tracker_manager_->initialized_tracker_views();
  for (const auto &view : tracker_views) {
    if (!view.tracker) continue;

    const auto result = view.tracker->assess_maneuver();
    const auto pos    = view.tracker->get_center_position();

    // Estimate robot top: center pos + half robot height (~0.25 m)
    const double top_z = pos.z() + 0.25;

    // ── Sphere marker (at robot top) ───────────────────────────
    visualization_msgs::msg::Marker sphere;
    sphere.header       = viz_header;
    sphere.ns           = "maneuver";
    sphere.id           = id++;
    sphere.type         = visualization_msgs::msg::Marker::SPHERE;
    sphere.action       = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = pos.x();
    sphere.pose.position.y = pos.y();
    sphere.pose.position.z = top_z;
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12;
    sphere.lifetime = rclcpp::Duration::from_seconds(0.15);
    if (result.is_maneuvering) {
      sphere.color.r = 0.9f; sphere.color.g = 0.1f;
      sphere.color.b = 0.1f; sphere.color.a = 0.8f;
    } else {
      sphere.color.r = 0.1f; sphere.color.g = 0.9f;
      sphere.color.b = 0.1f; sphere.color.a = 0.6f;
    }
    arr.markers.push_back(sphere);

    // ── Text marker (above sphere) ─────────────────────────────
    visualization_msgs::msg::Marker text;
    text.header    = viz_header;
    text.ns        = "maneuver_text";
    text.id        = id++;
    text.type      = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action    = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = pos.x();
    text.pose.position.y = pos.y();
    text.pose.position.z = top_z + 0.15;
    text.pose.orientation.w = 1.0;
    text.scale.z   = 0.08;
    text.color.r   = 1.0f; text.color.g = 1.0f;
    text.color.b   = 1.0f; text.color.a = 1.0f;
    text.lifetime  = rclcpp::Duration::from_seconds(0.15);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "NIS=%.0f\nIN=%.3f",
                  result.nis, result.innov_norm);
    text.text = buf;
    arr.markers.push_back(text);
  }

  if (!arr.markers.empty()) debug_maneuver_pub_->publish(arr);
}

void GimbalPipelineNode::publishArmorSelectionDebug(
    const gimbal_controller::GimbalControlContext &context,
    const std::string &strategy_name) {
  if (!debug_armor_selection_pub_ || !armor_selector_ || !position_calculator_) {
    return;
  }
  if (!context.is_tracking) {
    return;
  }

  const auto normalized_target = robot_description::TrackedRobotUsage::normalizeState(context.target_robot);
  const auto center_position = robot_description::TrackedRobotUsage::centerPosition(normalized_target);
  const double target_yaw = robot_description::TrackedRobotUsage::yaw(normalized_target);
  const double target_yaw_velocity = robot_description::TrackedRobotUsage::yawVelocity(normalized_target);

  std::vector<Eigen::Vector3d> armor_positions;
  Eigen::Vector3d select_center = center_position;
  double select_yaw = target_yaw;

  if (strategy_name == "mpc") {
    const double t_ahead = mpc_dt_debug_;
    armor_positions = position_calculator_->calculatePredicted(normalized_target, t_ahead);
    select_center = robot_description::TrackedRobotUsage::predictCenter(
      normalized_target,
      t_ahead,
      robot_description::TrackedRobotUsage::MotionModel::CONSTANT_VELOCITY);
    select_yaw = robot_description::TrackedRobotUsage::predictYaw(
      normalized_target,
      t_ahead,
      robot_description::TrackedRobotUsage::MotionModel::CONSTANT_VELOCITY);
  } else {
    armor_positions = position_calculator_->calculate(normalized_target);
  }

  if (armor_positions.empty()) {
    return;
  }

  auto debug_selector = *armor_selector_;
  const auto selection = debug_selector.selectBest(
    armor_positions,
    select_center,
    select_yaw,
    normalized_target.num_armors,
    target_yaw_velocity,
    context.current_yaw,
    context.current_pitch);

  std_msgs::msg::String out;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4)
      << "strategy=" << strategy_name
      << " sel=" << selection.selected_index
      << " real_sel=" << selection.real_selected_index
      << " is_virtual=" << (selection.is_virtual_target ? 1 : 0)
      << " fallback=" << (selection.is_center_fallback ? 1 : 0)
      << " dist=" << selection.distance
      << " move=" << selection.gimbal_movement
      << " v_yaw=" << selection.virtual_robot_yaw
      << " d_yaw=" << selection.virtual_delta_yaw
      << " pos=(" << selection.position.x() << "," << selection.position.y() << ","
      << selection.position.z() << ")"
      << " real=(" << selection.real_position.x() << "," << selection.real_position.y() << ","
      << selection.real_position.z() << ")";
  if (strategy_name == "mpc") {
    oss << " note=first_predicted_selection";
  }
  out.data = oss.str();
  debug_armor_selection_pub_->publish(out);
}

std::array<float, 4> GimbalPipelineNode::hsvToRgb(float h, float s,
                                                    float v) {
  float c = v * s;
  float x = c * (1 - std::abs(std::fmod(h / 60.0f, 2.0f) - 1));
  float m = v - c;
  float r, g, b;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }
  return {r + m, g + m, b + m, 1.0f};
}

}  // namespace fyt::auto_aim

// Register as composable node
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(fyt::auto_aim::GimbalPipelineNode)
