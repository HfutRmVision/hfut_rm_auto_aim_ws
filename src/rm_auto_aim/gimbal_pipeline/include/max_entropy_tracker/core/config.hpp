// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
#ifndef MAX_ENTROPY_TRACKER_CORE_CONFIG_HPP_
#define MAX_ENTROPY_TRACKER_CORE_CONFIG_HPP_

#include <cmath>
#include <string>

namespace fyt::auto_aim {

// ======================== Enums ========================

enum class TranslationModel { CV, CA, SINGER };

enum class RotationModel { CV, CA };

enum class FilterType { STANDARD, DECOMPOSED };

// ======================== Parameter Structs ========================

struct UKFParameters {
  // Sigma point sampling
  double alpha = 0.001;
  double beta = 2.0;
  double kappa = 0.0;

  // Single-observation noise
  double obs_noise_pos = 0.05;
  double obs_noise_yaw = 0.05;
  bool enable_ypd_observation_noise = false;
  double ypd_sigma_azi = 0.01;
  double ypd_sigma_ele = 0.01;
  double ypd_sigma_dist_coeff = 0.08;

  // Dual-observation noise
  double dual_obs_noise_pos = 0.01;
  double dual_obs_noise_yaw = 0.03;
  double dual_obs_geometry_noise_scale = 0.2;

  // Single-observation position update weight
  double single_obs_update_weight_pos = 0.05;

  // Innovation gating
  bool enable_innovation_gating = false;
  double innovation_gate_chi2_threshold = 9.49;
};

struct MotionModelParameters {
  TranslationModel translation_model = TranslationModel::CA;

  // CV
  double cv_process_noise_vel = 0.5;
  // CA
  double ca_process_noise_acc = 1.0;
  // Singer
  double singer_alpha = 0.5;
  double singer_sigma = 2.0;

  // Structural
  double process_noise_r = 0.02;
  double process_noise_dz = 0.005;
};

struct SpinModelParameters {
  double spin_process_noise_yaw_rate = 0.3;
  double spin_process_noise_yaw_acc = 1.0;
  double spin_process_noise_delta_rate = 0.3;
  double spin_process_noise_delta_acc = 3.0;
};

struct MaxEntropyParameters {
  double temperature = 2.0;
  bool use_adaptive = true;
  double k_prior_weight = 0.7;
};

struct TrackerParameters {
  // Tracker implementation for non-outpost robots:
  // "adaptive" | "norm4" | "norm4_v2"
  std::string implementation = "adaptive";

  int tracking_thres = 2;
  int lost_thres = 8;
  int temp_lost_thres = 3;

  double max_match_distance = 2.0;
  double max_match_yaw_diff = 1.0;

  int n_panels = 4;
  double panel_angle_step = M_PI / 2.0;

  // Optional periodic dz prior for standard 4-panel association.
  // Template: -dz, +dz, -dz, +dz (sign flips with spin direction).
  bool periodic_binding_enable = false;
  double periodic_binding_weight = 0.35;
  double periodic_binding_spin_rate_gate = 0.8;

  // Jump binder (adaptive tracker): sticky panel/layer in non-jump frames,
  // and transition-confirmed switching on reliable z-jump evidence.
  bool jump_binding_enable = true;
  int jump_binding_confirm_frames = 3;
  double jump_binding_z_jump_min = 0.015;
  double jump_binding_dz_match_tolerance = 0.03;
  double jump_binding_dz_gate = 0.010;
  double jump_binding_yaw_err_gate = 0.35;
  double jump_binding_cost_margin_min = 0.08;
  int jump_binding_switch_cooldown = 2;
  double jump_binding_dz_ema_alpha = 0.20;
  double jump_binding_confidence_floor = 0.15;

  // Legacy split-path single-observation degraded mode. Keep disabled for the
  // unified non-outpost hypothesis pipeline.
  bool degraded_single_obs_enable = false;
  int degraded_single_obs_streak = 8;
  double degraded_q_scale_r = 4.0;
  double degraded_q_scale_dza = 4.0;
};

struct ConstraintParameters {
  double min_radius = 0.12;
  double max_radius = 0.5;
  double min_dz = -1.0;
  double max_dz = 1.0;
};

struct PanelMismatchParameters {
  bool   enable        = true;
  int    window_size   = 8;       ///< rolling buffer length W (frames)
  double threshold_t1  = 0.0009;  ///< dz² mean threshold below which z is OK (3cm²)
  int    confirm_count = 3;       ///< consecutive suspect frames to trigger PATCH
  int    reinit_count  = 5;       ///< consecutive suspect frames to trigger REINIT
  bool   apply_correction = false;  ///< false=log only, true=allow PATCH/REINIT
};

struct OutpostParameters {
  // Outpost tracker implementation switch.
  // false: legacy OutpostArmorTracker
  // true : OutpostTrackerV2 (mode-aware pipeline)
  bool use_tracker_v2 = false;
  // If true, use OutpostTrackerV3 (hypothesis + InEKF pipeline).
  // Priority: use_tracker_v3 > use_tracker_v2 > legacy.
  bool use_tracker_v3 = false;

  // Outpost-specific tracker state machine thresholds
  int tracking_thres = 2;
  int lost_thres = 40;
  int temp_lost_thres = 30;

  // Outpost-specific matching gates (reserved for association logic)
  double max_match_distance = 2.0;
  double max_match_yaw_diff = 1.0;

  // Motion model selection (same style as 4-panel tracker)
  TranslationModel translation_model = TranslationModel::CV;
  RotationModel rotation_model = RotationModel::CV;

  // Optional outpost-specific Singer params (fallback to motion.* when <= 0)
  double singer_alpha = 0.0;
  double singer_sigma = 0.0;

  // Optional outpost-specific spin process noise (fallback to spin.* when <= 0)
  double spin_process_noise_theta_rate = 0.0;
  double spin_process_noise_theta_acc = 0.0;

  // Known geometric profile (relative to outpost center)
  // Semantic contract:
  //   panel 0 = highest, panel 1 = middle, panel 2 = lowest.
  // Top-down clockwise order: 0 deg(panel 0) -> panel 2 -> panel 1.
  double radius = 0.26;
  double z_offset_0 = 0.06;
  double z_offset_1 = 0.00;
  double z_offset_2 = -0.06;
  double panel_angle_step = 2.0 * M_PI / 3.0;

  // Max-entropy panel posterior
  double softmax_temperature = 1.5;
  double weight_yaw = 1.0;
  double weight_z_state = 6.0;
  double weight_z_history = 2.0;
  double weight_xy_residual = 2.5;
  double weight_switch_penalty = 0.05;

  // Hysteresis gating for mode switch (3-armors <-> single-armor)
  double entropy_enter = 0.75;
  double entropy_exit = 0.55;
  double max_prob_enter = 0.60;
  double max_prob_exit = 0.75;
  int stable_frames = 4;
  int z_history_window = 15;

  // Single-armor output confidence scaling
  double single_mode_confidence_scale = 0.70;

  // Binding engine controls (periodic evidence + transition confirmation)
  bool binding_use_new_binder_pipeline = false;
  bool binding_enable_multi_obs = true;
  int binding_transition_confirm_frames = 3;
  double binding_same_panel_yaw_gate = 0.35;
  double binding_same_panel_z_gate = 0.08;
  double binding_same_panel_xy_gate = 0.18;
  double binding_min_candidate_prob = 0.40;
  double binding_min_candidate_margin = 0.12;
  double binding_switch_strong_score = 0.60;
  int binding_period_window = 12;
  double binding_period_weight = 0.60;
  double binding_topology_prior_weight = 4.0;
  double binding_period_min_spin_rate = 0.8;
  int spin_direction_confirm_frames = 3;
  double binding_period_update_min_confidence = 0.55;
  double binding_period_update_min_jump = 0.015;
  double binding_dz_ema_alpha = 0.20;
  double binding_confidence_floor = 0.15;
  bool z_audit_rebind_enable = true;
  int z_audit_rebind_confirm_frames = 3;
  double z_audit_rebind_min_confidence = 0.60;
  double z_audit_rebind_min_jump = 0.015;
  double binding_conflict_position_scale = 0.10;

  // Kinematic smoothing gains
  double alpha_pos = 0.65;
  double beta_vel = 0.30;
  double alpha_yaw = 0.60;
  double beta_yaw_rate = 0.25;

  // Physical constraints / damping
  bool assume_static_center = true;
  double linear_velocity_damping = 0.90;
  double yaw_rate_damping = 0.98;
  double max_center_speed = 1.00;
  double max_yaw_rate = 12.0;
  double max_yaw_rate_step = 3.0;

  // ── Ambiguous semantics & backend control ──
  bool ambiguous_publish_single_armor_semantics = true;
  bool ambiguous_single_armor_zero_offset = true;
  bool ambiguous_backend_use_imm_adapter = false;

  // OutpostTrackerV2 ID warmup: publish ambiguous single-armor output while
  // collecting relative z-level evidence, then bind 0/1/2 after dz/2dz is observed.
  bool v2_warmup_enable = true;
  int v2_warmup_min_groups = 3;
  int v2_warmup_min_samples_per_group = 2;
  int v2_warmup_max_frames = 60;
  double v2_warmup_z_jump_gate = 0.025;
  double v2_warmup_yaw_jump_gate = 0.75;
  double v2_warmup_xyz_jump_gate = 0.18;
  double v2_warmup_ratio_min = 1.55;
  double v2_warmup_ratio_max = 2.45;
  double v2_warmup_min_large_diff = 0.06;

  // ── Outpost V3 config ──
  int v3_topk = 3;
  double v3_min_top1_confidence = 0.5;
  double v3_min_top1_top2_margin = 1.0;
  double v3_max_reconstruction_pos_error = 0.3;

  double v3_gate_single_total_nis = 11.34;
  double v3_gate_single_pos_chi2 = 9.0;

  double v3_posterior_max_center_jump = 0.5;
  double v3_posterior_max_yaw_jump = 0.5;
  double v3_posterior_max_yaw_rate = 15.0;
  double v3_posterior_max_yaw_acc = 30.0;

  double v3_mode_p_enter_structured = 0.7;
  double v3_mode_m_enter_structured = 1.5;
  int v3_mode_stable_frames = 5;
  double v3_mode_p_exit_structured = 0.4;
  double v3_mode_m_exit_structured = 0.5;
  int v3_mode_degraded_frames = 10;

  double v3_prior_panel_switch_penalty = 0.5;

  double v3_initial_p_pos = 0.01;
  double v3_initial_p_vel = 1.0;
  double v3_initial_p_acc = 10.0;
  double v3_initial_p_yaw = 0.1;
  double v3_initial_p_yaw_rate = 1.0;
  double v3_initial_p_yaw_acc = 5.0;

  double v3_process_noise_acc = 2.0;
  double v3_process_noise_yaw_acc = 3.0;

  double v3_observation_sigma_pos_xy = 0.02;
  double v3_observation_sigma_pos_z = 0.03;

  bool v3_warmup_enable = true;
  int v3_warmup_frames = 8;
  int v3_warmup_min_settle_frames = 3;
  double v3_warmup_min_margin_to_commit = 1.2;
  double v3_warmup_min_confidence_to_commit = 0.65;

  bool v3_phase_audit_enable = true;
  double v3_phase_audit_min_jump = 0.015;
  double v3_phase_audit_dz_gate = 0.035;
  int v3_phase_audit_confirm_frames = 2;

  // ModeFSM (OutpostTrackerV2)
  int mode_enter_confirm_frames = 3;
  int mode_exit_confirm_frames = 4;
  int mode_min_dwell_frames = 6;
  double mode_enter_threshold = 0.72;
  double mode_exit_threshold = 0.45;

  // Mode evidence fusion weights (OutpostTrackerV2)
  double mode_weight_jump = 0.30;
  double mode_weight_dual = 0.20;
  double mode_weight_margin = 0.20;
  double mode_weight_health = 0.20;
  double mode_weight_entropy = 0.10;
};

struct ManeuverDetectionParameters {
  bool   enable                      = true;
  double nis_threshold_single        = 238.807;
  double nis_threshold_dual          = 4132.110;
  double innov_norm_threshold_single = 0.1279;
  double innov_norm_threshold_dual   = 0.0613;

  // Optional MAD-based outlier filter applied before threshold comparison.
  // When enabled, nis and innov_norm are each filtered through a rolling
  // window of mad_window samples per update_type before the decision rule.
  bool   mad_filter_enable = false;
  int    mad_window        = 10;    ///< rolling window size (samples)
  double mad_k             = 3.0;   ///< outlier threshold = mad_k * MAD
};

// ======================== Binder Config ========================

struct BinderConfig {
  // ── Common / FSM ──
  int confirm_frames = 3;
  int lock_new_hold_frames = 2;
  int force_rebind_bad_frames = 10;
  int pending_window_frames = 0;
  double post_jump_min_confidence = 0.45;
  double confidence_floor = 0.15;

  // ── Decoder: PROXIMITY gates (4-panel) ──
  double z_jump_min = 0.015;
  double dz_match_tolerance = 0.03;
  double dz_gate = 0.010;
  double yaw_err_gate = 0.35;
  double cost_margin_min = 0.08;
  double dz_ema_alpha = 0.20;

  // ── Decoder: periodic evidence ──
  bool periodic_enable = false;
  int periodic_window = 12;
  double periodic_weight = 0.60;
  double periodic_min_spin_rate = 0.8;
  double periodic_update_min_jump = 0.015;
  double periodic_signature_threshold = 0.60;
  double reacquire_gap_dt_gate = 0.12;
  int reacquire_lost_frames_gate = 1;
  double z_cluster_ema_alpha = 0.25;
  double z_cluster_assign_gate = 0.10;

  // ── ID Binder: COST gates ──
  double min_candidate_prob = 0.40;
  double min_candidate_margin = 0.12;
  double switch_strong_score = 0.60;
  int single_obs_history_window = 8;
  bool dual_obs_enable = true;

  // ── Scorer ──
  bool scorer_enable = true;
  double same_panel_yaw_gate = 0.35;
  double same_panel_z_gate = 0.08;
  double same_panel_xy_gate = 0.18;

  // ── Scorer: z-audit rebind (outpost) ──
  bool z_audit_rebind_enable = false;
  int z_audit_rebind_confirm_frames = 3;
  double z_audit_rebind_min_confidence = 0.60;
  double z_audit_rebind_min_jump = 0.015;

  // ── Phase 6: soft fusion weights ──
  bool enable_soft_fusion = false;
  double soft_fusion_w_seq = 0.25;
  double soft_fusion_w_geo = 0.40;
  double soft_fusion_w_dyn = 0.20;
  double soft_fusion_w_continuity = 0.15;
  double soft_fusion_w_topology = 0.15;
};

// ======================== Norm4 V2 Config ========================

struct AntiPingPongConfig {
  int min_consistent_frames_to_commit = 3;
  double jerk_gate = 1.5;
  double yaw_rate_jump_gate = 2.0;
  double velocity_dir_cos_min = 0.2;
  int pending_timeout_frames = 12;
};

struct PhaseMemoryConfig {
  bool enable_phase_memory = true;
  bool enable_kinematic_anti_pingpong = true;
  int sequence_window_size = 10;
  double ping_pong_pattern_threshold = 0.7;
  bool enable_opposite_jump_detect = true;
  AntiPingPongConfig anti_pingpong;
};

struct Norm4V2UkfGateConfig {
  double single_total_nis = 25.0;
  double single_pos_chi2 = 16.0;
  double single_yaw_chi2 = 9.0;
  double dual_total_nis = 45.0;
  double dual_each_pos_chi2 = 16.0;
  double dual_each_yaw_chi2 = 9.0;
};

struct Norm4V2UkfSingleUpdateConfig {
  double structural_gain_r = 0.0;
  double structural_gain_dza = 0.0;
};

struct Norm4V2UkfDualUpdateConfig {
  double structural_gain_r = 0.05;
  double structural_gain_dza = 0.02;
};

struct Norm4V2UkfPosteriorSanityConfig {
  double max_center_jump = 0.25;
  double max_yaw_jump = 0.80;
  double min_r = 0.05;
  double max_r = 0.50;
  double max_r_jump = 0.05;
  double min_dza = 0.0;
  double max_dza = 0.15;
  double max_dza_jump = 0.03;
};

struct Norm4V2UkfConfig {
  bool enabled = true;
  bool force_rotation_ca = false;
  bool dual_raw_batch = true;

  double sigma_pos_xy = 0.06;
  double sigma_pos_z = 0.08;
  double sigma_yaw = 0.12;
  double dual_raw_R_scale = 1.5;

  Norm4V2UkfGateConfig gate;
  Norm4V2UkfSingleUpdateConfig single_update;
  Norm4V2UkfDualUpdateConfig dual_update;
  Norm4V2UkfPosteriorSanityConfig posterior_sanity;
};

struct Norm4V2SelectorConfig {
  int topk = 4;
  bool commit_top1_only = true;
  double min_top1_confidence = 0.55;
  double min_top1_top2_margin = 0.0;
  double ambiguous_margin = 1.0;
  bool include_rejected_in_debug = true;
  bool evidence_prior_enable = false;

  // Reconstruction error gate (meters)
  double max_reconstruction_pos_error = 0.30;
};

struct Norm4V2WarmupConfig {
  bool enable_dual_seed_01 = false;
  int warmup_frames = 8;
  int min_settle_frames = 3;
  double min_margin_to_commit = 1.5;
  double min_confidence_to_commit = 0.70;
};

struct Norm4V2ModeRoutingConfig {
  // "single_plate_3d" | "structured_ukf"
  std::string ambiguous_output = "single_plate_3d";
  std::string structured_output = "structured_ukf";
  // "shallow_or_predict" | "predict_only"
  std::string ambiguous_structured_backend_mode = "shallow_or_predict";
  // "shallow" | "predict_only"
  std::string structured_single_plate_mode = "shallow";
};

struct Norm4V2SinglePlateBridgeConfig {
  bool enable = false;
  std::string source_semantic = "track2d_id";
  std::string backend_type = "norm4_ambiguous_backend";
  int require_semantic_stable_frames = 2;
};

struct Norm4V2FallbackConfig {
  bool predict_only_on_reject = true;
  bool enable_ambiguous_single_fallback = true;
};

struct Norm4V2Config {
  bool enable_common_pipeline = false;
  bool enable_phase_memory = true;
  bool enable_kinematic_anti_pingpong = true;
  bool enable_2d_tracker = false;
  bool enable_proxy_manager = false;
  PhaseMemoryConfig phase_memory;

  Norm4V2UkfConfig ukf_v1;
  Norm4V2SelectorConfig hypothesis_selector;
  Norm4V2WarmupConfig warmup;
  Norm4V2ModeRoutingConfig mode_routing;
  Norm4V2SinglePlateBridgeConfig single_plate_bridge;
  Norm4V2FallbackConfig fallback;
};

using Norm4V3UkfGateConfig = Norm4V2UkfGateConfig;
using Norm4V3UkfSingleUpdateConfig = Norm4V2UkfSingleUpdateConfig;
using Norm4V3UkfDualUpdateConfig = Norm4V2UkfDualUpdateConfig;
using Norm4V3UkfPosteriorSanityConfig = Norm4V2UkfPosteriorSanityConfig;
using Norm4V3UkfConfig = Norm4V2UkfConfig;
using Norm4V3SelectorConfig = Norm4V2SelectorConfig;
using Norm4V3WarmupConfig = Norm4V2WarmupConfig;
using Norm4V3ModeRoutingConfig = Norm4V2ModeRoutingConfig;
using Norm4V3SinglePlateBridgeConfig = Norm4V2SinglePlateBridgeConfig;
using Norm4V3FallbackConfig = Norm4V2FallbackConfig;

// ── Norm4 V3 Observation Noise Config (Phase 1: YPD + BA dynamic R) ──

struct MeasurementNoiseCameraConfig {
  std::string source = "config";
  double fx = 1556.34704;
  double fy = 1557.43488;
  double cx = 610.59754;
  double cy = 503.80001;
  int image_width = 1280;
  int image_height = 1024;
};

struct MeasurementNoiseArmorGeometryConfig {
  double small_width = 0.135;
  double small_height = 0.055;
  double large_width = 0.230;
  double large_height = 0.055;
  double outpost_width = 0.230;
  double outpost_height = 0.055;
};

struct MeasurementNoiseRConfig {
  double sigma_x = 0.060;
  double sigma_y = 0.060;
  double sigma_z = 0.080;
  double sigma_yaw = 0.120;
};

struct MeasurementNoiseYpdPrior {
  double sigma_center_px = 2.0;
  double sigma_size_px = 2.0;
  double sigma_corner_px = 1.5;
  double sigma_azi_min = 0.0005;
  double sigma_azi_max = 0.020;
  double sigma_ele_min = 0.0005;
  double sigma_ele_max = 0.020;
  double sigma_dist_min = 0.02;
  double sigma_dist_max = 1.00;
  double sigma_yaw_min = 0.03;
  double sigma_yaw_max = 0.50;
  double sigma_yaw_scale = 10.0;
  double global_scale = 1.0;
};

struct MeasurementNoiseQualityScale {
  bool enable = true;
  double confidence_floor = 0.30;
  double min_scale = 1.0;
  double max_scale = 3.0;
};

struct MeasurementNoiseBaEigenClamp {
  double min = 1.0e-6;
  double max = 4.0;
};

struct MeasurementNoiseBaDiagClamp {
  double x_min = 1.0e-5;
  double y_min = 1.0e-5;
  double z_min = 4.0e-5;
  double yaw_min = 1.0e-5;
  double x_max = 1.0;
  double y_max = 1.0;
  double z_max = 4.0;
  double yaw_max = 1.0;
};

struct MeasurementNoiseBaCovarianceConfig {
  bool enable = true;
  bool require_cov_valid = true;
  bool require_frame_aligned = true;
  double min_confidence = 0.60;
  double max_reproj_rms_px = 3.0;
  double max_condition_number = 10000000.0;
  int min_observations = 4;
  double min_inlier_ratio = 0.75;
  double max_weight = 0.0;
  double weight_power = 2.0;
  double scale = 4.0;
  MeasurementNoiseBaEigenClamp eigen_clamp;
  MeasurementNoiseBaDiagClamp diag_clamp;
};

struct MeasurementNoiseDynamicBlend {
  double lambda = 0.30;
};

struct MeasurementNoiseDebugConfig {
  bool enable_snapshot = true;
  int log_throttle_ms = 500;
};

struct MeasurementNoiseConfig {
  std::string type = "fixed";
  MeasurementNoiseRConfig r_fixed;
  MeasurementNoiseRConfig r_floor;
  MeasurementNoiseCameraConfig camera;
  MeasurementNoiseArmorGeometryConfig armor_geometry;
  MeasurementNoiseYpdPrior ypd_prior;
  MeasurementNoiseQualityScale quality_scale;
  MeasurementNoiseBaCovarianceConfig ba_covariance;
  MeasurementNoiseDynamicBlend dynamic_blend;
  MeasurementNoiseDebugConfig debug;
};

// ── Norm4 V3 Backend / Selector ──

struct Norm4V3BackendConfig {
  std::string backend_type = "ukf_v1";       // "ukf_v1" | "ukf_v2" | "inekf"
  std::string motion_profile = "default";
  std::string noise_profile = "default";
  std::string structure_profile = "slow";
  bool enable_shadow_mode = false;
  int shadow_convergence_frames = 30;
};

struct Norm4V3InEKFRuntimeConfig {
  // Backend-local profile selection (overrides backend_config.* for inekf only).
  std::string motion_profile = "default";
  std::string noise_profile = "default";
  std::string structure_profile = "slow";

  // Optional motion-model overrides for inekf only.
  // Keep negative values to mean "inherit global motion/spin config".
  std::string translation_model = "";  // "CV" | "CA" | "Singer", empty=inherited
  double cv_process_noise_vel = -1.0;
  double ca_process_noise_acc = -1.0;
  double singer_alpha = -1.0;
  double singer_sigma = -1.0;
  double process_noise_r = -1.0;
  double process_noise_dz = -1.0;
  double spin_process_noise_delta_rate = -1.0;
  double spin_process_noise_delta_acc = -1.0;
};

struct Norm4V3SlowStructureConfig {
  bool enable = true;
  double q_theta_r1 = 1.0e-6;
  double q_theta_r2 = 1.0e-6;
  double q_theta_dza = 5.0e-7;

  double prior_r1 = 0.15;
  double prior_r2 = 0.20;
  double prior_dza = 0.0;
  double prior_sigma_r = 0.06;
  double prior_sigma_dza = 0.06;

  double alpha_r1_single = 0.00;
  double alpha_r2_single = 0.00;
  double alpha_dza_single = 0.00;
  double alpha_r1_dual = 0.05;
  double alpha_r2_dual = 0.05;
  double alpha_dza_dual = 0.02;

  double prior_pull_gain = 0.002;

  double min_r = 0.05;
  double max_r = 0.50;
  double min_dza = 0.0;
  double max_dza = 0.12;
};

struct Norm4V3DebugLogConfig {
  bool enable = false;
  int throttle_ms = 500;
  bool verbose = false;
};

struct Norm4V3Config {
  bool enable_common_pipeline = false;
  bool enable_phase_memory = true;
  bool enable_kinematic_anti_pingpong = true;
  bool enable_2d_tracker = false;
  bool enable_proxy_manager = false;
  PhaseMemoryConfig phase_memory;

  Norm4V3UkfConfig ukf_v1;
  Norm4V3UkfConfig ukf_v2;
  Norm4V3UkfConfig inekf;
  Norm4V3SlowStructureConfig slow_structure;
  Norm4V3SelectorConfig hypothesis_selector;
  Norm4V3WarmupConfig warmup;
  Norm4V3ModeRoutingConfig mode_routing;
  Norm4V3SinglePlateBridgeConfig single_plate_bridge;
  Norm4V3FallbackConfig fallback;
  Norm4V3BackendConfig backend_config;
  Norm4V3InEKFRuntimeConfig inekf_runtime;
  Norm4V3DebugLogConfig debug_log;
};

// ======================== Unified Config ========================

struct UnifiedConfig {
  FilterType filter_type = FilterType::DECOMPOSED;
  double dt = 0.05;

  UKFParameters ukf;
  MotionModelParameters motion;
  SpinModelParameters spin;
  MaxEntropyParameters entropy;
  TrackerParameters tracker;
  ConstraintParameters constraints;
  ManeuverDetectionParameters maneuver;
  PanelMismatchParameters panel_mismatch;
  OutpostParameters outpost;
  BinderConfig binder;
  Norm4V2Config norm4_v2;
  Norm4V3Config norm4_v3;

  static UnifiedConfig create_default() { return UnifiedConfig{}; }

  static UnifiedConfig create_optimized() {
    UnifiedConfig config;
    config.motion.ca_process_noise_acc = 1.5;
    config.spin.spin_process_noise_delta_rate = 1.2;
    config.spin.spin_process_noise_delta_acc = 12.0;
    config.ukf.obs_noise_pos = 0.008;
    config.ukf.obs_noise_yaw = 0.015;
    config.motion.process_noise_r = 0.008;
    config.motion.process_noise_dz = 0.003;
    config.entropy.temperature = 1.5;
    config.entropy.k_prior_weight = 0.6;
    return config;
  }
};

/// Parse TranslationModel from string
inline TranslationModel translation_model_from_string(const std::string &s) {
  if (s == "CV" || s == "cv") return TranslationModel::CV;
  if (s == "CA" || s == "ca") return TranslationModel::CA;
  if (s == "Singer" || s == "singer" || s == "SINGER")
    return TranslationModel::SINGER;
  return TranslationModel::CA;  // default
}

/// Parse RotationModel from string
inline RotationModel rotation_model_from_string(const std::string &s) {
  if (s == "CV" || s == "cv") return RotationModel::CV;
  if (s == "CA" || s == "ca") return RotationModel::CA;
  return RotationModel::CV;  // default
}

}  // namespace fyt::auto_aim

#endif  // MAX_ENTROPY_TRACKER_CORE_CONFIG_HPP_
