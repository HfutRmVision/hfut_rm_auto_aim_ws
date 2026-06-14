// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
//
// OutputSmoother — non-intrusive post-processing layer
//
// Architecture:
// ┌───────────────────────────────────────────────────────────┐
// │                      OutputSmoother                       │
// │                                                           │
// │  ┌──────────────────────┐  ┌────────────────────────┐    │
// │  │ OneEuro Position 3D  │  │ OneEuro Yaw (Angle)    │    │
// │  │ center x, y, z       │  │ center yaw             │    │
// │  └──────────────────────┘  └────────────────────────┘    │
// │                                                           │
// │  ┌──────────────────────┐  ┌────────────────────────┐    │
// │  │ OneEuro Velocity 3D  │  │ OneEuro Yaw Velocity   │    │
// │  │ vx, vy, vz           │  │ v_yaw                  │    │
// │  └──────────────────────┘  └────────────────────────┘    │
// │  ┌──────────────────────┐                                │
// │  │ Robbins-Monro        │                                │
// │  │ r1, r2, dza          │                                │
// │  └──────────────────────┘                                │
// └───────────────────────────────────────────────────────────┘
//
// Workflow:
// 1. Tracker completes predict + update → raw output
// 2. OutputSmoother::smooth() applies per-component filtering
// 3. Downstream (topic publish / serial) uses smoothed output
//
// IMPORTANT: Does NOT modify UKF internal state.

#ifndef MAX_ENTROPY_TRACKER_UTILS_OUTPUT_SMOOTHER_HPP_
#define MAX_ENTROPY_TRACKER_UTILS_OUTPUT_SMOOTHER_HPP_

#include <cmath>
#include <optional>

#include <Eigen/Dense>

#include "max_entropy_tracker/utils/one_euro_filter.hpp"
#include "max_entropy_tracker/utils/robbins_monro_estimator.hpp"

namespace fyt::auto_aim {

// ──────────────────────────────────────────────────────────
//  SmootherConfig — all tunable parameters
// ──────────────────────────────────────────────────────────
struct SmootherConfig {
  // Master switch
  bool enable = true;

  // Sub-component switches
  bool enable_position_smooth = true;
  bool enable_yaw_smooth = true;
  bool enable_velocity_smooth = true;
  bool enable_structural_convergence = true;

  // ---- Position OneEuro ----
  double pos_min_cutoff = 1.5;   // Hz, lower → more smooth
  double pos_beta = 0.01;        // speed response
  double pos_d_cutoff = 1.0;     // derivative filter cutoff

  // ---- Yaw OneEuro ----
  double yaw_min_cutoff = 1.0;
  double yaw_beta = 0.005;
  double yaw_d_cutoff = 1.0;

  // ---- Velocity OneEuro ----
  double vel_min_cutoff = 2.0;
  double vel_beta = 0.01;
  double vel_d_cutoff = 1.0;

  // ---- Yaw velocity OneEuro ----
  double yaw_vel_min_cutoff = 0.5;
  double yaw_vel_beta = 0.02;
  double yaw_vel_d_cutoff = 1.0;
  double yaw_vel_deadband = 0.0;

  // ---- Structural Robbins-Monro ----
  double rm_initial_step = 0.5;
  double rm_gamma = 0.75;
  int rm_n0 = 5;
  double rm_dual_obs_boost = 3.0;
  double rm_min_radius = 0.12;
  double rm_max_radius = 0.5;
  double rm_min_dz = -1.0;
  double rm_max_dz = 1.0;
  double rm_convergence_eps = 1e-4;

  // Default sampling frequency
  double default_freq = 30.0;

  // ---- Outlier filter (independent of smoother.enable) ----
  // Sits between UKF output and smooth(); when outlier is detected the
  // last valid SmoothedOutput is held (held-frame strategy).
  bool        enable_outlier_filter{false};
  std::string outlier_method{"mad"};      // "mad" | "iqr" | "mahalanobis"
  int         outlier_window_size{10};    // rolling window length
  int         outlier_min_samples{5};     // start filtering after N samples
  double      outlier_mad_k{3.5};         // MAD threshold multiplier
  double      outlier_iqr_k{1.5};         // IQR threshold multiplier
  double      outlier_mahal_threshold{9.21}; // chi²(3) threshold
};

// ──────────────────────────────────────────────────────────
//  SmoothedOutput — result of one smooth() call
// ──────────────────────────────────────────────────────────
struct SmoothedOutput {
  // Smoothed values
  Eigen::Vector3d center_position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double yaw_velocity{0.0};
  double r1{0.0};
  double r2{0.0};
  double dza{0.0};

  // Raw values (for debug comparison)
  Eigen::Vector3d raw_center_position{Eigen::Vector3d::Zero()};
  double raw_yaw{0.0};
  double raw_r1{0.0};
  double raw_r2{0.0};
  double raw_dza{0.0};

  bool structural_converged{false};
};

// ──────────────────────────────────────────────────────────
//  OutputSmoother — main class
// ──────────────────────────────────────────────────────────
class OutputSmoother {
 public:
  OutputSmoother() = default;
  explicit OutputSmoother(const SmootherConfig &cfg) { configure(cfg); }

  /// Apply configuration (can be called to reconfigure at runtime)
  void configure(const SmootherConfig &cfg) {
    cfg_ = cfg;

    pos_filter_ = OneEuroFilter3D(cfg.default_freq, cfg.pos_min_cutoff,
                                  cfg.pos_beta, cfg.pos_d_cutoff);
    yaw_filter_ = OneEuroFilterAngle(cfg.default_freq, cfg.yaw_min_cutoff,
                                     cfg.yaw_beta, cfg.yaw_d_cutoff);
    vel_filter_ = OneEuroFilter3D(cfg.default_freq, cfg.vel_min_cutoff,
                                  cfg.vel_beta, cfg.vel_d_cutoff);
    yaw_vel_filter_ = OneEuroFilter(cfg.default_freq, cfg.yaw_vel_min_cutoff,
                                    cfg.yaw_vel_beta, cfg.yaw_vel_d_cutoff);

    StructuralRMEstimator::Config rm_cfg;
    rm_cfg.initial_step = cfg.rm_initial_step;
    rm_cfg.gamma = cfg.rm_gamma;
    rm_cfg.n0 = cfg.rm_n0;
    rm_cfg.dual_obs_boost = cfg.rm_dual_obs_boost;
    rm_cfg.min_radius = cfg.rm_min_radius;
    rm_cfg.max_radius = cfg.rm_max_radius;
    rm_cfg.min_dz = cfg.rm_min_dz;
    rm_cfg.max_dz = cfg.rm_max_dz;
    rm_cfg.convergence_eps = cfg.rm_convergence_eps;
    struct_est_.configure(rm_cfg);
  }

  /// Initialize the smoother (call when tracker is first initialized)
  void initialize(double r1, double r2, double dza) {
    pos_filter_.reset();
    yaw_filter_.reset();
    vel_filter_.reset();
    yaw_vel_filter_.reset();

    if (cfg_.enable_structural_convergence) {
      struct_est_.initialize(r1, r2, dza);
    }

    initialized_ = true;
    frame_count_ = 0;
  }

  /// Smooth one frame of tracker output
  SmoothedOutput smooth(const Eigen::Vector3d &center_pos,
                        double yaw,
                        const Eigen::Vector3d &velocity,
                        double yaw_velocity,
                        double r1, double r2, double dza,
                        bool is_dual_obs,
                        std::optional<double> timestamp = std::nullopt) {
    SmoothedOutput out;

    // Always store raw values
    out.raw_center_position = center_pos;
    out.raw_yaw = yaw;
    out.raw_r1 = r1;
    out.raw_r2 = r2;
    out.raw_dza = dza;
    out.yaw_velocity = yaw_velocity;

    if (!initialized_ || !cfg_.enable) {
      // Pass-through
      out.center_position = center_pos;
      out.velocity = velocity;
      out.yaw = yaw;
      out.r1 = r1;
      out.r2 = r2;
      out.dza = dza;
      return out;
    }

    ++frame_count_;

    // ---- Position ----
    out.center_position = cfg_.enable_position_smooth
                              ? pos_filter_.filter(center_pos, timestamp)
                              : center_pos;

    // ---- Yaw ----
    out.yaw = cfg_.enable_yaw_smooth
                  ? yaw_filter_.filter(yaw, timestamp)
                  : yaw;

    // ---- Velocity ----
    out.velocity = cfg_.enable_velocity_smooth
                       ? vel_filter_.filter(velocity, timestamp)
                       : velocity;
    out.yaw_velocity = cfg_.enable_velocity_smooth
                           ? yaw_vel_filter_.filter(yaw_velocity, timestamp)
                           : yaw_velocity;
    if (cfg_.yaw_vel_deadband > 0.0 &&
        std::abs(out.yaw_velocity) < cfg_.yaw_vel_deadband) {
      out.yaw_velocity = 0.0;
    }

    // ---- Structural parameters (Robbins-Monro) ----
    if (cfg_.enable_structural_convergence && struct_est_.is_initialized()) {
      struct_est_.update(r1, r2, dza, is_dual_obs);
      auto [sr1, sr2, sdza] = struct_est_.get_smoothed_params();
      out.r1 = sr1;
      out.r2 = sr2;
      out.dza = sdza;
      out.structural_converged = struct_est_.is_converged();
    } else {
      out.r1 = r1;
      out.r2 = r2;
      out.dza = dza;
      out.structural_converged = false;
    }

    return out;
  }

  /// Full reset
  void reset() {
    pos_filter_.reset();
    yaw_filter_.reset();
    vel_filter_.reset();
    yaw_vel_filter_.reset();
    struct_est_.reset();
    initialized_ = false;
    frame_count_ = 0;
  }

  bool is_initialized() const { return initialized_; }
  int frame_count() const { return frame_count_; }
  const SmootherConfig &config() const { return cfg_; }
  const StructuralRMEstimator &structural_estimator() const {
    return struct_est_;
  }

 private:
  SmootherConfig cfg_;

  OneEuroFilter3D pos_filter_;
  OneEuroFilterAngle yaw_filter_;
  OneEuroFilter3D vel_filter_;
  OneEuroFilter yaw_vel_filter_;
  StructuralRMEstimator struct_est_;

  bool initialized_{false};
  int frame_count_{0};
};

}  // namespace fyt::auto_aim

#endif  // MAX_ENTROPY_TRACKER_UTILS_OUTPUT_SMOOTHER_HPP_
