// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
#include "max_entropy_tracker/trackers/norm4_v3/backends/norm4_inekf_backend.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "max_entropy_tracker/utils/angle_utils.hpp"
#include "max_entropy_tracker/utils/constraints.hpp"

namespace fyt::auto_aim::norm4_v3 {

// ═══════════════════════════════════════════════════════════════════
// Lie group operations (SO(2))
// ═══════════════════════════════════════════════════════════════════

Eigen::Matrix2d InvariantPoseBackend::adjoint_SO2(double psi) {
  (void)psi;
  // SO(2) is abelian: Ad_{R_z(ψ)} = I for all ψ.
  Eigen::Matrix2d Ad = Eigen::Matrix2d::Identity();
  return Ad;
}

double InvariantPoseBackend::left_jacobian_SO2(double dpsi) {
  // J_l(δψ) = sin(δψ)/δψ, limit → 1 as δψ → 0.
  if (std::abs(dpsi) < 1e-8) return 1.0;
  return std::sin(dpsi) / dpsi;
}

double InvariantPoseBackend::right_jacobian_SO2(double dpsi) {
  // J_r(δψ) = sin(δψ)/δψ (same as left for abelian SO(2)).
  return left_jacobian_SO2(dpsi);
}

// ═══════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════

InvariantPoseBackend::InvariantPoseBackend(
    std::unique_ptr<IMotionModelBundle> motion,
    std::unique_ptr<IMeasurementNoiseModel> noise,
    std::unique_ptr<IStructureProvider> structure,
    const Norm4V3UkfConfig &ukf_config,
    const UnifiedConfig &config, double dt)
    : config_(config),
      ukf_config_(ukf_config),
      dt_(dt),
      motion_(std::move(motion)),
      noise_(std::move(noise)),
      structure_(std::move(structure)) {
  int n = motion_->state_dim();
  x_ = Eigen::VectorXd::Zero(n);
  P_ = Eigen::MatrixXd::Identity(n, n) * 100.0;
  last_innov_xyz_ = Eigen::VectorXd::Zero(3);
}

// ═══════════════════════════════════════════════════════════════════
// Reset
// ═══════════════════════════════════════════════════════════════════

void InvariantPoseBackend::reset(const ObservationData &obs, int panel_id,
                                  double r1, double r2, double dza) {
  current_panel_id_ = ((panel_id % 4) + 4) % 4;
  phase_index_ = current_panel_id_;

  x_ = initialize_invariant_state(obs, current_panel_id_, r1, r2, dza);
  P_ = motion_->initial_covariance();

  auto idx = motion_->state_idx();
  const auto pp = get_panel_profile(current_panel_id_);
  double center_yaw = normalize_angle(obs.yaw - pp.phase_offset);
  k_ = phase_index_;
  last_k_ = phase_index_;
  x_(idx.DELTA()) = center_yaw;

  last_innov_xyz_ = Eigen::VectorXd::Zero(3);
  last_innov_yaw_ = 0.0;
  last_nis_ = -1.0;
  last_update_type_ = 0;

  if (structure_) structure_->reset(r1, r2, dza);

  initialized_ = true;
}

// ═══════════════════════════════════════════════════════════════════
// Predict
// ═══════════════════════════════════════════════════════════════════

void InvariantPoseBackend::predict(double dt) {
  if (!initialized_) return;
  last_update_type_ = 0;
  last_nis_ = -1.0;

  int n = motion_->state_dim();
  auto idx = motion_->state_idx();

  // ── Mean propagation (left-invariant nominal dynamics) ──
  // R_{k+1} = R_k · Exp(e_z · β · dt)   ⇔  yaw += yaw_rate · dt
  // p_{k+1} = p_k + v_k · dt  (+ ½·a·dt² for CA)
  // v_{k+1} = v_k  (+ a·dt for CA)
  // β_{k+1} = β_k
  // θ_{k+1} = θ_k
  Eigen::VectorXd x_pred = x_;
  const bool ca_xyz = idx.has("AX") && idx.has("AY") && idx.has("AZ");
  if (ca_xyz) {
    x_pred(idx.X()) += x_(idx.VX()) * dt + 0.5 * x_(idx.AX()) * dt * dt;
    x_pred(idx.Y()) += x_(idx.VY()) * dt + 0.5 * x_(idx.AY()) * dt * dt;
    x_pred(idx.Z()) += x_(idx.VZ()) * dt + 0.5 * x_(idx.AZ()) * dt * dt;
    x_pred(idx.VX()) += x_(idx.AX()) * dt;
    x_pred(idx.VY()) += x_(idx.AY()) * dt;
    x_pred(idx.VZ()) += x_(idx.AZ()) * dt;
  } else {
    x_pred(idx.X()) += x_(idx.VX()) * dt;
    x_pred(idx.Y()) += x_(idx.VY()) * dt;
    x_pred(idx.Z()) += x_(idx.VZ()) * dt;
  }

  if (idx.has("DELTA_ACC")) {
    x_pred(idx.DELTA()) = normalize_angle(
        x_(idx.DELTA()) + x_(idx.DELTA_RATE()) * dt +
        0.5 * x_(idx.get("DELTA_ACC")) * dt * dt);
    x_pred(idx.DELTA_RATE()) += x_(idx.get("DELTA_ACC")) * dt;
  } else {
    x_pred(idx.DELTA()) =
        normalize_angle(x_(idx.DELTA()) + x_(idx.DELTA_RATE()) * dt);
  }

  // ── Error-state transition matrix F (n × n) ──
  // For left-invariant error with world-frame velocity, F is identity + dt terms.
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n);
  F(idx.X(), idx.VX()) = dt;
  F(idx.Y(), idx.VY()) = dt;
  F(idx.Z(), idx.VZ()) = dt;
  F(idx.DELTA(), idx.DELTA_RATE()) = dt;

  if (ca_xyz) {
    F(idx.X(), idx.AX()) = 0.5 * dt * dt;
    F(idx.Y(), idx.AY()) = 0.5 * dt * dt;
    F(idx.Z(), idx.AZ()) = 0.5 * dt * dt;
    F(idx.VX(), idx.AX()) = dt;
    F(idx.VY(), idx.AY()) = dt;
    F(idx.VZ(), idx.AZ()) = dt;
  }

  if (idx.has("DELTA_ACC")) {
    const int dacc = idx.get("DELTA_ACC");
    F(idx.DELTA(), dacc) = 0.5 * dt * dt;
    F(idx.DELTA_RATE(), dacc) = dt;
  }

  // ── Covariance propagation ──
  // P lives on the Lie algebra (error-state covariance).
  Eigen::MatrixXd Q = build_invariant_Q(dt);
  P_ = F * P_ * F.transpose() + Q;

  x_ = x_pred;
  apply_state_constraints();
  P_ = ensure_positive_definite(P_);
}

// ═══════════════════════════════════════════════════════════════════
// PredictContext
// ═══════════════════════════════════════════════════════════════════

PredictContext InvariantPoseBackend::buildPredictContext() const {
  PredictContext ctx;
  ctx.x_prior = x_;
  ctx.P_prior = P_;
  ctx.k_prior = k_;
  ctx.last_k_prior = last_k_;
  ctx.hybrid_prior.panel_id = current_panel_id_;
  ctx.hybrid_prior.phase_index = phase_index_;
  return ctx;
}

// ═══════════════════════════════════════════════════════════════════
// Observation model (world-frame prediction)
// ═══════════════════════════════════════════════════════════════════

Eigen::Vector4d InvariantPoseBackend::obs_model_single(
    const Eigen::VectorXd &x, int k, int panel_id) const {
  (void)k;
  auto idx = motion_->state_idx();
  double x_c = x(idx.X());
  double y_c = x(idx.Y());
  double z_mean = x(idx.Z());

  const auto pp = get_panel_profile(panel_id);
  double radius, d_za;
  if (structure_ && structure_->converged()) {
    Eigen::Vector3d s = structure_->get_structure();
    radius = pp.use_r2 ? s(1) : s(0);
    d_za = s(2);
  } else {
    radius = pp.use_r2 ? x(idx.R2()) : x(idx.R1());
    d_za = x(idx.DZA());
  }
  double center_yaw = normalize_angle(x(idx.DELTA()));

  // Observation yaw is detector yaw plus pi; in this internal convention the
  // observed armor is on the positive radial direction from the estimated center.
  const double armor_yaw = normalize_angle(center_yaw + pp.phase_offset);
  const double x_obs = x_c + radius * std::cos(armor_yaw);
  const double y_obs = y_c + radius * std::sin(armor_yaw);
  const double z_obs = z_mean + pp.z_sign * d_za;

  Eigen::Vector4d z;
  z << x_obs, y_obs, z_obs, center_yaw;
  return z;
}

// ═══════════════════════════════════════════════════════════════════
// World-frame analytical Jacobian (4 × n)
// ═══════════════════════════════════════════════════════════════════

Eigen::MatrixXd InvariantPoseBackend::obs_jacobian_single_world(
    const Eigen::VectorXd &x, int k, int panel_id) const {
  (void)k;
  int n = motion_->state_dim();
  auto idx = motion_->state_idx();
  const int p = ((panel_id % 4) + 4) % 4;

  const auto pp = get_panel_profile(p);
  double radius = pp.use_r2 ? x(idx.R2()) : x(idx.R1());
  double center_yaw = normalize_angle(x(idx.DELTA()));
  double armor_yaw = normalize_angle(center_yaw + pp.phase_offset);

  double cos_a = std::cos(armor_yaw);
  double sin_a = std::sin(armor_yaw);

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, n);

  // Row 0: ∂x_obs/∂state  —  world-frame x
  H(0, idx.X()) = 1.0;
  H(0, idx.DELTA()) = -radius * sin_a;
  if (!pp.use_r2) {
    H(0, idx.R1()) = cos_a;
  } else {
    H(0, idx.R2()) = cos_a;
  }

  // Row 1: ∂y_obs/∂state  —  world-frame y
  H(1, idx.Y()) = 1.0;
  H(1, idx.DELTA()) = radius * cos_a;
  if (!pp.use_r2) {
    H(1, idx.R1()) = sin_a;
  } else {
    H(1, idx.R2()) = sin_a;
  }

  // Row 2: ∂z_obs/∂state  —  world-frame z
  H(2, idx.Z()) = 1.0;
  H(2, idx.DZA()) = pp.z_sign;

  // Row 3: ∂yaw_obs/∂state  —  center_yaw
  H(3, idx.DELTA()) = 1.0;

  return H;
}

// ═══════════════════════════════════════════════════════════════════
// Left-invariant innovation (body-frame)
//   ν_body = [R̂⁻¹·(p_obs − p_pred);  wrap(yaw_obs − yaw_pred)]
// ═══════════════════════════════════════════════════════════════════

Eigen::Vector4d InvariantPoseBackend::compute_left_invariant_innovation(
    const Eigen::Vector4d &z_obs, const Eigen::Vector4d &z_pred,
    double center_yaw_pred) const {
  Eigen::Vector4d innov;

  // Position residual in world frame
  double dx = z_obs(0) - z_pred(0);
  double dy = z_obs(1) - z_pred(1);

  // Rotate position residual to body frame: R_z(-center_yaw) · [dx, dy]ᵀ
  double cos_cy = std::cos(center_yaw_pred);
  double sin_cy = std::sin(center_yaw_pred);
  innov(0) =  cos_cy * dx + sin_cy * dy;
  innov(1) = -sin_cy * dx + cos_cy * dy;

  // z component: invariant under rotation about z-axis
  innov(2) = z_obs(2) - z_pred(2);

  // yaw: angle difference (invariant on SO(2))
  innov(3) = angle_difference(z_obs(3), z_pred(3));

  return innov;
}

// ═══════════════════════════════════════════════════════════════════
// Body-frame H Jacobian
//   Apply R̂⁻¹ to position rows; simplify yaw/radius derivatives to
//   body-frame form (depend only on panel phase φ, not center_yaw).
// ═══════════════════════════════════════════════════════════════════

Eigen::MatrixXd InvariantPoseBackend::compute_body_frame_H(
    const Eigen::MatrixXd &H_world, double center_yaw_pred, int panel_id,
    const Eigen::VectorXd &x) const {
  auto idx = motion_->state_idx();

  double cos_cy = std::cos(center_yaw_pred);
  double sin_cy = std::sin(center_yaw_pred);

  const auto pp = get_panel_profile(panel_id);

  // Use slow structure estimates for radius when converged
  double radius;
  if (structure_ && structure_->converged()) {
    Eigen::Vector3d s = structure_->get_structure();
    radius = pp.use_r2 ? s(1) : s(0);
  } else {
    radius = pp.use_r2 ? x(idx.R2()) : x(idx.R1());
  }

  // Start from world-frame H, rotate position rows
  Eigen::MatrixXd H_body = H_world;

  // Row 0 (body-frame x): cos(cy)*H(0,:) + sin(cy)*H(1,:)
  // This simplifies: H_body(0,DELTA) = -r·sin(φ), H_body(0,R1/R2) = cos(φ)
  H_body.row(0) = cos_cy * H_world.row(0) + sin_cy * H_world.row(1);

  // Row 1 (body-frame y): -sin(cy)*H(0,:) + cos(cy)*H(1,:)
  // This simplifies: H_body(1,DELTA) = r·cos(φ), H_body(1,R1/R2) = sin(φ)
  H_body.row(1) = -sin_cy * H_world.row(0) + cos_cy * H_world.row(1);

  // ── Replace yaw/radius derivatives with simplified body-frame form ──
  // These no longer depend on center_yaw — only on panel phase φ.
  double cos_phi = std::cos(pp.phase_offset);
  double sin_phi = std::sin(pp.phase_offset);

  // Row 0: ∂(body_x)/∂DELTA = -r·sin(φ),  ∂(body_x)/∂r = cos(φ)
  H_body(0, idx.DELTA()) = -radius * sin_phi;
  if (!pp.use_r2) {
    H_body(0, idx.R1()) = cos_phi;
    H_body(0, idx.R2()) = 0.0;
  } else {
    H_body(0, idx.R1()) = 0.0;
    H_body(0, idx.R2()) = cos_phi;
  }

  // Row 1: ∂(body_y)/∂DELTA = r·cos(φ),  ∂(body_y)/∂r = sin(φ)
  H_body(1, idx.DELTA()) = radius * cos_phi;
  if (!pp.use_r2) {
    H_body(1, idx.R1()) = sin_phi;
    H_body(1, idx.R2()) = 0.0;
  } else {
    H_body(1, idx.R1()) = 0.0;
    H_body(1, idx.R2()) = sin_phi;
  }

  return H_body;
}

// ═══════════════════════════════════════════════════════════════════
// Body-frame R matrix
//   R_body = Ad_Rz(−cy) · R_world · Ad_Rz(−cy)ᵀ
// where Ad_Rz(−cy) = blkdiag(R_z(−cy), 1, 1) on [x, y, z, yaw]
// ═══════════════════════════════════════════════════════════════════

Eigen::Matrix4d InvariantPoseBackend::rotate_R_to_body_frame(
    const Eigen::Matrix4d &R_world, double center_yaw_pred) const {
  double cos_cy = std::cos(center_yaw_pred);
  double sin_cy = std::sin(center_yaw_pred);

  // Rotation matrix on [x, y] subspace
  // Rz^T = [cos,  sin]   (applied to position block)
  //        [-sin, cos]
  //
  // Full 4×4 transformation:
  //   X_body = [cos,  sin, 0, 0] · X_world
  //            [-sin, cos, 0, 0]
  //            [0,    0,   1, 0]
  //            [0,    0,   0, 1]
  //
  //   R_body = Ad · R_world · Adᵀ

  Eigen::Matrix4d R_body = R_world;

  // Apply rotation to the 2×2 position block: R_body_xy = Rz^T · R_world_xy · Rz
  // This is equivalent to rotating the (x,y) rows and columns.
  // Rows 0,1: rotate by Rz^T
  Eigen::Matrix4d R_temp = R_world;
  for (int col = 0; col < 4; ++col) {
    double vx = R_world(0, col);
    double vy = R_world(1, col);
    R_temp(0, col) =  cos_cy * vx + sin_cy * vy;
    R_temp(1, col) = -sin_cy * vx + cos_cy * vy;
  }
  // Columns 0,1: rotate by Rz (transpose of Rz^T)
  for (int row = 0; row < 4; ++row) {
    double vx = R_temp(row, 0);
    double vy = R_temp(row, 1);
    R_body(row, 0) = cos_cy * vx + sin_cy * vy;
    R_body(row, 1) = -sin_cy * vx + cos_cy * vy;
  }

  return R_body;
}

// ═══════════════════════════════════════════════════════════════════
// evaluateSingle  —  body-frame invariant innovation
// ═══════════════════════════════════════════════════════════════════

MeasurementEval InvariantPoseBackend::evaluateSingle(
    const PredictContext &ctx, const ObservationData &obs,
    int panel_id) const {
  const int p = ((panel_id % 4) + 4) % 4;

  // Build observation in "center_yaw" coordinates
  double center_yaw_obs = normalize_angle(obs.yaw - p * (M_PI / 2.0));
  Eigen::Vector4d z_obs;
  z_obs << obs.x, obs.y, obs.z, center_yaw_obs;

  // World-frame prediction
  Eigen::Vector4d z_pred =
      obs_model_single(ctx.x_prior, ctx.k_prior, p);

  // World-frame Jacobian
  Eigen::MatrixXd H_world =
      obs_jacobian_single_world(ctx.x_prior, ctx.k_prior, p);

  // ── Body-frame innovation (left-invariant) ──
  double center_yaw_pred = z_pred(3);
  Eigen::Vector4d innov = compute_left_invariant_innovation(
      z_obs, z_pred, center_yaw_pred);

  // ── Body-frame H ──
  Eigen::MatrixXd H_body = compute_body_frame_H(
      H_world, center_yaw_pred, p, ctx.x_prior);

  // ── Body-frame R ──
  Eigen::Matrix4d R_world = noise_->build_single_R(obs);
  Eigen::Matrix4d R_body = rotate_R_to_body_frame(R_world, center_yaw_pred);

  // ── Innovation covariance S = H·P·Hᵀ + R  (all in body frame) ──
  Eigen::Matrix4d S = H_body * ctx.P_prior * H_body.transpose() + R_body;

  MeasurementEval eval;
  Eigen::LLT<Eigen::Matrix4d> llt(S);
  if (llt.info() != Eigen::Success) {
    eval.valid = false;
    eval.reject_reason = "S_not_spd";
    return eval;
  }
  auto solved = llt.solve(innov);
  double nis = innov.dot(solved);

  double logdet = 0.0;
  const auto &L = llt.matrixL();
  for (int i = 0; i < 4; ++i) logdet += std::log(L(i, i));
  logdet *= 2.0;
  double log_likelihood =
      -0.5 * (nis + logdet + 4.0 * std::log(2.0 * M_PI));

  // Per-component chi2 (body-frame)
  double chi2_yaw = (innov(3) * innov(3)) / S(3, 3);
  Eigen::Vector3d innov_pos = innov.head<3>();
  Eigen::Matrix3d S_pos = S.topLeftCorner<3, 3>();
  double chi2_pos =
      innov_pos.transpose() * S_pos.inverse() * innov_pos;

  const auto &gt = ukf_config_.gate;
  bool gate_pass = true;
  if (nis > gt.single_total_nis) gate_pass = false;
  if (chi2_pos > gt.single_pos_chi2) gate_pass = false;
  if (chi2_yaw > gt.single_yaw_chi2) gate_pass = false;

  eval.valid = true;
  eval.gate_pass = gate_pass;
  eval.nis = nis;
  eval.mahalanobis = std::sqrt(nis);
  eval.log_likelihood = log_likelihood;
  eval.score = log_likelihood;
  eval.chi2_pos = chi2_pos;
  eval.chi2_yaw = chi2_yaw;
  eval.innovation = innov;
  eval.S = S;
  eval.z_pred = z_pred;
  eval.z_obs = z_obs;

  if (!gate_pass) {
    std::ostringstream oss;
    oss << "gate_fail:nis=" << nis << ",chi2_pos=" << chi2_pos
        << ",chi2_yaw=" << chi2_yaw;
    eval.reject_reason = oss.str();
  }
  return eval;
}

// ═══════════════════════════════════════════════════════════════════
// evaluateDual  —  body-frame invariant innovation (dual observation)
// ═══════════════════════════════════════════════════════════════════

MeasurementEval InvariantPoseBackend::evaluateDual(
    const PredictContext &ctx, const ObservationData &obs0,
    const ObservationData &obs1, int panel_id_0, int panel_id_1) const {
  const int p0 = ((panel_id_0 % 4) + 4) % 4;
  const int p1 = ((panel_id_1 % 4) + 4) % 4;

  double cy0_obs = normalize_angle(obs0.yaw - p0 * (M_PI / 2.0));
  double cy1_obs = normalize_angle(obs1.yaw - p1 * (M_PI / 2.0));
  Eigen::Matrix<double, 8, 1> z_obs;
  z_obs << obs0.x, obs0.y, obs0.z, cy0_obs, obs1.x, obs1.y, obs1.z, cy1_obs;

  Eigen::Vector4d zp0 =
      obs_model_single(ctx.x_prior, ctx.k_prior, p0);
  Eigen::Vector4d zp1 =
      obs_model_single(ctx.x_prior, ctx.k_prior, p1);
  Eigen::Matrix<double, 8, 1> z_pred;
  z_pred << zp0, zp1;

  // World-frame Jacobians
  Eigen::MatrixXd H0_world =
      obs_jacobian_single_world(ctx.x_prior, ctx.k_prior, p0);
  Eigen::MatrixXd H1_world =
      obs_jacobian_single_world(ctx.x_prior, ctx.k_prior, p1);

  int n = motion_->state_dim();

  // ── Body-frame innovation for each panel ──
  double cy0_pred = zp0(3);
  double cy1_pred = zp1(3);
  Eigen::Vector4d innov0 = compute_left_invariant_innovation(
      z_obs.head<4>(), zp0, cy0_pred);
  Eigen::Vector4d innov1 = compute_left_invariant_innovation(
      z_obs.tail<4>(), zp1, cy1_pred);
  Eigen::Matrix<double, 8, 1> innov;
  innov << innov0, innov1;

  // ── Body-frame H for each panel ──
  Eigen::MatrixXd H0_body = compute_body_frame_H(
      H0_world, cy0_pred, p0, ctx.x_prior);
  Eigen::MatrixXd H1_body = compute_body_frame_H(
      H1_world, cy1_pred, p1, ctx.x_prior);
  Eigen::MatrixXd H_body(8, n);
  H_body << H0_body, H1_body;

  // ── Body-frame R ──
  Eigen::Matrix<double, 8, 8> R_world = noise_->build_dual_R(obs0, obs1);
  Eigen::Matrix4d R0_body = rotate_R_to_body_frame(
      R_world.topLeftCorner<4, 4>(), cy0_pred);
  Eigen::Matrix4d R1_body = rotate_R_to_body_frame(
      R_world.bottomRightCorner<4, 4>(), cy1_pred);
  Eigen::Matrix<double, 8, 8> R_body = Eigen::Matrix<double, 8, 8>::Zero();
  R_body.topLeftCorner<4, 4>() = R0_body;
  R_body.bottomRightCorner<4, 4>() = R1_body;

  Eigen::Matrix<double, 8, 8> S = H_body * ctx.P_prior * H_body.transpose() + R_body;

  MeasurementEval eval;
  Eigen::LLT<Eigen::Matrix<double, 8, 8>> llt(S);
  if (llt.info() != Eigen::Success) {
    eval.valid = false;
    eval.reject_reason = "S_not_spd";
    return eval;
  }
  auto solved = llt.solve(innov);
  double nis = innov.dot(solved);

  double logdet = 0.0;
  const auto &L = llt.matrixL();
  for (int i = 0; i < 8; ++i) logdet += std::log(L(i, i));
  logdet *= 2.0;
  double log_likelihood =
      -0.5 * (nis + logdet + 8.0 * std::log(2.0 * M_PI));

  double chi2_yaw0 = (innov(3) * innov(3)) / S(3, 3);
  double chi2_yaw1 = (innov(7) * innov(7)) / S(7, 7);
  double chi2_yaw = std::max(chi2_yaw0, chi2_yaw1);

  Eigen::Vector3d ip0 = innov.segment<3>(0);
  Eigen::Vector3d ip1 = innov.segment<3>(4);
  Eigen::Matrix3d Sp0 = S.block<3, 3>(0, 0);
  Eigen::Matrix3d Sp1 = S.block<3, 3>(4, 4);
  double chi2_pos0 = ip0.transpose() * Sp0.inverse() * ip0;
  double chi2_pos1 = ip1.transpose() * Sp1.inverse() * ip1;
  double chi2_pos = std::max(chi2_pos0, chi2_pos1);

  const auto &gt = ukf_config_.gate;
  bool gate_pass = true;
  if (nis > gt.dual_total_nis) gate_pass = false;
  if (chi2_pos0 > gt.dual_each_pos_chi2 ||
      chi2_pos1 > gt.dual_each_pos_chi2)
    gate_pass = false;
  if (chi2_yaw0 > gt.dual_each_yaw_chi2 ||
      chi2_yaw1 > gt.dual_each_yaw_chi2)
    gate_pass = false;

  eval.valid = true;
  eval.gate_pass = gate_pass;
  eval.nis = nis;
  eval.mahalanobis = std::sqrt(nis);
  eval.log_likelihood = log_likelihood;
  eval.score = log_likelihood;
  eval.chi2_pos = chi2_pos;
  eval.chi2_yaw = chi2_yaw;
  eval.innovation = innov;
  eval.S = S;
  eval.z_pred = z_pred;
  eval.z_obs = z_obs;

  if (!gate_pass) {
    std::ostringstream oss;
    oss << "gate_fail:nis=" << nis << ",chi2_pos=" << chi2_pos
        << ",chi2_yaw=" << chi2_yaw;
    eval.reject_reason = oss.str();
  }
  return eval;
}

// ═══════════════════════════════════════════════════════════════════
// tryUpdateSingle
// ═══════════════════════════════════════════════════════════════════

UkfTrial InvariantPoseBackend::tryUpdateSingle(
    const PredictContext &ctx, const ObservationData &obs,
    int panel_id) const {
  UkfTrial trial;
  const int p = ((panel_id % 4) + 4) % 4;

  trial.hypothesis.kind = HypothesisKind::Single;
  trial.hypothesis.assignments[0] = {0, p};
  trial.hypothesis.assignment_count = 1;

  MeasurementEval eval = evaluateSingle(ctx, obs, p);
  trial.eval = eval;
  if (!eval.valid) {
    trial.reject_reason = eval.reject_reason;
    return trial;
  }

  int n = motion_->state_dim();
  auto idx = motion_->state_idx();

  // ── Recompute body-frame H (matches evaluateSingle) ──
  double center_yaw_pred = eval.z_pred(3);
  Eigen::MatrixXd H_world =
      obs_jacobian_single_world(ctx.x_prior, ctx.k_prior, p);
  Eigen::MatrixXd H_body = compute_body_frame_H(
      H_world, center_yaw_pred, p, ctx.x_prior);

  // Innovation is already in body frame from evaluateSingle
  Eigen::Vector4d innov = eval.innovation;

  // ── Use body-frame R for Kalman gain ──
  Eigen::Matrix4d R_world = noise_->build_single_R(obs);
  Eigen::Matrix4d R_body = rotate_R_to_body_frame(R_world, center_yaw_pred);

  // S is already computed in body frame
  Eigen::Matrix4d S = eval.S;

  // Kalman gain: K = P·Hᵀ·S⁻¹  (n × 4)  — all in body frame
  Eigen::MatrixXd K = ctx.P_prior * H_body.transpose() * S.inverse();

  // Structural slow gain
  const auto &su = ukf_config_.single_update;
  K.row(idx.R1()) *= su.structural_gain_r;
  K.row(idx.R2()) *= su.structural_gain_r;
  K.row(idx.DZA()) *= su.structural_gain_dza;

  // Error-state correction on Lie algebra, then retract to nominal state
  Eigen::VectorXd dx = K * innov;
  Eigen::VectorXd x_post = retract_left_invariant(ctx.x_prior, dx);

  // Joseph form covariance update
  Eigen::MatrixXd I_KH =
      Eigen::MatrixXd::Identity(n, n) - K * H_body;
  Eigen::MatrixXd P_post =
      I_KH * ctx.P_prior * I_KH.transpose() + K * R_body * K.transpose();
  P_post = 0.5 * (P_post + P_post.transpose());
  P_post = ensure_positive_definite(P_post, 1e-6);

  trial.success = true;
  trial.x_post = x_post;
  trial.P_post = P_post;
  trial.k_post = p;
  trial.last_k_post = ctx.k_prior;
  trial.hybrid_post.panel_id = p;
  trial.hybrid_post.phase_index = p;

  trial.reconstruction_pos_error =
      compute_reconstruction_error(x_post, trial.k_post, obs, p);
  trial.posterior_sanity_pass =
      check_posterior_sanity(ctx.x_prior, x_post, P_post);

  if (!trial.posterior_sanity_pass) {
    trial.reject_reason = "posterior_sanity_fail";
    trial.success = false;
  }

  // Clamp structure params
  trial.x_post(idx.R1()) =
      std::clamp(trial.x_post(idx.R1()), 0.05, 0.50);
  trial.x_post(idx.R2()) =
      std::clamp(trial.x_post(idx.R2()), 0.05, 0.50);
  trial.x_post(idx.DZA()) =
      std::clamp(trial.x_post(idx.DZA()), 0.0, 0.15);

  return trial;
}

// ═══════════════════════════════════════════════════════════════════
// tryUpdateDual
// ═══════════════════════════════════════════════════════════════════

UkfTrial InvariantPoseBackend::tryUpdateDual(
    const PredictContext &ctx, const ObservationData &obs0,
    const ObservationData &obs1, int panel_id_0, int panel_id_1) const {
  UkfTrial trial;
  trial.hypothesis.kind = HypothesisKind::Dual;
  trial.hypothesis.assignments[0] = {0, panel_id_0};
  trial.hypothesis.assignments[1] = {1, panel_id_1};
  trial.hypothesis.assignment_count = 2;

  MeasurementEval eval =
      evaluateDual(ctx, obs0, obs1, panel_id_0, panel_id_1);
  trial.eval = eval;
  if (!eval.valid) {
    trial.reject_reason = eval.reject_reason;
    return trial;
  }

  const int p0 = ((panel_id_0 % 4) + 4) % 4;
  const int p1 = ((panel_id_1 % 4) + 4) % 4;
  int n = motion_->state_dim();
  auto idx = motion_->state_idx();

  // ── Recompute body-frame H (dual) ──
  double cy0_pred = eval.z_pred(3);
  double cy1_pred = eval.z_pred(7);
  Eigen::MatrixXd H0_world =
      obs_jacobian_single_world(ctx.x_prior, ctx.k_prior, p0);
  Eigen::MatrixXd H1_world =
      obs_jacobian_single_world(ctx.x_prior, ctx.k_prior, p1);
  Eigen::MatrixXd H0_body = compute_body_frame_H(
      H0_world, cy0_pred, p0, ctx.x_prior);
  Eigen::MatrixXd H1_body = compute_body_frame_H(
      H1_world, cy1_pred, p1, ctx.x_prior);
  Eigen::MatrixXd H_body(8, n);
  H_body << H0_body, H1_body;

  Eigen::Matrix<double, 8, 1> innov = eval.innovation;
  Eigen::Matrix<double, 8, 8> R_world = noise_->build_dual_R(obs0, obs1);
  Eigen::Matrix4d R0_body = rotate_R_to_body_frame(
      R_world.topLeftCorner<4, 4>(), cy0_pred);
  Eigen::Matrix4d R1_body = rotate_R_to_body_frame(
      R_world.bottomRightCorner<4, 4>(), cy1_pred);
  Eigen::Matrix<double, 8, 8> R_body = Eigen::Matrix<double, 8, 8>::Zero();
  R_body.topLeftCorner<4, 4>() = R0_body;
  R_body.bottomRightCorner<4, 4>() = R1_body;

  Eigen::Matrix<double, 8, 8> S = eval.S;

  // Kalman gain: K = P·Hᵀ·S⁻¹  (n × 8)
  Eigen::MatrixXd K = ctx.P_prior * H_body.transpose() * S.inverse();

  // Structural slow gain (dual: more permissive)
  const auto &du = ukf_config_.dual_update;
  K.row(idx.R1()) *= du.structural_gain_r;
  K.row(idx.R2()) *= du.structural_gain_r;
  K.row(idx.DZA()) *= du.structural_gain_dza;

  Eigen::VectorXd dx = K * innov;
  Eigen::VectorXd x_post = retract_left_invariant(ctx.x_prior, dx);

  // Joseph form
  Eigen::MatrixXd I_KH =
      Eigen::MatrixXd::Identity(n, n) - K * H_body;
  Eigen::MatrixXd P_post =
      I_KH * ctx.P_prior * I_KH.transpose() + K * R_body * K.transpose();
  P_post = 0.5 * (P_post + P_post.transpose());
  P_post = ensure_positive_definite(P_post, 1e-6);

  trial.success = true;
  trial.x_post = x_post;
  trial.P_post = P_post;
  trial.k_post = p0;
  trial.last_k_post = ctx.k_prior;
  trial.hybrid_post.panel_id = p0;
  trial.hybrid_post.phase_index = p0;

  double recon0 =
      compute_reconstruction_error(x_post, trial.k_post, obs0, p0);
  double recon1 =
      compute_reconstruction_error(x_post, trial.k_post, obs1, p1);
  trial.reconstruction_pos_error = std::max(recon0, recon1);
  trial.posterior_sanity_pass =
      check_posterior_sanity(ctx.x_prior, x_post, P_post);

  if (!trial.posterior_sanity_pass) {
    trial.reject_reason = "posterior_sanity_fail";
    trial.success = false;
  }

  trial.x_post(idx.R1()) =
      std::clamp(trial.x_post(idx.R1()), 0.05, 0.50);
  trial.x_post(idx.R2()) =
      std::clamp(trial.x_post(idx.R2()), 0.05, 0.50);
  trial.x_post(idx.DZA()) =
      std::clamp(trial.x_post(idx.DZA()), 0.0, 0.15);

  return trial;
}

// ═══════════════════════════════════════════════════════════════════
// commit  —  apply trial with hybrid state as first-class citizen
// ═══════════════════════════════════════════════════════════════════

void InvariantPoseBackend::commit(const UkfTrial &trial) {
  if (!trial.success) return;

  x_ = trial.x_post;
  P_ = trial.P_post;
  k_ = trial.k_post;
  last_k_ = trial.last_k_post;

  // ── Hybrid discrete state as first-class commit contract ──
  phase_index_ = trial.hybrid_post.phase_index;
  if (trial.hypothesis.kind == HypothesisKind::Single) {
    current_panel_id_ = trial.hypothesis.assignments[0].panel_id;
  } else {
    current_panel_id_ = trial.hypothesis.assignments[0].panel_id;
  }

  last_nis_ = trial.eval.nis;
  if (trial.eval.innovation.size() >= 3) {
    last_innov_xyz_ = trial.eval.innovation.head<3>();
  }
  if (trial.eval.innovation.size() >= 4) {
    last_innov_yaw_ = trial.eval.innovation(3);
  }
  last_update_type_ =
      (trial.hypothesis.kind == HypothesisKind::Single) ? 1 : 2;

  // Update slow structure estimator with dual-obs flag
  if (structure_) {
    bool is_dual = (trial.hypothesis.kind == HypothesisKind::Dual);
    structure_->update(x_, P_, k_, dt_, is_dual);
  }

  apply_state_constraints();
  P_ = ensure_positive_definite(P_);
}

// ═══════════════════════════════════════════════════════════════════
// snapshot  —  full state with hybrid discrete state
// ═══════════════════════════════════════════════════════════════════

BackendSnapshot InvariantPoseBackend::snapshot() const {
  BackendSnapshot snap;
  snap.x = x_;
  snap.P = P_;
  snap.k = k_;
  snap.last_k = last_k_;
  snap.current_panel_id = current_panel_id_;
  snap.hybrid.panel_id = current_panel_id_;
  snap.hybrid.phase_index = phase_index_;
  snap.last_nis = last_nis_;
  snap.last_innov_xyz = last_innov_xyz_;
  snap.last_innov_yaw = last_innov_yaw_;
  snap.last_update_type = last_update_type_;
  return snap;
}

// ═══════════════════════════════════════════════════════════════════
// SpinFilterInterface accessors
// ═══════════════════════════════════════════════════════════════════

Eigen::Vector3d InvariantPoseBackend::get_center_position() const {
  auto idx = motion_->state_idx();
  return Eigen::Vector3d(x_(idx.X()), x_(idx.Y()), x_(idx.Z()));
}

std::pair<double, double> InvariantPoseBackend::get_radii() const {
  if (structure_ && structure_->converged()) {
    Eigen::Vector3d s = structure_->get_structure();
    return {s(0), s(1)};
  }
  auto idx = motion_->state_idx();
  return {x_(idx.R1()), x_(idx.R2())};
}

double InvariantPoseBackend::get_dza() const {
  if (structure_ && structure_->converged()) {
    return structure_->get_structure()(2);
  }
  return x_(motion_->state_idx().DZA());
}

double InvariantPoseBackend::get_yaw() const {
  return normalize_angle(x_(motion_->state_idx().DELTA()));
}

double InvariantPoseBackend::get_delta() const {
  return x_(motion_->state_idx().DELTA());
}

// ═══════════════════════════════════════════════════════════════════
// Posterior sanity check
// ═══════════════════════════════════════════════════════════════════

bool InvariantPoseBackend::check_posterior_sanity(
    const Eigen::VectorXd &x_prior, const Eigen::VectorXd &x_post,
    const Eigen::MatrixXd &P_post) const {
  auto idx = motion_->state_idx();
  const auto &ps = ukf_config_.posterior_sanity;

  Eigen::Vector3d prior_center(x_prior(idx.X()), x_prior(idx.Y()),
                                x_prior(idx.Z()));
  Eigen::Vector3d post_center(x_post(idx.X()), x_post(idx.Y()),
                               x_post(idx.Z()));
  if ((post_center - prior_center).norm() > ps.max_center_jump) return false;

  double delta_jump = std::abs(
      angle_difference(x_post(idx.DELTA()), x_prior(idx.DELTA())));
  if (delta_jump > ps.max_yaw_jump) return false;

  const double max_yaw_rate = std::max(0.1, config_.outpost.max_yaw_rate);
  if (idx.has("DELTA_RATE") &&
      (!std::isfinite(x_post(idx.DELTA_RATE())) ||
       std::abs(x_post(idx.DELTA_RATE())) > max_yaw_rate)) {
    return false;
  }
  if (idx.has("DELTA_ACC")) {
    const double max_yaw_acc =
        std::max(0.1, config_.outpost.v3_posterior_max_yaw_acc);
    const int dacc = idx.get("DELTA_ACC");
    if (!std::isfinite(x_post(dacc)) || std::abs(x_post(dacc)) > max_yaw_acc) {
      return false;
    }
  }

  double r1 = x_post(idx.R1()), r2 = x_post(idx.R2());
  if (r1 < ps.min_r || r1 > ps.max_r || r2 < ps.min_r || r2 > ps.max_r)
    return false;
  if (std::abs(r1 - x_prior(idx.R1())) > ps.max_r_jump) return false;
  if (std::abs(r2 - x_prior(idx.R2())) > ps.max_r_jump) return false;

  double dza = x_post(idx.DZA());
  if (dza < ps.min_dza || dza > ps.max_dza) return false;
  if (std::abs(dza - x_prior(idx.DZA())) > ps.max_dza_jump) return false;

  if (!P_post.allFinite()) return false;
  return true;
}

double InvariantPoseBackend::compute_reconstruction_error(
    const Eigen::VectorXd &x_post, int k, const ObservationData &obs,
    int panel_id) const {
  Eigen::Vector4d z_rebuild = obs_model_single(x_post, k, panel_id);
  Eigen::Vector3d pos_rebuild = z_rebuild.head<3>();
  Eigen::Vector3d pos_obs(obs.x, obs.y, obs.z);
  return (pos_obs - pos_rebuild).norm();
}

// ═══════════════════════════════════════════════════════════════════
// Left-invariant retraction:  X̂⁺ = Exp(δξ̂) ∘ X̂
//   SO(2):  yaw⁺ = normalize_angle(yaw + δψ)
//   ℝⁿ:    x⁺ = x + δx
// ═══════════════════════════════════════════════════════════════════

Eigen::VectorXd InvariantPoseBackend::retract_left_invariant(
    const Eigen::VectorXd &x_prior, const Eigen::VectorXd &dx) const {
  const auto idx = motion_->state_idx();
  Eigen::VectorXd x_post = x_prior;

  // SE(2) part: p and R_yaw
  // Position: additive (world-frame, left-invariant for ℝⁿ part)
  x_post(idx.X()) += dx(idx.X());
  x_post(idx.Y()) += dx(idx.Y());
  x_post(idx.Z()) += dx(idx.Z());

  // Rotation: Exp(δψ) · R̂ ⇔ yaw⁺ = normalize_angle(yaw + δψ)
  // For SO(2), the Exp map is exactly addition with wrap.
  x_post(idx.DELTA()) = normalize_angle(x_prior(idx.DELTA()) + dx(idx.DELTA()));

  // Euclidean part: velocities, accelerations
  if (idx.has("VX")) x_post(idx.VX()) += dx(idx.VX());
  if (idx.has("VY")) x_post(idx.VY()) += dx(idx.VY());
  if (idx.has("VZ")) x_post(idx.VZ()) += dx(idx.VZ());
  if (idx.has("DELTA_RATE")) x_post(idx.DELTA_RATE()) += dx(idx.DELTA_RATE());
  if (idx.has("AX")) x_post(idx.AX()) += dx(idx.AX());
  if (idx.has("AY")) x_post(idx.AY()) += dx(idx.AY());
  if (idx.has("AZ")) x_post(idx.AZ()) += dx(idx.AZ());
  if (idx.has("DELTA_ACC")) x_post(idx.get("DELTA_ACC")) += dx(idx.get("DELTA_ACC"));

  // Slow structural state (θ): Euclidean
  x_post(idx.R1()) += dx(idx.R1());
  x_post(idx.R2()) += dx(idx.R2());
  x_post(idx.DZA()) += dx(idx.DZA());

  return x_post;
}

// ═══════════════════════════════════════════════════════════════════
// State initialization from observation
// ═══════════════════════════════════════════════════════════════════

Eigen::VectorXd InvariantPoseBackend::initialize_invariant_state(
    const ObservationData &obs, int panel_id, double r1, double r2,
    double dza) const {
  const auto idx = motion_->state_idx();
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(motion_->state_dim());
  const auto pp = get_panel_profile(panel_id);
  const double use_r = pp.use_r2 ? r2 : r1;
  const double center_yaw = normalize_angle(obs.yaw - pp.phase_offset);

  x0(idx.X()) = obs.x - use_r * std::cos(obs.yaw);
  x0(idx.Y()) = obs.y - use_r * std::sin(obs.yaw);
  x0(idx.Z()) = obs.z - pp.z_sign * dza;
  x0(idx.DELTA()) = center_yaw;
  x0(idx.R1()) = r1;
  x0(idx.R2()) = r2;
  x0(idx.DZA()) = dza;

  if (idx.has("VX")) x0(idx.VX()) = 0.0;
  if (idx.has("VY")) x0(idx.VY()) = 0.0;
  if (idx.has("VZ")) x0(idx.VZ()) = 0.0;
  if (idx.has("DELTA_RATE")) x0(idx.DELTA_RATE()) = 0.0;
  if (idx.has("AX")) x0(idx.AX()) = 0.0;
  if (idx.has("AY")) x0(idx.AY()) = 0.0;
  if (idx.has("AZ")) x0(idx.AZ()) = 0.0;
  if (idx.has("DELTA_ACC")) x0(idx.get("DELTA_ACC")) = 0.0;
  return x0;
}

// ═══════════════════════════════════════════════════════════════════
// Process noise covariance (on Lie algebra / error state)
// ═══════════════════════════════════════════════════════════════════

Eigen::MatrixXd InvariantPoseBackend::build_invariant_Q(double dt) const {
  const auto idx = motion_->state_idx();
  const int n = motion_->state_dim();
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, n);
  const double dt2 = dt * dt;

  const bool ca_xyz = idx.has("AX") && idx.has("AY") && idx.has("AZ");
  const bool ca_yaw = idx.has("DELTA_ACC");
  const double q_v_cv = config_.motion.cv_process_noise_vel *
                        config_.motion.cv_process_noise_vel;
  const double q_v_ca = config_.motion.ca_process_noise_acc *
                        config_.motion.ca_process_noise_acc;
  const double q_w_cv = config_.spin.spin_process_noise_delta_rate *
                        config_.spin.spin_process_noise_delta_rate;
  const double q_w_ca = config_.spin.spin_process_noise_delta_acc *
                        config_.spin.spin_process_noise_delta_acc;
  const double q_r = config_.motion.process_noise_r *
                     config_.motion.process_noise_r;
  const double q_dza = config_.motion.process_noise_dz *
                       config_.motion.process_noise_dz;

  if (ca_xyz) {
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    const double dt5 = dt4 * dt;
    auto fill_ca_block = [&](int p, int v, int a) {
      Q(p, p) = q_v_ca * dt5 / 20.0;
      Q(p, v) = q_v_ca * dt4 / 8.0;
      Q(v, p) = Q(p, v);
      Q(p, a) = q_v_ca * dt3 / 6.0;
      Q(a, p) = Q(p, a);
      Q(v, v) = q_v_ca * dt3 / 3.0;
      Q(v, a) = q_v_ca * dt2 / 2.0;
      Q(a, v) = Q(v, a);
      Q(a, a) = q_v_ca * dt;
    };
    fill_ca_block(idx.X(), idx.VX(), idx.AX());
    fill_ca_block(idx.Y(), idx.VY(), idx.AY());
    fill_ca_block(idx.Z(), idx.VZ(), idx.AZ());
  } else {
    Q(idx.X(), idx.X()) = std::max(1e-8, q_v_cv * dt2);
    Q(idx.Y(), idx.Y()) = std::max(1e-8, q_v_cv * dt2);
    Q(idx.Z(), idx.Z()) = std::max(1e-8, q_v_cv * dt2);
    Q(idx.VX(), idx.VX()) = std::max(1e-8, q_v_cv * dt);
    Q(idx.VY(), idx.VY()) = std::max(1e-8, q_v_cv * dt);
    Q(idx.VZ(), idx.VZ()) = std::max(1e-8, q_v_cv * dt);
  }

  if (ca_yaw) {
    const int dacc = idx.get("DELTA_ACC");
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    const double dt5 = dt4 * dt;
    Q(idx.DELTA(), idx.DELTA()) = q_w_ca * dt5 / 20.0;
    Q(idx.DELTA(), idx.DELTA_RATE()) = q_w_ca * dt4 / 8.0;
    Q(idx.DELTA_RATE(), idx.DELTA()) = Q(idx.DELTA(), idx.DELTA_RATE());
    Q(idx.DELTA(), dacc) = q_w_ca * dt3 / 6.0;
    Q(dacc, idx.DELTA()) = Q(idx.DELTA(), dacc);
    Q(idx.DELTA_RATE(), idx.DELTA_RATE()) = q_w_ca * dt3 / 3.0;
    Q(idx.DELTA_RATE(), dacc) = q_w_ca * dt2 / 2.0;
    Q(dacc, idx.DELTA_RATE()) = Q(idx.DELTA_RATE(), dacc);
    Q(dacc, dacc) = q_w_ca * dt;
  } else {
    Q(idx.DELTA(), idx.DELTA()) = std::max(1e-8, q_w_cv * dt2);
    Q(idx.DELTA_RATE(), idx.DELTA_RATE()) = std::max(1e-8, q_w_cv * dt);
  }
  Q(idx.R1(), idx.R1()) = std::max(1e-10, q_r * dt);
  Q(idx.R2(), idx.R2()) = std::max(1e-10, q_r * dt);
  Q(idx.DZA(), idx.DZA()) = std::max(1e-10, q_dza * dt);

  return Q;
}

void InvariantPoseBackend::apply_state_constraints() {
  auto idx = motion_->state_idx();
  x_ = fyt::auto_aim::apply_state_constraints(
      x_, idx.R1(), idx.R2(), idx.DZA(), config_.constraints.min_radius,
      config_.constraints.max_radius, 0.0, config_.constraints.max_dz);
  x_(idx.DELTA()) = normalize_angle(x_(idx.DELTA()));
  if (idx.has("DELTA_RATE")) {
    const double max_yaw_rate = std::max(0.1, config_.outpost.max_yaw_rate);
    x_(idx.DELTA_RATE()) =
        std::clamp(x_(idx.DELTA_RATE()), -max_yaw_rate, max_yaw_rate);
  }
  if (idx.has("DELTA_ACC")) {
    const double max_yaw_acc =
        std::max(0.1, config_.outpost.v3_posterior_max_yaw_acc);
    const int dacc = idx.get("DELTA_ACC");
    x_(dacc) = std::clamp(x_(dacc), -max_yaw_acc, max_yaw_acc);
  }
}

}  // namespace fyt::auto_aim::norm4_v3
