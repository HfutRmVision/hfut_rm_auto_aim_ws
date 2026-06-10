#include "armor_detector_nn/core/pose_refine/pose_refiner.hpp"

#include <algorithm>
#include <chrono>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <rm_utils/logger/log.hpp>

namespace fyt::auto_aim {

namespace {

constexpr double kDefaultDt = 1.0 / 30.0;  // assumed 30 fps

inline Eigen::Matrix3d yawPitchRollToImuMatrix(
    double yaw, double pitch, double roll) {
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cr = std::cos(roll), sr = std::sin(roll);
  Eigen::Matrix3d R_imu_armor;
  R_imu_armor <<
    cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,
    sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,
    -sp,      cp * sr,                 cp * cr;
  return R_imu_armor;
}

cv::Mat yawPitchRollToMatrix(
    double yaw, double pitch, double roll, const Eigen::Matrix3d& R_imu_camera) {
  const Eigen::Matrix3d R_imu_armor = yawPitchRollToImuMatrix(yaw, pitch, roll);
  const Eigen::Matrix3d R_camera_armor = R_imu_camera.transpose() * R_imu_armor;
  cv::Mat R;
  cv::eigen2cv(R_camera_armor, R);
  return R;
}

cv::Mat yawPitchRollToRvec(
    double yaw, double pitch, double roll, const Eigen::Matrix3d& R_imu_camera) {
  cv::Mat R = yawPitchRollToMatrix(yaw, pitch, roll, R_imu_camera);
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  return rvec;
}

// Compute per-point reprojection error (no Huber).
double perPointError(const cv::Vec3d& P_cam, double fx, double fy,
                     double cx, double cy,
                     const cv::Point2f& measured) {
  if (P_cam[2] <= 1e-8) return 1e9;
  double inv_z = 1.0 / P_cam[2];
  double u = fx * P_cam[0] * inv_z + cx;
  double v = fy * P_cam[1] * inv_z + cy;
  double du = measured.x - u;
  double dv = measured.y - v;
  return du * du + dv * dv;
}

double initialYawFromRotationLikeArmorDetector(
    const cv::Mat& rvec, const Eigen::Matrix3d& R_imu_camera) {
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);
  Eigen::Matrix3d R_camera_armor;
  cv::cv2eigen(R_cv, R_camera_armor);
  Eigen::Matrix3d R_imu_armor = R_imu_camera * R_camera_armor;

  double yaw = 0.0;
  auto theta_by_sin = std::asin(-R_imu_armor(0, 1));
  auto theta_by_cos = std::acos(std::clamp(R_imu_armor(1, 1), -1.0, 1.0));
  if (std::abs(theta_by_sin) > 1e-5) {
    yaw = theta_by_sin > 0 ? theta_by_cos : -theta_by_cos;
  } else {
    yaw = R_imu_armor(1, 1) > 0 ? 0.0 : CV_PI;
  }
  return yaw;
}

}  // namespace

SlidingWindowRefiner::SlidingWindowRefiner(
    const SlidingWindowConfig& sw_config,
    const SingleYawConfig& sy_config,
    const GateConfig& gate)
  : sw_config_(sw_config),
    sy_config_(sy_config),
    gate_(gate),
    fallback_refiner_(sy_config, gate)
{
}

double SlidingWindowRefiner::angleWrap(double angle) {
  while (angle > M_PI)  angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

SlidingWindowState& SlidingWindowRefiner::getWindow(int track_id) {
  auto it = windows_.find(track_id);
  if (it == windows_.end()) {
    auto [inserted, _] = windows_.emplace(
      track_id, SlidingWindowState(sw_config_.window_size));
    inserted->second.setMinFrames(sw_config_.min_frames);
    frame_counters_[track_id] = 0;
    return inserted->second;
  }
  return it->second;
}

void SlidingWindowRefiner::pushFrame(int track_id,
                                     const TrackWindowFrame& frame) {
  auto& win = getWindow(track_id);
  win.pushFrame(frame);
}

void SlidingWindowRefiner::removeTrack(int track_id) {
  windows_.erase(track_id);
  frame_counters_.erase(track_id);
}

size_t SlidingWindowRefiner::activeTrackCount() const {
  return windows_.size();
}

bool SlidingWindowRefiner::solveSlidingWindow(
    SlidingWindowState& window,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    Eigen::Vector3d& t_out,
    double& yaw_out,
    double& final_error)
{
  const auto solve_start = std::chrono::steady_clock::now();
  const double max_solver_time_ms = std::max(0.1, sw_config_.max_solver_time_ms);
  window.trimToMaxSpanMs(sw_config_.max_time_span_ms);
  const size_t N = window.size();
  if (N < static_cast<size_t>(sw_config_.min_frames)) {
    return false;
  }

  // Check time span (use frame counter as proxy if no real stamps).
  double span_ms = window.timeSpanMs();
  if (span_ms < 0.1) {
    // No real timestamps — estimate from frame count and assumed fps.
    span_ms = (N - 1) * kDefaultDt * 1000.0;
  }
  if (span_ms > sw_config_.max_time_span_ms) {
    FYT_DEBUG("armor_detector_nn",
              "SlidingWindowRefiner: time span too large ({:.1f} ms)", span_ms);
    return false;
  }

  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  // --- Initialize state vector from PnP results ---
  int state_dim = static_cast<int>(N) * 4;
  Eigen::VectorXd x(state_dim);
  for (size_t k = 0; k < N; ++k) {
    const auto& f = window.frames()[k];
    x[4 * static_cast<int>(k) + 0] = f.t_init.x();
    x[4 * static_cast<int>(k) + 1] = f.t_init.y();
    x[4 * static_cast<int>(k) + 2] = f.t_init.z();
    x[4 * static_cast<int>(k) + 3] = f.yaw_init;
  }

  // Pre-compute prior weights
  double wx_prior = 1.0 / (sw_config_.sigma_prior_xy * sw_config_.sigma_prior_xy);
  double wy_prior = wx_prior;
  double wz_prior = 1.0 / (sw_config_.sigma_prior_z * sw_config_.sigma_prior_z);
  double wyaw_prior = 1.0 / (sw_config_.sigma_prior_yaw * sw_config_.sigma_prior_yaw);

  // Smoothness weights (scaled by 1/dt so tighter coupling at high frame rate).
  double dt_smooth = kDefaultDt;
  double ws_xy  = dt_smooth / (sw_config_.sigma_smooth_xy * sw_config_.sigma_smooth_xy);
  double ws_z   = dt_smooth / (sw_config_.sigma_smooth_z * sw_config_.sigma_smooth_z);
  double ws_yaw = dt_smooth / (sw_config_.sigma_smooth_yaw * sw_config_.sigma_smooth_yaw);

  double huber_delta_sq = sw_config_.huber_delta * sw_config_.huber_delta;

  int executed_iters = 0;
  for (int iter = 0; iter < sw_config_.max_opt_iters; ++iter) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(now - solve_start).count();
    if (elapsed_ms > max_solver_time_ms) {
      FYT_DEBUG("armor_detector_nn",
                "SlidingWindowRefiner: solver budget exceeded ({:.2f} ms), early stop at iter {}",
                elapsed_ms, executed_iters);
      break;
    }
    ++executed_iters;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(state_dim);
    double total_cost = 0.0;

    // ---- 1. Reprojection edges ----
    for (size_t k = 0; k < N; ++k) {
      const auto& frame = window.frames()[k];
      int idx = 4 * static_cast<int>(k);

      double tx = x[idx + 0], ty = x[idx + 1], tz = x[idx + 2];
      double yaw = x[idx + 3];

      cv::Mat R_wc = yawPitchRollToMatrix(
        yaw, frame.pitch, frame.roll, frame.R_imu_camera);
      constexpr double kYawDiffEps = 1e-5;
      cv::Mat R_plus = yawPitchRollToMatrix(
        yaw + kYawDiffEps, frame.pitch, frame.roll, frame.R_imu_camera);
      cv::Mat R_minus = yawPitchRollToMatrix(
        yaw - kYawDiffEps, frame.pitch, frame.roll, frame.R_imu_camera);

      for (int i = 0; i < 4; ++i) {
        cv::Vec3d P_obj(object_points[i].x, object_points[i].y,
                        object_points[i].z);
        cv::Mat P_cam_mat = R_wc * cv::Mat(P_obj);
        double X_c = P_cam_mat.at<double>(0) + tx;
        double Y_c = P_cam_mat.at<double>(1) + ty;
        double Z_c = P_cam_mat.at<double>(2) + tz;

        if (Z_c <= 1e-8) {
          // Penalize negative depth strongly
          H(idx + 2, idx + 2) += 1000.0;
          b(idx + 2) += 1000.0 * 1.0;
          continue;
        }

        double inv_z = 1.0 / Z_c;
        double u_proj = fx * X_c * inv_z + cx;
        double v_proj = fy * Y_c * inv_z + cy;

        double du = frame.keypoints[i].x - u_proj;
        double dv = frame.keypoints[i].y - v_proj;
        double r2 = du * du + dv * dv;

        // Confidence-weighted Huber loss
        double conf = std::clamp(frame.keypoint_conf[i], 0.1f, 1.0f);
        double sigma_kp = sw_config_.sigma_kp_min +
          (1.0 - conf) * sw_config_.sigma_kp_scale;
        double w_kp = 1.0 / (sigma_kp * sigma_kp);

        double rho = 1.0;
        if (r2 > huber_delta_sq) {
          rho = sw_config_.huber_delta / std::sqrt(r2);
        }
        double weight = w_kp * rho;

        total_cost += weight * r2;

        // Jacobian of projection w.r.t [tx, ty, tz, yaw]
        // du/dtx = fx/z,   du/dty = 0,      du/dtz = -fx*x/z²
        // dv/dtx = 0,      dv/dty = fy/z,   dv/dtz = -fy*y/z²
        double J00 =  fx * inv_z;               // du/dtx
        double J01 =  0.0;                      // du/dty
        double J02 = -fx * X_c * inv_z * inv_z; // du/dtz
        double J10 =  0.0;                      // dv/dtx
        double J11 =  fy * inv_z;               // dv/dty
        double J12 = -fy * Y_c * inv_z * inv_z; // dv/dtz

        cv::Mat P_plus_mat = R_plus * cv::Mat(P_obj);
        cv::Mat P_minus_mat = R_minus * cv::Mat(P_obj);
        cv::Vec3d P_plus(P_plus_mat.at<double>(0) + tx,
                         P_plus_mat.at<double>(1) + ty,
                         P_plus_mat.at<double>(2) + tz);
        cv::Vec3d P_minus(P_minus_mat.at<double>(0) + tx,
                          P_minus_mat.at<double>(1) + ty,
                          P_minus_mat.at<double>(2) + tz);
        if (P_plus[2] <= 1e-8 || P_minus[2] <= 1e-8) continue;
        double uv_plus_u = fx * P_plus[0] / P_plus[2] + cx;
        double uv_plus_v = fy * P_plus[1] / P_plus[2] + cy;
        double uv_minus_u = fx * P_minus[0] / P_minus[2] + cx;
        double uv_minus_v = fy * P_minus[1] / P_minus[2] + cy;
        double J03 = (uv_plus_u - uv_minus_u) / (2.0 * kYawDiffEps);
        double J13 = (uv_plus_v - uv_minus_v) / (2.0 * kYawDiffEps);

        // Accumulate H += w * J^T * J  (symmetric)
        H(idx + 0, idx + 0) += weight * (J00 * J00 + J10 * J10);
        H(idx + 0, idx + 1) += weight * (J00 * J01 + J10 * J11);
        H(idx + 0, idx + 2) += weight * (J00 * J02 + J10 * J12);
        H(idx + 0, idx + 3) += weight * (J00 * J03 + J10 * J13);

        H(idx + 1, idx + 1) += weight * (J01 * J01 + J11 * J11);
        H(idx + 1, idx + 2) += weight * (J01 * J02 + J11 * J12);
        H(idx + 1, idx + 3) += weight * (J01 * J03 + J11 * J13);

        H(idx + 2, idx + 2) += weight * (J02 * J02 + J12 * J12);
        H(idx + 2, idx + 3) += weight * (J02 * J03 + J12 * J13);

        H(idx + 3, idx + 3) += weight * (J03 * J03 + J13 * J13);

        // b += w * J^T * r
        b(idx + 0) += weight * (J00 * du + J10 * dv);
        b(idx + 1) += weight * (J01 * du + J11 * dv);
        b(idx + 2) += weight * (J02 * du + J12 * dv);
        b(idx + 3) += weight * (J03 * du + J13 * dv);
      }
    }

    // ---- 2. PnP prior edges (weak, prevents depth drift) ----
    for (size_t k = 0; k < N; ++k) {
      const auto& frame = window.frames()[k];
      int idx = 4 * static_cast<int>(k);

      H(idx + 0, idx + 0) += wx_prior;
      H(idx + 1, idx + 1) += wy_prior;
      H(idx + 2, idx + 2) += wz_prior;
      H(idx + 3, idx + 3) += wyaw_prior;

      b(idx + 0) += wx_prior * (frame.t_init.x() - x[idx + 0]);
      b(idx + 1) += wy_prior * (frame.t_init.y() - x[idx + 1]);
      b(idx + 2) += wz_prior * (frame.t_init.z() - x[idx + 2]);

      double dyaw = angleWrap(x[idx + 3] - frame.yaw_init);
      b(idx + 3) += wyaw_prior * (-dyaw);
    }

    // ---- 3. Inter-frame smoothness edges ----
    for (size_t k = 1; k < N; ++k) {
      int i0 = 4 * static_cast<int>(k - 1);
      int i1 = 4 * static_cast<int>(k);

      for (int d = 0; d < 3; ++d) {
        double ws = (d == 2) ? ws_z : ws_xy;
        H(i0 + d, i0 + d) += ws;
        H(i0 + d, i1 + d) -= ws;
        H(i1 + d, i0 + d) -= ws;
        H(i1 + d, i1 + d) += ws;
        b(i0 + d) += ws * (x[i1 + d] - x[i0 + d]);
        b(i1 + d) -= ws * (x[i1 + d] - x[i0 + d]);
      }

      // Yaw smoothness with wrap handling
      double dyaw = angleWrap(x[i1 + 3] - x[i0 + 3]);
      H(i0 + 3, i0 + 3) += ws_yaw;
      H(i0 + 3, i1 + 3) -= ws_yaw;
      H(i1 + 3, i0 + 3) -= ws_yaw;
      H(i1 + 3, i1 + 3) += ws_yaw;
      b(i0 + 3) += ws_yaw * dyaw;
      b(i1 + 3) -= ws_yaw * dyaw;
    }

    // Fill in symmetric upper triangle
    for (int r = 0; r < state_dim; ++r) {
      for (int c = r + 1; c < state_dim; ++c) {
        H(c, r) = H(r, c);
      }
    }

    // ---- 4. Solve and update ----
    Eigen::VectorXd dx = H.ldlt().solve(b);

    // Line search
    double alpha = 1.0;
    for (int ls = 0; ls < 8; ++ls) {
      Eigen::VectorXd x_try = x + alpha * dx;
      // Normalize yaw angles
      for (size_t k = 0; k < N; ++k) {
        int i3 = 4 * static_cast<int>(k) + 3;
        x_try[i3] = std::atan2(std::sin(x_try[i3]), std::cos(x_try[i3]));
      }

      // Quick cost check on the latest frame only (heuristic)
      // Full cost recomputation is too expensive.
      // Accept if step is not too large.
      if (dx.norm() * alpha < 0.5) {
        x = x_try;
        break;
      }
      alpha *= 0.5;
      if (ls == 7) x = x + alpha * dx;  // accept anyway
    }
    if (alpha < 1e-8) {
      x = x + 1e-8 * dx;
    }

    // Normalize all yaw angles
    for (size_t k = 0; k < N; ++k) {
      int i3 = 4 * static_cast<int>(k) + 3;
      x[i3] = std::atan2(std::sin(x[i3]), std::cos(x[i3]));
    }

    if (dx.norm() < 1e-6) break;
  }

  if (executed_iters == 0) {
    return false;
  }

  // Return latest frame result
  size_t last = N - 1;
  int li = 4 * static_cast<int>(last);
  t_out = Eigen::Vector3d(x[li + 0], x[li + 1], x[li + 2]);
  yaw_out = x[li + 3];

  if (!std::isfinite(t_out.norm()) || !std::isfinite(yaw_out)) {
    return false;
  }

  // Final reprojection error (latest frame only, for quality gating)
  const auto& lf = window.frames()[last];
  cv::Mat R_final = yawPitchRollToMatrix(
    yaw_out, lf.pitch, lf.roll, lf.R_imu_camera);
  Eigen::Vector3d t_eigen = t_out;
  cv::Vec3d tvec_final(t_eigen.x(), t_eigen.y(), t_eigen.z());
  final_error = 0.0;
  for (int i = 0; i < 4; ++i) {
    cv::Vec3d P_obj(object_points[i].x, object_points[i].y, object_points[i].z);
    cv::Mat P_cam_mat = R_final * cv::Mat(P_obj);
    cv::Vec3d P_cam(P_cam_mat.at<double>(0) + tvec_final[0],
                    P_cam_mat.at<double>(1) + tvec_final[1],
                    P_cam_mat.at<double>(2) + tvec_final[2]);
    final_error += perPointError(P_cam, fx, fy, cx, cy,
                                  lf.keypoints[i]);
  }
  final_error /= 4.0;

  return true;
}

PoseEstimate SlidingWindowRefiner::refine(
    const PoseEstimate& pnp_result,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    const cv::Mat& D)
{
  int track_id = pnp_result.track_id;
  if (track_id < 0) {
    // No track yet — fall back to single-yaw BA
    return fallback_refiner_.refine(pnp_result, image_points,
                                    object_points, K, D);
  }

  // Build a TrackWindowFrame from the current PnP result.
  TrackWindowFrame frame;
  frame.track_id = track_id;
  frame.keypoints = image_points;
  frame.t_init = Eigen::Vector3d(
    pnp_result.tvec.at<double>(0),
    pnp_result.tvec.at<double>(1),
    pnp_result.tvec.at<double>(2));
  frame.R_imu_camera = pnp_result.R_imu_camera;
  frame.yaw_init = initialYawFromRotationLikeArmorDetector(
    pnp_result.rvec, frame.R_imu_camera);
  frame.pitch = pnp_result.pitch;
  if (!std::isfinite(frame.pitch)) {
    frame.pitch = sy_config_.pitch_deg_default * M_PI / 180.0;
    if (sy_config_.outpost_pitch_sign && pnp_result.publish_number == "outpost") {
      frame.pitch = -frame.pitch;
    }
  }
  frame.roll = pnp_result.roll;
  if (!std::isfinite(frame.roll)) {
    frame.roll = sy_config_.roll_deg_default * M_PI / 180.0;
  }
  // Prefer real observation timestamp; fallback to frame-counter proxy.
  if (pnp_result.observation_stamp.nanoseconds() > 0) {
    frame.stamp = pnp_result.observation_stamp;
  } else {
    frame.stamp = rclcpp::Time(
      static_cast<int64_t>(frame_counters_[track_id] * kDefaultDt * 1e9));
  }

  // Push to per-track window
  auto& window = getWindow(track_id);
  window.pushFrame(frame);
  frame_counters_[track_id]++;

  // Not enough frames yet — try single-yaw as intermediate output
  if (!window.isReady()) {
    return fallback_refiner_.refine(pnp_result, image_points,
                                    object_points, K, D);
  }

  // Solve sliding-window BA
  Eigen::Vector3d t_ba;
  double yaw_ba;
  double final_error = 0.0;
  bool ok = solveSlidingWindow(window, object_points, K,
                               t_ba, yaw_ba, final_error);

  // --- Fallback chain ---
  if (!ok) {
    FYT_DEBUG("armor_detector_nn",
              "SlidingWindowRefiner: BA failed for track {}, fallback to single_yaw",
              track_id);
    return fallback_refiner_.refine(pnp_result, image_points,
                                    object_points, K, D);
  }

  // Quality gate
  if (final_error > gate_.max_reproj_error) {
    FYT_DEBUG("armor_detector_nn",
              "SlidingWindowRefiner: reproj error {:.2f} > {:.2f}, falling back",
              final_error, gate_.max_reproj_error);
    return fallback_refiner_.refine(pnp_result, image_points,
                                    object_points, K, D);
  }

  Eigen::Vector3d t_pnp(pnp_result.tvec.at<double>(0),
                        pnp_result.tvec.at<double>(1),
                        pnp_result.tvec.at<double>(2));
  double pose_delta = (t_ba - t_pnp).norm();
  if (pose_delta > gate_.max_pose_delta_m) {
    FYT_DEBUG("armor_detector_nn",
              "SlidingWindowRefiner: pose delta {:.3f} m > {:.3f} m, falling back",
              pose_delta, gate_.max_pose_delta_m);
    return fallback_refiner_.refine(pnp_result, image_points,
                                    object_points, K, D);
  }

  double delta_yaw = std::abs(angleWrap(yaw_ba - frame.yaw_init));
  if (delta_yaw > gate_.max_yaw_delta_deg * M_PI / 180.0) {
    FYT_DEBUG("armor_detector_nn",
              "SlidingWindowRefiner: yaw delta {:.2f} deg, falling back",
              delta_yaw * 180.0 / M_PI);
    return fallback_refiner_.refine(pnp_result, image_points,
                                    object_points, K, D);
  }

  // --- Success: build refined PoseEstimate ---
  PoseEstimate result = pnp_result;
  result.yaw = yaw_ba;
  result.rvec = yawPitchRollToRvec(
    yaw_ba, frame.pitch, frame.roll, frame.R_imu_camera);
  result.tvec = (cv::Mat_<double>(3, 1) << t_ba.x(), t_ba.y(), t_ba.z());
  result.translation = t_ba;
  result.reproj_error_refined = final_error;
  result.mode = EstimateMode::SW_BA_VALID;
  result.quality_score = std::max(0.0, 1.0 - final_error / gate_.max_reproj_error);

  cv::Mat R;
  cv::Rodrigues(result.rvec, R);
  Eigen::Matrix3d eigen_R;
  cv::cv2eigen(R, eigen_R);
  result.rotation = Eigen::Quaterniond(eigen_R);

  return result;
}

}  // namespace fyt::auto_aim
