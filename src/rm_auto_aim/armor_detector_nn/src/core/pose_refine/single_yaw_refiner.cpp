#include "armor_detector_nn/core/pose_refine/pose_refiner.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <rm_utils/logger/log.hpp>

namespace fyt::auto_aim {

namespace {

inline Eigen::Matrix3d yawPitchRollToGimbalMatrix(
    double yaw, double pitch, double roll) {
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cr = std::cos(roll), sr = std::sin(roll);
  Eigen::Matrix3d R_gimbal_armor;
  R_gimbal_armor <<
    cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,
    sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,
    -sp,      cp * sr,                 cp * cr;
  return R_gimbal_armor;
}

// Build camera-frame rotation using BaSolver-consistent semantics:
// R_gimbal_armor = Rz(yaw_imu) * Ry(pitch_imu) * Rx(roll_imu)
// R_camera_armor = R_gimbal_camera^T * R_gimbal_armor
cv::Mat yawPitchRollToMatrix(
    double yaw, double pitch, double roll, const Eigen::Matrix3d& R_imu_camera) {
  const Eigen::Matrix3d R_imu_armor = yawPitchRollToGimbalMatrix(yaw, pitch, roll);
  const Eigen::Matrix3d R_camera_armor = R_imu_camera.transpose() * R_imu_armor;

  cv::Mat R;
  cv::eigen2cv(R_camera_armor, R);
  return R;
}

// Convert yaw/pitch/roll to a Rodrigues rotation vector.
cv::Mat yawPitchRollToRvec(
    double yaw, double pitch, double roll, const Eigen::Matrix3d& R_imu_camera) {
  cv::Mat R = yawPitchRollToMatrix(yaw, pitch, roll, R_imu_camera);
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  return rvec;
}

// Project a 3D point into the image plane.
inline cv::Point2d projectPoint(const cv::Vec3d& P_cam, double fx, double fy,
                                double cx, double cy) {
  double inv_z = 1.0 / P_cam[2];
  return {fx * P_cam[0] * inv_z + cx,
          fy * P_cam[1] * inv_z + cy};
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

SingleYawRefiner::SingleYawRefiner(const SingleYawConfig& config,
                                   const GateConfig& gate)
  : config_(config), gate_(gate)
{
}

double SingleYawRefiner::angleWrap(double angle) {
  while (angle > M_PI)  angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

double SingleYawRefiner::computeReprojError(
    double yaw,
    const cv::Vec3d& tvec,
    double pitch,
    double roll,
    const Eigen::Matrix3d& R_imu_camera,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K)
{
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  cv::Mat R_wc = yawPitchRollToMatrix(yaw, pitch, roll, R_imu_camera);
  double total = 0.0;

  for (int i = 0; i < 4; ++i) {
    cv::Vec3d P_obj(object_points[i].x, object_points[i].y, object_points[i].z);
    cv::Mat P_cam_mat = R_wc * cv::Mat(P_obj) + cv::Mat(tvec);
    cv::Vec3d P_cam(P_cam_mat.at<double>(0), P_cam_mat.at<double>(1), P_cam_mat.at<double>(2));

    if (P_cam[2] <= 1e-6) return 1e9;

    cv::Point2d proj = projectPoint(P_cam, fx, fy, cx, cy);
    double dx = image_points[i].x - proj.x;
    double dy = image_points[i].y - proj.y;
    double r2 = dx * dx + dy * dy;

    // Huber loss
    if (r2 > config_.huber_delta * config_.huber_delta) {
      r2 = 2.0 * config_.huber_delta * std::sqrt(r2)
           - config_.huber_delta * config_.huber_delta;
    }
    total += r2;
  }

  return total;
}

bool SingleYawRefiner::optimizeYaw(
    double& yaw,
    const cv::Vec3d& tvec,
    double pitch,
    double roll,
    const Eigen::Matrix3d& R_imu_camera,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K)
{
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < config_.max_iterations; ++iter) {
    double J_sum = 0.0;
    double b_sum = 0.0;
    double cost = 0.0;

    cv::Mat R_wc = yawPitchRollToMatrix(yaw, pitch, roll, R_imu_camera);
    constexpr double kYawDiffEps = 1e-5;
    cv::Mat R_plus =
      yawPitchRollToMatrix(yaw + kYawDiffEps, pitch, roll, R_imu_camera);
    cv::Mat R_minus =
      yawPitchRollToMatrix(yaw - kYawDiffEps, pitch, roll, R_imu_camera);

    for (int i = 0; i < 4; ++i) {
      cv::Vec3d P_obj(object_points[i].x, object_points[i].y, object_points[i].z);
      cv::Mat P_cam_mat = R_wc * cv::Mat(P_obj) + cv::Mat(tvec);
      cv::Vec3d P_cam(P_cam_mat.at<double>(0), P_cam_mat.at<double>(1), P_cam_mat.at<double>(2));

      if (P_cam[2] <= 1e-6) return false;

      double inv_z = 1.0 / P_cam[2];
      double u_proj = fx * P_cam[0] * inv_z + cx;
      double v_proj = fy * P_cam[1] * inv_z + cy;

      double du = image_points[i].x - u_proj;
      double dv = image_points[i].y - v_proj;
      double r2 = du * du + dv * dv;

      // Huber weight
      double rho = 1.0;
      if (r2 > config_.huber_delta * config_.huber_delta) {
        rho = config_.huber_delta / std::sqrt(r2);
      }
      cost += rho * r2;

      // Numerical Jacobian on yaw to keep consistency with the full
      // R_camera_armor = R_imu_camera^T * Rz(yaw)*Ry(pitch)*Rx(roll) chain.
      cv::Mat P_plus_mat = R_plus * cv::Mat(P_obj) + cv::Mat(tvec);
      cv::Mat P_minus_mat = R_minus * cv::Mat(P_obj) + cv::Mat(tvec);
      cv::Vec3d P_plus(P_plus_mat.at<double>(0),
                       P_plus_mat.at<double>(1),
                       P_plus_mat.at<double>(2));
      cv::Vec3d P_minus(P_minus_mat.at<double>(0),
                        P_minus_mat.at<double>(1),
                        P_minus_mat.at<double>(2));
      if (P_plus[2] <= 1e-6 || P_minus[2] <= 1e-6) return false;
      cv::Point2d uv_plus = projectPoint(P_plus, fx, fy, cx, cy);
      cv::Point2d uv_minus = projectPoint(P_minus, fx, fy, cx, cy);
      double du_dyaw = (uv_plus.x - uv_minus.x) / (2.0 * kYawDiffEps);
      double dv_dyaw = (uv_plus.y - uv_minus.y) / (2.0 * kYawDiffEps);

      J_sum += rho * (du_dyaw * du_dyaw + dv_dyaw * dv_dyaw);
      b_sum += rho * (du_dyaw * du + dv_dyaw * dv);
    }

    if (std::abs(J_sum) < 1e-10) return true;

    double delta = b_sum / J_sum;

    // Line search with step decay
    double alpha = 1.0;
    for (int ls = 0; ls < 10; ++ls) {
      double yaw_try = yaw + alpha * delta;
      double cost_try = computeReprojError(yaw_try, tvec, pitch, roll,
                                           R_imu_camera, image_points,
                                           object_points, K);
      if (cost_try < cost) {
        yaw = yaw_try;
        break;
      }
      alpha *= 0.5;
    }

    if (std::abs(delta) < 1e-6) return true;
  }

  return true;
}

PoseEstimate SingleYawRefiner::refine(
    const PoseEstimate& pnp_result,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    const cv::Mat& /*D*/)
{
  PoseEstimate result = pnp_result;

  // Step 1: determine pitch / roll.
  // Prefer the current PnP orientation to avoid pushing angle-dependent
  // geometry error into tz when the target is observed at large yaw.
  double pitch = pnp_result.pitch;
  if (!std::isfinite(pitch)) {
    pitch = config_.pitch_deg_default * M_PI / 180.0;
    if (config_.outpost_pitch_sign && pnp_result.publish_number == "outpost") {
      pitch = -pitch;
    }
  }
  double roll = pnp_result.roll;
  if (!std::isfinite(roll)) {
    roll = config_.roll_deg_default * M_PI / 180.0;
  }

  // Step 2: yaw initial value from PnP rotation matrix using the
  // same extraction logic as armor_detector::BaSolver.
  double yaw_init = initialYawFromRotationLikeArmorDetector(
    pnp_result.rvec, pnp_result.R_imu_camera);
  double yaw_opt  = yaw_init;

  // Step 3: 1-D yaw optimization
  cv::Vec3d tvec(pnp_result.tvec.at<double>(0),
                 pnp_result.tvec.at<double>(1),
                 pnp_result.tvec.at<double>(2));
  bool ok = optimizeYaw(yaw_opt, tvec, pitch, roll, pnp_result.R_imu_camera,
                        image_points, object_points, K);

  // Step 4: validity checks
  if (!ok || std::isnan(yaw_opt) || !std::isfinite(yaw_opt)) {
    FYT_DEBUG("armor_detector_nn",
              "SingleYawRefiner: optimization failed or produced NaN");
    result.mode = EstimateMode::PNP_VALID;
    result.quality_score = 0.5;
    result.reproj_error_refined = result.reproj_error_raw;
    return result;
  }

  // Step 5: check yaw delta
  double delta_yaw = std::abs(angleWrap(yaw_opt - yaw_init));
  if (delta_yaw > gate_.max_yaw_delta_deg * M_PI / 180.0) {
    FYT_DEBUG("armor_detector_nn",
              "SingleYawRefiner: yaw delta too large (%.2f deg)",
              delta_yaw * 180.0 / M_PI);
    result.mode = EstimateMode::PNP_VALID;
    result.quality_score = 0.5;
    result.reproj_error_refined = result.reproj_error_raw;
    return result;
  }

  // Step 6: quality gate
  double refined_error = computeReprojError(yaw_opt, tvec, pitch, roll,
                                             pnp_result.R_imu_camera,
                                             image_points, object_points, K);
  const double refined_error_per_point = refined_error / 4.0;
  if (gate_.require_finite && !std::isfinite(refined_error)) {
    result.mode = EstimateMode::PNP_VALID;
    result.quality_score = 0.5;
    result.reproj_error_refined = result.reproj_error_raw;
    return result;
  }
  if (refined_error_per_point > gate_.max_reproj_error) {
    result.mode = EstimateMode::PNP_VALID;
    result.quality_score =
      std::max(0.0, 1.0 - refined_error_per_point / gate_.max_reproj_error);
    result.reproj_error_refined = result.reproj_error_raw;
    return result;
  }

  // Step 7: success — update result
  result.yaw = yaw_opt;
  result.pitch = pitch;
  result.roll = roll;
  result.rvec = yawPitchRollToRvec(
    yaw_opt, pitch, roll, pnp_result.R_imu_camera);
  result.mode = EstimateMode::SINGLE_BA_VALID;
  result.reproj_error_refined = refined_error_per_point;
  result.reproj_error_raw = pnp_result.reproj_error_raw;  // preserve
  result.quality_score =
    std::max(0.0, 1.0 - refined_error_per_point / gate_.max_reproj_error);

  // Recompute rotation quaternion and translation
  result.translation = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  cv::Mat R;
  cv::Rodrigues(result.rvec, R);
  Eigen::Matrix3d eigen_R;
  cv::cv2eigen(R, eigen_R);
  result.rotation = Eigen::Quaterniond(eigen_R);

  return result;
}

}  // namespace fyt::auto_aim
