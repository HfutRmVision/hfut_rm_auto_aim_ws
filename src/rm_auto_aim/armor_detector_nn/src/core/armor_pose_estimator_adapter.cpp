#include "armor_detector_nn/core/armor_pose_estimator_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <rm_utils/logger/log.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include "armor_detector_nn/core/pose_refine/pose_refiner.hpp"

namespace fyt::auto_aim {

namespace {

// Match armor_detector::rotationMatrixToRPY() semantics:
// 1) convert rvec->R_camera_armor
// 2) transform by fixed R_gimbal_camera
// 3) tf2 getRPY(roll, pitch, yaw)
void rvecToEulerLikeArmorDetector(
    const cv::Mat& rvec, double& yaw, double& pitch, double& roll) {
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  Eigen::Matrix3d R_camera_armor;
  cv::cv2eigen(R_cv, R_camera_armor);

  Eigen::Matrix3d R_gimbal_camera = Eigen::Matrix3d::Identity();
  R_gimbal_camera << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  Eigen::Matrix3d R_gimbal_armor = R_gimbal_camera * R_camera_armor;

  Eigen::Quaterniond q(R_gimbal_armor);
  tf2::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
  tf2::Matrix3x3 m(tf_q);
  m.getRPY(roll, pitch, yaw);
}

double reprojectionErrorSum(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& K,
    const cv::Mat& D) {
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, K, D, projected);
  double err = 0.0;
  for (size_t j = 0; j < image_points.size(); ++j) {
    err += cv::norm(image_points[j] - projected[j]);
  }
  return err;
}

double objectHeightMeters(const std::vector<cv::Point3f>& object_points) {
  if (object_points.size() < 2) {
    return 0.0;
  }
  const double dx = static_cast<double>(object_points[1].x - object_points[0].x);
  const double dy = static_cast<double>(object_points[1].y - object_points[0].y);
  const double dz = static_cast<double>(object_points[1].z - object_points[0].z);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double lightbarHeightDepth(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points,
    const cv::Mat& K,
    const cv::Mat& D,
    const DepthCorrectionConfig& config) {
  if (image_points.size() < 4 || object_points.size() < 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double left_px = cv::norm(image_points[1] - image_points[0]);
  const double right_px = cv::norm(image_points[2] - image_points[3]);
  if (!std::isfinite(left_px) || !std::isfinite(right_px) ||
      left_px < config.min_lightbar_length_px ||
      right_px < config.min_lightbar_length_px) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double height_m = objectHeightMeters(object_points);
  if (height_m <= 0.0 || !std::isfinite(height_m)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::vector<cv::Point2f> normalized_points;
  cv::undistortPoints(image_points, normalized_points, K, D);
  if (normalized_points.size() < 4) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double left_norm = cv::norm(normalized_points[1] - normalized_points[0]);
  const double right_norm = cv::norm(normalized_points[2] - normalized_points[3]);
  const double mean_norm = 0.5 * (left_norm + right_norm);
  if (mean_norm <= 1e-9 || !std::isfinite(mean_norm)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return height_m / mean_norm;
}

bool applyDepthCorrection(
    PoseEstimate& result,
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points,
    const cv::Mat& K,
    const cv::Mat& D,
    const PoseConfig& pose_config) {
  const auto& config = pose_config.depth_correction;
  if (!config.enabled || !result.valid || result.tvec.empty() ||
      config.blend_alpha <= 0.0 || config.max_scale <= 1.0 ||
      config.max_correction_m <= 0.0) {
    return false;
  }

  const double pnp_z = result.tvec.at<double>(2);
  if (pnp_z <= 0.0 || !std::isfinite(pnp_z)) {
    return false;
  }

  const double height_z = lightbarHeightDepth(
    image_points, object_points, K, D, config);
  if (!std::isfinite(height_z) ||
      height_z <= pnp_z + config.min_depth_delta_m) {
    return false;
  }

  const double capped_z = std::min({
    height_z,
    pnp_z * config.max_scale,
    pnp_z + config.max_correction_m
  });
  const double corrected_z = pnp_z + config.blend_alpha * (capped_z - pnp_z);
  if (corrected_z <= pnp_z || !std::isfinite(corrected_z)) {
    return false;
  }

  const double scale = corrected_z / pnp_z;
  result.tvec.at<double>(0) *= scale;
  result.tvec.at<double>(1) *= scale;
  result.tvec.at<double>(2) *= scale;
  result.translation = Eigen::Vector3d(
    result.tvec.at<double>(0),
    result.tvec.at<double>(1),
    result.tvec.at<double>(2));

  const double corrected_error = reprojectionErrorSum(
    object_points, image_points, result.rvec, result.tvec, K, D);
  if (std::isfinite(corrected_error) && !image_points.empty()) {
    result.reprojection_error = corrected_error;
    result.reproj_error_refined =
      corrected_error / static_cast<double>(image_points.size());
  }
  result.quality_score *= 0.95;

  FYT_DEBUG("armor_detector_nn",
            "DepthCorrection: z %.3f -> %.3f, height_z %.3f, scale %.3f",
            pnp_z, corrected_z, height_z, scale);
  return true;
}

int selectIppeSolutionLikeArmorDetector(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& K,
    const cv::Mat& D,
    const std::string& publish_number) {
  // Baseline fallback: best positive-depth solution by reprojection error.
  int best_pos_z = -1;
  int best_any = -1;
  double best_pos_z_err = std::numeric_limits<double>::max();
  double best_any_err = std::numeric_limits<double>::max();
  std::vector<double> errors(rvecs.size(), std::numeric_limits<double>::max());

  for (size_t i = 0; i < rvecs.size(); ++i) {
    errors[i] = reprojectionErrorSum(object_points, image_points, rvecs[i], tvecs[i], K, D);
    double z = tvecs[i].at<double>(2);
    if (z > 0.0 && errors[i] < best_pos_z_err) {
      best_pos_z_err = errors[i];
      best_pos_z = static_cast<int>(i);
    }
    if (errors[i] < best_any_err) {
      best_any_err = errors[i];
      best_any = static_cast<int>(i);
    }
  }
  int fallback = (best_pos_z >= 0) ? best_pos_z : best_any;
  if (rvecs.size() < 2 || image_points.size() < 4) {
    return fallback;
  }

  constexpr double kProjectErrRatioThres = 3.0;
  // Keep armor_detector's current threshold semantics:
  // it compares radians against (10 * 180 / pi), i.e. effectively disabled.
  constexpr double kRollThresRad = 10.0 * 180.0 / M_PI;
  constexpr double kEps = 1e-9;

  double yaw1 = 0.0, pitch1 = 0.0, roll1 = 0.0;
  double yaw2 = 0.0, pitch2 = 0.0, roll2 = 0.0;
  rvecToEulerLikeArmorDetector(rvecs[0], yaw1, pitch1, roll1);
  rvecToEulerLikeArmorDetector(rvecs[1], yaw2, pitch2, roll2);

  const double err1 = errors[0];
  const double err2 = errors[1];
  if ((err2 / std::max(err1, kEps) > kProjectErrRatioThres) ||
      (std::abs(roll1) > kRollThresRad) ||
      (std::abs(roll2) > kRollThresRad)) {
    return fallback;
  }

  // Canonical order: [0]=left_bottom,[1]=left_top,[2]=right_top,[3]=right_bottom.
  const cv::Point2f left_axis = image_points[1] - image_points[0];
  const cv::Point2f right_axis = image_points[2] - image_points[3];
  double l_angle = std::atan2(left_axis.y, left_axis.x) * 180.0 / M_PI;
  double r_angle = std::atan2(right_axis.y, right_axis.x) * 180.0 / M_PI;
  double angle = (l_angle + r_angle) * 0.5 + 90.0;
  if (publish_number == "outpost") {
    angle = -angle;
  }

  // Match armor_detector::sortPnPResult sign disambiguation behavior.
  if ((angle > 0.0 && yaw1 > 0.0 && yaw2 < 0.0) ||
      (angle < 0.0 && yaw1 < 0.0 && yaw2 > 0.0)) {
    if (tvecs[1].at<double>(2) > 0.0) return 1;
    return fallback;
  }
  if (tvecs[0].at<double>(2) > 0.0) return 0;
  return fallback;
}

}  // namespace

ArmorPoseEstimatorAdapter::ArmorPoseEstimatorAdapter(const PoseConfig& config)
  : config_(config)
{
}

ArmorPoseEstimatorAdapter::~ArmorPoseEstimatorAdapter() = default;

void ArmorPoseEstimatorAdapter::setRefiner(
    std::shared_ptr<IPoseRefiner> refiner) {
  refiner_ = std::move(refiner);
}

PoseEstimate ArmorPoseEstimatorAdapter::estimate(
    const ArmorDetection& detection,
    const sensor_msgs::msg::CameraInfo& camera_info,
    const Eigen::Matrix3d& R_imu_camera)
{
  if (detection.publish_type == "invalid") {
    return PoseEstimate{};
  }

  auto object_pts = getObjectPoints(detection.publish_type,
      config_.small_armor_width, config_.small_armor_height,
      config_.large_armor_width, config_.large_armor_height);

  std::vector<cv::Point2f> image_pts(detection.keypoints.begin(),
                                      detection.keypoints.end());

  cv::Mat K = (cv::Mat_<double>(3, 3)
    << camera_info.k[0], camera_info.k[1], camera_info.k[2],
       camera_info.k[3], camera_info.k[4], camera_info.k[5],
       camera_info.k[6], camera_info.k[7], camera_info.k[8]);

  cv::Mat D;
  if (!camera_info.d.empty()) {
    D = cv::Mat(camera_info.d, true).reshape(1, 1);
  } else {
    D = cv::Mat::zeros(1, 5, CV_64F);
  }

  auto result = solvePnP(image_pts, object_pts, K, D, detection.publish_number);

  if (!result.valid) {
    return result;
  }

  if (config_.gate.require_finite && !std::isfinite(result.reproj_error_raw)) {
    FYT_DEBUG("armor_detector_nn",
              "Reject PnP result for %s: non-finite raw reprojection error",
              detection.publish_number.c_str());
    return PoseEstimate{};
  }

  if (config_.gate.max_raw_reproj_error > 0.0 &&
      result.reproj_error_raw > config_.gate.max_raw_reproj_error) {
    FYT_DEBUG("armor_detector_nn",
              "Reject PnP result for %s: raw reproj error %.3f > %.3f",
              detection.publish_number.c_str(),
              result.reproj_error_raw,
              config_.gate.max_raw_reproj_error);
    return PoseEstimate{};
  }

  // Propagate tracker/time context before refinement so phase-4 sliding BA
  // can build per-track windows with real timestamps.
  result.track_id = detection.track_id;
  result.observation_stamp = detection.stamp;
  result.publish_number = detection.publish_number;
  result.R_imu_camera = R_imu_camera;

  // Phase 1+: Run refiner (single_yaw or sliding_window) if configured
  if (refiner_ && config_.refiner.mode != "none") {
    std::array<cv::Point2f, 4> img_pts_arr;
    std::copy_n(detection.keypoints.begin(), 4, img_pts_arr.begin());
    auto refined = refiner_->refine(result, img_pts_arr,
        {object_pts[0], object_pts[1], object_pts[2], object_pts[3]}, K, D);
    if (refined.valid &&
        refined.mode >= EstimateMode::SINGLE_BA_VALID) {
      refined.track_id = detection.track_id;
      refined.observation_stamp = detection.stamp;
      refined.publish_number = detection.publish_number;
      refined.R_imu_camera = R_imu_camera;
      applyDepthCorrection(refined, image_pts, object_pts, K, D, config_);
      return refined;
    }
  }

  // Fallthrough: return PnP result
  result.mode = EstimateMode::PNP_VALID;
  result.quality_score = 0.5;
  result.track_id = detection.track_id;
  result.observation_stamp = detection.stamp;
  result.publish_number = detection.publish_number;
  result.R_imu_camera = R_imu_camera;
  applyDepthCorrection(result, image_pts, object_pts, K, D, config_);
  return result;
}

std::vector<PoseEstimate> ArmorPoseEstimatorAdapter::estimateBatch(
    const std::vector<ArmorDetection>& detections,
    const sensor_msgs::msg::CameraInfo& camera_info,
    const Eigen::Matrix3d& R_imu_camera)
{
  std::vector<PoseEstimate> results;
  results.reserve(detections.size());
  for (const auto& d : detections) {
    results.push_back(estimate(d, camera_info, R_imu_camera));
  }
  return results;
}

std::vector<cv::Point3f> ArmorPoseEstimatorAdapter::getObjectPoints(
    const std::string& publish_type,
    double small_w, double small_h,
    double large_w, double large_h)
{
  double w, h;
  if (publish_type == "large") {
    w = large_w;
    h = large_h;
  } else {
    w = small_w;
    h = small_h;
  }

  double half_w = w / 2.0;
  double half_h = h / 2.0;

  // Canonical order: left_bottom, left_top, right_top, right_bottom
  // X forward, Y left, Z up (ROS camera frame convention)
  return {
    cv::Point3f(0.0,  half_w, -half_h),  // left_bottom
    cv::Point3f(0.0,  half_w,  half_h),  // left_top
    cv::Point3f(0.0, -half_w,  half_h),  // right_top
    cv::Point3f(0.0, -half_w, -half_h),  // right_bottom
  };
}

double ArmorPoseEstimatorAdapter::distanceToImageCenter(
    const cv::Point2f& center,
    const cv::Point2f& image_center)
{
  return cv::norm(center - image_center);
}

PoseEstimate ArmorPoseEstimatorAdapter::solvePnP(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const std::string& publish_number)
{
  PoseEstimate result;

  if (image_points.size() != object_points.size() || image_points.size() < 4) {
    return result;
  }

  try {
    if (config_.pnp_method == "ippe") {
      std::vector<cv::Mat> rvecs, tvecs;
      cv::solvePnPGeneric(object_points, image_points,
                          camera_matrix, dist_coeffs,
                          rvecs, tvecs,
                          false,
                          cv::SOLVEPNP_IPPE);

      if (rvecs.empty()) return result;

      int best = selectIppeSolutionLikeArmorDetector(
        object_points, image_points, rvecs, tvecs,
        camera_matrix, dist_coeffs, publish_number);

      if (best >= 0) {
        result.rvec = rvecs[best];
        result.tvec = tvecs[best];
        result.reprojection_error = reprojectionErrorSum(
          object_points, image_points, result.rvec, result.tvec,
          camera_matrix, dist_coeffs);
        result.valid = true;
      }
    } else {
      cv::Mat rvec, tvec;
      if (cv::solvePnP(object_points, image_points,
                       camera_matrix, dist_coeffs,
                       rvec, tvec, false, cv::SOLVEPNP_ITERATIVE)) {
        result.rvec = rvec;
        result.tvec = tvec;
        result.valid = true;

        // Compute reprojection error
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points, rvec, tvec,
                          camera_matrix, dist_coeffs, projected);
        double err = 0.0;
        for (size_t j = 0; j < image_points.size(); ++j) {
          err += cv::norm(image_points[j] - projected[j]);
        }
        result.reprojection_error = err;
      }
    }
  } catch (const cv::Exception& e) {
    FYT_ERROR("armor_detector_nn", "PnP solve failed: %s", e.what());
    return result;
  }

  if (result.valid) {
    result.translation = Eigen::Vector3d(
      result.tvec.at<double>(0),
      result.tvec.at<double>(1),
      result.tvec.at<double>(2));

    cv::Mat R;
    cv::Rodrigues(result.rvec, R);
    Eigen::Matrix3d eigen_R;
    cv::cv2eigen(R, eigen_R);
    result.rotation = Eigen::Quaterniond(eigen_R);

    // Extract yaw/pitch/roll from rvec
    rvecToEulerLikeArmorDetector(
      result.rvec, result.yaw, result.pitch, result.roll);

    // If configured, only rotate yaw by 180 degrees (keep translation)
    if (config_.force_pnp_rotate_180) {
      result.yaw += M_PI;

      // Rebuild rotation from modified yaw/pitch/roll using tf2 semantics
      tf2::Matrix3x3 m;
      m.setRPY(result.roll, result.pitch, result.yaw);
      tf2::Quaternion tf_q;
      m.getRotation(tf_q);

      // Convert tf2 quaternion -> Eigen
      Eigen::Quaterniond new_q(tf_q.w(), tf_q.x(), tf_q.y(), tf_q.z());
      result.rotation = new_q;

      // Update eigen_R and rvec to remain consistent with modified rotation
      Eigen::Matrix3d new_R;
      new_R = new_q.toRotationMatrix();
      cv::Mat R_cv;
      cv::eigen2cv(new_R, R_cv);
      cv::Rodrigues(R_cv, result.rvec);
      // Keep result.tvec / translation unchanged
    }

    // Per-point average reprojection error
    result.reproj_error_raw = result.reprojection_error / 4.0;
    result.reproj_error_refined = result.reproj_error_raw;
    result.mode = EstimateMode::PNP_VALID;

  }

  return result;
}

}  // namespace fyt::auto_aim
