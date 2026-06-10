#ifndef ARMOR_DETECTOR_NN_ARMOR_POSE_ESTIMATOR_ADAPTER_HPP_
#define ARMOR_DETECTOR_NN_ARMOR_POSE_ESTIMATOR_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "armor_detector_nn/core/detection_types.hpp"
#include "armor_detector_nn/core/detector_config.hpp"

namespace fyt::auto_aim {

enum class EstimateMode {
  LOST,               // PnP failed
  PNP_VALID,          // PnP only
  SINGLE_BA_VALID,    // Single-frame yaw BA succeeded
  SW_BA_VALID         // Sliding-window BA succeeded (Phase 4+)
};

struct PoseEstimate {
  bool valid{false};
  EstimateMode mode{EstimateMode::LOST};
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
  cv::Mat rvec;
  cv::Mat tvec;
  double yaw{0.0};
  double pitch{0.0};
  double roll{0.0};
  double reprojection_error{0.0};
  double reproj_error_raw{0.0};
  double reproj_error_refined{0.0};
  double quality_score{0.0};
  int track_id{-1};
  rclcpp::Time observation_stamp{};
  std::string publish_number;
  Eigen::Matrix3d R_imu_camera{Eigen::Matrix3d::Identity()};

  // BA/PnP refiner covariance metadata (Phase 1: append-only)
  bool covariance_valid{false};
  Eigen::Matrix4d covariance_xyz_yaw{Eigen::Matrix4d::Identity()};
  double condition_number{0.0};
  int num_points{0};
  int num_inliers{0};
};

class IPoseRefiner;

class ArmorPoseEstimatorAdapter {
public:
  explicit ArmorPoseEstimatorAdapter(const PoseConfig& config);
  ~ArmorPoseEstimatorAdapter();

  // Set a pose refiner (Phase 1+: single_yaw / sliding_window).
  void setRefiner(std::shared_ptr<IPoseRefiner> refiner);

  PoseEstimate estimate(
    const ArmorDetection& detection,
    const sensor_msgs::msg::CameraInfo& camera_info,
    const Eigen::Matrix3d& R_imu_camera = Eigen::Matrix3d::Identity());

  std::vector<PoseEstimate> estimateBatch(
    const std::vector<ArmorDetection>& detections,
    const sensor_msgs::msg::CameraInfo& camera_info,
    const Eigen::Matrix3d& R_imu_camera = Eigen::Matrix3d::Identity());

  static std::vector<cv::Point3f>
  getObjectPoints(const std::string& publish_type,
                  double small_w, double small_h,
                  double large_w, double large_h);

  static double distanceToImageCenter(
    const cv::Point2f& center,
    const cv::Point2f& image_center);

  const PoseConfig& config() const { return config_; }

private:
  PoseEstimate solvePnP(
    const std::vector<cv::Point2f>& image_points,
    const std::vector<cv::Point3f>& object_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const std::string& publish_number = "");

  PoseConfig config_;
  std::shared_ptr<IPoseRefiner> refiner_;
};

}  // namespace fyt::auto_aim

#endif
