#ifndef ARMOR_DETECTOR_NN_POSE_REFINER_HPP_
#define ARMOR_DETECTOR_NN_POSE_REFINER_HPP_

#include <array>
#include <memory>
#include <unordered_map>

#include <opencv2/core.hpp>

#include "armor_detector_nn/core/armor_pose_estimator_adapter.hpp"
#include "armor_detector_nn/core/detector_config.hpp"
#include "armor_detector_nn/core/pose_refine/sliding_window_state.hpp"

namespace fyt::auto_aim {

// High-level refiner interface.
class IPoseRefiner {
public:
  virtual ~IPoseRefiner() = default;

  virtual PoseEstimate refine(
    const PoseEstimate& pnp_result,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    const cv::Mat& D) = 0;
};

// Single-frame yaw-only BA refiner.
// Fixed: tvec (from PnP), pitch, roll.
// Optimized: yaw.
class SingleYawRefiner : public IPoseRefiner {
public:
  explicit SingleYawRefiner(const SingleYawConfig& config,
                            const GateConfig& gate);

  PoseEstimate refine(
    const PoseEstimate& pnp_result,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    const cv::Mat& D) override;

private:
  SingleYawConfig config_;
  GateConfig gate_;

  static double angleWrap(double angle);

  double computeReprojError(
    double yaw,
    const cv::Vec3d& tvec,
    double pitch,
    double roll,
    const Eigen::Matrix3d& R_imu_camera,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K);

  bool optimizeYaw(
    double& yaw,
    const cv::Vec3d& tvec,
    double pitch,
    double roll,
    const Eigen::Matrix3d& R_imu_camera,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K);
};

// Per-track sliding-window BA refiner (Phase 4).
// Each track_id maintains an independent window.
// Optimizes [tx, ty, tz, yaw] over the window.
// Falls back to SingleYawRefiner → PnP on failure.
class SlidingWindowRefiner : public IPoseRefiner {
public:
  SlidingWindowRefiner(const SlidingWindowConfig& sw_config,
                       const SingleYawConfig& sy_config,
                       const GateConfig& gate);

  PoseEstimate refine(
    const PoseEstimate& pnp_result,
    const std::array<cv::Point2f, 4>& image_points,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    const cv::Mat& D) override;

  // Per-track window management
  void pushFrame(int track_id, const TrackWindowFrame& frame);
  SlidingWindowState& getWindow(int track_id);
  void removeTrack(int track_id);
  size_t activeTrackCount() const;

private:
  SlidingWindowConfig sw_config_;
  SingleYawConfig sy_config_;
  GateConfig gate_;

  // Per-track sliding windows, keyed by track_id.
  std::unordered_map<int, SlidingWindowState> windows_;

  // Fallback to single-frame yaw BA.
  SingleYawRefiner fallback_refiner_;

  // Internal frame counter used as dt proxy when real timestamps
  // are unavailable. Each call to refine() for a track increments it.
  std::unordered_map<int, int> frame_counters_;

  // Core solver: Gauss-Newton on the sliding window.
  bool solveSlidingWindow(
    SlidingWindowState& window,
    const std::array<cv::Point3f, 4>& object_points,
    const cv::Mat& K,
    Eigen::Vector3d& t_out,
    double& yaw_out,
    double& final_error);

  static double angleWrap(double angle);
};

}  // namespace fyt::auto_aim

#endif
