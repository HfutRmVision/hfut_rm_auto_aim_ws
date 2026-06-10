#ifndef ARMOR_DETECTOR_NN_ARMOR_DETECTOR_NN_NODE_HPP_
#define ARMOR_DETECTOR_NN_ARMOR_DETECTOR_NN_NODE_HPP_

#include <memory>
#include <string>

#include "armor_detector/armor_detector.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <image_transport/publisher.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "rm_interfaces/msg/armors.hpp"
#include "rm_interfaces/srv/set_mode.hpp"
#include "rm_utils/heartbeat.hpp"

#include "armor_detector_nn/core/armor_detector_nn.hpp"
#include "armor_detector_nn/core/armor_pose_estimator_adapter.hpp"
#include "armor_detector_nn/core/corner_refine/icorner_refiner.hpp"
#include "armor_detector_nn/core/detection_types.hpp"
#include "armor_detector_nn/core/detector_config.hpp"
#include "armor_detector_nn/core/tracker/itracker_strategy.hpp"
#include "armor_detector_nn/debug/debug_drawer.hpp"
#include "armor_detector_nn/debug/profiler.hpp"

namespace fyt::auto_aim {

class ArmorDetectorNNNode : public rclcpp::Node {
public:
  explicit ArmorDetectorNNNode(const rclcpp::NodeOptions& options);

private:
  void initializeParameters();
  void validateParameters();

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& ci_msg);

  void setModeCallback(
    const std::shared_ptr<rm_interfaces::srv::SetMode::Request> request,
    std::shared_ptr<rm_interfaces::srv::SetMode::Response> response);

  void publishEmptyArmors(const std_msgs::msg::Header& header);
  void publishMarkers(
    const std::vector<ArmorDetection>& detections,
    const std::vector<PoseEstimate>& poses,
    const std_msgs::msg::Header& header);
  void publishDebugImage(
    const cv::Mat& frame,
    const FrameDetections& fd,
    const std::vector<PoseEstimate>& poses);
  void createDebugPublishers();
  void initializeTraditionalDetector();
  void updateTraditionalDetectorColor();
  std::vector<ArmorDetection> detectTraditional(const cv::Mat& bgr_frame);
  std::vector<ArmorDetection> mergeDetections(
    const std::vector<ArmorDetection>& nn_detections,
    const std::vector<ArmorDetection>& traditional_detections) const;

  DetectorConfig config_;
  DetectMode current_mode_{DetectMode::DISABLED};

  // Subscriptions
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
  cv::Point2f cam_center_;

  // Publishers
  rclcpp::Publisher<rm_interfaces::msg::Armors>::SharedPtr armors_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // Debug
  bool debug_{false};
  image_transport::Publisher result_img_pub_;

  // Service
  rclcpp::Service<rm_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;

  // Heartbeat
  HeartBeatPublisher::SharedPtr heartbeat_;

  // Pipeline
  std::unique_ptr<ArmorDetectorNN> detector_;
  std::unique_ptr<ArmorPoseEstimatorAdapter> pose_estimator_adapter_;
  std::unique_ptr<ArmorPoseEstimatorAdapter> pose_estimator_reference_adapter_;
  std::unique_ptr<DebugDrawer> debug_drawer_;
  std::unique_ptr<Profiler> profiler_;

  bool debug_pose_compare_{false};
  bool publish_in_target_frame_{false};

  // Phase 2 — tracker
  std::shared_ptr<ITrackerStrategy> tracker_;

  // Phase 3 — corner refiner
  std::shared_ptr<ICornerRefiner> corner_refiner_;

  // Traditional fallback/fusion detector reused from armor_detector package.
  std::unique_ptr<Detector> traditional_detector_;

  // TF: target_frame (e.g. odom) -> camera frame rotation
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
};

}  // namespace fyt::auto_aim

#endif
