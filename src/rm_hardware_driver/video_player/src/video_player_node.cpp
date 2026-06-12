// Copyright (c) 2025
// Licensed under the MIT License.

// OpenCV
#include <opencv2/opencv.hpp>

// ROS
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

// C++ system
#include <fstream>
#include <memory>
#include <string>
#include <thread>

namespace video_player
{

namespace
{

std::string resolvePackageUrl(const std::string & path)
{
  const std::string prefix = "package://";
  if (path.rfind(prefix, 0) != 0) {
    return path;
  }

  const auto slash = path.find('/', prefix.size());
  if (slash == std::string::npos) {
    return path;
  }

  const auto package_name = path.substr(prefix.size(), slash - prefix.size());
  const auto relative_path = path.substr(slash + 1);
  try {
    return ament_index_cpp::get_package_share_directory(package_name) + "/" + relative_path;
  } catch (const std::exception &) {
    return path;
  }
}

bool isGitLfsPointer(const std::string & path)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string first_line;
  std::getline(file, first_line);
  return first_line == "version https://git-lfs.github.com/spec/v1";
}

}  // namespace

class VideoPlayerNode : public rclcpp::Node
{
public:
  explicit VideoPlayerNode(const rclcpp::NodeOptions & options) : Node("video_player", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting VideoPlayerNode!");

    // Declare parameters
    video_path_ = resolvePackageUrl(this->declare_parameter("video_path", ""));
    loop_playback_ = this->declare_parameter("loop_playback", true);
    fps_ = this->declare_parameter("fps", 30.0);
    camera_name_ = this->declare_parameter("camera_name", "camera");
    frame_id_ = this->declare_parameter("frame_id", "camera_optical_frame");
    flip_image_ = this->declare_parameter("flip_image", false);
    image_topic_ = this->declare_parameter("image_topic", "image_raw");
    
    if (video_path_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "video_path parameter is required!");
      return;
    }

    // Open video file
    cap_.open(video_path_);
    if (!cap_.isOpened()) {
      if (isGitLfsPointer(video_path_)) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Video file is a Git LFS pointer, not media data. Install/fix git-lfs and pull it: %s",
          video_path_.c_str());
        return;
      }
      RCLCPP_ERROR(this->get_logger(), "Failed to open video file: %s", video_path_.c_str());
      return;
    }

    // Get video properties
    video_width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    video_height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double video_fps = cap_.get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    
    RCLCPP_INFO(this->get_logger(), "Video file opened: %s", video_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d", video_width_, video_height_);
    RCLCPP_INFO(this->get_logger(), "Original FPS: %.2f", video_fps);
    RCLCPP_INFO(this->get_logger(), "Total frames: %d", total_frames_);
    RCLCPP_INFO(this->get_logger(), "Playback FPS: %.2f", fps_);
    RCLCPP_INFO(this->get_logger(), "Loop playback: %s", loop_playback_ ? "true" : "false");

    // Create camera publisher
    bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", false);
    auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    camera_pub_ = image_transport::create_camera_publisher(this, image_topic_, qos);
    RCLCPP_INFO(this->get_logger(), "Publishing images to topic: %s", image_topic_.c_str());

    // Load camera info
    camera_info_manager_ =
      std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    auto camera_info_url =
      this->declare_parameter("camera_info_url", "");
    if (!camera_info_url.empty() && camera_info_manager_->validateURL(camera_info_url)) {
      camera_info_manager_->loadCameraInfo(camera_info_url);
      camera_info_msg_ = camera_info_manager_->getCameraInfo();
      RCLCPP_INFO(this->get_logger(), "Camera info loaded from: %s", camera_info_url.c_str());
    } else {
      RCLCPP_WARN(this->get_logger(), "No valid camera info URL provided, using default camera info");
      camera_info_msg_.width = video_width_;
      camera_info_msg_.height = video_height_;
      // Fill reasonable default intrinsics to avoid downstream division-by-zero
      // and empty-intrinsic issues in consumers (e.g., PnP solvers).
      double fx = std::max(1.0, static_cast<double>(video_width_) * 0.8);
      double fy = fx; // assume square pixels
      double cx = static_cast<double>(video_width_) / 2.0;
      double cy = static_cast<double>(video_height_) / 2.0;
      camera_info_msg_.k[0] = fx;
      camera_info_msg_.k[1] = 0.0;
      camera_info_msg_.k[2] = cx;
      camera_info_msg_.k[3] = 0.0;
      camera_info_msg_.k[4] = fy;
      camera_info_msg_.k[5] = cy;
      camera_info_msg_.k[6] = 0.0;
      camera_info_msg_.k[7] = 0.0;
      camera_info_msg_.k[8] = 1.0;
      // Projection matrix P (3x4)
      camera_info_msg_.p[0] = fx;
      camera_info_msg_.p[1] = 0.0;
      camera_info_msg_.p[2] = cx;
      camera_info_msg_.p[3] = 0.0;
      camera_info_msg_.p[4] = 0.0;
      camera_info_msg_.p[5] = fy;
      camera_info_msg_.p[6] = cy;
      camera_info_msg_.p[7] = 0.0;
      camera_info_msg_.p[8] = 0.0;
      camera_info_msg_.p[9] = 0.0;
      camera_info_msg_.p[10] = 1.0;
      camera_info_msg_.p[11] = 0.0;
      camera_info_msg_.distortion_model = "plumb_bob";
      camera_info_msg_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
    }

    // Start capture thread
    capture_thread_ = std::thread{[this]() -> void {
      RCLCPP_INFO(this->get_logger(), "Starting video playback!");

      cv::Mat frame;
      int frame_delay_ms = static_cast<int>(1000.0 / fps_);
      auto loop_start = std::chrono::steady_clock::now();

      while (rclcpp::ok()) {
        auto frame_start = std::chrono::steady_clock::now();

        // Read frame
        if (!cap_.read(frame)) {
          if (loop_playback_) {
            RCLCPP_INFO(this->get_logger(), "End of video reached, restarting...");
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap_.read(frame)) {
              RCLCPP_ERROR(this->get_logger(), "Failed to restart video!");
              break;
            }
          } else {
            RCLCPP_INFO(this->get_logger(), "End of video reached, stopping...");
            break;
          }
        }

        if (frame.empty()) {
          RCLCPP_WARN(this->get_logger(), "Empty frame received!");
          continue;
        }

        // Flip image if required
        if (flip_image_) {
          cv::flip(frame, frame, -1);  // -1 means flip both horizontally and vertically
        }

        // Convert to ROS message
        auto now = this->now();
        std_msgs::msg::Header header;
        header.stamp = now;
        header.frame_id = frame_id_;

        cv_bridge::CvImage cv_image(header, "bgr8", frame);
        sensor_msgs::msg::Image::SharedPtr image_msg = cv_image.toImageMsg();

        //auto image_msg
        // Update camera info timestamp
        camera_info_msg_.header = header;

        // Publish
        camera_pub_.publish(*image_msg, camera_info_msg_);

        // Control frame rate
        auto frame_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        int sleep_time = frame_delay_ms - elapsed;
        
        if (sleep_time > 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }

        frame_count_++;
        if (frame_count_ % 100 == 0) {
          auto loop_end = std::chrono::steady_clock::now();
          auto loop_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
          double actual_fps = 100000.0 / loop_elapsed;
          RCLCPP_INFO(this->get_logger(), "Published %d frames, actual FPS: %.2f", frame_count_, actual_fps);
          loop_start = loop_end;
        }
      }

      RCLCPP_INFO(this->get_logger(), "Video playback stopped, total frames published: %d", frame_count_);
    }};
  }

  ~VideoPlayerNode() override
  {
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }

    if (cap_.isOpened()) {
      cap_.release();
    }

    RCLCPP_INFO(this->get_logger(), "VideoPlayerNode destroyed!");
  }

private:
  // Parameters
  std::string video_path_;
  bool loop_playback_;
  double fps_;
  std::string camera_name_;
  std::string frame_id_;
  bool flip_image_;
  std::string image_topic_;

  // OpenCV video capture
  cv::VideoCapture cap_;
  int video_width_;
  int video_height_;
  int total_frames_;
  int frame_count_ = 0;

  // ROS publishers
  image_transport::CameraPublisher camera_pub_;

  // Camera info
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;

  // Capture thread
  std::thread capture_thread_;
};

}  // namespace video_player

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(video_player::VideoPlayerNode)
