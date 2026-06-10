#include "armor_detector_nn/debug/debug_drawer.hpp"

#include <opencv2/imgproc.hpp>

namespace fyt::auto_aim {

DebugDrawer::DebugDrawer() {
  // Pre-assign colors for common labels
  class_colors_["B1"] = cv::Scalar(255, 100, 100);
  class_colors_["B2"] = cv::Scalar(255, 80, 80);
  class_colors_["B3"] = cv::Scalar(200, 60, 60);
  class_colors_["B4"] = cv::Scalar(200, 40, 40);
  class_colors_["B5"] = cv::Scalar(180, 20, 20);
  class_colors_["BO"] = cv::Scalar(150, 50, 50);
  class_colors_["BS"] = cv::Scalar(180, 80, 80);
  class_colors_["R1"] = cv::Scalar(100, 100, 255);
  class_colors_["R2"] = cv::Scalar(80, 80, 255);
  class_colors_["R3"] = cv::Scalar(60, 60, 200);
  class_colors_["R4"] = cv::Scalar(40, 40, 200);
  class_colors_["R5"] = cv::Scalar(20, 20, 180);
  class_colors_["RO"] = cv::Scalar(50, 50, 150);
  class_colors_["RS"] = cv::Scalar(80, 80, 180);
}

void DebugDrawer::drawDetections(
    cv::Mat& image,
    const std::vector<ArmorDetection>& detections,
    bool /*show_confidence*/) const
{
  for (const auto& d : detections) {
    cv::Scalar color = generateColor(d.publish_number + "_" +
        (d.color == fyt::EnemyColor::RED ? "R" : "B"));

    std::vector<cv::Point> outline;
    outline.reserve(4);
    for (const auto& kp : d.keypoints) {
      outline.emplace_back(cv::Point(cvRound(kp.x), cvRound(kp.y)));
    }

    cv::polylines(image, outline, true, color, 1, cv::LINE_AA);
  }
}

void DebugDrawer::drawProfiler(
    cv::Mat& image,
    double fps,
    double latency_ms,
    const std::string& backend_name,
    const std::string& precision) const
{
  int y = 30;
  auto putLine = [&](const std::string& text) {
    cv::putText(image, text, cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    y += 22;
  };

  std::ostringstream ss;
  ss << "Backend: " << backend_name << " | " << precision;
  putLine(ss.str());

  ss.str("");
  ss << "FPS: " << std::fixed << std::setprecision(1) << fps
     << "  Latency: " << std::setprecision(1) << latency_ms << " ms";
  putLine(ss.str());
}

void DebugDrawer::drawArmorsCount(cv::Mat& image, int count) const {
  std::ostringstream ss;
  ss << "Armors: " << count;
  cv::putText(image, ss.str(), cv::Point(10, 74),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

void DebugDrawer::setClassColor(const std::string& label, const cv::Scalar& color) {
  class_colors_[label] = color;
}

cv::Scalar DebugDrawer::generateColor(const std::string& label) {
  // Deterministic color from label hash
  std::hash<std::string> hasher;
  size_t h = hasher(label);
  return cv::Scalar(
    static_cast<int>((h >> 0) & 0xFF),
    static_cast<int>((h >> 8) & 0xFF),
    static_cast<int>((h >> 16) & 0xFF));
}

}  // namespace fyt::auto_aim
