#ifndef ARMOR_DETECTOR_NN_ARMOR_DETECTOR_NN_HPP_
#define ARMOR_DETECTOR_NN_ARMOR_DETECTOR_NN_HPP_

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <std_msgs/msg/header.hpp>

#include "armor_detector_nn/core/detection_types.hpp"
#include "armor_detector_nn/core/detector_config.hpp"
#include "armor_detector_nn/core/preprocessor.hpp"
#include "armor_detector_nn/backend/inference_backend.hpp"
#include "armor_detector_nn/postprocess/decode_strategy.hpp"
#include "armor_detector_nn/core/label_map.hpp"
#include "armor_detector_nn/debug/profiler.hpp"

namespace fyt::auto_aim {

class ArmorDetectorNN {
public:
  explicit ArmorDetectorNN(const DetectorConfig& config);
  ~ArmorDetectorNN();

  bool initialize();

  std::vector<FrameDetections> detectBatch(
    const std::vector<cv::Mat>& images,
    const std::vector<std_msgs::msg::Header>& headers);

  bool isInitialized() const { return initialized_; }

  void setTargetColor(fyt::EnemyColor color) { target_color_ = color; }

  const DetectorConfig& config() const { return config_; }
  BackendInfo backendInfo() const;
  const Preprocessor* preprocessor() const { return preprocessor_.get(); }
  const ProfilerEntry& lastProfiler() const { return last_profile_; }

private:
  DetectorConfig config_;
  std::unique_ptr<Preprocessor> preprocessor_;
  std::unique_ptr<IInferenceBackend> backend_;
  std::unique_ptr<IDecodeStrategy> decode_strategy_;
  std::unique_ptr<LabelMap> label_map_;
  fyt::EnemyColor target_color_{fyt::EnemyColor::RED};
  ProfilerEntry last_profile_;
  bool initialized_{false};
};

}  // namespace fyt::auto_aim

#endif
