#ifndef ARMOR_DETECTOR_NN_ONNXRUNTIME_BACKEND_HPP_
#define ARMOR_DETECTOR_NN_ONNXRUNTIME_BACKEND_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "armor_detector_nn/backend/inference_backend.hpp"

namespace Ort {
class Env;
class Session;
class MemoryInfo;
}  // namespace Ort

namespace fyt::auto_aim {

class OnnxRuntimeBackend : public IInferenceBackend {
public:
  OnnxRuntimeBackend();
  ~OnnxRuntimeBackend() override;

  void load(const BackendConfig& config) override;
  std::vector<TensorOutput> infer(const TensorInput& input) override;
  void warmup(int iterations) override;
  BackendInfo info() const override;

private:
  void validateModelIO(const BackendConfig& config);

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;

  std::vector<int64_t> input_shape_;
  std::vector<std::vector<int64_t>> output_shapes_;
  bool input_is_fp16_{false};

  BackendInfo info_;
  bool loaded_{false};

  std::mutex infer_mutex_;
};

}  // namespace fyt::auto_aim

#endif
