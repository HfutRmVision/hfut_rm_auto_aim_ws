#ifndef ARMOR_DETECTOR_NN_OPENVINO_BACKEND_HPP_
#define ARMOR_DETECTOR_NN_OPENVINO_BACKEND_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <openvino/openvino.hpp>

#include "armor_detector_nn/backend/inference_backend.hpp"

namespace ov {
class Core;
class CompiledModel;
class InferRequest;
}  // namespace ov

namespace fyt::auto_aim {

class OpenVINOBackend : public IInferenceBackend {
public:
  OpenVINOBackend();
  ~OpenVINOBackend() override;

  void load(const BackendConfig& config) override;
  std::vector<TensorOutput> infer(const TensorInput& input) override;
  void warmup(int iterations) override;
  BackendInfo info() const override;

private:
  void loadModel(const std::string& model_path, const std::string& bin_path,
                 const std::string& device, int num_threads);
  void validateModelIO(const BackendConfig& config);
  void detectQuantizationPrecision();
  std::string selectAvailableDevice(const std::string& preferred_device);
  ov::Tensor makeInputTensor(const TensorInput& input) const;

  std::unique_ptr<ov::Core> core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;
  std::unique_ptr<ov::InferRequest> infer_request_;

  std::string input_name_;
  std::vector<std::string> output_names_;
  std::vector<int64_t> input_shape_;
  std::vector<std::vector<int64_t>> output_shapes_;

  bool is_int8_quantized_{false};

  BackendInfo info_;
  bool loaded_{false};

  std::mutex infer_mutex_;
};

}  // namespace fyt::auto_aim

#endif
