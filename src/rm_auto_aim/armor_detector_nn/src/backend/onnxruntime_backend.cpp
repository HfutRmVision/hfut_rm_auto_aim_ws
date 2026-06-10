#include "armor_detector_nn/backend/onnxruntime_backend.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <onnxruntime_cxx_api.h>

#include "rm_utils/logger/log.hpp"

namespace fyt::auto_aim {

namespace {

std::string resolvePath(const std::string& raw_path) {
  if (raw_path.empty()) return raw_path;
  if (raw_path.compare(0, 10, "package://") == 0) {
    auto slash = raw_path.find('/', 10);
    std::string pkg = raw_path.substr(10, slash - 10);
    std::string rel = raw_path.substr(slash + 1);
    return ament_index_cpp::get_package_share_directory(pkg) + "/" + rel;
  }
  return raw_path;
}

}  // namespace

OnnxRuntimeBackend::OnnxRuntimeBackend() {
  info_.backend_name = "onnxruntime";
  info_.min_batch_size = 1;
  info_.max_batch_size = 1;
  info_.dynamic_batch = false;
}

OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

void OnnxRuntimeBackend::load(const BackendConfig& config) {
  std::string model_path = resolvePath(config.model_path);
  if (model_path.empty()) {
    throw std::runtime_error("OnnxRuntimeBackend: model_path is empty");
  }

  FYT_INFO("armor_detector_nn", "OnnxRuntimeBackend loading: {}", model_path.c_str());

  Ort::Env new_env(ORT_LOGGING_LEVEL_WARNING, "armor_detector_nn");
  env_ = std::make_unique<Ort::Env>(std::move(new_env));

  Ort::SessionOptions session_opts;
  session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_opts.SetIntraOpNumThreads(config.num_threads);
  session_opts.SetInterOpNumThreads(1);

  if (config.device == "cuda") {
    try {
      OrtCUDAProviderOptions cuda_opts;
      session_opts.AppendExecutionProvider_CUDA(cuda_opts);
      FYT_INFO("armor_detector_nn", "OnnxRuntimeBackend: CUDA EP enabled");
    } catch (const Ort::Exception& e) {
      FYT_WARN("armor_detector_nn",
               "CUDA EP not available ({}), falling back to CPU", e.what());
    }
  }

  try {
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("OnnxRuntimeBackend: failed to load model: " +
                             std::string(e.what()));
  }

  Ort::AllocatorWithDefaultOptions allocator;

  // Query input
  size_t num_inputs = session_->GetInputCount();
  if (num_inputs != 1) {
    throw std::runtime_error(
      "OnnxRuntimeBackend: expected 1 input, got " + std::to_string(num_inputs));
  }
  {
    auto name = session_->GetInputNameAllocated(0, allocator);
    input_names_.push_back(name.get());
    auto type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();
    const auto input_element_type = tensor_info.GetElementType();
    input_is_fp16_ = input_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (
      input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      input_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      throw std::runtime_error("OnnxRuntimeBackend: input must be float or float16");
    }
    FYT_INFO("armor_detector_nn", "  input: {} shape=[{},{},{},{}]",
             input_names_[0].c_str(),
             input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]);
  }

  // Query outputs
  size_t num_outputs = session_->GetOutputCount();
  output_names_.reserve(num_outputs);
  output_shapes_.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator);
    output_names_.push_back(name.get());
    auto type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    output_shapes_.push_back(tensor_info.GetShape());
    FYT_INFO("armor_detector_nn", "  output: {} shape=[{},{},{}]",
             output_names_[i].c_str(),
             output_shapes_[i][0], output_shapes_[i][1], output_shapes_[i][2]);
  }

  // Build C-string pointer arrays for Run()
  input_name_ptrs_.reserve(input_names_.size());
  for (const auto& n : input_names_) input_name_ptrs_.push_back(n.c_str());
  output_name_ptrs_.reserve(output_names_.size());
  for (const auto& n : output_names_) output_name_ptrs_.push_back(n.c_str());

  memory_info_ = std::make_unique<Ort::MemoryInfo>(
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

  validateModelIO(config);

  info_.precision = config.precision == Precision::FP32 ? "fp32" :
                    config.precision == Precision::FP16 ? "fp16" : "int8";

  loaded_ = true;
  FYT_INFO("armor_detector_nn", "OnnxRuntimeBackend loaded successfully");
}

std::vector<TensorOutput> OnnxRuntimeBackend::infer(const TensorInput& input) {
  if (!loaded_) {
    throw std::runtime_error("OnnxRuntimeBackend: not loaded");
  }

  std::lock_guard<std::mutex> lock(infer_mutex_);

  // Validate input shape
  const auto& in_shape = input.info.shape;
  if (in_shape.size() != input_shape_.size()) {
    throw std::runtime_error("OnnxRuntimeBackend: input rank mismatch");
  }

  std::vector<Ort::Float16_t> input_fp16_data;
  Ort::Value input_tensor{nullptr};
  if (input_is_fp16_) {
    input_fp16_data.resize(input.host_data.size());
    std::transform(
      input.host_data.begin(), input.host_data.end(), input_fp16_data.begin(),
      [](float value) { return Ort::Float16_t(value); });
    input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
      *memory_info_,
      input_fp16_data.data(),
      input_fp16_data.size(),
      input_shape_.data(),
      input_shape_.size());
  } else {
    // Wraps host_data, no copy.
    input_tensor = Ort::Value::CreateTensor<float>(
      *memory_info_,
      const_cast<float*>(input.host_data.data()),
      input.host_data.size(),
      input_shape_.data(),
      input_shape_.size());
  }

  try {
    auto outputs = session_->Run(
      Ort::RunOptions{nullptr},
      input_name_ptrs_.data(), &input_tensor, 1,
      output_name_ptrs_.data(), output_name_ptrs_.size());

    std::vector<TensorOutput> results;
    results.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto& out_val = outputs[i];
      auto type_info = out_val.GetTensorTypeAndShapeInfo();
      auto shape = type_info.GetShape();
      const auto element_type = type_info.GetElementType();
      size_t num_elements = type_info.GetElementCount();

      TensorOutput out;
      out.info.name = output_names_[i];
      out.info.shape = shape;
      out.info.dtype = TensorInfo::DType::FLOAT32;
      out.host_data.resize(num_elements);

      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::memcpy(out.host_data.data(),
                    out_val.GetTensorData<float>(),
                    num_elements * sizeof(float));
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* fp16_data = out_val.GetTensorData<Ort::Float16_t>();
        std::transform(
          fp16_data, fp16_data + num_elements, out.host_data.begin(),
          [](Ort::Float16_t value) { return static_cast<float>(value); });
      } else {
        throw std::runtime_error("OnnxRuntimeBackend: output must be float or float16");
      }

      results.push_back(std::move(out));
    }

    return results;
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("OnnxRuntimeBackend::infer failed: " +
                             std::string(e.what()));
  }
}

void OnnxRuntimeBackend::warmup(int iterations) {
  if (!loaded_) return;

  size_t num_elements = 1;
  for (auto d : input_shape_) num_elements *= d;
  std::vector<float> dummy(num_elements, 0.0F);

  std::vector<Ort::Float16_t> dummy_fp16;
  Ort::Value input_tensor{nullptr};
  if (input_is_fp16_) {
    dummy_fp16.assign(num_elements, Ort::Float16_t(0.0F));
    input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
      *memory_info_, dummy_fp16.data(), dummy_fp16.size(),
      input_shape_.data(), input_shape_.size());
  } else {
    input_tensor = Ort::Value::CreateTensor<float>(
      *memory_info_, dummy.data(), dummy.size(),
      input_shape_.data(), input_shape_.size());
  }

  for (int i = 0; i < iterations; ++i) {
    session_->Run(Ort::RunOptions{nullptr},
                  input_name_ptrs_.data(), &input_tensor, 1,
                  output_name_ptrs_.data(), output_name_ptrs_.size());
  }

  FYT_INFO("armor_detector_nn", "OnnxRuntimeBackend: warmup complete ({} iters)", iterations);
}

BackendInfo OnnxRuntimeBackend::info() const {
  return info_;
}

void OnnxRuntimeBackend::validateModelIO(const BackendConfig& config) {
  if (input_shape_.size() != 4) {
    throw std::runtime_error("OnnxRuntimeBackend: input must be 4D (NCHW)");
  }
  if (input_shape_[0] != 1) {
    FYT_WARN("armor_detector_nn",
             "Input batch dim is {}, expected 1. Batch scheduling will be limited.",
             input_shape_[0]);
  }
  if (output_shapes_.empty()) {
    throw std::runtime_error("OnnxRuntimeBackend: model has no outputs");
  }
}

}  // namespace fyt::auto_aim
