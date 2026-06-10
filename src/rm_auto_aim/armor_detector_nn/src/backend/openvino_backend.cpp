#include "armor_detector_nn/backend/openvino_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <openvino/openvino.hpp>

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

// ov::Shape = std::vector<size_t>, we store std::vector<int64_t>
std::vector<int64_t> shapeToVec(const ov::Shape& s) {
  return std::vector<int64_t>(s.begin(), s.end());
}

}  // namespace

OpenVINOBackend::OpenVINOBackend() {
  info_.backend_name = "openvino";
  info_.min_batch_size = 1;
  info_.max_batch_size = 1;
  info_.dynamic_batch = false;
}

OpenVINOBackend::~OpenVINOBackend() = default;

void OpenVINOBackend::load(const BackendConfig& config) {
  core_ = std::make_unique<ov::Core>();

  std::string model_path = resolvePath(config.openvino_xml_path);
  if (model_path.empty()) {
    model_path = resolvePath(config.model_path);
  }
  std::string bin_path = resolvePath(config.openvino_bin_path);

  if (model_path.empty()) {
    throw std::runtime_error("OpenVINOBackend: model path is empty (openvino_model_xml/model_path)");
  }

  std::string model_ext;
  const size_t ext_pos = model_path.find_last_of('.');
  if (ext_pos != std::string::npos) {
    model_ext = model_path.substr(ext_pos);
    std::transform(model_ext.begin(), model_ext.end(), model_ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  }

  if (model_ext == ".xml" && bin_path.empty()) {
    bin_path = model_path;
    const size_t xml_ext = bin_path.rfind(".xml");
    if (xml_ext != std::string::npos) {
      bin_path.replace(xml_ext, 4, ".bin");
    }
  }

  std::string device = selectAvailableDevice(config.device);
  if (device.empty()) {
    throw std::runtime_error("OpenVINOBackend: no available device found");
  }

  FYT_INFO("armor_detector_nn", "OpenVINOBackend loading: %s on %s",
           model_path.c_str(), device.c_str());

  loadModel(model_path, bin_path, device, config.num_threads);
  detectQuantizationPrecision();
  validateModelIO(config);

  info_.precision = is_int8_quantized_ ? "int8" :
                    config.precision == Precision::FP16 ? "fp16" : "fp32";

  loaded_ = true;
  FYT_INFO("armor_detector_nn", "OpenVINOBackend loaded successfully (precision: %s)",
           info_.precision.c_str());
}

std::vector<TensorOutput> OpenVINOBackend::infer(const TensorInput& input) {
  if (!loaded_) {
    throw std::runtime_error("OpenVINOBackend: not loaded");
  }

  std::lock_guard<std::mutex> lock(infer_mutex_);

  {
    const auto input_tensor = makeInputTensor(input);
    const auto& in_shape = input.info.shape;
    size_t num_elements = 1;
    for (auto d : in_shape) num_elements *= d;

    if (num_elements != input.host_data.size()) {
      throw std::runtime_error(
        "OpenVINOBackend::infer: input size mismatch. Expected " +
        std::to_string(num_elements) + ", got " +
        std::to_string(input.host_data.size()));
    }

    infer_request_->set_tensor(input_name_, input_tensor);
  }

  // Synchronous inference
  infer_request_->infer();

  // Extract outputs
  std::vector<TensorOutput> results;
  results.reserve(output_names_.size());

  for (size_t i = 0; i < output_names_.size(); ++i) {
    auto out_tensor = infer_request_->get_tensor(output_names_[i]);
    const auto& shape = output_shapes_[i];

    size_t num_elements = 1;
    for (auto d : shape) num_elements *= d;

    TensorOutput out;
    out.info.name = output_names_[i];
    out.info.shape = shape;
    out.info.dtype = TensorInfo::DType::FLOAT32;
    out.host_data.resize(num_elements);

    std::memcpy(out.host_data.data(),
                out_tensor.data<float>(),
                num_elements * sizeof(float));

    results.push_back(std::move(out));
  }

  FYT_DEBUG("armor_detector_nn", "OpenVINOBackend inference complete.");

  return results;
}

void OpenVINOBackend::warmup(int iterations) {
  if (!loaded_) return;

  size_t num_elements = 1;
  for (auto d : input_shape_) num_elements *= d;
  TensorInput dummy;
  dummy.info.shape = input_shape_;
  dummy.host_data.assign(num_elements, 0.0F);
  const auto input_tensor = makeInputTensor(dummy);
  infer_request_->set_tensor(input_name_, input_tensor);

  for (int i = 0; i < iterations; ++i) {
    infer_request_->infer();
  }

  FYT_INFO("armor_detector_nn", "OpenVINOBackend: warmup complete ({} iters)", iterations);
}

BackendInfo OpenVINOBackend::info() const {
  return info_;
}

void OpenVINOBackend::loadModel(const std::string& model_path,
                                 const std::string& bin_path,
                                 const std::string& device,
                                 int num_threads) {
  // Set thread count
  if (num_threads > 0) {
    core_->set_property("CPU", ov::inference_num_threads(num_threads));
  }

  // Read model
  std::shared_ptr<ov::Model> model;
  if (bin_path.empty()) {
    model = core_->read_model(model_path);
  } else {
    model = core_->read_model(model_path, bin_path);
  }
  if (!model) {
    throw std::runtime_error("OpenVINOBackend: failed to read model from " + model_path);
  }

  // Compile for latency — pass properties as variadic arguments to compile_model
  compiled_model_ = std::make_unique<ov::CompiledModel>(
    core_->compile_model(model, device,
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::streams::num(1)));

  // Create inference request
  infer_request_ = std::make_unique<ov::InferRequest>(
    compiled_model_->create_infer_request());

  // Query I/O
  {
    auto input = compiled_model_->input();
    input_name_ = input.get_any_name();
    input_shape_ = shapeToVec(input.get_shape());
    FYT_INFO("armor_detector_nn", "  OV input: {} shape=[{},{},{},{}]",
             input_name_.c_str(),
             input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]);
  }

  {
    auto outputs = compiled_model_->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      output_names_.push_back(outputs[i].get_any_name());
      output_shapes_.push_back(shapeToVec(outputs[i].get_shape()));
      FYT_INFO("armor_detector_nn", "  OV output: {} shape=[{},{},{}]",
               output_names_[i].c_str(),
               output_shapes_[i][0], output_shapes_[i][1], output_shapes_[i][2]);
    }
  }
}

void OpenVINOBackend::validateModelIO(const BackendConfig& config) {
  if (input_shape_.size() != 4) {
    throw std::runtime_error("OpenVINOBackend: input must be 4D (NCHW)");
  }
  if (input_shape_[0] != 1) {
    FYT_WARN("armor_detector_nn",
             "OV input batch dim is %ld, expected 1", input_shape_[0]);
  }
  if (output_shapes_.empty()) {
    throw std::runtime_error("OpenVINOBackend: model has no outputs");
  }
}

void OpenVINOBackend::detectQuantizationPrecision() {
  is_int8_quantized_ = false;

  // Check the original model ops (via runtime model from compiled model)
  auto runtime_model = compiled_model_->get_runtime_model();
  for (auto& op : runtime_model->get_ops()) {
    std::string type_name = op->get_type_name();
    if (type_name == "FakeQuantize") {
      is_int8_quantized_ = true;
      break;
    }
  }

  // Cross-check with NNCF rt_info
  try {
    auto rt_info = runtime_model->get_rt_info();
    if (rt_info.count("nncf")) {
      auto nncf = rt_info.at("nncf").as<ov::AnyMap>();
      if (nncf.count("version")) {
        auto ver = nncf.at("version").as<std::string>();
        FYT_INFO("armor_detector_nn", "NNCF quantization version: %s", ver.c_str());
      }
    }
  } catch (...) {
    // rt_info is optional
  }

  if (is_int8_quantized_) {
    info_.precision = "int8";
    FYT_INFO("armor_detector_nn", "INT8 quantized model detected (NNCF)");
  }
}

std::string OpenVINOBackend::selectAvailableDevice(
    const std::string& preferred_device) {
  auto available = core_->get_available_devices();

  // Exact match
  for (const auto& d : available) {
    if (d == preferred_device) return d;
  }

  // Case-insensitive
  std::string upper_pref = preferred_device;
  std::transform(upper_pref.begin(), upper_pref.end(),
                 upper_pref.begin(), ::toupper);
  for (const auto& d : available) {
    std::string upper = d;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == upper_pref) return d;
  }

  // Fallback: prefer CPU
  for (const auto& d : available) {
    std::string upper = d;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "CPU") return d;
  }

  return {};
}

ov::Tensor OpenVINOBackend::makeInputTensor(const TensorInput& input) const {
  const auto port = compiled_model_->input();
  const auto expected_shape = port.get_shape();
  const auto expected_type = port.get_element_type();

  size_t expected_elements = 1;
  for (const auto dim : expected_shape) expected_elements *= dim;
  if (expected_elements != input.host_data.size()) {
    throw std::runtime_error(
      "OpenVINOBackend::makeInputTensor: input size mismatch. Expected " +
      std::to_string(expected_elements) + ", got " +
      std::to_string(input.host_data.size()));
  }

  if (expected_type == ov::element::f32) {
    ov::Tensor tensor(expected_type, expected_shape);
    std::memcpy(tensor.data<float>(), input.host_data.data(),
                expected_elements * sizeof(float));
    return tensor;
  }

  if (expected_type == ov::element::f16) {
    ov::Tensor tensor(expected_type, expected_shape);
    auto* dst = tensor.data<ov::float16>();
    for (size_t i = 0; i < expected_elements; ++i) {
      dst[i] = ov::float16(input.host_data[i]);
    }
    return tensor;
  }

  throw std::runtime_error(
    "OpenVINOBackend::makeInputTensor: unsupported model input element type: " +
    expected_type.get_type_name());
}

}  // namespace fyt::auto_aim
