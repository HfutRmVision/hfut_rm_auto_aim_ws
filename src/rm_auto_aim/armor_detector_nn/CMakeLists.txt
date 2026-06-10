cmake_minimum_required(VERSION 3.16)
project(armor_detector_nn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-Wall)

# --- backend toggles ---
# Desktop NVIDIA profile: AMD CPU + NVIDIA GPU should use ONNX Runtime CUDA by default.
# OpenVINO is Intel-oriented and TensorRT requires a matching TensorRT SDK/engine, so both
# stay opt-in. BUILD_WITH_GPU is kept as a compatibility switch for old build commands.
option(BUILD_WITH_GPU         "Compatibility option: enable the NVIDIA GPU-capable ONNX Runtime backend" ON)
option(BUILD_ONNX_BACKEND     "Build ONNX Runtime backend; use onnxruntime-gpu at runtime for CUDA" ON)
option(BUILD_OPENVINO_BACKEND "Build OpenVINO backend" OFF)
option(BUILD_TENSORRT_BACKEND "Build TensorRT backend; requires CUDA Toolkit and TensorRT SDK" OFF)

if(BUILD_WITH_GPU)
  set(BUILD_ONNX_BACKEND ON CACHE BOOL "Build ONNX Runtime backend" FORCE)
endif()

message(STATUS "armor_detector_nn backends:")
message(STATUS "  ONNX Runtime: ${BUILD_ONNX_BACKEND}")
message(STATUS "  OpenVINO:      ${BUILD_OPENVINO_BACKEND}")
message(STATUS "  TensorRT:      ${BUILD_TENSORRT_BACKEND}")

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(yaml-cpp REQUIRED)

ament_auto_find_build_dependencies()

###########
## Build ##
###########

# --- core sources ---
set(CORE_SOURCES
  src/core/label_map.cpp
  src/core/preprocessor.cpp
  src/core/armor_pose_estimator_adapter.cpp
  src/core/armor_detector_nn.cpp
  src/core/pose_refine/single_yaw_refiner.cpp
  src/core/pose_refine/sliding_window_refiner.cpp
  src/core/tracker/internal_iou_tracker_strategy.cpp
  src/core/corner_refine/roi_pca_corner_refiner.cpp
)

# --- backend sources ---
set(BACKEND_SOURCES
  src/backend/inference_backend_factory.cpp
)

# --- ONNX Runtime backend ---
# Use the same backend for CPU and NVIDIA GPU. CUDA support is provided by the
# installed ONNX Runtime binary (onnxruntime-gpu), not by this CMake target.
if(BUILD_ONNX_BACKEND)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root directory")
  if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
  endif()
  if(NOT ONNXRUNTIME_ROOT AND EXISTS /opt/onnxruntime-gpu)
    set(ONNXRUNTIME_ROOT /opt/onnxruntime-gpu)
  endif()

  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      ${ONNXRUNTIME_ROOT}/include
      /opt/onnxruntime-gpu/include
      /usr/local/include
      /usr/include
    PATH_SUFFIXES
      onnxruntime
      onnxruntime/core/session
  )
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
      ${ONNXRUNTIME_ROOT}/lib
      ${ONNXRUNTIME_ROOT}/lib64
      /opt/onnxruntime-gpu/lib
      /usr/local/lib
      /usr/local/lib64
      /usr/lib
      /usr/lib/x86_64-linux-gnu
  )

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
      "BUILD_ONNX_BACKEND=ON but ONNX Runtime was not found. "
      "Install onnxruntime-gpu for NVIDIA CUDA, then set ONNXRUNTIME_ROOT "
      "to its install prefix if CMake cannot find it.")
  endif()

  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIB}")
  list(APPEND BACKEND_SOURCES src/backend/onnxruntime_backend.cpp)
endif()

# --- OpenVINO 后端 ---
if(BUILD_OPENVINO_BACKEND)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  list(APPEND BACKEND_SOURCES src/backend/openvino_backend.cpp)
endif()

# --- TensorRT backend ---
if(BUILD_TENSORRT_BACKEND)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "BUILD_TENSORRT_BACKEND=ON but CUDA not found. Set CUDAToolkit_ROOT or install CUDA.")
  endif()

  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
      /usr/include
      /usr/local/include
      /usr/include/x86_64-linux-gnu
      /usr/include/aarch64-linux-gnu
  )
  find_library(TENSORRT_NVINFER_LIB nvinfer
    HINTS
      /usr/lib
      /usr/local/lib
      /usr/lib/x86_64-linux-gnu
      /usr/lib/aarch64-linux-gnu
  )

  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
    message(FATAL_ERROR
      "BUILD_TENSORRT_BACKEND=ON but TensorRT headers/libs not found "
      "(need NvInfer.h and libnvinfer)")
  endif()

  list(APPEND BACKEND_SOURCES src/backend/tensorrt_backend.cpp)
endif()

# --- postprocess sources ---
set(POSTPROCESS_SOURCES
  src/postprocess/decode_strategy_factory.cpp
  src/postprocess/detection_quality_filter.cpp
  src/postprocess/ultralytics_pose_decode_strategy.cpp
  src/postprocess/robotpilots_landmark_decode_strategy.cpp
)

# --- geometry sources ---
set(GEOMETRY_SOURCES src/geometry/nms.cpp)

# --- debug sources ---
set(DEBUG_SOURCES
  src/debug/debug_drawer.cpp
  src/debug/profiler.cpp
)

# --- node sources ---
set(NODE_SOURCES src/armor_detector_nn_node.cpp)

ament_auto_add_library(${PROJECT_NAME} SHARED
  ${NODE_SOURCES}
  ${CORE_SOURCES}
  ${BACKEND_SOURCES}
  ${POSTPROCESS_SOURCES}
  ${GEOMETRY_SOURCES}
  ${DEBUG_SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  yaml-cpp
  ${armor_detector_LIBRARIES}
  Sophus::Sophus
)

if(BUILD_ONNX_BACKEND)
  target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
  target_compile_definitions(${PROJECT_NAME} PUBLIC ARMOR_DETECTOR_NN_HAS_ONNX=1)
endif()

if(BUILD_OPENVINO_BACKEND)
  target_link_libraries(${PROJECT_NAME} openvino::runtime)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ARMOR_DETECTOR_NN_HAS_OPENVINO=1)
endif()

if(BUILD_TENSORRT_BACKEND AND CUDAToolkit_FOUND)
  target_include_directories(${PROJECT_NAME} PUBLIC ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} ${TENSORRT_NVINFER_LIB} CUDA::cudart)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ARMOR_DETECTOR_NN_HAS_TENSORRT=1)
endif()

# --- node registration ---
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "fyt::auto_aim::ArmorDetectorNNNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

#############
## Install ##
#############

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  docs
  model
)
