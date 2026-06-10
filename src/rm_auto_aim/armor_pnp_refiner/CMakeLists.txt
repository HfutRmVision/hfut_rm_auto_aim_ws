cmake_minimum_required(VERSION 3.8)
project(armor_pnp_refiner)

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

# Find g2o - reuse existing FindG2O.cmake if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../armor_detector/cmake_modules/FindG2O.cmake")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../armor_detector/cmake_modules")
endif()
find_package(G2O REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/core/armor_pnp_refiner.cpp
  src/core/pnp_refiner_config.cpp
  src/adapters/armor_detector_adapter.cpp
  src/adapters/armor_detector_nn_adapter.cpp
  src/geometry/armor_geometry.cpp
  src/geometry/camera_model.cpp
  src/geometry/pose_parameterization.cpp
  src/graph/common/reprojection_edge.cpp
  src/graph/common/pnp_prior_edge.cpp
  src/graph/common/graph_diagnostics.cpp
  src/graph/single_yaw/yaw_vertex.cpp
  src/graph/single_yaw/single_yaw_optimizer.cpp
  src/graph/legacy_single_yaw/armor_detector_ba_solver_proxy.cpp
  src/graph/single_xyz_yaw/xyz_yaw_vertex.cpp
  src/graph/single_xyz_yaw/single_xyz_yaw_optimizer.cpp
  src/graph/sliding_window/window_state.cpp
  src/graph/sliding_window/temporal_prior_edges.cpp
  src/graph/sliding_window/sliding_window_optimizer.cpp
  src/association/short_term_associator.cpp
  src/association/track2d_bridge.cpp
  src/association/association_gate.cpp
  src/covariance/covariance_estimator.cpp
  src/covariance/covariance_clamp.cpp
  src/covariance/covariance_transform.cpp
  src/quality/quality_gate.cpp
  src/quality/confidence_estimator.cpp
  src/quality/fallback_manager.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${OpenCV_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
  ${G2O_DEFINITIONS}
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${G2O_LIBRARIES}
  fmt::fmt
)

ament_auto_package(
  INSTALL_TO_SHARE
  docs
)
