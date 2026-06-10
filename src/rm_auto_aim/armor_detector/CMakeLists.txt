cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)
add_definitions(-O3)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(G2O REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(armor_pnp_refiner REQUIRED)
ament_auto_find_build_dependencies()

###########
cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)
add_definitions(-O3)


## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(ABSEIL_DIR "/usr/local/include/absl" CACHE PATH "Path to Abseil libraries")  
#set(FMT_DIR "/usr/bin" CACHE PATH "Path to FMT libraries")  
#set(TBB_DIR "/usr/local/include/tbb" CACHE PATH "Path to TBB libraries")
#set(OpenVINO_DIR "/home/hfytcjrbomaster/openvino/bin/intel64/Release" CACHE PATH "Path to OpenVINO libraries")

#######################
## Find dependencies ##
#######################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXTERNAL_INCLUDE_DIRS ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${G2O_LIBRARIES})

#find_package(absl REQUIRED HINTS ${ABSEIL_DIR})
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(G2O REQUIRED)
#find_package(fmt REQUIRED HINTS ${FMT_DIR})
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(armor_pnp_refiner REQUIRED)
#find_package(openVINO REQUIRED HINTS ${OpenVINO_DIR} COMPONENTS Runtime ONNX)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC
${OpenCV_INCLUDE_DIRS}
${G2O_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
${G2O_DEFINITIONS}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${G2O_LIBRARIES}
fmt::fmt
tbb
)


rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN fyt::auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
    ament_cmake_lint_cmake
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)
  ament_add_gtest(test_detector test/test_detector.cpp)
  target_link_libraries(test_detector ${PROJECT_NAME})

endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  docs
  model
)
