// Copyright (C) FYT Vision Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gimbal_controller/armor_selector.hpp"
#include <angles/angles.h>
#include "gimbal_pipeline/common/robot_description/robot_description_facade.hpp"

#include <limits>

namespace gimbal_controller
{

void ArmorSelector::setSelectionMethod(SelectionMethod method)
{
  selection_method_ = method;
}

ArmorSelectionResult ArmorSelector::selectBest(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double target_yaw,
  int num_armors,
  double target_v_yaw,
  double current_yaw,
  double current_pitch)
{
  bool auto_switch_active = false;
  SelectionMethod effective_method = selection_method_;
  // Outpost uses a dedicated 3-panel tracker and selector semantics. Keep the
  // velocity-based virtual switch limited to normal robot geometries.
  const bool allow_virtual_auto_switch = virtual_auto_switch_enable_ && num_armors != 3;
  if (allow_virtual_auto_switch) {
    const double abs_v_yaw = std::abs(target_v_yaw);
    if (!abs_v_yaw_filter_initialized_) {
      filtered_abs_v_yaw_ = abs_v_yaw;
      abs_v_yaw_filter_initialized_ = true;
    } else {
      filtered_abs_v_yaw_ =
        abs_v_yaw_lpf_alpha_ * abs_v_yaw + (1.0 - abs_v_yaw_lpf_alpha_) * filtered_abs_v_yaw_;
    }

    if (virtual_mode_active_) {
      if (filtered_abs_v_yaw_ < virtual_auto_switch_exit_vyaw_) {
        virtual_mode_active_ = false;
      }
    } else if (filtered_abs_v_yaw_ > virtual_auto_switch_enter_vyaw_) {
      virtual_mode_active_ = true;
    }
    auto_switch_active = virtual_mode_active_;
  } else if (num_armors == 3) {
    virtual_mode_active_ = false;
  }

  if (auto_switch_active) {
    effective_method = virtual_auto_switch_method_;
  }

  switch (effective_method) {
    case SelectionMethod::VIRTUAL_POSE:
      return selectByVirtualPose(
        armor_positions,
        target_center,
        target_yaw,
        num_armors,
        target_v_yaw,
        current_yaw,
        current_pitch);

    case SelectionMethod::VIRTUAL_FIXED_ID: {
      const int fixed_id = auto_switch_active ? virtual_auto_switch_fixed_id_ : virtual_fixed_id_;
      return selectByVirtualFixedId(
        armor_positions,
        target_center,
        num_armors,
        fixed_id,
        current_yaw,
        current_pitch);
    }

    case SelectionMethod::FACING_OR_VIRTUAL_POSE:
      return selectByFacingOrVirtualPose(
        armor_positions,
        target_center,
        target_yaw,
        num_armors,
        target_v_yaw,
        current_yaw,
        current_pitch);

    case SelectionMethod::FACING_OR_VIRTUAL_FIXED_ID: {
      const int fixed_id = auto_switch_active ? virtual_auto_switch_fixed_id_ : virtual_fixed_id_;
      return selectByFacingOrVirtualFixedId(
        armor_positions,
        target_center,
        num_armors,
        fixed_id,
        current_yaw,
        current_pitch);
    }

    case SelectionMethod::SP_VISION_25:
      return selectBySpVision25(
        armor_positions,
        target_center,
        num_armors,
        target_v_yaw,
        current_yaw,
        current_pitch);

    default:
      break;
  }

  switch (effective_method) {
    case SelectionMethod::MIN_MOVEMENT:
      return selectByMinMovement(armor_positions, current_yaw, current_pitch);

    case SelectionMethod::MIN_MOVEMENT_WITH_RADIAL:
      return selectByMinMovementWithRadial(
        armor_positions, target_center, target_v_yaw, current_yaw, current_pitch);

    case SelectionMethod::DECISION_ANGLE: {
      int idx = selectByDecisionAngle(armor_positions, target_center, target_yaw, target_v_yaw);
      ArmorSelectionResult result;
      if (idx >= 0 && idx < static_cast<int>(armor_positions.size())) {
        result.selected_index = idx;
        result.real_selected_index = idx;
        result.position = armor_positions[idx];
        result.real_position = armor_positions[idx];
        result.distance = armor_positions[idx].norm();
        double yaw, pitch;
        calculateYawPitch(armor_positions[idx], current_yaw, yaw, pitch);
        double dy = angles::normalize_angle(yaw - current_yaw);
        double dp = pitch - current_pitch;
        result.gimbal_movement = dy * dy + dp * dp;
      } else {
        // fallback to center
        result.position = target_center;
        result.real_position = target_center;
        result.is_center_fallback = true;
        result.distance = target_center.norm();
      }
      return result;
    }

    case SelectionMethod::MIN_MOVEMENT_WITH_FACING:
    default:
      return selectByMinMovementWithFacing(
        armor_positions, target_center, target_yaw, num_armors,
        current_yaw, current_pitch);
  }
}

void ArmorSelector::setParameters(double side_angle, double min_switching_v_yaw)
{
  side_angle_ = side_angle;
  min_switching_v_yaw_ = min_switching_v_yaw;
}

void ArmorSelector::setFacingParameters(double enter_angle, double exit_angle)
{
  facing_enter_angle_ = enter_angle;
  facing_exit_angle_ = exit_angle;
}

void ArmorSelector::setRadialDynamicParameters(
  bool enable,
  double v_yaw_ref,
  double shrink_ratio,
  double min_angle_deg,
  double bias_gain_deg,
  double max_bias_deg)
{
  radial_dynamic_enable_ = enable;
  radial_dynamic_v_yaw_ref_ = std::max(v_yaw_ref, 1e-6);
  radial_dynamic_shrink_ratio_ = std::clamp(shrink_ratio, 0.0, 1.0);
  radial_dynamic_min_angle_deg_ = std::max(min_angle_deg, 0.0);
  radial_dynamic_bias_gain_deg_ = std::max(bias_gain_deg, 0.0);
  radial_dynamic_max_bias_deg_ = std::max(max_bias_deg, 0.0);
}

void ArmorSelector::setVirtualPoseParameters(
  bool auto_switch_enable,
  double auto_switch_enter_vyaw,
  double auto_switch_exit_vyaw)
{
  virtual_auto_switch_enable_ = auto_switch_enable;
  virtual_auto_switch_enter_vyaw_ = std::max(auto_switch_enter_vyaw, 0.0);
  virtual_auto_switch_exit_vyaw_ = std::max(auto_switch_exit_vyaw, 0.0);
  if (virtual_auto_switch_exit_vyaw_ > virtual_auto_switch_enter_vyaw_) {
    virtual_auto_switch_exit_vyaw_ = virtual_auto_switch_enter_vyaw_;
  }
  if (!virtual_auto_switch_enable_) {
    virtual_mode_active_ = false;
  }
}

void ArmorSelector::setVirtualAutoSwitchMethod(SelectionMethod method)
{
  if (method == SelectionMethod::VIRTUAL_POSE || method == SelectionMethod::VIRTUAL_FIXED_ID) {
    virtual_auto_switch_method_ = method;
  } else {
    virtual_auto_switch_method_ = SelectionMethod::VIRTUAL_POSE;
  }
}

void ArmorSelector::setVirtualAutoSwitchFixedId(int fixed_id)
{
  virtual_auto_switch_fixed_id_ = fixed_id;
}

void ArmorSelector::setVirtualFixedId(int fixed_id)
{
  virtual_fixed_id_ = fixed_id;
}

void ArmorSelector::setSpVisionParameters(
  double low_speed_vyaw,
  double shootable_angle_deg,
  double coming_angle_deg,
  double leaving_angle_deg,
  double outpost_coming_angle_deg,
  double outpost_leaving_angle_deg,
  bool hold_current_until_jump,
  bool zero_speed_fallback)
{
  sp_low_speed_vyaw_ = std::max(low_speed_vyaw, 0.0);
  sp_shootable_angle_deg_ = std::max(shootable_angle_deg, 0.0);
  sp_coming_angle_deg_ = std::max(coming_angle_deg, 0.0);
  sp_leaving_angle_deg_ = std::max(leaving_angle_deg, 0.0);
  sp_outpost_coming_angle_deg_ = std::max(outpost_coming_angle_deg, 0.0);
  sp_outpost_leaving_angle_deg_ = std::max(outpost_leaving_angle_deg, 0.0);
  sp_hold_current_until_jump_ = hold_current_until_jump;
  sp_zero_speed_fallback_ = zero_speed_fallback;
}

void ArmorSelector::resetState()
{
  last_selected_index_ = -1;
  sp_lock_id_ = -1;
  sp_initial_panel_id_ = -1;
  sp_last_front_panel_id_ = -1;
  sp_last_armor_count_ = 0;
  sp_has_jumped_ = false;
  virtual_mode_active_ = false;
  filtered_abs_v_yaw_ = 0.0;
  abs_v_yaw_filter_initialized_ = false;
}

ArmorSelectionResult ArmorSelector::selectByMinMovement(
  const std::vector<Eigen::Vector3d> & armor_positions,
  double current_yaw,
  double current_pitch) const
{
  ArmorSelectionResult result;
  result.selected_index = -1;
  result.gimbal_movement = std::numeric_limits<double>::max();
  result.distance = std::numeric_limits<double>::max();

  if (armor_positions.empty()) {
    return result;
  }

  // 先过滤掉距离最远的装甲板
  auto valid_indices = filterByDistance(armor_positions);

  for (int idx : valid_indices) {
    const auto & pos = armor_positions[idx];

    double yaw, pitch;
    calculateYawPitch(pos, current_yaw, yaw, pitch);

    double yaw_diff = angles::normalize_angle(yaw - current_yaw);
    double pitch_diff = pitch - current_pitch;

    double movement = yaw_diff * yaw_diff + pitch_diff * pitch_diff;
    double distance = pos.norm();

    // 选择云台移动最小的目标
    if (movement < result.gimbal_movement ||
        (std::abs(movement - result.gimbal_movement) < 0.01 && distance < result.distance))
    {
      result.selected_index = idx;
      result.real_selected_index = idx;
      result.position = pos;
      result.real_position = pos;
      result.gimbal_movement = movement;
      result.distance = distance;
    }
  }

  return result;
}

int ArmorSelector::selectByDecisionAngle(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double target_yaw,
  double target_v_yaw) const
{
  if (armor_positions.empty()) {
    return -1;
  }

  std::size_t armors_num = armor_positions.size();

  // 车中心与X轴的夹角
  double alpha = std::atan2(target_center.y(), target_center.x());
  // 观测到的装甲板正面与X轴的夹角
  double beta = target_yaw;

  Eigen::Matrix2d R_odom2center;
  Eigen::Matrix2d R_odom2armor;
  R_odom2center << std::cos(alpha), std::sin(alpha),
                   -std::sin(alpha), std::cos(alpha);
  R_odom2armor << std::cos(beta), std::sin(beta),
                  -std::sin(beta), std::cos(beta);

  Eigen::Matrix2d R_center2armor = R_odom2center.transpose() * R_odom2armor;

  // 决策角度
  double decision_angle = -std::asin(R_center2armor(0, 1));

  // 跳板角度阈值
  double theta = (target_v_yaw > 0 ? side_angle_ : -side_angle_) / 180.0 * M_PI;

  // 避免频繁切换
  if (std::abs(target_v_yaw) < min_switching_v_yaw_) {
    theta = 0;
  }

  double temp_angle = decision_angle + M_PI / armors_num - theta;

  if (temp_angle < 0) {
    temp_angle += 2 * M_PI;
  }

  int selected_id = static_cast<int>(temp_angle / (2 * M_PI / armors_num));
  return selected_id;
}

std::vector<int> ArmorSelector::filterByDistance(
  const std::vector<Eigen::Vector3d> & armor_positions) const
{
  std::vector<int> indices;

  if (armor_positions.size() <= 1) {
    for (size_t i = 0; i < armor_positions.size(); ++i) {
      indices.push_back(static_cast<int>(i));
    }
    return indices;
  }

  // 找到距离最远的装甲板
  double max_dist = 0;
  int max_idx = -1;
  for (size_t i = 0; i < armor_positions.size(); ++i) {
    double dist = armor_positions[i].head<2>().norm();  // 只考虑水平距离
    if (dist > max_dist) {
      max_dist = dist;
      max_idx = static_cast<int>(i);
    }
  }

  // 排除最远的装甲板
  for (size_t i = 0; i < armor_positions.size(); ++i) {
    if (static_cast<int>(i) != max_idx) {
      indices.push_back(static_cast<int>(i));
    }
  }

  return indices;
}

void ArmorSelector::calculateYawPitch(
  const Eigen::Vector3d & target_position,
  double /* current_yaw */,
  double & yaw,
  double & pitch)
{
  double distance_xy = std::sqrt(
    target_position.x() * target_position.x() +
    target_position.y() * target_position.y());

  yaw = std::atan2(target_position.y(), target_position.x());
  pitch = std::atan2(target_position.z(), distance_xy);
}

std::vector<double> ArmorSelector::computeFacingAngles(
  const std::vector<Eigen::Vector3d> & armor_positions,
  double target_yaw,
  int num_armors)
{
  std::vector<double> facing_angles;
  facing_angles.reserve(armor_positions.size());

  for (size_t i = 0; i < armor_positions.size(); ++i) {
    const auto & pos = armor_positions[i];

    // 装甲板法向量方向 (指向外侧)
    // 装甲板 i 相对于机器人正前方偏转 i*(2π/N),
    // 法向量 = target_yaw + i*(2π/N) + π (指向外侧)
    double armor_normal_angle = target_yaw + static_cast<double>(i) * (2.0 * M_PI / num_armors);

    // 云台指向装甲板的方向
    double view_angle = std::atan2(pos.y(), pos.x());

    // 法向量与视线的夹角 (0=正面朝向云台, π/2=侧面, π=背面)
    double facing = std::abs(angles::normalize_angle(armor_normal_angle - view_angle));
    facing_angles.push_back(facing);
  }

  return facing_angles;
}

double ArmorSelector::computeImpactFacingCos(
  const Eigen::Vector3d & target_center,
  const Eigen::Vector3d & armor_position)
{
  return fyt::auto_aim::robot_description::TrackedRobotUsage::computeFacingCos(
    target_center, armor_position);
}

std::vector<double> ArmorSelector::computeImpactFacingAngles(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center)
{
  std::vector<double> facing_angles;
  facing_angles.reserve(armor_positions.size());

  for (const auto & pos : armor_positions) {
    facing_angles.push_back(std::acos(computeImpactFacingCos(target_center, pos)));
  }

  return facing_angles;
}

std::vector<double> ArmorSelector::computeRadialAngles(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double centerline_bias_rad)
{
  std::vector<double> radial_angles;
  radial_angles.reserve(armor_positions.size());

  // 机器人中心->云台原点 (工作空间默认云台原点位于世界坐标原点)
  Eigen::Vector2d center_to_gimbal(-target_center.x(), -target_center.y());
  if (std::abs(centerline_bias_rad) > 1e-9) {
    const double c = std::cos(centerline_bias_rad);
    const double s = std::sin(centerline_bias_rad);
    Eigen::Vector2d rotated;
    rotated.x() = c * center_to_gimbal.x() - s * center_to_gimbal.y();
    rotated.y() = s * center_to_gimbal.x() + c * center_to_gimbal.y();
    center_to_gimbal = rotated;
  }
  const double center_to_gimbal_norm = center_to_gimbal.norm();

  constexpr double kEps = 1e-6;
  if (center_to_gimbal_norm < kEps) {
    radial_angles.assign(armor_positions.size(), M_PI);
    return radial_angles;
  }

  for (const auto & pos : armor_positions) {
    // 机器人中心->装甲板 (径向方向)
    Eigen::Vector2d radial(pos.x() - target_center.x(), pos.y() - target_center.y());
    const double radial_norm = radial.norm();

    if (radial_norm < kEps) {
      radial_angles.push_back(M_PI);
      continue;
    }

    double cos_theta = radial.dot(center_to_gimbal) / (radial_norm * center_to_gimbal_norm);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    radial_angles.push_back(std::acos(cos_theta));
  }

  return radial_angles;
}

std::vector<double> ArmorSelector::computeSignedRadialAngles(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center)
{
  std::vector<double> radial_angles;
  radial_angles.reserve(armor_positions.size());

  Eigen::Vector2d center_to_gimbal(-target_center.x(), -target_center.y());
  constexpr double kEps = 1e-6;
  if (center_to_gimbal.norm() < kEps) {
    radial_angles.assign(armor_positions.size(), 0.0);
    return radial_angles;
  }

  const double reference_angle = std::atan2(center_to_gimbal.y(), center_to_gimbal.x());

  for (const auto & pos : armor_positions) {
    Eigen::Vector2d radial(pos.x() - target_center.x(), pos.y() - target_center.y());
    if (radial.norm() < kEps) {
      radial_angles.push_back(M_PI);
      continue;
    }

    const double radial_angle = std::atan2(radial.y(), radial.x());
    radial_angles.push_back(angles::normalize_angle(radial_angle - reference_angle));
  }

  return radial_angles;
}

ArmorSelectionResult ArmorSelector::selectMinMovementFromIndices(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const std::vector<int> & candidate_indices,
  const std::vector<double> * facing_angles,
  double current_yaw,
  double current_pitch) const
{
  ArmorSelectionResult result;
  result.selected_index = -1;
  result.gimbal_movement = std::numeric_limits<double>::max();
  result.distance = std::numeric_limits<double>::max();

  for (int idx : candidate_indices) {
    if (idx < 0 || idx >= static_cast<int>(armor_positions.size())) {
      continue;
    }

    const auto & pos = armor_positions[idx];

    double yaw, pitch;
    calculateYawPitch(pos, current_yaw, yaw, pitch);

    const double yaw_diff = angles::normalize_angle(yaw - current_yaw);
    const double pitch_diff = pitch - current_pitch;
    const double movement = yaw_diff * yaw_diff + pitch_diff * pitch_diff;
    const double distance = pos.norm();

    if (movement < result.gimbal_movement ||
        (std::abs(movement - result.gimbal_movement) < 0.01 && distance < result.distance))
    {
      result.selected_index = idx;
      result.real_selected_index = idx;
      result.position = pos;
      result.real_position = pos;
      result.gimbal_movement = movement;
      result.distance = distance;
      if (facing_angles && idx < static_cast<int>(facing_angles->size())) {
        result.facing_angle = (*facing_angles)[idx];
      }
    }
  }

  return result;
}

ArmorSelectionResult ArmorSelector::selectByMinMovementWithFacing(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double target_yaw,
  int num_armors,
  double current_yaw,
  double current_pitch)
{
  ArmorSelectionResult result;
  result.selected_index = -1;
  result.gimbal_movement = std::numeric_limits<double>::max();
  result.distance = std::numeric_limits<double>::max();
  result.facing_angle = 0.0;
  result.is_center_fallback = false;

  if (armor_positions.empty()) {
    // Fallback to center
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    last_selected_index_ = -1;
    return result;
  }

  // 1. 距离过滤 (排除最远板)
  auto dist_valid_indices = filterByDistance(armor_positions);

  // 2. 计算 facing angles
  auto facing_angles = computeFacingAngles(armor_positions, target_yaw, num_armors);

  // 3. Facing 过滤 (Hysteresis 双阈值)
  double enter_rad = facing_enter_angle_ * M_PI / 180.0;
  double exit_rad = facing_exit_angle_ * M_PI / 180.0;

  std::vector<int> facing_valid_indices;
  for (int idx : dist_valid_indices) {
    double fa = facing_angles[idx];
    if (idx == last_selected_index_) {
      // 已锁定: 使用退出阈值 (更宽松)
      if (fa <= exit_rad) {
        facing_valid_indices.push_back(idx);
      }
    } else {
      // 未锁定: 使用进入阈值 (更严格)
      if (fa <= enter_rad) {
        facing_valid_indices.push_back(idx);
      }
    }
  }

  // 4. 如果过滤后为空, fallback 到目标中心
  if (facing_valid_indices.empty()) {
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    last_selected_index_ = -1;
    return result;
  }

  // 5. 在通过 facing 过滤的装甲板中, 选择云台移动最小的
  for (int idx : facing_valid_indices) {
    const auto & pos = armor_positions[idx];

    double yaw, pitch;
    calculateYawPitch(pos, current_yaw, yaw, pitch);

    double yaw_diff = angles::normalize_angle(yaw - current_yaw);
    double pitch_diff = pitch - current_pitch;

    double movement = yaw_diff * yaw_diff + pitch_diff * pitch_diff;
    double distance = pos.norm();

    if (movement < result.gimbal_movement ||
        (std::abs(movement - result.gimbal_movement) < 0.01 && distance < result.distance))
    {
      result.selected_index = idx;
      result.real_selected_index = idx;
      result.position = pos;
      result.real_position = pos;
      result.gimbal_movement = movement;
      result.distance = distance;
      result.facing_angle = facing_angles[idx];
    }
  }

  // 更新记忆
  last_selected_index_ = result.selected_index;

  return result;
}

ArmorSelectionResult ArmorSelector::selectByMinMovementWithRadial(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double target_v_yaw,
  double current_yaw,
  double current_pitch)
{
  ArmorSelectionResult result;
  result.selected_index = -1;
  result.gimbal_movement = std::numeric_limits<double>::max();
  result.distance = std::numeric_limits<double>::max();
  result.facing_angle = 0.0;
  result.is_center_fallback = false;

  if (armor_positions.empty()) {
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    last_selected_index_ = -1;
    return result;
  }

  // 1. 距离过滤 (排除最远板)
  auto dist_valid_indices = filterByDistance(armor_positions);

  double speed_norm = 0.0;
  if (radial_dynamic_enable_) {
    speed_norm = std::clamp(
      std::abs(target_v_yaw) / radial_dynamic_v_yaw_ref_, 0.0, 1.0);
  }

  // 2. 计算径向夹角（允许角平分线随角速度偏移）
  // 约定: target_v_yaw > 0 时角平分线向左(逆时针, +yaw)偏移。
  const double bias_mag_deg = std::min(
    radial_dynamic_bias_gain_deg_ * speed_norm,
    radial_dynamic_max_bias_deg_);
  const double bias_deg = (target_v_yaw >= 0.0 ? 1.0 : -1.0) * bias_mag_deg;
  const double bias_rad = bias_deg * M_PI / 180.0;
  auto radial_angles = computeRadialAngles(armor_positions, target_center, bias_rad);

  // 3. 径向夹角过滤 (Hysteresis 双阈值)
  // 角速度越大，阈值越小；角度单位为度，角速度单位为 rad/s。
  double enter_deg = facing_enter_angle_;
  double exit_deg = facing_exit_angle_;
  if (radial_dynamic_enable_) {
    const double scale = 1.0 - radial_dynamic_shrink_ratio_ * speed_norm;
    enter_deg = std::max(enter_deg * scale, radial_dynamic_min_angle_deg_);
    exit_deg = std::max(exit_deg * scale, radial_dynamic_min_angle_deg_);
  }
  double enter_rad = enter_deg * M_PI / 180.0;
  double exit_rad = exit_deg * M_PI / 180.0;

  std::vector<int> radial_valid_indices;
  for (int idx : dist_valid_indices) {
    double ra = radial_angles[idx];
    if (idx == last_selected_index_) {
      if (ra <= exit_rad) {
        radial_valid_indices.push_back(idx);
      }
    } else {
      if (ra <= enter_rad) {
        radial_valid_indices.push_back(idx);
      }
    }
  }

  // 4. 如果过滤后为空, fallback 到目标中心
  if (radial_valid_indices.empty()) {
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    last_selected_index_ = -1;
    return result;
  }

  // 5. 在通过径向过滤的装甲板中, 选择径向夹角最小的
  double best_radial_angle = std::numeric_limits<double>::max();
  constexpr double kAngleEps = 1e-6;
  for (int idx : radial_valid_indices) {
    const auto & pos = armor_positions[idx];

    double yaw, pitch;
    calculateYawPitch(pos, current_yaw, yaw, pitch);

    double yaw_diff = angles::normalize_angle(yaw - current_yaw);
    double pitch_diff = pitch - current_pitch;

    double movement = yaw_diff * yaw_diff + pitch_diff * pitch_diff;
    double distance = pos.norm();

    const double radial_angle = radial_angles[idx];

    if (radial_angle + kAngleEps < best_radial_angle ||
        (std::abs(radial_angle - best_radial_angle) <= kAngleEps &&
         (movement < result.gimbal_movement ||
          (std::abs(movement - result.gimbal_movement) < 0.01 && distance < result.distance))))
    {
      best_radial_angle = radial_angle;
      result.selected_index = idx;
      result.real_selected_index = idx;
      result.position = pos;
      result.real_position = pos;
      result.gimbal_movement = movement;
      result.distance = distance;
      result.facing_angle = radial_angle;
    }
  }

  // 更新记忆
  last_selected_index_ = result.selected_index;

  return result;
}

ArmorSelectionResult ArmorSelector::selectByFacingOrVirtualPose(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double target_yaw,
  int num_armors,
  double target_v_yaw,
  double current_yaw,
  double current_pitch)
{
  if (armor_positions.empty()) {
    last_selected_index_ = -1;
    return selectByVirtualPose(
      armor_positions, target_center, target_yaw, num_armors, target_v_yaw,
      current_yaw, current_pitch);
  }

  const auto valid_indices = filterByDistance(armor_positions);
  const auto facing_angles = computeImpactFacingAngles(armor_positions, target_center);
  const double enter_rad = facing_enter_angle_ * M_PI / 180.0;
  const double exit_rad = facing_exit_angle_ * M_PI / 180.0;

  std::vector<int> facing_valid_indices;
  facing_valid_indices.reserve(valid_indices.size());

  for (int idx : valid_indices) {
    if (idx < 0 || idx >= static_cast<int>(facing_angles.size())) {
      continue;
    }

    const double threshold = (idx == last_selected_index_) ? exit_rad : enter_rad;
    if (facing_angles[idx] <= threshold) {
      facing_valid_indices.push_back(idx);
    }
  }

  if (!facing_valid_indices.empty()) {
    auto result = selectMinMovementFromIndices(
      armor_positions, facing_valid_indices, &facing_angles, current_yaw, current_pitch);
    result.is_center_fallback = false;
    result.is_virtual_target = false;
    last_selected_index_ = result.selected_index;
    return result;
  }

  // No real armor is currently hittable under the same facing semantics used by fire_advice.
  // Keep the gimbal stable by aiming at an ideal virtual armor instead of the robot center.
  last_selected_index_ = -1;
  return selectByVirtualPose(
    armor_positions, target_center, target_yaw, num_armors, target_v_yaw,
    current_yaw, current_pitch);
}

ArmorSelectionResult ArmorSelector::selectByFacingOrVirtualFixedId(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  int num_armors,
  int fixed_id,
  double current_yaw,
  double current_pitch)
{
  if (armor_positions.empty()) {
    last_selected_index_ = -1;
    return selectByVirtualFixedId(
      armor_positions, target_center, num_armors, fixed_id, current_yaw, current_pitch);
  }

  const int armor_count = std::max(1, std::min(num_armors, static_cast<int>(armor_positions.size())));
  int fixed_idx = fixed_id % armor_count;
  if (fixed_idx < 0) {
    fixed_idx += armor_count;
  }

  if (fixed_idx >= 0 && fixed_idx < static_cast<int>(armor_positions.size())) {
    const auto facing_angles = computeImpactFacingAngles(armor_positions, target_center);
    const double threshold =
      (fixed_idx == last_selected_index_ ? facing_exit_angle_ : facing_enter_angle_) *
      M_PI / 180.0;

    if (fixed_idx < static_cast<int>(facing_angles.size()) &&
        facing_angles[fixed_idx] <= threshold)
    {
      std::vector<int> fixed_candidate{fixed_idx};
      auto result = selectMinMovementFromIndices(
        armor_positions, fixed_candidate, &facing_angles, current_yaw, current_pitch);
      result.is_center_fallback = false;
      result.is_virtual_target = false;
      last_selected_index_ = result.selected_index;
      return result;
    }
  }

  // The requested ID exists but is not facing enough to be fired at. Fall back to the
  // fixed-ID virtual armor so the controller keeps the same semantic target.
  last_selected_index_ = -1;
  return selectByVirtualFixedId(
    armor_positions, target_center, num_armors, fixed_id, current_yaw, current_pitch);
}

ArmorSelectionResult ArmorSelector::selectBySpVision25(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  int num_armors,
  double target_v_yaw,
  double current_yaw,
  double current_pitch)
{
  auto centerFallback = [&]() {
    ArmorSelectionResult result;
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    result.selected_index = -1;
    result.real_selected_index = -1;
    return result;
  };

  if (armor_positions.empty()) {
    resetState();
    return centerFallback();
  }

  int armor_count = static_cast<int>(armor_positions.size());
  if (num_armors > 0) {
    armor_count = std::min(num_armors, armor_count);
  }
  if (armor_count <= 0) {
    resetState();
    return centerFallback();
  }

  if (sp_last_armor_count_ != armor_count) {
    sp_lock_id_ = -1;
    sp_initial_panel_id_ = -1;
    sp_last_front_panel_id_ = -1;
    sp_has_jumped_ = false;
    sp_last_armor_count_ = armor_count;
  }

  const auto signed_radial_angles = computeSignedRadialAngles(armor_positions, target_center);
  if (signed_radial_angles.empty()) {
    return centerFallback();
  }

  int front_idx = 0;
  double best_abs_delta = std::numeric_limits<double>::max();
  for (int i = 0; i < armor_count; ++i) {
    const double abs_delta = std::abs(signed_radial_angles[static_cast<size_t>(i)]);
    if (abs_delta < best_abs_delta) {
      best_abs_delta = abs_delta;
      front_idx = i;
    }
  }

  const double shootable_rad = sp_shootable_angle_deg_ * M_PI / 180.0;
  if (sp_initial_panel_id_ < 0) {
    sp_initial_panel_id_ = front_idx;
    sp_last_front_panel_id_ = front_idx;
  } else if (
    front_idx != sp_last_front_panel_id_ &&
    front_idx != sp_initial_panel_id_ &&
    best_abs_delta <= shootable_rad)
  {
    sp_has_jumped_ = true;
    sp_last_front_panel_id_ = front_idx;
  } else {
    sp_last_front_panel_id_ = front_idx;
  }

  auto selectIndex = [&](int idx, double facing_angle) {
    std::vector<int> candidate{idx};
    std::vector<double> abs_angles = signed_radial_angles;
    for (double & angle : abs_angles) {
      angle = std::abs(angle);
    }
    auto result = selectMinMovementFromIndices(
      armor_positions, candidate, &abs_angles, current_yaw, current_pitch);
    if (result.selected_index < 0) {
      last_selected_index_ = -1;
      return centerFallback();
    }
    result.facing_angle = facing_angle;
    result.is_center_fallback = false;
    result.is_virtual_target = false;
    last_selected_index_ = result.selected_index;
    return result;
  };

  if (
    sp_hold_current_until_jump_ &&
    !sp_has_jumped_ &&
    sp_initial_panel_id_ >= 0 &&
    sp_initial_panel_id_ < armor_count)
  {
    const double facing_angle = std::abs(signed_radial_angles[sp_initial_panel_id_]);
    return selectIndex(sp_initial_panel_id_, facing_angle);
  }

  const bool is_outpost_like = (armor_count == 3);
  const bool low_speed = !is_outpost_like && std::abs(target_v_yaw) <= sp_low_speed_vyaw_;

  if (low_speed) {
    std::vector<int> shootable_indices;
    shootable_indices.reserve(static_cast<size_t>(armor_count));
    for (int i = 0; i < armor_count; ++i) {
      if (std::abs(signed_radial_angles[static_cast<size_t>(i)]) <= shootable_rad) {
        shootable_indices.push_back(i);
      }
    }

    if (shootable_indices.empty()) {
      sp_lock_id_ = -1;
      last_selected_index_ = -1;
      return centerFallback();
    }

    if (shootable_indices.size() > 1) {
      const bool lock_still_valid =
        std::find(shootable_indices.begin(), shootable_indices.end(), sp_lock_id_) !=
        shootable_indices.end();

      if (!lock_still_valid) {
        sp_lock_id_ = *std::min_element(
          shootable_indices.begin(),
          shootable_indices.end(),
          [&](int lhs, int rhs) {
            return std::abs(signed_radial_angles[static_cast<size_t>(lhs)]) <
                   std::abs(signed_radial_angles[static_cast<size_t>(rhs)]);
          });
      }

      const double facing_angle = std::abs(signed_radial_angles[sp_lock_id_]);
      return selectIndex(sp_lock_id_, facing_angle);
    }

    sp_lock_id_ = -1;
    const int selected_idx = shootable_indices.front();
    const double facing_angle = std::abs(signed_radial_angles[selected_idx]);
    return selectIndex(selected_idx, facing_angle);
  }

  sp_lock_id_ = -1;
  const double coming_angle =
    (is_outpost_like ? sp_outpost_coming_angle_deg_ : sp_coming_angle_deg_) * M_PI / 180.0;
  const double leaving_angle =
    (is_outpost_like ? sp_outpost_leaving_angle_deg_ : sp_leaving_angle_deg_) * M_PI / 180.0;

  if (std::abs(target_v_yaw) < 1e-6) {
    if (sp_zero_speed_fallback_) {
      return selectIndex(front_idx, best_abs_delta);
    }
    last_selected_index_ = -1;
    return centerFallback();
  }

  std::vector<int> coming_indices;
  coming_indices.reserve(static_cast<size_t>(armor_count));
  for (int i = 0; i < armor_count; ++i) {
    const double delta = signed_radial_angles[static_cast<size_t>(i)];
    if (std::abs(delta) > coming_angle) {
      continue;
    }

    if (target_v_yaw > 0.0 && delta < leaving_angle) {
      coming_indices.push_back(i);
    }
    if (target_v_yaw < 0.0 && delta > -leaving_angle) {
      coming_indices.push_back(i);
    }
  }

  if (!coming_indices.empty()) {
    std::vector<double> abs_angles = signed_radial_angles;
    for (double & angle : abs_angles) {
      angle = std::abs(angle);
    }
    auto result = selectMinMovementFromIndices(
      armor_positions, coming_indices, &abs_angles, current_yaw, current_pitch);
    if (result.selected_index >= 0) {
      result.facing_angle = std::abs(signed_radial_angles[static_cast<size_t>(result.selected_index)]);
      result.is_center_fallback = false;
      result.is_virtual_target = false;
      last_selected_index_ = result.selected_index;
      return result;
    }
  }

  last_selected_index_ = -1;
  return centerFallback();
}

ArmorSelectionResult ArmorSelector::selectByVirtualPose(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  double target_yaw,
  int num_armors,
  double target_v_yaw,
  double current_yaw,
  double current_pitch) const
{
  ArmorSelectionResult result;
  result.selected_index = -1;
  result.real_selected_index = -1;
  result.gimbal_movement = std::numeric_limits<double>::max();
  result.distance = std::numeric_limits<double>::max();

  if (armor_positions.empty()) {
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    return result;
  }

  const int armor_count = std::max(1, std::min(num_armors, static_cast<int>(armor_positions.size())));
  const double step = 2.0 * M_PI / static_cast<double>(armor_count);

  const double yaw_gap = angles::normalize_angle(target_yaw - current_yaw);
  double rotation_sign = 0.0;
  if (std::abs(yaw_gap) > 1e-6) {
    rotation_sign = (yaw_gap > 0.0) ? 1.0 : -1.0;
  } else if (std::abs(target_v_yaw) > 1e-6) {
    rotation_sign = (target_v_yaw > 0.0) ? 1.0 : -1.0;
  }

  int base_idx = -1;
  double best_abs_yaw_delta = std::numeric_limits<double>::max();

  for (int i = 0; i < armor_count; ++i) {
    const double armor_yaw = target_yaw + static_cast<double>(i) * step;
    const double yaw_delta = angles::normalize_angle(armor_yaw - current_yaw);
    const bool is_unturned =
      (rotation_sign == 0.0) ||
      (rotation_sign > 0.0 && yaw_delta >= -1e-6) ||
      (rotation_sign < 0.0 && yaw_delta <= 1e-6);
    if (!is_unturned) {
      continue;
    }
    const double abs_delta = std::abs(yaw_delta);
    if (abs_delta < best_abs_yaw_delta) {
      best_abs_yaw_delta = abs_delta;
      base_idx = i;
    }
  }

  if (base_idx < 0) {
    for (int i = 0; i < armor_count; ++i) {
      const double armor_yaw = target_yaw + static_cast<double>(i) * step;
      const double abs_delta = std::abs(angles::normalize_angle(armor_yaw - current_yaw));
      if (abs_delta < best_abs_yaw_delta) {
        best_abs_yaw_delta = abs_delta;
        base_idx = i;
      }
    }
  }

  if (base_idx < 0 || base_idx >= static_cast<int>(armor_positions.size())) {
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    return result;
  }

  const Eigen::Vector3d base_position = armor_positions[base_idx];
  const Eigen::Vector3d base_offset = base_position - target_center;
  const double base_offset_norm_xy = base_offset.head<2>().norm();

  Eigen::Vector2d center_to_gimbal_xy(-target_center.x(), -target_center.y());
  if (center_to_gimbal_xy.norm() < 1e-6 || base_offset_norm_xy < 1e-6) {
    result.selected_index = base_idx;
    result.real_selected_index = base_idx;
    result.position = base_position;
    result.real_position = base_position;
    result.distance = base_position.norm();
    result.gimbal_movement = best_abs_yaw_delta * best_abs_yaw_delta;
    return result;
  }

  center_to_gimbal_xy.normalize();
  const double desired_angle = std::atan2(center_to_gimbal_xy.y(), center_to_gimbal_xy.x());
  const double base_angle = std::atan2(base_offset.y(), base_offset.x());
  const double delta_yaw = angles::normalize_angle(desired_angle - base_angle);

  const double c = std::cos(delta_yaw);
  const double s = std::sin(delta_yaw);
  Eigen::Vector2d rotated_xy;
  rotated_xy.x() = c * base_offset.x() - s * base_offset.y();
  rotated_xy.y() = s * base_offset.x() + c * base_offset.y();

  Eigen::Vector3d virtual_offset(base_offset.x(), base_offset.y(), base_offset.z());
  virtual_offset.x() = rotated_xy.x();
  virtual_offset.y() = rotated_xy.y();

  const Eigen::Vector3d virtual_position = target_center + virtual_offset;

  double yaw, pitch;
  calculateYawPitch(virtual_position, current_yaw, yaw, pitch);
  const double yaw_diff = angles::normalize_angle(yaw - current_yaw);
  const double pitch_diff = pitch - current_pitch;

  result.selected_index = base_idx;
  result.real_selected_index = base_idx;
  result.position = virtual_position;
  result.real_position = base_position;
  result.distance = virtual_position.norm();
  result.gimbal_movement = yaw_diff * yaw_diff + pitch_diff * pitch_diff;
  result.facing_angle = 0.0;
  result.is_center_fallback = false;
  result.is_virtual_target = true;
  result.virtual_delta_yaw = delta_yaw;
  result.virtual_robot_yaw = angles::normalize_angle(target_yaw + delta_yaw);

  return result;
}

ArmorSelectionResult ArmorSelector::selectByVirtualFixedId(
  const std::vector<Eigen::Vector3d> & armor_positions,
  const Eigen::Vector3d & target_center,
  int num_armors,
  int fixed_id,
  double current_yaw,
  double current_pitch) const
{
  ArmorSelectionResult result;
  result.selected_index = -1;
  result.real_selected_index = -1;
  result.gimbal_movement = std::numeric_limits<double>::max();
  result.distance = std::numeric_limits<double>::max();

  if (armor_positions.empty()) {
    result.position = target_center;
    result.real_position = target_center;
    result.is_center_fallback = true;
    result.distance = target_center.norm();
    return result;
  }

  const int armor_count = std::max(1, std::min(num_armors, static_cast<int>(armor_positions.size())));
  int fixed_idx = fixed_id % armor_count;
  if (fixed_idx < 0) {
    fixed_idx += armor_count;
  }

  const Eigen::Vector3d base_position = armor_positions[fixed_idx];
  const Eigen::Vector3d base_offset = base_position - target_center;
  const double base_offset_norm_xy = base_offset.head<2>().norm();

  Eigen::Vector2d center_to_gimbal_xy(-target_center.x(), -target_center.y());
  if (center_to_gimbal_xy.norm() < 1e-6 || base_offset_norm_xy < 1e-6) {
    result.selected_index = fixed_idx;
    result.real_selected_index = fixed_idx;
    result.position = base_position;
    result.real_position = base_position;
    result.distance = base_position.norm();
    return result;
  }

  center_to_gimbal_xy.normalize();
  const double desired_angle = std::atan2(center_to_gimbal_xy.y(), center_to_gimbal_xy.x());
  const double base_angle = std::atan2(base_offset.y(), base_offset.x());
  const double delta_yaw = angles::normalize_angle(desired_angle - base_angle);

  const double c = std::cos(delta_yaw);
  const double s = std::sin(delta_yaw);
  Eigen::Vector2d rotated_xy;
  rotated_xy.x() = c * base_offset.x() - s * base_offset.y();
  rotated_xy.y() = s * base_offset.x() + c * base_offset.y();

  Eigen::Vector3d virtual_offset(base_offset.x(), base_offset.y(), base_offset.z());
  virtual_offset.x() = rotated_xy.x();
  virtual_offset.y() = rotated_xy.y();

  const Eigen::Vector3d virtual_position = target_center + virtual_offset;

  double yaw, pitch;
  calculateYawPitch(virtual_position, current_yaw, yaw, pitch);
  const double yaw_diff = angles::normalize_angle(yaw - current_yaw);
  const double pitch_diff = pitch - current_pitch;

  result.selected_index = fixed_idx;
  result.real_selected_index = fixed_idx;
  result.position = virtual_position;
  result.real_position = base_position;
  result.distance = virtual_position.norm();
  result.gimbal_movement = yaw_diff * yaw_diff + pitch_diff * pitch_diff;
  result.facing_angle = 0.0;
  result.is_center_fallback = false;
  result.is_virtual_target = true;
  result.virtual_delta_yaw = delta_yaw;
  result.virtual_robot_yaw = desired_angle;

  return result;
}

}  // namespace gimbal_controller
