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

#ifndef GIMBAL_CONTROLLER__ARMOR_SELECTOR_HPP_
#define GIMBAL_CONTROLLER__ARMOR_SELECTOR_HPP_

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace gimbal_controller
{

/**
 * @brief 装甲板选择结果
 */
struct ArmorSelectionResult
{
  int selected_index{-1};       // 选中的装甲板索引
  Eigen::Vector3d position;     // 选中装甲板的位置
  Eigen::Vector3d real_position;  // 真实装甲板位置（虚拟模式下用于开火参考）
  double gimbal_movement{0.0};  // 云台移动量 (yaw^2 + pitch^2)
  double distance{0.0};         // 目标距离
  double facing_angle{0.0};     // 选中装甲板的朝向角 (弧度, 0=正面)
  bool is_center_fallback{false}; // 是否因全部过滤而 fallback 到车体中心
  bool is_virtual_target{false};  // 当前控制目标是否为虚拟装甲板
  int real_selected_index{-1};    // 虚拟模式下对应的真实装甲板索引
  double virtual_robot_yaw{0.0};  // 虚拟机器人 yaw（用于诊断）
  double virtual_delta_yaw{0.0};  // 相对估计 yaw 的附加旋转量（用于诊断）
};

/**
 * @brief 装甲板选择器
 * 
 * 根据当前云台姿态和装甲板位置，选择最优的打击目标。
 * 支持:
 *  - 基础最小运动选板 (selectByMinMovement)
 *  - 带正面朝向过滤+hysteresis 的选板 (selectByMinMovementWithFacing)
 *  - 基于决策角的选板 (selectByDecisionAngle)
 */
class ArmorSelector
{
public:
  /**
   * @brief 选板策略枚举
   *  - MIN_MOVEMENT_WITH_FACING : 最小运动量 + 正面朝向 Hysteresis 过滤 (默认)
   *  - MIN_MOVEMENT_WITH_RADIAL : 径向夹角过滤后按最小径向夹角选板
   *  - MIN_MOVEMENT             : 最小运动量，无朝向过滤
   *  - DECISION_ANGLE           : 传统决策角算法 (与 armor_solver 原版一致)
   *  - VIRTUAL_POSE             : 虚拟姿态选板（最小旋转 + 中心连线朝向）
   *  - VIRTUAL_FIXED_ID         : 固定 ID 虚拟装甲板（仅生成指定 ID 的虚拟板）
   *  - FACING_OR_VIRTUAL_POSE   : 可打真实板优先，否则退回虚拟姿态
   *  - FACING_OR_VIRTUAL_FIXED_ID : 指定 ID 可打优先，否则退回该 ID 的虚拟姿态
   *  - SP_VISION_25           : 参考 sp_vision_25 Aimer::choose_aim_point 的来/离角选板
   */
  enum class SelectionMethod
  {
    MIN_MOVEMENT_WITH_FACING = 0,
    MIN_MOVEMENT             = 1,
    DECISION_ANGLE           = 2,
    MIN_MOVEMENT_WITH_RADIAL = 3,
    VIRTUAL_POSE             = 4,
    VIRTUAL_FIXED_ID         = 5,
    FACING_OR_VIRTUAL_POSE   = 6,
    FACING_OR_VIRTUAL_FIXED_ID = 7,
    SP_VISION_25             = 8,
  };

  ArmorSelector() = default;
  ~ArmorSelector() = default;

  /**
   * @brief 设置选板策略
   * @param method 选板方法枚举
   */
  void setSelectionMethod(SelectionMethod method);

  /**
   * @brief 统一选板入口，根据 setSelectionMethod 配置路由到对应算法
   * 
   * @param armor_positions 各装甲板的世界坐标位置
   * @param target_center   目标机器人中心位置
   * @param target_yaw      目标机器人 yaw 角 (弧度)
   * @param num_armors      装甲板总数
   * @param target_v_yaw    目标 yaw 角速度 (仅 DECISION_ANGLE 模式使用)
   * @param current_yaw     当前云台 yaw 角 (弧度)
   * @param current_pitch   当前云台 pitch 角 (弧度)
   * @return 选择结果
   */
  ArmorSelectionResult selectBest(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double target_yaw,
    int num_armors,
    double target_v_yaw,
    double current_yaw,
    double current_pitch);

  /**
   * @brief 设置基础选择参数
   * @param side_angle 侧向角度阈值 (度)
   * @param min_switching_v_yaw 最小切换角速度阈值
   */
  void setParameters(double side_angle, double min_switching_v_yaw);

  /**
   * @brief 设置 Facing 过滤参数 (hysteresis 双阈值)
   * @param enter_angle 进入阈值 (度): 未锁定时，facing angle ≤ enter 才允许选中
   * @param exit_angle 退出阈值 (度): 已锁定时，facing angle > exit 才释放
   */
  void setFacingParameters(double enter_angle, double exit_angle);

  /**
   * @brief 设置径向选板动态阈值参数
   * @param enable 是否启用动态阈值
   * @param v_yaw_ref 归一化参考角速度 (rad/s)
   * @param shrink_ratio 最大收缩比例 [0, 1]
   * @param min_angle_deg 动态收缩后的角度下限 (度)
   * @param bias_gain_deg 角平分线偏移增益 (度, 随速度线性增长)
   * @param max_bias_deg 角平分线偏移上限 (度)
   */
  void setRadialDynamicParameters(
    bool enable,
    double v_yaw_ref,
    double shrink_ratio,
    double min_angle_deg,
    double bias_gain_deg,
    double max_bias_deg);

  /**
   * @brief 设置虚拟姿态选板参数
   * @param auto_switch_enable 是否按角速度自动切换到虚拟模式
   * @param auto_switch_enter_vyaw 自动切入阈值 (rad/s)
   * @param auto_switch_exit_vyaw 自动退出阈值 (rad/s)
   */
  void setVirtualPoseParameters(
    bool auto_switch_enable,
    double auto_switch_enter_vyaw,
    double auto_switch_exit_vyaw);

  /**
   * @brief 设置自动切换时的虚拟模式选择方法
   * @param method 虚拟模式方法 (virtual_pose | virtual_fixed_id)
   */
  void setVirtualAutoSwitchMethod(SelectionMethod method);

  /**
   * @brief 设置自动切换时的固定虚拟装甲板 ID
   * @param fixed_id 指定的装甲板索引 ID
   */
  void setVirtualAutoSwitchFixedId(int fixed_id);

  /**
   * @brief 设置固定虚拟装甲板 ID
   * @param fixed_id 指定的装甲板索引 ID
   */
  void setVirtualFixedId(int fixed_id);

  /**
   * @brief 设置 sp_vision_25 风格选板参数
   * @param low_speed_vyaw 低速/非小陀螺角速度阈值(rad/s)
   * @param shootable_angle_deg 低速时可射击角范围(度)
   * @param coming_angle_deg 小陀螺来板角(度)
   * @param leaving_angle_deg 小陀螺离板角(度)
   * @param outpost_coming_angle_deg 前哨站来板角(度)
   * @param outpost_leaving_angle_deg 前哨站离板角(度)
   * @param hold_current_until_jump 是否在检测到正面板索引跳变前保持初始板
   * @param zero_speed_fallback 高速/前哨站分支角速度近零时是否回退到最正对板
   */
  void setSpVisionParameters(
    double low_speed_vyaw,
    double shootable_angle_deg,
    double coming_angle_deg,
    double leaving_angle_deg,
    double outpost_coming_angle_deg,
    double outpost_leaving_angle_deg,
    bool hold_current_until_jump,
    bool zero_speed_fallback);

  /**
   * @brief 重置内部记忆状态 (目标丢失时调用)
   */
  void resetState();

  /**
   * @brief 选择最佳装甲板 (基于云台移动最小, 无 facing 过滤)
   * @param armor_positions 各装甲板的世界坐标位置
   * @param current_yaw 当前云台yaw角 (弧度)
   * @param current_pitch 当前云台pitch角 (弧度)
   * @return 选择结果
   */
  ArmorSelectionResult selectByMinMovement(
    const std::vector<Eigen::Vector3d> & armor_positions,
    double current_yaw,
    double current_pitch) const;

  /**
   * @brief 选择最佳装甲板 (基于云台移动最小 + Facing 过滤 + Hysteresis)
   * 
   * 对每个装甲板计算其法向量与云台视线方向的夹角 (facing angle),
   * 使用双阈值 (enter/exit) 进行 hysteresis 过滤,
   * 然后在通过过滤的装甲板中选择云台移动最小的目标。
   * 如果所有装甲板都被过滤掉, 则 fallback 到目标中心。
   * 
   * @param armor_positions 各装甲板的世界坐标位置
   * @param target_center 目标机器人中心位置
   * @param target_yaw 目标机器人 yaw 角 (弧度)
   * @param num_armors 装甲板总数
   * @param current_yaw 当前云台 yaw 角 (弧度)
   * @param current_pitch 当前云台 pitch 角 (弧度)
   * @return 选择结果 (注意检查 is_center_fallback)
   */
  ArmorSelectionResult selectByMinMovementWithFacing(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double target_yaw,
    int num_armors,
    double current_yaw,
    double current_pitch);

  /**
   * @brief 选择最佳装甲板 (径向夹角过滤 + Hysteresis + 最小径向夹角)
   *
   * 径向夹角定义为: (机器人中心->装甲板) 与 (机器人中心->云台原点) 的夹角。
   * 先使用双阈值做 hysteresis 过滤，再在候选中选径向夹角最小的装甲板。
   * 若过滤后为空，则 fallback 到目标中心。
   */
  ArmorSelectionResult selectByMinMovementWithRadial(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double target_v_yaw,
    double current_yaw,
    double current_pitch);

  /**
   * @brief 选择虚拟装甲板 (基于估计 yaw 先选最近未转到位真实板，再最小旋转生成虚拟目标)
   * @param armor_positions 真实装甲板位置
   * @param target_center 目标中心
   * @param target_yaw 估计 yaw
   * @param num_armors 装甲板数量
   * @param target_v_yaw 估计 yaw 角速度
   * @param current_yaw 当前云台 yaw
   * @param current_pitch 当前云台 pitch
   */
  ArmorSelectionResult selectByVirtualPose(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double target_yaw,
    int num_armors,
    double target_v_yaw,
    double current_yaw,
    double current_pitch) const;

  /**
   * @brief 固定 ID 虚拟装甲板模式
   * @param armor_positions 真实装甲板位置
   * @param target_center 目标中心
   * @param num_armors 装甲板数量
   * @param current_yaw 当前云台 yaw
   * @param current_pitch 当前云台 pitch
   */
  ArmorSelectionResult selectByVirtualFixedId(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    int num_armors,
    int fixed_id,
    double current_yaw,
    double current_pitch) const;

  /**
   * @brief 真实可打板优先，否则生成虚拟姿态目标
   *
   * 该策略先使用与 FireAdviceEngine facing_only 等价的几何语义过滤真实板：
   * (中心->装甲板) 与 (中心->云台原点) 越同向，装甲板越正对我方。存在可打
   * 真实板时按最小云台运动量选板；全部不可打时退回 virtual_pose 稳定控云台。
   */
  ArmorSelectionResult selectByFacingOrVirtualPose(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double target_yaw,
    int num_armors,
    double target_v_yaw,
    double current_yaw,
    double current_pitch);

  /**
   * @brief 指定 ID 真实板可打优先，否则生成该 ID 的虚拟姿态目标
   */
  ArmorSelectionResult selectByFacingOrVirtualFixedId(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    int num_armors,
    int fixed_id,
    double current_yaw,
    double current_pitch);

  /**
   * @brief sp_vision_25 Aimer::choose_aim_point 风格选板
   *
   * 使用 (中心->云台原点) 与 (中心->装甲板) 的有符号水平夹角作为 delta_angle:
   * 低速普通目标在 shootable_angle 范围内选板并锁定双候选；高速/前哨站使用
   * coming/leaving 角和 yaw_velocity 方向选择转入视野的装甲板。
   */
  ArmorSelectionResult selectBySpVision25(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    int num_armors,
    double target_v_yaw,
    double current_yaw,
    double current_pitch);

  /**
   * @brief 选择最佳装甲板 (基于传统决策角)
   * @param armor_positions 各装甲板的世界坐标位置
   * @param target_center 目标中心位置
   * @param target_yaw 目标yaw角
   * @param target_v_yaw 目标yaw角速度
   * @return 选择结果索引
   */
  int selectByDecisionAngle(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double target_yaw,
    double target_v_yaw) const;

  /**
   * @brief 过滤掉距离最远的装甲板
   * @param armor_positions 装甲板位置列表
   * @return 过滤后的索引列表
   */
  std::vector<int> filterByDistance(
    const std::vector<Eigen::Vector3d> & armor_positions) const;

  /**
   * @brief 计算从当前云台位置到目标位置的yaw和pitch角
   * @param target_position 目标位置
   * @param current_yaw 当前云台yaw角 (用于参考)
   * @param[out] yaw 目标yaw角
   * @param[out] pitch 目标pitch角
   */
  static void calculateYawPitch(
    const Eigen::Vector3d & target_position,
    double current_yaw,
    double & yaw,
    double & pitch);

  /**
   * @brief 计算每个装甲板的 facing angle (法向量与视线夹角)
   * @param armor_positions 各装甲板位置
   * @param target_yaw 目标 yaw 角 (弧度)
   * @param num_armors 装甲板总数
   * @return 每个装甲板的 facing angle (弧度, 0=正面, π/2=侧面)
   */
  static std::vector<double> computeFacingAngles(
    const std::vector<Eigen::Vector3d> & armor_positions,
    double target_yaw,
    int num_armors);

  /**
   * @brief FireAdviceEngine facing_only 等价的正面朝向余弦
   *
   * 对比 (机器人中心->装甲板) 与 (机器人中心->云台原点) 的水平夹角。
   * 返回值越接近 1 表示装甲板越正对云台，越接近 -1 表示背向。
   */
  static double computeImpactFacingCos(
    const Eigen::Vector3d & target_center,
    const Eigen::Vector3d & armor_position);

  static std::vector<double> computeImpactFacingAngles(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center);

  /**
   * @brief 计算每个装甲板的径向夹角
   * @param armor_positions 各装甲板位置
   * @param target_center 目标机器人中心位置
   * @return 每个装甲板的径向夹角 (弧度, 0=最径向对齐)
   */
  static std::vector<double> computeRadialAngles(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center,
    double centerline_bias_rad = 0.0);

  static std::vector<double> computeSignedRadialAngles(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const Eigen::Vector3d & target_center);

  ArmorSelectionResult selectMinMovementFromIndices(
    const std::vector<Eigen::Vector3d> & armor_positions,
    const std::vector<int> & candidate_indices,
    const std::vector<double> * facing_angles,
    double current_yaw,
    double current_pitch) const;

private:
  double side_angle_{15.0};           // 侧向角度阈值 (度)
  double min_switching_v_yaw_{1.0};   // 最小切换角速度阈值

  // Facing hysteresis 参数
  double facing_enter_angle_{40.0};   // 进入阈值 (度)
  double facing_exit_angle_{55.0};    // 退出阈值 (度)

  // 径向选板动态阈值参数 (角度单位: 度, 角速度单位: rad/s)
  bool radial_dynamic_enable_{false};
  double radial_dynamic_v_yaw_ref_{8.0};
  double radial_dynamic_shrink_ratio_{0.6};
  double radial_dynamic_min_angle_deg_{5.0};
  double radial_dynamic_bias_gain_deg_{0.0};
  double radial_dynamic_max_bias_deg_{0.0};

  // 虚拟姿态自动启停参数
  bool virtual_auto_switch_enable_{false};
  double virtual_auto_switch_enter_vyaw_{8.0};
  double virtual_auto_switch_exit_vyaw_{6.0};
  int virtual_fixed_id_{0};
  SelectionMethod virtual_auto_switch_method_{SelectionMethod::VIRTUAL_POSE};
  int virtual_auto_switch_fixed_id_{0};

  // sp_vision_25 选板参数
  double sp_low_speed_vyaw_{2.0};
  double sp_shootable_angle_deg_{60.0};
  double sp_coming_angle_deg_{60.0};
  double sp_leaving_angle_deg_{20.0};
  double sp_outpost_coming_angle_deg_{70.0};
  double sp_outpost_leaving_angle_deg_{30.0};
  bool sp_hold_current_until_jump_{false};
  bool sp_zero_speed_fallback_{true};

  // 选板策略
  SelectionMethod selection_method_{SelectionMethod::MIN_MOVEMENT_WITH_FACING};

  // 记忆上次选择 (用于 hysteresis)
  mutable int last_selected_index_{-1};

  // sp_vision_25 风格锁定状态
  int sp_lock_id_{-1};
  int sp_initial_panel_id_{-1};
  int sp_last_front_panel_id_{-1};
  int sp_last_armor_count_{0};
  bool sp_has_jumped_{false};

  // 自动虚拟模式启停状态
  bool virtual_mode_active_{false};

  // 对 |target_v_yaw| 做一阶低通，降低噪声导致的抖动进出
  double filtered_abs_v_yaw_{0.0};
  bool abs_v_yaw_filter_initialized_{false};
  double abs_v_yaw_lpf_alpha_{0.2};
};

}  // namespace gimbal_controller

#endif  // GIMBAL_CONTROLLER__ARMOR_SELECTOR_HPP_
