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

#ifndef GIMBAL_CONTROLLER__STRATEGIES__MPC_CONTROL_STRATEGY_HPP_
#define GIMBAL_CONTROLLER__STRATEGIES__MPC_CONTROL_STRATEGY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>

#include <Eigen/Dense>

#include "gimbal_controller/delay_management/delay_semantic_manager.hpp"
#include "gimbal_controller/gimbal_control_strategy.hpp"
#include "gimbal_controller/mpc/gimbal_dynamics_model.hpp"
#include "gimbal_controller/mpc/qp_solver.hpp"
#include "gimbal_controller/mpc/mpc_reference_generator.hpp"

namespace gimbal_controller
{

/**
 * @brief MPC 控制策略
 *
 * 基于模型预测控制的云台控制策略:
 *   1. 差分估计云台角速度 (无下位机反馈)
 *   2. 调用 MpcReferenceGenerator 生成 N 步参考轨迹
 *   3. 构建 QP (含延迟补偿) 并用 qpOASES 求解
 *   4. 提取首步控制量推算目标 yaw/pitch
 *   5. 调用 FireAdvisor 判断开火
 */
class MpcControlStrategy : public GimbalControlStrategy
{
public:
  MpcControlStrategy();
  ~MpcControlStrategy() override = default;

  rm_interfaces::msg::GimbalCmd solve(const GimbalControlContext & context) override;

  std::string getName() const override { return "MpcControlStrategy"; }

  /**
   * @brief 设置 MPC 参数
   * @param N 预测步数
   * @param dt 时间步长 (秒)
   * @param control_delay_s 控制延迟 (秒)
   * @param max_accel 最大允许角加速度 (rad/s²)
   * @param q_yaw / q_pitch 位置跟踪权重
   * @param q_yaw_vel / q_pitch_vel 速度跟踪权重
   * @param r_yaw / r_pitch 控制量权重
   * @param s_yaw / s_pitch 控制平滑权重
   */
  void setMpcParameters(
    int N, double dt, double control_delay_s, double max_accel,
    double q_yaw, double q_pitch, double q_yaw_vel, double q_pitch_vel,
    double r_yaw, double r_pitch, double s_yaw, double s_pitch);

  /**
   * @brief 设置延时补偿参数
   * @param enable 是否启用延时补偿版本的参考轨迹生成
   * @param prediction_delay_s 额外预测延迟 (秒)
   * @param trigger_to_muzzle_s 从触发开火到子弹出膛的延迟 (秒)
   * @param allow_muzzle_compensation 是否将 trigger_to_muzzle_s 纳入 MPC 参考前瞻
   * @param flight_time_iters 飞行时间迭代次数
   * @param max_processing_delay_s 最大允许的 processing_delay 上限 (秒)，超出则被截断
   */
  void setDelayCompensation(
    bool enable, double prediction_delay_s, double trigger_to_muzzle_s,
    bool allow_muzzle_compensation, int flight_time_iters, double max_processing_delay_s);

  /**
   * @brief 设置 yaw 速度前馈参数
   *
   * 前馈以“等效额外前瞻时间”注入参考轨迹生成, 单位为秒。
   * k=0 时与原实现一致。
   *
   * @param yaw_feedforward_k_s 额外前瞻时间 (秒)
   */
  void setYawFeedforward(double yaw_feedforward_k_s);

  /**
   * @brief 设置手动角度补偿
   *
   * 参数单位与 predicted/current/state_machine 策略一致: 度。
   */
  void setManualOffset(double pitch_offset_deg, double yaw_offset_deg);

  /**
   * @brief 设置 MPC 机动自适应权重衰减参数
   *
   * 启用后，每帧通过对 UKF center_velocity 做时间戳感知差分计算机动因子 alpha，
   * 用 alpha 衰减远期 Q 权重并放大 R 正则项，在机动时减少控制量。
   * 禁用时 (enable=false) 与原实现完全一致。
   *
   * @param enable   是否启用
   * @param a_max    差分速度归一化上限 (m/s²)
   * @param eta      alpha EMA 平滑系数 (0.1~0.3)
   * @param tau      Q 衰减时间常数 (步数尺度)
   * @param r_scale  R 放大系数
   */
  void setManeuverAdaptParameters(
    bool enable, double a_max, double eta, double tau, double r_scale);

  /**
   * @brief 设置 MPC 基于命中概率的 Q 权重加权参数
   *
   * 仅影响 Q (状态跟踪)，不修改 R/S；禁用时行为与原实现一致。
   */
  void setWeightingParameters(
    bool enable, double alpha, double k_omega,
    double sigma_min, double sigma_max, double sigma_sys,
    double target_size, double delay_s, double max_w,
    double smooth_alpha, double min_distance, double bullet_speed,
    double sigma_beta, double gamma);

  /**
   * @brief 设置数值归一化参数
   *
   * - mode=rms: 对 Q/R/S 对应量做滑动窗口 RMS 归一化
   * - mode=typical: 用典型值做静态归一化
   *
   * 统一公式: w_norm = w / (scale^2 + eps)
   */
  void setNumericalNormalizationParameters(
    bool enable, int window_size, int min_samples, double rms_epsilon,
    const std::string & mode,
    const Eigen::Vector4d & state_typical,
    const Eigen::Vector2d & control_typical,
    const Eigen::Vector2d & delta_control_typical);

  /**
   * @brief 设置 Hessian 自适应对角正则参数
   *
   * H <- H + eps * I, eps = clamp(max(abs, rel*mean(diag(H))), abs, max)
   */
  void setHessianRegularizationParameters(
    bool enable, double epsilon_abs, double epsilon_rel, double epsilon_max,
    bool retry_on_fail, double retry_scale);

  /**
   * @brief 设置数值诊断参数
   *
   * 低成本指标可常开；高成本谱指标按采样周期计算。
   */
  void setDiagnosticsParameters(
    bool enable,
    bool low_cost_always,
    bool high_cost_enable,
    int high_cost_sample_every,
    int log_every,
    bool log_on_failure,
    double active_tol,
    double rank_tol_rel);

  /**
   * @brief 设置 FOV 软约束参数
   *
   * 启用后，通过 slack 变量在 QP 中惩罚预测轨迹超出相机视场角范围的行为。
   * FOV 半角通过 updateFov() 从 camera_info 动态获取，也可由 fallback 值提供。
   * 禁用时 (enable=false) 与原实现完全一致。
   *
   * @param enable           是否启用 FOV 约束
   * @param margin           静态安全裕度 (rad)
   * @param slack_weight     slack 惩罚权重
   * @param constraint_steps 约束步数 (0 表示约束全部 N 步)
   * @param dynamic_margin_enable  是否启用动态 margin
   * @param margin_vel_scale       速度→margin 缩放系数
   * @param fallback_fov_yaw       camera_info 未收到时的 fallback yaw FOV 半角 (rad)
   * @param fallback_fov_pitch     camera_info 未收到时的 fallback pitch FOV 半角 (rad)
   */
  void setFovConstraintParameters(
    bool enable, double margin, double slack_weight, int constraint_steps,
    bool dynamic_margin_enable, double margin_vel_scale,
    double fallback_fov_yaw, double fallback_fov_pitch);

  /**
   * @brief 从 camera_info 回调更新 FOV 半角
   */
  void updateFov(double fov_half_yaw, double fov_half_pitch);

  /**
   * @brief 在 setComponents() 之后调用, 将组件注入到 MpcReferenceGenerator
   */
  void initReferenceGenerator();

  /**
   * @brief 设置轨迹生成前的速度 clamp 参数
   */
  void setVelocityClamp(const mpc::VelocityClampConfig & cfg)
  {
    ref_generator_.setVelocityClamp(cfg);
  }

private:
  struct SlidingRms
  {
    int window_size{80};
    std::deque<double> values;
    double sum_sq{0.0};

    void setWindowSize(int size)
    {
      window_size = std::max(1, size);
      while (static_cast<int>(values.size()) > window_size) {
        const double old = values.front();
        values.pop_front();
        sum_sq -= old * old;
      }
      if (sum_sq < 0.0) {
        sum_sq = 0.0;
      }
    }

    void reset()
    {
      values.clear();
      sum_sq = 0.0;
    }

    void addSample(double value)
    {
      values.push_back(value);
      sum_sq += value * value;
      while (static_cast<int>(values.size()) > window_size) {
        const double old = values.front();
        values.pop_front();
        sum_sq -= old * old;
      }
      if (sum_sq < 0.0) {
        sum_sq = 0.0;
      }
    }

    int size() const { return static_cast<int>(values.size()); }

    double rms(double epsilon, double fallback = 1.0) const
    {
      if (values.empty()) {
        return fallback;
      }
      const double mean_sq = sum_sq / static_cast<double>(values.size());
      return std::sqrt(std::max(mean_sq, 0.0) + std::max(epsilon, 1e-12));
    }
  };

  struct DiagnosticsSnapshot
  {
    uint64_t cycle{0};
    bool maneuver_path{false};
    bool qp_success{false};
    int qp_iterations{0};
    int active_bound_size{0};
    int active_linear_size{0};
    int active_set_size{0};
    double qp_cost{0.0};
    double trace_q{0.0};
    double trace_r{0.0};
    double trace_s{0.0};
    double trace_q_over_r{0.0};
    double trace_s_over_r{0.0};
    double bu_over_u{0.0};
    double regularization_eps{0.0};
    bool high_cost_valid{false};
    double cond_h{0.0};
    double lambda_min_h{0.0};
    int rank_h{0};
  };

  enum class NormalizationMode
  {
    RMS,
    TYPICAL
  };

  Eigen::VectorXd buildWeightingVector(
    const GimbalControlContext & context,
    const Eigen::VectorXd & X_ref);

  void configureRmsWindows();
  void resetNumericalStates();
  Eigen::Vector4d updateAndGetStateRms(const Eigen::VectorXd & free_error);
  Eigen::Vector2d getControlRms() const;
  Eigen::Vector2d getDeltaControlRms() const;
  void updateControlHistory(const mpc::GimbalDynamicsModel::ControlVector & u_opt);
  double computeRegularizationEpsilon(const Eigen::MatrixXd & H) const;
  void applyHessianRegularization(Eigen::MatrixXd & H, double epsilon) const;
  void fillAndLogDiagnostics(
    bool maneuver_path,
    const Eigen::MatrixXd & H,
    const Eigen::MatrixXd & Q_eff,
    const Eigen::MatrixXd & R_eff,
    const Eigen::MatrixXd & S_eff,
    const Eigen::VectorXd & lb,
    const Eigen::VectorXd & ub,
    const mpc::QPResult & result,
    double applied_regularization);

  // MPC 核心模块
  mpc::GimbalDynamicsModel dynamics_model_;
  mpc::QPSolver qp_solver_;
  mpc::MpcReferenceGenerator ref_generator_;

  // 缓存的 QP 结构 (仅在参数变更时重建)
  Eigen::MatrixXd A_pred_;
  Eigen::MatrixXd B_ctrl_;    // 含延迟的 B_d (或无延迟时为 B_pred)
  Eigen::MatrixXd D_;
  Eigen::MatrixXd Q_blk_;
  Eigen::MatrixXd R_blk_;
  Eigen::MatrixXd S_blk_;
  bool matrices_dirty_{true};

  void rebuildMatrices();

  // MPC 参数
  int N_{20};
  double dt_{0.01};
  double control_delay_s_{0.0};
  double max_accel_{30.0};

  // 权重参数
  double q_yaw_{100.0};
  double q_pitch_{100.0};
  double q_yaw_vel_{10.0};
  double q_pitch_vel_{10.0};
  double r_yaw_{0.01};
  double r_pitch_{0.01};
  double s_yaw_{5.0};
  double s_pitch_{5.0};

  // 角速度差分估计状态
  double prev_yaw_{0.0};
  double prev_pitch_{0.0};
  rclcpp::Time prev_state_time_{0, 0, RCL_ROS_TIME};
  bool has_prev_state_{false};

  // 延时补偿参数
  bool enable_delay_compensation_{false};
  double prediction_delay_s_{0.0};
  double trigger_to_muzzle_s_{0.0};
  bool allow_muzzle_compensation_{true};
  int flight_time_iters_{2};
  double max_processing_delay_s_{0.5};  // processing_delay 上限 (秒)
  double yaw_feedforward_k_s_{0.0};     // yaw 速度前馈等效前瞻时间 (秒)
  double max_yaw_feedforward_s_{0.12};  // yaw 前馈上限 (秒)
  delay_management::DelaySemanticManager delay_manager_;
  bool uses_delayed_b_model_{false};
  bool warned_double_compensation_{false};

  // 上一步求解结果 (warmstart)
  Eigen::VectorXd U_prev_;
  // 机动自适应权重衰减参数
  bool enable_maneuver_adapt_{false};
  double a_max_{3.0};             // 差分速度归一化上限 (m/s²)
  double eta_{0.2};               // EMA 平滑系数
  double tau_{10.0};              // Q 衰减时间常数 (步数)
  double r_scale_maneuver_{10.0}; // R 放大系数

  // 命中概率权重参数
  bool enable_weighting_{false};
  double weighting_alpha_{0.0};
  double weighting_k_omega_{0.0};
  double weighting_sigma_min_{0.05};
  double weighting_sigma_max_{0.5};
  double weighting_sigma_sys_{0.02};
  double weighting_target_size_{0.135};
  double weighting_delay_s_{0.0};
  double weighting_max_w_{5.0};
  double weighting_smooth_alpha_{0.0};
  double weighting_min_distance_{0.1};
  double weighting_bullet_speed_{20.0};
  double weighting_sigma_beta_{0.0};
  double weighting_gamma_{1.0};
  Eigen::VectorXd prev_w_steps_;
  bool has_prev_w_steps_{false};

  // 数值归一化参数与状态
  bool enable_normalization_{false};
  NormalizationMode normalization_mode_{NormalizationMode::RMS};
  int rms_window_size_{80};
  int rms_min_samples_{10};
  double rms_epsilon_{1e-6};
  Eigen::Vector4d state_typical_{Eigen::Vector4d::Ones()};
  Eigen::Vector2d control_typical_{Eigen::Vector2d::Ones()};
  Eigen::Vector2d delta_control_typical_{Eigen::Vector2d::Ones()};
  std::array<SlidingRms, mpc::GimbalDynamicsModel::STATE_DIM> state_rms_trackers_;
  std::array<SlidingRms, mpc::GimbalDynamicsModel::CONTROL_DIM> control_rms_trackers_;
  std::array<SlidingRms, mpc::GimbalDynamicsModel::CONTROL_DIM> delta_control_rms_trackers_;
  mpc::GimbalDynamicsModel::ControlVector prev_applied_u_{
    mpc::GimbalDynamicsModel::ControlVector::Zero()};
  bool has_prev_applied_u_{false};

  // Hessian 自适应对角正则参数
  bool enable_hessian_regularization_{false};
  double hessian_reg_eps_abs_{1e-8};
  double hessian_reg_eps_rel_{1e-6};
  double hessian_reg_eps_max_{1e-2};
  bool hessian_reg_retry_on_fail_{true};
  double hessian_reg_retry_scale_{10.0};

  // 分层诊断参数与缓存
  bool enable_diagnostics_{false};
  bool diagnostics_low_cost_always_{true};
  bool diagnostics_high_cost_enable_{false};
  int diagnostics_high_cost_sample_every_{20};
  int diagnostics_log_every_{50};
  bool diagnostics_log_on_failure_{true};
  double diagnostics_active_tol_{1e-4};
  double diagnostics_rank_tol_rel_{1e-9};
  uint64_t diagnostics_cycle_{0};
  DiagnosticsSnapshot last_diagnostics_;

  // 机动 alpha EMA 状态
  double alpha_ema_{0.0};

  // 目标速度差分历史（时间戳感知）
  Eigen::Vector3d prev_target_velocity_{Eigen::Vector3d::Zero()};
  rclcpp::Time prev_target_stamp_{0, 0, RCL_ROS_TIME};
  bool has_prev_velocity_{false};
  // FOV 软约束参数
  bool enable_fov_constraint_{false};
  double fov_half_yaw_{0.35};         // 视场半角 yaw (rad) — 从 camera_info 或 fallback 获取
  double fov_half_pitch_{0.26};       // 视场半角 pitch (rad)
  double fov_margin_{0.05};           // 静态安全裕度 (rad)
  double fov_slack_weight_{1000.0};   // slack 惩罚权重
  int fov_constraint_steps_{0};       // 约束步数 (0 = 全部 N 步)
  bool enable_dynamic_margin_{false}; // 动态 margin
  double margin_vel_scale_{0.01};     // 速度→margin 缩放
  double fallback_fov_yaw_{0.35};     // camera_info 未收到时的 fallback
  double fallback_fov_pitch_{0.26};
  bool camera_info_received_{false};  // 是否已收到 camera_info

  // FOV 约束 QP 扩展专用求解器（维度不同于原始 QP，需独立实例）
  mpc::QPSolver qp_solver_fov_;

  /**
   * @brief 构建 FOV 软约束扩展 QP 并求解, 返回原始控制变量维度的结果
   */
  mpc::QPResult solveFovConstrainedQP(
    const Eigen::MatrixXd & H,
    const Eigen::VectorXd & f,
    const Eigen::VectorXd & lb,
    const Eigen::VectorXd & ub,
    const mpc::GimbalDynamicsModel::StateVector & x0,
    const Eigen::VectorXd & X_ref,
    const GimbalControlContext & context);

  /**
   * @brief QP 求解失败时回退到直接瞄准参考轨迹首步
   */
  rm_interfaces::msg::GimbalCmd fallbackDirectAim(
    const GimbalControlContext & context,
    const Eigen::VectorXd & X_ref);
};

}  // namespace gimbal_controller

#endif  // GIMBAL_CONTROLLER__STRATEGIES__MPC_CONTROL_STRATEGY_HPP_
