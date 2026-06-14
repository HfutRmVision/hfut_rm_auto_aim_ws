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

#ifndef GIMBAL_CONTROLLER__GIMBAL_CONTROL_ORCHESTRATOR_HPP_
#define GIMBAL_CONTROLLER__GIMBAL_CONTROL_ORCHESTRATOR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "gimbal_controller/fire_advice_engine.hpp"
#include "gimbal_controller/gimbal_control_strategy.hpp"

namespace gimbal_controller
{

class FireAdvisor;

struct FireDecisionConfig
{
  double prediction_delay_s{0.0};
  double control_latency_s{0.0};
  double trigger_to_muzzle_s{0.0};
  double max_processing_delay_s{0.5};
  double yaw_offset_rad{0.0};
  double pitch_offset_rad{0.0};
  bool include_processing_delay{true};
  bool include_control_latency_in_target_prediction{false};
};

struct FireAdviceDebugSnapshot
{
  bool evaluated{false};
  bool valid{false};
  bool fire_advice{false};
  int32_t best_candidate_index{-1};
  double yaw_error{0.0};
  double pitch_error{0.0};
  bool best_candidate_facing_ok{false};
  int32_t candidate_count_total{0};
  int32_t candidate_count_facing_eligible{0};
  int32_t candidate_count_facing_rejected{0};
  bool probability_enabled{false};
  double p_hit_window{0.0};
  double fire_score{0.0};
  double burst_probability{0.0};
  double log_evidence{0.0};
  double evidence_sum{0.0};
  double evidence_strength{0.0};
  int gate_strategy{0};
  int gate_state{0};
  double best_tau_ms{0.0};
  double e_u{0.0};
  double e_v{0.0};
  double sigma_u{0.0};
  double sigma_v{0.0};
  double armor_width_m{0.135};
  double armor_height_m{0.125};
  std::vector<fire_advice::TauDebugSample> tau_samples;
  Eigen::Vector3d armor_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d armor_right = Eigen::Vector3d::UnitY();
  Eigen::Vector3d armor_up = Eigen::Vector3d::UnitZ();
  int8_t mode{rm_interfaces::msg::GimbalCmd::MODE_UNKNOWN};
  uint8_t track_state{rm_interfaces::msg::TrackedRobot::DETECTING};
  std::string target_id;
};

class GimbalControlOrchestrator
{
public:
  void setFireModules(
    std::shared_ptr<FireAdviceEngine> fire_advice_engine,
    std::shared_ptr<FireAdvisor> fire_advisor);

  void setFireDecisionConfig(const FireDecisionConfig & config)
  {
    fire_cfg_ = config;
  }

  rm_interfaces::msg::GimbalCmd buildIdleCmd(const GimbalControlContext & context) const;

  rm_interfaces::msg::GimbalCmd finalize(
    const GimbalControlContext & context,
    const rm_interfaces::msg::GimbalCmd & control_cmd) const;

  const FireAdviceDebugSnapshot & lastFireAdviceDebug() const
  {
    return last_fire_debug_;
  }

private:
  int8_t decideMode(const GimbalControlContext & context) const;

  bool evaluateFireAdvice(
    const GimbalControlContext & context,
    const rm_interfaces::msg::GimbalCmd & cmd,
    double distance) const;

  std::shared_ptr<FireAdviceEngine> fire_advice_engine_;
  std::shared_ptr<FireAdvisor> fire_advisor_;
  FireDecisionConfig fire_cfg_;
  mutable FireAdviceDebugSnapshot last_fire_debug_;
};

}  // namespace gimbal_controller

#endif  // GIMBAL_CONTROLLER__GIMBAL_CONTROL_ORCHESTRATOR_HPP_
