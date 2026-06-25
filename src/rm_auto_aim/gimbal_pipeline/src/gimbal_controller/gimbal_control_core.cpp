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

#include "gimbal_controller/gimbal_control_core.hpp"

#include <utility>

#include "gimbal_controller/strategies/mpc_control_strategy.hpp"

namespace gimbal_controller
{

namespace
{

rm_interfaces::msg::GimbalCmd makeFallbackIdleCmd(const GimbalControlContext & context)
{
  rm_interfaces::msg::GimbalCmd cmd;
  cmd.header = context.target_robot.header;
  cmd.target_id = context.target_robot.robot_id;
  cmd.yaw_diff = 0.0;
  cmd.pitch_diff = 0.0;
  cmd.distance = 0.0;
  cmd.fire_advice = false;
  cmd.mode = rm_interfaces::msg::GimbalCmd::MODE_NO_VALID_MEASUREMENT;
  return cmd;
}

}  // namespace

GimbalControlStrategy::SharedPtr GimbalControlCore::findStrategy(const std::string & name) const
{
  if (!strategies_) {
    return nullptr;
  }
  auto it = strategies_->find(name);
  if (it == strategies_->end()) {
    return nullptr;
  }
  return it->second;
}

GimbalControlCoreOutput GimbalControlCore::compute(
  const GimbalControlContext & context,
  const std::string & strategy_name,
  const std::string & selected_target_id,
  bool enable)
{
  GimbalControlCoreOutput output;
  output.has_tracking = context.is_tracking;

  if (!enable) {
    output.cmd = orchestrator_.buildIdleCmd(context);
    if (output.cmd.mode == rm_interfaces::msg::GimbalCmd::MODE_UNKNOWN) {
      output.cmd = makeFallbackIdleCmd(context);
    }
    output.fire_advice_debug = orchestrator_.lastFireAdviceDebug();
    output.delay_audit = DelayAuditSnapshot{};
    output.delay_audit.strategy_name = strategy_name;
    output.delay_audit.tracking = false;

    filter_.reset();
    prev_tracking_target_id_.clear();
    return output;
  }

  auto strategy = findStrategy(strategy_name);
  if (!strategy) {
    output.strategy_found = false;
    output.cmd = orchestrator_.buildIdleCmd(context);
    if (output.cmd.mode == rm_interfaces::msg::GimbalCmd::MODE_UNKNOWN) {
      output.cmd = makeFallbackIdleCmd(context);
    }
    output.fire_advice_debug = orchestrator_.lastFireAdviceDebug();

    output.delay_audit = DelayAuditSnapshot{};
    output.delay_audit.strategy_name = strategy_name;
    output.delay_audit.tracking = context.is_tracking;
    return output;
  }

  auto cmd = strategy->solve(context);
  cmd = orchestrator_.finalize(context, cmd);
  output.fire_advice_debug = orchestrator_.lastFireAdviceDebug();

  // Like the reference tracker pipeline, TEMP_LOST is still a predictive
  // continuation of the same target. Keep output-filter history across this
  // state; reset only after the target is truly unavailable or changed.
  const bool target_continues = context.is_tracking || context.is_temp_lost;
  const std::string current_target = target_continues ? selected_target_id : std::string();
  if (current_target != prev_tracking_target_id_) {
    filter_.reset();
  }
  filter_.filter(cmd);
  prev_tracking_target_id_ = current_target;

  output.cmd = std::move(cmd);
  output.delay_audit = strategy->getLastDelayAudit();
  return output;
}

void GimbalControlCore::updateFov(double fov_half_yaw, double fov_half_pitch)
{
  auto mpc_strategy = std::dynamic_pointer_cast<MpcControlStrategy>(findStrategy("mpc"));
  if (!mpc_strategy) {
    return;
  }
  mpc_strategy->updateFov(fov_half_yaw, fov_half_pitch);
}

}  // namespace gimbal_controller
