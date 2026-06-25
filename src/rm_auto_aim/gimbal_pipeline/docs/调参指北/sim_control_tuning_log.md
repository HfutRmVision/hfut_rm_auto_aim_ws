# Sim Control Tuning Log

本文件记录 Webots sim 下 `gimbal_pipeline` 控制链路调参过程。当前约束：只修改
`src/rm_bringup/config/sim/*.yaml`；除非 config 已到明显瓶颈，否则不改控制代码。

## 固定条件

- 记录开始时间：2026-06-25 15:16 CST
- 配置文件：`src/rm_bringup/config/sim/gimbal_pipeline.yaml`
- 控制频率：`controller.control_rate = 250 Hz`
- 输出步进单位：`deg/control_tick`
- 相机机器人 yaw/pitch 物理极速：`20 rad/s`
- 250 Hz 下 `20 rad/s ~= 4.58 deg/control_tick`
- 云台理想加速度上限按实车条件保留，不继续增大
- outpost 三板逻辑保持特殊处理，不并入普通单板/多板链路

## 采样工具

- 控制抖动：`python3 /root/code_for_test/sample_control_jitter.py <seconds> --csv <path>`
- 目标转速：`python3 /root/code_for_test/sample_target_vyaw.py <seconds> --csv <path>`
- 真值误差：`python3 /root/code_for_test/sample_truth_aim_error.py <seconds> --csv <path>`

## 基线观测

采样窗口：150 s，文件：`/tmp/control_tune_baseline.csv`、
`/tmp/target_vyaw_tune_baseline.csv`。

关键结果：

- `valid_matched = 12032`
- `fire_ratio = 0.478`
- `target_abs_vyaw_rad_s`: mean `3.03`, p90 `3.68`, max `6.68`
- `yaw_pos_abs_err_rad_valid_mode`: mean `0.0303`, p95 `0.0769`, p99 `0.1032`, max `0.3367`
- `pitch_pos_abs_err_rad_valid_mode`: mean `0.0047`, p95 `0.0130`, p99 `0.0302`, max `0.1025`
- `yaw_cmd_abs_vel_rad_s_valid_mode`: p99/max `10.91`
- `pitch_cmd_abs_vel_rad_s_valid_mode`: p99 `1.88`, max `5.24`
- 实际 yaw/pitch 加速度多次顶到 `120 rad/s^2`

判断：

- 尖峰主要不是单帧命令断裂，而是多帧连续给出较大速度命令后，云台加速度饱和并冲过目标。
- 旧输出限速 `max_yaw_rate = 2.5 deg/tick` 对应 `10.91 rad/s`，低于当前物理极速 `20 rad/s`。
- 旧 `sp_vision.low_speed_vyaw = 6.5` 会让当前 3 rad/s 左右目标大多走低速锁板分支，与参考 `sp_vision_25` 的小陀螺分支不一致。

## 参数记录

| 时间 | 修改 | 原因 | 待验证 |
| --- | --- | --- | --- |
| 2026-06-25 15:16 | `prediction_delay/prediction_extra_s/mpc.prediction_delay_s: 0.018 -> 0.014` | 降低前瞻，避免目标点领先过多导致加速度饱和后过冲 | p99/max yaw 误差是否下降，fire_ratio 是否维持 |
| 2026-06-25 15:16 | `sp_vision.low_speed_vyaw: 6.5 -> 2.0` | 按参考实现，当前约 3 rad/s 目标应更多进入来板/离板逻辑 | 切板窗口尖峰是否减少 |
| 2026-06-25 15:16 | `sp_vision.shootable_angle_deg: 55 -> 60`, `coming_angle_deg: 75 -> 60`, `leaving_angle_deg: 35 -> 20` | 普通四板参数回到参考 `sp_vision_25` 附近；outpost 70/30 不变 | 高速/移动靶下切板是否更早且更稳定 |
| 2026-06-25 15:16 | `output_filter.max_yaw_rate: 2.5 -> 4.0`, `max_pitch_rate: 0.9 -> 3.0` | 当前物理极速 20 rad/s 对应 4.58 deg/tick；先留余量不直接顶满 | DPS 是否提高，是否重新出现甩飞 |
| 2026-06-25 15:24 | `sp_vision.low_speed_vyaw: 2.0 -> 6.5`, `max_yaw_rate: 4.0 -> 3.0`, `max_pitch_rate: 3.0 -> 1.2` | 4.0/3.0 让有效跟踪时 yaw 约 35% 帧、pitch 约 11% 帧贴近加速度上限；`low_speed_vyaw=2.0` 后 `mode=-1` 比例升高，开火比例下降 | 验证是否恢复开火比例并压低 yaw/pitch 尖峰 |

## 2026-06-25 15:24 采样结果

配置：`prediction_extra_s = 0.014`，`low_speed_vyaw = 2.0`，
`max_yaw_rate = 4.0 deg/tick`，`max_pitch_rate = 3.0 deg/tick`。

采样文件：

- `/tmp/control_tune_after_rate_4_3.csv`
- `/tmp/target_vyaw_after_rate_4_3.csv`

关键结果：

- `fire_ratio = 0.290`，低于基线 `0.478`
- `valid_matched = 10964`，`invalid_matched = 5044`，`mode=-1` 比例约 `31.5%`
- `target_abs_vyaw_rad_s`: mean `2.96`, p90 `3.86`, max `7.76`
- `yaw_pos_abs_err_rad_valid_mode`: mean `0.0318`, p95 `0.0808`, p99 `0.1147`, max `0.4551`
- `pitch_pos_abs_err_rad_valid_mode`: mean `0.0082`, p95 `0.0303`, p99 `0.0757`, max `0.1894`
- `yaw_cmd_abs_vel_rad_s_valid_mode`: p99 `13.96`, max `17.45`
- `pitch_cmd_abs_vel_rad_s_valid_mode`: p99 `3.53`, max `13.09`
- 有效跟踪中 `actual_yaw_acc >= 119 rad/s^2` 比例约 `35%`
- 有效跟踪中 `actual_pitch_acc >= 119 rad/s^2` 比例约 `11%`

判断：

- 4.0/3.0 的输出步进虽然接近物理极速，但对当前加速度上限过激；尖峰窗口多次出现连续
  `cmd_yaw_v = 17.45 rad/s` 或 `cmd_pitch_v = 13.09 rad/s`。
- `low_speed_vyaw = 2.0` 让当前约 3 rad/s 目标大量进入来板/离板分支，开火窗口减少。
- 下一轮先恢复普通目标低速锁板分支，同时把输出限幅回收到
  `max_yaw_rate = 3.0 deg/tick`、`max_pitch_rate = 1.2 deg/tick`。

## 甩飞归因

当前离线数据来自 `/tmp/control_tune_after_rate_4_3.csv`。该组配置下出现的甩飞主要归因到
控制输出超过云台加速度能力；识别不是主因，预测/选板是触发因素之一。

证据：

- 大 yaw 误差帧 `abs(yaw_err) > 0.20 rad` 中，`cmd_yaw_v` 中位数已经达到
  `17.45 rad/s`，也就是 `max_yaw_rate = 4.0 deg/tick` 的输出上限。
- 大 pitch 误差帧 `abs(pitch_err) > 0.10 rad` 中，`cmd_pitch_v` 中位数约
  `10.33 rad/s`，p90 达到 `13.09 rad/s`，接近 `max_pitch_rate = 3.0 deg/tick` 的输出上限。
- 这些大误差帧中，实际 yaw/pitch 加速度中位数基本贴近 `120 rad/s^2` 上限。
- 以最大 yaw 峰值为例，误差峰值前 120ms 内常见 `cap_y=8~16` 帧、`sat_y=13~16`
  帧，即控制命令连续顶速，同时云台连续加速度饱和。
- `cmd_age` 在大误差帧中通常为 `0.03~0.07s`，没有表现为识别消息长时间陈旧导致的追踪错误。
- `mode=-1` 在大误差峰中占比较高，说明切板/临时丢失会触发参考点变化；但真正把视角甩远的是
  后续控制命令连续顶速和加速度饱和。

结论：

- 主问题：控制侧。`output_filter.max_yaw_rate/max_pitch_rate` 按速度能力放得过大，但没有同时限制
  加速度/jerk，导致 MPC/输出滤波给出当前云台加速度无法跟上的命令。
- 次问题：预测/选板侧。`low_speed_vyaw = 2.0` 让当前约 3 rad/s 目标更多进入来板/离板分支，
  `mode=-1` 比例从基线约 `24%` 升到约 `31.5%`，开火比例下降，并增加参考点切换触发。
- 识别侧：现有采样没有看到识别是主因。若要确认识别，需要在程序运行时同步采
  `/detector` 输出、`tracked_robot` 和 Webots 真值；目前控制数据已经能解释甩飞。

## 严重超前归因

用户说明这里的“甩飞”指瞄准点严重超前目标，而不是视角脱离目标。为此新增采样脚本：
`/root/code_for_test/sample_truth_lead_breakdown.py`，按 Webots 四块真值装甲板的视线角速度
统计云台实际指向/命令指向相对目标运动方向的 signed lead。

当前运行参数：

- `controller.delay.prediction_extra_s = 0.014`
- `controller.output_filter.max_yaw_rate = 3.0 deg/tick`
- `controller.output_filter.max_pitch_rate = 1.2 deg/tick`
- `controller.solver.sp_vision.low_speed_vyaw = 6.5`

采样结果：

- `sample_truth_aim_error.py 120`
  - `fire_cmd_count = 15394`
  - `shot_samples = 1465`
  - `center_angle_err_rad`: mean `0.0358`, p95 `0.0978`, p99 `0.1590`, max `0.2010`
  - `signed_yaw_err_rad`: mean `0.0041`, p95 `0.0775`, p99 `0.1507`
- `sample_control_jitter.py 120`
  - `fire_ratio = 0.517`
  - `yaw_pos_abs_err_rad_valid_mode`: mean `0.0280`, p95 `0.0736`, p99 `0.0891`, max `0.1874`
  - `pitch_pos_abs_err_rad_valid_mode`: mean `0.0042`, p95 `0.0111`, p99 `0.0237`, max `0.0861`
  - 控制跟随误差已明显低于 4.0/3.0 激进配置，不再是主要解释。
- `sample_truth_lead_breakdown.py 120`
  - 全部样本：`actual_lead_gt_0p08 = 15.9%`
  - 开火样本：`actual_lead_gt_0p08 = 5.9%`
  - 正常测量样本：`actual_lead_gt_0p08 = 10.4%`
  - top lead 段主要是 `fire=0`，大量集中在 `mode=-1` / `track_state=TEMP_LOST`
  - 开火样本 p95 lead time 约 `0.136s`，p99 约 `0.227s`
  - 正常测量样本 p95 lead time 约 `0.178s`，p99 约 `0.380s`

代码定位：

- `MpcReferenceGenerator::generateWithDelay()` 中每个 MPC 参考点使用
  `t_predict = base_delay + (k + 1) * dt`，之后还会迭代加入子弹飞行时间。
- 当前 `base_delay ~= prediction_extra_s = 0.014s`，`controller.mpc.dt = 0.05s`，
  距离约 2.2m、弹速 22.5m/s 时飞行时间约 0.10s。
- 因此第一步参考点已经约为 `0.014 + 0.05 + 0.10 = 0.16s` 后的装甲板，和实测开火样本
  p95 lead time `0.136s` 同量级。

结论：

- 这类“严重超前”的主因是预测/选板链路，不是识别。
- 控制侧当前主要是在跟随 MPC 给出的未来参考点；回收限幅后，控制跟随误差已不是主导问题。
- 识别侧暂未看到错目标证据；严重超前段里 tracked yaw velocity 与 target yaw velocity 一致，
  但控制目标被推进到未来可打板/未来命中点。
- `TEMP_LOST` 时只靠预测继续外推，容易把超前放大；该段通常 `fire=0`，说明火控已经认为不可打，
  但 MPC 仍在引导云台提前追未来板。

## 下一轮验证

用户重启 launch 后采样当前配置：

```bash
python3 /root/code_for_test/sample_control_jitter.py 150 --csv /tmp/control_tune_after_rate_3_1p2_lowspeed6p5.csv
python3 /root/code_for_test/sample_target_vyaw.py 150 --csv /tmp/target_vyaw_after_rate_3_1p2_lowspeed6p5.csv
```

重点看：

- `yaw_pos_abs_err_rad_valid_mode` 的 p95/p99/max
- `pitch_pos_abs_err_rad_valid_mode` 的 p95/p99/max
- `fire_ratio` 是否从 `0.290` 恢复到接近基线
- `yaw_cmd_abs_vel_rad_s_valid_mode` max 是否约 `13.09 rad/s`
- `pitch_cmd_abs_vel_rad_s_valid_mode` max 是否约 `5.24 rad/s`
- `actual_yaw_acc` / `actual_pitch_acc` 顶满窗口是否缩短
- `top_segments yaw_err/pitch_err` 是否仍集中在切板或 `mode=-1` 附近
