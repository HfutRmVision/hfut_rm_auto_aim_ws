// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
#ifndef MAX_ENTROPY_TRACKER_TRACKER_MANAGER_HPP_
#define MAX_ENTROPY_TRACKER_TRACKER_MANAGER_HPP_

#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "max_entropy_tracker/core/config.hpp"
#include "max_entropy_tracker/core/observation.hpp"
#include "max_entropy_tracker/trackers/base_tracker.hpp"
#include "max_entropy_tracker/trackers/adaptive_armor_tracker.hpp"
#include "max_entropy_tracker/trackers/norm_4armor_tracker.hpp"
#include "max_entropy_tracker/trackers/norm4_v3/tracker/norm4_tracker_v2.hpp"
#include "max_entropy_tracker/trackers/outpost_armor_tracker.hpp"
#include "max_entropy_tracker/trackers/outpost_tracker_v2.hpp"
#include "max_entropy_tracker/trackers/outpost_v3/outpost_tracker_v3.hpp"
#include "max_entropy_tracker/utils/observation_outlier_filter.hpp"
#include "max_entropy_tracker/utils/output_smoother.hpp"

namespace fyt::auto_aim {

/// Manages per-robot tracker instances.
class TrackerManager {
 public:
  struct TrackerEntry {
    std::unique_ptr<BaseTracker> tracker;
    double last_update_time = 0.0;
    int observation_count = 0;
  };

  struct FrameProcessResult {
    std::vector<std::string> removed_stale_ids;
    std::vector<std::string> removed_lost_ids;
  };

  struct PostProcessResult {
    bool has_smoothed = false;
    bool is_outlier = false;
    SmoothedOutput smoothed;
  };

  struct TrackerConstView {
    std::string robot_id;
    const BaseTracker *tracker = nullptr;
  };

  explicit TrackerManager(const UnifiedConfig &config, double dt = 0.01,
                          double default_r1 = 0.15, double default_r2 = 0.20,
                          double default_dza = 0.0,
                          double timeout_seconds = 0.5,
                          bool enable_oscillation = false)
      : config_(config),
        dt_(dt),
        default_r1_(default_r1),
        default_r2_(default_r2),
        default_dza_(default_dza),
        timeout_(timeout_seconds),
        enable_osc_(enable_oscillation) {}

  /// Unified per-frame entry point for tracker lifecycle orchestration.
  FrameProcessResult process_frame(
      const std::unordered_map<std::string, std::vector<ObservationData>>
          &obs_by_robot,
      double current_time,
      const SmootherConfig &smoother_cfg) {
    FrameProcessResult result;

    // Step 1: predict + stale cleanup
    predict_all(current_time);
    result.removed_stale_ids = remove_stale(current_time);

    // Step 2: update observed trackers and ensure per-robot post-process state
    for (const auto &[rid, obs_list] : obs_by_robot) {
      last_obs_counts_[rid] = static_cast<int>(obs_list.size());
      last_dual_obs_[rid] = (obs_list.size() >= 2);

      const bool is_ok = update(rid, obs_list, current_time);
      if (!is_ok) {
        continue;
      }

      auto *tracker = get(rid);
      if (tracker && tracker->is_initialized()) {
        ensure_postprocess_state(rid, smoother_cfg, *tracker);
      }
    }

    // Step 3: advance state machine for missing robots, then remove LOST trackers
    std::set<std::string> observed_ids;
    for (const auto &[rid, _] : obs_by_robot) {
      observed_ids.insert(rid);
    }
    notify_missing(observed_ids, current_time);
    result.removed_lost_ids = remove_lost();

    // Step 4: clear post-process resources for removed trackers
    std::vector<std::string> removed_ids;
    removed_ids.reserve(result.removed_stale_ids.size() +
                        result.removed_lost_ids.size());
    removed_ids.insert(removed_ids.end(), result.removed_stale_ids.begin(),
                       result.removed_stale_ids.end());
    removed_ids.insert(removed_ids.end(), result.removed_lost_ids.begin(),
                       result.removed_lost_ids.end());
    for (const auto &rid : removed_ids) {
      if (get(rid) != nullptr) {
        continue;
      }
      erase_runtime_cache(rid);
    }

    return result;
  }

  /// Apply outlier filter + smoother for one robot and return processed output.
  PostProcessResult post_process_output(const std::string &robot_id,
                                        double timestamp_seconds,
                                        const SmootherConfig &smoother_cfg) {
    PostProcessResult result;

    auto *tracker = get(robot_id);
    if (!tracker || !tracker->is_initialized()) {
      return result;
    }

    ensure_postprocess_state(robot_id, smoother_cfg, *tracker);

    const auto pos = tracker->get_center_position();
    const auto &filter = tracker->spin_filter();
    const auto idx = filter.state_idx();
    const auto &x = filter.x();
    const Eigen::Vector3d vel = tracker->get_publish_velocity();
    const double yaw = tracker->get_yaw();
    const double v_yaw = x(idx.DELTA_RATE());
    const auto [r1, r2] = tracker->get_radii();
    const double dza = filter.get_dza();
    const bool is_dual = is_last_dual_observation(robot_id);

    if (smoother_cfg.enable_outlier_filter) {
      auto of_it = outlier_filters_.find(robot_id);
      if (of_it != outlier_filters_.end()) {
        result.is_outlier = of_it->second.update(pos, yaw);
      }
    }

    if (result.is_outlier) {
      auto prev_it = last_smoothed_outputs_.find(robot_id);
      if (prev_it != last_smoothed_outputs_.end()) {
        result.smoothed = prev_it->second;
        result.has_smoothed = true;
      }
      return result;
    }

    auto sm_it = smoothers_.find(robot_id);
    if (sm_it != smoothers_.end() && smoother_cfg.enable) {
      result.smoothed = sm_it->second.smooth(
          pos, yaw, vel, v_yaw, r1, r2, dza, is_dual, timestamp_seconds);
      result.has_smoothed = true;
      last_smoothed_outputs_[robot_id] = result.smoothed;
    }

    return result;
  }

  /// Get existing tracker or create+initialize a new one.
  BaseTracker *get_or_create(
      const std::string &robot_id,
      const std::vector<ObservationData> &initial_obs,
      double current_time) {
    auto it = trackers_.find(robot_id);
    if (it != trackers_.end()) return it->second.tracker.get();
    if (initial_obs.empty()) return nullptr;

    std::unique_ptr<BaseTracker> t;
    if (robot_id == "outpost") {
      if (config_.outpost.use_tracker_v3) {
        t = std::make_unique<OutpostTrackerV3>(config_, dt_, enable_osc_);
      } else if (config_.outpost.use_tracker_v2) {
        t = std::make_unique<OutpostTrackerV2>(config_, dt_, enable_osc_);
      } else {
        t = std::make_unique<OutpostArmorTracker>(config_, dt_, enable_osc_);
      }
    } else {
      if (config_.tracker.implementation == "norm4") {
        t = std::make_unique<Norm4ArmorTrackerV2>(config_, dt_, enable_osc_);
      } else if (config_.tracker.implementation == "norm4_v2") {
        t = std::make_unique<Norm4ArmorTrackerV2>(config_, dt_, enable_osc_);
      } else {
        t = std::make_unique<AdaptiveArmorTracker>(config_, dt_, enable_osc_);
      }
    }
    t->initialize(initial_obs, default_r1_, default_r2_, default_dza_);

    auto *ptr = t.get();
    TrackerEntry entry;
    entry.tracker = std::move(t);
    entry.last_update_time = current_time;
    entry.observation_count = static_cast<int>(initial_obs.size());
    trackers_[robot_id] = std::move(entry);
    return ptr;
  }

  /// Update a robot's tracker; auto-creates if needed.
  bool update(const std::string &robot_id,
              const std::vector<ObservationData> &obs,
              double current_time) {
    std::cout << "Updating tracker for robot_id=" << robot_id
              << " with obs_count=" << obs.size() << std::endl;
    if (obs.empty()) return false;
    double t = current_time;
    auto it = trackers_.find(robot_id);
    if (it == trackers_.end()) {
      std::cout << "No existing tracker for robot_id=" << robot_id
                << ", creating new one." << std::endl;
      return get_or_create(robot_id, obs, current_time) != nullptr;
    }

    bool ok = it->second.tracker->update(obs);
    if (ok) {
      it->second.last_update_time = t;
      it->second.observation_count += static_cast<int>(obs.size());
      std::cout << "Updated tracker for robot_id=" << robot_id
                << ", total_obs_count=" << it->second.observation_count
                << std::endl;
    }

    std::cout << "Tracker state for robot_id=" << robot_id
              << " is now " << (it->second.tracker->is_tracking() ? "TRACKING" : "OTHER")
              << std::endl;
    return ok;
  }

  /// Notify trackers that were NOT observed in the current frame.
  /// This drives the state machine: TRACKING → TEMP_LOST → LOST.
  void notify_missing(const std::set<std::string> &observed_ids,
                      double /*current_time*/) {
    for (auto &[id, entry] : trackers_) {
      if (observed_ids.count(id) == 0 && entry.tracker->is_initialized()) {
        // Call update with empty observations to trigger handle_observation_loss
        entry.tracker->update({});
        std::cout << "[TrackerManager] notify_missing: robot_id=" << id
                  << " state=" << tracker_state_to_string(entry.tracker->state())
                  << " lost_count=" << entry.tracker->lost_count() << std::endl;
      }
    }
  }

  /// Remove trackers that have transitioned to LOST state.
  std::vector<std::string> remove_lost() {
    std::vector<std::string> removed;
    for (auto it = trackers_.begin(); it != trackers_.end();) {
      if (it->second.tracker->is_lost()) {
        std::cout << "[TrackerManager] Removing LOST tracker: robot_id=" << it->first << std::endl;
        removed.push_back(it->first);
        erase_runtime_cache(it->first);
        it = trackers_.erase(it);
      } else {
        ++it;
      }
    }
    return removed;
  }

  /// Predict all active trackers to target_time.
  void predict_all(std::optional<double> target_time = std::nullopt) {
    for (auto &[id, entry] : trackers_) {
      if (entry.tracker->is_initialized())
        entry.tracker->predict(target_time);
    }
  }

  /// Remove trackers not updated within timeout.
  std::vector<std::string> remove_stale(double current_time) {
    double t = current_time;
    std::vector<std::string> removed;
    for (auto it = trackers_.begin(); it != trackers_.end();) {
      if (t - it->second.last_update_time > timeout_) {
        std::cout << "Removing stale tracker for robot_id=" << it->first
                  << ", last_update_time=" << it->second.last_update_time
                  << ", current_time=" << t << std::endl;
        std::cout << "delay=" << (t - it->second.last_update_time) << "s exceeds timeout=" << timeout_ << "s" << std::endl;
        removed.push_back(it->first);
        erase_runtime_cache(it->first);
        it = trackers_.erase(it);
      } else {
        ++it;
      }
    }
    return removed;
  }

  BaseTracker *get(const std::string &id) {
    auto it = trackers_.find(id);
    return (it != trackers_.end()) ? it->second.tracker.get() : nullptr;
  }

  const std::unordered_map<std::string, TrackerEntry> &trackers() const {
    return trackers_;
  }

  std::vector<TrackerConstView> initialized_tracker_views() const {
    std::vector<TrackerConstView> views;
    views.reserve(trackers_.size());
    for (const auto &[robot_id, entry] : trackers_) {
      if (!entry.tracker || !entry.tracker->is_initialized()) {
        continue;
      }
      views.push_back(TrackerConstView{robot_id, entry.tracker.get()});
    }
    return views;
  }

  /// Returns IDs of trackers in TRACKING state only.
  std::vector<std::string> tracking_robot_ids() const {
    std::vector<std::string> ids;
    for (const auto &[id, e] : trackers_)
      if (e.tracker->is_tracking()) ids.push_back(id);
    return ids;
  }

  /// Returns IDs of trackers in TRACKING or TEMP_LOST state (active trackers).
  std::vector<std::string> active_robot_ids() const {
    std::vector<std::string> ids;
    for (const auto &[id, e] : trackers_)
      if (e.tracker->is_tracking() || e.tracker->is_temp_lost())
        ids.push_back(id);
    return ids;
  }

  int visible_observation_count(const std::string &robot_id) const {
    auto it = last_obs_counts_.find(robot_id);
    if (it == last_obs_counts_.end()) {
      return 0;
    }
    return it->second;
  }

  bool is_last_dual_observation(const std::string &robot_id) const {
    auto it = last_dual_obs_.find(robot_id);
    if (it == last_dual_obs_.end()) {
      return false;
    }
    return it->second;
  }

  int num_trackers() const { return static_cast<int>(trackers_.size()); }
  void clear() {
    trackers_.clear();
    smoothers_.clear();
    last_obs_counts_.clear();
    last_dual_obs_.clear();
    outlier_filters_.clear();
    last_smoothed_outputs_.clear();
  }

 private:
  static double now() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch())
        .count();
  }

  UnifiedConfig config_;
  double dt_, default_r1_, default_r2_, default_dza_, timeout_;
  bool enable_osc_;
  std::unordered_map<std::string, TrackerEntry> trackers_;

  // Per-robot post-process resources and caches.
  std::unordered_map<std::string, OutputSmoother> smoothers_;
  std::unordered_map<std::string, int> last_obs_counts_;
  std::unordered_map<std::string, bool> last_dual_obs_;
  std::unordered_map<std::string, ObservationOutlierFilter> outlier_filters_;
  std::unordered_map<std::string, SmoothedOutput> last_smoothed_outputs_;

  static OutlierFilterConfig make_outlier_config(const SmootherConfig &cfg) {
    OutlierFilterConfig ocfg;
    ocfg.enable = cfg.enable_outlier_filter;
    ocfg.method = cfg.outlier_method;
    ocfg.window_size = cfg.outlier_window_size;
    ocfg.min_samples = cfg.outlier_min_samples;
    ocfg.mad_k = cfg.outlier_mad_k;
    ocfg.iqr_k = cfg.outlier_iqr_k;
    ocfg.mahal_threshold = cfg.outlier_mahal_threshold;
    return ocfg;
  }

  void ensure_postprocess_state(const std::string &robot_id,
                                const SmootherConfig &smoother_cfg,
                                BaseTracker &tracker) {
    if (smoothers_.find(robot_id) == smoothers_.end()) {
      OutputSmoother sm(smoother_cfg);
      auto [r1, r2] = tracker.get_radii();
      const double dza = tracker.spin_filter().get_dza();
      sm.initialize(r1, r2, dza);
      smoothers_.emplace(robot_id, std::move(sm));
    }

    if (outlier_filters_.find(robot_id) == outlier_filters_.end()) {
      outlier_filters_.emplace(robot_id,
                               ObservationOutlierFilter(
                                   make_outlier_config(smoother_cfg)));
    }
  }

  void erase_runtime_cache(const std::string &robot_id) {
    smoothers_.erase(robot_id);
    last_obs_counts_.erase(robot_id);
    last_dual_obs_.erase(robot_id);
    outlier_filters_.erase(robot_id);
    last_smoothed_outputs_.erase(robot_id);
  }
};

}  // namespace fyt::auto_aim

#endif  // MAX_ENTROPY_TRACKER_TRACKER_MANAGER_HPP_
