// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
#include "max_entropy_tracker/trackers/norm4_v3/tracker/norm4_tracker_v2.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace fyt::auto_aim {

Norm4ArmorTrackerV2::Norm4ArmorTrackerV2(const UnifiedConfig &config, double dt,
                                         bool /*enable_oscillation*/)
    : BaseTracker(dt),
      config_(config),
      backend_(norm4_v3::create_backend(
          norm4_v3::backend_type_from_string(
              config_.norm4_v3.backend_config.backend_type),
          config, dt)),
      maneuver_detector_(config.maneuver) {}

void Norm4ArmorTrackerV2::initialize(const std::vector<ObservationData> &obs,
                                     double r1, double r2, double dza) {
  if (obs.empty())
    throw std::invalid_argument("At least one observation required");

  default_r1_ = r1;
  default_r2_ = r2;
  default_dza_ = dza;
  warmup_last_obs_ = obs[0];

  const auto &warmup_cfg = config_.norm4_v3.warmup;
  bool use_warmup = warmup_cfg.enable_dual_seed_01 && !obs[0].panel_id.has_value();

  if (use_warmup) {
    init_warmup(obs, r1, r2, dza);
    set_mode(norm4_v3::Norm4V2Mode::AMBIGUOUS);
  } else {
    int init_panel = obs[0].panel_id.value_or(0);
    backend_->reset(obs[0], init_panel, default_r1_, default_r2_, default_dza_);
    current_panel_id_ = init_panel;
    set_mode(norm4_v3::Norm4V2Mode::AMBIGUOUS);
  }

  transition_to(TrackerState::INITIALIZING);
  mark_initialized();
  update_time(obs[0].timestamp.value_or(0.0));
}

void Norm4ArmorTrackerV2::predict(std::optional<double> target_time) {
  if (!is_initialized()) return;
  double dt = compute_dt(target_time);
  if (dt <= 0.0) return;
  backend_->predict(dt);
  if (target_time.has_value()) update_time(target_time.value());
}

bool Norm4ArmorTrackerV2::update(const std::vector<ObservationData> &obs) {
  if (!is_initialized() || obs.empty()) {
    handle_observation_loss(config_.tracker.tracking_thres,
                            config_.tracker.lost_thres);
    return false;
  }

  double obs_ts =
      obs[0].timestamp.value_or(current_time_.value_or(0.0));

  // Predict to observation time.
  if (current_time_.has_value() && obs_ts > current_time_.value()) {
    backend_->predict(obs_ts - current_time_.value());
  }

  // Route through warmup if active (0/1 dual-seed settling phase).
  if (warmup_state_.active) {
    bool warmup_ok = run_warmup(obs);
    update_time(obs_ts);
    increment_frame();
    if (warmup_ok) {
      handle_observation_received(config_.tracker.tracking_thres);
    } else {
      handle_observation_loss(config_.tracker.tracking_thres,
                              config_.tracker.lost_thres);
    }

    last_hypothesis_debug_.valid = true;
    last_hypothesis_debug_.obs_count = static_cast<int>(obs.size());
    last_hypothesis_debug_.committed = false;
    last_hypothesis_debug_.degraded = true;
    last_hypothesis_debug_.decision_reason = warmup_state_.warmup_reason;

    populate_debug_snapshot();
    return warmup_ok;
  }

  // Build prior snapshot — all hypotheses evaluated from this single prior.
  auto ctx = backend_->buildPredictContext();

  // Generate hypotheses: 1 obs → 4 single, 2+ obs → 8 dual.
  auto hypotheses = hypothesis_generator_.generate(obs);
  hypothesis_generator_.attach_prior(&hypotheses);

  // Evaluate all hypotheses from the same prior.
  std::vector<norm4_v3::MeasurementEval> evals;
  evals.reserve(hypotheses.size());

  for (const auto &hyp : hypotheses) {
    norm4_v3::MeasurementEval eval;
    if (hyp.kind == norm4_v3::HypothesisKind::Single) {
      int panel = hyp.assignments[0].panel_id;
      eval = backend_->evaluateSingle(ctx, obs[hyp.assignments[0].obs_index],
                                       panel);
    } else {
      int p0 = hyp.assignments[0].panel_id;
      int p1 = hyp.assignments[1].panel_id;
      eval = backend_->evaluateDual(ctx, obs[hyp.assignments[0].obs_index],
                                     obs[hyp.assignments[1].obs_index], p0, p1);
    }
    // Combine prior log weight into score.
    eval.score = eval.log_likelihood + hyp.prior_log_weight;
    evals.push_back(eval);
  }

  // Select TopK, compute confidence.
  const auto &sel_cfg = config_.norm4_v3.hypothesis_selector;
  int topk_count = std::max(1, sel_cfg.topk);
  std::vector<norm4_v3::TopKEntry> topk;
  double top1_confidence = 0.0, top1_top2_margin = 0.0;
  select_topk(evals, hypotheses, topk_count, &topk,
              &top1_confidence, &top1_top2_margin);

  bool committed = false;
  std::string decision_reason;

  // Find the first gate-passing hypothesis in ranked order.
  int best_idx = -1;
  for (size_t i = 0; i < topk.size(); ++i) {
    if (topk[i].eval.gate_pass && topk[i].eval.valid) {
      best_idx = static_cast<int>(i);
      break;
    }
  }

  bool allow_commit = (mode_ == norm4_v3::Norm4V2Mode::STRUCTURED);
  if (!allow_commit) {
    decision_reason = "ambiguous_mode_predict_only";
  } else if (best_idx >= 0) {
    // Commit gate: confidence / margin check before attempting trial.
    bool commit_gate_pass = true;
    std::ostringstream gate_oss;

    if (top1_confidence < sel_cfg.min_top1_confidence) {
      commit_gate_pass = false;
      gate_oss << "conf=" << top1_confidence
               << "<" << sel_cfg.min_top1_confidence;
    }
    if (top1_top2_margin < sel_cfg.min_top1_top2_margin) {
      commit_gate_pass = false;
      if (!gate_oss.str().empty()) gate_oss << ",";
      gate_oss << "margin=" << top1_top2_margin
               << "<" << sel_cfg.min_top1_top2_margin;
    }

    if (commit_gate_pass) {
      const auto &best_hyp = topk[best_idx].hypothesis;
      norm4_v3::UkfTrial trial;

      if (best_hyp.kind == norm4_v3::HypothesisKind::Single) {
        trial = backend_->tryUpdateSingle(
            ctx, obs[best_hyp.assignments[0].obs_index],
            best_hyp.assignments[0].panel_id);
      } else {
        trial = backend_->tryUpdateDual(
            ctx, obs[best_hyp.assignments[0].obs_index],
            obs[best_hyp.assignments[1].obs_index],
            best_hyp.assignments[0].panel_id,
            best_hyp.assignments[1].panel_id);
      }

      // Triple gate: trial success + posterior sanity + reconstruction error.
      bool trial_ok = trial.success && trial.posterior_sanity_pass;
      if (trial_ok &&
          trial.reconstruction_pos_error > sel_cfg.max_reconstruction_pos_error) {
        trial_ok = false;
        std::ostringstream oss;
        oss << "reconstruction=" << trial.reconstruction_pos_error
            << ">" << sel_cfg.max_reconstruction_pos_error;
        trial.reject_reason = oss.str();
      }

      if (trial_ok) {
        backend_->commit(trial);
        committed = true;

        if (best_hyp.kind == norm4_v3::HypothesisKind::Single) {
          current_panel_id_ = best_hyp.assignments[0].panel_id;
        }

        std::ostringstream oss;
        oss << "committed_" << best_hyp.debug_name
            << "_nis=" << trial.eval.nis
            << "_conf=" << top1_confidence
            << "_margin=" << top1_top2_margin
            << "_recon=" << trial.reconstruction_pos_error;
        decision_reason = oss.str();
      } else {
        std::ostringstream oss;
        oss << "trial_rejected:" << trial.reject_reason;
        decision_reason = oss.str();
      }
    } else {
      decision_reason = "commit_gate_fail:" + gate_oss.str();
    }
  } else if (allow_commit) {
    decision_reason = "all_gate_fail";
  }

  // Update state tracking.
  update_time(obs_ts);
  increment_frame();

  if (committed) {
    handle_observation_received(config_.tracker.tracking_thres);
  } else {
    handle_observation_loss(config_.tracker.tracking_thres,
                            config_.tracker.lost_thres);
  }

  // Populate debug frame.
  last_hypothesis_debug_.valid = true;
  last_hypothesis_debug_.obs_count = static_cast<int>(obs.size());
  last_hypothesis_debug_.committed = committed;
  last_hypothesis_debug_.degraded = !committed;
  last_hypothesis_debug_.topk = topk;
  last_hypothesis_debug_.top1_confidence = top1_confidence;
  last_hypothesis_debug_.top1_top2_margin = top1_top2_margin;
  last_hypothesis_debug_.decision_reason = decision_reason;

  populate_debug_snapshot();

  // Fix: return true when observations were processed, even if not committed.
  // This prevents TrackerManager from treating reject frames as missing observations
  // and prematurely marking the tracker as stale/lost.
  return true;
}

Eigen::Vector3d Norm4ArmorTrackerV2::get_center_position() const {
  return backend_->spin_filter().get_center_position();
}

double Norm4ArmorTrackerV2::get_yaw() const {
  // Published TrackedRobot uses the shared armors_offset convention where
  // armor 0 is at local (-r, 0). Shift the internal center yaw by pi so
  // downstream projection reconstructs armor positions consistently.
  return normalize_angle(backend_->spin_filter().get_yaw() + M_PI);
}

std::pair<double, double> Norm4ArmorTrackerV2::get_radii() const {
  return backend_->spin_filter().get_radii();
}

SpinFilterInterface &Norm4ArmorTrackerV2::spin_filter() {
  return backend_->spin_filter();
}

const SpinFilterInterface &Norm4ArmorTrackerV2::spin_filter() const {
  return backend_->spin_filter();
}

ManeuverResult Norm4ArmorTrackerV2::assess_maneuver() const {
  const auto &sf = backend_->spin_filter();
  return maneuver_detector_.detect(sf.last_nis(),
                                   sf.last_innov_xyz().norm(),
                                   sf.last_update_type());
}

Eigen::Vector3d Norm4ArmorTrackerV2::get_publish_velocity() const {
  const auto &sf = backend_->spin_filter();
  const auto &idx = sf.state_idx();
  const auto &x = sf.x();
  Eigen::Vector3d vel(x(idx.VX()), x(idx.VY()), x(idx.VZ()));
  return vel;
}

bool Norm4ArmorTrackerV2::is_ambiguous_single_mode() const {
  return mode_ == norm4_v3::Norm4V2Mode::AMBIGUOUS;
}

int Norm4ArmorTrackerV2::effective_num_armors() const { return 4; }

double Norm4ArmorTrackerV2::confidence_scale() const {
  return (mode_ == norm4_v3::Norm4V2Mode::AMBIGUOUS) ? 0.7 : 1.0;
}

std::vector<geometry_msgs::msg::Pose>
Norm4ArmorTrackerV2::build_armors_offset_for_message() const {
  return {};
}

void Norm4ArmorTrackerV2::select_topk(
    std::vector<norm4_v3::MeasurementEval> &evals,
    const std::vector<norm4_v3::Hypothesis> &hyps,
    int topk_count,
    std::vector<norm4_v3::TopKEntry> *topk_out,
    double *confidence_out, double *margin_out) const {
  const int n = static_cast<int>(evals.size());
  if (n == 0) {
    *confidence_out = 0.0;
    *margin_out = 0.0;
    return;
  }

  // Create indexed list.
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  // Sort: gate_pass first, then by score descending.
  std::sort(indices.begin(), indices.end(),
            [&evals](int a, int b) {
              if (evals[a].gate_pass != evals[b].gate_pass)
                return evals[a].gate_pass;
              return evals[a].score > evals[b].score;
            });

  // Take TopK.
  const int k = std::min(topk_count, n);
  topk_out->clear();
  topk_out->reserve(k);

  // Compute log-sum-exp over topk for softmax.
  double max_score = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < k; ++i) {
    max_score = std::max(max_score, evals[indices[i]].score);
  }

  double logsumexp = 0.0;
  std::vector<double> weights(k);
  for (int i = 0; i < k; ++i) {
    weights[i] = std::exp(evals[indices[i]].score - max_score);
    logsumexp += weights[i];
  }

  for (int i = 0; i < k; ++i) {
    weights[i] /= logsumexp;
    norm4_v3::TopKEntry entry;
    entry.hypothesis = hyps[indices[i]];
    entry.eval = evals[indices[i]];
    entry.normalized_weight = weights[i];
    topk_out->push_back(entry);
  }

  *confidence_out = topk_out->empty() ? 0.0 : topk_out->front().normalized_weight;
  *margin_out = (k >= 2)
                    ? (topk_out->at(0).eval.score - topk_out->at(1).eval.score)
                    : std::numeric_limits<double>::infinity();
}

void Norm4ArmorTrackerV2::populate_debug_snapshot() {
  debug_snapshot_.valid = last_hypothesis_debug_.valid;
  debug_snapshot_.track_mode = (mode_ == norm4_v3::Norm4V2Mode::AMBIGUOUS) ? 1 : 0;
  debug_snapshot_.current_panel_id = current_panel_id_;
  debug_snapshot_.mode_state = static_cast<int>(mode_);

  if (!last_hypothesis_debug_.topk.empty()) {
    const auto &top = last_hypothesis_debug_.topk.front();
    debug_snapshot_.candidate_panel_id = top.hypothesis.assignments[0].panel_id;
    debug_snapshot_.candidate_prob = top.normalized_weight;
    debug_snapshot_.entropy_norm = 1.0 - top.normalized_weight;
    debug_snapshot_.max_prob = top.normalized_weight;

    if (last_hypothesis_debug_.topk.size() >= 2) {
      const auto &second = last_hypothesis_debug_.topk[1];
      debug_snapshot_.candidate_margin =
          top.eval.score - second.eval.score;
    }
  }

  debug_snapshot_.binding_confidence = last_hypothesis_debug_.top1_confidence;
  debug_snapshot_.degraded_single_obs_mode = last_hypothesis_debug_.degraded;
  debug_snapshot_.committed = last_hypothesis_debug_.committed;
  debug_snapshot_.top1_confidence = last_hypothesis_debug_.top1_confidence;
  debug_snapshot_.top1_top2_margin = last_hypothesis_debug_.top1_top2_margin;
  debug_snapshot_.decision_reason = last_hypothesis_debug_.decision_reason;
  debug_snapshot_.warmup_active = warmup_state_.active ? 1 : 0;
  if (!last_hypothesis_debug_.topk.empty()) {
    debug_snapshot_.top1_nis = last_hypothesis_debug_.topk.front().eval.nis;
  }
}

// ── Warmup / Mode Routing ──

void Norm4ArmorTrackerV2::init_warmup(const std::vector<ObservationData> &obs,
                                       double r1, double r2, double dza) {
  warmup_state_ = norm4_v3::WarmupState{};
  warmup_state_.active = true;

  warmup_state_.h0.seed_panel = 0;
  warmup_state_.h0.r1 = r1;
  warmup_state_.h0.r2 = r2;
  warmup_state_.h0.dza = dza;

  warmup_state_.h1.seed_panel = 1;
  warmup_state_.h1.r1 = r1;
  warmup_state_.h1.r2 = r2;
  warmup_state_.h1.dza = dza;

  // Initialize the structured backend with H0 (panel 0) for predict continuity.
  // The warmup internally evaluates both H0 and H1 via evaluateSingle.
  backend_->reset(obs[0], 0, r1, r2, dza);
  warmup_last_obs_ = obs[0];
  current_panel_id_ = -1;  // ambiguous during warmup

  warmup_state_.warmup_reason = "warmup_init_seed_01";
}

bool Norm4ArmorTrackerV2::run_warmup(const std::vector<ObservationData> &obs) {
  const auto &warmup_cfg = config_.norm4_v3.warmup;
  warmup_state_.total_frames++;

  // Predict and get prior context.
  auto ctx = backend_->buildPredictContext();

  // Use first observation for single-obs shallow evaluation on both seeds.
  const auto &primary_obs = obs[0];
  warmup_last_obs_ = primary_obs;

  // Evaluate H0 (panel 0 assumption).
  auto eval_h0 = backend_->evaluateSingle(ctx, primary_obs, 0);
  if (eval_h0.gate_pass && eval_h0.valid) {
    warmup_state_.h0.gate_pass_count++;
    warmup_state_.h0.accumulated_score += eval_h0.log_likelihood;
  }
  warmup_state_.h0.total_frames++;

  // Evaluate H1 (panel 1 assumption).
  auto eval_h1 = backend_->evaluateSingle(ctx, primary_obs, 1);
  if (eval_h1.gate_pass && eval_h1.valid) {
    warmup_state_.h1.gate_pass_count++;
    warmup_state_.h1.accumulated_score += eval_h1.log_likelihood;
  }
  warmup_state_.h1.total_frames++;

  if (!std::isfinite(eval_h0.log_likelihood) ||
      !std::isfinite(eval_h1.log_likelihood)) {
    warmup_state_.settle_frames = 0;
    warmup_state_.warmup_reason = "warmup_nonfinite_score";
    return true;
  }

  // Compute margin between H0 and H1.
  double margin = eval_h0.log_likelihood - eval_h1.log_likelihood;
  double abs_margin = std::abs(margin);

  // Compute confidence via softmax between the two.
  double max_score = std::max(eval_h0.log_likelihood, eval_h1.log_likelihood);
  double w0 = std::exp(eval_h0.log_likelihood - max_score);
  double w1 = std::exp(eval_h1.log_likelihood - max_score);
  double confidence = std::max(w0, w1) / (w0 + w1);

  bool h0_better = eval_h0.log_likelihood > eval_h1.log_likelihood;

  std::ostringstream oss;
  oss << "warmup_f" << warmup_state_.total_frames
      << "_h0=" << eval_h0.log_likelihood
      << "_h1=" << eval_h1.log_likelihood
      << "_margin=" << abs_margin
      << "_conf=" << confidence;
  warmup_state_.warmup_reason = oss.str();

  // Check convergence conditions.
  bool has_min_frames =
      warmup_state_.total_frames >= warmup_cfg.warmup_frames;

  if (has_min_frames &&
      abs_margin > warmup_cfg.min_margin_to_commit &&
      confidence > warmup_cfg.min_confidence_to_commit) {
    warmup_state_.settle_frames++;

    if (warmup_state_.settle_frames >= warmup_cfg.min_settle_frames) {
      // Converged: promote winner.
      warmup_state_.active = false;
      warmup_state_.winning_branch = h0_better ? 0 : 1;
      warmup_state_.final_margin = abs_margin;
      warmup_state_.final_confidence = confidence;

      promote_warmup_winner();
      return true;
    }
  } else {
    warmup_state_.settle_frames = 0;
  }

  // Timeout: stay ambiguous, keep shallow predict-only.
  if (warmup_state_.total_frames > warmup_cfg.warmup_frames * 3) {
    warmup_state_.active = false;
    warmup_state_.warmup_reason += "_timeout";
    // Stay in AMBIGUOUS mode with current backend state.
    // backend_ keeps its last state; continue predict-only.
  }

  // During warmup, do not commit — backend predict-only.
  return true;
}

void Norm4ArmorTrackerV2::promote_warmup_winner() {
  int winner_panel = warmup_state_.winning_branch == 0 ? 0 : 1;
  if (warmup_last_obs_.has_value()) {
    backend_->reset(warmup_last_obs_.value(), winner_panel, default_r1_,
                    default_r2_, default_dza_);
  }
  current_panel_id_ = winner_panel;
  set_mode(norm4_v3::Norm4V2Mode::STRUCTURED);

  std::ostringstream oss;
  oss << "warmup_promoted_panel" << winner_panel
      << "_margin=" << warmup_state_.final_margin
      << "_conf=" << warmup_state_.final_confidence;
  warmup_state_.warmup_reason = oss.str();
}

void Norm4ArmorTrackerV2::set_mode(norm4_v3::Norm4V2Mode m) {
  mode_ = m;
  apply_mode_routing();
}

void Norm4ArmorTrackerV2::apply_mode_routing() {
  const auto &routing = config_.norm4_v3.mode_routing;
  // Phase-1 routing contract in this tracker:
  // AMBIGUOUS: non-destructive predict/evaluate path (no commit in update).
  // STRUCTURED: enable commit path.
  // Output-side single-plate bridge is still external to this class.
  (void)routing;
}

}  // namespace fyt::auto_aim
