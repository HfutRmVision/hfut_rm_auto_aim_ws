// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
#include "max_entropy_tracker/trackers/norm4_v3/hypothesis/norm4_hypothesis_generator.hpp"

#include <sstream>

namespace fyt::auto_aim::norm4_v3 {

std::vector<Hypothesis> HypothesisGenerator::generate(
    const std::vector<ObservationData> &observations) const {
  return generate_all_candidates(observations);
}

std::vector<Hypothesis> HypothesisGenerator::generate_single(
    int obs_index) const {
  std::vector<Hypothesis> hyps;
  hyps.reserve(4);

  for (int panel = 0; panel < 4; ++panel) {
    Hypothesis h;
    h.kind = HypothesisKind::Single;
    h.assignments[0] = {obs_index, panel};
    h.assignment_count = 1;
    h.prior_log_weight = 0.0;

    std::ostringstream oss;
    oss << "single_obs" << obs_index << "_panel" << panel;
    h.debug_name = oss.str();

    hyps.push_back(h);
  }
  return hyps;
}

std::vector<Hypothesis> HypothesisGenerator::generate_dual(
    int obs0, int obs1) const {
  // Ordered adjacent pairs: (0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,0),(0,3)
  static const int pairs[8][2] = {
      {0, 1}, {1, 0}, {1, 2}, {2, 1},
      {2, 3}, {3, 2}, {3, 0}, {0, 3},
  };

  std::vector<Hypothesis> hyps;
  hyps.reserve(8);

  for (const auto &p : pairs) {
    Hypothesis h;
    h.kind = HypothesisKind::Dual;
    h.assignments[0] = {obs0, p[0]};
    h.assignments[1] = {obs1, p[1]};
    h.assignment_count = 2;
    h.prior_log_weight = 0.0;

    std::ostringstream oss;
    oss << "dual_o" << obs0 << "p" << p[0] << "_o" << obs1 << "p" << p[1];
    h.debug_name = oss.str();

    hyps.push_back(h);
  }
  return hyps;
}

std::vector<Hypothesis> HypothesisGenerator::generate_all_candidates(
    const std::vector<ObservationData> &observations) const {
  std::vector<Hypothesis> hyps;
  if (observations.empty()) return hyps;

  const int obs_count = static_cast<int>(observations.size());
  hyps.reserve(static_cast<size_t>(obs_count * 4 + obs_count * (obs_count - 1) * 4));

  for (int obs_index = 0; obs_index < obs_count; ++obs_index) {
    auto single_hyps = generate_single(obs_index);
    hyps.insert(hyps.end(), single_hyps.begin(), single_hyps.end());
  }

  for (int obs0 = 0; obs0 < obs_count; ++obs0) {
    for (int obs1 = obs0 + 1; obs1 < obs_count; ++obs1) {
      auto dual_hyps = generate_dual(obs0, obs1);
      hyps.insert(hyps.end(), dual_hyps.begin(), dual_hyps.end());
    }
  }

  return hyps;
}

}  // namespace fyt::auto_aim::norm4_v3
