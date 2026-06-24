// Copyright (C) Max Entropy Tracker. Licensed under the MIT License.
#ifndef MAX_ENTROPY_TRACKER_TRACKERS_NORM4_V3_NORM4_HYPOTHESIS_GENERATOR_HPP_
#define MAX_ENTROPY_TRACKER_TRACKERS_NORM4_V3_NORM4_HYPOTHESIS_GENERATOR_HPP_

#include <string>
#include <vector>

#include "max_entropy_tracker/core/observation.hpp"
#include "max_entropy_tracker/trackers/norm4_v3/hypothesis/norm4_hypothesis_types.hpp"

namespace fyt::auto_aim::norm4_v3 {

class HypothesisGenerator {
 public:
  HypothesisGenerator() = default;

  std::vector<Hypothesis> generate(
      const std::vector<ObservationData> &observations) const;

  void attach_prior(std::vector<Hypothesis> * /*hypotheses*/) const {
    // Phase 1: prior_log_weight stays 0.0; evidence integration in later phases.
  }

 private:
  std::vector<Hypothesis> generate_single(int obs_index) const;
  std::vector<Hypothesis> generate_dual(int obs0, int obs1) const;
  std::vector<Hypothesis> generate_all_candidates(
      const std::vector<ObservationData> &observations) const;
};

}  // namespace fyt::auto_aim::norm4_v3

#endif  // MAX_ENTROPY_TRACKER_TRACKERS_NORM4_V3_NORM4_HYPOTHESIS_GENERATOR_HPP_
