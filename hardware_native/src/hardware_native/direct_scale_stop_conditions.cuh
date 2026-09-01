#ifndef HARDWARE_NATIVE_DIRECT_SCALE_STOP_CONDITIONS_CUH
#define HARDWARE_NATIVE_DIRECT_SCALE_STOP_CONDITIONS_CUH

#include "hardware_native/direct_scale_frontier_meters.cuh"

namespace substrate::direct_adult {

// Observer-side stop predicates for active-frontier scaling, condensation,
// and contradiction reopen on one continuing canonical adult. Host snapshots
// may read DirectBrain.resource_ecology without feeding measurements back in:
//   DirectResourceEcologyState host{};
//   cudaMemcpy(&host, brain.resource_ecology, sizeof(host), cudaMemcpyDeviceToHost);
// Refuse mixed adult identity or fixture promotion at the caller boundary.
// A verdict is GREEN only when every executable stop flag is clear.

struct DirectScaleProbeSamples {
  DirectScaleFrontierMeters baseline;
  DirectScaleFrontierMeters after_corpus;
  DirectScaleFrontierMeters first_contact;
  DirectScaleFrontierMeters tenth_contact;
  DirectScaleFrontierMeters after_contradiction;
  std::uint64_t baseline_corpus_records, after_corpus_records;
};

struct DirectScaleStopFlags {
  bool abort_wrong_root;
  bool abort_insufficient_samples;
  bool frontier_tracks_corpus;
  bool repeat_not_cheaper;
  bool contradiction_did_not_reopen;
  bool new_module_installed;
};

inline bool scale_live_frontier_grew(const DirectScaleFrontierMeters& before,
                                    const DirectScaleFrontierMeters& after) {
  return after.s3_live_unresolved_units > before.s3_live_unresolved_units ||
         after.s4_active_closures > before.s4_active_closures ||
         after.frontier_work > before.frontier_work ||
         after.live_occurrences > before.live_occurrences;
}

inline bool scale_frontier_tracks_corpus(const DirectScaleProbeSamples& samples) {
  const bool corpus_grew = samples.after_corpus_records > samples.baseline_corpus_records;
  const bool vram_grew = samples.after_corpus.vram_charged_bytes > samples.baseline.vram_charged_bytes;
  return corpus_grew && (vram_grew || scale_live_frontier_grew(samples.baseline, samples.after_corpus));
}

inline bool scale_repeat_not_cheaper(const DirectScaleFrontierMeters& first,
                                    const DirectScaleFrontierMeters& tenth) {
  return tenth.frontier_work >= first.frontier_work &&
         tenth.s3_live_unresolved_units >= first.s3_live_unresolved_units;
}

inline bool scale_contradiction_did_not_reopen(const DirectScaleFrontierMeters& tenth,
                                              const DirectScaleFrontierMeters& after) {
  return after.frontier_work <= tenth.frontier_work &&
         after.s3_live_unresolved_units <= tenth.s3_live_unresolved_units &&
         after.s4_active_closures <= tenth.s4_active_closures;
}

inline DirectScaleStopFlags evaluate_scale_stop_conditions(
    const DirectScaleProbeSamples& samples, bool samples_complete, bool birth_root_mismatch,
    bool new_module_installed) {
  DirectScaleStopFlags flags{};
  flags.abort_wrong_root = birth_root_mismatch;
  flags.abort_insufficient_samples = !samples_complete;
  flags.new_module_installed = new_module_installed;
  if (flags.abort_wrong_root || flags.abort_insufficient_samples) return flags;
  flags.frontier_tracks_corpus = scale_frontier_tracks_corpus(samples);
  flags.repeat_not_cheaper =
      scale_repeat_not_cheaper(samples.first_contact, samples.tenth_contact);
  flags.contradiction_did_not_reopen =
      scale_contradiction_did_not_reopen(samples.tenth_contact, samples.after_contradiction);
  return flags;
}

inline bool scale_must_stop_filming(const DirectScaleStopFlags& flags) {
  return flags.abort_wrong_root || flags.abort_insufficient_samples ||
         flags.frontier_tracks_corpus || flags.repeat_not_cheaper ||
         flags.contradiction_did_not_reopen || flags.new_module_installed;
}

inline bool scale_verdict_is_green(const DirectScaleStopFlags& flags) {
  return !scale_must_stop_filming(flags);
}

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_SCALE_STOP_CONDITIONS_CUH
