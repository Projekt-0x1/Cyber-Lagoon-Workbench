__device__ inline bool valid_staging(const DeviceInputs* inputs) {
  return inputs != nullptr &&
         ((inputs->predict_staged != 0u) != (inputs->observe_staged != 0u));
}

__device__ inline bool step_available(const SiteWord* words, const DeviceInputs* inputs) {
  if (inputs == nullptr)
    return true;
  const bool requested =
      inputs->predict_staged != 0u || inputs->observe_staged != 0u;
  if (requested && !valid_staging(inputs))
    return false;
  if (!requested && !has_live_eligibility(words))
    return true;
  return read_unsigned(words, pair_index(kGlobalBase, kJournalCount, 0u)) < kJournalDepth;
}

static __global__ void preflight_kernel(const SiteWord* words, const DeviceInputs* inputs,
                                        std::uint32_t* available) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && available != nullptr &&
      !step_available(words, inputs))
    *available = 0u;
}

static __global__ void reserve_kernel(SiteWord* words, const DeviceLayout& layout,
                                      const DeviceInputs* inputs, DeviceScratch* scratch,
                                      Receipt* receipt, const std::uint32_t* available) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr)
    return;
  scratch->reserved = 0u;
  scratch->accepted = 0u;
  scratch->current_form = kFormAssemblyCount;
  scratch->observed_form = kFormAssemblyCount;
  scratch->predicted_form = kFormAssemblyCount;
  scratch->transition_record = kTransitionRecordCount;
  scratch->candidate_low = 0u;
  scratch->candidate_high = 0u;
  if (receipt != nullptr) {
    receipt->selected_trajectory = kTrajectoryAssemblyCount;
    receipt->trajectory_phase = kTrajectoryPhaseCount;
  }
  const bool requested = inputs != nullptr &&
                         (inputs->predict_staged != 0u ||
                          inputs->observe_staged != 0u);
  if (available == nullptr || *available == 0u ||
      (requested && !valid_staging(inputs))) {
    if (receipt != nullptr && requested && valid_staging(inputs))
      ++receipt->journal_exhausted;
    return;
  }
  if (!requested && (inputs == nullptr || !has_live_eligibility(words)))
    return;
  if (requested) {
    const RawByteDecode raw = inputs->observe_staged != 0u
                                  ? current_observation_raw(words, layout)
                                  : current_prediction_raw(words, layout);
    if (!raw.valid) {
      if (receipt != nullptr)
        ++receipt->invalid_contact;
      return;
    }
  }
  const std::uint32_t count =
      read_unsigned(words, pair_index(kGlobalBase, kJournalCount, 0u));
  const std::uint32_t event =
      read_unsigned(words, pair_index(kGlobalBase, kEventId, 0u)) + 1u;
  scratch->journal_slot = count;
  scratch->event_id = event;
  scratch->reserved = 1u;
}

static __global__ void snapshot_kernel(SiteWord* words, const DeviceLayout& layout,
                                       const DeviceScratch* scratch) {
  if (scratch == nullptr || scratch->reserved == 0u)
    return;
  const std::uint32_t slot = scratch->journal_slot;
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    write_resident(words, journal_index(slot, 0u), scratch->event_id & (kJournalDepth - 1u));
  }
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = tid; index < kMutableRailCount; index += stride)
    words[fixed_physical_slot(journal_index(slot, kJournalMetaRailCount + index))] =
        read_resident(words, index);
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t offset = kJournalMetaRailCount + kMutableRailCount;
    words[fixed_physical_slot(journal_index(slot, offset))] = words[layout.raw_motor_zero_slot];
    words[fixed_physical_slot(journal_index(slot, offset + 1u))] = ~words[layout.raw_motor_zero_slot];
    words[fixed_physical_slot(journal_index(slot, offset + 2u))] = words[layout.raw_motor_one_slot];
    words[fixed_physical_slot(journal_index(slot, offset + 3u))] = ~words[layout.raw_motor_one_slot];
  }
}

static __global__ void mutate_kernel(SiteWord* words, const DeviceLayout& layout,
                                      const DeviceInputs* inputs, DeviceScratch* scratch,
                                      Receipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr ||
      scratch->reserved == 0u || inputs == nullptr)
    return;
  scratch->accepted = 1u;
  age_prediction_eligibility(words, receipt);
  if (inputs->predict_staged == 0u && inputs->observe_staged == 0u) {
    write_resident(words, pair_index(kGlobalBase, kEventId, 0u),
                   scratch->event_id);
    if (receipt != nullptr) {
      receipt->step_mode = kStepAged;
      receipt->event_id = scratch->event_id;
    }
    return;
  }
  if (inputs->predict_staged != 0u) {
    const RawByteDecode raw = current_prediction_raw(words, layout);
    if (raw.valid && raw.value == static_cast<std::uint8_t>('^')) {
      write_resident(words, pair_index(kGlobalBase, kPreviousTrajectory, 0u),
                     kTrajectoryStartSentinel);
      write_resident(
          words, pair_index(kGlobalBase, kPreviousObservedTrajectory, 0u),
          kTrajectoryStartSentinel);
    }
    const std::uint32_t current = find_or_recruit_form(words, raw, receipt);
    std::uint32_t record = kTransitionRecordCount;
    std::uint32_t successor = kFormAssemblyCount;
    const std::uint32_t active = read_unsigned(
        words, pair_index(kGlobalBase, kActiveTrajectory, 0u));
    if (active < kTrajectoryAssemblyCount) {
      const std::uint32_t length = read_unsigned(
          words, trajectory_meta_index(active, kTrajectoryLength, 0u));
      const std::uint32_t phase = read_unsigned(
          words, pair_index(kGlobalBase, kActiveTrajectoryPhase, 0u));
      const bool active_valid =
          read_unsigned(words,
                        trajectory_meta_index(active, kTrajectoryActive, 0u)) != 0u &&
          phase > 0u && phase < length && length <= kTrajectoryPhaseCount &&
          read_unsigned(words,
                        trajectory_phase_index(active, phase - 1u, 0u)) == current;
      if (active_valid) {
        const std::uint32_t candidate =
            read_unsigned(words, trajectory_phase_index(active, phase, 0u));
        record = find_transition(words, current, candidate);
        if (record < kTransitionRecordCount &&
            decode_form_raw(words, candidate).valid) {
          successor = candidate;
          scratch->candidate_low = record < 32u ? (1u << record) : 0u;
          scratch->candidate_high =
              record >= 32u ? (1u << (record - 32u)) : 0u;
          if (receipt != nullptr) {
            receipt->selected_trajectory = active;
            receipt->trajectory_phase = phase;
          }
          if (phase + 1u >= length) {
            write_resident(words,
                           pair_index(kGlobalBase, kPreviousTrajectory, 0u),
                           active);
            clear_active_trajectory(words);
          } else
            write_resident(
                words, pair_index(kGlobalBase, kActiveTrajectoryPhase, 0u),
                phase + 1u);
        } else {
          clear_active_trajectory(words);
          if (receipt != nullptr)
            receipt->trajectory_abstained = 1u;
        }
      } else {
        clear_active_trajectory(words);
        if (receipt != nullptr)
          receipt->trajectory_abstained = 1u;
      }
    } else if (active == kTrajectoryAssemblyCount) {
      const std::uint32_t previous = read_unsigned(
          words, pair_index(kGlobalBase, kPreviousTrajectory, 0u));
      if (has_learned_trajectory_adjacency(words)) {
        if (previous >= kTrajectoryPredecessorCount) {
          if (receipt != nullptr)
            receipt->trajectory_abstained = 1u;
        } else {
          bool ambiguous = false;
          const std::uint32_t trajectory = select_adjacent_trajectory(
              words, layout, current, previous, &record,
              &scratch->candidate_low, &scratch->candidate_high, &ambiguous);
          if (trajectory < kTrajectoryAssemblyCount &&
              arm_selected_trajectory(words, trajectory, receipt)) {
            successor = read_unsigned(
                words, trajectory_phase_index(trajectory, 0u, 0u));
          } else {
            successor = kFormAssemblyCount;
            record = kTransitionRecordCount;
            scratch->candidate_low = 0u;
            scratch->candidate_high = 0u;
            if (receipt != nullptr)
              receipt->trajectory_abstained = 1u;
          }
        }
      } else {
        successor = select_successor(words, layout, current, &record,
                                     &scratch->candidate_low,
                                     &scratch->candidate_high);
        if (successor < kFormAssemblyCount &&
            decode_form_raw(words, successor).valid) {
          if (!arm_trajectory_after_first_form(words, layout, successor, receipt)) {
            successor = kFormAssemblyCount;
            record = kTransitionRecordCount;
            scratch->candidate_low = 0u;
            scratch->candidate_high = 0u;
          }
        } else {
          successor = kFormAssemblyCount;
          record = kTransitionRecordCount;
          scratch->candidate_low = 0u;
          scratch->candidate_high = 0u;
        }
      }
    } else {
      clear_active_trajectory(words);
      if (receipt != nullptr)
        receipt->trajectory_abstained = 1u;
    }
    scratch->current_form = current;
    scratch->predicted_form = successor;
    scratch->transition_record = record;
    retain_prediction_eligibility(words, current, record,
                                  scratch->candidate_low,
                                  scratch->candidate_high);
    write_resident(words, pair_index(kGlobalBase, kPendingValid, 0u), 1u);
    write_resident(words, pair_index(kGlobalBase, kPendingPredecessor, 0u), current);
    write_resident(words, pair_index(kGlobalBase, kPendingSuccessor, 0u), successor);
    if (successor < kFormAssemblyCount)
      publish_form(words, layout, successor);
    else {
      publish_form(words, layout, kFormAssemblyCount);
      if (receipt != nullptr)
        receipt->abstained = 1u;
    }
    write_resident(words, pair_index(kGlobalBase, kEventId, 0u), scratch->event_id);
    write_resident(words, pair_index(kGlobalBase, kPredictionCount, 0u),
                   read_unsigned(words, pair_index(kGlobalBase, kPredictionCount, 0u)) + 1u);
    if (receipt != nullptr) {
      receipt->step_mode = kStepPredicted;
      receipt->event_id = scratch->event_id;
      receipt->current_form = current;
      receipt->predicted_form = successor;
      receipt->predecessor_form = current;
      receipt->transition_record = record;
    }
    return;
  }

  const RawByteDecode raw = current_observation_raw(words, layout);
  if (raw.valid && raw.value == static_cast<std::uint8_t>('^'))
    write_resident(
        words, pair_index(kGlobalBase, kPreviousObservedTrajectory, 0u),
        kTrajectoryStartSentinel);
  const std::uint32_t observed = find_or_recruit_form(words, raw, receipt);
  scratch->observed_form = observed;
  apply_oldest_prediction_credit(words, observed, receipt);
  const std::uint32_t pending =
      read_unsigned(words, pair_index(kGlobalBase, kPendingValid, 0u));
  std::uint32_t actual_record = kTransitionRecordCount;
  if (pending != 0u) {
    const std::uint32_t predecessor =
        read_unsigned(words, pair_index(kGlobalBase, kPendingPredecessor, 0u));
    const std::uint32_t predicted =
        read_unsigned(words, pair_index(kGlobalBase, kPendingSuccessor, 0u));
    if (predicted != observed)
      clear_active_trajectory(words);
    actual_record = find_transition(words, predecessor, observed);
    if (actual_record == kTransitionRecordCount)
      actual_record = recruit_transition(words, layout, predecessor, observed, receipt);
    if (actual_record < kTransitionRecordCount) {
      const std::uint32_t support =
          read_unsigned(words, transition_meta_index(actual_record, kTransitionSupport, 0u));
      write_resident(words, transition_meta_index(actual_record, kTransitionSupport, 0u),
                     support + 1u);
      update_transition_observation(words, layout, actual_record);
      if (receipt != nullptr)
        ++receipt->support_updates;
    }
    if (predicted < kFormAssemblyCount && predicted != observed) {
      const std::uint32_t wrong = find_transition(words, predecessor, predicted);
      if (wrong < kTransitionRecordCount) {
        const std::uint32_t evidence = read_unsigned(
            words, transition_meta_index(wrong, kTransitionCounterevidence, 0u));
        write_resident(words, transition_meta_index(wrong, kTransitionCounterevidence, 0u),
                       evidence + 1u);
        write_resident(words, pair_index(kGlobalBase, kCounterevidenceCount, 0u),
                       read_unsigned(words, pair_index(kGlobalBase, kCounterevidenceCount, 0u)) +
                           1u);
        if (receipt != nullptr)
          ++receipt->counterevidence_updates;
      }
    }
    observe_trajectory_phase(words, layout, predecessor, observed, receipt);
  }
  write_resident(words, pair_index(kGlobalBase, kPendingValid, 0u), 0u);
  write_resident(words, pair_index(kGlobalBase, kEventId, 0u), scratch->event_id);
  write_resident(words, pair_index(kGlobalBase, kObservationCount, 0u),
                 read_unsigned(words, pair_index(kGlobalBase, kObservationCount, 0u)) + 1u);
  if (receipt != nullptr) {
    receipt->step_mode = kStepObserved;
    receipt->event_id = scratch->event_id;
    receipt->observed_form = observed;
    receipt->transition_record = actual_record;
  }
}

static __global__ void finalize_kernel(SiteWord* words, const DeviceScratch* scratch,
                                       const DeviceInputs* inputs, Receipt* receipt,
                                       std::uint32_t* advanced, std::uint32_t advancement_bit) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr || receipt == nullptr)
    return;
  if (scratch->accepted != 0u)
    write_resident(words, pair_index(kGlobalBase, kJournalCount, 0u),
                   read_unsigned(words, pair_index(kGlobalBase, kJournalCount, 0u)) + 1u);
  if (advanced != nullptr) {
    if (inputs != nullptr && inputs->observe_staged != 0u &&
        scratch->accepted == 0u)
      *advanced = 0u;
    else if (scratch->accepted != 0u)
      *advanced |= advancement_bit;
  }
}

static __global__ void prepare_inverse_kernel(SiteWord* words, InverseScratch* scratch) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr)
    return;
  scratch->valid = 0u;
  scratch->error = 0u;
  const std::uint32_t count = read_unsigned(words, pair_index(kGlobalBase, kJournalCount, 0u));
  if (count == 0u || count > kJournalDepth) {
    scratch->error = 1u;
    return;
  }
  scratch->journal_slot = count - 1u;
  scratch->event_slot = read_unsigned(words, journal_index(scratch->journal_slot, 0u));
  if (scratch->event_slot >= kJournalDepth) {
    scratch->error = 1u;
    return;
  }
  scratch->valid = 1u;
}

static __global__ void restore_inverse_kernel(SiteWord* words, const DeviceLayout& layout,
                                              const InverseScratch* scratch) {
  if (scratch == nullptr || scratch->valid == 0u)
    return;
  const std::uint32_t slot = scratch->journal_slot;
  const std::uint32_t offset = kJournalMetaRailCount;
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = tid; index < kMutableRailCount; index += stride)
    words[fixed_physical_slot(index)] =
        words[fixed_physical_slot(journal_index(slot, offset + index))];
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t motor = offset + kMutableRailCount;
    words[layout.raw_motor_zero_slot] =
        words[fixed_physical_slot(journal_index(slot, motor))];
    words[layout.raw_motor_one_slot] =
        words[fixed_physical_slot(journal_index(slot, motor + 2u))];
  }
}

static __global__ void clear_inverse_journal_kernel(SiteWord* words,
                                                    const InverseScratch* scratch) {
  if (scratch == nullptr || scratch->valid == 0u)
    return;
  const std::uint32_t slot = scratch->journal_slot;
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = tid; index < kJournalRecordRailCount; index += stride)
    words[fixed_physical_slot(journal_index(slot, index))] =
        founder_value(journal_index(slot, index));
}

static __global__ void validate_kernel(const SiteWord* words, ValidationReceipt* receipt) {
  if (receipt == nullptr)
    return;
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    receipt->marker_valid =
        read_unsigned(words, pair_index(kGlobalBase, kFactorMarker, 0u)) == kFactorMarkerValue;
    receipt->version_valid =
        read_unsigned(words, pair_index(kGlobalBase, kLayoutVersion, 0u)) == kLayoutVersionValue;
    const std::uint32_t journal =
        read_unsigned(words, pair_index(kGlobalBase, kJournalCount, 0u));
    receipt->journal_valid = journal <= kJournalDepth;
    receipt->form_valid = 1u;
    receipt->transition_valid = 1u;
    receipt->trajectory_valid =
        read_unsigned(words, pair_index(kGlobalBase, kTrajectoryCount, 0u)) <=
        kTrajectoryAssemblyCount;
    const std::uint32_t previous = read_unsigned(
        words, pair_index(kGlobalBase, kPreviousTrajectory, 0u));
    if (previous > kTrajectoryNoneSentinel)
      receipt->trajectory_valid = 0u;
    const std::uint32_t observed_previous = read_unsigned(
        words,
        pair_index(kGlobalBase, kPreviousObservedTrajectory, 0u));
    if (observed_previous > kTrajectoryNoneSentinel)
      receipt->trajectory_valid = 0u;
    receipt->acquisition_valid = 1u;
    receipt->eligibility_valid = 1u;
    for (std::uint32_t form = 0u; form < kFormAssemblyCount; ++form) {
      if (read_unsigned(words, form_active_index(form, 0u)) > 1u) {
        receipt->form_valid = 0u;
        break;
      }
    }
    for (std::uint32_t record = 0u; record < kTransitionRecordCount; ++record) {
      const std::uint32_t active =
          read_unsigned(words, transition_meta_index(record, kTransitionActive, 0u));
      if (active > 1u ||
          (active != 0u &&
           (read_unsigned(words, transition_meta_index(record, kTransitionPredecessor, 0u)) >=
                kFormAssemblyCount ||
            read_unsigned(words, transition_meta_index(record, kTransitionSuccessor, 0u)) >=
                kFormAssemblyCount))) {
        receipt->transition_valid = 0u;
        break;
      }
    }
    for (std::uint32_t trajectory = 0u;
         trajectory < kTrajectoryAssemblyCount; ++trajectory) {
      const std::uint32_t active = read_unsigned(
          words, trajectory_meta_index(trajectory, kTrajectoryActive, 0u));
      const std::uint32_t length = read_unsigned(
          words, trajectory_meta_index(trajectory, kTrajectoryLength, 0u));
      const std::uint32_t admitted = read_unsigned(
          words,
          trajectory_meta_index(trajectory, kTrajectoryAdmittedCount, 0u));
      std::uint32_t represented_admitted = 0u;
      for (std::uint32_t channel = 0u; channel < kSituationChannelCount;
           ++channel) {
        const std::uint32_t value = read_unsigned(
            words, trajectory_admit_index(trajectory, channel, 0u));
        if (value > 1u)
          receipt->trajectory_valid = 0u;
        represented_admitted += value != 0u ? 1u : 0u;
      }
      if (active > 1u ||
          (active != 0u &&
           (length == 0u || length > kTrajectoryPhaseCount ||
            admitted != represented_admitted))) {
        receipt->trajectory_valid = 0u;
        continue;
      }
      if (active == 0u)
        continue;
      for (std::uint32_t phase = 0u; phase < length; ++phase) {
        const std::uint32_t form = read_unsigned(
            words, trajectory_phase_index(trajectory, phase, 0u));
        if (!decode_form_raw(words, form).valid) {
          receipt->trajectory_valid = 0u;
          break;
        }
      }
    }
    for (std::uint32_t predecessor = 0u;
         predecessor < kTrajectoryPredecessorCount; ++predecessor) {
      for (std::uint32_t successor = 0u;
           successor < kTrajectoryAssemblyCount; ++successor) {
        const std::uint32_t support = trajectory_adjacency_support(
            words, predecessor, successor);
        if (support == 0u)
          continue;
        const bool predecessor_valid =
            predecessor == kTrajectoryStartSentinel ||
            read_unsigned(words, trajectory_meta_index(
                                      predecessor, kTrajectoryActive, 0u)) != 0u;
        const bool successor_valid =
            read_unsigned(words, trajectory_meta_index(
                                      successor, kTrajectoryActive, 0u)) != 0u;
        if (!predecessor_valid || !successor_valid) {
          receipt->trajectory_valid = 0u;
          break;
        }
      }
    }
    const std::uint32_t active_trajectory = read_unsigned(
        words, pair_index(kGlobalBase, kActiveTrajectory, 0u));
    const std::uint32_t active_phase = read_unsigned(
        words, pair_index(kGlobalBase, kActiveTrajectoryPhase, 0u));
    if (active_trajectory > kTrajectoryAssemblyCount) {
      receipt->trajectory_valid = 0u;
    } else if (active_trajectory < kTrajectoryAssemblyCount) {
      const std::uint32_t active = read_unsigned(
          words,
          trajectory_meta_index(active_trajectory, kTrajectoryActive, 0u));
      const std::uint32_t length = read_unsigned(
          words,
          trajectory_meta_index(active_trajectory, kTrajectoryLength, 0u));
      if (active == 0u || active_phase == 0u || active_phase >= length)
        receipt->trajectory_valid = 0u;
    } else if (active_phase != 0u) {
      receipt->trajectory_valid = 0u;
    }
    const std::uint32_t acquisition_length = read_unsigned(
        words, pair_index(kGlobalBase, kAcquisitionLength, 0u));
    if (acquisition_length > kTrajectoryPhaseCount) {
      receipt->acquisition_valid = 0u;
    } else {
      for (std::uint32_t phase = 0u; phase < acquisition_length; ++phase) {
        const std::uint32_t form =
            read_unsigned(words, acquisition_phase_index(phase, 0u));
        if (!decode_form_raw(words, form).valid) {
          receipt->acquisition_valid = 0u;
          break;
        }
      }
    }
    if (receipt->trajectory_valid == 0u || receipt->acquisition_valid == 0u)
      receipt->form_valid = 0u;
    const std::uint32_t prediction_count =
        read_unsigned(words, pair_index(kGlobalBase, kPredictionId, 0u));
    for (std::uint32_t slot = 0u; slot < kEligibilityDepth; ++slot) {
      const std::uint32_t event =
          read_unsigned(words, eligibility_index(slot, kEligibilityEvent, 0u));
      const std::uint32_t status =
          read_unsigned(words, eligibility_index(slot, kEligibilityStatus, 0u));
      const std::uint32_t age =
          read_unsigned(words, eligibility_index(slot, kEligibilityAge, 0u));
      const std::uint32_t selected = read_unsigned(
          words, eligibility_index(slot, kEligibilitySelectedRecord, 0u));
      if (event > prediction_count || status > 2u || age > kEligibilityDepth ||
          selected > kTransitionRecordCount) {
        receipt->eligibility_valid = 0u;
        ++receipt->invalid_eligibility;
      }
    }
  }
  const std::uint32_t journal =
      read_unsigned(words, pair_index(kGlobalBase, kJournalCount, 0u));
  const std::uint32_t represented = journal < kJournalDepth ? journal : kJournalDepth;
  const std::uint32_t limit = kMutableRailCount + represented * kJournalRecordRailCount;
  for (std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < limit / 2u;
       pair += blockDim.x * gridDim.x) {
    const SiteWord value = words[fixed_physical_slot(pair * 2u)];
    const SiteWord complement = words[fixed_physical_slot(pair * 2u + 1u)];
    if (complement != ~value)
      atomicAdd(&receipt->invalid_pairs, 1u);
  }
  for (std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < represented;
       slot += blockDim.x * gridDim.x) {
    if (read_unsigned(words, journal_index(slot, 0u)) >= kJournalDepth)
      atomicAdd(&receipt->invalid_journal_events, 1u);
  }
}

static __global__ void lesion_transition_kernel(SiteWord* words,
                                                std::uint32_t record,
                                                LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr)
    return;
  *receipt = LesionReceipt{};
  if (record >= kTransitionRecordCount)
    return;
  const std::uint32_t active_index =
      transition_meta_index(record, kTransitionActive, 0u);
  const SiteWord before = read_resident(words, active_index);
  const SiteWord before_complement = read_resident(words, active_index + 1u);
  if (before == 0u)
    return;
  receipt->record = record;
  receipt->prior_active = before;
  receipt->matter_before = static_cast<std::uint32_t>(
      __popc(before) + __popc(before_complement));
  write_resident(words, active_index, 0u);
  const SiteWord after = read_resident(words, active_index);
  const SiteWord after_complement = read_resident(words, active_index + 1u);
  receipt->changed_bits = static_cast<std::uint32_t>(
      __popc(before ^ after) + __popc(before_complement ^ after_complement));
  receipt->matter_after = static_cast<std::uint32_t>(
      __popc(after) + __popc(after_complement));
  receipt->valid = 1u;
}

static __global__ void restore_transition_lesion_kernel(
    SiteWord* words, const LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr ||
      receipt->valid == 0u || receipt->record >= kTransitionRecordCount)
    return;
  write_resident(words,
                 transition_meta_index(receipt->record, kTransitionActive, 0u),
                 receipt->prior_active);
}

static __global__ void lesion_trajectory_adjacency_kernel(
    SiteWord* words, std::uint32_t predecessor, std::uint32_t successor,
    TrajectoryAdjacencyLesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr)
    return;
  *receipt = TrajectoryAdjacencyLesionReceipt{};
  if (predecessor >= kTrajectoryPredecessorCount ||
      successor >= kTrajectoryAssemblyCount)
    return;
  const std::uint32_t index =
      trajectory_adjacency_index(predecessor, successor, 0u);
  const SiteWord before = read_resident(words, index);
  const SiteWord before_complement = read_resident(words, index + 1u);
  if (before == 0u)
    return;
  receipt->predecessor = predecessor;
  receipt->successor = successor;
  receipt->prior_support = before;
  receipt->matter_before = static_cast<std::uint32_t>(
      __popc(before) + __popc(before_complement));
  write_resident(words, index, 0u);
  const SiteWord after = read_resident(words, index);
  const SiteWord after_complement = read_resident(words, index + 1u);
  receipt->changed_bits = static_cast<std::uint32_t>(
      __popc(before ^ after) + __popc(before_complement ^ after_complement));
  receipt->matter_after = static_cast<std::uint32_t>(
      __popc(after) + __popc(after_complement));
  receipt->valid = 1u;
}

static __global__ void find_remote_trajectory_adjacency_kernel(
    const SiteWord* words, std::uint32_t excluded_mask,
    std::uint32_t* predecessor, std::uint32_t* successor) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || predecessor == nullptr ||
      successor == nullptr)
    return;
  *predecessor = kTrajectoryPredecessorCount;
  *successor = kTrajectoryAssemblyCount;
  for (std::uint32_t left = 0u; left < kTrajectoryAssemblyCount; ++left) {
    if ((excluded_mask & (1u << left)) != 0u)
      continue;
    for (std::uint32_t right = 0u; right < kTrajectoryAssemblyCount; ++right) {
      if ((excluded_mask & (1u << right)) == 0u &&
          trajectory_adjacency_support(words, left, right) != 0u) {
        *predecessor = left;
        *successor = right;
        return;
      }
    }
  }
}

static_assert(kTrajectoryAssemblyCount * kTrajectoryAssemblyCount <= 64u,
             "edge-exclusive mask must address every (left, right) pair "
             "as a single bit in a 64-bit word");

// find_remote_trajectory_adjacency_kernel (above) is NODE-exclusive: a
// candidate edge is rejected whenever EITHER endpoint trajectory appears
// anywhere in the baseline walk, even if that specific edge was never
// traversed. That is too coarse whenever two candidate edges share a
// decision point -- e.g. predecessor node P with learned edges P->A and
// P->B, where the baseline only ever traversed P->A. A node-exclusive
// predicate excludes P (because it appears in the baseline) and so it can
// never find the sibling edge P->B, even though P->B is exactly the
// matched sham a lesion/dissociation control needs: same predecessor, same
// decision point, different (untraversed) successor. This kernel is
// EDGE-exclusive instead: excluded_edge_mask carries one bit per directed
// edge (left, right) -- bit (left * kTrajectoryAssemblyCount + right) --
// set only for edges actually traversed in the baseline, so an untraversed
// sibling edge at a shared decision point remains eligible.
static __global__ void find_remote_trajectory_adjacency_edge_exclusive_kernel(
    const SiteWord* words, std::uint64_t excluded_edge_mask,
    std::uint32_t* predecessor, std::uint32_t* successor) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || predecessor == nullptr ||
      successor == nullptr)
    return;
  *predecessor = kTrajectoryPredecessorCount;
  *successor = kTrajectoryAssemblyCount;
  for (std::uint32_t left = 0u; left < kTrajectoryAssemblyCount; ++left) {
    for (std::uint32_t right = 0u; right < kTrajectoryAssemblyCount; ++right) {
      const std::uint64_t edge_bit =
          1ull << (left * kTrajectoryAssemblyCount + right);
      if ((excluded_edge_mask & edge_bit) == 0u &&
          trajectory_adjacency_support(words, left, right) != 0u) {
        *predecessor = left;
        *successor = right;
        return;
      }
    }
  }
}

static __global__ void lesion_trajectory_kernel(
    SiteWord* words, std::uint32_t trajectory,
    TrajectoryLesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr)
    return;
  *receipt = TrajectoryLesionReceipt{};
  if (trajectory >= kTrajectoryAssemblyCount)
    return;
  const std::uint32_t active_index =
      trajectory_meta_index(trajectory, kTrajectoryActive, 0u);
  const SiteWord before = read_resident(words, active_index);
  const SiteWord before_complement = read_resident(words, active_index + 1u);
  if (before == 0u)
    return;
  receipt->trajectory = trajectory;
  receipt->prior_active = before;
  receipt->matter_before = static_cast<std::uint32_t>(
      __popc(before) + __popc(before_complement));
  write_resident(words, active_index, 0u);
  const SiteWord after = read_resident(words, active_index);
  const SiteWord after_complement = read_resident(words, active_index + 1u);
  receipt->changed_bits = static_cast<std::uint32_t>(
      __popc(before ^ after) + __popc(before_complement ^ after_complement));
  receipt->matter_after = static_cast<std::uint32_t>(
      __popc(after) + __popc(after_complement));
  receipt->valid = 1u;
}

static __global__ void restore_trajectory_lesion_kernel(
    SiteWord* words, const TrajectoryLesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr ||
      receipt->valid == 0u ||
      receipt->trajectory >= kTrajectoryAssemblyCount)
    return;
  write_resident(
      words,
      trajectory_meta_index(receipt->trajectory, kTrajectoryActive, 0u),
      receipt->prior_active);
}

static __global__ void find_remote_trajectory_kernel(
    const SiteWord* words, std::uint32_t excluded_mask,
    std::uint32_t* trajectory) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || trajectory == nullptr)
    return;
  *trajectory = kTrajectoryAssemblyCount;
  for (std::uint32_t candidate = 0u;
       candidate < kTrajectoryAssemblyCount; ++candidate) {
    if ((excluded_mask & (1u << candidate)) == 0u &&
        read_unsigned(
            words,
            trajectory_meta_index(candidate, kTrajectoryActive, 0u)) != 0u) {
      *trajectory = candidate;
      return;
    }
  }
}

// 0X1-283: the thirteen scalar (<<<1,1>>>) launches below each ran as an
// independent host-side cudaLaunchKernel call, paying full launch-configuration
// overhead every invocation even though grid/block shape, argument count and
// argument types never change between calls at the same site. Each site now
// caches a single-node CUDA graph (created once, params refreshed in place on
// every later call via cudaGraphExecKernelNodeSetParams) and replays it with
// cudaGraphLaunch instead of re-issuing <<<1,1>>> -- same kernel, same
// grid/block, same synchronous default-stream semantics, fewer host-side
// launch calls. Mirrors the graph-dispatch idiom already landed in
// bcc32_resident_readout_plastic_owner.cuh (commit 3bb87f725c).
namespace detail {

struct KernelGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaGraphNode_t node = nullptr;

  void destroy() noexcept {
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
      executable = nullptr;
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
      graph = nullptr;
    }
    node = nullptr;
  }

  ~KernelGraph() { destroy(); }
};

template <typename T>
inline void* kernel_argument(T& value) {
  return const_cast<void*>(static_cast<const void*>(&value));
}

inline cudaError_t launch_graph(KernelGraph& graph, void* function,
                                void** arguments) {
  cudaKernelNodeParams params{};
  params.func = function;
  params.gridDim = dim3{1u, 1u, 1u};
  params.blockDim = dim3{1u, 1u, 1u};
  params.sharedMemBytes = 0u;
  params.kernelParams = arguments;
  params.extra = nullptr;
  cudaError_t status;
  if (graph.executable == nullptr) {
    status = cudaGraphCreate(&graph.graph, 0u);
    if (status != cudaSuccess)
      return status;
    status = cudaGraphAddKernelNode(&graph.node, graph.graph, nullptr, 0u,
                                    &params);
    if (status != cudaSuccess)
      return status;
    status = cudaGraphInstantiate(&graph.executable, graph.graph, nullptr,
                                  nullptr, 0u);
    if (status != cudaSuccess)
      return status;
  } else {
    status = cudaGraphExecKernelNodeSetParams(graph.executable, graph.node,
                                              &params);
    if (status != cudaSuccess)
      return status;
  }
  return cudaGraphLaunch(graph.executable, 0);
}

}  // namespace detail

inline void launch_step(SiteWord* words, const DeviceLayout& layout, DeviceInputs* inputs,
                        DeviceScratch* scratch, Receipt* receipt, std::uint32_t* advanced,
                        std::uint32_t advancement_bit) {
  static detail::KernelGraph preflight_graph;
  static detail::KernelGraph reserve_graph;
  static detail::KernelGraph mutate_graph;
  static detail::KernelGraph finalize_graph;
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(inputs),
                    detail::kernel_argument(advanced)};
    (void)detail::launch_graph(preflight_graph,
                               reinterpret_cast<void*>(preflight_kernel), args);
  }
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(layout),
                    detail::kernel_argument(inputs), detail::kernel_argument(scratch),
                    detail::kernel_argument(receipt), detail::kernel_argument(advanced)};
    (void)detail::launch_graph(reserve_graph,
                               reinterpret_cast<void*>(reserve_kernel), args);
  }
  snapshot_kernel<<<64, 256>>>(words, layout, scratch);
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(layout),
                    detail::kernel_argument(inputs), detail::kernel_argument(scratch),
                    detail::kernel_argument(receipt)};
    (void)detail::launch_graph(mutate_graph,
                               reinterpret_cast<void*>(mutate_kernel), args);
  }
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(scratch),
                    detail::kernel_argument(inputs), detail::kernel_argument(receipt),
                    detail::kernel_argument(advanced), detail::kernel_argument(advancement_bit)};
    (void)detail::launch_graph(finalize_graph,
                               reinterpret_cast<void*>(finalize_kernel), args);
  }
}

inline void launch_prepare_inverse(SiteWord* words, InverseScratch* scratch) {
  static detail::KernelGraph prepare_inverse_graph;
  void* args[] = {detail::kernel_argument(words), detail::kernel_argument(scratch)};
  (void)detail::launch_graph(prepare_inverse_graph,
                             reinterpret_cast<void*>(prepare_inverse_kernel), args);
}

inline void launch_restore_inverse(SiteWord* words, const DeviceLayout& layout,
                                   const InverseScratch* scratch) {
  restore_inverse_kernel<<<64, 256>>>(words, layout, scratch);
  clear_inverse_journal_kernel<<<64, 256>>>(words, scratch);
}

inline void launch_validate(const SiteWord* words, ValidationReceipt* receipt) {
  validate_kernel<<<64, 256>>>(words, receipt);
}

inline LesionReceipt lesion_transition(SiteWord* words, std::uint32_t record) {
  LesionReceipt* device_receipt = nullptr;
  cudaError_t status = cudaMalloc(&device_receipt, sizeof(*device_receipt));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("allocate sparse lesion receipt: ") +
                             cudaGetErrorString(status));
  static detail::KernelGraph lesion_transition_graph;
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(record),
                    detail::kernel_argument(device_receipt)};
    status = detail::launch_graph(lesion_transition_graph,
                                  reinterpret_cast<void*>(lesion_transition_kernel), args);
  }
  if (status == cudaSuccess) {
    LesionReceipt host{};
    status = cudaMemcpy(&host, device_receipt, sizeof(host),
                        cudaMemcpyDeviceToHost);
    (void)cudaFree(device_receipt);
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("copy sparse lesion receipt: ") +
                               cudaGetErrorString(status));
    return host;
  }
  (void)cudaFree(device_receipt);
  throw std::runtime_error(std::string("launch sparse transition lesion: ") +
                           cudaGetErrorString(status));
}

inline void restore_transition_lesion(SiteWord* words,
                                      const LesionReceipt& receipt) {
  LesionReceipt* device_receipt = nullptr;
  cudaError_t status = cudaMalloc(&device_receipt, sizeof(*device_receipt));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("allocate sparse lesion restore: ") +
                             cudaGetErrorString(status));
  status = cudaMemcpy(device_receipt, &receipt, sizeof(receipt),
                      cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    static detail::KernelGraph restore_transition_lesion_graph;
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(device_receipt)};
    status = detail::launch_graph(
        restore_transition_lesion_graph,
        reinterpret_cast<void*>(restore_transition_lesion_kernel), args);
  }
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  (void)cudaFree(device_receipt);
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("restore sparse transition lesion: ") +
                             cudaGetErrorString(status));
}

inline TrajectoryAdjacencyLesionReceipt lesion_trajectory_adjacency(
    SiteWord* words, std::uint32_t predecessor, std::uint32_t successor) {
  TrajectoryAdjacencyLesionReceipt* device_receipt = nullptr;
  cudaError_t status = cudaMalloc(&device_receipt, sizeof(*device_receipt));
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("allocate trajectory adjacency lesion receipt: ") +
        cudaGetErrorString(status));
  static detail::KernelGraph lesion_trajectory_adjacency_graph;
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(predecessor),
                    detail::kernel_argument(successor), detail::kernel_argument(device_receipt)};
    status = detail::launch_graph(
        lesion_trajectory_adjacency_graph,
        reinterpret_cast<void*>(lesion_trajectory_adjacency_kernel), args);
  }
  TrajectoryAdjacencyLesionReceipt host{};
  if (status == cudaSuccess)
    status = cudaMemcpy(&host, device_receipt, sizeof(host),
                        cudaMemcpyDeviceToHost);
  (void)cudaFree(device_receipt);
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("lesion trajectory adjacency: ") +
        cudaGetErrorString(status));
  return host;
}

inline bool find_remote_trajectory_adjacency(
    const SiteWord* words, std::uint32_t excluded_mask,
    std::uint32_t* predecessor, std::uint32_t* successor) {
  std::uint32_t* device_pair = nullptr;
  cudaError_t status = cudaMalloc(&device_pair, 2u * sizeof(*device_pair));
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("allocate remote adjacency witness: ") +
        cudaGetErrorString(status));
  static detail::KernelGraph find_remote_trajectory_adjacency_graph;
  {
    std::uint32_t* successor_slot = device_pair + 1u;
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(excluded_mask),
                    detail::kernel_argument(device_pair), detail::kernel_argument(successor_slot)};
    status = detail::launch_graph(
        find_remote_trajectory_adjacency_graph,
        reinterpret_cast<void*>(find_remote_trajectory_adjacency_kernel), args);
  }
  std::uint32_t host[2]{kTrajectoryPredecessorCount,
                        kTrajectoryAssemblyCount};
  if (status == cudaSuccess)
    status = cudaMemcpy(host, device_pair, sizeof(host), cudaMemcpyDeviceToHost);
  (void)cudaFree(device_pair);
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("find remote trajectory adjacency: ") +
        cudaGetErrorString(status));
  if (predecessor != nullptr)
    *predecessor = host[0];
  if (successor != nullptr)
    *successor = host[1];
  return host[0] < kTrajectoryAssemblyCount &&
         host[1] < kTrajectoryAssemblyCount;
}

// Edge-exclusive counterpart of find_remote_trajectory_adjacency (see the
// kernel comment above for why node-exclusive is too coarse for a matched
// sham at a shared decision point). excluded_edge_mask bit
// (predecessor * kTrajectoryAssemblyCount + successor) means that directed
// edge was traversed in the baseline and must not be selected as the
// "remote" control edge.
inline bool find_remote_trajectory_adjacency_edge_exclusive(
    const SiteWord* words, std::uint64_t excluded_edge_mask,
    std::uint32_t* predecessor, std::uint32_t* successor) {
  std::uint32_t* device_pair = nullptr;
  cudaError_t status = cudaMalloc(&device_pair, 2u * sizeof(*device_pair));
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("allocate remote adjacency edge-exclusive witness: ") +
        cudaGetErrorString(status));
  static detail::KernelGraph find_remote_trajectory_adjacency_edge_exclusive_graph;
  {
    std::uint32_t* successor_slot = device_pair + 1u;
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(excluded_edge_mask),
                    detail::kernel_argument(device_pair), detail::kernel_argument(successor_slot)};
    status = detail::launch_graph(
        find_remote_trajectory_adjacency_edge_exclusive_graph,
        reinterpret_cast<void*>(find_remote_trajectory_adjacency_edge_exclusive_kernel), args);
  }
  std::uint32_t host[2]{kTrajectoryPredecessorCount,
                        kTrajectoryAssemblyCount};
  if (status == cudaSuccess)
    status = cudaMemcpy(host, device_pair, sizeof(host), cudaMemcpyDeviceToHost);
  (void)cudaFree(device_pair);
  if (status != cudaSuccess)
    throw std::runtime_error(
        std::string("find remote trajectory adjacency edge-exclusive: ") +
        cudaGetErrorString(status));
  if (predecessor != nullptr)
    *predecessor = host[0];
  if (successor != nullptr)
    *successor = host[1];
  return host[0] < kTrajectoryAssemblyCount &&
         host[1] < kTrajectoryAssemblyCount;
}

inline TrajectoryLesionReceipt lesion_trajectory(
    SiteWord* words, std::uint32_t trajectory) {
  TrajectoryLesionReceipt* device_receipt = nullptr;
  cudaError_t status = cudaMalloc(&device_receipt, sizeof(*device_receipt));
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("allocate trajectory lesion receipt: ") +
                             cudaGetErrorString(status));
  static detail::KernelGraph lesion_trajectory_graph;
  {
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(trajectory),
                    detail::kernel_argument(device_receipt)};
    status = detail::launch_graph(lesion_trajectory_graph,
                                  reinterpret_cast<void*>(lesion_trajectory_kernel), args);
  }
  TrajectoryLesionReceipt host{};
  if (status == cudaSuccess)
    status = cudaMemcpy(&host, device_receipt, sizeof(host),
                        cudaMemcpyDeviceToHost);
  (void)cudaFree(device_receipt);
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("lesion resident trajectory: ") +
                             cudaGetErrorString(status));
  return host;
}

inline void restore_trajectory_lesion(
    SiteWord* words, const TrajectoryLesionReceipt& receipt) {
  TrajectoryLesionReceipt* device_receipt = nullptr;
  cudaError_t status = cudaMalloc(&device_receipt, sizeof(*device_receipt));
  if (status == cudaSuccess)
    status = cudaMemcpy(device_receipt, &receipt, sizeof(receipt),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    static detail::KernelGraph restore_trajectory_lesion_graph;
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(device_receipt)};
    status = detail::launch_graph(
        restore_trajectory_lesion_graph,
        reinterpret_cast<void*>(restore_trajectory_lesion_kernel), args);
  }
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  (void)cudaFree(device_receipt);
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("restore resident trajectory lesion: ") +
                             cudaGetErrorString(status));
}

inline std::uint32_t find_remote_trajectory(const SiteWord* words,
                                            std::uint32_t excluded_mask) {
  std::uint32_t* device = nullptr;
  cudaError_t status = cudaMalloc(&device, sizeof(*device));
  if (status == cudaSuccess) {
    static detail::KernelGraph find_remote_trajectory_graph;
    void* args[] = {detail::kernel_argument(words), detail::kernel_argument(excluded_mask),
                    detail::kernel_argument(device)};
    status = detail::launch_graph(
        find_remote_trajectory_graph,
        reinterpret_cast<void*>(find_remote_trajectory_kernel), args);
  }
  std::uint32_t host = kTrajectoryAssemblyCount;
  if (status == cudaSuccess)
    status = cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost);
  (void)cudaFree(device);
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("find remote resident trajectory: ") +
                             cudaGetErrorString(status));
  return host;
}

inline ValidationReceipt validate_resident(const SiteWord* words) {
  auto require = [](cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  };
  ValidationReceipt* device = nullptr;
  require(cudaMalloc(&device, sizeof(*device)), "allocate sparse event validation");
  require(cudaMemset(device, 0, sizeof(*device)), "clear sparse event validation");
  launch_validate(words, device);
  require(cudaGetLastError(), "launch sparse event validation");
  ValidationReceipt host{};
  require(cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost),
          "copy sparse event validation");
  (void)cudaFree(device);
  return host;
}

}  // namespace substrate::bcc32::grown_sparse_event_memory
