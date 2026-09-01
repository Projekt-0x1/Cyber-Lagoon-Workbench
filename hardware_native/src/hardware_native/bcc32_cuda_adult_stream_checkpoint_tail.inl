#if defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)
__global__ void initialize_query_answer_state_kernel(
    discourse_plan::ResidentDiscoursePlanState* plan,
    proposition_chain::OrderedSettlementResult* settlement,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (plan != nullptr) *plan = discourse_plan::ResidentDiscoursePlanState{};
  if (settlement != nullptr)
    *settlement = proposition_chain::OrderedSettlementResult{};
  if (receipt != nullptr) *receipt = QueryAnswerReceipt{};
}

#endif

struct StreamCheckpointHeader {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t drive_bytes;
  std::uint32_t appraisal_bytes;
  std::uint32_t chunk_capacity;
  std::uint32_t emission_capacity;
  std::uint32_t appraisal_holdout_bytes;
  std::uint64_t adult_checkpoint_bytes;
};

// V4 persists only settled contact-action-later-contact evidence. An active
// trace is intentionally excluded: after restore there is no physical basis
// to claim that an interrupted action received a later external contact.
struct StreamCheckpointExtensionV4 {
  std::uint32_t action_transition_bytes;
  std::uint32_t action_transition_scalars_bytes;
  std::uint64_t action_transition_hash;
  std::uint64_t action_transition_scalars_hash;
  std::uint64_t header_hash;
  std::uint64_t drive_hash;
  std::uint64_t appraisal_hash;
};

// V5 carried a complete AdultState checkpoint plus plan-local transient state.
// The current stream has no authority to replay that obsolete plan state, but
// it can migrate the authenticated adult matter and initialize its own empty
// query transport.
struct StreamCheckpointPolicyMetadataV5 {
  std::uint32_t plan_bytes;
  std::uint32_t policy_flags;
};

struct StreamCheckpointExtensionV5 {
  StreamCheckpointPolicyMetadataV5 policy;
  std::uint32_t eligibility_bytes;
  std::uint32_t reserved;
  std::uint64_t policy_hash;
  std::uint64_t plan_hash;
  std::uint64_t eligibility_hash;
  std::uint64_t header_hash;
  std::uint64_t drive_hash;
  std::uint64_t appraisal_hash;
};

struct StreamCheckpointExtensionV6 {
  std::uint32_t action_transition_bytes;
  std::uint32_t action_transition_scalars_bytes;
  std::uint64_t conditioned_matter_bytes;
  std::uint64_t action_transition_hash;
  std::uint64_t action_transition_scalars_hash;
  std::uint64_t conditioned_matter_hash;
  std::uint64_t conditioned_physical_hash;
  std::uint64_t header_hash;
  std::uint64_t drive_hash;
  std::uint64_t appraisal_hash;
};

struct StreamCheckpointExtensionV7 {
  std::uint32_t action_transition_bytes;
  std::uint32_t action_transition_scalars_bytes;
  std::uint64_t conditioned_matter_bytes;
  std::uint64_t conditioned_device_owner_bytes;
  std::uint64_t action_transition_hash;
  std::uint64_t action_transition_scalars_hash;
  std::uint64_t conditioned_matter_hash;
  std::uint64_t conditioned_device_owner_hash;
  std::uint64_t conditioned_physical_hash;
  std::uint64_t conditioned_device_physical_hash;
  std::uint64_t header_hash;
  std::uint64_t drive_hash;
  std::uint64_t appraisal_hash;
};

inline std::vector<std::uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open stream checkpoint part: " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  if (size < 0) throw std::runtime_error("cannot size stream checkpoint part: " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  if (!input) throw std::runtime_error("cannot read stream checkpoint part: " + path);
  return bytes;
}

inline void write_binary_file(const std::string& path,
                              const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create stream checkpoint part: " + path);
  if (!bytes.empty()) {
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  if (!output) throw std::runtime_error("cannot write stream checkpoint part: " + path);
}

inline std::string base_checkpoint_part(const std::string& path) {
  return path + ".bcc32-adult-v1-part";
}

#if !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)

struct CompleteAdultCheckpointHeader {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t unit_words;
  std::uint32_t mass_budget;
  std::uint32_t unit_occurrences;
  std::uint32_t unit_count;
  std::uint32_t unit_capacity;
  std::uint32_t bigram_count;
  std::uint32_t trigram_count;
  std::uint32_t online_bigram_count;
  std::uint32_t online_trigram_count;
  std::uint32_t online_association_count;
  std::uint32_t online_episode_count;
  std::uint32_t online_episode_break_count;
  std::uint32_t transitions_lesioned;
};

inline void save_complete_adult_checkpoint(const adult::AdultState& state,
                                           const std::string& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create complete adult checkpoint: " + path);
  const CompleteAdultCheckpointHeader header{
      0x3154504d43414342ull, 1u, adult::kUnitWords, adult::kResidentMassBudget,
      state.unit_occurrences, state.unit_count, state.unit_capacity,
      state.bigram_count, state.trigram_count,
      state.online_bigram_count, state.online_trigram_count,
      state.online_association_count, state.online_episode_count,
      state.online_episode_break_count, state.transitions_lesioned ? 1u : 0u};
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  adult::checkpoint_write_device(output, state.boundary_mask.get(), 256u);
  adult::checkpoint_write_device(output, state.boundary_bytes.get(), adult::kBoundaryCount);
  adult::checkpoint_write_device(output, state.boundary_histogram.get(), 256u);
  adult::checkpoint_write_device(output, state.boundary_pairs.get(), 256u * 256u);
  adult::checkpoint_write_device(output, state.unit_lengths.get(), state.unit_count);
  adult::checkpoint_write_device(output, state.unit_content.get(),
      static_cast<std::size_t>(state.unit_count) * adult::kUnitWords);
  adult::checkpoint_write_device(output, state.unit_vitality.get(), state.unit_count);
  adult::checkpoint_write_device(output, state.unigram_top_ids.get(), adult::kUnigramTop);
  adult::checkpoint_write_device(output, state.bigrams.get(), state.bigram_count);
  adult::checkpoint_write_device(output, state.bigram_counts.get(), state.bigram_count);
  adult::checkpoint_write_device(output, state.trigrams.get(), state.trigram_count);
  adult::checkpoint_write_device(output, state.trigram_counts.get(), state.trigram_count);
  adult::checkpoint_write_device(output, state.online_bigrams.get(), state.online_bigram_count);
  adult::checkpoint_write_device(output, state.online_bigram_counts.get(),
                                  state.online_bigram_count);
  adult::checkpoint_write_device(output, state.online_trigrams.get(),
                                  state.online_trigram_count);
  adult::checkpoint_write_device(output, state.online_trigram_counts.get(),
                                  state.online_trigram_count);
  adult::checkpoint_write_device(output, state.online_associations.get(),
                                  state.online_association_count);
  adult::checkpoint_write_device(output, state.online_association_counts.get(),
                                  state.online_association_count);
  adult::checkpoint_write_device(output, state.online_episode_units.get(),
                                  state.online_episode_count);
  adult::checkpoint_write_device(output, state.online_episode_breaks.get(),
                                  state.online_episode_break_count);
  adult::checkpoint_write_device(output, state.mutable_sizes.get(), 6u);
  adult::checkpoint_write_device(output, state.ledger.get(), 4u);
  adult::checkpoint_write_device(output, state.rng.get(), 1u);
}

inline adult::AdultState load_complete_adult_checkpoint(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open complete adult checkpoint: " + path);
  CompleteAdultCheckpointHeader header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input || header.magic != 0x3154504d43414342ull || header.version != 1u ||
      header.unit_words != adult::kUnitWords ||
      header.mass_budget != adult::kResidentMassBudget) {
    throw std::runtime_error("incompatible complete adult stream checkpoint");
  }
  if (header.unit_capacity < header.unit_count ||
      header.unit_capacity - header.unit_count > adult::kOnlineUnitReserve ||
      header.online_bigram_count > adult::kOnlineNgramCapacity ||
      header.online_trigram_count > adult::kOnlineNgramCapacity ||
      header.online_association_count > adult::kOnlineAssociationCapacity ||
      header.online_episode_count > adult::kOnlineEpisodeCapacity ||
      header.online_episode_break_count > adult::kOnlineEpisodeBreakCapacity) {
    throw std::runtime_error("complete adult checkpoint extents exceed capacities");
  }

  adult::AdultState state;
  state.unit_occurrences = header.unit_occurrences;
  state.unit_count = header.unit_count;
  state.bigram_count = header.bigram_count;
  state.trigram_count = header.trigram_count;
  state.online_bigram_count = header.online_bigram_count;
  state.online_trigram_count = header.online_trigram_count;
  state.online_association_count = header.online_association_count;
  state.online_episode_count = header.online_episode_count;
  state.online_episode_break_count = header.online_episode_break_count;
  state.transitions_lesioned = header.transitions_lesioned != 0u;
  state.unit_capacity = header.unit_capacity;

  state.boundary_mask.allocate(256u);
  state.boundary_bytes.allocate(adult::kBoundaryCount);
  state.boundary_histogram.allocate(256u);
  state.boundary_pairs.allocate(256u * 256u);
  state.unit_lengths.allocate(state.unit_capacity);
  state.unit_content.allocate(static_cast<std::size_t>(state.unit_capacity) *
                              adult::kUnitWords);
  state.unit_vitality.allocate(state.unit_capacity);
  state.unigram_top_ids.allocate(adult::kUnigramTop);
  state.bigrams.allocate(state.bigram_count);
  state.bigram_counts.allocate(state.bigram_count);
  state.trigrams.allocate(state.trigram_count);
  state.trigram_counts.allocate(state.trigram_count);
  state.online_bigrams.allocate(adult::kOnlineNgramCapacity);
  state.online_bigram_counts.allocate(adult::kOnlineNgramCapacity);
  state.online_trigrams.allocate(adult::kOnlineNgramCapacity);
  state.online_trigram_counts.allocate(adult::kOnlineNgramCapacity);
  state.online_associations.allocate(adult::kOnlineAssociationCapacity);
  state.online_association_counts.allocate(adult::kOnlineAssociationCapacity);
  state.online_episode_units.allocate(adult::kOnlineEpisodeCapacity);
  state.online_episode_breaks.allocate(adult::kOnlineEpisodeBreakCapacity);
  state.mutable_sizes.allocate(6u);
  state.motor_context.allocate(6u);
  state.motor_completion.allocate(adult::kEpisodeCompletionUnits);
  state.ledger.allocate(4u);
  state.rng.allocate(1u);

  adult::cuda_require(cudaMemset(state.unit_lengths.get(), 0, state.unit_lengths.bytes()),
                      "clear complete checkpoint unit reserve");
  adult::cuda_require(cudaMemset(state.unit_content.get(), 0, state.unit_content.bytes()),
                      "clear complete checkpoint content reserve");
  adult::cuda_require(cudaMemset(state.unit_vitality.get(), 0, state.unit_vitality.bytes()),
                      "clear complete checkpoint vitality reserve");
  adult::cuda_require(cudaMemset(state.online_bigram_counts.get(), 0,
                                 state.online_bigram_counts.bytes()),
                      "clear complete checkpoint bigram reserve");
  adult::cuda_require(cudaMemset(state.online_trigram_counts.get(), 0,
                                 state.online_trigram_counts.bytes()),
                      "clear complete checkpoint trigram reserve");
  adult::cuda_require(cudaMemset(state.online_association_counts.get(), 0,
                                 state.online_association_counts.bytes()),
                      "clear complete checkpoint association reserve");

  adult::checkpoint_read_device(input, state.boundary_mask.get(), 256u);
  adult::checkpoint_read_device(input, state.boundary_bytes.get(), adult::kBoundaryCount);
  adult::checkpoint_read_device(input, state.boundary_histogram.get(), 256u);
  adult::checkpoint_read_device(input, state.boundary_pairs.get(), 256u * 256u);
  adult::checkpoint_read_device(input, state.unit_lengths.get(), state.unit_count);
  adult::checkpoint_read_device(input, state.unit_content.get(),
      static_cast<std::size_t>(state.unit_count) * adult::kUnitWords);
  adult::checkpoint_read_device(input, state.unit_vitality.get(), state.unit_count);
  adult::checkpoint_read_device(input, state.unigram_top_ids.get(), adult::kUnigramTop);
  adult::checkpoint_read_device(input, state.bigrams.get(), state.bigram_count);
  adult::checkpoint_read_device(input, state.bigram_counts.get(), state.bigram_count);
  adult::checkpoint_read_device(input, state.trigrams.get(), state.trigram_count);
  adult::checkpoint_read_device(input, state.trigram_counts.get(), state.trigram_count);
  adult::checkpoint_read_device(input, state.online_bigrams.get(), state.online_bigram_count);
  adult::checkpoint_read_device(input, state.online_bigram_counts.get(),
                                 state.online_bigram_count);
  adult::checkpoint_read_device(input, state.online_trigrams.get(),
                                 state.online_trigram_count);
  adult::checkpoint_read_device(input, state.online_trigram_counts.get(),
                                 state.online_trigram_count);
  adult::checkpoint_read_device(input, state.online_associations.get(),
                                 state.online_association_count);
  adult::checkpoint_read_device(input, state.online_association_counts.get(),
                                 state.online_association_count);
  adult::checkpoint_read_device(input, state.online_episode_units.get(),
                                 state.online_episode_count);
  adult::checkpoint_read_device(input, state.online_episode_breaks.get(),
                                 state.online_episode_break_count);
  adult::checkpoint_read_device(input, state.mutable_sizes.get(), 6u);
  adult::checkpoint_read_device(input, state.ledger.get(), 4u);
  adult::checkpoint_read_device(input, state.rng.get(), 1u);
  if (input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("trailing complete adult checkpoint bytes");
  }
  adult::cuda_require(cudaMemset(state.motor_context.get(), 0,
                                 state.motor_context.bytes()),
                      "clear complete checkpoint motor context");
  adult::cuda_require(cudaMemset(state.motor_completion.get(), 0,
                                 state.motor_completion.bytes()),
                      "clear complete checkpoint motor completion");
  adult::build_generation_indexes(state);

  state.resident_bytes = state.boundary_mask.bytes() + state.boundary_bytes.bytes() +
      state.boundary_histogram.bytes() + state.boundary_pairs.bytes() +
      state.unit_lengths.bytes() + state.unit_content.bytes() + state.unit_vitality.bytes() +
      state.unigram_top_ids.bytes() + state.bigrams.bytes() + state.bigram_counts.bytes() +
      state.trigrams.bytes() + state.trigram_counts.bytes() +
      state.cached_bigram_contexts.bytes() + state.cached_bigram_entries.bytes() +
      state.cached_trigram_contexts.bytes() + state.cached_trigram_entries.bytes() +
      state.online_bigrams.bytes() + state.online_bigram_counts.bytes() +
      state.online_trigrams.bytes() + state.online_trigram_counts.bytes() +
      state.online_associations.bytes() + state.online_association_counts.bytes() +
      state.online_episode_units.bytes() + state.online_episode_breaks.bytes() +
      state.mutable_sizes.bytes() + state.motor_context.bytes() +
      state.motor_completion.bytes() + state.ledger.bytes() + state.rng.bytes();
  adult::audit_ledger_kernel<<<1u, adult::kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      state.online_episode_count, state.boundary_histogram.get(),
      state.boundary_pairs.get(), state.ledger.get());
  adult::cuda_require(cudaGetLastError(), "launch complete adult checkpoint audit");
  adult::cuda_require(cudaDeviceSynchronize(), "complete adult checkpoint restore");
  return state;
}

#endif  // !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)

inline void save_checkpoint(const StreamState& state, const std::string& path) {
  const std::string part = base_checkpoint_part(path);
  complete_checkpoint::save_checkpoint(state.adult, part);
  std::vector<std::uint8_t> adult_bytes;
  try {
    adult_bytes = read_binary_file(part);
  } catch (...) {
    std::remove(part.c_str());
    throw;
  }
  std::remove(part.c_str());

  DriveState drive{};
  appraisal::ResidentAppraisal resident_appraisal{};
  adult::cuda_require(cudaMemcpy(&drive, state.drive.get(), sizeof(drive),
                                 cudaMemcpyDeviceToHost),
                      "stage resident stream checkpoint");
  adult::cuda_require(cudaMemcpy(&resident_appraisal, state.appraisal.get(),
                                 sizeof(resident_appraisal), cudaMemcpyDeviceToHost),
                      "stage resident appraisal checkpoint");
  std::vector<ActionTransitionEvidence> transitions(
      state.action_transitions.size());
  ActionTransitionScalars transition_scalars{};
  adult::cuda_require(cudaMemcpy(transitions.data(), state.action_transitions.get(),
                                 state.action_transitions.bytes(),
                                 cudaMemcpyDeviceToHost),
                      "stage action transition checkpoint");
  adult::cuda_require(cudaMemcpy(&transition_scalars,
                                 state.action_transition_scalars.get(),
                                 sizeof(transition_scalars),
                                 cudaMemcpyDeviceToHost),
                      "stage action transition scalar checkpoint");
  std::ostringstream matter_output(std::ios::binary | std::ios::out);
  state.conditioned_learning_matter.save(matter_output);
  const std::string matter_bytes = matter_output.str();
  std::ostringstream device_owner_output(std::ios::binary | std::ios::out);
  state.conditioned_device_owner.save(device_owner_output);
  const std::string device_owner_bytes = device_owner_output.str();
  const StreamCheckpointHeader header{kDriveMagic, kStreamVersion,
      static_cast<std::uint32_t>(sizeof(DriveState)),
      static_cast<std::uint32_t>(sizeof(appraisal::ResidentAppraisal)),
      state.chunk_capacity, state.emission_capacity, state.appraisal_holdout_bytes,
      static_cast<std::uint64_t>(adult_bytes.size())};
  const StreamCheckpointExtensionV7 extension{
      static_cast<std::uint32_t>(state.action_transitions.bytes()),
      static_cast<std::uint32_t>(sizeof(transition_scalars)),
      static_cast<std::uint64_t>(matter_bytes.size()),
      static_cast<std::uint64_t>(device_owner_bytes.size()),
      complete_checkpoint::hash_bytes(transitions.data(),
                                      state.action_transitions.bytes()),
      complete_checkpoint::hash_bytes(&transition_scalars,
                                      sizeof(transition_scalars)),
      complete_checkpoint::hash_bytes(matter_bytes.data(),
                                      matter_bytes.size()),
      complete_checkpoint::hash_bytes(device_owner_bytes.data(),
                                      device_owner_bytes.size()),
      state.conditioned_learning_matter.physical_hash(),
      state.conditioned_device_owner.physical_hash(),
      complete_checkpoint::hash_bytes(&header, sizeof(header)),
      complete_checkpoint::hash_bytes(&drive, sizeof(drive)),
      complete_checkpoint::hash_bytes(&resident_appraisal,
                                      sizeof(resident_appraisal))};
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create stream checkpoint: " + path);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(&extension), sizeof(extension));
  output.write(reinterpret_cast<const char*>(&drive), sizeof(drive));
  output.write(reinterpret_cast<const char*>(&resident_appraisal),
               sizeof(resident_appraisal));
  output.write(reinterpret_cast<const char*>(transitions.data()),
               state.action_transitions.bytes());
  output.write(reinterpret_cast<const char*>(&transition_scalars),
               sizeof(transition_scalars));
  if (!matter_bytes.empty()) {
    output.write(matter_bytes.data(),
                 static_cast<std::streamsize>(matter_bytes.size()));
  }
  if (!device_owner_bytes.empty()) {
    output.write(device_owner_bytes.data(),
                 static_cast<std::streamsize>(device_owner_bytes.size()));
  }
  if (!adult_bytes.empty()) {
    output.write(reinterpret_cast<const char*>(adult_bytes.data()), adult_bytes.size());
  }
  if (!output) throw std::runtime_error("stream checkpoint write failed");
}

inline StreamState load_v7_checkpoint(std::ifstream& input,
                                      const std::string& path,
                                      const StreamCheckpointHeader& header) {
  StreamCheckpointExtensionV7 extension{};
  DriveState drive{};
  appraisal::ResidentAppraisal resident_appraisal{};
  input.read(reinterpret_cast<char*>(&extension), sizeof(extension));
  input.read(reinterpret_cast<char*>(&drive), sizeof(drive));
  input.read(reinterpret_cast<char*>(&resident_appraisal),
             sizeof(resident_appraisal));
  if (!input || header.magic != kDriveMagic || header.version != 7u ||
      header.drive_bytes != sizeof(DriveState) || drive.magic != kDriveMagic ||
      header.appraisal_bytes != sizeof(appraisal::ResidentAppraisal) ||
      header.chunk_capacity == 0u || header.emission_capacity == 0u ||
      header.appraisal_holdout_bytes == 0u ||
      extension.action_transition_bytes !=
          kActionTransitionCapacity * sizeof(ActionTransitionEvidence) ||
      extension.action_transition_scalars_bytes !=
          sizeof(ActionTransitionScalars) ||
      header.adult_checkpoint_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.conditioned_matter_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.conditioned_device_owner_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.header_hash !=
          complete_checkpoint::hash_bytes(&header, sizeof(header)) ||
      extension.drive_hash !=
          complete_checkpoint::hash_bytes(&drive, sizeof(drive)) ||
      extension.appraisal_hash != complete_checkpoint::hash_bytes(
          &resident_appraisal, sizeof(resident_appraisal))) {
    throw std::runtime_error("incompatible bcc32 adult stream v7 checkpoint");
  }

  const std::uint64_t prefix_bytes =
      sizeof(StreamCheckpointHeader) + sizeof(StreamCheckpointExtensionV7) +
      sizeof(DriveState) + sizeof(appraisal::ResidentAppraisal);
  const std::uint64_t evidence_bytes =
      static_cast<std::uint64_t>(extension.action_transition_bytes) +
      static_cast<std::uint64_t>(extension.action_transition_scalars_bytes);
  std::uint64_t expected_bytes = prefix_bytes;
  const auto add_extent = [&expected_bytes](std::uint64_t extent) {
    if (extent > std::numeric_limits<std::uint64_t>::max() - expected_bytes)
      throw std::runtime_error("oversized bcc32 adult stream v7 checkpoint");
    expected_bytes += extent;
  };
  add_extent(evidence_bytes);
  add_extent(extension.conditioned_matter_bytes);
  add_extent(extension.conditioned_device_owner_bytes);
  add_extent(header.adult_checkpoint_bytes);
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0 ||
      static_cast<std::uint64_t>(file_size) != expected_bytes) {
    throw std::runtime_error(
        "truncated or trailing bcc32 adult stream v7 checkpoint");
  }
  input.seekg(static_cast<std::streamoff>(prefix_bytes), std::ios::beg);

  std::vector<ActionTransitionEvidence> transitions(kActionTransitionCapacity);
  ActionTransitionScalars transition_scalars{};
  input.read(reinterpret_cast<char*>(transitions.data()),
             extension.action_transition_bytes);
  input.read(reinterpret_cast<char*>(&transition_scalars),
             sizeof(transition_scalars));
  std::string matter_bytes(
      static_cast<std::size_t>(extension.conditioned_matter_bytes), '\0');
  std::string device_owner_bytes(
      static_cast<std::size_t>(extension.conditioned_device_owner_bytes), '\0');
  if (!matter_bytes.empty()) {
    input.read(matter_bytes.data(),
               static_cast<std::streamsize>(matter_bytes.size()));
  }
  if (!device_owner_bytes.empty()) {
    input.read(device_owner_bytes.data(),
               static_cast<std::streamsize>(device_owner_bytes.size()));
  }
  if (!input ||
      extension.action_transition_hash != complete_checkpoint::hash_bytes(
          transitions.data(), extension.action_transition_bytes) ||
      extension.action_transition_scalars_hash !=
          complete_checkpoint::hash_bytes(&transition_scalars,
                                          sizeof(transition_scalars)) ||
      extension.conditioned_matter_hash != complete_checkpoint::hash_bytes(
          matter_bytes.data(), matter_bytes.size()) ||
      extension.conditioned_device_owner_hash !=
          complete_checkpoint::hash_bytes(device_owner_bytes.data(),
                                          device_owner_bytes.size())) {
    throw std::runtime_error("corrupt bcc32 adult stream v7 physical state");
  }

  std::istringstream matter_input(matter_bytes,
                                  std::ios::binary | std::ios::in);
  substrate::bcc32::ConditionedLearningMatter conditioned_matter =
      substrate::bcc32::ConditionedLearningMatter::load(
          matter_input,
          substrate::bcc32::make_paged_conditioned_matter_executor());
  std::istringstream device_owner_input(device_owner_bytes,
                                        std::ios::binary | std::ios::in);
  bcc32::paged_conditioned_owner::PagedConditionedOwner conditioned_device =
      bcc32::paged_conditioned_owner::PagedConditionedOwner::load(
          device_owner_input);
  if (matter_input.peek() != std::istringstream::traits_type::eof() ||
      device_owner_input.peek() != std::istringstream::traits_type::eof() ||
      conditioned_matter.physical_hash() !=
          extension.conditioned_physical_hash ||
      conditioned_device.physical_hash() !=
          extension.conditioned_device_physical_hash) {
    throw std::runtime_error(
        "conditioned owners checkpoint did not restore exactly");
  }

  std::vector<std::uint8_t> adult_bytes(
      static_cast<std::size_t>(header.adult_checkpoint_bytes));
  if (!adult_bytes.empty())
    input.read(reinterpret_cast<char*>(adult_bytes.data()), adult_bytes.size());
  if (!input || input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("truncated bcc32 adult stream v7 adult payload");

  const std::string part = base_checkpoint_part(path);
  write_binary_file(part, adult_bytes);
  adult::AdultState restored;
  try {
    restored = complete_checkpoint::load_checkpoint(part);
  } catch (...) {
    std::remove(part.c_str());
    throw;
  }
  std::remove(part.c_str());

  StreamState state;
  state.adult = std::move(restored);
  state.conditioned_learning_matter = std::move(conditioned_matter);
  state.conditioned_device_owner = std::move(conditioned_device);
  allocate_transport(state, header.chunk_capacity, header.emission_capacity,
                     header.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemcpy(state.drive.get(), &drive, sizeof(drive),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream v7 drive");
  adult::cuda_require(cudaMemcpy(state.appraisal.get(), &resident_appraisal,
                                 sizeof(resident_appraisal),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream v7 appraisal");
  adult::cuda_require(cudaMemcpy(state.action_transitions.get(),
                                 transitions.data(),
                                 state.action_transitions.bytes(),
                                 cudaMemcpyHostToDevice),
                      "restore v7 action transition evidence");
  adult::cuda_require(cudaMemcpy(state.action_transition_scalars.get(),
                                 &transition_scalars,
                                 sizeof(transition_scalars),
                                 cudaMemcpyHostToDevice),
                      "restore v7 action transition scalars");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear restored v7 appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear restored v7 contact summary");
  auto* query_plan = state.query_plan.get();
  auto* query_settlement = state.query_settlement.get();
  auto* query_answer_receipt = state.query_answer_receipt.get();
  void* query_transport_arguments[] = {&query_plan, &query_settlement,
                                       &query_answer_receipt};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(initialize_query_answer_state_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, query_transport_arguments, 0u,
          nullptr),
      "initialize restored v7 query transport");
  adult::cuda_require(cudaGetLastError(),
                      "initialize restored v7 query transport");
  adult::cuda_require(cudaDeviceSynchronize(),
                      "complete resident stream v7 restore");
  state.chronological_bytes = drive.chronological_bytes;
  state.plasticity_disabled = drive.plasticity_enabled == 0u;
  publish_conditioned_conductance(state);
  return state;
}

inline StreamState load_v6_checkpoint(std::ifstream& input,
                                      const std::string& path,
                                      const StreamCheckpointHeader& header) {
  StreamCheckpointExtensionV6 extension{};
  DriveState drive{};
  appraisal::ResidentAppraisal resident_appraisal{};
  input.read(reinterpret_cast<char*>(&extension), sizeof(extension));
  input.read(reinterpret_cast<char*>(&drive), sizeof(drive));
  input.read(reinterpret_cast<char*>(&resident_appraisal),
             sizeof(resident_appraisal));
  if (!input || header.magic != kDriveMagic || header.version != 6u ||
      header.drive_bytes != sizeof(DriveState) || drive.magic != kDriveMagic ||
      header.appraisal_bytes != sizeof(appraisal::ResidentAppraisal) ||
      header.chunk_capacity == 0u || header.emission_capacity == 0u ||
      header.appraisal_holdout_bytes == 0u ||
      extension.action_transition_bytes !=
          kActionTransitionCapacity * sizeof(ActionTransitionEvidence) ||
      extension.action_transition_scalars_bytes !=
          sizeof(ActionTransitionScalars) ||
      header.adult_checkpoint_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.conditioned_matter_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.header_hash !=
          complete_checkpoint::hash_bytes(&header, sizeof(header)) ||
      extension.drive_hash !=
          complete_checkpoint::hash_bytes(&drive, sizeof(drive)) ||
      extension.appraisal_hash != complete_checkpoint::hash_bytes(
          &resident_appraisal, sizeof(resident_appraisal))) {
    throw std::runtime_error("incompatible bcc32 adult stream v6 checkpoint");
  }
  const std::uint64_t prefix_bytes =
      sizeof(StreamCheckpointHeader) + sizeof(StreamCheckpointExtensionV6) +
      sizeof(DriveState) + sizeof(appraisal::ResidentAppraisal);
  const std::uint64_t evidence_bytes =
      static_cast<std::uint64_t>(extension.action_transition_bytes) +
      static_cast<std::uint64_t>(extension.action_transition_scalars_bytes);
  if (evidence_bytes > std::numeric_limits<std::uint64_t>::max() -
                           prefix_bytes ||
      extension.conditioned_matter_bytes >
          std::numeric_limits<std::uint64_t>::max() - prefix_bytes -
              evidence_bytes ||
      header.adult_checkpoint_bytes >
          std::numeric_limits<std::uint64_t>::max() - prefix_bytes -
              evidence_bytes - extension.conditioned_matter_bytes) {
    throw std::runtime_error("oversized bcc32 adult stream v6 checkpoint");
  }
  const std::uint64_t expected_bytes =
      prefix_bytes + evidence_bytes + extension.conditioned_matter_bytes +
      header.adult_checkpoint_bytes;
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0 || static_cast<std::uint64_t>(file_size) != expected_bytes)
    throw std::runtime_error(
        "truncated or trailing bcc32 adult stream v6 checkpoint");
  input.seekg(static_cast<std::streamoff>(prefix_bytes), std::ios::beg);

  std::vector<ActionTransitionEvidence> transitions(kActionTransitionCapacity);
  ActionTransitionScalars transition_scalars{};
  input.read(reinterpret_cast<char*>(transitions.data()),
             extension.action_transition_bytes);
  input.read(reinterpret_cast<char*>(&transition_scalars),
             sizeof(transition_scalars));
  std::string matter_bytes(
      static_cast<std::size_t>(extension.conditioned_matter_bytes), '\0');
  if (!matter_bytes.empty()) {
    input.read(matter_bytes.data(),
               static_cast<std::streamsize>(matter_bytes.size()));
  }
  if (!input ||
      extension.action_transition_hash != complete_checkpoint::hash_bytes(
          transitions.data(), extension.action_transition_bytes) ||
      extension.action_transition_scalars_hash !=
          complete_checkpoint::hash_bytes(&transition_scalars,
                                          sizeof(transition_scalars)) ||
      extension.conditioned_matter_hash != complete_checkpoint::hash_bytes(
          matter_bytes.data(), matter_bytes.size())) {
    throw std::runtime_error("corrupt bcc32 adult stream v6 physical state");
  }
  std::istringstream matter_input(matter_bytes,
                                  std::ios::binary | std::ios::in);
  substrate::bcc32::ConditionedLearningMatter conditioned_matter =
      substrate::bcc32::ConditionedLearningMatter::load(
          matter_input,
          substrate::bcc32::make_paged_conditioned_matter_executor());
  if (matter_input.peek() != std::istringstream::traits_type::eof() ||
      conditioned_matter.physical_hash() !=
          extension.conditioned_physical_hash) {
    throw std::runtime_error(
        "conditioned matter checkpoint did not restore exactly");
  }

  std::vector<std::uint8_t> adult_bytes(
      static_cast<std::size_t>(header.adult_checkpoint_bytes));
  if (!adult_bytes.empty())
    input.read(reinterpret_cast<char*>(adult_bytes.data()), adult_bytes.size());
  if (!input || input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("truncated bcc32 adult stream v6 adult payload");

  const std::string part = base_checkpoint_part(path);
  write_binary_file(part, adult_bytes);
  adult::AdultState restored;
  try {
    restored = complete_checkpoint::load_checkpoint(part);
  } catch (...) {
    std::remove(part.c_str());
    throw;
  }
  std::remove(part.c_str());

  StreamState state;
  state.adult = std::move(restored);
  state.conditioned_learning_matter = std::move(conditioned_matter);
  state.conditioned_device_owner =
      bcc32::paged_conditioned_owner::PagedConditionedOwner::migrate_legacy(
          state.conditioned_learning_matter);
  allocate_transport(state, header.chunk_capacity, header.emission_capacity,
                     header.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemcpy(state.drive.get(), &drive, sizeof(drive),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream v6 drive");
  adult::cuda_require(cudaMemcpy(state.appraisal.get(), &resident_appraisal,
                                 sizeof(resident_appraisal),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream v6 appraisal");
  adult::cuda_require(cudaMemcpy(state.action_transitions.get(),
                                 transitions.data(),
                                 state.action_transitions.bytes(),
                                 cudaMemcpyHostToDevice),
                      "restore v6 action transition evidence");
  adult::cuda_require(cudaMemcpy(state.action_transition_scalars.get(),
                                 &transition_scalars,
                                 sizeof(transition_scalars),
                                 cudaMemcpyHostToDevice),
                      "restore v6 action transition scalars");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear restored v6 appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear restored v6 contact summary");
  auto* query_plan = state.query_plan.get();
  auto* query_settlement = state.query_settlement.get();
  auto* query_answer_receipt = state.query_answer_receipt.get();
  void* query_transport_arguments[] = {&query_plan, &query_settlement,
                                       &query_answer_receipt};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(initialize_query_answer_state_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, query_transport_arguments, 0u,
          nullptr),
      "initialize restored v6 query transport");
  adult::cuda_require(cudaGetLastError(),
                      "initialize restored v6 query transport");
  adult::cuda_require(cudaDeviceSynchronize(),
                      "complete resident stream v6 restore");
  state.chronological_bytes = drive.chronological_bytes;
  state.plasticity_disabled = drive.plasticity_enabled == 0u;
  publish_conditioned_conductance(state);
  return state;
}

inline StreamState load_v4_checkpoint(std::ifstream& input,
                                      const std::string& path,
                                      const StreamCheckpointHeader& header) {
  StreamCheckpointExtensionV4 extension{};
  DriveState drive{};
  appraisal::ResidentAppraisal resident_appraisal{};
  input.read(reinterpret_cast<char*>(&extension), sizeof(extension));
  input.read(reinterpret_cast<char*>(&drive), sizeof(drive));
  input.read(reinterpret_cast<char*>(&resident_appraisal),
             sizeof(resident_appraisal));
  if (!input || header.magic != kDriveMagic || header.version != 4u ||
      header.drive_bytes != sizeof(DriveState) || drive.magic != kDriveMagic ||
      header.appraisal_bytes != sizeof(appraisal::ResidentAppraisal) ||
      header.chunk_capacity == 0u || header.emission_capacity == 0u ||
      header.appraisal_holdout_bytes == 0u ||
      extension.action_transition_bytes !=
          kActionTransitionCapacity * sizeof(ActionTransitionEvidence) ||
      extension.action_transition_scalars_bytes != sizeof(ActionTransitionScalars) ||
      header.adult_checkpoint_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.header_hash != complete_checkpoint::hash_bytes(&header, sizeof(header)) ||
      extension.drive_hash != complete_checkpoint::hash_bytes(&drive, sizeof(drive)) ||
      extension.appraisal_hash != complete_checkpoint::hash_bytes(
          &resident_appraisal, sizeof(resident_appraisal))) {
    throw std::runtime_error("incompatible bcc32 adult stream v4 checkpoint");
  }

  const std::uint64_t prefix_bytes =
      sizeof(StreamCheckpointHeader) + sizeof(StreamCheckpointExtensionV4) +
      sizeof(DriveState) + sizeof(appraisal::ResidentAppraisal);
  const std::uint64_t evidence_bytes =
      static_cast<std::uint64_t>(extension.action_transition_bytes) +
      static_cast<std::uint64_t>(extension.action_transition_scalars_bytes);
  if (evidence_bytes > std::numeric_limits<std::uint64_t>::max() - prefix_bytes ||
      header.adult_checkpoint_bytes >
          std::numeric_limits<std::uint64_t>::max() - prefix_bytes -
              evidence_bytes) {
    throw std::runtime_error("oversized bcc32 adult stream v4 checkpoint");
  }
  const std::uint64_t expected_bytes =
      prefix_bytes + evidence_bytes + header.adult_checkpoint_bytes;
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0 || static_cast<std::uint64_t>(file_size) != expected_bytes) {
    throw std::runtime_error("truncated or trailing bcc32 adult stream v4 checkpoint");
  }
  input.seekg(static_cast<std::streamoff>(prefix_bytes), std::ios::beg);

  std::vector<ActionTransitionEvidence> transitions(kActionTransitionCapacity);
  ActionTransitionScalars transition_scalars{};
  input.read(reinterpret_cast<char*>(transitions.data()),
             extension.action_transition_bytes);
  input.read(reinterpret_cast<char*>(&transition_scalars),
             sizeof(transition_scalars));
  if (!input ||
      extension.action_transition_hash != complete_checkpoint::hash_bytes(
          transitions.data(), extension.action_transition_bytes) ||
      extension.action_transition_scalars_hash != complete_checkpoint::hash_bytes(
          &transition_scalars, sizeof(transition_scalars))) {
    throw std::runtime_error("corrupt bcc32 adult stream v4 action evidence");
  }

  std::vector<std::uint8_t> adult_bytes(
      static_cast<std::size_t>(header.adult_checkpoint_bytes));
  if (!adult_bytes.empty()) {
    input.read(reinterpret_cast<char*>(adult_bytes.data()), adult_bytes.size());
  }
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("truncated bcc32 adult stream v4 adult payload");
  }

  const std::string part = base_checkpoint_part(path);
  write_binary_file(part, adult_bytes);
  adult::AdultState restored;
  try {
    restored = complete_checkpoint::load_checkpoint(part);
  } catch (...) {
    std::remove(part.c_str());
    throw;
  }
  std::remove(part.c_str());

  StreamState state;
  state.adult = std::move(restored);
  allocate_transport(state, header.chunk_capacity, header.emission_capacity,
                     header.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemcpy(state.drive.get(), &drive, sizeof(drive),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream v4 drive");
  adult::cuda_require(cudaMemcpy(state.appraisal.get(), &resident_appraisal,
                                 sizeof(resident_appraisal),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream v4 appraisal");
  adult::cuda_require(cudaMemcpy(state.action_transitions.get(), transitions.data(),
                                 state.action_transitions.bytes(),
                                 cudaMemcpyHostToDevice),
                      "restore action transition evidence");
  adult::cuda_require(cudaMemcpy(state.action_transition_scalars.get(),
                                 &transition_scalars, sizeof(transition_scalars),
                                 cudaMemcpyHostToDevice),
                      "restore action transition scalars");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear restored v4 appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear restored v4 contact summary");
  auto* query_plan = state.query_plan.get();
  auto* query_settlement = state.query_settlement.get();
  auto* query_answer_receipt = state.query_answer_receipt.get();
  void* query_transport_arguments[] = {&query_plan, &query_settlement,
                                       &query_answer_receipt};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(initialize_query_answer_state_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, query_transport_arguments, 0u,
          nullptr),
      "initialize restored v4 query transport");
  adult::cuda_require(cudaGetLastError(), "initialize restored v4 query transport");
  adult::cuda_require(cudaDeviceSynchronize(),
                      "complete resident stream v4 restore");
  state.chronological_bytes = drive.chronological_bytes;
  state.plasticity_disabled = drive.plasticity_enabled == 0u;
  publish_conditioned_conductance(state);
  return state;
}

inline StreamState load_v5_checkpoint(std::ifstream& input,
                                      const std::string& path,
                                      const StreamCheckpointHeader& header) {
  StreamCheckpointExtensionV5 extension{};
  DriveState drive{};
  appraisal::ResidentAppraisal resident_appraisal{};
  input.read(reinterpret_cast<char*>(&extension), sizeof(extension));
  input.read(reinterpret_cast<char*>(&drive), sizeof(drive));
  input.read(reinterpret_cast<char*>(&resident_appraisal),
             sizeof(resident_appraisal));
  if (!input || header.magic != kDriveMagic || header.version != 5u ||
      header.drive_bytes != sizeof(DriveState) || drive.magic != kDriveMagic ||
      header.appraisal_bytes != sizeof(appraisal::ResidentAppraisal) ||
      header.chunk_capacity == 0u || header.emission_capacity == 0u ||
      header.appraisal_holdout_bytes == 0u ||
      header.adult_checkpoint_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      extension.policy_hash != complete_checkpoint::hash_bytes(
          &extension.policy, sizeof(extension.policy)) ||
      extension.header_hash != complete_checkpoint::hash_bytes(&header,
                                                                sizeof(header)) ||
      extension.drive_hash != complete_checkpoint::hash_bytes(&drive,
                                                               sizeof(drive)) ||
      extension.appraisal_hash != complete_checkpoint::hash_bytes(
          &resident_appraisal, sizeof(resident_appraisal))) {
    throw std::runtime_error("incompatible bcc32 adult stream v5 checkpoint");
  }

  const std::uint64_t prefix_bytes =
      sizeof(StreamCheckpointHeader) + sizeof(StreamCheckpointExtensionV5) +
      sizeof(DriveState) + sizeof(appraisal::ResidentAppraisal);
  const std::uint64_t transient_bytes =
      static_cast<std::uint64_t>(extension.policy.plan_bytes) +
      static_cast<std::uint64_t>(extension.eligibility_bytes);
  if (transient_bytes > std::numeric_limits<std::uint64_t>::max() - prefix_bytes ||
      header.adult_checkpoint_bytes >
          std::numeric_limits<std::uint64_t>::max() - prefix_bytes -
              transient_bytes) {
    throw std::runtime_error("oversized bcc32 adult stream v5 checkpoint");
  }
  const std::uint64_t expected_bytes =
      prefix_bytes + transient_bytes + header.adult_checkpoint_bytes;
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0 || static_cast<std::uint64_t>(file_size) != expected_bytes) {
    throw std::runtime_error("truncated or trailing bcc32 adult stream v5 checkpoint");
  }
  input.seekg(static_cast<std::streamoff>(prefix_bytes), std::ios::beg);

  std::vector<std::uint8_t> plan_bytes(extension.policy.plan_bytes);
  std::vector<std::uint8_t> eligibility_bytes(extension.eligibility_bytes);
  if (!plan_bytes.empty()) {
    input.read(reinterpret_cast<char*>(plan_bytes.data()), plan_bytes.size());
  }
  if (!eligibility_bytes.empty()) {
    input.read(reinterpret_cast<char*>(eligibility_bytes.data()),
               eligibility_bytes.size());
  }
  if (!input || extension.plan_hash != complete_checkpoint::hash_bytes(
                                       plan_bytes.data(), plan_bytes.size()) ||
      extension.eligibility_hash != complete_checkpoint::hash_bytes(
                                      eligibility_bytes.data(),
                                      eligibility_bytes.size())) {
    throw std::runtime_error("corrupt bcc32 adult stream v5 transient state");
  }

  std::vector<std::uint8_t> adult_bytes(
      static_cast<std::size_t>(header.adult_checkpoint_bytes));
  if (!adult_bytes.empty()) {
    input.read(reinterpret_cast<char*>(adult_bytes.data()), adult_bytes.size());
  }
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("truncated bcc32 adult stream v5 adult payload");
  }

  const std::string part = base_checkpoint_part(path);
  write_binary_file(part, adult_bytes);
  adult::AdultState restored;
  try {
    restored = complete_checkpoint::load_checkpoint(part);
  } catch (...) {
    std::remove(part.c_str());
    throw;
  }
  std::remove(part.c_str());

  StreamState state;
  state.adult = std::move(restored);
  allocate_transport(state, header.chunk_capacity, header.emission_capacity,
                     header.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemcpy(state.drive.get(), &drive, sizeof(drive),
                                 cudaMemcpyHostToDevice),
                      "restore migrated resident stream drive");
  adult::cuda_require(cudaMemcpy(state.appraisal.get(), &resident_appraisal,
                                 sizeof(resident_appraisal),
                                 cudaMemcpyHostToDevice),
                      "restore migrated resident stream appraisal");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear migrated appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear migrated contact summary");
  auto* query_plan = state.query_plan.get();
  auto* query_settlement = state.query_settlement.get();
  auto* query_answer_receipt = state.query_answer_receipt.get();
  void* query_transport_arguments[] = {&query_plan, &query_settlement,
                                       &query_answer_receipt};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(initialize_query_answer_state_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, query_transport_arguments, 0u,
          nullptr),
      "initialize migrated query transport");
  adult::cuda_require(cudaGetLastError(), "initialize migrated query transport");
  adult::cuda_require(cudaDeviceSynchronize(),
                      "complete resident stream v5 migration");
  state.chronological_bytes = drive.chronological_bytes;
  state.plasticity_disabled = drive.plasticity_enabled == 0u;
  state.legacy_generator_enabled =
      (extension.policy.policy_flags & 1u) == 0u;
  publish_conditioned_conductance(state);
  return state;
}

inline StreamState load_checkpoint(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open stream checkpoint: " + path);
  StreamCheckpointHeader header{};
  DriveState drive{};
  appraisal::ResidentAppraisal resident_appraisal{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (input && header.version == 7u) {
    return load_v7_checkpoint(input, path, header);
  }
#if defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)
  throw std::runtime_error(
      "focused stream checkpoint projection accepts v7 only");
#else
  if (input && header.version == 6u) {
    return load_v6_checkpoint(input, path, header);
  }
  if (input && header.version == 5u) {
    return load_v5_checkpoint(input, path, header);
  }
  if (input && header.version == 4u) {
    return load_v4_checkpoint(input, path, header);
  }
  input.read(reinterpret_cast<char*>(&drive), sizeof(drive));
  input.read(reinterpret_cast<char*>(&resident_appraisal),
             sizeof(resident_appraisal));
  // Stream-v2/v3 stored the same fixed header and adult payload. Their query
  // transport was ephemeral, so migration initializes empty action evidence
  // rather than pretending the older checkpoint carried an interaction trace.
  if (!input || header.magic != kDriveMagic ||
      (header.version != 2u && header.version != 3u) ||
      header.drive_bytes != sizeof(DriveState) || drive.magic != kDriveMagic ||
      header.appraisal_bytes != sizeof(appraisal::ResidentAppraisal) ||
      header.chunk_capacity == 0u || header.emission_capacity == 0u ||
      header.appraisal_holdout_bytes == 0u ||
      header.adult_checkpoint_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("incompatible bcc32 adult stream v1 checkpoint");
  }
  std::vector<std::uint8_t> adult_bytes(
      static_cast<std::size_t>(header.adult_checkpoint_bytes));
  if (!adult_bytes.empty()) {
    input.read(reinterpret_cast<char*>(adult_bytes.data()), adult_bytes.size());
  }
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("truncated or trailing stream checkpoint bytes");
  }

  const std::string part = base_checkpoint_part(path);
  write_binary_file(part, adult_bytes);
  adult::AdultState restored;
  try {
    restored = adult::load_checkpoint(part);
  } catch (...) {
    std::remove(part.c_str());
    throw;
  }
  std::remove(part.c_str());

  StreamState state;
  state.adult = std::move(restored);
  allocate_transport(state, header.chunk_capacity, header.emission_capacity,
                     header.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemcpy(state.drive.get(), &drive, sizeof(drive),
                                 cudaMemcpyHostToDevice),
                      "restore resident stream drive");
  adult::cuda_require(cudaMemcpy(state.appraisal.get(), &resident_appraisal,
                                 sizeof(resident_appraisal), cudaMemcpyHostToDevice),
                      "restore resident stream appraisal");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear restored appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear restored contact summary");
  adult::cuda_require(cudaDeviceSynchronize(), "complete resident stream restore");
  state.chronological_bytes = drive.chronological_bytes;
  state.plasticity_disabled = drive.plasticity_enabled == 0u;
  publish_conditioned_conductance(state);
  return state;
#endif
}

}  // namespace bcc32_cuda_adult_stream_v1
