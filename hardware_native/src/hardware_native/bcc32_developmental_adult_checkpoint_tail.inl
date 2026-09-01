  void save_checkpoint(const std::string& path) const {
    require_sensorimotor_checkpoint_boundary();
    require_readout_checkpoint_boundary();
    require_selective_state_checkpoint_boundary();
    require_sparse_event_memory_checkpoint_boundary();
    require_sparse_event_memory_checkpoint_valid();
    const Snapshot state = snapshot();
    CheckpointHeader header{};
    header.magic = kCheckpointMagic;
    header.version = kCheckpointVersion;
    header.header_bytes = sizeof(header);
    header.founder_hash = founder_hash_;
    header.germ_site_count = founder_snapshot_.entries.size();
    header.aperture_edge_chunks = kApertureEdgeChunks;
    header.completed_ticks = completed_ticks_;
    header.site_count = executor_.site_count();
    header.entry_count = state.entries.size();
    header.law_identity = law_identity_;
    header.genesis_manifest_identity = genesis_manifest_identity_;
    header.genesis_manifest_entry_count = genesis_manifest_entry_count_;
    header.genesis_recipe_kind = genesis_recipe_kind_;
    header.founder_entry_count = founder_snapshot_.entries.size();
    header.founder_state_hash = state_hash(founder_snapshot_);
    header.state_hash = bound_state_hash(
        state, founder_snapshot_, founder_hash_, header.law_identity,
        header.genesis_manifest_identity,
        header.genesis_recipe_kind, header.genesis_manifest_entry_count);
    const BoundarySnapshot boundary_state = boundary_snapshot();
    header.boundary_hash =
        developmental_adult::boundary_hash(boundary_state);
    header.topology_hash =
        developmental_adult::topology_hash(map_.host);
    header.boundary = boundary_;
    header.boundary_words = boundary_state.words;
    header.resident_history_count = resident_step_history_.size();
    header.resident_history_hash = developmental_adult::resident_history_hash(
        resident_step_history_);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot create grown-adult checkpoint");
    }
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const StateEntry& entry : state.entries) {
      const CheckpointEntry stored{entry.slot, entry.word, 0u};
      output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
    }
    for (const StateEntry& entry : founder_snapshot_.entries) {
      const CheckpointEntry stored{entry.slot, entry.word, 0u};
      output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
    }
    if (!resident_step_history_.empty()) {
      output.write(
          reinterpret_cast<const char*>(resident_step_history_.data()),
          static_cast<std::streamsize>(resident_step_history_.size() *
                                       sizeof(std::uint32_t)));
    }
    if (!output) {
      throw std::runtime_error("cannot write grown-adult checkpoint");
    }
  }

  [[nodiscard]] static std::unique_ptr<GrownAdult> load_checkpoint(
      const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open grown-adult checkpoint");
    }
    struct CheckpointPrefix {
      std::array<char, 8> magic{};
      std::uint32_t version = 0u;
      std::uint32_t header_bytes = 0u;
    };
    CheckpointPrefix prefix{};
    input.read(reinterpret_cast<char*>(&prefix), sizeof(prefix));
    if (!input || prefix.magic != kCheckpointMagic) {
      throw std::runtime_error("invalid grown-adult checkpoint header");
    }
    input.seekg(0);
    CheckpointHeader header{};
    if (prefix.version == kCheckpointVersion &&
        prefix.header_bytes == sizeof(CheckpointHeader)) {
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
    } else {
      throw std::runtime_error(
          "grown-adult checkpoint law epoch unsupported; explicit migration required");
    }
    if (!input) {
      throw std::runtime_error("invalid grown-adult checkpoint header");
    }
    if (header.germ_site_count != header.founder_entry_count ||
        header.aperture_edge_chunks != kApertureEdgeChunks) {
      throw std::runtime_error("grown-adult checkpoint recipe mismatch");
    }
    const ContentAddress current_law = canonical_law_identity();
    if (header.law_identity != current_law ||
        !is_valid_content_address(header.genesis_manifest_identity) ||
        header.reserved != 0u ||
        (header.genesis_recipe_kind != GenesisRecipeKind::developmental_hash &&
         header.genesis_recipe_kind != GenesisRecipeKind::explicit_manifest) ||
        header.genesis_manifest_entry_count == 0u ||
        header.genesis_manifest_entry_count > header.site_count) {
      throw std::runtime_error(
          "grown-adult checkpoint law/Genesis manifest mismatch");
    }
    if (header.genesis_recipe_kind == GenesisRecipeKind::developmental_hash &&
        header.germ_site_count != kGermSiteCount) {
      throw std::runtime_error(
          "grown-adult hash genesis does not contain the complete germ");
    }
    auto adult = std::unique_ptr<GrownAdult>(new GrownAdult(kRestoreTag));
    adult->founder_hash_ = header.founder_hash;
    adult->law_identity_ = header.law_identity;
    adult->genesis_manifest_identity_ = header.genesis_manifest_identity;
    adult->genesis_manifest_entry_count_ =
        header.genesis_manifest_entry_count;
    adult->genesis_recipe_kind_ = header.genesis_recipe_kind;
    if (header.site_count != adult->executor_.site_count() ||
        header.topology_hash !=
            developmental_adult::topology_hash(adult->map_.host)) {
      throw std::runtime_error("grown-adult checkpoint topology mismatch");
    }
    if (header.entry_count > header.site_count ||
        header.founder_entry_count > header.site_count ||
        header.founder_entry_count == 0u) {
      throw std::runtime_error(
          "grown-adult checkpoint state/founder entry count invalid");
    }
    if (header.resident_history_count > header.completed_ticks) {
      throw std::runtime_error(
          "grown-adult checkpoint resident history count invalid");
    }
    std::vector<CheckpointEntry> entries(header.entry_count);
    input.read(reinterpret_cast<char*>(entries.data()),
               static_cast<std::streamsize>(entries.size() *
                                            sizeof(CheckpointEntry)));
    std::vector<CheckpointEntry> founder_entries(header.founder_entry_count);
    input.read(reinterpret_cast<char*>(founder_entries.data()),
               static_cast<std::streamsize>(founder_entries.size() *
                                            sizeof(CheckpointEntry)));
    std::vector<std::uint32_t> resident_history(
        header.resident_history_count);
    if (!resident_history.empty()) {
      input.read(reinterpret_cast<char*>(resident_history.data()),
                 static_cast<std::streamsize>(resident_history.size() *
                                              sizeof(std::uint32_t)));
    }
    char trailing = 0;
    if (!input || input.read(&trailing, 1)) {
      throw std::runtime_error("grown-adult checkpoint payload invalid");
    }
    std::vector<std::uint64_t> slots;
    slots.reserve(entries.size());
    std::vector<StateEntry> restored_entries;
    restored_entries.reserve(entries.size());
    std::uint64_t previous = 0u;
    bool first = true;
    for (const CheckpointEntry& entry : entries) {
      if (entry.slot >= header.site_count || entry.word == kQ ||
          entry.reserved != 0u ||
          (!first && entry.slot <= previous)) {
        throw std::runtime_error("grown-adult checkpoint state invalid");
      }
      slots.push_back(entry.slot);
      restored_entries.push_back({entry.slot, entry.word});
      previous = entry.slot;
      first = false;
    }
    Snapshot restored_founder;
    restored_founder.entries.reserve(founder_entries.size());
    previous = 0u;
    first = true;
    for (const CheckpointEntry& entry : founder_entries) {
      if (entry.slot >= header.site_count || entry.word == kQ ||
          entry.reserved != 0u || (!first && entry.slot <= previous)) {
        throw std::runtime_error(
            "grown-adult checkpoint founder state invalid");
      }
      restored_founder.entries.push_back({entry.slot, entry.word});
      previous = entry.slot;
      first = false;
    }
    if (restored_founder.completed_ticks != 0u ||
        state_hash(restored_founder) != header.founder_state_hash) {
      throw std::runtime_error(
          "grown-adult checkpoint founder state hash mismatch");
    }
    if (!restored_entries.empty()) {
      StateEntry* device_entries = nullptr;
      require_cuda(cudaMalloc(&device_entries,
                              restored_entries.size() * sizeof(StateEntry)),
                   "allocate checkpoint state scatter");
      require_cuda(cudaMemcpy(device_entries, restored_entries.data(),
                              restored_entries.size() * sizeof(StateEntry),
                              cudaMemcpyHostToDevice),
                   "upload checkpoint state scatter");
      const std::uint32_t blocks = static_cast<std::uint32_t>(
          std::min<std::size_t>((restored_entries.size() + 255u) / 256u,
                                4096u));
      scatter_state_entries_kernel<<<blocks, 256>>>(
          adult->executor_.mutable_device_words(), device_entries,
          restored_entries.size());
      require_cuda(cudaGetLastError(), "launch checkpoint state scatter");
      require_cuda(cudaDeviceSynchronize(),
                   "synchronize checkpoint state scatter");
      require_cuda(cudaFree(device_entries),
                   "release checkpoint state scatter");
    }
    adult->support_ = std::make_unique<CudaBcc32ActiveSupport>(
        adult->executor_, adult->map_.host, slots);
    adult->founder_snapshot_ = std::move(restored_founder);
    adult->completed_ticks_ = header.completed_ticks;
    constexpr std::uint32_t kResidentStepKnownBits =
        kResidentStepSucceeded | kResidentStepAdvancedInstance |
        kResidentStepJournaledFormCredit | kResidentStepAgedFormCredit |
        kResidentStepSelectiveState | kResidentStepSparseEventMemory;
    for (const std::uint32_t advancement : resident_history) {
      const std::uint32_t form_modes =
          advancement & (kResidentStepJournaledFormCredit |
                         kResidentStepAgedFormCredit);
      if ((advancement & kResidentStepSucceeded) == 0u ||
          (advancement & ~kResidentStepKnownBits) != 0u ||
          form_modes == (kResidentStepJournaledFormCredit |
                         kResidentStepAgedFormCredit)) {
        throw std::runtime_error(
            "grown-adult checkpoint resident history invalid");
      }
    }
    if (developmental_adult::resident_history_hash(resident_history) !=
        header.resident_history_hash) {
      throw std::runtime_error(
          "grown-adult checkpoint resident history hash mismatch");
    }
    adult->resident_step_history_ = std::move(resident_history);
    grown_cloud_factor::DeviceLayout cloud_layout{};
    for (std::uint32_t index = 0u;
         index < grown_cloud_factor::kResidentRailCount; ++index) {
      const grown_cloud_factor::PhysicalOffset offset =
          grown_cloud_factor::physical_offset(index);
      cloud_layout.rails[index] =
          adult->physical_slot({offset.x, offset.y, offset.z});
    }
    const std::uint64_t cloud_marker_slot =
        cloud_layout.rails[grown_cloud_factor::global_rail(
            grown_cloud_factor::kCloudMarker, 0u)];
    if (adult->executor_.read_word(cloud_marker_slot) ==
        grown_cloud_factor::kCloudMarkerValue) {
      adult->cloud_factor_enabled_ = true;
      adult->cloud_factor_layout_ = cloud_layout;
      adult->upload_cloud_factor_layout();
    }
    grown_instance_basin_factor::DeviceLayout instance_layout{};
    for (std::uint32_t index = 0u;
         index < grown_instance_basin_factor::kPhysicalRailCount; ++index) {
      const grown_instance_basin_factor::PhysicalOffset offset =
          grown_instance_basin_factor::physical_offset(index);
      instance_layout.rails[index] =
          adult->physical_slot({offset.x, offset.y, offset.z});
    }
    const std::uint64_t instance_marker_slot =
        instance_layout.rails[grown_instance_basin_factor::resident_index(
            grown_instance_basin_factor::kFactorMarker)];
    if (adult->executor_.read_word(instance_marker_slot) ==
        grown_instance_basin_factor::kFactorMarkerValue) {
      if (adult->executor_.read_word(
              instance_layout.rails[grown_instance_basin_factor::resident_index(
                  grown_instance_basin_factor::kLayoutVersion)]) !=
          grown_instance_basin_factor::kLayoutVersionValue) {
        throw std::runtime_error(
            "grown-adult checkpoint instance basin layout unsupported");
      }
      adult->instance_basin_factor_enabled_ = true;
      adult->instance_basin_layout_ = instance_layout;
      adult->upload_instance_basin_layout();
      adult->ensure_factor_advanced();
    }
    grown_sensorimotor_factor::DeviceLayout sensorimotor_layout{};
    for (std::uint32_t index = 0u;
         index < grown_sensorimotor_factor::kPhysicalRailCount; ++index) {
      const grown_sensorimotor_factor::PhysicalOffset offset =
          grown_sensorimotor_factor::physical_offset(index);
      sensorimotor_layout.rails[index] =
          adult->physical_slot({offset.x, offset.y, offset.z});
    }
    for (std::uint32_t index = 0u;
         index < grown_instance_basin_factor::kPhysicalRailCount; ++index) {
      const grown_instance_basin_factor::PhysicalOffset offset =
          grown_instance_basin_factor::physical_offset(index);
      sensorimotor_layout.context.rails[index] =
          adult->physical_slot({offset.x, offset.y, offset.z});
    }
    const std::uint64_t sensorimotor_marker_slot =
        sensorimotor_layout.rails[grown_sensorimotor_factor::value_index(
            grown_sensorimotor_factor::kFactorMarker)];
    if (adult->executor_.read_word(sensorimotor_marker_slot) ==
        grown_sensorimotor_factor::kFactorMarkerValue) {
      if (adult->executor_.read_word(
              sensorimotor_layout.rails[
                  grown_sensorimotor_factor::value_index(
                      grown_sensorimotor_factor::kLayoutVersion)]) !=
          grown_sensorimotor_factor::kLayoutVersionValue) {
        throw std::runtime_error(
            "grown-adult checkpoint sensorimotor layout unsupported");
      }
      const SiteWord journal_count = adult->executor_.read_word(
          sensorimotor_layout
              .rails[grown_sensorimotor_factor::value_index(
                  grown_sensorimotor_factor::kJournalCount)]);
      if (journal_count > grown_sensorimotor_factor::kJournalDepth) {
        throw std::runtime_error(
            "grown-adult checkpoint sensorimotor journal unsupported");
      }
      adult->sensorimotor_factor_enabled_ = true;
      adult->sensorimotor_factor_layout_ = sensorimotor_layout;
      adult->upload_sensorimotor_factor_layout();
      adult->ensure_factor_advanced();
    }
    resident_readout_f_route::DeviceLayout readout_layout{};
    for (std::uint32_t index = 0u;
         index < resident_readout_f_route::kPhysicalRailCount; ++index) {
      const resident_readout_f_route::PhysicalOffset offset =
          resident_readout_f_route::physical_offset(index);
      readout_layout.rails[index] =
          adult->physical_slot({offset.x, offset.y, offset.z});
    }
    const std::uint64_t readout_marker_slot = readout_layout.rails[
        resident_readout_f_route::global_rail(
            resident_readout_f_route::kFactorMarker, 0u)];
    if (adult->executor_.read_word(readout_marker_slot) ==
        resident_readout_f_route::kFactorMarkerValue) {
      const SiteWord event_count = adult->executor_.read_word(
          readout_layout.rails[resident_readout_f_route::global_rail(
              resident_readout_f_route::kEventCount, 0u)]);
      const SiteWord journal_count = adult->executor_.read_word(
          readout_layout.rails[resident_readout_f_route::global_rail(
              resident_readout_f_route::kJournalCount, 0u)]);
      if (event_count > resident_readout_f_route::kEventDepth ||
          journal_count > resident_readout_f_route::kJournalCapacity) {
        throw std::runtime_error(
            "grown-adult checkpoint resident readout journal unsupported");
      }
      adult->readout_factor_enabled_ = true;
      adult->readout_factor_layout_ = readout_layout;
      adult->upload_readout_factor_layout();
      adult->ensure_factor_advanced();
    }
    const grown_form_credit_factor::DeviceLayout form_credit_layout =
        grown_form_credit_factor::make_layout(*adult);
    if (adult->executor_.read_word(form_credit_layout.rails[
            grown_form_credit_factor::global_index(0u, 0u)]) ==
        grown_form_credit_factor::kFactorMarkerValue) {
      const SiteWord journal_count = adult->executor_.read_word(
          form_credit_layout.rails[grown_form_credit_factor::global_index(1u, 0u)]);
      if (journal_count > grown_form_credit_factor::kJournalDepth) {
        throw std::runtime_error(
            "grown-adult checkpoint form credit journal unsupported");
      }
      adult->form_credit_factor_enabled_ = true;
      adult->form_credit_layout_ = form_credit_layout;
      adult->upload_form_credit_layout();
      adult->ensure_factor_advanced();
    }
    if (adult->executor_.read_word(
            grown_selective_state_space::fixed_physical_slot(
                grown_selective_state_space::pair_index(
                    grown_selective_state_space::kGlobalBase,
                    grown_selective_state_space::kFactorMarker, 0u))) ==
        grown_selective_state_space::kFactorMarkerValue) {
      if (adult->sensorimotor_factor_enabled_ ||
          adult->form_credit_factor_enabled_) {
        throw std::runtime_error(
            "checkpoint has multiple owners of the raw-motor aperture");
      }
      if (!adult->instance_basin_factor_enabled_)
        throw std::runtime_error("checkpoint recurrence lacks situation tissue");
      if (adult->executor_.read_word(
              grown_selective_state_space::fixed_physical_slot(
                  grown_selective_state_space::pair_index(
                      grown_selective_state_space::kGlobalBase,
                      grown_selective_state_space::kLayoutVersion, 0u))) !=
          grown_selective_state_space::kLayoutVersionValue) {
        throw std::runtime_error(
            "grown-adult checkpoint resident recurrent layout unsupported");
      }
      adult->selective_state_layout_ =
          grown_selective_state_space::connect_resident_layout(
              adult->boundary_port_slot(kRawSensoryZeroPort), adult->boundary_port_slot(kRawSensoryOnePort),
              adult->boundary_port_slot(kRawMotorZeroPort), adult->boundary_port_slot(kRawMotorOnePort),
              adult->instance_basin_layout_);
      const std::uint64_t motor_slots[]{adult->selective_state_layout_.raw_motor_zero_slot,
                                        adult->selective_state_layout_.raw_motor_one_slot};
      adult->include_physical_support(motor_slots);
      adult->selective_state_factor_enabled_ = true;
      adult->ensure_factor_advanced();
      adult->ensure_selective_inverse_scratch();
      adult->require_selective_state_checkpoint_valid();
    }
    if (adult->executor_.read_word(
            grown_sparse_event_memory::fixed_physical_slot(
                grown_sparse_event_memory::pair_index(
                    grown_sparse_event_memory::kGlobalBase,
                    grown_sparse_event_memory::kFactorMarker, 0u))) ==
        grown_sparse_event_memory::kFactorMarkerValue) {
      // The former exclusion against selective_state_factor_enabled_ was a
      // workaround for reverse() unwinding resident factors in forward (not
      // LIFO) order; see the comment in configure_selective_state_factor()
      // above. reverse() now unwinds LIFO, so selective_state and
      // sparse_event may coexist across a checkpoint round trip too.
      // sensorimotor and form_credit are untouched by that fix and must
      // still be refused here.
      if (adult->sensorimotor_factor_enabled_ ||
          adult->form_credit_factor_enabled_) {
        throw std::runtime_error(
            "checkpoint has multiple owners of the raw-motor aperture");
      }
      if (!adult->instance_basin_factor_enabled_)
        throw std::runtime_error(
            "checkpoint sparse event memory lacks situation tissue");
      if (adult->executor_.read_word(
              grown_sparse_event_memory::fixed_physical_slot(
                  grown_sparse_event_memory::pair_index(
                      grown_sparse_event_memory::kGlobalBase,
                      grown_sparse_event_memory::kLayoutVersion, 0u))) !=
          grown_sparse_event_memory::kLayoutVersionValue) {
        throw std::runtime_error(
            "grown-adult checkpoint sparse event layout unsupported");
      }
      adult->sparse_event_memory_layout_ =
          grown_sparse_event_memory::connect_resident_layout(
              adult->boundary_port_slot(kRawSensoryZeroPort),
              adult->boundary_port_slot(kRawSensoryOnePort),
              adult->boundary_port_slot(kRawMotorZeroPort),
              adult->boundary_port_slot(kRawMotorOnePort),
              adult->instance_basin_layout_);
      const std::uint64_t motor_slots[]{
          adult->sparse_event_memory_layout_.raw_motor_zero_slot,
          adult->sparse_event_memory_layout_.raw_motor_one_slot};
      adult->include_physical_support(motor_slots);
      adult->sparse_event_memory_factor_enabled_ = true;
      adult->ensure_factor_advanced();
      adult->ensure_sparse_event_memory_inverse_scratch();
      adult->require_sparse_event_memory_checkpoint_valid();
    }
    adult->boundary_ = header.boundary;
    require_cuda(
        cudaMemcpy(adult->boundary_words_.device,
                   header.boundary_words.data(),
                   header.boundary_words.size() * sizeof(SiteWord),
                   cudaMemcpyHostToDevice),
        "restore grown-adult boundary");
    Snapshot restored = adult->snapshot();
    if (bound_state_hash(restored, adult->founder_snapshot_,
                         adult->founder_hash_,
                         adult->law_identity_,
                         adult->genesis_manifest_identity_,
                         adult->genesis_recipe_kind_,
                         adult->genesis_manifest_entry_count_) !=
        header.state_hash) {
      throw std::runtime_error("grown-adult checkpoint hash mismatch");
    }
    if (developmental_adult::boundary_hash(adult->boundary_snapshot()) !=
        header.boundary_hash) {
      throw std::runtime_error(
          "grown-adult checkpoint boundary hash mismatch");
    }
    return adult;
  }

 private:
  void ensure_factor_advanced() {
    if (factor_advanced_device_ == nullptr) {
      require_cuda(cudaMalloc(&factor_advanced_device_, sizeof(std::uint32_t)),
                   "allocate factor advancement receipt");
    }
    if (resident_association_outcome_device_ == nullptr) {
      require_cuda(cudaMalloc(&resident_association_outcome_device_,
                              sizeof(*resident_association_outcome_device_)),
                   "allocate resident association outcome");
    }
  }

  // Two counters and one identity fold, allocated once. A per-transaction
  // cudaMalloc inside a boundary exchange would put an allocator on the hot path
  // §15 measures.
  void ensure_contact_census_scratch() {
    if (contact_census_scratch_ == nullptr) {
      require_cuda(cudaMalloc(&contact_census_scratch_,
                              sizeof(unsigned int) * 4u +
                                  sizeof(unsigned long long)),
                   "allocate grown-adult contact stage census scratch");
    }
  }

  void ensure_selective_inverse_scratch() {
    if (selective_state_inverse_scratch_ == nullptr) {
      require_cuda(cudaMalloc(&selective_state_inverse_scratch_,
                              sizeof(*selective_state_inverse_scratch_)),
                   "allocate resident recurrent inverse scratch");
    }
  }

  void ensure_sparse_event_memory_inverse_scratch() {
    if (sparse_event_memory_inverse_scratch_ == nullptr) {
      require_cuda(cudaMalloc(&sparse_event_memory_inverse_scratch_,
                              sizeof(*sparse_event_memory_inverse_scratch_)),
                   "allocate sparse event memory inverse scratch");
    }
  }

  std::uint32_t require_factor_advanced() const {
    std::uint32_t advanced = 0u;
    require_cuda(cudaMemcpy(&advanced, factor_advanced_device_,
                            sizeof(advanced), cudaMemcpyDeviceToHost),
                 "copy factor advancement receipt");
    if ((advanced & kResidentStepSucceeded) == 0u) {
      throw std::runtime_error(
          "resident factor journal exhausted before adult tick");
    }
    return advanced;
  }

  void require_sensorimotor_checkpoint_boundary() const {
    if (!sensorimotor_factor_enabled_) return;
    if (sensorimotor_inputs_ != nullptr) {
      grown_sensorimotor_factor::DeviceInputs inputs{};
      require_cuda(cudaMemcpy(&inputs, sensorimotor_inputs_, sizeof(inputs),
                              cudaMemcpyDeviceToHost),
                   "inspect sensorimotor checkpoint boundary");
      if (inputs.staged != 0u || inputs.transform_mode != 0u) {
        throw std::logic_error(
            "sensorimotor checkpoint requires an idle contact boundary");
      }
    }
    const SiteWord eligibility = executor_.read_word(
        sensorimotor_factor_layout_
            .rails[grown_sensorimotor_factor::value_index(
                grown_sensorimotor_factor::kEligibility)]);
    const SiteWord motor = executor_.read_word(
        sensorimotor_factor_layout_
            .rails[grown_sensorimotor_factor::value_index(
                grown_sensorimotor_factor::kMotor)]);
    if (eligibility != 0u || motor != 0u) {
      throw std::logic_error(
          "sensorimotor checkpoint cannot capture eligible action");
    }
  }

  void require_readout_checkpoint_boundary() const {
    if (!readout_factor_enabled_ || readout_inputs_ == nullptr) return;
    resident_readout_f_route::DeviceInputs inputs{};
    require_cuda(cudaMemcpy(&inputs, readout_inputs_, sizeof(inputs),
                            cudaMemcpyDeviceToHost),
                 "inspect resident readout F checkpoint boundary");
    if (inputs.staged != 0u) {
      throw std::logic_error(
          "resident readout F checkpoint requires an idle contact boundary");
    }
  }

  void require_selective_state_checkpoint_boundary() const {
    if (!selective_state_factor_enabled_ || selective_state_inputs_ == nullptr)
      return;
    grown_selective_state_space::DeviceInputs inputs{};
    require_cuda(cudaMemcpy(&inputs, selective_state_inputs_, sizeof(inputs),
                            cudaMemcpyDeviceToHost),
                 "inspect resident recurrent checkpoint boundary");
    if (inputs.predict_staged != 0u || inputs.observe_staged != 0u) {
      throw std::logic_error(
          "resident recurrent checkpoint requires an idle contact boundary");
    }
  }

  void require_selective_state_checkpoint_valid() const {
    if (!selective_state_factor_enabled_) return;
    const SiteWord journal_count = executor_.read_word(
        grown_selective_state_space::fixed_physical_slot(
            grown_selective_state_space::pair_index(
                grown_selective_state_space::kGlobalBase,
                grown_selective_state_space::kJournalCount, 0u)));
    const std::size_t witnessed = static_cast<std::size_t>(std::count_if(
        resident_step_history_.begin(), resident_step_history_.end(),
        [](std::uint32_t step) {
          return (step & kResidentStepSelectiveState) != 0u;
        }));
    if (journal_count > grown_selective_state_space::kJournalDepth ||
        witnessed != static_cast<std::size_t>(journal_count)) {
      throw std::runtime_error(
          "resident recurrent checkpoint history/journal mismatch");
    }
    const grown_selective_state_space::ValidationReceipt host =
        grown_selective_state_space::validate_resident(
            executor_.device_words());
    if (host.marker_valid == 0u || host.version_valid == 0u ||
        host.journal_valid == 0u || host.active_bank_valid == 0u ||
        host.fifo_valid == 0u || host.invalid_pairs != 0u ||
        host.invalid_journal_events != 0u ||
        host.invalid_eligibility != 0u || host.invalid_ledger != 0u) {
      throw std::runtime_error(
          "resident recurrent checkpoint invariant failure");
    }
  }

  void require_sparse_event_memory_checkpoint_boundary() const {
    if (!sparse_event_memory_factor_enabled_ ||
        sparse_event_memory_inputs_ == nullptr)
      return;
    grown_sparse_event_memory::DeviceInputs inputs{};
    require_cuda(cudaMemcpy(&inputs, sparse_event_memory_inputs_, sizeof(inputs),
                            cudaMemcpyDeviceToHost),
                 "inspect sparse event checkpoint boundary");
    if (inputs.predict_staged != 0u || inputs.observe_staged != 0u) {
      throw std::logic_error(
          "sparse event checkpoint requires an idle contact boundary");
    }
  }

  void require_sparse_event_memory_checkpoint_valid() const {
    if (!sparse_event_memory_factor_enabled_) return;
    const SiteWord journal_count = executor_.read_word(
        grown_sparse_event_memory::fixed_physical_slot(
            grown_sparse_event_memory::pair_index(
                grown_sparse_event_memory::kGlobalBase,
                grown_sparse_event_memory::kJournalCount, 0u)));
    const std::size_t witnessed = static_cast<std::size_t>(std::count_if(
        resident_step_history_.begin(), resident_step_history_.end(),
        [](std::uint32_t step) {
          return (step & kResidentStepSparseEventMemory) != 0u;
        }));
    if (journal_count > grown_sparse_event_memory::kJournalDepth ||
        witnessed != static_cast<std::size_t>(journal_count)) {
      throw std::runtime_error(
          "sparse event checkpoint history/journal mismatch");
    }
    const grown_sparse_event_memory::ValidationReceipt host =
        grown_sparse_event_memory::validate_resident(executor_.device_words());
    if (host.marker_valid == 0u || host.version_valid == 0u ||
        host.journal_valid == 0u || host.form_valid == 0u ||
        host.transition_valid == 0u || host.eligibility_valid == 0u ||
        host.invalid_pairs != 0u || host.invalid_journal_events != 0u ||
        host.invalid_eligibility != 0u) {
      throw std::runtime_error("sparse event checkpoint invariant failure");
    }
  }

  void upload_cloud_factor_layout() {
    if (cloud_factor_layout_device_ == nullptr) {
      require_cuda(cudaMalloc(&cloud_factor_layout_device_,
                              sizeof(cloud_factor_layout_)),
                   "allocate cloud factor layout");
    }
    require_cuda(cudaMemcpy(cloud_factor_layout_device_, &cloud_factor_layout_,
                            sizeof(cloud_factor_layout_),
                            cudaMemcpyHostToDevice),
                 "upload cloud factor layout");
  }

  void upload_instance_basin_layout() {
    if (instance_basin_layout_device_ == nullptr) {
      require_cuda(cudaMalloc(&instance_basin_layout_device_,
                              sizeof(instance_basin_layout_)),
                   "allocate instance basin layout");
    }
    require_cuda(cudaMemcpy(instance_basin_layout_device_,
                            &instance_basin_layout_,
                            sizeof(instance_basin_layout_),
                            cudaMemcpyHostToDevice),
                 "upload instance basin layout");
  }

  void upload_edge_bank_layout() {
    if (edge_bank_layout_device_ == nullptr) {
      require_cuda(cudaMalloc(&edge_bank_layout_device_,
                              sizeof(edge_bank_layout_)),
                   "allocate edge bank layout");
    }
    require_cuda(cudaMemcpy(edge_bank_layout_device_, &edge_bank_layout_,
                            sizeof(edge_bank_layout_),
                            cudaMemcpyHostToDevice),
                 "upload edge bank layout");
  }

  void upload_sensorimotor_factor_layout() {
    if (sensorimotor_factor_layout_device_ == nullptr) {
      require_cuda(cudaMalloc(&sensorimotor_factor_layout_device_,
                              sizeof(sensorimotor_factor_layout_)),
                   "allocate sensorimotor factor layout");
    }
    require_cuda(cudaMemcpy(sensorimotor_factor_layout_device_,
                            &sensorimotor_factor_layout_,
                            sizeof(sensorimotor_factor_layout_),
                            cudaMemcpyHostToDevice),
                 "upload sensorimotor factor layout");
  }

  void upload_readout_factor_layout() {
    if (readout_factor_layout_device_ == nullptr) {
      require_cuda(cudaMalloc(&readout_factor_layout_device_,
                              sizeof(readout_factor_layout_)),
                   "allocate resident readout F layout");
    }
    require_cuda(cudaMemcpy(readout_factor_layout_device_,
                            &readout_factor_layout_,
                            sizeof(readout_factor_layout_),
                            cudaMemcpyHostToDevice),
                 "upload resident readout F layout");
  }

  void upload_form_credit_layout() {
    if (form_credit_layout_device_ == nullptr) {
      require_cuda(cudaMalloc(&form_credit_layout_device_,
                              sizeof(form_credit_layout_)),
                   "allocate form credit layout");
    }
    require_cuda(cudaMemcpy(form_credit_layout_device_, &form_credit_layout_,
                            sizeof(form_credit_layout_),
                            cudaMemcpyHostToDevice),
                 "upload form credit layout");
  }

  static void require_distinct_boundary_pair(std::uint32_t zero_index,
                                             std::uint32_t one_index) {
    if (zero_index >= kBoundaryWordCount ||
        one_index >= kBoundaryWordCount) {
      throw std::out_of_range("grown-adult raw-byte boundary index invalid");
    }
    if (zero_index == one_index) {
      throw std::invalid_argument(
          "grown-adult raw-byte rails require distinct boundary words");
    }
  }

  // Exchange one exact byte frame between two fixed physical adult words and
  // an explicit dual-rail boundary tape. Public entry points fix the physical
  // sensory and motor topology; callers cannot select resident addresses.
  // ⭐ FULL REQUIREMENT-3 RECEIPT, second of four transactions.
  //
  // A raw byte crosses on TWO rails, so `port` names the zero rail and
  // before/after carry the DECODED BYTE rather than one rail's word -- the byte
  // is the matter that crossed, and reporting a single rail would describe half
  // of it. Conservation is over the pair of rails on both sides, for the same
  // reason entry 170 had to correct it: a reciprocal exchange leaves neither
  // owner individually invariant.
  MembraneReceipt exchange_boundary_raw_byte(std::uint32_t zero_index,
                                             std::uint32_t one_index) {
    require_distinct_boundary_pair(zero_index, one_index);
    const std::array<std::uint64_t, 2> slots{
        boundary_port_slot(zero_index), boundary_port_slot(one_index)};
    MembraneReceipt receipt;
    receipt.epoch = completed_ticks_;
    receipt.port = zero_index;
    std::array<SiteWord, 2> rails_before{};
    require_cuda(cudaMemcpy(&rails_before[0], boundary_words_.device + zero_index,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte zero rail before exchange");
    require_cuda(cudaMemcpy(&rails_before[1], boundary_words_.device + one_index,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte one rail before exchange");
    const RawByteDecode decoded_before =
        decode_raw_byte_faces({rails_before[0], rails_before[1]});
    receipt.before = decoded_before.valid ? decoded_before.value : 0u;
    std::array<SiteWord, 2> resident_before{};
    require_cuda(cudaMemcpy(&resident_before[0], executor_.device_words() + slots[0],
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte resident zero slot before exchange");
    require_cuda(cudaMemcpy(&resident_before[1], executor_.device_words() + slots[1],
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte resident one slot before exchange");
    ++boundary_transactions_;
    SiteWord* world_words = executor_.mutable_device_words();
    std::uint64_t zero_slot = slots[0];
    std::uint64_t one_slot = slots[1];
    SiteWord* boundary_words = boundary_words_.device;
    std::uint32_t zero_boundary_index = zero_index;
    std::uint32_t one_boundary_index = one_index;
    void* exchange_args[] = {&world_words,
                             &zero_slot,
                             &one_slot,
                             &boundary_words,
                             &zero_boundary_index,
                             &one_boundary_index};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(exchange_boundary_raw_byte_kernel),
                     dim3(1u, 1u, 1u), dim3(2u, 1u, 1u), exchange_args, 0u,
                     nullptr),
                 "launch grown-adult raw-byte boundary exchange");
    require_cuda(cudaGetLastError(),
                 "launch grown-adult raw-byte boundary exchange");
    require_cuda(cudaDeviceSynchronize(),
                 "synchronize grown-adult raw-byte boundary exchange");
    reconcile_external_contacts(slots);
    std::array<SiteWord, 2> rails_after{};
    require_cuda(cudaMemcpy(&rails_after[0], boundary_words_.device + zero_index,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte zero rail after exchange");
    require_cuda(cudaMemcpy(&rails_after[1], boundary_words_.device + one_index,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte one rail after exchange");
    const RawByteDecode decoded_after =
        decode_raw_byte_faces({rails_after[0], rails_after[1]});
    receipt.after = decoded_after.valid ? decoded_after.value : 0u;
    // ⛔ THE PAIR AGAIN. The first version summed only the two RAILS and reported
    // conserved=0 on every correct exchange -- before=255 after=0 -- because the
    // byte crosses INTO the organism and the rails legitimately lose it. This is
    // the identical error entry 170 corrected for exchange_boundary_word, made
    // again two entries later: one side of a two-owner exchange is never
    // invariant. The resident slots are the other side.
    std::array<SiteWord, 2> resident_after{};
    require_cuda(cudaMemcpy(&resident_after[0], executor_.device_words() + slots[0],
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte resident zero slot after exchange");
    require_cuda(cudaMemcpy(&resident_after[1], executor_.device_words() + slots[1],
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult raw-byte resident one slot after exchange");
    receipt.conserved =
        __builtin_popcount(rails_before[0]) + __builtin_popcount(rails_before[1]) +
            __builtin_popcount(resident_before[0]) +
            __builtin_popcount(resident_before[1]) ==
        __builtin_popcount(rails_after[0]) + __builtin_popcount(rails_after[1]) +
            __builtin_popcount(resident_after[0]) +
            __builtin_popcount(resident_after[1]);
    // pairs stays 1: this transaction names exactly one declared port, and
    // `declaration` repeats it so a consumer reads one field either way.
    receipt.declaration_fingerprint = receipt.port;
    receipt.transaction = boundary_transactions_;
    return receipt;
  }

  [[nodiscard]] RawByteDecode boundary_raw_byte(
      std::uint32_t zero_index, std::uint32_t one_index) const {
    require_distinct_boundary_pair(zero_index, one_index);
    const BoundarySnapshot boundary = boundary_snapshot();
    return decode_raw_byte_faces(
        {boundary.words[zero_index], boundary.words[one_index]});
  }

  struct RestoreTag {};
  inline static constexpr RestoreTag kRestoreTag{};

  explicit GrownAdult(RestoreTag)
      : aperture_edge_chunks_(kApertureEdgeChunks),
        map_(make_cube(aperture_edge_chunks_)),
        executor_(CudaBcc32Executor::testing(
            static_cast<std::uint32_t>(map_.host.size()))),
        founder_hash_(kFounderHash),
        law_identity_(canonical_law_identity()),
        genesis_manifest_identity_(genesis_manifest_address(
            "g1-developmental-hash-v1",
            developmental_hash_manifest(kFounderHash), map_.host)),
        genesis_manifest_entry_count_(kGermSiteCount) {
    executor_.initialize_q();
    initialize_boundary();
  }

  void mint() {
    executor_.initialize_q();
    initialize_boundary();
    const auto manifest = developmental_hash_manifest(founder_hash_, variant_);
    std::vector<std::uint64_t> seeds;
    seeds.reserve(manifest.size());
    for (const GenesisManifestEntry& entry : manifest) {
      const std::uint64_t slot = physical_slot(entry.relative);
      executor_.write_word(slot, entry.word);
      seeds.push_back(slot);
    }
    support_ = std::make_unique<CudaBcc32ActiveSupport>(
        executor_, map_.host, seeds, seeds);
    founder_snapshot_ = snapshot();
  }

  void sample_residency() {
    const std::vector<std::uint64_t>& slots = support_->active_slots();
    const std::vector<SiteWord> words = support_->download_active_words();
    for (std::size_t index = 0; index < slots.size(); ++index) {
      if (words[index] != 0u && words[index] != kQ) {
        ++residency_hits_[slots[index]];
      }
    }
  }

  void initialize_boundary() {
    if (boundary_words_.device == nullptr) {
      require_cuda(cudaMalloc(&boundary_words_.device,
                              kBoundaryWordCount * sizeof(SiteWord)),
                   "allocate grown-adult boundary");
    }
    std::array<SiteWord, kBoundaryWordCount> empty{};
    empty.fill(kQ);
    require_cuda(cudaMemcpy(boundary_words_.device, empty.data(),
                            empty.size() * sizeof(SiteWord),
                            cudaMemcpyHostToDevice),
                 "initialize grown-adult boundary");
  }

  std::int64_t aperture_edge_chunks_ = kApertureEdgeChunks;
  HostChunkMap map_;
  CudaBcc32Executor executor_;
  std::uint64_t boundary_transactions_ = 0u;
  std::uint64_t interventions_ = 0u;
  InterventionReceipt last_intervene_{};
  std::uint64_t resident_stages_ = 0u;
  ResidentStageReceipt last_resident_stage_{};
  std::unique_ptr<CudaBcc32ActiveSupport> support_;
  DevelopmentalHash founder_hash_ = kFounderHash;
  FounderVariant variant_ = FounderVariant::intact;
  ContentAddress law_identity_{};
  ContentAddress genesis_manifest_identity_{};
  GenesisRecipeKind genesis_recipe_kind_ =
      GenesisRecipeKind::developmental_hash;
  std::uint64_t genesis_manifest_entry_count_ = 0u;
  std::uint64_t completed_ticks_ = 0u;
  Snapshot founder_snapshot_;
  std::map<std::uint64_t, int> residency_hits_;
  BoundaryCounters boundary_{};
  bool cloud_factor_enabled_ = false;
  grown_cloud_factor::DeviceLayout cloud_factor_layout_{};
  grown_cloud_factor::DeviceLayout* cloud_factor_layout_device_ = nullptr;
  grown_cloud_factor::ContactReceipt* cloud_factor_receipt_ = nullptr;
  bool instance_basin_factor_enabled_ = false;
  grown_instance_basin_factor::DeviceLayout instance_basin_layout_{};
  grown_instance_basin_factor::DeviceLayout*
      instance_basin_layout_device_ = nullptr;
  grown_instance_basin_factor::DeviceInputs* instance_basin_inputs_ = nullptr;
  grown_instance_basin_factor::StepReceipt* instance_basin_receipt_ = nullptr;
  bool sensorimotor_factor_enabled_ = false;
  grown_sensorimotor_factor::DeviceLayout sensorimotor_factor_layout_{};
  grown_sensorimotor_factor::DeviceLayout*
      sensorimotor_factor_layout_device_ = nullptr;
  grown_sensorimotor_factor::DeviceInputs* sensorimotor_inputs_ = nullptr;
  grown_sensorimotor_factor::PredictionReceipt* sensorimotor_prediction_ =
      nullptr;
  grown_sensorimotor_factor::ConsequenceReceipt* sensorimotor_consequence_ =
      nullptr;
  grown_sensorimotor_factor::TransformReceipt* sensorimotor_transform_ =
      nullptr;
  bool readout_factor_enabled_ = false;
  resident_readout_f_route::DeviceLayout readout_factor_layout_{};
  resident_readout_f_route::DeviceLayout* readout_factor_layout_device_ =
      nullptr;
  resident_readout_f_route::DeviceInputs* readout_inputs_ = nullptr;
  resident_readout_f_route::CreditReceipt* readout_receipt_ = nullptr;
  bool form_credit_factor_enabled_ = false;
  grown_form_credit_factor::DeviceLayout form_credit_layout_{};
  grown_form_credit_factor::DeviceLayout* form_credit_layout_device_ = nullptr;
  grown_form_credit_factor::DeviceInputs* form_credit_inputs_ = nullptr;
  grown_form_credit_factor::Receipt* form_credit_receipt_ = nullptr;
  bool selective_state_factor_enabled_ = false;
  grown_selective_state_space::DeviceLayout selective_state_layout_{};
  grown_selective_state_space::DeviceInputs* selective_state_inputs_ = nullptr;
  grown_selective_state_space::DeviceScratch* selective_state_scratch_ =
      nullptr;
  grown_selective_state_space::Receipt* selective_state_receipt_ = nullptr;
  grown_selective_state_space::InverseScratch*
      selective_state_inverse_scratch_ = nullptr;
  bool sparse_event_memory_factor_enabled_ = false;
  grown_sparse_event_memory::DeviceLayout sparse_event_memory_layout_{};
  grown_sparse_event_memory::DeviceInputs* sparse_event_memory_inputs_ = nullptr;
  grown_sparse_event_memory::DeviceScratch* sparse_event_memory_scratch_ =
      nullptr;
  grown_sparse_event_memory::Receipt* sparse_event_memory_receipt_ = nullptr;
  grown_sparse_event_memory::InverseScratch*
      sparse_event_memory_inverse_scratch_ = nullptr;
  std::uint32_t* factor_advanced_device_ = nullptr;
  void* contact_census_scratch_ = nullptr;
  std::vector<std::uint32_t> resident_step_history_;
  AssociationOutcome* resident_association_outcome_device_ = nullptr;
  std::vector<AssociationOutcome> resident_association_history_;
  bool edge_bank_factor_enabled_ = false;
  bool production_timeline_claimed_ = false;
  resident_edge_bank::DeviceLayout edge_bank_layout_{};
  resident_edge_bank::DeviceLayout* edge_bank_layout_device_ = nullptr;
  DeviceBoundaryWords boundary_words_{};
