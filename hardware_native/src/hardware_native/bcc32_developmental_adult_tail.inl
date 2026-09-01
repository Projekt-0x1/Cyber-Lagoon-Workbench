class GrownAdult {
 public:
  explicit GrownAdult(
      DevelopmentalHash founder_hash = kFounderHash,
      FounderVariant variant = FounderVariant::intact,
      std::int64_t aperture_edge_chunks = kApertureEdgeChunks)
      : aperture_edge_chunks_(aperture_edge_chunks),
        map_(make_cube(aperture_edge_chunks_)),
        executor_(CudaBcc32Executor::testing(
            static_cast<std::uint32_t>(map_.host.size()))),
        founder_hash_(founder_hash),
        variant_(variant),
        law_identity_(canonical_law_identity()),
        genesis_manifest_identity_(genesis_manifest_address(
            "g1-developmental-hash-v1",
            developmental_hash_manifest(founder_hash_, variant_), map_.host)),
        genesis_manifest_entry_count_(kGermSiteCount) {
    if (aperture_edge_chunks_ <= 0 ||
        aperture_edge_chunks_ > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("grown-adult aperture edge is invalid");
    }
    mint();
  }

  [[nodiscard]] static std::unique_ptr<GrownAdult> from_genesis_manifest(
      std::span<const GenesisManifestEntry> entries,
      std::string_view recipe_id) {
    if (recipe_id.empty() || entries.empty()) {
      throw std::invalid_argument(
          "grown-adult explicit Genesis manifest must name a nonempty recipe");
    }
    std::vector<GenesisManifestEntry> ordered(entries.begin(), entries.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const GenesisManifestEntry& left,
                 const GenesisManifestEntry& right) {
                if (left.relative.x != right.relative.x)
                  return left.relative.x < right.relative.x;
                if (left.relative.y != right.relative.y)
                  return left.relative.y < right.relative.y;
                return left.relative.z < right.relative.z;
              });
    auto adult = std::unique_ptr<GrownAdult>(new GrownAdult(kRestoreTag));
    std::vector<StateEntry> state;
    state.reserve(ordered.size());
    for (std::size_t index = 0u; index < ordered.size(); ++index) {
      const GenesisManifestEntry& entry = ordered[index];
      if (entry.word == kQ ||
          (index > 0u && ordered[index - 1u].relative == entry.relative)) {
        throw std::invalid_argument(
            "grown-adult explicit Genesis manifest contains Q or overlap");
      }
      state.push_back(
          {adult->physical_slot(entry.relative), entry.word});
    }
    std::sort(state.begin(), state.end(),
              [](const StateEntry& left, const StateEntry& right) {
                return left.slot < right.slot;
              });
    for (std::size_t index = 1u; index < state.size(); ++index)
      if (state[index - 1u].slot == state[index].slot)
        throw std::invalid_argument(
            "grown-adult explicit Genesis manifest aliases physical slots");

    StateEntry* device_entries = nullptr;
    require_cuda(cudaMalloc(&device_entries, state.size() * sizeof(StateEntry)),
                 "allocate explicit Genesis manifest scatter");
    require_cuda(cudaMemcpy(device_entries, state.data(),
                            state.size() * sizeof(StateEntry),
                            cudaMemcpyHostToDevice),
                 "upload explicit Genesis manifest scatter");
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        std::min<std::size_t>((state.size() + 255u) / 256u, 4096u));
    scatter_state_entries_kernel<<<blocks, 256>>>(
        adult->executor_.mutable_device_words(), device_entries, state.size());
    require_cuda(cudaGetLastError(),
                 "launch explicit Genesis manifest scatter");
    require_cuda(cudaDeviceSynchronize(),
                 "synchronize explicit Genesis manifest scatter");
    require_cuda(cudaFree(device_entries),
                 "release explicit Genesis manifest scatter");
    std::vector<std::uint64_t> slots;
    slots.reserve(state.size());
    for (const StateEntry& entry : state) slots.push_back(entry.slot);
    adult->support_ = std::make_unique<CudaBcc32ActiveSupport>(
        adult->executor_, adult->map_.host, slots);
    adult->genesis_recipe_kind_ = GenesisRecipeKind::explicit_manifest;
    adult->genesis_manifest_identity_ =
        genesis_manifest_address(recipe_id, ordered,
                                 adult->map_.host);
    adult->genesis_manifest_entry_count_ = ordered.size();
    adult->founder_snapshot_ = adult->snapshot();
    return adult;
  }

  GrownAdult(const GrownAdult&) = delete;
  GrownAdult& operator=(const GrownAdult&) = delete;
  ~GrownAdult() {
    (void)cudaFree(sensorimotor_factor_layout_device_);
    (void)cudaFree(cloud_factor_layout_device_);
    (void)cudaFree(instance_basin_layout_device_);
    (void)cudaFree(readout_factor_layout_device_);
    (void)cudaFree(form_credit_layout_device_);
    (void)cudaFree(selective_state_inverse_scratch_);
    (void)cudaFree(sparse_event_memory_inverse_scratch_);
    (void)cudaFree(factor_advanced_device_);
    (void)cudaFree(contact_census_scratch_);
    (void)cudaFree(resident_association_outcome_device_);
    (void)cudaFree(edge_bank_layout_device_);
  }

  [[nodiscard]] std::unique_ptr<
      device_ordinary_f_timeline::DeviceOrdinaryFTimeline>
  claim_device_ordinary_f_timeline(
      std::uint32_t capacity = device_ordinary_f_timeline::kCapacity,
      std::uint32_t history_capacity = device_ordinary_f_timeline::kHistoryCapacity) {
    if (production_timeline_claimed_)
      throw std::logic_error(
          "grown adult ordinary-F production timeline already claimed");
    if (completed_ticks_ != 0u || cloud_factor_enabled_ ||
        instance_basin_factor_enabled_ || sensorimotor_factor_enabled_ ||
        readout_factor_enabled_ || form_credit_factor_enabled_ ||
        selective_state_factor_enabled_ ||
        sparse_event_memory_factor_enabled_ ||
        edge_bank_factor_enabled_)
      throw std::logic_error(
          "ordinary-F production ownership requires an untouched hash-grown world");
    device_ordinary_f_timeline::DeviceBoundaryBinding boundary{};
    boundary.boundary_words = boundary_words_.device;
    boundary.sensory_zero_boundary = kRawSensoryZeroPort;
    boundary.sensory_one_boundary = kRawSensoryOnePort;
    boundary.sensory_zero_slot = boundary_port_slot(kRawSensoryZeroPort);
    boundary.sensory_one_slot = boundary_port_slot(kRawSensoryOnePort);
    boundary.motor_zero_slot = boundary_port_slot(kRawMotorZeroPort);
    boundary.motor_one_slot = boundary_port_slot(kRawMotorOnePort);
    auto timeline = std::make_unique<
        device_ordinary_f_timeline::DeviceOrdinaryFTimeline>(
        executor_, map_.view(), boundary, capacity, history_capacity);
    production_timeline_claimed_ = true;
    return timeline;
  }

  void develop(std::uint64_t ticks) {
    if (production_timeline_claimed_)
      throw std::logic_error(
          "host development is disabled after ordinary-F production claim");
    resident_step_history_.reserve(resident_step_history_.size() + ticks);
    resident_association_history_.reserve(resident_association_history_.size() +
                                          ticks);
    for (std::uint64_t tick = 0u; tick < ticks; ++tick) {
      // KNOWN UNFIXED, measured 2026-08-04. Making this apply_superstep()
      // call unconditional (composing ordinary F with every resident-factor
      // tick, not just the non-resident one) fails in TWO independent ways.
      //
      // 1. F OVERWRITES RESIDENT STORAGE. A single unconditional call takes
      //    instance_basin's journal count 0 -> 134283663 and sparse event
      //    memory's 0 -> 151060879 in ONE tick, because those factors' fixed
      //    slots sit inside the active support that F transforms. The step
      //    then reports "resident factor journal exhausted before adult
      //    tick" -- a MISNOMER: nothing filled up, journal_available() is
      //    comparing a clobbered word against kJournalDepth. Gated, the same
      //    counts rise legitimately (ib to 4, sem to 111 over 183 ticks).
      //
      // 2. CLOSURE IS CLOSE BUT NOT MET. Relocating every resident region
      //    into the [26,473] halo window (d6164a20f4, 69a8ff3ee6) moved the
      //    worst margin from depth 10 to DEPTH 25 against
      //    kSpatialMacroClosureRadius=26 -- one hop short. An earlier commit
      //    message overstated this as fixed; it is not.
      //    NB the payoff test is sensitive to which paths you hoist: hoisting
      //    the forward path alone hides (2) behind (1). Hoist both this call
      //    and apply_superstep_inverse, and say which you did.
      //
      // Raising kApertureEdgeChunks to 8 satisfies closure but OOMs the P
      // carrier snapshot even while gated. Every fixed_physical_slot() now
      // reads edge_chunks from the shared bcc32_aperture_geometry.cuh
      // definition, so raising it would no longer silently address the
      // wrong geometry -- but the OOM above remains. The remaining fix is
      // separating growing developmental support from fixed-address
      // resident storage in the closure-scanned active set, so ordinary F
      // cannot transform resident words.
      if (cloud_factor_enabled_ || instance_basin_factor_enabled_ ||
          sensorimotor_factor_enabled_ ||
          readout_factor_enabled_ || form_credit_factor_enabled_ ||
          selective_state_factor_enabled_ ||
          sparse_event_memory_factor_enabled_ || edge_bank_factor_enabled_) {
        combined_resident_factors_step_kernel<<<1, 1>>>(
            executor_.mutable_device_words(), cloud_factor_enabled_,
            cloud_factor_layout_device_, cloud_factor_receipt_,
            instance_basin_factor_enabled_, instance_basin_layout_device_,
            instance_basin_inputs_, instance_basin_receipt_,
            sensorimotor_factor_enabled_, sensorimotor_factor_layout_device_,
            boundary_port_slot(kRawMotorZeroPort),
            boundary_port_slot(kRawMotorOnePort),
            sensorimotor_inputs_, sensorimotor_prediction_,
            sensorimotor_consequence_, sensorimotor_transform_,
            readout_factor_enabled_, readout_factor_layout_device_,
            readout_inputs_, readout_receipt_, form_credit_factor_enabled_,
            form_credit_layout_device_, form_credit_inputs_, form_credit_receipt_,
            selective_state_factor_enabled_, selective_state_inputs_,
            sparse_event_memory_factor_enabled_, sparse_event_memory_inputs_,
            edge_bank_factor_enabled_, edge_bank_layout_device_,
            factor_advanced_device_, resident_association_outcome_device_);
        require_cuda(cudaGetLastError(),
                     "launch combined resident-factor superstep");
        if (selective_state_factor_enabled_) {
          grown_selective_state_space::launch_step(
              executor_.mutable_device_words(), selective_state_layout_,
              selective_state_inputs_, selective_state_scratch_,
              selective_state_receipt_, factor_advanced_device_,
              kResidentStepSelectiveState);
          require_cuda(cudaGetLastError(),
                       "launch resident recurrent superstep");
        }
        if (sparse_event_memory_factor_enabled_) {
          grown_sparse_event_memory::launch_step(
              executor_.mutable_device_words(), sparse_event_memory_layout_,
              sparse_event_memory_inputs_, sparse_event_memory_scratch_,
              sparse_event_memory_receipt_, factor_advanced_device_,
              kResidentStepSparseEventMemory);
          require_cuda(cudaGetLastError(),
                       "launch sparse event memory superstep");
        }
        require_cuda(cudaDeviceSynchronize(),
                     "synchronize combined resident-factor superstep");
        resident_step_history_.push_back(require_factor_advanced());
        // Recorded in lockstep with resident_step_history_ above (same
        // push-forward/pop-reverse lifecycle); see AssociationOutcome's
        // comment for why this cannot be recomputed later from state alone.
        // Deliberately NOT persisted across save_checkpoint/load_checkpoint
        // (unlike resident_step_history_) -- out of scope for this change;
        // a checkpoint-restored adult reversing pre-restore ticks will treat
        // this association as not-fired for those ticks.
        AssociationOutcome association_outcome{};
        if (resident_association_outcome_device_ != nullptr) {
          require_cuda(
              cudaMemcpy(&association_outcome,
                        resident_association_outcome_device_,
                        sizeof(association_outcome), cudaMemcpyDeviceToHost),
              "copy resident association outcome");
        }
        resident_association_history_.push_back(association_outcome);
      } else {
        support_->apply_superstep(map_.view());
      }
      ++completed_ticks_;
      if (completed_ticks_ > static_cast<std::uint64_t>(kAutonomousGrowthTicks) &&
          completed_ticks_ <=
              static_cast<std::uint64_t>(kCanonicalDevelopmentTicks)) {
        sample_residency();
      }
    }
  }

  void reverse(std::uint64_t ticks) {
    if (production_timeline_claimed_)
      throw std::logic_error(
          "host reversal is disabled after ordinary-F production claim");
    if (ticks > completed_ticks_) {
      throw std::invalid_argument(
          "cannot reverse grown adult before its genesis boundary");
    }
    for (std::uint64_t tick = 0u; tick < ticks; ++tick) {
      if (cloud_factor_enabled_ || instance_basin_factor_enabled_ ||
          sensorimotor_factor_enabled_ ||
          readout_factor_enabled_ || form_credit_factor_enabled_ ||
          selective_state_factor_enabled_ ||
          sparse_event_memory_factor_enabled_ || edge_bank_factor_enabled_) {
        std::uint32_t advancement = kResidentStepSucceeded;
        if (resident_step_history_.empty() &&
            (instance_basin_factor_enabled_ || form_credit_factor_enabled_ ||
             selective_state_factor_enabled_ ||
             sparse_event_memory_factor_enabled_)) {
          throw std::logic_error(
              "resident inverse lacks a factor-advancement witness");
        }
        if (!resident_step_history_.empty()) {
          advancement = resident_step_history_.back();
        }
        AssociationOutcome association_outcome{};
        if (!resident_association_history_.empty()) {
          association_outcome = resident_association_history_.back();
        }
        // Resident factors must unwind LIFO with respect to develop()'s
        // forward dispatch order (combined kernel, then selective_state,
        // then sparse_event -- see develop() above). sparse_event's inverse
        // therefore runs BEFORE selective_state's here: sparse_event
        // journaled the intermediate raw-motor value that selective_state's
        // own forward write produced, so sparse_event must restore that
        // intermediate first; only then does selective_state's restore land
        // on the true pre-tick value. Running them in forward order (as
        // before) left the raw-motor pair one factor's write short of exact
        // inverse whenever both were live. The combined inverse kernel below
        // stays last, mirroring its position first in the forward order.
        if (sparse_event_memory_factor_enabled_ &&
            (advancement & kResidentStepSparseEventMemory) != 0u) {
          grown_sparse_event_memory::launch_prepare_inverse(
              executor_.mutable_device_words(), sparse_event_memory_inverse_scratch_);
          require_cuda(cudaGetLastError(),
                       "prepare sparse event memory inverse");
          grown_sparse_event_memory::InverseScratch inverse{};
          require_cuda(cudaMemcpy(&inverse,
                                  sparse_event_memory_inverse_scratch_,
                                  sizeof(inverse), cudaMemcpyDeviceToHost),
                       "inspect sparse event memory inverse reservation");
          if (inverse.error != 0u || inverse.valid == 0u) {
            throw std::runtime_error(
                "sparse event memory inverse journal metadata invalid");
          }
          grown_sparse_event_memory::launch_restore_inverse(
              executor_.mutable_device_words(), sparse_event_memory_layout_,
              sparse_event_memory_inverse_scratch_);
          require_cuda(cudaGetLastError(),
                       "launch sparse event memory inverse");
        }
        if (selective_state_factor_enabled_ &&
            (advancement & kResidentStepSelectiveState) != 0u) {
          grown_selective_state_space::launch_prepare_inverse(
              executor_.mutable_device_words(), selective_state_inverse_scratch_);
          require_cuda(cudaGetLastError(),
                       "prepare resident recurrent inverse");
          grown_selective_state_space::InverseScratch inverse{};
          require_cuda(cudaMemcpy(&inverse, selective_state_inverse_scratch_,
                                  sizeof(inverse), cudaMemcpyDeviceToHost),
                       "inspect resident recurrent inverse reservation");
          if (inverse.error != 0u || inverse.valid == 0u) {
            throw std::runtime_error(
                "resident recurrent inverse journal metadata invalid");
          }
          grown_selective_state_space::launch_restore_inverse(
              executor_.mutable_device_words(), selective_state_layout_,
              selective_state_inverse_scratch_);
          require_cuda(cudaGetLastError(),
                       "launch resident recurrent inverse");
        }
        combined_resident_factors_inverse_kernel<<<1, 1>>>(
            executor_.mutable_device_words(), cloud_factor_enabled_,
            cloud_factor_layout_device_,
            instance_basin_factor_enabled_ &&
                (advancement & kResidentStepAdvancedInstance) != 0u,
            instance_basin_layout_device_, sensorimotor_factor_enabled_,
            sensorimotor_factor_layout_device_,
            boundary_port_slot(kRawMotorZeroPort),
            boundary_port_slot(kRawMotorOnePort), readout_factor_enabled_,
            readout_factor_layout_device_,
            form_credit_factor_enabled_,
            form_credit_factor_enabled_ &&
                (advancement & kResidentStepJournaledFormCredit) != 0u,
            form_credit_factor_enabled_ &&
                (advancement & kResidentStepAgedFormCredit) != 0u,
            form_credit_layout_device_, edge_bank_factor_enabled_,
            edge_bank_layout_device_, association_outcome.edge_action,
            association_outcome.edge_index,
            association_outcome.owner_value_before,
            association_outcome.dest_value_before,
            association_outcome.recall_cell,
            association_outcome.recall_quantum);
        require_cuda(cudaGetLastError(),
                     "launch combined resident-factor inverse");
        require_cuda(cudaDeviceSynchronize(),
                     "synchronize combined resident-factor inverse");
        if (!resident_step_history_.empty()) {
          resident_step_history_.pop_back();
        }
        if (!resident_association_history_.empty()) {
          resident_association_history_.pop_back();
        }
      } else {
        // Mirrors the forward revert above; see that comment for why the
        // ordinary-substrate step is branch-gated again rather than composed.
        support_->apply_superstep_inverse(map_.view());
      }
      --completed_ticks_;
    }
  }

  [[nodiscard]] Snapshot snapshot() const {
    Snapshot result;
    result.completed_ticks = completed_ticks_;
    const std::vector<std::uint64_t>& slots = support_->active_slots();
    const std::vector<SiteWord> words = support_->download_active_words();
    result.entries.reserve(slots.size());
    for (std::size_t index = 0; index < slots.size(); ++index) {
      if (words[index] != kQ) {
        result.entries.push_back({slots[index], words[index]});
      }
    }
    return result;
  }

  [[nodiscard]] Residency residency() const {
    Residency result;
    for (const auto& [slot, hits] : residency_hits_) {
      if (hits * 2 < kResidencyWindowTicks) continue;
      result.slots.push_back(slot);
      ++result.occupied;
      const SiteWord word = executor_.read_word(slot);
      if (word == 0u || word == 0xffffffffu) ++result.saturated;
    }
    return result;
  }

  [[nodiscard]] const Snapshot& founder_snapshot() const {
    return founder_snapshot_;
  }

  [[nodiscard]] std::size_t active_support_size() const {
    return support_->active_slots().size();
  }

  [[nodiscard]] std::vector<std::uint64_t> active_support_slots() const {
    return support_->active_slots();
  }

  [[nodiscard]] std::size_t reversible_supersteps() const {
    return support_->reversible_supersteps();
  }

  [[nodiscard]] std::uint64_t completed_ticks() const {
    return completed_ticks_;
  }

  [[nodiscard]] const DevelopmentalHash& founder_hash() const {
    return founder_hash_;
  }

  [[nodiscard]] const ContentAddress& law_identity() const {
    return law_identity_;
  }

  [[nodiscard]] const ContentAddress& genesis_manifest_identity() const {
    return genesis_manifest_identity_;
  }

  [[nodiscard]] GenesisRecipeKind genesis_recipe_kind() const {
    return genesis_recipe_kind_;
  }

  [[nodiscard]] std::uint64_t genesis_manifest_entry_count() const {
    return genesis_manifest_entry_count_;
  }

  [[nodiscard]] static std::uint64_t manifest_physical_slot(Int3 relative) {
    const std::int64_t extent =
        kApertureEdgeChunks * static_cast<std::int64_t>(kChunkEdge);
    const std::int64_t centre = extent / 2;
    const std::int64_t coordinates[3] = {
        centre + relative.x, centre + relative.y, centre + relative.z};
    for (const std::int64_t coordinate : coordinates)
      if (coordinate < 0 || coordinate >= extent)
        throw std::out_of_range(
            "grown-adult manifest coordinate is outside aperture");
    const std::uint64_t chunk_x = coordinates[0] / kChunkEdge;
    const std::uint64_t chunk_y = coordinates[1] / kChunkEdge;
    const std::uint64_t chunk_z = coordinates[2] / kChunkEdge;
    const std::uint64_t chunk =
        (chunk_x * kApertureEdgeChunks + chunk_y) * kApertureEdgeChunks +
        chunk_z;
    const std::uint64_t local =
        ((coordinates[0] % kChunkEdge) * kChunkEdge +
         coordinates[1] % kChunkEdge) *
            kChunkEdge +
        coordinates[2] % kChunkEdge;
    return chunk * kChunkSites + local;
  }

  [[nodiscard]] const BoundaryCounters& boundary_counters() const {
    return boundary_;
  }

  [[nodiscard]] BoundarySnapshot boundary_snapshot() const {
    BoundarySnapshot result;
    require_cuda(cudaMemcpy(result.words.data(), boundary_words_.device,
                            result.words.size() * sizeof(SiteWord),
                            cudaMemcpyDeviceToHost),
                 "download grown-adult boundary");
    return result;
  }

  void initialize_boundary_word(std::uint32_t index, SiteWord word) {
    if (completed_ticks_ != 0u) {
      throw std::logic_error(
          "grown-adult boundary can only be initialized before time starts");
    }
    if (index >= kBoundaryWordCount) {
      throw std::out_of_range("grown-adult boundary index invalid");
    }
    require_cuda(cudaMemcpy(boundary_words_.device + index, &word,
                            sizeof(word), cudaMemcpyHostToDevice),
                 "initialize grown-adult boundary word");
    boundary_.sensory_bytes += sizeof(SiteWord);
  }

  // A contact is an exact reciprocal exchange between one declared boundary
  // reservoir word and its fixed physical adult port. The host can request a
  // contact but cannot select its world address or rewrite exchanged content.
  // ⭐ THE FULL §0.12 REQUIREMENT-3 RECEIPT, for one transaction.
  //
  // 169 left every boundary transaction emitting only a COUNT. Doctrine asks for
  // "tick epoch, declared boundary port, matter before/after, conservation
  // result, transaction identity", because "a host cannot preload a local
  // mailbox without creating such a receipt". A count records that something
  // happened; these fields record WHAT.
  //
  // ⚠ The return value is added, not substituted: existing callers ignore it and
  // are unaffected, so this cannot silently change any contract's behaviour.
  MembraneReceipt exchange_boundary_word(std::uint32_t index) {
    if (index >= kBoundaryWordCount) {
      throw std::out_of_range("grown-adult boundary index invalid");
    }
    MembraneReceipt receipt;
    receipt.epoch = completed_ticks_;
    receipt.port = index;
    // `before` and `after` are the boundary word itself -- the matter that
    // actually crossed -- read either side of the exchange rather than inferred.
    require_cuda(cudaMemcpy(&receipt.before, boundary_words_.device + index,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult boundary word before exchange");
    SiteWord resident_before = 0u;
    require_cuda(cudaMemcpy(&resident_before,
                            executor_.device_words() + boundary_port_slot(index),
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult resident slot before exchange");
    exchange_resident_slot_with_boundary(boundary_port_slot(index), index);
    require_cuda(cudaMemcpy(&receipt.after, boundary_words_.device + index,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult boundary word after exchange");
    // ⛔ CONSERVATION IS OVER THE PAIR, NOT THE CROSSING WORD.
    //
    // The first version compared popcount(before) with popcount(after) on the
    // boundary word alone and reported conserved=0 on a legitimate exchange:
    // before=0xef after=0xeffffe1f. That is what a reciprocal exchange DOES --
    // it moves matter between two owners, so neither side is individually
    // invariant. Only the pair is.
    const std::uint64_t slot = boundary_port_slot(index);
    SiteWord resident_after = 0u;
    require_cuda(cudaMemcpy(&resident_after, executor_.device_words() + slot,
                            sizeof(SiteWord), cudaMemcpyDeviceToHost),
                 "read grown-adult resident slot after exchange");
    receipt.conserved =
        __builtin_popcount(resident_before) + __builtin_popcount(receipt.before) ==
        __builtin_popcount(resident_after) + __builtin_popcount(receipt.after);
    // pairs stays 1: this transaction names exactly one declared port, and
    // `declaration` repeats it so a consumer reads one field either way.
    receipt.declaration_fingerprint = receipt.port;
    receipt.transaction = boundary_transactions_;
    return receipt;
  }

  void initialize_sensory_raw_byte(std::uint8_t value) {
    if (completed_ticks_ != 0u) {
      throw std::logic_error(
          "grown-adult raw-byte boundary can only be initialized before time "
          "starts");
    }
    const RawByteRails tape = with_raw_byte_faces(RawByteRails{}, value);
    const std::array<SiteWord, 2> words{tape.zero, tape.one};
    require_cuda(cudaMemcpy(boundary_words_.device + kRawSensoryZeroPort,
                            &words[0], sizeof(SiteWord),
                            cudaMemcpyHostToDevice),
                 "initialize grown-adult raw-byte zero rail");
    require_cuda(cudaMemcpy(boundary_words_.device + kRawSensoryOnePort,
                            &words[1], sizeof(SiteWord),
                            cudaMemcpyHostToDevice),
                 "initialize grown-adult raw-byte one rail");
    ++boundary_.sensory_bytes;
  }

  MembraneReceipt exchange_sensory_raw_byte() {
    return exchange_boundary_raw_byte(kRawSensoryZeroPort, kRawSensoryOnePort);
  }

  // Exchange one exact sensory event with caller-owned device escrow at the
  // fixed sensory ports. Repeating the same exchange restores both sides.
  // ⭐ FULL REQUIREMENT-3 RECEIPT, third of four transactions.
  //
  // The other owner here is CALLER-OWNED DEVICE ESCROW rather than a boundary
  // word, so conservation sums the world slots and the escrow rails on both
  // sides. Entries 170 and 171 both got this wrong by summing only the side they
  // touched; a reciprocal exchange leaves neither owner individually invariant.
  MembraneReceipt exchange_sensory_contact(RawByteRails* device_escrow) {
    if (device_escrow == nullptr) {
      throw std::invalid_argument("grown-adult sensory escrow must be non-null");
    }
    const std::array<std::uint64_t, 2> slots{
        boundary_port_slot(kRawSensoryZeroPort),
        boundary_port_slot(kRawSensoryOnePort)};
    MembraneReceipt receipt;
    receipt.epoch = completed_ticks_;
    receipt.port = kRawSensoryZeroPort;
    const auto read_pair = [&](const std::uint64_t* from_slots,
                               RawByteRails* escrow) {
      std::array<SiteWord, 4> words{};
      require_cuda(cudaMemcpy(&words[0], executor_.device_words() + from_slots[0],
                              sizeof(SiteWord), cudaMemcpyDeviceToHost),
                   "read grown-adult sensory contact world zero");
      require_cuda(cudaMemcpy(&words[1], executor_.device_words() + from_slots[1],
                              sizeof(SiteWord), cudaMemcpyDeviceToHost),
                   "read grown-adult sensory contact world one");
      RawByteRails rails{};
      require_cuda(cudaMemcpy(&rails, escrow, sizeof(RawByteRails),
                              cudaMemcpyDeviceToHost),
                   "read grown-adult sensory contact escrow");
      words[2] = rails.zero;
      words[3] = rails.one;
      return words;
    };
    const std::array<SiteWord, 4> before = read_pair(slots.data(), device_escrow);
    const RawByteDecode escrow_before = decode_raw_byte_faces({before[2], before[3]});
    receipt.before = escrow_before.valid ? escrow_before.value : 0u;
    ++boundary_transactions_;
    exchange_sensory_contact_kernel<<<1, 1>>>(
        executor_.mutable_device_words(), slots[0], slots[1], device_escrow);
    require_cuda(cudaGetLastError(),
                 "launch grown-adult sensory contact exchange");
    require_cuda(cudaDeviceSynchronize(),
                 "synchronize grown-adult sensory contact exchange");
    reconcile_external_contacts(slots);
    const std::array<SiteWord, 4> after = read_pair(slots.data(), device_escrow);
    const RawByteDecode escrow_after = decode_raw_byte_faces({after[2], after[3]});
    receipt.after = escrow_after.valid ? escrow_after.value : 0u;
    int sum_before = 0, sum_after = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
      sum_before += __builtin_popcount(before[i]);
      sum_after += __builtin_popcount(after[i]);
    }
    receipt.conserved = sum_before == sum_after;
    // pairs stays 1: this transaction names exactly one declared port, and
    // `declaration` repeats it so a consumer reads one field either way.
    receipt.declaration_fingerprint = receipt.port;
    receipt.transaction = boundary_transactions_;
    return receipt;
  }

  void exchange_motor_raw_byte() {
    exchange_boundary_raw_byte(kRawMotorZeroPort, kRawMotorOnePort);
  }

  [[nodiscard]] RawByteDecode extract_motor_raw_byte() {
    exchange_motor_raw_byte();
    const RawByteDecode result =
        boundary_raw_byte(kRawMotorZeroPort, kRawMotorOnePort);
    if (result.valid) ++boundary_.motor_bytes;
    return result;
  }

  // Reciprocal focal contact between represented resident matter and one
  // declared boundary escrow word. Applying the same exchange again restores
  // both sides exactly; no host-visible word is synthesized or discarded.
  void exchange_resident_slot_with_boundary(
      std::uint64_t slot, std::uint32_t boundary_index) {
    if (slot >= site_count()) {
      throw std::out_of_range("grown-adult resident slot invalid");
    }
    if (boundary_index >= kBoundaryWordCount) {
      throw std::out_of_range("grown-adult boundary index invalid");
    }
    ++boundary_transactions_;
    exchange_boundary_word_kernel<<<1, 1>>>(
        executor_.mutable_device_words(), slot, boundary_words_.device, boundary_index);
    require_cuda(cudaGetLastError(),
                 "launch grown-adult resident-boundary exchange");
    require_cuda(cudaDeviceSynchronize(),
                 "synchronize grown-adult resident-boundary exchange");
    const std::array<std::uint64_t, 1> touched{slot};
    reconcile_external_contacts(touched);
  }

  [[nodiscard]] std::uint64_t boundary_port_slot(
      std::uint32_t index) const {
    return physical_slot(boundary_port_relative(index));
  }

  [[nodiscard]] std::uint64_t site_count() const {
    return executor_.site_count();
  }

  // The most recent tick's coincidence-association outcome (see
  // AssociationOutcome above), for test/instrumentation inspection. Default
  // (fired=0) if no resident-factor tick has run yet.
  [[nodiscard]] AssociationOutcome last_resident_association() const {
    return resident_association_history_.empty()
               ? AssociationOutcome{}
               : resident_association_history_.back();
  }

  [[nodiscard]] std::span<const DeviceChunkSlot> topology() const {
    return map_.host;
  }

  [[nodiscard]] std::uint64_t physical_slot(Int3 relative) const {
    const std::int64_t centre =
        aperture_edge_chunks_ * static_cast<std::int64_t>(kChunkEdge) / 2;
    return slot_at(map_, centre + relative.x, centre + relative.y,
                   centre + relative.z);
  }

  // ⛔ NO LONGER PUBLIC. The last external caller was a test that held a batched
  // contact kernel; that mechanism moved into this class as apply_contact_stage()
  // and the route came with it. A caller that needs to mutate organism memory
  // must now go through a named method whose class is one of the two permitted
  // mutation classes -- it can no longer take the pointer and decide for itself.
  //
  // ⚠ This is scoped: the adult's OWN internals still use the executor's mutable
  // accessor, and CudaBcc32ActiveSupport still holds one. Those are counted by
  // tools/audit_world_write_authority.sh and are the remaining work.
 private:
  [[nodiscard]] SiteWord* mutable_device_words() { return executor_.mutable_device_words(); }

 public:
  [[nodiscard]] const SiteWord* device_words() const {
    return executor_.device_words();
  }

  // External represented contacts can reactivate a site that dynamic sparse
  // scheduling previously dropped at Q. Re-enrol every physically touched
  // slot before the next F step so the dense world remains the authority.
  void include_physical_support(std::span<const std::uint64_t> slots) {
    support_->include(slots);
  }

  // Use immediately after any external reciprocal exchange.  The dense field
  // remains material authority while sparse scheduling is updated without
  // invalidating reversible F-phase history.
  // The adult owns the batched contact stage, so no caller needs its world
  // pointer to run one. Pair with `reconcile_external_contacts` exactly as the
  // single-slot path does: this IS an external reciprocal exchange.
  //
  // ⭐ AND IT EMITS A RECEIPT. §0.12 requirement 3: "each boundary transaction
  // needs a receipt containing tick epoch, declared boundary port, matter
  // before/after, conservation result, transaction identity. A host cannot
  // preload a local mailbox without creating such a receipt."
  //
  // ⛔ BEFORE THIS, `BoundaryCounters::semantic_host_writes` was declared and
  // NEVER INCREMENTED, while eleven contracts printed `semantic_host_writes=0`
  // as literal text -- a receipt asserting the absence of the forbidden thing in
  // a way that could not report anything else. A counter that cannot rise is not
  // evidence that nothing happened.
  // ⛔ AND ITS RECEIPT IS NOT A COPY OF THE OTHER THREE. The three single-port
  // transactions read one word from each owner. This one exchanges `count` pairs
  // in one launch, so `before`/`after` are the POPULATION of the declared world
  // bits, `port` is kNoSinglePort because no single index names the batch, and
  // `declaration` identifies which batch ran. `conserved` is the pair sum across
  // BOTH owners -- the only field that keeps its meaning unchanged from the
  // single-port case, and the only one whose definition survives batching.
  MembraneReceipt apply_contact_stage(SiteWord* tape,
                                      const ContactExchange* device_exchanges,
                                      std::uint32_t count) {
    MembraneReceipt receipt;
    receipt.epoch = completed_ticks_;
    receipt.port = kNoSinglePort;
    receipt.pairs = count;
    if (count == 0u) {
      // Nothing crossed, so nothing to conserve. The transaction counter does
      // NOT advance: a receipt for a batch that never launched would make
      // `boundary_transactions()` count intentions rather than exchanges.
      receipt.transaction = boundary_transactions_;
      return receipt;
    }
    ensure_contact_census_scratch();
    constexpr std::uint32_t kBlock = 128u;
    const std::uint32_t blocks = (count + kBlock - 1u) / kBlock;
    const auto census = [&](unsigned int& world_population,
                            unsigned int& pair_population,
                            unsigned long long& identity,
                            unsigned int& alias_count) {
      require_cuda(cudaMemset(contact_census_scratch_, 0,
                              sizeof(unsigned int) * 4u +
                                  sizeof(unsigned long long)),
                   "clear grown-adult contact stage census");
      auto* counts = static_cast<unsigned int*>(contact_census_scratch_);
      auto* ident = reinterpret_cast<unsigned long long*>(counts + 4);
      contact_stage_census_kernel<<<blocks, kBlock>>>(
          executor_.device_words(), tape, device_exchanges, count, counts,
          counts + 1, ident, counts + 2);
      require_cuda(cudaGetLastError(), "launch grown-adult contact stage census");
      require_cuda(cudaDeviceSynchronize(),
                   "synchronize grown-adult contact stage census");
      unsigned int host_counts[3] = {0u, 0u, 0u};
      require_cuda(cudaMemcpy(host_counts, counts, sizeof(host_counts),
                              cudaMemcpyDeviceToHost),
                   "read grown-adult contact stage census counts");
      require_cuda(cudaMemcpy(&identity, ident, sizeof(identity),
                              cudaMemcpyDeviceToHost),
                   "read grown-adult contact stage fingerprint");
      world_population = host_counts[0];
      pair_population = host_counts[1];
      alias_count = host_counts[2];
    };
    unsigned int world_before = 0u, pair_before = 0u, alias_before = 0u;
    unsigned long long identity_before = 0ull;
    census(world_before, pair_before, identity_before, alias_before);
    // ⭐ REFUSED, NOT EXECUTED-AND-REPORTED. Two declared pairs naming the same
    // world word and the same world bit make the exchange kernel's two atomicXors
    // cancel: the world keeps a bit it should have given away and the tape gives
    // one to nobody. Reporting that afterwards as `conserved=0` leaves the world
    // already corrupted. The batch does not launch, the world is untouched, and
    // the transaction counter does not advance -- a refusal is not an exchange.
    if (alias_before != 0u) {
      receipt.aliased = alias_before;
      receipt.before = static_cast<SiteWord>(world_before);
      receipt.after = static_cast<SiteWord>(world_before);
      receipt.conserved = true;   // nothing crossed, so nothing to conserve
      receipt.declaration_fingerprint = identity_before;
      receipt.transaction = boundary_transactions_;
      return receipt;
    }
    apply_contact_stage_kernel<<<blocks, kBlock>>>(
        executor_.mutable_device_words(), tape, device_exchanges, count);
    require_cuda(cudaGetLastError(), "launch grown-adult contact stage");
    require_cuda(cudaDeviceSynchronize(), "synchronize grown-adult contact stage");
    unsigned int world_after = 0u, pair_after = 0u, alias_after = 0u;
    unsigned long long identity_after = 0ull;
    census(world_after, pair_after, identity_after, alias_after);
    // ⛔ SEALED INPUTS, CHECKED RATHER THAN ASSUMED. The exchange kernel does not
    // touch the descriptor list, so the fingerprint must be identical on both
    // sides. If it is not, the declared batch changed WHILE the transaction ran
    // -- §0.12 requirement 4, and a fault of a different kind than conservation.
    // Folding it into `conserved` would let a sealed-input violation be reported
    // as lost matter, so it is raised on its own terms.
    if (identity_before != identity_after) {
      throw std::runtime_error(
          "grown-adult contact descriptors changed during the transaction");
    }
    boundary_.sensory_bytes += count;
    ++boundary_transactions_;
    receipt.before = static_cast<SiteWord>(world_before);
    receipt.after = static_cast<SiteWord>(world_after);
    // ⚠ ENTAILED, AND SAID SO. With aliasing refused above, a launched batch
    // conserves the pair sum by construction, so this reads 1 for a reason that
    // is not evidence. It is kept because a future change to the exchange kernel
    // could break it; it is NOT the arm that can fail. `aliased` is.
    receipt.conserved = pair_before == pair_after;
    receipt.aliased = 0u;
    receipt.declaration_fingerprint = identity_before;
    receipt.transaction = boundary_transactions_;
    return receipt;
  }

  // Read-only provenance. The count rises with every boundary transaction the
  // adult performs, so a contract can print a MEASURED number instead of a
  // literal, and a run in which no transaction occurred reports 0 because none
  // occurred rather than because nothing counts.
  [[nodiscard]] std::uint64_t boundary_transactions() const {
    return boundary_transactions_;
  }

  // ⭐ THE INTERVENTION BROKER. The one named authority through which an
  // experimenter's manipulation -- a lesion, a matched remote perturbation, a
  // route ablation -- reaches organism memory.
  //
  // ⛔ WHY IT EXISTS, AND WHY HERE RATHER THAN BESIDE THE ADULT. §13 forbids
  // building a new tissue next to an existing one. This is not tissue: it is the
  // OWNER's intervention authority, and the owner is this class. Before it, six
  // tissues each took the world pointer and launched their own lesion kernel,
  // which is the "third instrumentation route" §0.12 requirement 2 forbids --
  // and after `mutable_device_words()` became private, 28 translation units
  // stopped compiling because that is exactly what they were doing.
  //
  // ⭐ THE POINTER IS SCOPED TO THE OPERATION, NOT HANDED OVER. A tissue passes
  // what it wants done; it never receives a pointer it can keep. That is
  // strictly stronger than the previous arrangement and strictly weaker than
  // requirement 5 -- a callable could still copy the pointer out. It is a choke
  // point and a ledger, not a proof, and the audit counts it as one write route
  // rather than six.
  //
  // ⚠ EPOCH-FORMING. Every intervention stamps the tick at which it happened and
  // takes the next index, so a claim can say WHICH trajectory it is talking
  // about. `interventions()` is measured, never asserted: a run with no
  // intervention reports 0 because none occurred.
  template <typename Operation>
    // ⭐ THE BROKER FORWARDS THE OPERATION'S RESULT, and that is a measurement
  // result rather than a preference. Nine call sites consume a helper's return
  // value (`auto r = sparse::lesion_transition(words, ...)`), and a broker that
  // returns a receipt cannot wrap them -- the caller would receive the receipt
  // where it expects the helper's value. Twice already a transform produced
  // exactly that and would not have compiled.
  //
  // ⚠ The receipt is not lost, it is moved off the return path: `last_intervene()`
  // exposes it, and no caller consumed the returned receipt before this change
  // (checked, not assumed). Counting and epoch-stamping are unchanged.
  decltype(auto) intervene(InterventionReason reason,
                         Operation&& operation) {
    InterventionReceipt receipt;
    receipt.epoch = completed_ticks_;
    receipt.intervention = ++interventions_;
    receipt.reason = reason;
    last_intervene_ = receipt;
    return std::forward<Operation>(operation)(mutable_device_words());
  }
  [[nodiscard]] std::uint64_t interventions() const { return interventions_; }

  // ⭐ THE OTHER BROKER, AND IT IS DELIBERATELY A SEPARATE ONE.
  //
  // `intervene` is the experimenter reaching in. This is the organism's own
  // machinery writing OUTSIDE the canonical tick -- a contact through no
  // declared port, a learned trajectory, a bound context, a staged motor byte.
  // Routing both through one method would have been less code and a worse
  // measurement: the mutation-class census could no longer tell the two apart,
  // and requirement 2's 23 would stay a single opaque number.
  //
  // ⚠ SEPARATE COUNTERS FOR THE SAME REASON. `interventions()` and
  // `resident_stages()` are asked different questions by different claims: one
  // says which trajectory a result belongs to, the other says how much of the
  // organism still runs outside its own law.
  template <typename Operation>
    // ⭐ THE BROKER FORWARDS THE OPERATION'S RESULT, and that is a measurement
  // result rather than a preference. Nine call sites consume a helper's return
  // value (`auto r = sparse::lesion_transition(words, ...)`), and a broker that
  // returns a receipt cannot wrap them -- the caller would receive the receipt
  // where it expects the helper's value. Twice already a transform produced
  // exactly that and would not have compiled.
  //
  // ⚠ The receipt is not lost, it is moved off the return path: `last_resident_stage()`
  // exposes it, and no caller consumed the returned receipt before this change
  // (checked, not assumed). Counting and epoch-stamping are unchanged.
  decltype(auto) resident_stage(ResidentStageReason reason,
                              Operation&& operation) {
    ResidentStageReceipt receipt;
    receipt.epoch = completed_ticks_;
    receipt.stage = ++resident_stages_;
    receipt.reason = reason;
    last_resident_stage_ = receipt;
    return std::forward<Operation>(operation)(mutable_device_words());
  }
  [[nodiscard]] std::uint64_t resident_stages() const { return resident_stages_; }

  void reconcile_external_contacts(
      std::span<const std::uint64_t> slots,
      cudaStream_t stream = nullptr) {
    support_->reconcile_external_contacts(slots, stream);
  }

  // Selects the fixed version-1 local cloud law for subsequent ordinary adult
  // ticks. The layout contains only physical resident addresses; the receipt is
  // transient instrumentation and never supplies a target or semantic role.
  void configure_cloud_factor(
      const grown_cloud_factor::DeviceLayout& layout,
      grown_cloud_factor::ContactReceipt* receipt) {
    for (std::uint32_t index = 0u;
         index < grown_cloud_factor::kResidentRailCount; ++index) {
      const grown_cloud_factor::PhysicalOffset offset =
          grown_cloud_factor::physical_offset(index);
      if (layout.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "cloud_F_v1 layout is not the fixed physical aperture");
      }
    }
    const std::uint64_t marker_slot =
        layout.rails[grown_cloud_factor::global_rail(
            grown_cloud_factor::kCloudMarker, 0u)];
    const SiteWord marker = executor_.read_word(marker_slot);
    if (marker != grown_cloud_factor::kCloudMarkerValue) {
      throw std::logic_error(
          "cloud_F_v1 founder matter is not resident");
    }
    cloud_factor_layout_ = layout;
    upload_cloud_factor_layout();
    cloud_factor_receipt_ = receipt;
    ensure_factor_advanced();
    cloud_factor_enabled_ = true;
  }

  void detach_cloud_factor_receipt() {
    cloud_factor_receipt_ = nullptr;
  }

  void configure_instance_basin_factor(
      const grown_instance_basin_factor::DeviceLayout& layout,
      grown_instance_basin_factor::DeviceInputs* inputs,
      grown_instance_basin_factor::StepReceipt* receipt) {
    for (std::uint32_t index = 0u;
         index < grown_instance_basin_factor::kPhysicalRailCount; ++index) {
      const grown_instance_basin_factor::PhysicalOffset offset =
          grown_instance_basin_factor::physical_offset(index);
      if (layout.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "instance basin layout is not the fixed physical aperture");
      }
    }
    const std::uint64_t marker_slot =
        layout.rails[grown_instance_basin_factor::resident_index(
            grown_instance_basin_factor::kFactorMarker)];
    if (executor_.read_word(marker_slot) !=
        grown_instance_basin_factor::kFactorMarkerValue) {
      throw std::logic_error("instance basin founder matter is not resident");
    }
    instance_basin_layout_ = layout;
    upload_instance_basin_layout();
    instance_basin_inputs_ = inputs;
    instance_basin_receipt_ = receipt;
    ensure_factor_advanced();
    instance_basin_factor_enabled_ = true;
  }

  void detach_instance_basin_buffers() {
    instance_basin_inputs_ = nullptr;
    instance_basin_receipt_ = nullptr;
  }

  // The edge-bank ancilla (bcc32_resident_edge_bank.cuh) that records
  // coincidence associations. It has no external inputs or receipt -- it is
  // driven entirely by combined_resident_factors_step_kernel's own
  // instance-basin/cloud coincidence detection above.
  void configure_edge_bank_factor(
      const resident_edge_bank::DeviceLayout& layout) {
    for (std::uint32_t index = 0u;
         index < resident_edge_bank::kPhysicalRailCount; ++index) {
      const resident_edge_bank::PhysicalOffset offset =
          resident_edge_bank::physical_offset(index);
      if (layout.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "edge bank layout is not the fixed physical aperture");
      }
    }
    const std::uint64_t marker_slot = layout.rails[resident_edge_bank::
        resident_index(resident_edge_bank::kFactorMarker)];
    if (executor_.read_word(marker_slot) !=
        resident_edge_bank::kFactorMarkerValue) {
      throw std::logic_error("edge bank founder matter is not resident");
    }
    edge_bank_layout_ = layout;
    upload_edge_bank_layout();
    edge_bank_factor_enabled_ = true;
  }

  void configure_sensorimotor_factor(
      const grown_sensorimotor_factor::DeviceLayout& layout,
      grown_sensorimotor_factor::DeviceInputs* inputs,
      grown_sensorimotor_factor::PredictionReceipt* prediction,
      grown_sensorimotor_factor::ConsequenceReceipt* consequence,
      grown_sensorimotor_factor::TransformReceipt* transform) {
    if (selective_state_factor_enabled_) {
      throw std::logic_error(
          "sensorimotor and resident recurrent factors need an explicit motor mux");
    }
    for (std::uint32_t index = 0u;
         index < grown_sensorimotor_factor::kPhysicalRailCount; ++index) {
      const grown_sensorimotor_factor::PhysicalOffset offset =
          grown_sensorimotor_factor::physical_offset(index);
      if (layout.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "sensorimotor_F_v1 layout is not the fixed physical aperture");
      }
    }
    for (std::uint32_t index = 0u;
         index < grown_instance_basin_factor::kPhysicalRailCount; ++index) {
      const grown_instance_basin_factor::PhysicalOffset offset =
          grown_instance_basin_factor::physical_offset(index);
      if (layout.context.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "sensorimotor context is not the fixed instance aperture");
      }
    }
    const std::uint64_t marker_slot =
        layout.rails[grown_sensorimotor_factor::value_index(
            grown_sensorimotor_factor::kFactorMarker)];
    if (executor_.read_word(marker_slot) !=
        grown_sensorimotor_factor::kFactorMarkerValue) {
      throw std::logic_error(
          "sensorimotor_F_v1 founder matter is not resident");
    }
    const std::uint64_t version_slot =
        layout.rails[grown_sensorimotor_factor::value_index(
            grown_sensorimotor_factor::kLayoutVersion)];
    if (executor_.read_word(version_slot) !=
        grown_sensorimotor_factor::kLayoutVersionValue) {
      throw std::logic_error("sensorimotor factor layout version unsupported");
    }
    sensorimotor_factor_layout_ = layout;
    upload_sensorimotor_factor_layout();
    sensorimotor_inputs_ = inputs;
    sensorimotor_prediction_ = prediction;
    sensorimotor_consequence_ = consequence;
    sensorimotor_transform_ = transform;
    ensure_factor_advanced();
    sensorimotor_factor_enabled_ = true;
  }

  void detach_sensorimotor_factor_buffers() {
    sensorimotor_inputs_ = nullptr;
    sensorimotor_prediction_ = nullptr;
    sensorimotor_consequence_ = nullptr;
    sensorimotor_transform_ = nullptr;
  }

  void configure_readout_factor(
      const resident_readout_f_route::DeviceLayout& layout,
      resident_readout_f_route::DeviceInputs* inputs,
      resident_readout_f_route::CreditReceipt* receipt) {
    for (std::uint32_t index = 0u;
         index < resident_readout_f_route::kPhysicalRailCount; ++index) {
      const resident_readout_f_route::PhysicalOffset offset =
          resident_readout_f_route::physical_offset(index);
      if (layout.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "resident readout F layout is not the fixed physical aperture");
      }
    }
    const std::uint64_t marker_slot = layout.rails[
        resident_readout_f_route::global_rail(
            resident_readout_f_route::kFactorMarker, 0u)];
    if (executor_.read_word(marker_slot) !=
        resident_readout_f_route::kFactorMarkerValue) {
      throw std::logic_error(
          "resident readout F founder matter is not resident");
    }
    readout_factor_layout_ = layout;
    upload_readout_factor_layout();
    readout_inputs_ = inputs;
    readout_receipt_ = receipt;
    ensure_factor_advanced();
    readout_factor_enabled_ = true;
  }

  void detach_readout_factor_buffers() {
    readout_inputs_ = nullptr;
    readout_receipt_ = nullptr;
  }

  void configure_form_credit_factor(
      const grown_form_credit_factor::DeviceLayout& layout,
      grown_form_credit_factor::DeviceInputs* inputs,
      grown_form_credit_factor::Receipt* receipt) {
    if (selective_state_factor_enabled_) {
      throw std::logic_error(
          "form-credit and resident recurrent factors need an explicit motor mux");
    }
    for (std::uint32_t index = 0u;
         index < grown_form_credit_factor::kFormRailCount; ++index) {
      const std::uint32_t region = index /
          (grown_form_credit_factor::kFormFieldsPerRegion * 2u);
      const std::uint32_t field = (index / 2u) %
          grown_form_credit_factor::kFormFieldsPerRegion;
      const grown_form_credit_factor::PhysicalOffset offset =
          grown_form_credit_factor::form_physical_offset(
              region, field, index & 1u);
      if (layout.form.rails[index] !=
          physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "form credit layout is not the fixed form aperture");
      }
    }
    for (std::uint32_t index = 0u;
         index < grown_form_credit_factor::kCreditResidentRailCount; ++index) {
      const grown_form_credit_factor::PhysicalOffset offset =
          grown_form_credit_factor::physical_offset(index);
      if (layout.rails[index] != physical_slot({offset.x, offset.y, offset.z})) {
        throw std::invalid_argument(
            "form credit layout is not the fixed credit aperture");
      }
    }
    if (executor_.read_word(layout.rails[
            grown_form_credit_factor::global_index(0u, 0u)]) !=
        grown_form_credit_factor::kFactorMarkerValue) {
      throw std::logic_error("form credit founder matter is not resident");
    }
    form_credit_layout_ = layout;
    upload_form_credit_layout();
    form_credit_inputs_ = inputs;
    form_credit_receipt_ = receipt;
    ensure_factor_advanced();
    form_credit_factor_enabled_ = true;
  }

  void detach_form_credit_factor_buffers() {
    form_credit_inputs_ = nullptr;
    form_credit_receipt_ = nullptr;
  }

  void configure_selective_state_factor(
      grown_selective_state_space::DeviceInputs* inputs,
      grown_selective_state_space::DeviceScratch* scratch,
      grown_selective_state_space::Receipt* receipt) {
    if (inputs == nullptr || scratch == nullptr || receipt == nullptr) {
      throw std::invalid_argument(
          "resident recurrent factor requires device contact buffers");
    }
    // 04a21d8769 added sparse_event_memory_factor_enabled_ to this predicate
    // to make the one-sided exclusion symmetric, on the theory that two
    // factors journaling/restoring the same raw-motor pair at their own
    // snapshots could not be made exactly reversible together: forward,
    // selective_state snapshots X0 and publishes X1, then sparse_event
    // snapshots X1 (an intermediate, never the true prior) and publishes X2;
    // inverse ran in the SAME order (selective_state then sparse_event), so
    // selective_state restored X0 correctly and sparse_event then overwrote
    // it with X1 -- landing at X1, not X0.
    //
    // That diagnosis was of the DISPATCH ORDER, not of coexistence itself.
    // reverse() now unwinds resident factors LIFO with respect to develop()'s
    // forward order (sparse_event's inverse block now runs before
    // selective_state's -- see reverse()), so sparse_event correctly restores
    // its intermediate X1 first and selective_state then restores the true
    // prior X0. With the ordering fixed, the exclusion above is no longer
    // load-bearing, so it is deliberately removed here: selective_state and
    // sparse_event may coexist. sensorimotor and form_credit are untouched --
    // this diagnosis never covered them.
    if (sensorimotor_factor_enabled_ || form_credit_factor_enabled_) {
      throw std::logic_error(
          "resident recurrent factor needs exclusive raw-motor ownership");
    }
    if (!instance_basin_factor_enabled_)
      throw std::logic_error("resident recurrence requires situation tissue");
    if (executor_.read_word(grown_selective_state_space::fixed_physical_slot(
            grown_selective_state_space::pair_index(
                grown_selective_state_space::kGlobalBase,
                grown_selective_state_space::kFactorMarker, 0u))) !=
        grown_selective_state_space::kFactorMarkerValue) {
      throw std::logic_error(
          "resident recurrent founder matter is not resident");
    }
    if (executor_.read_word(grown_selective_state_space::fixed_physical_slot(
            grown_selective_state_space::pair_index(
                grown_selective_state_space::kGlobalBase,
                grown_selective_state_space::kLayoutVersion, 0u))) !=
        grown_selective_state_space::kLayoutVersionValue) {
      throw std::logic_error(
          "resident recurrent layout version unsupported");
    }
    selective_state_layout_ = grown_selective_state_space::connect_resident_layout(
        boundary_port_slot(kRawSensoryZeroPort), boundary_port_slot(kRawSensoryOnePort),
        boundary_port_slot(kRawMotorZeroPort), boundary_port_slot(kRawMotorOnePort),
        instance_basin_layout_);
    const std::uint64_t motor_slots[]{selective_state_layout_.raw_motor_zero_slot,
                                      selective_state_layout_.raw_motor_one_slot};
    include_physical_support(motor_slots);
    selective_state_inputs_ = inputs;
    selective_state_scratch_ = scratch;
    selective_state_receipt_ = receipt;
    ensure_factor_advanced();
    ensure_selective_inverse_scratch();
    selective_state_factor_enabled_ = true;
  }

  void configure_sparse_event_memory_factor(
      grown_sparse_event_memory::DeviceInputs* inputs,
      grown_sparse_event_memory::DeviceScratch* scratch,
      grown_sparse_event_memory::Receipt* receipt) {
    if (inputs == nullptr || scratch == nullptr || receipt == nullptr) {
      throw std::invalid_argument(
          "sparse event memory requires device contact buffers");
    }
    // See the comment in configure_selective_state_factor() above: the
    // former exclusion against selective_state was a workaround for the
    // inverse dispatch running in forward (not LIFO) order, which is now
    // fixed in reverse(). selective_state and sparse_event may coexist.
    if (sensorimotor_factor_enabled_ || form_credit_factor_enabled_) {
      throw std::logic_error(
          "sparse event memory needs exclusive raw-motor ownership");
    }
    if (!instance_basin_factor_enabled_)
      throw std::logic_error("sparse event memory requires situation tissue");
    if (executor_.read_word(grown_sparse_event_memory::fixed_physical_slot(
            grown_sparse_event_memory::pair_index(
                grown_sparse_event_memory::kGlobalBase,
                grown_sparse_event_memory::kFactorMarker, 0u))) !=
        grown_sparse_event_memory::kFactorMarkerValue) {
      throw std::logic_error("sparse event founder matter is not resident");
    }
    if (executor_.read_word(grown_sparse_event_memory::fixed_physical_slot(
            grown_sparse_event_memory::pair_index(
                grown_sparse_event_memory::kGlobalBase,
                grown_sparse_event_memory::kLayoutVersion, 0u))) !=
        grown_sparse_event_memory::kLayoutVersionValue) {
      throw std::logic_error("sparse event layout version unsupported");
    }
    sparse_event_memory_layout_ =
        grown_sparse_event_memory::connect_resident_layout(
            boundary_port_slot(kRawSensoryZeroPort),
            boundary_port_slot(kRawSensoryOnePort),
            boundary_port_slot(kRawMotorZeroPort),
            boundary_port_slot(kRawMotorOnePort), instance_basin_layout_);
    const std::uint64_t motor_slots[]{
        sparse_event_memory_layout_.raw_motor_zero_slot,
        sparse_event_memory_layout_.raw_motor_one_slot};
    include_physical_support(motor_slots);
    sparse_event_memory_inputs_ = inputs;
    sparse_event_memory_scratch_ = scratch;
    sparse_event_memory_receipt_ = receipt;
    ensure_factor_advanced();
    ensure_sparse_event_memory_inverse_scratch();
    sparse_event_memory_factor_enabled_ = true;
  }

  void detach_sparse_event_memory_buffers() {
    sparse_event_memory_inputs_ = nullptr;
    sparse_event_memory_scratch_ = nullptr;
    sparse_event_memory_receipt_ = nullptr;
  }

  void detach_selective_state_buffers() {
    selective_state_inputs_ = nullptr;
    selective_state_scratch_ = nullptr;
    selective_state_receipt_ = nullptr;
  }

  [[nodiscard]] resident_readout_f_route::DeviceView readout_factor_view(
      resident_readout_f_route::CaptureView capture) {
    if (!readout_factor_enabled_ || readout_factor_layout_device_ == nullptr) {
      throw std::logic_error("resident readout F factor is not configured");
    }
    // ⚠ THE TREE'S ONE AGGREGATE ESCAPE, AND IT IS LEFT OPEN ON PURPOSE. Making
    // the view's `words` const closes it and costs nothing -- measured: the sole
    // use of the field is a read. But it adds two compile errors to a file that
    // is ALREADY red, and 28 translation units are already red from the accessor
    // privatisation (entry 177). Landing a closure on top of an unrepaired break
    // is landing broken code, so the escape stays counted at 1 until the repair
    // arc lands and closes it with the callers intact.
    return {executor_.mutable_device_words(), readout_factor_layout_device_, capture};
  }

  // Adds mechanism-only founder matter before the first tick. The exact M0
  // constructor remains unchanged; later milestones use this seam to place
  // additional developmental tissue in the same resident field.
  void attach_founder_matter(std::span<const StateEntry> entries) {
    if (completed_ticks_ != 0u) {
      throw std::logic_error(
          "grown-adult founder matter can only be attached before time starts");
    }
    if (genesis_recipe_kind_ == GenesisRecipeKind::explicit_manifest) {
      throw std::logic_error(
          "grown-adult explicit Genesis manifest is sealed; construct one "
          "complete re-addressed manifest instead of attaching founder matter");
    }
    std::vector<StateEntry> ordered(entries.begin(), entries.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const StateEntry& left, const StateEntry& right) {
                return left.slot < right.slot;
              });
    std::vector<std::uint64_t> slots;
    slots.reserve(ordered.size());
    if (ordered.empty()) return;
    std::vector<std::uint64_t> current_slots = support_->active_slots();
    std::sort(current_slots.begin(), current_slots.end());
    std::size_t current_index = 0u;
    for (std::size_t index = 0u; index < ordered.size(); ++index) {
      const StateEntry& entry = ordered[index];
      if (entry.slot >= executor_.site_count() || entry.word == kQ ||
          (index > 0u && ordered[index - 1u].slot == entry.slot)) {
        throw std::invalid_argument(
            "grown-adult adjunct founder matter is invalid");
      }
      while (current_index < current_slots.size() &&
             current_slots[current_index] < entry.slot) {
        ++current_index;
      }
      if (current_index < current_slots.size() &&
          current_slots[current_index] == entry.slot) {
        throw std::invalid_argument(
            "grown-adult adjunct founder matter overlaps the measured germ");
      }
      slots.push_back(entry.slot);
    }
    StateEntry* device_entries = nullptr;
    require_cuda(cudaMalloc(&device_entries,
                            ordered.size() * sizeof(StateEntry)),
                 "allocate adjunct founder scatter");
    require_cuda(cudaMemcpy(device_entries, ordered.data(),
                            ordered.size() * sizeof(StateEntry),
                            cudaMemcpyHostToDevice),
                 "upload adjunct founder scatter");
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        std::min<std::size_t>((ordered.size() + 255u) / 256u, 4096u));
    scatter_state_entries_kernel<<<blocks, 256>>>(
        executor_.mutable_device_words(), device_entries, ordered.size());
    require_cuda(cudaGetLastError(), "launch adjunct founder scatter");
    require_cuda(cudaDeviceSynchronize(),
                 "synchronize adjunct founder scatter");
    require_cuda(cudaFree(device_entries), "release adjunct founder scatter");
    support_->include(slots);
    founder_snapshot_ = snapshot();
  }

#include "bcc32_developmental_adult_checkpoint_tail.inl"
};

}  // namespace substrate::bcc32::developmental_adult
