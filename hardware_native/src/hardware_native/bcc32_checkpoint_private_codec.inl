// Private implementation fragment included inside bcc32_checkpoint.cpp's
// anonymous namespace. Keep it translation-unit local.

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

class PublicationLock {
  public:
    PublicationLock(const std::filesystem::path& root, std::string* error) {
        descriptor_ = ::open((root / ".publish.lock").c_str(),
                             O_CREAT | O_RDWR | O_CLOEXEC,
                             0644);
        if (descriptor_ < 0) {
            fail(error, "cannot open BCC-32 publication lock: " +
                            std::string(std::strerror(errno)));
            return;
        }
        int lock_status = 0;
        do {
            lock_status = ::flock(descriptor_, LOCK_EX);
        } while (lock_status != 0 && errno == EINTR);
        if (lock_status != 0) {
            fail(error, "cannot acquire BCC-32 publication lock: " +
                            std::string(std::strerror(errno)));
            ::close(descriptor_);
            descriptor_ = -1;
        }
    }

    PublicationLock(const PublicationLock&) = delete;
    PublicationLock& operator=(const PublicationLock&) = delete;

    ~PublicationLock() {
        if (descriptor_ >= 0) {
            (void)::flock(descriptor_, LOCK_UN);
            (void)::close(descriptor_);
        }
    }

    [[nodiscard]] bool acquired() const { return descriptor_ >= 0; }

  private:
    int descriptor_ = -1;
};

[[nodiscard]] std::span<const std::byte> as_bytes(const Bytes& bytes) {
    return {reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()};
}

[[nodiscard]] Bytes copy_bytes(std::span<const std::byte> bytes) {
    Bytes copied;
    copied.reserve(bytes.size());
    for (const std::byte byte : bytes) {
        copied.push_back(std::to_integer<std::uint8_t>(byte));
    }
    return copied;
}

void put_u32(Bytes* bytes, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32u; shift += 8u) {
        bytes->push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void put_u64(Bytes* bytes, std::uint64_t value) {
    for (std::uint32_t shift = 0; shift < 64u; shift += 8u) {
        bytes->push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void put_i64(Bytes* bytes, std::int64_t value) {
    put_u64(bytes, std::bit_cast<std::uint64_t>(value));
}

void put_coordinate_component(Bytes* bytes, const CoordinateComponent& value) {
    const bool negative = value < 0;
    const CoordinateComponent magnitude = negative ? -value : value;
    std::vector<std::uint8_t> encoded;
    if (magnitude != 0) {
        boost::multiprecision::export_bits(
            magnitude, std::back_inserter(encoded), 8u, true);
    }
    bytes->push_back(negative ? 1u : 0u);
    put_u64(bytes, encoded.size());
    bytes->insert(bytes->end(), encoded.begin(), encoded.end());
}

void put_hash(Bytes* bytes, const Hash256& hash) {
    bytes->insert(bytes->end(), hash.begin(), hash.end());
}

void put_address(Bytes* bytes, const ContentAddress& address) {
    put_hash(bytes, address.digest);
    put_u64(bytes, address.byte_count);
}

bool take_u32(std::span<const std::uint8_t> bytes,
              std::size_t* cursor,
              std::uint32_t* value) {
    if (*cursor > bytes.size() || bytes.size() - *cursor < 4u) return false;
    *value = 0u;
    for (std::uint32_t shift = 0; shift < 32u; shift += 8u) {
        *value |= static_cast<std::uint32_t>(bytes[*cursor + shift / 8u]) << shift;
    }
    *cursor += 4u;
    return true;
}

bool take_u64(std::span<const std::uint8_t> bytes,
              std::size_t* cursor,
              std::uint64_t* value) {
    if (*cursor > bytes.size() || bytes.size() - *cursor < 8u) return false;
    *value = 0u;
    for (std::uint32_t shift = 0; shift < 64u; shift += 8u) {
        *value |= static_cast<std::uint64_t>(bytes[*cursor + shift / 8u]) << shift;
    }
    *cursor += 8u;
    return true;
}

bool take_i64(std::span<const std::uint8_t> bytes,
              std::size_t* cursor,
              std::int64_t* value) {
    std::uint64_t raw = 0u;
    if (!take_u64(bytes, cursor, &raw)) return false;
    *value = std::bit_cast<std::int64_t>(raw);
    return true;
}

bool take_coordinate_component(std::span<const std::uint8_t> bytes,
                               std::size_t* cursor,
                               CoordinateComponent* value) {
    if (*cursor >= bytes.size()) return false;
    const std::uint8_t sign = bytes[(*cursor)++];
    std::uint64_t magnitude_bytes = 0u;
    if (sign > 1u || !take_u64(bytes, cursor, &magnitude_bytes) ||
        magnitude_bytes > bytes.size() - *cursor) {
        return false;
    }
    if (magnitude_bytes == 0u) {
        if (sign != 0u) return false;
        *value = 0;
        return true;
    }
    if (bytes[*cursor] == 0u) return false;
    CoordinateComponent magnitude = 0;
    const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(*cursor);
    boost::multiprecision::import_bits(
        magnitude,
        begin,
        begin + static_cast<std::ptrdiff_t>(magnitude_bytes),
        8u,
        true);
    *cursor += static_cast<std::size_t>(magnitude_bytes);
    *value = sign == 0u ? magnitude : -magnitude;
    return true;
}

Hash256 domain_hash(std::string_view domain, std::span<const std::byte> bytes) {
    ContentHasher hasher;
    hasher.update({reinterpret_cast<const std::byte*>(domain.data()), domain.size()});
    hasher.update(bytes);
    return hasher.finish();
}

bool take_hash(std::span<const std::uint8_t> bytes,
               std::size_t* cursor,
               Hash256* hash) {
    if (*cursor > bytes.size() || bytes.size() - *cursor < hash->size()) return false;
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(*cursor),
                hash->size(),
                hash->begin());
    *cursor += hash->size();
    return true;
}

bool take_address(std::span<const std::uint8_t> bytes,
                  std::size_t* cursor,
                  ContentAddress* address) {
    return take_hash(bytes, cursor, &address->digest) &&
           take_u64(bytes, cursor, &address->byte_count);
}

bool write_all(int descriptor,
               const std::uint8_t* data,
               std::size_t bytes,
               std::string* error) {
    while (bytes != 0u) {
        const ssize_t written = ::write(descriptor, data, bytes);
        if (written < 0) {
            if (errno == EINTR) continue;
            return fail(error, "BCC-32 durable write failed: " +
                                   std::string(std::strerror(errno)));
        }
        if (written == 0) return fail(error, "BCC-32 durable write made no progress");
        data += written;
        bytes -= static_cast<std::size_t>(written);
    }
    return true;
}

bool write_synced_file(const std::filesystem::path& path,
                       std::span<const std::uint8_t> bytes,
                       std::string* error) {
    const int descriptor =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
    if (descriptor < 0) {
        return fail(error, "cannot open BCC-32 artifact file: " + path.string());
    }
    const bool wrote = write_all(descriptor, bytes.data(), bytes.size(), error);
    const bool synced = wrote && ::fdatasync(descriptor) == 0;
    const int saved_errno = errno;
    ::close(descriptor);
    if (!wrote) return false;
    if (!synced) {
        return fail(error, "cannot synchronize BCC-32 artifact file: " +
                               std::string(std::strerror(saved_errno)));
    }
    return true;
}

std::filesystem::path unique_temporary_path(
    const std::filesystem::path& destination) {
    static std::atomic<std::uint64_t> sequence{0u};
    return destination.string() + ".tmp." +
           std::to_string(static_cast<unsigned long long>(::getpid())) + "." +
           std::to_string(sequence.fetch_add(1u, std::memory_order_relaxed));
}

bool sync_directory(const std::filesystem::path& path, std::string* error) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) return fail(error, "cannot open BCC-32 artifact directory");
    const bool synced = ::fsync(descriptor) == 0;
    const int saved_errno = errno;
    ::close(descriptor);
    return synced ? true
                  : fail(error, "cannot synchronize BCC-32 artifact directory: " +
                                    std::string(std::strerror(saved_errno)));
}

bool read_file(const std::filesystem::path& path, Bytes* bytes, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return fail(error, "cannot open BCC-32 artifact file: " + path.string());
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) >
                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return fail(error, "BCC-32 artifact file size is invalid");
    }
    bytes->resize(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes->data()), size);
    if (input.gcount() != size) return fail(error, "BCC-32 artifact file is truncated");
    return true;
}

bool read_exact_regular_file(const std::filesystem::path& path,
                             std::uint64_t expected_bytes,
                             Bytes* bytes,
                             std::string* error) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        return fail(error, "cannot open bounded BCC-32 artifact file: " +
                               std::string(std::strerror(errno)));
    }
    struct stat status {};
    const bool valid_size =
        ::fstat(descriptor, &status) == 0 && S_ISREG(status.st_mode) &&
        status.st_size >= 0 &&
        static_cast<std::uint64_t>(status.st_size) == expected_bytes &&
        static_cast<std::uint64_t>(status.st_size) <=
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (!valid_size) {
        (void)::close(descriptor);
        return fail(error, "BCC-32 artifact is not a regular file of its declared size");
    }

    bytes->resize(static_cast<std::size_t>(status.st_size));
    std::size_t offset = 0u;
    bool complete = true;
    while (offset < bytes->size()) {
        const ssize_t count =
            ::read(descriptor, bytes->data() + offset, bytes->size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            complete = false;
            break;
        }
        offset += static_cast<std::size_t>(count);
    }
    struct stat final_status {};
    const bool unchanged = ::fstat(descriptor, &final_status) == 0 &&
                           S_ISREG(final_status.st_mode) &&
                           final_status.st_dev == status.st_dev &&
                           final_status.st_ino == status.st_ino &&
                           final_status.st_size == status.st_size;
    (void)::close(descriptor);
    if (!complete || offset != bytes->size() || !unchanged) {
        return fail(error, "BCC-32 artifact changed or was truncated while being read");
    }
    return true;
}

bool verify_existing_immutable_object(const std::filesystem::path& path,
                                      std::span<const std::uint8_t> bytes,
                                      std::string* error) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (descriptor < 0) {
        return fail(error, "cannot open existing BCC-32 immutable object: " +
                               std::string(std::strerror(errno)));
    }
    struct stat status {};
    const bool regular = ::fstat(descriptor, &status) == 0 && S_ISREG(status.st_mode);
    const bool same_size = regular && status.st_size >= 0 &&
                           static_cast<std::uint64_t>(status.st_size) == bytes.size();
    Bytes existing(same_size ? bytes.size() : 0u);
    std::size_t offset = 0u;
    while (same_size && offset < existing.size()) {
        const ssize_t count =
            ::read(descriptor, existing.data() + offset, existing.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += static_cast<std::size_t>(count);
    }
    const bool same_bytes = same_size && offset == existing.size() &&
                            std::equal(existing.begin(), existing.end(), bytes.begin());
    const bool synced = same_bytes && ::fsync(descriptor) == 0;
    const int saved_errno = errno;
    ::close(descriptor);
    if (!regular) {
        return fail(error, "BCC-32 immutable object is not a regular file: " +
                               path.string());
    }
    if (!same_bytes) {
        return fail(error, "BCC-32 immutable object collision: " + path.string());
    }
    if (!synced) {
        return fail(error, "cannot synchronize existing BCC-32 immutable object: " +
                               std::string(std::strerror(saved_errno)));
    }
    return sync_directory(path.parent_path(), error);
}

bool write_immutable_object(const std::filesystem::path& path,
                            std::span<const std::uint8_t> bytes,
                            std::string* error) {
    std::error_code filesystem_error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, filesystem_error);
    if (!filesystem_error && std::filesystem::exists(status)) {
        return verify_existing_immutable_object(path, bytes, error);
    }
    if (filesystem_error &&
        filesystem_error != std::errc::no_such_file_or_directory) {
        return fail(error, "cannot inspect BCC-32 immutable object");
    }
    const std::filesystem::path temporary = unique_temporary_path(path);
    if (!write_synced_file(temporary, bytes, error)) return false;
    if (::link(temporary.c_str(), path.c_str()) != 0) {
        const int link_errno = errno;
        std::filesystem::remove(temporary, filesystem_error);
        if (link_errno == EEXIST) {
            return verify_existing_immutable_object(path, bytes, error);
        }
        return fail(error, "cannot publish BCC-32 immutable object: " +
                               std::string(std::strerror(link_errno)));
    }
    std::filesystem::remove(temporary, filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot remove BCC-32 immutable-object temporary");
    }
    return sync_directory(path.parent_path(), error);
}

[[nodiscard]] bool valid_kind(ArtifactKind kind) {
    return kind == ArtifactKind::propagule_capsule ||
           kind == ArtifactKind::adult_continuity_checkpoint ||
           kind == ArtifactKind::cultural_capsule;
}

bool validate_metadata(const ArtifactMetadata& metadata,
                       ArtifactKind expected,
                       std::string* error) {
    if (metadata.schema_version != 4u || metadata.kind != expected ||
        !valid_kind(metadata.kind)) {
        return fail(error, "BCC-32 artifact type or schema does not match the endpoint");
    }
    if (metadata.site_word_bits != kBitsPerSite || metadata.chunk_edge != kChunkEdge ||
        metadata.production_sites != kProductionSites ||
        metadata.production_bits != kProductionBits) {
        return fail(error, "BCC-32 artifact physical word contract is incompatible");
    }
    if (metadata.provenance.law != canonical_law_identity()) {
        return fail(error, "BCC-32 artifact law epoch does not match this executable");
    }
    const ReplayBoundary& boundary = metadata.replay_boundary;
    if (boundary.next_factor != 0u) {
        return fail(error, "BCC-32 artifacts require a complete-superstep boundary");
    }
    if (boundary.contact != ContactBoundary::closed &&
        boundary.contact != ContactBoundary::reversible_external_tape) {
        return fail(error, "BCC-32 contact boundary is invalid");
    }
    if (boundary.taint != ContinuationTaint::natural &&
        boundary.taint != ContinuationTaint::operator_intervention &&
        boundary.taint != ContinuationTaint::synthetic_import) {
        return fail(error, "BCC-32 continuation taint is invalid");
    }
    const bool has_predecessor = boundary.predecessor_commit != ContentAddress{};
    if (has_predecessor &&
        (!is_valid_content_address(boundary.predecessor_commit) ||
         boundary.predecessor_commit.byte_count == 0u)) {
        return fail(error, "BCC-32 predecessor commit address is malformed");
    }
    const bool has_external_state =
        is_valid_content_address(boundary.external_tape_state);
    if ((boundary.contact == ContactBoundary::closed && has_external_state) ||
        (boundary.contact == ContactBoundary::reversible_external_tape &&
         !has_external_state)) {
        return fail(error, "BCC-32 external tape commitment disagrees with contact mode");
    }
    return validate_provenance(metadata.provenance, error);
}

void put_support(Bytes* bytes, const WorldSupport& support) {
    put_u32(bytes, support.has_chunks ? 1u : 0u);
    put_coordinate_component(bytes, support.minimum.x);
    put_coordinate_component(bytes, support.minimum.y);
    put_coordinate_component(bytes, support.minimum.z);
    put_coordinate_component(bytes, support.maximum.x);
    put_coordinate_component(bytes, support.maximum.y);
    put_coordinate_component(bytes, support.maximum.z);
    put_u64(bytes, support.materialized_chunks);
    put_u64(bytes, support.non_quiescent_sites);
    put_u64(bytes, support.direct_word_bytes);
}

bool take_support(std::span<const std::uint8_t> bytes,
                  std::size_t* cursor,
                  WorldSupport* support) {
    std::uint32_t has_chunks = 0;
    if (!take_u32(bytes, cursor, &has_chunks) || has_chunks > 1u ||
        !take_coordinate_component(bytes, cursor, &support->minimum.x) ||
        !take_coordinate_component(bytes, cursor, &support->minimum.y) ||
        !take_coordinate_component(bytes, cursor, &support->minimum.z) ||
        !take_coordinate_component(bytes, cursor, &support->maximum.x) ||
        !take_coordinate_component(bytes, cursor, &support->maximum.y) ||
        !take_coordinate_component(bytes, cursor, &support->maximum.z) ||
        !take_u64(bytes, cursor, &support->materialized_chunks) ||
        !take_u64(bytes, cursor, &support->non_quiescent_sites) ||
        !take_u64(bytes, cursor, &support->direct_word_bytes)) {
        return false;
    }
    support->has_chunks = has_chunks != 0u;
    return true;
}

void put_replay_boundary(Bytes* bytes, const ReplayBoundary& boundary) {
    put_u64(bytes, boundary.completed_supersteps);
    put_u32(bytes, boundary.next_factor);
    put_u32(bytes, static_cast<std::uint32_t>(boundary.contact));
    put_u32(bytes, static_cast<std::uint32_t>(boundary.taint));
    put_address(bytes, boundary.predecessor_commit);
    put_address(bytes, boundary.external_tape_state);
}

bool take_replay_boundary(std::span<const std::uint8_t> bytes,
                          std::size_t* cursor,
                          ReplayBoundary* boundary) {
    std::uint32_t contact = 0u;
    std::uint32_t taint = 0u;
    if (!take_u64(bytes, cursor, &boundary->completed_supersteps) ||
        !take_u32(bytes, cursor, &boundary->next_factor) ||
        !take_u32(bytes, cursor, &contact) ||
        !take_u32(bytes, cursor, &taint) ||
        !take_address(bytes, cursor, &boundary->predecessor_commit) ||
        !take_address(bytes, cursor, &boundary->external_tape_state)) {
        return false;
    }
    boundary->contact = static_cast<ContactBoundary>(contact);
    boundary->taint = static_cast<ContinuationTaint>(taint);
    return true;
}

void put_provenance(Bytes* bytes, const Provenance& provenance) {
    put_u32(bytes, static_cast<std::uint32_t>(provenance.genesis_class));
    put_u64(bytes, provenance.biological_parents.size());
    for (const ContentAddress& parent : provenance.biological_parents) {
        put_address(bytes, parent);
    }
    put_u64(bytes, provenance.causal_inputs.size());
    for (const ContentAddress& input : provenance.causal_inputs) {
        put_address(bytes, input);
    }
    put_address(bytes, provenance.law);
    put_address(bytes, provenance.genesis);
    put_address(bytes, provenance.environment_contact_manifest);
    put_u32(bytes, static_cast<std::uint32_t>(provenance.entry_event.kind));
    put_address(bytes, provenance.entry_event.genesis_entry);
    put_address(bytes, provenance.entry_event.evaluator_transcript);
    put_u64(bytes, provenance.entry_event.next_population_slot);
    put_address(bytes, provenance.replay_commitment);
}

bool take_provenance(std::span<const std::uint8_t> bytes,
                     std::size_t* cursor,
                     Provenance* provenance) {
    std::uint32_t genesis_class = 0;
    std::uint64_t parent_count = 0;
    if (!take_u32(bytes, cursor, &genesis_class) ||
        !take_u64(bytes, cursor, &parent_count) ||
        parent_count > (bytes.size() - *cursor) / 40u) {
        return false;
    }
    provenance->genesis_class = static_cast<GenesisClass>(genesis_class);
    provenance->biological_parents.resize(static_cast<std::size_t>(parent_count));
    for (ContentAddress& parent : provenance->biological_parents) {
        if (!take_address(bytes, cursor, &parent)) return false;
    }
    std::uint64_t causal_count = 0;
    if (!take_u64(bytes, cursor, &causal_count) ||
        causal_count > (bytes.size() - *cursor) / 40u) {
        return false;
    }
    provenance->causal_inputs.resize(static_cast<std::size_t>(causal_count));
    for (ContentAddress& input : provenance->causal_inputs) {
        if (!take_address(bytes, cursor, &input)) return false;
    }
    std::uint32_t entry_kind = 0u;
    if (!take_address(bytes, cursor, &provenance->law) ||
        !take_address(bytes, cursor, &provenance->genesis) ||
        !take_address(bytes, cursor, &provenance->environment_contact_manifest) ||
        !take_u32(bytes, cursor, &entry_kind) ||
        !take_address(bytes, cursor, &provenance->entry_event.genesis_entry) ||
        !take_address(bytes, cursor, &provenance->entry_event.evaluator_transcript) ||
        !take_u64(bytes, cursor, &provenance->entry_event.next_population_slot) ||
        !take_address(bytes, cursor, &provenance->replay_commitment)) {
        return false;
    }
    provenance->entry_event.kind =
        static_cast<EntryEventKind>(entry_kind);
    return true;
}

Bytes serialize_manifest(const Manifest& manifest) {
    Bytes bytes;
    bytes.insert(bytes.end(), kManifestMagic.begin(), kManifestMagic.end());
    const ArtifactMetadata& metadata = manifest.metadata;
    put_u32(&bytes, metadata.schema_version);
    put_u32(&bytes, static_cast<std::uint32_t>(metadata.kind));
    put_u32(&bytes, metadata.site_word_bits);
    put_u32(&bytes, metadata.chunk_edge);
    put_u64(&bytes, metadata.production_sites);
    put_u64(&bytes, metadata.production_bits);
    put_provenance(&bytes, metadata.provenance);
    put_replay_boundary(&bytes, metadata.replay_boundary);
    put_support(&bytes, metadata.world_support);
    put_u64(&bytes, metadata.byte_counts.direct_chunk_bytes);
    put_u64(&bytes, metadata.byte_counts.manifest_bytes);
    put_u64(&bytes, metadata.byte_counts.durable_bytes);
    put_u64(&bytes, manifest.chunks.size());
    for (const ChunkObjectReference& chunk : manifest.chunks) {
        put_coordinate_component(&bytes, chunk.coordinate.x);
        put_coordinate_component(&bytes, chunk.coordinate.y);
        put_coordinate_component(&bytes, chunk.coordinate.z);
        put_u64(&bytes, kChunkBytes);
        put_hash(&bytes, chunk.digest);
        put_u64(&bytes, chunk.non_quiescent_sites);
        put_i64(&bytes, chunk.delta_n_q);
    }
    return bytes;
}
