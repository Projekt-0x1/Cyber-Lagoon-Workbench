// Included inside bcc32_cuda_adult_complete_checkpoint_v2. This file owns
// checkpoint transport only; schema identity and organism state stay in the
// including header.

constexpr std::size_t kCheckpointTransferStageBytes = 16u * 1024u * 1024u;

class CheckpointTransferWorkspace {
public:
  explicit CheckpointTransferWorkspace(std::size_t bytes) : capacity_(bytes) {
    adult::cuda_require(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "create complete adult checkpoint transfer stream");
    if (capacity_ != 0u) {
      adult::cuda_require(
          cudaMallocHost(reinterpret_cast<void **>(&bytes_), capacity_),
          "allocate complete adult checkpoint pinned staging");
    }
  }

  CheckpointTransferWorkspace(const CheckpointTransferWorkspace &) = delete;
  CheckpointTransferWorkspace &
  operator=(const CheckpointTransferWorkspace &) = delete;

  ~CheckpointTransferWorkspace() {
    if (stream_ != nullptr)
      (void)cudaStreamDestroy(stream_);
    if (bytes_ != nullptr)
      (void)cudaFreeHost(bytes_);
  }

  [[nodiscard]] std::uint8_t *data() { return bytes_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }

  void clear_async(void *destination, std::size_t bytes) {
    if (bytes != 0u) {
      adult::cuda_require(cudaMemsetAsync(destination, 0, bytes, stream_),
                          "enqueue complete adult checkpoint padding clear");
    }
  }

  void copy_batch_and_wait(const std::vector<void *> &destinations,
                           const std::vector<const void *> &sources,
                           const std::vector<std::size_t> &sizes,
                           cudaMemcpyKind fallback_kind) {
    if (destinations.size() != sources.size() || sources.size() != sizes.size())
      throw std::logic_error(
          "complete adult checkpoint copy batch shape mismatch");
    if (!destinations.empty()) {
#if CUDART_VERSION >= 13000
      (void)fallback_kind;
      cudaMemcpyAttributes attributes{};
      attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      std::size_t attributes_index = 0u;
      adult::cuda_require(cudaMemcpyBatchAsync(destinations.data(),
                                               sources.data(), sizes.data(),
                                               sizes.size(), &attributes,
                                               &attributes_index, 1u, stream_),
                          "enqueue complete adult checkpoint copy batch");
#else
      for (std::size_t index = 0u; index < sizes.size(); ++index) {
        adult::cuda_require(cudaMemcpyAsync(destinations[index], sources[index],
                                            sizes[index], fallback_kind,
                                            stream_),
                            "enqueue complete adult checkpoint copy");
      }
#endif
    }
    adult::cuda_require(cudaStreamSynchronize(stream_),
                        "complete adult checkpoint copy batch");
  }

private:
  std::uint8_t *bytes_ = nullptr;
  std::size_t capacity_ = 0u;
  cudaStream_t stream_ = nullptr;
};

struct CheckpointTransferArena {
  CheckpointTransferArena() : workspace(kCheckpointTransferStageBytes) {}

  CheckpointTransferWorkspace workspace;
  std::mutex mutex;
};

inline CheckpointTransferArena &checkpoint_transfer_arena() {
  static CheckpointTransferArena arena;
  return arena;
}

class CheckpointTransferLease {
public:
  CheckpointTransferLease()
      : arena_(checkpoint_transfer_arena()), lock_(arena_.mutex) {}

  [[nodiscard]] CheckpointTransferWorkspace &workspace() {
    return arena_.workspace;
  }

  void release() {
    if (lock_.owns_lock())
      lock_.unlock();
  }

private:
  CheckpointTransferArena &arena_;
  std::unique_lock<std::mutex> lock_;
};

struct CheckpointDeviceField {
  const char *name = nullptr;
  FieldClass classification = FieldClass::persistent_learned_matter;
  const void *device_data = nullptr;
  std::uint32_t element_bytes = 0u;
  std::uint64_t count = 0u;
  std::size_t bytes = 0u;
};

inline void update_receipt(Receipt *receipt, FieldClass classification,
                           std::uint64_t name, std::uint64_t data,
                           std::uint64_t bytes);

class CheckpointDeviceFields {
public:
  template <typename T>
  void add(const char *name, FieldClass classification,
           const adult::DeviceArray<T> &array, std::size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) <= std::numeric_limits<std::uint32_t>::max());
    if (count > array.size() || count > kMaximumFieldBytes / sizeof(T))
      throw std::runtime_error(
          "complete adult checkpoint stage exceeds allocation: " +
          std::string(name));
    const std::size_t bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<std::size_t>::max() - total_bytes_)
      throw std::overflow_error(
          "complete adult checkpoint staging extent overflow");
    fields_.push_back({name, classification, array.get(),
                       static_cast<std::uint32_t>(sizeof(T)), count, bytes});
    total_bytes_ += bytes;
  }

  void write(std::ofstream *output, Receipt *receipt) const {
    std::streampos large_header_position{};
    ArrayHeader large_header{};
    std::uint64_t large_hash = 1469598103934665603ull;
    visit(
        [&](const CheckpointDeviceField &field, const std::uint8_t *bytes) {
          const ArrayHeader header{
              hash_name(field.name),
              static_cast<std::uint32_t>(field.classification),
              field.element_bytes, field.count, hash_bytes(bytes, field.bytes)};
          output->write(reinterpret_cast<const char *>(&header),
                        sizeof(header));
          if (field.bytes != 0u)
            output->write(reinterpret_cast<const char *>(bytes), field.bytes);
          if (!*output)
            throw std::runtime_error(
                "complete adult checkpoint field write failed");
          update_receipt(receipt, field.classification, header.name_hash,
                         header.data_hash, field.bytes);
        },
        [&](const CheckpointDeviceField &field) {
          large_hash = 1469598103934665603ull;
          large_header = {hash_name(field.name),
                          static_cast<std::uint32_t>(field.classification),
                          field.element_bytes, field.count, 0u};
          large_header_position = output->tellp();
          output->write(reinterpret_cast<const char *>(&large_header),
                        sizeof(large_header));
        },
        [&](const std::uint8_t *bytes, std::size_t count) {
          large_hash = hash_bytes(bytes, count, large_hash);
          output->write(reinterpret_cast<const char *>(bytes), count);
          if (!*output)
            throw std::runtime_error(
                "complete adult checkpoint field write failed");
        },
        [&](const CheckpointDeviceField &field) {
          const std::streampos end = output->tellp();
          large_header.data_hash = large_hash;
          output->seekp(large_header_position);
          output->write(reinterpret_cast<const char *>(&large_header),
                        sizeof(large_header));
          output->seekp(end);
          if (!*output)
            throw std::runtime_error(
                "complete adult checkpoint field header patch failed");
          update_receipt(receipt, field.classification, large_header.name_hash,
                         large_header.data_hash, field.bytes);
        });
  }

  void hash_into(Receipt *receipt) const {
    std::uint64_t large_hash = 1469598103934665603ull;
    visit(
        [&](const CheckpointDeviceField &field, const std::uint8_t *bytes) {
          update_receipt(receipt, field.classification, hash_name(field.name),
                         hash_bytes(bytes, field.bytes), field.bytes);
        },
        [&](const CheckpointDeviceField &) {
          large_hash = 1469598103934665603ull;
        },
        [&](const std::uint8_t *bytes, std::size_t count) {
          large_hash = hash_bytes(bytes, count, large_hash);
        },
        [&](const CheckpointDeviceField &field) {
          update_receipt(receipt, field.classification, hash_name(field.name),
                         large_hash, field.bytes);
        });
  }

private:
  template <typename Whole, typename BeginLarge, typename LargeChunk,
            typename EndLarge>
  void visit(Whole whole, BeginLarge begin_large, LargeChunk large_chunk,
             EndLarge end_large) const {
    if (fields_.empty())
      return;
    CheckpointTransferLease transfer;
    adult::cuda_require(cudaDeviceSynchronize(),
                        "quiesce complete adult checkpoint device fields");
    CheckpointTransferWorkspace &workspace = transfer.workspace();
    std::size_t cursor = 0u;
    while (cursor < fields_.size()) {
      const CheckpointDeviceField &first = fields_[cursor];
      if (first.bytes > workspace.capacity()) {
        begin_large(first);
        std::size_t copied = 0u;
        while (copied < first.bytes) {
          const std::size_t chunk =
              std::min(workspace.capacity(), first.bytes - copied);
          std::vector<void *> destinations{workspace.data()};
          std::vector<const void *> sources{
              static_cast<const std::uint8_t *>(first.device_data) + copied};
          std::vector<std::size_t> sizes{chunk};
          workspace.copy_batch_and_wait(destinations, sources, sizes,
                                        cudaMemcpyDeviceToHost);
          large_chunk(workspace.data(), chunk);
          copied += chunk;
        }
        end_large(first);
        ++cursor;
        continue;
      }

      const std::size_t begin = cursor;
      std::size_t used = 0u;
      std::vector<std::size_t> offsets;
      std::vector<void *> destinations;
      std::vector<const void *> sources;
      std::vector<std::size_t> sizes;
      while (cursor < fields_.size()) {
        const CheckpointDeviceField &field = fields_[cursor];
        if (field.bytes > workspace.capacity() ||
            field.bytes > workspace.capacity() - used)
          break;
        offsets.push_back(used);
        if (field.bytes != 0u) {
          destinations.push_back(workspace.data() + used);
          sources.push_back(field.device_data);
          sizes.push_back(field.bytes);
        }
        used += field.bytes;
        ++cursor;
      }
      if (!destinations.empty())
        workspace.copy_batch_and_wait(destinations, sources, sizes,
                                      cudaMemcpyDeviceToHost);
      for (std::size_t index = begin; index < cursor; ++index) {
        const std::size_t offset = offsets[index - begin];
        whole(fields_[index], workspace.data() == nullptr
                                  ? nullptr
                                  : workspace.data() + offset);
      }
    }
  }

  std::vector<CheckpointDeviceField> fields_;
  std::size_t total_bytes_ = 0u;
};

template <typename T>
inline std::vector<std::uint8_t> stage_array(const adult::DeviceArray<T> &array,
                                             std::size_t count) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (count > array.size())
    throw std::runtime_error(
        "complete adult checkpoint stage exceeds allocation");
  std::vector<std::uint8_t> bytes(count * sizeof(T));
  if (!bytes.empty()) {
    adult::cuda_require(cudaMemcpy(bytes.data(), array.get(), bytes.size(),
                                   cudaMemcpyDeviceToHost),
                        "stage complete adult checkpoint field");
  }
  return bytes;
}

inline void update_receipt(Receipt *receipt, FieldClass classification,
                           std::uint64_t name, std::uint64_t data,
                           std::uint64_t bytes) {
  std::uint64_t *target = &receipt->persistent_hash;
  if (classification == FieldClass::transient_workspace)
    target = &receipt->workspace_hash;
  else if (classification == FieldClass::observer_diagnostic)
    target = &receipt->diagnostic_hash;
  mix_hash(target, name);
  mix_hash(target, data);
  mix_hash(target, bytes);
  receipt->payload_bytes += bytes;
  ++receipt->array_fields;
}

template <typename T>
inline void write_array(std::ofstream *output, const char *name,
                        FieldClass classification,
                        const adult::DeviceArray<T> &array, std::size_t count,
                        Receipt *receipt) {
  const std::vector<std::uint8_t> bytes = stage_array(array, count);
  const ArrayHeader header{hash_name(name),
                           static_cast<std::uint32_t>(classification),
                           static_cast<std::uint32_t>(sizeof(T)), count,
                           hash_bytes(bytes.data(), bytes.size())};
  output->write(reinterpret_cast<const char *>(&header), sizeof(header));
  if (!bytes.empty())
    output->write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!*output)
    throw std::runtime_error("complete adult checkpoint field write failed");
  update_receipt(receipt, classification, header.name_hash, header.data_hash,
                 bytes.size());
}

// Legacy-fixture writer: historical checkpoints encoded allocation extent.
template <typename T>
inline void write_array(std::ofstream *output, const char *name,
                        FieldClass classification,
                        const adult::DeviceArray<T> &array, Receipt *receipt) {
  write_array(output, name, classification, array, array.size(), receipt);
}

template <typename T>
inline void write_host_vector(std::ofstream *output, const char *name,
                              FieldClass classification,
                              const std::vector<T> &values, Receipt *receipt) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.size() > kMaximumFieldBytes / sizeof(T))
    throw std::runtime_error("complete adult host vector exceeds limit: " +
                             std::string(name));
  const std::size_t byte_count = values.size() * sizeof(T);
  const ArrayHeader header{hash_name(name),
                           static_cast<std::uint32_t>(classification),
                           static_cast<std::uint32_t>(sizeof(T)), values.size(),
                           hash_bytes(values.data(), byte_count)};
  output->write(reinterpret_cast<const char *>(&header), sizeof(header));
  if (byte_count != 0u)
    output->write(reinterpret_cast<const char *>(values.data()), byte_count);
  if (!*output)
    throw std::runtime_error("complete adult host vector write failed");
  update_receipt(receipt, classification, header.name_hash, header.data_hash,
                 byte_count);
}

class CheckpointFieldReader {
public:
  explicit CheckpointFieldReader(Receipt *receipt)
      : receipt_(receipt), workspace_(transfer_.workspace()) {}

  template <typename T>
  void read(std::ifstream *input, const char *name, FieldClass classification,
            adult::DeviceArray<T> *array) {
    static_assert(std::is_trivially_copyable_v<T>);
    ArrayHeader header{};
    input->read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!*input || header.name_hash != hash_name(name) ||
        header.classification != static_cast<std::uint32_t>(classification) ||
        header.element_bytes != sizeof(T) ||
        header.count > std::numeric_limits<std::size_t>::max() ||
        header.count > kMaximumFieldBytes / sizeof(T)) {
      throw std::runtime_error("invalid complete adult checkpoint field: " +
                               std::string(name));
    }
    const std::size_t bytes =
        static_cast<std::size_t>(header.count) * sizeof(T);
    if (bytes > workspace_.capacity()) {
      flush();
      read_large(input, name, classification, header, array, bytes);
      return;
    }
    if (bytes > workspace_.capacity() - used_)
      flush();
    const std::size_t offset = used_;
    if (bytes != 0u)
      input->read(reinterpret_cast<char *>(workspace_.data() + offset), bytes);
    if (!*input ||
        hash_bytes(workspace_.data() + offset, bytes) != header.data_hash) {
      throw std::runtime_error("corrupt complete adult checkpoint field: " +
                               std::string(name));
    }
    const std::size_t capacity =
        restored_array_capacity(name, static_cast<std::size_t>(header.count));
    array->allocate(capacity);
    pending_.push_back({array->get(), capacity * sizeof(T), bytes, offset,
                        classification, header.name_hash, header.data_hash});
    used_ += bytes;
  }

  void flush() {
    if (pending_.empty())
      return;
    std::vector<void *> destinations;
    std::vector<const void *> sources;
    std::vector<std::size_t> sizes;
    for (const PendingField &field : pending_) {
      if (field.capacity_bytes > field.payload_bytes)
        workspace_.clear_async(field.destination, field.capacity_bytes);
      if (field.payload_bytes != 0u) {
        destinations.push_back(field.destination);
        sources.push_back(workspace_.data() + field.offset);
        sizes.push_back(field.payload_bytes);
      }
    }
    workspace_.copy_batch_and_wait(destinations, sources, sizes,
                                   cudaMemcpyHostToDevice);
    for (const PendingField &field : pending_) {
      update_receipt(receipt_, field.classification, field.name_hash,
                     field.data_hash, field.payload_bytes);
    }
    pending_.clear();
    used_ = 0u;
  }

  void finish() {
    flush();
    transfer_.release();
  }

private:
  struct PendingField {
    void *destination = nullptr;
    std::size_t capacity_bytes = 0u;
    std::size_t payload_bytes = 0u;
    std::size_t offset = 0u;
    FieldClass classification = FieldClass::persistent_learned_matter;
    std::uint64_t name_hash = 0u;
    std::uint64_t data_hash = 0u;
  };

  template <typename T>
  void read_large(std::ifstream *input, const char *name,
                  FieldClass classification, const ArrayHeader &header,
                  adult::DeviceArray<T> *array, std::size_t bytes) {
    std::vector<std::uint8_t> staged(bytes);
    input->read(reinterpret_cast<char *>(staged.data()), staged.size());
    if (!*input || hash_bytes(staged.data(), staged.size()) != header.data_hash)
      throw std::runtime_error("corrupt complete adult checkpoint field: " +
                               std::string(name));
    const std::size_t capacity =
        restored_array_capacity(name, static_cast<std::size_t>(header.count));
    array->allocate(capacity);
    if (capacity > static_cast<std::size_t>(header.count)) {
      adult::cuda_require(cudaMemset(array->get(), 0, array->bytes()),
                          "clear compact complete adult checkpoint field");
    }
    if (!staged.empty()) {
      adult::cuda_require(cudaMemcpy(array->get(), staged.data(), staged.size(),
                                     cudaMemcpyHostToDevice),
                          "restore complete adult checkpoint field");
    }
    update_receipt(receipt_, classification, header.name_hash, header.data_hash,
                   bytes);
  }

  Receipt *receipt_ = nullptr;
  CheckpointTransferLease transfer_;
  CheckpointTransferWorkspace &workspace_;
  std::vector<PendingField> pending_;
  std::size_t used_ = 0u;
};

template <typename T>
inline void read_host_vector(std::ifstream *input, const char *name,
                             FieldClass classification, std::vector<T> *values,
                             Receipt *receipt) {
  static_assert(std::is_trivially_copyable_v<T>);
  ArrayHeader header{};
  input->read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!*input || header.name_hash != hash_name(name) ||
      header.classification != static_cast<std::uint32_t>(classification) ||
      header.element_bytes != sizeof(T) ||
      header.count > std::numeric_limits<std::size_t>::max() ||
      header.count > kMaximumFieldBytes / sizeof(T)) {
    throw std::runtime_error("invalid complete adult host vector: " +
                             std::string(name));
  }
  values->resize(static_cast<std::size_t>(header.count));
  const std::size_t byte_count = values->size() * sizeof(T);
  if (byte_count != 0u)
    input->read(reinterpret_cast<char *>(values->data()), byte_count);
  if (!*input || hash_bytes(values->data(), byte_count) != header.data_hash)
    throw std::runtime_error("corrupt complete adult host vector: " +
                             std::string(name));
  update_receipt(receipt, classification, header.name_hash, header.data_hash,
                 byte_count);
}
