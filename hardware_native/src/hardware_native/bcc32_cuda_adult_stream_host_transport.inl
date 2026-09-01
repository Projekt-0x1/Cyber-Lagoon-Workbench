// Included inside bcc32_cuda_adult_stream_v1. This host-only staging buffer
// publishes already-selected motor bytes; it has no resident semantic or
// action-selection authority.

class PinnedEmissionPublication {
 public:
  PinnedEmissionPublication() = default;
  PinnedEmissionPublication(const PinnedEmissionPublication&) = delete;
  PinnedEmissionPublication& operator=(const PinnedEmissionPublication&) = delete;
  PinnedEmissionPublication(PinnedEmissionPublication&&) = delete;
  PinnedEmissionPublication& operator=(PinnedEmissionPublication&&) = delete;

  ~PinnedEmissionPublication() {
    if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    if (ready_event_ != nullptr) (void)cudaEventDestroy(ready_event_);
    reset_buffer();
  }

  void allocate(std::size_t payload_capacity) {
    if (bytes_ != nullptr && payload_capacity <= payload_capacity_) return;
    ensure_runtime();
    reset_buffer();
    adult::cuda_require(
        cudaMallocHost(reinterpret_cast<void**>(&bytes_),
                       kCountsBytes + payload_capacity),
        "allocate pinned resident emission publication");
    payload_capacity_ = payload_capacity;
  }

  [[nodiscard]] bool can_stage(std::size_t payload_capacity) const {
    return bytes_ != nullptr && payload_capacity <= payload_capacity_;
  }

  void copy_from_device_and_wait(const std::uint32_t* device_counts,
                                 const std::uint8_t* device_payload,
                                 std::size_t payload_bytes) {
    if (!can_stage(payload_bytes))
      throw std::logic_error("resident emission publication exceeds staging");

    std::array<void*, 2u> destinations{bytes_, bytes_ + kCountsBytes};
    std::array<const void*, 2u> sources{device_counts, device_payload};
    std::array<std::size_t, 2u> sizes{kCountsBytes, payload_bytes};
    adult::cuda_require(cudaEventRecord(ready_event_, nullptr),
                        "record resident emission readiness");
    adult::cuda_require(cudaStreamWaitEvent(stream_, ready_event_, 0u),
                        "order resident emission publication");
#if CUDART_VERSION >= 13000
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t attributes_index = 0u;
    adult::cuda_require(
        cudaMemcpyBatchAsync(destinations.data(), sources.data(), sizes.data(),
                             sizes.size(), &attributes, &attributes_index, 1u,
                             stream_),
        "enqueue resident emission publication batch");
#else
    for (std::size_t index = 0u; index < sizes.size(); ++index) {
      adult::cuda_require(
          cudaMemcpyAsync(destinations[index], sources[index], sizes[index],
                          cudaMemcpyDeviceToHost, stream_),
          "enqueue resident emission publication copy");
    }
#endif
    adult::cuda_require(cudaStreamSynchronize(stream_),
                        "complete resident emission publication");
  }

  void read_counts(std::uint32_t (&counts)[2]) const {
    std::memcpy(counts, bytes_, kCountsBytes);
  }

  [[nodiscard]] const std::uint8_t* payload() const {
    return bytes_ + kCountsBytes;
  }

 private:
  static constexpr std::size_t kCountsBytes = 2u * sizeof(std::uint32_t);

  void ensure_runtime() {
    if (stream_ == nullptr) {
      adult::cuda_require(
          cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
          "create resident emission publication stream");
    }
    if (ready_event_ == nullptr) {
      adult::cuda_require(
          cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming),
          "create resident emission publication event");
    }
  }

  void reset_buffer() noexcept {
    if (bytes_ != nullptr) (void)cudaFreeHost(bytes_);
    bytes_ = nullptr;
    payload_capacity_ = 0u;
  }

  std::uint8_t* bytes_ = nullptr;
  std::size_t payload_capacity_ = 0u;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ready_event_ = nullptr;
};

inline PinnedEmissionPublication& pinned_emission_publication() {
  thread_local PinnedEmissionPublication publication;
  return publication;
}
