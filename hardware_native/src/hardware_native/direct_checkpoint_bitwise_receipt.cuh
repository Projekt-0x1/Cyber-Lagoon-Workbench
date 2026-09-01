#ifndef HARDWARE_NATIVE_DIRECT_CHECKPOINT_BITWISE_RECEIPT_CUH
#define HARDWARE_NATIVE_DIRECT_CHECKPOINT_BITWISE_RECEIPT_CUH

// github #1445 -- bitwise checkpoint receipt.
//
// The V02 capture already carries a payload digest and its restore refuses a
// mismatch before any allocation. What the continuity law still needs is a
// SERIALIZED form: one flat byte blob whose every flipped byte is refused,
// whose bytes are a pure function of captured state (two captures of one
// subject at one tick are byte-identical), and whose parser validates magic,
// versions, framing, and the outer digest before it writes a single field of
// the output checkpoint. This header is that layer and nothing more.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "hardware_native/direct_adult_checkpoint.cuh"
#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kDirectCheckpointBlobMagic = 0x31424358u;  // "XCB1"
inline constexpr std::uint32_t kDirectCheckpointBlobVersion = 1u;

struct DirectCheckpointBitwiseReceipt {
  substrate::direct_network::DirectSha256Address blob_sha256;
  substrate::direct_network::DirectSha256Address payload_sha256;
  std::uint32_t format_version = 0u;
  std::size_t blob_bytes = 0u;
};

namespace checkpoint_blob_detail {

using substrate::direct_network::detail::DirectSha256State;

struct Writer {
  std::vector<std::uint8_t> out;
  void u32(std::uint32_t value) {
    for (std::uint32_t i = 0u; i < 4u; ++i)
      out.push_back(static_cast<std::uint8_t>(value >> (i * 8u)));
  }
  void u64(std::uint64_t value) {
    for (std::uint32_t i = 0u; i < 8u; ++i)
      out.push_back(static_cast<std::uint8_t>(value >> (i * 8u)));
  }
  void trivial(const void* value, std::size_t size) {
    const auto* begin = static_cast<const std::uint8_t*>(value);
    out.insert(out.end(), begin, begin + size);
  }
  void bytes(const void* value, std::size_t size) {
    u64(size);
    trivial(value, size);
  }
};

struct Reader {
  const std::uint8_t* data;
  std::size_t size;
  std::size_t cursor = 0u;
  void take(std::size_t count, void* out) {
    if (size - cursor < count) throw std::runtime_error(
        "checkpoint blob: truncated");
    std::memcpy(out, data + cursor, count);
    cursor += count;
  }
  std::uint32_t u32() {
    std::uint8_t raw[4];
    take(sizeof(raw), raw);
    std::uint32_t value = 0u;
    for (std::uint32_t i = 0u; i < 4u; ++i)
      value |= static_cast<std::uint32_t>(raw[i]) << (i * 8u);
    return value;
  }
  std::uint64_t u64() {
    std::uint8_t raw[8];
    take(sizeof(raw), raw);
    std::uint64_t value = 0u;
    for (std::uint32_t i = 0u; i < 8u; ++i)
      value |= static_cast<std::uint64_t>(raw[i]) << (i * 8u);
    return value;
  }
  void trivial(void* out, std::size_t size) { take(size, out); }
  void bytes(std::vector<std::uint8_t>& out) {
    const std::uint64_t size = u64();
    if (size > this->size - cursor) throw std::runtime_error(
        "checkpoint blob: truncated vector length");
    out.resize(static_cast<std::size_t>(size));
    if (size != 0u) take(static_cast<std::size_t>(size), out.data());
  }
};

// Outer digest over the framed blob minus its trailing digest trailer.
DirectSha256State outer_state(const std::uint8_t* data, std::size_t framed) {
  static constexpr char kDomain[] = "0x1-direct-checkpoint-bitwise-blob";
  DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  state.update(data, framed);
  return state;
}

}  // namespace checkpoint_blob_detail

// Deterministic field-wise dump in exactly the payload-digest order, framed
// by magic, blob version, adult format version, payload length, and closed
// by the outer SHA-256 over everything before it.
inline void serialize_direct_checkpoint_bitwise(
    const DirectAdultCheckpoint& checkpoint,
    std::vector<std::uint8_t>& blob) {
  namespace detail = checkpoint_blob_detail;
  detail::Writer payload{};
  payload.u32(checkpoint.format_version);
  payload.trivial(&checkpoint.brain, sizeof(checkpoint.brain));
  payload.u32(checkpoint.has_resource_ecology ? 1u : 0u);
  if (checkpoint.has_resource_ecology)
    payload.trivial(&checkpoint.resource_ecology,
                    sizeof(checkpoint.resource_ecology));
  payload.trivial(checkpoint.arena_pointer_offsets.data(),
                  checkpoint.arena_pointer_offsets.size() *
                      sizeof(std::uint64_t));
  payload.bytes(checkpoint.arena.data(),
                checkpoint.arena.size() * sizeof(std::byte));
  for (const auto& buffer : checkpoint.device_buffers)
    payload.bytes(buffer.data(), buffer.size() * sizeof(std::byte));
  payload.bytes(checkpoint.host_ingress_staging.data(),
                checkpoint.host_ingress_staging.size() *
                    sizeof(ActivityEvent));
  payload.bytes(checkpoint.host_ingress_contact_staging.data(),
                checkpoint.host_ingress_contact_staging.size() *
                    sizeof(ResidentContactEpochReceipt));
  payload.bytes(checkpoint.host_consequence_staging.data(),
                checkpoint.host_consequence_staging.size() *
                    sizeof(ConsequenceIngressEvent));
  payload.trivial(&checkpoint.config, sizeof(checkpoint.config));
  payload.u64(checkpoint.resident_development_epochs);
  payload.u32(checkpoint.current_tick);
  payload.u32(checkpoint.participation_staging_capacity);
  payload.u32(checkpoint.host_ingress_write_tail);
  payload.u32(checkpoint.host_ingress_publish_tail);
  payload.u32(checkpoint.host_ingress_observed_head);
  payload.u32(checkpoint.host_ingress_dispatched_tail);
  payload.u64(checkpoint.host_ingress_overflow_drops);
  payload.u64(checkpoint.host_ingress_protocol_faults);
  payload.u32(checkpoint.host_consequence_write_tail);
  payload.u32(checkpoint.host_consequence_publish_tail);
  payload.u32(checkpoint.host_consequence_observed_head);
  payload.u64(checkpoint.host_consequence_overflow_drops);
  payload.u64(checkpoint.host_consequence_protocol_faults);
  payload.u32(checkpoint.host_ingress_head_snapshot);
  payload.u32(checkpoint.host_ingress_publish_slot);
  payload.u32(checkpoint.host_consequence_head_snapshot);
  payload.u32(checkpoint.host_consequence_publish_slot);
  payload.trivial(&checkpoint.payload_sha256,
                  sizeof(checkpoint.payload_sha256));

  blob.clear();
  detail::Writer frame{};
  frame.u32(kDirectCheckpointBlobMagic);
  frame.u32(kDirectCheckpointBlobVersion);
  frame.u32(kDirectCheckpointBlobVersion);  // adult format echoed at parse
  frame.u32(checkpoint.format_version);
  frame.u64(payload.out.size());
  blob = std::move(frame.out);
  blob.insert(blob.end(), payload.out.begin(), payload.out.end());
  auto state = detail::outer_state(blob.data(), blob.size());
  const auto digest = state.finish();
  for (std::uint8_t byte : digest.byte) blob.push_back(byte);
}

inline DirectCheckpointBitwiseReceipt direct_checkpoint_bitwise_receipt(
    const DirectAdultCheckpoint& checkpoint,
    const std::vector<std::uint8_t>& blob) {
  DirectCheckpointBitwiseReceipt receipt{};
  receipt.format_version = checkpoint.format_version;
  receipt.payload_sha256 = checkpoint.payload_sha256;
  auto state = checkpoint_blob_detail::outer_state(blob.data(),
                                                   blob.size() - 32u);
  receipt.blob_sha256 = state.finish();
  receipt.blob_bytes = blob.size();
  return receipt;
}

// Validate framing, versions, and the outer digest BEFORE any field of `out`
// is written; only a fully verified payload mutates the caller's checkpoint.
inline void parse_direct_checkpoint_bitwise(
    const std::uint8_t* data, std::size_t size,
    DirectAdultCheckpoint& out) {
  namespace detail = checkpoint_blob_detail;
  if (size < 24u + 32u) throw std::runtime_error(
      "checkpoint blob: truncated header");
  detail::Reader header{data, size, 0u};
  const std::uint32_t magic = header.u32();
  if (magic != kDirectCheckpointBlobMagic)
    throw std::runtime_error("checkpoint blob: foreign magic");
  const std::uint32_t blob_version = header.u32();
  if (blob_version != kDirectCheckpointBlobVersion)
    throw std::runtime_error("checkpoint blob: foreign blob version");
  (void)header.u32();  // reserved echo slot
  const std::uint32_t format_version = header.u32();
  if (format_version != kDirectAdultCheckpointVersion)
    throw std::runtime_error("checkpoint blob: foreign adult format version");
  const std::uint64_t payload_size = header.u64();
  if (payload_size > size - 24u - 32u)
    throw std::runtime_error("checkpoint blob: truncated payload");
  const std::size_t framed = 24u + static_cast<std::size_t>(payload_size);
  auto state = detail::outer_state(data, framed);
  const auto expected = state.finish();
  std::uint8_t stored[32];
  std::memcpy(stored, data + framed, sizeof(stored));
  if (std::memcmp(stored, expected.byte, sizeof(stored)) != 0)
    throw std::runtime_error("checkpoint blob: digest mismatch");

  // Verified: now materialize the checkpoint. Any residual defect surfaces
  // in the payload digest the canonical restore rechecks before allocating.
  detail::Reader payload{data + 24u, static_cast<std::size_t>(payload_size), 0u};
  DirectAdultCheckpoint parsed{};
  parsed.format_version = payload.u32();
  payload.trivial(&parsed.brain, sizeof(parsed.brain));
  const std::uint32_t has_ecology = payload.u32();
  parsed.has_resource_ecology = has_ecology != 0u;
  if (parsed.has_resource_ecology)
    payload.trivial(&parsed.resource_ecology,
                    sizeof(parsed.resource_ecology));
  payload.trivial(parsed.arena_pointer_offsets.data(),
                  parsed.arena_pointer_offsets.size() *
                      sizeof(std::uint64_t));
  {
    std::vector<std::uint8_t> raw;
    payload.bytes(raw);
    parsed.arena.resize(raw.size());
    std::memcpy(parsed.arena.data(), raw.data(), raw.size());
  }
  for (auto& buffer : parsed.device_buffers) {
    std::vector<std::uint8_t> raw;
    payload.bytes(raw);
    buffer.resize(raw.size());
    std::memcpy(buffer.data(), raw.data(), raw.size());
  }
  {
    std::vector<std::uint8_t> raw;
    payload.bytes(raw);
    parsed.host_ingress_staging.resize(raw.size() / sizeof(ActivityEvent));
    if (!raw.empty())
      std::memcpy(parsed.host_ingress_staging.data(), raw.data(), raw.size());
  }
  {
    std::vector<std::uint8_t> raw;
    payload.bytes(raw);
    if (raw.size() % sizeof(ResidentContactEpochReceipt) != 0u)
      throw std::runtime_error(
          "checkpoint blob: malformed contact receipt staging");
    parsed.host_ingress_contact_staging.resize(
        raw.size() / sizeof(ResidentContactEpochReceipt));
    if (!raw.empty())
      std::memcpy(parsed.host_ingress_contact_staging.data(), raw.data(),
                  raw.size());
  }
  {
    std::vector<std::uint8_t> raw;
    payload.bytes(raw);
    parsed.host_consequence_staging.resize(
        raw.size() / sizeof(ConsequenceIngressEvent));
    if (!raw.empty())
      std::memcpy(parsed.host_consequence_staging.data(), raw.data(),
                  raw.size());
  }
  payload.trivial(&parsed.config, sizeof(parsed.config));
  parsed.resident_development_epochs = payload.u64();
  parsed.current_tick = payload.u32();
  parsed.participation_staging_capacity = payload.u32();
  parsed.host_ingress_write_tail = payload.u32();
  parsed.host_ingress_publish_tail = payload.u32();
  parsed.host_ingress_observed_head = payload.u32();
  parsed.host_ingress_dispatched_tail = payload.u32();
  parsed.host_ingress_overflow_drops = payload.u64();
  parsed.host_ingress_protocol_faults = payload.u64();
  parsed.host_consequence_write_tail = payload.u32();
  parsed.host_consequence_publish_tail = payload.u32();
  parsed.host_consequence_observed_head = payload.u32();
  parsed.host_consequence_overflow_drops = payload.u64();
  parsed.host_consequence_protocol_faults = payload.u64();
  parsed.host_ingress_head_snapshot = payload.u32();
  parsed.host_ingress_publish_slot = payload.u32();
  parsed.host_consequence_head_snapshot = payload.u32();
  parsed.host_consequence_publish_slot = payload.u32();
  payload.trivial(&parsed.payload_sha256, sizeof(parsed.payload_sha256));
  if (parsed.format_version != kDirectAdultCheckpointVersion ||
      !(parsed.payload_sha256 ==
        direct_adult_checkpoint_payload_digest(parsed)))
    throw std::runtime_error("checkpoint blob: payload digest mismatch");
  out = std::move(parsed);
}

}  // namespace substrate::direct_adult_core

#endif
