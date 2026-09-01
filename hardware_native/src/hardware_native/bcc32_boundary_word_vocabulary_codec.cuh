#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_BOUNDARY_WORD_HD __host__ __device__
#else
#define BCC32_BOUNDARY_WORD_HD
#endif

// Lossless framing for one raw boundary word.  The codec deliberately carries
// only channel and payload bits: it has no vocabulary table, token identity,
// parser, semantic label, or host-selected rendering rule.  The private frame
// nibble is an internal transport distinction, not evidence about who produced
// a word.  Packet seals detect accidental/stale use only; they are neither
// cryptographic authentication nor a production-reachability receipt.
namespace substrate::bcc32::boundary_word_codec {

using BoundaryWord = std::uint32_t;

inline constexpr std::uint32_t kPayloadBits = 24u;
inline constexpr std::uint32_t kChannelBits = 4u;
inline constexpr std::uint32_t kChannelShift = kPayloadBits;
inline constexpr std::uint32_t kFrameShift = kPayloadBits + kChannelBits;
inline constexpr BoundaryWord kPayloadMask = 0x00ffffffu;
inline constexpr BoundaryWord kChannelMask = 0x0f000000u;
inline constexpr BoundaryWord kRawBoundaryMask = kPayloadMask | kChannelMask;
inline constexpr BoundaryWord kFrameMask = 0xf0000000u;
inline constexpr BoundaryWord kContactFrame = 0xa0000000u;
inline constexpr BoundaryWord kGeneratedFrame = 0xb0000000u;
inline constexpr std::uint32_t kMaximumChannel = 0x0fu;

struct RawBoundaryWord {
  std::uint32_t channel = 0u;
  std::uint32_t payload = 0u;
};

struct ResidentBoundaryCodecState {
  // This key merely makes accidental packet transplantation detectable.  Its
  // resident ownership and ingress provenance must be established elsewhere.
  std::uint32_t resident_key = 0u;
  std::uint32_t epoch = 0u;
};

struct ResidentBoundaryPacket {
  BoundaryWord framed_word = 0u;
  std::uint32_t epoch = 0u;
  std::uint32_t seal = 0u;
};

[[nodiscard]] BCC32_BOUNDARY_WORD_HD constexpr bool raw_boundary_word_valid(
    const RawBoundaryWord& word) {
  return word.channel <= kMaximumChannel &&
         (word.payload & ~kPayloadMask) == 0u;
}

// A public/raw word always has a zero private-frame nibble.  Consequently any
// externally supplied word carrying private frame bits fails closed.
[[nodiscard]] BCC32_BOUNDARY_WORD_HD constexpr bool external_public_word_valid(
    BoundaryWord word) {
  return (word & kFrameMask) == 0u;
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool raw_public_word(
    const RawBoundaryWord& raw, BoundaryWord* public_word) {
  if (public_word == nullptr || !raw_boundary_word_valid(raw)) return false;
  *public_word = (raw.channel << kChannelShift) | raw.payload;
  return true;
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool unpack_external_public_word(
    BoundaryWord public_word, RawBoundaryWord* raw) {
  if (raw == nullptr || !external_public_word_valid(public_word)) return false;
  raw->channel = (public_word & kChannelMask) >> kChannelShift;
  raw->payload = public_word & kPayloadMask;
  return true;
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD constexpr bool resident_codec_state_valid(
    const ResidentBoundaryCodecState& state) {
  return state.resident_key != 0u && state.epoch != 0u;
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline std::uint32_t seal_mix32(
    std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

// This checksum is intentionally non-cryptographic.  It binds the complete
// framed word and epoch to one nonzero resident key so stale or corrupted
// packets fail closed at this codec boundary.
[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline std::uint32_t packet_seal(
    const ResidentBoundaryCodecState& state, BoundaryWord framed_word,
    std::uint32_t epoch) {
  const std::uint32_t keyed_epoch =
      seal_mix32(state.resident_key ^ epoch ^ 0x9e3779b9u);
  const std::uint32_t keyed_word =
      seal_mix32(framed_word ^ (state.resident_key << 13u) ^
                 (state.resident_key >> 19u) ^ 0x85ebca6bu);
  return seal_mix32(keyed_epoch ^ keyed_word ^ 0xc2b2ae35u);
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD constexpr bool private_frame_valid(
    BoundaryWord framed_word) {
  const BoundaryWord frame = framed_word & kFrameMask;
  return frame == kContactFrame || frame == kGeneratedFrame;
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool resident_packet_valid(
    const ResidentBoundaryCodecState* state,
    const ResidentBoundaryPacket* packet) {
  if (state == nullptr || packet == nullptr ||
      !resident_codec_state_valid(*state) || packet->epoch != state->epoch ||
      !private_frame_valid(packet->framed_word))
    return false;
  return packet->seal ==
         packet_seal(*state, packet->framed_word, packet->epoch);
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool resident_make_packet(
    const ResidentBoundaryCodecState* state, BoundaryWord frame,
    const RawBoundaryWord& raw, ResidentBoundaryPacket* packet) {
  if (state == nullptr || packet == nullptr ||
      !resident_codec_state_valid(*state) || !raw_boundary_word_valid(raw) ||
      (frame != kContactFrame && frame != kGeneratedFrame))
    return false;

  BoundaryWord public_word = 0u;
  if (!raw_public_word(raw, &public_word)) return false;
  packet->framed_word = frame | public_word;
  packet->epoch = state->epoch;
  packet->seal = packet_seal(*state, packet->framed_word, packet->epoch);
  return true;
}

// External contact must arrive as a public/raw word.  A caller cannot smuggle
// either private frame kind through this ingress function.
[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool resident_frame_external_contact(
    const ResidentBoundaryCodecState* state, BoundaryWord external_word,
    ResidentBoundaryPacket* packet) {
  RawBoundaryWord raw{};
  if (!unpack_external_public_word(external_word, &raw)) return false;
  return resident_make_packet(state, kContactFrame, raw, packet);
}

[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool resident_frame_generated(
    const ResidentBoundaryCodecState* state, const RawBoundaryWord& generated,
    ResidentBoundaryPacket* packet) {
  return resident_make_packet(state, kGeneratedFrame, generated, packet);
}

// Contact is reversible only back into private raw channel/payload structure;
// this function does not authorize contact matter for public egress.
[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool resident_unpack_contact(
    const ResidentBoundaryCodecState* state,
    const ResidentBoundaryPacket* packet, RawBoundaryWord* raw) {
  if (raw == nullptr || !resident_packet_valid(state, packet) ||
      (packet->framed_word & kFrameMask) != kContactFrame)
    return false;
  raw->channel = (packet->framed_word & kChannelMask) >> kChannelShift;
  raw->payload = packet->framed_word & kPayloadMask;
  return true;
}

// The sole public decoder accepts a currently valid generated frame.  Contact
// packets and corrupted/stale packets cannot cross this egress boundary.
[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool decode_generated_public_word(
    const ResidentBoundaryCodecState* state,
    const ResidentBoundaryPacket* packet, BoundaryWord* public_word) {
  if (public_word == nullptr || !resident_packet_valid(state, packet) ||
      (packet->framed_word & kFrameMask) != kGeneratedFrame)
    return false;
  *public_word = packet->framed_word & kRawBoundaryMask;
  return true;
}

// Source withdrawal advances the nonzero epoch, invalidating every previously
// sealed packet.  Wraparound deliberately skips the reserved invalid epoch 0.
[[nodiscard]] BCC32_BOUNDARY_WORD_HD inline bool resident_withdraw_source(
    ResidentBoundaryCodecState* state) {
  if (state == nullptr || !resident_codec_state_valid(*state)) return false;
  ++state->epoch;
  if (state->epoch == 0u) state->epoch = 1u;
  return true;
}

}  // namespace substrate::bcc32::boundary_word_codec

#undef BCC32_BOUNDARY_WORD_HD
