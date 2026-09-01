#pragma once

// Bounded resident-contact integrity receipt.  This is not a transport,
// physical-authentication mechanism, language model, or cognition claim.

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_MCR_HD __host__ __device__
#else
#define BCC32_MCR_HD
#endif

namespace substrate::bcc32::multilingual_contact {

constexpr std::uint32_t kPayloadBytes = 32u;
constexpr std::uint32_t kContactCapacity = 16u;
constexpr std::uint32_t kInvalid = 0xffffffffu;

enum class Language : std::uint8_t {
  kChin = 1, kMandarin = 2, kEnglish = 3, kFrench = 4, kSpanish = 5,
  kDenglish = 6
};

enum class Status : std::uint8_t {
  kAccepted, kForgedTicket, kPayloadMismatch, kMetadataMismatch,
  kOutOfOrder, kReplay, kNoReturnCommitment, kSnapshotTamper,
  kCapacity
};

struct Contact {
  std::uint32_t ticket = 0u;
  std::uint32_t revision = 0u;
  std::uint32_t ingress_order = 0u;
  std::uint32_t source_id = 0u;
  std::uint32_t referent_id = 0u;
  std::uint32_t return_commitment = 0u;
  Language language = Language::kEnglish;
  std::uint8_t size = 0u;
  std::uint8_t payload[kPayloadBytes]{};
};

struct Receipt {
  std::uint32_t payload_identity = 0u;
  std::uint32_t referent_identity = 0u;
  std::uint32_t ingress_order = 0u;
  std::uint32_t source_id = 0u;
  std::uint32_t ticket = 0u;
  std::uint32_t revision = 0u;
  Language declared_language = Language::kEnglish;
  bool returned = false;
};

struct ResidentState {
  std::uint32_t next_ingress = 0u;
  std::uint32_t count = 0u;
  std::uint32_t last_ticket = 0u;
  std::uint32_t checkpoint_identity = 0u;
  Receipt receipts[kContactCapacity]{};
};

BCC32_MCR_HD inline std::uint32_t mix(std::uint32_t h, std::uint32_t v) {
  h ^= v + 0x9e3779b9u + (h << 6u) + (h >> 2u);
  return h * 0x85ebca6bu + 0xc2b2ae35u;
}

BCC32_MCR_HD inline std::uint32_t payload_identity(const Contact& c) {
  std::uint32_t h = 0x811c9dc5u;
  h = mix(h, c.size);
  for (std::uint32_t i = 0u; i < c.size && i < kPayloadBytes; ++i)
    h = mix(h, c.payload[i]);
  return h;
}

BCC32_MCR_HD inline std::uint32_t ticket_for(const Contact& c) {
  std::uint32_t h = mix(c.source_id, c.referent_id);
  h = mix(h, c.revision);
  h = mix(h, c.ingress_order);
  h = mix(h, payload_identity(c));
  return mix(h, c.return_commitment);
}

// Choice is deliberately independent of declared language metadata.
BCC32_MCR_HD inline std::uint32_t resident_choice(const Contact& c) {
  return mix(c.referent_id, payload_identity(c));
}

BCC32_MCR_HD inline std::uint32_t state_identity(const ResidentState& s) {
  std::uint32_t h = mix(s.next_ingress, s.count);
  for (std::uint32_t i = 0u; i < s.count && i < kContactCapacity; ++i) {
    h = mix(h, s.receipts[i].payload_identity);
    h = mix(h, s.receipts[i].referent_identity);
    h = mix(h, s.receipts[i].ingress_order);
    h = mix(h, s.receipts[i].ticket ^ s.receipts[i].revision);
  }
  return h;
}

BCC32_MCR_HD inline Status accept(ResidentState* s, const Contact& c,
                                  Receipt* out) {
  if (s == nullptr || out == nullptr || c.size > kPayloadBytes)
    return Status::kPayloadMismatch;
  if (c.ticket != ticket_for(c)) return Status::kForgedTicket;
  if (c.return_commitment == 0u) return Status::kNoReturnCommitment;
  for (std::uint32_t i = 0u; i < s->count; ++i)
    if (s->receipts[i].ticket == c.ticket) return Status::kReplay;
  if (c.ingress_order != s->next_ingress) return Status::kOutOfOrder;
  if (s->count >= kContactCapacity) return Status::kCapacity;
  Receipt r{};
  r.payload_identity = payload_identity(c);
  r.referent_identity = resident_choice(c);
  r.ingress_order = c.ingress_order;
  r.source_id = c.source_id;
  r.ticket = c.ticket;
  r.revision = c.revision;
  r.declared_language = c.language;
  s->receipts[s->count++] = r;
  s->last_ticket = c.ticket;
  ++s->next_ingress;
  s->checkpoint_identity = state_identity(*s);
  *out = r;
  return Status::kAccepted;
}

BCC32_MCR_HD inline bool commit_return(ResidentState* s,
                                       std::uint32_t ticket,
                                       std::uint32_t commitment) {
  if (s == nullptr || commitment == 0u) return false;
  for (std::uint32_t i = 0u; i < s->count; ++i)
    if (s->receipts[i].ticket == ticket) {
      s->receipts[i].returned = true;
      return true;
    }
  return false;
}

BCC32_MCR_HD inline bool checkpoint(const ResidentState& s,
                                    ResidentState* snapshot) {
  if (snapshot == nullptr) return false;
  *snapshot = s;
  snapshot->checkpoint_identity = state_identity(*snapshot);
  return true;
}

BCC32_MCR_HD inline Status restore(const ResidentState& snapshot,
                                   ResidentState* s) {
  if (s == nullptr || snapshot.checkpoint_identity != state_identity(snapshot))
    return Status::kSnapshotTamper;
  *s = snapshot;
  return Status::kAccepted;
}

}  // namespace substrate::bcc32::multilingual_contact

#undef BCC32_MCR_HD
