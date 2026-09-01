#ifndef HARDWARE_NATIVE_DIRECT_ADULT_NODE_PARTICIPATION_TYPES_INL
#define HARDWARE_NATIVE_DIRECT_ADULT_NODE_PARTICIPATION_TYPES_INL
// Mid-include splice for substrate::direct_adult_core.
// Keeps the bounded current participation bank usable by narrow Network seams
// without pulling in the complete Direct Adult state header. Eight slots let
// several unresolved lived claims coexist at one physical boundary node; the
// executor still touches only that node-local bank rather than scanning matter.
inline constexpr std::uint32_t kMaxProvenanceSlotsPerNode = 8u;

struct alignas(16) NodeCausalParticipation {
  std::uint64_t ticket_id;
  std::uint32_t expiry_tick;
  std::uint32_t last_refresh_tick;
  DirectParticipationAuthority authority;
  std::uint32_t commit_generation;
  std::uint32_t authority_incarnation;
  std::uint32_t claim_incarnation;
  std::uint32_t current_drive;
  std::uint32_t reserved0, reserved1, reserved2;
};
static_assert(sizeof(NodeCausalParticipation) == 48 &&
              std::is_standard_layout_v<NodeCausalParticipation> &&
              std::is_trivial_v<NodeCausalParticipation>);
static_assert(std::has_unique_object_representations_v<NodeCausalParticipation>);
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_NODE_PARTICIPATION_TYPES_INL
