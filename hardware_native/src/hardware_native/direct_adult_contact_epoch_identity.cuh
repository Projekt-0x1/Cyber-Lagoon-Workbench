#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CONTACT_EPOCH_IDENTITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CONTACT_EPOCH_IDENTITY_CUH

// Included inside substrate::direct_adult_core after direct_adult_core.cuh.
// Stable contact identity mathematics stays available without materializing
// the transient resident-frontier mechanism in every caller.
DIRECT_ADULT_HD inline std::uint64_t resident_contact_payload_identity(
    const ActivityEvent& event) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x636f6e7461637462ull, event.channel);
  identity = exact_history_fold_word(identity, event.word);
  identity = exact_history_fold_word(identity, event.node);
  identity = exact_history_fold_word(identity, event.context);
  identity = exact_history_fold_word(identity, event.timestamp);
  identity = exact_history_fold_word(identity, event.ticket_id);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint64_t resident_contact_source_identity(
    const DirectBrain& brain, const DirectBoundaryPort& port) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = 0x636f6e7461637473ull;
  for (std::uint32_t i = 0u; i < 8u; ++i)
    identity = exact_history_fold_word(identity, brain.body_root.word[i]);
  identity = exact_history_fold_word(identity, port.node);
  identity = exact_history_fold_word(identity, port.channel);
  identity = exact_history_fold_word(identity, port.physical_route);
  identity = exact_history_fold_word(identity, port.parent_route);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint64_t resident_contact_codec_identity(
    const DirectBrain& brain, const DirectBoundaryPort& port) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x636f6e7461637463ull, resident_contact_source_identity(brain, port));
  identity = exact_history_fold_word(identity, port.channel);
  identity = exact_history_fold_word(identity, port.role_mask);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint64_t resident_contact_session_identity(
    const DirectBrain& brain, const DirectBoundaryPort& port) {
  using direct_network::exact_history_fold_word;
  (void)port;
  std::uint64_t identity = 0x636f6e7461637465ull;
  for (std::uint32_t i = 0u; i < 8u; ++i)
    identity = exact_history_fold_word(identity, brain.birth_root.word[i]);
  for (std::uint32_t i = 0u; i < 8u; ++i)
    identity = exact_history_fold_word(identity, brain.body_root.word[i]);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint32_t resident_physical_binding_identity(
    const DirectBrain& brain, std::uint32_t node,
    std::uint32_t context_signature, std::uint64_t participation_identity,
    std::uint32_t authority_incarnation) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = 0x7068797362696e64ull;
  for (std::uint32_t i = 0u; i < 8u; ++i)
    identity = exact_history_fold_word(identity, brain.birth_root.word[i]);
  identity = exact_history_fold_word(identity, node);
  identity = exact_history_fold_word(identity, context_signature);
  identity = exact_history_fold_word(identity, participation_identity);
  identity = exact_history_fold_word(identity, authority_incarnation);
  const std::uint32_t folded = static_cast<std::uint32_t>(identity) ^
      static_cast<std::uint32_t>(identity >> 32);
  return folded == 0u ? 1u : folded;
}

DIRECT_ADULT_HD inline std::uint32_t resident_contact_authority_incarnation(
    const ResidentContactEpochReceipt& receipt, const ActivityEvent& event) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x636f6e7461637461ull, receipt.boundary_session_epoch);
  identity = exact_history_fold_word(identity, event.ticket_id);
  identity = exact_history_fold_word(identity, event.context);
  const std::uint32_t folded = static_cast<std::uint32_t>(identity) ^
      static_cast<std::uint32_t>(identity >> 32);
  return folded == 0u ? 1u : folded;
}

DIRECT_ADULT_HD inline std::uint64_t resident_contact_receipt_identity(
    const ResidentContactEpochReceipt& receipt) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x636f6e7461637472ull, receipt.source_identity);
  identity = exact_history_fold_word(identity, receipt.codec_identity);
  identity = exact_history_fold_word(identity, receipt.payload_identity);
  identity = exact_history_fold_word(identity, receipt.boundary_session_epoch);
  identity = exact_history_fold_word(identity, receipt.ingress_sequence);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(receipt.selection));
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(receipt.integration));
  identity = exact_history_fold_word(identity, receipt.source_available);
  identity = exact_history_fold_word(identity, receipt.port_index);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline ResidentContactEpochReceipt
make_resident_contact_epoch_receipt(
    const DirectBrain& brain, const DirectBoundaryPort& port,
    std::uint32_t port_index, const ActivityEvent& event,
    std::uint64_t ingress_sequence) {
  ResidentContactEpochReceipt receipt{};
  receipt.source_identity = resident_contact_source_identity(brain, port);
  receipt.codec_identity = resident_contact_codec_identity(brain, port);
  receipt.payload_identity = resident_contact_payload_identity(event);
  receipt.boundary_session_epoch = resident_contact_session_identity(brain, port);
  receipt.ingress_sequence = ingress_sequence;
  receipt.selection = ResidentContactSelection::resident_owned;
  receipt.integration = ResidentContactIntegration::canonical;
  receipt.source_available = 1u;
  receipt.port_index = port_index;
  receipt.identity = resident_contact_receipt_identity(receipt);
  return receipt;
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_CONTACT_EPOCH_IDENTITY_CUH
