// This unit owns the public commit and exact raw-consequence return phase for
// one already charged participation component. Charge derivation and rollback
// remain in the parent so the component has one mutable state owner.
template <typename CanonicalCommit>
inline bool commit_consumed_participation_component(
    rewrite::ResidentRewriteState* world,
    ParticipationComponentCharge* charge,
    CanonicalCommit&& canonical_commit,
    guarded::CanonicalPublicActionReceipt* publication) {
  if (world == nullptr || charge == nullptr || publication == nullptr ||
      charge->status != ParticipationComponentStatus::component_consumed ||
      !exact_component_charge(*world, *charge))
    return false;
  guarded::CanonicalTicketedActionView view{};
  view.ticket = charge->ticket;
  view.selected_extent = charge->extent;
  view.purpose = pressure::PaymentPurpose::action_proposal;
  guarded::CanonicalPublicActionReceipt published{};
  if (!canonical_commit(view, &published) ||
      !guarded::is_exact_public_action_receipt(published, view)) {
    rollback_unpublished_participation_component(world, charge);
    return false;
  }
  charge->public_sequence = published.public_sequence;
  charge->status = ParticipationComponentStatus::publicly_committed;
  *publication = published;
  return true;
}
// Exact bytes return only after the matching accepted raw consequence.
BCC32_RESOURCE_ATTENTION_HD inline bool
release_participation_component_after_raw_consequence(
    rewrite::ResidentRewriteState* world,
    ParticipationComponentCharge* charge,
    const guarded::CanonicalPublicActionReceipt& publication,
    const guarded::GuardedActionConsequence& consequence) {
  if (world == nullptr || charge == nullptr ||
      charge->status != ParticipationComponentStatus::publicly_committed ||
      publication.public_sequence != charge->public_sequence ||
      charge->extent < 4u ||
      charge->extent > pressure::kMaximumChargedExtent ||
      charge->binding == 0u || charge->transaction_tag == 0u ||
      charge->public_sequence == 0u)
    return false;
  guarded::CanonicalTicketedActionView view{};
  view.ticket = charge->ticket;
  view.selected_extent = charge->extent;
  view.purpose = pressure::PaymentPurpose::action_proposal;
  if (!guarded::is_exact_public_action_receipt(publication, view) ||
      !guarded::is_accepted_raw_action_return(
          consequence.accepted_return, consequence.accepted_before,
          charge->ticket) ||
      consequence.resident.consequence_id == 0u ||
      consequence.resident.actual_extent != charge->extent ||
      consequence.resident.purpose !=
          pressure::PaymentPurpose::action_proposal ||
      consequence.resident.transaction_tag != charge->transaction_tag ||
      consequence.resident.commit_revision != charge->commit_revision ||
      consequence.resident.resident_tick <= charge->commit_revision ||
      !rewrite::matter_account_is_closed(*world))
    return false;
  for (std::uint32_t index = 0u; index < charge->extent; ++index) {
    const std::uint32_t slot = charge->slot[index];
    if (slot >= rewrite::live_record_capacity(world) ||
        !participation::is_participation(charge->before[index]) ||
        charge->before[index].matter_q8 != rewrite::kRecordMatterQ8 ||
        !pressure::live_owned_form(world->records[slot],
                                   pressure::kFormCommittedMatter,
                                   charge->before[index].lane[4]) ||
        world->records[slot].matter_q8 != rewrite::kRecordMatterQ8 ||
        world->records[slot].lane[3] !=
            static_cast<std::uint32_t>(
                pressure::PaymentPurpose::action_proposal) ||
        world->records[slot].lane[4] != charge->transaction_tag ||
        world->records[slot].lane[5] !=
            static_cast<std::uint32_t>(charge->commit_revision) ||
        world->records[slot].lane[6] !=
            static_cast<std::uint32_t>(charge->commit_revision >> 32u) ||
        world->records[slot].reserved[0] !=
            guarded::ticket_sequence_low(charge->ticket) ||
        world->records[slot].reserved[1] !=
            guarded::ticket_sequence_high(charge->ticket))
      return false;
    for (std::uint32_t prior = 0u; prior < index; ++prior)
      if (charge->slot[prior] == slot) return false;
  }
  for (std::uint32_t index = 0u; index < charge->extent; ++index)
    world->records[charge->slot[index]] = charge->before[index];
  pressure::increment_revision(world);
  if (!rewrite::matter_account_is_closed(*world)) return false;
  charge->status = ParticipationComponentStatus::component_released;
  return true;
}
