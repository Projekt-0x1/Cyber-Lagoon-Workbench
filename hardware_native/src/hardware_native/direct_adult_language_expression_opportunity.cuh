#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_EXPRESSION_OPPORTUNITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_EXPRESSION_OPPORTUNITY_CUH

#include <cstdint>

struct CUstream_st;

namespace substrate::direct_network {
struct DirectResidentLanguageRuntimeBlock;
struct DirectExactHistoryRecord;
struct DirectEfferenceCopy;
struct DirectBrain;
struct ResidentDevelopmentState;
}  // namespace substrate::direct_network

namespace substrate::direct_adult_core {
struct MotorEvent;
struct AsynchronousTicket;
struct ResidentActualFrontier;
struct DirectActionOccurrence;
struct DirectActionParticipationLink;
struct ResidentPublicMotorTrajectory;

__device__ bool resident_language_bind_emitted_step(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    MotorEvent* event, AsynchronousTicket* ticket,
    direct_network::DirectExactHistoryRecord* history_record,
    direct_network::DirectEfferenceCopy* efference);

__device__ bool resident_language_commit_expression_opportunity(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const direct_network::DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links);

__device__ bool resident_language_record_network_span_candidate(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const DirectActionOccurrence& action,
    const ResidentPublicMotorTrajectory& trajectory);

__device__ bool resident_language_current_recipe_nomination(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const direct_network::DirectBrain& brain, std::uint32_t current_tick,
    std::uint64_t* logical_recipe_id, std::uint64_t* revision_identity);

__device__ std::uint32_t resident_language_current_recipe_nominations(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const direct_network::DirectBrain& brain, std::uint32_t current_tick,
    std::uint64_t* logical_recipe_ids, std::uint64_t* revision_identities,
    std::uint32_t capacity, std::uint64_t* recruitment_identity);

__device__ std::uint64_t resident_language_current_discourse_recruitment(
    const direct_network::DirectResidentLanguageRuntimeBlock* language);

__device__ void finalize_resident_language_motor_owned(
    direct_network::DirectResidentLanguageRuntimeBlock* resident_language,
    direct_network::DirectBrain brain,
    const ResidentActualFrontier* actual_frontier, MotorEvent* egress_queue,
    const std::uint32_t* egress_head, const std::uint32_t* egress_tail,
    AsynchronousTicket* ticket_table, DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    direct_network::ResidentDevelopmentState* development,
    direct_network::DirectEfferenceCopy* efference_ring,
    const std::uint32_t* efference_head, const std::uint32_t* efference_tail,
    std::uint32_t route_efference_copies, std::uint32_t current_tick);

void launch_finalize_resident_language_motor(
    direct_network::DirectResidentLanguageRuntimeBlock* resident_language,
    direct_network::DirectBrain brain,
    const ResidentActualFrontier* actual_frontier, MotorEvent* egress_queue,
    const std::uint32_t* egress_head, const std::uint32_t* egress_tail,
    AsynchronousTicket* ticket_table, DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    direct_network::ResidentDevelopmentState* development,
    direct_network::DirectEfferenceCopy* efference_ring,
    const std::uint32_t* efference_head, const std::uint32_t* efference_tail,
    std::uint32_t route_efference_copies, std::uint32_t current_tick,
    CUstream_st* stream);

// Project one consequence-earned resident relation into the existing contextual
// RecipeRevision nomination lane. `relation_blocks_fallback` is set whenever
// relation matter exists, including ambiguous/source-incompatible matter, so a
// failed reference cannot silently fall back to an older recruitment focus.
__device__ std::uint32_t resident_language_current_discourse_relation_nominations(
    const direct_network::DirectResidentLanguageRuntimeBlock* language,
    std::uint64_t* logical_recipe_ids, std::uint64_t* revision_identities,
    std::uint32_t capacity, bool* relation_blocks_fallback);

}  // namespace substrate::direct_adult_core

#endif
