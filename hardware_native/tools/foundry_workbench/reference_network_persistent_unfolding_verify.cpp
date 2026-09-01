#include <cstdio>

#include "hardware_native/direct_adult_core.cuh"
namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_resident_relational_network.cuh"
}  // namespace substrate::direct_adult_core

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

static bool make_network(std::uint64_t episode,
                         const std::int32_t eligibility[3],
                         ResidentRelationalNetworkClosure* out) {
  ResidentRecipeCell recipes[3]{};
  ResidentRecipeDerivation derivations[3]{};
  ResidentRecipeOccurrence occurrences[3]{};
  for (std::uint32_t i = 0u; i < 3u; ++i) {
    recipes[i].logical_recipe_id = 100u + i;
    recipes[i].revision_identity = 200u + i;
    derivations[i].logical_recipe_id = recipes[i].logical_recipe_id;
    derivations[i].revision_identity = recipes[i].revision_identity;
    derivations[i].generation = 1u + i;
    derivations[i].port_count = 2u;
    derivations[i].ports[0] = {
        10u + i, ResidentRecipePortDomain::q16_scalar,
        ResidentRecipePortDirection::input, 1u};
    derivations[i].ports[1] = {
        20u + i, ResidentRecipePortDomain::q16_scalar,
        ResidentRecipePortDirection::output, 1u};
    const std::uint32_t variables[2] = {10u + i, 20u + i};
    if (!bind_resident_recipe_occurrence(
            recipes[i], derivations[i], variables, 2u,
            1000u + episode + i, 2000u + episode + i,
            3000u + episode + i, 1u + i,
            ResidentOccurrenceLineageKind::actual,
            DirectParticipationAuthority::independent_external,
            7u, 1u, 40u, eligibility[i], &occurrences[i]))
      return false;
  }
  occurrences[0].bindings[1].variable_identity = 77u;
  occurrences[1].bindings[0].variable_identity = 77u;
  occurrences[1].bindings[1].variable_identity = 88u;
  occurrences[2].bindings[0].variable_identity = 88u;
  ResidentOccurrenceCoupling couplings[2]{};
  if (!bind_resident_occurrence_coupling(
          occurrences[0], derivations[0], 1u,
          occurrences[1], derivations[1], 0u, &couplings[0]) ||
      !bind_resident_occurrence_coupling(
          occurrences[1], derivations[1], 1u,
          occurrences[2], derivations[2], 0u, &couplings[1]))
    return false;
  return bind_resident_relational_network_closure(
      recipes, derivations, occurrences, 3u, couplings, 2u, out);
}

int main() {
  const std::int32_t first_eligibility[3] = {10, 20, 30};
  const std::int32_t fresh_eligibility[3] = {30, 40, 50};
  const std::int32_t suppressed_eligibility[3] = {-100, 20, 30};
  ResidentRelationalNetworkClosure first{}, fresh{}, suppressed{};
  if (!make_network(0u, first_eligibility, &first) ||
      !make_network(100u, fresh_eligibility, &fresh) ||
      !make_network(200u, suppressed_eligibility, &suppressed))
    return 2;
  const std::uint64_t rid =
      resident_relational_network_recruitment_identity(first);
  if (rid == 0u || rid !=
          resident_relational_network_recruitment_identity(fresh) ||
      first.identity == fresh.identity)
    return 3;

  ResidentDevelopmentState development{};
  ResidentRelationalNetworkSet first_set{};
  first_set.network_count = 1u;
  first_set.networked_occurrence_count = 3u;
  first_set.coupling_count = 2u;
  first_set.networks[0] = first;
  if (!recruit_resident_relational_network_set(&development, first_set, 10u))
    return 4;
  const auto activation_count =
      development.recruited_networks.incidences[0].activation_count;
  if (!recruit_resident_relational_network_set(&development, first_set, 10u) ||
      activation_count != 1u ||
      development.recruited_networks.incidences[0].activation_count != 1u)
    return 5;

  ResidentRecruitedNetworkCreditPlan plan{};
  if (!plan_resident_recruited_network_credit(
          &development, rid, first.identity, 1234, &plan) ||
      !apply_resident_recruited_network_credit(&development, plan, 11u))
    return 6;

  ResidentRelationalNetworkSet active{};
  active.network_count = 1u;
  active.networked_occurrence_count = 3u;
  active.coupling_count = 2u;
  active.networks[0] = fresh;
  std::uint64_t selected_network = 0u, selected_recruitment = 0u;
  std::int64_t selected_credit = 0;
  const bool unfolded = select_resident_recruited_network(
      development.recruited_networks, active, &selected_network,
      &selected_recruitment, &selected_credit);

  active.networks[0] = suppressed;
  std::uint64_t vetoed_network = 1u, vetoed_recruitment = 1u;
  std::int64_t vetoed_credit = 1;
  const bool vetoed = select_resident_recruited_network(
      development.recruited_networks, active, &vetoed_network,
      &vetoed_recruitment, &vetoed_credit);

  constexpr std::size_t kLegacyPersistentRowBytes = 392u;
  const std::size_t persistent_bytes = sizeof(ResidentRecruitedNetworkState);
  const std::size_t legacy_bytes =
      kLegacyPersistentRowBytes * kResidentRecruitedNetworkCapacity + 24u;
  const bool green = unfolded && selected_network == fresh.identity &&
      selected_recruitment == rid && selected_credit == 1234 && vetoed &&
      vetoed_network == 0u && vetoed_recruitment == 0u && vetoed_credit == 0 &&
      fresh.eligibility_signed_q16 == 120 &&
      sizeof(ResidentRecruitedNetworkIncidence) == 48u &&
      persistent_bytes == 3096u && persistent_bytes < legacy_bytes;

  std::printf(
      "FOUNDRY_NETWORK_PERSISTENT_UNFOLDING %s recruitment=%llu "
      "first_active=%llu fresh_active=%llu credit=%lld current_eligibility=%lld "
      "transient_veto=%u row_bytes=%zu transient_closure_bytes=%zu "
      "checkpoint_network_bytes=%zu legacy_checkpoint_network_bytes=%zu "
      "checkpoint_bytes_saved=%zu development_state_bytes=%zu runtime_llm=0\n",
      green ? "GREEN" : "RED", static_cast<unsigned long long>(rid),
      static_cast<unsigned long long>(first.identity),
      static_cast<unsigned long long>(fresh.identity),
      static_cast<long long>(selected_credit),
      static_cast<long long>(fresh.eligibility_signed_q16),
      vetoed_network == 0u ? 1u : 0u,
      sizeof(ResidentRecruitedNetworkIncidence),
      sizeof(ResidentRelationalNetworkClosure), persistent_bytes, legacy_bytes,
      legacy_bytes - persistent_bytes, sizeof(ResidentDevelopmentState));
  return green ? 0 : 1;
}
