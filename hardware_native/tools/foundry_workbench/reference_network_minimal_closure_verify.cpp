#include <cstdio>
#include <cstring>
#include <utility>

#include "hardware_native/direct_adult_core.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

static bool make_member(std::uint32_t index, std::uint32_t input,
                        std::uint32_t output, ResidentRecipeCell* recipe,
                        ResidentRecipeDerivation* derivation,
                        ResidentRecipeOccurrence* occurrence,
                        std::uint64_t episode_offset = 0u) {
  *recipe = {};
  recipe->logical_recipe_id = 100u + index;
  recipe->revision_identity = 200u + index;
  *derivation = {};
  derivation->logical_recipe_id = recipe->logical_recipe_id;
  derivation->revision_identity = recipe->revision_identity;
  derivation->generation = 1u + index;
  derivation->port_count = 2u;
  derivation->ports[0] = {input, ResidentRecipePortDomain::q16_scalar,
                          ResidentRecipePortDirection::input, 1u};
  derivation->ports[1] = {output, ResidentRecipePortDomain::q16_scalar,
                          ResidentRecipePortDirection::output, 1u};
  const std::uint32_t bindings[] = {input, output};
  return bind_resident_recipe_occurrence(
      *recipe, *derivation, bindings, 2u, 300u + index + episode_offset,
      400u + index + episode_offset, 500u + index + episode_offset,
      10u + index + static_cast<std::uint32_t>(episode_offset),
      ResidentOccurrenceLineageKind::actual,
      DirectParticipationAuthority::independent_external, 7u, 1u, 20u, 0,
      occurrence);
}

static bool contains_network(const ResidentRelationalNetworkSet& set,
                             std::uint64_t identity) {
  for (std::uint16_t i = 0u; i < set.network_count; ++i)
    if (set.networks[i].identity == identity) return true;
  return false;
}

int main() {
  ResidentRecipeCell recipes[5]{};
  ResidentRecipeDerivation derivations[5]{};
  ResidentRecipeOccurrence occurrences[5]{};
  const std::uint32_t ports[5][2] = {
      {11u, 77u}, {77u, 22u}, {33u, 88u}, {88u, 44u}, {55u, 66u}};
  for (std::uint32_t i = 0u; i < 5u; ++i)
    if (!make_member(i, ports[i][0], ports[i][1], &recipes[i],
                     &derivations[i], &occurrences[i]))
      return 2;
  const std::int32_t eligibility[] = {10, 20, 5, -2, 99};
  for (std::uint32_t i = 0u; i < 5u; ++i)
    occurrences[i].eligibility_q16 = eligibility[i];

  ResidentOccurrenceCoupling couplings[2]{};
  if (!bind_resident_occurrence_coupling(
          occurrences[0], derivations[0], 1u, occurrences[1], derivations[1],
          0u, &couplings[0]) ||
      !bind_resident_occurrence_coupling(
          occurrences[2], derivations[2], 1u, occurrences[3], derivations[3],
          0u, &couplings[1]))
    return 3;

  ResidentRelationalNetworkClosure expected_a{}, expected_b{};
  if (!bind_resident_relational_network_closure(
          recipes, derivations, occurrences, 2u, couplings, 1u, &expected_a) ||
      !bind_resident_relational_network_closure(
          recipes + 2, derivations + 2, occurrences + 2, 2u, couplings + 1,
          1u, &expected_b))
    return 4;

  ResidentRelationalNetworkSet set{};
  const bool coactive = bind_resident_relational_network_set(
      recipes, derivations, occurrences, 5u, couplings, 2u, &set);
  const bool split = coactive && set.network_count == 2u &&
      set.networked_occurrence_count == 4u &&
      set.isolated_occurrence_count == 1u && set.coupling_count == 2u &&
      contains_network(set, expected_a.identity) &&
      contains_network(set, expected_b.identity) &&
      expected_a.eligibility_signed_q16 == 30 &&
      expected_a.eligibility_l1_q16 == 30u &&
      expected_b.eligibility_signed_q16 == 3 &&
      expected_b.eligibility_l1_q16 == 7u;

  // S2-shaped action topology: two formal child chains become one unfolded
  // computation through a current causal intersection. Motor binding names only
  // the two terminal Occurrences; causal ancestors remain legitimate members of
  // the same Network. A foreign terminal must still refuse.
  ResidentOccurrenceCoupling action_edges[3] = {couplings[0], couplings[1], {}};
  ResidentRelationalNetworkClosure action_closure{};
  const std::uint64_t action_terminals[2] = {
      occurrences[1].occurrence_identity, occurrences[3].occurrence_identity};
  const std::uint64_t foreign_terminals[2] = {
      occurrences[1].occurrence_identity, occurrences[4].occurrence_identity};
  const bool terminal_subset_closure =
      bind_resident_occurrence_causal_intersection_coupling(
          occurrences[1], derivations[1], occurrences[3], derivations[3], 999u,
          &action_edges[2]) &&
      bind_resident_relational_network_closure(
          recipes, derivations, occurrences, 4u, action_edges, 3u,
          &action_closure) &&
      action_closure.occurrence_count == 4u &&
      resident_relational_network_contains_occurrences(
          action_closure, action_terminals, 2u) &&
      !resident_relational_network_contains_occurrences(
          action_closure, foreign_terminals, 2u);

  // Generic reconvergence relation: different ticket-local variables may still
  // form a Network when their exact output boundaries terminate on the same
  // resident node/domain/arity. This must not masquerade as formal-variable
  // identity, and B_N must therefore keep all four local variables exposed.
  ResidentRecipeCell shared_recipes[2]{};
  ResidentRecipeDerivation shared_derivations[2]{};
  ResidentRecipeOccurrence shared_occurrences[2]{};
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    shared_recipes[i].logical_recipe_id = 800u + i;
    shared_recipes[i].revision_identity = 900u + i;
    shared_derivations[i].logical_recipe_id = shared_recipes[i].logical_recipe_id;
    shared_derivations[i].revision_identity = shared_recipes[i].revision_identity;
    shared_derivations[i].generation = 5u + i;
    shared_derivations[i].port_count = 2u;
    shared_derivations[i].ports[0] = {
        40u + i, ResidentRecipePortDomain::q16_scalar,
        ResidentRecipePortDirection::input, 1u};
    shared_derivations[i].ports[1] = {
        99u, ResidentRecipePortDomain::q16_scalar,
        ResidentRecipePortDirection::output, 1u};
    const std::uint32_t vars[2] = {1001u + i, 2001u + i};
    if (!bind_resident_recipe_occurrence(
            shared_recipes[i], shared_derivations[i], vars, 2u, 700u + i,
            1700u + i, 2700u + i, 30u + i,
            ResidentOccurrenceLineageKind::actual,
            DirectParticipationAuthority::independent_external, 17u, 2u, 40u,
            0, &shared_occurrences[i]))
      return 5;
  }
  ResidentOccurrenceCoupling should_not_be_formal{};
  const bool shared_not_formal = !bind_resident_occurrence_coupling(
      shared_occurrences[0], shared_derivations[0], 1u,
      shared_occurrences[1], shared_derivations[1], 0u,
      &should_not_be_formal);
  ResidentOccurrenceCoupling shared_coupling{};
  ResidentRelationalNetworkClosure shared_closure{};
  const bool causal_intersection_network = shared_not_formal &&
      bind_resident_occurrence_causal_intersection_coupling(
          shared_occurrences[0], shared_derivations[0],
          shared_occurrences[1], shared_derivations[1], 99u,
          &shared_coupling) &&
      shared_coupling.kind == ResidentOccurrenceCouplingKind::causal_intersection &&
      shared_coupling.reserved2 == 99u &&
      bind_resident_relational_network_closure(
          shared_recipes, shared_derivations, shared_occurrences, 2u,
          &shared_coupling, 1u, &shared_closure) &&
      shared_closure.boundary_count == 4u &&
      shared_closure.reconvergence_count == 1u;
  const std::uint64_t shared_recruitment =
      resident_relational_network_recruitment_identity(shared_closure);
  ResidentRecipeOccurrence shared_fresh[2]{};
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    const std::uint32_t vars[2] = {3001u + i, 4001u + i};
    if (!bind_resident_recipe_occurrence(
            shared_recipes[i], shared_derivations[i], vars, 2u, 900u + i,
            1900u + i, 2900u + i, 50u + i,
            ResidentOccurrenceLineageKind::actual,
            DirectParticipationAuthority::independent_external, 17u, 3u, 60u,
            0, &shared_fresh[i]))
      return 6;
  }
  ResidentOccurrenceCoupling shared_fresh_coupling{};
  ResidentRelationalNetworkClosure shared_fresh_closure{};
  const bool causal_intersection_recruits_stably =
      causal_intersection_network && shared_recruitment != 0u &&
      bind_resident_occurrence_causal_intersection_coupling(
          shared_fresh[1], shared_derivations[1],
          shared_fresh[0], shared_derivations[0], 99u,
          &shared_fresh_coupling) &&
      bind_resident_relational_network_closure(
          shared_recipes, shared_derivations, shared_fresh, 2u,
          &shared_fresh_coupling, 1u, &shared_fresh_closure) &&
      shared_fresh_closure.identity != shared_closure.identity &&
      resident_relational_network_recruitment_identity(shared_fresh_closure) ==
          shared_recruitment;

  const std::uint64_t recruitment_a =
      resident_relational_network_recruitment_identity(expected_a);
  ResidentRecipeCell recurrent_recipes[2]{};
  ResidentRecipeDerivation recurrent_derivations[2]{};
  ResidentRecipeOccurrence recurrent_occurrences[2]{};
  if (!make_member(0u, 11u, 77u, &recurrent_recipes[0],
                   &recurrent_derivations[0], &recurrent_occurrences[0], 10000u) ||
      !make_member(1u, 77u, 22u, &recurrent_recipes[1],
                   &recurrent_derivations[1], &recurrent_occurrences[1], 10000u))
    return 12;
  recurrent_occurrences[0].eligibility_q16 = eligibility[0];
  recurrent_occurrences[1].eligibility_q16 = eligibility[1];
  ResidentOccurrenceCoupling recurrent_coupling{};
  if (!bind_resident_occurrence_coupling(
          recurrent_occurrences[0], recurrent_derivations[0], 1u,
          recurrent_occurrences[1], recurrent_derivations[1], 0u,
          &recurrent_coupling))
    return 13;
  ResidentRelationalNetworkClosure recurrent_closure{};
  if (!bind_resident_relational_network_closure(
          recurrent_recipes, recurrent_derivations, recurrent_occurrences, 2u,
          &recurrent_coupling, 1u, &recurrent_closure))
    return 14;
  const bool stable_recruitment_identity = recruitment_a != 0u &&
      recurrent_closure.identity != expected_a.identity &&
      resident_relational_network_recruitment_identity(recurrent_closure) ==
          recruitment_a;
  ResidentDevelopmentState development{};
  ResidentRelationalNetworkSet first_recruit{};
  first_recruit.network_count = 1u;
  first_recruit.networked_occurrence_count = 2u;
  first_recruit.coupling_count = 1u;
  first_recruit.networks[0] = expected_a;
  ResidentRelationalNetworkSet second_recruit{};
  second_recruit.network_count = 1u;
  second_recruit.networked_occurrence_count = 2u;
  second_recruit.coupling_count = 1u;
  second_recruit.networks[0] = recurrent_closure;
  const bool recurrent_recruitment = stable_recruitment_identity &&
      recruit_resident_relational_network_set(&development, first_recruit, 11u) &&
      recruit_resident_relational_network_set(&development, second_recruit, 12u) &&
      development.recruited_networks.incidence_count == 1u &&
      development.recruited_networks.incidences[0].recruitment_identity ==
          recruitment_a &&
      development.recruited_networks.incidences[0].last_active_tick == 12u &&
      development.recruited_networks.incidences[0].activation_count == 2u;

  ResidentRecruitedNetworkCreditPlan network_credit{};
  const bool network_credit_lands_on_recruitment = recurrent_recruitment &&
      plan_resident_recruited_network_credit(
          &development, recruitment_a, recurrent_closure.identity, 1234,
          &network_credit) && network_credit.valid != 0u &&
      resident_recruited_network_credit_plan_current(&development, network_credit);
  if (network_credit_lands_on_recruitment)
    commit_resident_recruited_network_credit(&development, network_credit, 13u);
  const bool network_credit_committed = network_credit_lands_on_recruitment &&
      development.recruited_networks.incidences[0].credit_q16 == 1234 &&
      development.recruited_networks.incidences[0].last_credit_tick == 13u;

  // Eligibility belongs to the current active Network geometry. It may change
  // on a later activation without erasing causal credit already earned by the
  // stable recruitment morphology.
  recurrent_occurrences[1].eligibility_q16 += 7;
  ResidentRelationalNetworkClosure reeligible_closure{};
  const bool reeligible_bound = bind_resident_relational_network_closure(
      recurrent_recipes, recurrent_derivations, recurrent_occurrences, 2u,
      &recurrent_coupling, 1u, &reeligible_closure);
  ResidentRelationalNetworkSet reeligible_set{};
  reeligible_set.network_count = 1u;
  reeligible_set.networked_occurrence_count = 2u;
  reeligible_set.coupling_count = 1u;
  reeligible_set.networks[0] = reeligible_closure;
  const bool eligibility_refresh_preserves_credit = reeligible_bound &&
      resident_relational_network_recruitment_identity(reeligible_closure) ==
          recruitment_a && reeligible_closure.identity != recurrent_closure.identity &&
      recruit_resident_relational_network_set(&development, reeligible_set, 14u) &&
      development.recruited_networks.incidences[0].last_active_tick == 14u &&
      development.recruited_networks.incidences[0].credit_q16 == 1234 &&
      development.recruited_networks.incidences[0].last_credit_tick == 13u;

  // Whole-Network selection ranks only already-formed active closures. It must
  // never combine the best member from A with the best member from B.
  ResidentDevelopmentState selection_development{};
  const bool selection_recruited =
      recruit_resident_relational_network_set(&selection_development, set, 20u) &&
      selection_development.recruited_networks.incidence_count == 2u;
  const std::uint64_t recruitment_b =
      resident_relational_network_recruitment_identity(expected_b);
  ResidentRecruitedNetworkCreditPlan selection_credit_a{}, selection_credit_b{};
  bool selection_credit_ready = selection_recruited && recruitment_a != 0u &&
      recruitment_b != 0u && recruitment_a != recruitment_b &&
      plan_resident_recruited_network_credit(
          &selection_development, recruitment_a, expected_a.identity, 1000,
          &selection_credit_a) && selection_credit_a.valid != 0u &&
      plan_resident_recruited_network_credit(
          &selection_development, recruitment_b, expected_b.identity, 2000,
          &selection_credit_b) && selection_credit_b.valid != 0u;
  if (selection_credit_ready) {
    commit_resident_recruited_network_credit(
        &selection_development, selection_credit_a, 21u);
    commit_resident_recruited_network_credit(
        &selection_development, selection_credit_b, 21u);
  }
  std::uint64_t selected_network = 0u, selected_recruitment = 0u;
  std::int64_t selected_credit = 0;
  const bool whole_network_selection = selection_credit_ready &&
      select_resident_recruited_network(
          selection_development.recruited_networks, set, &selected_network,
          &selected_recruitment, &selected_credit) &&
      selected_network == expected_b.identity &&
      selected_recruitment == recruitment_b && selected_credit == 2000;
  ResidentRecruitedNetworkCreditPlan tie_credit{};
  const bool tie_ready = whole_network_selection &&
      plan_resident_recruited_network_credit(
          &selection_development, recruitment_a, expected_a.identity, 1000,
          &tie_credit) && tie_credit.valid != 0u;
  if (tie_ready)
    commit_resident_recruited_network_credit(
        &selection_development, tie_credit, 22u);
  selected_network = selected_recruitment = 0u;
  selected_credit = 0;
  const bool exact_tie_unresolved = tie_ready &&
      select_resident_recruited_network(
          selection_development.recruited_networks, set, &selected_network,
          &selected_recruitment, &selected_credit) &&
      selected_network == 0u && selected_recruitment == 0u && selected_credit == 0;
  ResidentRelationalNetworkSet eligibility_suppressed = set;
  for (std::uint16_t n = 0u; n < eligibility_suppressed.network_count; ++n)
    if (resident_relational_network_recruitment_identity(
            eligibility_suppressed.networks[n]) == recruitment_b) {
      eligibility_suppressed.networks[n].eligibility_signed_q16 = -1;
      eligibility_suppressed.networks[n].eligibility_l1_q16 = 1u;
    }
  selected_network = selected_recruitment = 0u;
  selected_credit = 0;
  const bool eligibility_gates_credit_without_erasing_it = exact_tie_unresolved &&
      select_resident_recruited_network(
          selection_development.recruited_networks, eligibility_suppressed,
          &selected_network, &selected_recruitment, &selected_credit) &&
      selected_network == expected_a.identity &&
      selected_recruitment == recruitment_a && selected_credit == 2000;

  ResidentRecipeOccurrence eligibility_variant_occurrences[5]{};
  std::memcpy(eligibility_variant_occurrences, occurrences,
              sizeof(eligibility_variant_occurrences));
  ++eligibility_variant_occurrences[0].eligibility_q16;
  ResidentRelationalNetworkSet eligibility_variant{};
  const bool eligibility_is_geometry = bind_resident_relational_network_set(
      recipes, derivations, eligibility_variant_occurrences, 5u, couplings, 2u,
      &eligibility_variant) && eligibility_variant.network_count == 2u &&
      eligibility_variant.networked_occurrence_count == 4u &&
      eligibility_variant.identity != set.identity &&
      std::memcmp(&eligibility_variant, &set, sizeof(set)) != 0;

  std::swap(recipes[0], recipes[4]);
  std::swap(derivations[0], derivations[4]);
  std::swap(occurrences[0], occurrences[4]);
  std::swap(recipes[1], recipes[3]);
  std::swap(derivations[1], derivations[3]);
  std::swap(occurrences[1], occurrences[3]);
  ResidentRelationalNetworkSet permuted{};
  const bool permutation = bind_resident_relational_network_set(
      recipes, derivations, occurrences, 5u, couplings, 2u, &permuted) &&
      permuted.identity == set.identity &&
      std::memcmp(&permuted, &set, sizeof(set)) == 0;

  ResidentRelationalNetworkSet sentinel{};
  sentinel.identity = 0xfeedu;
  ResidentRecipeOccurrence isolated_stale_occurrences[5]{};
  std::memcpy(isolated_stale_occurrences, occurrences,
              sizeof(isolated_stale_occurrences));
  isolated_stale_occurrences[0].state = kResidentRecipeOccurrenceSettled;
  const bool atomic_isolated_stale = !bind_resident_relational_network_set(
      recipes, derivations, isolated_stale_occurrences, 5u, couplings, 2u,
      &sentinel) && sentinel.identity == 0xfeedu;
  occurrences[2].state = kResidentRecipeOccurrenceSettled;
  const bool atomic_stale = !bind_resident_relational_network_set(
      recipes, derivations, occurrences, 5u, couplings, 2u, &sentinel) &&
      sentinel.identity == 0xfeedu;

  const bool green = split && terminal_subset_closure && causal_intersection_network &&
      causal_intersection_recruits_stably && stable_recruitment_identity &&
      recurrent_recruitment && network_credit_committed &&
      eligibility_refresh_preserves_credit && whole_network_selection &&
      exact_tie_unresolved && eligibility_gates_credit_without_erasing_it &&
      eligibility_is_geometry && permutation && atomic_isolated_stale && atomic_stale;
  std::printf(
      "FOUNDRY_NETWORK_COACTIVE_CLOSURES %s networks=%u networked=%u "
      "isolated=%u touched_occurrences=5 touched_couplings=2 "
      "whole_frontier_single_network=0 permutation=%u atomic_stale=%u "
      "atomic_isolated_stale=%u "
      "eligibility_a_signed=30 eligibility_b_signed=3 "
      "eligibility_geometry=%u eligibility_changes_active_identity=1 "
      "eligibility_is_credit=0 recruitment_stable=%u recruitment_reuse=%u "
      "network_credit=%u eligibility_refresh_preserves_credit=%u "
      "network_credit_q16=%lld recruitment_activations=%llu identity=%llu "
      "recipe_selector=0 graph_flip=0\n",
      green ? "GREEN" : "RED", set.network_count,
      set.networked_occurrence_count, set.isolated_occurrence_count,
      permutation ? 1u : 0u, atomic_stale ? 1u : 0u,
      atomic_isolated_stale ? 1u : 0u,
      eligibility_is_geometry ? 1u : 0u,
      stable_recruitment_identity ? 1u : 0u,
      recurrent_recruitment ? 1u : 0u,
      network_credit_lands_on_recruitment ? 1u : 0u,
      eligibility_refresh_preserves_credit ? 1u : 0u,
      static_cast<long long>(
          development.recruited_networks.incidences[0].credit_q16),
      static_cast<unsigned long long>(
          development.recruited_networks.incidences[0].activation_count),
      static_cast<unsigned long long>(set.identity));
  return green ? 0 : 1;
}
