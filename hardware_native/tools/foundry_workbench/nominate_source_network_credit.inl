#include "hardware_native/direct_adult_action_binding_api.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"

namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_motor_affect_helpers.cuh"
}

struct NominateCreditCtx {
  const ResidentRecipeCell* rec_src;
  const ResidentRecipeDerivation* der_src;
  ResidentRecipeOccurrence* occ_src;
  const ResidentOccurrenceCoupling* e_src;
  const ResidentRecipeCell* rec_alt;
  const ResidentRecipeDerivation* der_alt;
  ResidentRecipeOccurrence* occ_alt;
  const ResidentOccurrenceCoupling* e_alt;
  substrate::direct_network::ResidentDevelopmentState* development;
  std::uint32_t src_slot, alt_slot;
  std::uint64_t rid_src, rid_alt;
  ResidentRelationalNetworkClosure n_src_e, n_alt_e, n_src_r, n_surf;
  ResidentRelationalNetworkSet nominated, only_src;
  ResidentRecipeOccurrence occ_surf[2];
  ResidentOccurrenceCoupling e_surf[1];
};

static bool nominate_credit_source_alt(NominateCreditCtx* c) {
  ResidentRecruitedNetworkCreditPlan plan{};
  std::uint64_t selected_n = 0, selected_r = 0;
  std::int64_t selected_c = 0;
  if (!recruit_resident_relational_network_set(c->development, c->nominated, 30u) ||
      c->development->recruited_networks.incidence_count != 2u ||
      !plan_resident_recruited_network_credit(c->development, c->rid_src, c->n_src_e.identity, 2000,
                                              &plan) ||
      plan.valid == 0u || !apply_resident_recruited_network_credit(c->development, plan, 31u))
    return false;
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    const auto rid = c->development->recruited_networks.incidences[i].recruitment_identity;
    if (rid == c->rid_src)
      c->src_slot = i;
    if (rid == c->rid_alt)
      c->alt_slot = i;
  }
  if (c->src_slot >= 2u || c->alt_slot >= 2u || c->src_slot == c->alt_slot ||
      c->development->recruited_networks.incidences[c->src_slot].credit_q16 != 2000 ||
      c->development->recruited_networks.incidences[c->alt_slot].credit_q16 != 0 ||
      !select_resident_recruited_network(c->development->recruited_networks, c->nominated,
                                         &selected_n, &selected_r, &selected_c) ||
      selected_r != c->rid_src || selected_n != c->n_src_e.identity || selected_c != 2000)
    return false;
  c->occ_src[1].eligibility_q16 += 7;
  ResidentRelationalNetworkSet refresh{};
  refresh.network_count = 1u;
  refresh.networked_occurrence_count = 2u;
  refresh.coupling_count = 1u;
  if (!bind_resident_relational_network_closure(c->rec_src, c->der_src, c->occ_src, 2u, c->e_src,
                                                1u, &c->n_src_r) ||
      c->n_src_r.identity == c->n_src_e.identity ||
      resident_relational_network_recruitment_identity(c->n_src_r) != c->rid_src)
    return false;
  refresh.networks[0] = c->n_src_r;
  return recruit_resident_relational_network_set(c->development, refresh, 32u) &&
         c->development->recruited_networks.incidences[c->src_slot].credit_q16 == 2000 &&
         c->development->recruited_networks.incidences[c->alt_slot].credit_q16 == 0;
}

static bool nominate_withdraw_and_tie(NominateCreditCtx* c) {
  c->only_src.network_count = 1u;
  c->only_src.networked_occurrence_count = 2u;
  c->only_src.coupling_count = 1u;
  c->only_src.networks[0] = c->n_src_r;
  ResidentRelationalNetworkSet only_alt{};
  only_alt.network_count = 1u;
  only_alt.networked_occurrence_count = 2u;
  only_alt.coupling_count = 1u;
  only_alt.networks[0] = c->n_alt_e;
  std::uint64_t src_sel_n = 0, src_sel_r = 0, alt_sel_n = 1, alt_sel_r = 1;
  std::int64_t src_sel_c = 0, alt_sel_c = 1;
  if (!select_resident_recruited_network(c->development->recruited_networks, c->only_src,
                                         &src_sel_n, &src_sel_r, &src_sel_c) ||
      src_sel_r != c->rid_src || src_sel_n != c->n_src_r.identity || src_sel_c != 2000 ||
      !select_resident_recruited_network(c->development->recruited_networks, only_alt, &alt_sel_n,
                                         &alt_sel_r, &alt_sel_c) ||
      alt_sel_n != 0u || alt_sel_r != 0u || alt_sel_c != 0 ||
      c->development->recruited_networks.incidences[c->src_slot].credit_q16 != 2000 ||
      c->development->recruited_networks.incidences[c->alt_slot].credit_q16 != 0)
    return false;
  ResidentRecruitedNetworkCreditPlan alt_plan{}, break_plan{};
  std::uint64_t tie_n = 1, tie_r = 1, broken_n = 0, broken_r = 0;
  std::int64_t tie_c = 1, broken_c = 0;
  return plan_resident_recruited_network_credit(c->development, c->rid_alt, c->n_alt_e.identity,
                                                2000, &alt_plan) &&
         alt_plan.valid != 0u &&
         apply_resident_recruited_network_credit(c->development, alt_plan, 33u) &&
         c->development->recruited_networks.incidences[c->src_slot].credit_q16 == 2000 &&
         c->development->recruited_networks.incidences[c->alt_slot].credit_q16 == 2000 &&
         select_resident_recruited_network(c->development->recruited_networks, c->nominated, &tie_n,
                                           &tie_r, &tie_c) &&
         tie_n == 0u && tie_r == 0u && tie_c == 0 &&
         plan_resident_recruited_network_credit(c->development, c->rid_src, c->n_src_e.identity, 1,
                                                &break_plan) &&
         break_plan.valid != 0u &&
         apply_resident_recruited_network_credit(c->development, break_plan, 34u) &&
         c->development->recruited_networks.incidences[c->src_slot].credit_q16 == 2001 &&
         c->development->recruited_networks.incidences[c->alt_slot].credit_q16 == 2000 &&
         select_resident_recruited_network(c->development->recruited_networks, c->nominated,
                                           &broken_n, &broken_r, &broken_c) &&
         broken_r == c->rid_src && broken_n == c->n_src_e.identity && broken_c == 2001;
}

static bool nominate_surface_and_hybrid(NominateCreditCtx* c) {
  const std::uint32_t surf0[] = {150u, 160u};
  const std::uint32_t surf1[] = {160u, 170u};
  ResidentRelationalNetworkSet across{};
  std::uint64_t surf_n = 0, surf_r = 0;
  std::int64_t surf_c = 0;
  if (!bind_live(c->rec_src[0], c->der_src[0], surf0, 0x1610, &c->occ_surf[0], 13u, 13 << 16) ||
      !bind_live(c->rec_src[1], c->der_src[1], surf1, 0x1611, &c->occ_surf[1], 13u, 13 << 16))
    return false;
  c->occ_surf[0].eligibility_q16 = 11;
  c->occ_surf[1].eligibility_q16 = 13;
  if (!bind_resident_occurrence_coupling(c->occ_surf[0], c->der_src[0], 1u, c->occ_surf[1],
                                         c->der_src[1], 0u, &c->e_surf[0]) ||
      !bind_resident_relational_network_closure(c->rec_src, c->der_src, c->occ_surf, 2u, c->e_surf,
                                                1u, &c->n_surf) ||
      c->n_surf.identity == 0u || c->n_surf.identity == c->n_src_e.identity ||
      resident_relational_network_recruitment_identity(c->n_surf) != c->rid_src ||
      resident_relational_network_recruitment_identity(c->n_surf) == c->rid_alt ||
      resident_network_boundary_relation_equal(c->n_src_e, c->n_surf))
    return false;
  across.network_count = 2u;
  across.networked_occurrence_count = 4u;
  across.coupling_count = 2u;
  across.networks[0] = c->n_surf;
  across.networks[1] = c->n_alt_e;
  if (!recruit_resident_relational_network_set(c->development, across, 35u) ||
      c->development->recruited_networks.incidence_count != 2u ||
      c->development->recruited_networks.incidences[c->src_slot].credit_q16 != 2001 ||
      c->development->recruited_networks.incidences[c->src_slot].recruitment_identity !=
          c->rid_src ||
      c->development->recruited_networks.incidences[c->alt_slot].credit_q16 != 2000)
    return false;
  std::uint32_t surf_slot = 2u;
  const std::uint64_t rid_surf = resident_relational_network_recruitment_identity(c->n_surf);
  for (std::uint32_t i = 0u; i < 2u; ++i)
    if (c->development->recruited_networks.incidences[i].recruitment_identity == rid_surf)
      surf_slot = i;
  if (surf_slot != c->src_slot || surf_slot == c->alt_slot ||
      c->development->recruited_networks.incidences[surf_slot].credit_q16 != 2001 ||
      !select_resident_recruited_network(c->development->recruited_networks, across, &surf_n,
                                         &surf_r, &surf_c) ||
      surf_r != c->rid_src || surf_n != c->n_surf.identity || surf_c != 2001)
    return false;
  ResidentRecipeCell rec_hy[2] = {c->rec_src[1], c->rec_alt[1]};
  ResidentRecipeDerivation der_hy[2] = {c->der_src[1], c->der_alt[1]};
  der_hy[0].logical_recipe_id = rec_hy[0].logical_recipe_id;
  der_hy[0].revision_identity = rec_hy[0].revision_identity;
  der_hy[1].logical_recipe_id = rec_hy[1].logical_recipe_id;
  der_hy[1].revision_identity = rec_hy[1].revision_identity;
  const std::uint32_t hy0[] = {170u, 180u};
  const std::uint32_t hy1[] = {180u, 190u};
  ResidentRecipeOccurrence occ_hy[2]{};
  ResidentOccurrenceCoupling e_hy[1]{};
  ResidentRelationalNetworkClosure n_hy{};
  ResidentRelationalNetworkSet with_hy{};
  ResidentRelationalNetworkSet only_hy{};
  std::uint64_t hy_mix_n = 0, hy_mix_r = 0, hy_only_n = 1, hy_only_r = 1;
  std::int64_t hy_mix_c = 0, hy_only_c = 1;
  if (!bind_live(rec_hy[0], der_hy[0], hy0, 0x1710, &occ_hy[0], 14u, 14 << 16) ||
      !bind_live(rec_hy[1], der_hy[1], hy1, 0x1711, &occ_hy[1], 14u, 14 << 16))
    return false;
  occ_hy[0].eligibility_q16 = 11;
  occ_hy[1].eligibility_q16 = 13;
  if (!bind_resident_occurrence_coupling(occ_hy[0], der_hy[0], 1u, occ_hy[1], der_hy[1], 0u,
                                         &e_hy[0]) ||
      !bind_resident_relational_network_closure(rec_hy, der_hy, occ_hy, 2u, e_hy, 1u, &n_hy) ||
      n_hy.identity == 0u || resident_relational_network_recruitment_identity(n_hy) == c->rid_src ||
      resident_relational_network_recruitment_identity(n_hy) == c->rid_alt)
    return false;
  with_hy.network_count = 2u;
  with_hy.networked_occurrence_count = 4u;
  with_hy.coupling_count = 2u;
  with_hy.networks[0] = c->n_surf;
  with_hy.networks[1] = n_hy;
  if (!recruit_resident_relational_network_set(c->development, with_hy, 36u) ||
      c->development->recruited_networks.incidence_count != 3u ||
      c->development->recruited_networks.incidences[c->src_slot].credit_q16 != 2001 ||
      c->development->recruited_networks.incidences[c->alt_slot].credit_q16 != 2000)
    return false;
  std::uint32_t hy_slot = 3u;
  const std::uint64_t rid_hy = resident_relational_network_recruitment_identity(n_hy);
  for (std::uint32_t i = 0u; i < 3u; ++i)
    if (c->development->recruited_networks.incidences[i].recruitment_identity == rid_hy)
      hy_slot = i;
  only_hy.network_count = 1u;
  only_hy.networked_occurrence_count = 2u;
  only_hy.coupling_count = 1u;
  only_hy.networks[0] = n_hy;
  return hy_slot < 3u && hy_slot != c->src_slot && hy_slot != c->alt_slot &&
         c->development->recruited_networks.incidences[hy_slot].credit_q16 == 0 &&
         select_resident_recruited_network(c->development->recruited_networks, with_hy, &hy_mix_n,
                                           &hy_mix_r, &hy_mix_c) &&
         hy_mix_n == c->n_surf.identity && hy_mix_r == c->rid_src && hy_mix_c == 2001 &&
         select_resident_recruited_network(c->development->recruited_networks, only_hy, &hy_only_n,
                                           &hy_only_r, &hy_only_c) &&
         hy_only_n == 0u && hy_only_r == 0u && hy_only_c == 0;
}

static bool nominate_express_and_motor(NominateCreditCtx* c) {
  const std::uint32_t ex_vars[] = {150u, 170u};
  const std::uint32_t surf0[] = {150u, 160u};
  const std::uint32_t surf1[] = {160u, 170u};
  ResidentRecipeOccurrence ex_a{}, ex_b{};
  ResidentOccurrenceCoupling e_ex{};
  ResidentRelationalNetworkClosure n_ex{};
  ResidentRecipeCell rec_ex[2] = {c->rec_alt[1], c->rec_alt[1]};
  ResidentRecipeDerivation der_ex[2] = {c->der_alt[1], c->der_alt[1]};
  der_ex[0].logical_recipe_id = rec_ex[0].logical_recipe_id;
  der_ex[0].revision_identity = rec_ex[0].revision_identity;
  der_ex[1].logical_recipe_id = rec_ex[1].logical_recipe_id;
  der_ex[1].revision_identity = rec_ex[1].revision_identity;
  ResidentRecipeOccurrence occ_ex[2]{};
  DirectWhiteboxCondensationV1 w_ex{};
  if (!bind_live(c->rec_alt[1], c->der_alt[1], ex_vars, 0xE250, &ex_a) ||
      !bind_live(c->rec_alt[1], c->der_alt[1], ex_vars, 0xE251, &ex_b) ||
      ex_a.logical_recipe_id != c->rec_alt[1].logical_recipe_id ||
      ex_b.logical_recipe_id != c->rec_alt[1].logical_recipe_id ||
      ex_a.occurrence_identity == ex_b.occurrence_identity ||
      bind_resident_occurrence_coupling(ex_a, c->der_alt[1], 1u, ex_b, c->der_alt[1], 0u, &e_ex))
    return false;
  occ_ex[0] = ex_a;
  occ_ex[1] = ex_b;
  if (bind_resident_relational_network_closure(rec_ex, der_ex, occ_ex, 2u, &e_ex, 0u, &n_ex) ||
      observe_resident_mixed_rank_whitebox(c->rec_src, c->der_src, c->occ_surf, occ_ex, 2u,
                                           c->e_surf, &e_ex, 1u, &w_ex))
    return false;
  ResidentRecipeOccurrence occ_w2[2]{};
  ResidentOccurrenceCoupling e_w2[1]{};
  DirectWhiteboxCondensationV1 w_world{};
  std::uint64_t old_n = 0, old_r = 0;
  std::int64_t old_c = 0;
  if (!bind_live(c->rec_src[0], c->der_src[0], surf0, 0x1620, &occ_w2[0], 16u, 16 << 16) ||
      !bind_live(c->rec_src[1], c->der_src[1], surf1, 0x1621, &occ_w2[1], 16u, 16 << 16) ||
      !bind_resident_occurrence_coupling(occ_w2[0], c->der_src[0], 1u, occ_w2[1], c->der_src[1], 0u,
                                         &e_w2[0]) ||
      !observe_resident_mixed_rank_whitebox(c->rec_src, c->der_src, c->occ_surf, occ_w2, 2u,
                                            c->e_surf, e_w2, 1u, &w_world) ||
      w_world.witness_identity == 0u ||
      !select_resident_recruited_network(c->development->recruited_networks, c->only_src, &old_n,
                                         &old_r, &old_c) ||
      old_r != c->rid_src || old_n != c->n_src_r.identity || old_c != 2001 ||
      c->development->recruited_networks.incidences[c->src_slot].credit_q16 != 2001)
    return false;
  const std::uint32_t ch0[] = {150u, 170u};
  const std::uint32_t ch1[] = {170u, 190u};
  ResidentRecipeOccurrence occ_ch[2]{};
  ResidentOccurrenceCoupling e_ch[1]{};
  ResidentRelationalNetworkClosure n_ch{};
  DirectWhiteboxCondensationV1 w_ch{};
  if (!bind_live(c->rec_alt[1], c->der_alt[1], ch0, 0xE260, &occ_ch[0], 17u, 17 << 16) ||
      !bind_live(c->rec_alt[1], c->der_alt[1], ch1, 0xE261, &occ_ch[1], 17u, 17 << 16) ||
      !bind_resident_occurrence_coupling(occ_ch[0], c->der_alt[1], 1u, occ_ch[1], c->der_alt[1], 0u,
                                         &e_ch[0]) ||
      !bind_resident_relational_network_closure(rec_ex, der_ex, occ_ch, 2u, e_ch, 1u, &n_ch) ||
      n_ch.identity == 0u || n_ch.actual_count != 2u ||
      resident_relational_network_recruitment_identity(n_ch) == c->rid_src ||
      resident_relational_network_recruitment_identity(n_ch) == c->rid_alt ||
      observe_resident_mixed_rank_whitebox(c->rec_src, c->der_src, c->occ_surf, occ_ch, 2u,
                                           c->e_surf, e_ch, 1u, &w_ch))
    return false;
  bool ch_in = false, ch_out = false;
  for (std::uint32_t i = 0u; i < n_ch.boundary_count; ++i) {
    if (n_ch.boundary[i].variable_identity == 150u &&
        n_ch.boundary[i].direction == ResidentRecipePortDirection::input)
      ch_in = true;
    if (n_ch.boundary[i].variable_identity == 190u &&
        n_ch.boundary[i].direction == ResidentRecipePortDirection::output)
      ch_out = true;
  }
  if (!ch_in || !ch_out)
    return false;
  DirectParticipationDescriptor whole[8]{};
  DirectParticipationDescriptor bits[8]{};
  const auto& src_inc = c->development->recruited_networks.incidences[c->src_slot];
  const auto& src_n = c->only_src.networks[0];
  std::uint32_t whole_n = src_n.occurrence_count;
  for (std::uint16_t m = 0u; m < src_n.occurrence_count; ++m) {
    whole[m].ticket_id = src_n.members[m].participation_identity;
    whole[m].target_node = 7u;
    whole[m].authority = src_n.members[m].authority;
    whole[m].expiry_tick = src_inc.last_active_tick;
  }
  bits[0] = whole[0];
  bits[1] = whole[1];
  bits[1].ticket_id ^= 1ull;
  std::uint32_t bits_n = 2u;
  std::int32_t p_whole = -1, p_bits = -1;
  return src_inc.last_active_tick != 0u && src_n.occurrence_count >= 2u &&
         src_n.eligibility_l1_q16 != 0u && src_n.eligibility_signed_q16 > 0 &&
         resident_motor_candidate_network_credit_q16(c->development->recruited_networks,
                                                     c->only_src, whole, &whole_n, 8u, 7u,
                                                     src_inc.last_active_tick, &p_whole) &&
         p_whole == 2001 &&
         resident_motor_candidate_network_credit_q16(c->development->recruited_networks,
                                                     c->only_src, bits, &bits_n, 8u, 7u,
                                                     src_inc.last_active_tick, &p_bits) &&
         p_bits == 0;
}

static bool nominate_source_network_credit(
    const ResidentRecipeCell* rec_src, const ResidentRecipeDerivation* der_src,
    ResidentRecipeOccurrence* occ_src, const ResidentOccurrenceCoupling* e_src,
    const ResidentRecipeCell* rec_alt, const ResidentRecipeDerivation* der_alt,
    ResidentRecipeOccurrence* occ_alt, const ResidentOccurrenceCoupling* e_alt,
    const ResidentRelationalNetworkClosure& n_src, const ResidentRelationalNetworkClosure& n_alt) {
  if (!rec_src || !der_src || !occ_src || !e_src || !rec_alt || !der_alt || !occ_alt || !e_alt ||
      n_src.identity == 0u || n_alt.identity == 0u || n_src.identity == n_alt.identity)
    return false;
  NominateCreditCtx ctx{};
  ctx.rec_src = rec_src;
  ctx.der_src = der_src;
  ctx.occ_src = occ_src;
  ctx.e_src = e_src;
  ctx.rec_alt = rec_alt;
  ctx.der_alt = der_alt;
  ctx.occ_alt = occ_alt;
  ctx.e_alt = e_alt;
  ctx.src_slot = 2u;
  ctx.alt_slot = 2u;
  occ_src[0].eligibility_q16 = 11;
  occ_src[1].eligibility_q16 = 13;
  occ_alt[0].eligibility_q16 = 11;
  occ_alt[1].eligibility_q16 = 13;
  if (!bind_resident_relational_network_closure(rec_src, der_src, occ_src, 2u, e_src, 1u,
                                                &ctx.n_src_e) ||
      !bind_resident_relational_network_closure(rec_alt, der_alt, occ_alt, 2u, e_alt, 1u,
                                                &ctx.n_alt_e) ||
      ctx.n_src_e.identity == 0u || ctx.n_alt_e.identity == 0u ||
      ctx.n_src_e.identity == n_src.identity || ctx.n_alt_e.identity == n_alt.identity ||
      ctx.n_src_e.eligibility_signed_q16 <= 0 || ctx.n_alt_e.eligibility_signed_q16 <= 0)
    return false;
  ctx.rid_src = resident_relational_network_recruitment_identity(ctx.n_src_e);
  ctx.rid_alt = resident_relational_network_recruitment_identity(ctx.n_alt_e);
  if (ctx.rid_src == 0u || ctx.rid_alt == 0u || ctx.rid_src == ctx.rid_alt)
    return false;
  ctx.nominated.network_count = 2u;
  ctx.nominated.networked_occurrence_count = 4u;
  ctx.nominated.coupling_count = 2u;
  ctx.nominated.networks[0] = ctx.n_src_e;
  ctx.nominated.networks[1] = ctx.n_alt_e;
  ctx.development = static_cast<substrate::direct_network::ResidentDevelopmentState*>(
      std::calloc(1, sizeof(substrate::direct_network::ResidentDevelopmentState)));
  if (ctx.development == nullptr)
    return false;
  const bool ok = nominate_credit_source_alt(&ctx) && nominate_withdraw_and_tie(&ctx) &&
                  nominate_surface_and_hybrid(&ctx) && nominate_express_and_motor(&ctx);
  std::free(ctx.development);
  return ok;
}
