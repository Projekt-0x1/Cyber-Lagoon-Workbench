struct LanguageAfterPn2 {
  ResidentRecipeCell* cells;
  std::uint32_t cell_count;
  ResidentRecipeDerivation* ders;
  std::uint32_t der_count;
  std::uint32_t i_anc, i_src, i_pn1, i_pn2;
  const ResidentRecipeCell* rec_r;
  const ResidentRecipeDerivation* der_r;
  ResidentRecipeOccurrence* occ_r1;
  ResidentRecipeOccurrence* occ_r2;
  ResidentOccurrenceCoupling* e_r1;
  ResidentOccurrenceCoupling* e_r2;
  ResidentRelationalNetworkClosure n_r;
  ResidentRelationalNetworkClosure n_next;
  const DirectWhiteboxCondensationV1* again;
  const ResidentNetworkCondensationEvidence* ev2;
  std::uint64_t again_logical;
  FoundryCondensationWorkspace* workspace;
};

static bool bind_endogenous(const ResidentRecipeCell& cell, const ResidentRecipeDerivation& der,
                            const std::uint32_t* vars, std::uint64_t oid,
                            ResidentRecipeOccurrence* out, std::uint32_t incarnation = 1u,
                            std::int32_t activation_q16 = 0) {
  if (!bind_resident_recipe_occurrence(cell, der, vars, 2u, oid, oid + 1000u, 77u, incarnation,
                                       ResidentOccurrenceLineageKind::endogenous,
                                       DirectParticipationAuthority::none, 9u, 1u, 100u, 0, out))
    return false;
  out->activation_q16 = activation_q16;
  return true;
}

static bool language_boundary_has(const ResidentRelationalNetworkClosure& closure,
                                  std::uint32_t var, ResidentRecipePortDirection direction);

static bool language_recipe_opportunity(const LanguageAfterPn2& a,
                                        const ResidentRecipeOccurrence& expressed) {
  const std::uint32_t learned_surface[] = {0x6c616e67u, 0x72656369u, 0x70656f70u};
  const std::uint32_t wrong_surface[] = {0x6c616e67u, 0x72656369u, 0x70656f71u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_content_identity(3u, learned_surface);
  const std::uint64_t wrong =
      substrate::direct_network::surface_ecology_content_identity(3u, wrong_surface);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence candidate{};
  if (surface == 0u || wrong == 0u || surface == wrong ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(&bank, surface,
                                                                                 expressed) ||
      bank.count != 1u || bank.rows[0].logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id ||
      bank.rows[0].revision_identity != a.cells[a.i_pn2].revision_identity)
    return false;
  if (!substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, a.cell_count, a.ders, a.der_count, 0xE160, &candidate) ||
      candidate.logical_recipe_id != bank.rows[0].logical_recipe_id ||
      candidate.revision_identity != bank.rows[0].revision_identity ||
      candidate.lineage_kind != ResidentOccurrenceLineageKind::endogenous ||
      candidate.authority != DirectParticipationAuthority::none || candidate.eligibility_q16 != 0)
    return false;
  if (substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, wrong, a.cells, a.cell_count, a.ders, a.der_count, 0xE161, &candidate))
    return false;
  ResidentRecipeCell stale_cells[8]{};
  const std::uint32_t copy_count = a.cell_count < 8u ? a.cell_count : 8u;
  for (std::uint32_t i = 0; i < copy_count; ++i)
    stale_cells[i] = a.cells[i];
  if (a.i_pn2 >= copy_count)
    return false;
  stale_cells[a.i_pn2].revision_identity ^= 0x1ull;
  return !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
      bank, surface, stale_cells, copy_count, a.ders, a.der_count, 0xE162, &candidate);
}

static bool language_outer_imagine(
    const LanguageAfterPn2& a, const ResidentRelationalNetworkClosure& n_s,
    ResidentRecipeOccurrence* occ_s1, ResidentRecipeOccurrence* occ_s2,
    ResidentOccurrenceCoupling* e_s1, ResidentOccurrenceCoupling* e_s2,
    const ResidentNetworkCondensationEvidence* ev_s, const DirectWhiteboxCondensationV1* w_s) {
  const std::uint32_t s0[] = {150u, 160u};
  const std::uint32_t s1[] = {160u, 170u};
  ResidentRecipeOccurrence fake{};
  if (bind_resident_recipe_occurrence(a.cells[a.i_src], a.der_r[0], s0, 2u, 0x1930, 0x1930 + 1000u,
                                      77u, 1u, ResidentOccurrenceLineageKind::endogenous,
                                      DirectParticipationAuthority::independent_external, 9u, 1u,
                                      100u, 0, &fake))
    return false;
  ResidentRecipeOccurrence imag_a[2]{}, imag_b[2]{};
  ResidentOccurrenceCoupling e_ia[1]{}, e_ib[1]{};
  ResidentRelationalNetworkClosure n_imag{};
  DirectWhiteboxCondensationV1 w_imag{};
  ResidentRelationalNetworkSet heard_and_imag{};
  ResidentRelationalNetworkClosure whole{};
  if (!bind_endogenous(a.cells[a.i_src], a.der_r[0], s0, 0x1910, &imag_a[0], 21u, 21 << 16) ||
      !bind_endogenous(a.cells[a.i_pn1], a.der_r[1], s1, 0x1911, &imag_a[1], 21u, 21 << 16) ||
      !bind_endogenous(a.cells[a.i_src], a.der_r[0], s0, 0x1920, &imag_b[0], 22u, 22 << 16) ||
      !bind_endogenous(a.cells[a.i_pn1], a.der_r[1], s1, 0x1921, &imag_b[1], 22u, 22 << 16) ||
      !bind_resident_occurrence_coupling(imag_a[0], a.der_r[0], 1u, imag_a[1], a.der_r[1], 0u,
                                         &e_ia[0]) ||
      !bind_resident_occurrence_coupling(imag_b[0], a.der_r[0], 1u, imag_b[1], a.der_r[1], 0u,
                                         &e_ib[0]) ||
      !bind_resident_relational_network_closure(a.rec_r, a.der_r, imag_a, 2u, e_ia, 1u, &n_imag) ||
      n_imag.identity == 0u || n_imag.actual_count != 0u || n_imag.identity == n_s.identity ||
      n_imag.identity == a.n_r.identity ||
      resident_relational_network_recruitment_identity(n_imag) == 0u ||
      resident_relational_network_recruitment_identity(n_imag) !=
          resident_relational_network_recruitment_identity(n_s) ||
      !language_boundary_has(n_imag, 150u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n_imag, 170u, ResidentRecipePortDirection::output) ||
      observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, occ_s1, imag_a, 2u, e_s1, e_ia, 1u,
                                           &w_imag) ||
      observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, imag_a, imag_b, 2u, e_ia, e_ib, 1u,
                                           &w_imag))
    return false;
  ResidentRecipeCell rec_mix[4] = {a.rec_r[0], a.rec_r[1], a.rec_r[0], a.rec_r[1]};
  ResidentRecipeDerivation der_mix[4] = {a.der_r[0], a.der_r[1], a.der_r[0], a.der_r[1]};
  ResidentRecipeOccurrence occ_mix[4] = {occ_s1[0], occ_s1[1], imag_a[0], imag_a[1]};
  ResidentOccurrenceCoupling edges_mix[2] = {e_s1[0], e_ia[0]};
  if (!bind_resident_relational_network_set(rec_mix, der_mix, occ_mix, 4u, edges_mix, 2u,
                                            &heard_and_imag) ||
      heard_and_imag.network_count != 2u || heard_and_imag.networked_occurrence_count != 4u ||
      !((heard_and_imag.networks[0].identity == n_s.identity &&
         heard_and_imag.networks[1].identity == n_imag.identity) ||
        (heard_and_imag.networks[0].identity == n_imag.identity &&
         heard_and_imag.networks[1].identity == n_s.identity)) ||
      bind_resident_relational_network_closure(rec_mix, der_mix, occ_mix, 4u, edges_mix, 2u,
                                               &whole))
    return false;
  const std::uint64_t rid_s = resident_relational_network_recruitment_identity(n_s);
  const std::uint64_t src_logical = a.cells[a.i_src].logical_recipe_id;
  const std::uint64_t pn1_logical = a.cells[a.i_pn1].logical_recipe_id;
  ResidentRecipeOccurrence heard_dead[2] = {occ_s1[0], occ_s1[1]};
  heard_dead[0].state = kResidentRecipeOccurrenceSettled;
  heard_dead[1].state = kResidentRecipeOccurrenceSettled;
  ResidentRelationalNetworkClosure n_dead{};
  ResidentRecipeOccurrence imag_after[2]{}, heard_after[2]{};
  ResidentOccurrenceCoupling e_after[1]{}, e_heard[1]{};
  ResidentRelationalNetworkClosure n_imag_after{}, n_heard_after{};
  DirectWhiteboxCondensationV1 w_dead{};
  std::uint64_t ev_logical = 0, ev_rev = 0;
  return rid_s != 0u && src_logical != 0u && pn1_logical != 0u &&
         !bind_resident_relational_network_closure(a.rec_r, a.der_r, heard_dead, 2u, e_s1, 1u,
                                                   &n_dead) &&
         bind_endogenous(a.cells[a.i_src], a.der_r[0], s0, 0x1940, &imag_after[0], 23u, 23 << 16) &&
         bind_endogenous(a.cells[a.i_pn1], a.der_r[1], s1, 0x1941, &imag_after[1], 23u, 23 << 16) &&
         bind_resident_occurrence_coupling(imag_after[0], a.der_r[0], 1u, imag_after[1], a.der_r[1],
                                           0u, &e_after[0]) &&
         bind_resident_relational_network_closure(a.rec_r, a.der_r, imag_after, 2u, e_after, 1u,
                                                  &n_imag_after) &&
         n_imag_after.actual_count == 0u &&
         resident_relational_network_recruitment_identity(n_imag_after) == rid_s &&
         bind_live(a.cells[a.i_src], a.der_r[0], s0, 0x1950, &heard_after[0], 24u, 24 << 16) &&
         bind_live(a.cells[a.i_pn1], a.der_r[1], s1, 0x1951, &heard_after[1], 24u, 24 << 16) &&
         bind_resident_occurrence_coupling(heard_after[0], a.der_r[0], 1u, heard_after[1],
                                           a.der_r[1], 0u, &e_heard[0]) &&
         bind_resident_relational_network_closure(a.rec_r, a.der_r, heard_after, 2u, e_heard, 1u,
                                                  &n_heard_after) &&
         n_heard_after.actual_count == 2u &&
         resident_relational_network_recruitment_identity(n_heard_after) == rid_s &&
         a.cells[a.i_src].logical_recipe_id == src_logical &&
         a.cells[a.i_pn1].logical_recipe_id == pn1_logical &&
         !admit_resident_network_parent(a.rec_r, a.der_r, imag_after, 2u, e_after, 1u,
                                        n_imag_after.identity, pn1_logical) &&
         admit_resident_network_parent(a.rec_r, a.der_r, heard_after, 2u, e_heard, 1u,
                                       n_heard_after.identity, pn1_logical) &&
         !admit_resident_network_parent(a.rec_r, a.der_r, imag_after, 2u, e_after, 1u,
                                        n_imag_after.identity,
                                        a.cells[a.i_pn2].logical_recipe_id) &&
         ev_s != nullptr && w_s != nullptr && occ_s2 != nullptr && e_s2 != nullptr &&
         !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, heard_dead, occ_s2, 2u, e_s1, e_s2,
                                               1u, &w_dead) &&
         replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count, *ev_s, w_s,
                                           &ev_logical, &ev_rev) &&
         ev_logical == a.again_logical && ev_logical == a.cells[a.i_pn2].logical_recipe_id;
}

static bool language_boundary_has(const ResidentRelationalNetworkClosure& closure,
                                  std::uint32_t var, ResidentRecipePortDirection direction) {
  for (std::uint32_t i = 0; i < closure.boundary_count; ++i)
    if (closure.boundary[i].variable_identity == var && closure.boundary[i].direction == direction)
      return true;
  return false;
}

static bool language_triple_linear(const LanguageAfterPn2& a) {
  ResidentRecipeCell rec[3] = {a.cells[a.i_src], a.cells[a.i_pn1], a.cells[a.i_src]};
  ResidentRecipeDerivation der[3] = {a.ders[a.i_src], a.ders[a.i_pn1], a.ders[a.i_src]};
  for (std::uint32_t i = 0u; i < 3u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
    if (der[i].parameter_count != 2u || der[i].relation_count != 1u)
      return false;
  }
  const std::uint32_t t0[] = {210u, 220u};
  const std::uint32_t t1[] = {220u, 230u};
  const std::uint32_t t2[] = {230u, 240u};
  ResidentRecipeOccurrence occ_a[3]{}, occ_b[3]{};
  ResidentOccurrenceCoupling e_a[2]{}, e_b[2]{};
  ResidentRelationalNetworkClosure n3{};
  DirectWhiteboxCondensationV1 w3{};
  ResidentNetworkCondensationEvidence ev3{};
  std::uint64_t t_logical = 0, t_rev = 0;
  if (!bind_live(rec[0], der[0], t0, 0x1810, &occ_a[0], 18u, 18 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1811, &occ_a[1], 18u, 18 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1812, &occ_a[2], 18u, 18 << 16) ||
      !bind_live(rec[0], der[0], t0, 0x1820, &occ_b[0], 19u, 19 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1821, &occ_b[1], 19u, 19 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1822, &occ_b[2], 19u, 19 << 16) ||
      !bind_resident_occurrence_coupling(occ_a[0], der[0], 1u, occ_a[1], der[1], 0u, &e_a[0]) ||
      !bind_resident_occurrence_coupling(occ_a[1], der[1], 1u, occ_a[2], der[2], 0u, &e_a[1]) ||
      !bind_resident_occurrence_coupling(occ_b[0], der[0], 1u, occ_b[1], der[1], 0u, &e_b[0]) ||
      !bind_resident_occurrence_coupling(occ_b[1], der[1], 1u, occ_b[2], der[2], 0u, &e_b[1]) ||
      !bind_resident_relational_network_closure(rec, der, occ_a, 3u, e_a, 2u, &n3) ||
      n3.identity == 0u || n3.actual_count != 3u ||
      !language_boundary_has(n3, 210u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n3, 240u, ResidentRecipePortDirection::output) ||
      !observe_resident_mixed_rank_whitebox(rec, der, occ_a, occ_b, 3u, e_a, e_b, 2u, &w3) ||
      w3.witness_identity == 0u || !a.again || w3.witness_identity == a.again->witness_identity ||
      resident_whitebox_recipe_logical_identity(w3) == a.again_logical ||
      replay_resident_whitebox_recipe(w3, a.cells[a.i_pn2], a.ders[a.i_pn2]) ||
      replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count, *a.ev2, &w3,
                                        &t_logical, &t_rev))
    return false;
  return !observe_resident_mixed_rank_evidence(rec, der, occ_a, occ_b, 3u, e_a, e_b, 2u, &w3, &ev3);
}

static bool language_c4_refuse(const LanguageAfterPn2& a) {
  ResidentRecipeCell rec[4] = {a.cells[a.i_src], a.cells[a.i_pn1], a.cells[a.i_src],
                               a.cells[a.i_pn1]};
  ResidentRecipeDerivation der[4] = {a.ders[a.i_src], a.ders[a.i_pn1], a.ders[a.i_src],
                                     a.ders[a.i_pn1]};
  for (std::uint32_t i = 0u; i < 4u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
    if (der[i].parameter_count != 2u || der[i].relation_count != 1u)
      return false;
  }
  const std::uint32_t t0[] = {210u, 220u};
  const std::uint32_t t1[] = {220u, 230u};
  const std::uint32_t t2[] = {230u, 240u};
  const std::uint32_t t3[] = {240u, 250u};
  ResidentRecipeOccurrence occ_a[4]{}, occ_b[4]{};
  ResidentOccurrenceCoupling e_a[3]{}, e_b[3]{};
  ResidentRelationalNetworkClosure n4{};
  DirectWhiteboxCondensationV1 w4{};
  if (!bind_live(rec[0], der[0], t0, 0x1B10, &occ_a[0], 28u, 28 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1B11, &occ_a[1], 28u, 28 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1B12, &occ_a[2], 28u, 28 << 16) ||
      !bind_live(rec[3], der[3], t3, 0x1B13, &occ_a[3], 28u, 28 << 16) ||
      !bind_live(rec[0], der[0], t0, 0x1B20, &occ_b[0], 29u, 29 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1B21, &occ_b[1], 29u, 29 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1B22, &occ_b[2], 29u, 29 << 16) ||
      !bind_live(rec[3], der[3], t3, 0x1B23, &occ_b[3], 29u, 29 << 16) ||
      !bind_resident_occurrence_coupling(occ_a[0], der[0], 1u, occ_a[1], der[1], 0u, &e_a[0]) ||
      !bind_resident_occurrence_coupling(occ_a[1], der[1], 1u, occ_a[2], der[2], 0u, &e_a[1]) ||
      !bind_resident_occurrence_coupling(occ_a[2], der[2], 1u, occ_a[3], der[3], 0u, &e_a[2]) ||
      !bind_resident_occurrence_coupling(occ_b[0], der[0], 1u, occ_b[1], der[1], 0u, &e_b[0]) ||
      !bind_resident_occurrence_coupling(occ_b[1], der[1], 1u, occ_b[2], der[2], 0u, &e_b[1]) ||
      !bind_resident_occurrence_coupling(occ_b[2], der[2], 1u, occ_b[3], der[3], 0u, &e_b[2]) ||
      !bind_resident_relational_network_closure(rec, der, occ_a, 4u, e_a, 3u, &n4) ||
      n4.identity == 0u || n4.actual_count != 4u ||
      !language_boundary_has(n4, 210u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n4, 250u, ResidentRecipePortDirection::output))
    return false;
  return !fold_resident_mixed_rank_affine(der, occ_a, e_a, 4u, 3u, n4, &w4) &&
         !observe_resident_mixed_rank_whitebox(rec, der, occ_a, occ_b, 4u, e_a, e_b, 3u, &w4);
}

static bool language_w5_c2(const LanguageAfterPn2& a) {
  ResidentRecipeCell rec[5] = {a.cells[a.i_anc], a.cells[a.i_src], a.cells[a.i_anc],
                               a.cells[a.i_pn1], a.cells[a.i_anc]};
  ResidentRecipeDerivation der[5] = {a.ders[a.i_anc], a.ders[a.i_src], a.ders[a.i_anc],
                                     a.ders[a.i_pn1], a.ders[a.i_anc]};
  for (std::uint32_t i = 0u; i < 5u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
  }
  if (der[0].parameter_count != 0u || der[0].relation_count != 0u || der[1].parameter_count != 2u ||
      der[1].relation_count != 1u || der[3].parameter_count != 2u || der[3].relation_count != 1u)
    return false;
  const std::uint32_t t0[] = {410u, 420u};
  const std::uint32_t t1[] = {420u, 430u};
  const std::uint32_t t2[] = {430u, 440u};
  const std::uint32_t t3[] = {440u, 450u};
  const std::uint32_t t4[] = {450u, 460u};
  ResidentRecipeOccurrence occ_a[5]{}, occ_b[5]{};
  ResidentOccurrenceCoupling e_a[4]{}, e_b[4]{};
  ResidentRelationalNetworkClosure n5{};
  DirectWhiteboxCondensationV1 w5{};
  ResidentNetworkCondensationEvidence ev5{};
  std::uint64_t w5_logical = 0, w5_rev = 0;
  if (!bind_live(rec[0], der[0], t0, 0x1C10, &occ_a[0], 30u, 30 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1C11, &occ_a[1], 30u, 30 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1C12, &occ_a[2], 30u, 30 << 16) ||
      !bind_live(rec[3], der[3], t3, 0x1C13, &occ_a[3], 30u, 30 << 16) ||
      !bind_live(rec[4], der[4], t4, 0x1C14, &occ_a[4], 30u, 30 << 16) ||
      !bind_live(rec[0], der[0], t0, 0x1C20, &occ_b[0], 31u, 31 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1C21, &occ_b[1], 31u, 31 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1C22, &occ_b[2], 31u, 31 << 16) ||
      !bind_live(rec[3], der[3], t3, 0x1C23, &occ_b[3], 31u, 31 << 16) ||
      !bind_live(rec[4], der[4], t4, 0x1C24, &occ_b[4], 31u, 31 << 16) ||
      !bind_resident_occurrence_coupling(occ_a[0], der[0], 1u, occ_a[1], der[1], 0u, &e_a[0]) ||
      !bind_resident_occurrence_coupling(occ_a[1], der[1], 1u, occ_a[2], der[2], 0u, &e_a[1]) ||
      !bind_resident_occurrence_coupling(occ_a[2], der[2], 1u, occ_a[3], der[3], 0u, &e_a[2]) ||
      !bind_resident_occurrence_coupling(occ_a[3], der[3], 1u, occ_a[4], der[4], 0u, &e_a[3]) ||
      !bind_resident_occurrence_coupling(occ_b[0], der[0], 1u, occ_b[1], der[1], 0u, &e_b[0]) ||
      !bind_resident_occurrence_coupling(occ_b[1], der[1], 1u, occ_b[2], der[2], 0u, &e_b[1]) ||
      !bind_resident_occurrence_coupling(occ_b[2], der[2], 1u, occ_b[3], der[3], 0u, &e_b[2]) ||
      !bind_resident_occurrence_coupling(occ_b[3], der[3], 1u, occ_b[4], der[4], 0u, &e_b[3]) ||
      !bind_resident_relational_network_closure(rec, der, occ_a, 5u, e_a, 4u, &n5) ||
      n5.identity == 0u || n5.actual_count != 5u ||
      !language_boundary_has(n5, 410u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n5, 460u, ResidentRecipePortDirection::output) ||
      !fold_resident_mixed_rank_affine(der, occ_a, e_a, 5u, 4u, n5, &w5) ||
      w5.witness_identity == 0u || !a.again ||
      resident_whitebox_recipe_logical_identity(w5) != a.again_logical ||
      !observe_resident_mixed_rank_whitebox(rec, der, occ_a, occ_b, 5u, e_a, e_b, 4u, &w5) ||
      w5.witness_identity != a.again->witness_identity)
    return false;
  return observe_resident_mixed_rank_evidence(rec, der, occ_a, occ_b, 5u, e_a, e_b, 4u, &w5,
                                              &ev5) &&
         replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count, ev5, &w5,
                                           &w5_logical, &w5_rev) &&
         w5_logical == a.again_logical;
}

#include "language_phrase_bank.inl"
#include "language_phrase_bank_ambiguous.inl"
#include "language_phrase_bank_lesion.inl"
#include "language_phrase_bank_relearn.inl"
#include "language_phrase_merge.inl"
#include "language_phrase_revise_authority.inl"
#include "language_phrase_revise_pair.inl"
#include "language_phrase_revise_rerecord.inl"
#include "language_phrase_revise_world.inl"
#include "language_phrase_world.inl"
#include "language_phrase_world_lesion.inl"
#include "language_phrase_world_revise.inl"
#include "language_phrase_world_two.inl"

static bool language_outer_cross(const LanguageAfterPn2& a, ResidentRecipeOccurrence* occ_s1,
                                 ResidentOccurrenceCoupling* e_s1,
                                 ResidentRecipeOccurrence* occ_world,
                                 ResidentOccurrenceCoupling* e_world,
                                 const ResidentRecipeCell* rec_world,
                                 const ResidentRecipeDerivation* der_world) {
  if (!occ_s1 || !e_s1 || !occ_world || !e_world || !rec_world || !der_world)
    return false;
  ResidentOccurrenceCoupling cross{};
  ResidentRelationalNetworkClosure n_cross{};
  DirectWhiteboxCondensationV1 w_cross{};
  ResidentRelationalNetworkSet two{};
  ResidentRecipeCell rec4[4] = {a.rec_r[0], a.rec_r[1], rec_world[0], rec_world[1]};
  ResidentRecipeDerivation der4[4] = {a.der_r[0], a.der_r[1], der_world[0], der_world[1]};
  ResidentRecipeOccurrence occ4[4] = {occ_s1[0], occ_s1[1], occ_world[0], occ_world[1]};
  if (!bind_resident_occurrence_causal_intersection_coupling(occ_s1[0], a.der_r[0], occ_world[0],
                                                             der_world[0], 310u, &cross) ||
      cross.kind != ResidentOccurrenceCouplingKind::causal_intersection || cross.reserved2 != 310u)
    return false;
  ResidentOccurrenceCoupling edges3[3] = {e_s1[0], e_world[0], cross};
  ResidentOccurrenceCoupling edges2[2] = {e_s1[0], e_world[0]};
  return bind_resident_relational_network_closure(rec4, der4, occ4, 4u, edges3, 3u, &n_cross) &&
         n_cross.identity != 0u && n_cross.actual_count == 4u &&
         n_cross.reconvergence_count >= 1u && n_cross.identity != a.n_r.identity &&
         !fold_resident_mixed_rank_affine(der4, occ4, edges3, 4u, 3u, n_cross, &w_cross) &&
         bind_resident_relational_network_set(rec4, der4, occ4, 4u, edges2, 2u, &two) &&
         two.network_count == 2u && two.networked_occurrence_count == 4u;
}

static bool language_outer_asif(const LanguageAfterPn2& a, ResidentRecipeOccurrence* occ_world,
                                const ResidentRecipeDerivation* der_world) {
  if (!occ_world || !der_world)
    return false;
  const std::uint32_t s0[] = {150u, 160u};
  ResidentRecipeOccurrence imag{};
  ResidentOccurrenceCoupling asif{};
  ResidentRelationalNetworkClosure n_asif{};
  DirectWhiteboxCondensationV1 w_asif{};
  if (!bind_endogenous(a.cells[a.i_src], a.der_r[0], s0, 0x1A10, &imag, 25u, 25 << 16) ||
      !bind_resident_occurrence_causal_intersection_coupling(imag, a.der_r[0], occ_world[0],
                                                             der_world[0], 311u, &asif) ||
      asif.kind != ResidentOccurrenceCouplingKind::causal_intersection || asif.reserved2 != 311u)
    return false;
  ResidentRecipeCell rec2[2] = {a.rec_r[0], a.cells[a.i_anc]};
  ResidentRecipeDerivation der2[2] = {a.der_r[0], der_world[0]};
  ResidentRecipeOccurrence occ2[2] = {imag, occ_world[0]};
  return bind_resident_relational_network_closure(rec2, der2, occ2, 2u, &asif, 1u, &n_asif) &&
         n_asif.identity != 0u && n_asif.actual_count == 1u && n_asif.reconvergence_count >= 1u &&
         !admit_resident_network_parent(rec2, der2, occ2, 2u, &asif, 1u, n_asif.identity,
                                        a.cells[a.i_anc].logical_recipe_id) &&
         !fold_resident_mixed_rank_affine(der2, occ2, &asif, 2u, 1u, n_asif, &w_asif);
}

static bool language_outer_two_cue(const LanguageAfterPn2& a, ResidentRecipeOccurrence* occ_world,
                                   const ResidentRecipeDerivation* der_world) {
  if (!occ_world || !der_world)
    return false;
  const std::uint32_t s0[] = {150u, 160u};
  const std::uint32_t t0[] = {210u, 220u};
  ResidentRecipeOccurrence imag_a{}, imag_b{};
  ResidentOccurrenceCoupling asif_a{}, asif_b{};
  ResidentRelationalNetworkClosure n_a{}, n_b{}, n_all{};
  DirectWhiteboxCondensationV1 w_all{};
  if (!bind_endogenous(a.cells[a.i_src], a.der_r[0], s0, 0x1A20, &imag_a, 26u, 26 << 16) ||
      !bind_endogenous(a.cells[a.i_pn1], a.der_r[1], t0, 0x1A21, &imag_b, 27u, 27 << 16) ||
      !bind_resident_occurrence_causal_intersection_coupling(imag_a, a.der_r[0], occ_world[0],
                                                             der_world[0], 311u, &asif_a) ||
      !bind_resident_occurrence_causal_intersection_coupling(imag_b, a.der_r[1], occ_world[0],
                                                             der_world[0], 311u, &asif_b) ||
      asif_a.reserved2 != 311u || asif_b.reserved2 != 311u)
    return false;
  ResidentRecipeCell rec_a[2] = {a.rec_r[0], a.cells[a.i_anc]};
  ResidentRecipeCell rec_b[2] = {a.rec_r[1], a.cells[a.i_anc]};
  ResidentRecipeDerivation der_a[2] = {a.der_r[0], der_world[0]};
  ResidentRecipeDerivation der_b[2] = {a.der_r[1], der_world[0]};
  ResidentRecipeOccurrence occ_a[2] = {imag_a, occ_world[0]};
  ResidentRecipeOccurrence occ_b[2] = {imag_b, occ_world[0]};
  ResidentRecipeCell rec3[3] = {a.rec_r[0], a.rec_r[1], a.cells[a.i_anc]};
  ResidentRecipeDerivation der3[3] = {a.der_r[0], a.der_r[1], der_world[0]};
  ResidentRecipeOccurrence occ3[3] = {imag_a, imag_b, occ_world[0]};
  ResidentOccurrenceCoupling edges2[2] = {asif_a, asif_b};
  return bind_resident_relational_network_closure(rec_a, der_a, occ_a, 2u, &asif_a, 1u, &n_a) &&
         bind_resident_relational_network_closure(rec_b, der_b, occ_b, 2u, &asif_b, 1u, &n_b) &&
         n_a.identity != 0u && n_b.identity != 0u && n_a.identity != n_b.identity &&
         n_a.actual_count == 1u && n_b.actual_count == 1u &&
         bind_resident_relational_network_closure(rec3, der3, occ3, 3u, edges2, 2u, &n_all) &&
         n_all.identity != n_a.identity && n_all.identity != n_b.identity &&
         n_all.actual_count == 1u &&
         !admit_resident_network_parent(rec3, der3, occ3, 3u, edges2, 2u, n_all.identity,
                                        a.cells[a.i_anc].logical_recipe_id) &&
         !fold_resident_mixed_rank_affine(der3, occ3, edges2, 3u, 2u, n_all, &w_all);
}

static bool language_outer_multilingual(const LanguageAfterPn2& a,
                                        ResidentRecipeOccurrence* occ_world,
                                        const ResidentRecipeDerivation* der_world) {
  if (!occ_world || !der_world || !a.again)
    return false;
  const std::uint32_t a0[] = {510u, 520u};
  const std::uint32_t a1[] = {520u, 530u};
  const std::uint32_t b0[] = {610u, 620u};
  const std::uint32_t b1[] = {620u, 630u};
  ResidentRecipeOccurrence occ_a1[2]{}, occ_a2[2]{}, occ_b1[2]{}, occ_b2[2]{};
  ResidentOccurrenceCoupling e_a1[1]{}, e_a2[1]{}, e_b1[1]{}, e_b2[1]{};
  ResidentRelationalNetworkClosure n_a{}, n_b{};
  DirectWhiteboxCondensationV1 w_a{}, w_b{};
  ResidentOccurrenceCoupling world_a{}, world_b{};
  ResidentRelationalNetworkClosure n_aw{}, n_bw{};
  DirectWhiteboxCondensationV1 w_aw{};
  ResidentRelationalNetworkSet two{};
  if (!bind_live(a.cells[a.i_src], a.der_r[0], a0, 0x1D10, &occ_a1[0], 32u, 32 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], a1, 0x1D11, &occ_a1[1], 32u, 32 << 16) ||
      !bind_live(a.cells[a.i_src], a.der_r[0], a0, 0x1D12, &occ_a2[0], 34u, 34 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], a1, 0x1D13, &occ_a2[1], 34u, 34 << 16) ||
      !bind_live(a.cells[a.i_src], a.der_r[0], b0, 0x1D20, &occ_b1[0], 33u, 33 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], b1, 0x1D21, &occ_b1[1], 33u, 33 << 16) ||
      !bind_live(a.cells[a.i_src], a.der_r[0], b0, 0x1D22, &occ_b2[0], 35u, 35 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], b1, 0x1D23, &occ_b2[1], 35u, 35 << 16) ||
      !bind_resident_occurrence_coupling(occ_a1[0], a.der_r[0], 1u, occ_a1[1], a.der_r[1], 0u,
                                         &e_a1[0]) ||
      !bind_resident_occurrence_coupling(occ_a2[0], a.der_r[0], 1u, occ_a2[1], a.der_r[1], 0u,
                                         &e_a2[0]) ||
      !bind_resident_occurrence_coupling(occ_b1[0], a.der_r[0], 1u, occ_b1[1], a.der_r[1], 0u,
                                         &e_b1[0]) ||
      !bind_resident_occurrence_coupling(occ_b2[0], a.der_r[0], 1u, occ_b2[1], a.der_r[1], 0u,
                                         &e_b2[0]) ||
      !bind_resident_relational_network_closure(a.rec_r, a.der_r, occ_a1, 2u, e_a1, 1u, &n_a) ||
      !bind_resident_relational_network_closure(a.rec_r, a.der_r, occ_b1, 2u, e_b1, 1u, &n_b) ||
      n_a.identity == 0u || n_b.identity == 0u || n_a.identity == n_b.identity ||
      n_a.actual_count != 2u || n_b.actual_count != 2u ||
      resident_relational_network_recruitment_identity(n_a) !=
          resident_relational_network_recruitment_identity(n_b) ||
      resident_network_boundary_relation_equal(n_a, n_b) ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, occ_a1, occ_a2, 2u, e_a1, e_a2, 1u,
                                            &w_a) ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, occ_b1, occ_b2, 2u, e_b1, e_b2, 1u,
                                            &w_b) ||
      w_a.witness_identity != a.again->witness_identity ||
      w_b.witness_identity != a.again->witness_identity ||
      !bind_resident_occurrence_causal_intersection_coupling(occ_a1[0], a.der_r[0], occ_world[0],
                                                             der_world[0], 312u, &world_a) ||
      !bind_resident_occurrence_causal_intersection_coupling(occ_b1[0], a.der_r[0], occ_world[0],
                                                             der_world[0], 312u, &world_b) ||
      world_a.reserved2 != 312u || world_b.reserved2 != 312u)
    return false;
  ResidentRecipeCell rec3[3] = {a.rec_r[0], a.rec_r[1], a.cells[a.i_anc]};
  ResidentRecipeDerivation der3[3] = {a.der_r[0], a.der_r[1], der_world[0]};
  ResidentRecipeOccurrence occ_aw[3] = {occ_a1[0], occ_a1[1], occ_world[0]};
  ResidentRecipeOccurrence occ_bw[3] = {occ_b1[0], occ_b1[1], occ_world[0]};
  ResidentOccurrenceCoupling edges_aw[2] = {e_a1[0], world_a};
  ResidentOccurrenceCoupling edges_bw[2] = {e_b1[0], world_b};
  ResidentRecipeCell rec4[4] = {a.rec_r[0], a.rec_r[1], a.rec_r[0], a.rec_r[1]};
  ResidentRecipeDerivation der4[4] = {a.der_r[0], a.der_r[1], a.der_r[0], a.der_r[1]};
  ResidentRecipeOccurrence occ4[4] = {occ_a1[0], occ_a1[1], occ_b1[0], occ_b1[1]};
  ResidentOccurrenceCoupling edges2[2] = {e_a1[0], e_b1[0]};
  return bind_resident_relational_network_closure(rec3, der3, occ_aw, 3u, edges_aw, 2u, &n_aw) &&
         bind_resident_relational_network_closure(rec3, der3, occ_bw, 3u, edges_bw, 2u, &n_bw) &&
         n_aw.identity != 0u && n_bw.identity != 0u && n_aw.identity != n_bw.identity &&
         n_aw.actual_count == 3u && n_bw.actual_count == 3u &&
         !fold_resident_mixed_rank_affine(der3, occ_aw, edges_aw, 3u, 2u, n_aw, &w_aw) &&
         bind_resident_relational_network_set(rec4, der4, occ4, 4u, edges2, 2u, &two) &&
         two.network_count == 2u && two.networked_occurrence_count == 4u;
}

static bool language_outer_discourse(const LanguageAfterPn2& a, ResidentRecipeOccurrence* occ_world,
                                     const ResidentRecipeDerivation* der_world) {
  if (!occ_world || !der_world || !a.again)
    return false;
  const std::uint32_t a0[] = {710u, 720u};
  const std::uint32_t a1[] = {720u, 730u};
  const std::uint32_t b0[] = {810u, 820u};
  const std::uint32_t b1[] = {820u, 830u};
  const std::uint64_t src_logical = a.cells[a.i_src].logical_recipe_id;
  const std::uint64_t pn1_logical = a.cells[a.i_pn1].logical_recipe_id;
  ResidentRecipeOccurrence occ_a1[2]{}, occ_a2[2]{}, occ_b1[2]{}, occ_b2[2]{};
  ResidentOccurrenceCoupling e_a1[1]{}, e_a2[1]{}, e_b1[1]{}, e_b2[1]{};
  ResidentRelationalNetworkClosure n_a{}, n_dead{}, n_aw{}, n_aw_dead{}, n_bw{};
  DirectWhiteboxCondensationV1 w_a{}, w_dead{}, w_b{};
  ResidentOccurrenceCoupling world_a{}, world_b{};
  if (!bind_live(a.cells[a.i_src], a.der_r[0], a0, 0x1E10, &occ_a1[0], 36u, 36 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], a1, 0x1E11, &occ_a1[1], 36u, 36 << 16) ||
      !bind_live(a.cells[a.i_src], a.der_r[0], a0, 0x1E12, &occ_a2[0], 38u, 38 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], a1, 0x1E13, &occ_a2[1], 38u, 38 << 16) ||
      !bind_resident_occurrence_coupling(occ_a1[0], a.der_r[0], 1u, occ_a1[1], a.der_r[1], 0u,
                                         &e_a1[0]) ||
      !bind_resident_occurrence_coupling(occ_a2[0], a.der_r[0], 1u, occ_a2[1], a.der_r[1], 0u,
                                         &e_a2[0]) ||
      !bind_resident_relational_network_closure(a.rec_r, a.der_r, occ_a1, 2u, e_a1, 1u, &n_a) ||
      n_a.actual_count != 2u ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, occ_a1, occ_a2, 2u, e_a1, e_a2, 1u,
                                            &w_a) ||
      w_a.witness_identity != a.again->witness_identity ||
      !bind_resident_occurrence_causal_intersection_coupling(occ_a1[0], a.der_r[0], occ_world[0],
                                                             der_world[0], 313u, &world_a) ||
      world_a.reserved2 != 313u)
    return false;
  ResidentRecipeCell rec3[3] = {a.rec_r[0], a.rec_r[1], a.cells[a.i_anc]};
  ResidentRecipeDerivation der3[3] = {a.der_r[0], a.der_r[1], der_world[0]};
  ResidentRecipeOccurrence occ_aw[3] = {occ_a1[0], occ_a1[1], occ_world[0]};
  ResidentOccurrenceCoupling edges_aw[2] = {e_a1[0], world_a};
  if (!bind_resident_relational_network_closure(rec3, der3, occ_aw, 3u, edges_aw, 2u, &n_aw) ||
      n_aw.actual_count != 3u)
    return false;
  ResidentRecipeOccurrence dead[2] = {occ_a1[0], occ_a1[1]};
  dead[0].state = kResidentRecipeOccurrenceSettled;
  dead[1].state = kResidentRecipeOccurrenceSettled;
  ResidentRecipeOccurrence occ_aw_dead[3] = {dead[0], dead[1], occ_world[0]};
  if (bind_resident_relational_network_closure(a.rec_r, a.der_r, dead, 2u, e_a1, 1u, &n_dead) ||
      observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, dead, occ_a2, 2u, e_a1, e_a2, 1u,
                                           &w_dead) ||
      bind_resident_relational_network_closure(rec3, der3, occ_aw_dead, 3u, edges_aw, 2u,
                                               &n_aw_dead) ||
      a.cells[a.i_src].logical_recipe_id != src_logical ||
      a.cells[a.i_pn1].logical_recipe_id != pn1_logical)
    return false;
  if (!bind_live(a.cells[a.i_src], a.der_r[0], b0, 0x1E20, &occ_b1[0], 37u, 37 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], b1, 0x1E21, &occ_b1[1], 37u, 37 << 16) ||
      !bind_live(a.cells[a.i_src], a.der_r[0], b0, 0x1E22, &occ_b2[0], 39u, 39 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], b1, 0x1E23, &occ_b2[1], 39u, 39 << 16) ||
      !bind_resident_occurrence_coupling(occ_b1[0], a.der_r[0], 1u, occ_b1[1], a.der_r[1], 0u,
                                         &e_b1[0]) ||
      !bind_resident_occurrence_coupling(occ_b2[0], a.der_r[0], 1u, occ_b2[1], a.der_r[1], 0u,
                                         &e_b2[0]) ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, occ_b1, occ_b2, 2u, e_b1, e_b2, 1u,
                                            &w_b) ||
      w_b.witness_identity != a.again->witness_identity ||
      !bind_resident_occurrence_causal_intersection_coupling(occ_b1[0], a.der_r[0], occ_world[0],
                                                             der_world[0], 313u, &world_b) ||
      world_b.reserved2 != 313u)
    return false;
  ResidentRecipeOccurrence occ_bw[3] = {occ_b1[0], occ_b1[1], occ_world[0]};
  ResidentOccurrenceCoupling edges_bw[2] = {e_b1[0], world_b};
  DirectWhiteboxCondensationV1 w_bw{};
  return bind_resident_relational_network_closure(rec3, der3, occ_bw, 3u, edges_bw, 2u, &n_bw) &&
         n_bw.identity != 0u && n_bw.identity != n_aw.identity && n_bw.actual_count == 3u &&
         !fold_resident_mixed_rank_affine(der3, occ_bw, edges_bw, 3u, 2u, n_bw, &w_bw);
}

static bool language_after_pn2(const LanguageAfterPn2& a) {
  if (!a.cells || !a.ders || !a.rec_r || !a.der_r || !a.occ_r1 || !a.occ_r2 || !a.e_r1 || !a.e_r2 ||
      !a.again || !a.ev2 || a.cell_count <= a.i_pn2 || a.n_r.identity == 0u ||
      a.again_logical == 0u || a.again_logical != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t z0[] = {110u, 120u};
  const std::uint32_t z1[] = {120u, 130u};
  const std::uint32_t r0[] = {50u, 60u};
  const std::uint32_t r1[] = {60u, 70u};
  ResidentRecipeOccurrence q0{}, q2{};
  ResidentOccurrenceCoupling edge_pn2{};
  ResidentRelationalNetworkClosure n_pn2{};
  ResidentRecipeCell rec_pn2[2] = {a.cells[a.i_anc], a.cells[a.i_pn2]};
  ResidentRecipeDerivation der_pn2[2] = {a.ders[a.i_anc], a.ders[a.i_pn2]};
  der_pn2[0].logical_recipe_id = a.cells[a.i_anc].logical_recipe_id;
  der_pn2[0].revision_identity = a.cells[a.i_anc].revision_identity;
  der_pn2[1].logical_recipe_id = a.cells[a.i_pn2].logical_recipe_id;
  der_pn2[1].revision_identity = a.cells[a.i_pn2].revision_identity;
  if (!bind_live(a.cells[a.i_anc], der_pn2[0], z0, 0xD10, &q0, 8u, 8 << 16) ||
      !bind_live(a.cells[a.i_pn2], der_pn2[1], z1, 0xD11, &q2, 8u, 8 << 16) ||
      !bind_resident_occurrence_coupling(q0, der_pn2[0], 1u, q2, der_pn2[1], 0u, &edge_pn2))
    return false;
  ResidentRecipeOccurrence occ_pn2[2] = {q0, q2};
  if (!bind_resident_relational_network_closure(rec_pn2, der_pn2, occ_pn2, 2u, &edge_pn2, 1u,
                                                &n_pn2) ||
      n_pn2.identity == 0u || n_pn2.identity == a.n_r.identity ||
      n_pn2.identity == a.n_next.identity || n_pn2.actual_count != 2u ||
      !language_boundary_has(n_pn2, 110u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n_pn2, 130u, ResidentRecipePortDirection::output) ||
      !admit_resident_network_parent(rec_pn2, der_pn2, occ_pn2, 2u, &edge_pn2, 1u, n_pn2.identity,
                                     a.cells[a.i_pn2].logical_recipe_id) ||
      admit_resident_network_parent(rec_pn2, der_pn2, occ_pn2, 2u, &edge_pn2, 1u, a.n_r.identity,
                                    a.cells[a.i_pn2].logical_recipe_id))
    return false;
  ResidentRelationalNetworkClosure n_src{};
  DirectWhiteboxCondensationV1 source_w{};
  if (!bind_resident_relational_network_closure(a.rec_r, a.der_r, a.occ_r1, 2u, a.e_r1, 1u,
                                                &n_src) ||
      n_src.identity != a.n_r.identity ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, a.occ_r1, a.occ_r2, 2u, a.e_r1,
                                            a.e_r2, 1u, &source_w) ||
      source_w.witness_identity != a.again->witness_identity)
    return false;
  ResidentRecipeOccurrence q_lesion{};
  ResidentOccurrenceCoupling edge_lesion{};
  ResidentRelationalNetworkClosure n_lesion{};
  ResidentRecipeCell rec_lesion[2] = {a.cells[a.i_anc], a.cells[a.i_pn1]};
  ResidentRecipeDerivation der_lesion[2] = {a.ders[a.i_anc], a.ders[a.i_pn1]};
  der_lesion[0].logical_recipe_id = a.cells[a.i_anc].logical_recipe_id;
  der_lesion[0].revision_identity = a.cells[a.i_anc].revision_identity;
  der_lesion[1].logical_recipe_id = a.cells[a.i_pn1].logical_recipe_id;
  der_lesion[1].revision_identity = a.cells[a.i_pn1].revision_identity;
  if (!bind_live(a.cells[a.i_pn1], der_lesion[1], z1, 0xD21, &q_lesion, 8u, 8 << 16) ||
      !bind_resident_occurrence_coupling(q0, der_lesion[0], 1u, q_lesion, der_lesion[1], 0u,
                                         &edge_lesion))
    return false;
  ResidentRecipeOccurrence occ_lesion[2] = {q0, q_lesion};
  if (!bind_resident_relational_network_closure(rec_lesion, der_lesion, occ_lesion, 2u,
                                                &edge_lesion, 1u, &n_lesion) ||
      n_lesion.identity == 0u || n_lesion.identity == n_pn2.identity ||
      admit_resident_network_parent(rec_lesion, der_lesion, occ_lesion, 2u, &edge_lesion, 1u,
                                    n_pn2.identity, a.cells[a.i_pn2].logical_recipe_id))
    return false;
  ResidentRecipeCell rec_alt[2] = {a.cells[a.i_src], a.cells[a.i_pn2]};
  ResidentRecipeDerivation der_b2[2] = {a.ders[a.i_src], a.ders[a.i_pn2]};
  der_b2[0].logical_recipe_id = a.cells[a.i_src].logical_recipe_id;
  der_b2[0].revision_identity = a.cells[a.i_src].revision_identity;
  der_b2[1].logical_recipe_id = a.cells[a.i_pn2].logical_recipe_id;
  der_b2[1].revision_identity = a.cells[a.i_pn2].revision_identity;
  ResidentRecipeOccurrence occ_b1[2]{}, occ_b2[2]{};
  ResidentOccurrenceCoupling e_b1[1]{}, e_b2[1]{};
  ResidentRelationalNetworkClosure n_amb{};
  ResidentRelationalNetworkSet amb{};
  ResidentRelationalNetworkClosure whole_amb{};
  DirectWhiteboxCondensationV1 w_b{};
  ResidentNetworkCondensationEvidence ev_b{};
  std::uint64_t b_logical = 0, b_rev = 0;
  if (!bind_live(a.cells[a.i_src], der_b2[0], r0, 0xAB10, &occ_b1[0], 9u, 9 << 16) ||
      !bind_live(a.cells[a.i_pn2], der_b2[1], r1, 0xAB11, &occ_b1[1], 9u, 9 << 16) ||
      !bind_live(a.cells[a.i_src], der_b2[0], r0, 0xAB20, &occ_b2[0], 10u, 10 << 16) ||
      !bind_live(a.cells[a.i_pn2], der_b2[1], r1, 0xAB21, &occ_b2[1], 10u, 10 << 16) ||
      !bind_resident_occurrence_coupling(occ_b1[0], der_b2[0], 1u, occ_b1[1], der_b2[1], 0u,
                                         &e_b1[0]) ||
      !bind_resident_occurrence_coupling(occ_b2[0], der_b2[0], 1u, occ_b2[1], der_b2[1], 0u,
                                         &e_b2[0]) ||
      !bind_resident_relational_network_closure(rec_alt, der_b2, occ_b1, 2u, e_b1, 1u, &n_amb) ||
      n_amb.identity == 0u || n_amb.identity == a.n_r.identity ||
      !language_boundary_has(n_amb, 50u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n_amb, 70u, ResidentRecipePortDirection::output) ||
      !language_boundary_has(a.n_r, 50u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(a.n_r, 70u, ResidentRecipePortDirection::output))
    return false;
  ResidentRecipeCell rec_amb[4] = {a.cells[a.i_src], a.cells[a.i_pn1], a.cells[a.i_src],
                                   a.cells[a.i_pn2]};
  ResidentRecipeDerivation der_amb[4] = {a.der_r[0], a.der_r[1], der_b2[0], der_b2[1]};
  ResidentRecipeOccurrence occ_amb[4] = {a.occ_r1[0], a.occ_r1[1], occ_b1[0], occ_b1[1]};
  ResidentOccurrenceCoupling edges_amb[2] = {a.e_r1[0], e_b1[0]};
  if (!bind_resident_relational_network_set(rec_amb, der_amb, occ_amb, 4u, edges_amb, 2u, &amb) ||
      amb.network_count != 2u || amb.networked_occurrence_count != 4u ||
      (amb.networks[0].identity != a.n_r.identity && amb.networks[1].identity != a.n_r.identity) ||
      (amb.networks[0].identity != n_amb.identity && amb.networks[1].identity != n_amb.identity) ||
      bind_resident_relational_network_closure(rec_amb, der_amb, occ_amb, 4u, edges_amb, 2u,
                                               &whole_amb) ||
      observe_resident_mixed_rank_whitebox(rec_alt, der_b2, occ_b1, occ_b1, 2u, e_b1, e_b1, 1u,
                                           &w_b) ||
      !observe_resident_mixed_rank_whitebox(rec_alt, der_b2, occ_b1, occ_b2, 2u, e_b1, e_b2, 1u,
                                            &w_b) ||
      w_b.witness_identity == 0u || w_b.witness_identity == a.again->witness_identity ||
      !observe_resident_mixed_rank_evidence(rec_alt, der_b2, occ_b1, occ_b2, 2u, e_b1, e_b2, 1u,
                                            &w_b, &ev_b) ||
      !replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count, ev_b, &w_b,
                                         &b_logical, &b_rev) ||
      b_logical == 0u || b_logical == a.cells[a.i_pn2].logical_recipe_id ||
      b_logical == a.again_logical)
    return false;
  ResidentRelationalNetworkSet only_source{};
  ResidentRelationalNetworkSet only_alt{};
  DirectWhiteboxCondensationV1 w_src{};
  if (!bind_resident_relational_network_set(rec_amb, der_amb, occ_amb, 4u, a.e_r1, 1u,
                                            &only_source) ||
      only_source.network_count != 1u || only_source.networked_occurrence_count != 2u ||
      only_source.isolated_occurrence_count != 2u ||
      only_source.networks[0].identity != a.n_r.identity ||
      !bind_resident_relational_network_set(rec_amb, der_amb, occ_amb, 4u, e_b1, 1u, &only_alt) ||
      only_alt.network_count != 1u || only_alt.networked_occurrence_count != 2u ||
      only_alt.isolated_occurrence_count != 2u || only_alt.networks[0].identity != n_amb.identity ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, a.occ_r1, a.occ_r2, 2u, a.e_r1,
                                            a.e_r2, 1u, &w_src) ||
      w_src.witness_identity != a.again->witness_identity ||
      !observe_resident_mixed_rank_whitebox(rec_alt, der_b2, occ_b1, occ_b2, 2u, e_b1, e_b2, 1u,
                                            &w_src) ||
      w_src.witness_identity != w_b.witness_identity)
    return false;
  const std::uint32_t s0[] = {150u, 160u};
  const std::uint32_t s1[] = {160u, 170u};
  ResidentRecipeOccurrence occ_s1[2]{}, occ_s2[2]{};
  ResidentOccurrenceCoupling e_s1[1]{}, e_s2[1]{};
  ResidentRelationalNetworkClosure n_s{};
  DirectWhiteboxCondensationV1 w_s{};
  if (!bind_live(a.cells[a.i_src], a.der_r[0], s0, 0x1510, &occ_s1[0], 11u, 11 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], s1, 0x1511, &occ_s1[1], 11u, 11 << 16) ||
      !bind_live(a.cells[a.i_src], a.der_r[0], s0, 0x1520, &occ_s2[0], 12u, 12 << 16) ||
      !bind_live(a.cells[a.i_pn1], a.der_r[1], s1, 0x1521, &occ_s2[1], 12u, 12 << 16) ||
      !bind_resident_occurrence_coupling(occ_s1[0], a.der_r[0], 1u, occ_s1[1], a.der_r[1], 0u,
                                         &e_s1[0]) ||
      !bind_resident_occurrence_coupling(occ_s2[0], a.der_r[0], 1u, occ_s2[1], a.der_r[1], 0u,
                                         &e_s2[0]) ||
      !bind_resident_relational_network_closure(a.rec_r, a.der_r, occ_s1, 2u, e_s1, 1u, &n_s) ||
      n_s.identity == 0u || n_s.identity == a.n_r.identity ||
      !language_boundary_has(n_s, 150u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n_s, 170u, ResidentRecipePortDirection::output) ||
      resident_network_boundary_relation_equal(a.n_r, n_s) ||
      observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, a.occ_r1, occ_s1, 2u, a.e_r1, e_s1, 1u,
                                           &w_s) ||
      !observe_resident_mixed_rank_whitebox(a.rec_r, a.der_r, occ_s1, occ_s2, 2u, e_s1, e_s2, 1u,
                                            &w_s) ||
      w_s.witness_identity != a.again->witness_identity)
    return false;
  ResidentNetworkCondensationEvidence ev_s{};
  std::uint64_t s_logical = 0, s_rev = 0, s_alt = 0, s_alt_rev = 0;
  ResidentRecipeOccurrence express{};
  const std::uint32_t express_vars[] = {150u, 170u};
  const bool ok = observe_resident_mixed_rank_evidence(a.rec_r, a.der_r, occ_s1, occ_s2, 2u, e_s1,
                                                       e_s2, 1u, &w_s, &ev_s) &&
                  ev_s.observation_identities[0] != a.ev2->observation_identities[0] &&
                  ev_s.observation_identities[1] != a.ev2->observation_identities[1] &&
                  ev_s.witness_identity != a.ev2->witness_identity && ev_s.witness_identity != 0u &&
                  replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count,
                                                    ev_s, &w_s, &s_logical, &s_rev) &&
                  s_logical == a.again_logical && s_logical == a.cells[a.i_pn2].logical_recipe_id &&
                  !replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count,
                                                     ev_s, &w_b, &s_alt, &s_alt_rev) &&
                  bind_live(a.cells[a.i_pn2], a.ders[a.i_pn2], express_vars, 0xE150, &express) &&
                  express.logical_recipe_id == a.cells[a.i_pn2].logical_recipe_id &&
                  express.bindings[0].variable_identity == 150u &&
                  express.bindings[1].variable_identity == 170u &&
                  nominate_source_network_credit(a.rec_r, a.der_r, a.occ_r1, a.e_r1, rec_alt,
                                                 der_b2, occ_b1, e_b1, a.n_r, n_amb);
  return ok && language_recipe_opportunity(a, express) &&
         language_outer_cross(a, occ_s1, e_s1, occ_pn2, &edge_pn2, rec_pn2, der_pn2) &&
         language_outer_asif(a, occ_pn2, der_pn2) && language_outer_two_cue(a, occ_pn2, der_pn2) &&
         language_outer_multilingual(a, occ_pn2, der_pn2) &&
         language_outer_discourse(a, occ_pn2, der_pn2) &&
         language_outer_imagine(a, n_s, occ_s1, occ_s2, e_s1, e_s2, &ev_s, &w_s) &&
         language_triple_linear(a) && language_c4_refuse(a) && language_w5_c2(a) &&
         language_recursive_wide(a) && language_pn3_world(a, occ_pn2, der_pn2) &&
         language_pn4_mint(a) && language_phrase_compose(a) && language_phrase_reuse(a) &&
         language_phrase_opportunity(a) && language_phrase_two_surfaces(a) &&
         language_phrase_bank(a) && language_phrase_bank_ambiguous(a) && language_phrase_world(a) &&
         language_phrase_world_two(a) && language_phrase_world_lesion(a) &&
         language_phrase_bank_lesion(a) && language_phrase_bank_relearn(a) &&
         language_phrase_world_revise(a) && language_phrase_revise_rerecord(a) &&
         language_phrase_revise_world(a) && language_phrase_revise_pair(a) &&
         language_phrase_revise_authority(a);
}
