static bool language_recursive_wide(const LanguageAfterPn2& a) {
  ResidentRecipeCell rec[5] = {a.cells[a.i_anc], a.cells[a.i_src], a.cells[a.i_anc],
                               a.cells[a.i_pn2], a.cells[a.i_anc]};
  ResidentRecipeDerivation der[5] = {a.ders[a.i_anc], a.ders[a.i_src], a.ders[a.i_anc],
                                     a.ders[a.i_pn2], a.ders[a.i_anc]};
  for (std::uint32_t i = 0u; i < 5u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
  }
  if (der[0].parameter_count != 0u || der[0].relation_count != 0u ||
      der[1].parameter_count != 2u || der[1].relation_count != 1u ||
      der[3].parameter_count != 2u || der[3].relation_count != 1u)
    return false;
  const std::uint32_t t0[] = {910u, 920u};
  const std::uint32_t t1[] = {920u, 930u};
  const std::uint32_t t2[] = {930u, 940u};
  const std::uint32_t t3[] = {940u, 950u};
  const std::uint32_t t4[] = {950u, 960u};
  ResidentRecipeOccurrence occ_a[5]{}, occ_b[5]{};
  ResidentOccurrenceCoupling e_a[4]{}, e_b[4]{};
  ResidentRelationalNetworkClosure n5{};
  DirectWhiteboxCondensationV1 w5{};
  ResidentNetworkCondensationEvidence ev5{};
  std::uint64_t w5_logical = 0, w5_rev = 0;
  if (!bind_live(rec[0], der[0], t0, 0x1F10, &occ_a[0], 40u, 40 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1F11, &occ_a[1], 40u, 40 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1F12, &occ_a[2], 40u, 40 << 16) ||
      !bind_live(rec[3], der[3], t3, 0x1F13, &occ_a[3], 40u, 40 << 16) ||
      !bind_live(rec[4], der[4], t4, 0x1F14, &occ_a[4], 40u, 40 << 16) ||
      !bind_live(rec[0], der[0], t0, 0x1F20, &occ_b[0], 41u, 41 << 16) ||
      !bind_live(rec[1], der[1], t1, 0x1F21, &occ_b[1], 41u, 41 << 16) ||
      !bind_live(rec[2], der[2], t2, 0x1F22, &occ_b[2], 41u, 41 << 16) ||
      !bind_live(rec[3], der[3], t3, 0x1F23, &occ_b[3], 41u, 41 << 16) ||
      !bind_live(rec[4], der[4], t4, 0x1F24, &occ_b[4], 41u, 41 << 16) ||
      !bind_resident_occurrence_coupling(occ_a[0], der[0], 1u, occ_a[1], der[1], 0u,
                                        &e_a[0]) ||
      !bind_resident_occurrence_coupling(occ_a[1], der[1], 1u, occ_a[2], der[2], 0u,
                                        &e_a[1]) ||
      !bind_resident_occurrence_coupling(occ_a[2], der[2], 1u, occ_a[3], der[3], 0u,
                                        &e_a[2]) ||
      !bind_resident_occurrence_coupling(occ_a[3], der[3], 1u, occ_a[4], der[4], 0u,
                                        &e_a[3]) ||
      !bind_resident_occurrence_coupling(occ_b[0], der[0], 1u, occ_b[1], der[1], 0u,
                                        &e_b[0]) ||
      !bind_resident_occurrence_coupling(occ_b[1], der[1], 1u, occ_b[2], der[2], 0u,
                                        &e_b[1]) ||
      !bind_resident_occurrence_coupling(occ_b[2], der[2], 1u, occ_b[3], der[3], 0u,
                                        &e_b[2]) ||
      !bind_resident_occurrence_coupling(occ_b[3], der[3], 1u, occ_b[4], der[4], 0u,
                                        &e_b[3]) ||
      !bind_resident_relational_network_closure(rec, der, occ_a, 5u, e_a, 4u, &n5) ||
      n5.identity == 0u || n5.actual_count != 5u ||
      !language_boundary_has(n5, 910u, ResidentRecipePortDirection::input) ||
      !language_boundary_has(n5, 960u, ResidentRecipePortDirection::output) ||
      !fold_resident_mixed_rank_affine(der, occ_a, e_a, 5u, 4u, n5, &w5) ||
      w5.witness_identity == 0u || !a.again ||
      resident_whitebox_recipe_logical_identity(w5) == a.again_logical ||
      resident_whitebox_recipe_logical_identity(w5) ==
          a.cells[a.i_pn2].logical_recipe_id ||
      !observe_resident_mixed_rank_whitebox(rec, der, occ_a, occ_b, 5u, e_a, e_b, 4u,
                                           &w5))
    return false;
  if (!observe_resident_mixed_rank_evidence(rec, der, occ_a, occ_b, 5u, e_a, e_b, 4u,
                                           &w5, &ev5) ||
      !replay_resident_network_candidate(a.cells, a.cell_count, a.ders, a.der_count,
                                         ev5, &w5, &w5_logical, &w5_rev) ||
      w5_logical == 0u || w5_logical == a.again_logical ||
      w5_logical == a.cells[a.i_pn2].logical_recipe_id || !a.workspace ||
      a.workspace->cells != a.cells || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr)
    return false;
  const ResidentRecipeCell parent = a.cells[a.i_pn2];
  a.workspace->claimed_identity = n5.identity;
  a.workspace->network_cells = rec;
  a.workspace->network_ders = der;
  a.workspace->occurrences = occ_a;
  a.workspace->couplings = e_a;
  a.workspace->occurrence_count = 5u;
  a.workspace->coupling_count = 4u;
  a.workspace->parent_cell = a.i_pn2;
  a.workspace->parent_logical = parent.logical_recipe_id;
  a.workspace->witness = &w5;
  a.workspace->evidence = &ev5;
  const std::uint32_t before = *a.workspace->cell_count;
  ResidentRecipeOccurrence unfold{};
  const std::uint32_t held[] = {910u, 960u};
  if (!foundry_workspace_ready(a.workspace) ||
      !foundry_materialize_from_workspace(a.workspace, nullptr) ||
      *a.workspace->cell_count != before + 1u)
    return false;
  const std::uint32_t i_pn3 = *a.workspace->cell_count - 1u;
  if (a.cells[a.i_pn2].logical_recipe_id != parent.logical_recipe_id ||
      a.cells[a.i_pn2].revision_identity != parent.revision_identity ||
      a.cells[i_pn3].logical_recipe_id != w5_logical ||
      a.ders[i_pn3].parent_logical_recipe_id != parent.logical_recipe_id ||
      !replay_resident_whitebox_recipe(w5, a.cells[i_pn3], a.ders[i_pn3]) ||
      !bind_live(a.cells[i_pn3], a.ders[i_pn3], held, 0x1F30, &unfold) ||
      unfold.logical_recipe_id != w5_logical ||
      unfold.bindings[0].variable_identity != 910u ||
      unfold.bindings[1].variable_identity != 960u)
    return false;
  ResidentRecipeOccurrence dead[5] = {occ_a[0], occ_a[1], occ_a[2], occ_a[3], occ_a[4]};
  for (std::uint32_t i = 0u; i < 5u; ++i)
    dead[i].state = kResidentRecipeOccurrenceSettled;
  ResidentRelationalNetworkClosure n_dead{};
  ResidentRecipeOccurrence kept{};
  if (bind_resident_relational_network_closure(rec, der, dead, 5u, e_a, 4u, &n_dead) ||
      !bind_live(a.cells[i_pn3], a.ders[i_pn3], held, 0x1F31, &kept) ||
      kept.logical_recipe_id != w5_logical)
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[a.i_src], a.cells[i_pn3]};
  ResidentRecipeDerivation der2[2] = {a.ders[a.i_src], a.ders[i_pn3]};
  der2[0].logical_recipe_id = rec2[0].logical_recipe_id;
  der2[0].revision_identity = rec2[0].revision_identity;
  der2[1].logical_recipe_id = rec2[1].logical_recipe_id;
  der2[1].revision_identity = rec2[1].revision_identity;
  const std::uint32_t u0[] = {1110u, 1120u};
  const std::uint32_t u1[] = {1120u, 1130u};
  ResidentRecipeOccurrence occ_u1[2]{}, occ_u2[2]{};
  ResidentOccurrenceCoupling e_u1[1]{}, e_u2[1]{};
  DirectWhiteboxCondensationV1 w_u{};
  ResidentNetworkCondensationEvidence ev_u{};
  std::uint64_t u_logical = 0, u_rev = 0;
  if (!bind_live(rec2[0], der2[0], u0, 0x2110, &occ_u1[0], 42u, 42 << 16) ||
      !bind_live(rec2[1], der2[1], u1, 0x2111, &occ_u1[1], 42u, 42 << 16) ||
      !bind_live(rec2[0], der2[0], u0, 0x2120, &occ_u2[0], 43u, 43 << 16) ||
      !bind_live(rec2[1], der2[1], u1, 0x2121, &occ_u2[1], 43u, 43 << 16) ||
      !bind_resident_occurrence_coupling(occ_u1[0], der2[0], 1u, occ_u1[1], der2[1], 0u,
                                        &e_u1[0]) ||
      !bind_resident_occurrence_coupling(occ_u2[0], der2[0], 1u, occ_u2[1], der2[1], 0u,
                                        &e_u2[0]) ||
      !observe_resident_mixed_rank_whitebox(rec2, der2, occ_u1, occ_u2, 2u, e_u1, e_u2,
                                           1u, &w_u) ||
      !observe_resident_mixed_rank_evidence(rec2, der2, occ_u1, occ_u2, 2u, e_u1, e_u2,
                                           1u, &w_u, &ev_u) ||
      !replay_resident_network_candidate(a.cells, *a.workspace->cell_count, a.ders,
                                         a.workspace->state->derivation_count, ev_u,
                                         &w_u, &u_logical, &u_rev))
    return false;
  return u_logical != 0u && u_logical != w5_logical && u_logical != a.again_logical;
}

static bool language_pn3_world(const LanguageAfterPn2& a,
                              ResidentRecipeOccurrence* occ_world,
                              const ResidentRecipeDerivation* der_world) {
  if (!occ_world || !der_world || !a.workspace || a.workspace->cell_count == nullptr ||
      *a.workspace->cell_count <= a.i_pn2)
    return false;
  const std::uint32_t i_pn3 = *a.workspace->cell_count - 1u;
  const std::uint64_t pn3_logical = a.cells[i_pn3].logical_recipe_id;
  if (pn3_logical == 0u || pn3_logical == a.again_logical ||
      pn3_logical == a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {910u, 960u};
  ResidentRecipeOccurrence phrase{};
  ResidentOccurrenceCoupling world{};
  ResidentRelationalNetworkClosure n_w{};
  DirectWhiteboxCondensationV1 w_w{};
  if (!bind_live(a.cells[i_pn3], a.ders[i_pn3], held, 0x1F32, &phrase, 44u, 44 << 16) ||
      phrase.logical_recipe_id != pn3_logical ||
      !bind_resident_occurrence_causal_intersection_coupling(
          phrase, a.ders[i_pn3], occ_world[0], der_world[0], 314u, &world) ||
      world.reserved2 != 314u)
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[i_pn3], a.cells[a.i_anc]};
  ResidentRecipeDerivation der2[2] = {a.ders[i_pn3], der_world[0]};
  ResidentRecipeOccurrence occ2[2] = {phrase, occ_world[0]};
  if (!bind_resident_relational_network_closure(rec2, der2, occ2, 2u, &world, 1u,
                                               &n_w) ||
      n_w.identity == 0u || n_w.actual_count != 2u ||
      fold_resident_mixed_rank_affine(der2, occ2, &world, 2u, 1u, n_w, &w_w) ||
      a.cells[i_pn3].logical_recipe_id != pn3_logical)
    return false;
  const std::uint32_t other[] = {1310u, 1360u};
  ResidentRecipeOccurrence phrase_b{};
  ResidentOccurrenceCoupling world_b{};
  ResidentRelationalNetworkClosure n_b{};
  if (!bind_live(a.cells[i_pn3], a.ders[i_pn3], other, 0x1F40, &phrase_b, 45u,
                 45 << 16) ||
      phrase_b.logical_recipe_id != pn3_logical ||
      phrase_b.bindings[0].variable_identity != 1310u ||
      !bind_resident_occurrence_causal_intersection_coupling(
          phrase_b, a.ders[i_pn3], occ_world[0], der_world[0], 314u, &world_b) ||
      world_b.reserved2 != 314u)
    return false;
  ResidentRecipeOccurrence occ_b[2] = {phrase_b, occ_world[0]};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_b, 2u, &world_b, 1u,
                                               &n_b) ||
      n_b.identity == 0u || n_b.identity == n_w.identity || n_b.actual_count != 2u)
    return false;
  ResidentRecipeOccurrence dead = phrase;
  dead.state = kResidentRecipeOccurrenceSettled;
  ResidentRecipeOccurrence occ_dead[2] = {dead, occ_world[0]};
  ResidentRelationalNetworkClosure n_dead{}, n_keep{};
  ResidentRecipeOccurrence keep{};
  ResidentOccurrenceCoupling world_keep{};
  if (bind_resident_relational_network_closure(rec2, der2, occ_dead, 2u, &world, 1u,
                                              &n_dead) ||
      a.cells[i_pn3].logical_recipe_id != pn3_logical ||
      !bind_live(a.cells[i_pn3], a.ders[i_pn3], other, 0x1F41, &keep, 46u, 46 << 16) ||
      keep.logical_recipe_id != pn3_logical ||
      !bind_resident_occurrence_causal_intersection_coupling(
          keep, a.ders[i_pn3], occ_world[0], der_world[0], 314u, &world_keep) ||
      world_keep.reserved2 != 314u)
    return false;
  ResidentRecipeOccurrence occ_keep[2] = {keep, occ_world[0]};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_keep, 2u, &world_keep,
                                               1u, &n_keep) ||
      n_keep.identity == 0u || n_keep.actual_count != 2u ||
      n_keep.identity == n_w.identity)
    return false;
  ResidentRecipeOccurrence keep_dead = keep;
  keep_dead.state = kResidentRecipeOccurrenceSettled;
  ResidentRecipeOccurrence occ_keep_dead[2] = {keep_dead, occ_world[0]};
  ResidentRelationalNetworkClosure n_keep_dead{}, n_world{};
  ResidentOccurrenceCoupling edge_world{};
  if (bind_resident_relational_network_closure(rec2, der2, occ_keep_dead, 2u,
                                              &world_keep, 1u, &n_keep_dead) ||
      !bind_resident_occurrence_coupling(occ_world[0], der_world[0], 1u, occ_world[1],
                                        der_world[1], 0u, &edge_world))
    return false;
  ResidentRecipeCell rec_w[2] = {a.cells[a.i_anc], a.cells[a.i_pn2]};
  ResidentRecipeDerivation der_w[2] = {der_world[0], der_world[1]};
  if (!bind_resident_relational_network_closure(rec_w, der_w, occ_world, 2u,
                                               &edge_world, 1u, &n_world) ||
      n_world.identity == 0u || n_world.actual_count != 2u ||
      a.cells[i_pn3].logical_recipe_id != pn3_logical)
    return false;
  const std::uint32_t again_vars[] = {1410u, 1460u};
  ResidentRecipeOccurrence again_phrase{};
  ResidentOccurrenceCoupling again_world{};
  ResidentRelationalNetworkClosure n_again{};
  if (!bind_live(a.cells[i_pn3], a.ders[i_pn3], again_vars, 0x1F50, &again_phrase, 47u,
                 47 << 16) ||
      again_phrase.logical_recipe_id != pn3_logical ||
      !bind_resident_occurrence_causal_intersection_coupling(
          again_phrase, a.ders[i_pn3], occ_world[0], der_world[0], 314u,
          &again_world) ||
      again_world.reserved2 != 314u)
    return false;
  ResidentRecipeOccurrence occ_again[2] = {again_phrase, occ_world[0]};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_again, 2u,
                                               &again_world, 1u, &n_again) ||
      n_again.identity == 0u || n_again.actual_count != 2u ||
      n_again.identity == n_w.identity || n_again.identity == n_keep.identity)
    return false;
  const std::uint32_t imag_vars[] = {1510u, 1560u};
  ResidentRecipeOccurrence imag{};
  ResidentOccurrenceCoupling imag_world{};
  ResidentRelationalNetworkClosure n_imag{};
  DirectWhiteboxCondensationV1 w_imag{};
  if (!bind_endogenous(a.cells[i_pn3], a.ders[i_pn3], imag_vars, 0x1A30, &imag, 48u,
                       48 << 16) ||
      imag.logical_recipe_id != pn3_logical ||
      !bind_resident_occurrence_causal_intersection_coupling(
          imag, a.ders[i_pn3], occ_world[0], der_world[0], 314u, &imag_world) ||
      imag_world.reserved2 != 314u)
    return false;
  ResidentRecipeOccurrence occ_imag[2] = {imag, occ_world[0]};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_imag, 2u, &imag_world,
                                               1u, &n_imag) ||
      n_imag.identity == 0u || n_imag.actual_count != 1u ||
      n_imag.identity == n_again.identity ||
      admit_resident_network_parent(rec2, der2, occ_imag, 2u, &imag_world, 1u,
                                   n_imag.identity, pn3_logical) ||
      fold_resident_mixed_rank_affine(der2, occ_imag, &imag_world, 2u, 1u, n_imag,
                                     &w_imag))
    return false;
  ResidentRecipeCell rec3[3] = {a.cells[i_pn3], a.cells[i_pn3], a.cells[a.i_anc]};
  ResidentRecipeDerivation der3[3] = {a.ders[i_pn3], a.ders[i_pn3], der_world[0]};
  ResidentRecipeOccurrence occ3[3] = {again_phrase, imag, occ_world[0]};
  ResidentOccurrenceCoupling edges2[2] = {again_world, imag_world};
  ResidentRelationalNetworkSet two{};
  ResidentRelationalNetworkClosure n_all{};
  DirectWhiteboxCondensationV1 w_all{};
  return bind_resident_relational_network_set(rec3, der3, occ3, 3u, edges2, 2u, &two) &&
         two.networked_occurrence_count == 3u &&
         bind_resident_relational_network_closure(rec3, der3, occ3, 3u, edges2, 2u,
                                                 &n_all) &&
         n_all.identity != n_again.identity && n_all.identity != n_imag.identity &&
         n_all.actual_count == 2u &&
         !admit_resident_network_parent(rec3, der3, occ3, 3u, edges2, 2u, n_all.identity,
                                       pn3_logical) &&
         !fold_resident_mixed_rank_affine(der3, occ3, edges2, 3u, 2u, n_all, &w_all);
}

static bool language_pn4_mint(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr || a.workspace->state == nullptr ||
      a.workspace->cells != a.cells)
    return false;
  const std::uint32_t i_pn3 = *a.workspace->cell_count - 1u;
  const std::uint64_t pn3_logical = a.cells[i_pn3].logical_recipe_id;
  if (pn3_logical == 0u || pn3_logical == a.again_logical ||
      pn3_logical == a.cells[a.i_pn2].logical_recipe_id)
    return false;
  a.workspace->state->recipe_cell_capacity =
      *a.workspace->cell_count + 1u > a.workspace->state->recipe_cell_capacity
          ? *a.workspace->cell_count + 1u
          : a.workspace->state->recipe_cell_capacity;
  a.workspace->state->derivation_capacity =
      a.workspace->state->derivation_count + 1u > a.workspace->state->derivation_capacity
          ? a.workspace->state->derivation_count + 1u
          : a.workspace->state->derivation_capacity;
  a.workspace->cell_capacity = a.workspace->state->recipe_cell_capacity;
  ResidentRecipeCell rec2[2] = {a.cells[a.i_src], a.cells[i_pn3]};
  ResidentRecipeDerivation der2[2] = {a.ders[a.i_src], a.ders[i_pn3]};
  der2[0].logical_recipe_id = rec2[0].logical_recipe_id;
  der2[0].revision_identity = rec2[0].revision_identity;
  der2[1].logical_recipe_id = rec2[1].logical_recipe_id;
  der2[1].revision_identity = rec2[1].revision_identity;
  const std::uint32_t u0[] = {1610u, 1620u};
  const std::uint32_t u1[] = {1620u, 1630u};
  ResidentRecipeOccurrence occ_u1[2]{}, occ_u2[2]{};
  ResidentOccurrenceCoupling e_u1[1]{}, e_u2[1]{};
  ResidentRelationalNetworkClosure n4{};
  DirectWhiteboxCondensationV1 w4{};
  ResidentNetworkCondensationEvidence ev4{};
  std::uint64_t w4_logical = 0, w4_rev = 0;
  if (!bind_live(rec2[0], der2[0], u0, 0x2210, &occ_u1[0], 49u, 49 << 16) ||
      !bind_live(rec2[1], der2[1], u1, 0x2211, &occ_u1[1], 49u, 49 << 16) ||
      !bind_live(rec2[0], der2[0], u0, 0x2220, &occ_u2[0], 50u, 50 << 16) ||
      !bind_live(rec2[1], der2[1], u1, 0x2221, &occ_u2[1], 50u, 50 << 16) ||
      !bind_resident_occurrence_coupling(occ_u1[0], der2[0], 1u, occ_u1[1], der2[1], 0u,
                                        &e_u1[0]) ||
      !bind_resident_occurrence_coupling(occ_u2[0], der2[0], 1u, occ_u2[1], der2[1], 0u,
                                        &e_u2[0]) ||
      !bind_resident_relational_network_closure(rec2, der2, occ_u1, 2u, e_u1, 1u, &n4) ||
      n4.actual_count != 2u ||
      !observe_resident_mixed_rank_whitebox(rec2, der2, occ_u1, occ_u2, 2u, e_u1, e_u2,
                                           1u, &w4) ||
      !observe_resident_mixed_rank_evidence(rec2, der2, occ_u1, occ_u2, 2u, e_u1, e_u2,
                                           1u, &w4, &ev4) ||
      !replay_resident_network_candidate(a.cells, *a.workspace->cell_count, a.ders,
                                         a.workspace->state->derivation_count, ev4, &w4,
                                         &w4_logical, &w4_rev) ||
      w4_logical == 0u || w4_logical == pn3_logical || w4_logical == a.again_logical)
    return false;
  const ResidentRecipeCell parent = a.cells[i_pn3];
  a.workspace->claimed_identity = n4.identity;
  a.workspace->network_cells = rec2;
  a.workspace->network_ders = der2;
  a.workspace->occurrences = occ_u1;
  a.workspace->couplings = e_u1;
  a.workspace->occurrence_count = 2u;
  a.workspace->coupling_count = 1u;
  a.workspace->parent_cell = i_pn3;
  a.workspace->parent_logical = parent.logical_recipe_id;
  a.workspace->witness = &w4;
  a.workspace->evidence = &ev4;
  const std::uint32_t before = *a.workspace->cell_count;
  ResidentRecipeOccurrence unfold{};
  const std::uint32_t held[] = {1610u, 1630u};
  if (!foundry_workspace_ready(a.workspace) ||
      !foundry_materialize_from_workspace(a.workspace, nullptr) ||
      *a.workspace->cell_count != before + 1u)
    return false;
  const std::uint32_t i_pn4 = *a.workspace->cell_count - 1u;
  return a.cells[i_pn3].logical_recipe_id == parent.logical_recipe_id &&
         a.cells[i_pn4].logical_recipe_id == w4_logical &&
         a.ders[i_pn4].parent_logical_recipe_id == pn3_logical &&
         replay_resident_whitebox_recipe(w4, a.cells[i_pn4], a.ders[i_pn4]) &&
         bind_live(a.cells[i_pn4], a.ders[i_pn4], held, 0x1F60, &unfold) &&
         unfold.logical_recipe_id == w4_logical;
}

static bool language_phrase_compose(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t i_pn4 = *a.workspace->cell_count - 1u;
  const std::uint64_t src_logical = a.cells[a.i_src].logical_recipe_id;
  const std::uint64_t pn1_logical = a.cells[a.i_pn1].logical_recipe_id;
  const std::uint64_t pn2_logical = a.cells[a.i_pn2].logical_recipe_id;
  const std::uint64_t pn4_logical = a.cells[i_pn4].logical_recipe_id;
  if (pn1_logical == 0u || pn2_logical == 0u || pn4_logical == 0u ||
      pn1_logical == src_logical || pn2_logical == src_logical ||
      pn4_logical == pn2_logical || a.i_pn1 == a.i_src || a.i_pn2 == a.i_src ||
      a.ders[a.i_pn1].parameter_count != 2u || a.ders[a.i_pn2].parameter_count != 2u ||
      a.ders[i_pn4].parameter_count != 2u || a.ders[a.i_pn1].relation_count != 1u ||
      a.ders[a.i_pn2].relation_count != 1u || a.ders[i_pn4].relation_count != 1u)
    return false;
  ResidentRecipeCell rec_s[2] = {a.cells[a.i_pn1], a.cells[a.i_pn2]};
  ResidentRecipeDerivation der_s[2] = {a.ders[a.i_pn1], a.ders[a.i_pn2]};
  der_s[0].logical_recipe_id = rec_s[0].logical_recipe_id;
  der_s[0].revision_identity = rec_s[0].revision_identity;
  der_s[1].logical_recipe_id = rec_s[1].logical_recipe_id;
  der_s[1].revision_identity = rec_s[1].revision_identity;
  const std::uint32_t s0[] = {1710u, 1720u};
  const std::uint32_t s1[] = {1720u, 1730u};
  ResidentRecipeOccurrence occ_s1[2]{}, occ_s2[2]{};
  ResidentOccurrenceCoupling e_s1[1]{}, e_s2[1]{};
  ResidentRelationalNetworkClosure n_s{};
  DirectWhiteboxCondensationV1 w_s{};
  ResidentNetworkCondensationEvidence ev_s{};
  std::uint64_t s_logical = 0, s_rev = 0;
  if (!bind_live(rec_s[0], der_s[0], s0, 0x2310, &occ_s1[0], 51u, 51 << 16) ||
      !bind_live(rec_s[1], der_s[1], s1, 0x2311, &occ_s1[1], 51u, 51 << 16) ||
      !bind_live(rec_s[0], der_s[0], s0, 0x2320, &occ_s2[0], 52u, 52 << 16) ||
      !bind_live(rec_s[1], der_s[1], s1, 0x2321, &occ_s2[1], 52u, 52 << 16) ||
      !bind_resident_occurrence_coupling(occ_s1[0], der_s[0], 1u, occ_s1[1], der_s[1], 0u,
                                        &e_s1[0]) ||
      !bind_resident_occurrence_coupling(occ_s2[0], der_s[0], 1u, occ_s2[1], der_s[1], 0u,
                                        &e_s2[0]) ||
      !bind_resident_relational_network_closure(rec_s, der_s, occ_s1, 2u, e_s1, 1u,
                                               &n_s) ||
      n_s.actual_count != 2u ||
      !fold_resident_mixed_rank_affine(der_s, occ_s1, e_s1, 2u, 1u, n_s, &w_s) ||
      resident_whitebox_recipe_logical_identity(w_s) == pn1_logical ||
      resident_whitebox_recipe_logical_identity(w_s) == pn2_logical ||
      resident_whitebox_recipe_logical_identity(w_s) == a.again_logical ||
      !observe_resident_mixed_rank_whitebox(rec_s, der_s, occ_s1, occ_s2, 2u, e_s1,
                                           e_s2, 1u, &w_s) ||
      !observe_resident_mixed_rank_evidence(rec_s, der_s, occ_s1, occ_s2, 2u, e_s1,
                                           e_s2, 1u, &w_s, &ev_s) ||
      !replay_resident_network_candidate(a.cells, *a.workspace->cell_count, a.ders,
                                         a.workspace->state->derivation_count, ev_s,
                                         &w_s, &s_logical, &s_rev) ||
      s_logical == 0u || s_logical == pn1_logical || s_logical == pn2_logical ||
      s_logical == pn4_logical || s_logical == a.again_logical)
    return false;
  a.workspace->state->recipe_cell_capacity =
      *a.workspace->cell_count + 1u > a.workspace->state->recipe_cell_capacity
          ? *a.workspace->cell_count + 1u
          : a.workspace->state->recipe_cell_capacity;
  a.workspace->state->derivation_capacity =
      a.workspace->state->derivation_count + 1u > a.workspace->state->derivation_capacity
          ? a.workspace->state->derivation_count + 1u
          : a.workspace->state->derivation_capacity;
  a.workspace->cell_capacity = a.workspace->state->recipe_cell_capacity;
  const ResidentRecipeCell parent = a.cells[a.i_pn2];
  a.workspace->claimed_identity = n_s.identity;
  a.workspace->network_cells = rec_s;
  a.workspace->network_ders = der_s;
  a.workspace->occurrences = occ_s1;
  a.workspace->couplings = e_s1;
  a.workspace->occurrence_count = 2u;
  a.workspace->coupling_count = 1u;
  a.workspace->parent_cell = a.i_pn2;
  a.workspace->parent_logical = parent.logical_recipe_id;
  a.workspace->witness = &w_s;
  a.workspace->evidence = &ev_s;
  const std::uint32_t before = *a.workspace->cell_count;
  ResidentRecipeOccurrence unfold{};
  const std::uint32_t held[] = {1710u, 1730u};
  if (!foundry_workspace_ready(a.workspace) ||
      !foundry_materialize_from_workspace(a.workspace, nullptr) ||
      *a.workspace->cell_count != before + 1u)
    return false;
  const std::uint32_t i_phrase = *a.workspace->cell_count - 1u;
  if (a.cells[a.i_pn2].logical_recipe_id != parent.logical_recipe_id ||
      a.cells[i_phrase].logical_recipe_id != s_logical ||
      a.ders[i_phrase].parent_logical_recipe_id != pn2_logical ||
      !replay_resident_whitebox_recipe(w_s, a.cells[i_phrase], a.ders[i_phrase]) ||
      !bind_live(a.cells[i_phrase], a.ders[i_phrase], held, 0x1F70, &unfold) ||
      unfold.logical_recipe_id != s_logical)
    return false;
  ResidentRecipeCell rec_d[2] = {a.cells[a.i_pn2], a.cells[i_pn4]};
  ResidentRecipeDerivation der_d[2] = {a.ders[a.i_pn2], a.ders[i_pn4]};
  der_d[0].logical_recipe_id = rec_d[0].logical_recipe_id;
  der_d[0].revision_identity = rec_d[0].revision_identity;
  der_d[1].logical_recipe_id = rec_d[1].logical_recipe_id;
  der_d[1].revision_identity = rec_d[1].revision_identity;
  const std::uint32_t d0[] = {1750u, 1760u};
  const std::uint32_t d1[] = {1760u, 1770u};
  ResidentRecipeOccurrence occ_d1[2]{}, occ_d2[2]{};
  ResidentOccurrenceCoupling e_d1[1]{}, e_d2[1]{};
  ResidentRelationalNetworkClosure n_d{};
  DirectWhiteboxCondensationV1 w_d{};
  return bind_live(rec_d[0], der_d[0], d0, 0x2410, &occ_d1[0], 53u, 53 << 16) &&
         bind_live(rec_d[1], der_d[1], d1, 0x2411, &occ_d1[1], 53u, 53 << 16) &&
         bind_live(rec_d[0], der_d[0], d0, 0x2420, &occ_d2[0], 54u, 54 << 16) &&
         bind_live(rec_d[1], der_d[1], d1, 0x2421, &occ_d2[1], 54u, 54 << 16) &&
         bind_resident_occurrence_coupling(occ_d1[0], der_d[0], 1u, occ_d1[1], der_d[1],
                                          0u, &e_d1[0]) &&
         bind_resident_occurrence_coupling(occ_d2[0], der_d[0], 1u, occ_d2[1], der_d[1],
                                          0u, &e_d2[0]) &&
         bind_resident_relational_network_closure(rec_d, der_d, occ_d1, 2u, e_d1, 1u,
                                                 &n_d) &&
         n_d.actual_count == 2u &&
         !fold_resident_mixed_rank_affine(der_d, occ_d1, e_d1, 2u, 1u, n_d, &w_d) &&
         !observe_resident_mixed_rank_whitebox(rec_d, der_d, occ_d1, occ_d2, 2u, e_d1,
                                              e_d2, 1u, &w_d);
}

static bool language_phrase_reuse(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t i_phrase = *a.workspace->cell_count - 1u;
  const std::uint64_t phrase_logical = a.cells[i_phrase].logical_recipe_id;
  const std::uint64_t src_logical = a.cells[a.i_src].logical_recipe_id;
  const std::uint64_t pn2_logical = a.cells[a.i_pn2].logical_recipe_id;
  if (phrase_logical == 0u || phrase_logical == src_logical ||
      phrase_logical == pn2_logical || phrase_logical == a.again_logical ||
      a.ders[i_phrase].parent_logical_recipe_id != pn2_logical ||
      a.ders[i_phrase].parameter_count != 2u || a.ders[a.i_src].parameter_count != 2u ||
      a.ders[i_phrase].relation_count != 1u || a.ders[a.i_src].relation_count != 1u)
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[a.i_src], a.cells[i_phrase]};
  ResidentRecipeDerivation der2[2] = {a.ders[a.i_src], a.ders[i_phrase]};
  der2[0].logical_recipe_id = rec2[0].logical_recipe_id;
  der2[0].revision_identity = rec2[0].revision_identity;
  der2[1].logical_recipe_id = rec2[1].logical_recipe_id;
  der2[1].revision_identity = rec2[1].revision_identity;
  const std::uint32_t r0[] = {1850u, 1860u};
  const std::uint32_t r1[] = {1860u, 1870u};
  ResidentRecipeOccurrence occ_a[2]{}, occ_b[2]{};
  ResidentOccurrenceCoupling e_a[1]{}, e_b[1]{};
  ResidentRelationalNetworkClosure n_r{};
  DirectWhiteboxCondensationV1 w_r{};
  ResidentNetworkCondensationEvidence ev_r{};
  std::uint64_t r_logical = 0, r_rev = 0;
  if (!bind_live(rec2[0], der2[0], r0, 0x2510, &occ_a[0], 56u, 56 << 16) ||
      !bind_live(rec2[1], der2[1], r1, 0x2511, &occ_a[1], 56u, 56 << 16) ||
      !bind_live(rec2[0], der2[0], r0, 0x2520, &occ_b[0], 57u, 57 << 16) ||
      !bind_live(rec2[1], der2[1], r1, 0x2521, &occ_b[1], 57u, 57 << 16) ||
      !bind_resident_occurrence_coupling(occ_a[0], der2[0], 1u, occ_a[1], der2[1], 0u,
                                        &e_a[0]) ||
      !bind_resident_occurrence_coupling(occ_b[0], der2[0], 1u, occ_b[1], der2[1], 0u,
                                        &e_b[0]) ||
      !bind_resident_relational_network_closure(rec2, der2, occ_a, 2u, e_a, 1u, &n_r) ||
      n_r.actual_count != 2u)
    return false;
  const bool folded =
      fold_resident_mixed_rank_affine(der2, occ_a, e_a, 2u, 1u, n_r, &w_r);
  const bool observed = observe_resident_mixed_rank_whitebox(
      rec2, der2, occ_a, occ_b, 2u, e_a, e_b, 1u, &w_r);
  if (folded != observed) return false;
  if (!folded) return true;
  return observe_resident_mixed_rank_evidence(rec2, der2, occ_a, occ_b, 2u, e_a, e_b,
                                             1u, &w_r, &ev_r) &&
         replay_resident_network_candidate(a.cells, *a.workspace->cell_count, a.ders,
                                           a.workspace->state->derivation_count, ev_r,
                                           &w_r, &r_logical, &r_rev) &&
         r_logical != 0u && r_logical != phrase_logical && r_logical != src_logical &&
         r_logical != pn2_logical && r_logical != a.again_logical;
}

static bool language_phrase_opportunity(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t i_phrase = *a.workspace->cell_count - 1u;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint64_t phrase_logical = a.cells[i_phrase].logical_recipe_id;
  if (phrase_logical == 0u || phrase_logical == a.cells[a.i_pn2].logical_recipe_id ||
      a.ders[i_phrase].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id ||
      live_cells <= a.cell_count || i_phrase >= live_cells)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence expressed{};
  if (!bind_live(a.cells[i_phrase], a.ders[i_phrase], held, 0xE170, &expressed, 58u,
                 58 << 16) ||
      expressed.logical_recipe_id != phrase_logical)
    return false;
  const std::uint32_t learned_surface[] = {0x70687261u, 0x73656d72u, 0x67656e74u};
  const std::uint32_t wrong_surface[] = {0x70687261u, 0x73656d72u, 0x67656e74u + 1u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(8u, 3u, learned_surface);
  const std::uint64_t wrong =
      substrate::direct_network::surface_ecology_payload_identity(8u, 3u, wrong_surface);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityV1 receipt{};
  ResidentRecipeOccurrence candidate{};
  if (surface == 0u || wrong == 0u || surface == wrong ||
      !substrate::direct_adult_core::earn_resident_language_recipe_opportunity(
          surface, expressed, &receipt) ||
      receipt.logical_recipe_id != phrase_logical ||
      receipt.revision_identity != a.cells[i_phrase].revision_identity ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          receipt, surface, a.cells, live_cells, a.ders, live_ders, 0xE171,
          &candidate) ||
      candidate.logical_recipe_id != phrase_logical ||
      candidate.lineage_kind != ResidentOccurrenceLineageKind::endogenous ||
      candidate.authority != DirectParticipationAuthority::none ||
      candidate.eligibility_q16 != 0 ||
      substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          receipt, wrong, a.cells, live_cells, a.ders, live_ders, 0xE172,
          &candidate) ||
      substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          receipt, surface, a.cells, a.cell_count, a.ders, a.der_count, 0xE173,
          &candidate))
    return false;
  return true;
}

static bool language_phrase_two_surfaces(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t i_phrase = *a.workspace->cell_count - 1u;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint64_t phrase_logical = a.cells[i_phrase].logical_recipe_id;
  if (phrase_logical == 0u ||
      a.ders[i_phrase].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence express_a{};
  ResidentRecipeOccurrence express_b{};
  if (!bind_live(a.cells[i_phrase], a.ders[i_phrase], held, 0xE180, &express_a, 59u,
                 59 << 16) ||
      !bind_live(a.cells[i_phrase], a.ders[i_phrase], held, 0xE181, &express_b, 60u,
                 60 << 16) ||
      express_a.logical_recipe_id != phrase_logical ||
      express_b.logical_recipe_id != phrase_logical)
    return false;
  const std::uint32_t surface_a_bytes[] = {0x64657574u, 0x73636831u, 0x61626364u};
  const std::uint32_t surface_b_bytes[] = {0x7a686f75u, 0x73636832u, 0x7768797au};
  const std::uint64_t surface_a =
      substrate::direct_network::surface_ecology_payload_identity(8u, 3u, surface_a_bytes);
  const std::uint64_t surface_b =
      substrate::direct_network::surface_ecology_payload_identity(9u, 3u, surface_b_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityV1 rec_a{};
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityV1 rec_b{};
  ResidentRecipeOccurrence nom_a{};
  ResidentRecipeOccurrence nom_b{};
  if (surface_a == 0u || surface_b == 0u || surface_a == surface_b ||
      !substrate::direct_adult_core::earn_resident_language_recipe_opportunity(
          surface_a, express_a, &rec_a) ||
      rec_a.logical_recipe_id != phrase_logical ||
      substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          rec_a, surface_b, a.cells, live_cells, a.ders, live_ders, 0xE182, &nom_a) ||
      !substrate::direct_adult_core::earn_resident_language_recipe_opportunity(
          surface_b, express_b, &rec_b) ||
      rec_b.logical_recipe_id != phrase_logical ||
      rec_b.surface_identity == rec_a.surface_identity ||
      rec_b.logical_recipe_id != rec_a.logical_recipe_id ||
      rec_b.revision_identity != rec_a.revision_identity ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          rec_a, surface_a, a.cells, live_cells, a.ders, live_ders, 0xE183, &nom_a) ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          rec_b, surface_b, a.cells, live_cells, a.ders, live_ders, 0xE184, &nom_b) ||
      nom_a.logical_recipe_id != phrase_logical ||
      nom_b.logical_recipe_id != phrase_logical ||
      nom_a.logical_recipe_id != nom_b.logical_recipe_id ||
      substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          rec_a, surface_b, a.cells, live_cells, a.ders, live_ders, 0xE185, &nom_a) ||
      substrate::direct_adult_core::nominate_resident_language_recipe_opportunity(
          rec_b, surface_a, a.cells, live_cells, a.ders, live_ders, 0xE186, &nom_b))
    return false;
  return true;
}
