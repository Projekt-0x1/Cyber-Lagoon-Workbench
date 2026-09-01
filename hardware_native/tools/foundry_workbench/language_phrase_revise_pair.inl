static bool language_phrase_revise_pair(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  const std::uint64_t der_rev = a.ders[i_last].revision_identity;
  if (phrase_logical == 0u ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  if (a.cells[i_last].revision_identity == der_rev) {
    const std::uint32_t map_vars[] = {3410u, 3420u};
    ResidentRecipeOccurrence occ_rev{};
    if (!bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE240, &occ_rev, 90u,
                   90 << 16) ||
        !commit_resident_causal_difference_revision(&occ_rev, &a.cells[i_last], i_last,
                                                    true, 1, 0))
      return false;
  }
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence before{};
  if (a.cells[i_last].revision_identity == a.ders[i_last].revision_identity ||
      !bind_live(a.cells[i_last], a.ders[i_last], held, 0xE241, &before, 91u,
                 91 << 16) ||
      resident_relational_network_occurrence_current(a.cells[i_last], a.ders[i_last],
                                                     before))
    return false;
  a.ders[i_last].revision_identity = a.cells[i_last].revision_identity;
  ResidentRecipeOccurrence after{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE242, &after, 92u,
                 92 << 16) ||
      !resident_relational_network_occurrence_current(a.cells[i_last], a.ders[i_last],
                                                      after))
    return false;
  const std::uint32_t heard_bytes[] = {0x70616972u, 0x72656d61u, 0x73726631u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(24u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence nom{};
  const std::uint32_t src_vars[] = {3430u, 3440u};
  const std::uint32_t world_vars[] = {3450u, 3460u};
  ResidentRecipeOccurrence occ_src{}, occ_map{};
  ResidentOccurrenceCoupling world{};
  ResidentRelationalNetworkClosure n_w{};
  if (surface == 0u ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, after) ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE243, &nom) ||
      nom.logical_recipe_id != phrase_logical ||
      !bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE244, &occ_src, 93u,
                 93 << 16) ||
      !bind_live(a.cells[i_last], a.ders[i_last], world_vars, 0xE245, &occ_map, 93u,
                 93 << 16) ||
      !bind_resident_occurrence_causal_intersection_coupling(
          occ_map, a.ders[i_last], occ_src, a.ders[a.i_src], 323u, &world))
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[i_last], a.cells[a.i_src]};
  ResidentRecipeDerivation der2[2] = {a.ders[i_last], a.ders[a.i_src]};
  ResidentRecipeOccurrence occ2[2] = {occ_map, occ_src};
  return bind_resident_relational_network_closure(rec2, der2, occ2, 2u, &world, 1u,
                                                 &n_w) &&
         n_w.identity != 0u && n_w.actual_count == 2u &&
         *a.workspace->cell_count == live_cells;
}
