static bool language_phrase_world_lesion(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  if (phrase_logical == 0u ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence expressed{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1D0, &expressed, 75u,
                 75 << 16))
    return false;
  const std::uint32_t heard_bytes[] = {0x6c657331u, 0x6d617035u, 0x72656331u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(17u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  if (surface == 0u ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      bank.count != 1u)
    return false;
  expressed = {};
  ResidentRecipeOccurrence nom{};
  if (!substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1D1, &nom) ||
      nom.logical_recipe_id != phrase_logical)
    return false;
  const std::uint32_t src_vars[] = {2910u, 2920u};
  const std::uint32_t map_vars[] = {2930u, 2940u};
  ResidentRecipeOccurrence src_a{}, map_a{}, src_b{}, map_b{};
  ResidentOccurrenceCoupling world_a{}, world_b{};
  ResidentRelationalNetworkClosure n_a{}, n_dead{}, n_b{};
  if (!bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE1D2, &src_a, 76u,
                 76 << 16) ||
      !bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE1D3, &map_a, 76u,
                 76 << 16) ||
      !bind_resident_occurrence_causal_intersection_coupling(
          map_a, a.ders[i_last], src_a, a.ders[a.i_src], 318u, &world_a))
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[i_last], a.cells[a.i_src]};
  ResidentRecipeDerivation der2[2] = {a.ders[i_last], a.ders[a.i_src]};
  ResidentRecipeOccurrence occ_a[2] = {map_a, src_a};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_a, 2u, &world_a, 1u,
                                               &n_a) ||
      n_a.identity == 0u)
    return false;
  ResidentRecipeOccurrence settled = map_a;
  settled.state = kResidentRecipeOccurrenceSettled;
  ResidentRecipeOccurrence occ_dead[2] = {settled, src_a};
  if (bind_resident_relational_network_closure(rec2, der2, occ_dead, 2u, &world_a,
                                              1u, &n_dead) ||
      n_dead.identity != 0u)
    return false;
  if (!bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE1D4, &src_b, 77u,
                 77 << 16) ||
      !bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE1D5, &map_b, 77u,
                 77 << 16) ||
      !bind_resident_occurrence_causal_intersection_coupling(
          map_b, a.ders[i_last], src_b, a.ders[a.i_src], 319u, &world_b))
    return false;
  ResidentRecipeOccurrence occ_b[2] = {map_b, src_b};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_b, 2u, &world_b, 1u,
                                               &n_b) ||
      n_b.identity == 0u || n_b.identity == n_a.identity || n_b.actual_count != 2u)
    return false;
  ResidentRecipeOccurrence nom_after{};
  return a.cells[i_last].logical_recipe_id == phrase_logical &&
         substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
             bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1D6,
             &nom_after) &&
         nom_after.logical_recipe_id == phrase_logical &&
         nom_after.occurrence_identity != map_a.occurrence_identity &&
         nom_after.occurrence_identity != map_b.occurrence_identity;
}
