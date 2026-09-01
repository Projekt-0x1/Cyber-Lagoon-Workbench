static bool language_phrase_world_two(const LanguageAfterPn2& a) {
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
  ResidentRecipeOccurrence express_a{};
  ResidentRecipeOccurrence express_b{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1C0, &express_a, 69u,
                 69 << 16) ||
      !bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1C1, &express_b, 70u,
                 70 << 16))
    return false;
  const std::uint32_t bytes_a[] = {0x74776f31u, 0x73726661u, 0x61616161u};
  const std::uint32_t bytes_b[] = {0x74776f32u, 0x73726662u, 0x62626262u};
  const std::uint64_t surface_a =
      substrate::direct_network::surface_ecology_payload_identity(15u, 3u, bytes_a);
  const std::uint64_t surface_b =
      substrate::direct_network::surface_ecology_payload_identity(16u, 3u, bytes_b);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  if (surface_a == 0u || surface_b == 0u || surface_a == surface_b ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface_a, express_a) ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface_b, express_b) ||
      bank.count != 2u)
    return false;
  express_a = {};
  express_b = {};
  ResidentRecipeOccurrence nom_a{};
  ResidentRecipeOccurrence nom_b{};
  if (!substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface_a, a.cells, live_cells, a.ders, live_ders, 0xE1C2,
          &nom_a) ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface_b, a.cells, live_cells, a.ders, live_ders, 0xE1C3,
          &nom_b) ||
      nom_a.logical_recipe_id != phrase_logical ||
      nom_b.logical_recipe_id != phrase_logical ||
      nom_a.logical_recipe_id != nom_b.logical_recipe_id ||
      nom_a.occurrence_identity == nom_b.occurrence_identity)
    return false;
  std::uint32_t i_map = live_cells;
  for (std::uint32_t i = 0u; i < live_cells; ++i) {
    if (a.cells[i].logical_recipe_id != nom_a.logical_recipe_id ||
        a.cells[i].revision_identity != nom_a.revision_identity)
      continue;
    if (i_map != live_cells) return false;
    i_map = i;
  }
  if (i_map >= live_cells) return false;
  const std::uint32_t src_vars[] = {2810u, 2820u};
  const std::uint32_t map_vars[] = {2830u, 2840u};
  ResidentRecipeOccurrence src_a{}, map_a{}, src_b{}, map_b{};
  ResidentOccurrenceCoupling world_a{}, world_b{};
  ResidentRelationalNetworkClosure n_a{}, n_b{};
  if (!bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE1C4, &src_a, 71u,
                 71 << 16) ||
      !bind_live(a.cells[i_map], a.ders[i_map], map_vars, 0xE1C5, &map_a, 71u,
                 71 << 16) ||
      !bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE1C6, &src_b, 72u,
                 72 << 16) ||
      !bind_live(a.cells[i_map], a.ders[i_map], map_vars, 0xE1C7, &map_b, 72u,
                 72 << 16) ||
      !bind_resident_occurrence_causal_intersection_coupling(
          map_a, a.ders[i_map], src_a, a.ders[a.i_src], 316u, &world_a) ||
      !bind_resident_occurrence_causal_intersection_coupling(
          map_b, a.ders[i_map], src_b, a.ders[a.i_src], 317u, &world_b))
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[i_map], a.cells[a.i_src]};
  ResidentRecipeDerivation der2[2] = {a.ders[i_map], a.ders[a.i_src]};
  ResidentRecipeOccurrence occ_a[2] = {map_a, src_a};
  ResidentRecipeOccurrence occ_b[2] = {map_b, src_b};
  if (!bind_resident_relational_network_closure(rec2, der2, occ_a, 2u, &world_a, 1u,
                                               &n_a) ||
      !bind_resident_relational_network_closure(rec2, der2, occ_b, 2u, &world_b, 1u,
                                               &n_b) ||
      n_a.identity == 0u || n_b.identity == 0u || n_a.identity == n_b.identity ||
      n_a.actual_count != 2u || n_b.actual_count != 2u)
    return false;
  const std::uint64_t rid_a = resident_relational_network_recruitment_identity(n_a);
  const std::uint64_t rid_b = resident_relational_network_recruitment_identity(n_b);
  bool saw_map_a = false, saw_map_b = false;
  for (std::uint16_t i = 0u; i < n_a.occurrence_count; ++i) {
    if (n_a.members[i].logical_recipe_id == phrase_logical) saw_map_a = true;
    if (n_a.members[i].occurrence_identity == nom_a.occurrence_identity ||
        n_a.members[i].occurrence_identity == 0xE1C0)
      return false;
  }
  for (std::uint16_t i = 0u; i < n_b.occurrence_count; ++i) {
    if (n_b.members[i].logical_recipe_id == phrase_logical) saw_map_b = true;
    if (n_b.members[i].occurrence_identity == nom_b.occurrence_identity ||
        n_b.members[i].occurrence_identity == 0xE1C1)
      return false;
  }
  return saw_map_a && saw_map_b && rid_a != 0u && rid_b != 0u && rid_a != rid_b &&
         a.cells[i_map].logical_recipe_id == phrase_logical;
}
