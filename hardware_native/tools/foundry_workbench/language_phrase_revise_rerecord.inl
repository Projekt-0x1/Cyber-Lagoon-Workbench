static bool language_phrase_revise_rerecord(const LanguageAfterPn2& a) {
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
  const std::uint32_t held[] = {1710u, 1730u};
  if (a.cells[i_last].revision_identity == der_rev) {
    ResidentRecipeOccurrence occ_map{};
    const std::uint32_t map_vars[] = {3210u, 3220u};
    if (!bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE222, &occ_map, 85u,
                   85 << 16) ||
        !commit_resident_causal_difference_revision(&occ_map, &a.cells[i_last], i_last,
                                                    true, 1, 0))
      return false;
  }
  if (a.cells[i_last].revision_identity == der_rev ||
      a.ders[i_last].revision_identity != der_rev)
    return false;
  ResidentRecipeOccurrence again{};
  ResidentRecipeOccurrence nom_again{};
  const std::uint32_t heard_bytes[] = {0x72726364u, 0x73726661u, 0x64657231u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(22u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  if (surface == 0u ||
      !bind_live(a.cells[i_last], a.ders[i_last], held, 0xE223, &again, 86u,
                 86 << 16) ||
      again.revision_identity != a.cells[i_last].revision_identity ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, again) ||
      bank.count != 1u ||
      substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE224,
          &nom_again))
    return false;
  return *a.workspace->cell_count == live_cells &&
         a.cells[i_last].logical_recipe_id == phrase_logical;
}
