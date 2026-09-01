static bool language_phrase_bank_relearn(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  const std::uint64_t phrase_rev = a.cells[i_last].revision_identity;
  if (phrase_logical == 0u ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence first{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1F0, &first, 80u,
                 80 << 16))
    return false;
  const std::uint32_t heard_bytes[] = {0x726c726eu, 0x776f7264u, 0x70617468u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(19u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence nom{};
  if (surface == 0u ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, first) ||
      bank.count != 1u ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1F1, &nom) ||
      nom.logical_recipe_id != phrase_logical)
    return false;
  bank.count = 0u;
  if (substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1F2, &nom))
    return false;
  ResidentRecipeOccurrence again{};
  ResidentRecipeOccurrence nom_again{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1F3, &again, 81u,
                 81 << 16) ||
      again.occurrence_identity == first.occurrence_identity ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, again) ||
      bank.count != 1u ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1F4,
          &nom_again) ||
      nom_again.logical_recipe_id != phrase_logical ||
      nom_again.revision_identity != phrase_rev)
    return false;
  return *a.workspace->cell_count == live_cells &&
         a.workspace->state->derivation_count == live_ders &&
         a.cells[i_last].logical_recipe_id == phrase_logical &&
         a.cells[i_last].revision_identity == phrase_rev;
}
