static bool language_phrase_bank_ambiguous(const LanguageAfterPn2& a) {
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
  const std::uint32_t bind_a[] = {1710u, 1730u};
  const std::uint32_t bind_b[] = {2010u, 2030u};
  ResidentRecipeOccurrence express_a{};
  ResidentRecipeOccurrence express_b{};
  ResidentRecipeOccurrence express_u{};
  if (!bind_live(a.cells[i_phrase], a.ders[i_phrase], bind_a, 0xE1A0, &express_a, 63u,
                 63 << 16) ||
      !bind_live(a.cells[i_phrase], a.ders[i_phrase], bind_b, 0xE1A1, &express_b, 64u,
                 64 << 16) ||
      !bind_live(a.cells[i_phrase], a.ders[i_phrase], bind_a, 0xE1A2, &express_u, 65u,
                 65 << 16) ||
      express_a.logical_recipe_id != phrase_logical ||
      express_b.bindings[0].variable_identity == express_a.bindings[0].variable_identity)
    return false;
  const std::uint32_t collide_bytes[] = {0x616d6231u, 0x62696e64u, 0x636f6c6cu};
  const std::uint32_t unique_bytes[] = {0x616d6231u, 0x756e6971u, 0x726f7731u};
  const std::uint64_t collide =
      substrate::direct_network::surface_ecology_payload_identity(12u, 3u, collide_bytes);
  const std::uint64_t unique =
      substrate::direct_network::surface_ecology_payload_identity(13u, 3u, unique_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence nom{};
  if (collide == 0u || unique == 0u || collide == unique ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, collide, express_a) ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, collide, express_b) ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, unique, express_u) ||
      bank.count != 3u || bank.refusals != 0u ||
      bank.rows[0].surface_identity != bank.rows[1].surface_identity ||
      bank.rows[0].identity == bank.rows[1].identity ||
      bank.rows[0].logical_recipe_id != phrase_logical ||
      bank.rows[2].surface_identity != unique ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, unique, a.cells, live_cells, a.ders, live_ders, 0xE1A3, &nom) ||
      nom.logical_recipe_id != phrase_logical ||
      substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, collide, a.cells, live_cells, a.ders, live_ders, 0xE1A4, &nom))
    return false;
  return bank.count == 3u;
}
