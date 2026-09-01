static bool language_phrase_bank(const LanguageAfterPn2& a) {
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
  ResidentRecipeOccurrence expressed{};
  if (!bind_live(a.cells[i_phrase], a.ders[i_phrase], held, 0xE190, &expressed, 61u,
                 61 << 16) ||
      expressed.logical_recipe_id != phrase_logical)
    return false;
  const std::uint32_t heard_bytes[] = {0x6b6e6f77u, 0x6d617031u, 0x61626364u};
  const std::uint32_t other_bytes[] = {0x6b6e6f77u, 0x6d617032u, 0x65666768u};
  const std::uint32_t unheard_bytes[] = {0x6b6e6f77u, 0x6d617033u, 0x696a6b6cu};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(10u, 3u, heard_bytes);
  const std::uint64_t surface_b =
      substrate::direct_network::surface_ecology_payload_identity(11u, 3u, other_bytes);
  const std::uint64_t unheard =
      substrate::direct_network::surface_ecology_payload_identity(10u, 3u, unheard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  if (surface == 0u || surface_b == 0u || unheard == 0u || surface == surface_b ||
      surface == unheard ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      bank.count != 1u || bank.refusals != 0u ||
      bank.rows[0].logical_recipe_id != phrase_logical ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      bank.count != 1u)
    return false;
  expressed = {};
  ResidentRecipeOccurrence nom{};
  if (expressed.logical_recipe_id != 0u || expressed.occurrence_identity != 0u ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE191, &nom) ||
      nom.logical_recipe_id != phrase_logical ||
      nom.lineage_kind != ResidentOccurrenceLineageKind::endogenous ||
      nom.authority != DirectParticipationAuthority::none ||
      nom.eligibility_q16 != 0 ||
      substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, unheard, a.cells, live_cells, a.ders, live_ders, 0xE192, &nom))
    return false;
  ResidentRecipeOccurrence express_b{};
  if (!bind_live(a.cells[i_phrase], a.ders[i_phrase], held, 0xE193, &express_b, 62u,
                 62 << 16) ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface_b, express_b) ||
      bank.count != 2u ||
      bank.rows[1].logical_recipe_id != phrase_logical ||
      bank.rows[1].surface_identity == bank.rows[0].surface_identity)
    return false;
  express_b = {};
  ResidentRecipeOccurrence nom_b{};
  if (express_b.logical_recipe_id != 0u ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface_b, a.cells, live_cells, a.ders, live_ders, 0xE194,
          &nom_b) ||
      nom_b.logical_recipe_id != phrase_logical ||
      nom_b.logical_recipe_id != nom.logical_recipe_id ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE195, &nom) ||
      nom.logical_recipe_id != phrase_logical)
    return false;
  const std::uint32_t count_before = bank.count;
  const std::uint32_t refusals_before = bank.refusals;
  if (substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, nom) ||
      substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      bank.count != count_before ||
      bank.refusals != refusals_before + 2u)
    return false;
  ResidentRecipeOccurrence nom_after{};
  return substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
             bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE196,
             &nom_after) &&
         nom_after.logical_recipe_id == phrase_logical &&
         nom_after.lineage_kind == ResidentOccurrenceLineageKind::endogenous &&
         nom_after.authority == DirectParticipationAuthority::none;
}
