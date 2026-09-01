static bool language_phrase_revise_authority(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  if (phrase_logical == 0u ||
      a.ders[i_last].revision_identity != a.cells[i_last].revision_identity ||
      resident_recipe_current_revision_authority(a.cells[i_last]) !=
          ResidentRecipeRevisionAuthority::experience)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  const std::uint32_t heard_bytes[] = {0x61757468u, 0x65787031u, 0x73726631u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(25u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence expressed{};
  ResidentRecipeOccurrence nom{};
  if (surface == 0u ||
      !bind_live(a.cells[i_last], a.ders[i_last], held, 0xE250, &expressed, 94u,
                 94 << 16) ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE251, &nom) ||
      nom.logical_recipe_id != phrase_logical ||
      nom.authority != DirectParticipationAuthority::none ||
      nom.lineage_kind != ResidentOccurrenceLineageKind::endogenous)
    return false;
  return resident_recipe_current_revision_authority(a.cells[i_last]) ==
         ResidentRecipeRevisionAuthority::experience;
}
