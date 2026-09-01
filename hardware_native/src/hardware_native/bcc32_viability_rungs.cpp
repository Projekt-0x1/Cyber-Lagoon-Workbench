#include <algorithm>
#include <string>
#include <vector>

#include "bcc32_g1_blueprint.hpp"
#include "bcc32_genesis.hpp"
#include "bcc32_law_identity.hpp"
#include "bcc32_reference.hpp"
#include "bcc32_types.cuh"
#include "bcc32_viability.hpp"

namespace substrate::bcc32::viability {

bool seal_motif(const G1Motif& motif, const std::string& instance,
                std::vector<ReferenceSite>& support, std::vector<Port>& ports,
                std::string* error) {
  G1BlueprintBuilder builder(instance + "-viability");
  if (!builder.place(motif, instance, G1AuthoredRole::viability)) {
    if (error != nullptr) {
      *error = "motif failed to place: " + std::string(builder.error());
    }
    return false;
  }
  const auto sealed = builder.seal(canonical_law_identity());
  if (!sealed.has_value()) {
    if (error != nullptr) {
      *error = "blueprint failed to seal: " + std::string(builder.error());
    }
    return false;
  }
  const auto capsule = open_g1_capsule(sealed->capsule, canonical_law_identity());
  if (!capsule.has_value()) {
    if (error != nullptr) {
      *error = "sealed capsule failed to reopen: " + std::string(builder.error());
    }
    return false;
  }
  const Genesis genesis = compile_g1(*capsule);

  support.clear();
  for (const SitePlacement& site : genesis.sites) {
    support.push_back(
        {Z3Coordinate{site.coordinate.x, site.coordinate.y, site.coordinate.z}, site.word});
  }

  ports.clear();
  for (const G1PlacedRegion& region : builder.regions()) {
    for (const G1PhysicalPort& port : region.ports) {
      if (port.kind == G1PortKind::observed_output) {
        ports.push_back(Port{Z3Coordinate{port.coordinate.x, port.coordinate.y,
                                          port.coordinate.z},
                             port.observed_mask, port.tick});
      }
    }
  }
  return true;
}

namespace {

std::uint32_t max_tick(const std::vector<Port>& ports) {
  std::uint32_t horizon = 0u;
  for (const Port& port : ports) {
    horizon = std::max(horizon, port.tick);
  }
  return horizon;
}

}  // namespace

RungCatalog build_rungs() {
  RungCatalog catalog;

  // Rungs 1 and 2 declare no ports; their witness is transcribed from the
  // bond-restore assertions committed in bcc32_g1_blueprint_contract.cpp.
  const struct {
    G1Motif motif;
    std::string id;
    std::uint32_t horizon;
  } transcribed[] = {
      {g1_period_two_bond_orbit_motif(), "period_two_bond_orbit", 2u},
      {g1_flux_powered_period_four_boundary_motif(), "flux_powered_period_four_boundary", 4u}};

  for (const auto& item : transcribed) {
    std::vector<ReferenceSite> support;
    std::vector<Port> declared;
    std::string error;
    if (!seal_motif(item.motif, item.id, support, declared, &error)) {
      catalog.skipped.push_back({item.id, error});
      continue;
    }
    Rung rung;
    rung.id = item.id;
    rung.seed = support;
    rung.ports = transcribe_bond_restore_ports(support, item.horizon);
    rung.horizon = item.horizon;
    catalog.rungs.push_back(std::move(rung));
  }

  // Rungs 3 and 4 declare their own ports. payload > 3 returns nullopt.
  for (std::uint32_t payload = 0u; payload <= 3u; ++payload) {
    const std::string id = "cpair_port_relay_" + std::to_string(payload);
    const auto relay = g1_cpair_port_relay_motif(payload);
    if (!relay.has_value()) {
      catalog.skipped.push_back({id, "motif factory returned nullopt"});
      continue;
    }
    std::vector<ReferenceSite> support;
    std::vector<Port> declared;
    std::string error;
    if (!seal_motif(*relay, relay->id, support, declared, &error)) {
      catalog.skipped.push_back({relay->id, error});
      continue;
    }
    catalog.rungs.push_back(Rung{relay->id, support, declared, max_tick(declared)});
  }
  for (std::uint32_t payload = 0u; payload <= 3u; ++payload) {
    const std::string id = "cpair_receiver_transaction_" + std::to_string(payload);
    const auto transaction = g1_cpair_receiver_transaction_motif(payload);
    if (!transaction.has_value()) {
      catalog.skipped.push_back({id, "motif factory returned nullopt"});
      continue;
    }
    std::vector<ReferenceSite> support;
    std::vector<Port> declared;
    std::string error;
    if (!seal_motif(*transaction, transaction->id, support, declared, &error)) {
      catalog.skipped.push_back({transaction->id, error});
      continue;
    }
    catalog.rungs.push_back(Rung{transaction->id, support, declared, max_tick(declared)});
  }

  return catalog;
}

std::vector<SiteWord> catalog_alphabet(const std::vector<Rung>& rungs) {
  std::vector<SiteWord> words{kQuiescentWord};
  for (const Rung& rung : rungs) {
    for (const ReferenceSite& site : rung.seed) {
      words.push_back(site.word);
    }
  }
  std::sort(words.begin(), words.end());
  words.erase(std::unique(words.begin(), words.end()), words.end());
  return words;
}

}  // namespace substrate::bcc32::viability
