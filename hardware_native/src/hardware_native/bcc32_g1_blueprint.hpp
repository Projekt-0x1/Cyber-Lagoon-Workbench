#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "bcc32_genesis.hpp"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {

// These labels exist only in the offline genesis compiler and its audit
// manifest. They are never serialized into world matter or consulted by F.
enum class G1AuthoredRole : std::uint8_t {
  viability,
  development,
  heredity,
  raw_contact,
  generic_tissue,
  blind_mutation,
  waste_history,
};

[[nodiscard]] std::string_view g1_authored_role_name(G1AuthoredRole role);

// A motif constrains deviations from the ordinary Q word. Explicit set/clear
// masks make composition order-independent and expose contradictory overlaps.
struct G1MaterialConstraint {
  Int3 coordinate{};
  SiteWord set_mask = 0u;
  SiteWord clear_mask = 0u;

  friend bool operator==(const G1MaterialConstraint&, const G1MaterialConstraint&) = default;
};

enum class G1PortKind : std::uint8_t {
  initial_material,
  observed_output,
};

[[nodiscard]] std::string_view g1_port_kind_name(G1PortKind kind);

struct G1PhysicalPort {
  std::string name;
  Int3 coordinate{};
  SiteWord observed_mask = 0u;
  G1PortKind kind = G1PortKind::initial_material;
  std::uint32_t tick = 0u;

  friend bool operator==(const G1PhysicalPort&, const G1PhysicalPort&) = default;
};

struct G1Motif {
  std::string id;
  std::string proven_scope;
  std::vector<G1MaterialConstraint> material;
  std::vector<G1PhysicalPort> ports;
};

struct G1PoweredProducerFlags {
  bool boundary = true;
  bool flux = true;
  bool adapter = true;
};

struct G1MotifPlacement {
  Int3 origin{};
  BasisPermutation basis_permutation{0u, 1u, 2u, 3u};
};

struct G1PlacedRegion {
  std::string instance_id;
  G1AuthoredRole role = G1AuthoredRole::viability;
  std::string motif_id;
  std::string proven_scope;
  Int3 origin{};
  BasisPermutation basis_permutation{0u, 1u, 2u, 3u};
  std::vector<SitePlacement> material_sites;
  std::vector<G1PhysicalPort> ports;
};

struct G1SealedBlueprint {
  EncodedGenesis capsule;
  std::string manifest_json;
  ContentAddress manifest_identity{};
};

class G1BlueprintBuilder {
 public:
  explicit G1BlueprintBuilder(std::string blueprint_id);

  [[nodiscard]] bool place(const G1Motif& motif, std::string instance_id, G1AuthoredRole role,
                           Int3 origin = {}, BasisPermutation permutation = {0u, 1u, 2u, 3u});

  [[nodiscard]] std::optional<G1SealedBlueprint> seal(const ContentAddress& law_identity) const;
  [[nodiscard]] const std::vector<G1PlacedRegion>& regions() const { return regions_; }
  [[nodiscard]] std::string_view error() const { return error_; }

 private:
  struct MaterialState {
    SiteWord set_mask = 0u;
    SiteWord clear_mask = 0u;

    friend bool operator==(const MaterialState&, const MaterialState&) = default;
  };

  [[nodiscard]] std::optional<Genesis> compile_material() const;
  [[nodiscard]] std::string manifest_json(const Genesis& compiled,
                                          const ContentAddress& capsule_identity) const;

  std::string blueprint_id_;
  std::vector<G1PlacedRegion> regions_;
  std::vector<std::pair<Int3, MaterialState>> material_;
  std::string error_;
};

// Exact current-head motifs. Their names describe only the measured contract;
// catalog membership does not promote them into organs or a complete G1.
[[nodiscard]] G1Motif g1_period_two_bond_orbit_motif();
[[nodiscard]] G1Motif g1_flux_powered_period_four_boundary_motif();
[[nodiscard]] G1Motif g1_plasticity_homeostasis_identity_motif();
[[nodiscard]] G1Motif g1_plasticity_homeostasis_forward_fallback_motif();
[[nodiscard]] G1Motif g1_outcome_mediation_addon_motif();
[[nodiscard]] bool place_g1_outcome_recovery_profile_v0(G1BlueprintBuilder& builder,
                                                        std::string_view instance_prefix);
[[nodiscard]] G1Motif g1_c0_period23_turnover_motif();
[[nodiscard]] G1Motif g1_reanchor_seed_motif();
[[nodiscard]] std::optional<G1Motif> g1_integrated_two_bit_constructor_motif(std::uint32_t payload);
[[nodiscard]] G1Motif g1_metabolic_work_waste_motif();
[[nodiscard]] std::optional<G1Motif> g1_cpair_port_relay_motif(std::uint32_t payload);
[[nodiscard]] G1Motif g1_cpair_receiver_scaffold_motif();
[[nodiscard]] std::optional<G1Motif> g1_cpair_receiver_transaction_motif(std::uint32_t payload);
[[nodiscard]] std::optional<G1Motif> g1_single_locus_heredity_relay_motif(std::uint32_t value);
[[nodiscard]] std::optional<G1Motif> g1_powered_producer_arm_motif(
    std::uint32_t value, G1PoweredProducerFlags flags = {});
[[nodiscard]] std::optional<G1Motif> g1_powered_dual_producer_motif(
    std::uint32_t payload, G1PoweredProducerFlags first = {}, G1PoweredProducerFlags second = {});
[[nodiscard]] std::optional<G1Motif> g1_fixed_flux_chassis_motif(
    G1PoweredProducerFlags first = {}, G1PoweredProducerFlags second = {});
[[nodiscard]] const std::array<G1MotifPlacement, 2>& g1_decoupled_heredity_placements();
[[nodiscard]] std::optional<G1Motif> g1_decoupled_chassis_heredity_motif(
    std::uint32_t payload, G1PoweredProducerFlags first = {}, G1PoweredProducerFlags second = {});
[[nodiscard]] std::optional<G1Motif> g1_successor_body_motif(std::uint32_t payload);

}  // namespace substrate::bcc32
