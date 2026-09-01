#include "bcc32_g1_blueprint.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "bcc32_law.cuh"

namespace substrate::bcc32 {
namespace {

struct Int3Less {
  bool operator()(Int3 left, Int3 right) const {
    if (left.x != right.x)
      return left.x < right.x;
    if (left.y != right.y)
      return left.y < right.y;
    return left.z < right.z;
  }
};

[[nodiscard]] bool fits_i32(const CoordinateComponent& value) {
  return value >= std::numeric_limits<std::int32_t>::min() &&
         value <= std::numeric_limits<std::int32_t>::max();
}

[[nodiscard]] std::optional<Int3> transform_coordinate(Int3 coordinate, Int3 origin,
                                                       const BasisPermutation& permutation) {
  Z3Coordinate transformed =
      transformed_coordinate(Z3Coordinate{coordinate.x, coordinate.y, coordinate.z}, permutation);
  transformed.x += origin.x;
  transformed.y += origin.y;
  transformed.z += origin.z;
  if (!fits_i32(transformed.x) || !fits_i32(transformed.y) || !fits_i32(transformed.z)) {
    return std::nullopt;
  }
  return Int3{transformed.x.convert_to<std::int32_t>(), transformed.y.convert_to<std::int32_t>(),
              transformed.z.convert_to<std::int32_t>()};
}

[[nodiscard]] std::string escape_json(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (c < 0x20u) {
          escaped += "\\u00";
          escaped += digits[c >> 4u];
          escaped += digits[c & 0x0fu];
        } else {
          escaped += static_cast<char>(c);
        }
        break;
    }
  }
  return escaped;
}

[[nodiscard]] std::string word_hex(SiteWord word) {
  constexpr char digits[] = "0123456789abcdef";
  std::string output = "0x00000000";
  for (std::uint32_t index = 0u; index < 8u; ++index) {
    output[9u - index] = digits[(word >> (4u * index)) & 0x0fu];
  }
  return output;
}

[[nodiscard]] bool hash_is_zero(const Hash256& hash) {
  return std::all_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte == 0u; });
}

[[nodiscard]] G1MaterialConstraint exact_word(Int3 coordinate, SiteWord word) {
  return {coordinate, word & ~kQ, (~word) & kQ};
}

[[nodiscard]] SiteWord material_word(const G1MaterialConstraint& material) {
  return (kQ | material.set_mask) & ~material.clear_mask;
}

[[nodiscard]] bool merge_word(std::map<Int3, SiteWord, Int3Less>& material, Int3 coordinate,
                              SiteWord word) {
  const auto [entry, inserted] = material.emplace(coordinate, word);
  return inserted || entry->second == word;
}

[[nodiscard]] bool add_transformed_material(std::map<Int3, SiteWord, Int3Less>& destination,
                                            const std::map<Int3, SiteWord, Int3Less>& source,
                                            Int3 origin, const BasisPermutation& permutation) {
  for (const auto& [coordinate, word] : source) {
    const std::optional<Int3> transformed = transform_coordinate(coordinate, origin, permutation);
    if (!transformed.has_value() ||
        !merge_word(destination, *transformed, transformed_word(word, permutation))) {
      return false;
    }
  }
  return true;
}

void append_cpair_locus(G1Motif& motif, std::int32_t z, bool value, std::uint32_t logical_bit) {
  const Int3 p0{-1, 0, z};
  const Int3 p1{0, -1, z};
  const Int3 p2{0, 0, z - 1};
  const Int3 p3{1, 1, z + 1};
  SiteWord p0_word = kQ | kOwnedBondMask;
  SiteWord p1_word = kQ | kOwnedBondMask;
  if (value)
    p1_word |= energy_bit(1u);
  else
    p0_word |= energy_bit(0u);
  motif.material.push_back(exact_word(p0, p0_word));
  motif.material.push_back(exact_word(p1, p1_word));
  motif.material.push_back(exact_word(p2, kQ | kOwnedBondMask | energy_bit(2u)));
  motif.material.push_back(exact_word(p3, kQ | kOwnedBondMask));
  motif.material.push_back(exact_word({-3, 0, z}, kQ & ~carrier_bit(0u)));
  motif.material.push_back(exact_word({-1, 0, z - 2}, kQ & ~carrier_bit(2u)));
  motif.material.push_back(exact_word({0, -3, z}, kQ & ~carrier_bit(1u)));
  motif.material.push_back(exact_word({0, -1, z - 2}, kQ & ~carrier_bit(2u)));
  motif.material.push_back(exact_word({-1, 0, z - 4}, kQ & ~carrier_bit(2u)));
  motif.material.push_back(exact_word({3, 4, z + 4}, kQ & ~carrier_bit(3u)));
  motif.material.push_back(exact_word({0, -1, z - 4}, kQ & ~carrier_bit(2u)));
  motif.material.push_back(exact_word({4, 3, z + 4}, kQ & ~carrier_bit(3u)));
  motif.material.push_back(exact_word({0, -1, z + 1}, kQ | channel_bit(kConformationShift, 0u) |
                                                          channel_bit(kConformationShift, 1u)));

  motif.ports.push_back(
      {"payload_" + std::to_string(logical_bit), value ? p1 : p0, energy_bit(value ? 1u : 0u)});
  motif.ports.push_back({"enabling_e2_" + std::to_string(logical_bit), p2, energy_bit(2u)});
  motif.ports.push_back(
      {"bridge_" + std::to_string(logical_bit),
       {0, -1, z + 1},
       channel_bit(kConformationShift, 0u) | channel_bit(kConformationShift, 1u)});
  motif.ports.push_back({"c2_" + std::to_string(logical_bit),
                         {-1, 0, z + 1},
                         channel_bit(kConformationShift, 2u) | channel_bit(kReactiveShift, 2u),
                         G1PortKind::observed_output,
                         3u});
  motif.ports.push_back({"germ_payload_" + std::to_string(logical_bit),
                         {0, -1, z},
                         energy_bit(logical_bit),
                         G1PortKind::observed_output,
                         7u});
  motif.ports.push_back({"germ_anchor_" + std::to_string(logical_bit),
                         {0, -1, z},
                         owned_bond_bit(logical_bit),
                         G1PortKind::observed_output,
                         7u});
}

[[nodiscard]] bool add_powered_producer_arm(std::map<Int3, SiteWord, Int3Less>& arm,
                                            std::uint32_t value, G1PoweredProducerFlags flags) {
  const std::optional<G1Motif> transaction = g1_cpair_receiver_transaction_motif(value);
  if (!transaction.has_value())
    return false;

  for (const G1MaterialConstraint& material : transaction->material) {
    if (material.coordinate.z > 16)
      continue;
    SiteWord word = material_word(material);
    if (material.coordinate == Int3{0, 0, -1})
      word &= ~energy_bit(2u);
    if (!merge_word(arm, material.coordinate, word))
      return false;
  }

  if (flags.boundary) {
    const G1Motif boundary = g1_flux_powered_period_four_boundary_motif();
    constexpr BasisPermutation boundary_permutation{1u, 2u, 3u, 0u};
    constexpr Int3 boundary_origin{-2, -2, 0};
    for (const G1MaterialConstraint& material : boundary.material) {
      const std::optional<Int3> coordinate =
          transform_coordinate(material.coordinate, boundary_origin, boundary_permutation);
      if (!coordinate.has_value())
        return false;
      SiteWord word = transformed_word(material_word(material), boundary_permutation);
      if (!flags.flux && material.coordinate == Int3{-1, -1, -1})
        word &= ~transformed_word(energy_bit(3u), boundary_permutation);
      if (!merge_word(arm, *coordinate, word))
        return false;
    }
  }

  if (flags.adapter) {
    constexpr Int3 adapter_coordinate{0, 0, 0};
    const auto found = arm.find(adapter_coordinate);
    const SiteWord before = found == arm.end() ? kQ : found->second;
    const SiteWord after = before ^ channel_bit(kConformationShift, 3u);
    if (found == arm.end())
      arm.emplace(adapter_coordinate, after);
    else
      found->second = after;
  }
  return true;
}

[[nodiscard]] G1Motif plasticity_homeostasis_motif(bool forward_fallback) {
  constexpr Int3 kTissueOrigin{-1, -2, 0};
  constexpr Int3 kIntake{-1, -1, -1};
  constexpr BasisPermutation kTissuePermutation{0u, 1u, 3u, 2u};

  std::map<Int3, SiteWord, Int3Less> material{
      {{0, 0, 0}, kQ | owned_bond_bit(3u)},
      {{1, 0, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(2u)},
      {{0, 1, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(1u)},
      {{0, 0, 1}, kQ | owned_bond_bit(2u)},
      {kIntake, (kQ & ~carrier_bit(3u)) | owned_bond_bit(3u)},
  };
  const std::map<Int3, SiteWord, Int3Less> tissue{
      {{0, 0, 0}, kQ | 0x00008000u},
      {{0, 1, 0}, kQ | 0x18001400u},
  };
  if (!add_transformed_material(material, tissue, kTissueOrigin, kTissuePermutation))
    throw std::logic_error("plasticity tissue unexpectedly overlaps its authored body");
  if (forward_fallback)
    material.at(kIntake) |= face_bit(4u);

  G1Motif motif;
  motif.id = forward_fallback ? "plasticity_homeostasis_forward_fallback"
                              : "plasticity_homeostasis_identity";
  motif.proven_scope = forward_fallback
                           ? "authored language-blank G1 body and plasticity tissue with one "
                             "additional face quantum that reconstructs a relocated primary face "
                             "under forward F; damage detection, proposal, appraisal, failure "
                             "learning, broad healing, self-redesign, and language remain unproved"
                           : "authored language-blank G1 body and plasticity tissue without the "
                             "forward-reconstruction face quantum; it is the identity-matched "
                             "control, not a healing or self-redesign claim";
  motif.material.reserve(material.size());
  for (const auto& [coordinate, word] : material)
    motif.material.push_back(exact_word(coordinate, word));

  const Int3 first_tissue = *transform_coordinate({0, 0, 0}, kTissueOrigin, kTissuePermutation);
  const Int3 second_tissue = *transform_coordinate({0, 1, 0}, kTissueOrigin, kTissuePermutation);
  motif.ports = {
      {"intake_bond", kIntake, owned_bond_bit(3u)},
      {"plasticity_seed_0", first_tissue, transformed_word(0x00008000u, kTissuePermutation)},
      {"plasticity_seed_1", second_tissue, transformed_word(0x18001400u, kTissuePermutation)},
  };
  if (forward_fallback)
    motif.ports.push_back({"forward_fallback_face", kIntake, face_bit(4u)});
  return motif;
}

}  // namespace

std::string_view g1_authored_role_name(G1AuthoredRole role) {
  switch (role) {
    case G1AuthoredRole::viability:
      return "viability";
    case G1AuthoredRole::development:
      return "development";
    case G1AuthoredRole::heredity:
      return "heredity";
    case G1AuthoredRole::raw_contact:
      return "raw_contact";
    case G1AuthoredRole::generic_tissue:
      return "generic_tissue";
    case G1AuthoredRole::blind_mutation:
      return "blind_mutation";
    case G1AuthoredRole::waste_history:
      return "waste_history";
  }
  return "unknown";
}

std::string_view g1_port_kind_name(G1PortKind kind) {
  switch (kind) {
    case G1PortKind::initial_material:
      return "initial_material";
    case G1PortKind::observed_output:
      return "observed_output";
  }
  return "unknown";
}

G1BlueprintBuilder::G1BlueprintBuilder(std::string blueprint_id)
    : blueprint_id_(std::move(blueprint_id)) {
  if (blueprint_id_.empty())
    error_ = "blueprint id must be nonempty";
}

bool G1BlueprintBuilder::place(const G1Motif& motif, std::string instance_id, G1AuthoredRole role,
                               Int3 origin, BasisPermutation permutation) {
  if (!error_.empty())
    return false;
  if (motif.id.empty() || motif.proven_scope.empty() || motif.material.empty() ||
      instance_id.empty() || !is_basis_permutation(permutation)) {
    error_ = "motif placement is incomplete or has an invalid basis permutation";
    return false;
  }
  if (std::any_of(regions_.begin(), regions_.end(), [&](const G1PlacedRegion& region) {
        return region.instance_id == instance_id;
      })) {
    error_ = "motif instance ids must be unique";
    return false;
  }

  std::map<Int3, MaterialState, Int3Less> combined;
  for (const auto& [coordinate, state] : material_)
    combined.emplace(coordinate, state);
  std::map<Int3, MaterialState, Int3Less> local;

  G1PlacedRegion region;
  region.instance_id = std::move(instance_id);
  region.role = role;
  region.motif_id = motif.id;
  region.proven_scope = motif.proven_scope;
  region.origin = origin;
  region.basis_permutation = permutation;

  for (const G1MaterialConstraint& constraint : motif.material) {
    if ((constraint.set_mask & constraint.clear_mask) != 0u ||
        (constraint.set_mask | constraint.clear_mask) == 0u) {
      error_ = "material constraint must be nonempty and internally consistent";
      return false;
    }
    const std::optional<Int3> coordinate =
        transform_coordinate(constraint.coordinate, origin, permutation);
    if (!coordinate.has_value()) {
      error_ = "transformed motif coordinate exceeds the genesis coordinate range";
      return false;
    }
    const SiteWord set_mask = transformed_word(constraint.set_mask, permutation);
    const SiteWord clear_mask = transformed_word(constraint.clear_mask, permutation);
    const MaterialState requirement{set_mask, clear_mask};
    const auto [local_it, local_inserted] = local.emplace(*coordinate, requirement);
    if (!local_inserted && local_it->second != requirement) {
      error_ = "one motif must declare one complete material requirement per coordinate";
      return false;
    }
    const auto existing = combined.find(*coordinate);
    if (existing != combined.end() && existing->second != requirement) {
      error_ = "overlapping motifs may share only an identical complete material word";
      return false;
    }
    combined[*coordinate] = requirement;
    const SiteWord word = (kQ | requirement.set_mask) & ~requirement.clear_mask;
    region.material_sites.push_back({*coordinate, word});
  }

  std::set<std::string> port_names;
  for (const G1PhysicalPort& port : motif.ports) {
    const bool known_kind =
        port.kind == G1PortKind::initial_material || port.kind == G1PortKind::observed_output;
    if (port.name.empty() || port.observed_mask == 0u || !known_kind ||
        (port.kind == G1PortKind::initial_material && port.tick != 0u) ||
        (port.kind == G1PortKind::observed_output && port.tick == 0u)) {
      error_ = "physical ports require a name, mask, and kind-consistent tick";
      return false;
    }
    if (!port_names.insert(port.name).second) {
      error_ = "physical port names must be unique within one motif";
      return false;
    }
    const std::optional<Int3> coordinate =
        transform_coordinate(port.coordinate, origin, permutation);
    if (!coordinate.has_value()) {
      error_ = "transformed port coordinate exceeds the genesis coordinate range";
      return false;
    }
    if (port.kind == G1PortKind::initial_material) {
      const auto material = local.find(*coordinate);
      if (material == local.end()) {
        error_ = "initial material ports must name material declared by their motif";
        return false;
      }
    }
    region.ports.push_back({port.name, *coordinate,
                            transformed_word(port.observed_mask, permutation), port.kind,
                            port.tick});
  }

  material_.assign(combined.begin(), combined.end());
  regions_.push_back(std::move(region));
  return true;
}

std::optional<Genesis> G1BlueprintBuilder::compile_material() const {
  if (!error_.empty() || blueprint_id_.empty() || regions_.empty())
    return std::nullopt;
  Genesis genesis{};
  genesis.metadata.genesis_class = GenesisClass::G1;
  genesis.metadata.artifact_type = ArtifactType::authored_material;
  genesis.metadata.contact_boundary_mask = 0xffu;
  for (const auto& [coordinate, state] : material_) {
    const SiteWord word = (kQ | state.set_mask) & ~state.clear_mask;
    if (word != kQ)
      genesis.sites.push_back({coordinate, word});
  }
  if (genesis.sites.empty())
    return std::nullopt;
  return genesis;
}

std::optional<G1SealedBlueprint> G1BlueprintBuilder::seal(
    const ContentAddress& law_identity) const {
  const std::optional<Genesis> draft = compile_material();
  if (!draft.has_value())
    return std::nullopt;
  std::optional<EncodedGenesis> capsule = seal_genesis(*draft, law_identity);
  if (!capsule.has_value())
    return std::nullopt;
  const std::optional<G1Capsule> opened = open_g1_capsule(*capsule, law_identity);
  if (!opened.has_value())
    return std::nullopt;
  const Genesis compiled = compile_g1(*opened);
  const ContentAddress capsule_identity = genesis_artifact_identity(*capsule);
  std::string manifest = manifest_json(compiled, capsule_identity);
  if (manifest.empty())
    return std::nullopt;
  const ContentAddress manifest_identity =
      content_address({reinterpret_cast<const std::byte*>(manifest.data()), manifest.size()});
  if (!is_valid_content_address(manifest_identity))
    return std::nullopt;
  return G1SealedBlueprint{std::move(*capsule), std::move(manifest), manifest_identity};
}

std::string G1BlueprintBuilder::manifest_json(const Genesis& compiled,
                                              const ContentAddress& capsule_identity) const {
  if (compiled.metadata.law_identity.byte_count == 0u ||
      hash_is_zero(compiled.metadata.law_identity.digest) ||
      hash_is_zero(compiled.metadata.content_hash) || capsule_identity.byte_count == 0u ||
      hash_is_zero(capsule_identity.digest)) {
    return {};
  }
  std::vector<G1PlacedRegion> regions = regions_;
  std::sort(regions.begin(), regions.end(),
            [](const G1PlacedRegion& left, const G1PlacedRegion& right) {
              return left.instance_id < right.instance_id;
            });
  for (G1PlacedRegion& region : regions) {
    std::sort(region.material_sites.begin(), region.material_sites.end(),
              [](const SitePlacement& left, const SitePlacement& right) {
                return Int3Less{}(left.coordinate, right.coordinate);
              });
    std::sort(region.ports.begin(), region.ports.end(),
              [](const G1PhysicalPort& left, const G1PhysicalPort& right) {
                return left.name < right.name;
              });
  }
  std::ostringstream out;
  out << "{\"schema\":\"bcc32-g1-blueprint-manifest-v1\",\"blueprint\":\""
      << escape_json(blueprint_id_)
      << "\",\"runtime_semantics\":false,\"law_identity\":{\"sha256\":\""
      << hash_hex(compiled.metadata.law_identity.digest)
      << "\",\"length\":" << compiled.metadata.law_identity.byte_count
      << "},\"material_content_sha256\":\"" << hash_hex(compiled.metadata.content_hash)
      << "\",\"capsule_identity\":{\"sha256\":\"" << hash_hex(capsule_identity.digest)
      << "\",\"length\":" << capsule_identity.byte_count
      << "},\"site_count\":" << compiled.sites.size() << ",\"regions\":[";
  for (std::size_t index = 0u; index < regions.size(); ++index) {
    if (index != 0u)
      out << ',';
    const G1PlacedRegion& region = regions[index];
    out << "{\"instance\":\"" << escape_json(region.instance_id) << "\",\"role\":\""
        << g1_authored_role_name(region.role) << "\",\"motif\":\"" << escape_json(region.motif_id)
        << "\",\"proven_scope\":\"" << escape_json(region.proven_scope) << "\",\"origin\":["
        << region.origin.x << ',' << region.origin.y << ',' << region.origin.z
        << "],\"basis_permutation\":[";
    for (std::size_t basis = 0u; basis < region.basis_permutation.size(); ++basis) {
      if (basis != 0u)
        out << ',';
      out << region.basis_permutation[basis];
    }
    out << "],\"material\":[";
    for (std::size_t site_index = 0u; site_index < region.material_sites.size(); ++site_index) {
      if (site_index != 0u)
        out << ',';
      const SitePlacement& site = region.material_sites[site_index];
      out << "{\"coordinate\":[" << site.coordinate.x << ',' << site.coordinate.y << ','
          << site.coordinate.z << "],\"word\":\"" << word_hex(site.word) << "\"}";
    }
    out << "],\"ports\":[";
    for (std::size_t port_index = 0u; port_index < region.ports.size(); ++port_index) {
      if (port_index != 0u)
        out << ',';
      const G1PhysicalPort& port = region.ports[port_index];
      out << "{\"name\":\"" << escape_json(port.name) << "\",\"coordinate\":[" << port.coordinate.x
          << ',' << port.coordinate.y << ',' << port.coordinate.z << "],\"mask\":\""
          << word_hex(port.observed_mask) << "\",\"kind\":\"" << g1_port_kind_name(port.kind)
          << "\",\"tick\":" << port.tick << '}';
    }
    out << "]}";
  }
  out << "]}";
  return out.str();
}

G1Motif g1_period_two_bond_orbit_motif() {
  G1Motif motif;
  motif.id = "period_two_bond_orbit";
  motif.proven_scope =
      "three-site bounded active structural orbit; not metabolism, heredity, or healing";
  motif.material = {
      exact_word({0, 0, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(1u)),
      exact_word({1, 0, 0}, kQ | owned_bond_bit(2u)),
      exact_word({0, 1, 0}, kQ | owned_bond_bit(3u)),
  };
  motif.ports = {{"bond_phase", {0, 0, 0}, kOwnedBondMask}};
  return motif;
}

G1Motif g1_flux_powered_period_four_boundary_motif() {
  G1Motif motif;
  motif.id = "flux_powered_period_four_boundary";
  motif.proven_scope =
      "five-site period-four bond boundary with represented pair-exchange capture; not a "
      "complete-state oscillator, waste-export witness, or producer";
  motif.material = {
      exact_word({0, 0, 0}, kQ | owned_bond_bit(3u)),
      exact_word({1, 0, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(2u)),
      exact_word({0, 1, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(1u)),
      exact_word({0, 0, 1}, kQ | owned_bond_bit(2u)),
      exact_word({-1, -1, -1}, (kQ & ~carrier_bit(3u)) | owned_bond_bit(3u) | energy_bit(3u)),
  };
  motif.ports = {
      {"intake", {-1, -1, -1}, carrier_bit(3u) | energy_bit(3u)},
      {"bond_phase", {0, 0, 0}, kOwnedBondMask},
  };
  return motif;
}

G1Motif g1_plasticity_homeostasis_identity_motif() {
  return plasticity_homeostasis_motif(false);
}

G1Motif g1_plasticity_homeostasis_forward_fallback_motif() {
  return plasticity_homeostasis_motif(true);
}

G1Motif g1_outcome_mediation_addon_motif() {
  G1Motif motif;
  motif.id = "outcome_mediation_addon";
  motif.proven_scope =
      "three-site language-blank add-on whose measured outcome-conditioned mediation requires "
      "the exact plasticity/homeostasis chassis and declared developmental contact history; the "
      "three sites have no standalone functional claim and are not a distinct record, selector, "
      "appraisal policy, self-edit, or learning claim";
  motif.material = {
      exact_word({-2, 0, 0}, 0x002002f6u),
      exact_word({0, 1, 1}, 0x002080dbu),
      exact_word({1, 1, 2}, 0x800010ebu),
  };
  motif.ports = {
      {"mediation_seed_0", {-2, 0, 0}, 0x002002f6u ^ kQ},
      {"mediation_seed_1", {0, 1, 1}, 0x002080dbu ^ kQ},
      {"mediation_seed_2", {1, 1, 2}, 0x800010ebu ^ kQ},
  };
  return motif;
}

bool place_g1_outcome_recovery_profile_v0(G1BlueprintBuilder& builder,
                                          std::string_view instance_prefix) {
  const std::string prefix(instance_prefix);
  return builder.place(g1_plasticity_homeostasis_forward_fallback_motif(), prefix + ".body",
                       G1AuthoredRole::viability) &&
         builder.place(g1_outcome_mediation_addon_motif(), prefix + ".outcome_addon",
                       G1AuthoredRole::generic_tissue);
}

G1Motif g1_c0_period23_turnover_motif() {
  G1Motif motif;
  motif.id = "c0_period23_turnover";
  motif.proven_scope =
      "five-site exact period-23 C0 catalytic return under complete F; it is an authored "
      "turnover affordance, not a translated body, constructor, propagule, or lineage";
  motif.material = {
      exact_word({-1, -1, -1}, kQ | channel_bit(kReactiveShift, 0u)),
      exact_word({-1, 0, 0}, kQ | owned_bond_bit(0u)),
      exact_word({0, 0, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(1u) | owned_bond_bit(2u) |
                                energy_bit(0u)),
      exact_word({1, 0, 0}, kQ | channel_bit(kReactiveShift, 0u) | face_bit(1u)),
      exact_word({2, 0, 0}, kQ | face_bit(4u)),
  };
  motif.ports = {
      {"turnover_input_e0", {0, 0, 0}, energy_bit(0u)},
      {"turnover_c0_out",
       {1, 0, 0},
       channel_bit(kConformationShift, 0u),
       G1PortKind::observed_output,
       6u},
      {"turnover_c0_reentry",
       {1, 0, 0},
       channel_bit(kConformationShift, 0u),
       G1PortKind::observed_output,
       12u},
      {"turnover_e0_restored", {0, 0, 0}, energy_bit(0u), G1PortKind::observed_output, 14u},
  };
  return motif;
}

G1Motif g1_reanchor_seed_motif() {
  G1Motif motif;
  motif.id = "reanchor_seed";
  motif.proven_scope =
      "two ordinary B|E parents recruit two fresh X anchors after two uninterrupted full-F "
      "supersteps; constructor reconstruction, daughter release, and lineage remain unproved";
  motif.material = {
      exact_word({-1, 0, 0}, kQ | owned_bond_bit(0u) | energy_bit(0u)),
      exact_word({0, -1, 0}, kQ | owned_bond_bit(1u) | energy_bit(1u)),
  };
  motif.ports = {
      {"parent_0", {-1, 0, 0}, owned_bond_bit(0u) | energy_bit(0u)},
      {"parent_1", {0, -1, 0}, owned_bond_bit(1u) | energy_bit(1u)},
      {"child_cr", {0, 0, 0}, kConformationMask | kReactiveMask, G1PortKind::observed_output, 1u},
      {"fresh_anchor_2", {0, 0, -1}, face_bit(2u), G1PortKind::observed_output, 2u},
      {"fresh_anchor_3", {1, 1, 1}, face_bit(3u), G1PortKind::observed_output, 2u},
  };
  return motif;
}

std::optional<G1Motif> g1_integrated_two_bit_constructor_motif(std::uint32_t payload) {
  if (payload > 3u)
    return std::nullopt;

  SiteWord parent = 0x0200005fu;
  if ((payload & 1u) != 0u)
    parent |= channel_bit(kConformationShift, 0u);
  if ((payload & 2u) != 0u)
    parent |= channel_bit(kConformationShift, 1u);

  G1Motif motif;
  motif.id = "integrated_two_bit_constructor_" + std::to_string(payload);
  motif.proven_scope =
      "one five-site feedstock circuit copies two arbitrary parent C bits into distributed E "
      "daughters at F2 and releases represented daughters at F3; it does not reconstruct its "
      "feedstock, produce a second constructor, or establish lineage";
  motif.material = {
      exact_word({-1, -1, -1}, 0x008000ffu), exact_word({-1, 0, 0}, 0xc2400a66u),
      exact_word({0, -1, 0}, 0x401c04f2u),   exact_word({0, 0, 0}, parent),
      exact_word({1, 1, 1}, 0x000008f7u),
  };
  motif.ports = {
      {"parent_payload_0", {0, 0, 0}, channel_bit(kConformationShift, 0u)},
      {"parent_payload_1", {0, 0, 0}, channel_bit(kConformationShift, 1u)},
      {"daughter_payload_0", {-1, 0, 0}, energy_bit(0u), G1PortKind::observed_output, 2u},
      {"daughter_payload_1", {0, -1, 0}, energy_bit(1u), G1PortKind::observed_output, 2u},
      {"release_0", {-1, 0, 0}, owned_bond_bit(0u), G1PortKind::observed_output, 3u},
      {"release_1", {0, -1, 0}, energy_bit(1u), G1PortKind::observed_output, 3u},
  };
  return motif;
}

G1Motif g1_metabolic_work_waste_motif() {
  G1Motif motif;
  motif.id = "metabolic_work_waste";
  motif.proven_scope =
      "four-site resource-to-work-to-waste transaction whose two catalysts restore exactly at "
      "F2; no-flux and waste-replay controls separate, but boundary maintenance, heredity, and "
      "reproduction remain unproved";
  motif.material = {
      exact_word({-1, 0, 0}, 0xf0011effu),
      exact_word({0, -1, 0}, 0x222220ffu),
  };
  motif.ports = {
      {"resource_0", {-1, 0, 0}, energy_bit(0u)},
      {"resource_1", {0, -1, 0}, energy_bit(1u)},
      {"work", {0, 0, 0}, 0x0fff0000u, G1PortKind::observed_output, 2u},
      {"waste", {1, 1, 1}, 0x00000800u, G1PortKind::observed_output, 2u},
  };
  return motif;
}

std::optional<G1Motif> g1_cpair_port_relay_motif(std::uint32_t payload) {
  if (payload > 3u)
    return std::nullopt;

  G1Motif motif;
  motif.id = "cpair_port_relay_" + std::to_string(payload);
  motif.proven_scope =
      "two-locus dual-rail full-F relay through C2/R2, bonded E germ observations, and fresh "
      "distributed C/R outputs; downstream consumption, a repeatable producer, an independent "
      "propagule, and lineage remain unproved";

  append_cpair_locus(motif, 0, (payload & 1u) != 0u, 0u);
  append_cpair_locus(motif, 32, (payload & 2u) != 0u, 1u);
  motif.ports.push_back({"fresh_output_0",
                         {2, 1, 1},
                         channel_bit(kConformationShift, 0u) | channel_bit(kReactiveShift, 0u),
                         G1PortKind::observed_output,
                         8u});
  motif.ports.push_back({"fresh_output_1",
                         {-1, 1, 32},
                         channel_bit(kConformationShift, 1u) | channel_bit(kReactiveShift, 1u),
                         G1PortKind::observed_output,
                         8u});
  return motif;
}

std::optional<G1Motif> g1_single_locus_heredity_relay_motif(std::uint32_t value) {
  if (value > 1u)
    return std::nullopt;

  G1Motif motif;
  motif.id = "single_locus_heredity_relay_" + std::to_string(value);
  motif.proven_scope =
      "one payload-independent thirteen-site full-F relay whose dual-rail E input produces one "
      "fresh C/R state; two transformed copies carry independently decodable loci in one jointly "
      "coupled fixed G1 body, while repeatable reconstruction and lineage remain unproved";
  append_cpair_locus(motif, 0, value != 0u, 0u);
  motif.ports.clear();
  motif.ports.push_back({"payload_zero_rail", {-1, 0, 0}, energy_bit(0u)});
  motif.ports.push_back({"payload_one_rail", {0, -1, 0}, energy_bit(1u)});
  motif.ports.push_back({"c2",
                         {-1, 0, 1},
                         channel_bit(kConformationShift, 2u) | channel_bit(kReactiveShift, 2u),
                         G1PortKind::observed_output,
                         3u});
  motif.ports.push_back(
      {"germ_payload", {0, -1, 0}, energy_bit(0u), G1PortKind::observed_output, 7u});
  motif.ports.push_back(
      {"germ_anchor", {0, -1, 0}, owned_bond_bit(0u), G1PortKind::observed_output, 7u});
  motif.ports.push_back({"fresh_output",
                         {2, 1, 1},
                         channel_bit(kConformationShift, 0u) | channel_bit(kReactiveShift, 0u),
                         G1PortKind::observed_output,
                         8u});
  return motif;
}

G1Motif g1_cpair_receiver_scaffold_motif() {
  constexpr std::array<std::pair<Int3, SiteWord>, 8> shared_receiver{{
      {{-1, -1, -1}, 0x000080ffu},
      {{-1, 0, 0}, 0x000310feu},
      {{0, -1, 0}, 0x000320fdu},
      {{0, 0, -1}, 0x000840fbu},
      {{0, 0, 0}, 0x00c20000u},
      {{0, 0, 1}, 0x000040ffu},
      {{1, 0, 0}, 0x000010ffu},
      {{1, 1, 1}, 0x000430fcu},
  }};

  G1Motif motif;
  motif.id = "cpair_receiver_scaffold";
  motif.proven_scope =
      "two payload-independent copies of one eight-site ordinary-matter receiver scaffold; "
      "co-resident full-F transduction is proven by the transaction motif, while receiver "
      "reconstruction, reproduction, lineage, metabolism, and learning remain unproved";

  const auto add_receiver = [&](Int3 origin, const BasisPermutation& permutation) {
    for (const auto& [local_coordinate, word] : shared_receiver) {
      const std::optional<Int3> coordinate =
          transform_coordinate(local_coordinate, origin, permutation);
      if (!coordinate.has_value())
        throw std::overflow_error("G1 receiver coordinate exceeds int32 range");
      motif.material.push_back(exact_word(*coordinate, transformed_word(word, permutation)));
    }
  };

  add_receiver({2, 0, 1}, {0u, 1u, 2u, 3u});
  add_receiver({0, 1, 32}, {0u, 1u, 3u, 2u});
  motif.ports = {
      {"receiver_arm_0_seed",
       {2, 0, 1},
       channel_bit(kConformationShift, 2u) | channel_bit(kConformationShift, 3u)},
      {"receiver_arm_1_seed",
       {0, 1, 32},
       channel_bit(kConformationShift, 2u) | channel_bit(kConformationShift, 3u)},
  };
  return motif;
}

std::optional<G1Motif> g1_cpair_receiver_transaction_motif(std::uint32_t payload) {
  std::optional<G1Motif> relay = g1_cpair_port_relay_motif(payload);
  if (!relay.has_value())
    return std::nullopt;

  const G1Motif receiver = g1_cpair_receiver_scaffold_motif();
  G1Motif motif = std::move(*relay);
  motif.id = "cpair_receiver_transaction_" + std::to_string(payload);
  motif.proven_scope =
      "two-bit full-F relay-receiver composite with causal F9 gates and distributed E outputs at "
      "F10; one downstream bonded value-cell is load-bearing for a later retained value, while the "
      "later observed attachment survives the parent lesions and is not yet a demonstrated "
      "construction port; separate data and structural paths then form an F24 conjunction that is "
      "necessary for F26 C state at a seed-initially-Q coordinate, but no equivalent reusable "
      "daughter head is established; receiver "
      "reconstruction, "
      "a repeatable producer, an "
      "independent propagule, lineage, metabolism, learning, and life remain unproved";
  motif.material.insert(motif.material.end(), receiver.material.begin(), receiver.material.end());
  motif.ports.insert(motif.ports.end(), receiver.ports.begin(), receiver.ports.end());
  motif.ports.push_back(
      {"receiver_gate_0_b1", {2, -1, 1}, owned_bond_bit(1u), G1PortKind::observed_output, 9u});
  motif.ports.push_back({"receiver_gate_1_c1",
                         {0, 1, 32},
                         channel_bit(kConformationShift, 1u),
                         G1PortKind::observed_output,
                         9u});
  motif.ports.push_back(
      {"receiver_output_0", {2, -1, 1}, energy_bit(0u), G1PortKind::observed_output, 10u});
  motif.ports.push_back(
      {"receiver_output_1", {0, 0, 32}, energy_bit(1u), G1PortKind::observed_output, 10u});
  motif.ports.push_back({"receiver_reentry_0_c0",
                         {3, -1, 1},
                         channel_bit(kConformationShift, 0u),
                         G1PortKind::observed_output,
                         12u});
  motif.ports.push_back({"receiver_reentry_1_c1",
                         {0, 0, 32},
                         channel_bit(kConformationShift, 1u),
                         G1PortKind::observed_output,
                         13u});
  motif.ports.push_back({"bonded_value_1_c3",
                         {-1, 0, 30},
                         channel_bit(kConformationShift, 3u),
                         G1PortKind::observed_output,
                         17u});
  motif.ports.push_back(
      {"bonded_value_1_b3", {0, 1, 31}, owned_bond_bit(3u), G1PortKind::observed_output, 17u});
  motif.ports.push_back({"downstream_value_1_r0",
                         {0, 1, 32},
                         channel_bit(kReactiveShift, 0u),
                         G1PortKind::observed_output,
                         21u});
  motif.ports.push_back(
      {"downstream_arm_1_b2", {0, 1, 31}, owned_bond_bit(2u), G1PortKind::observed_output, 21u});
  motif.ports.push_back({"structural_parent_1_b0",
                         {-1, 1, 32},
                         owned_bond_bit(0u),
                         G1PortKind::observed_output,
                         17u});
  motif.ports.push_back(
      {"serial_joint_1_x1", {0, 0, 32}, face_bit(1u), G1PortKind::observed_output, 24u});
  motif.ports.push_back({"serial_result_1_c3",
                         {-1, -1, 31},
                         channel_bit(kConformationShift, 3u),
                         G1PortKind::observed_output,
                         26u});
  return motif;
}

std::optional<G1Motif> g1_powered_dual_producer_motif(std::uint32_t payload,
                                                      G1PoweredProducerFlags first_flags,
                                                      G1PoweredProducerFlags second_flags) {
  if (payload > 3u)
    return std::nullopt;

  std::map<Int3, SiteWord, Int3Less> first;
  std::map<Int3, SiteWord, Int3Less> second;
  if (!add_powered_producer_arm(first, payload & 1u, first_flags) ||
      !add_powered_producer_arm(second, (payload >> 1u) & 1u, second_flags)) {
    return std::nullopt;
  }

  std::map<Int3, SiteWord, Int3Less> body;
  constexpr BasisPermutation first_permutation{2u, 1u, 3u, 0u};
  constexpr BasisPermutation second_permutation{1u, 2u, 0u, 3u};
  if (!add_transformed_material(body, first, {-1, 0, -2}, first_permutation) ||
      !add_transformed_material(body, second, {0, -3, 0}, second_permutation)) {
    return std::nullopt;
  }

  G1Motif motif;
  motif.id = "powered_dual_producer_" + std::to_string(payload);
  motif.proven_scope =
      "two symmetry-related flux-powered relay/receiver arms that form the exact two-bit C/R "
      "child at F11 and recruit both virgin anchors at F12; producer reconstruction, an "
      "independently executable daughter, a granddaughter, and lineage remain unproved";
  motif.material.reserve(body.size());
  for (const auto& [coordinate, word] : body)
    motif.material.push_back(exact_word(coordinate, word));
  motif.ports = {
      {"heredity_0_zero_rail", {-1, 0, -3}, energy_bit(2u), G1PortKind::initial_material, 0u},
      {"heredity_0_one_rail", {-1, -1, -2}, energy_bit(1u), G1PortKind::initial_material, 0u},
      {"heredity_1_zero_rail", {0, -4, 0}, energy_bit(1u), G1PortKind::initial_material, 0u},
      {"heredity_1_one_rail", {0, -3, -1}, energy_bit(2u), G1PortKind::initial_material, 0u},
      {"child_cr", {0, 0, 0}, kConformationMask | kReactiveMask, G1PortKind::observed_output, 11u},
      {"child_anchor_2", {0, 0, -1}, face_bit(2u), G1PortKind::observed_output, 12u},
      {"child_anchor_3", {1, 1, 1}, face_bit(3u), G1PortKind::observed_output, 12u},
  };
  return motif;
}

std::optional<G1Motif> g1_fixed_flux_chassis_motif(G1PoweredProducerFlags first_flags,
                                                   G1PoweredProducerFlags second_flags) {
  const std::optional<G1Motif> powered =
      g1_powered_dual_producer_motif(3u, first_flags, second_flags);
  if (!powered.has_value())
    return std::nullopt;

  G1Motif motif;
  motif.id = "fixed_flux_chassis";
  motif.proven_scope =
      "one fixed language-blank flux chassis whose two structural arm keys are ordinary mutable "
      "genesis matter, not hereditary values; independent two-bit heredity is supplied only by "
      "separate single-locus relays, while complete daughter reconstruction and lineage remain "
      "unproved";
  motif.material = powered->material;
  motif.ports = {
      {"chassis_child_cr",
       {0, 0, 0},
       kConformationMask | kReactiveMask,
       G1PortKind::observed_output,
       11u},
      {"chassis_anchor_2", {0, 0, -1}, face_bit(2u), G1PortKind::observed_output, 12u},
      {"chassis_anchor_3", {1, 1, 1}, face_bit(3u), G1PortKind::observed_output, 12u},
  };
  return motif;
}

const std::array<G1MotifPlacement, 2>& g1_decoupled_heredity_placements() {
  static const std::array<G1MotifPlacement, 2> placements{{
      {{-2, 0, -3}, {1u, 0u, 2u, 3u}},
      {{0, 2, 3}, {3u, 0u, 2u, 1u}},
  }};
  return placements;
}

std::optional<G1Motif> g1_decoupled_chassis_heredity_motif(std::uint32_t payload,
                                                           G1PoweredProducerFlags first_flags,
                                                           G1PoweredProducerFlags second_flags) {
  if (payload > 3u)
    return std::nullopt;
  const std::optional<G1Motif> chassis = g1_fixed_flux_chassis_motif(first_flags, second_flags);
  const std::optional<G1Motif> first = g1_single_locus_heredity_relay_motif(payload & 1u);
  const std::optional<G1Motif> second = g1_single_locus_heredity_relay_motif((payload >> 1u) & 1u);
  if (!chassis.has_value() || !first.has_value() || !second.has_value())
    return std::nullopt;

  std::map<Int3, SiteWord, Int3Less> body;
  for (const G1MaterialConstraint& material : chassis->material) {
    if (!merge_word(body, material.coordinate, material_word(material)))
      return std::nullopt;
  }
  const auto& placements = g1_decoupled_heredity_placements();
  const auto add_relay = [&](const G1Motif& relay, const G1MotifPlacement& placement) {
    std::map<Int3, SiteWord, Int3Less> material;
    for (const G1MaterialConstraint& constraint : relay.material) {
      if (!merge_word(material, constraint.coordinate, material_word(constraint)))
        return false;
    }
    return add_transformed_material(body, material, placement.origin, placement.basis_permutation);
  };
  if (!add_relay(*first, placements[0]) || !add_relay(*second, placements[1]))
    return std::nullopt;

  const auto transformed_port =
      [&](std::string name, const G1PhysicalPort& port,
          const G1MotifPlacement& placement) -> std::optional<G1PhysicalPort> {
    const std::optional<Int3> coordinate =
        transform_coordinate(port.coordinate, placement.origin, placement.basis_permutation);
    if (!coordinate.has_value())
      return std::nullopt;
    return G1PhysicalPort{std::move(name), *coordinate,
                          transformed_word(port.observed_mask, placement.basis_permutation),
                          port.kind, port.tick};
  };

  G1Motif motif;
  motif.id = "decoupled_chassis_heredity_" + std::to_string(payload);
  motif.proven_scope =
      "one fixed flux chassis plus two symmetry-related one-bit relays: all four genesis values "
      "share one chassis and equal represented matter, while exact independent C/R rails at F13 "
      "require the chassis and survive matched sibling-relay lesions; complete daughter "
      "reconstruction, autonomous branching, selection, learning, and lineage remain unproved";
  motif.material.reserve(body.size());
  for (const auto& [coordinate, value] : body)
    motif.material.push_back(exact_word(coordinate, value));
  motif.ports = chassis->ports;
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const G1Motif& relay = index == 0u ? *first : *second;
    const G1MotifPlacement& placement = placements[index];
    const auto zero =
        std::find_if(relay.ports.begin(), relay.ports.end(),
                     [](const G1PhysicalPort& port) { return port.name == "payload_zero_rail"; });
    const auto one =
        std::find_if(relay.ports.begin(), relay.ports.end(),
                     [](const G1PhysicalPort& port) { return port.name == "payload_one_rail"; });
    if (zero == relay.ports.end() || one == relay.ports.end())
      return std::nullopt;
    const std::optional<G1PhysicalPort> transformed_zero =
        transformed_port("heredity_" + std::to_string(index) + "_zero_rail", *zero, placement);
    const std::optional<G1PhysicalPort> transformed_one =
        transformed_port("heredity_" + std::to_string(index) + "_one_rail", *one, placement);
    if (!transformed_zero.has_value() || !transformed_one.has_value())
      return std::nullopt;
    motif.ports.push_back(*transformed_zero);
    motif.ports.push_back(*transformed_one);
  }
  motif.ports.push_back({"daughter_heredity_0_cr",
                         {-1, 2, -2},
                         channel_bit(kConformationShift, 1u) | channel_bit(kReactiveShift, 1u),
                         G1PortKind::observed_output,
                         13u});
  motif.ports.push_back({"daughter_heredity_1_cr",
                         {-1, 0, 2},
                         channel_bit(kConformationShift, 3u) | channel_bit(kReactiveShift, 3u),
                         G1PortKind::observed_output,
                         13u});
  motif.ports.push_back(
      {"daughter_valid_0", {-3, 0, -3}, face_bit(2u), G1PortKind::observed_output, 13u});
  motif.ports.push_back({"daughter_valid_1",
                         {0, 2, 3},
                         channel_bit(kReactiveShift, 1u),
                         G1PortKind::observed_output,
                         13u});
  motif.ports.push_back(
      {"daughter_anchor_2", {0, 0, -1}, face_bit(2u), G1PortKind::observed_output, 13u});
  motif.ports.push_back(
      {"daughter_anchor_3", {1, 1, 1}, face_bit(3u), G1PortKind::observed_output, 13u});
  return motif;
}

std::optional<G1Motif> g1_successor_body_motif(std::uint32_t payload) {
  if (payload > 3u)
    return std::nullopt;
  const std::optional<G1Motif> body = g1_decoupled_chassis_heredity_motif(payload);
  // The constructor has the same zero-default input state in all four bodies.
  // Zero is a real two-bit value, not a fifth neutral symbol; therefore this
  // apparatus cannot establish inheritance by itself. The four-payload gate
  // requires the balanced relay matter to drive every nonzero value physically.
  const std::optional<G1Motif> constructor = g1_integrated_two_bit_constructor_motif(0u);
  if (!body.has_value() || !constructor.has_value())
    return std::nullopt;

  struct Component {
    const G1Motif* motif = nullptr;
    std::string_view prefix;
    Int3 origin{};
    BasisPermutation permutation{0u, 1u, 2u, 3u};
  };
  const G1Motif reanchor = g1_reanchor_seed_motif();
  const G1Motif metabolism = g1_metabolic_work_waste_motif();
  const G1Motif turnover = g1_c0_period23_turnover_motif();
  const std::array<Component, 5> components{{
      {&*body, "body", {0, 0, 0}, {0u, 1u, 2u, 3u}},
      {&*constructor, "constructor", {0, 0, -1}, {0u, 1u, 3u, 2u}},
      {&reanchor, "reanchor", {-1, 0, 0}, {0u, 1u, 2u, 3u}},
      {&metabolism, "metabolism", {0, -1, 0}, {1u, 3u, 0u, 2u}},
      {&turnover, "turnover", {0, 0, 1}, {1u, 0u, 3u, 2u}},
  }};

  std::map<Int3, SiteWord, Int3Less> combined;
  G1Motif result;
  result.id = "successor_body_" + std::to_string(payload);
  result.proven_scope =
      "one 93-site language-blank authored material body that co-locates the proven flux chassis, "
      "two-locus heredity, fixed zero-default constructor apparatus, reanchor, metabolism/waste, "
      "and C0 turnover affordances; only the balanced relay varies across the four payload bodies, "
      "and their material compatibility is exact, while autonomous coupling, successor "
      "construction, serial descent, branching, selection, learning, and language remain unproved";

  for (const Component& component : components) {
    std::map<Int3, SiteWord, Int3Less> local;
    for (const G1MaterialConstraint& material : component.motif->material)
      if (!merge_word(local, material.coordinate, material_word(material)))
        return std::nullopt;
    if (!add_transformed_material(combined, local, component.origin, component.permutation))
      return std::nullopt;

    for (const G1PhysicalPort& port : component.motif->ports) {
      const std::optional<Int3> coordinate =
          transform_coordinate(port.coordinate, component.origin, component.permutation);
      if (!coordinate.has_value())
        return std::nullopt;
      result.ports.push_back({std::string(component.prefix) + "." + port.name, *coordinate,
                              transformed_word(port.observed_mask, component.permutation),
                              port.kind, port.tick});
    }
  }

  result.material.reserve(combined.size());
  for (const auto& [coordinate, word] : combined)
    result.material.push_back(exact_word(coordinate, word));
  if (result.material.size() != 93u)
    return std::nullopt;
  return result;
}

std::optional<G1Motif> g1_powered_producer_arm_motif(std::uint32_t value,
                                                     G1PoweredProducerFlags flags) {
  if (value > 1u)
    return std::nullopt;
  std::map<Int3, SiteWord, Int3Less> material;
  if (!add_powered_producer_arm(material, value, flags))
    return std::nullopt;

  G1Motif motif;
  motif.id = "powered_producer_arm_" + std::to_string(value);
  motif.proven_scope =
      "one language-blank flux-powered relay/receiver arm used by the verified dual-producer "
      "child transaction; an isolated arm is not a complete child, reproducer, or lineage";
  motif.material.reserve(material.size());
  for (const auto& [coordinate, word] : material)
    motif.material.push_back(exact_word(coordinate, word));
  return motif;
}

}  // namespace substrate::bcc32
