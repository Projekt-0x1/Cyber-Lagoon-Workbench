#ifndef HARDWARE_NATIVE_DIRECT_RELATIONAL_SEQUENCE_COMPOSITION_CUH
#define HARDWARE_NATIVE_DIRECT_RELATIONAL_SEQUENCE_COMPOSITION_CUH
#include <climits>
#include <cstdint>
#include <type_traits>
#if defined(__CUDACC__)
#define DIRECT_RELSEQ_HD __host__ __device__
#else
#define DIRECT_RELSEQ_HD
#endif
namespace substrate::direct_network {
inline constexpr std::uint32_t kDirectRelSeqMaxPieces = 32u;
inline constexpr std::uint32_t kDirectRelSeqMaxPorts = 12u;
inline constexpr std::uint32_t kDirectRelSeqMaxOutputUnits = 256u;
inline constexpr std::uint32_t kDirectRelSeqMaxContextFeatures = 8u;
inline constexpr std::uint32_t kDirectRelSeqMaxDepth = 16u;
enum class DirectRelSeqPieceKind : std::uint16_t { fixed_unit = 0u, port = 1u };
struct DirectRelSeqPiece {
  std::uint32_t value;
  std::uint16_t kind;
  std::uint16_t trim_right;
};
struct DirectRelSeqRecipe {
  std::uint64_t logical_recipe_id, revision_identity, relation_identity;
  DirectRelSeqPiece pieces[kDirectRelSeqMaxPieces];
  std::uint32_t piece_count, port_count;
  std::uint32_t context_features[kDirectRelSeqMaxContextFeatures];
  std::uint32_t context_feature_count, support, active;
};
struct DirectRelSeqPortBinding {
  std::uint32_t formal_port, unit_identity;
  std::uint64_t child_occurrence_identity;
};
struct DirectRelSeqOccurrence {
  std::uint64_t occurrence_identity, logical_recipe_id, revision_identity;
  DirectRelSeqPortBinding bindings[kDirectRelSeqMaxPorts];
  std::uint32_t binding_count;
  std::uint32_t context_features[kDirectRelSeqMaxContextFeatures];
  std::uint32_t context_feature_count;
};
struct DirectRelSeqOutput {
  std::uint32_t units[kDirectRelSeqMaxOutputUnits];
  std::uint32_t count, refused, depth_peak;
};
static_assert(std::is_trivially_copyable_v<DirectRelSeqRecipe>);
static_assert(std::is_trivially_copyable_v<DirectRelSeqOccurrence>);
DIRECT_RELSEQ_HD inline std::uint64_t direct_relseq_fold(std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
  return h ? h : 1u;
}
DIRECT_RELSEQ_HD inline std::uint64_t direct_relseq_recipe_revision_identity(
    const DirectRelSeqRecipe& r) {
  std::uint64_t h = direct_relseq_fold(0x72656c7365717631ull, r.logical_recipe_id);
  h = direct_relseq_fold(h, r.relation_identity);
  h = direct_relseq_fold(h, r.piece_count);
  h = direct_relseq_fold(h, r.port_count);
  for (std::uint32_t i = 0; i < r.piece_count; ++i) {
    h = direct_relseq_fold(h, r.pieces[i].value);
    h = direct_relseq_fold(h, r.pieces[i].kind);
    h = direct_relseq_fold(h, r.pieces[i].trim_right);
  }
  for (std::uint32_t i = 0; i < r.context_feature_count; ++i)
    h = direct_relseq_fold(h, r.context_features[i]);
  return h;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_recipe_valid(const DirectRelSeqRecipe& r) {
  if (!r.logical_recipe_id || !r.relation_identity || !r.active || !r.piece_count ||
      r.piece_count > kDirectRelSeqMaxPieces || r.port_count > kDirectRelSeqMaxPorts ||
      r.context_feature_count > kDirectRelSeqMaxContextFeatures ||
      r.revision_identity != direct_relseq_recipe_revision_identity(r))
    return false;
  for (std::uint32_t i = 0; i < r.piece_count; ++i) {
    auto k = static_cast<DirectRelSeqPieceKind>(r.pieces[i].kind);
    if (k == DirectRelSeqPieceKind::fixed_unit) {
      if (!r.pieces[i].value || r.pieces[i].trim_right)
        return false;
    } else if (k == DirectRelSeqPieceKind::port) {
      if (r.pieces[i].value >= r.port_count)
        return false;
    } else
      return false;
  }
  return true;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_context_has(const std::uint32_t* v, std::uint32_t n,
                                                       std::uint32_t f) {
  for (std::uint32_t i = 0; i < n; ++i)
    if (v[i] == f)
      return true;
  return false;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_copy_context(const std::uint32_t* src, std::uint32_t n,
                                                        std::uint32_t* dst, std::uint32_t* dn) {
  if (!dst || !dn || n > kDirectRelSeqMaxContextFeatures || (n && !src))
    return false;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (!src[i])
      return false;
    dst[i] = src[i];
  }
  *dn = n;
  return true;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_context_equal(const std::uint32_t* a, std::uint32_t na,
                                                         const std::uint32_t* b, std::uint32_t nb) {
  if (na != nb)
    return false;
  for (std::uint32_t i = 0; i < na; ++i)
    if (!direct_relseq_context_has(b, nb, a[i]))
      return false;
  return true;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_inherit_context(const DirectRelSeqOccurrence& child,
                                                           DirectRelSeqOccurrence* parent) {
  return parent &&
         direct_relseq_copy_context(child.context_features, child.context_feature_count,
                                    parent->context_features, &parent->context_feature_count);
}
DIRECT_RELSEQ_HD inline std::int32_t direct_relseq_candidate_score(
    const DirectRelSeqRecipe& r, const DirectRelSeqOccurrence& o) {
  if (!direct_relseq_recipe_valid(r) || o.binding_count != r.port_count)
    return INT32_MIN;
  std::int32_t s = static_cast<std::int32_t>(r.support);
  for (std::uint32_t i = 0; i < r.context_feature_count; ++i) {
    if (!direct_relseq_context_has(o.context_features, o.context_feature_count,
                                   r.context_features[i]))
      return INT32_MIN;
    s += 8;
  }
  return s;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_induce_flat_recipe(
    std::uint64_t logical, std::uint64_t relation, const std::uint32_t* observed,
    std::uint32_t observed_count, const std::uint32_t* port_units, std::uint32_t port_count,
    const std::uint32_t* ctx, std::uint32_t ctx_count, DirectRelSeqRecipe* out) {
  if (!out || !logical || !relation || !observed || !observed_count ||
      observed_count > kDirectRelSeqMaxPieces || port_count > kDirectRelSeqMaxPorts ||
      ctx_count > kDirectRelSeqMaxContextFeatures || (port_count && !port_units))
    return false;
  bool used[kDirectRelSeqMaxPieces]{};
  std::uint32_t pos[kDirectRelSeqMaxPorts]{};
  for (std::uint32_t p = 0; p < port_count; ++p) {
    bool found = false;
    for (std::uint32_t i = 0; i < observed_count; ++i)
      if (!used[i] && observed[i] == port_units[p]) {
        used[i] = true;
        pos[p] = i;
        found = true;
        break;
      }
    if (!found)
      return false;
  }
  DirectRelSeqRecipe r{};
  r.logical_recipe_id = logical;
  r.relation_identity = relation;
  r.port_count = port_count;
  r.support = 1u;
  r.active = 1u;
  r.context_feature_count = ctx_count;
  for (std::uint32_t i = 0; i < ctx_count; ++i) {
    if (!ctx || !ctx[i])
      return false;
    r.context_features[i] = ctx[i];
  }
  for (std::uint32_t i = 0; i < observed_count; ++i) {
    DirectRelSeqPiece x{};
    bool is_port = false;
    for (std::uint32_t p = 0; p < port_count; ++p)
      if (pos[p] == i) {
        x.kind = static_cast<std::uint16_t>(DirectRelSeqPieceKind::port);
        x.value = p;
        is_port = true;
        break;
      }
    if (!is_port) {
      if (!observed[i])
        return false;
      x.kind = static_cast<std::uint16_t>(DirectRelSeqPieceKind::fixed_unit);
      x.value = observed[i];
    }
    r.pieces[r.piece_count++] = x;
  }
  r.revision_identity = direct_relseq_recipe_revision_identity(r);
  *out = r;
  return true;
}
DIRECT_RELSEQ_HD inline const DirectRelSeqRecipe* direct_relseq_select_recipe(
    const DirectRelSeqRecipe* rs, std::uint32_t n, std::uint64_t relation,
    const DirectRelSeqOccurrence& o) {
  if (!rs || !relation)
    return nullptr;
  const DirectRelSeqRecipe* best = nullptr;
  std::int32_t best_score = INT32_MIN;
  bool tie = false;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (rs[i].relation_identity != relation)
      continue;
    auto score = direct_relseq_candidate_score(rs[i], o);
    if (score > best_score) {
      best = &rs[i];
      best_score = score;
      tie = false;
    } else if (score == best_score && score != INT32_MIN && best &&
               rs[i].logical_recipe_id != best->logical_recipe_id)
      tie = true;
  }
  return tie ? nullptr : best;
}
DIRECT_RELSEQ_HD inline const DirectRelSeqPortBinding* direct_relseq_binding(
    const DirectRelSeqOccurrence& o, std::uint32_t p) {
  for (std::uint32_t i = 0; i < o.binding_count; ++i)
    if (o.bindings[i].formal_port == p)
      return &o.bindings[i];
  return nullptr;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_append(DirectRelSeqOutput* out, std::uint32_t v) {
  if (!out || !v || out->count >= kDirectRelSeqMaxOutputUnits)
    return false;
  out->units[out->count++] = v;
  return true;
}
DIRECT_RELSEQ_HD inline bool direct_relseq_evaluate_occurrence(
    const DirectRelSeqRecipe* rs, std::uint32_t rn, const DirectRelSeqOccurrence* os,
    std::uint32_t on, std::uint64_t oid, DirectRelSeqOutput* out, std::uint32_t depth = 0u) {
  if (!rs || !os || !out || !oid || depth > kDirectRelSeqMaxDepth)
    return false;
  if (depth > out->depth_peak)
    out->depth_peak = depth;
  const DirectRelSeqOccurrence* o = nullptr;
  for (std::uint32_t i = 0; i < on; ++i)
    if (os[i].occurrence_identity == oid) {
      o = &os[i];
      break;
    }
  if (!o)
    return false;
  const DirectRelSeqRecipe* r = nullptr;
  for (std::uint32_t i = 0; i < rn; ++i)
    if (rs[i].logical_recipe_id == o->logical_recipe_id &&
        rs[i].revision_identity == o->revision_identity) {
      r = &rs[i];
      break;
    }
  if (!r || !direct_relseq_recipe_valid(*r) || o->binding_count != r->port_count)
    return false;
  for (std::uint32_t i = 0; i < r->piece_count; ++i) {
    auto& x = r->pieces[i];
    if (static_cast<DirectRelSeqPieceKind>(x.kind) == DirectRelSeqPieceKind::fixed_unit) {
      if (!direct_relseq_append(out, x.value))
        return false;
      continue;
    }
    auto* b = direct_relseq_binding(*o, x.value);
    if (!b)
      return false;
    auto before = out->count;
    if (b->child_occurrence_identity) {
      if (!direct_relseq_evaluate_occurrence(rs, rn, os, on, b->child_occurrence_identity, out,
                                             depth + 1u))
        return false;
    } else if (!direct_relseq_append(out, b->unit_identity))
      return false;
    if (x.trim_right > out->count - before)
      return false;
    out->count -= x.trim_right;
  }
  return true;
}
}  // namespace substrate::direct_network
#undef DIRECT_RELSEQ_HD
#endif
