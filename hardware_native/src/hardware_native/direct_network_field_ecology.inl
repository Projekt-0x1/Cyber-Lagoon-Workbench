// Included inside direct_network_life_function.cu's anonymous namespace after
// the field potential primitives are defined: the first-order spatial term of
// the developmental field ecology (#1289 b.local_fields).
//
// The canonical object is the authored falloff relation itself, and its
// gradient is exact rather than sampled: influence =
// decayed_strength * (radius3 - manhattan) / radius3 is piecewise linear in
// every coordinate, so inside reach
//   d influence / d coord_a = -decayed_strength * sign(coord_a - center_a) / radius3
// negated for repulsion and inhibition exactly as field_influence_q16 negates
// their value. A growth cone at `cone_origin` extending toward `growth_site`
// reads the gradient AT THE SITE projected along the cone direction:
//   tilt = grad(site) . (site - cone_origin).
// Positive tilt therefore means "extending uphill", toward higher carrier
// concentration; attraction draws cones up-gradient, repulsion and inhibition
// push them down-gradient. Multiplying before the single division keeps the
// fixed-point exact for far-centred carriers whose per-unit slope would
// otherwise truncate to zero.

DIRECT_NETWORK_HD inline bool field_gradient_participates(const FieldBlock& field) {
  return ((field.polarity /
           (kDevelopmentFieldKindCount * kFieldDecayClassCount *
            kFieldGainParticipationCount)) %
          kFieldGradientParticipationCount) != 0u;
}

DIRECT_NETWORK_HD inline std::int64_t field_gradient_tilt_q16(
    const FieldBlock& field, const std::int32_t growth_site[3],
    const std::int32_t cone_origin[3], std::uint32_t chemotype,
    std::uint32_t logical_tick) {
  if (!field_gradient_participates(field)) return 0;
  if (logical_tick < field.begin_tick) return 0;
  if (field.end_tick != 0u && logical_tick >= field.end_tick) return 0;
  if (!chemistry_matches(chemotype, field.require_mask, field.require_value)) return 0;
  const std::uint64_t radius = static_cast<std::uint64_t>(field.radius) * 3ull;
  if (radius == 0u) return 0;
  std::uint64_t distance = 0u;
  std::int64_t projection = 0;
  for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
    const std::int64_t delta = static_cast<std::int64_t>(growth_site[axis]) -
                               static_cast<std::int64_t>(field.center[axis]);
    const std::int64_t magnitude = delta < 0 ? -delta : delta;
    distance += static_cast<std::uint64_t>(magnitude);
    if (magnitude != 0) {
      projection += (delta < 0 ? -1 : 1) * (static_cast<std::int64_t>(growth_site[axis]) -
                                            static_cast<std::int64_t>(cone_origin[axis]));
    }
  }
  if (distance > radius) return 0;
  std::int64_t strength = field_decayed_strength_q16(field, logical_tick);
  const DevelopmentFieldKind kind = field_kind(field);
  if (kind == DevelopmentFieldKind::repel || kind == DevelopmentFieldKind::inhibition) {
    strength = -strength;
  }
  return (-strength * projection) / static_cast<std::int64_t>(radius);
}

DIRECT_NETWORK_HD inline std::int64_t slot_gradient_tilt_q16(
    const GammaV1& gamma, std::uint32_t field_index, const std::int32_t growth_site[3],
    const std::int32_t cone_origin[3], std::uint32_t chemotype,
    std::uint32_t logical_tick) {
  if (field_index >= gamma.header.field_count) return 0;  // kInvalidField included
  return field_gradient_tilt_q16(gamma.fields[field_index], growth_site, cone_origin,
                                 chemotype, logical_tick);
}

// Same canonical bound set and weights as combined_developmental_score_q16:
// the gradient is the first-order term of exactly that landscape.
DIRECT_NETWORK_HD inline std::int32_t combined_gradient_tilt_q16(
    const GammaV1& gamma, const TerritoryPlan& plan, const std::int32_t growth_site[3],
    const std::int32_t cone_origin[3], std::uint32_t logical_tick) {
  std::int64_t tilt = 0;
  for (std::uint32_t i = 0u; i < plan.bound_field_count; ++i) {
    const std::uint32_t field_index = plan.bound_field_indices[i];
    std::int64_t contribution = slot_gradient_tilt_q16(
        gamma, field_index, growth_site, cone_origin, plan.chemotype, logical_tick);
    const DevelopmentFieldKind kind = field_kind(gamma.fields[field_index]);
    if (kind == DevelopmentFieldKind::repair) continue;
    if (kind == DevelopmentFieldKind::maturation ||
        kind == DevelopmentFieldKind::inhibition)
      contribution /= 4;
    tilt += contribution;
  }
  return clamp_i32(static_cast<std::int32_t>(tilt), -(8 << 16), 8 << 16);
}
