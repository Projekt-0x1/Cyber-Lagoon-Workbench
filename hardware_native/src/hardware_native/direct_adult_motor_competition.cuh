// Mid-include splice for substrate::direct_adult_core via direct_adult_core.cuh.
// Owns body-modulated motor candidate ranking and affect-preservation threshold
// restore (#1517). Requires DirectNode / DirectBoundaryPort already in scope.

// #1517: body-modulated value ranks competing motor candidates before raw tie-breakers.
DIRECT_ADULT_HD inline bool affect_motor_competition_score_precedes(
    std::int32_t left_value_q16, const DirectNode& left,
    const DirectBoundaryPort& left_port, std::uint32_t left_index,
    std::int32_t right_value_q16, const DirectNode& right,
    const DirectBoundaryPort& right_port, std::uint32_t right_index) {
  if (left_value_q16 != right_value_q16) return left_value_q16 > right_value_q16;
  if (left.activation_q16 != right.activation_q16) return left.activation_q16 > right.activation_q16;
  if (left.activity_ema_q16 != right.activity_ema_q16) return left.activity_ema_q16 > right.activity_ema_q16;
  if (left.credit_ema_q16 != right.credit_ema_q16) return left.credit_ema_q16 > right.credit_ema_q16;
  if (left_port.physical_route != right_port.physical_route) return left_port.physical_route < right_port.physical_route;
  if (left_port.channel != right_port.channel) return left_port.channel < right_port.channel;
  if (left_port.node != right_port.node) return left_port.node < right_port.node;
  return left_index < right_index;
}

// A negative body-consequence bias blocks wanting from buying its bounded 1/8
// pursuit-threshold discount for that same candidate. This restores only the
// pre-wanting threshold; it does not mint a veto or mutate any authority state.
DIRECT_ADULT_HD inline std::int32_t affect_preservation_base_threshold_q16(
    std::int32_t wanting_threshold_q16, std::int32_t affect_bias_q16,
    std::uint32_t live_pursuit) {
  if (affect_bias_q16 >= 0 || live_pursuit == 0u) return wanting_threshold_q16;
  const std::int32_t restored = wanting_threshold_q16 + kQ16One / 8;
  return restored > kQ16One ? kQ16One : restored;
}
