#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WB="$ROOT/hardware_native/tools/foundry_workbench"
cd "$ROOT"

python3 -m py_compile \
  "$WB/reference_developmental_predictive_amplifier_v1.py" \
  "$WB/reference_developmental_predictive_amplifier_v2.py" \
  "$WB/reference_developmental_predictive_amplifier_v3.py" \
  "$WB/reference_recursive_self_culture_control_v1.py" \
  "$WB/reference_recursive_self_culture_control_v2.py" \
  "$WB/reference_recursive_self_culture_control_v3.py" \
  "$WB/reference_recursive_policy_metacontrol_v1.py" \
  "$WB/reference_recursive_policy_metacontrol_v2.py" \
  "$WB/reference_recursive_policy_metacontrol_v3.py" \
  "$WB/reference_recursive_causal_experiment_v1.py" \
  "$WB/reference_recursive_causal_experiment_v2.py" \
  "$WB/reference_recursive_causal_experiment_v3.py" \
  "$WB/reference_recursive_causal_experiment_v4.py" \
  "$WB/reference_recursive_experiment_strategy_v1.py" \
  "$WB/reference_recursive_experiment_strategy_v2.py" \
  "$WB/reference_recursive_experiment_policy_v2.py" \
  "$WB/reference_recursive_experiment_policy_v3.py" \
  "$WB/reference_multiaxis_policy_branch_v2.py" \
  "$WB/reference_action_outcome_decomposition_v1.py" \
  "$WB/reference_source_prediction_calibration_v1.py" \
  "$WB/reference_recursive_causal_regime_v1.py" \
  "$WB/reference_recursive_context_partition_v2.py" \
  "$WB/reference_recursive_contextual_owners_v1.py" \
  "$WB/reference_recursive_partner_access_v1.py" \
  "$WB/reference_recursive_partner_access_v2.py" \
  "$WB/reference_reason_action_recommendation_v1.py" \
  "$WB/reference_organism_v2_consolidation_v1.py" \
  "$WB/reference_organism_v2_selfculture_v1.py" \
  "$WB/reference_organism_v2_selfculture_v2.py" \
  "$WB/reference_organism_v2_execution_v3.py" \
  "$WB/reference_organism_v2_metacontrol_v1.py" \
  "$WB/reference_organism_v2_causal_experiment_v1.py" \
  "$WB/reference_organism_v2_experiment_strategy_v1.py" \
  "$WB/reference_organism_v2_experiment_policy_v2.py" \
  "$WB/reference_organism_v2_latent_regime_v1.py" \
  "$WB/reference_organism_v2_overloaded_context_v1.py" \
  "$WB/reference_organism_v2_category_split_v1.py" \
  "$WB/reference_organism_v2_noncompensatory_v1.py" \
  "$WB/reference_organism_v2_outcome_semantics_v1.py" \
  "$WB/reference_organism_v2_consequence_split_v1.py" \
  "$WB/reference_organism_v2_source_prediction_v1.py" \
  "$WB/reference_organism_v2.py" \
  "$WB/reference_causal_reason_experiment_integration_verify.py" \
  "$WB/reference_experiment_strategy_integration_verify.py" \
  "$WB/reference_experiment_policy_integration_verify.py" \
  "$WB/reference_latent_causal_regime_integration_verify.py" \
  "$WB/reference_partner_access_category_verify.py" \
  "$WB/reference_action_outcome_category_integration_verify.py" \
  "$WB/reference_source_prediction_category_integration_verify.py" \
  "$WB/reference_prospective_execution_valence_integration_verify.py" \
  "$WB/reference_endogenous_prospection_verify.py" \
  "$WB/reference_life_function_curriculum_v1.py" \
  "$WB/reference_life_extension_relational_productive_surplus_v1.py"

python3 "$WB/reference_causal_reason_experiment_integration_verify.py"
python3 "$WB/reference_experiment_strategy_integration_verify.py"
python3 "$WB/reference_experiment_policy_integration_verify.py"
python3 "$WB/reference_latent_causal_regime_integration_verify.py"
python3 "$WB/reference_partner_access_category_verify.py"
python3 "$WB/reference_action_outcome_category_integration_verify.py"
python3 "$WB/reference_source_prediction_category_integration_verify.py"
python3 "$WB/reference_prospective_execution_valence_integration_verify.py"
python3 "$WB/reference_organism_frontier_curriculum_verify.py"
bash "$WB/run_life_function_curriculum_fast.sh"
printf '%s\n' 'FOUNDRY_FRONTIER_UNION_GREEN'
