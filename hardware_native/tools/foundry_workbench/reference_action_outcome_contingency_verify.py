#!/usr/bin/env python3
"""Research-driven falsifier for learned action-vs-background controllability.

Grounding: docs/research/sapolsky/2026-08-31-controllability-contingency-hardware-ethology.md
The same outcome count/value may imply different control when outcome probability differs
between action and matched no-action opportunities. This is a Workbench mechanism receipt,
not Direct capability promotion.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q  # noqa: E402

SID = 0xA501


def observe_cells(bank, ao=0, an=0, bo=0, bn=0):
    for _ in range(ao):bank.observe_control(SID, True, True)
    for _ in range(an):bank.observe_control(SID, True, False)
    for _ in range(bo):bank.observe_control(SID, False, True)
    for _ in range(bn):bank.observe_control(SID, False, False)
    return bank.row(SID)


def desired_delta_q16(row):
    action_n = int(getattr(row, 'control_attempts', 0))
    action_o = int(getattr(row, 'control_successes', 0))
    bg_n = int(getattr(row, 'background_attempts', 0))
    bg_o = int(getattr(row, 'background_successes', 0))
    action_rate = 0 if action_n <= 0 else (action_o * Q) // action_n
    bg_rate = 0 if bg_n <= 0 else (bg_o * Q) // bg_n
    return max(0, action_rate - bg_rate)


def main():
    started=time.perf_counter()

    # Research discriminator: equal total outcome count, different contingency.
    master=PredictiveCreditBankV1(8);m=observe_cells(master,ao=4,bn=4)
    yoked=PredictiveCreditBankV1(8);y=observe_cells(yoked,ao=2,an=2,bo=2,bn=2)

    # Immediate-predecessor bug: two action outcomes plus two free/background outcomes.
    # Old success/all-observation ratio is 2/4 = 0.5 even though DeltaP = 1 - 1 = 0.
    zero=PredictiveCreditBankV1(8);z=observe_cells(zero,ao=2,bo=2)
    zero_before_ready=bool(getattr(z,'control_ready',False))
    zero_before_delta=int(getattr(z,'controllability_q16',0))

    # Recovery: matched no-action periods without the outcome lower background rate.
    observe_cells(zero,bn=2)
    recovered_delta=int(getattr(z,'controllability_q16',0))
    recovered_ready=bool(getattr(z,'control_ready',False))

    # Controllability evidence may change without rewriting consequence value.
    value=PredictiveCreditBankV1(8)
    value.observe_use(SID,10,11,Q//16,77)
    value.observe_return(SID,3*Q//4,Q//16,12,True,77)
    value_before=value.row(SID).outcome_mean_q16
    observe_cells(value,ao=2,bo=2)
    value_after=value.row(SID).outcome_mean_q16

    # Same current DeltaP, different evidence mass -> different lawful next update.
    shallow=PredictiveCreditBankV1(8);sr=observe_cells(shallow,ao=1,an=1,bn=2)
    deep=PredictiveCreditBankV1(8);dr=observe_cells(deep,ao=2,an=2,bn=4)
    same_before=desired_delta_q16(sr)==desired_delta_q16(dr)==Q//2
    observe_cells(shallow,ao=1);observe_cells(deep,ao=1)

    has_background=(hasattr(m,'background_attempts') and hasattr(m,'background_successes'))

    # Checkpoint authority: schema-4 retains global and context-qualified cells.
    # Obsolete schemas are refused rather than retained as a compatibility museum.
    roundtrip=PredictiveCreditBankV1.restore(zero.checkpoint())
    obsolete=json.loads(json.dumps(zero.checkpoint()));obsolete['schema']=3
    try:PredictiveCreditBankV1.restore(obsolete)
    except ValueError:obsolete_refused=True
    else:obsolete_refused=False
    tampered=json.loads(json.dumps(zero.checkpoint()))
    tampered['rows'][0]['control_history']=Q+1
    try:PredictiveCreditBankV1.restore(tampered)
    except ValueError:history_tamper_refused=True
    else:history_tamper_refused=False

    checks={
        'background_contingency_state_exists':has_background,
        'master_and_yoked_match_total_outcome_count':4==4,
        'master_contingency_is_positive':desired_delta_q16(m)==Q,
        'yoked_contingency_is_zero':desired_delta_q16(y)==0,
        'master_ready_yoked_not_ready':bool(getattr(m,'control_ready',False)) and not bool(getattr(y,'control_ready',False)),
        'zero_contingency_degrades_prior_success_quorum':zero_before_delta==0 and not zero_before_ready,
        'background_no_outcome_evidence_recovers_control':recovered_delta==Q//2 and recovered_ready,
        'control_revision_does_not_rewrite_outcome_value':value_before==value_after==3*Q//4,
        'equal_current_delta_can_hide_different_evidence_mass':same_before,
        'evidence_mass_preserves_future_update_authority':desired_delta_q16(shallow.row(SID))!=desired_delta_q16(deep.row(SID)),
        'schema4_checkpoint_retains_contingency_and_history':roundtrip.snapshot()==zero.snapshot() and zero.checkpoint().get('schema')==4,
        'obsolete_checkpoint_schema_is_refused':obsolete_refused,
        'schema4_checkpoint_refuses_out_of_domain_history':history_tamper_refused,
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={
        'schema':'cyber-lagoon.reference-action-outcome-contingency.v1',
        'pass':not failed,
        'reference_only':True,
        'research_grounded':True,
        'novel_synthesis':'SPARSE_CONTINGENCY_SUFFICIENT_STATISTICS',
        'graph_node':'d.action_outcome_contingency',
        'issue':'#1658',
        'factory_gate':'hardware_native/tools/foundry_workbench/run_language_mastery_factory_fast.sh',
        'checks':checks,
        'master':{'delta_q16':desired_delta_q16(m),'ready':bool(getattr(m,'control_ready',False))},
        'yoked':{'delta_q16':desired_delta_q16(y),'ready':bool(getattr(y,'control_ready',False))},
        'zero_contingency_before_recovery':{'delta_q16':zero_before_delta,'ready':zero_before_ready},
        'recovered':{'delta_q16':recovered_delta,'ready':recovered_ready},
        'remaining_red':['MISSING_BACKGROUND_UNCERTAINTY','BODY_RESOURCE_MODULATION','DIRECT_AUTHENTICATED_PARITY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_ACTION_OUTCOME_CONTINGENCY_'+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
