#!/usr/bin/env python3
"""R1/R2: causal event boundaries become separate resident discourse trajectories."""
from __future__ import annotations

import copy
import json
import tempfile
import time

from life_function_factory_v1 import build_cache,load_mark
from reference_life_extension_history_matrix_v1 import A,B,C
from reference_mathematical_adult_workbench_v1 import MathematicalWorkbenchAdultV1
from reference_partner_uptake_causal_paragraph_verify import A as UPTAKE_A,teach_prefix

EFFECT=0xA104
Q=1<<16


def clone(adult):
    return type(adult).restore(copy.deepcopy(adult.checkpoint()))


def projection(adult,channel=0):
    leaf=adult.language_adult.leaf(100,(EFFECT,))
    surface,programs=adult.compose_causal_component(leaf.identity,channel=channel)
    groups=adult.compose_causal_groups(leaf.identity,channel=channel)
    return bytes(surface),tuple(programs),tuple((bytes(s),tuple(p)) for s,p in groups)


def main():
    started=time.perf_counter();checks={};failure='';metrics={};visible={}
    try:
        with tempfile.TemporaryDirectory(prefix='foundry-paragraph-trajectory-') as directory:
            manifest=build_cache(directory)
            base=load_mark(directory,'causal_discourse_form_diversity').adult
            loaded=load_mark(directory,'open_state_prompt_loaded').adult
            recovered_runtime=load_mark(directory,'open_state_prompt_recovered');recovered=recovered_runtime.adult

            base_surface,base_programs,base_groups=projection(clone(base))
            flat=tuple(pid for _surface,programs in base_groups for pid in programs)
            observer_paragraphs=b'\n\n'.join(surface for surface,_programs in base_groups)
            checks['branched_causal_closure_splits_on_structural_discontinuity']=(
                len(base_programs)==10 and len(base_groups)==4
                and tuple(len(programs) for _surface,programs in base_groups)==(6,1,1,2)
                and flat==base_programs)
            checks['certified_sibling_coordination_stays_local_to_structural_group']=(
                len(base_groups[-1][1])==2
                and base_groups[-1][0].count(b'warm air dries the soil')==1
                and b'plants need more water and the greenhouse needs more humidity' in base_groups[-1][0])
            checks['grouping_preserves_every_selected_program_once']=(
                len(flat)==len(set(flat))==len(base_programs)
                and sum(surface.count(b'.') for surface,_programs in base_groups)>=9)
            checks['observer_paragraph_render_is_not_resident_newline_content']=(
                b'\n\n' in observer_paragraphs and b'\n\n' not in base_surface
                and observer_paragraphs.replace(b'\n\n',b' ')!=b'' )

            chain=clone(base);leaf=chain.language_adult.leaf(100,(EFFECT,));rows=chain.causal_chain_rows(leaf.identity)
            branch=rows[-1];receipt=int(branch[4]);ecology=chain.language_adult.world_causal_learning.ecology.pending[receipt]
            branch_sources=tuple(sorted({int(row.source) for row in ecology.evidence if row.active}))
            for source in branch_sources:chain.language_adult.world_causal_learning.withdraw_source(source)
            chain_surface,chain_programs,chain_groups=projection(chain)
            checks['focal_sibling_source_lesion_deoptimizes_only_local_group']=(
                len(branch_sources)>=3 and len(chain_programs)==9
                and tuple(len(programs) for _surface,programs in chain_groups)==(6,1,1,1)
                and chain_groups[:3]==base_groups[:3]
                and b'plants need more water' in chain_groups[-1][0]
                and b'greenhouse needs more humidity' not in chain_surface)

            loaded_rows={partner:projection(clone(loaded),partner) for partner in (A,B,C)}
            la,lb,lc=(loaded_rows[A],loaded_rows[B],loaded_rows[C])
            checks['matched_world_and_load_partner_history_changes_group_reach']=(
                tuple(map(len,(la[1],lb[1],lc[1])))==(10,2,1)
                and tuple(map(len,(la[2],lb[2],lc[2])))==(4,1,1)
                and loaded.language_adult.world_causal_learning.current_resolutions()==
                    clone(loaded).language_adult.world_causal_learning.current_resolutions())
            checks['partner_truncation_preserves_self_contained_first_group']=(
                lb[2] and lc[2] and lb[2][0][0].startswith(b'Because ')
                and lc[2][0][0].startswith(b'Because ')
                and b'plants need more water' not in lb[2][0][0]
                and b'plants need more water' not in lc[2][0][0])

            recovered_rows={partner:projection(clone(recovered),partner) for partner in (A,B,C)}
            checks['quiet_body_recovery_restores_structural_groups_without_reteaching']=(
                all(len(row[1])==10 and tuple(len(p) for _s,p in row[2])==(6,1,1,2)
                    for row in recovered_rows.values())
                and recovered.language_adult.slow_resource_history.pressure_q16()==0)

            body_cut=clone(loaded);world_before=copy.deepcopy(body_cut.language_adult.world_causal_learning.checkpoint())
            body_cut.language_adult.slow_resource_history.lesion_history()
            _bc_surface,bc_programs,bc_groups=projection(body_cut,C)
            checks['body_history_lesion_restores_groups_not_world_truth']=(
                len(bc_programs)==10 and tuple(len(p) for _s,p in bc_groups)==(6,1,1,2)
                and body_cut.language_adult.world_causal_learning.checkpoint()==world_before)

            staged=clone(base);leaf=staged.language_adult.leaf(100,(EFFECT,));surfaces,receipts=staged.externalize_causal_groups(leaf.identity,0xEE01,0)
            staged_blob=json.dumps(staged.checkpoint(),sort_keys=True).lower()
            checks['each_structural_group_is_own_resident_motor_action']=(
                tuple(surfaces)==tuple(surface for surface,_programs in base_groups)
                and len(receipts)==4 and tuple(len(r.programs) for r in receipts)==(6,1,1,2)
                and len(staged.pending_causal_dialogue_actions)==4
                and all(r.identity in staged.pending_causal_dialogue_actions for r in receipts))
            checks['checkpoint_keeps_pending_action_witness_not_completed_paragraph']=(
                'paragraph' not in staged_blob and '\\n\\n' not in staged_blob
                and all(surface.decode(errors='ignore').lower() not in staged_blob for surface in surfaces))
            restored=MathematicalWorkbenchAdultV1.restore(copy.deepcopy(staged.checkpoint()))
            checks['checkpoint_replays_group_plan_exactly']=(
                projection(restored)[2]==base_groups
                and set(restored.pending_causal_dialogue_actions)==set(staged.pending_causal_dialogue_actions))
            settled=all(restored.settle_causal_dialogue_return(
                restored.pending_causal_dialogue_actions[r.identity],0xEF00+i,Q,0,True)
                for i,r in enumerate(receipts))
            checks['independent_return_settles_each_group_through_existing_credit_boundary']=(
                settled and not restored.pending_causal_dialogue_actions
                and all(r.identity in restored.recent_causal_dialogue_actions for r in receipts))

            uptake_runtime=type(recovered_runtime).restore(
                recovered_runtime.program,copy.deepcopy(recovered_runtime.checkpoint()))
            acknowledged=teach_prefix(uptake_runtime,UPTAKE_A,True)
            uptake_adult=clone(uptake_runtime.adult)
            uptake_surface,uptake_programs,uptake_groups=projection(uptake_adult,UPTAKE_A)
            uptake_flat=tuple(pid for _surface,programs in uptake_groups for pid in programs)
            checks['partner_uptake_replans_nonprefix_frontier_into_valid_groups']=(
                len(uptake_programs)==9 and tuple(len(p) for _s,p in uptake_groups)==(5,1,1,2)
                and uptake_flat==uptake_programs
                and acknowledged not in b' '.join(surface for surface,_programs in uptake_groups)
                and b'plants need more water and the greenhouse needs more humidity' in uptake_groups[-1][0])
            uptake_leaf=uptake_adult.language_adult.leaf(100,(EFFECT,))
            uptake_surfaces,uptake_receipts=uptake_adult.externalize_causal_groups(
                uptake_leaf.identity,0xF911,UPTAKE_A)
            uptake_restored=MathematicalWorkbenchAdultV1.restore(copy.deepcopy(uptake_adult.checkpoint()))
            checks['uptake_filtered_groups_externalize_and_checkpoint_with_correct_rows']=(
                tuple(uptake_surfaces)==tuple(surface for surface,_programs in uptake_groups)
                and tuple(len(r.programs) for r in uptake_receipts)==(5,1,1,2)
                and uptake_restored.compose_causal_groups(uptake_leaf.identity,UPTAKE_A)==uptake_groups
                and set(uptake_restored.pending_causal_dialogue_actions)==
                    {r.identity for r in uptake_receipts})

            visible={
                'before_serial':base_surface.decode(errors='replace'),
                'after_groups':[surface.decode(errors='replace') for surface,_programs in base_groups],
                'observer_rendered_paragraphs':observer_paragraphs.decode(errors='replace'),
                'loaded_partner_group_counts':{str(k):len(v[2]) for k,v in loaded_rows.items()},
                'recovered_partner_group_counts':{str(k):len(v[2]) for k,v in recovered_rows.items()},
                'after_partner_uptake_groups':[surface.decode(errors='replace') for surface,_programs in uptake_groups],
            }
            metrics={
                'curriculum_events':manifest['events'],
                'selected_programs':len(base_programs),
                'structural_groups':len(base_groups),
                'group_program_widths':[len(programs) for _surface,programs in base_groups],
                'serial_bytes':len(base_surface),
                'group_bytes':[len(surface) for surface,_programs in base_groups],
                'pending_actions_after_group_externalization':len(receipts),
                'persistent_paragraph_objects':0,
            }
    except Exception as exc:
        failure=f'{type(exc).__name__}:{exc}'
    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.sapolsky-causal-paragraph-trajectory.v1',
        'contract':'FOUNDRY_SAPOLSKY_CAUSAL_PARAGRAPH_TRAJECTORY_'+('GREEN' if not failed and not failure else 'RED'),
        'pass':not failed and not failure,
        'reference_only':True,
        'mechanism_change':True,
        'language_phenotype_improved':not failed and not failure,
        'visible_language_gain':'STRUCTURAL_EVENT_GROUPING_WITH_LOCAL_CERTIFIED_SIBLING_COORDINATION',
        'visible':visible,'metrics':metrics,'checks':checks,'failed':failed,'runtime_failure':failure,
        'remaining_red':['THREE_PLUS_SIBLING_COORDINATION','ADVERSE_UPTAKE_REVERSAL','OPEN_ENDED_QUD_GROUPING','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
