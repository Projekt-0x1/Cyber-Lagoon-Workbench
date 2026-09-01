#!/usr/bin/env python3
"""R1/R2: delayed contact can revise an earlier causal discourse group after later groups were emitted."""
from __future__ import annotations
import copy,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
from reference_partner_multisource_dispute_repair_verify import projection,relation_surfaces
Q=1<<16;EFFECT=0xA104;PARTNER=0xDC01;SOURCE=0xDC11;MOTOR_SOURCE=0xDC21;WORLD_SOURCE=0xDC31

def main():
    started=time.perf_counter();checks={};failure='';visible={};metrics={}
    try:
        with tempfile.TemporaryDirectory(prefix='foundry-delayed-earlier-group-') as directory:
            build_cache(directory);adult=load_mark(directory,'relational_surplus_recovered').adult
            leaf=adult.language_adult.leaf(100,(EFFECT,));rows=adult.causal_focus_rows(leaf.identity)
            baseline_surface,baseline_programs,baseline_groups=projection(adult,PARTNER)
            surfaces,receipts=adult.externalize_causal_groups(leaf.identity,MOTOR_SOURCE,PARTNER)
            group_widths=tuple(len(r.programs) for r in receipts)
            checks['paragraph_is_multiple_ordered_recent_actions']=(len(receipts)>=2 and sum(group_widths)==len(baseline_programs) and all(width>0 for width in group_widths) and len({pid for r in receipts for pid in r.programs})==len(baseline_programs))
            checks['all_group_actions_settle_before_delayed_partner_contact']=all(adult.settle_causal_dialogue_return(receipt,WORLD_SOURCE+i,Q,0,True) for i,receipt in enumerate(receipts))
            first_row=rows[0];first_program=receipts[0].programs[0];accepted,_reversed=relation_surfaces(adult,first_row,first_program);first_receipt=int(first_row[4])
            checks['opening_relation_surface_is_structurally_certified']=bool(accepted and first_program>0 and first_receipt>0)
            latest=max(adult.recent_causal_dialogue_actions.values(),key=lambda r:(int(r.born_tick),int(r.identity)))
            checks['opening_relation_is_not_in_latest_action']=(first_program not in latest.programs and latest.identity==receipts[-1].identity)
            first=adult.observe_authenticated_causal_dialogue_contact(accepted,SOURCE,channel=PARTNER)
            second=adult.observe_authenticated_causal_dialogue_contact(accepted,SOURCE,channel=PARTNER)
            after_surface,after_programs,after_groups=projection(adult,PARTNER);after_blob=b' '.join(s for s,_p in after_groups)
            checks['delayed_opening_acknowledgement_revises_earlier_group']=(first==1 and second==1 and adult.causal_dialogue_uptake_support(PARTNER,first_receipt)==2 and len(after_programs)==len(baseline_programs)-1 and accepted not in after_blob)
            other=type(adult).restore(copy.deepcopy(adult.checkpoint()));other_before=projection(other,0xDC02);other.observe_authenticated_causal_dialogue_contact(accepted,SOURCE,channel=0xDC02)
            checks['channel_mismatch_cannot_revise_unemitted_partner_context']=(projection(other,0xDC02)==other_before)
            checkpoint=copy.deepcopy(adult.checkpoint());restart=type(adult).restore(copy.deepcopy(checkpoint))
            checks['delayed_uptake_survives_checkpoint']=(restart.checkpoint()==checkpoint and projection(restart,PARTNER)==(after_surface,after_programs,after_groups))
            checks['bounded_reconciliation_window_only']=len(adult.recent_causal_dialogue_actions)<=8
            visible={'before':baseline_surface.decode(errors='replace'),'delayed_contact':accepted.decode(errors='replace'),'after':[s.decode(errors='replace') for s,_p in after_groups]}
            metrics={'recent_actions_examined':len(adult.recent_causal_dialogue_actions),'group_program_widths':group_widths,'baseline_programs':len(baseline_programs),'after_delayed_uptake_programs':len(after_programs),'persistent_uptake_rows':len(adult._causal_dialogue_uptake_evidence)}
    except Exception as exc:failure=f'{type(exc).__name__}:{exc}'
    failed=[name for name,passed in checks.items() if not passed]
    result={'schema':'cyber-lagoon.delayed-earlier-causal-group-uptake.v1','contract':'FOUNDRY_DELAYED_EARLIER_CAUSAL_GROUP_UPTAKE_'+('GREEN' if not failed and not failure else 'RED'),'pass':not failed and not failure,'reference_only':True,'runtime_llm':False,'mechanism_change':True,'language_phenotype_improved':not failed and not failure,'visible_language_gain':'PARTNER_CAN_ACKNOWLEDGE_AN_OPENING_CAUSAL_CLAUSE_AFTER_LATER_PARAGRAPH_GROUPS_AND_CHANGE_COMMON_GROUND','visible':visible,'metrics':metrics,'checks':checks,'failed':failed,'runtime_failure':failure,'remaining_red':['RECONCILIATION_BEYOND_BOUNDED_RECENT_ACTION_WINDOW','RECURSIVE_EMBEDDED_CAUSAL_CONTACT','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
