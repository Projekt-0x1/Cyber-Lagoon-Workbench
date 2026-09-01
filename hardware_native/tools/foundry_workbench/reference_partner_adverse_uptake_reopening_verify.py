#!/usr/bin/env python3
"""R1/R2: adverse partner contact reopens only the disputed accepted causal claim."""
from __future__ import annotations
import copy,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
Q=1<<16;EFFECT=0xA104;PARTNER=0xDA01;OTHER_PARTNER=0xDA02;MOTOR_SOURCE=0xDA11;WORLD_SOURCE=0xDA21

def clone(adult):return type(adult).restore(copy.deepcopy(adult.checkpoint()))
def projection(adult,channel):
    leaf=adult.language_adult.leaf(100,(EFFECT,));surface,programs=adult.compose_causal_component(leaf.identity,channel=channel);groups=adult.compose_causal_groups(leaf.identity,channel=channel)
    return bytes(surface),tuple(programs),tuple((bytes(s),tuple(p)) for s,p in groups)
def relation_surfaces(adult,row,program,factor):
    factor=int(factor);accepted=bytes(adult._causal_self_contained_surface(row,factor) or b'');pieces=adult.language.historical_span_pieces(factor);orientation=adult.language_adult.world_causal_learning.grounding.orientation(factor);cause,effect=int(row[2]),int(row[3]);children=((effect,cause) if orientation>0 else (cause,effect));reversed_surface=(bytes(adult.language_adult._render_pieces(pieces,tuple(tuple(adult.language_adult._leaf_surface(child)) for child in reversed(children)))) if pieces else b'');return accepted,reversed_surface

def verify_loaded(subject):
    started=time.perf_counter();checks={};failure='';visible={};metrics={}
    try:
        if True:
            adult=clone(subject);leaf=adult.language_adult.leaf(100,(EFFECT,));world_before=copy.deepcopy(adult.language_adult.world_causal_learning.checkpoint());language_before=copy.deepcopy(adult.language.checkpoint())
            baseline_surface,baseline_programs,baseline_groups=projection(clone(adult),PARTNER);surfaces,receipts=adult.externalize_causal_groups(leaf.identity,MOTOR_SOURCE,PARTNER)
            rows=adult.causal_message_rows(leaf.identity)
            checks['resident_group_action_precedes_partner_revision']=(bool(receipts) and sum(len(r.programs) for r in receipts)==len(rows)==len(baseline_programs) and all(r.programs for r in receipts))
            checks['independent_world_returns_settle_without_partner_semantics']=all(adult.settle_causal_dialogue_return(receipt,WORLD_SOURCE+i,Q,0,True) for i,receipt in enumerate(receipts))
            causes={int(row[2]) for row in rows};by_cause={}
            for row in rows:by_cause.setdefault(int(row[2]),[]).append(row)
            terminal_pairs=[]
            for group in by_cause.values():
                terminal=tuple(row for row in group if int(row[3]) not in causes)
                terminal_pairs.extend((terminal[left],terminal[right]) for left in range(len(terminal)) for right in range(left+1,len(terminal)))
            sibling_rows=min(terminal_pairs,key=lambda pair:tuple(sorted(int(row[4]) for row in pair))) if terminal_pairs else ()
            program_coordinates={}
            for receipt in receipts:
                for program,factor in zip(receipt.programs,receipt.factors):
                    members=adult._causal_program_members(program,factor);orientation=adult.language_adult.world_causal_learning.grounding.orientation(factor)
                    if len(members)==2 and orientation:
                        cause,effect=((members[1],members[0]) if orientation>0 else members);program_coordinates[(int(cause),int(effect))]=(int(program),int(factor))
            sibling_programs=tuple(program_coordinates.get((int(row[2]),int(row[3])),(0,0)) for row in sibling_rows);accepted=[];reversed_surfaces=[]
            for row,(program,factor) in zip(sibling_rows,sibling_programs):good,bad=relation_surfaces(adult,row,program,factor);accepted.append(good);reversed_surfaces.append(bad)
            water,humidity=accepted;_water_reversed,humidity_reversed=reversed_surfaces
            water_effect,humidity_effect=tuple(bytes(adult.language_adult._leaf_surface(int(row[3]))) for row in sibling_rows)
            checks['heldout_sibling_surfaces_are_certified_and_distinct']=(all(accepted) and humidity_reversed and humidity_reversed!=humidity and sibling_rows[0][2]==sibling_rows[1][2])
            checks['two_acceptances_each_change_partner_history']=all(adult.observe_authenticated_causal_dialogue_contact(surface,PARTNER)==1 and adult.observe_authenticated_causal_dialogue_contact(surface,PARTNER)==1 for surface in accepted)
            accepted_surface,accepted_programs,accepted_groups=projection(adult,PARTNER);accepted_blob=b' '.join(surface for surface,_programs in accepted_groups)
            checks['accepted_siblings_are_compressed_for_only_that_partner']=(len(accepted_programs)==len(baseline_programs)-2 and all(surface not in accepted_blob for surface in accepted))
            other_surface,other_programs,other_groups=projection(adult,OTHER_PARTNER)
            checks['matched_other_partner_retains_full_common_cause_closure']=(len(other_programs)==len(baseline_programs) and any(water_effect in group_surface and humidity_effect in group_surface for group_surface,_programs in other_groups))
            dispute_changed=(adult.observe_authenticated_causal_dialogue_contact(humidity_reversed,PARTNER),adult.observe_authenticated_causal_dialogue_contact(humidity_reversed,PARTNER));disputed_surface,disputed_programs,disputed_groups=projection(adult,PARTNER);disputed_blob=b' '.join(surface for surface,_programs in disputed_groups);humidity_receipt=int(sibling_rows[1][4]);water_receipt=int(sibling_rows[0][4])
            checks['adverse_contact_reopens_only_the_disputed_accepted_sibling']=(dispute_changed==(0,0) and adult.causal_dialogue_uptake_support(PARTNER,water_receipt)==2 and adult.causal_dialogue_dispute_support(PARTNER,water_receipt)==0 and adult.causal_dialogue_uptake_support(PARTNER,humidity_receipt)==2 and adult.causal_dialogue_dispute_support(PARTNER,humidity_receipt)==2 and len(disputed_programs)==len(accepted_programs)+1 and water_effect not in disputed_blob and humidity_effect in disputed_blob)
            checks['revision_changes_discourse_not_world_truth_or_language_competence']=(adult.language_adult.world_causal_learning.checkpoint()==world_before and adult.language.checkpoint()==language_before)
            disputed_cp=copy.deepcopy(adult.checkpoint());restart=type(adult).restore(copy.deepcopy(disputed_cp))
            checks['partner_specific_reopening_survives_checkpoint_without_transcript']=(restart.checkpoint()==disputed_cp and projection(restart,PARTNER)==(disputed_surface,disputed_programs,disputed_groups) and projection(restart,OTHER_PARTNER)==(other_surface,other_programs,other_groups))
            repair_changed=restart.observe_authenticated_causal_dialogue_contact(humidity,PARTNER);repaired_surface,repaired_programs,repaired_groups=projection(restart,PARTNER);repaired_blob=b' '.join(surface for surface,_programs in repaired_groups)
            checks['matched_correct_contact_clears_dispute_and_recompresses_claim']=(repair_changed==1 and restart.causal_dialogue_dispute_support(PARTNER,humidity_receipt)==0 and len(repaired_programs)==len(accepted_programs) and humidity_effect not in repaired_blob)
            withdrawn=clone(restart);withdrawn.language.withdraw_source(PARTNER);_withdrawn_surface,withdrawn_programs,withdrawn_groups=projection(withdrawn,PARTNER);withdrawn.language.restore_source(PARTNER)
            checks['source_lesion_reopens_all_partner_claims_and_restoration_recovers_focus']=(len(withdrawn_programs)==len(baseline_programs) and projection(withdrawn,PARTNER)==(repaired_surface,repaired_programs,repaired_groups))
            checks['revision_state_and_work_are_relation_local_and_bounded']=(len(restart._causal_dialogue_uptake_evidence)==2 and len(restart._causal_dialogue_dispute_evidence)==0)
            visible={'baseline':baseline_surface.decode(errors='replace'),'after_acceptance':[s.decode(errors='replace') for s,_p in accepted_groups],'adverse_contact':humidity_reversed.decode(errors='replace'),'after_dispute':[s.decode(errors='replace') for s,_p in disputed_groups],'after_repair':[s.decode(errors='replace') for s,_p in repaired_groups]}
            metrics={'baseline_programs':len(baseline_programs),'accepted_programs':len(accepted_programs),'disputed_programs':len(disputed_programs),'repaired_programs':len(repaired_programs),'persistent_uptake_rows':len(restart._causal_dialogue_uptake_evidence),'persistent_dispute_rows_after_repair':len(restart._causal_dialogue_dispute_evidence)}
    except Exception as exc:failure=f'{type(exc).__name__}:{exc}'
    failed=[name for name,passed in checks.items() if not passed];result={'schema':'cyber-lagoon.partner-adverse-uptake-reopening.v1','contract':'FOUNDRY_PARTNER_ADVERSE_UPTAKE_REOPENING_'+('GREEN' if not failed and not failure else 'RED'),'pass':not failed and not failure,'reference_only':True,'runtime_llm':False,'mechanism_change':True,'language_phenotype_improved':not failed and not failure,'visible_language_gain':'PARTNER_SPECIFIC_CONTRADICTION_REOPENS_ONLY_THE_DISPUTED_ACCEPTED_CAUSAL_CLAIM','visible':visible,'metrics':metrics,'checks':checks,'failed':failed,'runtime_failure':failure,'remaining_red':['THREE_PLUS_SIBLING_COORDINATION','OPEN_ENDED_QUD_GROUPING','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    return result
def main():
    with tempfile.TemporaryDirectory(prefix='foundry-adverse-uptake-') as directory:
        build_cache(directory);result=verify_loaded(load_mark(directory,'relational_surplus_recovered').adult)
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
