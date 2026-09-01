#!/usr/bin/env python3
"""Chomsky-style RED/GREEN: structurally select a distant reply constituent over a local distractor."""
from __future__ import annotations
import copy,json,time
from pathlib import Path
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_source_reliability_productive_construction_transfer_verify import (
    CLAUSE,X,ALT_AMB,ALT_ANS,ALT_X,ALT_Y,P_ACCEPT,P_VERIFY,REL,teach_alt,
    response_ecology,settle,emit_all,
)
from reference_recursive_testimony_span_repair_verify import calibrated

LD=0x7B01
# Locally plausible competing answer: same clause family/action relation, different target.
DISTRACTOR=b'after review, the valve is what the careful engineer tests.'

def wrap(target,distractor,prefix=b'answer slot: ',middle=b' | distractor: '):
    return prefix+target+middle+distractor

def teach_long_distance_wrapper(a):
    # Two independent source witnesses establish only the structural surface relation.
    s1=wrap(ALT_X,ALT_Y);s2=wrap(ALT_Y,ALT_X)
    assert a.language.observe_span(LD,(ALT_X,ALT_Y),s1,0x8D01)
    assert a.language.observe_span(LD,(ALT_Y,ALT_X),s2,0x8D02)

def current_reply(a,answer,counter):
    counter[0]+=8;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
    gap=m.contact(CONTACT_UTTERANCE,tuple(ALT_AMB),base,REL);q=settle(a)
    if q!=P_VERIFY:raise RuntimeError('longdep:question')
    emit_all(a,q);scene=m.contact(CONTACT_UTTERANCE,tuple(answer),base+1,REL)
    return int(scene),int(a.choose()),int(gap)

def teach_reply_role(a,counter,port=0):
    sources=(0x8E01,0x8E02);irrelevant=(b'unrelated field note.',b'separate maintenance note.')
    for source,other in zip(sources,irrelevant):
        counter[0]+=8;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
        gap=m.contact(CONTACT_UTTERANCE,tuple(ALT_AMB),base,REL);q=settle(a)
        if q!=P_VERIFY or not gap:raise RuntimeError('longdep:teach_question')
        emit_all(a,q);children=(ALT_ANS,other) if int(port)==0 else (other,ALT_ANS)
        scene=m.contact(CONTACT_UTTERANCE,tuple(wrap(*children)),source,REL)
        if not scene or a.choose()!=P_ACCEPT:raise RuntimeError('longdep:teach_reply_role')
    return sources

def main():
    t=time.perf_counter();a,obj_rep,counter=calibrated();teach_alt(a);response_ecology(a,ALT_AMB,(CLAUSE,X));teach_long_distance_wrapper(a)
    pre_role_cp=copy.deepcopy(a.checkpoint())
    target_only=LanguageMasteryAdultV1.restore(copy.deepcopy(pre_role_cp));ts,tc,_=current_reply(target_only,ALT_ANS,counter)
    heldout=wrap(ALT_ANS,DISTRACTOR,prefix=b'answer slot: ',middle=b' | distractor: ')

    no_role=LanguageMasteryAdultV1.restore(copy.deepcopy(pre_role_cp));nrs,nrc,_=current_reply(no_role,heldout,counter)

    trained=LanguageMasteryAdultV1.restore(copy.deepcopy(pre_role_cp));role_sources=teach_reply_role(trained,counter,0);role_cp=copy.deepcopy(trained.checkpoint())
    tested=LanguageMasteryAdultV1.restore(copy.deepcopy(role_cp));scene,choice,gap=current_reply(tested,heldout,counter)

    reversed_branch=LanguageMasteryAdultV1.restore(copy.deepcopy(pre_role_cp));teach_reply_role(reversed_branch,counter,1);rs,rc,_=current_reply(reversed_branch,heldout,counter)

    withdrawn=LanguageMasteryAdultV1.restore(copy.deepcopy(role_cp))
    for source in role_sources:withdrawn.language.withdraw_source(source)
    ws,wc,_=current_reply(withdrawn,heldout,counter)

    direct=tuple(tested.language.invert_surface(tuple(heldout)));spans=tuple(tested.language.invert_span(tuple(heldout)))
    role_rows=role_cp['language'].get('span_reply_roles',[])
    checks={
      'flat_target_baseline_closes':bool(ts) and tc==P_ACCEPT,
      'heldout_wrapper_has_no_direct_flat_binding':not direct,
      'learned_wrapper_recovers_one_structural_span':len(spans)==1 and len(spans[0].children)==2,
      'same_heldout_wrapper_without_role_history_refuses':nrs==0 and nrc==0,
      'reply_role_is_learned_from_two_independent_dialogue_sources':len(role_rows)==1 and role_rows[0]['port']==0 and role_rows[0]['sources']==list(role_sources),
      'distant_target_constituent_beats_intervening_distractor':bool(scene) and choice==P_ACCEPT and gap!=0,
      'reversing_learned_structural_role_changes_controlling_constituent':bool(rs) and rc!=P_ACCEPT,
      'withdrawing_reply_role_sources_restores_refusal':ws==0 and wc==0,
      'checkpoint_contains_no_heldout_wrapper':heldout.decode() not in json.dumps(role_cp),
      'bounded_fast_path':time.perf_counter()-t<1.0,
      'conversation_frontier_hook_present':'reference_long_distance_structural_reply_verify.py' in (Path(__file__).parent/'run_conversation_frontier_fast.sh').read_text(),
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-long-distance-structural-reply.v2','contract':'FOUNDRY_LONG_DISTANCE_STRUCTURAL_REPLY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'visible_language_gain':'LEARNED_STRUCTURAL_REPLY_ROLE_SELECTS_DISTANT_TARGET_OVER_LOCALLY_VALID_DISTRACTOR','checks':checks,'failed':failed,'current_choice':choice,'reversed_choice':rc,'remaining_red':['NESTED_STRUCTURAL_REPLY_ROLE_COMPOSITION','DIRECT_LONG_DISTANCE_DIALOGUE_PARITY','PHYSICAL_LONG_DISTANCE_DIALOGUE'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
