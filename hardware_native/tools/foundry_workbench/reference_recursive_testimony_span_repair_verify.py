#!/usr/bin/env python3
"""Destructive audit for recursive learned-span recovery of source-qualified repair testimony."""
from __future__ import annotations
import copy,json,time
from pathlib import Path
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_predictive_credit_profile_v1 import Q
from reference_source_reliability_productive_construction_transfer_verify import (
    CLAUSE,X,BASE_AMB,BASE_ANS,ALT_X,ALT_Y,ALT_AMB,ALT_ANS,OTHER_AMB,
    P_ACCEPT,P_VERIFY,REL,UNREL,NOVEL,teach_lexemes,teach_base,teach_alt,
    response_ecology,repair,emit_all,settle,relations,
)
language_phenotype_improved=True
future_update_authority_preserved=True
INNER=0x7A01;OUTER=0x7A02
ALT_VALVE=b'after review, the valve is what the careful engineer tests.'

def inner(left,right):return b'report: '+left+b' / answer: '+right
def outer(left,right):return b'context: '+left+b' | testimony: '+right

def teach_inner(a):
    a1=inner(b'alpha filler',ALT_X);a2=inner(b'beta filler',ALT_Y)
    assert a.language.observe_span(INNER,(b'alpha filler',ALT_X),a1,0x8101)
    assert a.language.observe_span(INNER,(b'beta filler',ALT_Y),a2,0x8102)

def teach_outer(a):
    i1=inner(b'alpha filler',ALT_X);i2=inner(b'beta filler',ALT_Y)
    assert a.language.observe_span(OUTER,(b'left alpha',i1),outer(b'left alpha',i1),0x8201)
    assert a.language.observe_span(OUTER,(b'left beta',i2),outer(b'left beta',i2),0x8202)

def nested(answer=ALT_ANS,left=b'novel outer',inside=b'novel inner'):
    return outer(left,inner(inside,answer))

def current_reply(a,channel,ambiguity,answer,counter):
    counter[0]+=8;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
    gap=m.contact(CONTACT_UTTERANCE,tuple(ambiguity),base,channel);q=settle(a)
    if q!=P_VERIFY:raise RuntimeError('recursive_testimony:question')
    emit_all(a,q);scene=m.contact(CONTACT_UTTERANCE,tuple(answer),base+1,channel)
    return int(scene),int(a.choose(AdultStateV1())),int(a._current_partner_context),int(gap)

def calibrated():
    a=LanguageMasteryAdultV1();teach_lexemes(a);teach_base(a)
    _comp,rep=response_ecology(a,BASE_AMB,(CLAUSE,X));counter=[0x9000]
    for _ in range(2):
        ch,*_=repair(a,REL,BASE_AMB,BASE_ANS,counter);a.experience_partner_choice(ch,Q);a.experience_partner_background(ch,False)
    for _ in range(2):
        ch,*_=repair(a,UNREL,BASE_AMB,BASE_ANS,counter);a.experience_partner_choice(ch,-Q);a.experience_partner_background(ch,False)
    teach_alt(a)
    return a,int(rep),counter

def main():
    started=time.perf_counter();checks={};a,obj_rep,counter=calibrated();base_cp=copy.deepcopy(a.checkpoint())
    direct_rel,*_=repair(LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp)),REL,ALT_AMB,ALT_ANS,counter)
    direct_unrel,*_=repair(LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp)),UNREL,ALT_AMB,ALT_ANS,counter)
    checks['flat_heldout_baseline_survives']=direct_rel==P_ACCEPT and direct_unrel==P_VERIFY

    full=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));partner_before=json.dumps(full.partner_credit.checkpoint(),sort_keys=True,separators=(',',':'))
    teach_inner(full);teach_outer(full);partner_after=json.dumps(full.partner_credit.checkpoint(),sort_keys=True,separators=(',',':'))
    nested_answer=nested();direct_rows=relations(full,nested_answer)
    checks['span_learning_does_not_rewrite_source_history']=partner_before==partner_after
    checks['full_nested_reply_has_no_direct_flat_repair_match']=(CLAUSE,X) not in direct_rows
    full_cp=copy.deepcopy(full.checkpoint())
    checks['checkpoint_has_factors_but_no_current_nested_parse']=('current_partner_context' not in full_cp and nested_answer.decode() not in json.dumps(full_cp) and LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp)).choose()==0)

    tested=LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp))
    rs,rc,rctx,_=current_reply(tested,REL,ALT_AMB,nested_answer,counter)
    us,uc,uctx,_=current_reply(tested,UNREL,ALT_AMB,nested_answer,counter)
    ns,nc,nctx,_=current_reply(tested,NOVEL,ALT_AMB,nested_answer,counter)
    checks['reliable_unreliable_novel_nested_policy']=(rs and us and ns and rc==P_ACCEPT and uc==P_VERIFY and nc==P_ACCEPT)
    checks['nested_policy_uses_same_repair_coordinate']=(rctx==full._repair_partner_context(REL,obj_rep) and uctx==full._repair_partner_context(UNREL,obj_rep) and nctx==full._repair_partner_context(NOVEL,obj_rep))

    no_outer=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));teach_inner(no_outer)
    os,oc,_,_=current_reply(no_outer,REL,ALT_AMB,nested_answer,counter)
    checks['missing_outer_span_blocks_nested_closure']=os==0 and oc==0 and no_outer._language_reply_open(REL)
    no_inner=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));teach_outer(no_inner)
    ins,inc,_,_=current_reply(no_inner,REL,ALT_AMB,nested_answer,counter)
    checks['missing_inner_span_blocks_nested_closure']=ins==0 and inc==0 and no_inner._language_reply_open(REL)

    conflict=LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp))
    conflict_reply=outer(inner(b'left one',ALT_ANS),inner(b'left two',ALT_VALVE))
    cs,cc,_,_=current_reply(conflict,REL,ALT_AMB,conflict_reply,counter)
    checks['two_embedded_pending_answers_refuse_atomically']=cs==0 and cc==0 and conflict._language_reply_open(REL)

    withdrawn=LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp));withdrawn.language.withdraw_source(0x8201)
    ws,wc,_,_=current_reply(withdrawn,REL,ALT_AMB,nested_answer,counter)
    flat_after_withdraw,*_=repair(withdrawn,REL,ALT_AMB,ALT_ANS,counter)
    checks['focal_outer_span_withdrawal_blocks_nested_but_spares_flat']=(ws==0 and wc==0 and flat_after_withdraw==P_ACCEPT)

    local=LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp));_other_comp,other_rep=response_ecology(local,OTHER_AMB,(CLAUSE,X))
    ls,lc,lctx,_=current_reply(local,UNREL,OTHER_AMB,nested_answer,counter)
    checks['same_source_nested_other_relation_remains_generic']=(ls and int(other_rep)!=obj_rep and lctx==local._repair_partner_context(UNREL,other_rep) and lc==P_ACCEPT)

    deep=LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp));too_deep=ALT_ANS
    for n in range(5):too_deep=outer(('depth-%d'%n).encode(),too_deep)
    ds,dc,_,_=current_reply(deep,REL,ALT_AMB,too_deep,counter)
    checks['depth_beyond_four_refuses']=ds==0 and dc==0 and deep._language_reply_open(REL)

    checks['recursive_recovery_is_reply_open_only']=LanguageMasteryContactAdapterV1(LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp))).contact(CONTACT_UTTERANCE,tuple(nested_answer),0xA100,REL)==0
    checks['both_fast_unions_run_recursive_testimony_audit']=(
        'recursive-testimony:reference_recursive_testimony_span_repair_verify.py' in (Path(__file__).parent/'run_language_mastery_fast.sh').read_text()
        and 'reference_recursive_testimony_span_repair_verify.py' in (Path(__file__).parent/'run_language_mastery_factory_fast.sh').read_text())
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-recursive-testimony-span-repair.v1','pass':not failed,'reference_only':True,'mechanism_change':True,'language_phenotype_improved':language_phenotype_improved,'visible_language_gain':'SOURCE_RELIABILITY_GOVERNS_DEPTH2_LEARNED_SPAN_EMBEDDED_HELDOUT_TESTIMONY','future_update_authority_preserved':future_update_authority_preserved,'choices':{'reliable_nested':rc,'unreliable_nested':uc,'novel_nested':nc,'other_relation':lc},'checks':checks,'failed':failed,'remaining_red':['NON_ADJACENT_DEPENDENCY_REQUIRING_RECURSIVE_CONSTITUENT_INTERPRETATION','MULTISOURCE_CONSENSUS_AND_DECEPTION','DIRECT_RECURSIVE_TESTIMONY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_RECURSIVE_TESTIMONY_SPAN_REPAIR_'+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
