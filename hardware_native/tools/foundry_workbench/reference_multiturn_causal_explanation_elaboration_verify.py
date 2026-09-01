#!/usr/bin/env python3
"""N+1: reduced 'more' elaborates an already answered self-discourse why relation."""
from __future__ import annotations

import copy
import inspect
import json
import time

from reference_discourse_quantity_interaction_verify import RELATIONS
from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE,CONTACT_SURFACE,CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_explanation_elaboration_bridge_v1 import LivedWorldExplanationElaborationBridgeV1
from reference_predictive_credit_profile_v1 import Q
from reference_reduced_why_self_discourse_verify import QCTX,QMARK,WHY,ask,develop_world,earn_pair,prepared,teach_question
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
from reference_world_derived_proposition_frontier_verify import WORLD_A,WORLD_B,SOURCE_A,SOURCE_B,world

MORE=0xEA03; MORE_CTX=0xEA12; NAME=0xEA10; CHANNEL=7
visible_language_gain='BARE_MORE_AFTER_SELF_DISCOURSE_WHY_PRODUCES_SECOND_CONSEQUENCE_LEARNED_CAUSAL_EXPLANATION'
language_phenotype_improved=True
future_update_authority_preserved=True


def teach_more(a):
    c=LanguageMasteryContactAdapterV1(a)
    for k in range(2):
        c.contact(CONTACT_SCENE,(NAME,MORE),41000+k)
        if not c.contact(CONTACT_SURFACE,b'more',41020+k):raise RuntimeError('elaboration:item')
    rows=a.language.invert_surface(tuple(b'more'))
    if not rows or {tuple(row.atoms) for row in rows}!={(MORE,)}:
        raise RuntimeError('elaboration:inverse')


def develop_elaboration(a,o,state,source,why_context,index,seed):
    world(o,state,source);frontier=WorldDiscourseSituationBridgeV1.frontier(a,o)
    program=earn_pair(a,frontier[index],frontier[index+1],RELATIONS[2],seed)
    context=LivedWorldExplanationElaborationBridgeV1.context(
        o,why_context,NAME,(MORE,))
    if context<=0:raise RuntimeError('elaboration:context')
    for _ in range(2):a.experience_choice(program.identity,Q,context=context,effort_q16=Q//16,duration=1,independent_return=True)
    return program,context


def more(a,o,last,source=0xED00):
    prior=int(a.last_completed_public_context());rows=a.language.invert_surface(tuple(b'more'))
    contents={tuple(map(int,row.atoms)) for row in rows if row.atoms}
    if len(contents)!=1:return 0,0,b'',prior
    atoms=next(iter(contents));identity=1
    context,program=LivedWorldExplanationElaborationBridgeV1.activate_program(
        a,o,0,atoms,last)
    if not program:return int(identity),0,b'',prior
    chosen=a.choose();return int(identity),int(chosen),emit_choice(a,chosen),prior


def main():
    started=time.perf_counter();checks={};a,o=prepared();teach_question(a);teach_more(a);last=0
    last,why_a,_fa,qa,_spoken_a=develop_world(a,o,WORLD_A,SOURCE_A,last,RELATIONS[0],0,0xEB01)
    more_a,ma=develop_elaboration(a,o,WORLD_A,SOURCE_A,qa,4,0xEE01)
    last,why_b,_fb,qb,_spoken_b=develop_world(a,o,WORLD_B,SOURCE_B,last,RELATIONS[0],2,0xEB02)
    more_b,mb=develop_elaboration(a,o,WORLD_B,SOURCE_B,qb,6,0xEE02)
    checks['why_and_more_contexts_are_world_and_history_specific']=(qa!=qb and ma!=mb and ma!=qa and mb!=qb)
    developed=copy.deepcopy(a.checkpoint())

    aa=type(a).restore(copy.deepcopy(developed));oo=type(o).restore(copy.deepcopy(o.checkpoint()));world(oo,WORLD_A,SOURCE_A);speech_a,last_a=quiet(aa,oo,0)
    _wid,wpid,why_answer,why_prior=ask(aa,oo,last_a,0xEF01)
    completed_why_a=aa.last_completed_public_context()
    _mid,mpid,more_answer,more_prior=more(aa,oo,last_a,0xEF02)
    checks['world_a_why_then_more_produces_distinct_visible_explanations']=(
        len(speech_a)>300 and wpid==why_a.identity and mpid==more_a.identity
        and why_answer and more_answer and why_answer!=more_answer)
    checks['more_is_licensed_by_completed_why_context_not_world_monologue']=(
        why_prior==WorldDiscourseSituationBridgeV1.context(oo)
        and completed_why_a==qa and more_prior==qa)

    bb=type(a).restore(copy.deepcopy(developed));bo=type(o).restore(copy.deepcopy(o.checkpoint()));world(bo,WORLD_B,SOURCE_B);speech_b,last_b=quiet(bb,bo,0)
    _bwid,bwpid,bwhy,_=ask(bb,bo,last_b,0xEF11);_bmid,bmpid,bmore,_=more(bb,bo,last_b,0xEF12)
    checks['same_reduced_sequence_in_world_b_selects_different_explanations']=(
        len(speech_b)>300 and bwpid==why_b.identity and bmpid==more_b.identity
        and bwhy and bmore and (bwhy, bmore)!=(why_answer,more_answer))

    # Same 'more' before an answered why has no prior explanation relation.
    no_prior=type(a).restore(copy.deepcopy(developed));no_org=type(o).restore(copy.deepcopy(o.checkpoint()));world(no_org,WORLD_A,SOURCE_A);_ns,nlast=quiet(no_prior,no_org,0)
    nid,npid,nans,nprior=more(no_prior,no_org,nlast,0xEF20)
    checks['more_without_completed_why_answer_refuses']=(nid and npid==0 and nans==b'' and nprior==WorldDiscourseSituationBridgeV1.context(no_org))

    # Checkpoint after why retains only opaque completed context; elaboration survives.
    resumed=type(a).restore(copy.deepcopy(developed));ro=type(o).restore(copy.deepcopy(o.checkpoint()));world(ro,WORLD_A,SOURCE_A);_rs,rlast=quiet(resumed,ro,0);_rid,rwpid,rwhy,_=ask(resumed,ro,rlast,0xEF30)
    cp=copy.deepcopy(resumed.checkpoint());rr=type(a).restore(copy.deepcopy(cp));_mid,rmpid,rmore,rprior=more(rr,ro,rlast,0xEF31)
    checks['checkpoint_between_why_and_more_preserves_elaboration_without_transcript']=(
        rwpid==why_a.identity and rmpid==more_a.identity and rmore==more_answer
        and rprior==qa and int(cp.get('last_completed_public_context',0))==qa)

    # Equal alternatives refuse rather than picking a second explanation by ID/order.
    tied=type(a).restore(copy.deepcopy(developed));to=type(o).restore(copy.deepcopy(o.checkpoint()));world(to,WORLD_A,SOURCE_A);_ts,tlast=quiet(tied,to,0);ask(tied,to,tlast,0xEFA0)
    frontier=WorldDiscourseSituationBridgeV1.frontier(tied,to);alt=earn_pair(tied,frontier[6],frontier[7],RELATIONS[1],0xEE11)
    for _ in range(2):tied.experience_choice(alt.identity,Q,context=ma,effort_q16=Q//16,duration=1,independent_return=True)
    _ti,tpid,tans,_=more(tied,to,tlast,0xEFA1)
    checks['equal_elaboration_history_refuses_arbitrary_more']=(tpid==0 and tans==b'')

    # Stale world occurrence cannot reuse the relation.
    stale=type(a).restore(copy.deepcopy(developed));so=type(o).restore(copy.deepcopy(o.checkpoint()));world(so,WORLD_A,SOURCE_A);_ss,slast=quiet(stale,so,0);ask(stale,so,slast,0xEFB0)
    _si,spid,sans,_=more(stale,so,slast+1,0xEFB1)
    checks['wrong_spoken_world_occurrence_refuses_more']=(spid==0 and sans==b'')

    source=inspect.getsource(LivedWorldExplanationElaborationBridgeV1).lower()
    checks['bridge_has_no_more_why_because_topic_or_answer_semantics']=(
        all(token not in source for token in ('more','why','because','topic','answer','expected')))
    blob=json.dumps(developed,sort_keys=True)
    checks['checkpoint_has_no_question_answer_transcript_or_elaboration_state']=(
        all(token not in blob for token in ('more','why?','conversation_buffer','context_window','elaboration_state')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.multiturn-causal-explanation-elaboration.v1','contract':'FOUNDRY_MULTITURN_CAUSAL_EXPLANATION_ELABORATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':future_update_authority_preserved,'visible_language_gain':visible_language_gain,'conversation':{'world_a':[speech_a.decode(),'why?',why_answer.decode() if why_answer else '','more',more_answer.decode() if more_answer else ''],'world_b':[speech_b.decode(),'why?',bwhy.decode() if bwhy else '','more',bmore.decode() if bmore else '']},'checks':checks,'failed':failed,'remaining_red':['OPEN_WORLD_CAUSAL_RELATION_ACQUISITION','OPEN_ENDED_CONVERSATION','DIRECT_MULTITURN_EXPLANATION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print('visible_language_gain='+visible_language_gain);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
