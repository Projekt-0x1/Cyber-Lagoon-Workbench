#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
# Preload exact committed core modules before older helper assays reorder sys.path.
import reference_language_learning_v1
import reference_predictive_credit_profile_v1
import reference_population_v1
import reference_organism_v2
import reference_language_mastery_adult_v1
import reference_language_mastery_contact_adapter_v1
from reference_discourse_quantity_interaction_verify import RELATIONS
from reference_language_mastery_contact_adapter_v1 import LanguageMasteryContactAdapterV1,CONTACT_SCENE,CONTACT_SURFACE,CONTACT_UTTERANCE
from reference_language_mastery_terminal_v1 import emit_choice
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_relation_question_bridge_v1 import LivedWorldRelationQuestionBridgeV1
from reference_predictive_credit_profile_v1 import Q
from reference_self_initiated_world_followup_verify import prepared
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
from reference_world_derived_proposition_frontier_verify import WORLD_A,WORLD_B,SOURCE_A,SOURCE_B,world

WHY=0xEA01;QMARK=0xEA02;NAME=0xEA10;QCTX=0xEA11;CHANNEL=7

def teach_question(a):
    c=LanguageMasteryContactAdapterV1(a)
    for feature,text,base in ((WHY,b'why',40000),(QMARK,b'?',40100)):
        for k in range(2):
            c.contact(CONTACT_SCENE,(NAME,feature),base+k)
            if not c.contact(CONTACT_SURFACE,text,base+20+k):raise RuntimeError('reduced-why:item')
    for k in range(2):
        c.contact(CONTACT_SCENE,(QCTX,WHY,QMARK),40200+k)
        if not c.contact(CONTACT_SURFACE,b'why?',40300+k):raise RuntimeError('reduced-why:construction')
    rows=a.language.invert_surface(tuple(b'why?'))
    if len(rows)!=1 or rows[0].context!=QCTX or tuple(rows[0].atoms)!=(WHY,QMARK):
        raise RuntimeError('reduced-why:inverse')

def earn_pair(a,left,right,relation,seed):
    witness=a.compose(relation,left,right);program=None
    for _ in range(3):
        program=a.experience_program((left.identity,right.identity),witness,3*Q//4,Q//8,seed,Q//8,True)
    if program is None:raise RuntimeError('reduced-why:program')
    return program

def develop_world(a,o,state,source,last,relation,index,seed):
    a._clear_current_occurrence();world(o,state,source);spoken,last=quiet(a,o,last)
    if not spoken:raise RuntimeError('reduced-why:self-speech')
    frontier=WorldDiscourseSituationBridgeV1.frontier(a,o)
    program=earn_pair(a,frontier[index],frontier[index+1],relation,seed)
    qctx=LivedWorldRelationQuestionBridgeV1.context(o,QCTX,(WHY,QMARK))
    for _ in range(2):a.experience_choice(program.identity,Q,context=qctx,effort_q16=Q//16,duration=1,independent_return=True)
    return last,program,tuple(int(x.identity) for x in frontier),qctx,spoken

def ask(a,o,last,source):
    prior=int(a.last_completed_public_context());c=LanguageMasteryContactAdapterV1(a)
    identity=c.contact(CONTACT_UTTERANCE,tuple(b'why?'),source,CHANNEL)
    scene=c.scenes.get(int(identity)) if int(identity)>0 else None
    if scene is None:return int(identity),0,b'',prior
    context,program=LivedWorldRelationQuestionBridgeV1.activate_program(
        a,o,scene.context,scene.atoms,last)
    if not program:return int(identity),0,b'',prior
    chosen=a.choose();return int(identity),int(chosen),emit_choice(a,chosen),prior

def main():
    started=time.perf_counter();checks={};a,o=prepared();teach_question(a);last=0
    last,pa,fa,qa,spoken_a=develop_world(a,o,WORLD_A,SOURCE_A,last,RELATIONS[0],0,0xEB01)
    last,pb,fb,qb,spoken_b=develop_world(a,o,WORLD_B,SOURCE_B,last,RELATIONS[0],2,0xEB02)
    checks['same_reduced_question_has_one_learned_nonreferential_binding']=(len(a.language.invert_surface(tuple(b'why?')))==1)
    checks['world_question_contexts_are_distinct_without_topic_id']=(qa!=qb and pa.identity!=pb.identity)
    developed=copy.deepcopy(a.checkpoint())

    aa=type(a).restore(copy.deepcopy(developed));oo=type(o).restore(copy.deepcopy(o.checkpoint()));world(oo,WORLD_A,SOURCE_A);speech_a,last_a=quiet(aa,oo,0)
    aid,apid,answer_a,prior_a=ask(aa,oo,last_a,0xEC01)
    bb=type(a).restore(copy.deepcopy(developed));bo=type(o).restore(copy.deepcopy(o.checkpoint()));world(bo,WORLD_B,SOURCE_B);speech_b,last_b=quiet(bb,bo,0)
    bid,bpid,answer_b,prior_b=ask(bb,bo,last_b,0xEC02)
    checks['same_bare_why_after_world_a_selects_a_explanation']=(aid and apid==pa.identity and answer_a==bytes(a.public_surface(pa.identity)) and b' because ' in answer_a)
    checks['same_bare_why_after_world_b_selects_b_explanation']=(bid and bpid==pb.identity and answer_b==bytes(a.public_surface(pb.identity)) and b' because ' in answer_b)
    checks['same_question_different_self_discourse_changes_visible_answer']=bool(answer_a and answer_b and answer_a!=answer_b)
    checks['prior_context_is_exact_current_world_discourse']=(prior_a==WorldDiscourseSituationBridgeV1.context(oo) and prior_b==WorldDiscourseSituationBridgeV1.context(bo))

    no_speech=type(a).restore(copy.deepcopy(developed));no_org=type(o).restore(copy.deepcopy(o.checkpoint()));world(no_org,WORLD_A,SOURCE_A)
    nid,npid,nans,_=ask(no_speech,no_org,int(no_org.world_state_occurrence),0xEC10)
    checks['same_why_without_prior_self_initiated_turn_refuses']=(nid and npid==0 and nans==b'')

    checkpointed=type(a).restore(copy.deepcopy(developed));co=type(o).restore(copy.deepcopy(o.checkpoint()));world(co,WORLD_A,SOURCE_A);_s,clast=quiet(checkpointed,co,0)
    mid=copy.deepcopy(checkpointed.checkpoint());resumed=type(a).restore(mid)
    rid,rpid,rans,rprior=ask(resumed,co,clast,0xEC20)
    checks['checkpoint_retains_completed_discourse_license_for_bare_why']=(
        rid and rpid==pa.identity and rans==bytes(a.public_surface(pa.identity))
        and rprior==WorldDiscourseSituationBridgeV1.context(co))

    stale=type(a).restore(copy.deepcopy(developed));so=type(o).restore(copy.deepcopy(o.checkpoint()));world(so,WORLD_A,SOURCE_A);_ss,slast=quiet(stale,so,0)
    sid,spid,sans,_=ask(stale,so,slast+1,0xEC30)
    checks['wrong_spoken_world_occurrence_refuses']=(sid and spid==0 and sans==b'')

    tied=type(a).restore(copy.deepcopy(developed));to=type(o).restore(copy.deepcopy(o.checkpoint()));world(to,WORLD_A,SOURCE_A);_ts,tlast=quiet(tied,to,0)
    frontier=WorldDiscourseSituationBridgeV1.frontier(tied,to);alt=earn_pair(tied,frontier[2],frontier[3],RELATIONS[1],0xEB11)
    tq=LivedWorldRelationQuestionBridgeV1.context(to,QCTX,(WHY,QMARK))
    for _ in range(2):tied.experience_choice(alt.identity,Q,context=tq,effort_q16=Q//16,duration=1,independent_return=True)
    tid,tpid,tans,_=ask(tied,to,tlast,0xEC40)
    checks['equal_competing_relation_answers_refuse_arbitrary_why']=(tid and tpid==0 and tans==b'')

    foreign=type(a).restore(copy.deepcopy(developed));fo=type(o).restore(copy.deepcopy(o.checkpoint()));world(fo,WORLD_A,SOURCE_A);_fs,flast=quiet(foreign,fo,0)
    # Make B explanation dominate A's question context; bridge must reject its out-of-world leaves.
    for _ in range(8):foreign.experience_choice(pb.identity,Q,context=qa,effort_q16=Q//32,duration=1,independent_return=True)
    fid,fpid,fans,_=ask(foreign,fo,flast,0xEC50)
    checks['out_of_world_winning_program_cannot_answer_current_why']=(fid and fpid==0 and fans==b'')

    source=inspect.getsource(LivedWorldRelationQuestionBridgeV1)
    checks['bridge_has_no_why_because_topic_or_answer_semantics']=(all(x not in source.lower() for x in ('why','because','topic','answer','expected')))
    blob=json.dumps(developed,sort_keys=True)
    checks['adult_checkpoint_has_no_reduced_question_or_topic_state']=(all(x not in blob for x in ('why?','active_topic','question_under_discussion','relation_question')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reduced-why-self-discourse.v1','contract':'FOUNDRY_REDUCED_WHY_SELF_DISCOURSE_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'visible_language_gain':'BARE_WHY_REUSES_IMMEDIATE_SELF_INITIATED_WORLD_DISCOURSE_TO_SELECT_CONTEXT_SPECIFIC_EXPLANATION','conversation':{'world_a':[speech_a.decode(), 'why?',answer_a.decode() if answer_a else ''],'world_b':[speech_b.decode(),'why?',answer_b.decode() if answer_b else '']},'checks':checks,'failed':failed,'remaining_red':['MULTI_PARTNER_REDUCED_QUESTION_CONTEXT','MULTI_TURN_CAUSAL_EXPLANATION_ELABORATION','OPEN_WORLD_CAUSAL_RELATION_ACQUISITION','OPEN_ENDED_CONVERSATION'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
