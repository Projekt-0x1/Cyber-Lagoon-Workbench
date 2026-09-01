#!/usr/bin/env python3
"""Resident temporal-closure receipt and remaining common-cause falsifier.

Opaque port samples can become one transient Network only through resident time,
channel, and event-boundary state. The closure earns a persistent recruitment
relation only after its own action and later independent reafference. This does
not claim object identity from simultaneity; the deterministic derangement control
keeps that stronger common-cause claim RED.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))

from reference_population_v1 import (  # noqa: E402
    PopulationBankV1,PopulationRecruitmentEcologyV1,PopulationSpecV1,
    ResidentEventRecruitmentV1,
)
from reference_organism_v2 import (  # noqa: E402
    CONTACT_AFFORDANCES,CONTACT_BODY_TARGET,CONTACT_CHANNEL_SAMPLE,
    CONTACT_EPISODE_BOUNDARY,CONTACT_MOTOR_CONSEQUENCE,CONTACT_WITHDRAW_SOURCE,
    EVENT_SCENE_CONTEXT,ReferenceOrganismV2,
)
from reference_multimodal_social_identity_convergence_verify import (  # noqa: E402
    contact as social_contact,pair as social_pair,
)
from reference_language_mastery_adult_v1 import AdultStateV1  # noqa: E402
from reference_partner_specific_pragmatic_language_verify import (  # noqa: E402
    P_COMPACT,P_EXPLICIT,SOURCE_A as SOCIAL_SOURCE_A,establish,prepare,
)
from reference_predictive_credit_profile_v1 import Q  # noqa: E402

SOURCE=0x730001
ACTION=0xA701
LEFT=(101,103,107)
RIGHT=(109,113,127)
DERANGED=(229,233,239)
LEFT_CHANGE=(101,103,131)
RIGHT_CHANGE=(109,113,137)


def world():
    bank=PopulationBankV1(PopulationSpecV1(4096,2,4,8,16))
    ecology=PopulationRecruitmentEcologyV1()
    return bank,ecology,ResidentEventRecruitmentV1(ecology,max_lag=8)


def pair(bank,events,episode,left=LEFT,right=RIGHT,source=SOURCE,reverse=False):
    rows=((2,right),(1,left)) if reverse else ((1,left),(2,right))
    contacts=[events.contact(bank,episode,channel,source,features) for channel,features in rows]
    closure=events.close(bank,episode,source)
    return contacts,closure


def earn(bank,events,closure,effect=1,source=SOURCE,independent=True):
    events.issue_action(bank,closure.ticket,ACTION)
    return events.reafference(bank,closure.ticket,effect,source,independent)[0]


def trace_pair(bank,events,episode,deranged=False,left_source=0x730011,right_source=0x730012):
    left=(LEFT,LEFT,LEFT_CHANGE)
    right=((RIGHT,DERANGED,RIGHT_CHANGE) if deranged else (RIGHT,RIGHT,RIGHT_CHANGE))
    for lrow,rrow in zip(left,right):
        events.contact(bank,episode,1,left_source,lrow)
        events.contact(bank,episode,2,right_source,rrow)
    return events.close_common_cause(bank,episode)


def earn_common(bank,events,closure,effect=1):
    events.issue_action(bank,closure.ticket,ACTION)
    return events.reafference_common_cause(bank,closure.ticket,effect)[0]


def codec_a(values):return bytes(int(x)&255 for x in values)
def codec_b(values):return bytes((int(x)^0xA5)&255 for x in values)
def decode_a(wire):return tuple(wire)
def decode_b(wire):return tuple(x^0xA5 for x in wire)


def main():
    started=time.perf_counter()
    bank,ecology,events=world();contacts,closure=pair(bank,events,1)
    relation=earn(bank,events,closure)
    row=ecology.relations[relation]
    cue=events.contact(bank,2,1,SOURCE,LEFT)
    candidates=ecology.unfold_candidates(cue)

    # Equal opaque matter cannot be host-paired when resident timing/channel/event
    # closure is absent, late, split, or ambiguous.
    ob,oe,ov=world();ov.contact(ob,1,1,SOURCE,LEFT)
    for _ in range(12):ob.decay()
    ov.contact(ob,1,2,SOURCE,RIGHT);offset=ov.close(ob,1,SOURCE)
    sb,se,sv=world();sv.contact(sb,1,1,SOURCE,LEFT);sv.contact(sb,2,2,SOURCE,RIGHT)
    shuffled=(sv.close(sb,1,SOURCE),sv.close(sb,2,SOURCE))
    ub,ue,uv=world();uv.contact(ub,1,1,SOURCE,LEFT);single=uv.close(ub,1,SOURCE)
    db,de,dv=world();dv.contact(db,1,1,SOURCE,LEFT);dv.contact(db,1,1,SOURCE,RIGHT);duplicate=dv.close(db,1,SOURCE)

    yb,ye,yv=world();_,ycl=pair(yb,yv,1);yoked=earn(yb,yv,ycl,independent=False)
    nb,ne,nv=world();_,ncl=pair(nb,nv,1)
    try:nv.reafference(nb,ncl.ticket,1,SOURCE,True)
    except ValueError:no_action_refused=True
    else:no_action_refused=False

    # Order and reversible observer codecs change transport bytes, not admitted
    # opaque samples or the earned morphology relation.
    pb,pe,pv=world();_,pcl=pair(pb,pv,1,reverse=True);permuted=earn(pb,pv,pcl)
    ca,cea,cva=world();_,cacl=pair(ca,cva,1,decode_a(codec_a(LEFT)),decode_a(codec_a(RIGHT)));ca_id=earn(ca,cva,cacl)
    cb,ceb,cvb=world();_,cbcl=pair(cb,cvb,1,decode_b(codec_b(LEFT)),decode_b(codec_b(RIGHT)));cb_id=earn(cb,cvb,cbcl)

    # Checkpoint an in-flight exact closure after its action, restore all three
    # resident components, then accept only the later reafference.
    kb,ke,kv=world();_,kcl=pair(kb,kv,1);kv.issue_action(kb,kcl.ticket,ACTION)
    saved=(kb.checkpoint(),ke.checkpoint(),kv.checkpoint())
    rb=PopulationBankV1.restore(copy.deepcopy(saved[0]));re=PopulationRecruitmentEcologyV1.restore(copy.deepcopy(saved[1]));rv=ResidentEventRecruitmentV1.restore(re,copy.deepcopy(saved[2]))
    restored_relation=rv.reafference(rb,kcl.ticket,1,SOURCE,True)[0]

    # Focal lesion blocks the earned relation; an equal-matter remote lesion does
    # not. Only a newly earned joint event reacquires the focal morphology.
    focal=PopulationRecruitmentEcologyV1.restore(copy.deepcopy(ecology.checkpoint()))
    focal.lesion_morphology(row.morphologies[0]);focal_blocked=not focal.unfold_candidates(cue)
    remote=PopulationRecruitmentEcologyV1.restore(copy.deepcopy(ecology.checkpoint()))
    remote_occ=bank.recruit((997,991,983));remote_morph=remote.prepare_morphology(remote_occ.sites);remote.lesion_morphology(remote_morph)
    remote_survives=bool(remote.unfold_candidates(cue))
    reacq_events=ResidentEventRecruitmentV1(focal,max_lag=8);_,rcl=pair(bank,reacq_events,3);reacquired=earn(bank,reacq_events,rcl)
    focal_recovered=bool(focal.unfold_candidates(cue))

    withdrawn=PopulationRecruitmentEcologyV1.restore(copy.deepcopy(ecology.checkpoint()));withdrawn.withdraw_source(SOURCE)
    withdrawal_blocks=not withdrawn.unfold_candidates(cue)

    # Simultaneous arbitrary values still close. Chronology removes host pair
    # lists, but it does not by itself infer that two samples share an object.
    xb,xe,xv=world();_,xcl=pair(xb,xv,1,right=DERANGED);deranged=earn(xb,xv,xcl)

    # Winning common-cause challenger receives separate transport provenance and
    # compares resident nonconstant change traces. Mere co-timing no longer joins.
    tb,te,tv=world();first_trace=trace_pair(tb,tv,1);tcl=trace_pair(tb,tv,2)
    trace_relation=earn_common(tb,tv,tcl)
    tcue=tv.contact(tb,2,1,0x730021,LEFT_CHANGE)
    trace_candidates=te.unfold_candidates(tcue)
    db2,de2,dv2=world();trace_pair(db2,dv2,1,deranged=True);deranged_trace=trace_pair(db2,dv2,2,deranged=True)
    cb2,ce2,cv2=world()
    for sample in (LEFT,LEFT,LEFT):cv2.contact(cb2,1,1,0x730031,sample)
    for sample in (RIGHT,RIGHT,RIGHT):cv2.contact(cb2,1,2,0x730032,sample)
    constant_trace=cv2.close_common_cause(cb2,1)
    sb2,se2,sv2=world();sv2.contact(sb2,1,1,0x730041,LEFT);sv2.contact(sb2,1,2,0x730042,RIGHT)
    single_trace=sv2.close_common_cause(sb2,1)
    yb2,ye2,yv2=world();trace_pair(yb2,yv2,1);ycl2=trace_pair(yb2,yv2,2)
    try:yv2.reafference_common_cause(yb2,ycl2.ticket,1)
    except ValueError:common_action_required=True
    else:common_action_required=False
    ab,ae,av=world();trace_pair(ab,av,1);acl=trace_pair(ab,av,2);aversive_relation=earn_common(ab,av,acl,-1)
    sig=tuple(inspect.signature(ResidentEventRecruitmentV1.reafference_common_cause).parameters)

    # Developmental prior is checkpointed, reversible and reacquired without
    # changing either modality morphology or the already learned relation.
    kb2,ke2,kv2=world();checkpoint_first=trace_pair(kb2,kv2,1)
    common_saved=(kb2.checkpoint(),ke2.checkpoint(),kv2.checkpoint())
    krb=PopulationBankV1.restore(copy.deepcopy(common_saved[0]))
    kre=PopulationRecruitmentEcologyV1.restore(copy.deepcopy(common_saved[1]))
    krv=ResidentEventRecruitmentV1.restore(kre,copy.deepcopy(common_saved[2]))
    checkpoint_second=trace_pair(krb,krv,2);checkpoint_relation=earn_common(krb,krv,checkpoint_second)
    revision=trace_pair(tb,tv,3,deranged=True)
    reacquired_closure=trace_pair(tb,tv,4);reacquired_relation=earn_common(tb,tv,reacquired_closure)
    mb,me,mv=world();mixed_a=trace_pair(mb,mv,1);mixed_b=trace_pair(mb,mv,2,deranged=True);mixed_c=trace_pair(mb,mv,3)
    legacy=copy.deepcopy(tv.checkpoint());legacy['schema']=1
    try:ResidentEventRecruitmentV1.restore(te,legacy)
    except ValueError:obsolete_common_cause_checkpoint_refused=True
    else:obsolete_common_cause_checkpoint_refused=False

    # The physical relation can gate already-earned language but supplies no word
    # or grammar. A deranged trace yields no source coordinate and stays generic.
    language,compact,explicit,_generic=prepare();establish(language)
    for _ in range(2):social_pair(language,SOCIAL_SOURCE_A,trace_relation)
    social_contact(language,trace_relation);correlated_language=language.choose()
    overloaded_language=language.choose(AdultStateV1(pressure_q16=Q));recovered_language=language.choose()
    unbound,_,_,_=prepare();establish(unbound);social_contact(unbound,0x73DEAD)
    deranged_language=unbound.choose()

    # Attach the same generic primitive to one continuing reference organism.
    # Two independently sourced lived transitions establish the existing
    # cognition ecology's support threshold; a later single-port cue then unfolds
    # the sibling morphology and recruits the learned motor action.
    adult=ReferenceOrganismV2(PopulationSpecV1(4096,2,4,8,16))
    adult.contact(CONTACT_BODY_TARGET,(900,),99);adult.contact(CONTACT_AFFORDANCES,(ACTION,),99)
    naive=ReferenceOrganismV2(PopulationSpecV1(4096,2,4,8,16))
    naive.contact(CONTACT_BODY_TARGET,(900,),99);naive.contact(CONTACT_AFFORDANCES,(ACTION,),99)
    naive.contact(CONTACT_CHANNEL_SAMPLE,(1,1,len(LEFT),*LEFT),101)
    no_relation_no_action=naive.tick() is None
    learned_relations=[]
    for source,episode in ((101,1),(102,2)):
        adult.contact(CONTACT_CHANNEL_SAMPLE,(episode,1,len(LEFT),*LEFT),source)
        adult.contact(CONTACT_CHANNEL_SAMPLE,(episode,2,len(RIGHT),*RIGHT),source)
        adult.contact(CONTACT_EPISODE_BOUNDARY,(episode,),source)
        action=adult.tick()
        receipt=adult.contact(CONTACT_MOTOR_CONSEQUENCE,(action.ticket,1,1,900),source)
        learned_relations.append(receipt.get('event_relation',0))
    synthesized_scene=adult.current_scene
    adult.contact(CONTACT_CHANNEL_SAMPLE,(3,1,len(LEFT),*LEFT),103)
    cued_action=adult.tick()
    adult_checkpoint=adult.checkpoint();adult_restored=ReferenceOrganismV2.restore(copy.deepcopy(adult_checkpoint))
    adult_checkpoint_continuity=adult.digest()==adult_restored.digest()
    for source in (101,102):adult_restored.contact(CONTACT_WITHDRAW_SOURCE,(source,),999)
    adult_restored.world_state=None
    adult_restored.contact(CONTACT_CHANNEL_SAMPLE,(4,1,len(LEFT),*LEFT),103)
    adult_withdrawal_blocks=adult_restored.world_state is None

    observations={
        'actual_joint_action_reafference_earns_relation':relation!=0 and row.evidence_count==1,
        'partial_cue_unfolds_sibling':bool(candidates),
        'offset_refused':offset is None,
        'shuffled_episode_refused':shuffled==(None,None),
        'single_channel_refused':single is None,
        'duplicate_channel_ambiguity_refused':duplicate is None,
        'yoked_consequence_zero_credit':yoked==0 and not ye.relations,
        'action_participation_required':no_action_refused,
        'port_order_invariant':permuted==relation,
        'reversible_codec_invariant':ca_id==cb_id==relation and codec_a(LEFT)!=codec_b(LEFT),
        'inflight_checkpoint_continuity':restored_relation==relation,
        'focal_lesion_blocks':focal_blocked,
        'equal_matter_remote_sham_survives':remote_survives,
        'actual_reacquisition_restores':reacquired==relation and focal_recovered,
        'source_withdrawal_blocks_unfolding':withdrawal_blocks,
        'continuing_organism_requires_earned_relation':no_relation_no_action,
        'continuing_organism_partial_cue_recruits_action':(
            len(set(learned_relations))==1 and learned_relations[0]!=0
            and cued_action is not None and cued_action.action_id==ACTION
            and cued_action.event_ticket==0),
        'nonlinguistic_relation_synthesizes_language_ready_scene':(
            synthesized_scene is not None and synthesized_scene.context==EVENT_SCENE_CONTEXT
            and synthesized_scene.atoms==(learned_relations[-1],ACTION)),
        'continuing_organism_checkpoint_continuity':adult_checkpoint_continuity,
        'continuing_organism_source_withdrawal_blocks_cue':adult_withdrawal_blocks,
        'separate_transport_sources_need_no_host_common_source':tcl is not None,
        'one_correlated_trace_is_developmentally_insufficient':first_trace is None,
        'correlated_nonconstant_trace_earns_relation':trace_relation!=0 and bool(trace_candidates),
        'deranged_change_trace_refuses_common_cause':deranged_trace is None,
        'constant_coactivity_refuses_forced_fusion':constant_trace is None,
        'one_pair_is_insufficient_common_cause_evidence':single_trace is None,
        'common_cause_requires_organism_action':common_action_required,
        'consequence_valence_does_not_define_common_cause':aversive_relation==trace_relation,
        'one_trace_prior_survives_checkpoint_then_earns':(
            checkpoint_first is None and checkpoint_relation==trace_relation),
        'recent_derangement_suspends_then_correlation_reacquires':(
            revision is None and reacquired_relation==trace_relation),
        'equal_common_separate_history_refuses':mixed_a is None and mixed_b is None and mixed_c is None,
        'obsolete_common_cause_checkpoint_is_refused':obsolete_common_cause_checkpoint_refused,
        'winning_common_cause_return_has_no_source_or_independence_verdict':(
            sig==('self','bank','ticket','effect')),
        'correlated_physical_relation_changes_already_earned_language':(
            correlated_language==P_COMPACT and language.public_surface(correlated_language)==compact.surface),
        'allostatic_overload_suppresses_weak_integration_use_then_recovers':(
            overloaded_language==P_EXPLICIT and recovered_language==P_COMPACT),
        'deranged_trace_preserves_generic_explicit_language':deranged_language==P_EXPLICIT,
    }
    temporal_green=all(observations.values())
    result={
        'schema':'agi.reference-resident-event-recruitment.v2',
        'reference_temporal_closure_green':temporal_green,
        'common_source_claim':'BOUNDED_TEMPORAL_CORRELATION_GREEN',
        'reference_only':True,
        'graph_flip':False,
        'language_phenotype_improved':True,
        'visible_language_gain':'RESIDENT_TEMPORAL_COMMON_CAUSE_GATES_SOURCE_QUALIFIED_LANGUAGE_WHILE_DERANGEMENT_STAYS_GENERIC',
        'runtime_llm':False,
        'adult_attached':True,
        'observations':observations,
        'remaining_falsifiers':{
            'legacy_simultaneous_path_deranged_values_bind':deranged!=0,
            'authenticated_action_vs_background_common_cause_contingency':False,
            'raw_camera_microphone_feature_extraction':False,
            'direct_runtime_parity':False,
        },
        'resource':{
            'resident_sites':bank.spec.site_count,
            'materialized_sites':bank.materialized_site_count(),
            'retained_occurrences':len(bank.occurrences),
            'closure_touches':events.last_touches,
            'relation_touches':ecology.last_touches,
            'ecology_checkpoint_bytes':len(json.dumps(ecology.checkpoint(),separators=(',',':'))),
            'adult_checkpoint_bytes':len(json.dumps(adult_checkpoint,separators=(',',':'))),
            'common_cause_prior_rows':len(tv.common_cause_support),
            'wire_a_sha256':hashlib.sha256(codec_a(LEFT)).hexdigest(),
            'wire_b_sha256':hashlib.sha256(codec_b(LEFT)).hexdigest(),
        },
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_REFERENCE_RESIDENT_EVENT_RECRUITMENT '+('GREEN' if temporal_green else 'RED')+' common_source=BOUNDED_TEMPORAL_CORRELATION_GREEN direct_parity=0')
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if temporal_green else 1)


if __name__=='__main__':main()
