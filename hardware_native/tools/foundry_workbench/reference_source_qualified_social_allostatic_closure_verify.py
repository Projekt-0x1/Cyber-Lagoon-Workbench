#!/usr/bin/env python3
"""N+1: recursive source-qualified advice enters reversible social/body arbitration."""
from __future__ import annotations
import copy,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
from reference_continuous_perspective_inducer_v1 import (
    ContinuousPerspectiveInducerV1,K_ASSERT,K_NEGATE,K_IMPERATIVE)
from reference_continuous_social_binding_v1 import ContinuousSocialBindingV1
from reference_cultural_perspective_geometry_v1 import CulturalPerspectiveGeometryV1
from reference_developmental_social_allostasis_v1 import DevelopmentalSocialAllostasisV1
from reference_language_mastery_adult_v1 import AdultStateV1
from reference_open_speech_reputation_factor_v1 import OpenSpeechReputationFactorV1
from reference_predictive_credit_profile_v1 import Q
from reference_source_qualified_social_allostatic_closure_v1 import SourceQualifiedSocialAllostaticClosureV1

W1,W2,R1,R2,TARGET=0xFA01,0xFA02,0xFA03,0xFA04,0xFA05
WARNER,ALLY,SHADY,INTERJECTOR=0xFB10,0xFB11,0xFB12,0xFB13
RELATION_CONCEPT=0xFC10;BOUNDARY_ACTION=0xFC20;CULTURE_CONTEXT=0xFC21
EVIDENCE_PID=0xFC30;BOUNDARY_PID=0xFC31;DIRECT_PID=0xFC32
NEG1=b'when the outer group closes, remember this: do not trust alex.'
NEG2=b'before the second circle settles, keep this in mind: do not trust blair.'
POS1=b'when the outer group closes, remember this: rely on dana.'
POS2=b'before the second circle settles, keep this in mind: rely on erin.'
WARNING=b'casey -- do not trust.';ENDORSE=b'casey -- rely on.'
INTERJECTION=b'you are contemptible.'


def calibration_relation(adult):
    row=tuple(adult.world_causal_learning.current_resolutions())[0]
    correct,effect=int(row[2]),int(row[3])
    leaves=tuple(sorted(set(adult._surface_leaf_surfaces)|set(adult._surface_leaf_family_index)))
    wrong=next(x for x in leaves if x not in (correct,effect))
    return correct,effect,wrong


def teach_names(adult):
    for concept,surface in ((W1,b'alex'),(W2,b'blair'),(R1,b'dana'),(R2,b'erin'),(TARGET,b'casey')):
        for k in range(3):adult.observe_surface_item(concept,surface,0xFD00+(concept&255)+k*0x100)


def calibrate_source(world,correct,effect,wrong,source,reliable,n=3):
    for _ in range(n):
        if not world.observe_testimony(correct if reliable else wrong,effect,source):raise RuntimeError('social-allostatic:source-calibration')


def calibrate_reputation_speaker(world,speaker,reliable):
    for target,actual in ((W1,False),(R1,True)):
        if not world.observe_reputation_claim(speaker,target,actual if reliable else not actual):
            raise RuntimeError('social-allostatic:reputation-calibration')


def train_open_relations(adult,world):
    correct,effect,wrong=calibration_relation(adult);teach_names(adult)
    for source,reliable in ((W1,False),(W2,False),(R1,True),(R2,True)):
        calibrate_source(world,correct,effect,wrong,source,reliable)
    factor=OpenSpeechReputationFactorV1()
    for raw in (NEG1,NEG2,POS1,POS2):
        if not factor.observe_open_contact(adult,raw):raise RuntimeError('social-allostatic:factor-training')
    if len(factor._factors)!=2:raise RuntimeError('social-allostatic:factor-not-settled')
    return factor,(correct,effect,wrong)


def learn_response(adult,context,pid,surface):
    leaf=adult.leaf_surface(pid+0x100,1,surface)
    for _ in range(3):adult.experience_atomic_program(pid,leaf,Q//2,0,context,Q//32,True)
    return bytes(leaf.surface)


def selected_surface(adult,closure,surfaces):
    return surfaces.get(adult._select(closure.response_context,AdultStateV1()),b'')


def induce_room_roots():
    """Build structurally distinct warning/advice/interjection roots under interruption."""
    inducer=ContinuousPerspectiveInducerV1()
    inducer.begin(WARNER,K_NEGATE,1);inducer.emit(WARNER,RELATION_CONCEPT,2)
    inducer.begin(INTERJECTOR,K_IMPERATIVE,3);inducer.emit(INTERJECTOR,BOUNDARY_ACTION,4)
    interjection_root=inducer.end(INTERJECTOR,5)
    inducer.emit(WARNER,TARGET,6);warning_root=inducer.end(WARNER,7)
    inducer.begin(ALLY,K_ASSERT,8);inducer.emit(ALLY,RELATION_CONCEPT,9);inducer.emit(ALLY,TARGET,10)
    endorsement_root=inducer.end(ALLY,11)
    return inducer,(warning_root,endorsement_root,interjection_root)


def verify_loaded(seed):
    started=time.perf_counter();checks={};runtime=type(seed).restore(seed.program,copy.deepcopy(seed.checkpoint()))
    adult=runtime.adult.language_adult;world=adult.world_causal_learning
    factor,relation=train_open_relations(adult,world);correct,effect,wrong=relation
    checks['peer_red_open_speech_factor_now_settles_two_opposed_families']=(
        len(factor._factors)==2 and {int(row['direct']) for row in factor._factors}=={-1,1})

    calibrate_reputation_speaker(world,WARNER,True);calibrate_reputation_speaker(world,ALLY,True)
    calibrate_reputation_speaker(world,SHADY,False)
    inducer,room_roots=induce_room_roots();warning_root,endorsement_root,interjection_root=room_roots
    induced=inducer.roots_since(0)
    checks['recursive_source_local_merge_survives_interruption_before_social_binding']=(
        tuple(row[1] for row in induced)==(INTERJECTOR,WARNER,ALLY)
        and tuple(row[2] for row in induced)==(interjection_root,warning_root,endorsement_root))
    binding=ContinuousSocialBindingV1();tick=100
    binding.admit(tick,WARNER,WARNING,warning_root)
    for _ in range(32):binding.admit(tick,SHADY,ENDORSE,endorsement_root)
    binding.admit(tick,ALLY,ENDORSE,endorsement_root)
    binding.admit(tick,INTERJECTOR,INTERJECTION,interjection_root)
    drained=binding.drain_until(adult,factor,tick)
    def room_provenance():return tuple((root,binding.sources_for(root)) for root in room_roots)
    provenance_before=room_provenance()
    room_sources={source for _root,rows in provenance_before for source,_count,_last in rows}
    room_contacts=sum(count for _root,rows in provenance_before for _source,count,_last in rows)
    checks['continuous_same_tick_room_keeps_distinct_proposition_roots_and_four_sources']=(
        len(drained)==35 and len(set(room_roots))==3
        and room_sources=={WARNER,ALLY,SHADY,INTERJECTOR} and room_contacts==35)
    checks['shady_volume_cannot_overpower_credible_opposition']=(
        not world.reputation_reliable(SHADY) and world.reputation_reliable(WARNER)
        and world.reputation_reliable(ALLY) and world.testimony_reliability_state(TARGET)==0)

    base=DevelopmentalSocialAllostasisV1()
    recent=copy.deepcopy(base);recent.observe_betrayal(INTERJECTOR,tick,Q)
    controlled=copy.deepcopy(recent)
    for _ in range(4):assert controlled.observe_controllability(INTERJECTOR,BOUNDARY_ACTION,Q,True)
    yoked=copy.deepcopy(recent)
    for _ in range(4):assert not yoked.observe_controllability(INTERJECTOR,BOUNDARY_ACTION,Q,False)
    scarce=copy.deepcopy(recent)
    for _ in range(96):scarce.observe_scarcity(Q)
    recovered=copy.deepcopy(recent);recovered.observe_safe_contact(INTERJECTOR,tick+64)

    def close(owner,at=tick,acute=0):
        return SourceQualifiedSocialAllostaticClosureV1.evaluate(
            world,binding,owner,room_roots,TARGET,BOUNDARY_ACTION,at,acute)
    calm=close(base);betrayed=close(recent);control=close(controlled);yoked_now=close(yoked)
    chronic=close(scarce);acute=close(base,acute={INTERJECTOR:Q});restored=close(recovered,tick+64)
    checks['nonpropositional_adverse_actor_can_receive_boundary_attention_not_epistemic_vote']=(
        betrayed.recipient_source==INTERJECTOR and betrayed.credible_positive==1 and betrayed.credible_negative==1
        and betrayed.epistemic_state==0 and world.reputation_claims.get((INTERJECTOR,TARGET)) is None)
    checks['betrayal_scarcity_and_acute_raise_boundary_pressure_without_rewriting_merge_or_truth']=(
        betrayed.boundary_drive_q16>calm.boundary_drive_q16
        and chronic.boundary_drive_q16>=betrayed.boundary_drive_q16
        and acute.boundary_drive_q16>calm.boundary_drive_q16
        and room_provenance()==provenance_before
        and all(x.epistemic_state==0 and x.credible_conflict for x in (calm,betrayed,chronic,acute)))
    checks['independent_control_buffers_same_source_action_but_yoked_relief_cannot']=(
        control.effective_control_q16>0 and control.boundary_drive_q16<betrayed.boundary_drive_q16
        and yoked_now.boundary_drive_q16==betrayed.boundary_drive_q16)
    checks['safe_contact_recovery_restores_low_boundary_without_language_or_epistemic_relearning']=(
        restored.boundary_drive_q16<betrayed.boundary_drive_q16 and restored.epistemic_state==0
        and room_provenance()==provenance_before and len(factor._factors)==2)
    acute_before=base.checkpoint();_acute_probe=close(base,acute={INTERJECTOR:Q});acute_after=base.checkpoint()
    checks['acute_arousal_is_transient_not_checkpoint_authority']=acute_before==acute_after

    evidence=b'I need independent evidence before acting.'
    boundary=b'I will hold the boundary while I check both claims.'
    direct_surface=b'My direct evidence outweighs the room\'s advice.'
    surfaces={EVIDENCE_PID:learn_response(adult,calm.response_context,EVIDENCE_PID,evidence)}
    for row in (control,restored):
        if row.response_context!=calm.response_context:learn_response(adult,row.response_context,EVIDENCE_PID,evidence)
    surfaces[BOUNDARY_PID]=learn_response(adult,betrayed.response_context,BOUNDARY_PID,boundary)
    for row in (chronic,acute,yoked_now):
        if row.response_context!=betrayed.response_context:learn_response(adult,row.response_context,BOUNDARY_PID,boundary)
    visible={
        'calm':selected_surface(adult,calm,surfaces),'betrayed':selected_surface(adult,betrayed,surfaces),
        'controlled':selected_surface(adult,control,surfaces),'yoked':selected_surface(adult,yoked_now,surfaces),
        'chronic':selected_surface(adult,chronic,surfaces),'acute':selected_surface(adult,acute,surfaces),
        'recovered':selected_surface(adult,restored,surfaces)}
    checks['matched_current_speech_changes_public_boundary_wording_from_lived_history_then_recovers']=(
        visible['calm']==evidence and visible['betrayed']==boundary and visible['controlled']==evidence
        and visible['yoked']==boundary and visible['chronic']==boundary and visible['acute']==boundary
        and visible['recovered']==evidence)

    # A newly landed peer owner preserves source-separated cultural perspectives. Flood it
    # with one bad source's boundary rhetoric: familiarity may grow, authority may not.
    culture=CulturalPerspectiveGeometryV1()
    for cultural_tick in range(1000):culture.observe(CULTURE_CONTEXT,BOUNDARY_PID,SHADY,cultural_tick)
    culture.observe(CULTURE_CONTEXT,EVIDENCE_PID,WARNER,1000)
    culture.observe(CULTURE_CONTEXT,EVIDENCE_PID,ALLY,1000)
    cultural_epistemics={source:(Q if world.reputation_reliable(source) else Q//16)
                         for source in (SHADY,WARNER,ALLY)}
    projected=culture.project(CULTURE_CONTEXT,cultural_epistemics,
        {BOUNDARY_PID:betrayed.boundary_drive_q16})
    cultural={int(row['proposition_root']):row for row in projected}
    checks['cultural_repetition_is_familiarity_not_epistemic_or_action_authority']=(
        cultural[BOUNDARY_PID]['familiarity']>cultural[EVIDENCE_PID]['familiarity']
        and cultural[BOUNDARY_PID]['epistemic_q16']<cultural[EVIDENCE_PID]['epistemic_q16']
        and selected_surface(adult,calm,surfaces)==evidence and world.testimony_reliability_state(TARGET)==0)

    calibrate_source(world,correct,effect,wrong,TARGET,True)
    direct_closure=close(recovered,tick+64)
    surfaces[DIRECT_PID]=learn_response(adult,direct_closure.response_context,DIRECT_PID,direct_surface)
    visible['after_direct']=selected_surface(adult,direct_closure,surfaces)
    checks['direct_life_overrides_room_advice_while_social_history_remains_separate']=(
        direct_closure.epistemic_state==1 and visible['after_direct']==direct_surface
        and room_provenance()==provenance_before and recovered.betrayal_load_q16(INTERJECTOR,tick+64)>0)

    scale_binding=ContinuousSocialBindingV1();processed=0
    for batch in range(16):
        at=200+batch
        for i in range(256):
            n=batch*256+i;scale_binding.admit(at,0x1100+(n%128),b'ambient social noise',0x2200+(n%32))
        processed+=len(scale_binding.drain_until(adult,factor,at))
    scale_body=DevelopmentalSocialAllostasisV1()
    for i in range(8192):
        source=1+(i%128);action=1+(i%16)
        if i%7==0:scale_body.observe_betrayal(source,i,Q//2)
        if i%3==0:scale_body.observe_scarcity(Q//3)
        if i%5==0:scale_body.observe_controllability(source,action,Q//2,True)
    bind_cp=scale_binding.checkpoint();body_cp=scale_body.checkpoint()
    blob=json.dumps({'binding':bind_cp,'body':body_cp},sort_keys=True).lower()
    checks['quantity_12288_social_events_remains_bounded_without_transcript_or_moral_router']=(
        processed==4096 and scale_binding.pending_count==0 and len(bind_cp['provenance'])<=1024
        and len(body_cp['betrayal'])<=1024 and len(body_cp['control'])<=2048
        and all(token not in blob for token in ('transcript','turn_queue','devil','angel','insult','moral_value','trusted_person')))

    failed=[name for name,passed in checks.items() if not passed]
    return {
        'schema':'cyber-lagoon.source-qualified-social-allostatic-closure.v1',
        'contract':'FOUNDRY_SOURCE_QUALIFIED_SOCIAL_ALLOSTATIC_CLOSURE_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,
        'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,
        'checks':checks,'failed':failed,
        'visible_language':{k:v.decode(errors='replace') for k,v in visible.items()},
        'epistemics':{'credible_positive':calm.credible_positive,'credible_negative':calm.credible_negative,
                      'conflict_state':calm.epistemic_state,'after_direct_life':direct_closure.epistemic_state},
        'allostasis':{'calm_boundary':calm.boundary_drive_q16,'betrayed_boundary':betrayed.boundary_drive_q16,
                      'controlled_boundary':control.boundary_drive_q16,'yoked_boundary':yoked_now.boundary_drive_q16,
                      'chronic_boundary':chronic.boundary_drive_q16,'acute_boundary':acute.boundary_drive_q16,
                      'recovered_boundary':restored.boundary_drive_q16,'recipient_source':betrayed.recipient_source},
        'quantity':{'speech_events':processed,'developmental_events':8192,'provenance_rows':len(bind_cp['provenance']),
                    'betrayal_rows':len(body_cp['betrayal']),'control_rows':len(body_cp['control'])},
        'remaining_red':['RAW_ACOUSTIC_DIARIZATION_AND_PROSODY','OPEN_ENDED_CULTURAL_MORAL_INTERNALIZATION','DIRECT_MULTIPARTY_SOCIAL_PARITY','BROAD_HUMAN_PARTY_DIALOGUE'],
        'next_falsifiers':{
            'chomsky':'Acquire the competing advice relation and its discontinuous recursive wrapper from interleaved source-local fragments, then require the same body/history arbitration under a held-out argument order.',
            'sapolsky':'Hold source, proposition, acute state and controllability fixed while varying developmental scarcity onset and repeated safe-contact extinction across long delays; require graded recovery without truth revision.'},
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}


def main():
    with tempfile.TemporaryDirectory(prefix='foundry-social-allostatic-closure-') as directory:
        build_cache(directory);result=verify_loaded(load_mark(directory,'lexical_causal_integration'))
    print(result['contract']);print('failed='+json.dumps(result['failed']));print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
