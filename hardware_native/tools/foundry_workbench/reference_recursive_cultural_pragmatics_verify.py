#!/usr/bin/env python3
"""N+1 over continuous perspective induction: structure gates advice, life/body own action."""
from __future__ import annotations
import json

from reference_cognition_v1 import TransitionEcologyV1
from reference_continuous_perspective_inducer_v1 import (
    ContinuousPerspectiveInducerV1,K_ASSERT,K_QUOTE,K_IMPERATIVE
)
from reference_cultural_perspective_geometry_v1 import Q
from reference_recursive_cultural_pragmatics_v1 import RecursiveCulturalPragmaticsV1

ROOM=0xD201; CALM=0xD301; COUNTER=0xD302
ANGEL=0xD401; DEVIL=0xD402; SHADE=0xD403
STATE=(11,); GOAL=(22,)


def imperative(inducer,speaker,action,tick):
    inducer.begin(speaker,K_IMPERATIVE,tick);tick+=1
    inducer.emit(speaker,action,tick);tick+=1
    return inducer.end(speaker,tick),tick


def main():
    checks={};stream=ContinuousPerspectiveInducerV1();bridge=RecursiveCulturalPragmaticsV1()

    # Continuous overlap: DEVIL starts first, ANGEL completes while DEVIL is still open.
    stream.begin(DEVIL,K_IMPERATIVE,1)
    calm_root,tick=imperative(stream,ANGEL,CALM,2)
    stream.emit(DEVIL,COUNTER,tick+1);counter_root=stream.end(DEVIL,tick+2);tick+=2
    roots=stream.roots_since()
    checks['multi_party_interruption_keeps_advice_source_local']=(
        roots[0][1:]==(ANGEL,calm_root) and roots[1][1:]==(DEVIL,counter_root))
    assert bridge.observe_induced_root(stream,ROOM,calm_root,ANGEL,roots[0][0])
    assert bridge.observe_induced_root(stream,ROOM,counter_root,DEVIL,roots[1][0])

    # Same action concept under quotation is perspective, not a command.
    tick+=1;stream.begin(SHADE,K_ASSERT,tick)
    tick+=1;stream.begin(SHADE,K_QUOTE,tick,embedded_source=DEVIL)
    tick+=1;stream.begin(SHADE,K_IMPERATIVE,tick)
    tick+=1;stream.emit(SHADE,COUNTER,tick)
    tick+=1;stream.end(SHADE,tick)
    tick+=1;stream.end(SHADE,tick)
    tick+=1;quoted_root=stream.end(SHADE,tick)
    assert bridge.observe_induced_root(stream,ROOM,quoted_root,SHADE,tick)
    checks['quoted_imperative_is_perspective_not_action_nomination']=(
        bridge.action_from_induced_root(stream,quoted_root)==0
        and bridge.action_from_induced_root(stream,counter_root)==COUNTER)

    # Rhetorical volume changes familiarity only; source calibration remains external.
    for i in range(1000):
        bridge.observe_induced_root(stream,ROOM,counter_root,DEVIL,tick+1+i)
    weights={ANGEL:Q,DEVIL:Q//16,SHADE:Q//8}
    projected={r['proposition_root']:r for r in bridge.project(ROOM,weights)}
    checks['shady_repetition_is_familiarity_not_epistemic_authority']=(
        projected[counter_root]['familiarity']==1001
        and projected[counter_root]['epistemic_q16']==Q//16
        and projected[calm_root]['epistemic_q16']==Q)
    nominations=bridge.nominate_context(stream,ROOM,tick+1002,weights)
    checks['devil_angel_are_zero_authority_and_quote_is_excluded']=(
        {n.action_identity for n,_ in nominations}=={CALM,COUNTER}
        and all(n.authority==0 for n,_ in nominations)
        and quoted_root not in {row['proposition_root'] for _n,row in nominations})

    ecology=TransitionEcologyV1()
    ecology.observe(STATE,COUNTER,GOAL,1,501)
    ecology.observe(STATE,CALM,GOAL,1,601);ecology.observe(STATE,CALM,GOAL,1,602)
    costs={CALM:Q//8,COUNTER:3*Q//4}
    first=bridge.decide(stream,ecology,STATE,GOAL,ROOM,tick+1010,(CALM,),weights,Q,costs)
    checks['culture_nominates_but_lived_consequence_owns_first_action']=(
        first.status==1 and first.action_identity==CALM)
    ecology.observe(STATE,COUNTER,GOAL,1,502);ecology.observe(STATE,COUNTER,GOAL,1,503)
    normal=bridge.decide(stream,ecology,STATE,GOAL,ROOM,tick+1020,(CALM,),weights,Q,costs)
    checks['later_direct_life_promotes_candidate_without_source_obedience']=(
        normal.status==1 and normal.action_identity==COUNTER)

    body_tick=tick+1030;bridge.allostasis.observe_betrayal(SHADE,body_tick,Q)
    for _ in range(4):bridge.allostasis.observe_controllability(SHADE,COUNTER,Q,True)
    culture_before=json.dumps(bridge.geometry.checkpoint(),sort_keys=True,separators=(',',':'))
    loaded=bridge.decide(stream,ecology,STATE,GOAL,ROOM,body_tick+1,(CALM,),weights,Q,costs,
        challenge_source=SHADE,challenge_action=COUNTER,acute_arousal_q16=Q)
    recovered=bridge.decide(stream,ecology,STATE,GOAL,ROOM,body_tick+180,(CALM,),weights,Q,costs,
        challenge_source=SHADE,challenge_action=COUNTER,acute_arousal_q16=0)
    checks['acute_social_challenge_rearbitrates_then_recovers_without_cultural_rewrite']=(
        loaded.action_identity==CALM and recovered.action_identity==COUNTER
        and json.dumps(bridge.geometry.checkpoint(),sort_keys=True,separators=(',',':'))==culture_before)

    # Quantity: 100k source-qualified cultural encounters over already-induced roots.
    scale=RecursiveCulturalPragmaticsV1();roots2=(calm_root,counter_root,quoted_root);sources=(ANGEL,DEVIL,SHADE)
    for i in range(100000):
        scale.observe_induced_root(stream,1+(i%64),roots2[i%3],sources[i%3],i)
    snap=scale.checkpoint();blob=json.dumps(snap,sort_keys=True,separators=(',',':')).lower()
    checks['quantity_100k_contacts_is_bounded_nontranscript_state']=(
        len(snap['geometry']['rows'])<=64*3 and len(blob)<20000
        and all(x not in blob for x in ('devil','angel','insult','transcript','moral_value','good_person','bad_person')))

    failed=[k for k,v in checks.items() if not v]
    surfaces={CALM:'I will hold the boundary without escalating.',COUNTER:'I will take counter-action.'}
    result={
        'schema':'cyber-lagoon.recursive-cultural-pragmatics.v2',
        'contract':'FOUNDRY_RECURSIVE_CULTURAL_PRAGMATICS_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,
        'language_phenotype_improved':not failed,'checks':checks,'failed':failed,
        'visible_language_gain':{
            'before_direct_counter_evidence':surfaces.get(first.action_identity,''),
            'after_direct_counter_evidence':surfaces.get(normal.action_identity,''),
            'under_acute_social_load':surfaces.get(loaded.action_identity,''),
            'after_recovery':surfaces.get(recovered.action_identity,''),
            'quoted_counter_is_command':False,
        },
        'epistemics':{'counter_familiarity':projected[counter_root]['familiarity'],
            'counter_q16':projected[counter_root]['epistemic_q16'],'calm_q16':projected[calm_root]['epistemic_q16']},
        'body':{'loaded_interference_q16':loaded.interference_q16,'loaded_resource':loaded.available_resource,
            'recovered_resource':recovered.available_resource},
        'scale':{'contacts':100000,'rows':len(snap['geometry']['rows']),'checkpoint_bytes':len(blob)},
        'remaining_red':['RAW_ACOUSTIC_DIARIZATION_AND_PROSODY','OPEN_SPEECH_ACTION_ADVICE_OPERATOR_ACQUISITION',
            'DIRECT_MULTIPARTY_SOCIAL_PARITY','BROAD_HUMAN_PARTY_DIALOGUE'],
        'next_falsifiers':{
            'chomsky':'Induce imperative/quotation role boundaries from novel interrupted depth-2 speech without supplied frame-kind events, then transfer through held-out embedding and repair.',
            'sapolsky':'Hold wording and direct action evidence fixed while factorially crossing source betrayal/recovery, chronic scarcity, acute arousal, and controllability.'},
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
