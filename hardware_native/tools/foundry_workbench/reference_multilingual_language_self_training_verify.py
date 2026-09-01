#!/usr/bin/env python3
from __future__ import annotations
import copy,json
from reference_cognition_v1 import TransitionEcologyV1
from reference_developmental_social_allostasis_v1 import DevelopmentalSocialAllostasisV1,Q
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_open_language_action_affordance_v1 import OpenLanguageActionAffordanceV1
from reference_recursive_cultural_pragmatics_v1 import RecursiveCulturalPragmaticsV1
CALM=0xE701;COUNTER=0xE702;ROOM=0xE703;STATE=(101,);GOAL=(202,);TEACHERS=(0xE710,0xE711)
CURRICULUM={
'english':{CALM:("hold the boundary calmly please","hold the boundary calmly now","hold the boundary calmly today"),COUNTER:("take counter action firmly please","take counter action firmly now","take counter action firmly today")},
'german':{CALM:("halte die grenze ruhig bitte","halte die grenze ruhig jetzt","halte die grenze ruhig heute"),COUNTER:("setz eine gegenaktion klar bitte","setz eine gegenaktion klar jetzt","setz eine gegenaktion klar heute")},
'russian':{CALM:("держи границу спокойно пожалуйста","держи границу спокойно сейчас","держи границу спокойно сегодня"),COUNTER:("предприми ответное действие твердо пожалуйста","предприми ответное действие твердо сейчас","предприми ответное действие твердо сегодня")},
'japanese':{CALM:("境界を静かに守ってください","境界を静かに守って今","境界を静かに守ってね"),COUNTER:("対抗行動をはっきり取ってください","対抗行動をはっきり取って今","対抗行動をはっきり取ってね")},
'mandarin':{CALM:("请冷静地守住边界现在","请冷静地守住边界一下","请冷静地守住边界吧"),COUNTER:("请明确采取反制行动现在","请明确采取反制行动一下","请明确采取反制行动吧")},
'mixed':{CALM:("bitte hold the boundary 冷静に please","bitte hold the boundary 冷静に now","bitte hold the boundary 冷静に heute"),COUNTER:("bitte take counter action はっきり please","bitte take counter action はっきり now","bitte take counter action はっきり heute")},
'denglish_jugend':{CALM:("bro halt die boundary entspannt bitte","bro halt die boundary entspannt jetzt","bro halt die boundary entspannt safe"),COUNTER:("bro mach counter action stabil bitte","bro mach counter action stabil jetzt","bro mach counter action stabil safe")}}
def u(text):return text.encode('utf-8')
def train_one_life(learner):
    tick=1;timeline=[];varieties=tuple(CURRICULUM)
    # Interleave varieties inside each source/action sweep: there is no language phase or reset.
    for source_index,source in enumerate(TEACHERS):
        if source_index==0:order=varieties
        else:
            rev=tuple(reversed(varieties));order=rev[1:]+rev[:1]
        for action in ((CALM,COUNTER) if source_index==0 else (COUNTER,CALM)):
            for variety in order:
                text=CURRICULUM[variety][action][source_index]
                assert learner.observe_language(u(text),source,tick);timeline.append((tick,variety,'language',source,text));tick+=1
                assert learner.observe_action(action,source,tick);timeline.append((tick,variety,'action',source,action));tick+=1
    return tick,timeline

def main():
    checks={};bridge=RecursiveCulturalPragmaticsV1();learner=bridge.affordances;tick,timeline=train_one_life(learner)
    language_sequence=[row[1] for row in timeline if row[2]=='language']
    checks['one_continuing_multilingual_curriculum_chain']=(len(timeline)==56 and tick==57 and set(language_sequence)==set(CURRICULUM))
    checks['curriculum_interleaves_varieties_without_language_phase']=(all(language_sequence[i]!=language_sequence[i+1] for i in range(len(language_sequence)-1)))
    checks['mechanism_has_no_language_or_imperative_classifier']=(not hasattr(learner,'language_id') and not hasattr(learner,'imperative') and not hasattr(learner,'translation'))
    heldout={};wrapped={}
    for variety,actions in CURRICULUM.items():
        for action in (CALM,COUNTER):
            heldout[(variety,action)]=learner.candidates(u(actions[action][2]))
            wrapped[(variety,action)]=learner.candidates(u('someone said: '+actions[action][2]))
    checks['same_mechanism_transfers_in_all_seven_varieties']=(all(rows and rows[0][0]==action for (_v,action),rows in heldout.items()) and len(heldout)==14)
    checks['unseen_outer_quote_wrapper_does_not_inherit_action_force']=all(not rows for rows in wrapped.values())

    ecology=TransitionEcologyV1();bank=LanguageActionNominationBankV1()
    ecology.observe(STATE,CALM,GOAL,1,0xE720);ecology.observe(STATE,CALM,GOAL,1,0xE721)
    sample=u(CURRICULUM['mixed'][COUNTER][2]);noms=learner.nominate(bank,sample,ROOM,tick)
    first=bank.arbitrate(ecology,STATE,GOAL,(CALM,),Q,{CALM:Q//8,COUNTER:Q//8})
    checks['language_nominates_without_motor_authority']=(len(noms)==1 and noms[0].action_identity==COUNTER and noms[0].authority==0 and first.status==1 and first.action_identity==CALM)
    ecology.observe(STATE,COUNTER,GOAL,2,0xE722);ecology.observe(STATE,COUNTER,GOAL,2,0xE723)
    second=bank.arbitrate(ecology,STATE,GOAL,(CALM,),Q,{CALM:Q//8,COUNTER:Q//8})
    checks['later_lived_action_evidence_not_language_promotes_counter_action']=(second.status==1 and second.action_identity==COUNTER)

    allostasis=DevelopmentalSocialAllostasisV1();challenge=0xE730
    allostasis.observe_betrayal(challenge,tick,Q);before=copy.deepcopy(learner.checkpoint())
    loaded=allostasis.appraise(challenge,COUNTER,tick,Q);available=max(0,Q-loaded['interference_q16'])
    bank.clear();learner.nominate(bank,sample,ROOM,tick+1)
    under_load=bank.arbitrate(ecology,STATE,GOAL,(CALM,),available,{CALM:Q//8,COUNTER:3*Q//4})
    recovered=allostasis.appraise(challenge,COUNTER,tick+160,0);available2=max(0,Q-recovered['interference_q16'])
    bank.clear();learner.nominate(bank,sample,ROOM,tick+161)
    after_recovery=bank.arbitrate(ecology,STATE,GOAL,(CALM,),available2,{CALM:Q//8,COUNTER:3*Q//4})
    checks['body_history_rearbitrates_then_recovers_without_language_rewrite']=(under_load.status==0 and under_load.action_identity==0 and after_recovery.action_identity==COUNTER and learner.checkpoint()==before)

    ephemeral=0xE740;assert learner.observe_language(b'unrelated unfinished speech',ephemeral,tick+200)
    cp=learner.checkpoint();restored=OpenLanguageActionAffordanceV1.restore(copy.deepcopy(cp))
    checks['pending_language_occurrence_is_not_checkpoint_transcript']=(learner.pending_count==1 and restored.pending_count==0 and 'unfinished' not in json.dumps(cp))

    scale=OpenLanguageActionAffordanceV1();st=1
    for i in range(100):
        for source,suffix in ((0xF000+i*2,'alpha'),(0xF001+i*2,'beta')):
            raw=(f'form{i:03d}-shared-action-pattern-{suffix}').encode();assert scale.observe_language(raw,source,st);st+=1;assert scale.observe_action(CALM,source,st);st+=1
    transfer=0;max_touches=0
    for i in range(100):
        rows=scale.candidates((f'form{i:03d}-shared-action-pattern-gamma').encode());transfer+=int(bool(rows and rows[0][0]==CALM));max_touches=max(max_touches,scale.last_match_touches)
    scp=scale.checkpoint();checks['n_plus_100_same_law_transfers_without_new_semantic_classes']=(transfer==100 and scale.factor_count>=100 and scale.factor_count<=512 and max_touches<=scale.factor_count and all(token not in json.dumps(scp).lower() for token in ('english','german','russian','japanese','mandarin','imperative','translation','transcript')))

    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.multilingual-language-self-training.v1','contract':'FOUNDRY_MULTILINGUAL_LANGUAGE_SELF_TRAINING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,'checks':checks,'failed':failed,'curriculum_chain':tuple(CURRICULUM),'heldout':{f'{v}:{"calm" if a==CALM else "counter"}':bool(rows and rows[0][0]==a) for (v,a),rows in heldout.items()},'visible_language':{v:CURRICULUM[v][CALM][2] for v in CURRICULUM},'action_arbitration':{'before_counter_life':first.action_identity,'after_counter_life':second.action_identity,'under_load':under_load.action_identity,'after_recovery':after_recovery.action_identity},'quantity':{'additional_surface_families':100,'transfer':transfer,'factor_rows':scale.factor_count,'max_match_touches':max_touches},'remaining_red':['RAW_AUDIO_PHONOLOGICAL_OPERATOR_INDUCTION','RECURSIVE_ROLE_INDUCTION_WITHOUT_FRAME_KINDS','DIRECT_MULTILINGUAL_SOCIAL_PARITY','BROAD_HUMAN_DIALOGUE'],'next_falsifiers':{'chomsky':'Learn quotation/negation/action-role distinctions from raw continuous multilingual speech and lived consequences, then transfer through unseen recursive embedding without frame-kind events.','sapolsky':'Hold one multilingual phrase and transition ecology fixed while crossing source betrayal/recovery, chronic scarcity, acute arousal, controllable versus yoked consequences, and long-delay extinction.'}}
    print(result['contract']);print('failed='+json.dumps(failed));print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
