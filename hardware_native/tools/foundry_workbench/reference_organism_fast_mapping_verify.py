#!/usr/bin/env python3
"""Consequence-stabilized one-exposure word learning in one continuing organism."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801
REMOTE,INSPECT,SENSOR,NEW,ALT,CATEGORY,LOW,UNRELATED,OTHER=601,303,403,7777,7778,7779,7780,7781,7782
MOTOR_TEST,MOTOR_INSPECT=7302,8303


def u(text):return tuple(text.encode())


PROFILES=(
    ('english',41001,'zoe','inspects','sensor','dax','zoe inspects the sensor.','zoe inspects the dax.'),
    ('german',42001,'lena','prüft','fühler','blicket','lena prüft den fühler.','lena prüft den blicket.'),
    ('chin',43001,'mi','zoh','sen','wug','mi nih sen kha a zoh.','mi nih wug kha a zoh.'),
    ('mandarin',44001,'卓伊','检查','传感器','达克斯','卓伊检查传感器。','卓伊检查达克斯。'),
    ('russian',45001,'зоя','проверяет','датчик','дакс','зоя проверяет датчик.','зоя проверяет дакс.'),
)


def features(o,entity,values,source):
    o.contact(CONTACT_ENTITY_FEATURES,(entity,len(values),*values),source,True,True)


def name(o,entity,text,source):
    o.contact(CONTACT_SCENE,(7,0,1,entity),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def clause(o,atoms,text,source):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def prepare(spec):
    o=ReferenceOrganismV2(spec)
    for entity,values,source in (
        (REMOTE,(71,72,73,74),30101),(INSPECT,(35,36),30102),
        (SENSOR,(45,46),30103),(NEW,(81,82,83),30104),
    ):features(o,entity,values,source)
    for index,profile in enumerate(PROFILES):
        _label,partner,subject,action,known,novel,demonstration,_command=profile
        second=partner+5000
        for entity,text in ((REMOTE,subject),(INSPECT,action),(SENSOR,known)):
            name(o,entity,text,partner);name(o,entity,text,second)
        clause(o,(REMOTE,INSPECT,SENSOR),demonstration,partner)
        clause(o,(REMOTE,INSPECT,SENSOR),demonstration,second)
        name(o,NEW,novel,partner) # One actual naming encounter: eligibility, not credit.
        lexeme=o.language.lexeme_identity(INSPECT,u(action))
        if not o._ground_language_action_recruitment(lexeme,MOTOR_INSPECT,60000+index,1,True):
            raise AssertionError('action grounding')
    return o


def stage(o,state,source,actions=(MOTOR_TEST,MOTOR_INSPECT),prime=True,target=NEW):
    o.contact(CONTACT_WORLD_STATE,(state,),source,True,True)
    o.contact(CONTACT_BODY_TARGET,(target,),source+1000,True,True)
    o.contact(CONTACT_AFFORDANCES,actions,source+2000,True,True)
    if prime:
        # Neutral resident trials establish a counterfactual without world credit.
        for _ in range(len(actions)):
            if o._exploration_candidate()==MOTOR_TEST:break
            trial=o.tick()
            if not isinstance(trial,MotorActionV2):raise AssertionError('resident exploration')
            o.contact(CONTACT_MOTOR_CONSEQUENCE,(trial.ticket,0,1,state),source,True,True)
    return o._exploration_candidate()


def command(o,profile,state,source,effect=1,independent=True,actions=(MOTOR_TEST,MOTOR_INSPECT),prime=True,target=NEW):
    baseline=stage(o,state,source,actions,prime,target)
    assertion=o.contact(CONTACT_SOURCE_UTTERANCE,u(profile[7]),profile[1],True,True)
    action=o.tick()
    if not isinstance(action,MotorActionV2):return assertion,action,{},baseline
    nxt=(state,NEW) if effect>0 else (state,)
    learned=o.contact(CONTACT_MOTOR_CONSEQUENCE,(action.ticket,effect,len(nxt),*nxt),source,True,independent)
    return assertion,action,learned,baseline


def outward(o,profile,source,entity=NEW):
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,profile[1]),source+1,True,True)
    sid=o.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,entity),source,True,True)
    action=o.tick()
    payload=() if not isinstance(action,ActionV2) else action.payload
    if isinstance(action,ActionV2):o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    return sid,payload


def outcome(o,profile,feature=NEW):
    key=(feature,u(profile[5]))
    return len(o.language._lexeme_positive.get(key,())),len(o.language._lexeme_counter.get(key,()))


def refused(checkpoint):
    try:ReferenceOrganismV2.restore(checkpoint);return False
    except ValueError:return True


def main():
    started=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8)
    base=prepare(spec);base_checkpoint=copy.deepcopy(base.checkpoint())
    checks['one_exposure_is_provisional_not_outward_truth']=(
        base.language.lexeme(NEW) is None
        and all(len(base.language.invert_surface(u(row[7])))==1 for row in PROFILES))

    adult=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint));causal_rows=[]
    for index,profile in enumerate(PROFILES):
        assertion,action,learned,baseline=command(adult,profile,9100+index,7100+index)
        causal_rows.append((assertion,action,learned,baseline,outcome(adult,profile)))
    checks['five_conditions_change_resident_action'] = all(
        assertion>0 and isinstance(action,MotorActionV2)
        and action.action_id==MOTOR_INSPECT and baseline==MOTOR_TEST
        and action.source_counterfactual_action==MOTOR_TEST
        and learned.get('source_credit',0)>0 and learned.get('lexeme_settlement')==u(profile[5])
        and result==(1,0)
        for profile,(assertion,action,learned,baseline,result) in zip(PROFILES,causal_rows))
    checks['shared_feature_remains_globally_ambiguous']=adult.language.lexeme(NEW) is None

    outputs={}
    for index,profile in enumerate(PROFILES):
        _sid,payload=outward(adult,profile,8100+index);outputs[profile[0]]=bytes(payload).decode()
    checks['five_partner_histories_reconstruct_correct_surface']=all(outputs[row[0]]==row[7] for row in PROFILES)
    positive_checkpoint=copy.deepcopy(adult.checkpoint())
    restored=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint))
    checks['checkpoint_exact']=restored.digest()==adult.digest() and restored.language.digest()==adult.language.digest()

    extension=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint));features(extension,CATEGORY,(81,82,83,84),9701)
    extended={}
    for index,profile in enumerate(PROFILES):
        _sid,payload=outward(extension,profile,9710+index,CATEGORY);extended[profile[0]]=bytes(payload).decode()
    checks['distributed_novel_exemplar_extends_five_partner_lexicons']=(
        NEW in extension._overlapping_entities(CATEGORY)
        and all(extended[row[0]]==row[7] for row in PROFILES))
    extension_before=outcome(extension,PROFILES[0]);baseline=stage(extension,9720,7720,target=CATEGORY)
    assertion=extension.contact(CONTACT_SOURCE_UTTERANCE,u(PROFILES[0][7]),PROFILES[0][1],True,True);category_action=extension.tick()
    row=next(item for item in extension.source_assertions if item.identity==assertion)
    category_learning=extension.contact(CONTACT_MOTOR_CONSEQUENCE,(category_action.ticket,1,2,9720,CATEGORY),7720,True,True)
    checks['raw_word_resolves_current_distributed_target_without_new_lexeme']=(
        baseline==MOTOR_TEST and row.binding_atoms==(REMOTE,INSPECT,CATEGORY)
        and isinstance(category_action,MotorActionV2) and category_action.action_id==MOTOR_INSPECT
        and outcome(extension,PROFILES[0])==extension_before and 'lexeme_settlement' not in category_learning)
    features(extension,LOW,(81,82,90,91),9702);features(extension,UNRELATED,(91,92,93,94),9703)
    extension.contact(CONTACT_PARTNER_CONTEXT,(1,7,PROFILES[0][1]),9704,True,True)
    checks['subthreshold_and_unrelated_entities_do_not_extend']=(
        extension._realized_lexeme(LOW) is None and extension._realized_lexeme(UNRELATED) is None)

    competing=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint));alt_profile=(
        'english-alt',PROFILES[0][1],PROFILES[0][2],PROFILES[0][3],PROFILES[0][4],'mip',
        PROFILES[0][6],'zoe inspects the mip.')
    features(competing,OTHER,(81,82,83,85),9730);competing.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),9730,True,True);name(competing,OTHER,'mip',PROFILES[0][1])
    _a,_action,_learned,_baseline=command(competing,alt_profile,9731,7731,target=OTHER)
    features(competing,CATEGORY,(81,82,83,84),9732)
    competing.contact(CONTACT_PARTNER_CONTEXT,(1,7,PROFILES[0][1]),9733,True,True)
    checks['competing_distributed_words_preserve_outward_ambiguity']=(
        outcome(competing,alt_profile,OTHER)==(1,0) and competing._realized_lexeme(CATEGORY) is None)

    extension_checkpoint=copy.deepcopy(extension.checkpoint());extension_replay=ReferenceOrganismV2.restore(copy.deepcopy(extension_checkpoint))
    checks['distributed_extension_is_derived_and_checkpoint_exact']=(
        extension_replay.digest()==extension.digest()
        and 'category_extension' not in json.dumps(extension_checkpoint)
        and NEW in extension_replay._overlapping_entities(CATEGORY))
    feature_cut=ReferenceOrganismV2.restore(copy.deepcopy(extension_checkpoint));feature_cut.contact(CONTACT_WITHDRAW_SOURCE,(9701,),9740,True,True)
    feature_cut.contact(CONTACT_PARTNER_CONTEXT,(1,7,PROFILES[0][1]),9741,True,True)
    name_cut=ReferenceOrganismV2.restore(copy.deepcopy(extension_checkpoint));name_cut.contact(CONTACT_WITHDRAW_SOURCE,(PROFILES[0][1],),9742,True,True)
    name_cut.contact(CONTACT_PARTNER_CONTEXT,(1,7,PROFILES[0][1]),9743,True,True)
    extension_lesion=ReferenceOrganismV2.restore(copy.deepcopy(extension_checkpoint));extension_lesion.language._lexeme_positive.pop((NEW,u(PROFILES[0][5])))
    extension_lesion.contact(CONTACT_PARTNER_CONTEXT,(1,7,PROFILES[0][1]),9744,True,True)
    checks['extension_requires_live_feature_source_name_source_and_confirmation']=(
        feature_cut._realized_lexeme(CATEGORY) is None
        and name_cut._realized_lexeme(CATEGORY) is None
        and extension_lesion._realized_lexeme(CATEGORY) is None)
    legacy=copy.deepcopy(base_checkpoint);legacy['language']['schema']=1;legacy['language'].pop('lexeme_outcomes',None)
    legacy_restored=ReferenceOrganismV2.restore(legacy)
    checks['schema_one_checkpoint_migrates_without_credit_invention']=(
        legacy_restored.language.lexeme(NEW) is None
        and len(legacy_restored.language.invert_surface(u(PROFILES[0][7])))==1
        and not legacy_restored.language._lexeme_positive and not legacy_restored.language._lexeme_counter)

    revised=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint));first=PROFILES[0]
    _a,negative_action,negative_learning,_b=command(revised,first,9401,7401,-1,True)
    revised.contact(CONTACT_PARTNER_CONTEXT,(1,7,first[1]),9402,True,True)
    checks['confirmed_mapping_is_causally_revisable']=(
        isinstance(negative_action,MotorActionV2) and bool(negative_action.lexical_occurrences)
        and negative_learning.get('lexeme_settlement') is None and outcome(revised,first)==(1,1)
        and revised._realized_lexeme(NEW) is None)
    _a,reconfirm_action,reconfirm_learning,_b=command(revised,first,9404,7404,1,True)
    _sid,reconfirmed_surface=outward(revised,first,9405)
    checks['later_independent_confirmation_restores_mapping']=(
        isinstance(reconfirm_action,MotorActionV2) and reconfirm_learning.get('lexeme_settlement')==u(first[5])
        and outcome(revised,first)==(2,1) and reconfirmed_surface==u(first[7]))

    stale=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint));baseline=stage(stale,9501,7501)
    stale_action=stale.tick();stale_learning=stale.contact(
        CONTACT_MOTOR_CONSEQUENCE,(stale_action.ticket,-1,1,9501),7501,True,True)
    checks['reinstated_old_assertion_has_no_fresh_lexical_eligibility']=(
        baseline==MOTOR_TEST and isinstance(stale_action,MotorActionV2)
        and stale_action.action_id==MOTOR_INSPECT and not stale_action.lexical_occurrences
        and 'lexeme_settlement' not in stale_learning and outcome(stale,first)==(1,0))

    inflight=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint));stage(inflight,9601,7601)
    inflight.contact(CONTACT_SOURCE_UTTERANCE,u(first[7]),first[1],True,True);pending=inflight.tick()
    inflight_checkpoint=copy.deepcopy(inflight.checkpoint());inflight_restored=ReferenceOrganismV2.restore(copy.deepcopy(inflight_checkpoint))
    replay_action=next(row for row in inflight_restored.motor_actions if row.ticket==pending.ticket)
    replay_learning=inflight_restored.contact(CONTACT_MOTOR_CONSEQUENCE,(replay_action.ticket,1,2,9601,NEW),7601,True,True)
    tampered=copy.deepcopy(inflight_checkpoint);lexical=list(tampered['motor_actions'][-1]['lexical_occurrences']);member=list(lexical[0]);member[2]+=1;lexical[0]=tuple(member);tampered['motor_actions'][-1]['lexical_occurrences']=tuple(lexical)
    checks['inflight_eligibility_checkpoint_exact_and_committed']=(
        replay_learning.get('lexeme_settlement')==u(first[5]) and outcome(inflight_restored,first)==(1,0)
        and refused(tampered))

    negative=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint))
    _a,action,learned,_b=command(negative,PROFILES[0],9201,7201,-1,True)
    checks['negative_consequence_counters_without_promotion']=(
        isinstance(action,MotorActionV2) and learned.get('lexeme_settlement') is None
        and outcome(negative,PROFILES[0])==(0,1)
        and not negative.language.invert_surface(u(PROFILES[0][7])))

    yoked=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint))
    _a,action,learned,_b=command(yoked,PROFILES[0],9202,7202,1,False)
    checks['yoked_positive_return_cannot_promote']=(
        isinstance(action,MotorActionV2) and 'lexeme_settlement' not in learned
        and outcome(yoked,PROFILES[0])==(0,0)
        and len(yoked.language.invert_surface(u(PROFILES[0][7])))==1)

    same=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint))
    _a,action,learned,baseline=command(same,PROFILES[0],9203,7203,1,True,(MOTOR_INSPECT,),False)
    checks['same_counterfactual_action_cannot_promote']=(
        isinstance(action,MotorActionV2) and baseline==MOTOR_INSPECT
        and action.source_counterfactual_action==MOTOR_INSPECT
        and 'lexeme_settlement' not in learned and outcome(same,PROFILES[0])==(0,0))

    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint))
    features(ambiguous,ALT,(91,92,93),9300);name(ambiguous,ALT,PROFILES[0][5],PROFILES[0][1])
    baseline=stage(ambiguous,9301,7301)
    ambiguous.contact(CONTACT_BODY_TARGET,(NEW,ALT),9302,True,True)
    bindings=ambiguous.language.invert_surface(u(PROFILES[0][7]))
    assertion=ambiguous.contact(CONTACT_SOURCE_UTTERANCE,u(PROFILES[0][7]),PROFILES[0][1],True,True)
    checks['referential_ambiguity_refuses_credit']=(
        baseline==MOTOR_TEST and len(bindings)==2 and assertion==0
        and isinstance((fallback:=ambiguous.tick()),MotorActionV2)
        and fallback.action_id==MOTOR_TEST and not fallback.source_assertion_ids
        and outcome(ambiguous,PROFILES[0])==(0,0))

    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(first[1],),99901,True,True)
    withdrawn.contact(CONTACT_PARTNER_CONTEXT,(1,7,first[1]),99902,True,True)
    withdrawn.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,NEW),99903,True,True)
    checks['sole_source_withdrawal_blocks_use_but_retains_history']=(
        outcome(withdrawn,first)==(1,0) and not withdrawn.language.invert_surface(u(first[7]))
        and withdrawn.tick() is None)

    lesion=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint));lesion.language._lexeme_positive.pop((NEW,u(first[5])))
    lesion.contact(CONTACT_PARTNER_CONTEXT,(1,7,first[1]),99801,True,True)
    lesion.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,NEW),99802,True,True)
    failed=lesion.tick()
    _sid,spared=outward(lesion,PROFILES[1],99803)
    checks['focal_confirmation_lesion_is_selective']=failed is None and bytes(spared).decode()==PROFILES[1][7]

    corrupt=copy.deepcopy(positive_checkpoint);row=corrupt['language']['lexeme_outcomes'][0]
    row['counter']=list(row['positive'])
    missing=copy.deepcopy(positive_checkpoint);victim=missing['language']['lexeme_outcomes'][0]
    missing['language']['lexemes']=[row for row in missing['language']['lexemes'] if not (row['feature']==victim['feature'] and row['units']==victim['units'])]
    checks['corrupt_outcome_checkpoint_refused']=refused(corrupt) and refused(missing)

    quantity=ReferenceOrganismV2.restore(copy.deepcopy(base_checkpoint));probe=u(PROFILES[0][7])
    before=quantity.language.invert_surface(probe);touches_before=quantity.language.last_lookup_touches
    bytes_before=len(json.dumps(quantity.checkpoint(),separators=(',',':')).encode())
    for index in range(512):name(quantity,20000+index,f'~q{index:04d}',88000+index)
    after=quantity.language.invert_surface(probe);touches_after=quantity.language.last_lookup_touches
    bytes_after=len(json.dumps(quantity.checkpoint(),separators=(',',':')).encode())
    checks['sparse_inverse_work_survives_512_decoys']=(before==after and touches_before==touches_after)
    category_quantity=ReferenceOrganismV2.restore(copy.deepcopy(positive_checkpoint));features(category_quantity,CATEGORY,(81,82,83,84),9810)
    category_quantity.contact(CONTACT_BODY_TARGET,(CATEGORY,),9811,True,True)
    aliases_before=category_quantity._command_form_aliases(PROFILES[0][1],probe);category_touches_before=category_quantity.last_entity_candidate_touches
    for index in range(512):features(category_quantity,30000+index,(100000+index*4,100001+index*4,100002+index*4,100003+index*4),99000+index)
    aliases_after=category_quantity._command_form_aliases(PROFILES[0][1],probe);category_touches_after=category_quantity.last_entity_candidate_touches
    checks['sparse_category_nomination_survives_512_feature_decoys']=(
        aliases_before==aliases_after and aliases_before.get(u(PROFILES[0][5]))=={CATEGORY}
        and category_touches_before==category_touches_after)

    result={
        'schema':'agi.reference-organism-fast-mapping.v1','pass':all(checks.values()),'checks':checks,
        'languages':list(outputs),'outputs':outputs,'positive_mappings':sum(len(v) for v in adult.language._lexeme_positive.values()),
        'inverse_touches':{'before':touches_before,'after':touches_after},
        'quantity':{'decoys':512,'checkpoint_growth_bytes':bytes_after-bytes_before,
                    'category_touches_before':category_touches_before,'category_touches_after':category_touches_after},
        'claim':'REFERENCE_FAST_MAPPING_CAUSAL_ELIGIBILITY_ONLY_NOT_DIRECT_PARITY_OR_HUMAN_LANGUAGE_MASTERY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_REFERENCE_ORGANISM_FAST_MAPPING '+('GREEN' if result['pass'] else 'RED')+
          f" languages={len(outputs)} positive={result['positive_mappings']} decoys=512 touches={touches_before}->{touches_after}")
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))
    raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
