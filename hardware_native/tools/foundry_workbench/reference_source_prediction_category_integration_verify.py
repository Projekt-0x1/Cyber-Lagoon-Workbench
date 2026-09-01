#!/usr/bin/env python3
"""Whole-organism receipt separating source prediction accuracy from advice consequence valence."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

WORLD=6101;SRC=6102;TARGET_SRC=6103;AFF=6104;A=31;B=32
S0=(101,);PRED=(201,);WRONG=(301,);TARGET_A=(901,);TARGET_B=(902,)

def organism():return ReferenceOrganismV2(PopulationSpecV1(32768,fanout=2,sites_per_feature=4,eligibility_horizon=8))
def setup(o,target=TARGET_A):
    o.contact(CONTACT_WORLD_STATE,S0,WORLD,True,True);o.contact(CONTACT_BODY_TARGET,target,TARGET_SRC,True,True);o.contact(CONTACT_AFFORDANCES,(A,B),AFF,True,True)
def settle_selected(o,nxt,effect):
    motor=o.tick()
    if not isinstance(motor,MotorActionV2) or int(motor.action_id)!=A:raise AssertionError(('source-category:motor',motor))
    result=o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,int(effect),len(nxt),*nxt),WORLD,True,True)
    return motor,result

def main():
    started=time.perf_counter();checks={}

    # Action-only recommendation: pleasant outcome is not epistemic source accuracy.
    advice=organism();setup(advice);advice.contact(CONTACT_SOURCE_ASSERTION,(A,),SRC,True,True)
    ctx=advice._source_context_signature();_m,res=settle_selected(advice,PRED,1)
    checks['positive_action_advice_creates_no_prediction_calibration']=(
        advice.source_prediction_calibration.evidence_for(SRC,ctx)==0
        and advice._source_calibration(SRC,ctx)==0
        and int(res.get('source_updates',0))==0)
    # Repetition by the same source remains repetition, not independent accuracy evidence.
    advice.contact(CONTACT_WORLD_STATE,S0,WORLD,True,True);advice.contact(CONTACT_BODY_TARGET,TARGET_A,TARGET_SRC,True,True);advice.contact(CONTACT_AFFORDANCES,(A,B),AFF,True,True)
    advice.contact(CONTACT_SOURCE_ASSERTION,(A,),SRC,True,True);_m,res2=settle_selected(advice,PRED,-1)
    checks['negative_action_advice_also_creates_no_prediction_calibration']=advice.source_prediction_calibration.evidence_for(SRC,ctx)==0 and int(res2.get('source_updates',0))==0

    # Explicit prediction: match is epistemic support even when consequence valence is negative.
    match=organism();setup(match);ctxm=match._source_context_signature()
    match._record_source_assertion(A,SRC,predicted_state=PRED)
    _m,mres=settle_selected(match,PRED,-1)
    checks['explicit_prediction_match_calibrates_despite_negative_valence']=(
        match._source_calibration(SRC,ctxm)>0 and match.developmental_source_credibility_q16(SRC)>0
        and int(mres.get('source_updates',0))==1)

    # Explicit prediction: mismatch is counterevidence even when consequence valence is positive.
    mismatch=organism();setup(mismatch);ctxw=mismatch._source_context_signature()
    mismatch._record_source_assertion(A,SRC,predicted_state=PRED)
    _m,wres=settle_selected(mismatch,WRONG,1)
    checks['explicit_prediction_mismatch_calibrates_despite_positive_valence']=(
        mismatch._source_calibration(SRC,ctxw)<0 and mismatch.developmental_source_credibility_q16(SRC)<0
        and int(wres.get('source_updates',0))==1)

    # Pertinence: source accuracy in one target context is neutral in a new target context.
    match.contact(CONTACT_WORLD_STATE,S0,WORLD,True,True);match.contact(CONTACT_BODY_TARGET,TARGET_B,TARGET_SRC,True,True);match.contact(CONTACT_AFFORDANCES,(A,B),AFF,True,True)
    ctx_b=match._source_context_signature()
    checks['source_predictive_quality_is_target_context_local']=(ctx_b!=ctxm and match._source_calibration(SRC,ctx_b)==0 and match.developmental_source_credibility_q16(SRC)==0)

    # Legacy reward-contaminated rows cannot control the typed adapter after restore/migration.
    legacy=organism();setup(legacy);legacy.source_calibrations.append(SourceCalibrationV2(SRC,legacy._source_context_signature(),support=99,counter=0,revision=99,active=True))
    checks['legacy_source_calibration_rows_do_not_become_live_epistemic_quality']=legacy.developmental_source_credibility_q16(SRC)==0 and legacy._source_calibration(SRC,legacy._source_context_signature())==0

    cp=match.checkpoint();restored=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    # Return to context A to compare the persisted typed relation.
    restored.contact(CONTACT_WORLD_STATE,S0,WORLD,True,True);restored.contact(CONTACT_BODY_TARGET,TARGET_A,TARGET_SRC,True,True);restored.contact(CONTACT_AFFORDANCES,(A,B),AFF,True,True)
    checks['typed_source_prediction_history_survives_checkpoint']=restored._source_calibration(SRC,ctxm)==match.source_prediction_calibration.calibration(SRC,ctxm)>0

    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.source-prediction-category-separation.v1','pass':not failed,'checks':checks,'failed':failed,
        'match_context':ctxm,'other_context':ctx_b,'match_quality_q16':match.source_prediction_calibration.quality_q16(SRC,ctxm),
        'claim':'SOURCE_PREDICTION_ACCURACY_RECOMMENDATION_VALENCE_REPETITION_AND_CONTEXT_PERTINENCE_ARE_DISTINCT',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_SOURCE_PREDICTION_CATEGORY_SEPARATION '+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
