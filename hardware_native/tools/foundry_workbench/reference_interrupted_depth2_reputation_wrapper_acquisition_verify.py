#!/usr/bin/env python3
from __future__ import annotations
import copy,json,tempfile,time
from pathlib import Path
from life_function_factory_v1 import build_cache,load_mark
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_natural_reputation_relation_verify import (
    ALLY,R1,R2,TARGET,W1,W2,WARNER,WARNING_CTX,ENDORSE_CTX,
    calibrate_meta,calibrate_source,calibration_relation,surface,teach_names)


def clone(seed):return type(seed).restore(seed.program,copy.deepcopy(seed.checkpoint()))
def fragmented_example(a,ctx,target,left,right,source,stream):
    first=a.observe_fragmented_natural_reputation_example(ctx,target,left,source,stream,False)
    final=a.observe_fragmented_natural_reputation_example(ctx,target,right,source,stream,True)
    return first,final
def wrapper(left,right):return b'field note < '+left+b' > beside < '+right+b' >'
def fragmented_structure(a,raw,source,stream):
    cut=max(1,len(raw)//2)
    return (a.observe_fragmented_natural_structure_contact(raw[:cut],source,stream,False),
            a.observe_fragmented_natural_structure_contact(raw[cut:],source,stream,True))
def fragmented_claim(a,raw,speaker,stream,interject=False):
    cut=max(1,len(raw)//2)
    first=a.observe_fragmented_natural_reputation_contact(raw[:cut],speaker,stream,False)
    if interject:
        # Another authenticated source occupies its own opaque occurrence. Its raw
        # content is irrelevant to the held source's unfinished structural stream.
        a.natural_reputation_relation.close_fragmented_surface(
            b'other actor',0xE940,0xE941,a._advance(),True)
    final=a.observe_fragmented_natural_reputation_contact(raw[cut:],speaker,stream,True)
    return first,final


def verify_loaded(seed):
    started=time.perf_counter();checks={};runtime=clone(seed);a=runtime.adult.language_adult;w=a.world_causal_learning
    correct,effect,wrong=calibration_relation(a);teach_names(a)
    for source,reliable in ((W1,False),(W2,False),(R1,True),(R2,True)):
        calibrate_source(w,correct,effect,wrong,source,reliable)
    for ctx,target,left,right,source,stream in (
        (WARNING_CTX,W1,b'do not ',b'trust alex.',0xA711,0xB711),
        (WARNING_CTX,W2,b'do not ',b'trust blair.',0xA712,0xB712),
        (ENDORSE_CTX,R1,b'rely ',b'on dana.',0xA713,0xB713),
        (ENDORSE_CTX,R2,b'rely ',b'on erin.',0xA714,0xB714)):
        fragmented_example(a,ctx,target,left,right,source,stream)
    checks['leaf_reputation_relations_are_independently_earned_prerequisites']=(
        a.natural_reputation_relation.relation_polarity(WARNING_CTX) is False and
        a.natural_reputation_relation.relation_polarity(ENDORSE_CTX) is True)

    # Same learned wrapper literals; the warning child appears first in one raw
    # developmental contact and second in the other. No wrapper context is supplied.
    train_left=wrapper(surface(a,WARNING_CTX,W1),b'dana')
    train_right=wrapper(b'erin',surface(a,WARNING_CTX,W2))
    first_train=fragmented_structure(a,train_left,0xA701,0xB701)
    one_cp=copy.deepcopy(a.checkpoint())
    held=wrapper(b'alex',surface(a,WARNING_CTX,TARGET))
    one=LanguageMasteryAdultV1.restore(copy.deepcopy(one_cp));one_spans=tuple(one.language.invert_span(tuple(held)))
    one_exact=tuple(span for span in one_spans if tuple(span.children)==(tuple(b'alex'),tuple(surface(one,WARNING_CTX,TARGET))))
    second_train=fragmented_structure(a,train_right,0xA702,0xB702)
    trained_cp=copy.deepcopy(a.checkpoint());held_spans=tuple(a.language.invert_span(tuple(held)))
    held_exact=tuple(span for span in held_spans if tuple(span.children)==(tuple(b'alex'),tuple(surface(a,WARNING_CTX,TARGET))))
    checks['two_interrupted_raw_examples_acquire_wrapper_without_direct_context_teaching']=(
        first_train[0]==0 and second_train[0]==0 and not one_exact and len(held_exact)==1)
    checks['heldout_argument_order_rebinds_warning_on_second_structural_port']=len(held_exact)==1

    flat=tuple(a.language.invert_surface(tuple(held)));recursive=tuple(a.natural_reputation_relation._all_bindings(a,held))
    checks['heldout_wrapper_requires_recursive_span_not_flat_reputation_match']=(
        not any(int(row.context)==WARNING_CTX and tuple(map(int,row.atoms))==(TARGET,) for row in flat)
        and any(c==WARNING_CTX and target==TARGET and depth>0 for c,target,depth in recursive))

    calibrate_meta(w,correct,effect,wrong,WARNER,0xC710)
    first,finish=fragmented_claim(a,held,WARNER,0xD710,True)
    checks['interrupted_depth2_warning_survives_other_actor_and_updates_original_source']=(
        not first[1] and finish[1] and tuple(finish[0])==(WARNING_CTX,TARGET,False)
        and w.testimony_reliability_state(TARGET)==-1)

    stable_binding=a.natural_reputation_relation.natural_claim_binding(a,held)

    # Independently adjudicated wrong reputation predictions can make the same
    # speaker lose authority; later correct predictions can restore it. Syntax is
    # held fixed throughout and must not absorb that history.
    for base,actual in ((0xC720,True),(0xC730,False),(0xC740,True)):
        target=base;calibrate_source(w,correct,effect,wrong,target,actual)
        assert w.observe_reputation_claim(WARNER,target,not actual)
    betrayed=bool(w.reputation_reliable(WARNER))
    for base,actual in ((0xC750,True),(0xC760,False),(0xC770,True),(0xC780,False)):
        target=base;calibrate_source(w,correct,effect,wrong,target,actual)
        assert w.observe_reputation_claim(WARNER,target,actual)
    recovered=bool(w.reputation_reliable(WARNER))
    checks['betrayal_and_recovery_change_source_authority_not_hierarchical_binding']=(
        not betrayed and recovered and a.natural_reputation_relation.natural_claim_binding(a,held)==stable_binding)

    isolated=LanguageMasteryAdultV1.restore(copy.deepcopy(trained_cp));iw=isolated.world_causal_learning
    calibrate_meta(iw,correct,effect,wrong,WARNER,0xC790);before=copy.deepcopy(iw.reputation_claims)
    cut=max(1,len(held)//2)
    isolated.observe_fragmented_natural_reputation_contact(held[:cut],WARNER,0xD720,False)
    wrong_source=isolated.observe_fragmented_natural_reputation_contact(held[cut:],ALLY,0xD720,True)
    wrong_stream=isolated.observe_fragmented_natural_reputation_contact(held[cut:],WARNER,0xD721,True)
    malformed=b'field note beside < alex > < '+surface(isolated,WARNING_CTX,TARGET)+b' >'
    checks['cross_source_cross_stream_and_unlearned_order_refuse']=(
        not wrong_source[1] and not wrong_stream[1] and iw.reputation_claims==before
        and not isolated.natural_reputation_relation.natural_claim_binding(isolated,malformed))

    source_lines=Path(__file__).read_text().splitlines();cptext=json.dumps(trained_cp,sort_keys=True).lower()
    direct_construction='observe_surface_'+'construction(';direct_span='.observe_'+'span('
    checks['assay_contains_no_direct_wrapper_construction_or_span_teaching']=not any(
        direct_construction in line or direct_span in line for line in source_lines)
    checks['heldout_complete_wrapper_was_not_checkpointed']=held.decode() not in cptext
    failed=[k for k,v in checks.items() if not v]
    return {'schema':'cyber-lagoon.interrupted-depth2-reputation-wrapper-acquisition.v1',
            'contract':'FOUNDRY_INTERRUPTED_DEPTH2_REPUTATION_WRAPPER_ACQUISITION_'+('GREEN' if not failed else 'RED'),
            'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,
            'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,
            'checks':checks,'failed':failed,'heldout_wrapper':held.decode(),
            'binding':list(stable_binding) if stable_binding else [],
            'epistemics':{'after_wrapped_warning':w.testimony_reliability_state(TARGET),'betrayed':betrayed,'recovered':recovered},
            'structure':{'one_example_span_count':len(one_spans),'heldout_span_count':len(held_spans),'recursive_bindings':len(recursive)},
            'remaining_red':['RAW_ACOUSTIC_DIARIZATION_AND_PROSODY','OPEN_ENDED_CULTURAL_MORAL_INTERNALIZATION','DIRECT_PARITY','BROAD_HUMAN_PARTY_DIALOGUE'],
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}

def main():
    with tempfile.TemporaryDirectory(prefix='foundry-interrupted-depth2-reputation-') as d:
        build_cache(d);r=verify_loaded(load_mark(d,'social_prediction_ready'))
    print(r['contract']);print('failed='+json.dumps(r['failed']));print(json.dumps(r,sort_keys=True));return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
