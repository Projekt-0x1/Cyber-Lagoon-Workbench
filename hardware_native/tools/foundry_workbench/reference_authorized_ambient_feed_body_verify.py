#!/usr/bin/env python3
"""Focused transport gate for the authorized ambient-feed body membrane."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from reference_authorized_ambient_feed_body_v1 import AuthorizedAmbientFeedBodyV1
from reference_life_extension_endogenous_state_inquiry_v1 import TESTIMONY_BODY_SOURCE
from reference_life_function_curriculum_v1 import (
    LifeCurriculumEventV2,ReferenceLifeFunctionRuntimeV2,canonical_species_program_v2,
)
from reference_predictive_credit_profile_v1 import Q


def root(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime():return ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())


def main():
    started=time.perf_counter();checks={};body=AuthorizedAmbientFeedBodyV1
    pool_a=((0x31001,b'first candidate'),(0x31002,b'second candidate'),(0x31003,b'third candidate'))
    pool_b=((0x41001,b'XXXXXXXXXXXXXXXX'),(0x41002,b'YYYYYYYYYYYYYYYYY'),(0x41003,b'ZZZZZZZZZZZZZZZ'))

    a=runtime();b=runtime();entropy=0xA17E5EED
    ia=body.choose_index(entropy,a.cursor+1,len(pool_a));ib=body.choose_index(entropy,b.cursor+1,len(pool_b))
    ra=body.admit_one(a,100,pool_a,entropy);rb=body.admit_one(b,100,pool_b,entropy)
    checks['selection_depends_on_entropy_life_sequence_and_pool_size_not_payload_or_source_values']=(ia==ib==ra['index']==rb['index'])
    checks['selection_receipt_binds_entropy_cardinality_and_exact_ordered_candidate_pool']=(
        ra['entropy']==rb['entropy']==entropy and ra['pool_count']==rb['pool_count']==3
        and len(ra['pool_sha256'])==len(rb['pool_sha256'])==64 and ra['pool_sha256']!=rb['pool_sha256'])

    selected_source,selected_raw=pool_a[ia];pending=a.ambient_stream.checkpoint();row=pending['pending'][0]
    checks['selected_exact_bytes_enter_one_ordinary_life_event_and_pending_sensory_state']=(
        a.cursor==1 and a.ambient_stream.pending_count==1 and int(row['source'])==selected_source
        and bytes(row['raw'])==selected_raw and ra['sha256']==hashlib.sha256(selected_raw).hexdigest())
    cp=copy.deepcopy(a.checkpoint());restored=ReferenceLifeFunctionRuntimeV2.restore(a.program,copy.deepcopy(cp))
    checks['selected_pending_contact_restart_is_exact']=restored.checkpoint()==cp and restored.ambient_stream.pending_count==1

    before_world=root(a.adult.language_adult.world_causal_learning.checkpoint());processed=body.drain(a,100);after_world=root(a.adult.language_adult.world_causal_learning.checkpoint())
    checks['drain_retires_transport_without_minting_world_truth']=(a.cursor==2 and a.ambient_stream.pending_count==0 and before_world==after_world and len(processed)==1)

    loaded=runtime();L=loaded.adult.language_adult
    for index in range(8):loaded.apply(LifeCurriculumEventV2(loaded.cursor+1,'body_load',TESTIMONY_BODY_SOURCE,(0xE0+index,Q)))
    pressure=int(L.slow_resource_history.pressure_q16());receipt=body.admit_one(loaded,500,pool_a,entropy);blocked=body.drain(loaded,500)
    blocked_cp=copy.deepcopy(loaded.ambient_stream.checkpoint());world_loaded=root(L.world_causal_learning.checkpoint())
    loaded.apply(LifeCurriculumEventV2(loaded.cursor+1,'quiet',0,(64,)));recovered=body.drain(loaded,500)
    checks['max_load_preserves_same_sampled_bytes_until_quiet_recovery_without_refetch']=(
        pressure==Q and not blocked and len(blocked_cp['pending'])==1
        and bytes(blocked_cp['pending'][0]['raw'])==pool_a[receipt['index']][1]
        and loaded.ambient_stream.pending_count==0 and len(recovered)==1)
    checks['load_defer_and_recovery_do_not_create_world_truth']=world_loaded==root(L.world_causal_learning.checkpoint())

    source=Path(__import__('reference_authorized_ambient_feed_body_v1').__file__).read_text().lower()
    checks['body_contains_no_provider_specific_fetcher_or_semantic_selector']=all(token not in source for token in (
        'requests.','urllib','http://','https://','reddit','twitter','twitch','x.com','expected_answer','topic_score','toxicity_score','language_router'))
    checks['body_owns_no_persistent_feed_memory_or_daemon']=not any(token in source for token in ('threading','asyncio','while true','feed_memory','transcript'))
    over_capacity=tuple((0x60000+i,b'x') for i in range(513));capacity_refused=False
    try:body.admit_one(runtime(),700,over_capacity,entropy)
    except RuntimeError as exc:capacity_refused=str(exc)=='ambient-feed-body:pool-capacity'
    checks['authorized_candidate_pool_quantity_fails_closed_before_large_transient_growth']=capacity_refused

    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.authorized-ambient-feed-body.v1','contract':'FOUNDRY_AUTHORIZED_AMBIENT_FEED_BODY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runtime_llm':False,'checks':checks,'failed':failed,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
