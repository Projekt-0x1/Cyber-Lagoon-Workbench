#!/usr/bin/env python3
from __future__ import annotations
import json,time
from reference_packed_selection_revision_v1 import PackedSelectionRevisionV1
from reference_organism_v2 import PREF_TEMPLATE,PREF_LEXEME

CTX=9001

def config(i,width=5):
    i=int(i);rows=[(PREF_TEMPLATE,0,100_000_000+i)]
    rows.extend((PREF_LEXEME,slot,200_000_000+slot*10_000_000+i) for slot in range(1,width))
    return tuple(rows)

def refused(fn):
    try:fn();return False
    except ValueError:return True


def main():
    t0=time.perf_counter();checks={};s=PackedSelectionRevisionV1();target=config(7)
    s.record(CTX,target,501,1);s.record(CTX,target,502,1);s.record(CTX,target,501,-1)
    checks['signed_source_rows_match_language_law']=s.evidence(CTX,target)==(1,3) and s.last_revision_rows==2
    checks['source_withdrawal_is_view_not_history_rewrite']=s.evidence(CTX,target,(502,))==(0,2) and s.evidence(CTX,target)==(1,3)
    checks['body_roundtrip_matches_live_configuration']=s.configuration_at(s._find_config(CTX,target,False))==(CTX,target)
    checks['iter_revisions_exposes_source_rows']=list(s.iter_revisions())==[(CTX,target,501,1,1),(CTX,target,502,1,0)]
    s.drop(CTX,target,502);checks['drop_removes_one_source_row']=s.evidence(CTX,target)==(0,2) and s.row_count==1
    s.record(CTX,target,502,1)
    cp=s.checkpoint();r=PackedSelectionRevisionV1.restore(cp)
    checks['restored_body_matches']=r.configuration_at(r._find_config(CTX,target,False))==(CTX,target)
    checks['checkpoint_exact']=r.checkpoint()==cp and r.evidence(CTX,target)==(1,3)
    bad=bytearray(cp);bad[len(bad)//2]^=1
    checks['checkpoint_tamper_refuses']=refused(lambda:PackedSelectionRevisionV1.restore(bad))
    a=config(1);b=config(2);checks['content_addressed_configuration_identity']=PackedSelectionRevisionV1.configuration_root(CTX,a)!=PackedSelectionRevisionV1.configuration_root(CTX,b)

    scale=PackedSelectionRevisionV1();scale.record(CTX,target,501,1)
    rows=[]
    for total in (1_000,10_000,100_000,250_000):
        start=scale.config_count;t=time.perf_counter()
        for i in range(start,total):scale.record(CTX,config(i+10),800_000+i,1)
        insert_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        for _ in range(10_000):value,evidence=scale.evidence(CTX,target)
        lookup_us=(time.perf_counter()-t)*1e6/10_000
        rows.append({'configurations':scale.config_count,'revision_rows':scale.row_count,'persistent_bytes':scale.persistent_bytes,'insert_ms':round(insert_ms,3),'lookup_us':round(lookup_us,3),'config_hash_candidates':scale.last_config_candidates,'max_bucket_occupancy':scale.max_bucket_occupancy,'runtime_index_payload_bytes':scale.runtime_index_payload_bytes,'revision_rows_touched':scale.last_revision_rows,'value':value,'evidence':evidence})
    checks['quarter_million_lookup_touches_bounded']=all(x['config_hash_candidates']<16 and x['revision_rows_touched']==1 and x['value']==1 for x in rows)
    checks['quarter_million_persistent_under_48mb']=rows[-1]['persistent_bytes']<48*1024*1024
    checks['quarter_million_runtime_index_payload_under_4mb']=rows[-1]['runtime_index_payload_bytes']<4*1024*1024
    checks['lookup_flat_enough']=max(x['lookup_us'] for x in rows)<2*max(1.0,min(x['lookup_us'] for x in rows)) and rows[-1]['lookup_us']<15

    # Compare exact serialized selection-history bytes at a bounded common scale.
    common=50_000
    legacy=[]
    for i in range(common):
        legacy.append({'context':CTX,'configuration':[list(x) for x in config(i+20)],'source':900_000+i,'support':1,'counter':0})
    legacy_bytes=len(json.dumps(legacy,separators=(',',':'),sort_keys=True).encode())
    packed_common=PackedSelectionRevisionV1()
    for i in range(common):packed_common.record(CTX,config(i+20),900_000+i,1)
    packed_bytes=packed_common.persistent_bytes
    checks['packed_checkpoint_smaller_than_json_authority']=packed_bytes<legacy_bytes
    restored=PackedSelectionRevisionV1.restore(scale.checkpoint())
    checks['large_checkpoint_roundtrip']=restored.config_count==scale.config_count and restored.row_count==scale.row_count and restored.evidence(CTX,target)==(1,1)
    elapsed=(time.perf_counter()-t0)*1000;checks['rapid_bounded_assay']=elapsed<10_000
    semantic_checks={k:v for k,v in checks.items() if k not in {'rapid_bounded_assay','lookup_flat_enough'}}
    result={'schema':'0x1.reference-packed-selection-revision.v1','pass':all(semantic_checks.values()),'checks':checks,'scale':rows,'common_scale':common,'legacy_json_bytes':legacy_bytes,'packed_bytes':packed_bytes,'compression_ratio':round(legacy_bytes/packed_bytes,3),'elapsed_ms':round(elapsed,3),'semantic_change':False,'runtime_llm':False,'physical_direct_parity':'NOT_RUN/RED','claim':'PACKED_PAGEABLE_SELECTION_NETWORK_REVISION_STORAGE_REFERENCE_ONLY'}
    print('FOUNDRY_PACKED_SELECTION_REVISION '+('GREEN' if result['pass'] else 'RED')+f" configs={rows[-1]['configurations']} bytes={rows[-1]['persistent_bytes']} touch={rows[-1]['revision_rows_touched']} lookup_us={rows[-1]['lookup_us']} ratio={result['compression_ratio']}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
