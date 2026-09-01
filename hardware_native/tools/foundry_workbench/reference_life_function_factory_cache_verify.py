#!/usr/bin/env python3
from __future__ import annotations
import json,tempfile,time
from pathlib import Path
from life_function_factory_v1 import build_cache,load_mark
from reference_life_function_curriculum_v1 import canonical_developmental_probe_v2,canonical_life_function_curriculum_v2,canonical_species_program_v2,source_semantics_root_v2

def main():
    t=time.perf_counter();checks={};program=canonical_species_program_v2();curriculum=canonical_life_function_curriculum_v2();marks=tuple(e.payload[0] for e in curriculum.events if e.lane=='checkpoint_mark');context,_examples,heldout,_features=canonical_developmental_probe_v2(curriculum)
    naive_event_work=sum(curriculum.mark_cursor(mark) for mark in marks);shared_event_work=len(curriculum.events)
    with tempfile.TemporaryDirectory() as d:
        manifest=build_cache(d);loaded={mark:load_mark(d,mark) for mark in marks};files=tuple(Path(d).iterdir())
        checks['one_pass_materializes_every_canonical_prefix']=tuple(row['mark'] for row in manifest['checkpoints'])==marks
        checks['snapshot_keys_bind_species_curriculum_source_semantics_and_mark']=all(program.root()[:12] in row['file'] and curriculum.root()[:12] in row['file'] and source_semantics_root_v2()[:12] in row['file'] and row['mark'] in row['file'] for row in manifest['checkpoints'])
        checks['loaded_prefixes_are_real_continuing_adult_state']=all(loaded[mark].cursor==curriculum.mark_cursor(mark) and loaded[mark].marks.get(mark)==loaded[mark].cursor for mark in marks)
        productive=loaded['productive'].adult.language_adult
        held_surface=bytes(productive.leaf(context,heldout).surface);positions=[]
        for feature in heldout:
            units=bytes(productive.language.lexeme(feature) or b'')
            hits=tuple(i for i in range(len(held_surface)-len(units)+1)
                       if units and held_surface[i:i+len(units)]==units)
            if len(hits)==1:positions.append((hits[0],hits[0]+len(units)))
        checks['productive_checkpoint_is_immediately_probeable_without_retraining']=(
            len(positions)==len(heldout) and positions==sorted(positions)
            and all(left[1]<=right[0] for left,right in zip(positions,positions[1:])))
        checks['snapshot_manifest_contains_no_alternate_cognitive_authority']=set(manifest)=={'schema','species_root','curriculum_root','source_semantics_root','events','checkpoints'} and manifest['species_root']==program.root() and manifest['curriculum_root']==curriculum.root() and manifest['source_semantics_root']==source_semantics_root_v2()
        checks['snapshot_store_is_derivative_files_plus_manifest_only']=len(files)==len(marks)+1
        manifest_path=Path(d)/'manifest.json';canonical_manifest=manifest_path.read_text()
        stale=json.loads(canonical_manifest);stale['species_root']='0'*64;manifest_path.write_text(json.dumps(stale));refused=False
        try:load_mark(d,'productive')
        except ValueError:refused=True
        checks['stale_species_snapshot_refuses']=refused
        manifest_path.write_text(canonical_manifest);stale=json.loads(canonical_manifest);stale['source_semantics_root']='0'*64;manifest_path.write_text(json.dumps(stale));source_refused=False
        try:load_mark(d,'productive')
        except ValueError:source_refused=True
        checks['stale_source_semantics_snapshot_refuses']=source_refused
        manifest_path.write_text(canonical_manifest)
    checks['shared_chain_reduces_prefix_training_event_work_by_more_than_3x']=naive_event_work>3*shared_event_work
    checks['bounded_fast_path']=time.perf_counter()-t<5.0
    failed=[k for k,v in checks.items() if not v]
    out={'contract':'FOUNDRY_LIFE_FUNCTION_FACTORY_CACHE_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'marks':marks,'naive_event_work':naive_event_work,'shared_event_work':shared_event_work,'event_work_reduction':round(naive_event_work/shared_event_work,3),'checks':checks,'failed':failed,'remaining_red':['MIGRATE_ASSAY_LOCAL_PRIVATE_TRAINING_TO_CANONICAL_MARK_PROBES','MULTI_MECHANIC_CI_SHARDING_OVER_ONE_SNAPSHOT_ROOT'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(out['contract']);print(json.dumps(out,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
