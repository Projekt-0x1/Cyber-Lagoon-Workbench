#!/usr/bin/env python3
"""Economic policy: every promoted language lane uses the same canonical Life Function."""
from __future__ import annotations
import importlib,json,time
from pathlib import Path
from reference_life_function_curriculum_v1 import (
    CANONICAL_TAIL_MODULES_V2,_local_source_semantic_files_v2,
    canonical_life_function_curriculum_v2,canonical_tail_builders_v2,
)

def verify():
    root=Path(__file__).parent;primary='run_life_function_curriculum_fast.sh';checks={};rows=[]
    paths=set(root.glob('run_*fast.sh'));paths.add(root/'run_language_workbench.sh')
    primary_calls=[
        'python3 hardware_native/tools/foundry_workbench/reference_life_function_factory_verify.py',
    ]
    for path in sorted(paths):
        text=path.read_text();python_calls=[line.strip() for line in text.splitlines() if line.strip().startswith('python3 ')]
        if path.name==primary:ok=python_calls==primary_calls
        else:ok=(not python_calls and 'run_life_function_curriculum_fast.sh' in text)
        rows.append({'runner':path.name,'python_calls':python_calls,'delegates_to_primary':('run_life_function_curriculum_fast.sh' in text),'pass':ok})
    checks['one_and_only_one_developmental_python_authority']=all(row['pass'] for row in rows)
    checks['every_nonprimary_promoted_lane_consumes_same_growing_curriculum']=all(row['runner']==primary or row['delegates_to_primary'] for row in rows)

    # Every canonical tail is external contact for this one life, never a private Adult.
    # This is deliberately an R0 source/chronology guard: it does not emulate cognition.
    semantic_files=set(_local_source_semantic_files_v2())
    curriculum=canonical_life_function_curriculum_v2()
    canonical_marks=tuple(event.payload[0] for event in curriculum.events if event.lane=='checkpoint_mark')
    canonical_modules={module for module,_builder in canonical_tail_builders_v2()}
    extension_rows=[]
    for path in sorted(root.glob('reference_life_extension_*.py')):
        module=path.stem;source=path.read_text();lower=source.lower();registered=module in canonical_modules
        marks=()
        if registered:
            loaded=importlib.import_module(module)
            builder=getattr(loaded,'build',None) or getattr(loaded,'build_extension',None)
            emitted=tuple(builder(0)) if callable(builder) else ()
            marks=tuple(event.payload[0] for event in emitted if getattr(event,'lane',None)=='checkpoint_mark')
        private_tokens=tuple(token for token in (
            'birth_language_mastery_adult(', 'referencelifefunctionruntimev2(',
            'languagemasteryadultv1(',
        ) if token in lower)
        extension_rows.append({
            'module':module,'marks':marks,'canonical_reference':registered,
            'source_hashed':path.name in semantic_files,'private_adult_tokens':private_tokens,
        })
    checks['canonical_tail_registry_is_explicit']=canonical_modules==set(CANONICAL_TAIL_MODULES_V2)
    checks['registered_life_extensions_are_consumed_by_canonical_chronology']=all(
        (not row['canonical_reference']) or (row['marks'] and all(mark in canonical_marks for mark in row['marks']))
        for row in extension_rows)
    checks['only_registered_life_extensions_change_source_semantics_root']=all(
        row['source_hashed']==row['canonical_reference'] for row in extension_rows)
    checks['life_extensions_cannot_birth_or_train_private_adults']=all(not row['private_adult_tokens'] for row in extension_rows)
    rows.extend({'runner':'extension:'+row['module'],'python_calls':[],'delegates_to_primary':row['canonical_reference'],'pass':(row['source_hashed']==row['canonical_reference']) and not row['private_adult_tokens']} for row in extension_rows)
    return checks,rows

def main():
    t=time.perf_counter();checks,rows=verify();failed=[k for k,v in checks.items() if not v];out={'contract':'FOUNDRY_SINGLE_GROWING_CURRICULUM_POLICY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runners':rows,'checks':checks,'failed':failed,'elapsed_ms':round((time.perf_counter()-t)*1000,3)};print(out['contract']);print(json.dumps(out,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
