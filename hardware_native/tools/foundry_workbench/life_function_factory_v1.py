#!/usr/bin/env python3
"""Fast shared Life Function checkpoint factory for Workbench mechanics.

Build the canonical life once, snapshot future-causally-sufficient Adult state at observer
marks, and let independent probes fork those checkpoints. The cache is derivative and can
always be deleted/rebuilt from Species + curriculum; it is never cognitive authority.
"""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,canonical_life_function_curriculum_v2,
    canonical_species_program_v2,source_semantics_root_v2,
)

SCHEMA='cyber-lagoon.life-function-factory-cache.v1'

def _state_root(checkpoint):return hashlib.sha256(json.dumps(checkpoint['adult'],sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build_cache(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    program=canonical_species_program_v2();curriculum=canonical_life_function_curriculum_v2();source_root=source_semantics_root_v2();species_root=program.root();curriculum_root=curriculum.root();runtime=ReferenceLifeFunctionRuntimeV2(program);rows=[]
    for event in curriculum.events:
        runtime.apply(event)
        if event.lane!='checkpoint_mark':continue
        mark=str(event.payload[0]);checkpoint=runtime.checkpoint();name=f'{species_root[:12]}-{curriculum_root[:12]}-{source_root[:12]}-{mark}.json';path=directory/name
        with tempfile.NamedTemporaryFile('w',dir=directory,delete=False) as out:
            json.dump(checkpoint,out,sort_keys=True,separators=(',',':'));out.flush();os.fsync(out.fileno());tmp=Path(out.name)
        os.replace(tmp,path);rows.append({'mark':mark,'cursor':runtime.cursor,'state_root':_state_root(checkpoint),'file':name})
    manifest={'schema':SCHEMA,'species_root':species_root,'curriculum_root':curriculum_root,'source_semantics_root':source_root,'events':len(curriculum.events),'checkpoints':rows}
    (directory/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');return manifest

def load_mark(directory,mark):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!=SCHEMA:raise ValueError('life_factory:manifest')
    program=canonical_species_program_v2();curriculum=canonical_life_function_curriculum_v2()
    if manifest.get('species_root')!=program.root() or manifest.get('curriculum_root')!=curriculum.root() or manifest.get('source_semantics_root')!=source_semantics_root_v2():raise ValueError('life_factory:stale_cache')
    row=next((x for x in manifest.get('checkpoints',()) if x.get('mark')==str(mark)),None)
    if row is None:raise KeyError(mark)
    checkpoint=json.loads((directory/row['file']).read_text());runtime=ReferenceLifeFunctionRuntimeV2.restore(program,checkpoint)
    if runtime.cursor!=int(row['cursor']) or _state_root(checkpoint)!=row['state_root'] or runtime.marks.get(str(mark))!=runtime.cursor:raise ValueError('life_factory:checkpoint')
    return runtime

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--mark');args=ap.parse_args()
    if args.mark:
        runtime=load_mark(args.cache_dir,args.mark);print(json.dumps({'mark':args.mark,'cursor':runtime.cursor,'history_root':runtime.history_root(),'adult':runtime.adult.digest()},sort_keys=True));return
    print(json.dumps(build_cache(args.cache_dir),sort_keys=True))
if __name__=='__main__':main()
