#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_SCENE,CONTACT_SURFACE_STREAM
from reference_population_v1 import PopulationSpecV1

def u(s):return tuple(s.encode())
def expose(o,f,raw,src):
    o.contact(CONTACT_SCENE,(7,7001,1,f),src,True,True)
    return o.contact(CONTACT_SURFACE_STREAM,u(raw),src,True,True)
def main():
    t=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(32768,2,3,42,8));f=9101
    a=expose(o,f,'akmipzo',7001);b=expose(o,f,'qumipte',7002);c=expose(o,f,'romipva',7003)
    checks['three_context_threshold']=a==0 and b==0 and c!=0
    checks['undelimited_chunk']=o.language.lexeme(f)==u('mip')
    rep=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,42,8));last=0
    for i in range(3):last=expose(rep,9201,'afoob',7100+i)
    checks['identical_frame_not_boundary']=last==0 and rep.language.lexeme(9201) is None
    amb=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,42,8));expose(amb,9301,'afooXbarb',7201);expose(amb,9301,'cfooYbard',7202)
    try:expose(amb,9301,'efooZbarf',7203)
    except ValueError as e:checks['equal_candidates_refuse']=str(e)=='language:stream_segmentation_ambiguous'
    else:checks['equal_candidates_refuse']=False
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));checks['checkpoint']=r.digest()==o.digest() and r.language.lexeme(f)==u('mip')
    checks['no_tokenizer']=not hasattr(o,'tokenize') and not hasattr(o.language,'tokenize')
    checks['bounded_work']=o.language.last_segment_touches<512
    checks['no_external_choice_seed']=all('seed' not in k.lower() and 'salt' not in k.lower() for k in o.checkpoint())
    out={'schema':'0x1.reference-organism-surface-segmentation.v2','pass':all(checks.values()),'checks':checks,'learned':'mip','candidate_touches':o.language.last_segment_touches,'claim':'UNDELIMITED_SURFACE_SEGMENTATION_REFERENCE_ONLY_NOT_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SURFACE_SEGMENTATION '+('GREEN' if out['pass'] else 'RED')+' undelimited=1 tokenizer=0');print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
if __name__=='__main__':main()
