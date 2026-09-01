#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
A,B,C=4101,4102,4103;X,Y=101,202;WORLD=9001
def setup(o,state,goal,actions=(X,Y),source=WORLD):
    o.contact(CONTACT_WORLD_STATE,state,source,True,True);o.contact(CONTACT_BODY_TARGET,goal,8001,True,True);o.contact(CONTACT_AFFORDANCES,actions,8002,True,True)
def claim(o,a,s):return o.contact(CONTACT_SOURCE_ASSERTION,(a,),s,True,True)
def settle(o,a,e,nxt,ind=True,source=WORLD):return o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,e,len(nxt),*nxt),source,True,ind)
def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8)
    r=ReferenceOrganismV2(spec);setup(r,(11,),(99,));aid=claim(r,X,A)
    for _ in range(19):assert claim(r,X,A)==aid
    claim(r,Y,B);checks['repetition_not_votes']=r.tick() is None and r.information_need==(3,X,Y) and next(x for x in r.source_assertions if x.identity==aid).repetitions==20
    r.information_need=();claim(r,X,C);a=r.tick();checks['independent_sources_nominate']=isinstance(a,MotorActionV2) and a.action_id==X and len(a.source_assertion_ids)==2
    checks['nonindependent_no_calibration']=settle(r,a,1,(99,),False)['source_updates']==0 and not r.source_calibrations
    o=ReferenceOrganismV2(spec);setup(o,(21,),(501,));base=o._exploration_candidate();chosen=Y if base==X else X;cid=claim(o,chosen,A);a=o.tick();checks['nomination_not_truth']=a.source_assertion_ids==(cid,) and a.source_counterfactual_action==base
    learned=settle(o,a,1,(501,),True);ctx=o._source_context_signature();checks['decisive_independent_calibrates']=learned['source_updates']==1 and o._source_calibration(A,ctx)==1
    setup(o,(22,),(501,));claim(o,chosen,A);claim(o,base,B);b=o.tick();checks['context_transfer']=isinstance(b,MotorActionV2) and b.action_id==chosen;settle(o,b,0,(23,),True)
    setup(o,(31,),(777,));claim(o,chosen,A);claim(o,base,B);checks['context_isolation']=o.tick() is None and o.information_need and o.information_need[0]==3
    d=ReferenceOrganismV2(spec);S=(61,);G=(62,)
    for src in (9101,9102):setup(d,S,G,(Y,),src);m=d.tick();settle(d,m,1,G,True,src)
    setup(d,S,G,(X,Y),9103)
    for src in range(5000,5020):claim(d,X,src)
    m=d.tick();checks['direct_world_precedes_testimony']=isinstance(m,MotorActionV2) and m.action_id==Y and not m.source_assertion_ids
    s=ReferenceOrganismV2(spec)
    for i in range(128):setup(s,(10000+i,),(888,),(X,));claim(s,X,6000+i)
    setup(s,(20000,),(888,));claim(s,X,A);claim(s,Y,B);s._source_nomination(False);checks['sparse_lookup']=s.last_source_touches==2 and len(s.source_assertions)==130
    cp=o.checkpoint();q=ReferenceOrganismV2.restore(copy.deepcopy(cp));checks['checkpoint']=q.digest()==o.digest()
    checks['no_global_trust_scalar']=not hasattr(o,'trust')
    out={'schema':'0x1.reference-organism-source-epistemics.v2','pass':all(checks.values()),'checks':checks,'source_touches':s.last_source_touches,'claim':'CONTEXT_CALIBRATED_TESTIMONY_REFERENCE_ONLY_NOT_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOURCE_EPISTEMICS '+('GREEN' if out['pass'] else 'RED')+' repetition_vote=0 context_calibration=1 sparse=1');print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
if __name__=='__main__':main()
