#!/usr/bin/env python3
"""GPU quantity gate for the current V2 continuing organism.

The resident population namespace scales procedurally; only sparse lived deltas
are materialized. This is a scale/portability assay, not a human-language claim.
"""
from __future__ import annotations
import hashlib,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from pycuda_population_exec import GpuPopulationSpecV1,PyCudaPopulationExecV1
from reference_population_v1 import PopulationSpecV1
from reference_organism_v2 import *

SCALES=(131_072,1_000_000_000,80_000_000_000)
SITE_DELTA_CAPACITY=1<<18;EDGE_DELTA_CAPACITY=1<<19
NAME=100;CLAUSE=9001;CAUSE=51001;CONTRAST=51002;ELAB=51003;SEQUENCE=51004;PARTNER=9901
BRIDGES={CAUSE:' therefore, ',CONTRAST:' however, ',ELAB:' also, ',SEQUENCE:' then, '}
M={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
S0=(80001,80011);S1=(80002,80012);S2=(80003,80013);A1=90001;A2=90002

def u(s):return tuple(s.encode())
def sentence(a):return f"the {M[a[0]]} {M[a[1]]} {M[a[2]]} the {M[a[3]]}."
def gpu_population(n):return PyCudaPopulationExecV1(GpuPopulationSpecV1(n,fanout=2,sites_per_feature=3,eligibility_horizon=8,site_delta_capacity=SITE_DELTA_CAPACITY,edge_delta_capacity=EDGE_DELTA_CAPACITY))
def organism(n):
    o=ReferenceOrganismV2(PopulationSpecV1(1024,2,3,42,8));o.population=gpu_population(n);return o

def scene(o,a,source,context=CLAUSE):return o.contact(CONTACT_SCENE,(7,context,len(a),*a),source,True,True)
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def partner(o):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),70000,True,True)
def settle_speech(o,a,effect=1):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,effect),PARTNER,True,True)
def world(o,s,source):o.contact(CONTACT_WORLD_STATE,s,source,True,True)
def target(o,s):o.contact(CONTACT_BODY_TARGET,s,71000,True,True)
def afford(o,*a):o.contact(CONTACT_AFFORDANCES,a,72000,True,True)
def settle_motor(o,a,effect,nxt,source):return o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,effect,len(nxt),*nxt),source,True,True)

def train_language(o):
    for f,text in M.items():
        for k in range(2):scene(o,(f,),10000+f*10+k,NAME);surface(o,text,20000+f*10+k)
    a=(101,201,301,401);b=(102,202,302,402);c=(101,202,302,401);d=(102,201,301,402)
    for k,row in enumerate((a,b)):
        scene(o,row,30001+k);surface(o,sentence(row),31001+k)
    assert o.language.template(CLAUSE,4) is not None
    for j,(relation,bridge) in enumerate(BRIDGES.items()):
        for k,(left,right) in enumerate(((a,b),(c,d))):
            src=40000+j*100+k;lid=scene(o,left,src+1000);surface(o,sentence(left),src+2000);rid=scene(o,right,src+1100);surface(o,sentence(right),src+2100)
            o.contact(CONTACT_SCENE_LINK,(lid,rid,relation),src,True,True)
            o.contact(CONTACT_DISCOURSE_SURFACE,u(sentence(left)+bridge+sentence(right)),src,True,True)
        assert o.language.span_template(relation,2) is not None
    return a,b,c,d

def run_language(o,rows):
    partner(o);relations=(CAUSE,CONTRAST,ELAB,SEQUENCE,CAUSE,CONTRAST,ELAB);ids=[scene(o,rows[i%4],50000+i) for i in range(8)]
    for i,r in enumerate(relations):o.contact(CONTACT_SCENE_LINK,(ids[i],ids[i+1],r),60000+i,True,True)
    out=[];ticks=[]
    for i in range(8):
        t=time.perf_counter();a=o.tick();ticks.append((time.perf_counter()-t)*1000);assert isinstance(a,ActionV2);out.append(bytes(a.payload).decode());settle_speech(o,a)
    quiet=o.tick() is None
    # Efference/reafference repair on one later utterance.
    sid=scene(o,rows[1],61000);o.inject_output_fault(0,ord('X'));t=time.perf_counter();bad=o.tick();ticks.append((time.perf_counter()-t)*1000);assert bad and bad.payload!=bad.planned_payload;settle_speech(o,bad,-1)
    t=time.perf_counter();repair=o.tick();ticks.append((time.perf_counter()-t)*1000);assert repair and repair.repair and repair.payload==bad.planned_payload;settle_speech(o,repair,1)
    return {'outputs':out,'ticks':ticks,'terminal':quiet,'repair':bytes(repair.payload).decode()}

def experience(o,start,action,nxt,source):
    world(o,start,source);target(o,nxt);afford(o,action);a=o.tick();assert isinstance(a,MotorActionV2) and a.action_id==action;settle_motor(o,a,1,nxt,source)
def run_plan(o):
    for src in (9101,9102):experience(o,S0,A1,S1,src)
    for src in (9201,9202):experience(o,S1,A2,S2,src)
    world(o,S0,9301);target(o,S2);afford(o,A1,A2);ticks=[];actions=[]
    for nxt in (S1,S2):
        t=time.perf_counter();a=o.tick();ticks.append((time.perf_counter()-t)*1000);assert isinstance(a,MotorActionV2);actions.append(a.action_id);settle_motor(o,a,1,nxt,9301)
    return {'actions':actions,'ticks':ticks,'terminal':o.tick() is None}

def scale_row(n):
    t=time.perf_counter();o=organism(n);rows=train_language(o);train_ms=(time.perf_counter()-t)*1000
    lang=run_language(o,rows);plan=run_plan(o);pop=o.population;counts=pop.sparse_counts();ticks=lang['ticks']+plan['ticks']
    phenotype={'outputs':lang['outputs'],'repair':lang['repair'],'language_terminal':lang['terminal'],'plan':plan['actions'],'plan_terminal':plan['terminal']}
    digest=hashlib.sha256(json.dumps(phenotype,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    row={'resident_sites':n,'procedural_edges':n*pop.spec.fanout,'device_bytes':pop.required_device_bytes,'train_ms':round(train_ms,3),'tick_mean_ms':round(statistics.mean(ticks),3),'tick_max_ms':round(max(ticks),3),'signature_kernel_ms_total':round(pop.signature_kernel_ms_total,3),'recruit_kernel_ms_total':round(pop.recruit_kernel_ms_total,3),'settle_kernel_ms_total':round(pop.settle_kernel_ms_total,3),'life_changed_sites':counts['site_deltas'],'life_changed_edges':counts['edge_deltas'],'site_overflow':counts['site_overflow'],'edge_overflow':counts['edge_overflow'],'behavior_digest':digest,'credit_events':pop.credit_events,'revision_events':pop.revision_events,'phenotype':phenotype}
    pop.close();return row

def main():
    started=time.perf_counter();warm_t=time.perf_counter();warm=gpu_population(1024);warm_ms=(time.perf_counter()-warm_t)*1000;compile_count=warm.module_compile_count;gpu_name=warm.cuda.Device(0).name();gpu_total=int(warm.cuda.Device(0).total_memory());warm.close()
    rows=[scale_row(n) for n in SCALES];digests={r['behavior_digest'] for r in rows}
    checks={'one_generic_module_compile':compile_count==1,'same_behavior_all_scales':len(digests)==1,'no_population_linear_device_state':len({r['device_bytes'] for r in rows})==1,'no_sparse_overflow':all(not r['site_overflow'] and not r['edge_overflow'] for r in rows),'eighty_billion_exercised':rows[-1]['resident_sites']==80_000_000_000,'eighty_billion_fast_training':rows[-1]['train_ms']<250.0,'eighty_billion_fast_ticks':rows[-1]['tick_max_ms']<20.0,'scale_does_not_inflate_tick_10x':rows[-1]['tick_mean_ms']<max(20.0,rows[0]['tick_mean_ms']*10),'life_state_actually_changed':rows[-1]['life_changed_sites']>200 and rows[-1]['life_changed_edges']>400,'causal_learning_exercised':rows[-1]['credit_events']>0 and rows[-1]['revision_events']>0}
    result={'schema':'0x1.pycuda-organism-scale.v2','pass':all(checks.values()),'gpu':gpu_name,'gpu_total_bytes':gpu_total,'module_warm_ms':round(warm_ms,3),'module_compile_count':compile_count,'rows':rows,'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'claim':'V2_80B_PROCEDURAL_POPULATION_LANGUAGE_PLANNING_SCALE_GATE_NOT_GENERAL_AGI'}
    print('PYCUDA_ORGANISM_SCALE_V2 '+('GREEN' if result['pass'] else 'RED')+f" gpu={gpu_name!r} tick80b_ms={rows[-1]['tick_mean_ms']} train80b_ms={rows[-1]['train_ms']}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
