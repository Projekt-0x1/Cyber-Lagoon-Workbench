#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,AdultStateV1
from reference_language_mastery_contact_adapter_v1 import *
from reference_predictive_credit_profile_v1 import Q


def main():
    t=time.perf_counter();a=LanguageMasteryAdultV1();contact=LanguageMasteryContactAdapterV1(a);C=9001;J=9101;CTX=0x1A61
    A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402
    names={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
    for f,text in names.items():
        contact.contact(CONTACT_SCENE,(100,f),1000+f);contact.contact(CONTACT_SURFACE,tuple(text.encode()),1000+f)
        contact.contact(CONTACT_SCENE,(100,f),2000+f);contact.contact(CONTACT_SURFACE,tuple(text.encode()),2000+f)
    x=(A1,G1,V1,O1);y=(A2,G2,V2,O2)
    sx=contact.contact(CONTACT_SCENE,(C,*x),3001);contact.contact(CONTACT_SURFACE,tuple(b'the careful engineer tests the sensor.'),3001)
    sy=contact.contact(CONTACT_SCENE,(C,*y),3002);contact.contact(CONTACT_SURFACE,tuple(b'the quiet technician inspects the valve.'),3002)
    l1=a.leaf(C,x);l2=a.leaf(C,y);l3=a.leaf(C,(A2,G1,V1,O2));l4=a.leaf(C,(A1,G2,V2,O1))
    contact.contact(CONTACT_RELATION,(J,sx,sy),5001);contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(l1.surface)+(32,)+tuple(l2.surface),5001)
    s3=contact.contact(CONTACT_SCENE,(C,A2,G1,V1,O2),5002);contact.contact(CONTACT_RELATION,(J,sy,s3),5002);contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(l2.surface)+(32,)+tuple(l3.surface),5002)
    old=a.compose(J,l3,l4)
    oldp=None
    for _ in range(3):oldp=a.experience_program((l3.identity,l4.identity),old,Q,Q//8,CTX,Q//3,True)
    deeper=a.compose(J,oldp.identity,l1)
    for _ in range(4):a.experience_choice(oldp.identity,Q,Q//8,CTX,Q//3,6,True)
    BAD=0xBAD
    a.experience_atomic_program(BAD,l1,-Q,-Q//4,0xBADD,Q//16,True)
    # Saturate the same selection bank with consequence-cold, low-exposure matter.
    filler=0x100000
    while len(a.credit.rows)<a.credit.capacity:
        pid=filler;filler+=1
        a.programs.bind_factor(pid,a.programs.factor(BAD))  # carrier reuse; no new language bytes taught
        start=a._advance();a.credit.observe_use(pid,start,start+1,Q//32,0x9000+(pid&0xffff));a._tick=start+2
    before_capacity=a.credit.capacity;before_rows=len(a.credit.rows)
    old_surface=a.public_surface(oldp.identity);old_emit=[];ex=a.expression(oldp.identity)
    while True:
        p=ex.emit()
        if p is None:break
        old_emit.append(p.value);assert ex.reafference(p,p.value)
    # New deeper program arrives at saturation and must displace low-evidence filler.
    newp=None
    for _ in range(3):newp=a.experience_program((oldp.identity,l1.identity),deeper,3*Q//2,Q//8,CTX,Q//2,True)
    for _ in range(3):a.experience_choice(newp.identity,3*Q//2,Q//8,CTX,Q//2,9,True)
    chosen=a._probe_choice(CTX,AdultStateV1())
    new_surface=a.public_surface(newp.identity);new_emit=[];nx=a.expression(newp.identity)
    while True:
        p=nx.emit()
        if p is None:break
        new_emit.append(p.value);assert nx.reafference(p,p.value)
    checks={
      'fixed_capacity':a.credit.capacity==before_capacity and len(a.credit.rows)==before_rows,
      'eviction_occurred_instead_of_capacity_cliff':a.credit.evictions>0 and a.credit.capacity_refusals==0,
      'old_useful_language_survives':oldp.identity in a.credit.rows and tuple(old_emit)==tuple(old_surface),
      'negative_avoidance_survives':BAD in a.credit.rows and a.credit.row(BAD).outcome_mean_q16<0,
      'new_deeper_language_admitted':newp.identity in a.credit.rows and len(new_surface)>len(old_surface) and newp.depth>oldp.depth,
      'consequence_selects_new_deeper_program':chosen==newp.identity,
      'new_deeper_program_emits_exactly':tuple(new_emit)==tuple(new_surface),
      'no_capacity_growth':before_capacity==256,
      'bounded_fast_path':time.perf_counter()-t<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_LANGUAGE_MASTERY_INTERFERENCE_RED '+','.join(failed))
    pth=Path(__file__);result={'contract':'FOUNDRY_AGI_LANGUAGE_MASTERY_INTERFERENCE_GREEN','reference_only':True,'capacity':a.credit.capacity,'evictions':a.credit.evictions,'old_bytes':len(old_surface),'new_bytes':len(new_surface),'old_depth':oldp.depth,'new_depth':newp.depth,'checks':checks,'sha256':hashlib.sha256(pth.read_bytes()).hexdigest(),'remaining_red':['DIRECT_CAUSAL_PROGRAM_MIGRATION']}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
