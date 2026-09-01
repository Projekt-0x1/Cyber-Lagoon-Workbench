#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
CTX=9401;WARM=9400;P_FORMAL=7001;P_TERSE=7002;P_NEW=7003
M={101:'careful',201:'engineer',301:'checks',401:'sensor'}
def u(s):return tuple(s.encode())
def scene(o,src):return o.contact(CONTACT_SCENE,(7,CTX,4,101,201,301,401),src,True,True)
def surf(o,text,src):return o.contact(CONTACT_SURFACE,u(text),src,True,True)
def partner(o,p):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),80000+p,True,True)
def off(o):return o.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),89999,True,True)
def settle(o,a,e,p):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,e),p,True,True)
def config(a):return tuple(row[:3] for row in a.selection_occurrences)
def value(o,ctx,cfg):return o._selection_configuration_evidence(ctx,cfg)[0]
def train(o):
 for f,t in M.items():
  for j in range(2):o.contact(CONTACT_SCENE,(7,100,1,f),10000+f*3+j,True,True);surf(o,t,20000+f*3+j)
 # Equal developmental support for two construction alternatives.
 for src in (30001,30002):scene(o,src);surf(o,'the careful engineer checks the sensor.',src)
 for src in (31001,31002):scene(o,src);surf(o,'careful engineer: checks sensor.',src)
 # Separate unambiguous context used only to establish real partner-specific shared history.
 for src in (32001,32002):o.contact(CONTACT_SCENE,(7,WARM,4,101,201,301,401),src,True,True);surf(o,'the careful engineer checks the sensor.',src)
 return o
def context_for(o,p):
 # Establish shared history through an unambiguous learned interaction, then
 # derive the target-context reinstatement signature from the same semantic atoms.
 partner(o,p);o.contact(CONTACT_SCENE,(7,WARM,4,101,201,301,401),40000+p,True,True);a=o.tick();assert a is not None
 o.contact(CONTACT_CONSEQUENCE,(a.ticket,1),p,True,True)
 scene(o,41000+p);ctx=o._selection_preference_context(o.current_scene);o.current_scene.demonstrated=True
 assert ctx!=0
 return ctx
def template_ids(o):
 rows=o.language.template_candidates(CTX,4);out={}
 for t in rows:
  r=o.language.render_template(t,tuple(u(M[x]) for x in (101,201,301,401)))
  out[bytes(r).decode()]=int(t.identity[:15],16)
 return out

def main():
 t=time.perf_counter();checks={};base=train(ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8)));ids=template_ids(base)
 checks['two_equal_learned_constructions']=len(ids)==2 and len({x.support for x in base.language.template_candidates(CTX,4)})==1
 # Seed partner-specific lived preference by recording consequence-qualified preference rows directly through an actual selected action path.
 def learn_partner(p,target_text):
  o=ReferenceOrganismV2.restore(copy.deepcopy(base.checkpoint()));ctx=context_for(o,p);tid=ids[target_text]
  # Developmental support remains tied. A temporary one-source support advantage lets the target participate once in this context.
  off(o);scene(o,50000+p);surf(o,target_text,50000+p);partner(o,p);scene(o,51000+p);a=o.tick();assert a is not None and a.template_identity==tid and a.selection_context==ctx;settle(o,a,1,p)
  # Equalize other construction support outside social context.
  off(o);other=next(k for k,v in ids.items() if v!=tid);scene(o,52000+p);surf(o,other,52000+p);partner(o,p)
  return o,ctx,config(a)
 formal,fctx,fcfg=learn_partner(P_FORMAL,'the careful engineer checks the sensor.')
 terse,tctx,tcfg=learn_partner(P_TERSE,'careful engineer: checks sensor.')
 checks['partner_configurations_are_separate']=value(formal,fctx,fcfg)==1 and value(formal,fctx,tcfg)==0 and value(terse,tctx,tcfg)==1
 # Same semantic closure, different partner histories.
 scene(formal,60001);fa=formal.tick();scene(terse,60002);ta=terse.tick()
 checks['formal_partner_selects_formal']=fa is not None and fa.payload==u('the careful engineer checks the sensor.')
 checks['terse_partner_selects_terse']=ta is not None and ta.payload==u('careful engineer: checks sensor.')
 # Settle terse use so later work is causally permitted; this strengthens only terse partner history.
 settle(terse,ta,1,P_TERSE)
 # A new partner has no earned configuration. Because the two developmental
 # configurations remain exactly tied, the organism must refuse rather than
 # invent a social/register preference; novelty nomination requires unique local geometry.
 new=ReferenceOrganismV2.restore(copy.deepcopy(base.checkpoint()));partner(new,P_NEW);scene(new,60003);checks['unfamiliar_partner_equal_geometry_refuses']=new.tick() is None and not new.selection_configuration_revisions
 # Negative consequence can reopen the partner-specific choice without changing the other partner.
 settle(formal,fa,-1,P_FORMAL);scene(formal,60004);reopened=formal.tick();checks['negative_reopens_formal_partner']=reopened is not None and config(reopened)!=fcfg
 scene(terse,60005);still=terse.tick();checks['other_partner_preference_unaffected']=still is not None and still.payload==u('careful engineer: checks sensor.')
 checks['no_style_register_persona_api']=all(not hasattr(base,n) for n in ('style','register','persona','formal','terse'))
 checks['developmental_support_remains_equal']=len({x.support for x in formal.language.template_candidates(CTX,4)})==1 and len({x.support for x in terse.language.template_candidates(CTX,4)})==1
 out={'schema':'0x1.reference-organism-register.v1','pass':all(checks.values()),'checks':checks,'formal':bytes(fa.payload).decode() if fa else '', 'terse':bytes(ta.payload).decode() if ta else '', 'claim':'PARTNER_CONDITIONED_LEARNED_PARAPHRASE_REFERENCE_ONLY_NOT_HUMAN_STYLE_OR_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print('FOUNDRY_REFERENCE_ORGANISM_REGISTER '+('GREEN' if out['pass'] else 'RED')+' partner_specific=1 style_api=0 consequence_revision=1');print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
if __name__=='__main__':main()
