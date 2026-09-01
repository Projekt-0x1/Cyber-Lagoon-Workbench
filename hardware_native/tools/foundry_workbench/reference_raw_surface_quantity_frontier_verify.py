#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_raw_surface_recipe_critic_v1 import RawSurfaceRecipeCriticV1
import reference_ephemeral_language_recipe_verify as alice_recipe

CTX=0xA610
TRAIN=((101,'careful'),(201,'engineer'),(301,'checks'),(401,'sensor'))
HELD=((102,'senior'),(202,'auditor'),(302,'checks'),(402,'record'))
# Authored generic construction-network aperture: port permutation plus article
# placement. These are candidate mechanics, not desired outputs or capabilities.
PATTERNS=(
 ((0,1,2,3),(1,0,0,1)),
 ((0,1,2,3),(1,1,0,0)),
 ((0,1,3,2),(1,1,0,0)),
 ((0,1,3,2),(1,0,1,0)),
 ((0,2,1,3),(1,0,1,0)),
 ((0,2,3,1),(1,0,1,0)),
 ((0,3,1,2),(1,1,0,0)),
 ((0,3,2,1),(1,0,1,0)),
 ((1,0,2,3),(1,0,1,0)),
 ((1,0,3,2),(1,0,1,0)),
 ((1,2,0,3),(1,0,1,0)),
 ((3,0,2,1),(1,0,1,0)),
)

def surface(words,pattern):
 order,articles=pattern;out=[]
 for slot in order:
  if articles[slot]:out.append('the')
  out.append(words[slot])
 return ' '.join(out)+'.'

def build_organism():
 o=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8))
 for base,rows in ((10000,TRAIN),(20000,HELD)):
  for i,(atom,text) in enumerate(rows):
   for witness in range(2):
    src=base+i*10+witness
    o.contact(CONTACT_SCENE,(7,100,1,atom),src,True,True)
    o.contact(CONTACT_SURFACE,tuple(text.encode()),src+1000,True,True)
 train_words=tuple(x[1] for x in TRAIN)
 train_atoms=tuple(x[0] for x in TRAIN)
 for index,pattern in enumerate(PATTERNS):
  text=surface(train_words,pattern)
  for witness in range(2):
   src=30000+index*10+witness
   o.contact(CONTACT_SCENE,(7,CTX,4,*train_atoms),src,True,True)
   o.contact(CONTACT_SURFACE,tuple(text.encode()),src,True,True)
 rows=o.language.template_candidates(CTX,4)
 assert len(rows)==len(PATTERNS) and len({r.support for r in rows})==1
 return o

def build_raw():
 raw=(Path(__file__).resolve().parents[2]/'data'/'alice.txt').read_bytes()[:131072]
 _surface,ecology,_state,_sm,_rm=alice_recipe._alice_build(raw,0)
 ecology.compact_training_buffer();return ecology

def main():
 started=time.perf_counter();checks={};o=build_organism();raw=build_raw();critic=RawSurfaceRecipeCriticV1(raw)
 held_atoms=tuple(x[0] for x in HELD);held_words=tuple(x[1] for x in HELD)
 o.contact(CONTACT_PARTNER_CONTEXT,(1,7,99001),80001,True,True)
 sid=o.contact(CONTACT_SCENE,(7,CTX,4,*held_atoms),90001,True,True);scene=o._scene_by_id[sid]
 candidates=critic.current_candidates(o,scene);selection=critic.select(o,scene)
 signatures={}
 for row in candidates:
  occ=raw.unfold_all(row.surface)
  signatures[row.surface]=(len(occ),sum(raw.recipes[x.recipe_identity].anchor_count*raw.recipes[x.recipe_identity].support for x in occ),tuple(sorted(x.recipe_identity for x in occ)))
 peak=max((sig[0],sig[1]) for sig in signatures.values())
 peak_rows=[surface for surface,sig in signatures.items() if (sig[0],sig[1])==peak]
 expected={surface(held_words,p).encode() for p in PATTERNS}
 checks['twelve_authored_construction_networks_are_live']=len(candidates)==12 and set(signatures)==expected
 checks['developmental_support_is_equal']=len({row.developmental_support for row in candidates})==1
 checks['quantity_exposes_higher_recipe_aliasing']=len(peak_rows)>=2 and peak[0]>0
 checks['raw_surface_layer_refuses_tied_higher_structure']=selection is None and o.tick() is None
 checks['no_candidate_was_pruned_before_raw_evaluation']=critic.last_surface_candidates==12
 checks['bounded_current_raw_work']=critic.last_raw_touches<256
 checks['no_answer_or_semantic_policy_api']=all(not hasattr(critic,n) for n in ('answer','semantic_policy','expected','prompt','llm'))
 report={'schema':'0x1.reference-raw-surface-quantity-frontier.v1','pass':all(checks.values()),'checks':checks,
         'candidate_count':len(candidates),'peak_signature':[peak[0],peak[1]],'peak_ties':len(peak_rows),
         'peak_surfaces':[x.decode() for x in sorted(peak_rows)[:12]],'raw_candidate_touches':critic.last_raw_touches,
         'raw_recipe_bytes':len(raw.packed_state()),'runtime_llm':False,'graph_flip':False,
         'claim':'AUTHORED_SURFACE_NETWORK_QUANTITY_EXPOSES_RAW_RECIPE_ALIASING_AND_REFUSES_SEMANTIC_OVERREACH_REFERENCE_ONLY',
         'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
 print('FOUNDRY_RAW_SURFACE_QUANTITY_FRONTIER '+('GREEN' if report['pass'] else 'RED')+f" candidates={len(candidates)} peak_ties={len(peak_rows)} touches={critic.last_raw_touches}")
 print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if report['pass'] else 1)

if __name__=='__main__':main()
