#!/usr/bin/env python3
"""Generic conditioned-form and cross-port compatibility checks."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1

def u(s):return tuple(s.encode())
def support_lex(e,f,s,a,b):e.observe_naming(f,u(s),a);e.observe_naming(f,u(s),b)
def support_form(e,f,c,s,a,b):e.observe_form(f,(c,),u(s),a);e.observe_form(f,(c,),u(s),b)

def main():
 t=time.perf_counter();e=LearnedSurfaceEcologyV1();checks={}
 CAT,DOG,SEE,ENGINEER=101,102,103,104;SING,PLUR,GIVEN=7001,7002,8001;CTX=9001
 for f,s in ((CAT,'cat'),(DOG,'dog'),(SEE,'see'),(ENGINEER,'engineer')):support_lex(e,f,s,1000+f,2000+f)
 # Learn the surface scaffold from ordinary raw contacts over the unconditioned forms.
 a=(CAT,SEE,DOG);raw=u('the cat see the dog.')
 checks['construction_observation_1']=e.observe_construction(CTX,a,raw,3001)
 checks['construction_observation_2']=e.observe_construction(CTX,a,raw,3002) and e.template(CTX,3) is not None
 # Conditioned form alternatives are contact-learned numeric state, not born categories.
 support_form(e,CAT,SING,'cat',3101,3102);support_form(e,CAT,PLUR,'cats',3103,3104)
 support_form(e,DOG,SING,'dog',3201,3202);support_form(e,DOG,PLUR,'dogs',3203,3204)
 support_form(e,SEE,SING,'sees',3301,3302);support_form(e,SEE,PLUR,'see',3303,3304)
 # Two distinct condition values with the same equality structure induce one abstract compatibility pattern.
 p1=e.observe_compatibility(CTX,(SING,SING,0),4001);p2=e.observe_compatibility(CTX,(PLUR,PLUR,0),4002)
 checks['abstract_equality_pattern']=p1==p2==(1,1,0) and e.compatible(CTX,(SING,SING,0)) and e.compatible(CTX,(PLUR,PLUR,0))
 plural=e.realize_conditioned(CTX,a,((PLUR,),(PLUR,),(SING,)),(PLUR,PLUR,0))
 singular=e.realize_conditioned(CTX,a,((SING,),(SING,),(PLUR,)),(SING,SING,0))
 checks['conditioned_surface_realization']=plural==u('the cats see the dog.') and singular==u('the cat sees the dogs.')
 saved_templates=e.template_candidates;saved_lexemes=e.lexeme_candidates
 e.template_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('conditioned_contact_must_not_rank_templates'))
 e.lexeme_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('conditioned_contact_must_not_rank_lexemes'))
 try:hit=e.observe_conditioned_contact(CTX,a,((PLUR,),(),()),u('the cats see the dog.'),3001)
 finally:
  e.template_candidates=saved_templates;e.lexeme_candidates=saved_lexemes
 checks['conditioned_contact_does_not_rank_slates']=hit
 saved_forms=e.form_candidates
 e.form_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('conditioned_construction_must_not_rank_forms'))
 try:built=e.observe_conditioned_construction(CTX,a,((PLUR,),(PLUR,),(SING,)),u('the cats see the dog.'),3001)
 finally:e.form_candidates=saved_forms
 checks['conditioned_construction_does_not_rank_form_slates']=built
 iso=LearnedSurfaceEcologyV1()
 for f,s in ((CAT,'cat'),(DOG,'dog'),(SEE,'see')):support_lex(iso,f,s,1000+f,2000+f)
 iso.observe_construction(CTX,a,raw,3001);iso.observe_construction(CTX,a,raw,3002)
 support_form(iso,CAT,SING,'cat',3101,3102);support_form(iso,CAT,PLUR,'cats',3103,3104)
 support_form(iso,DOG,SING,'dog',3201,3202);support_form(iso,DOG,PLUR,'dogs',3203,3204)
 support_form(iso,SEE,SING,'sees',3301,3302);support_form(iso,SEE,PLUR,'see',3303,3304)
 saved_roster=iso._active_sources
 iso._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('conditioned_observe_must_not_materialize_source_roster'))
 try:
  roster_built=iso.observe_conditioned_construction(CTX,a,((PLUR,),(PLUR,),(SING,)),u('the cats see the dog.'),3001)
  roster_hit=iso.observe_conditioned_contact(CTX,a,((PLUR,),(),()),u('the cats see the dog.'),3001)
 finally:iso._active_sources=saved_roster
 checks['conditioned_observe_does_not_materialize_source_roster']=roster_built and roster_hit
 checks['nonlocal_mismatch_refuses']=e.realize_conditioned(CTX,a,((PLUR,),(SING,),(SING,)),(PLUR,SING,0)) is None
 # Equality generalizes structurally to an unseen opaque value; no linguistic label is encoded.
 checks['compatibility_generalizes_to_unseen_value']=e.compatible(CTX,(9991,9991,0)) and not e.compatible(CTX,(9991,9992,0))
 saved_index=e._compat_index
 class _Boom(dict):
  def get(self,*a,**k):raise RuntimeError('compat_must_not_scan_index')
  def __contains__(self,k):raise RuntimeError('compat_must_not_scan_index')
 e._compat_index=_Boom()
 try:keyed=e.compatible(CTX,(SING,SING,0)) and e.compatible(CTX,(9991,9991,0)) and not e.compatible(CTX,(9991,9992,0))
 finally:e._compat_index=saved_index
 checks['unique_compatibility_does_not_scan_index']=keyed
 fill=LearnedSurfaceEcologyV1()
 fill.observe_compatibility(CTX,(SING,SING,0),4001);fill.observe_compatibility(CTX,(PLUR,PLUR,0),4002)
 checks['unique_compatibility_completes_zero_port']=fill.complete_compatibility(CTX,(SING,0,0))==(SING,SING,0)
 fill.observe_compatibility(CTX,(SING,0,SING),4101);fill.observe_compatibility(CTX,(SING,0,SING),4102)
 checks['compatibility_completion_tie_refuses']=fill.complete_compatibility(CTX,(SING,0,0)) is None
 # Same generic form mechanism can express discourse-given reference without a pronoun opcode.
 support_form(e,ENGINEER,GIVEN,'she',5101,5102)
 checks['reference_as_conditioned_form']=e.form(ENGINEER,())==u('engineer') and e.form(ENGINEER,(GIVEN,))==u('she')
 # Equal evidence for two conditioned forms remains ambiguous.
 amb=LearnedSurfaceEcologyV1();support_lex(amb,ENGINEER,'engineer',1,2);support_form(amb,ENGINEER,GIVEN,'she',3,4);support_form(amb,ENGINEER,GIVEN,'they',5,6)
 checks['conditioned_form_tie_refuses']=amb.form(ENGINEER,(GIVEN,)) is None
 # Source withdrawal removes only the affected form evidence.
 e.withdraw_source(5102);checks['reference_source_withdrawal']=e.form(ENGINEER,(GIVEN,))==u('engineer')
 cp=e.checkpoint();r=LearnedSurfaceEcologyV1.restore(cp);checks['checkpoint_replay']=r.digest()==e.digest() and r.realize_conditioned(CTX,a,((PLUR,),(PLUR,),(SING,)),(PLUR,PLUR,0))==plural
 result={'schema':'0x1.reference-language-control.v1','pass':all(checks.values()),'language_phenotype_improved':True,'checks':checks,'compatibility_pattern':list(p1),'plural_bytes':len(plural or ()), 'singular_bytes':len(singular or ()), 'claim':'GENERIC_CONDITIONED_SURFACE_AND_COMPATIBILITY_NOT_LANGUAGE_CATEGORY_OPCODE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print('FOUNDRY_REFERENCE_LANGUAGE_CONTROL '+('GREEN' if result['pass'] else 'RED')+' conditioned_forms=1 nonlocal_compatibility=1 reference=1 category_opcodes=0')
 print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
