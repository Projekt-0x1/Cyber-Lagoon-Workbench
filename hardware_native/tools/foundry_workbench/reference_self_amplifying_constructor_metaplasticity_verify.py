#!/usr/bin/env python3
"""N+1: language and mathematics recursively improve one generic constructor."""
from __future__ import annotations
import copy,json,time
from reference_constructor_metaplasticity_v1 import GenericConstructorMetaplasticityV1,Q,_identity
from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1
from reference_heterogeneous_exact_relation_algebra_v1 import compose_affine
from reference_hierarchical_composition_v1 import CONDENSE_PROBATION_PASSES,HierarchicalConstructionV1
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_learned_heterogeneous_mathematical_composition_verify import boundary

CLAUSE=0xCA01;JOIN=0xCA02;JOINER=b' then '
A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402
NAMES={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
LEAF_ATOMS=((A1,G1,V1,O1),(A2,G2,V2,O2),(A2,G1,V1,O2),(A1,G2,V2,O1),(A1,G1,V2,O2),(A2,G2,V1,O1),(A1,G2,V1,O2),(A2,G1,V2,O1),(A1,G2,V1,O1))

def u(text):return tuple(text.encode())
def clause_text(atoms):
 a,g,v,o=(NAMES[x] for x in atoms);return f'the {a} {g} {v} the {o}.'

def language_base():
 e=LearnedSurfaceEcologyV1()
 for feature,text in NAMES.items():
  e.observe_naming(feature,u(text),0x10000+feature);e.observe_naming(feature,u(text),0x20000+feature)
 for atoms,source in ((LEAF_ATOMS[0],0x31001),(LEAF_ATOMS[1],0x31002)):
  e.observe_construction(CLAUSE,atoms,u(clause_text(atoms)),source)
 h=HierarchicalConstructionV1(e);leaves=tuple(h.leaf(CLAUSE,atoms) for atoms in LEAF_ATOMS)
 assert h.observe(JOIN,leaves[:2],(*leaves[0].surface,*JOINER,*leaves[1].surface),0x41001)
 assert h.observe(JOIN,leaves[1:3],(*leaves[1].surface,*JOINER,*leaves[2].surface),0x41002)
 template=e.span_template(JOIN,2)
 if template is None:raise RuntimeError('constructor_meta:join_template')
 return e,h,tuple(int(x) for x in template.sources)

def restore_language(ecp,hcp):
 e=LearnedSurfaceEcologyV1.restore(copy.deepcopy(ecp));h=HierarchicalConstructionV1.restore(e,copy.deepcopy(hcp));leaves=tuple(h.leaf(CLAUSE,atoms) for atoms in LEAF_ATOMS);return e,h,leaves

def guarded_language_chain(constructor,h,leaves,count,roots):
 cur=leaves[0]
 for leaf in leaves[1:int(count)]:
  depth=1+max(int(cur.depth),int(leaf.depth))
  if not constructor.admits(depth):return None,None
  cur=h.compose(JOIN,(cur,leaf))
 return cur,constructor.receipt(cur.identity,cur.depth,cur.ancestry,roots)

def learned_affine(rows):
 owner=HeadingCoordinateCalibrationV1(window=8,min_support=2)
 for x,y in rows:owner.observe_pair(x,y);owner.observe_pair(x,y)
 b=boundary(owner)
 if b is None:raise RuntimeError('constructor_meta:learned_affine')
 return b,_identity('constructor-learned-affine-witness-v1',owner.checkpoint())

def mathematical_evidence():
 transforms=((2,1),(3,-2),(-1,4),(1,5),(2,-3),(-1,2),(1,7),(2,-1))
 specs=tuple((((-2,g*-2+b),(3,g*3+b))) for g,b in transforms)
 learned=tuple(learned_affine(rows) for rows in specs)
 return tuple(row[0] for row in learned),tuple(row[1] for row in learned)

def guarded_math_chain(constructor,relations,roots):
 depth=len(relations)-1
 if depth<0 or not constructor.admits(depth):return None,None
 cur=relations[0];ancestry=[int(roots[0])]
 for index,nxt in enumerate(relations[1:],1):
  cur=compose_affine(cur,nxt)
  if cur is None:return None,None
  ancestry.append(int(roots[index]))
 structure=_identity('constructor-affine-chain-result-v1',(relations,cur))
 return cur,constructor.receipt(structure,depth,ancestry,roots)

def cold(h,closure):
 h.drop_hot_cache();surface=tuple(closure.surface);return surface,int(h.last_materialization_touches),int(h.last_condensed_recipe)

def main():
 started=time.perf_counter();checks={};e0,h0,join_roots=language_base();ecp=copy.deepcopy(e0.checkpoint());hcp=copy.deepcopy(h0.checkpoint())
 experimental=GenericConstructorMetaplasticityV1(base_depth=6,max_depth=8);yoked_control=GenericConstructorMetaplasticityV1(base_depth=6,max_depth=8)
 # A: same language, same shallow held-out construction, different returned consequence authority.
 _ee,eh,eleaves=restore_language(ecp,hcp);_ce,ch,cleaves=restore_language(ecp,hcp)
 shallow_e,receipt_e=guarded_language_chain(experimental,eh,eleaves,7,join_roots);shallow_c,receipt_c=guarded_language_chain(yoked_control,ch,cleaves,7,join_roots)
 shallow_surface=b'' if shallow_e is None else bytes(shallow_e.surface)
 demonstrated={bytes(eleaves[0].surface)+JOINER+bytes(eleaves[1].surface),bytes(eleaves[1].surface)+JOINER+bytes(eleaves[2].surface)}
 checks['identical_siblings_have_same_heldout_shallow_language_competence']=(shallow_e is not None and shallow_c is not None and shallow_e.depth==shallow_c.depth==6 and bytes(shallow_e.surface)==bytes(shallow_c.surface) and shallow_surface.count(JOINER)==6)
 checks['shallow_seven_clause_surface_was_never_demonstrated']=shallow_surface not in demonstrated and shallow_surface.count(JOINER)==6
 changed=experimental.settle_child(receipt_e,Q,0x51001,True,True);yoked_changed=yoked_control.settle_child(receipt_c,Q,0x51001,False,True)
 checks['only_action_contingent_constructed_language_child_revises_constructor']=(changed and not yoked_changed and experimental.max_dependency_depth==7 and yoked_control.max_dependency_depth==6 and experimental.revision==1 and yoked_control.revision==0)
 once_revised_cp=copy.deepcopy(experimental.checkpoint())
 before=experimental.digest();rehearsal_changed=experimental.settle_child(receipt_e,Q,0x51002,True,True,True)
 checks['endogenous_rehearsal_cannot_self_amplify_constructor']=not rehearsal_changed and experimental.digest()==before
 # A hostile sub-frontier success is useful experience but cannot justify more
 # structural reach. Only a child exercising the current frontier may ratchet it.
 subfrontier=experimental.receipt(shallow_e.identity,2,shallow_e.ancestry,join_roots);sub_before=experimental.digest()
 checks['successful_subfrontier_child_cannot_expand_current_constructor_horizon']=not experimental.settle_child(subfrontier,Q,0x51003,True,True) and experimental.digest()==sub_before
 # B: same constructor state controls deeper nonlinguistic composition over independently learned affine laws.
 relations,math_roots=mathematical_evidence();math_e,math_receipt=guarded_math_chain(experimental,relations,math_roots);math_c,_=guarded_math_chain(yoked_control,relations,math_roots)
 checks['first_language_revision_unlocks_deeper_learned_mathematical_closure']=(math_e is not None and math_c is None and experimental.max_dependency_depth==7 and yoked_control.max_dependency_depth==6)
 x=2*Q;endpoint=(((math_e[0]*x)>>16)+math_e[1]) if math_e is not None else 0
 checks['deeper_math_is_exact_unseen_endpoint_not_authored_answer']=endpoint==39*Q
 # C: successful math child re-arms the same constructor; a yoked equal outcome does not.
 before_math=copy.deepcopy(experimental.checkpoint());math_yoked=GenericConstructorMetaplasticityV1.restore(copy.deepcopy(before_math))
 math_changed=experimental.settle_child(math_receipt,Q,0x52001,True,True);math_yoked_changed=math_yoked.settle_child(math_receipt,Q,0x52001,False,True)
 checks['mathematical_child_recursively_revises_same_constructor_but_yoke_cannot']=(math_changed and not math_yoked_changed and experimental.revision==2 and experimental.max_dependency_depth==8 and math_yoked.revision==1 and math_yoked.max_dependency_depth==7)
 # D: unchanged language evidence now crosses the shared depth-6 floor to depth 8.
 de,dh,dleaves=restore_language(ecp,hcp);_oe,oh,oleaves=restore_language(ecp,hcp);_ce2,ch2,cleaves2=restore_language(ecp,hcp)
 twice_deep,twice_receipt=guarded_language_chain(experimental,dh,dleaves,9,join_roots);once=GenericConstructorMetaplasticityV1.restore(once_revised_cp);once_deep,_=guarded_language_chain(once,oh,oleaves,9,join_roots);control_deep,_=guarded_language_chain(yoked_control,ch2,cleaves2,9,join_roots)
 deep_surface=b'' if twice_deep is None else bytes(twice_deep.surface)
 checks['recursive_cross_domain_gain_unlocks_never_demonstrated_nine_clause_discourse']=(twice_deep is not None and twice_receipt is not None and twice_deep.depth==8 and deep_surface.count(JOINER)==8 and once_deep is None and control_deep is None)
 checks['deep_language_gain_uses_same_fixed_terminal_evidence']=de.checkpoint()==ecp and set(int(x) for x in de.span_template(JOIN,2).sources)==set(join_roots)
 checks['final_structural_depth_exceeds_shared_reference_depth6_floor']=twice_deep is not None and twice_deep.depth==8 and twice_deep.depth>6
 constructor_cp=copy.deepcopy(experimental.checkpoint());restored_constructor=GenericConstructorMetaplasticityV1.restore(constructor_cp);_re,rh,rleaves=restore_language(ecp,hcp);restored_deep,_=guarded_language_chain(restored_constructor,rh,rleaves,9,join_roots)
 lesioned=GenericConstructorMetaplasticityV1.restore(copy.deepcopy(constructor_cp));lesioned.lesion_meta_state();_le,lh,lleaves=restore_language(ecp,hcp);lesioned_deep,_=guarded_language_chain(lesioned,lh,lleaves,9,join_roots);lesioned_shallow,_=guarded_language_chain(lesioned,lh,lleaves,7,join_roots)
 checks['checkpoint_preserves_cross_domain_constructor_gain']=restored_constructor.digest()==experimental.digest() and restored_deep is not None and bytes(restored_deep.surface)==deep_surface
 checks['meta_lesion_removes_depth_gain_but_preserves_shallow_language_competence']=lesioned_deep is None and lesioned_shallow is not None and bytes(lesioned_shallow.surface)==shallow_surface
 forged=type(twice_receipt)(twice_receipt.constructor_identity,twice_receipt.constructor_revision,twice_receipt.structure_identity+1,twice_receipt.dependency_depth,twice_receipt.ancestry_digest,twice_receipt.ancestry_roots,twice_receipt.identity);forged_before=experimental.digest()
 checks['forged_child_receipt_cannot_author_metaplasticity']=not experimental.settle_child(forged,Q,0x53001,True,True) and experimental.digest()==forged_before
 # E: existing automaticity machinery condenses the newly reachable depth-8 language closure.
 baseline=[]
 for _ in range(3):
  surface,touches,selected=cold(dh,twice_deep);baseline.append(touches)
  if surface!=tuple(deep_surface) or selected!=0:raise RuntimeError('constructor_meta:condense_baseline')
 recipe=dh._condensed.get(twice_deep.identity)
 for _ in range(CONDENSE_PROBATION_PASSES):cold(dh,twice_deep)
 active_surface,active_touches,active_recipe=cold(dh,twice_deep);meta_before_auto=experimental.digest()
 checks['new_deep_language_competence_condenses_into_existing_recipe_automaticity']=(recipe is not None and recipe.active==1 and active_surface==tuple(deep_surface) and active_recipe==recipe.identity and active_touches<min(baseline))
 checks['automaticity_cannot_self_credit_constructor']=experimental.digest()==meta_before_auto
 before_deopt=recipe.deoptimizations;de.withdraw_source(join_roots[0]);dh.drop_hot_cache();slow_surface=tuple(twice_deep.surface);slow_touches=int(dh.last_materialization_touches)
 checks['owning_source_withdrawal_deoptimizes_to_slower_exact_network_like_path']=slow_surface==tuple(deep_surface) and slow_touches>active_touches and recipe.active==0 and recipe.deoptimizations==before_deopt+1 and experimental.revision==2
 de.restore_source(join_roots[0])
 checks['constructor_state_contains_no_language_math_or_task_semantics']=all(token not in json.dumps(constructor_cp,sort_keys=True).lower() for token in ('language','math','sentence','grammar','affine','answer','task','join'))
 checks['two_successes_expand_reachable_dependency_depth_by_exactly_two']=constructor_cp['revision']==2 and constructor_cp['max_dependency_depth']==8 and len(constructor_cp['exact_constructor_evidence_lineage'])==2
 checks['bounded_fast_path']=time.perf_counter()-started<1.0
 failed=[k for k,v in checks.items() if not v]
 result={'schema':'cyber-lagoon.self-amplifying-constructor-metaplasticity.v1','contract':'FOUNDRY_SELF_AMPLIFYING_CONSTRUCTOR_METAPLASTICITY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'novel_synthesis':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'LANGUAGE_CONSEQUENCE_EXPANDS_GENERIC_CONSTRUCTION_DEPTH_THEN_DEEPER_MATHEMATICS_EXPANDS_IT_AGAIN_THEN_THE_SAME_FIXED_LANGUAGE_EVIDENCE_YIELDS_A_NEVER_DEMONSTRATED_NINE_CLAUSE_DEPTH8_DISCOURSE_AND_CONDENSES','constructor':constructor_cp,'math':{'learned_relation_count':len(relations),'required_dependency_depth':7,'heldout_endpoint_q16':endpoint,'shared_reference_depth_floor':6},'conversation':{'shallow_before_meta_learning':shallow_surface.decode(),'shallow_depth':0 if shallow_e is None else int(shallow_e.depth),'shallow_clause_count':0 if shallow_e is None else shallow_surface.count(JOINER)+1,'shallow_bytes':len(shallow_surface),'deep_after_language_math_language_loop':deep_surface.decode(),'deep_depth':0 if twice_deep is None else int(twice_deep.depth),'deep_clause_count':0 if twice_deep is None else deep_surface.count(JOINER)+1,'deep_bytes':len(deep_surface)},'automaticity':{'baseline_touches':baseline,'active_touches':active_touches,'deoptimized_touches':slow_touches,'touch_reduction_vs_cold_min':0 if not baseline else min(baseline)-active_touches,'touch_ratio_vs_cold_min':0.0 if not baseline or active_touches<=0 else round(min(baseline)/active_touches,3),'recipe_identity':0 if recipe is None else int(recipe.identity)},'checks':checks,'failed':failed,'remaining_red':['FUTURE_UNAVAILABLE_ACTION_HYPOTHESIS_APERTURE','ENDOGENOUS_CURIOSITY_OVER_CONSTRUCTED_HYPOTHESES','NATURAL_PARTNER_DIALOGUE_DRIVING_CONSTRUCTOR_CREDIT','DIRECT_GENERIC_CONSTRUCTOR_METAPLASTICITY_PARITY','GRAPH_PROMOTION'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
 print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
