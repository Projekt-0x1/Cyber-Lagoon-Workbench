#!/usr/bin/env python3
"""N+4: unordered exact reduction incidence recruits cross-family composition."""
from __future__ import annotations
from dataclasses import replace
import hashlib,inspect,json,random,time
from reference_automatic_exact_reduction_program_v1 import AutomaticExactReductionProgramV1,ExactReductionNodeV1,ReductionTermV1
from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1
from reference_heterogeneous_exact_relation_algebra_v1 import AFFINE,BISIMULATION,POLYNOMIAL,SCHUR,Q,algebraic_source,qadd,qmul
from reference_variable_arity_relation_graph_v1 import RelationEdgeV1,VariableArityRelationGraphV1

A,B,C,D=0xA701,0xA702,0xA703,0xA704
N_AFF,N_POLY,N_SCHUR,N_RECONV,N_BISIM,N_FINAL,N_DISTRACTOR=range(0x7101,0x7108)

def _id(value):
 raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode();return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big') or 1

def learned_boundary(gain,offset):
 owner=HeadingCoordinateCalibrationV1(window=8,min_support=2)
 for x in (-2,3):
  y=gain*x+offset;owner.observe_pair(x,y);owner.observe_pair(x,y)
 row=owner.relation()
 if row is None:raise RuntimeError('automatic_reduction:learn')
 x0,y0,num,den=map(int,row);scaled=num*Q
 if den==0 or scaled%den:raise RuntimeError('automatic_reduction:inexact_learned_boundary')
 g=scaled//den;gx=qmul(g,x0*Q);b=None if gx is None else qadd(y0*Q,-gx)
 if b is None:raise RuntimeError('automatic_reduction:learned_boundary')
 return (g,b),_id(owner.checkpoint())

def discovered_affine(last_offset=4):
 specs=((A,B,2,1),(B,C,3,-2),(C,D,-1,int(last_offset)));edges=[]
 for left,right,gain,offset in specs:
  bound,evidence=learned_boundary(gain,offset);edge=RelationEdgeV1.make(left,right,bound,evidence)
  if edge is None:raise RuntimeError('automatic_reduction:edge')
  edges.append(edge)
 random.Random(1701).shuffle(edges);closure=VariableArityRelationGraphV1.resolve(edges,A,D,max_depth=3)
 if closure is None:raise RuntimeError('automatic_reduction:closure')
 return tuple(edges),closure

def program(boundary):
 lit=ReductionTermV1.literal;qref=ReductionTermV1.q16;select=ReductionTermV1.selected_q16
 affine=ExactReductionNodeV1(N_AFF,AFFINE,(lit(boundary[0]),lit(boundary[1]),lit(Q),lit(0)))
 poly=ExactReductionNodeV1(N_POLY,POLYNOMIAL,(qref(N_AFF,0),qref(N_AFF,1),lit(Q//2),lit(3*Q)))
 schur=ExactReductionNodeV1(N_SCHUR,SCHUR,(qref(N_AFF,0),lit(Q//2),lit(Q//2),lit(0)))
 reconv=ExactReductionNodeV1(N_RECONV,AFFINE,(qref(N_POLY,0),qref(N_POLY,2),qref(N_SCHUR,0),qref(N_POLY,1)))
 bisim=ExactReductionNodeV1(N_BISIM,BISIMULATION,successors=(2,3,2,3),outputs_q16=(0,0,Q,Q))
 final=ExactReductionNodeV1(N_FINAL,AFFINE,(select(N_RECONV,N_BISIM,0),lit(0),lit(Q),qref(N_BISIM,1)))
 distractor=ExactReductionNodeV1(N_DISTRACTOR,POLYNOMIAL,(lit(2*Q),lit(Q),lit(Q),lit(7*Q)))
 return (affine,poly,schur,reconv,bisim,final,distractor)

def witness_map(result):return {} if result is None else {int(k):v for k,v in result.evaluated}

def main():
 started=time.perf_counter();checks={};edges,closure=discovered_affine(4);nodes=list(program(closure.boundary_q16));random.Random(9917).shuffle(nodes);snapshot=tuple(nodes)
 result=AutomaticExactReductionProgramV1.resolve(nodes,N_FINAL);wm=witness_map(result);causal={N_AFF,N_POLY,N_SCHUR,N_RECONV,N_BISIM,N_FINAL}
 checks['learned_discovered_affine_boundary_is_program_source']=(closure.boundary_q16==(-6*Q,3*Q) and tuple(map(len,closure.paths))==(3,) and all(not(e.left_space==A and e.right_space==D) for e in edges) and wm.get(N_AFF) is not None and wm[N_AFF].result_q16==closure.boundary_q16)
 checks['final_request_recruits_all_and_only_causal_ancestors']=result is not None and set(wm)==causal
 checks['unused_distractor_is_never_evaluated']=N_DISTRACTOR not in wm
 checks['polynomial_and_schur_read_actual_affine_slots']=(wm[N_POLY].source.coefficients_q16[:2]==wm[N_AFF].result_q16 and wm[N_SCHUR].source.coefficients_q16[0]==wm[N_AFF].result_q16[0])
 checks['second_affine_reconverges_polynomial_and_schur_results']=(wm[N_RECONV].source.coefficients_q16==(wm[N_POLY].result_q16[0],wm[N_POLY].result_q16[2],wm[N_SCHUR].result_q16[0],wm[N_POLY].result_q16[1]))
 selector=wm[N_BISIM].result_u32[0]
 checks['bisimulation_quotient_selects_reconverged_output']=(selector==1 and wm[N_FINAL].source.coefficients_q16[0]==wm[N_RECONV].result_q16[selector] and wm[N_FINAL].source.coefficients_q16[3]==wm[N_BISIM].result_q16[1])
 permuted=list(nodes);random.Random(13).shuffle(permuted);replay=AutomaticExactReductionProgramV1.resolve(permuted,N_FINAL)
 checks['node_insertion_and_sibling_permutation_preserve_exact_witness']=replay==result

 final=next(n for n in nodes if n.identity==N_FINAL)
 missing_final=replace(final,coefficients=(ReductionTermV1.selected_q16(0xDEAD,N_BISIM,0),*final.coefficients[1:]))
 missing_nodes=tuple(missing_final if n.identity==N_FINAL else n for n in nodes)
 checks['missing_dependency_refuses_atomically']=AutomaticExactReductionProgramV1.resolve(missing_nodes,N_FINAL) is None and tuple(nodes)==snapshot
 poly=next(n for n in nodes if n.identity==N_POLY)
 wrong_poly=replace(poly,coefficients=(ReductionTermV1.q16(N_AFF,99),*poly.coefficients[1:]))
 wrong_nodes=tuple(wrong_poly if n.identity==N_POLY else n for n in nodes)
 checks['wrong_result_slot_refuses_atomically']=AutomaticExactReductionProgramV1.resolve(wrong_nodes,N_FINAL) is None and tuple(nodes)==snapshot
 affine=next(n for n in nodes if n.identity==N_AFF)
 cyclic_affine=replace(affine,coefficients=(ReductionTermV1.q16(N_FINAL,0),ReductionTermV1.literal(0),ReductionTermV1.literal(Q),ReductionTermV1.literal(0)))
 cyclic_nodes=tuple(cyclic_affine if n.identity==N_AFF else n for n in nodes)
 checks['dependency_cycle_refuses_atomically']=AutomaticExactReductionProgramV1.resolve(cyclic_nodes,N_FINAL) is None and tuple(nodes)==snapshot
 sealed=algebraic_source(AFFINE,Q,0,Q,0)
 sealed_ok=ExactReductionNodeV1(0x710F,AFFINE,(ReductionTermV1.literal(Q),ReductionTermV1.literal(0),ReductionTermV1.literal(Q),ReductionTermV1.literal(0)),source_identity=sealed.source_identity)
 checks['matching_source_identity_admits_exact_materialization']=AutomaticExactReductionProgramV1.resolve((sealed_ok,),sealed_ok.identity) is not None and tuple(nodes)==snapshot
 corrupt=ExactReductionNodeV1(0x7110,AFFINE,(ReductionTermV1.literal(Q),ReductionTermV1.literal(0),ReductionTermV1.literal(Q),ReductionTermV1.literal(0)),source_identity=sealed.source_identity^1)
 checks['corrupted_source_identity_refuses_atomically']=AutomaticExactReductionProgramV1.resolve((corrupt,),corrupt.identity) is None and tuple(nodes)==snapshot
 inexact=ExactReductionNodeV1(0x7111,SCHUR,tuple(ReductionTermV1.literal(v) for v in (0,Q,Q,1)))
 checks['inexact_arithmetic_refuses_instead_of_rounding']=AutomaticExactReductionProgramV1.resolve((inexact,),inexact.identity) is None and tuple(nodes)==snapshot

 _changed_edges,changed_closure=discovered_affine(5);changed=AutomaticExactReductionProgramV1.resolve(program(changed_closure.boundary_q16),N_FINAL);cwm=witness_map(changed)
 changed_ids={k for k in causal if wm[k].witness_identity!=cwm[k].witness_identity}
 checks['one_upstream_relation_changes_precise_downstream_dependency_cone']=(changed_closure.boundary_q16==(-6*Q,4*Q) and changed_ids=={N_AFF,N_POLY,N_RECONV,N_FINAL} and wm[N_SCHUR].witness_identity==cwm[N_SCHUR].witness_identity and wm[N_BISIM].witness_identity==cwm[N_BISIM].witness_identity)
 source=inspect.getsource(AutomaticExactReductionProgramV1.resolve).lower()
 checks['api_contains_no_stage_list_or_host_topological_order']=(list(inspect.signature(AutomaticExactReductionProgramV1.resolve).parameters)==['nodes','output_identity','max_nodes'] and 'topological' not in source and 'stage' not in source and 'schedule' not in source)
 module_source=inspect.getsource(__import__('reference_automatic_exact_reduction_program_v1')).lower()
 checks['program_state_contains_no_language_answer_or_task_semantics']=all(token not in module_source for token in ('expected_answer','target_text','language','grammar','task_label'))
 checks['bounded_cheap_lane']=time.perf_counter()-started<0.1
 failed=[k for k,v in checks.items() if not v]
 report={'schema':'cyber-lagoon.reference-automatic-exact-reduction-program.v1','contract':'FOUNDRY_AUTOMATIC_EXACT_REDUCTION_PROGRAM_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'unordered_nodes':len(nodes),'evaluated_nodes':sorted(wm),'unused_nodes':[N_DISTRACTOR],'discovered_affine_q16':list(closure.boundary_q16),'final_q16':[] if result is None else list(result.witness.result_q16),'selector_class':int(selector) if result is not None else -1,'changed_dependency_cone':sorted(changed_ids),'checks':checks,'failed':failed,'remaining_red':['LEARNED_NONLINEAR_REDUCTION_DEFINITIONS','OPEN_RECURRENT_FIXED_POINT_REDUCTION_PROGRAMS','THEOREM_PROVING','ADULT_LANGUAGE_INTEGRATION','DIRECT_AUTOMATIC_EXACT_REDUCTION_PARITY'],'claim':'UNORDERED_TYPED_EXACT_REDUCTION_GRAPH_AUTOMATICALLY_RECRUITS_CROSS_FAMILY_COMPOSITION','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
 print(report['contract']);print(json.dumps(report,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
