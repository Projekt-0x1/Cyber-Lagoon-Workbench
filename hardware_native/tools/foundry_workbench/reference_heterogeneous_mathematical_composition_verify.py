#!/usr/bin/env python3
"""N+1: exact relation-of-relations composition across four mathematical families."""
from __future__ import annotations
import copy,json,time
import reference_heterogeneous_exact_relation_algebra_v1 as algebra
from reference_heterogeneous_exact_relation_algebra_v1 import *


def eval_poly(coeffs,x):
    x2=qmul(x,x)
    if x2 is None:return None
    a=qmul(coeffs[0],x2);b=qmul(coeffs[1],x)
    if a is None or b is None:return None
    s=qadd(a,b)
    return None if s is None else qadd(s,coeffs[2])


def sequential(x):
    u=qmul(2*Q,x);u=None if u is None else qadd(u,Q)
    v=None if u is None else qmul(3*Q,u);v=None if v is None else qadd(v,-2*Q)
    sq=None if v is None else qmul(v,v);half=None if sq is None else qmul(Q//2,sq)
    poly=None if half is None else qadd(half,3*Q)
    second=None if poly is None else qmul(6*Q+Q//4,poly)
    return poly,second


def build(affine_source):
    aw=reduce_exact(affine_source)
    if aw is None or len(aw.result_q16)!=2:return None
    A,B=aw.result_q16
    pw=reduce_exact(algebraic_source(POLYNOMIAL,A,B,Q//2,3*Q))
    sw=reduce_exact(algebraic_source(SCHUR,A,Q//2,Q//2,0))
    qw=reduce_exact(bisimulation_source((2,3,2,3),(0,0,Q,Q)))
    if None in (pw,sw,qw):return None
    ident=algebra._identity('heterogeneous-composite-v1',(aw.witness_identity,pw.witness_identity,sw.witness_identity,qw.witness_identity))
    return aw,pw,sw,qw,ident


def mixed(comp,x,start=0):
    aw,pw,sw,qw,_=comp
    first=eval_poly(pw.result_q16,x)
    if first is None:return None
    cls=qw.class_by_state[start];nxt=qw.result_u32[cls]
    second=qmul(sw.result_q16[0],first) if nxt==1 else first
    return first,second


def fold_left(maps):
    row=maps[0]
    for nxt in maps[1:]:
        row=compose_affine(row,nxt)
        if row is None:return None
    return row


def fold_balanced(maps):
    left=compose_affine(maps[0],maps[1]);right=compose_affine(maps[2],maps[3])
    return None if left is None or right is None else compose_affine(left,right)


def main():
    t=time.perf_counter();checks={}
    source=algebraic_source(AFFINE,2*Q,Q,3*Q,-2*Q);comp=build(source)
    if comp is None:raise RuntimeError('heterogeneous_math:build')
    aw,pw,sw,qw,cid=comp
    checks['affine_eliminates_internal_variable']=aw.result_q16==(6*Q,Q) and aw.eliminated_count==1
    checks['actual_affine_result_drives_polynomial_substitution']=pw.source.coefficients_q16[:2]==aw.result_q16 and pw.result_q16==(18*Q,6*Q,3*Q+Q//2)
    checks['actual_affine_result_drives_schur_elimination']=sw.source.coefficients_q16[0]==aw.result_q16[0] and sw.result_q16==(6*Q+Q//4,)
    checks['finite_state_behavior_minimizes_four_states_to_two']=qw.class_by_state==(0,0,1,1) and qw.eliminated_count==2 and qw.result_u32==(1,1)
    held=(-2*Q,-Q,0,Q,2*Q)
    checks['five_heldout_values_match_uncondensed_relation_graph']=all(mixed(comp,x)==sequential(x) for x in held)

    # Four-map variable-depth composition. Parenthesization is changed while the exact boundary is invariant.
    maps=((2*Q,Q),(3*Q,-2*Q),(-Q,4*Q),(Q//2,2*Q))
    left=fold_left(maps);balanced=fold_balanced(maps)
    checks['four_relation_associative_rebracketing_has_same_boundary']=left is not None and left==balanced
    probes=(-3*Q,-Q,0,Q,3*Q)
    checks['rebracketed_boundary_has_same_heldout_action']=all(
        qadd(qmul(left[0],x),left[1])==qadd(qmul(balanced[0],x),balanced[1]) for x in probes)

    replay=build(source)
    checks['same_constituents_replay_exact_composite']=replay==comp
    alt=build(algebraic_source(AFFINE,2*Q,Q,3*Q,-Q))
    checks['offset_change_propagates_only_to_dependent_downstream_relation']=(alt is not None and alt[-1]!=cid and alt[0].result_q16[0]==aw.result_q16[0] and alt[0].result_q16[1]!=aw.result_q16[1] and alt[1].witness_identity!=pw.witness_identity and alt[2].witness_identity==sw.witness_identity and mixed(alt,Q)!=mixed(comp,Q))
    gain_alt=build(algebraic_source(AFFINE,3*Q,Q,3*Q,-2*Q))
    checks['gain_change_propagates_to_both_polynomial_and_schur_branches']=(gain_alt is not None and gain_alt[0].result_q16[0]!=aw.result_q16[0] and gain_alt[1].witness_identity!=pw.witness_identity and gain_alt[2].witness_identity!=sw.witness_identity and mixed(gain_alt,Q)!=mixed(comp,Q))
    corrupt=copy.deepcopy(source);corrupt=ExactRelationSourceV1(corrupt.kind,corrupt.coefficients_q16,corrupt.successors,corrupt.outputs_q16,corrupt.state_count,corrupt.source_identity^1)
    checks['corrupted_required_source_refuses_whole_composite']=build(corrupt) is None
    inexact=algebraic_source(SCHUR,0,Q,Q,1)
    checks['nonexact_reduction_refuses_instead_of_rounding']=reduce_exact(inexact) is None
    malformed=bisimulation_source((1,0),(0,0));bad=ExactRelationSourceV1(malformed.kind,malformed.coefficients_q16,(3,0,0,0),malformed.outputs_q16,2,0).sealed()
    checks['invalid_state_incidence_refuses']=reduce_exact(bad) is None
    checks['no_surface_or_expected_answer_state']=all(tok not in open(__import__('reference_heterogeneous_exact_relation_algebra_v1').__file__).read().lower() for tok in ('expected_answer','target_text','sympy','equation_parser','language_mastery'))
    checks['bounded_cheap_lane']=time.perf_counter()-t<0.1
    failed=[k for k,v in checks.items() if not v]
    res={'schema':'cyber-lagoon.reference-heterogeneous-mathematical-composition.v1','contract':'FOUNDRY_HETEROGENEOUS_MATHEMATICAL_COMPOSITION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'composition_rank':{'exact_reduction_families':4,'affine_chain_arity':4,'heldout_numeric_probes':len(held)+len(probes),'eliminated_internal_states_or_variables':aw.eliminated_count+pw.eliminated_count+sw.eliminated_count+qw.eliminated_count},'derived':{'affine_q16':list(aw.result_q16),'polynomial_q16':list(pw.result_q16),'schur_q16':list(sw.result_q16),'quotient_classes':list(qw.class_by_state),'composite_identity':cid},'checks':checks,'failed':failed,'remaining_red':['LEARNED_RELATION_FAMILY_INDUCTION','UNBOUNDED_VARIABLE_ARITY','CROSS_FAMILY_AUTOMATIC_CONSTRUCTOR','ADULT_LANGUAGE_BINDING_TO_MATHEMATICAL_CLOSURE','DIRECT_ADULT_HETEROGENEOUS_COMPOSITION_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
