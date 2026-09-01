#!/usr/bin/env python3
"""N+3: shuffled learned relation incidence discovers exact variable-arity closure."""
from __future__ import annotations
import copy,hashlib,inspect,json,random,time
from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1
from reference_heterogeneous_exact_relation_algebra_v1 import Q,AFFINE,algebraic_source,qadd,qmul
from reference_heterogeneous_mathematical_composition_verify import build,mixed
from reference_variable_arity_relation_graph_v1 import RelationEdgeV1,VariableArityRelationGraphV1

A,B,C,D,E,F,G,H,J=(0xA101,0xA102,0xA103,0xA104,0xA105,0xA106,0xA107,0xA108,0xA109)


def learn(rows):
    owner=HeadingCoordinateCalibrationV1()
    for x,y in rows:
        owner.observe_pair(x,y);owner.observe_pair(x,y)
    return owner


def boundary(owner):
    row=owner.relation()
    if row is None:return None
    x0,y0,num,den=map(int,row);scaled=num*Q
    if den==0 or scaled%den:return None
    gain=scaled//den;gx=qmul(gain,x0*Q)
    return None if gx is None else (gain,qadd(y0*Q,-gx))


def evidence(owner):
    raw=json.dumps(owner.checkpoint(),sort_keys=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big') or 1


def edge(left,right,rows):
    owner=learn(rows);b=boundary(owner)
    if b is None:raise RuntimeError('relation_graph:learn')
    return RelationEdgeV1.make(left,right,b,evidence(owner)),owner


def apply(bound,x):
    v=qmul(bound[0],x*Q)
    return None if v is None else qadd(v,bound[1])


def main():
    t=time.perf_counter();checks={}
    # Long A-B-C-D-E route and independently learned short A-F-E route agree at E.
    specs=(
      (A,B,((-4,-5),(4,11))),          # 2x+3
      (B,C,((-5,-11),(11,37))),        # 3x+4
      (C,D,((-11,13),(37,-35))),       # -x+2
      (D,E,((-35,-69),(13,27))),       # 2x+1
      (A,F,((-4,-10),(4,14))),         # 3x+2
      (F,E,((-10,27),(14,-69))),       # -4x-13
      (G,H,((-2,3),(2,7))),            # unrelated x+5
      (B,A,((-5,-4),(11,4))),          # inverse cycle 0.5x-1.5
    )
    learned=[edge(*spec) for spec in specs];edges=[row[0] for row in learned]
    random.Random(941).shuffle(edges)
    closure=VariableArityRelationGraphV1.resolve(edges,A,E)
    checks['shuffled_incidence_discovers_agreeing_two_and_four_edge_paths']=(closure is not None and closure.boundary_q16==(-12*Q,-21*Q) and sorted(map(len,closure.paths))==[2,4])
    checks['no_direct_endpoint_relation_was_learned']=all(not (e.left_space==A and e.right_space==E) for e in edges)

    c1=VariableArityRelationGraphV1.resolve(edges,B,C);c2=VariableArityRelationGraphV1.resolve(edges,A,C);c3=VariableArityRelationGraphV1.resolve(edges,A,D);c4=closure
    checks['one_constructor_handles_arity_one_two_three_four']=(c1 is not None and c2 is not None and c3 is not None and c4 is not None and [len(c1.paths[0]),len(c2.paths[0]),len(c3.paths[0]),max(map(len,c4.paths))]==[1,2,3,4])
    checks['variable_arity_boundaries_are_exact']=(c1.boundary_q16==(3*Q,4*Q) and c2.boundary_q16==(6*Q,13*Q) and c3.boundary_q16==(-6*Q,-11*Q) and c4.boundary_q16==(-12*Q,-21*Q))

    probes=(-3,-1,0,1,3)
    checks['heldout_endpoint_matches_both_independent_paths']=all(apply(closure.boundary_q16,x)==(-12*x-21)*Q for x in probes)
    perm=list(reversed(edges));permuted=VariableArityRelationGraphV1.resolve(perm,A,E)
    checks['insertion_permutation_cannot_change_closure']=permuted==closure

    no_unrelated=[e for e in edges if not (e.left_space==G and e.right_space==H)]
    checks['disconnected_relation_has_zero_authority']=VariableArityRelationGraphV1.resolve(no_unrelated,A,E)==closure

    # Remove a long-route edge: the independently supported short route remains sufficient.
    long_cut=[e for e in edges if not (e.left_space==C and e.right_space==D)]
    after_cut=VariableArityRelationGraphV1.resolve(long_cut,A,E)
    checks['long_path_lesion_leaves_independent_short_path']=after_cut is not None and after_cut.boundary_q16==closure.boundary_q16 and tuple(map(len,after_cut.paths))==(2,)
    all_cut=[e for e in edges if e.left_space not in (A,)]
    checks['all_source_paths_removed_refuses']=VariableArityRelationGraphV1.resolve(all_cut,A,E) is None

    # Supported conflicting alternate path must reopen ambiguity, never rank by path length/insertion.
    aj,_=edge(A,J,((-4,-4),(4,4)))
    je,_=edge(J,E,((-4,-4),(4,4)))
    conflict=VariableArityRelationGraphV1.resolve(edges+[aj,je],A,E)
    checks['independent_conflicting_path_refuses_instead_of_ranking']=conflict is None

    # Existing B->A cycle is genuine supported incidence. Simple-path search must terminate and not create endpoint authority.
    checks['cycle_is_bounded_and_cannot_self_author_endpoint']=(VariableArityRelationGraphV1.resolve(edges,A,A) is None and closure is not None)

    # The DISCOVERED A->C closure, not a supplied path, drives the heterogeneous branch.
    discovered_ac=VariableArityRelationGraphV1.resolve(edges,A,C)
    hetero=build(algebraic_source(AFFINE,discovered_ac.boundary_q16[0],discovered_ac.boundary_q16[1],Q,0)) if discovered_ac else None
    checks['discovered_closure_feeds_heterogeneous_math_without_authored_path']=(hetero is not None and hetero[0].result_q16==discovered_ac.boundary_q16 and all(mixed(hetero,x*Q) is not None for x in probes))

    source=inspect.getsource(VariableArityRelationGraphV1).lower()
    checks['constructor_api_contains_no_path_or_expected_boundary_argument']=list(inspect.signature(VariableArityRelationGraphV1.resolve).parameters)==['edges','left_space','right_space','max_depth']
    checks['constructor_has_no_semantic_or_ranking_policy']=all(token not in source for token in ('sensor','answer','expected','shortest','preferred','score','language'))
    checks['constructor_persists_no_graph_or_path_cache']=not vars(VariableArityRelationGraphV1()) and not hasattr(VariableArityRelationGraphV1,'checkpoint')
    checks['bounded_cheap_lane']=time.perf_counter()-t<0.1
    failed=[k for k,v in checks.items() if not v]
    res={'schema':'cyber-lagoon.reference-variable-arity-relation-graph-discovery.v1','contract':'FOUNDRY_VARIABLE_ARITY_RELATION_GRAPH_DISCOVERY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'edge_count':len(edges),'endpoint':None if closure is None else {'boundary_q16':list(closure.boundary_q16),'path_lengths':sorted(map(len,closure.paths))},'arity_boundaries':{str(k):list(v.boundary_q16) for k,v in ((1,c1),(2,c2),(3,c3),(4,c4)) if v is not None},'checks':checks,'failed':failed,'remaining_red':['LEARNED_NONLINEAR_RELATION_GRAPH_EDGES','AUTOMATIC_CROSS_FAMILY_OPERATOR_RECRUITMENT','RECURRENT_FIXED_POINT_GRAPH_CLOSURE','ADULT_LANGUAGE_BINDING_TO_DISCOVERED_MATHEMATICAL_CLOSURE','DIRECT_ADULT_VARIABLE_ARITY_RELATION_GRAPH_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
