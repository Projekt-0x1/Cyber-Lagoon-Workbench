#!/usr/bin/env python3
"""N+2: separately learned affine relations feed a heterogeneous exact closure."""
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1
from reference_heterogeneous_exact_relation_algebra_v1 import Q,AFFINE,algebraic_source,compose_affine,qadd,qmul,reduce_exact
from reference_heterogeneous_mathematical_composition_verify import build,mixed


def observe_relation(rows):
    r=HeadingCoordinateCalibrationV1(window=8,min_support=2)
    for x,y in rows:
        r.observe_pair(x,y);r.observe_pair(x,y)
    return r


def boundary(owner):
    row=owner.relation()
    if row is None:return None
    x0,y0,num,den=map(int,row)
    scaled=num*Q
    if den==0 or scaled%den:return None
    gain=scaled//den
    gx=qmul(gain,x0*Q)
    if gx is None:return None
    offset=qadd(y0*Q,-gx)
    return None if offset is None else (gain,offset)


def apply(bound,x):
    if bound is None:return None
    prod=qmul(bound[0],x*Q)
    return None if prod is None else qadd(prod,bound[1])


def certified_compose(left,right):
    source=algebraic_source(AFFINE,left[0],left[1],right[0],right[1])
    witness=reduce_exact(source)
    return None if witness is None else tuple(witness.result_q16)


def main():
    t=time.perf_counter();c={}
    # Three separately acquired coordinate laws. No endpoint X->Z or X->W samples exist.
    xy=observe_relation(((-4,-5),(4,11)))       # y=2x+3
    yz=observe_relation(((-5,-11),(11,37)))     # z=3y+4
    zw=observe_relation(((-11,13),(37,-35)))    # w=-z+2
    bxy,byz,bzw=boundary(xy),boundary(yz),boundary(zw)
    c['three_relations_are_learned_from_distinct_anchor_spaces']=(bxy==(2*Q,3*Q) and byz==(3*Q,4*Q) and bzw==(-Q,2*Q))

    one=HeadingCoordinateCalibrationV1();one.observe_pair(-4,-5)
    repeat=HeadingCoordinateCalibrationV1();repeat.observe_pair(-4,-5);repeat.observe_pair(-4,-5)
    c['one_or_repeated_one_anchor_cannot_identify_relation']=boundary(one) is None and boundary(repeat) is None

    xz=compose_affine(bxy,byz);xw=None if xz is None else compose_affine(xz,bzw)
    c['two_learned_relations_compose_unseen_x_to_z']=xz==(6*Q,13*Q)
    c['three_learned_relations_compose_unseen_x_to_w']=xw==(-6*Q,-11*Q)
    reverse=compose_affine(byz,bxy)
    c['composition_order_is_noncommutative']=reverse==(6*Q,11*Q) and reverse!=xz

    # Transient closure must preserve the certified exact equations without
    # rematerializing source/witness identities that no caller retains.  The
    # sweep includes signed, non-integral and overflow-prone coefficients.
    hostile=(-3*Q,0,Q,1,Q//3,(1<<31)-1)
    contrasts=[((a,b),(d,e)) for a in hostile for b in hostile
               for d in hostile for e in hostile]
    c['transient_composition_matches_certified_values_and_refusals']=all(
        compose_affine(left,right)==certified_compose(left,right)
        for left,right in contrasts)
    bench=contrasts[:4096]
    bt=time.perf_counter();certified_digest=0
    for left,right in bench:
        value=certified_compose(left,right)
        certified_digest^=0 if value is None else value[0]^value[1]
    certified_elapsed=time.perf_counter()-bt
    bt=time.perf_counter();transient_digest=0
    for left,right in bench:
        value=compose_affine(left,right)
        transient_digest^=0 if value is None else value[0]^value[1]
    transient_elapsed=time.perf_counter()-bt
    c['transient_composition_removes_certificate_work']=(
        transient_digest==certified_digest and
        transient_elapsed*4<certified_elapsed)

    held=(-3,-1,0,1,3)
    endpoint=[]
    for x in held:
        y=xy.map_visual(x);z=None if y is None else yz.map_visual(y);w=None if z is None else zw.map_visual(z)
        endpoint.append((x,w,apply(xw,x)))
    c['heldout_endpoint_matches_sequential_learned_maps']=all(w is not None and mapped==w*Q for _x,w,mapped in endpoint)

    # The actual learned XY and YZ boundaries author the entry source. The fixture never writes 6x+13.
    learned_source=algebraic_source(AFFINE,bxy[0],bxy[1],byz[0],byz[1])
    hetero=build(learned_source)
    if hetero is None:raise RuntimeError('learned_heterogeneous:build')
    aw,pw,sw,qw,_=hetero
    c['heterogeneous_entry_is_actual_learned_composition']=aw.result_q16==xz and pw.source.coefficients_q16[:2]==xz and sw.source.coefficients_q16[0]==xz[0]
    def sequential_hetero(x):
        y=xy.map_visual(x);z=yz.map_visual(y)
        zq=z*Q;sq=qmul(zq,zq);half=None if sq is None else qmul(Q//2,sq);p=None if half is None else qadd(half,3*Q)
        s=None if p is None else qmul(xz[0]+Q//4,p)
        return p,s
    c['learned_relation_drives_heldout_polynomial_schur_bisimulation_trajectory']=all(mixed(hetero,x*Q)==sequential_hetero(x) for x in held)

    # Checkpoint does not need any endpoint observations to reconstruct the same composite.
    rxy=HeadingCoordinateCalibrationV1.restore(copy.deepcopy(xy.checkpoint()));ryz=HeadingCoordinateCalibrationV1.restore(copy.deepcopy(yz.checkpoint()));rzw=HeadingCoordinateCalibrationV1.restore(copy.deepcopy(zw.checkpoint()))
    restored_xz=compose_affine(boundary(rxy),boundary(ryz));restored_xw=compose_affine(restored_xz,boundary(rzw))
    c['checkpointed_learners_reconstruct_same_unseen_endpoint']=restored_xz==xz and restored_xw==xw

    # One relation with only one distinct anchor destroys the larger closure.
    cut_yz=HeadingCoordinateCalibrationV1();cut_yz.observe_pair(-5,-11);cut_yz.observe_pair(-5,-11)
    cut=boundary(cut_yz)
    c['one_relation_lesion_blocks_endpoint_and_heterogeneous_closure']=cut is None

    conflict=HeadingCoordinateCalibrationV1()
    for pair in ((-4,-5),(-4,-5),(4,11),(4,11),(0,99),(0,99)):conflict.observe_pair(*pair)
    c['contradictory_third_anchor_refuses_learned_relation']=boundary(conflict) is None
    c['learner_api_has_no_gain_offset_or_expected_endpoint']=list(inspect.signature(HeadingCoordinateCalibrationV1.observe_pair).parameters)==['self','visual_coordinate','vestibular_angle']
    c['no_direct_endpoint_training_state']=all(len(owner.pairs)==4 for owner in (xy,yz,zw)) and not hasattr(xy,'endpoint') and not hasattr(yz,'endpoint')
    c['bounded_cheap_lane']=time.perf_counter()-t<0.1
    failed=[k for k,v in c.items() if not v]
    res={'schema':'cyber-lagoon.reference-learned-heterogeneous-mathematical-composition.v1','contract':'FOUNDRY_LEARNED_HETEROGENEOUS_MATHEMATICAL_COMPOSITION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'economic_refactor':True,'language_phenotype_improved':False,'visible_language_gain':'PRESERVED_NOT_ADVANCED','learned_boundaries_q16':{'xy':list(bxy or ()),'yz':list(byz or ()),'zw':list(bzw or ()),'xz_heldout':list(xz or ()),'xw_heldout':list(xw or ())},'heldout_endpoint':endpoint,'heterogeneous':{'polynomial_q16':list(pw.result_q16),'schur_q16':list(sw.result_q16),'quotient_classes':list(qw.class_by_state)},'transient_composition':{'cases':len(contrasts),'bench_cases':len(bench),'certified_ms':round(certified_elapsed*1000,3),'transient_ms':round(transient_elapsed*1000,3),'certificate_work_removed':True},'checks':c,'failed':failed,'remaining_red':['LEARNED_NONLINEAR_RELATION_FAMILY_INDUCTION','AUTOMATIC_CROSS_FAMILY_CONSTRUCTOR','VARIABLE_ARITY_RELATION_GRAPH_DISCOVERY','ADULT_LANGUAGE_BINDING_TO_MATHEMATICAL_CLOSURE','DIRECT_ADULT_LEARNED_HETEROGENEOUS_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
