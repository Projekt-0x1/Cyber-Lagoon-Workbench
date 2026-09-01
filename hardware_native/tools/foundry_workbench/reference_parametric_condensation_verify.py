#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1,HierarchicalRefuse,rematerialize_transient_plan
from reference_parametric_condensation_v1 import ParametricCondensationV1,OP_CALL,OP_SLOT,PROBATION_PASSES

def u(s):return tuple(s.encode())
WORDS=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','mu','nu','xi','omicron','pi','rho','sigma','tau','upsilon','phi','chi','psi','omega','amber','birch','cedar','dune','ember','frost','glint','haze','iris','jade','kelp','lumen','moss','nova','onyx','pearl']
CLAUSE=10;JOIN=11

def lexicon():
    e=LearnedSurfaceEcologyV1()
    for f,text in enumerate(WORDS,1):
        e.observe_naming(f,u(text),1000+f);e.observe_naming(f,u(text),2000+f)
    for base in (3000,4000):
        for f in (1,2,3):e.observe_construction(CLAUSE,(f,),u(WORDS[f-1]),base+f)
    h=HierarchicalConstructionV1(e);leaves={f:h.leaf(CLAUSE,(f,)) for f in range(1,len(WORDS)+1)}
    return e,h,leaves

def teach_span_language(e,L):
    examples=((1,2),(3,4))
    for source,(a,b) in zip((5001,5002,6001,6002),examples+examples):
        assert e.observe_span(JOIN,(L[a].surface,L[b].surface),L[a].surface+u(' then ')+L[b].surface,source)
    assert e.span_template(JOIN,2) is not None

def build():
    e,h,leaves=lexicon();teach_span_language(e,leaves)
    return e,h,leaves

def wrap(plan,surface):
    return SimpleNamespace(identity=plan.identity,context=plan.context,template_identity=plan.template_identity,child_identities=plan.child_identities,depth=plan.depth,surface=surface)
def pair(e,L,a,b):return wrap(*rematerialize_transient_plan(e,JOIN,(L[a],L[b])))
def superpair(e,left,right):return wrap(*rematerialize_transient_plan(e,JOIN,(left,right)))

def main():
    started=time.perf_counter();checks={};e,h,L=build()
    inc=ParametricCondensationV1(h)
    ipairs=[pair(e,L,37,38),pair(e,L,39,40)]
    for row in ipairs:inc._remember(row);inc.materialize(row)
    ishape=inc.shape_digest(ipairs[0]);icp=copy.deepcopy(inc.checkpoint())
    ie=LearnedSurfaceEcologyV1.restore(copy.deepcopy(e.checkpoint()));ih=HierarchicalConstructionV1.restore(ie,copy.deepcopy(h.checkpoint()));i2=ParametricCondensationV1.restore(ih,icp)
    def ileaf(i):return ih.closure(L[i].identity)
    third=wrap(*rematerialize_transient_plan(ie,JOIN,(ileaf(1),ileaf(2))));i2._remember(third);i2.materialize(third)
    ir=i2.recipes.get(i2.by_shape.get(ishape,0));cold=ParametricCondensationV1(ih);cold.materialize(third)
    inode_ids={int(row['identity']) for row in icp.get('nodes',())}
    checks['incomplete_cluster_survives_checkpoint']=(
        ishape not in inc.by_shape and len(inc.pressure.get(ishape,()))==2
        and ishape in icp.get('pressure',{}) and len(icp['pressure'][ishape])==2
        and {int(x) for x in icp['pressure'][ishape]}<=inode_ids
        and ir is not None and not ir.active and len(ir.witness_closures)==3 and third.identity in ir.witness_closures)
    checks['matched_third_without_history_does_not_nominate']=ishape not in cold.by_shape and len(cold.pressure.get(ishape,()))==1
    more=[wrap(*rematerialize_transient_plan(ie,JOIN,(ileaf(i),ileaf(i+1)))) for i in range(3,17,2)]
    for row in more:i2._remember(row)
    for row in more[:2]:i2.materialize(row)
    held_pair=more[2];held_pair_out=i2.materialize(held_pair)
    r37=wrap(*rematerialize_transient_plan(ie,JOIN,(ileaf(37),ileaf(38))));r39=wrap(*rematerialize_transient_plan(ie,JOIN,(ileaf(39),ileaf(40))))
    i2._remember(r37);i2._remember(r39)
    rsupers=[superpair(ie,r37,r39),superpair(ie,third,more[0]),superpair(ie,more[1],held_pair),superpair(ie,more[3],more[4]),superpair(ie,more[5],more[6]),superpair(ie,more[2],more[3])]
    for row in rsupers:i2._remember(row)
    for row in rsupers[:2]:i2.materialize(row)
    sshape=i2.shape_digest(rsupers[0])
    checks['n2_cluster_incomplete_before_rest']=sshape not in i2.by_shape and len(i2.pressure.get(sshape,()))==2 and ir.active
    n2cp=copy.deepcopy(i2.checkpoint())
    ie3=LearnedSurfaceEcologyV1.restore(copy.deepcopy(ie.checkpoint()));ih3=HierarchicalConstructionV1.restore(ie3,copy.deepcopy(ih.checkpoint()));i3=ParametricCondensationV1.restore(ih3,n2cp)
    def jleaf(i):return ih3.closure(L[i].identity)
    def jpair(a,b):
        row=wrap(*rematerialize_transient_plan(ie3,JOIN,(jleaf(a),jleaf(b))));i3._remember(row);return row
    hp=jpair(7,8);hp_out=i3.materialize(hp);ir3=i3.recipes[i3.by_shape[ishape]]
    checks['delayed_n2_does_not_erase_n1']=ir3.active and hp_out==hp.surface and i3.last_recipe==ir3.identity and sshape in i3.pressure
    n1u=ir3.uses;probe=superpair(ie3,jpair(37,38),jpair(39,40));i3._remember(probe)
    probe_out,_=i3._original(probe)
    checks['index_load_survives_checkpoint']=(
        probe_out==probe.surface and ir3.uses==n1u and ir3.active
        and sshape in i3.pressure and len(i3.pressure.get(sshape,()))==2)
    rsup3=[superpair(ie3,jpair(1,2),jpair(3,4)),superpair(ie3,jpair(5,6),hp),superpair(ie3,jpair(9,10),jpair(11,12)),superpair(ie3,jpair(13,14),jpair(15,16)),superpair(ie3,hp,jpair(9,10)),superpair(ie3,jpair(37,38),jpair(39,40))]
    for row in rsup3:i3._remember(row)
    i3.materialize(rsup3[2]);rn2=i3.recipes[i3.by_shape[sshape]]
    for row in rsup3[3:5]:i3.materialize(row)
    r2out=i3.materialize(rsup3[5])
    extras=[jpair(i,i+1) for i in range(17,33,2)]
    for row in extras:i3.materialize(row)
    es=[superpair(ie3,extras[i],extras[i+1]) for i in range(0,8,2)]
    es += [superpair(ie3,jpair(9,10),jpair(13,14)),superpair(ie3,jpair(11,12),jpair(15,16))]
    for row in es:i3._remember(row);i3.materialize(row)
    rthirds=[superpair(ie3,es[0],es[1]),superpair(ie3,es[2],es[3]),superpair(ie3,es[4],es[5]),superpair(ie3,es[0],es[2]),superpair(ie3,es[1],es[3]),superpair(ie3,es[4],es[0])]
    for row in rthirds:i3._remember(row)
    for row in rthirds[:3]:i3.materialize(row)
    tshape=i3.shape_digest(rthirds[0]);rn3=i3.recipes[i3.by_shape[tshape]]
    for row in rthirds[3:5]:i3.materialize(row)
    r3out=i3.materialize(rthirds[5])
    cold2=ParametricCondensationV1(ih)
    for row in (*more,r37,r39,third,*rsupers):cold2._remember(row)
    for row in rsupers[:3]:cold2.materialize(row)
    cn2=cold2.recipes.get(cold2.by_shape.get(sshape,0))
    checks['resumed_cluster_earns_recursive_n2']=(
        ir3.active and rn2.rank==2 and rn2.active and r2out==rsup3[5].surface
        and sum(op.kind==OP_CALL and op.reference_identity==ir3.identity for op in rn2.ops)==2)
    checks['interrupted_n2_resumes_into_n3']=(
        rn3.rank==3 and rn3.active and r3out==rthirds[5].surface and i3.last_rank==3
        and sum(op.kind==OP_CALL and op.reference_identity==rn2.identity for op in rn3.ops)==2)
    checks['matched_supers_without_pair_history_are_not_n2']=(
        cn2 is not None and cn2.rank==1 and all(op.reference_identity!=ir.identity for op in cn2.ops))
    p=ParametricCondensationV1(h)
    pairs=[pair(e,L,i,i+1) for i in range(1,33,2)]
    for row in pairs:p._remember(row)
    training_pair_surfaces={tuple(x.surface) for x in pairs[:5]}
    for row in pairs[:3]:p.materialize(row)
    shape=p.shape_digest(pairs[0]);n1=p.recipes[p.by_shape[shape]]
    checks['distinct_lived_shapes_nominate_n1']=len(n1.witness_closures)==3 and shape not in p.pressure and n1.rank==1 and not n1.active and all(h.closure(row.identity) is None for row in pairs)
    checks['n1_is_parametric_not_leaf_cache']=all(op.kind!=OP_SLOT or op.reference_identity==0 for op in n1.ops) and all(op.reference_identity not in {L[i].identity for i in range(1,7)} for op in n1.ops)
    for row in pairs[3:5]:p.materialize(row)
    checks['heldout_shadow_probation_earns_n1']=n1.active and n1.probation_passes==PROBATION_PASSES
    original1,orig1_touches=p._original(pairs[5]);out1=p.materialize(pairs[5]);n1_touches=p.last_touches
    checks['heldout_fillers_execute_through_n1']=out1==original1 and tuple(out1) not in training_pair_surfaces and p.last_recipe==n1.identity and p.last_rank==1
    checks['n1_reduces_internal_touched_work']=n1_touches<orig1_touches
    tplan,tsurface=rematerialize_transient_plan(e,JOIN,(L[35],L[36]));texec,t_touches=p._execute(n1,tplan)
    checks['transient_heldout_pair_executes_n1_without_composite_closure']=(
        texec==tsurface and h.closure(tplan.identity) is None and t_touches>0 and n1.active)
    q=ParametricCondensationV1(h);trows=[]
    for i in range(1,13,2):
        plan,surface=rematerialize_transient_plan(e,JOIN,(L[i],L[i+1]));assert h.closure(plan.identity) is None;trows.append((plan,surface))
    for plan,_ in trows[:3]:q.materialize(plan)
    qshape=q.shape_digest(trows[0][0]);qn1=q.recipes[q.by_shape[qshape]]
    checks['transient_plans_nominate_n1']=len(qn1.witness_closures)==3 and qshape not in q.pressure and qn1.rank==1 and not qn1.active and all(h.closure(plan.identity) is None for plan,_ in trows[:3])
    for plan,_ in trows[3:5]:q.materialize(plan)
    checks['transient_plans_earn_n1']=qn1.active and qn1.probation_passes==PROBATION_PASSES
    qout=q.materialize(trows[5][0])
    checks['transient_heldout_pair_selected_through_n1']=qout==trows[5][1] and q.last_recipe==qn1.identity and h.closure(trows[5][0].identity) is None
    more=[]
    for i in range(13,41,2):
        plan,surface=rematerialize_transient_plan(e,JOIN,(L[i],L[i+1]));q.materialize(plan);more.append((plan,surface))
    wpairs=[wrap(plan,surface) for plan,surface in (*trows,*more)];qsupers=[]
    for i in range(0,20,2):
        plan,surface=rematerialize_transient_plan(e,JOIN,(wpairs[i],wpairs[i+1]));assert h.closure(plan.identity) is None;qsupers.append((plan,surface))
    for left,right in ((0,2),(1,3)):
        plan,surface=rematerialize_transient_plan(e,JOIN,(wpairs[left],wpairs[right]));assert h.closure(plan.identity) is None;qsupers.append((plan,surface))
    for plan,_ in qsupers[:3]:q.materialize(plan)
    qsshape=q.shape_digest(qsupers[0][0]);qn2=q.recipes[q.by_shape[qsshape]]
    checks['transient_superpairs_nominate_n2']=qn2.rank==2 and not qn2.active and sum(op.kind==OP_CALL and op.reference_identity==qn1.identity for op in qn2.ops)==2
    for plan,_ in qsupers[3:5]:q.materialize(plan)
    checks['transient_superpairs_earn_n2']=qn2.active and qn2.probation_passes==PROBATION_PASSES
    q2out=q.materialize(qsupers[5][0])
    checks['transient_heldout_superpair_selected_through_n2']=q2out==qsupers[5][1] and q.last_recipe==qn2.identity and q.last_rank==2 and h.closure(qsupers[5][0].identity) is None
    for plan,_ in qsupers[6:]:
        q.materialize(plan);assert q.last_recipe==qn2.identity
    wqsupers=[wrap(plan,surface) for plan,surface in qsupers];qthird=[]
    for i in range(0,12,2):
        plan,surface=rematerialize_transient_plan(e,JOIN,(wqsupers[i],wqsupers[i+1]));assert h.closure(plan.identity) is None;qthird.append((plan,surface))
    for plan,_ in qthird[:3]:q.materialize(plan)
    qtshape=q.shape_digest(qthird[0][0]);qn3=q.recipes[q.by_shape[qtshape]]
    checks['transient_thirds_nominate_n3']=qn3.rank==3 and not qn3.active and sum(op.kind==OP_CALL and op.reference_identity==qn2.identity for op in qn3.ops)==2
    for plan,_ in qthird[3:5]:q.materialize(plan)
    checks['transient_thirds_earn_n3']=qn3.active and qn3.probation_passes==PROBATION_PASSES
    q3out=q.materialize(qthird[5][0])
    checks['transient_heldout_third_selected_through_n3']=q3out==qthird[5][1] and q.last_recipe==qn3.identity and q.last_rank==3 and h.closure(qthird[5][0].identity) is None
    le,lh,lL=lexicon();teach_span_language(le,lL)
    checks['language_span_does_not_write_template_witnesses']=le.span_template(JOIN,2) is not None and not hasattr(lh,'_template_witnesses')
    lp=ParametricCondensationV1(lh);lrows=[]
    for i in range(1,13,2):
        plan,surface=rematerialize_transient_plan(le,JOIN,(lL[i],lL[i+1]));assert lh.closure(plan.identity) is None;lrows.append((plan,surface))
    for plan,_ in lrows[:3]:lp.materialize(plan)
    lshape=lp.shape_digest(lrows[0][0]);ln1=lp.recipes[lp.by_shape[lshape]]
    for plan,_ in lrows[3:5]:lp.materialize(plan)
    lout=lp.materialize(lrows[5][0])
    checks['language_only_span_earns_n1']=ln1.active and lout==lrows[5][1] and lp.last_recipe==ln1.identity and not hasattr(lh,'_template_witnesses')
    checks['visible_discussion_improvement']=checks['language_only_span_earns_n1'] and checks['transient_heldout_third_selected_through_n3']

    # N+2 witnesses are rematerialized plans whose children have the already-earned N+1 shape.
    supers=[superpair(e,pairs[i],pairs[i+1]) for i in range(0,11)]
    for row in supers:p._remember(row)
    n1.active=0;raw_super,raw_super_touches=p._original(supers[0]);n1.active=1
    n1_uses=n1.uses;first_super_out=p.materialize(supers[0]);sshape=p.shape_digest(supers[0])
    checks['schema_congruent_super_uses_n1_before_n2']=(
        first_super_out==raw_super==supers[0].surface and p.last_recipe==0
        and sshape not in p.by_shape and len(p.pressure.get(sshape,()))==1
        and n1.active and n1.uses==n1_uses+2 and p.last_touches<raw_super_touches)
    p.materialize(supers[1])
    n1_uses_loaded=n1.uses
    n1.active=0;raw_loaded,raw_loaded_touches=p._original(supers[2]);n1.active=1
    loaded_out,loaded_touches=p._original(supers[2])
    checks['index_load_delays_schema_use']=(
        loaded_out==raw_loaded==supers[2].surface and n1.uses==n1_uses_loaded
        and loaded_touches==raw_loaded_touches and len(p.pressure.get(sshape,()))==2)
    held_under_load=p.materialize(pairs[5])
    checks['index_load_does_not_erase_n1']=(
        held_under_load==pairs[5].surface and p.last_recipe==n1.identity and n1.active)
    p.materialize(supers[2])
    n2=p.recipes[p.by_shape[sshape]]
    checks['recurrent_networks_nominate_n2']=n2.rank==2 and not n2.active and sum(op.kind==OP_CALL and op.reference_identity==n1.identity for op in n2.ops)==2
    n1_uses_recovered=n1.uses
    recovered_out=p.materialize(supers[3])
    checks['schema_use_recovers_when_index_load_pops']=(
        recovered_out==supers[3].surface and n1.uses==n1_uses_recovered+2
        and sshape not in p.pressure and not n2.active)
    p.materialize(supers[4])
    checks['heldout_shadow_probation_earns_n2']=n2.active and n2.probation_passes==PROBATION_PASSES
    original2,orig2_touches=p._original(supers[5]);out2=p.materialize(supers[5]);n2_touches=p.last_touches
    checks['heldout_recursive_closure_executes_through_n2']=out2==original2 and p.last_recipe==n2.identity and p.last_rank==2
    checks['n2_calls_n1_and_reduces_work']=n2_touches<raw_super_touches

    # N+3 witnesses call the already-earned N+2 relation twice; rank is acquired,
    # not a fixed depth or grammar role installed by the assay.
    thirds=[superpair(e,supers[i],supers[i+1]) for i in range(6)]
    for row in thirds:p._remember(row)
    n1.active=0;n2.active=0;raw_third,raw_third_touches=p._original(thirds[0]);n1.active=1;n2.active=1
    n2_uses=n2.uses;first_third_out=p.materialize(thirds[0]);tshape=p.shape_digest(thirds[0])
    checks['schema_congruent_third_uses_n2_before_n3']=(
        first_third_out==raw_third==thirds[0].surface and p.last_recipe==0
        and tshape not in p.by_shape and len(p.pressure.get(tshape,()))==1
        and n2.active and n2.uses==n2_uses+2 and p.last_touches<raw_third_touches)
    for row in thirds[1:3]:p.materialize(row)
    n3=p.recipes[p.by_shape[tshape]]
    checks['recurrent_networks_nominate_n3']=n3.rank==3 and not n3.active and sum(op.kind==OP_CALL and op.reference_identity==n2.identity for op in n3.ops)==2
    for row in thirds[3:5]:p.materialize(row)
    checks['heldout_shadow_probation_earns_n3']=n3.active and n3.probation_passes==PROBATION_PASSES
    n1.active=0;n2.active=0;raw3,raw3_touches=p._original(thirds[5]);n1.active=1;n2.active=1
    original3,orig3_touches=p._original(thirds[5]);out3=p.materialize(thirds[5]);n3_touches=p.last_touches
    checks['heldout_recursive_closure_executes_through_n3']=out3==original3==raw3 and p.last_recipe==n3.identity and p.last_rank==3
    checks['n3_calls_n2_and_reduces_work']=n3_touches<raw3_touches

    # Exact checkpoint retains Recipes/evidence but not their deterministic hot index.
    ecp=copy.deepcopy(e.checkpoint());hcp=copy.deepcopy(h.checkpoint());pcp=copy.deepcopy(p.checkpoint())
    e2=LearnedSurfaceEcologyV1.restore(ecp);h2=HierarchicalConstructionV1.restore(e2,hcp);p2=ParametricCondensationV1.restore(h2,pcp)
    checkpoint_exact=p2.checkpoint()==pcp
    def _leaf2(i):return h2.closure(L[i].identity)
    rp5r=wrap(*rematerialize_transient_plan(e2,JOIN,(_leaf2(11),_leaf2(12))))
    rp6r=wrap(*rematerialize_transient_plan(e2,JOIN,(_leaf2(13),_leaf2(14))))
    rp7r=wrap(*rematerialize_transient_plan(e2,JOIN,(_leaf2(15),_leaf2(16))))
    rs5r=wrap(*rematerialize_transient_plan(e2,JOIN,(rp5r,rp6r)))
    rs6r=wrap(*rematerialize_transient_plan(e2,JOIN,(rp6r,rp7r)))
    rt5r=wrap(*rematerialize_transient_plan(e2,JOIN,(rs5r,rs6r)))
    for node in (rp5r,rp6r,rp7r,rs5r,rs6r):p2._remember(node)
    rout=p2.materialize(rt5r)
    node_ids={int(row['identity']) for row in pcp.get('nodes',())}
    witness_ids={int(i) for r in p.recipes.values() for i in r.witness_closures}
    checks['checkpoint_preserves_parametric_n3']=checkpoint_exact and rout==original3 and p2.last_rank==3 and h2.closure(thirds[5].identity) is None and thirds[5].identity not in node_ids
    checks['checkpoint_nodes_are_witness_subtrees']=('nodes' in pcp and thirds[5].identity not in node_ids and pairs[5].identity not in node_ids and witness_ids<=node_ids and len(pcp['nodes'])<len(p.nodes))
    checks['checkpoint_pressure_omits_nominated_shapes']=('pressure' not in pcp and not p.pressure and all(len(r.witness_closures)==3 for r in p.recipes.values()))
    checks['checkpoint_omits_derived_shape_index']='by_shape' not in pcp
    legacy=copy.deepcopy(pcp);legacy['by_shape']=dict(sorted(p.by_shape.items()))
    compact_bytes=len(json.dumps(pcp,sort_keys=True,separators=(',',':')));legacy_bytes=len(json.dumps(legacy,sort_keys=True,separators=(',',':')))
    checks['derived_shape_index_bytes_deleted']=legacy_bytes>compact_bytes
    forged=copy.deepcopy(pcp);forged['by_shape']={'forged-shape':n1.identity}
    try:
        e3=LearnedSurfaceEcologyV1.restore(ecp);h3=HierarchicalConstructionV1.restore(e3,hcp);p3=ParametricCondensationV1.restore(h3,forged)
        forged_ignored=p3.by_shape==p.by_shape and p3.checkpoint()==pcp
    except Exception:
        forged_ignored=False
    checks['forged_shape_index_has_no_checkpoint_authority']=forged_ignored

    retired=h.retire_unreferenced_composites()
    rp5,rs5,rt5=pairs[5],supers[5],thirds[5]
    checks['persist_bank_nominates_without_compose']=(
        retired==0 and all(h.closure(row.identity) is None for row in (*pairs,*supers,*thirds))
        and all(node.depth==0 for node in h._closures.values())
        and p.materialize(rt5)==original3 and p.last_recipe==n3.identity and p.last_rank==3)
    we,wh,wL=lexicon();teach_span_language(we,wL);w=ParametricCondensationV1(wh)
    wpairs=[pair(we,wL,i,i+1) for i in range(1,13,2)]
    for row in wpairs:w._remember(row)
    for row in wpairs[:5]:w.materialize(row)
    wn1=w.recipes[w.by_shape[w.shape_digest(wpairs[0])]]
    wsupers=[superpair(we,wpairs[i],wpairs[i+1]) for i in range(0,5)]
    for row in wsupers:w._remember(row)
    for row in wsupers[:2]:w.materialize(row)
    wshape=w.shape_digest(wsupers[0]);before_recipes=len(w.recipes)
    for source in (5002,6001,6002):we.withdraw_source(source)
    w.materialize(wpairs[5])
    w.materialize(wsupers[2])
    checks['withdrawn_relation_does_not_nominate_under_load']=(
        not wn1.active and wshape not in w.by_shape and len(w.recipes)==before_recipes
        and len(w.pressure.get(wshape,()))>=2)
    wsnap=(copy.deepcopy(we.checkpoint()),copy.deepcopy(wh.checkpoint()),copy.deepcopy(w.checkpoint()))
    we2=LearnedSurfaceEcologyV1.restore(wsnap[0]);wh2=HierarchicalConstructionV1.restore(we2,wsnap[1]);w2=ParametricCondensationV1.restore(wh2,wsnap[2])
    wr2=w2.recipes[wn1.identity]
    checks['withdrawn_load_survives_checkpoint']=(
        not wr2.active and wshape not in w2.by_shape and len(w2.pressure.get(wshape,()))>=2)
    forged=copy.deepcopy(wsnap[2])
    for row in forged.get('recipes',()):
        row['active']=1
    try:
        ParametricCondensationV1.restore(wh2,forged);forged_refused=False
    except HierarchicalRefuse:
        forged_refused=True
    checks['forged_active_recipe_without_relation_refused']=forged_refused
    def cleaf(i):return wh2.closure(wL[i].identity)
    for source,(a,b) in zip((8101,8102,9101,9102),((1,2),(3,4),(1,2),(3,4))):
        assert we2.observe_span(JOIN,(cleaf(a).surface,cleaf(b).surface),cleaf(a).surface+u(' then ')+cleaf(b).surface,source)
    rpair=wrap(*rematerialize_transient_plan(we2,JOIN,(cleaf(11),cleaf(12))));w2._remember(rpair)
    for _ in range(PROBATION_PASSES):w2.materialize(rpair)
    w2.materialize(w2._node(wsupers[2].identity));wn2r=w2.recipes.get(w2.by_shape.get(wshape,0))
    checks['reacquired_after_rest_nominates_deferred_n2']=(
        wr2.active and wn2r is not None and wn2r.rank==2 and not wn2r.active
        and sum(op.kind==OP_CALL and op.reference_identity==wr2.identity for op in wn2r.ops)==2)
    for source,(a,b) in zip((8001,8002,9001,9002),((1,2),(3,4),(1,2),(3,4))):
        assert we.observe_span(JOIN,(wL[a].surface,wL[b].surface),wL[a].surface+u(' then ')+wL[b].surface,source)
    for _ in range(PROBATION_PASSES):w.materialize(wpairs[5])
    w.materialize(wsupers[2]);wn2=w.recipes.get(w.by_shape.get(wshape,0))
    checks['reacquired_relation_nominates_deferred_n2']=(
        wn1.active and wn2 is not None and wn2.rank==2 and not wn2.active
        and sum(op.kind==OP_CALL and op.reference_identity==wn1.identity for op in wn2.ops)==2)
    checks['visible_discussion_improvement']=checks['forged_active_recipe_without_relation_refused'] and checks['reacquired_after_rest_nominates_deferred_n2'] and checks['withdrawn_load_survives_checkpoint'] and checks['reacquired_relation_nominates_deferred_n2'] and checks['withdrawn_relation_does_not_nominate_under_load'] and checks['index_load_survives_checkpoint'] and checks['schema_use_recovers_when_index_load_pops'] and checks['index_load_does_not_erase_n1'] and checks['index_load_delays_schema_use'] and checks['schema_congruent_third_uses_n2_before_n3'] and checks['schema_congruent_super_uses_n1_before_n2'] and checks['interrupted_n2_resumes_into_n3'] and checks['delayed_n2_does_not_erase_n1'] and checks['resumed_cluster_earns_recursive_n2'] and checks['matched_supers_without_pair_history_are_not_n2'] and checks['incomplete_cluster_survives_checkpoint'] and checks['matched_third_without_history_does_not_nominate'] and checks['checkpoint_pressure_omits_nominated_shapes'] and checks['checkpoint_nodes_are_witness_subtrees'] and checks['persist_bank_nominates_without_compose'] and checks['language_only_span_earns_n1'] and checks['transient_heldout_third_selected_through_n3']

    # Shape mismatch does not recruit a compact Recipe just because some bytes overlap.
    leaf_out=p.materialize(L[33])
    checks['different_shape_does_not_alias_recipe']=leaf_out==L[33].surface and p.last_recipe==0

    # Contradict the learned JOIN relation with an equally supported alternative.
    conflict_sources=(7001,7002,7003,7004)
    for source in conflict_sources:
        assert h.observe(JOIN,(L[1],L[2]),L[1].surface+u(' / ')+L[2].surface,source)
    before1=n1.deoptimizations;before2=n2.deoptimizations;before3=n3.deoptimizations
    fallback3=p.materialize(rt5);fallback2=p.materialize(rs5);p.materialize(rp5)
    checks['template_contradiction_deoptimizes_recursive_shortcuts']=fallback3==original3 and fallback2==original2 and not n3.active and not n2.active and not n1.active and n3.deoptimizations==before3+1 and n2.deoptimizations==before2+1 and n1.deoptimizations==before1+1
    checks['contradiction_preserves_lived_closure']=tuple(rs5.surface)==original2

    # Remove only the contradictory source evidence; all levels shadow-prove again.
    for source in conflict_sources:e.withdraw_source(source)
    for _ in range(PROBATION_PASSES):p.materialize(rp5)
    for _ in range(PROBATION_PASSES):p.materialize(rs5)
    for _ in range(PROBATION_PASSES):p.materialize(rt5)
    checks['reopen_requires_reprobation']=n1.active and n2.active and n3.active
    p.materialize(rt5);checks['reopened_n3_selected']=p.last_recipe==n3.identity and p.last_rank==3

    # Withdraw enough original JOIN evidence to remove the current relation; shortcuts deopt.
    for source in (5002,6001,6002):e.withdraw_source(source)
    before=n2.deoptimizations;out=p.materialize(rs5)
    checks['owning_relation_withdrawal_deoptimizes']=out==original2 and not n2.active and n2.deoptimizations==before+1

    q={'recipes':len(p.recipes),'n1_ops':len(n1.ops),'n2_ops':len(n2.ops),'n3_ops':len(n3.ops),'n1_uses':n1.uses,'n2_uses':n2.uses,'n3_uses':n3.uses,'n1_touches':n1_touches,'n1_original_touches':orig1_touches,'n2_touches':n2_touches,'n2_original_touches':raw_super_touches,'n3_touches':n3_touches,'n3_original_touches':raw3_touches,'closures':h.closure_count,'retired_composites':retired,'max_depth':h.max_depth_seen,'lived_nodes':len(p.nodes),'checkpoint_nodes':len(pcp.get('nodes',())),'checkpoint_pressure_shapes':len(pcp.get('pressure',{})),'legacy_checkpoint_bytes':legacy_bytes,'compact_checkpoint_bytes':compact_bytes,'shape_index_bytes_deleted':legacy_bytes-compact_bytes}
    checks['bounded_parameterized_state']=q['recipes']==3 and q['n1_ops']<16 and q['n2_ops']<16 and q['n3_ops']<16
    checks['no_semantic_grammar_or_runtime_llm']=not any(hasattr(p,x) for x in ('grammar','parse','answer','think','reward','dopamine'))
    elapsed=(time.perf_counter()-started)*1000;checks['rapid_foundry_runtime']=elapsed<1500
    report={'schema':'cyber-lagoon.reference-parametric-condensation.v2','pass':all(checks.values()),'checks':checks,'quantity':q,'heldout_n1_surface':bytes(out1).decode(),'heldout_n2_surface':bytes(out2).decode(),'heldout_n3_surface':bytes(out3).decode(),'heldout_transient_n3_surface':bytes(q3out).decode(),'n1_identity':n1.identity,'n2_identity':n2.identity,'n3_identity':n3.identity,'elapsed_ms':round(elapsed,3),'reference_only':True,'physical_direct_parity':'NOT_RUN/RED','runtime_llm':False,'claim':'PARAMETRIC_N_PLUS_ONE_THROUGH_N_PLUS_THREE_LANGUAGE_CONDENSATION_REFERENCE_PROPERTY_ONLY'}
    print('FOUNDRY_PARAMETRIC_CONDENSATION '+('GREEN' if report['pass'] else 'RED')+f" n1={int(checks['heldout_fillers_execute_through_n1'])} n2={int(checks['heldout_recursive_closure_executes_through_n2'])} n3={int(checks['heldout_recursive_closure_executes_through_n3'])} transient_n3={int(checks['transient_heldout_third_selected_through_n3'])} index_bytes_deleted={q['shape_index_bytes_deleted']} deopt={int(checks['template_contradiction_deoptimizes_recursive_shortcuts'])} ms={report['elapsed_ms']}")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if report['pass'] else 1)
if __name__=='__main__':main()
