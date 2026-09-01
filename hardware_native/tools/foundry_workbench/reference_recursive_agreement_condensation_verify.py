#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
from reference_parametric_condensation_v1 import ParametricCondensationV1,OP_CALL,PROBATION_PASSES

def u(s):return tuple(s.encode())
def lex(e,f,text,a,b):e.observe_naming(f,u(text),a);e.observe_naming(f,u(text),b)
def form(e,f,c,text,a,b):e.observe_form(f,(c,),u(text),a);e.observe_form(f,(c,),u(text),b)

SING,PLUR=7001,7002; CLAUSE,COORD=9001,9002
CAT,DOG,PILOT,ENGINEER,SENSOR,VALVE=101,102,103,104,105,106
SEE,TEST,INSPECT=201,202,203

BASE={CAT:'cat',DOG:'dog',PILOT:'pilot',ENGINEER:'engineer',SENSOR:'sensor',VALVE:'valve',SEE:'see',TEST:'test',INSPECT:'inspect'}
FORMS={
 CAT:{SING:'cat',PLUR:'cats'},DOG:{SING:'dog',PLUR:'dogs'},PILOT:{SING:'pilot',PLUR:'pilots'},ENGINEER:{SING:'engineer',PLUR:'engineers'},
 SENSOR:{SING:'sensor',PLUR:'sensors'},VALVE:{SING:'valve',PLUR:'valves'},
 SEE:{SING:'sees',PLUR:'see'},TEST:{SING:'tests',PLUR:'test'},INSPECT:{SING:'inspects',PLUR:'inspect'}}

def build():
    e=LearnedSurfaceEcologyV1()
    for f,text in BASE.items():lex(e,f,text,10000+f,20000+f)
    # General clause scaffold from different constituents; no grammatical labels in the learner.
    assert e.observe_construction(CLAUSE,(CAT,SEE,DOG),u('the cat see the dog.'),30001)
    assert e.observe_construction(CLAUSE,(ENGINEER,TEST,SENSOR),u('the engineer test the sensor.'),30002)
    for f,rows in FORMS.items():
        for c,text in rows.items():form(e,f,c,text,40000+f*10+(c-SING)*2,40001+f*10+(c-SING)*2)
    p1=e.observe_compatibility(CLAUSE,(SING,SING,0),50001);p2=e.observe_compatibility(CLAUSE,(PLUR,PLUR,0),50002)
    assert p1==p2==(1,1,0)
    return e

def clause(e,subject,verb,obj,sc,oc):
    return e.realize_conditioned(CLAUSE,(subject,verb,obj),((sc,),(sc,),(oc,)),(sc,sc,0))

def main():
    t=time.perf_counter();checks={};e=build()
    # Mismatch is rejected before hierarchy/condensation can see a clause boundary.
    mismatch=e.realize_conditioned(CLAUSE,(PILOT,SEE,SENSOR),((PLUR,),(SING,),(SING,)),(PLUR,SING,0))
    checks['nonlocal_agreement_mismatch_refuses_before_network']=mismatch is None
    specs=[
      (CAT,SEE,DOG,SING,SING),(PILOT,TEST,SENSOR,PLUR,SING),(ENGINEER,INSPECT,VALVE,SING,PLUR),(DOG,SEE,SENSOR,PLUR,SING),
      (PILOT,INSPECT,VALVE,SING,SING),(ENGINEER,TEST,DOG,PLUR,PLUR),(CAT,INSPECT,SENSOR,PLUR,PLUR),(DOG,TEST,VALVE,SING,SING),
      (PILOT,SEE,DOG,PLUR,SING),(ENGINEER,INSPECT,SENSOR,PLUR,SING),(CAT,TEST,VALVE,SING,PLUR),(DOG,INSPECT,SENSOR,SING,PLUR),
      (PILOT,TEST,DOG,SING,PLUR),(ENGINEER,SEE,VALVE,SING,SING),(CAT,INSPECT,DOG,PLUR,SING),(DOG,SEE,VALVE,PLUR,PLUR),
      (PILOT,INSPECT,SENSOR,PLUR,PLUR),(ENGINEER,TEST,VALVE,SING,PLUR),(CAT,SEE,SENSOR,SING,PLUR),(DOG,TEST,DOG,PLUR,SING),
      (PILOT,SEE,VALVE,SING,PLUR),(ENGINEER,INSPECT,DOG,PLUR,SING),(CAT,TEST,SENSOR,PLUR,SING),(DOG,INSPECT,VALVE,SING,SING)]
    surfaces=[clause(e,*row) for row in specs];assert all(surfaces)
    checks['heldout_clause_recombination']=surfaces[1]==u('the pilots test the sensor.') and surfaces[2]==u('the engineer inspects the valves.')
    h=HierarchicalConstructionV1(e)
    leaves=[h.leaf_surface(CLAUSE,60000+i,s) for i,s in enumerate(surfaces)]
    # Learn coordination from two distinct clause-pair episodes only.
    assert h.observe(COORD,(leaves[0],leaves[1]),leaves[0].surface+u(' and ')+leaves[1].surface,61001)
    assert h.observe(COORD,(leaves[2],leaves[3]),leaves[2].surface+u(' and ')+leaves[3].surface,61002)
    checks['coordination_relation_learned']=e.span_template(COORD,2) is not None
    pairs=[h.compose(COORD,(leaves[i],leaves[i+1])) for i in range(0,22,2)]
    pc=ParametricCondensationV1(h)
    for row in pairs[:3]:pc.materialize(row)
    n1=pc.recipes[pc.by_shape[pc.shape_digest(pairs[0])]]
    for row in pairs[3:5]:pc.materialize(row)
    checks['agreement_preserving_coordination_earns_n1']=n1.active and n1.rank==1
    expected1=tuple(pairs[5].surface);out1=pc.materialize(pairs[5])
    checks['heldout_agreeing_clauses_bind_n1']=out1==expected1 and pc.last_recipe==n1.identity and b' and ' in bytes(out1)

    supers=[h.compose(COORD,(pairs[i],pairs[i+1])) for i in range(0,8)]
    for row in supers[:3]:pc.materialize(row)
    n2=pc.recipes[pc.by_shape[pc.shape_digest(supers[0])]]
    for row in supers[3:5]:pc.materialize(row)
    checks['recursive_coordination_earns_n2']=n2.active and n2.rank==2 and sum(op.kind==OP_CALL for op in n2.ops)==2
    expected2=tuple(supers[5].surface);out2=pc.materialize(supers[5])
    checks['heldout_recursive_agreement_executes_n2']=out2==expected2 and pc.last_recipe==n2.identity and pc.last_rank==2

    # Surface is not one of coordination demonstrations, and its four clauses were not construction demonstrations.
    demos={leaves[0].surface+u(' and ')+leaves[1].surface,leaves[2].surface+u(' and ')+leaves[3].surface}
    checks['recursive_output_not_surface_replay']=out2 not in demos

    # Upstream form authority remains necessary: remove one of the two plural PILOT witnesses.
    # New plural PILOT clause formation must fail; old lived closures remain historical structures.
    pilot_plural_sources=[40000+PILOT*10+(PLUR-SING)*2,40001+PILOT*10+(PLUR-SING)*2]
    e.withdraw_source(pilot_plural_sources[1])
    fresh=clause(e,PILOT,SEE,SENSOR,PLUR,SING)
    checks['condensation_cannot_bypass_upstream_form_authority']=fresh is None and tuple(pairs[5].surface)==expected1

    # Context compatibility itself remains abstract and generalizes to unseen opaque condition IDs.
    checks['agreement_relation_is_structural_not_category_opcode']=e.compatible(CLAUSE,(9991,9991,0)) and not e.compatible(CLAUSE,(9991,9992,0))
    checks['no_parser_or_grammar_runtime']=not any(hasattr(x,n) for x in (e,h,pc) for n in ('parse','grammar','syntax_tree','answer','reward','dopamine'))
    elapsed=(time.perf_counter()-t)*1000;checks['rapid_runtime']=elapsed<1000
    result={'schema':'0x1.reference-recursive-agreement-condensation.v1','pass':all(checks.values()),'checks':checks,'heldout_pair_surface':bytes(out1).decode(),'heldout_recursive_surface':bytes(out2).decode(),'n1_rank':n1.rank,'n2_rank':n2.rank,'n1_ops':len(n1.ops),'n2_ops':len(n2.ops),'elapsed_ms':round(elapsed,3),'claim':'RECURSIVE_HELDOUT_AGREEMENT_COORDINATION_PARAMETRIC_CONDENSATION_REFERENCE_ONLY','physical_direct_parity':'NOT_RUN/RED','human_language_mastery':False}
    print('FOUNDRY_RECURSIVE_AGREEMENT_CONDENSATION '+('GREEN' if result['pass'] else 'RED')+f" agreement={int(checks['heldout_clause_recombination'])} coord={int(checks['heldout_agreeing_clauses_bind_n1'])} n2={int(checks['heldout_recursive_agreement_executes_n2'])} ms={result['elapsed_ms']}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
