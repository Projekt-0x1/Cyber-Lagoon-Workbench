#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1,CONDENSE_PROBATION_PASSES


def u(s):return tuple(s.encode())


def fixture(depth=20):
    e=LearnedSurfaceEcologyV1();A,B=1,2;CLAUSE,JOIN=10,11
    for feature,text in ((A,'alpha'),(B,'beta')):
        e.observe_naming(feature,u(text),100+feature);e.observe_naming(feature,u(text),200+feature)
    e.observe_construction(CLAUSE,(A,),u('alpha'),301);e.observe_construction(CLAUSE,(B,),u('beta'),302)
    h=HierarchicalConstructionV1(e);a=h.leaf(CLAUSE,(A,));b=h.leaf(CLAUSE,(B,))
    h.observe(JOIN,(a,b),u('alpha beta'),401);h.observe(JOIN,(b,a),u('beta alpha'),402)
    cur=a
    for _ in range(depth):cur=h.compose(JOIN,(cur,b))
    h.drop_hot_cache()
    return e,h,a,b,cur,JOIN


def cold(h,closure):
    h.drop_hot_cache();surface=closure.surface
    return surface,h.last_materialization_touches,h.last_condensed_recipe


def main():
    started=time.perf_counter();checks={};e,h,a,b,deep,JOIN=fixture();expected=u('alpha'+' beta'*20);ancestry=deep.ancestry
    baseline=[]
    for _ in range(3):
        surface,touches,selected=cold(h,deep);baseline.append(touches)
        checks.setdefault('pre_nomination_exact',surface==expected and selected==0)
    recipe=h._condensed.get(deep.identity)
    checks['repeated_expensive_use_nominates']=recipe is not None and recipe.active==0 and h._condense_evidence[deep.identity][0]>=3
    checks['candidate_not_output_cache']=recipe is not None and all(op.kind in (1,2) for op in recipe.ops) and tuple(expected) not in tuple(getattr(op,'reference_identity',0) for op in recipe.ops)
    checks['candidate_has_exact_structural_dependencies']=recipe is not None and recipe.dependencies and recipe.ancestry_digest==h._ancestry_digest(ancestry)

    shadow=[]
    for index in range(CONDENSE_PROBATION_PASSES):
        surface,touches,selected=cold(h,deep);shadow.append(touches)
        checks[f'probation_{index+1}_exact']=surface==expected and selected==0
    checks['probation_earns_activation']=recipe is not None and recipe.active==1 and recipe.probation_passes==CONDENSE_PROBATION_PASSES

    surface,condensed_touches,selected=cold(h,deep)
    checks['active_n_plus_one_selected']=surface==expected and selected==recipe.identity and recipe.uses>=CONDENSE_PROBATION_PASSES+1
    checks['condensed_reduces_touched_work']=condensed_touches<min(baseline) and condensed_touches*2<min(baseline)
    checks['ancestry_equivalence_preserved']=deep.ancestry==ancestry and h._ancestry_digest(deep.ancestry)==recipe.ancestry_digest

    cp=copy.deepcopy(h.checkpoint());e2=LearnedSurfaceEcologyV1.restore(e.checkpoint());r=HierarchicalConstructionV1.restore(e2,cp);rr=r._condensed.get(deep.identity);restored=r.closure(deep.identity)
    checkpoint_exact=r.checkpoint()==cp and r.digest()==h.digest()
    rs,rt,ri=cold(r,restored)
    checks['checkpoint_preserves_earned_recipe']=checkpoint_exact and rr is not None and rr.active==1 and rs==expected and ri==rr.identity and rt==condensed_touches

    # Contradictory construction evidence creates an equal supported alternative.
    # This does not rewrite the lived closure; it removes authority for the N+1
    # shortcut, which must deopt and fall back to the structural witness graph.
    e.observe_span(JOIN,(a.surface,b.surface),u('alpha / beta'),403)
    e.observe_span(JOIN,(b.surface,a.surface),u('beta / alpha'),404)
    before_deopt=recipe.deoptimizations
    surface,conflict_touches,selected=cold(h,deep)
    checks['contradiction_deoptimizes']=surface==expected and selected==0 and recipe.active==0 and recipe.deoptimizations==before_deopt+1 and conflict_touches>=min(baseline)
    checks['contradiction_does_not_erase_lived_closure']=deep.identity in h._closures and deep.surface==expected

    # Remove the contradictory sources. Two new shadow-equivalence passes are
    # required before the shortcut can reactivate; restoration is not a host flip.
    e.withdraw_source(403);e.withdraw_source(404)
    for _ in range(CONDENSE_PROBATION_PASSES):cold(h,deep)
    checks['withdrawal_reopens_then_reearns']=recipe.active==1
    _,_,selected=cold(h,deep);checks['reearned_candidate_selected']=selected==recipe.identity

    # Withdrawing one of the original construction witnesses independently strips
    # current source authority from the condensed candidate while retaining the
    # exact historical closure and its frozen witness.
    e.withdraw_source(402);before=recipe.deoptimizations
    surface,withdraw_touches,selected=cold(h,deep)
    checks['owning_source_withdrawal_deoptimizes']=surface==expected and selected==0 and recipe.active==0 and recipe.deoptimizations==before+1 and withdraw_touches>=min(baseline)
    checks['no_credit_minted_by_optimization']=not hasattr(recipe,'credit') and not hasattr(recipe,'reward') and not hasattr(recipe,'effect')

    q=h.persistent_quantity()
    out={'schema':'0x1.reference-constructor-condensation.v1','pass':all(checks.values()),'checks':checks,'closure_depth':deep.depth,'closure_bytes':len(expected),'baseline_cold_touches':baseline,'shadow_touches':shadow,'condensed_touches':condensed_touches,'conflict_fallback_touches':conflict_touches,'withdraw_fallback_touches':withdraw_touches,'recipe_identity':0 if recipe is None else recipe.identity,'recipe_ops':0 if recipe is None else len(recipe.ops),'recipe_dependencies':0 if recipe is None else len(recipe.dependencies),'recipe_uses':0 if recipe is None else recipe.uses,'deoptimizations':0 if recipe is None else recipe.deoptimizations,'persistent_quantity':q,'runtime_llm':False,'semantic_summary_cache':False,'causal_credit_minted':False,'physical_direct_parity':'NOT_RUN/RED','claim':'RESIDENT_CONSTRUCTOR_CONDENSATION_AND_DEOPTIMIZATION_REFERENCE_PROPERTY_ONLY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_CONSTRUCTOR_CONDENSATION '+('GREEN' if out['pass'] else 'RED')+f" nominate={int(checks['repeated_expensive_use_nominates'])} probation={int(checks['probation_earns_activation'])} cheaper={int(checks['condensed_reduces_touched_work'])} deopt={int(checks['contradiction_deoptimizes'])} direct_parity=RED")
    print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)


if __name__=='__main__':main()
