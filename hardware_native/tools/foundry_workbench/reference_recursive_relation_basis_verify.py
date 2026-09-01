#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_recursive_relation_basis_v1 import *
from reference_resident_variable_depth_endogenous_unfolding_v1 import Q

def chain(offset=0):
    b=RecursiveRelationBasisV1()
    for i in range(1,9):b.observe_primitive(offset+i,offset+i+1,(Q,i*Q),offset+100+i)
    return b

def main():
    t=time.perf_counter();checks={};cold=chain();base=cold.resolve(1,9);base_boundary=base.boundary_q16
    hot=RecursiveRelationBasisV1.restore(copy.deepcopy(cold.checkpoint()));earned=[]
    for _ in range(7):earned.append(hot.silent_wave())
    best=hot.best_derived();fast=hot.resolve(1,9)
    checks['silent_closure_recursively_reenters_same_relation_basis']=(all(earned) and hot.generation(best)>=3 and hot.expand_spaces(best)==tuple(range(1,10)))
    checks['retained_basis_changes_later_work_not_answer']=(base_boundary==fast.boundary_q16 and base.relation_touches==8 and fast.relation_touches==1)
    checks['derived_relation_carries_all_premise_roots_but_no_direct_evidence']=(not hot.relations[best].direct_evidence and hot.relations[best].root_evidence==tuple(range(101,109)))
    lesioned=RecursiveRelationBasisV1.restore(copy.deepcopy(hot.checkpoint()));lesioned.remove_derived(best);slower=lesioned.resolve(1,9)
    checks['highest_abstraction_lesion_restores_slower_lower_basis']=(slower is not None and 1<slower.relation_touches<base.relation_touches)
    primitive=RecursiveRelationBasisV1.restore(copy.deepcopy(hot.checkpoint()))
    for rid in tuple(sorted(primitive.relations)):
        if rid in primitive.relations and primitive.relations[rid].derived:primitive.remove_derived(rid)
    raw=primitive.resolve(1,9)
    checks['all_abstraction_lesion_restores_primitive_competence']=(raw is not None and raw.boundary_q16==base_boundary and raw.relation_touches==base.relation_touches)
    cut=RecursiveRelationBasisV1.restore(copy.deepcopy(hot.checkpoint()));cut.withdraw_evidence(104)
    checks['primitive_withdrawal_shatters_exact_dependent_descendants']=cut.resolve(1,9) is None and best not in cut.active

    alt=RecursiveRelationBasisV1()
    for l,r,e in ((20,21,201),(21,22,202),(20,23,203),(23,22,204)):alt.observe_primitive(l,r,(Q,0),e)
    closure=alt.resolve(20,22);arid=alt.retain(closure);alt.withdraw_evidence(202);survives=alt.resolve(20,22)
    checks['independent_alternate_support_preserves_derived_relation']=(len(closure.paths)==2 and arid in alt.active and survives is not None and survives.boundary_q16==(Q,0))
    alt.observe_primitive(20,24,(Q,0),205);alt.observe_primitive(24,22,(Q,Q),206)
    checks['new_live_conflict_defeats_stale_abstraction']=alt.resolve(20,22) is None and (20,22) in alt.ambiguous_pairs

    cp=copy.deepcopy(hot.checkpoint());restored=RecursiveRelationBasisV1.restore(cp);replay=restored.resolve(1,9)
    blob=json.dumps(cp,sort_keys=True).lower()
    checks['checkpoint_preserves_ancestry_not_active_traversal']=(replay==fast and 'active' not in cp and 'ambiguous_pairs' not in cp and 'last_path' not in cp and 'traversal' not in blob)
    renamed=chain(1000)
    for _ in range(7):renamed.silent_wave()
    rr=renamed.resolve(1001,1009)
    checks['opaque_space_renaming_preserves_relation_math']=(rr.boundary_q16==base_boundary and rr.relation_touches==1 and renamed.generation(renamed.best_derived())==hot.generation(best))
    src=inspect.getsource(RecursiveRelationBasisV1)
    checks['no_depth_rank_or_constructor_admission_state']=all(token not in src for token in ('max_dependency_depth','constructor_horizon','base_depth','max_depth_cap'))
    checks['bounded_fast_path']=time.perf_counter()-t<0.25
    failed=[k for k,v in checks.items() if not v]
    out={'contract':'FOUNDRY_RECURSIVE_RELATION_BASIS_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'base_touches':base.relation_touches,'retained_touches':fast.relation_touches,'generation':hot.generation(best),'retained_relations':len(hot.relations)-8,'checks':checks,'failed':failed,'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(out['contract']);print(json.dumps(out,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
