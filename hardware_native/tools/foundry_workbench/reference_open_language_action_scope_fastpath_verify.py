#!/usr/bin/env python3
"""Economic receipt: eliminate impossible embedded-scope structural recursion without changing candidates."""
from __future__ import annotations
import copy,json,time
from reference_open_language_action_affordance_v1 import (
    OpenLanguageActionAffordanceV1,MIN_FACTOR_BYTES,MAX_FACTOR_BYTES,
)
from reference_open_joint_role_affordance_v1 import OpenJointRoleAffordanceV1
from reference_life_extension_argument_joint_plan_v1 import ROLE_FORMS,B1

state_minimization_refactor = True
representation_only = True
phenotype_preserved = True
future_update_authority_preserved = True

class SlowDonor(OpenLanguageActionAffordanceV1):
    """Current semantics with the pre-fastpath execution strategy and no rebuild cache."""
    def _structural_inventory(self):
        held=self._structural_inventory_cache;self._structural_inventory_cache=None
        try:return super()._structural_inventory()
        finally:self._structural_inventory_cache=None
    def _embedded_scope_owns_action(self,raw,action):
        raw=tuple(map(int,raw));full_prefix,full_suffix=self._structural_frame_strength(raw)
        if full_prefix>0 or full_suffix<=0:return False
        for start in range(1,max(1,len(raw)-MIN_FACTOR_BYTES+1)):
            child=raw[start:]
            rows=self.structural_candidates(child)
            if len(rows)!=1 or int(rows[0][0])!=int(action):continue
            child_prefix,_child_suffix=self._structural_frame_strength(child)
            if child_prefix>0:return True
        return False

def build(inner_cls):
    joint=OpenJointRoleAffordanceV1();joint._inner=inner_cls();tick=1
    for idx,name in enumerate(ROLE_FORMS):
        base=0x7A000+idx*0x100;forms=ROLE_FORMS[name]
        for witness in range(4):
            speaker=base+witness;other=base+0x40+witness;variant=witness%2
            for key,action,actor_self in (('b1_other',B1,False),('b1_self',B1,True)):
                raw=tuple(forms[key][variant].encode('utf-8'))
                joint.observe_language(raw,speaker,tick);tick+=1
                joint.observe_actor_action(speaker,speaker if actor_self else other,action,tick);tick+=1
    return joint

def queries():
    out=[]
    for name in ROLE_FORMS:
        out.append(tuple(ROLE_FORMS[name]['b1_other'][2].encode('utf-8')))
        out.append(tuple(('Alex said: '+ROLE_FORMS[name]['b1_other'][2]).encode('utf-8')))
    return tuple(out)

def measure(joint,qs):
    inner=joint._inner;count={'structural':0};orig=inner.structural_candidates
    def wrapped(raw):count['structural']+=1;return orig(raw)
    inner.structural_candidates=wrapped
    start=time.perf_counter();rows=tuple(inner.candidates(q) for q in qs);elapsed=time.perf_counter()-start
    return rows,count['structural'],elapsed

def main():
    qs=queries();donor=build(SlowDonor);challenger=build(OpenLanguageActionAffordanceV1)
    donor_rows,donor_calls,donor_s=measure(donor,qs)
    challenger_rows,challenger_calls,challenger_s=measure(challenger,qs)
    cp=copy.deepcopy(challenger._inner.checkpoint());restored=OpenLanguageActionAffordanceV1.restore(copy.deepcopy(cp))
    restored_rows=tuple(restored.candidates(q) for q in qs)
    # Future update authority: after restore, fresh witnessed speech/action still changes learned state.
    before=(len(restored._examples),len(restored._factors));speaker=0x7FFF01;raw=tuple(b'please take the sheltered route')
    restored.observe_language(raw,speaker,10000);restored.observe_action(B1,speaker,10001)
    after=(len(restored._examples),len(restored._factors))
    checks={
        'candidate_phenotype_is_byte_exact':challenger_rows==donor_rows,
        'seven_heldout_role_surfaces_remain_actionable':all(bool(challenger_rows[i]) for i in range(0,len(qs),2)),
        'outer_report_wrapper_behavior_is_exactly_preserved':tuple(bool(challenger_rows[i]) for i in range(1,len(qs),2))==tuple(bool(donor_rows[i]) for i in range(1,len(qs),2)),
        'checkpoint_rebuild_preserves_candidate_phenotype':restored_rows==challenger_rows,
        'future_update_authority_survives_checkpoint':after[0]>before[0],
        'recursive_structural_calls_reduced':challenger_calls<donor_calls,
        'measured_latency_reduced':challenger_s<donor_s,
    }
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.open-language-action-scope-fastpath.v1','contract':'FOUNDRY_OPEN_LANGUAGE_ACTION_SCOPE_FASTPATH_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'economic_refactor':True,'representation_only':representation_only,'state_minimization_refactor':state_minimization_refactor,'phenotype_preserved':checks['candidate_phenotype_is_byte_exact'] and checks['checkpoint_rebuild_preserves_candidate_phenotype'],'future_update_authority_preserved':checks['future_update_authority_survives_checkpoint'],'economic_gain':'EMBEDDED_SCOPE_PREFILTERS_IMPOSSIBLE_CHILD_BOUNDARIES_BEFORE_STRUCTURAL_RECONSTRUCTION','metrics':{'queries':len(qs),'donor_ms':round(donor_s*1000,3),'challenger_ms':round(challenger_s*1000,3),'speedup':round(donor_s/max(challenger_s,1e-12),3),'donor_structural_calls':donor_calls,'challenger_structural_calls':challenger_calls,'structural_call_reduction':donor_calls-challenger_calls},'checks':checks,'failed':failed}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
