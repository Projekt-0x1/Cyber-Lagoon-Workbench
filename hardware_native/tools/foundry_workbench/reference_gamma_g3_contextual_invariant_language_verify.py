#!/usr/bin/env python3
"""G3 research-grounded N+1: context diversity earns reusable recursive discourse.

The underlying language learner is unchanged. A reference Life Function keeps exact
relation episodes in a contextual-invariant bank and consolidates them into the
ordinary learned span relation only after the resident evidence supports invariance.
Same-source-count/same-exposure controls separate context diversity from repetition.
"""
from __future__ import annotations
import copy,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from reference_contextual_invariant_consolidation_v1 import ContextualInvariantBankV1
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE,CONTACT_SURFACE,LanguageMasteryContactAdapterV1
from reference_hierarchical_composition_v1 import HierarchicalRefuse

C=9401;JOIN=9402;SEP=b' then '
A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402
NAMES={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
CLAUSES=(
    ((A1,G1,V1,O1),'the careful engineer tests the sensor.'),
    ((A2,G2,V2,O2),'the quiet technician inspects the valve.'),
    ((A1,G2,V2,O1),'the careful technician inspects the sensor.'),
    ((A2,G1,V1,O2),'the quiet engineer tests the valve.'),
)

class ContextualLanguageLifeV1:
    def __init__(self):
        self.adult=LanguageMasteryAdultV1();self.invariants=ContextualInvariantBankV1();self.pending=[];self.promoted=set()
    def relation_episode(self,left,right,context_identity,source_identity):
        witness=((int(left.identity)&0xffffffff)<<32)^(int(right.identity)&0xffffffff)^int(source_identity)
        episode=self.invariants.observe(JOIN,context_identity,source_identity,witness)
        self.pending.append((episode,int(source_identity),left,right))
        self._materialize_if_earned()
        return episode
    def consolidate(self,plasticity_q16=65536,resource_q16=65536):
        self.invariants.consolidate(plasticity_q16,resource_q16);self._materialize_if_earned()
    def _materialize_if_earned(self):
        if not self.invariants.reusable(JOIN):return False
        for episode,source,left,right in self.pending:
            if episode in self.promoted:continue
            self.adult.observe_join(JOIN,left,right,source,SEP);self.promoted.add(episode)
        return True


def contact(adult,scene,text,source):
    c=LanguageMasteryContactAdapterV1(adult);c.contact(CONTACT_SCENE,scene,source);c.contact(CONTACT_SURFACE,tuple(text.encode()),source)

def prepare(life):
    a=life.adult
    for feature,text in NAMES.items():
        for base in (1000,2000):contact(a,(100,feature),text,base+feature)
    for i,(atoms,text) in enumerate(CLAUSES):
        for base in (3000,4000):contact(a,(C,*atoms),text,base+i)
    return [a.leaf(C,atoms) for atoms,_ in CLAUSES]

def refuses(a,left,right):
    try:a.compose(JOIN,left,right);return False
    except HierarchicalRefuse:return True

def main():
    started=time.perf_counter();variable=ContextualLanguageLifeV1();repeated=ContextualLanguageLifeV1();vl=prepare(variable);rl=prepare(repeated)
    checks={
      'same_initial_language_capability':all(bytes(vl[i].surface)==bytes(rl[i].surface) for i in range(4)),
      'relation_absent_before_lived_relation_episodes':refuses(variable.adult,vl[3],vl[0]) and refuses(repeated.adult,rl[3],rl[0]),
    }
    # Equal total exposure and equal independent source count. Only context varies.
    vepisodes=[];repisodes=[]
    for trial,(li,ri) in enumerate(((0,1),(2,3)),1):
        source=7000+trial
        vepisodes.append(variable.relation_episode(vl[li],vl[ri],8000+trial,source))
        repisodes.append(repeated.relation_episode(rl[li],rl[ri],8100,source))
    checks['matched_exposure_count']=variable.invariants.invariants[JOIN].total_evidence==repeated.invariants.invariants[JOIN].total_evidence==2
    checks['matched_source_diversity']=variable.invariants.invariants[JOIN].source_diversity==repeated.invariants.invariants[JOIN].source_diversity==2
    checks['only_context_diversity_differs']=variable.invariants.invariants[JOIN].context_diversity==2 and repeated.invariants.invariants[JOIN].context_diversity==1
    checks['variable_context_invariant_is_reusable']=variable.invariants.reusable(JOIN)
    checks['repeated_context_invariant_is_not_reusable']=not repeated.invariants.reusable(JOIN)
    first=variable.adult.compose(JOIN,vl[3],vl[0]);nested=variable.adult.compose(JOIN,first,vl[2])
    visible=bytes(nested.surface).decode()
    checks['variable_context_history_unlocks_recursive_discourse']=nested.depth==2 and bytes(nested.surface).count(SEP)==2
    checks['matched_repeated_context_history_still_refuses']=refuses(repeated.adult,rl[3],rl[0])

    # Replay/consolidation can strengthen what happened but cannot forge new context.
    before_diversity=repeated.invariants.invariants[JOIN].context_diversity
    for _ in range(64):repeated.consolidate()
    checks['consolidation_strengthens_without_forging_diversity']=(
        repeated.invariants.invariants[JOIN].consolidated_strength>0 and
        repeated.invariants.invariants[JOIN].context_diversity==before_diversity==1 and
        not repeated.invariants.reusable(JOIN) and refuses(repeated.adult,rl[3],rl[0]))

    # Episode/invariant lesions are dissociable in mechanism state.
    bank=ContextualInvariantBankV1.restore(copy.deepcopy(variable.invariants.checkpoint()))
    checks['checkpoint_replay_exact']=bank.checkpoint()==variable.invariants.checkpoint()
    checks['one_episode_lesion_preserves_other_exact_episode']=bank.lesion_episode(vepisodes[0]) and len(bank.episodes)==1
    lesion_bank=ContextualInvariantBankV1.restore(copy.deepcopy(variable.invariants.checkpoint()))
    checks['invariant_lesion_preserves_episode_evidence']=lesion_bank.lesion_invariant(JOIN) and not lesion_bank.reusable(JOIN) and len(lesion_bank.episodes)==2
    checks['invariant_repair_uses_existing_evidence_not_new_contact']=lesion_bank.repair_invariant(JOIN) and lesion_bank.invariants[JOIN].context_diversity==2 and len(lesion_bank.episodes)==2

    # Sapolsky/allostatic axis: resource/plasticity can modulate consolidation work,
    # but cannot author context diversity or semantic relation evidence.
    low=ContextualInvariantBankV1.restore(copy.deepcopy(variable.invariants.checkpoint()));high=ContextualInvariantBankV1.restore(copy.deepcopy(variable.invariants.checkpoint()))
    low_before=low.invariants[JOIN].context_diversity;high_before=high.invariants[JOIN].context_diversity
    low.consolidate(plasticity_q16=65536,resource_q16=0);high.consolidate(plasticity_q16=65536,resource_q16=65536)
    checks['resource_state_modulates_consolidation_not_evidence_identity']=(
        low.invariants[JOIN].consolidated_strength < high.invariants[JOIN].consolidated_strength and
        low.invariants[JOIN].context_diversity==low_before and high.invariants[JOIN].context_diversity==high_before)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_GAMMA_G3_CONTEXTUAL_INVARIANT_LANGUAGE_GREEN','pass':not failed,'reference_only':True,'graph_flip':False,
      'novel_synthesis':True,'visible_language_gain':'MATCHED_EXPOSURE_CONTEXT_DIVERSITY_UNLOCKS_RECURSIVE_DISCOURSE',
      'matched_exposures':2,'matched_sources':2,'variable_contexts':2,'repeated_contexts':1,
      'nested_surface':visible,'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3),
      'remaining_red':['DIRECT_CONTEXTUAL_INVARIANT_LOWERING','DIRECT_RUNTIME_OWNERSHIP','PHYSICAL_HISTORY_RESOURCE_ASSAY','GRAPH_PROMOTION'],
      'next_falsifiers':{
        'chomsky':'Held-out long-distance dependency/agreement with intervening distractors under matched local adjacency statistics.',
        'sapolsky':'Matched present relation after different controllability, social-source, resource and recovery histories; require lawful divergence plus reversible recovery.'}}
    print(result['contract'] if not failed else 'FOUNDRY_GAMMA_G3_CONTEXTUAL_INVARIANT_LANGUAGE_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
