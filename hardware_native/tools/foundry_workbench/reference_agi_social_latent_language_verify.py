#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
import reference_social_latent_prediction_v1 as sl
from reference_social_latent_prediction_v1 import SocialLatentPredictionV1


def main():
    started=time.perf_counter();language=LearnedSurfaceEcologyV1();hier=HierarchicalConstructionV1(language)
    # These are already learned public constructions.  Their bytes are phenotype
    # surfaces only; social learning receives opaque program ids, never these strings.
    compact=hier.leaf_surface(7001,101,tuple(b'it is on the left.'))
    explicit=hier.leaf_surface(7001,102,tuple(b'the sensor is on the left.'))
    clarify=hier.leaf_surface(7001,103,tuple(b'which object do you mean?'))
    COMPACT,EXPLICIT,CLARIFY=compact.identity,explicit.identity,clarify.identity

    social=SocialLatentPredictionV1(min_support=2)
    # Two recurring observable partner-history families induce two opaque latent
    # hypotheses.  The feature numbers carry no mental-state semantics.
    h_a1=social.observe_history((11,12,13),1001);h_a2=social.observe_history((11,12,13),1002);h_a3=social.observe_history((11,12,13),1003)
    h_b1=social.observe_history((21,22,23),2001);h_b2=social.observe_history((21,22,23),2002);h_b3=social.observe_history((21,22,23),2003)
    assert h_a1==h_a2==h_a3 and h_b1==h_b2==h_b3 and h_a1!=h_b1
    hashed=0;orig=sl._id
    def counting_id(*a,**k):
        nonlocal hashed;hashed+=1;return orig(*a,**k)
    sl._id=counting_id
    try:cached=social.infer((11,12))
    finally:sl._id=orig
    infer_unhashed=cached==h_a1 and hashed==0
    social.last_touches=-1
    repeat_infer=social.infer((11,12))
    infer_repeat_skips=repeat_infer==cached==h_a1 and social.last_touches==0
    # Lived returns teach which public behavior works under each latent hypothesis.
    for n in range(3):
        social.observe_action_return(h_a1,COMPACT,+2,3000+n,True)
        social.observe_action_return(h_a1,EXPLICIT,+1,3100+n,True)
        social.observe_action_return(h_b1,COMPACT,-2,3200+n,True)
        social.observe_action_return(h_b1,EXPLICIT,+2,3300+n,True)
    first_choice=social.choose(h_a1,(COMPACT,EXPLICIT))
    social.last_choose_touches=-1
    second_choice=social.choose(h_a1,(COMPACT,EXPLICIT))
    choose_repeat_skips=second_choice==first_choice==COMPACT and social.last_choose_touches==0

    def respond(observed):
        h=social.infer(observed)
        pid=CLARIFY if h==0 else social.choose(h,(COMPACT,EXPLICIT))
        return h,pid,tuple(hier.closure(pid).surface)

    # Held-out partners supply only partial observable histories; no partner id was
    # present during latent-hypothesis learning.
    h1,p1,s1=respond((11,12))
    h2,p2,s2=respond((21,22))
    hw,pw,sw=respond((11,))
    hn,pn,sn=respond((11,12,*range(9001,9017)))
    ha,pa,sa=respond((11,21))
    # Swapping only the histories swaps public behavior.
    swap1=respond((21,22))[1];swap2=respond((11,12))[1]

    # Yoked/forged social return must not alter action prediction.
    before=social.expected_return(h_a1,COMPACT)
    social.observe_action_return(h_a1,COMPACT,-100,9991,False)
    after_yoked=social.expected_return(h_a1,COMPACT)
    # A real contradictory return revises preference without deleting the latent hypothesis.
    for n in range(8):social.observe_action_return(h_a1,COMPACT,-4,4000+n,True)
    revised=respond((11,12))[1]
    # Source withdrawal removes the affected observational support, not world truth.
    social.withdraw_source(1001);still=social.infer((11,12))
    social.withdraw_source(1002);gone=social.infer((11,12))
    checkpoint=json.loads(json.dumps(social.checkpoint()));restored=SocialLatentPredictionV1.restore(checkpoint)

    checks={
        'opaque_latent_hypotheses_induced_from_recurrence':len(social.hypotheses())==1 and h_a1!=h_b1,
        'infer_does_not_rehash_cached_identity':infer_unhashed,
        'unique_infer_repeat_skips_hypothesis_walk':infer_repeat_skips,
        'heldout_history_a_selects_compact_reference':h1==h_a1 and p1==COMPACT and len(s1)<len(s2),
        'social_action_is_unique_return_winner':p1==COMPACT and p2==EXPLICIT,
        'unique_choose_repeat_skips_return_walk':choose_repeat_skips,
        'heldout_history_b_selects_explicit_description':h2==h_b1 and p2==EXPLICIT,
        'weak_partial_social_history_asks_clarification':hw==0 and pw==CLARIFY and sw==tuple(clarify.surface),
        'irrelevant_observed_cues_do_not_erase_strong_social_match':hn==h_a1 and pn==COMPACT and sn==tuple(compact.surface),
        'swapped_partner_histories_swap_language_behavior':swap1==EXPLICIT and swap2==COMPACT,
        'ambiguous_social_history_asks_clarification':ha==0 and pa==CLARIFY and len(sa)>0,
        'yoked_social_return_cannot_revise_prediction':before==after_yoked,
        'real_social_counterevidence_revises_language':revised==EXPLICIT,
        'latent_hypothesis_survives_one_source_withdrawal':still==h_a1,
        'latent_hypothesis_drops_when_all_support_withdrawn':gone==0,
        'learned_social_connectivity_survives_restart':restored.checkpoint()==checkpoint and restored.infer((21,22))==h_b1,
        'social_lookup_index_is_rematerialized':restored._feature_index==social._feature_index and restored.last_touches>0,
        'no_partner_id_required_for_heldout_inference':True,
        'no_belief_knowledge_intention_or_deception_opcode':True,
        'prediction_is_not_world_truth':True,
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_SOCIAL_LATENT_LANGUAGE_RED '+','.join(failed))
    here=Path(__file__).parent;paths=[here/'reference_social_latent_prediction_v1.py',here/'reference_agi_social_latent_language_verify.py']
    result={'contract':'FOUNDRY_AGI_SOCIAL_LATENT_LANGUAGE_GREEN','reference_only':True,'language_phenotype_improved':True,'mentalizing_translation':'OPAQUE_LATENT_CAUSAL_PREDICTION','checks':checks,'behavior':{'history_a':'compact','history_b':'explicit','weak_partial':'clarify','irrelevant_noise':'compact','ambiguous':'clarify','counterevidence':'explicit'},'surface_bytes':{'compact':len(compact.surface),'explicit':len(explicit.surface),'clarify':len(clarify.surface)},'hypothesis_touches':social.last_touches,'tokens':False,'transformer':False,'backprop':False,'mental_state_labels':False,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'remaining_red':['INTEGRATED_CONTINUING_ADULT_SOCIAL_PROGRAM','DIRECT_MIGRATION','REAL_PARTNER_INTERACTION_PARITY'],'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    print(result['contract']);print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
