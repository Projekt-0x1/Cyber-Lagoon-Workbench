#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,AdultStateV1
from reference_language_mastery_contact_adapter_v1 import *
from reference_predictive_credit_profile_v1 import Q


def refused_leaf(adult,ctx,atoms):
    try:
        adult.leaf(ctx,atoms)
        return False
    except RuntimeError as exc:
        return str(exc)=='adult:construction_not_productive'


def main():
    t=time.perf_counter(); a=LanguageMasteryAdultV1(); contact=LanguageMasteryContactAdapterV1(a); C=9001; JOIN=9101
    A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402
    names={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
    for f,text in names.items():
        contact.contact(CONTACT_SCENE,(100,f),1000+f);contact.contact(CONTACT_SURFACE,tuple(text.encode()),1000+f)
        contact.contact(CONTACT_SCENE,(100,f),2000+f);contact.contact(CONTACT_SURFACE,tuple(text.encode()),2000+f)
    x=(A1,G1,V1,O1); partial=(A2,G1,V1,O1); y=(A2,G2,V2,O2); held=(A1,G2,V1,O2)
    sx='the careful engineer tests the sensor.'
    sp='the quiet engineer tests the sensor.'
    sy='the quiet technician inspects the valve.'

    checks={}
    # First source cannot even establish a reusable construction.
    sxid=contact.contact(CONTACT_SCENE,(C,*x),3001);contact.contact(CONTACT_SURFACE,tuple(sx.encode()),3001)
    checks['single_source_no_abstract_construction']=refused_leaf(a,C,held)
    # Same exemplar from another source establishes the concrete pattern, not productivity.
    contact.contact(CONTACT_SCENE,(C,*x),3002);contact.contact(CONTACT_SURFACE,tuple(sx.encode()),3002)
    exact=a.leaf(C,x)
    checks['repeated_exemplar_supports_concrete_use']=len(exact.surface)>0
    checks['token_repetition_does_not_unlock_recombination']=refused_leaf(a,C,held)
    # Varying one port only still leaves three ports item-bound.
    spid=contact.contact(CONTACT_SCENE,(C,*partial),3003);contact.contact(CONTACT_SURFACE,tuple(sp.encode()),3003)
    checks['one_slot_variation_still_not_productive']=refused_leaf(a,C,held)
    checks['partial_diversity_score_below_productive']=a.construction_productivity(C,held)==0
    # A fully diverse exemplar now supplies variation in every port.
    syid=contact.contact(CONTACT_SCENE,(C,*y),3004);contact.contact(CONTACT_SURFACE,tuple(sy.encode()),3004)
    h=a.leaf(C,held)
    checks['type_diversity_unlocks_heldout_recombination']=bytes(h.surface).decode()=='the careful technician tests the valve.'
    checks['productivity_is_lived_diversity_not_rule_bit']=a.construction_productivity(C,held)>=2

    # Higher-order behavior appears only after lower structures exist.
    left=a.leaf(C,x);right=a.leaf(C,y)
    contact.contact(CONTACT_RELATION,(JOIN,sxid,syid),5001);contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(left.surface)+(32,)+tuple(right.surface),5001)
    heldid=contact.contact(CONTACT_SCENE,(C,*held),5002);contact.contact(CONTACT_RELATION,(JOIN,syid,heldid),5002);contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(right.surface)+(32,)+tuple(h.surface),5002)
    pair=a.compose(JOIN,left,right);held_pair=a.compose(JOIN,h,left)
    p=None
    for _ in range(3):p=a.experience_program((h.identity,left.identity),held_pair,Q//2,Q//16,0xA11,Q//8)
    checks['higher_program_emerges_after_composable_children']=p is not None and p.depth>=1
    pid=p.identity
    # No gradient/loss optimization: behavior changes only through lived use + returned consequence.
    before=a.credit.row(pid).outcome_mean_q16
    a.experience_choice(pid,3*Q//4,Q//8,0xA11,Q//8,3,True)
    after=a.credit.row(pid).outcome_mean_q16
    checks['returned_consequence_changes_prospective_value']=after!=before
    checks['normal_state_can_select_learned_program']=a._probe_choice(0xA11,AdultStateV1())==pid
    checks['no_optimizer_or_loss_state']=all(not hasattr(a,n) for n in ('optimizer','loss','gradient','backprop','target_text','grammar_rules','tokens'))
    checks['bounded_fast_curve']=time.perf_counter()-t<1.0
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_LANGUAGE_DEVELOPMENT_CURVE_RED '+','.join(failed))
    pth=Path(__file__)
    result={'contract':'FOUNDRY_AGI_LANGUAGE_DEVELOPMENT_CURVE_GREEN','reference_only':True,'optimizer':False,'grammar_rules':False,'token_objective':False,'stages':['concrete_item','supported_construction','partial_variation','productive_abstraction','hierarchical_program','consequence_sensitive_selection'],'heldout_surface':bytes(h.surface).decode(),'checks':checks,'elapsed_ms':round((time.perf_counter()-t)*1000,3),'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
