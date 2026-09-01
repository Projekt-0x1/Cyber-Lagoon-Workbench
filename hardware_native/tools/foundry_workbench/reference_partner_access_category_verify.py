#!/usr/bin/env python3
"""Whole-organism partner shared-access receipt; no proposition-level theory-of-mind fixture."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_social_language_verify import train_base,partner,scene,settle,CTX

P=9701;QPARTNER=9702;NEG=9703;YOKED=9704
W0=(11,12,13,14,15,16);W1=(21,22,23,24,25,26)

def organism():return ReferenceOrganismV2(PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8))
def establish_shared(o,p,atoms,source):
    partner(o,p);scene(o,CTX,atoms,source);a=o.tick()
    if not isinstance(a,ActionV2):raise AssertionError(('partner-access:action',a))
    settle(o,a,p,1,True);return a

def main():
    started=time.perf_counter();checks={};o=organism();sensor_prior,_=train_base(o)
    o.contact(CONTACT_WORLD_STATE,W0,8801,True,True);partner(o,P)
    checks['partner_presence_alone_creates_zero_shared_access']=o.recursive_partner_access.evidence_count==0
    establish_shared(o,P,sensor_prior,8802)
    checks['positive_independent_common_ground_creates_partner_local_access']=o.recursive_partner_access.evidence_count==1 and len(o.recursive_partner_access.partner_rows(P))==1
    stale_at_shared=o.partner_shared_access_staleness_q16(P)
    checks['current_shared_world_has_low_access_staleness']=stale_at_shared<CTXQ//2

    # World changes without a consequence-certified public update to P.
    o.contact(CONTACT_WORLD_STATE,W1,8803,True,True)
    stale_after_change=o.partner_shared_access_staleness_q16(P)
    checks['unshared_world_change_can_make_prior_access_stale']=stale_after_change>stale_at_shared
    partner(o,QPARTNER)
    checks['fresh_partner_cannot_borrow_other_partner_access']=o.partner_shared_access_staleness_q16(QPARTNER)==CTXQ//2 and not o.recursive_partner_access.partner_rows(QPARTNER)

    # Returning P has stale access until a new public episode is actually shared.
    partner(o,P);stale_on_return=o.partner_shared_access_staleness_q16(P)
    establish_shared(o,P,sensor_prior,8804)
    stale_after_reshare=o.partner_shared_access_staleness_q16(P)
    checks['partner_return_does_not_magically_refresh_access']=stale_on_return==stale_after_change
    checks['genuine_resharing_closes_access_staleness']=stale_after_reshare<stale_on_return and len(o.recursive_partner_access.partner_rows(P))==2

    # Negative and yoked public outcomes cannot mint shared access.
    n=organism();n_prior,_=train_base(n);n.contact(CONTACT_WORLD_STATE,W0,8810,True,True);partner(n,NEG);scene(n,CTX,n_prior,8811);na=n.tick();settle(n,na,NEG,-1,True)
    checks['negative_public_return_creates_no_shared_access']=n.recursive_partner_access.evidence_count==0
    y=organism();y_prior,_=train_base(y);y.contact(CONTACT_WORLD_STATE,W0,8820,True,True);partner(y,YOKED);scene(y,CTX,y_prior,8821);ya=y.tick();settle(y,ya,YOKED,1,False)
    checks['nonindependent_public_return_creates_no_shared_access']=y.recursive_partner_access.evidence_count==0

    category_guard=False
    try:o.recursive_partner_access.perspective_gap_q16(P,W1,0)
    except RuntimeError:category_guard=True
    checks['shared_access_api_refuses_perspective_or_belief_category']=category_guard
    before_rows=len(o.recursive_partner_access.partner_rows(P));o.contact(CONTACT_WITHDRAW_SOURCE,(P,),8899,True,True)
    checks['withdrawal_deactivates_access_without_erasing_history']=o.partner_access_applicability_q16(P)==0 and len(o.recursive_partner_access.partner_rows(P))==before_rows
    checks['shared_access_rows_have_zero_truth_authority']=all(int(row.authority)==0 for row in o.recursive_partner_access._rows)

    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['partner_access_survives_checkpoint']=r.recursive_partner_access.checkpoint()==o.recursive_partner_access.checkpoint()
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.partner-shared-access-category.v1','pass':not failed,'checks':checks,'failed':failed,
        'stale_shared_q16':stale_at_shared,'stale_missed_update_q16':stale_after_change,'stale_reshared_q16':stale_after_reshare,
        'claim':'CONSEQUENCE_CERTIFIED_SHARED_ACCESS_IS_PARTNER_LOCAL_AND_NEVER_PROMOTED_TO_BELIEF_CONTENT',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_PARTNER_SHARED_ACCESS_CATEGORY '+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
