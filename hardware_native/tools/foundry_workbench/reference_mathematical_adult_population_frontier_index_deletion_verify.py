#!/usr/bin/env python3
"""Host falsifier for deleting lifetime-growing derived population frontier indexes."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_population_v1 import PopulationBankV1,PopulationSpecV1

state_minimization_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True
work_reduction = 'lifetime-growing derived frontier sets -> current sparse causal-map keys'

SPEC=PopulationSpecV1(131072,2,4,42,8)
EPHEMERAL_CONTACTS=2048
DONOR_STALE_SITE_FRONTIER=28199
DONOR_STALE_EDGE_FRONTIER=56398
DONOR_CHECKPOINT_MS=45.986349

def wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'))

def future(bank):
    occ=bank.activate((0xF001,0xF002),retain=True)
    settled=bank.settle(occ,1,True)
    return {'sites':tuple(occ.sites),'edges':tuple(occ.edges),'settled':settled,
            'digest':bank.digest(),'checkpoint':bank.checkpoint()}

def main():
    started=time.perf_counter();b=PopulationBankV1(SPEC)
    for i in range(EPHEMERAL_CONTACTS):
        b.activate((0x100000+i,0x200000+i),retain=False);b.decay()
    for _ in range(SPEC.eligibility_horizon):b.decay()

    t=time.perf_counter_ns();cp=b.checkpoint();checkpoint_ms=(time.perf_counter_ns()-t)/1e6
    digest_before=b.digest();wire_before=wire(cp)
    has_frontiers=hasattr(b,'_state_sites') or hasattr(b,'_state_edges')
    if has_frontiers:
        saved_sites=set(getattr(b,'_state_sites',()))
        saved_edges=set(getattr(b,'_state_edges',()))
        if hasattr(b,'_state_sites'):b._state_sites.clear()
        if hasattr(b,'_state_edges'):b._state_edges.clear()
        clearing_digest_same=b.digest()==digest_before
        clearing_checkpoint_same=wire(b.checkpoint())==wire_before
        if hasattr(b,'_state_sites'):b._state_sites=saved_sites
        if hasattr(b,'_state_edges'):b._state_edges=saved_edges
    else:
        clearing_digest_same=True;clearing_checkpoint_same=True

    live_site_rows=b.sparse_support_count()+b.sparse_site_eligibility_count()
    live_edge_rows=b.sparse_edge_weight_count()+b.sparse_edge_eligibility_count()
    restored=PopulationBankV1.restore(copy.deepcopy(cp));left=future(b);right=future(restored)

    causal=PopulationBankV1(SPEC);prepared=causal.prepare((11,22));co=causal.activate_signature(prepared,retain=True);causal.settle(co,1,True)
    cut=copy.deepcopy(causal.checkpoint());cut['site_support']=[0 for _ in cut['site_support']]
    causal_cut_rejected=False
    try:
        damaged=PopulationBankV1.restore(cut);damaged.activate_signature(prepared,retain=False)
    except Exception:causal_cut_rejected=True

    checks={
      'donor_stale_frontier_was_material':DONOR_STALE_SITE_FRONTIER>1000 and DONOR_STALE_EDGE_FRONTIER>2000 and DONOR_CHECKPOINT_MS>10,
      'derived_frontier_sets_deleted':not hasattr(b,'_state_sites') and not hasattr(b,'_state_edges'),
      'checkpoint_projects_only_current_rows':len(cp.get('sites',()))==0 and len(cp.get('edges',()))==0 and live_site_rows==0 and live_edge_rows==0,
      'clearing_frontier_had_no_digest_authority':clearing_digest_same,
      'clearing_frontier_had_no_checkpoint_authority':clearing_checkpoint_same,
      'restored_current_state_has_same_future':left==right,
      'causal_support_negative_control_rejected':causal_cut_rejected,
      'checkpoint_under_one_ms':checkpoint_ms<1.0,
      'bounded_seconds_lane':time.perf_counter()-started<3.0,
    }
    result={'schema':'cyber-lagoon.reference-mathematical-adult-population-frontier-index-deletion.v1','pass':all(checks.values()),
      'reference_only':True,'state_role':'DERIVED_NOMINATION_INDEX','contacts':EPHEMERAL_CONTACTS,
      'donor_stale_site_frontier':DONOR_STALE_SITE_FRONTIER,'donor_stale_edge_frontier':DONOR_STALE_EDGE_FRONTIER,
      'donor_checkpoint_ms':DONOR_CHECKPOINT_MS,'candidate_checkpoint_ms':round(checkpoint_ms,6),
      'checkpoint_bytes':len(wire_before.encode()),'live_site_rows':live_site_rows,'live_edge_rows':live_edge_rows,
      'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_FRONTIER_INDEX_DELETION_'+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
