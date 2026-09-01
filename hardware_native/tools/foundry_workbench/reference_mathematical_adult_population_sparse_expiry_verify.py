#!/usr/bin/env python3
"""RED/contrast for sparse absolute eligibility expiry in PopulationBankV1."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_population_v1 import PopulationBankV1,PopulationSpecV1

SPEC=PopulationSpecV1(524288,2,4,42,8)

def numeric_bytes(bank):
    base=bank.numeric_allocation_bytes()
    if isinstance(bank.eligibility,(bytes,bytearray)):base+=len(bank.eligibility)
    if isinstance(bank.edge_eligibility,(bytes,bytearray)):base+=len(bank.edge_eligibility)
    return base

def legacy_checkpoint(bank):
    return {'schema':1,'spec':bank.spec.__dict__,'tick':bank.tick,'next_occurrence':bank.next_occurrence,
            'support':bank.support.tolist(),'last_tick':[0]*bank.spec.site_count,'eligibility':list(bank.eligibility),
            'edge_weight':bank.edge_weight.tolist(),'edge_eligibility':list(bank.edge_eligibility),
            'edge_target':[bank.edge_target_value(e) for e in range(bank.allocated_edge_count)],
            'territory':[bank.territory_value(s) for s in range(bank.spec.site_count)],
            'occurrences':[o.__dict__ for o in bank.occurrences],
            'credit_events':bank.credit_events,'revision_events':bank.revision_events}

def main():
    started=time.perf_counter(); b=PopulationBankV1(SPEC); o=b.recruit((101,202,303))
    H=SPEC.eligibility_horizon; site=o.sites[0]; edge=o.edges[0]
    immediate=(int(b.eligibility[site]),int(b.edge_eligibility[edge]))
    t=time.perf_counter_ns();b.decay();decay_ms=(time.perf_counter_ns()-t)/1e6
    after_one=(int(b.eligibility[site]),int(b.edge_eligibility[edge]));preexpiry_touches=int(getattr(b,'last_decay_touches',-1))

    final_live=PopulationBankV1(SPEC);fo=final_live.recruit((101,202,303))
    for _ in range(H-1):final_live.decay()
    pre=final_live.settle(fo,1,True)
    expired=PopulationBankV1(SPEC);eo=expired.recruit((101,202,303))
    for _ in range(H):expired.decay()
    expiry_touches=int(getattr(expired,'last_decay_touches',-1));post=expired.settle(eo,1,True)

    refresh=PopulationBankV1(SPEC);r1=refresh.recruit((11,22));
    for _ in range(H-1):refresh.decay()
    r2=refresh.activate((11,22),retain=False)
    refreshed=(int(refresh.eligibility[r2.sites[0]]),int(refresh.edge_eligibility[r2.edges[0]]))

    cp=b.checkpoint(); restored=PopulationBankV1.restore(copy.deepcopy(cp))
    legacy_source=PopulationBankV1(PopulationSpecV1(4096,2,4,42,H));legacy_source.recruit((31,41,59));legacy_source.decay()
    legacy=legacy_checkpoint(legacy_source); legacy_r=PopulationBankV1.restore(copy.deepcopy(legacy))
    sparse_site_rows=getattr(b,'sparse_site_eligibility_count',lambda:len(b.eligibility))()
    sparse_edge_rows=getattr(b,'sparse_edge_eligibility_count',lambda:len(b.edge_eligibility))()
    checks={
      'immediate_horizon_exact':immediate==(H,H),
      'one_decay_exact':after_one==(H-1,H-1),
      'final_live_tick_credits':pre['credit']==len(fo.sites) and pre['revisions']==len(fo.edges),
      'first_expired_tick_zero':post=={'credit':0,'revisions':0},
      'reactivation_refreshes_window':refreshed==(H,H),
      'checkpoint_replay_exact':restored.digest()==b.digest(),
      'legacy_restore_exact':legacy_r.digest()==legacy_source.digest(),
      'dense_site_eligibility_deleted':not isinstance(b.eligibility,(bytes,bytearray)),
      'dense_edge_eligibility_deleted':not isinstance(b.edge_eligibility,(bytes,bytearray)),
      'numeric_baseline_at_most_8_bytes_per_site':numeric_bytes(b)<=SPEC.site_count*8,
      'sparse_rows_match_active_set':sparse_site_rows<=len(o.sites) and sparse_edge_rows<=len(o.edges),
      'preexpiry_decay_touches_zero':preexpiry_touches==0,
      'expiry_boundary_touches_exact':expiry_touches==len(eo.sites)+len(eo.edges),
      'decay_under_half_ms':decay_ms<0.5,
      'bounded_seconds_lane':time.perf_counter()-started<3.0,
    }
    result={'schema':'cyber-lagoon.reference-mathematical-adult-population-sparse-expiry.v1','pass':all(checks.values()),
      'reference_only':True,'representation':'SPARSE_ABSOLUTE_ELIGIBILITY_EXPIRY',
      'site_count':SPEC.site_count,'active_sites':len(o.sites),'active_edges':len(o.edges),
      'numeric_bytes':numeric_bytes(b),'numeric_bytes_per_site':numeric_bytes(b)/SPEC.site_count,
      'decay_ms':round(decay_ms,6),'sparse_site_rows':sparse_site_rows,'sparse_edge_rows':sparse_edge_rows,
      'preexpiry_decay_touches':preexpiry_touches,'expiry_boundary_touches':expiry_touches,
      'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_SPARSE_EXPIRY_'+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
