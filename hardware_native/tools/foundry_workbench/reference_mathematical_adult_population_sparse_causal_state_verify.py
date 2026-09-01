#!/usr/bin/env python3
"""RED/contrast for sparse causal population support and learned edge weights."""
from __future__ import annotations
import copy,json,sys,time
from array import array
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_population_v1 import PopulationBankV1,PopulationSpecV1

state_minimization_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True
state_reduction = 'O(N) dense support/weight defaults -> sparse causal deviations'

SPEC=PopulationSpecV1(524288,2,4,42,8)

def main():
    started=time.perf_counter();b=PopulationBankV1(SPEC)
    fresh_bytes=b.numeric_allocation_bytes()
    first=b.recruit((101,202,303));before=tuple(int(b.edge_weight[e]) for e in first.edges)
    pos=b.settle(first,1,True);after_pos=tuple(int(b.edge_weight[e]) for e in first.edges)
    # A distinct negative experience exercises signed learned deviations.
    second=b.recruit((404,505,606));neg=b.settle(second,-1,True);after_neg=tuple(int(b.edge_weight[e]) for e in second.edges)
    prepared=b.prepare((7001,7002));activated=b.activate_signature(prepared,retain=True)
    cp=b.checkpoint();restored=PopulationBankV1.restore(copy.deepcopy(cp))
    legacy_source=PopulationBankV1(PopulationSpecV1(4096,2,4,42,8));lo=legacy_source.recruit((31,41,59));legacy_source.settle(lo,1,True);legacy_source.decay()
    legacy={'schema':1,'spec':legacy_source.spec.__dict__,'tick':legacy_source.tick,'next_occurrence':legacy_source.next_occurrence,
      'support':list(legacy_source.support),'last_tick':[0]*legacy_source.spec.site_count,'eligibility':list(legacy_source.eligibility),
      'edge_weight':list(legacy_source.edge_weight),'edge_eligibility':list(legacy_source.edge_eligibility),
      'edge_target':[legacy_source.edge_target_value(e) for e in range(legacy_source.allocated_edge_count)],
      'territory':[legacy_source.territory_value(s) for s in range(legacy_source.spec.site_count)],
      'occurrences':[o.__dict__ for o in legacy_source.occurrences],'credit_events':legacy_source.credit_events,'revision_events':legacy_source.revision_events}
    legacy_r=PopulationBankV1.restore(copy.deepcopy(legacy))
    sparse_support=getattr(b,'sparse_support_count',lambda:len(b.support))()
    sparse_weights=getattr(b,'sparse_edge_weight_count',lambda:len(b.edge_weight))()
    support_cut=copy.deepcopy(cp);support_cut['site_support']=[0 for _ in support_cut.get('site_support',())]
    support_negative=False
    try:
        cut=PopulationBankV1.restore(support_cut);cut.activate_signature(prepared,retain=False)
    except Exception:support_negative=True
    checks={
      'positive_credit_exact':pos['credit']==len(first.sites) and pos['revisions']==len(first.edges) and all(y==x+1 for x,y in zip(before,after_pos)),
      'negative_weight_exact':neg['revisions']==len(second.edges) and all(v==0 for v in after_neg),
      'prepared_support_still_load_bearing':all(int(b.support[s])>0 for s in prepared) and len(activated.sites)==len(prepared),
      'checkpoint_replay_exact':restored.digest()==b.digest(),
      'legacy_replay_exact':legacy_r.digest()==legacy_source.digest(),
      'support_negative_control_rejected':support_negative,
      'dense_support_array_deleted':not isinstance(b.support,array),
      'dense_edge_weight_array_deleted':not isinstance(b.edge_weight,array),
      'fresh_dense_numeric_allocation_zero':fresh_bytes==0,
      'sparse_support_rows_bounded':sparse_support<=len(b._state_sites),
      'sparse_weight_rows_bounded':sparse_weights<=len(b._state_edges),
      'bounded_seconds_lane':time.perf_counter()-started<3.0,
    }
    out={'schema':'cyber-lagoon.reference-mathematical-adult-population-sparse-causal-state.v1','pass':all(checks.values()),
      'reference_only':True,'representation':'SPARSE_CAUSAL_DELTAS_OVER_SPECIES_DEFAULTS',
      'site_count':SPEC.site_count,'fanout':SPEC.fanout,'fresh_numeric_bytes':fresh_bytes,
      'sparse_support_rows':sparse_support,'sparse_edge_weight_rows':sparse_weights,'checks':checks,
      'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_SPARSE_CAUSAL_STATE_'+('GREEN' if out['pass'] else 'RED'))
    print(json.dumps(out,indent=2,sort_keys=True))
    if not out['pass']:raise SystemExit(1)
if __name__=='__main__':main()
