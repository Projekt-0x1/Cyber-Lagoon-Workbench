#!/usr/bin/env python3
"""A/B falsifier for procedural Species topology plus sparse lesion overrides."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_population_v1 import PopulationBankV1,PopulationSpecV1,mix64

SPEC=PopulationSpecV1(131072,2,4,42,8)

def canonical(spec,edge):
    site,lane=divmod(int(edge),spec.fanout);n=spec.site_count
    delta=1+(mix64(site*0xD6E8FEB86659FD93+lane)%max(2,min(n-1,257)))
    target=(site+delta+mix64((lane+1)*0xA0761D6478BD642F+site)%n)%n
    return (site+1)%n if target==site else target

def edge_value(bank,edge):
    if hasattr(bank,'edge_target_value'):return int(bank.edge_target_value(edge))
    return int(bank.edge_target[edge])

def territory_value(bank,site):
    if hasattr(bank,'territory_value'):return int(bank.territory_value(site))
    return int(bank.territory[site])

def numeric_bytes(bank):
    total=bank.numeric_allocation_bytes()
    if hasattr(bank,'edge_target'):total+=len(bank.edge_target)*bank.edge_target.itemsize
    if hasattr(bank,'territory'):total+=len(bank.territory)
    return total

def main():
    started=time.perf_counter();t=time.perf_counter_ns();b=PopulationBankV1(SPEC);init_ms=(time.perf_counter_ns()-t)/1e6
    probes=(0,1,17,1234,SPEC.site_count*SPEC.fanout-1)
    canonical_exact=all(edge_value(b,e)==canonical(SPEC,e) for e in probes)
    features=tuple(range(1001,1101));t=time.perf_counter_ns()
    for r in range(2000):b.signature((features[r%100],features[(r+17)%100]))
    signature_us=((time.perf_counter_ns()-t)/2000)/1000.0
    # Lesion must remain individual state and survive sparse checkpoint/replay.
    f=(7001,7002);base=b.signature(f);seed=b.feature_sites(f[0])[0];edge=seed*b.spec.fanout
    old=edge_value(b,edge);b.set_edge_target(edge,(old+137)%b.spec.site_count)
    old_territory=territory_value(b,seed);b.set_territory(seed,(old_territory+1)%b.spec.territory_count)
    lesioned=b.signature(f);cp=b.checkpoint();r=PopulationBankV1.restore(copy.deepcopy(cp))
    checks={
      'canonical_edge_function_exact':canonical_exact,
      'canonical_territory_function_exact':all(territory_value(r,s)==(s%b.spec.territory_count) for s in (0,1,19,1000) if s!=seed),
      'materialized_edge_target_deleted':not hasattr(b,'edge_target'),
      'materialized_territory_deleted':not hasattr(b,'territory'),
      'numeric_bytes_per_site_at_most_11':numeric_bytes(b)<=SPEC.site_count*11,
      'topology_lesion_changes_signature':lesioned!=base,
      'topology_lesion_replays':edge_value(r,edge)==edge_value(b,edge) and r.signature(f)==lesioned,
      'territory_lesion_replays':territory_value(r,seed)==territory_value(b,seed),
      'checkpoint_sparse_overrides':len(cp.get('edge_target_overrides',()))==1 and len(cp.get('territory_overrides',()))==1,
      'signature_under_50us':signature_us<50.0,
      'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    result={'schema':'cyber-lagoon.reference-mathematical-adult-population-procedural-topology.v1','pass':all(checks.values()),
      'reference_only':True,'representation':'IMPLICIT_SPECIES_TOPOLOGY_PLUS_SPARSE_LESION_OVERRIDES',
      'site_count':SPEC.site_count,'fanout':SPEC.fanout,'numeric_bytes':numeric_bytes(b),
      'numeric_bytes_per_site':numeric_bytes(b)/SPEC.site_count,'init_ms':round(init_ms,3),
      'signature_us_avg':round(signature_us,3),'checkpoint_bytes':len(json.dumps(cp,separators=(',',':')).encode()),'checks':checks,
      'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_PROCEDURAL_TOPOLOGY_'+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
