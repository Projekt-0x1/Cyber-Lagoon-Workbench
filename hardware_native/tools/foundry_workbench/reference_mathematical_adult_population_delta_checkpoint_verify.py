#!/usr/bin/env python3
"""Host population checkpoint tournament: canonical Species substrate + sparse z_t deltas."""
from __future__ import annotations
import copy,json,sys,time
from array import array
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_population_v1 import PopulationBankV1,PopulationSpecV1

SPEC=PopulationSpecV1(32768,2,4,42,8)

def wire_bytes(x):return len(json.dumps(x,sort_keys=True,separators=(',',':')).encode())

def legacy_checkpoint(b):
    return {'schema':1,'spec':b.spec.__dict__,'tick':b.tick,'next_occurrence':b.next_occurrence,
            'support':b.support.tolist(),'last_tick':[0]*b.spec.site_count,'eligibility':list(b.eligibility),
            'edge_weight':b.edge_weight.tolist(),'edge_eligibility':list(b.edge_eligibility),
            'edge_target':[b.edge_target_value(edge) for edge in range(b.allocated_edge_count)],
            'territory':[b.territory_value(site) for site in range(b.spec.site_count)],
            'occurrences':[o.__dict__ for o in b.occurrences],
            'credit_events':b.credit_events,'revision_events':b.revision_events}

def lived_bank():
    b=PopulationBankV1(SPEC)
    for features,effect in (((101,201,301),1),((102,202,302),-1),((103,203,303),1)):
        o=b.recruit(features);b.settle(o,effect,True)
    live=b.recruit((777,888,999))
    return b,live

def main():
    started=time.perf_counter();b,live=lived_bank();legacy=legacy_checkpoint(b);cp=b.checkpoint()
    forbidden={'support','last_tick','eligibility','edge_target','territory'}
    restored=PopulationBankV1.restore(copy.deepcopy(cp))
    # One topology and one territory lesion are individual state, not Species defaults.
    lesion=PopulationBankV1.restore(copy.deepcopy(cp));features=(7001,7002);base_sig=lesion.signature(features)
    seed=lesion.feature_sites(features[0])[0];edge=seed*lesion.spec.fanout
    lesion.set_edge_target(edge,(lesion.edge_target_value(edge)+137)%lesion.spec.site_count)
    lesion.set_territory(seed,(lesion.territory_value(seed)+1)%lesion.spec.territory_count)
    lesion_sig=lesion.signature(features);lesion_cp=lesion.checkpoint();lesion_r=PopulationBankV1.restore(copy.deepcopy(lesion_cp))
    legacy_r=PopulationBankV1.restore(copy.deepcopy(legacy))
    # Future update after restore must remain identical.
    lo=restored.recruit((444,555));ro=legacy_r.recruit((444,555));lr=restored.settle(lo,1,True);rr=legacy_r.settle(ro,1,True)
    checks={
      'schema_two_sparse_checkpoint':cp.get('schema')==2,
      'full_default_arrays_deleted':not(forbidden & set(cp)) and len(cp.get('edge_weight',()))==len(cp.get('edges',())) and len(cp.get('edge_eligibility',()))==len(cp.get('edges',())),
      'material_compression':wire_bytes(cp)*20<wire_bytes(legacy),
      'restore_digest_exact':PopulationBankV1.restore(copy.deepcopy(cp)).digest()==b.digest(),
      'future_learning_exact':lr==rr and restored.digest()==legacy_r.digest(),
      'live_eligibility_survives':any(restored.eligibility),
      'learned_edge_weight_survives':any(int(x)!=1 for x in restored.edge_weight),
      'topology_lesion_changes_representation':lesion_sig!=base_sig,
      'topology_lesion_checkpointed':lesion_r.edge_target_value(edge)==lesion.edge_target_value(edge) and lesion_r.signature(features)==lesion_sig,
      'territory_lesion_checkpointed':lesion_r.territory_value(seed)==lesion.territory_value(seed),
      'lesions_are_sparse_overrides':len(lesion_cp.get('edge_target_overrides',()))==1 and len(lesion_cp.get('territory_overrides',()))==1,
      'legacy_full_checkpoint_restores':PopulationBankV1.restore(copy.deepcopy(legacy)).digest()==b.digest(),
      'legacy_canonicalizes_sparse':PopulationBankV1.restore(copy.deepcopy(legacy)).checkpoint().get('schema')==2,
      'bounded_seconds_lane':time.perf_counter()-started<2.0,
    }
    result={'schema':'cyber-lagoon.reference-mathematical-adult-population-delta-checkpoint.v1','pass':all(checks.values()),
      'reference_only':True,'representation':'CANONICAL_SPECIES_SUBSTRATE_PLUS_SPARSE_INDIVIDUAL_DELTAS',
      'legacy_checkpoint_bytes':wire_bytes(legacy),'compact_checkpoint_bytes':wire_bytes(cp),'bytes_saved':wire_bytes(legacy)-wire_bytes(cp),
      'compression_ratio':wire_bytes(cp)/wire_bytes(legacy),'site_rows':len(cp.get('sites',())),'edge_rows':len(cp.get('edges',())),
      'edge_target_overrides':len(cp.get('edge_target_overrides',())),'territory_overrides':len(cp.get('territory_overrides',())),
      'lesioned_checkpoint_bytes':wire_bytes(lesion_cp),'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_DELTA_CHECKPOINT_'+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
