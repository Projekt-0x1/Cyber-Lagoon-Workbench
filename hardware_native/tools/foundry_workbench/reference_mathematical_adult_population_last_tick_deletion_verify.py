#!/usr/bin/env python3
"""Host falsifier for deleting observational PopulationBankV1 last-tick state."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_population_v1 import PopulationBankV1,PopulationSpecV1
from language_development_v1 import LanguageDevelopmentV1

SPEC=PopulationSpecV1(131072,2,4,42,8)

def strip_observer(cp):
    out=copy.deepcopy(cp);out.pop('site_last_tick',None);return out

def future(bank,prepared_signature):
    occ=bank.activate_signature(prepared_signature,retain=True)
    bank.decay();settled=bank.settle(occ,1,True)
    cue=(501,502);mem=((1,bank.signature(cue)),(2,bank.signature((999,1000))))
    retrieved=bank.retrieve(cue,mem)
    return {
      'settled':settled,
      'retrieved':retrieved,
      'checkpoint':strip_observer(bank.checkpoint()),
    }

def main():
    started=time.perf_counter();b=PopulationBankV1(SPEC)
    prepared=b.prepare((11,22,33));b.recruit((101,202,303));b.decay()
    donor=b.checkpoint();compact=strip_observer(donor)
    legacy=copy.deepcopy(donor)
    sites=tuple(map(int,legacy.get('sites',())))
    if 'site_last_tick' not in legacy:legacy['site_last_tick']=[1 for _ in sites]
    forged=copy.deepcopy(legacy);forged['site_last_tick']=[0xffffffff for _ in sites]

    compact_restored=None;compact_error=''
    try:compact_restored=PopulationBankV1.restore(copy.deepcopy(compact))
    except Exception as exc:compact_error=f'{type(exc).__name__}:{exc}'
    legacy_restored=PopulationBankV1.restore(copy.deepcopy(legacy))
    forged_restored=PopulationBankV1.restore(copy.deepcopy(forged))

    legacy_future=future(legacy_restored,prepared)
    forged_future=future(forged_restored,prepared)
    compact_future=None if compact_restored is None else future(compact_restored,prepared)

    evidence_cut=copy.deepcopy(compact);evidence_cut.pop('site_support',None)
    support_cut_rejected=False
    try:
        cut=PopulationBankV1.restore(evidence_cut);future(cut,prepared)
    except Exception:support_cut_rejected=True

    raw=LanguageDevelopmentV1(4096);raw.population.prepare((7001,7002));raw_cp=raw.raw_ecology_checkpoint()
    raw_restored=LanguageDevelopmentV1.restore_raw_ecology(copy.deepcopy(raw_cp))
    raw_prepared=raw_cp['population'].get('prepared',())
    raw_rows_have_no_timestamp=all(len(row)==2 for row in raw_prepared)

    checks={
      'compact_checkpoint_deletes_site_last_tick':'site_last_tick' not in b.checkpoint(),
      'compact_restore_succeeds':compact_restored is not None,
      'forged_legacy_timestamp_has_no_future_authority':legacy_future==forged_future,
      'compact_future_matches_legacy':compact_future==legacy_future,
      'causal_support_negative_control_rejected':support_cut_rejected,
      'raw_language_checkpoint_deletes_timestamp':raw_rows_have_no_timestamp,
      'raw_language_restore_preserves_prepared_support':all(raw_restored.population.support[int(row[0])]==int(row[1]) for row in raw_prepared),
      'live_last_tick_array_deleted':not hasattr(b,'last_tick'),
      'numeric_baseline_at_most_4_bytes_per_site':b.numeric_allocation_bytes()<=SPEC.site_count*4,
      'bounded_fast_lane':time.perf_counter()-started<1.0,
    }
    result={'schema':'cyber-lagoon.reference-mathematical-adult-population-last-tick-deletion.v1','pass':all(checks.values()),
      'reference_only':True,'state_role':'OBSERVATIONAL','deleted_state':'PopulationBankV1.last_tick + checkpoint timestamps',
      'numeric_bytes':b.numeric_allocation_bytes(),'numeric_bytes_per_site':b.numeric_allocation_bytes()/SPEC.site_count,
      'site_rows':len(b.checkpoint().get('sites',())),'compact_restore_error':compact_error,
      'checks':checks,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_LAST_TICK_DELETION_'+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
