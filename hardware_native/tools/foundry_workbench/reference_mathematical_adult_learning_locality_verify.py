#!/usr/bin/env python3
"""Fast language-facing falsifier for local writes in the mathematical Adult.

A single independent confirmation after a provisional lexical exposure must update
only the directly implicated learned language relation state, without appending a
lifetime observer log, while changing several held-out future language compositions.
This measures the
operator/state-factorization goal rather than batch compression ratio.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1


CLAUSE = 9001
A1,A2,G1,G2,V1,V2,O1,O2 = 101,102,201,202,301,302,401,402
NEW = 999


def lexeme_rows(checkpoint):
    return {
        (int(row['feature']), tuple(map(int, row['units']))): tuple(map(int, row['sources']))
        for row in checkpoint['lexemes']
    }


def main():
    started=time.perf_counter(); adult=LanguageMasteryAdultV1()
    names={
        A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',
        V1:'tests',V2:'inspects',O1:'sensor',O2:'valve',
    }
    for feature,text in names.items():
        adult.observe_surface_item(feature,text.encode(),1000+feature)
        adult.observe_surface_item(feature,text.encode(),2000+feature)
    adult.observe_surface_construction(
        CLAUSE,(A1,G1,V1,O1),b'the careful engineer tests the sensor.',3001)
    adult.observe_surface_construction(
        CLAUSE,(A2,G2,V2,O2),b'the quiet technician inspects the valve.',3002)

    # First voice is provisional.  The existing productive construction cannot yet
    # use the new lexical item as outward language.
    adult.observe_surface_item(NEW,b'dax',7001)
    try:
        adult.leaf(CLAUSE,(A1,G1,V1,NEW))
        provisional_refused=False
    except Exception:
        provisional_refused=True

    language_before=copy.deepcopy(adult.language.checkpoint())
    factor_before=copy.deepcopy(adult.program_surface_checkpoint())
    program_counts_before=dict(adult.programs.counts)
    program_chunks_before=dict(adult.programs.chunks)
    credit_before=copy.deepcopy(adult.credit.snapshot())

    # This is the one new lived language event under measurement.
    adult.observe_surface_item(NEW,b'dax',7002)

    language_after=copy.deepcopy(adult.language.checkpoint())
    factor_after=copy.deepcopy(adult.program_surface_checkpoint())
    rows_before=lexeme_rows(language_before); rows_after=lexeme_rows(language_after)
    changed_rows=tuple(sorted(
        key for key in set(rows_before)|set(rows_after)
        if rows_before.get(key)!=rows_after.get(key)))

    probes=(
        (A1,G1,V1,NEW),
        (A2,G1,V2,NEW),
        (A1,G2,V2,NEW),
        (A2,G2,V1,NEW),
    )
    surfaces=tuple(bytes(adult.leaf(CLAUSE,atoms).surface).decode() for atoms in probes)
    distinct=len(set(surfaces))

    active_before=len(json.dumps(language_before['lexemes'],sort_keys=True,separators=(',',':')))
    active_after=len(json.dumps(language_after['lexemes'],sort_keys=True,separators=(',',':')))
    logical_growth=active_after-active_before

    checks={
        'provisional_single_source_does_not_publish_word':provisional_refused,
        'one_independent_confirmation_changes_one_lexeme_row':len(changed_rows)==1 and changed_rows[0][0]==NEW,
        'confirmation_does_not_append_resident_lifetime_log':'history' not in language_before and 'history' not in language_after,
        'unrelated_construction_store_unchanged_by_lexical_confirmation':factor_before==factor_after and not hasattr(adult,'hierarchy'),
        'causal_program_structure_unchanged_by_lexical_confirmation':program_counts_before==adult.programs.counts and program_chunks_before==adult.programs.chunks,
        'program_selection_credit_unchanged_by_lexical_confirmation':credit_before==adult.credit.snapshot(),
        'local_delta_unlocks_four_heldout_compositions':distinct==4 and all('dax' in surface for surface in surfaces),
        'active_lexical_growth_is_small':0<logical_growth<32,
        'behavioral_fanout_exceeds_changed_active_rows':distinct>len(changed_rows),
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[key for key,value in checks.items() if not value]
    if failed:
        raise SystemExit('FOUNDRY_MATHEMATICAL_ADULT_LEARNING_LOCALITY_RED '+','.join(failed))
    result={
        'contract':'FOUNDRY_MATHEMATICAL_ADULT_LEARNING_LOCALITY_GREEN',
        'reference_only':True,
        'language_phenotype_improved':True,
        'measured_event':'SECOND_INDEPENDENT_LEXICAL_CONFIRMATION',
        'changed_active_lexeme_rows':len(changed_rows),
        'resident_lifetime_history_rows':0,
        'active_lexical_json_growth_bytes':logical_growth,
        'heldout_language_fanout':distinct,
        'surfaces':surfaces,
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__':main()
