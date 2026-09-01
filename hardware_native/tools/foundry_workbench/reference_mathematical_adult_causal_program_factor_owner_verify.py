#!/usr/bin/env python3
"""RED/contrast: public program factors belong to the generic causal-program ecology.

The language Adult may interpret an opaque integer factor as a leaf/template reference,
but it must not own a second program->factor table. Binding or replaying a factor must
not alter generic causal chunk structure, prediction, credit, or executability.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_causal_program_chunk_v1 import CausalChunkBankV1
from reference_mathematical_adult_operator_factorization_verify import build
from reference_predictive_credit_profile_v1 import Q


def main():
    started=time.perf_counter()
    adult,leaves,_deep,top,programs,_roots=build(True)
    pid=top.identity
    before=adult.public_surface(pid)
    factor=int(adult.programs.factor(pid))

    generic=CausalChunkBankV1();chunk=None
    for n in range(3):
        chunk=generic.observe((11,22,33),10+n*5,13+n*5,Q//4,Q//2,Q//16,successor=77)
    if chunk is None:raise RuntimeError('factor-owner:generic_chunk')
    structure_before=copy.deepcopy(generic.chunks)
    prediction_before=copy.deepcopy(generic.predictive.snapshot())
    generic.bind_factor(chunk.identity,0x12345)
    generic_checkpoint=generic.factor_checkpoint()
    structure_after=copy.deepcopy(generic.chunks)
    prediction_after=copy.deepcopy(generic.predictive.snapshot())
    generic.factors={}
    generic.restore_factor_checkpoint(generic_checkpoint)

    checkpoint=adult.program_surface_checkpoint()
    adult.programs.factors={}
    adult.restore_program_surface_checkpoint(checkpoint)
    replay=adult.public_surface(pid)
    restored_factor=int(adult.programs.factor(pid))

    removed=adult.programs.factors.pop(pid)
    factor_lesion_surface=adult.public_surface(pid)
    chunk_survives=pid in adult.programs.chunks
    credit_survives=adult.credit.row(pid).outcome_samples>0
    adult.programs.bind_factor(pid,removed)
    rebound=adult.public_surface(pid)

    checks={
        'adult_has_no_surface_shadow_namespace':not hasattr(adult,'surface'),
        'recursive_factor_moved_without_value_change':factor==restored_factor==int(adult.programs.factor(pid)),
        'factor_checkpoint_owned_by_program_bank':checkpoint.get('program_factor_state')==adult.programs.factor_checkpoint(),
        'factor_checkpoint_replays_exact_surface':replay==before and rebound==before,
        'focal_factor_lesion_blocks_only_public_transduction':factor_lesion_surface is None and chunk_survives and credit_survives,
        'generic_factor_binding_preserves_chunk_structure':structure_before==structure_after,
        'generic_factor_binding_preserves_prediction_state':prediction_before==prediction_after,
        'generic_factor_checkpoint_replays':generic.factor(chunk.identity)==0x12345,
        'generic_factor_is_opaque_integer':all(type(v) is int and v!=0 for v in generic.factors.values()),
        'no_language_fields_in_generic_factor_checkpoint':set(generic_checkpoint)=={'schema','factors'} and all(set(row)=={'program','factor'} for row in generic_checkpoint['factors']),
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={
        'schema':'cyber-lagoon.reference-mathematical-adult-causal-program-factor-owner.v1',
        'pass':not failed,
        'reference_only':True,
        'state_boundary':'GENERIC_CAUSAL_PROGRAM_OWNS_OPAQUE_FACTOR_ADULT_INTERPRETS_IT',
        'programs':len(programs),
        'recursive_bytes':len(before),
        'factor':factor,
        'generic_factor':generic.factor(chunk.identity),
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000.0,3),
    }
    print('FOUNDRY_MATHEMATICAL_ADULT_CAUSAL_PROGRAM_FACTOR_OWNER_'+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
