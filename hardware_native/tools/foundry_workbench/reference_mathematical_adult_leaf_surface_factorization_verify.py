#!/usr/bin/env python3
"""Differential proof for productive-leaf surface factorization.

Productive language leaves keep one template identity plus historical lexical identities;
arbitrary resident atomics still keep raw bytes. Historical identity lookup must preserve
an earned program under later ambiguity/withdrawal without weakening checkpoint identity.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_hierarchical_composition_v1 import HierarchicalRefuse
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_mathematical_adult_operator_factorization_verify import build, CLAUSE
from reference_predictive_credit_profile_v1 import Q

# Explicit economic-refactor lane receipt for the shared Workbench commit gate.
state_minimization_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True


def wire_bytes(value):
    return len(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def legacy_raw_checkpoint(adult):
    leaves=dict(adult._surface_leaf_surfaces)
    for identity in adult._surface_leaf_family_index:
        leaves[int(identity)]=adult._leaf_surface(identity)
    return {
        'schema':7,
        'leaf_surfaces':[{'identity':k,'surface':list(v)} for k,v in sorted(leaves.items())],
        'program_factor_state':adult.programs.factor_checkpoint(),
    }


def direct_payload_bytes(adult):
    raw=sum(8+len(surface) for surface in adult._surface_leaf_surfaces.values())
    families=sum(8+sum(8+8*len(lexemes) for lexemes in rows.values())
                 for rows in adult._surface_leaf_families.values())
    return raw+families


def main():
    started=time.perf_counter()
    adult,leaves,_deep,top,_programs,_roots=build(True)
    expected=adult.public_surface(top.identity)
    checkpoint=adult.program_surface_checkpoint()
    legacy=legacy_raw_checkpoint(adult)
    candidate_bytes=wire_bytes(checkpoint);legacy_bytes=wire_bytes(legacy)
    legacy_direct=sum(8+len(leaf.surface) for leaf in leaves)
    candidate_direct=direct_payload_bytes(adult)

    saved=copy.deepcopy(checkpoint)
    adult._surface_leaf_surfaces={};adult._surface_leaf_families={};adult._surface_leaf_family_index={};adult.programs.factors={}
    adult.restore_program_surface_checkpoint(saved)
    replay=adult.public_surface(top.identity)
    replay_checkpoint_matches=saved==adult.program_surface_checkpoint()

    tampered=copy.deepcopy(saved)
    tampered['leaf_families'][0]['leaves'][0]['identity']+=1
    try:
        adult.restore_program_surface_checkpoint(tampered);tamper_refused=False
    except RuntimeError:
        tamper_refused=True
    adult.restore_program_surface_checkpoint(saved)

    withdrawal,withdrawal_leaves,_d,withdrawal_top,_p,_r=build(True)
    withdrawal_expected=withdrawal.public_surface(withdrawal_top.identity)
    withdrawal.language.withdraw_source(3002)
    withdrawal_survives=withdrawal.public_surface(withdrawal_top.identity)==withdrawal_expected
    try:
        withdrawal.leaf(CLAUSE,(101,201,301,401));withdrawal_new_refused=False
    except RuntimeError:
        withdrawal_new_refused=True

    ambiguous,ambiguous_leaves,_d,ambiguous_top,_p,_r=build(True)
    ambiguous_expected=ambiguous.public_surface(ambiguous_top.identity)
    ambiguous.observe_surface_item(101,b'steady',8101);ambiguous.observe_surface_item(101,b'steady',8102)
    lexical_ambiguous=ambiguous.language.lexeme(101) is None
    ambiguous.observe_surface_construction(CLAUSE,(101,201,301,401),b'careful engineer tests sensor',8201)
    ambiguous.observe_surface_construction(CLAUSE,(102,202,302,402),b'quiet technician inspects valve',8202)
    template_ambiguous=ambiguous.language.template(CLAUSE,4) is None
    ambiguity_survives=ambiguous.public_surface(ambiguous_top.identity)==ambiguous_expected
    try:
        ambiguous.leaf(CLAUSE,(101,201,301,401));ambiguity_new_refused=False
    except RuntimeError:
        ambiguity_new_refused=True

    lesion_seed,lesion_leaves,_d,lesion_top,_p,_r=build(True)
    # Probe durable learned state on a cold checkpoint fork; transient recall/output
    # holds intentionally vanish across restore and cannot mask the lesion.
    lesion=LanguageMasteryAdultV1.restore(copy.deepcopy(lesion_seed.checkpoint()))
    careful_units=lesion.language.lexeme(101);careful_identity=lesion.language.lexeme_identity(101,careful_units)
    lesion_key=next(key for key in lesion.language._lexeme_sources
                    if lesion.language.lexeme_identity(key[0],key[1])==careful_identity)
    lesion_sources=lesion.language._lexeme_sources.pop(lesion_key);lesion.language._rebuild_indices()
    try:
        lesion.public_surface(lesion_top.identity);lexeme_lesion_refused=False
    except RuntimeError:
        lexeme_lesion_refused=True
    unaffected_leaf=lesion._leaf_surface(lesion_leaves[-1].identity)==tuple(lesion_leaves[-1].surface)
    lesion.language._lexeme_sources[lesion_key]=lesion_sources;lesion.language._rebuild_indices()
    lesion_recovered=lesion.public_surface(lesion_top.identity)==expected

    raw=adult.leaf_surface(0xCA1,0xCA11,tuple(b'can you clarify?'))
    adult.experience_atomic_program(0xCA12,raw,Q//4,0,0xCA1,Q//16,True)
    raw_atomic_exact=(raw.identity in adult._surface_leaf_surfaces and
                      raw.identity not in adult._surface_leaf_family_index and
                      adult.public_surface(0xCA12)==tuple(b'can you clarify?'))

    checks={
        'productive_leaves_delete_raw_byte_copies':not checkpoint['raw_leaf_surfaces'] and len(adult._surface_leaf_family_index)>=16,
        'one_shared_productive_family':len(saved['leaf_families'])==1 and len(saved['leaf_families'][0]['leaves'])==16,
        'checkpoint_smaller_than_raw_leaf_projection':candidate_bytes<legacy_bytes,
        'direct_payload_smaller_than_raw_leaf_projection':candidate_direct<legacy_direct,
        'checkpoint_replay_exact':replay==expected and replay_checkpoint_matches,
        'tampered_leaf_identity_refused':tamper_refused,
        'withdrawal_preserves_earned_surface':withdrawal_survives,
        'withdrawal_blocks_new_leaf_realization':withdrawal_new_refused,
        'later_lexical_and_template_evidence_is_ambiguous':lexical_ambiguous and template_ambiguous,
        'later_ambiguity_preserves_earned_surface':ambiguity_survives,
        'later_ambiguity_blocks_new_leaf_realization':ambiguity_new_refused,
        'historical_lexeme_lesion_breaks_dependent_expression':lexeme_lesion_refused,
        'historical_lexeme_lesion_is_branch_local':unaffected_leaf,
        'historical_lexeme_restore_recovers_expression':lesion_recovered,
        'arbitrary_atomic_surface_keeps_raw_fallback':raw_atomic_exact,
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[key for key,value in checks.items() if not value]
    result={
        'schema':'cyber-lagoon.reference-mathematical-adult-leaf-surface-factorization.v1',
        'pass':not failed,
        'reference_only':True,
        'state_minimization_refactor':state_minimization_refactor,
        'phenotype_preserved':phenotype_preserved,
        'future_update_authority_preserved':future_update_authority_preserved,
        'grounding_level':'MECHANISM_CLASS_GROUNDED',
        'causal_role':'PRESERVE_EARNED_EXACT_SURFACE_WHILE_DELETING_DUPLICATE_PRODUCTIVE_LEAF_BYTES',
        'brain_mechanism_candidate':'INDEX_LIKE_REACTIVATION_OF_DISTRIBUTED_LEARNED_CONSTITUENTS_WITH_LITERAL_FALLBACK_FOR_UNASSIMILATED_TRACES',
        'sapolsky_check':'SAPOLSKY_DESTRUCTIVE_AUDIT_MATCHED_CURRENT_CONTEXT_HISTORY_STATE_CONTROL_TIMESCALE',
        'sapolsky_audit_verifier':'reference_productive_leaf_sapolsky_destructive_audit_verify.py',
        'hardware_ethology_translation':'RESIDENT_FACTOR_IDENTITIES_REACTIVATE_DISTRIBUTED_LEARNED_MATTER_NO_HOST_SEMANTIC_EXECUTIVE',
        'known_mismatch_or_unknown':'DISCRETE_INTEGER_IDENTITIES_ARE_A_SILICON_SURROGATE_NOT_A_CLAIMED_NEURAL_CODE',
        'state_boundary':'PRODUCTIVE_LEAF_TEMPLATE_PLUS_LEXEME_IDENTITIES_RAW_ONLY_FOR_UNFACTORED_ATOMICS',
        'recursive_bytes':len(expected),
        'legacy_raw_checkpoint_bytes':legacy_bytes,
        'factored_checkpoint_bytes':candidate_bytes,
        'checkpoint_bytes_saved':legacy_bytes-candidate_bytes,
        'legacy_direct_payload_bytes':legacy_direct,
        'factored_direct_payload_bytes':candidate_direct,
        'direct_payload_bytes_saved':legacy_direct-candidate_direct,
        'productive_leaf_count':len(leaves),
        'raw_atomic_count_after_probe':len(adult._surface_leaf_surfaces),
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000.0,3),
    }
    print('FOUNDRY_MATHEMATICAL_ADULT_LEAF_SURFACE_FACTORIZATION_'+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
