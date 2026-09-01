#!/usr/bin/env python3
"""Contact-only assay for surface-concord selection by learned Networks."""
from __future__ import annotations

import hashlib, inspect, json, time
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))

from reference_resident_channel_sequence_grounding_v1 import (
    admit_channel_sequence_boundary_v1,
)
from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse, ResidentParametricSpanNetworkV1,
)
from reference_resident_parametric_span_network_verify import acquire, arrange, put
from reference_resident_partial_network_unfold_v1 import (
    unfold_resident_partial_network_v1,
)


def b(value): return tuple(value.encode("utf-8"))

ALICE,CAROL,THEY,THOSE=map(b,("Alice","Carol","They","Those"))
SEES,SEE,BOB,DAVE=map(b,(" sees "," see ","Bob","Dave"))


def refuses(fn,fragment=""):
    try: fn()
    except (ParametricSpanNetworkRefuse,TypeError,ValueError) as exc:
        return fragment in str(exc)
    return False


def history_contains(machine,needle):
    groups={}
    for row in machine._inner.samples:
        groups.setdefault((row.source,row.channel),[]).append(row.unit)
    return any(any(tuple(units[i:i+len(needle)])==needle
                   for i in range(len(units)-len(needle)+1))
               for units in groups.values())


def feed(machine,boundary,source,*spans):
    for span in spans:
        for unit in span: put(boundary,machine,source,unit)


def main():
    started=time.perf_counter();checks={}
    boundary=admit_channel_sequence_boundary_v1()
    machine=ResidentParametricSpanNetworkV1(boundary)
    spans=(ALICE,CAROL,THEY,THOSE,SEES,SEE,BOB,DAVE)
    for index,span in enumerate(spans):
        acquire(boundary,machine,span,240+index)
    checks["all_surface_spans_acquired_from_numeric_contacts"]=(
        all(any(machine._inner._units(occurrence)==span
                and any((recipe.channel,recipe.length,recipe.span_hash)==(
                    occurrence.channel,occurrence.length,occurrence.span_hash)
                    for recipe in machine._inner.recipes.values())
                for occurrence in machine._inner.span_occurrences) for span in spans))

    for source,left,middle,right in (
        (1001,ALICE,SEES,BOB),(1001,ALICE,SEES,DAVE),(1002,CAROL,SEES,DAVE),
        (1101,THEY,SEE,BOB),(1101,THEY,SEE,DAVE),(1102,THOSE,SEE,DAVE)):
        arrange(boundary,machine,source,left,middle,right)
    constructors=tuple(machine.constructors.values())
    checks["two_compact_surface_concord_networks_recruited"]=(
        len(constructors)==2 and {row.middle_recipe for row in constructors}=={
            next(recipe.identity for recipe in machine._inner.recipes.values()
                 if any(machine._inner._units(occ)==SEES and
                        (occ.channel,occ.length,occ.span_hash)==(
                            recipe.channel,recipe.length,recipe.span_hash)
                        for occ in machine._inner.span_occurrences)),
            next(recipe.identity for recipe in machine._inner.recipes.values()
                 if any(machine._inner._units(occ)==SEE and
                        (occ.channel,occ.length,occ.span_hash)==(
                            recipe.channel,recipe.length,recipe.span_hash)
                        for occ in machine._inner.span_occurrences))})
    singular=CAROL+SEES+BOB;plural=THOSE+SEE+BOB
    checks["heldout_complete_surfaces_absent_from_contact_history"]=(
        not history_contains(machine,singular) and not history_contains(machine,plural))

    feed(machine,boundary,1201,CAROL,BOB)
    frozen=machine.checkpoint();singular_out=unfold_resident_partial_network_v1(machine)
    checks["heldout_singular_slots_select_learned_singular_inflection"]=(
        singular_out.units==singular
        and tuple(row.slot for row in singular_out.ancestry)==(
            (0,)*len(CAROL)+(-1,)*len(SEES)+(1,)*len(BOB)))
    checks["same_complete_state_reexecutes_exact_surface_and_ancestry"]=(
        machine.checkpoint()==frozen
        and unfold_resident_partial_network_v1(machine)==singular_out)

    feed(machine,boundary,1202,THOSE,BOB)
    plural_out=unfold_resident_partial_network_v1(machine)
    checks["heldout_plural_slots_select_distinct_learned_inflection"]=(
        plural_out.units==plural and plural_out.constructor_root!=singular_out.constructor_root)
    singular_constructor=next(row for row in machine.constructors.values()
                              if row.identity==singular_out.constructor_root)
    plural_constructor=next(row for row in machine.constructors.values()
                            if row.identity==plural_out.constructor_root)
    checks["counterfactual_wrong_inflection_network_cannot_bind_same_slots"]=(
        singular_out.left_recipe_root not in plural_constructor.left_bindings
        and plural_out.left_recipe_root not in singular_constructor.left_bindings
        and singular_constructor.middle_recipe!=plural_constructor.middle_recipe)
    checks["outputs_are_utf8_only_after_resident_network_selection"]=(
        bytes(singular_out.units).decode("utf-8")=="Carol sees Bob"
        and bytes(plural_out.units).decode("utf-8")=="Those see Bob")

    feed(machine,boundary,1204,b("Nora"),BOB)
    unfamiliar_state=machine.checkpoint()
    checks["unfamiliar_slot_has_no_host_fallback"]=(
        refuses(lambda:unfold_resident_partial_network_v1(machine),"ambiguous")
        and machine.checkpoint()==unfamiliar_state)

    singular_source=1001
    withdrawal=boundary.seal_withdrawal(machine.session,machine.next_sequence,
                                        1999,7,singular_source)
    machine.ingest_withdrawal(withdrawal)
    feed(machine,boundary,1205,CAROL,BOB)
    checks["singular_network_source_withdrawal_silences_singular"]=(
        refuses(lambda:unfold_resident_partial_network_v1(machine),"ambiguous"))
    feed(machine,boundary,1206,THOSE,BOB)
    checks["disjoint_plural_network_survives_singular_withdrawal"]=(
        unfold_resident_partial_network_v1(machine).units==plural)

    forbidden={"grammar","agreement","number","person","tense","semantic",
               "expected","surface","output","prompt","language"}
    public={name.lower() for name in dir(ResidentParametricSpanNetworkV1)
            if not name.startswith("_")}
    checks["runtime_has_no_grammar_or_semantic_selection_api"]=(
        not forbidden.intersection(public)
        and tuple(inspect.signature(unfold_resident_partial_network_v1).parameters)==("resident",))
    checks["bounded_state_and_cpu_runtime"]=(
        len(machine._inner.samples)<=576 and machine.work<=machine.work_limit
        and time.perf_counter()-started<60)
    failed=sorted(name for name,value in checks.items() if not value)
    if failed: raise SystemExit("FOUNDRY_REFERENCE_RESIDENT_STRUCTURAL_AGREEMENT_RED "+",".join(failed))

    path=Path(__file__);elapsed=time.perf_counter()-started
    receipt={"contract":"FOUNDRY_REFERENCE_RESIDENT_STRUCTURAL_AGREEMENT_GREEN",
      "claim":"CONTACT_LEARNED_SURFACE_CONCORD_NETWORK_SELECTION_REFERENCE_ONLY",
      "reference_only":True,"adult_attached":False,"runtime_llm":False,
      "fixture_known_expected_bytes":True,"host_fixture_current_slot_selection":True,
      "host_output_or_middle_selection":False,
      "complete_outputs_seen_in_training":False,"human_language_claim":False,
      "agreement_semantics_claim":False,"causal_claim":False,"graph_flip":False,
      "physical_direct_parity":"NOT_RUN/RED",
      "production_ir":"ResidentRecipeIrProgram.vcurrent",
      "translation_status":"UNDEFINED","parity_status":"NOT_RUN/RED",
      "runtime_limit_seconds":60,"elapsed_ms":round(elapsed*1000,3),
      "resident_samples":len(machine._inner.samples),
      "checks":checks,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
      "remaining_red":["GROUNDING_OF_AGREEMENT","MULTILINGUAL_ACQUISITION",
        "RECURSIVE_MULTI_CLAUSE_USE","CONSEQUENCE_CONDITIONED_PARTNER_STYLE",
        "SOURCE_REACQUISITION","PUBLIC_BODY_CHANNEL",
        "PRODUCTION_RECIPE_IR_TRANSLATION","DIRECT_PHYSICAL_PARITY"]}
    print("FOUNDRY_REFERENCE_RESIDENT_STRUCTURAL_AGREEMENT_GREEN")
    print(json.dumps(receipt,sort_keys=True,indent=2))


if __name__=="__main__":main()
