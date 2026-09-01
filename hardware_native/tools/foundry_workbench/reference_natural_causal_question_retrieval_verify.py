#!/usr/bin/env python3
"""Natural questions orient resident causality through learned construction."""
from __future__ import annotations

import copy
import tempfile
import time

from life_function_factory_v1 import build_cache, load_mark
from reference_life_function_curriculum_v1 import canonical_species_program_v2
from reference_life_extension_causal_depth_plus_v1 import A_ROOTS, A_WILT


def verify_runtime(runtime, before):
    adult = type(runtime.adult).restore(copy.deepcopy(runtime.adult.checkpoint()))
    before_language = before.adult.language_adult.language
    learned_language = runtime.adult.language_adult.language
    new_spans = set(learned_language._span_sources) - set(before_language._span_sources)
    new_span_sources = tuple(
        sorted(learned_language._span_sources[key]) for key in new_spans)

    rows = tuple(adult.language_adult.world_causal_learning.current_resolutions())
    candidates = []
    for row in rows:
        effect = int(row[3])
        signature = adult.language_adult.leaf_signature(effect)
        if signature is None or tuple(signature[1]) in ((A_ROOTS,), (A_WILT,)):
            continue
        surface = bytes(adult.language_adult._leaf_surface(effect) or b"")
        closure = adult.causal_focus_rows(effect)
        if surface and closure:
            candidates.append((len(closure), effect, surface))
    if not candidates:
        return {"natural_causal_question_has_lived_heldout_focus":False}, {}

    _depth, effect, proposition = max(candidates)
    def fork():
        return type(runtime).restore(runtime.program, copy.deepcopy(runtime.checkpoint()))
    def contact_surface(subject,raw,source):
        return subject.contact_utterance(raw,source)[0]

    declarative = contact_surface(fork(),proposition,0xFA01)
    questions = (
        b"why is it the case that " + proposition.rstrip(b".?").lower() + b"?",
        b"why " + proposition.rstrip(b".?").lower() + b"?",
        b"the case why is it that " + proposition.rstrip(b".?").lower() + b"?",
    )
    replies = tuple(
        contact_surface(fork(),question,0xFA10 + index)
        for index, question in enumerate(questions)
    )
    before_subject=type(before).restore(before.program,copy.deepcopy(before.checkpoint()))
    before_reply = contact_surface(before_subject,questions[0],0xFA20)
    restarted_reply = contact_surface(fork(),questions[0],0xFA21)
    withdrawn = fork()
    for source in (0xFA11, 0xFA12):
        withdrawn.adult.language.withdraw_source(source)
    withdrawn_question = contact_surface(withdrawn,questions[0],0xFA22)
    withdrawn_declarative = contact_surface(withdrawn,proposition,0xFA23)

    checks={
        "natural_causal_question_has_lived_heldout_focus":bool(proposition),
        "learned_declarative_control_is_retrievable":bool(declarative),
        "heldout_natural_causal_question_orients_same_certified_closure":replies[0]==declarative,
        "question_construction_requires_developmental_experience":not before_reply,
        "unearned_short_question_remains_silent":not replies[1],
        "same_word_bag_malformed_question_does_not_route":not replies[2],
        "question_contact_adds_no_whole_proposition_template":(
            learned_language._template_sources == before_language._template_sources),
        "question_contact_adds_one_generic_unary_span":(
            len(new_spans)==1
            and next(iter(new_spans))[1]==1
            and tuple(piece.kind for piece in next(iter(new_spans))[2])==(1,2,1)
            and new_span_sources==([0xFA11,0xFA12],)),
        "heldout_question_is_not_a_stored_surface_binding":(
            not learned_language.invert_surface(questions[0])),
        "question_construction_survives_restart_without_transcript":restarted_reply==declarative,
        "question_evidence_withdrawal_silences_question_not_declarative":(
            not withdrawn_question and withdrawn_declarative==declarative),
    }
    phenotype={
        "heldout_focus_identity":hex(effect),
        "heldout_focus_surface":proposition.decode(errors="replace"),
        "natural_question":questions[0].decode(errors="replace"),
        "natural_reply":replies[0].decode(errors="replace"),
        "question_reply_bytes":tuple(map(len,replies)),
        "pre_experience_reply_bytes":len(before_reply),
        "restart_reply_bytes":len(restarted_reply),
        "withdrawn_question_reply_bytes":len(withdrawn_question),
        "withdrawn_declarative_reply_bytes":len(withdrawn_declarative),
        "new_span_count":len(new_spans),
        "new_span_sources":new_span_sources,
    }
    return checks,phenotype


def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="foundry-natural-question-") as directory:
        manifest = build_cache(directory)
        marks = tuple(row["mark"] for row in manifest["checkpoints"])
        learned_mark = "natural_causal_question_experience"
        learned_index = marks.index(learned_mark)
        if learned_index <= 0:
            raise SystemExit("assay-invalid:question-experience-order")
        runtime = load_mark(directory, learned_mark)
        before = load_mark(directory, marks[learned_index - 1])
        checks,phenotype=verify_runtime(runtime,before)
    failed=sorted(name for name,passed in checks.items() if not passed)
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    print("FOUNDRY_NATURAL_CAUSAL_QUESTION_"+("GREEN" if not failed else "RED"))
    print({"checks":checks,"phenotype":phenotype,"elapsed_ms":elapsed,"failed":failed})
    if failed:raise SystemExit(1)


if __name__ == "__main__":
    main()
