#pragma once

// ---------------------------------------------------------------------------
// ANSWER-SIDE CAUSAL COMPATIBILITY -- a NEW field-identity signal, own name,
// own falsifier, NO inheritance of claims from the retired one-coordinate
// question-gap learner.
//
// The prior acquisition rule for "which relation field does this question
// request" required the question's and answer's extracted 4-tuple relation
// triples to differ in EXACTLY ONE coordinate. Measured across four corpora
// spanning expository prose, children's narrative, drama, and Socratic
// dialogue -- including material chosen specifically to maximize
// question/answer parallelism -- that condition fires ZERO times in over
// 12,000 evaluated pairs; mass sits entirely at 3-4 differing coordinates.
// The mechanism was retired on its own audit-stated retirement criterion,
// not patched.
//
// This header implements the audit's named replacement, unchanged in
// design intent: for each candidate field f, substitute the answer-side
// value into field f of the question-activated relation and ask whether
// the RESULT is resident-supported. The field whose substitution best
// closes the question-conditioned relation is the field the answer filled.
// Field identity comes from which substitution the store already supports,
// never from an authored reading of a wh-word -- no interrogative unit is
// ever inspected here.
//
// PREREQUISITE VERIFIED (per the audit): relation_triple_lookup is keyed on
// four plain uint32 unit ids with no event handle or slot index, so any
// candidate tuple can be synthesised and probed -- exactly what this needs,
// and exactly the retrieval path production already exercises for
// counterfactual substitution elsewhere (count_category_mates_kernel).
//
// WIRED 2026-08-14 (0X1-156): the vote is now the LIVE source of
// requested_field inside form_witnessed_relation_plan_kernel, replacing the
// retired learn_question_gap_fields_kernel/question_gap_field_support path
// at its one real production consumer. It does not touch
// question_answer_construction (a separate, still-unwired downstream
// mechanism sharing the retired writer's gate) -- that remains out of
// scope, unchanged.
//
// IMPLEMENTATION NOTE: CausalCompatibilityVote, vote_causal_compatibility_
// field, and requested_field_from_causal_compatibility are defined in
// bcc32_cuda_resident_construction_composer.cuh itself now, immediately
// after relation_triple_lookup, not in this file. Reason: composer.cuh's own
// tail (bcc32_cuda_resident_construction_composer_tail.inl, included at its
// end) needs to CALL requested_field_from_causal_compatibility from inside
// form_witnessed_relation_plan_kernel, while this header has always included
// composer.cuh to get RelationTriple/relation_triple_lookup -- a genuine
// mutual dependency, not a naming accident. Keeping the definitions split
// across two files that each include the other works only for whichever
// file happens to be the translation unit's FIRST include (pragma once
// silently skips the recursive one, before it reaches definitions the
// second file needs) -- verified broken in exactly that way when this
// header's OWN contract test (which includes only this file) tried to build
// against composer_tail.inl's call site. Defining the primitive at its
// natural non-circular home removes the hazard outright; this header keeps
// its full audit narrative and simply re-exposes the symbols by including
// composer.cuh, so every existing include of THIS file (this header's own
// contract test included) keeps compiling unchanged.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "bcc32_cuda_resident_construction_composer.cuh"
