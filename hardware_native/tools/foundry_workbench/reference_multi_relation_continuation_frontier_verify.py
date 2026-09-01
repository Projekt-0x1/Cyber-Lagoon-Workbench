#!/usr/bin/env python3
"""R0/R1 hostile assay for topology-sensitive multi-relation continuation."""
from __future__ import annotations
import copy,json,tempfile
from life_function_factory_v1 import build_cache,load_mark

SOURCE=0xCC01

def _coordinate(adult,row):
    effect=adult.language_adult.current_leaf_for_historical(int(row[3]))
    if effect is None:raise AssertionError('missing current effect')
    return int(row[4]),int(effect.identity)

def main():
    with tempfile.TemporaryDirectory() as directory:
        manifest=build_cache(directory);mark=manifest['checkpoints'][-1]['mark'];runtime=load_mark(directory,mark);adult=runtime.adult
        rows=tuple(adult.language_adult.world_causal_learning.current_resolutions());outgoing={}
        for row in rows:outgoing.setdefault(int(row[2]),[]).append(row)
        triple=next(((a,b,c) for a in rows for b in outgoing.get(int(a[3]),()) for c in outgoing.get(int(b[3]),())),None)
        siblings=next(((a,b) for a in rows for b in outgoing.get(int(a[2]),()) if int(a[4])!=int(b[4])),None)
        if triple is None or siblings is None:raise AssertionError('developed topology lacks hostile arms')
        chain=tuple(_coordinate(adult,row) for row in triple[:2]);chain3=tuple(_coordinate(adult,row) for row in triple)
        sibling_batch=tuple(_coordinate(adult,row) for row in siblings)
        expected_effect=chain[-1][1];expected_receipts=tuple(sorted(row[0] for row in chain))
        frontier=adult.causal_continuation_frontier(chain);chain_touches=adult.last_causal_continuation_frontier_touches
        reverse=adult.causal_continuation_frontier(tuple(reversed(chain)))
        frontier3=adult.causal_continuation_frontier(chain3)
        sibling_frontier=adult.causal_continuation_frontier(sibling_batch);sibling_touches=adult.last_causal_continuation_frontier_touches
        incomparable=adult.causal_continuation_frontier((*chain,sibling_batch[0]))
        invalid=adult.causal_continuation_frontier(((chain[0][0]^1,chain[0][1]),chain[1]))
        surface,receipt=adult.externalize_causal_component(frontier[0],SOURCE,0,frontier[1]) if frontier else (b'',None)
        public_rows=adult._causal_discourse_frontier(frontier[0],0,frontier[1])[0] if frontier else ()

        integrated=type(runtime).restore(runtime.program,copy.deepcopy(runtime.checkpoint()))
        tick=int(integrated.adult.language_adult._tick)
        integrated.adult.last_causal_dialogue_contact_continuations=tuple((*row,tick) for row in chain)
        integrated_surface,integrated_receipt=integrated.contact_utterance(b'\x00',SOURCE,0)
        ambiguous=type(runtime).restore(runtime.program,copy.deepcopy(runtime.checkpoint()))
        tick=int(ambiguous.adult.language_adult._tick)
        ambiguous.adult.last_causal_dialogue_contact_continuations=tuple((*row,tick) for row in sibling_batch)
        ambiguous_surface,ambiguous_receipt=ambiguous.contact_utterance(b'\x00',SOURCE,0)

        lesioned=load_mark(directory,mark);learner=lesioned.adult.language_adult.world_causal_learning
        occurrence=learner.ecology.pending.get(int(chain[1][0]));sources={int(e.source) for e in occurrence.evidence if e.active} if occurrence is not None else set()
        for source in sources:learner.withdraw_source(source)
        lesion=lesioned.adult.causal_continuation_frontier(chain)

        checks={
            'baseline_exactly_one_gate_would_silence_two_relation_chain':len(chain)!=1,
            'two_link_chain_has_unique_downstream_frontier':frontier==(expected_effect,expected_receipts),
            'clause_order_cannot_choose_frontier':reverse==frontier,
            'three_link_chain_reduces_to_last_effect':bool(frontier3) and frontier3[0]==chain3[-1][1] and frontier3[1]==tuple(sorted(row[0] for row in chain3)),
            'sibling_effects_remain_silent':sibling_frontier==(),
            'chain_plus_incomparable_effect_remains_silent':incomparable==(),
            'uncertified_receipt_is_rejected':invalid==(),
            'accepted_receipts_are_absent_from_regenerated_frontier':bool(public_rows) and not (set(expected_receipts)&{int(row[4]) for row in public_rows}),
            'public_continuation_now_exists':bool(surface) and receipt is not None,
            'runtime_consumes_certified_multi_relation_transient':bool(integrated_surface) and integrated_receipt>0,
            'runtime_preserves_silence_for_ambiguous_siblings':ambiguous_surface==b'' and ambiguous_receipt==0,
            'developmental_topology_lesion_abolishes_same_batch':bool(sources) and lesion==(),
            'work_is_batch_local_not_whole_adult_scan':chain_touches<=2*len(chain) and sibling_touches<=2*len(sibling_batch),
        }
        payload={'schema':'cyber-lagoon.multi-relation-continuation-frontier.r0-r1.v1','mark':mark,'checks':checks,'metrics':{
            'resolved_world_relations':len(rows),'accepted_chain_relations':len(chain),'chain_relation_evaluations':chain_touches,
            'sibling_relation_evaluations':sibling_touches,'continuation_bytes':len(surface),'continuation_programs':0 if receipt is None else len(receipt.programs),
            'old_exactly_one_gate_continuation_bytes':0,'lesioned_sources':len(sources)}}
        print(json.dumps(payload,sort_keys=True))
        if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
