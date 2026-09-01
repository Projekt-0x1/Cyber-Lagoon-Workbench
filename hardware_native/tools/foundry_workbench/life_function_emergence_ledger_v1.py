#!/usr/bin/env python3
"""Observation-only phenotype ledger over canonical Life Function checkpoints.

The ledger defines no capability. It exposes what the newest shared Adult actually has
at every observer mark so peer changes cannot hide behind stale assay expectations.
"""
from __future__ import annotations
import copy,hashlib,json


def _root(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _longest_causal_component(rows):
    outgoing={};nodes=set()
    for row in rows:
        cause=int(row[2]);effect=int(row[3]);outgoing.setdefault(cause,set()).add(effect);nodes.update((cause,effect))
    memo={}
    def walk(node,seen=()):
        if node in seen:return 0
        if node in memo:return memo[node]
        children=outgoing.get(node,())
        value=0 if not children else 1+max(walk(child,seen+(node,)) for child in children)
        memo[node]=value;return value
    return max((walk(node) for node in nodes),default=0)

def _relation_public_frontier(adult):
    max_clauses=0;max_bytes=0;contexts=set()
    for rid in tuple(adult.relation_basis.active):
        seq=adult.relation_basis.expand_spaces(rid)
        if not seq:continue
        for context in adult.relation_language_contexts_for_order(seq):
            if int(context)>0:contexts.add(int(context))
        surface=adult.project_relation_basis(rid)
        if surface:
            raw=bytes(surface);max_clauses=max(max_clauses,raw.count(b'.'));max_bytes=max(max_bytes,len(raw))
    return max_clauses,max_bytes,len(contexts)

def _causal_public_frontier(adult,rows):
    if not rows:return 0,0
    # Choose one proposition in the longest component; resident conversation itself
    # rematerializes the whole unique component. Probe only a restored observer clone.
    by_node={}
    for row in rows:
        by_node[int(row[2])]=row;by_node[int(row[3])]=row
    for identity in tuple(by_node):
        try:raw=bytes(adult.language_adult._leaf_surface(identity))
        except Exception:continue
        clone=type(adult).restore(copy.deepcopy(adult.checkpoint()));out=clone.compose_causal_component(identity)[0]
        if out:return out.count(b'.'),len(out)
    return 0,0

def observe(mark,runtime):
    adult=runtime.adult;language=adult.language_adult.language;basis=adult.relation_basis
    active_lexemes=0
    for feature in language._lexeme_index:
        try:active_lexemes+=1 if language.lexeme(int(feature)) is not None else 0
        except Exception:pass
    active_spans=[]
    for rid in sorted(basis.active):
        seq=basis.expand_spaces(rid)
        if seq:active_spans.append((len(seq),basis.generation(rid)))
    winner=adult.operator_run_until_settled();projected=adult.project_operator_trace(winner.trace) if winner is not None else None
    world=adult.language_adult.world_causal_learning
    world_resolutions=tuple(world.current_resolutions())
    control_rows=[]
    for profile in adult.language_adult.credit.rows.values():
        if profile.control_history_q16>0:control_rows.append((int(profile.control_history_q16),int(profile.controllability_q16)))
        for local in profile.contexts.values():
            if local.control_history_q16>0:control_rows.append((int(local.control_history_q16),int(local.controllability_q16)))
    learned_control=max((row[0] for row in control_rows),default=0)
    matched_current=max((row[1] for row in control_rows if row[0]==learned_control),default=0)
    relation_clauses,relation_bytes,relation_contexts=_relation_public_frontier(adult)
    causal_clauses,causal_bytes=_causal_public_frontier(adult,world_resolutions)
    reliable_sources=sum(1 for source in world.testimony_accuracy if world.testimony_reliable(source))
    conflicts=len(world.current_testimony_conflicts()) if hasattr(world,'current_testimony_conflicts') else 0
    source_blocks=[world.complete_source_blocks(int(row[4])) for row in world_resolutions]
    causal_factors=sum(1 for factor in world.grounding.rows if world.grounding.orientation(factor)!=0) if hasattr(world,'grounding') else 0
    return {
        'mark':str(mark),'cursor':int(runtime.cursor),'adult_root':adult.digest(),
        'language':{
            'active_lexemes':active_lexemes,
            'support_epoch':int(language._support_epoch),
            'template_epoch':int(language._template_epoch),
            'checkpoint_root':_root(language.checkpoint()),
            'grounded_causal_operators':causal_factors,
        },
        'operator':{
            'needs':len(adult.operators.needs()),'revision':int(adult.operators.revision),
            'public_count':int(adult.operator_public_count),
            'winner_operations':0 if winner is None else int(winner.operations),
            'projected_clauses':0 if projected is None else int(bytes(projected).count(b'.')),
            'projected_bytes':0 if projected is None else len(projected),
        },
        'relations':{
            'active':len(basis.active),'retained':len(basis.relations),
            'max_generation':max((g for _n,g in active_spans),default=0),
            'expressible_widths':sorted(set(n for n,_g in active_spans)),
            'public_count':int(adult.relation_public_count),
            'checkpoint_root':_root(basis.checkpoint()),
            'max_public_clauses':relation_clauses,'max_public_bytes':relation_bytes,
            'discourse_contexts':relation_contexts,
        },
        'organism':{'somatic_pressure_q16':int(adult.language_adult.slow_resource_history.pressure_q16()),'learned_control_history_q16':learned_control,'current_controllability_at_max_history_q16':matched_current,'learned_control_rows':len(control_rows)},
        'world':{
            'resolved_relations':len(world_resolutions),
            'provisional_testimony_relations':len(world.current_testimony_resolutions()),
            'checkpoint_root':_root(world.checkpoint()),
            'causal_component_links':_longest_causal_component(world_resolutions),
            'minimum_source_blocks':min(source_blocks,default=0),
            'reliable_testimony_sources':reliable_sources,'testimony_conflicts':conflicts,
            'public_causal_clauses':causal_clauses,'public_causal_bytes':causal_bytes,
        },
    }


def ledger(mark_runtimes):
    rows=[observe(mark,runtime) for mark,runtime in mark_runtimes]
    return {'schema':'cyber-lagoon.life-function-emergence-ledger.v1','marks':rows,
            'ledger_root':_root(rows)}
