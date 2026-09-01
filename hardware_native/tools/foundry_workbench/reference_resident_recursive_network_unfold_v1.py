#!/usr/bin/env python3
"""Authored recursive fold over learned span Networks and current chronology."""
from __future__ import annotations

from dataclasses import dataclass,replace

from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse, ResidentParametricSpanNetworkV1, _identity,
    _strict_numeric,
)
from reference_resident_partial_network_unfold_v1 import (
    unfold_resident_partial_network_v1,
)


MAX_DEPTH=3
MAX_BYTES=512


@dataclass(frozen=True)
class RecursiveNetworkByteAncestryV1:
    offset:int
    unit:int
    leaf_rank:int
    role:int
    clause_index:int
    clause_root:int
    closure_rank:int
    closure_roots:tuple[int,...]
    raw_contact_root:int
    sample_root:int
    contact_sequence:int
    source:int
    channel:int
    provenance:tuple[int,...]
    span_occurrence_root:int
    span_recipe_root:int
    prediction_witness_root:int
    constructor_root:int
    constructor_evidence_root:int
    constructor_witness_roots:tuple[int,...]
    source_roots:tuple[int,...]


@dataclass(frozen=True)
class ResidentRecursiveNetworkTrajectoryV1:
    identity:int
    rank:int
    clause_roots:tuple[int,...]
    connective_recipe_roots:tuple[int,...]
    closure_roots:tuple[int,...]
    units:tuple[int,...]
    source_roots:tuple[int,...]
    work_units:int
    ancestry:tuple[RecursiveNetworkByteAncestryV1,...]


def unfold_resident_recursive_network_v1(
        resident:ResidentParametricSpanNetworkV1)->ResidentRecursiveNetworkTrajectoryV1:
    if not isinstance(resident,ResidentParametricSpanNetworkV1):
        raise ParametricSpanNetworkRefuse("recursive_network:resident")
    base=resident._inner
    if base.pending or not base.samples:
        raise ParametricSpanNetworkRefuse("recursive_network:pending_or_empty")
    work=0
    def charge(amount):
        nonlocal work
        if amount<0 or work+int(amount)>resident.work_limit:
            raise ParametricSpanNetworkRefuse("recursive_network:resource")
        work+=int(amount)
    charge(len(base.samples)+len(base.span_occurrences)+len(base.recipes)
           +len(base.prediction_witnesses)+len(resident.constructors))

    newest=base.samples[-1]
    samples=tuple(row for row in base.samples
                  if row.source==newest.source and row.channel==newest.channel)
    if not (1<=len(samples)<=MAX_BYTES):
        raise ParametricSpanNetworkRefuse("recursive_network:frontier_bound")
    positions={row.identity:index for index,row in enumerate(samples)}
    recipes={(row.channel,row.length,row.span_hash):row for row in base.recipes.values()}
    spans=[]
    for occurrence in base.span_occurrences:
        charge(1)
        if occurrence.source!=newest.source or occurrence.channel!=newest.channel:
            continue
        recipe=recipes.get((occurrence.channel,occurrence.length,occurrence.span_hash))
        if recipe is None or not occurrence.sample_roots: continue
        start=positions.get(occurrence.sample_roots[0]);end=positions.get(occurrence.sample_roots[-1])
        if start is not None and end is not None and end-start+1==len(occurrence.sample_roots):
            spans.append((start,end+1,occurrence,recipe))
    by_start={}
    for row in spans: by_start.setdefault(row[0],[]).append(row)
    paths={0:[((),(0,0,0))]}
    for start in range(len(samples)):
        for path,score in paths.get(start,()):
            for row in by_start.get(start,()):
                charge(len(path)+1)
                recipe=row[3]
                candidate=(path+(row,),(
                    score[0]+recipe.length,score[1]+recipe.prediction_gain,
                    score[2]+recipe.retention_margin))
                paths.setdefault(row[1],[]).append(candidate)
    complete=paths.get(len(samples),())
    if not complete:
        raise ParametricSpanNetworkRefuse("recursive_network:unsegmented")
    peak=max(score for _,score in complete)
    winners={tuple(row[2].identity for row in path):path
             for path,score in complete if score==peak}
    if len(winners)!=1:
        raise ParametricSpanNetworkRefuse("recursive_network:segmentation_ambiguous")
    segments=next(iter(winners.values()))

    def eligible(left,right):
        rows=[]
        for constructor in resident.constructors.values():
            charge(4+len(constructor.binding_pairs)**2)
            pair=(left[3].identity,right[3].identity)
            sources={(row[0],row[1]):row[2] for row in constructor.binding_pair_sources}
            if (pair not in constructor.binding_pairs
                    and left[3].identity in constructor.left_bindings
                    and right[3].identity in constructor.right_bindings
                    and resident._three_corner_support(set(constructor.binding_pairs),*pair)
                    and resident._cross_source_three_corner(sources,*pair)):
                rows.append(constructor)
        return rows

    groups=[];connectives=[];index=0
    while index<len(segments):
        if index+1>=len(segments):
            raise ParametricSpanNetworkRefuse("recursive_network:incomplete_clause")
        choices=eligible(segments[index],segments[index+1])
        if len(choices)!=1:
            raise ParametricSpanNetworkRefuse("recursive_network:clause_ambiguous")
        groups.append((segments[index],segments[index+1],choices[0]));index+=2
        if index<len(segments):
            connectives.append(segments[index]);index+=1
    if len(groups) not in (2,3) or len(connectives)!=len(groups)-1:
        raise ParametricSpanNetworkRefuse("recursive_network:depth")

    # The final pair exercises the already-hostile-reviewed constructor and
    # live-witness validator over this exact resident state.
    original_limit=resident.work_limit
    resident.work_limit=original_limit-work
    try:
        final=unfold_resident_partial_network_v1(resident)
    finally:
        resident.work_limit=original_limit
    charge(final.work_units)
    if (final.left_occurrence_root,final.right_occurrence_root)!=(
            groups[-1][0][2].identity,groups[-1][1][2].identity):
        raise ParametricSpanNetworkRefuse("recursive_network:final_pair")

    sample_by_root={row.identity:row for row in base.samples}
    witness_by_root={row.identity:row for row in base.prediction_witnesses}
    def materialize(segment):
        _,_,occurrence,recipe=segment
        charge(12+len(occurrence.sample_roots)*8+len(base.prediction_witnesses))
        try: rows=tuple(sample_by_root[root] for root in occurrence.sample_roots)
        except KeyError as exc:
            raise ParametricSpanNetworkRefuse("recursive_network:sample") from exc
        units=tuple(row.unit for row in rows)
        witnesses=[witness_by_root.get(root) for root in recipe.prediction_witness_roots]
        if (len(units)!=recipe.length or _identity(b"variable-span-content-v1",units)!=recipe.span_hash
                or recipe.identity!=_identity(b"variable-span-recipe-v1",(
                    recipe.channel,recipe.length,recipe.span_hash))
                or occurrence.identity!=_identity(b"variable-span-occurrence-v1",(
                    occurrence.channel,occurrence.sample_roots))
                or any(not 0<=unit<=255 for unit in units)
                or any(row is None or row.difference<=0 or row.channel!=recipe.channel
                       or row.source in base.withdrawn_sources or row.source<=0
                       or row.source not in row.source_roots
                       or row.source not in recipe.source_roots
                       or row.observed_sample not in sample_by_root
                       or sample_by_root[row.observed_sample].source!=row.source
                       or sample_by_root[row.observed_sample].channel!=row.channel
                       or row.identity!=_identity(b"variable-span-prediction-witness-v1",(
                           row.ticket,row.observed_sample,row.difference))
                       for row in witnesses)
                or any(row.source!=occurrence.source or row.channel!=occurrence.channel
                       or row.contact_root!=_identity(b"variable-span-contact-v1",(
                           resident.session,row.contact_sequence,row.source,row.channel,
                           (row.unit,),row.provenance))
                       or row.identity!=_identity(b"variable-span-sample-v1",(
                           row.contact_root,row.contact_sequence)) for row in rows)):
            raise ParametricSpanNetworkRefuse("recursive_network:witness")
        return rows,units,min(row.identity for row in witnesses)

    recipe_by_root={row.identity:row for row in base.recipes.values()}
    materialized={}
    def middle_part(constructor):
        recipe=recipe_by_root.get(constructor.middle_recipe)
        if recipe is None:
            raise ParametricSpanNetworkRefuse("recursive_network:middle_recipe")
        candidates=[row for row in base.span_occurrences
                    if row.identity in recipe.occurrence_witness_roots]
        if len(candidates)!=len(recipe.occurrence_witness_roots):
            raise ParametricSpanNetworkRefuse("recursive_network:middle_currentness")
        parts=[(-1,-1,row,recipe) for row in candidates]
        rows=[materialize(part) for part in parts]
        if len({row[1] for row in rows})!=1:
            raise ParametricSpanNetworkRefuse("recursive_network:middle_ambiguity")
        chosen=min(zip(parts,rows),key=lambda row:row[0][2].identity)
        materialized[chosen[0][2].identity]=chosen[1]
        return chosen[0]

    units=[];ancestry=[];clause_roots=[];connective_roots=[];source_roots=set()
    for clause_index,(left,right,constructor) in enumerate(groups):
        parts=(left,middle_part(constructor),right)
        clause_units=[];clause_ancestry=[]
        for position,part in enumerate(parts):
            rows,part_units,prediction=materialized.get(part[2].identity) or materialize(part)
            role=0 if position==0 else 1 if position==2 else -1
            for sample in rows:
                clause_ancestry.append((sample,part[2],part[3],prediction,role))
            clause_units.extend(part_units);source_roots.update(part[3].source_roots)
            source_roots.update(row.source for row in rows)
        clause_root=_identity(b"resident-recursive-network-clause-v1",(
            constructor.identity,constructor.evidence_root,
            constructor.witness_network_roots,left[2].identity,right[2].identity,
            tuple(clause_units),tuple((sample.identity,occurrence.identity,
                recipe.identity,prediction,role) for sample,occurrence,recipe,
                prediction,role in clause_ancestry)))
        clause_roots.append(clause_root)
        for sample,occurrence,recipe,prediction,role in clause_ancestry:
            ancestry.append(RecursiveNetworkByteAncestryV1(len(units),sample.unit,1,role,
                clause_index,clause_root,0,(),
                sample.contact_root,sample.identity,sample.contact_sequence,sample.source,
                sample.channel,sample.provenance,occurrence.identity,recipe.identity,prediction,
                constructor.identity,constructor.evidence_root,
                constructor.witness_network_roots,tuple(sorted(source_roots))))
            units.append(sample.unit)
        if clause_index<len(connectives):
            rows,join_units,prediction=materialize(connectives[clause_index])
            connective_roots.append(connectives[clause_index][3].identity)
            source_roots.update(connectives[clause_index][3].source_roots)
            source_roots.update(row.source for row in rows)
            for sample in rows:
                ancestry.append(RecursiveNetworkByteAncestryV1(len(units),sample.unit,1,2,
                    clause_index,0,0,(),
                    sample.contact_root,sample.identity,sample.contact_sequence,sample.source,
                    sample.channel,sample.provenance,connectives[clause_index][2].identity,
                    connectives[clause_index][3].identity,prediction,0,0,
                    (),tuple(sorted(source_roots))))
                units.append(sample.unit)
    closures=[];current=clause_roots[0]
    for index,other in enumerate(clause_roots[1:]):
        current=_identity(b"resident-recursive-network-closure-v1",(
            current,connective_roots[index],other,index+2))
        closures.append(current)
    if len(units)>MAX_BYTES:
        raise ParametricSpanNetworkRefuse("recursive_network:byte_bound")
    source_tuple=tuple(sorted(source_roots));rank=len(groups)
    ancestry=tuple(replace(row,closure_rank=rank,
        closure_roots=tuple(closures[(row.clause_index if row.role==2
                                      else max(0,row.clause_index-1)):]),
        source_roots=source_tuple) for row in ancestry)
    identity=_identity(b"resident-recursive-network-trajectory-v1",(
        rank,tuple(clause_roots),tuple(connective_roots),tuple(closures),
        tuple(units),source_tuple,work,
        tuple(tuple(getattr(row,name) for name in row.__dataclass_fields__)
              for row in ancestry)))
    result=ResidentRecursiveNetworkTrajectoryV1(identity,rank,tuple(clause_roots),
        tuple(connective_roots),tuple(closures),tuple(units),source_tuple,work,
        ancestry)
    _strict_numeric(result)
    return result
