#!/usr/bin/env python3
"""Unfold a learned slot-call-slot Network from a current two-span frontier."""
from __future__ import annotations

from dataclasses import dataclass

from reference_resident_parametric_span_network_v1 import (
    NETWORK_ARITY, OP_CALL, OP_SLOT, ParametricSpanNetworkRefuse,
    ResidentParametricSpanNetworkV1, _identity, _strict_numeric,
)
from reference_resident_variable_span_v1 import MAX_SPAN


MAX_BYTES = 256
MAX_ROOTS = 1024


@dataclass(frozen=True)
class PartialNetworkByteAncestryV1:
    offset: int
    unit: int
    raw_contact_root: int
    sample_root: int
    contact_sequence: int
    source: int
    channel: int
    provenance: tuple[int, ...]
    span_occurrence_root: int
    span_recipe_root: int
    prediction_witness_root: int
    constructor_root: int
    constructor_evidence_root: int
    constructor_witness_roots: tuple[int, ...]
    network_position: int
    slot: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class ResidentPartialNetworkTrajectoryV1:
    identity: int
    constructor_root: int
    constructor_evidence_root: int
    left_recipe_root: int
    middle_recipe_root: int
    right_recipe_root: int
    left_occurrence_root: int
    middle_occurrence_root: int
    right_occurrence_root: int
    units: tuple[int, ...]
    constituent_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    work_units: int
    ancestry: tuple[PartialNetworkByteAncestryV1, ...]


def _unfold_resident_partial_network_v1(
        resident: ResidentParametricSpanNetworkV1,
        selected_constructor_root: int = 0) -> ResidentPartialNetworkTrajectoryV1:
    """Fill the learned CALL member; the host supplies neither Recipe nor bytes."""
    if not isinstance(resident, ResidentParametricSpanNetworkV1):
        raise ParametricSpanNetworkRefuse("partial_network:resident")
    base = resident._inner
    if base.pending:
        raise ParametricSpanNetworkRefuse("partial_network:pending")
    work = 0
    def charge(amount):
        nonlocal work
        amount=int(amount)
        if amount<0 or work+amount>resident.work_limit:
            raise ParametricSpanNetworkRefuse("partial_network:resource")
        work+=amount
    charge(len(base.samples)+len(base.span_occurrences)+len(base.recipes)
           +len(base.prediction_witnesses)+len(resident.constructors))
    if not base.samples:
        raise ParametricSpanNetworkRefuse("partial_network:frontier")

    newest = base.samples[-1]
    ordered = [row for row in base.samples
               if row.source == newest.source and row.channel == newest.channel]
    position = {row.identity:index for index,row in enumerate(ordered)}
    recipes = {(row.channel,row.length,row.span_hash):row
               for row in base.recipes.values()}
    frontier = []
    for occurrence in base.span_occurrences:
        charge(1)
        if (occurrence.source != newest.source or occurrence.channel != newest.channel
                or not occurrence.sample_roots):
            continue
        recipe = recipes.get((occurrence.channel,occurrence.length,occurrence.span_hash))
        if recipe is not None:
            frontier.append((occurrence,recipe))
    right = [(occ,recipe) for occ,recipe in frontier
             if occ.sample_roots[-1] == newest.identity]
    occurrence_by_root={row.identity:row for row in base.span_occurrences}
    recipe_by_root={row.identity:row for row in base.recipes.values()}
    charge(len(occurrence_by_root)+len(recipe_by_root))
    validated=[]
    for constructor in resident.constructors.values():
        charge(16+len(constructor.binding_pairs)+len(constructor.binding_pair_sources)
               +len(constructor.witness_network_roots)+len(constructor.source_roots))
        witness_networks=[resident.networks.get(root)
                          for root in constructor.witness_network_roots]
        if any(row is None for row in witness_networks):
            raise ParametricSpanNetworkRefuse("partial_network:constructor_witness")
        for root,network in zip(constructor.witness_network_roots,witness_networks):
            charge(12+len(network.member_occurrence_roots)+len(network.member_recipe_roots))
            occurrences=[occurrence_by_root.get(item)
                         for item in network.member_occurrence_roots]
            network_recipes=[recipe_by_root.get(item) for item in network.member_recipe_roots]
            if (network.identity!=root or len(network.member_occurrence_roots)!=NETWORK_ARITY
                    or len(network.member_recipe_roots)!=NETWORK_ARITY
                    or any(row is None for row in (*occurrences,*network_recipes))
                    or network.identity!=_identity(b"resident-parametric-span-network-v1",
                                                   network.member_occurrence_roots)
                    or any((occurrence.channel,occurrence.length,occurrence.span_hash)
                           !=(recipe.channel,recipe.length,recipe.span_hash)
                           for occurrence,recipe in zip(occurrences,network_recipes))
                    or any(occurrence.source!=network.source
                           or occurrence.channel!=network.channel
                           for occurrence in occurrences)
                    or network.born_sequence!=occurrences[-1].born_sequence
                    or network.covered_units!=sum(row.length for row in occurrences)
                    or network.evidence<=0):
                raise ParametricSpanNetworkRefuse("partial_network:constructor_witness")
        pairs=tuple(sorted({(row.member_recipe_roots[0],row.member_recipe_roots[2])
                            for row in witness_networks}))
        charge(max(1,len(pairs))*max(1,len(witness_networks)))
        pair_sources=tuple(sorted((left,right,tuple(sorted({row.source
            for row in witness_networks if (row.member_recipe_roots[0],
                row.member_recipe_roots[2])==(left,right)}))) for left,right in pairs))
        sources=tuple(sorted({row.source for row in witness_networks}))
        pair_hashes=tuple(sorted(resident._pair_hash(*pair) for pair in pairs))
        expected_evidence=_identity(b"resident-parametric-span-constructor-evidence-v1",(
            pair_hashes,sorted((left,right,sorted(values))
                for left,right,values in pair_sources),
            constructor.witness_network_roots,sources))
        if (constructor.operations!=((OP_SLOT,0),(OP_CALL,constructor.middle_recipe),
                                      (OP_SLOT,1))
                or constructor.arity!=NETWORK_ARITY
                or constructor.identity!=_identity(
                    b"resident-parametric-span-constructor-v1",
                    (NETWORK_ARITY,constructor.operations))
                or any(row.member_recipe_roots[1]!=constructor.middle_recipe
                       for row in witness_networks)
                or constructor.support!=len(witness_networks)
                or constructor.binding_pairs!=pairs
                or constructor.binding_pair_sources!=pair_sources
                or constructor.left_bindings!=tuple(sorted({row[0] for row in pairs}))
                or constructor.right_bindings!=tuple(sorted({row[1] for row in pairs}))
                or constructor.source_roots!=sources
                or constructor.binding_pair_hashes!=pair_hashes
                or constructor.evidence_root!=expected_evidence):
            raise ParametricSpanNetworkRefuse("partial_network:constructor")
        validated.append((constructor,set(constructor.binding_pairs),
            {(row[0],row[1]):row[2] for row in constructor.binding_pair_sources}))
    candidates = []
    for right_occ,right_recipe in right:
        right_start = position.get(right_occ.sample_roots[0],-1)
        for left_occ,left_recipe in frontier:
            charge(1)
            if position.get(left_occ.sample_roots[-1],-3)+1 != right_start:
                continue
            for constructor,binding_pairs,pair_sources in validated:
                charge(4+len(binding_pairs)*len(binding_pairs))
                pair=(left_recipe.identity,right_recipe.identity)
                if (constructor.middle_recipe
                        and left_recipe.identity in constructor.left_bindings
                        and right_recipe.identity in constructor.right_bindings
                        and pair not in binding_pairs
                        and resident._three_corner_support(binding_pairs,*pair)
                        and resident._cross_source_three_corner(pair_sources,*pair)):
                    if (not selected_constructor_root
                            or constructor.identity == selected_constructor_root):
                        candidates.append((constructor,left_occ,left_recipe,
                                           right_occ,right_recipe))
    unique={(row[0].identity,row[1].identity,row[3].identity):row for row in candidates}
    if len(unique)!=1:
        raise ParametricSpanNetworkRefuse("partial_network:ambiguous")
    constructor,left_occ,left_recipe,right_occ,right_recipe=next(iter(unique.values()))
    middle_recipe=base.recipes.get(constructor.middle_recipe)
    if middle_recipe is None:
        raise ParametricSpanNetworkRefuse("partial_network:middle_recipe")
    middle_occurrences=[row for row in base.span_occurrences
                        if row.identity in middle_recipe.occurrence_witness_roots]
    if len(middle_occurrences)!=len(middle_recipe.occurrence_witness_roots):
        raise ParametricSpanNetworkRefuse("partial_network:middle_currentness")
    sample_by_root={row.identity:row for row in base.samples}
    def materialize(occurrence,recipe):
        charge(12+len(occurrence.sample_roots)*(16+MAX_SPAN*4)
               +len(base.prediction_witnesses))
        try: samples=tuple(sample_by_root[root] for root in occurrence.sample_roots)
        except KeyError as exc:
            raise ParametricSpanNetworkRefuse("partial_network:sample") from exc
        units=tuple(row.unit for row in samples)
        if (len(units)!=recipe.length or any(not 0<=unit<=255 for unit in units)
                or _identity(b"variable-span-content-v1",units)!=recipe.span_hash
                or any(row.source!=occurrence.source or row.channel!=occurrence.channel
                    or row.source in base.withdrawn_sources
                    or row.contact_root!=_identity(b"variable-span-contact-v1",(
                        resident.session,row.contact_sequence,row.source,row.channel,
                        (row.unit,),row.provenance))
                    or row.identity!=_identity(b"variable-span-sample-v1",(
                        row.contact_root,row.contact_sequence)) for row in samples)
                or occurrence.identity!=_identity(b"variable-span-occurrence-v1",(
                    occurrence.channel,tuple(row.identity for row in samples)))):
            raise ParametricSpanNetworkRefuse("partial_network:witness")
        witnesses=[row for row in base.prediction_witnesses
                   if row.identity in recipe.prediction_witness_roots]
        if (len(witnesses)!=len(recipe.prediction_witness_roots)
                or any(row.difference<=0 or row.source in base.withdrawn_sources
                    or row.channel!=recipe.channel or row.source<=0
                    or row.source not in row.source_roots
                    or row.source not in recipe.source_roots
                    or row.observed_sample not in sample_by_root
                    or sample_by_root[row.observed_sample].source!=row.source
                    or sample_by_root[row.observed_sample].channel!=row.channel
                    or row.identity!=_identity(b"variable-span-prediction-witness-v1",(
                        row.ticket,row.observed_sample,row.difference)) for row in witnesses)):
            raise ParametricSpanNetworkRefuse("partial_network:prediction_witness")
        return samples,units,min(row.identity for row in witnesses)

    middle_materialized=[(row,*materialize(row,middle_recipe))
                         for row in middle_occurrences]
    if len({row[2] for row in middle_materialized})!=1:
        raise ParametricSpanNetworkRefuse("partial_network:middle_ambiguity")
    middle_occ,middle_samples,middle_units,middle_prediction=min(
        middle_materialized,key=lambda row:row[0].identity)
    parts=[]
    for position_index,(occ,recipe,cached) in enumerate(((left_occ,left_recipe,None),
            (middle_occ,middle_recipe,(middle_samples,middle_units,middle_prediction)),
            (right_occ,right_recipe,None))):
        samples,units,prediction=cached or materialize(occ,recipe)
        parts.append((position_index,occ,recipe,samples,units,prediction))
    units=tuple(unit for part in parts for unit in part[4])
    if not (1<=len(units)<=MAX_BYTES):
        raise ParametricSpanNetworkRefuse("partial_network:byte_bound")
    source_roots=tuple(sorted({*constructor.source_roots,
        *(source for part in parts for source in part[2].source_roots),
        *(part[1].source for part in parts)}))
    constituent_roots=tuple(sorted({constructor.identity,constructor.evidence_root,
        *constructor.witness_network_roots,
        *(part[1].identity for part in parts),*(part[2].identity for part in parts),
        *(part[5] for part in parts),
        *(sample.contact_root for part in parts for sample in part[3]),
        *(sample.identity for part in parts for sample in part[3])}))
    if (len(source_roots)>MAX_ROOTS or len(constituent_roots)>MAX_ROOTS
            or any(source in base.withdrawn_sources for source in source_roots)):
        raise ParametricSpanNetworkRefuse("partial_network:root_bound")
    charge(len(units)+len(source_roots)+len(constituent_roots)+len(units))
    ancestry=[]
    for position_index,occ,recipe,samples,_,prediction in parts:
        for sample in samples:
            ancestry.append(PartialNetworkByteAncestryV1(len(ancestry),sample.unit,
                sample.contact_root,sample.identity,sample.contact_sequence,sample.source,
                sample.channel,sample.provenance,occ.identity,recipe.identity,prediction,
                constructor.identity,constructor.evidence_root,
                constructor.witness_network_roots,position_index,
                0 if position_index==0 else 1 if position_index==2 else -1,source_roots))
    identity=_identity(b"resident-partial-network-trajectory-v1",(
        constructor.identity,constructor.evidence_root,left_recipe.identity,
        middle_recipe.identity,right_recipe.identity,left_occ.identity,
        middle_occ.identity,right_occ.identity,units,constituent_roots,source_roots,work,
        tuple(tuple(getattr(row,name) for name in row.__dataclass_fields__)
              for row in ancestry)))
    result=ResidentPartialNetworkTrajectoryV1(identity,constructor.identity,
        constructor.evidence_root,left_recipe.identity,middle_recipe.identity,
        right_recipe.identity,left_occ.identity,middle_occ.identity,right_occ.identity,
        units,constituent_roots,source_roots,work,tuple(ancestry))
    _strict_numeric(result,extent=MAX_ROOTS)
    return result


def unfold_resident_partial_network_v1(
        resident: ResidentParametricSpanNetworkV1) -> ResidentPartialNetworkTrajectoryV1:
    """Fill the uniquely eligible learned CALL member without a host selector."""
    return _unfold_resident_partial_network_v1(resident)
