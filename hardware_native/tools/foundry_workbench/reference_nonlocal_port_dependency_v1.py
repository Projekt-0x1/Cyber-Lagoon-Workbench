#!/usr/bin/env python3
"""Bounded numeric reference hypothesis for learned cross-port dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

MAX_OBSERVATIONS = 4096
# PopulationSpecV1 permits 16 sites per feature; the resident signature adds
# one propagated site and a scene entity carries at most 16 bounded features.
MAX_SITES = 16 * 16 * 2
MAX_SURFACE = 64


class Refuse(ValueError):
    pass


def _ints(values, limit):
    out = tuple(values)
    if len(out) > limit or any(type(value) is not int or value < 0 for value in out):
        raise Refuse("bounded numeric state required")
    return out


def _identity(parts):
    raw = json.dumps(parts, separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(b"nonlocal-port-v1\0" + raw).digest()[:8], "big") or 1


@dataclass
class ObservationV1:
    occurrence: int
    source: int
    exemplar: int
    context: int
    controller_port: int
    target_port: int
    controller_sites: tuple[int, ...]
    strip_suffix: int
    append: tuple[int, ...]
    active: int = 1


@dataclass(frozen=True)
class ResultV1:
    units: tuple[int, ...]
    recipe_identity: int
    contributors: tuple[int, ...]
    byte_ancestry: tuple[tuple[int, ...], ...]
    alternatives: int
    support: int = 0


class NonlocalPortDependencyV1:
    def __init__(self):
        self.observations: list[ObservationV1] = []
        self._incidence: dict[tuple[int,int,int], list[int]] = {}
        self.last_touches = 0

    def _index_row(self,row):
        key=(int(row.context),int(row.controller_port),int(row.target_port))
        self._incidence.setdefault(key,[]).append(len(self.observations)-1)

    def observe(self, occurrence, source, exemplar, context, controller_port,
                target_port, controller_sites, base, target):
        numeric = (occurrence, source, exemplar, context, controller_port, target_port)
        if any(type(value) is not int or value <= 0 for value in numeric):
            raise Refuse("positive identities required")
        sites = _ints(controller_sites, MAX_SITES)
        base, target = _ints(base, MAX_SURFACE), _ints(target, MAX_SURFACE)
        if not sites or len(self.observations) >= MAX_OBSERVATIONS:
            raise Refuse("observation bound")
        shared = 0
        while shared < min(len(base), len(target)) and base[shared] == target[shared]:
            shared += 1
        strip_suffix, append = len(base) - shared, target[shared:]
        if strip_suffix == 0 and not append:
            raise Refuse("identity edit")
        self.observations.append(ObservationV1(
            occurrence, source, exemplar, context, controller_port, target_port,
            tuple(sorted(set(sites))), strip_suffix, append));self._index_row(self.observations[-1])

    def candidates(self, context, controller_port, target_port, controller_sites, base):
        sites, base = set(_ints(controller_sites, MAX_SITES)), _ints(base, MAX_SURFACE)
        ids=self._incidence.get((int(context),int(controller_port),int(target_port)),());self.last_touches=len(ids);groups={}
        for idx in ids:
            row=self.observations[idx]
            if not row.active:continue
            groups.setdefault((row.strip_suffix,row.append),[]).append(row)
        out=[]
        for (strip_suffix,append),rows in groups.items():
            sources={row.source for row in rows};exemplars={row.exemplar for row in rows};shared=set(rows[0].controller_sites)
            for row in rows[1:]:shared.intersection_update(row.controller_sites)
            if len(sources)<2 or len(exemplars)<2 or not shared or not shared<=sites or strip_suffix>len(base):continue
            units=base[:len(base)-strip_suffix]+append;contributors=tuple(sorted(row.occurrence for row in rows));identity=_identity([context,controller_port,target_port,sorted(shared),strip_suffix,list(append)]);ancestry=tuple(contributors for _ in units)
            out.append(ResultV1(units,identity,contributors,ancestry,0,len(rows)))
        out.sort(key=lambda row:(-row.support,row.recipe_identity));n=len(out)
        return tuple(ResultV1(row.units,row.recipe_identity,row.contributors,row.byte_ancestry,n,row.support) for row in out)

    def apply(self, context, controller_port, target_port, controller_sites, base):
        candidates=self.candidates(context,controller_port,target_port,controller_sites,base)
        if not candidates:return None
        peak=candidates[0].support;top=[row for row in candidates if row.support==peak]
        if len({row.units for row in top})!=1:return None
        return top[0]

    def withdraw_source(self, source):
        for row in self.observations:
            if row.source == source:
                row.active = 0

    def restore_source(self, source):
        for row in self.observations:
            if row.source == source:
                row.active = 1

    def checkpoint(self):
        return json.dumps({"schema": 1, "observations": [asdict(row) for row in self.observations]},
                          separators=(",", ":"), sort_keys=True)

    @classmethod
    def restore(cls, blob):
        body = json.loads(blob)
        if set(body) != {"schema", "observations"} or body["schema"] != 1:
            raise Refuse("checkpoint schema")
        out = cls()
        for row in body["observations"]:
            if set(row) != set(ObservationV1.__dataclass_fields__):
                raise Refuse("checkpoint fields")
            row["controller_sites"] = _ints(row["controller_sites"], MAX_SITES)
            row["append"] = _ints(row["append"], MAX_SURFACE)
            out.observations.append(ObservationV1(**row));out._index_row(out.observations[-1])
        if len(out.observations) > MAX_OBSERVATIONS:
            raise Refuse("checkpoint bound")
        return out
