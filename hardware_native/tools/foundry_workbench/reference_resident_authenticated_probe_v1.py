#!/usr/bin/env python3
"""Authenticated reference probe wrapped around resident two-edge prediction.

This is a custody assay, not a physical-causality claim.  Learned structural
state originates an opaque probe command; a separate boundary authenticates
body acceptance and later raw scalar returns.  Only the wrapped structural
prediction witness changes relative action eligibility.
"""
from __future__ import annotations

import copy
from dataclasses import (asdict, dataclass, fields as dataclass_fields,
                         is_dataclass)
import hashlib
import hmac
import json
import secrets

import reference_resident_composite_cue_prediction_v1 as composite_module
import reference_resident_parametric_span_network_v1 as parametric_module
import reference_resident_variable_span_v1 as variable_module

from reference_resident_channel_sequence_grounding_v1 import (
    ReferenceChannelSequenceBoundaryV1,
)
from reference_resident_recursive_frontier_v1 import (
    OccurrenceContactV1, WithdrawalContactV1,
)
from reference_resident_composite_cue_prediction_v1 import (
    NODE_COMPOSITE, CompositeCueRefuse, ResidentCompositeCuePredictionV1,
    StructuralPredictionTicketV1,
)
from reference_resident_parametric_span_network_v1 import ResidentParametricSpanNetworkV1
from reference_resident_variable_span_v1 import ResidentVariableSpanV1


SCHEMA_VERSION, CHECKPOINT_VERSION = 0x41505231, 4
MAX_EVENTS, MAX_EVIDENCE = 4096, 512
MAX_WORK, MAX_CHECKPOINT_BYTES = 32768, 8 << 20
MAX_ROUTE_SOURCES = 16
MAX_SURFACE_ROUTES = 32
MAX_SURFACE_EVIDENCE_BYTES = 2 << 20
PROBE_HORIZON = 20
EVENT_SAMPLE, EVENT_TICK, EVENT_APPLY, EVENT_RETURN, EVENT_WITHDRAWAL, EVENT_MAPPING = range(1, 7)


class AuthenticatedProbeRefuse(RuntimeError):
    pass


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(domain: bytes, values) -> int:
    value = int.from_bytes(hashlib.sha256(
        domain + b"\0" + _canonical(values)).digest()[:8], "little") & ((1 << 63) - 1)
    return value or 1


def _numeric(value, path="$", extent=MAX_EVENTS):
    if (isinstance(value, bool) or value is None
            or isinstance(value, (str, bytes, bytearray, memoryview))):
        raise AuthenticatedProbeRefuse(f"probe:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise AuthenticatedProbeRefuse(f"probe:integer:{path}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _numeric(getattr(value, name), f"{path}.{name}", extent)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > extent:
            raise AuthenticatedProbeRefuse(f"probe:bound:{path}")
        for index, item in enumerate(value):
            _numeric(item, f"{path}[{index}]", extent)
        return
    raise AuthenticatedProbeRefuse(f"probe:type:{path}")


@dataclass(frozen=True)
class EventV1:
    kind: int
    values: tuple


@dataclass(frozen=True)
class ActionRecipeV1:
    identity: int
    capability: int
    structural_cost: int
    resource_cost: int


def _action_recipe(capability, structural_cost, resource_cost):
    values = (int(capability), int(structural_cost), int(resource_cost))
    return ActionRecipeV1(_identity(b"authenticated-probe-action-recipe-v1", values),
                          *values)


# Authored generic body capabilities, never stored in language/structural IR.
_AUTHORED_ACTIONS = (_action_recipe(1, 1, 1), _action_recipe(2, 2, 1))
ACTION_RECIPES = _AUTHORED_ACTIONS  # read-only assay alias; residents copy the tuple.


def _valid_action(row):
    return (isinstance(row, ActionRecipeV1) and row.capability > 0
            and row.structural_cost > 0 and row.resource_cost > 0
            and row.identity == _action_recipe(
                row.capability, row.structural_cost, row.resource_cost).identity)


@dataclass(frozen=True)
class BodyMappingContractV1:
    identity: int
    session: int
    device: int
    body_epoch: int
    port: int
    entries: tuple[tuple[int, int], ...]
    route_root: int
    auth_tag: int

    def signed_fields(self):
        return (self.identity, self.session, self.device, self.body_epoch,
                self.port, self.entries, self.route_root)


@dataclass(frozen=True)
class SurfaceResponseContractV1:
    identity: int
    mapping_root: int
    entries: tuple[tuple[int, int], ...]
    route_root: int
    auth_tag: int

    def signed_fields(self):
        return (self.identity, self.mapping_root, self.entries, self.route_root)


@dataclass(frozen=True)
class ProbeCommandV1:
    identity: int
    episode: int
    ticket: int
    ticket_envelope_root: int
    session: int
    incarnation: int
    action_recipe: int
    capability: int
    command_value: int
    mapping_root: int
    selection_state_root: int
    candidate_set_root: int
    resource_before: int
    resource_cost: int
    opened_sequence: int
    deadline_sequence: int
    opened_resident_tick: int
    deadline_resident_tick: int

    def content_fields(self):
        return (self.episode, self.ticket, self.ticket_envelope_root,
                self.session, self.incarnation,
                self.action_recipe, self.capability, self.command_value,
                self.mapping_root, self.selection_state_root,
                self.candidate_set_root, self.resource_before,
                self.resource_cost, self.opened_sequence,
                self.deadline_sequence, self.opened_resident_tick,
                self.deadline_resident_tick)


@dataclass(frozen=True)
class BodyApplyReceiptV1:
    identity: int
    command_hash: int
    mapping_root: int
    device: int
    body_epoch: int
    port: int
    apply_tick: int
    accepted: int
    auth_tag: int

    def signed_fields(self):
        return (self.identity, self.command_hash, self.mapping_root, self.device,
                self.body_epoch, self.port, self.apply_tick, self.accepted)


@dataclass(frozen=True)
class SurfaceApplyReceiptV1:
    identity: int
    apply_receipt: int
    command_hash: int
    mapping_root: int
    action_recipe: int
    response_contract_root: int
    trial_root: int
    constructor_root: int
    trajectory_root: int
    unit_digest: int
    ancestry_root: int
    apply_tick: int
    auth_tag: int

    def signed_fields(self):
        return (self.identity, self.apply_receipt, self.command_hash,
                self.mapping_root, self.action_recipe,
                self.response_contract_root,
                self.trial_root, self.constructor_root, self.trajectory_root,
                self.unit_digest, self.ancestry_root, self.apply_tick)


@dataclass(frozen=True)
class BodyReturnContactV1:
    identity: int
    apply_receipt: int
    command_hash: int
    mapping_root: int
    device: int
    body_epoch: int
    apply_tick: int
    arrival_tick: int
    occurrence: OccurrenceContactV1
    auth_tag: int

    def signed_fields(self):
        return (self.identity, self.apply_receipt, self.command_hash,
                self.mapping_root, self.device, self.body_epoch,
                self.apply_tick, self.arrival_tick,
                self.occurrence.signed_fields(), self.occurrence.auth_tag)


@dataclass(frozen=True)
class ProbeEvidenceV1:
    identity: int
    episode: int
    action_recipe: int
    command_hash: int
    mapping_root: int
    apply_receipt: int
    structural_ticket: int
    structural_witness: int
    difference: int
    return_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass
class PendingProbeV1:
    episode: int
    ticket: StructuralPredictionTicketV1
    command: ProbeCommandV1
    apply_receipt: BodyApplyReceiptV1 | None = None
    return_roots: tuple[int, ...] = ()
    return_source_roots: tuple[int, ...] = ()
    last_arrival_tick: int = 0


@dataclass(frozen=True)
class ProbeTickResultV1:
    identity: int
    inner_result_root: int
    command: ProbeCommandV1 | None


_BODY_ADMISSION = object()
_SURFACE_AUTHORITY = object()
_INNER_TYPES = (ResidentCompositeCuePredictionV1,
                ResidentParametricSpanNetworkV1, ResidentVariableSpanV1)
_INNER_KEYS = (
    ("actual_nodes", "current_epoch", "deferred_sample_roots", "events",
     "hypotheses", "last_eligible_sequence", "next_ticket",
     "nomination_eligible", "nominations", "pending", "recipes", "trace",
     "witnesses", "work", "work_limit"),
    ("constructors", "events", "networks", "processed_networks", "rebinds",
     "trace", "work", "work_limit"),
    ("deferred_sample_roots", "events", "incarnation", "next_sequence",
     "next_ticket", "pending", "prediction_witnesses", "recipes", "samples",
     "session", "span_occurrences", "trace", "withdrawn_sources", "work",
     "work_limit"),
)
_STATE_TYPES = tuple(sorted({value for module in (
    composite_module, parametric_module, variable_module)
    for value in vars(module).values()
    if isinstance(value, type) and is_dataclass(value)},
    key=lambda value: (value.__module__, value.__name__)))
_STATE_TYPE_INDEX = {value: index for index, value in enumerate(_STATE_TYPES)}
MAX_STATE_NODES = 1 << 20


def _state_encode(value, budget, depth=0):
    budget[0] += 1
    if budget[0] > MAX_STATE_NODES or depth > 64:
        raise AuthenticatedProbeRefuse("probe:inner_snapshot_bound")
    if value is None: return [0]
    if isinstance(value, bool): return [1, int(value)]
    if isinstance(value, int):
        _numeric(value); return [2, value]
    if isinstance(value, tuple):
        return [3, [_state_encode(row, budget, depth + 1) for row in value]]
    if isinstance(value, list):
        return [4, [_state_encode(row, budget, depth + 1) for row in value]]
    if isinstance(value, dict):
        rows = [(_state_encode(key, budget, depth + 1),
                 _state_encode(item, budget, depth + 1))
                for key, item in value.items()]
        rows.sort(key=lambda row: _canonical(row[0]))
        return [5, [[key, item] for key, item in rows]]
    if isinstance(value, set):
        rows = [_state_encode(row, budget, depth + 1) for row in value]
        rows.sort(key=_canonical); return [6, rows]
    kind = type(value)
    if is_dataclass(value) and kind in _STATE_TYPE_INDEX:
        return [7, _STATE_TYPE_INDEX[kind],
                [_state_encode(getattr(value, row.name), budget, depth + 1)
                 for row in dataclass_fields(kind)]]
    raise AuthenticatedProbeRefuse("probe:inner_snapshot_type")


def _state_decode(value, budget, depth=0):
    budget[0] += 1
    if (budget[0] > MAX_STATE_NODES or depth > 64
            or not isinstance(value, list) or not value
            or isinstance(value[0], bool) or not isinstance(value[0], int)):
        raise AuthenticatedProbeRefuse("probe:inner_snapshot_shape")
    tag = value[0]
    if tag == 0 and len(value) == 1: return None
    if tag == 1 and len(value) == 2 and value[1] in (0, 1): return bool(value[1])
    if tag == 2 and len(value) == 2:
        _numeric(value[1]); return int(value[1])
    if tag in (3, 4, 6) and len(value) == 2 and isinstance(value[1], list):
        rows = [_state_decode(row, budget, depth + 1) for row in value[1]]
        return tuple(rows) if tag == 3 else (rows if tag == 4 else set(rows))
    if tag == 5 and len(value) == 2 and isinstance(value[1], list):
        out = {}
        for row in value[1]:
            if not isinstance(row, list) or len(row) != 2:
                raise AuthenticatedProbeRefuse("probe:inner_snapshot_shape")
            key = _state_decode(row[0], budget, depth + 1)
            if key in out: raise AuthenticatedProbeRefuse("probe:inner_snapshot_duplicate")
            out[key] = _state_decode(row[1], budget, depth + 1)
        return out
    if (tag == 7 and len(value) == 3 and isinstance(value[1], int)
            and 0 <= value[1] < len(_STATE_TYPES) and isinstance(value[2], list)):
        kind = _STATE_TYPES[value[1]]
        if len(value[2]) != len(dataclass_fields(kind)):
            raise AuthenticatedProbeRefuse("probe:inner_snapshot_shape")
        return kind(*(_state_decode(row, budget, depth + 1) for row in value[2]))
    raise AuthenticatedProbeRefuse("probe:inner_snapshot_shape")


def _inner_states(machine, deep=False):
    rows = []
    for expected, keys in zip(_INNER_TYPES, _INNER_KEYS):
        derived = ({"_checkpoint_event_bytes"}
                   if expected is ResidentParametricSpanNetworkV1 else set())
        if (type(machine) is not expected
                or set(machine.__dict__) != set(keys) | derived | {"_boundary"}
                    | ({"_inner"} if hasattr(machine, "_inner") else set())):
            raise AuthenticatedProbeRefuse("probe:inner_snapshot_state")
        row = tuple(machine.__dict__[key] for key in keys)
        rows.append(copy.deepcopy(row) if deep else row)
        machine = getattr(machine, "_inner", None)
    if machine is not None: raise AuthenticatedProbeRefuse("probe:inner_snapshot_state")
    return tuple(rows)


def _pack_inner(machine):
    budget = [0]
    return tuple(tuple(_state_encode(value, budget) for value in row)
                 for row in _inner_states(machine))


def _unpack_inner(rows, boundary):
    if (not isinstance(rows, (tuple, list)) or len(rows) != len(_INNER_TYPES)
            or any(not isinstance(row, (tuple, list)) for row in rows)):
        raise AuthenticatedProbeRefuse("probe:inner_snapshot_shape")
    budget = [0]; machines = []
    for expected, keys, row in zip(_INNER_TYPES, _INNER_KEYS, rows):
        if len(row) != len(keys):
            raise AuthenticatedProbeRefuse("probe:inner_snapshot_shape")
        state = {key: _state_decode(value, budget) for key, value in zip(keys, row)}
        machine = object.__new__(expected)
        machine.__dict__.update(state); machine._boundary = boundary
        machines.append(machine)
    machines[1]._checkpoint_event_bytes = sum(len(
        parametric_module._canonical(asdict(event)))
        for event in machines[1].events)
    machines[0]._inner = machines[1]; machines[1]._inner = machines[2]
    return machines[0]


def _copy_inner(machine, boundary):
    _pack_inner(machine)  # enforce the same explicit node bound before deepcopy
    rows = _inner_states(machine, deep=True); machines = []
    for expected, keys, row in zip(_INNER_TYPES, _INNER_KEYS, rows):
        item = object.__new__(expected); item._boundary = boundary
        item.__dict__.update(zip(keys, row)); machines.append(item)
    machines[1]._checkpoint_event_bytes = sum(len(
        parametric_module._canonical(asdict(event)))
        for event in machines[1].events)
    machines[0]._inner = machines[1]; machines[1]._inner = machines[2]
    return machines[0]


class BodyProbeBoundaryV1:
    """Per-instance reference body custody; it does not simulate causality."""

    def __init__(self, contact_boundary, admission=None):
        if (admission is not _BODY_ADMISSION
                or not isinstance(contact_boundary, ReferenceChannelSequenceBoundaryV1)):
            raise AuthenticatedProbeRefuse("probe:body_boundary_admission")
        self._contact_boundary = contact_boundary
        self.__contact_key = secrets.token_bytes(32)
        self.__checkpoint_key = secrets.token_bytes(32)
        self.__clock = 0
        self.__routes = {}
        self.__applications = {}
        self.__route_uses = {}
        self.__surface_routes = {}
        self.__surface_applications = {}
        self.__surface_route_uses = {}
        self.__surface_evidence = {}
        self.__surface_evidence_bytes = 0

    def __tag(self, kind, fields):
        return int.from_bytes(hmac.new(
            self.__contact_key, _canonical((int(kind), fields)), hashlib.sha256,
        ).digest()[:8], "little") & ((1 << 63) - 1)

    def _valid(self, kind, fields, tag):
        expected = self.__tag(kind, fields)
        return hmac.compare_digest(int(tag).to_bytes(8, "little"),
                                   expected.to_bytes(8, "little"))

    def __checkpoint_tag(self, body):
        return hmac.new(self.__checkpoint_key, _canonical(body), hashlib.sha256).hexdigest()

    def _seal_resident_checkpoint(self, resident):
        if (not isinstance(resident, ResidentAuthenticatedProbeV1)
                or resident._body_boundary is not self):
            raise AuthenticatedProbeRefuse("probe:checkpoint_authority")
        resident._bounded()
        body = resident._checkpoint_body()
        blob = _canonical({"version": CHECKPOINT_VERSION, "body": body,
                           "hmac": self.__checkpoint_tag(body)})
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise AuthenticatedProbeRefuse("probe:checkpoint_bound")
        return blob

    def _valid_checkpoint(self, body, tag):
        return isinstance(tag, str) and hmac.compare_digest(
            tag, self.__checkpoint_tag(body))

    def precommit(self, session: int, device: int, body_epoch: int,
                  port: int, routes) -> BodyMappingContractV1:
        if isinstance(routes, (str, bytes, bytearray, memoryview)):
            raise AuthenticatedProbeRefuse("probe:mapping_type")
        routes = tuple(tuple(row) for row in routes)
        _numeric((session, device, body_epoch, port, routes))
        normalized = []
        for row in routes:
            if len(row) != 5:
                raise AuthenticatedProbeRefuse("probe:mapping_shape")
            source_value = row[2]
            sources = ((int(source_value),) if isinstance(source_value, int)
                       and not isinstance(source_value, bool) else
                       tuple(map(int, source_value))
                       if isinstance(source_value, (tuple, list)) else ())
            if (not 1 <= len(sources) <= MAX_ROUTE_SOURCES
                    or any(source <= 0 for source in sources)
                    or len(set(sources)) != len(sources)):
                raise AuthenticatedProbeRefuse("probe:mapping_sources")
            normalized.append((int(row[0]), int(row[1]), sources,
                               int(row[3]), tuple(map(int, row[4]))))
        routes = tuple(normalized)
        expected = {row.identity for row in _AUTHORED_ACTIONS}
        if (len(routes) != len(_AUTHORED_ACTIONS)
                or {row[0] for row in routes} != expected
                or any(row[1] < 0 or row[3] <= 0 or not row[4]
                       for row in routes)):
            raise AuthenticatedProbeRefuse("probe:mapping_shape")
        ordered = tuple(sorted((int(row[0]), int(row[1])) for row in routes))
        private_rows = tuple(sorted(routes))
        route_root = _identity(b"authenticated-probe-private-route-v1", private_rows)
        values = (int(session), int(device), int(body_epoch), int(port),
                  ordered, route_root)
        identity = _identity(b"authenticated-probe-body-mapping-v1", values)
        fields = (identity, *values)
        contract = BodyMappingContractV1(*fields, self.__tag(1, fields))
        if identity not in self.__routes and len(self.__routes) >= MAX_EVENTS:
            raise AuthenticatedProbeRefuse("probe:mapping_bound")
        self.__routes[identity] = {
            row[0]: (row[1], row[2], row[3], row[4]) for row in routes}
        return contract

    @staticmethod
    def surface_unit_digest(units):
        units = tuple(map(int, units))
        if (not units or len(units) > MAX_EVENTS
                or any(not 0 <= unit <= 255 for unit in units)):
            raise AuthenticatedProbeRefuse("probe:surface_units")
        return _identity(b"authenticated-voicebox-unit-trajectory-v1", units)

    def precommit_surface_responses(self, mapping, routes):
        if (not self.valid_mapping(mapping)
                or isinstance(routes, (str, bytes, bytearray, memoryview))):
            raise AuthenticatedProbeRefuse("probe:surface_mapping")
        normalized = []
        for raw in tuple(routes):
            row = tuple(raw)
            if len(row) != 6:
                raise AuthenticatedProbeRefuse("probe:surface_route_shape")
            action, digest, response_code, source_value, channel, units = row
            sources = ((int(source_value),) if isinstance(source_value, int)
                       and not isinstance(source_value, bool) else
                       tuple(map(int, source_value))
                       if isinstance(source_value, (tuple, list)) else ())
            units = tuple(map(int, units))
            normalized.append((int(action), int(digest), int(response_code),
                               sources, int(channel), units))
        routes = tuple(sorted(normalized))
        _numeric(routes)
        if (not 1 <= len(routes) <= MAX_SURFACE_ROUTES
                or len({row[:2] for row in routes}) != len(routes)
                or any(row[0] not in dict(mapping.entries)
                    or row[1] <= 0 or row[2] <= 0 or row[4] <= 0
                    or not 1 <= len(row[3]) <= MAX_ROUTE_SOURCES
                    or len(set(row[3])) != len(row[3])
                    or any(source <= 0 for source in row[3])
                    or not row[5] or any(not 0 <= unit <= 255 for unit in row[5])
                    for row in routes)):
            raise AuthenticatedProbeRefuse("probe:surface_route")
        public = tuple((row[0], row[1], row[2]) for row in routes)
        route_root = _identity(b"authenticated-surface-private-route-v1", routes)
        values = (mapping.identity, public, route_root)
        identity = _identity(b"authenticated-surface-response-contract-v1", values)
        fields = (identity, *values)
        contract = SurfaceResponseContractV1(*fields, self.__tag(4, fields))
        if identity not in self.__surface_routes and self.__surface_routes:
            raise AuthenticatedProbeRefuse("probe:surface_route_bound")
        self.__surface_routes[identity] = {
            (row[0], row[1]): (row[2], row[3], row[4], row[5])
            for row in routes}
        return contract

    def _bind_resident_surface(self, mapping, contract, apply, trial,
                               trajectory, authority):
        if authority is not _SURFACE_AUTHORITY:
            raise AuthenticatedProbeRefuse("probe:surface_bind_authority")
        if (not self.valid_mapping(mapping)
                or not self.valid_surface_contract(contract)
                or not self.valid_apply(apply)
                or apply.mapping_root != mapping.identity
                or contract.mapping_root != mapping.identity
                or apply.identity not in self.__applications
                or apply.identity in self.__surface_applications):
            raise AuthenticatedProbeRefuse("probe:surface_bind_state")
        try:
            units = tuple(map(int, trajectory.units))
            digest = self.surface_unit_digest(units)
            ancestry = tuple(tuple(getattr(row, name)
                for name in row.__dataclass_fields__) for row in trajectory.ancestry)
            ancestry_root = _identity(
                b"authenticated-voicebox-ancestry-v1", ancestry)
            trial_root = int(trial.identity)
            command_root = int(trial.command_root)
            constructor_root = int(trial.constructor_root)
            trajectory_values = (
                int(trajectory.constructor_root),
                int(trajectory.constructor_evidence_root),
                int(trajectory.left_recipe_root),
                int(trajectory.middle_recipe_root),
                int(trajectory.right_recipe_root),
                int(trajectory.left_occurrence_root),
                int(trajectory.middle_occurrence_root),
                int(trajectory.right_occurrence_root), units,
                tuple(map(int, trajectory.constituent_roots)),
                tuple(map(int, trajectory.source_roots)),
                int(trajectory.work_units), ancestry)
        except (AttributeError, TypeError, ValueError) as exc:
            raise AuthenticatedProbeRefuse("probe:surface_bind_shape") from exc
        application = self.__applications[apply.identity]
        action_recipe = application[1]
        route = self.__surface_routes.get(contract.identity, {}).get(
            (action_recipe, digest))
        if (route is None or command_root != apply.command_hash
                or constructor_root != int(trajectory.constructor_root)
                or int(trajectory.identity) != _identity(
                    b"resident-partial-network-trajectory-v1", trajectory_values)
                or not ancestry):
            raise AuthenticatedProbeRefuse("probe:surface_bind_lineage")
        if self.__clock >= (1 << 63) - 1:
            raise AuthenticatedProbeRefuse("probe:body_bound")
        next_tick = self.__clock + 1
        values = (apply.identity, apply.command_hash, mapping.identity,
            action_recipe, contract.identity, trial_root, constructor_root,
            int(trajectory.identity), digest, ancestry_root, next_tick)
        identity = _identity(b"authenticated-surface-apply-v1", values)
        fields = (identity, *values)
        receipt = SurfaceApplyReceiptV1(*fields, self.__tag(5, fields))
        evidence_row = (receipt.identity, int(trajectory.identity), trajectory_values)
        evidence_bytes = len(_canonical(evidence_row))
        if (len(self.__surface_evidence) >= MAX_EVENTS
                or evidence_bytes > MAX_SURFACE_EVIDENCE_BYTES
                or self.__surface_evidence_bytes
                    > MAX_SURFACE_EVIDENCE_BYTES - evidence_bytes):
            raise AuthenticatedProbeRefuse("probe:surface_evidence_bound")
        self.__clock = next_tick
        use_key = (contract.identity, action_recipe, digest)
        slot = self.__surface_route_uses.get(use_key, 0) % len(route[1])
        self.__surface_route_uses[use_key] = (
            self.__surface_route_uses.get(use_key, 0) + 1)
        self.__surface_applications[apply.identity] = (
            contract.identity, action_recipe, digest, 0, slot, receipt.identity)
        self.__surface_evidence[receipt.identity] = (
            int(trajectory.identity), trajectory_values)
        self.__surface_evidence_bytes += evidence_bytes
        return receipt

    def _dispatch_resident(self, resident) -> BodyApplyReceiptV1:
        if (not isinstance(resident, ResidentAuthenticatedProbeV1)
                or resident._body_boundary is not self
                or resident.pending is None
                or resident.pending.apply_receipt is not None):
            raise AuthenticatedProbeRefuse("probe:dispatch_authority")
        mapping, command = resident.mapping, resident.pending.command
        if (len(resident.events) >= MAX_EVENTS
                or not command.opened_sequence <= resident.next_sequence
                <= command.deadline_sequence
                or command.resource_cost > resident.resource):
            raise AuthenticatedProbeRefuse("probe:dispatch_state")
        body_before = self._snapshot_state()
        outer_before = (copy.deepcopy(resident.pending), resident.resource,
                        resident.last_apply_tick, resident.work,
                        copy.deepcopy(resident.events))
        try:
            receipt = self.__seal_accept(mapping, command)
            if receipt.apply_tick <= resident.last_apply_tick:
                raise AuthenticatedProbeRefuse("probe:apply_tick")
            resident.work = 0; resident._charge(1)
            resident.resource -= command.resource_cost
            resident.last_apply_tick = receipt.apply_tick
            resident.pending.apply_receipt = receipt
            resident._record(EventV1(
                EVENT_APPLY, (receipt.signed_fields(), receipt.auth_tag)))
            resident._bounded(); return receipt
        except Exception:
            self._restore_state(body_before)
            (resident.pending, resident.resource, resident.last_apply_tick,
             resident.work, resident.events) = outer_before
            raise

    def __seal_accept(self, mapping: BodyMappingContractV1,
                      command: ProbeCommandV1) -> BodyApplyReceiptV1:
        if not isinstance(mapping, BodyMappingContractV1) or not self.valid_mapping(mapping):
            raise AuthenticatedProbeRefuse("probe:mapping_authentication")
        if not isinstance(command, ProbeCommandV1):
            raise AuthenticatedProbeRefuse("probe:command_type")
        _numeric(command)
        if (command.identity != _identity(b"authenticated-probe-command-v1",
                                         command.content_fields())
                or command.mapping_root != mapping.identity
                or (command.action_recipe, command.command_value) not in mapping.entries):
            raise AuthenticatedProbeRefuse("probe:command_content")
        route = self.__routes.get(mapping.identity, {}).get(command.action_recipe)
        if route is None or route[0] != command.command_value:
            raise AuthenticatedProbeRefuse("probe:command_route")
        if len(self.__applications) >= MAX_EVENTS or self.__clock >= (1 << 63) - 1:
            raise AuthenticatedProbeRefuse("probe:body_bound")
        self.__clock += 1
        values = (command.identity, mapping.identity, mapping.device,
                  mapping.body_epoch, mapping.port, self.__clock, 1)
        identity = _identity(b"authenticated-probe-apply-v1", values)
        fields = (identity, *values)
        receipt = BodyApplyReceiptV1(*fields, self.__tag(2, fields))
        use_key = (mapping.identity, command.action_recipe)
        source_slot = self.__route_uses.get(use_key, 0) % len(route[1])
        self.__route_uses[use_key] = self.__route_uses.get(use_key, 0) + 1
        self.__applications[identity] = (mapping.identity,
                                         command.action_recipe, 0, source_slot)
        return receipt

    def advance(self, steps: int = 1):
        _numeric(steps)
        if not 1 <= steps <= PROBE_HORIZON:
            raise AuthenticatedProbeRefuse("probe:body_advance")
        if self.__clock > (1 << 63) - 1 - steps:
            raise AuthenticatedProbeRefuse("probe:body_bound")
        self.__clock += int(steps)
        return self.__clock

    def seal_return(self, mapping: BodyMappingContractV1,
                    apply: BodyApplyReceiptV1,
                    sequence: int) -> BodyReturnContactV1:
        if (not self.valid_mapping(mapping) or not self.valid_apply(apply)
                or apply.mapping_root != mapping.identity):
            raise AuthenticatedProbeRefuse("probe:return_apply")
        _numeric(sequence)
        application = self.__applications.get(apply.identity)
        if application is None or application[0] != mapping.identity:
            raise AuthenticatedProbeRefuse("probe:return_application")
        surface = self.__surface_applications.get(apply.identity)
        if surface is None:
            route = self.__routes.get(mapping.identity, {}).get(application[1])
            cursor = application[2]
            surface_apply_root = 0
        else:
            route = self.__surface_routes.get(surface[0], {}).get(
                (surface[1], surface[2]))
            cursor = surface[3]
            surface_apply_root = surface[5]
        if route is None or cursor >= len(route[3]):
            raise AuthenticatedProbeRefuse("probe:return_exhausted")
        if self.__clock >= (1 << 63) - 1:
            raise AuthenticatedProbeRefuse("probe:body_bound")
        _command, sources, channel, units = route
        source = sources[application[3] if surface is None else surface[4]]
        unit = units[cursor]
        self.__clock += 1
        arrival_tick = self.__clock
        occurrence = self._contact_boundary.seal_sample(
            mapping.session, int(sequence), int(source), int(channel),
            (int(unit),), (mapping.device, mapping.body_epoch,
                           apply.apply_tick, int(arrival_tick), apply.identity,
                           *((surface_apply_root,) if surface_apply_root else ())))
        values = (apply.identity, apply.command_hash, mapping.identity,
                  mapping.device, mapping.body_epoch, apply.apply_tick,
                  int(arrival_tick), occurrence.signed_fields(), occurrence.auth_tag)
        identity = _identity(b"authenticated-probe-return-v1", values)
        fields = (identity, *values)
        contact = BodyReturnContactV1(
            identity, apply.identity, apply.command_hash, mapping.identity,
            mapping.device, mapping.body_epoch, apply.apply_tick,
            int(arrival_tick), occurrence, self.__tag(3, fields))
        if cursor + 1 == len(units):
            self.__applications.pop(apply.identity, None)
            self.__surface_applications.pop(apply.identity, None)
        else:
            if surface is None:
                self.__applications[apply.identity] = (
                    application[0], application[1], cursor + 1, application[3])
            else:
                self.__surface_applications[apply.identity] = (
                    surface[0], surface[1], surface[2], cursor + 1,
                    surface[4], surface[5])
        return contact

    def valid_mapping(self, row):
        if (not isinstance(row, BodyMappingContractV1)
                or not self._valid(1, row.signed_fields(), row.auth_tag)):
            return False
        routes = self.__routes.get(row.identity)
        if routes is None:
            return False
        private_rows = tuple(sorted((action, command, sources, channel, units)
            for action, (command, sources, channel, units) in routes.items()))
        return (row.route_root == _identity(
            b"authenticated-probe-private-route-v1", private_rows)
            and row.entries == tuple(sorted((action, values[0])
                                            for action, values in routes.items())))

    def valid_apply(self, row):
        return (isinstance(row, BodyApplyReceiptV1)
                and self._valid(2, row.signed_fields(), row.auth_tag))

    def valid_surface_contract(self, row):
        if (not isinstance(row, SurfaceResponseContractV1)
                or not self._valid(4, row.signed_fields(), row.auth_tag)):
            return False
        routes = self.__surface_routes.get(row.identity)
        if routes is None:
            return False
        private = tuple(sorted((action, digest, response, sources, channel, units)
            for (action, digest), (response, sources, channel, units)
            in routes.items()))
        return (row.route_root == _identity(
            b"authenticated-surface-private-route-v1", private)
            and row.entries == tuple((item[0], item[1], item[2])
                                     for item in private))

    def valid_surface_apply(self, row):
        return (isinstance(row, SurfaceApplyReceiptV1)
                and self._valid(5, row.signed_fields(), row.auth_tag))

    def valid_surface_apply_evidence(self, row):
        if not self.valid_surface_apply(row):
            return False
        evidence = self.__surface_evidence.get(row.identity)
        if evidence is None:
            return False
        trajectory_root, values = evidence
        units, ancestry = values[8], values[12]
        return (trajectory_root == row.trajectory_root
            and row.constructor_root == values[0]
            and row.unit_digest == self.surface_unit_digest(units)
            and row.ancestry_root == _identity(
                b"authenticated-voicebox-ancestry-v1", ancestry)
            and trajectory_root == _identity(
                b"resident-partial-network-trajectory-v1", values))

    def valid_return(self, row):
        return (isinstance(row, BodyReturnContactV1)
                and self._valid(3, row.signed_fields(), row.auth_tag))

    def _snapshot_state(self):
        routes = tuple(sorted((root, tuple(sorted(
            (action, command, sources, channel, units)
            for action, (command, sources, channel, units) in rows.items())))
            for root, rows in self.__routes.items()))
        applications = tuple(sorted((root, *values)
                                    for root, values in self.__applications.items()))
        route_uses = tuple(sorted((mapping, action, count)
                                  for (mapping, action), count
                                  in self.__route_uses.items()))
        surface_routes = tuple(sorted((root, tuple(sorted(
            (action, digest, response, sources, channel, units)
            for (action, digest), (response, sources, channel, units)
            in rows.items())))
            for root, rows in self.__surface_routes.items()))
        surface_applications = tuple(sorted((root, *values) for root, values
            in self.__surface_applications.items()))
        surface_uses = tuple(sorted((contract, action, digest, count)
            for (contract, action, digest), count
            in self.__surface_route_uses.items()))
        surface_evidence = tuple(sorted((root, trajectory, values)
            for root, (trajectory, values) in self.__surface_evidence.items()))
        state = (self.__clock, routes, applications, route_uses,
                 surface_routes, surface_applications, surface_uses,
                 surface_evidence, self.__surface_evidence_bytes)
        _numeric(state)
        return state

    def _restore_state(self, state):
        _numeric(state)
        if not isinstance(state, (tuple, list)) or len(state) not in (4, 8, 9):
            raise AuthenticatedProbeRefuse("probe:body_snapshot")
        if len(state) == 4:
            clock, routes, applications, route_uses = state
            surface_routes = surface_applications = surface_uses = surface_evidence = ()
            surface_evidence_bytes = 0
        else:
            (clock, routes, applications, route_uses, surface_routes,
             surface_applications, surface_uses, surface_evidence) = state[:8]
            surface_evidence_bytes = (int(state[8]) if len(state) == 9 else
                sum(len(_canonical(row)) for row in surface_evidence))
        if (clock < 0 or len(routes) > MAX_EVENTS
                or len(applications) > MAX_EVENTS or len(route_uses) > MAX_EVENTS
                or len(surface_routes) > MAX_EVENTS
                or len(surface_applications) > MAX_EVENTS
                or len(surface_uses) > MAX_EVENTS
                or len(surface_evidence) > MAX_EVENTS
                or not 0 <= surface_evidence_bytes
                    <= MAX_SURFACE_EVIDENCE_BYTES):
            raise AuthenticatedProbeRefuse("probe:body_snapshot")
        self.__clock = int(clock)
        self.__routes = {int(root): {int(row[0]): (
            int(row[1]), tuple(map(int, row[2])), int(row[3]),
            tuple(map(int, row[4])))
            for row in rows} for root, rows in routes}
        self.__applications = {int(row[0]): tuple(map(int, row[1:]))
                               for row in applications}
        self.__route_uses = {(int(row[0]), int(row[1])): int(row[2])
                             for row in route_uses}
        self.__surface_routes = {int(root): {(int(row[0]), int(row[1])): (
            int(row[2]), tuple(map(int, row[3])), int(row[4]),
            tuple(map(int, row[5]))) for row in rows}
            for root, rows in surface_routes}
        self.__surface_applications = {int(row[0]): tuple(map(int, row[1:]))
                                       for row in surface_applications}
        self.__surface_route_uses = {(int(row[0]), int(row[1]), int(row[2])):
                                     int(row[3])
                                     for row in surface_uses}
        self.__surface_evidence = {int(row[0]): (
            int(row[1]), _tuplify(row[2])) for row in surface_evidence}
        recomputed_evidence_bytes = sum(len(_canonical(row))
            for row in surface_evidence)
        if recomputed_evidence_bytes != surface_evidence_bytes:
            raise AuthenticatedProbeRefuse("probe:surface_evidence_size")
        self.__surface_evidence_bytes = surface_evidence_bytes
        if (any(len(row) != 5 for row in applications)
                or any(len(row) != 3 or row[2] < 0 for row in route_uses)
                or any(len(row) != 7 for row in surface_applications)
                or any(len(row) != 4 or row[3] < 0 for row in surface_uses)
                or any(len(row) != 3 for row in surface_evidence)
                or any(not sources or len(sources) > MAX_ROUTE_SOURCES
                       or len(set(sources)) != len(sources)
                       or any(source <= 0 for source in sources)
                       or channel <= 0 or not units
                       for rows in self.__routes.values()
                       for _command, sources, channel, units in rows.values())
                or any(not sources or len(sources) > MAX_ROUTE_SOURCES
                       or len(set(sources)) != len(sources)
                       or any(source <= 0 for source in sources)
                       or response <= 0 or channel <= 0 or not units
                       for rows in self.__surface_routes.values()
                       for response, sources, channel, units in rows.values())):
            raise AuthenticatedProbeRefuse("probe:body_snapshot")


def admit_body_probe_boundary_v1(contact_boundary) -> BodyProbeBoundaryV1:
    return BodyProbeBoundaryV1(contact_boundary, _BODY_ADMISSION)


class ResidentAuthenticatedProbeV1:
    def __init__(self, contact_boundary: ReferenceChannelSequenceBoundaryV1,
                 body_boundary: BodyProbeBoundaryV1,
                 mapping: BodyMappingContractV1, session: int = 1,
                 resource_limit: int = 64, work_limit: int = MAX_WORK,
                 initial_inner: ResidentCompositeCuePredictionV1 | None = None):
        if (not isinstance(body_boundary, BodyProbeBoundaryV1)
                or body_boundary._contact_boundary is not contact_boundary
                or not body_boundary.valid_mapping(mapping)
                or mapping.session != session):
            raise AuthenticatedProbeRefuse("probe:configuration")
        _numeric((session, resource_limit, work_limit))
        if not 1 <= resource_limit <= 1_000_000 or not 1 <= work_limit <= MAX_WORK:
            raise AuthenticatedProbeRefuse("probe:configuration")
        self._contact_boundary, self._body_boundary = contact_boundary, body_boundary
        self.mapping = mapping
        self._actions = tuple(_AUTHORED_ACTIONS)
        if initial_inner is None:
            self._inner = ResidentCompositeCuePredictionV1(
                contact_boundary, session, work_limit)
        elif (not isinstance(initial_inner, ResidentCompositeCuePredictionV1)
              or initial_inner._boundary is not contact_boundary
              or initial_inner.session != session
              or initial_inner.work_limit != work_limit or initial_inner.pending):
            raise AuthenticatedProbeRefuse("probe:initial_inner")
        else:
            # Foundry construction transfers sole ownership of the authored
            # starting organism.  No runtime host retains a second live copy.
            self._inner = initial_inner
        _pack_inner(self._inner)
        self.resource_limit = int(resource_limit)
        self.resource = int(resource_limit)
        self.work_limit = int(work_limit)
        self.pending: PendingProbeV1 | None = None
        self.evidence: list[ProbeEvidenceV1] = []
        self.events: list[EventV1] = []
        self.last_apply_tick = 0
        self.resident_tick = 0
        self.work = 0

    @property
    def session(self): return self._inner.session

    @property
    def next_sequence(self): return self._inner.next_sequence

    def _snapshot(self):
        return (_copy_inner(self._inner, self._contact_boundary), copy.deepcopy(self.pending),
                copy.deepcopy(self.evidence), tuple(self._actions), self.resource,
                self.last_apply_tick, self.resident_tick, self.work,
                copy.deepcopy(self.events))

    def _rollback(self, state):
        (inner_state, self.pending, self.evidence, self._actions, self.resource,
         self.last_apply_tick, self.resident_tick, self.work, self.events) = state
        self._inner = inner_state

    def _record(self, event):
        _numeric(event)
        if len(self.events) >= MAX_EVENTS:
            raise AuthenticatedProbeRefuse("probe:event_bound")
        self.events.append(event)

    def _charge(self, amount=1):
        self.work += int(amount)
        if self.work > self.work_limit:
            raise AuthenticatedProbeRefuse("probe:resource")

    @staticmethod
    def _contact_event(kind, contact):
        return EventV1(kind, (contact.signed_fields(), contact.auth_tag))

    def _eligible(self, ticket):
        recipes = [self._inner.recipes.get(root)
                   for root in ticket.relation_recipe_roots]
        self._charge(len(recipes))
        if (len(recipes) != 2 or any(row is None or row.credit <= 0 for row in recipes)
                or ticket.prospective_middle_kind != NODE_COMPOSITE):
            return ()
        scores = []
        for action in self._actions:
            self._charge(len(self.evidence) + 1)
            returned = sum(row.difference for row in self.evidence
                           if row.action_recipe == action.identity
                           and row.mapping_root == self.mapping.identity)
            score = 4 * returned - action.structural_cost
            if action.resource_cost <= self.resource:
                scores.append((score, action))
        return tuple(scores)

    def _policy_uniqueness(self):
        scores = []
        for action in self._actions:
            returned = sum(row.difference for row in self.evidence
                           if row.action_recipe == action.identity
                           and row.mapping_root == self.mapping.identity)
            if action.resource_cost <= self.resource:
                scores.append(4 * returned - action.structural_cost)
        return (bool(scores) and scores.count(max(scores)) == 1,
                len(self._actions) * (len(self.evidence) + 1))

    def _select(self, ticket):
        scored = self._eligible(ticket)
        self._charge(len(scored))
        if not scored:
            raise AuthenticatedProbeRefuse("probe:no_eligible_action")
        peak = max(row[0] for row in scored)
        winners = [row for score, row in scored if score == peak]
        if len(winners) != 1:
            raise AuthenticatedProbeRefuse("probe:action_ambiguous")
        action = winners[0]
        mapping = dict(self.mapping.entries)
        candidate_rows = tuple(sorted((row.identity, row.capability,
                                       row.structural_cost, row.resource_cost)
                                      for _score, row in scored))
        candidate_root = _identity(b"authenticated-probe-candidate-set-v1", candidate_rows)
        score_rows = tuple(sorted((row.identity, score) for score, row in scored))
        state_values = (ticket.envelope_root, tuple(ticket.relation_recipe_roots),
                        tuple(ticket.relation_witness_roots), candidate_rows, score_rows,
                        self.resource, tuple(row.identity for row in self.evidence))
        state_root = _identity(b"authenticated-probe-selection-state-v1", state_values)
        episode = _identity(b"authenticated-probe-episode-v1", [
            ticket.envelope_root, state_root, candidate_root, len(self.events)])
        values = (episode, ticket.ticket, ticket.envelope_root,
                  ticket.session, ticket.incarnation, action.identity,
                  action.capability, mapping[action.identity], self.mapping.identity,
                  state_root, candidate_root, self.resource, action.resource_cost,
                  self.next_sequence, ticket.deadline_sequence,
                  self.resident_tick, self.resident_tick + PROBE_HORIZON)
        return ProbeCommandV1(_identity(b"authenticated-probe-command-v1", values),
                              *values)

    def ingest_sample(self, contact: OccurrenceContactV1):
        if self.pending is not None:
            raise AuthenticatedProbeRefuse("probe:pending_episode")
        if len(self.events) >= MAX_EVENTS:
            raise AuthenticatedProbeRefuse("probe:event_bound")
        # The inner transaction authenticates and rolls itself back.  All outer
        # postconditions below are prebounded numeric appends, so serializing the
        # entire trained organism a second time would add no atomicity.
        self.work = 0; self._charge(1)
        result = self._inner.ingest_sample(contact)
        self._record(self._contact_event(EVENT_SAMPLE, contact))
        self._bounded(); return result

    def tick(self) -> ProbeTickResultV1:
        if self.pending is not None:
            if len(self.events) >= MAX_EVENTS:
                raise AuthenticatedProbeRefuse("probe:event_bound")
            deadline = self.pending.command.deadline_resident_tick
            next_tick = self.resident_tick + 1
            if next_tick > deadline:
                raise AuthenticatedProbeRefuse("probe:pending_deadline")
            episode, ticket = self.pending.episode, self.pending.ticket.ticket
            expired = next_tick == deadline
            identity = _identity(b"authenticated-probe-wait-v1",
                                 [episode, next_tick, deadline, int(expired)])
            self.work = 0; self._charge(1); self.resident_tick = next_tick
            if expired:
                self._inner.pending.pop(ticket, None)
                self.pending = None
            out = ProbeTickResultV1(identity, 0, None)
            self._record(EventV1(EVENT_TICK, (identity, 0, 0)))
            self._bounded(); return out
        # Only a policy-side refusal after the inner tick needs cross-layer
        # rollback.  A unique pre-state policy makes selection a bounded,
        # infallible projection of any chained ticket the inner tick creates.
        unique_policy, preflight_work = self._policy_uniqueness()
        before = None if unique_policy else self._snapshot()
        old_tick, old_work = self.resident_tick, self.work
        try:
            self.work = 0; self._charge(preflight_work); self.resident_tick += 1
            result = self._inner.tick(); command = None
            tickets = [self._inner.pending.get(ticket)
                       for ticket in result.prediction_tickets]
            chained = [row for row in tickets if row is not None
                       and len(row.relation_recipe_roots) == 2
                       and row.prospective_middle_kind == NODE_COMPOSITE]
            if len(chained) > 1 or (chained and len(tickets) != 1):
                raise AuthenticatedProbeRefuse("probe:chained_ticket_unique")
            if chained:
                command = self._select(chained[0])
                self.pending = PendingProbeV1(command.episode, chained[0], command)
            identity = _identity(b"authenticated-probe-tick-v1", [
                result.identity, 0 if command is None else command.identity])
            out = ProbeTickResultV1(identity, result.identity, command)
            self._record(EventV1(EVENT_TICK, (identity, result.identity,
                                              0 if command is None else command.identity)))
            self._bounded(); return out
        except Exception:
            if before is not None:
                self._rollback(before)
            else:
                self.resident_tick, self.work = old_tick, old_work
            raise

    def dispatch(self):
        return self._body_boundary._dispatch_resident(self)

    def ingest_return(self, contact: BodyReturnContactV1):
        if self.pending is None or self.pending.apply_receipt is None:
            raise AuthenticatedProbeRefuse("probe:return_episode")
        if len(self.events) >= MAX_EVENTS or len(self.evidence) >= MAX_EVIDENCE:
            raise AuthenticatedProbeRefuse("probe:event_bound")
        before = self._snapshot()
        try:
            self.work = 0
            episode = self.pending; apply = episode.apply_receipt
            occurrence = contact.occurrence
            if (not self._body_boundary.valid_return(contact)
                    or contact.apply_receipt != apply.identity
                    or contact.command_hash != episode.command.identity
                    or contact.mapping_root != self.mapping.identity
                    or (contact.device, contact.body_epoch, contact.apply_tick)
                    != (apply.device, apply.body_epoch, apply.apply_tick)
                    or contact.arrival_tick <= max(apply.apply_tick,
                                                   episode.last_arrival_tick)
                    or occurrence.session != self.session
                    or occurrence.sequence != self.next_sequence
                    or occurrence.channel != episode.ticket.target_channel
                    or self.next_sequence > episode.ticket.deadline_sequence):
                raise AuthenticatedProbeRefuse("probe:return_authentication")
            prior = {row.identity for row in self._inner.witnesses}
            self._charge(len(prior))
            self._inner.ingest_sample(occurrence)
            episode.return_roots += (contact.identity,)
            episode.return_source_roots = tuple(sorted(set(
                episode.return_source_roots + (occurrence.source,))))
            episode.last_arrival_tick = contact.arrival_tick
            created = [row for row in self._inner.witnesses
                       if row.identity not in prior and row.ticket == episode.ticket.ticket]
            assert len(created) <= 1
            if created:
                witness = created[0]
                sources = tuple(sorted(set(episode.ticket.source_roots
                                           + witness.source_roots)))
                values = (episode.episode, episode.command.action_recipe,
                          episode.command.identity, self.mapping.identity,
                          apply.identity, episode.ticket.ticket, witness.identity,
                          witness.difference, episode.return_roots, sources)
                evidence = ProbeEvidenceV1(
                    _identity(b"authenticated-probe-evidence-v1", values), *values)
                self.evidence.append(evidence); self.pending = None
            elif episode.ticket.ticket not in self._inner.pending:
                raise AuthenticatedProbeRefuse("probe:return_without_witness")
            self._record(EventV1(EVENT_RETURN,
                (contact.signed_fields(), contact.auth_tag)))
            self._bounded(); return None if not created else self.evidence[-1]
        except Exception:
            self._rollback(before)
            raise

    def ingest_withdrawal(self, contact: WithdrawalContactV1):
        before = self._snapshot()
        try:
            self.work = 0
            self._inner.ingest_withdrawal(contact); target = contact.target_source
            self._charge(len(self.evidence) + 1)
            self.evidence = [row for row in self.evidence
                             if target not in row.source_roots]
            if self.pending is not None and target in self.pending.ticket.source_roots:
                self._inner.pending.pop(self.pending.ticket.ticket, None)
                self.pending = None
            elif (self.pending is not None
                  and target in self.pending.return_source_roots):
                self._inner.pending.pop(self.pending.ticket.ticket, None)
                self.pending = None
            self._record(self._contact_event(EVENT_WITHDRAWAL, contact))
            self._bounded()
        except Exception:
            self._rollback(before); raise

    def adopt_mapping(self, mapping: BodyMappingContractV1):
        if self.pending is not None:
            raise AuthenticatedProbeRefuse("probe:mapping_pending")
        if (len(self.events) >= MAX_EVENTS
                or not self._body_boundary.valid_mapping(mapping)
                or mapping.session != self.session
                or (mapping.device, mapping.port, mapping.entries)
                != (self.mapping.device, self.mapping.port, self.mapping.entries)
                or mapping.body_epoch <= self.mapping.body_epoch):
            raise AuthenticatedProbeRefuse("probe:mapping_transition")
        self.work = 0; self._charge(len(self.evidence) + 1)
        self.evidence = [row for row in self.evidence
                         if row.mapping_root == mapping.identity]
        self.mapping = mapping
        self._record(EventV1(EVENT_MAPPING, (mapping.signed_fields(), mapping.auth_tag)))
        self._bounded(); return mapping.identity

    def _bounded(self):
        if (len(self.events) > MAX_EVENTS or len(self.evidence) > MAX_EVIDENCE
                or not 0 <= self.resource <= self.resource_limit
                or not 0 <= self.work <= self.work_limit
                or self.resident_tick < 0
                or tuple(self._actions) != _AUTHORED_ACTIONS
                or any(not _valid_action(row) for row in self._actions)):
            raise AuthenticatedProbeRefuse("probe:state_bound")
        pending = (() if self.pending is None else (
            self.pending.episode, self.pending.ticket, self.pending.command,
            () if self.pending.apply_receipt is None
            else (self.pending.apply_receipt,), self.pending.return_roots,
            self.pending.return_source_roots, self.pending.last_arrival_tick,
            self.pending.command.deadline_resident_tick))
        _numeric((self.mapping, self._actions, self.resource,
                  self.resource_limit, self.work_limit,
                  self.last_apply_tick, self.resident_tick, self.work, pending,
                  tuple(self.evidence), tuple(self.events)))

    def _checkpoint_body(self):
        return {"schema": SCHEMA_VERSION, "session": self.session,
                "resource_limit": self.resource_limit, "work_limit": self.work_limit,
                "resource": self.resource, "last_apply_tick": self.last_apply_tick,
                "resident_tick": self.resident_tick, "work": self.work,
                "mapping": asdict(self.mapping),
                "actions": [asdict(row) for row in self._actions],
                "inner_layers": _pack_inner(self._inner),
                "body_boundary": self._body_boundary._snapshot_state(),
                "pending": None if self.pending is None else asdict(self.pending),
                "evidence": [asdict(row) for row in self.evidence],
                "events": [asdict(row) for row in self.events]}

    def checkpoint(self):
        return self._body_boundary._seal_resident_checkpoint(self)

    @classmethod
    def restore(cls, blob, contact_boundary, body_boundary):
        try:
            raw = bytes(blob)
            if len(raw) > MAX_CHECKPOINT_BYTES:
                raise AuthenticatedProbeRefuse("probe:checkpoint_bound")
            envelope = json.loads(raw); body = envelope["body"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AuthenticatedProbeRefuse("probe:checkpoint") from exc
        if (not isinstance(envelope, dict) or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] not in (3, CHECKPOINT_VERSION)
                or not body_boundary._valid_checkpoint(body, envelope["hmac"])):
            raise AuthenticatedProbeRefuse("probe:checkpoint_authentication")
        if (not isinstance(body, dict)
                or set(body) != {"schema", "session", "resource_limit",
                                 "work_limit", "resource", "last_apply_tick",
                                 "mapping", "actions", "inner_layers", "body_boundary", "pending",
                                 "evidence", "resident_tick", "work", "events"}
                or body["schema"] != SCHEMA_VERSION
                or any(isinstance(body[key], bool) or not isinstance(body[key], int)
                       for key in ("schema", "session", "resource_limit",
                                   "work_limit", "resource", "last_apply_tick",
                                   "resident_tick", "work"))
                or not isinstance(body["events"], list)
                or not isinstance(body["evidence"], list)
                or len(body["events"]) > MAX_EVENTS
                or len(body["evidence"]) > MAX_EVIDENCE):
            raise AuthenticatedProbeRefuse("probe:checkpoint_schema")
        body_before = body_boundary._snapshot_state()
        try:
            raw_mapping = body["mapping"]
            mapping = BodyMappingContractV1(
                *(int(raw_mapping[key]) for key in
                  ("identity", "session", "device", "body_epoch", "port")),
                tuple(tuple(map(int, row)) for row in raw_mapping["entries"]),
                int(raw_mapping["route_root"]),
                int(raw_mapping["auth_tag"]))
            body_boundary._restore_state(_tuplify(body["body_boundary"]))
            inner = _unpack_inner(body["inner_layers"], contact_boundary)
            inner_pending = inner.pending; inner.pending = {}
            out = cls(contact_boundary, body_boundary, mapping,
                      int(body["session"]), int(body["resource_limit"]),
                      int(body["work_limit"]), inner)
            out._inner.pending = inner_pending
            if not isinstance(body["actions"], list):
                raise AuthenticatedProbeRefuse("probe:checkpoint_actions")
            out._actions = tuple(_from_dict(ActionRecipeV1, row)
                                 for row in body["actions"])
            out.resource = int(body["resource"])
            out.last_apply_tick = int(body["last_apply_tick"])
            out.resident_tick = int(body["resident_tick"])
            out.work = int(body["work"])
            out.evidence = [_from_dict(ProbeEvidenceV1, row)
                            for row in body["evidence"]]
            out.events = [_from_dict(EventV1, row) for row in body["events"]]
            if body["pending"] is not None:
                raw_pending = body["pending"]
                out.pending = PendingProbeV1(
                    int(raw_pending["episode"]),
                    _from_dict(StructuralPredictionTicketV1, raw_pending["ticket"]),
                    _from_dict(ProbeCommandV1, raw_pending["command"]),
                    None if raw_pending["apply_receipt"] is None else
                    _from_dict(BodyApplyReceiptV1, raw_pending["apply_receipt"]),
                    _tuplify(raw_pending["return_roots"]),
                    _tuplify(raw_pending["return_source_roots"]),
                    int(raw_pending["last_arrival_tick"]))
            out._bounded()
            if envelope["version"] == CHECKPOINT_VERSION and out.checkpoint() != raw:
                raise AuthenticatedProbeRefuse("probe:checkpoint_noncanonical")
        except Exception as exc:
            body_boundary._restore_state(body_before)
            if isinstance(exc, AuthenticatedProbeRefuse):
                raise
            raise AuthenticatedProbeRefuse("probe:checkpoint_schema") from exc
        return out


def _occurrence(fields, tag):
    return OccurrenceContactV1(int(fields[0]), int(fields[1]), int(fields[2]),
        int(fields[3]), tuple(map(int, fields[4])), tuple(map(int, fields[5])), int(tag))


def _tuplify(value):
    return tuple(_tuplify(row) for row in value) if isinstance(value, list) else int(value)


def _from_dict(cls, raw):
    if not isinstance(raw, dict) or set(raw) != {row.name for row in dataclass_fields(cls)}:
        raise AuthenticatedProbeRefuse("probe:checkpoint_row")
    return cls(*(_tuplify(raw[row.name]) for row in dataclass_fields(cls)))
