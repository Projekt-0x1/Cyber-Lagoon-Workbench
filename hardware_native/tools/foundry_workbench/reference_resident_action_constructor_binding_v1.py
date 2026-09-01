#!/usr/bin/env python3
"""Resident action-to-constructor Networks learned from authenticated trials."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import secrets

from reference_resident_parametric_span_network_v1 import _identity, _strict_numeric


SCHEMA_VERSION = 0x41434233
MAX_CANDIDATES = 16
MAX_TRIALS = 128
MAX_WITNESSES = 128
MAX_NETWORKS = 32
MAX_ROOTS = 1024
MAX_WORK = 65536
MAX_CHECKPOINT_BYTES = 2 << 20
_BINDING_AUTHORITY = object()
_CHECKPOINT_AUTHORITY = object()


class ActionConstructorBindingRefuse(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionConstructorTrialV1:
    identity: int
    command_root: int
    action_recipe: int
    context_nomination_root: int
    candidate_set_root: int
    surface_state_root: int
    candidate_roots: tuple[int, ...]
    constructor_root: int
    source_roots: tuple[int, ...]
    opened_tick: int
    deadline_tick: int


@dataclass
class PendingActionConstructorTrialV1:
    trial: ActionConstructorTrialV1
    trajectory_root: int = 0
    trajectory_constituent_roots: tuple[int, ...] = ()
    surface_apply_fields: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActionConstructorWitnessV1:
    identity: int
    trial_root: int
    command_root: int
    action_recipe: int
    constructor_root: int
    candidate_set_root: int
    trajectory_root: int
    trajectory_constituent_roots: tuple[int, ...]
    surface_apply_fields: tuple[int, ...]
    catalyst_witness_root: int
    probe_evidence_root: int
    difference: int
    outcome_source: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class ActionConstructorNetworkV1:
    identity: int
    action_recipe: int
    constructor_root: int
    candidate_set_root: int
    support: int
    credit: int
    witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


class ActionConstructorCheckpointBoundaryV1:
    def __init__(self, authority):
        if authority is not _CHECKPOINT_AUTHORITY:
            raise ActionConstructorBindingRefuse("action_binding:boundary_authority")
        self.__key = secrets.token_bytes(32)

    @staticmethod
    def _body(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def _seal(self, body, authority):
        if authority is not _CHECKPOINT_AUTHORITY:
            raise ActionConstructorBindingRefuse("action_binding:checkpoint_authority")
        return hmac.new(self.__key, self._body(body), hashlib.sha256).hexdigest()

    def valid(self, body, tag):
        expected = hmac.new(self.__key, self._body(body), hashlib.sha256).hexdigest()
        return isinstance(tag, str) and hmac.compare_digest(expected, tag)


def admit_action_constructor_checkpoint_boundary_v1():
    return ActionConstructorCheckpointBoundaryV1(_CHECKPOINT_AUTHORITY)


class ResidentActionConstructorBindingV1:
    def __init__(self, boundary: ActionConstructorCheckpointBoundaryV1,
                 work_limit=MAX_WORK):
        if not isinstance(boundary, ActionConstructorCheckpointBoundaryV1):
            raise ActionConstructorBindingRefuse("action_binding:boundary")
        self._boundary = boundary
        self.work_limit = int(work_limit)
        self.work = 0
        self.tick = 0
        self.pending: PendingActionConstructorTrialV1 | None = None
        self.trials: list[ActionConstructorTrialV1] = []
        self.witnesses: list[ActionConstructorWitnessV1] = []
        self.networks: dict[int, ActionConstructorNetworkV1] = {}
        self.withdrawn_sources: set[int] = set()
        self._bounded()

    def _charge(self, amount=1):
        self.work += int(amount)
        if self.work > self.work_limit:
            raise ActionConstructorBindingRefuse("action_binding:resource")

    def _validation_cost(self, pending=None, trials=None, witnesses=None,
                         networks=None, withdrawn=None):
        pending = self.pending if pending is None else pending
        trials = self.trials if trials is None else trials
        witnesses = self.witnesses if witnesses is None else witnesses
        networks = self.networks if networks is None else networks
        withdrawn = self.withdrawn_sources if withdrawn is None else withdrawn
        return (8 + len(withdrawn)
            + sum(12 + len(row.candidate_roots) + len(row.source_roots)
                  for row in trials)
            + sum(12 + len(row.trajectory_constituent_roots)
                  + len(row.surface_apply_fields) + len(row.source_roots)
                  for row in witnesses)
            + sum(8 + len(row.witness_roots) + len(row.source_roots)
                  for row in networks.values())
            + (0 if pending is None else 12
               + len(pending.trial.candidate_roots)
               + len(pending.trial.source_roots)
               + len(pending.trajectory_constituent_roots)
               + len(pending.surface_apply_fields)))

    def _conservative_validation_cost(self):
        return (8 + len(self.withdrawn_sources)
            + len(self.trials) * (12 + MAX_CANDIDATES + MAX_ROOTS)
            + len(self.witnesses) * (25 + 2 * MAX_ROOTS)
            + len(self.networks) * (8 + MAX_WITNESSES + MAX_ROOTS)
            + (0 if self.pending is None else
               25 + MAX_CANDIDATES + 2 * MAX_ROOTS))

    def _nomination_preflight_cost(self, candidate_count, source_count):
        validation = self._conservative_validation_cost()
        return (2 * validation + int(candidate_count)
            + len(self.networks) * (2 + MAX_ROOTS)
            + len(self.witnesses) * (1 + MAX_ROOTS)
            + int(candidate_count) + int(source_count) + 4
            + 25 + MAX_CANDIDATES + 2 * MAX_ROOTS)

    def _snapshot(self):
        pending = None if self.pending is None else PendingActionConstructorTrialV1(
            self.pending.trial, self.pending.trajectory_root,
            self.pending.trajectory_constituent_roots,
            self.pending.surface_apply_fields)
        return (self.work, self.tick, pending, list(self.trials),
                list(self.witnesses), dict(self.networks),
                set(self.withdrawn_sources))

    def _rollback(self, snapshot):
        (self.work, self.tick, self.pending, self.trials, self.witnesses,
         self.networks, self.withdrawn_sources) = snapshot

    @staticmethod
    def _candidate_root(candidates):
        return _identity(b"action-constructor-candidate-set-v1", candidates)

    def resolve(self, action_recipe, candidates, authority):
        if authority is not _BINDING_AUTHORITY:
            raise ActionConstructorBindingRefuse("action_binding:resolve_authority")
        if (not isinstance(candidates, tuple)
                or not 2 <= len(candidates) <= MAX_CANDIDATES
                or self._conservative_validation_cost()
                    + len(candidates) + len(self.networks) * (2 + MAX_ROOTS)
                    > self.work_limit):
            raise ActionConstructorBindingRefuse("action_binding:resource")
        candidates = tuple(sorted(map(int, candidates)))
        self._bounded()
        root = self._candidate_root(candidates)
        rows = [row for row in self.networks.values()
                if row.action_recipe == int(action_recipe)
                and row.candidate_set_root == root
                and row.constructor_root in candidates
                and not set(row.source_roots).intersection(self.withdrawn_sources)]
        cost = (self._validation_cost() + len(candidates)
                + sum(2 + len(row.source_roots) for row in self.networks.values()))
        if cost > self.work_limit:
            raise ActionConstructorBindingRefuse("action_binding:resource")
        if len(rows) != 1:
            raise ActionConstructorBindingRefuse("action_binding:unresolved")
        return rows[0]

    def _nominate(self, command_root, action_recipe, context_nomination_root,
                  candidates, surface_state_root, source_roots, deadline_tick,
                  authority):
        if authority is not _BINDING_AUTHORITY or self.pending is not None:
            raise ActionConstructorBindingRefuse("action_binding:nomination_authority")
        candidates = tuple(sorted(map(int, candidates)))
        sources = tuple(sorted(map(int, source_roots)))
        if (not 2 <= len(candidates) <= MAX_CANDIDATES
                or len(set(candidates)) != len(candidates)
                or not sources or len(sources) > MAX_ROOTS
                or any(source in self.withdrawn_sources for source in sources)
                or self.tick > int(deadline_tick)):
            raise ActionConstructorBindingRefuse("action_binding:nomination")
        preflight = self._nomination_preflight_cost(
            len(candidates), len(sources))
        if preflight > self.work_limit:
            raise ActionConstructorBindingRefuse("action_binding:resource")
        candidate_root = self._candidate_root(candidates)
        resolution_work = (self._validation_cost() + len(candidates)
            + sum(2 + len(row.source_roots)
                  for row in self.networks.values()))
        prior_work = 0
        try:
            network = self.resolve(action_recipe, candidates, _BINDING_AUTHORITY)
            constructor = network.constructor_root
        except ActionConstructorBindingRefuse as exc:
            if "unresolved" not in str(exc):
                raise
            prior_work = sum(1 + len(row.source_roots)
                             for row in self.witnesses)
            prior = [row for row in self.witnesses
                     if row.action_recipe == int(action_recipe)
                     and row.candidate_set_root == candidate_root
                     and not set(row.source_roots).intersection(
                         self.withdrawn_sources)]
            if prior:
                last = prior[-1]
                index = candidates.index(last.constructor_root)
                constructor = candidates[index if last.difference > 0
                                         else (index + 1) % len(candidates)]
            else:
                constructor = candidates[_identity(
                    b"action-constructor-endogenous-exploration-v1",
                    (int(action_recipe), candidate_root)) % len(candidates)]
        snapshot = self._snapshot()
        self.work = 0
        values = (int(command_root), int(action_recipe),
            int(context_nomination_root), candidate_root,
            int(surface_state_root), candidates, constructor, sources,
            self.tick, int(deadline_tick))
        trial = ActionConstructorTrialV1(
            _identity(b"action-constructor-trial-v1", values), *values)
        try:
            pending = PendingActionConstructorTrialV1(trial)
            self._charge(resolution_work + prior_work + len(candidates)
                         + len(sources) + 4
                         + self._validation_cost(pending=pending))
            self.pending = pending
            self._bounded(); return trial
        except Exception:
            self._rollback(snapshot); raise

    def _mark_surface_apply(self, trial, fields, authority):
        if (authority is not _BINDING_AUTHORITY or self.pending is None
                or self.pending.trial != trial
                or not self.pending.trajectory_root
                or self.pending.surface_apply_fields):
            raise ActionConstructorBindingRefuse("action_binding:surface_authority")
        fields = tuple(map(int, fields))
        if (len(fields) != 13 or any(value <= 0 for value in fields)
                or fields[2] != trial.command_root
                or fields[4] != trial.action_recipe
                or fields[6] != trial.identity
                or fields[7] != trial.constructor_root
                or fields[8] != self.pending.trajectory_root):
            raise ActionConstructorBindingRefuse("action_binding:surface_lineage")
        snapshot = self._snapshot()
        try:
            projected = PendingActionConstructorTrialV1(
                trial, self.pending.trajectory_root,
                self.pending.trajectory_constituent_roots, fields)
            self.work = 0; self._charge(len(fields)
                + self._validation_cost(pending=projected))
            self.pending = projected; self._bounded()
        except Exception:
            self._rollback(snapshot); raise

    def _mark_trajectory(self, trial, trajectory_root, constituent_roots,
                         authority):
        if (authority is not _BINDING_AUTHORITY or self.pending is None
                or self.pending.trial != trial or self.pending.trajectory_root):
            raise ActionConstructorBindingRefuse("action_binding:trajectory_authority")
        roots = tuple(sorted(map(int, constituent_roots)))
        if not int(trajectory_root) > 0 or not roots or len(roots) > MAX_ROOTS:
            raise ActionConstructorBindingRefuse("action_binding:trajectory")
        snapshot = self._snapshot()
        try:
            projected = PendingActionConstructorTrialV1(
                trial, int(trajectory_root), roots)
            self.work = 0; self._charge(len(roots) + 2
                + self._validation_cost(pending=projected))
            self.pending = projected
            self._bounded()
        except Exception:
            self._rollback(snapshot); raise

    def _settle(self, trial, catalyst_witness_root, probe_evidence_root,
                difference, outcome_source,
                evidence_source_roots, authority):
        if (authority is not _BINDING_AUTHORITY or self.pending is None
                or self.pending.trial != trial
                or not self.pending.trajectory_root
                or not self.pending.surface_apply_fields):
            raise ActionConstructorBindingRefuse("action_binding:settlement_authority")
        sources = tuple(sorted({*trial.source_roots,
            *map(int, evidence_source_roots), int(outcome_source),
            self.pending.surface_apply_fields[0]}))
        if (len(self.trials) >= MAX_TRIALS
                or len(self.witnesses) >= MAX_WITNESSES or not int(difference)
                or int(catalyst_witness_root) <= 0
                or int(outcome_source) <= 0
                or len(sources) > MAX_ROOTS
                or any(source in self.withdrawn_sources for source in sources)):
            raise ActionConstructorBindingRefuse("action_binding:settlement")
        snapshot = self._snapshot()
        self.work = 0
        values = (trial.identity, trial.command_root, trial.action_recipe,
            trial.constructor_root, trial.candidate_set_root,
            self.pending.trajectory_root,
            self.pending.trajectory_constituent_roots,
            self.pending.surface_apply_fields,
            int(catalyst_witness_root),
            int(probe_evidence_root), int(difference), int(outcome_source), sources)
        witness = ActionConstructorWitnessV1(
            _identity(b"action-constructor-witness-v1", values), *values)
        try:
            projected_trials = [*self.trials, trial]
            projected_witnesses = [*self.witnesses, witness]
            self._charge(len(self.witnesses) + len(sources) + 8
                + self._validation_cost(pending=None, trials=projected_trials,
                    witnesses=projected_witnesses))
            self.trials = projected_trials; self.witnesses = projected_witnesses
            self.pending = None; self.tick += 1
            self._rebuild(); self._bounded(); return witness
        except Exception:
            self._rollback(snapshot); raise

    def _rebuild(self):
        groups = {}
        for row in self.witnesses:
            self._charge(1 + len(row.source_roots))
            if not set(row.source_roots).intersection(self.withdrawn_sources):
                groups.setdefault((row.action_recipe, row.constructor_root,
                                   row.candidate_set_root), []).append(row)
        networks = {}
        for (action, constructor, candidate_root), rows in groups.items():
            self._charge(4 + len(rows)
                + sum(len(row.source_roots) for row in rows))
            positive = {row.outcome_source for row in rows if row.difference > 0}
            credit = sum(row.difference for row in rows)
            if len(positive) < 2 or credit <= 0:
                continue
            roots = tuple(sorted(row.identity for row in rows))
            sources = tuple(sorted({source for row in rows for source in row.source_roots}))
            values = (action, constructor, candidate_root,
                      len(rows), credit, roots, sources)
            network = ActionConstructorNetworkV1(
                _identity(b"action-constructor-network-v1", values), *values)
            networks[network.identity] = network
        if len(networks) > MAX_NETWORKS:
            raise ActionConstructorBindingRefuse("action_binding:network_bound")
        self.networks = networks

    def withdraw_source(self, source):
        source = int(source)
        if source <= 0 or self.pending is not None:
            raise ActionConstructorBindingRefuse("action_binding:withdrawal")
        snapshot = self._snapshot()
        try:
            projected = {*self.withdrawn_sources, source}
            self.work = 0; self._charge(len(self.witnesses) + 1
                + self._validation_cost(withdrawn=projected))
            self.withdrawn_sources = projected; self.tick += 1
            self._rebuild(); self._bounded()
        except Exception:
            self._rollback(snapshot); raise

    def _bounded(self):
        if (not 0 < self.work_limit <= MAX_WORK or not 0 <= self.work <= self.work_limit
                or len(self.trials) > MAX_TRIALS
                or len(self.witnesses) > MAX_WITNESSES
                or len(self.networks) > MAX_NETWORKS):
            raise ActionConstructorBindingRefuse("action_binding:bound")
        _strict_numeric((self.tick, self.work, self.work_limit,
            tuple(self.trials), tuple(self.witnesses), tuple(self.networks.values()),
            tuple(sorted(self.withdrawn_sources)),
            () if self.pending is None else (self.pending.trial,
                self.pending.trajectory_root,
                self.pending.trajectory_constituent_roots,
                self.pending.surface_apply_fields)), extent=MAX_ROOTS)
        witness_by_root = {row.identity: row for row in self.witnesses}
        trial_by_root = {row.identity: row for row in self.trials}
        if len(trial_by_root) != len(self.trials):
            raise ActionConstructorBindingRefuse("action_binding:trial_duplicate")
        for row in self.trials:
            self._validate_trial(row)
        if len(witness_by_root) != len(self.witnesses):
            raise ActionConstructorBindingRefuse("action_binding:witness_duplicate")
        for row in self.witnesses:
            values = (row.trial_root, row.command_root, row.action_recipe,
                row.constructor_root, row.candidate_set_root,
                row.trajectory_root, row.trajectory_constituent_roots,
                row.surface_apply_fields,
                row.catalyst_witness_root, row.probe_evidence_root, row.difference,
                row.outcome_source, row.source_roots)
            trial = trial_by_root.get(row.trial_root)
            if (row.identity != _identity(b"action-constructor-witness-v1", values)
                    or trial is None or trial.command_root != row.command_root
                    or trial.action_recipe != row.action_recipe
                    or trial.constructor_root != row.constructor_root
                    or trial.candidate_set_root != row.candidate_set_root
                    or not row.trajectory_root
                    or not row.trajectory_constituent_roots
                    or len(row.surface_apply_fields) != 13
                    or row.surface_apply_fields[0] not in row.source_roots
                    or row.surface_apply_fields[2] != row.command_root
                    or row.surface_apply_fields[4] != row.action_recipe
                    or row.surface_apply_fields[6] != row.trial_root
                    or row.surface_apply_fields[7] != row.constructor_root
                    or row.surface_apply_fields[8] != row.trajectory_root):
                raise ActionConstructorBindingRefuse("action_binding:witness_identity")
        for row in self.networks.values():
            witnesses = [witness_by_root.get(root) for root in row.witness_roots]
            values = (row.action_recipe, row.constructor_root,
                row.candidate_set_root, row.support, row.credit,
                row.witness_roots, row.source_roots)
            if (row.identity != _identity(b"action-constructor-network-v1", values)
                    or row.support != len(witnesses)
                    or any(item is None or item.action_recipe != row.action_recipe
                        or item.constructor_root != row.constructor_root
                        or item.candidate_set_root != row.candidate_set_root
                        for item in witnesses)
                    or row.credit != sum(item.difference for item in witnesses)
                    or row.source_roots != tuple(sorted({source
                        for item in witnesses for source in item.source_roots}))):
                raise ActionConstructorBindingRefuse("action_binding:network_identity")
        if self.pending is not None:
            self._validate_trial(self.pending.trial)

    def _validate_trial(self, row):
        values = (row.command_root, row.action_recipe,
            row.context_nomination_root, row.candidate_set_root,
            row.surface_state_root, row.candidate_roots,
            row.constructor_root, row.source_roots,
            row.opened_tick, row.deadline_tick)
        if (row.identity != _identity(b"action-constructor-trial-v1", values)
                or row.candidate_set_root != self._candidate_root(row.candidate_roots)
                or row.constructor_root not in row.candidate_roots):
            raise ActionConstructorBindingRefuse("action_binding:trial_identity")

    def checkpoint(self):
        body = {"schema": SCHEMA_VERSION, "tick": self.tick,
            "work_limit": self.work_limit, "work": self.work,
            "trials": [asdict(row) for row in self.trials],
            "witnesses": [asdict(row) for row in self.witnesses],
            "networks": [asdict(row) for row in self.networks.values()],
            "withdrawn_sources": sorted(self.withdrawn_sources),
            "pending": None if self.pending is None else {
                "trial": asdict(self.pending.trial),
                "trajectory_root": self.pending.trajectory_root,
                "trajectory_constituent_roots": self.pending.trajectory_constituent_roots,
                "surface_apply_fields": self.pending.surface_apply_fields}}
        blob = json.dumps({"body": body,
                          "hmac": self._boundary._seal(body, _CHECKPOINT_AUTHORITY)},
                          sort_keys=True, separators=(",", ":")).encode()
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise ActionConstructorBindingRefuse("action_binding:checkpoint_bound")
        return blob

    @classmethod
    def restore(cls, blob, boundary):
        try:
            raw = bytes(blob); envelope = json.loads(raw); body = envelope["body"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ActionConstructorBindingRefuse("action_binding:checkpoint") from exc
        if (len(raw) > MAX_CHECKPOINT_BYTES or set(envelope) != {"body", "hmac"}
                or not isinstance(boundary, ActionConstructorCheckpointBoundaryV1)
                or not boundary.valid(body, envelope["hmac"])
                or set(body) != {"schema", "tick", "work_limit", "work",
                    "trials", "witnesses", "networks", "withdrawn_sources", "pending"}
                or body["schema"] != SCHEMA_VERSION):
            raise ActionConstructorBindingRefuse("action_binding:checkpoint_authentication")
        try:
            out = cls(boundary, int(body["work_limit"])); out.tick = int(body["tick"])
            out.work = int(body["work"])
            out.trials = [ActionConstructorTrialV1(int(row["identity"]),
                int(row["command_root"]), int(row["action_recipe"]),
                int(row["context_nomination_root"]),
                int(row["candidate_set_root"]), int(row["surface_state_root"]),
                tuple(map(int, row["candidate_roots"])),
                int(row["constructor_root"]), tuple(map(int, row["source_roots"])),
                int(row["opened_tick"]), int(row["deadline_tick"]))
                for row in body["trials"]]
            out.witnesses = [ActionConstructorWitnessV1(
                int(row["identity"]), int(row["trial_root"]),
                int(row["command_root"]), int(row["action_recipe"]),
                int(row["constructor_root"]), int(row["candidate_set_root"]),
                int(row["trajectory_root"]),
                tuple(map(int, row["trajectory_constituent_roots"])),
                tuple(map(int, row["surface_apply_fields"])),
                int(row["catalyst_witness_root"]),
                int(row["probe_evidence_root"]),
                int(row["difference"]), int(row["outcome_source"]),
                tuple(map(int, row["source_roots"]))) for row in body["witnesses"]]
            networks = [ActionConstructorNetworkV1(
                int(row["identity"]), int(row["action_recipe"]),
                int(row["constructor_root"]), int(row["candidate_set_root"]),
                int(row["support"]), int(row["credit"]),
                tuple(map(int, row["witness_roots"])),
                tuple(map(int, row["source_roots"]))) for row in body["networks"]]
            out.networks = {row.identity: row for row in networks}
            out.withdrawn_sources = set(map(int, body["withdrawn_sources"]))
            pending = body["pending"]
            if pending is not None:
                row = pending["trial"]
                trial = ActionConstructorTrialV1(int(row["identity"]),
                    int(row["command_root"]), int(row["action_recipe"]),
                    int(row["context_nomination_root"]),
                    int(row["candidate_set_root"]), int(row["surface_state_root"]),
                    tuple(map(int, row["candidate_roots"])),
                    int(row["constructor_root"]), tuple(map(int, row["source_roots"])),
                    int(row["opened_tick"]), int(row["deadline_tick"]))
                out.pending = PendingActionConstructorTrialV1(trial,
                    int(pending["trajectory_root"]),
                    tuple(map(int, pending["trajectory_constituent_roots"])),
                    tuple(map(int, pending["surface_apply_fields"])))
            out._bounded()
            if out.checkpoint() != raw:
                raise ActionConstructorBindingRefuse(
                    "action_binding:checkpoint_noncanonical")
            return out
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionConstructorBindingRefuse("action_binding:checkpoint_schema") from exc
