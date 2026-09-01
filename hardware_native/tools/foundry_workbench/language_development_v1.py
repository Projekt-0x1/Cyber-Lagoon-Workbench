#!/usr/bin/env python3
"""Compact developmental language ecology for the graph-neutral Foundry organism.

New design, not compatibility code. Language-specific content is empty at birth.
Grounding, construction order, and discourse connectors are learned from separate
resident/context and raw-surface episode streams over the sparse population bank.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from reference_population_v1 import PopulationBankV1, PopulationSpecV1


class LanguageRefuse(RuntimeError):
    pass


def _unit_id(raw: tuple[int, ...]) -> int:
    if not raw or any(not 0 <= x <= 255 for x in raw):
        raise LanguageRefuse("surface:bytes")
    value = int.from_bytes(hashlib.sha256(bytes(raw)).digest()[:8], "little") & ((1 << 63) - 1)
    return value or 1


@dataclass
class SurfaceUnitV1:
    identity: int
    raw: tuple[int, ...]
    support: int = 1
    structural: int = 0


@dataclass
class RawChunkEvidenceV1:
    unit: int
    source: int
    count: int
    left_diversity: int
    right_diversity: int
    utility: int
    resident_signature: tuple[int, ...] = ()


@dataclass
class AssociationV1:
    unit: int
    signature: tuple[int, ...]
    support: int = 1


@dataclass
class SequenceObservationV1:
    context: tuple[int, ...]
    bindings: tuple[int, ...]
    surface_units: tuple[int, ...]


@dataclass
class ConstructionV1:
    identity: int
    context: tuple[int, ...]
    arity: int
    pieces: tuple[int, ...]  # negative => formal port -(index+1), positive => learned unit
    support: int


@dataclass
class SpanObservationV1:
    context: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]
    surface_units: tuple[int, ...]


@dataclass
class SpanConstructionV1:
    identity: int
    context: tuple[int, ...]
    arity: int
    pieces: tuple[tuple[int, tuple[int, ...]], ...]  # kind 1=child port, 2=constant units
    support: int


@dataclass(frozen=True)
class ClosureV1:
    kind: int  # 1 clause leaf, 2 discourse span
    context_features: tuple[int, ...]
    entity_features: tuple[tuple[int, ...], ...] = ()
    children: tuple["ClosureV1", ...] = ()


class LanguageDevelopmentV1:
    def __init__(self, population_sites: int = 32768):
        self.population = PopulationBankV1(PopulationSpecV1(
            population_sites, fanout=2, sites_per_feature=2))
        self.surface_units: dict[int, SurfaceUnitV1] = {}
        self.raw_chunk_evidence: dict[tuple[int, int], RawChunkEvidenceV1] = {}
        self.withdrawn_raw_sources: set[int] = set()
        self.associations: list[AssociationV1] = []
        self.sequence_observations: list[SequenceObservationV1] = []
        self.constructions: list[ConstructionV1] = []
        self.span_observations: list[SpanObservationV1] = []
        self.span_constructions: list[SpanConstructionV1] = []
        self.episode_features: dict[int, tuple[int, ...]] = {}
        self.episode_surfaces: dict[int, tuple[int, ...]] = {}
        self.utterance_context: dict[int, tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = {}
        self.discourse_context: dict[int, tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = {}
        # Derived execution indices only; never checkpoint/semantic authority.
        self._site_units: dict[int, set[int]] = {}
        self._trie: dict = {}
        self._raw_chunk_first: dict[int, set[int]] = {}
        self._raw_chunk_unit_evidence: dict[int, dict[int, RawChunkEvidenceV1]] = {}
        self.last_candidate_touches = 0
        self.last_raw_chunk_touches = 0
        self.last_raw_stream_bytes = 0

    # ----- separate lived contact streams -----
    def contact_episode_features(self, episode: int, features) -> None:
        features = tuple(int(x) for x in features)
        if not features or episode in self.episode_features:
            raise LanguageRefuse("episode:features")
        self.episode_features[int(episode)] = features

    def contact_surface(self, episode: int, raw) -> None:
        raw = tuple(int(x) for x in raw)
        if not raw or episode in self.episode_surfaces:
            raise LanguageRefuse("episode:surface")
        _unit_id(raw)
        self.episode_surfaces[int(episode)] = raw

    def contact_utterance_context(self, episode: int, context_features, entity_features) -> None:
        context = tuple(int(x) for x in context_features)
        entities = tuple(tuple(int(y) for y in row) for row in entity_features)
        if not context or not entities or episode in self.utterance_context:
            raise LanguageRefuse("episode:utterance_context")
        self.utterance_context[int(episode)] = (context, entities)

    def contact_discourse_context(self, episode: int, context_features, child_units) -> None:
        context = tuple(int(x) for x in context_features)
        children = tuple(tuple(int(y) for y in row) for row in child_units)
        if not context or len(children) < 2 or episode in self.discourse_context:
            raise LanguageRefuse("episode:discourse_context")
        self.discourse_context[int(episode)] = (context, children)

    # ----- generic surface / sparse population learning -----
    def _learn_surface(self, raw: tuple[int, ...], structural: int) -> int:
        identity = _unit_id(raw)
        current = self.surface_units.get(identity)
        if current is not None:
            if current.raw != raw:
                raise LanguageRefuse("surface:hash_collision")
            current.support += 1
            current.structural = max(current.structural, int(bool(structural)))
            return identity
        unit = SurfaceUnitV1(identity, raw, 1, int(bool(structural)))
        self.surface_units[identity] = unit
        node = self._trie
        for byte in raw:
            node = node.setdefault(byte, {})
        node.setdefault(None, set()).add(identity)
        return identity

    def learn_structural_chunk(self, raw) -> int:
        return self._learn_surface(tuple(int(x) for x in raw), 1)

    # ----- raw continuous surface chunk acquisition -----
    # This is deliberately byte-level. Packet boundaries are transport only: all
    # supplied packets are concatenated before recurrence statistics are measured,
    # and no token/word/sentence/language label enters the law. Retained state is
    # bounded reusable chunks plus source-qualified statistics, never the transcript.
    def consolidate_raw_stream(self, packets, source: int, *, max_chunk: int = 12,
                               min_count: int = 8, min_contexts: int = 3,
                               max_units: int = 1024):
        source = int(source)
        if (source <= 0 or not 2 <= max_chunk <= 32 or min_count < 2 or
                min_contexts < 1 or not 32 <= max_units <= 8192):
            raise LanguageRefuse("raw_stream:parameters")
        parts = [bytes(int(x) for x in packet) for packet in packets]
        if not parts or any(not part for part in parts):
            raise LanguageRefuse("raw_stream:packets")
        raw = b"".join(parts)
        if len(raw) < min_count:
            raise LanguageRefuse("raw_stream:short")
        self.last_raw_stream_bytes = len(raw)

        candidates = []
        # Single bytes are physical fallback units. They make unseen combinations
        # representable without installing a tokenizer or expected segmentation.
        byte_counts = [0] * 256
        for value in raw:
            byte_counts[value] += 1
        for value, count in enumerate(byte_counts):
            if count:
                candidates.append((0, bytes((value,)), count, 1, 1))

        # Bounded n-gram census. Context diversity rejects exact repeated phrases
        # whose only recurrence is one fixed frame; compression utility nominates
        # resident matter, but does not itself mint causal/action credit.
        for width in range(2, max_chunk + 1):
            counts = {}
            limit = len(raw) - width + 1
            for i in range(limit):
                chunk = raw[i:i + width]
                counts[chunk] = counts.get(chunk, 0) + 1
            eligible = {chunk: count for chunk, count in counts.items() if count >= min_count}
            if not eligible:
                continue
            left = {chunk: set() for chunk in eligible}
            right = {chunk: set() for chunk in eligible}
            for i in range(limit):
                chunk = raw[i:i + width]
                if chunk not in eligible:
                    continue
                left[chunk].add(-1 if i == 0 else raw[i - 1])
                end = i + width
                right[chunk].add(-1 if end == len(raw) else raw[end])
            for chunk, count in eligible.items():
                ld, rd = len(left[chunk]), len(right[chunk])
                if ld < min_contexts or rd < min_contexts:
                    continue
                utility = (width - 1) * count
                if utility <= width + 8:
                    continue
                candidates.append((utility, chunk, count, ld, rd))

        # Keep all observed byte fallbacks, then the highest-utility reusable chunks.
        singles = [row for row in candidates if len(row[1]) == 1]
        complex_rows = [row for row in candidates if len(row[1]) > 1]
        # Capacity admission is groupwise. Exact utility+extent peers are either
        # all resident or all omitted; raw byte value/address never breaks a tie.
        groups = {}
        for row in complex_rows:
            groups.setdefault((row[0], len(row[1])), []).append(row)
        selected = list(singles)
        for key in sorted(groups, key=lambda k: (-k[0], -k[1])):
            group = groups[key]
            if len(group) <= max_units - len(selected):
                selected.extend(group)
            if len(selected) >= max_units:
                break
        selected_units = []
        for utility, chunk, count, ld, rd in selected:
            units = tuple(chunk)
            uid = _unit_id(units)
            current = self.surface_units.get(uid)
            if current is None:
                self._learn_surface(units, 1)
            elif current.raw != units:
                raise LanguageRefuse("surface:hash_collision")
            resident_signature = self.population.prepare((0x53555246, uid, len(units)))
            evidence = RawChunkEvidenceV1(uid, source, int(count), int(ld), int(rd), int(utility), resident_signature)
            self.raw_chunk_evidence[(uid, source)] = evidence
            self._raw_chunk_unit_evidence.setdefault(uid, {})[source] = evidence
            self._raw_chunk_first.setdefault(chunk[0], set()).add(uid)
            selected_units.append(uid)
        self.withdrawn_raw_sources.discard(source)
        return tuple(selected_units)

    def withdraw_raw_source(self, source: int) -> None:
        self.withdrawn_raw_sources.add(int(source))

    def _raw_unit_utility(self, unit: int):
        rows = [row for source, row in self._raw_chunk_unit_evidence.get(int(unit), {}).items()
                if source not in self.withdrawn_raw_sources]
        self.last_raw_chunk_touches += len(rows)
        if not rows:
            return None
        return sum(row.utility for row in rows), sum(row.count for row in rows)

    def segment_raw(self, raw, active_by_first=None) -> tuple[int, ...]:
        data = bytes(int(x) for x in raw)
        if not data:
            raise LanguageRefuse("raw_segment:empty")
        self.last_raw_chunk_touches = 0
        # Resolve current source-qualified chunk statistics once for this
        # computation. This is a disposable execution cache; it is rebuilt after
        # every withdrawal/reacquisition state and is never checkpoint authority.
        if active_by_first is None:
            active_by_first = {}
            for first, uids in self._raw_chunk_first.items():
                rows = []
                for uid in uids:
                    unit = self.surface_units.get(uid)
                    if unit is None:
                        continue
                    chunk = bytes(unit.raw)
                    learned = self._raw_unit_utility(uid)
                    if chunk and learned is not None:
                        rows.append((int(uid), chunk, int(learned[0]), int(learned[1])))
                if rows:
                    active_by_first[int(first)] = tuple(rows)
        # DP competition uses only learned physical statistics. Order:
        # 1) maximize compression utility, 2) minimize resident unit count,
        # 3) maximize recurrence support. Exact remaining ties refuse.
        #
        # Store one backpointer per byte position instead of copying the complete
        # suffix tuple into every DP cell. This preserves the exact competition law
        # while making long developmental streams O(bytes * local candidates).
        n = len(data)
        scores = [None] * (n + 1)
        chosen_uid = [0] * (n + 1)
        next_pos = [-1] * (n + 1)
        ambiguous = bytearray(n + 1)
        scores[n] = (0, 0, 0)
        next_pos[n] = n
        for i in range(n - 1, -1, -1):
            peak = None
            winner_uid = 0
            winner_next = -1
            winner_ambiguous = False
            tied = False
            for uid, chunk, utility, count in active_by_first.get(data[i], ()):
                if not data.startswith(chunk, i):
                    continue
                j = i + len(chunk)
                tail_score = scores[j]
                if tail_score is None:
                    continue
                score = (tail_score[0] + utility, tail_score[1] - 1,
                         tail_score[2] + min(count, 65535))
                if peak is None or score > peak:
                    peak = score
                    winner_uid = int(uid)
                    winner_next = j
                    winner_ambiguous = bool(ambiguous[j])
                    tied = False
                elif score == peak and (int(uid) != winner_uid or j != winner_next):
                    tied = True
            if peak is None:
                continue
            scores[i] = peak
            chosen_uid[i] = winner_uid
            next_pos[i] = winner_next
            ambiguous[i] = int(winner_ambiguous or tied)
        if scores[0] is None or ambiguous[0]:
            raise LanguageRefuse("raw_segment:ambiguous_or_missing")
        out = []
        cursor = 0
        while cursor < n:
            uid = int(chosen_uid[cursor]);nxt = int(next_pos[cursor])
            if uid <= 0 or not cursor < nxt <= n:
                raise LanguageRefuse("raw_segment:backpointer")
            out.append(uid);cursor = nxt
        return tuple(out)

    def raw_segmentation_stats(self, raw):
        data = bytes(int(x) for x in raw)
        units = self.segment_raw(data)
        multi = sum(len(self.surface_units[uid].raw) for uid in units
                    if len(self.surface_units[uid].raw) > 1)
        return {"bytes": len(data), "units": len(units), "multi_byte_coverage": multi,
                "coverage_fraction": 0.0 if not data else multi / len(data),
                "candidate_touches": self.last_raw_chunk_touches}

    def raw_chunk_lattice_stats(self, raw, max_candidates_per_position: int = 64):
        data = bytes(int(x) for x in raw)
        if not data or not 1 <= max_candidates_per_position <= 256:
            raise LanguageRefuse("raw_lattice:shape")
        self.last_raw_chunk_touches = 0
        cover_delta = [0] * (len(data) + 1)
        positions = ambiguous = total_candidates = maximum = 0
        multi_candidates = 0
        for i, first in enumerate(data):
            rows = []
            for uid in self._raw_chunk_first.get(first, ()):
                unit = self.surface_units.get(uid)
                if unit is None:
                    continue
                chunk = bytes(unit.raw)
                if not data.startswith(chunk, i):
                    continue
                learned = self._raw_unit_utility(uid)
                if learned is None:
                    continue
                utility, count = learned
                rows.append((utility, count, len(chunk), uid))
            if not rows:
                continue
            # Resource bound is selected by learned physical utility only; equal rows
            # remain coactive rather than being ordered by identity. The cap refuses
            # instead of silently pruning a semantically inconvenient alternative.
            if len(rows) > max_candidates_per_position:
                raise LanguageRefuse("raw_lattice:candidate_bound")
            positions += 1
            total_candidates += len(rows)
            maximum = max(maximum, len(rows))
            if len(rows) > 1:
                ambiguous += 1
            for _utility, _count, width, _uid in rows:
                if width <= 1:
                    continue
                multi_candidates += 1
                cover_delta[i] += 1
                cover_delta[min(len(data), i + width)] -= 1
        active = covered = 0
        for i in range(len(data)):
            active += cover_delta[i]
            covered += active > 0
        return {
            "bytes": len(data),
            "positions_with_candidates": positions,
            "ambiguous_positions": ambiguous,
            "total_candidates": total_candidates,
            "multi_byte_candidates": multi_candidates,
            "max_candidates_per_position": maximum,
            "multi_byte_covered_bytes": covered,
            "coverage_fraction": covered / len(data),
            "candidate_touches": self.last_raw_chunk_touches,
        }

    def raw_chunk_occurrence_network_stats(self, raw, max_edges_per_position: int = 16):
        data = bytes(int(x) for x in raw)
        if not data or not 1 <= max_edges_per_position <= 64:
            raise LanguageRefuse("raw_network:shape")
        # Learned chunks are persistent prepared matter; every present match recruits
        # a FRESH ephemeral Occurrence. The active span Network is computation, not
        # a sentence/word object and not lifetime history.
        retained_before = len(self.population.occurrences)
        active = []
        for start, first in enumerate(data):
            for uid in self._raw_chunk_first.get(first, ()):
                unit = self.surface_units.get(uid)
                if unit is None:
                    continue
                chunk = bytes(unit.raw)
                if not data.startswith(chunk, start):
                    continue
                evidence_rows = [row for source, row in self._raw_chunk_unit_evidence.get(uid, {}).items()
                                 if source not in self.withdrawn_raw_sources]
                if not evidence_rows:
                    continue
                occurrence = self.population.activate(
                    (0x53555246, uid, start, start + len(chunk)), retain=False)
                active.append((start, start + len(chunk), uid, occurrence))
        edges = 0; touched = 0; max_degree = 0; connected = set()
        # Explicit observer wire for the state we deliberately do NOT checkpoint:
        # <occurrence_id:u64,tick:u64,unit:u64,begin:u32,end:u32,feature_count:u32,
        #   site_count:u32,edge_count:u32,sites:u32[],edges:u32[]> per Occurrence.
        unfolded_occurrence_wire_bytes = 0
        for begin, end, uid, occurrence in active:
            unfolded_occurrence_wire_bytes += 44 + 4 * len(occurrence.sites) + 4 * len(occurrence.edges)
        by_start = {}
        for index, row in enumerate(active):
            by_start.setdefault(row[0], []).append(index)
        for index, (begin, end, uid, occurrence) in enumerate(active):
            local = []
            for next_start in range(begin + 1, min(len(data), end + 1) + 1):
                for other in by_start.get(next_start, ()):
                    if other == index:
                        continue
                    obegin, oend, ouid, ooccurrence = active[other]
                    local.append((other, obegin < end, obegin == end))
            if len(local) > max_edges_per_position:
                # Current resource pressure refuses this local fanout; it never picks
                # a token/sentence path by host ordering.
                continue
            max_degree = max(max_degree, len(local)); touched += len(local)
            for other, overlap, adjacent in local:
                if not (overlap or adjacent):
                    continue
                edges += 1; connected.add(index); connected.add(other)
        return {
          "bytes":len(data), "active_chunk_occurrences":len(active),
          "networked_chunk_occurrences":len(connected), "span_edges":edges,
          "max_local_degree":max_degree, "edge_candidate_touches":touched,
          # Edge wire is <source_index:u32,target_index:u32,relation_flags:u8>.
          "unfolded_occurrence_wire_bytes":unfolded_occurrence_wire_bytes,
          "unfolded_edge_wire_bytes":edges * 9,
          "unfolded_total_wire_bytes":unfolded_occurrence_wire_bytes + edges * 9,
          "retained_occurrences_before":retained_before,
          "retained_occurrences_after":len(self.population.occurrences),
          "ephemeral_network":len(self.population.occurrences)==retained_before,
        }

    def consolidate_grounded_episode(self, episode: int) -> int:
        if episode not in self.episode_features or episode not in self.episode_surfaces:
            raise LanguageRefuse("grounding:incomplete_episode")
        features = self.episode_features[episode]
        raw = self.episode_surfaces[episode]
        occurrence = self.population.recruit(features)
        unit = self._learn_surface(raw, 0)
        signature = occurrence.sites
        existing = next((a for a in self.associations
                         if a.unit == unit and a.signature == signature), None)
        if existing is None:
            existing = AssociationV1(unit, signature, 1)
            self.associations.append(existing)
            for site in signature:
                self._site_units.setdefault(site, set()).add(unit)
        else:
            existing.support += 1
        return unit

    def _association_rows(self, unit: int):
        return [a for a in self.associations if a.unit == unit]

    def resolve_features(self, features, minimum_overlap: int = 2) -> int:
        cue = self.population.signature(tuple(int(x) for x in features))
        if len(cue) < minimum_overlap:
            raise LanguageRefuse("resolve:cue")
        posting = sorted((len(self._site_units.get(site, ())), site) for site in cue)
        nominated: set[int] = set()
        touches = 0
        for _, site in posting[:minimum_overlap]:
            rows = self._site_units.get(site, ())
            touches += len(rows)
            nominated.update(rows)
        scored = []
        for unit in nominated:
            for assoc in self._association_rows(unit):
                score = self.population.overlap(cue, assoc.signature)
                touches += len(assoc.signature)
                if score >= minimum_overlap:
                    scored.append((score, assoc.support, unit))
        self.last_candidate_touches = touches
        if not scored:
            raise LanguageRefuse("resolve:none")
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
        peak = scored[0][:2]
        winners = sorted(set(row[2] for row in scored if row[:2] == peak))
        if len(winners) != 1:
            raise LanguageRefuse("resolve:ambiguous")
        return winners[0]

    def segment(self, raw) -> tuple[int, ...]:
        raw = tuple(int(x) for x in raw)
        paths: dict[int, list[tuple[int, ...]]] = {0: [()]}
        for start in range(len(raw)):
            if start not in paths:
                continue
            node = self._trie
            for end in range(start, len(raw)):
                node = node.get(raw[end])
                if node is None:
                    break
                for unit in node.get(None, ()):
                    target = end + 1
                    bucket = paths.setdefault(target, [])
                    for prefix in paths[start]:
                        candidate = prefix + (unit,)
                        if candidate not in bucket:
                            bucket.append(candidate)
                        if len(bucket) > 1:
                            bucket[:] = bucket[:2]
        finals = paths.get(len(raw), [])
        if len(finals) != 1:
            raise LanguageRefuse("segment:ambiguous_or_missing")
        return finals[0]

    # ----- learned clause constructions -----
    def observe_utterance_episode(self, episode: int) -> None:
        if episode not in self.utterance_context or episode not in self.episode_surfaces:
            raise LanguageRefuse("utterance:incomplete_episode")
        context_features, entities = self.utterance_context[episode]
        context = self.population.signature(context_features)
        bindings = tuple(self.resolve_features(row) for row in entities)
        surface = self.segment(self.episode_surfaces[episode])
        self.sequence_observations.append(SequenceObservationV1(context, bindings, surface))

    def _induce_sequence(self, rows: list[SequenceObservationV1]):
        arity = len(rows[0].bindings)
        width = len(rows[0].surface_units)
        if any(len(r.bindings) != arity or len(r.surface_units) != width for r in rows):
            raise LanguageRefuse("construction:shape")
        pieces = []
        used = [0] * arity
        for pos in range(width):
            ports = [i for i in range(arity)
                     if all(r.surface_units[pos] == r.bindings[i] for r in rows)]
            if len(ports) == 1:
                used[ports[0]] += 1
                pieces.append(-(ports[0] + 1))
                continue
            if len(ports) > 1:
                raise LanguageRefuse("construction:ambiguous_port")
            fixed = rows[0].surface_units[pos]
            if any(r.surface_units[pos] != fixed for r in rows[1:]):
                raise LanguageRefuse("construction:unexplained_variation")
            pieces.append(fixed)
        if any(x == 0 for x in used):
            raise LanguageRefuse("construction:unused_binding")
        return arity, tuple(pieces)

    def consolidate_construction(self, context_features, minimum_support: int = 2) -> ConstructionV1:
        context = self.population.signature(tuple(int(x) for x in context_features))
        rows = [r for r in self.sequence_observations if r.context == context]
        if len(rows) < minimum_support:
            raise LanguageRefuse("construction:support")
        arity, pieces = self._induce_sequence(rows)
        identity = int.from_bytes(hashlib.sha256(repr((context, arity, pieces)).encode()).digest()[:8], "little") or 1
        found = next((c for c in self.constructions if c.identity == identity), None)
        if found:
            found.support = max(found.support, len(rows))
            return found
        c = ConstructionV1(identity, context, arity, pieces, len(rows))
        self.constructions.append(c)
        return c

    def _select_context(self, rows, context_features):
        cue = self.population.signature(tuple(int(x) for x in context_features))
        scored = [(self.population.overlap(cue, row.context), row.support, row.identity, row)
                  for row in rows]
        scored = [x for x in scored if x[0] > 0]
        if not scored:
            raise LanguageRefuse("context:none")
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        peak = scored[0][:2]
        winners = [x for x in scored if x[:2] == peak]
        if len(winners) != 1:
            raise LanguageRefuse("context:ambiguous")
        return winners[0][3]

    def realize_clause_units(self, context_features, entity_features) -> tuple[int, ...]:
        construction = self._select_context(self.constructions, context_features)
        bindings = tuple(self.resolve_features(row) for row in entity_features)
        if len(bindings) != construction.arity:
            raise LanguageRefuse("construction:arity")
        out = []
        for piece in construction.pieces:
            if piece < 0:
                out.append(bindings[-piece - 1])
            else:
                out.append(piece)
        return tuple(out)

    # ----- learned higher discourse constructions -----
    @staticmethod
    def _subsequence_hits(sequence, needle):
        n = len(needle)
        return [i for i in range(len(sequence) - n + 1)
                if tuple(sequence[i:i+n]) == tuple(needle)]

    def observe_discourse_episode(self, episode: int) -> None:
        if episode not in self.discourse_context or episode not in self.episode_surfaces:
            raise LanguageRefuse("discourse:incomplete_episode")
        context_features, children = self.discourse_context[episode]
        context = self.population.signature(context_features)
        surface = self.segment(self.episode_surfaces[episode])
        self.span_observations.append(SpanObservationV1(context, children, surface))

    def _span_pattern(self, row: SpanObservationV1):
        spans = []
        for index, child in enumerate(row.children):
            hits = self._subsequence_hits(row.surface_units, child)
            if len(hits) != 1:
                raise LanguageRefuse("span:child_cardinality")
            spans.append((hits[0], hits[0] + len(child), index))
        spans.sort()
        if any(spans[i][1] > spans[i+1][0] for i in range(len(spans)-1)):
            raise LanguageRefuse("span:overlap")
        pieces = []
        cursor = 0
        for begin, end, index in spans:
            if begin > cursor:
                pieces.append((2, tuple(row.surface_units[cursor:begin])))
            pieces.append((1, (index,)))
            cursor = end
        if cursor < len(row.surface_units):
            pieces.append((2, tuple(row.surface_units[cursor:])))
        return tuple(pieces)

    def consolidate_span(self, context_features, minimum_support: int = 2) -> SpanConstructionV1:
        context = self.population.signature(tuple(int(x) for x in context_features))
        rows = [r for r in self.span_observations if r.context == context]
        if len(rows) < minimum_support:
            raise LanguageRefuse("span:support")
        patterns = [self._span_pattern(r) for r in rows]
        if any(p != patterns[0] for p in patterns[1:]):
            raise LanguageRefuse("span:variation")
        arity = len(rows[0].children)
        identity = int.from_bytes(hashlib.sha256(repr((context, arity, patterns[0])).encode()).digest()[:8], "little") or 1
        c = SpanConstructionV1(identity, context, arity, patterns[0], len(rows))
        existing = next((x for x in self.span_constructions if x.identity == identity), None)
        if existing:
            existing.support = max(existing.support, len(rows))
            return existing
        self.span_constructions.append(c)
        return c

    def realize_span_units(self, context_features, children) -> tuple[int, ...]:
        c = self._select_context(self.span_constructions, context_features)
        children = tuple(tuple(x) for x in children)
        if len(children) != c.arity:
            raise LanguageRefuse("span:arity")
        out = []
        for kind, data in c.pieces:
            if kind == 1:
                out.extend(children[data[0]])
            elif kind == 2:
                out.extend(data)
            else:
                raise LanguageRefuse("span:piece")
        return tuple(out)

    def realize_closure_units(self, closure: ClosureV1, depth: int = 0) -> tuple[int, ...]:
        if depth > 32:
            raise LanguageRefuse("closure:depth")
        if closure.kind == 1:
            if closure.children:
                raise LanguageRefuse("closure:leaf_children")
            return self.realize_clause_units(closure.context_features, closure.entity_features)
        if closure.kind == 2:
            if not closure.children:
                raise LanguageRefuse("closure:span_children")
            children = tuple(self.realize_closure_units(x, depth + 1) for x in closure.children)
            return self.realize_span_units(closure.context_features, children)
        raise LanguageRefuse("closure:kind")

    def render_units(self, units) -> bytes:
        out = bytearray()
        for unit in units:
            row = self.surface_units.get(int(unit))
            if row is None:
                raise LanguageRefuse("surface:missing_unit")
            out.extend(row.raw)
        return bytes(out)

    def render_closure(self, closure: ClosureV1) -> bytes:
        return self.render_units(self.realize_closure_units(closure))

    def raw_ecology_checkpoint(self):
        """Persist the compact reusable law, not deterministic/unfolded state."""
        # This compact form is intentionally for the raw prepared ecology. If actual
        # causally settled population history exists, use the organism checkpoint path
        # rather than silently dropping it here.
        if (self.population.occurrences or self.population.credit_events or
                self.population.revision_events or self.population.live_eligibility_count()):
            raise LanguageRefuse("raw_checkpoint:causal_state_present")
        if self.population.sparse_edge_weight_count():
            raise LanguageRefuse("raw_checkpoint:revised_edge_present")
        prepared = [[site, support] for site,support in self.population.sparse_support_items()]
        return {
          "schema": 2,
          "population": {
            "spec": self.population.spec.__dict__,
            "tick": self.population.tick,
            "prepared": prepared,
          },
          "surface_units": [
            [row.identity, list(row.raw), row.support, row.structural]
            for row in sorted(self.surface_units.values(), key=lambda row: row.identity)
          ],
          "raw_chunk_evidence": [
            [row.unit, row.source, row.count, row.left_diversity,
             row.right_diversity, row.utility, list(row.resident_signature)]
            for row in sorted(self.raw_chunk_evidence.values(),
                              key=lambda row: (row.unit, row.source))
          ],
          "withdrawn_raw_sources": sorted(self.withdrawn_raw_sources),
          "last_raw_stream_bytes": self.last_raw_stream_bytes,
        }

    @classmethod
    def restore_raw_ecology(cls, checkpoint):
        if checkpoint.get("schema") != 2:
            raise LanguageRefuse("raw_checkpoint:schema")
        pop = checkpoint.get("population", {})
        spec = PopulationSpecV1(**pop.get("spec", {}))
        spec.validate()
        population = PopulationBankV1(spec)
        population.tick = int(pop.get("tick", 0))
        population.next_occurrence = 1
        seen_sites = set()
        for values in pop.get("prepared", ()):
            if len(values) not in (2,3):
                raise LanguageRefuse("raw_checkpoint:prepared_shape")
            site, support = map(int, values[:2])
            if (site in seen_sites or not 0 <= site < spec.site_count or
                    not 1 <= support <= 65535):
                raise LanguageRefuse("raw_checkpoint:prepared_value")
            seen_sites.add(site)
            population.support[site] = support
        machine = cls.__new__(cls)
        machine.population = population
        machine.surface_units = {}
        machine.raw_chunk_evidence = {}
        machine.withdrawn_raw_sources = set(map(int, checkpoint.get("withdrawn_raw_sources", ())))
        machine.associations = []
        machine.sequence_observations = []
        machine.constructions = []
        machine.span_observations = []
        machine.span_constructions = []
        machine.episode_features = {}
        machine.episode_surfaces = {}
        machine.utterance_context = {}
        machine.discourse_context = {}
        machine._site_units = {}
        machine._trie = {}
        machine._raw_chunk_first = {}
        machine._raw_chunk_unit_evidence = {}
        machine.last_candidate_touches = 0
        machine.last_raw_chunk_touches = 0
        machine.last_raw_stream_bytes = int(checkpoint.get("last_raw_stream_bytes", 0))
        seen_units = set()
        for identity, raw, support, structural in checkpoint.get("surface_units", ()):
            units = tuple(map(int, raw)); identity = int(identity)
            if identity in seen_units or _unit_id(units) != identity:
                raise LanguageRefuse("raw_checkpoint:surface_identity")
            seen_units.add(identity)
            row = SurfaceUnitV1(identity, units, int(support), int(structural))
            machine.surface_units[row.identity] = row
            node = machine._trie
            for byte in units:
                node = node.setdefault(byte, {})
            node.setdefault(None, set()).add(row.identity)
        for values in checkpoint.get("raw_chunk_evidence", ()):
            if len(values) != 7:
                raise LanguageRefuse("raw_checkpoint:evidence_shape")
            unit, source, count, ld, rd, utility, signature = values
            unit = int(unit); source = int(source); signature = tuple(map(int, signature))
            surface = machine.surface_units.get(unit)
            if surface is None or not signature or machine.population.signature(
                    (0x53555246, unit, len(surface.raw))) != signature:
                raise LanguageRefuse("raw_checkpoint:evidence_binding")
            row = RawChunkEvidenceV1(unit, source, int(count), int(ld), int(rd),
                                     int(utility), signature)
            key = (unit, source)
            if key in machine.raw_chunk_evidence:
                raise LanguageRefuse("raw_checkpoint:evidence_duplicate")
            machine.raw_chunk_evidence[key] = row
            machine._raw_chunk_unit_evidence.setdefault(unit, {})[source] = row
            machine._raw_chunk_first.setdefault(surface.raw[0], set()).add(unit)
        return machine

    def quantity(self):
        return {
            "population_sites": self.population.spec.site_count,
            "population_edges": self.population.allocated_edge_count,
            "materialized_sites": self.population.materialized_site_count(),
            "surface_units": len(self.surface_units),
            "raw_chunk_evidence": len(self.raw_chunk_evidence),
            "raw_chunk_sources": len({source for _unit, source in self.raw_chunk_evidence}),
            "raw_chunk_prepared_signatures": len({row.resident_signature for row in self.raw_chunk_evidence.values() if row.resident_signature}),
            "last_raw_chunk_touches": self.last_raw_chunk_touches,
            "last_raw_stream_bytes": self.last_raw_stream_bytes,
            "associations": len(self.associations),
            "constructions": len(self.constructions),
            "span_constructions": len(self.span_constructions),
            "last_candidate_touches": self.last_candidate_touches,
        }
