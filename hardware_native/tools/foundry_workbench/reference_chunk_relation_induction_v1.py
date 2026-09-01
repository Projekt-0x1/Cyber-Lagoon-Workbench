#!/usr/bin/env python3
"""Compact language Recipes that unfold into ephemeral computational Networks.

The lower surface ecology (`LanguageDevelopmentV1`) learns reusable byte chunks
from raw contact without a tokenizer.  This layer persists only compact Recipe
mathematics over those resident terms.  A current contact unfolds the Recipes
into a transient NetworkOccurrence containing the current anchor Occurrences,
port intervals and competition receipt.  Words/sentences/thoughts are not
persistent runtime object classes.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import struct

from language_development_v1 import LanguageDevelopmentV1, LanguageRefuse


class ChunkRelationRefuse(ValueError):
    pass


@dataclass(frozen=True)
class ChunkAnchorOccurrenceV1:
    start: int
    end: int
    unit: int
    utility: int
    support: int


@dataclass(frozen=True)
class PortGuardV1:
    min_bytes: int
    max_bytes: int
    may_be_empty: int


@dataclass(frozen=True)
class PortPairGuardV1:
    left_port: int
    right_port: int
    relation: int  # 1=same current binding, 2=different current binding


@dataclass(frozen=True)
class ChunkRecipeV1:
    identity: int
    anchors: tuple[int, ...]
    support: int
    # The Recipe stores only relation shape and compact guards. Actual bound
    # values belong to transient current/reinstated Network state.
    port_guards: tuple[PortGuardV1, ...]
    pair_guards: tuple[PortPairGuardV1, ...]
    min_bytes: int
    max_bytes: int

    @property
    def anchor_count(self) -> int:
        return len(self.anchors)


@dataclass(frozen=True)
class ChunkNetworkOccurrenceV1:
    identity: int
    recipe_identity: int
    anchor_occurrences: tuple[tuple[int, int, int], ...]
    port_intervals: tuple[tuple[int, int], ...]
    candidate_touches: int
    lattice_occurrences: int


@dataclass(frozen=True)
class ChunkRelationScoreV1:
    recipe_identity: int
    support: int
    anchor_count: int
    structure_score: float
    candidate_touches: int
    lattice_occurrences: int
    multi_byte_coverage: float

    @property
    def relation_identity(self) -> int:  # observer vocabulary only
        return self.recipe_identity


class ChunkRelationInductionV1:
    NETWORK_MORPHOLOGY_TAG = 0x43524E31  # "CRN1", generic lowering only
    RECURSIVE_MORPHOLOGY_TAG = 0x43524E32  # "CRN2", nested Recipe geometry only
    CLOSURE_MORPHOLOGY_TAG = 0x43524E33  # "CRN3", coactive Recipe closure only
    MIN_ANCHORS = 2
    MAX_ANCHORS = 4
    MIN_SUPPORT = 2
    MAX_RECIPES = 20
    MAX_RECIPE_CANDIDATES = 512
    MAX_OCCURRENCES_PER_CONTACT = 192
    MAX_PAIR_CANDIDATES = 250_000
    MIN_PAIR_NOMINATION_ANCHORS = 6
    MAX_PAIR_NOMINATION_ANCHORS = 10
    MAX_PORT_ALTERNATIVES = 32
    MAX_NOMINATED_CURRENT = 384
    STREAM_WINDOW_APERTURES = (96,)
    STREAM_MIN_WINDOW_BYTES = 24

    def __init__(self, surface: LanguageDevelopmentV1):
        self.surface = surface
        # Ingress packet boundaries are transport only. Construction observes one
        # continuous raw chronology, then derives bounded overlapping windows from
        # byte position. Those windows are disposable work, not mature state.
        self._stream = bytearray()
        self._windows: list[bytes] = []
        self._window_offsets: list[tuple[int, int]] = []
        self._ingress_packet_count = 0
        self.contact_count = 0
        self.contact_bytes = 0
        self.construction_root = bytes(32)
        self.recipes: dict[int, ChunkRecipeV1] = {}
        self._anchor_recipes: dict[int, dict[int, tuple[int, ...]]] = {}
        self._recipe_bit_order: tuple[int, ...] = ()
        self._anchor_recipe_bits: dict[int, int] = {}
        self._active_surface: dict[int, tuple[tuple[int, bytes, int, int], ...]] = {}
        self._surface_support: dict[int, int] = {}
        self.last_candidate_touches = 0
        self.last_lattice_occurrences = 0
        self.last_multi_byte_coverage = 0.0
        self.pair_candidates_examined = 0

    @property
    def relations(self):  # observer-facing compatibility while this Foundry hypothesis is evaluated
        return self.recipes

    @staticmethod
    def _identity(tag: bytes, values) -> int:
        body = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(tag + b"\0" + body).digest()[:8], "big") or 1

    def ingest(self, payload) -> int:
        raw = bytes(int(x) & 0xFF for x in payload)
        if not raw:
            raise ChunkRelationRefuse("chunk_recipe:empty")
        self._stream.extend(raw)
        self._ingress_packet_count += 1
        self.contact_bytes += len(raw)
        return self._ingress_packet_count

    def _derive_windows(self) -> None:
        data = bytes(self._stream)
        if len(data) < self.STREAM_MIN_WINDOW_BYTES:
            raise ChunkRelationRefuse("chunk_recipe:insufficient_stream")
        windows = []
        seen = set()
        for width in self.STREAM_WINDOW_APERTURES:
            stride = max(16, int(width) // 2)
            starts = list(range(
                0, max(1, len(data) - self.STREAM_MIN_WINDOW_BYTES + 1), stride,
            ))
            tail_start = max(0, len(data) - int(width))
            if tail_start not in starts:
                starts.append(tail_start)
            for start in starts:
                row = data[start:start + int(width)]
                key = (int(start), len(row))
                if len(row) >= self.STREAM_MIN_WINDOW_BYTES and key not in seen:
                    seen.add(key);windows.append((int(start), row))
        windows.sort(key=lambda item: (item[0], len(item[1])))
        if len(windows) < 8:
            raise ChunkRelationRefuse("chunk_recipe:insufficient_windows")
        self._window_offsets = [(start, start + len(row)) for start, row in windows]
        self._windows = [row for _start, row in windows]
        # These apertures are derived from continuous byte chronology rather than
        # ingress packets. Packetization therefore cannot author language units.
        self.contact_count = len(windows)

    def compact_training_buffer(self) -> int:
        released = len(self._stream)
        self._stream.clear()
        self._windows.clear()
        self._window_offsets.clear()
        return released

    def _build_active_surface(self) -> None:
        active = {}
        support_by_uid = {}
        for first, uids in self.surface._raw_chunk_first.items():
            rows = []
            for uid in uids:
                unit = self.surface.surface_units.get(uid)
                if unit is None:
                    continue
                learned = self.surface._raw_unit_utility(uid)
                if learned is None:
                    continue
                utility, count = learned
                rows.append((int(uid), bytes(unit.raw), int(utility), int(count)))
                support_by_uid[int(uid)] = max(1, int(count))
            active[int(first)] = tuple(rows)
        self._active_surface = active
        self._surface_support = support_by_uid

    def _anchor_support(self, uid: int) -> int:
        if not self._active_surface:
            self._build_active_surface()
        return self._surface_support.get(int(uid), 1 << 30)

    def _anchor_nomination_value(self, uid: int) -> float:
        """Recurrent information aperture for higher-rank construction search.

        Lower surface admission has already required recurrence and contextual
        diversity.  Higher-rank nomination must therefore not rank raw frequency
        again, which collapses resident Recipes onto spaces/common bigrams.  Byte
        extent times log recurrence preserves frequent structural glue while
        giving recurrent longer chunks enough aperture to enter N+1 relations.
        """
        unit = self.surface.surface_units.get(int(uid))
        if unit is None:
            return 0.0
        raw = bytes(unit.raw)
        extent = max(0, len(raw) - 1)
        if extent == 0:
            return 0.0
        counts = Counter(raw)
        entropy = -sum(
            (count / len(raw)) * math.log2(count / len(raw))
            for count in counts.values()
        )
        entropy_cap = math.log2(min(256, len(raw)))
        entropy_fraction = entropy / entropy_cap if entropy_cap > 0 else 0.0
        return (extent * math.log2(1 + self._anchor_support(uid))
                * (0.25 + entropy_fraction))

    def lattice(self, raw) -> tuple[ChunkAnchorOccurrenceV1, ...]:
        data = bytes(int(x) & 0xFF for x in raw)
        if not self._active_surface:
            self._build_active_surface()
        rows = []
        for start, first in enumerate(data):
            for uid, chunk, utility, support in self._active_surface.get(first, ()):
                if len(chunk) <= 1 or not data.startswith(chunk, start):
                    continue
                rows.append(ChunkAnchorOccurrenceV1(
                    start, start + len(chunk), uid, utility, support,
                ))
        rows.sort(key=lambda row: (row.start, -row.utility, -(row.end - row.start), row.unit))
        if len(rows) > self.MAX_OCCURRENCES_PER_CONTACT:
            groups = defaultdict(list)
            for row in rows:
                groups[(row.utility, row.support, row.end-row.start)].append(row)
            selected = []
            for key in sorted(groups, reverse=True):
                group = groups[key]
                if len(group) > self.MAX_OCCURRENCES_PER_CONTACT - len(selected):
                    break
                selected.extend(group)
            rows = sorted(selected, key=lambda row: (row.start, row.end, row.unit))
        self.last_lattice_occurrences = len(rows)
        if not data or not rows:
            self.last_multi_byte_coverage = 0.0
        else:
            covered = 0
            begin = end = -1
            for row in rows:
                if row.start > end:
                    if end > begin:
                        covered += end - begin
                    begin, end = row.start, row.end
                elif row.end > end:
                    end = row.end
            if end > begin:
                covered += end - begin
            self.last_multi_byte_coverage = covered / len(data)
        return tuple(rows)

    @staticmethod
    def _ordered_common(left, right, cap):
        positions = defaultdict(list)
        for index, uid in enumerate(right):
            positions[int(uid)].append(index)
        cursors = {uid: 0 for uid in positions}
        out = []
        floor = -1
        for uid in left:
            values = positions.get(int(uid))
            if not values:
                continue
            cursor = cursors[int(uid)]
            while cursor < len(values) and values[cursor] <= floor:
                cursor += 1
            cursors[int(uid)] = cursor
            if cursor >= len(values):
                continue
            floor = values[cursor]
            cursors[int(uid)] = cursor + 1
            out.append(int(uid))
            if len(out) >= cap:
                break
        return tuple(out)

    def _align(self, raw: bytes, anchors):
        cursor = 0
        rows = []
        for uid in anchors:
            unit = self.surface.surface_units.get(int(uid))
            if unit is None:
                return None
            chunk = bytes(unit.raw)
            pos = raw.find(chunk, cursor)
            if pos < 0:
                return None
            rows.append((pos, pos + len(chunk), int(uid)))
            cursor = pos + len(chunk)
        return tuple(rows)

    @staticmethod
    def _port_intervals(length: int, aligned):
        if not aligned:
            return ()
        intervals = [(0, aligned[0][0])]
        for left, right in zip(aligned, aligned[1:]):
            intervals.append((left[1], right[0]))
        intervals.append((aligned[-1][1], length))
        return tuple(intervals)

    def _rebuild_index(self) -> None:
        index = defaultdict(lambda: defaultdict(list))
        bit_order = tuple(sorted(self.recipes))
        bit_for = {rid: 1 << pos for pos, rid in enumerate(bit_order)}
        anchor_bits = defaultdict(int)
        for recipe in self.recipes.values():
            distinct = sorted(
                set(recipe.anchors),
                key=lambda uid: (self._anchor_support(uid), uid),
            )
            if len(distinct) < 2:
                continue
            first, second = distinct[:2]
            index[first][second].append(recipe.identity)
            bit = bit_for[recipe.identity]
            anchor_bits[first] |= bit
            anchor_bits[second] |= bit
        self._anchor_recipes = {
            first: {
                second: tuple(sorted(ids, key=lambda rid: (
                    -(self.recipes[rid].support * self.recipes[rid].anchor_count), rid,
                )))
                for second, ids in inner.items()
            }
            for first, inner in index.items()
        }
        self._recipe_bit_order = bit_order
        self._anchor_recipe_bits = dict(anchor_bits)

    def consolidate(self) -> None:
        self._derive_windows()
        self._build_active_surface()
        construction = hashlib.sha256(b"chunk-recipe-construction-v3\0")
        construction.update(bytes(self._stream))
        full_posting = defaultdict(list)
        lattice_wires = []
        # Construction sees the same ambiguous interval lattice as runtime. A
        # persistent order may be nominated only by Occurrences that can coexist
        # physically (left.end <= right.start). We therefore neither flatten all
        # overlaps into a fake token sequence nor force one global segmentation.
        # Lattices are disposable construction work. Preserve their exact ambiguous
        # interval geometry once in a compact <start:u16,end:u16,unit:u64> wire,
        # rather than retaining Python Occurrence objects or recomputing the lattice
        # during the recurrence pass.
        for index, raw in enumerate(self._windows):
            lattice = self.lattice(raw)
            wire = bytearray()
            for row in lattice:
                wire.extend(struct.pack("<HHQ", int(row.start), int(row.end), int(row.unit)))
            lattice_wires.append(bytes(wire))
            for uid in {int(row.unit) for row in lattice}:
                full_posting[uid].append(index)
        self.construction_root = construction.digest()
        roots = [hashlib.sha256(raw).digest() for raw in self._windows]

        # Mature state is a small set of recurrent local relations, not a cache of
        # windows. A bounded informative frontier nominates candidates; interval
        # compatibility supplies the local ordering witness; full incidence below
        # still verifies support. Held-out bytes never participate.
        # A candidate cannot mature until a second independent window witnesses it.
        # Do not retain singleton hypotheses at all: use the already-built exact
        # per-anchor window postings to test a newly observed ordered tuple against
        # every window in which all of its anchors co-occur. Only genuinely recurrent
        # ordered relations enter candidate state; lifetime novelty therefore does not
        # create an unbounded singleton ledger in an online learner.
        candidate_votes = Counter()
        posting_sets = {uid: set(ids) for uid, ids in full_posting.items()}
        wire_stride = struct.calcsize("<HHQ")

        def unpack_lattice(wire):
            return [
                struct.unpack_from("<HHQ", wire, offset)
                for offset in range(0, len(wire), wire_stride)
            ]

        def admitted_for(lattice):
            available = {int(row[2]) for row in lattice}
            recurrent_available = sum(
                len(full_posting[uid]) >= self.MIN_SUPPORT for uid in available
            )
            nomination_budget = min(
                self.MAX_PAIR_NOMINATION_ANCHORS,
                max(
                    self.MIN_PAIR_NOMINATION_ANCHORS,
                    4 + math.ceil(math.log2(1 + recurrent_available)),
                ),
            )
            structural_budget = nomination_budget // 2
            structural = sorted(
                available,
                key=lambda uid: (-len(full_posting[uid]), self._anchor_support(uid), uid),
            )[:structural_budget]
            admitted = set(structural)
            for uid in sorted(
                    available,
                    key=lambda uid: (-self._anchor_nomination_value(uid),
                                     -len(full_posting[uid]), uid)):
                if len(admitted) >= nomination_budget:
                    break
                admitted.add(uid)
            return frozenset(admitted)

        def admitted_geometry(wire):
            lattice = unpack_lattice(wire)
            admitted = admitted_for(lattice)
            rows = [row for row in lattice if int(row[2]) in admitted]
            rows.sort(key=lambda row: (row[0], row[1], row[2]))
            starts_by_uid = {}
            suffix_min_end_by_uid = {}
            for uid in admitted:
                unit_rows = [row for row in rows if int(row[2]) == uid]
                starts = tuple(int(row[0]) for row in unit_rows)
                suffix_min = [0] * len(unit_rows)
                best = 1 << 30
                for pos in range(len(unit_rows) - 1, -1, -1):
                    best = min(best, int(unit_rows[pos][1]))
                    suffix_min[pos] = best
                starts_by_uid[uid] = starts
                suffix_min_end_by_uid[uid] = tuple(suffix_min)
            return starts_by_uid, suffix_min_end_by_uid

        # This is transient construction geometry, not mature memory. Its size is
        # bounded by windows × the local 6–10-unit nomination aperture and it is
        # discarded after consolidation. It lets recurrence query the same interval
        # DAG that nominated the candidate instead of rescanning raw bytes.
        geometry_by_window = tuple(admitted_geometry(wire) for wire in lattice_wires)
        lattice_wires = ()

        # Exact recurrent ordered-pair backbone in fixed transient matter. Surface
        # units are already capped; a dense uint32 matrix is therefore bounded by
        # that developmental aperture rather than by lifetime novelty. Higher-rank
        # recurrence can only traverse transitions that themselves recur.
        active_uids = tuple(sorted(full_posting))
        uid_ordinal = {uid: pos for pos, uid in enumerate(active_uids)}
        pair_width = len(active_uids)
        pair_counts = array("I", [0]) * (pair_width * pair_width)
        for starts_by_uid, suffix_min_end_by_uid in geometry_by_window:
            admitted = tuple(starts_by_uid)
            for left in admitted:
                left_suffix = suffix_min_end_by_uid.get(left, ())
                if not left_suffix:
                    continue
                left_end = left_suffix[0]
                base = uid_ordinal[left] * pair_width
                for right in admitted:
                    starts = starts_by_uid.get(right, ())
                    pos = bisect_left(starts, left_end)
                    if pos < len(starts):
                        pair_counts[base + uid_ordinal[right]] += 1

        def pair_support(left, right):
            return pair_counts[uid_ordinal[int(left)] * pair_width + uid_ordinal[int(right)]]

        for left in active_uids:
            base = uid_ordinal[left] * pair_width
            for right in active_uids:
                if left == right:
                    continue
                support = pair_counts[base + uid_ordinal[right]]
                if support >= self.MIN_SUPPORT:
                    candidate_votes[(left, right)] = int(support)

        for starts_by_uid, suffix_min_end_by_uid in geometry_by_window:
            admitted = frozenset(starts_by_uid)
            pressure_q16 = (len(candidate_votes) << 16) // self.MAX_PAIR_CANDIDATES
            pressure_drop = (1 if pressure_q16 >= int(0.40 * (1 << 16)) else 0) \
                          + (1 if pressure_q16 >= int(0.65 * (1 << 16)) else 0) \
                          + (1 if pressure_q16 >= int(0.82 * (1 << 16)) else 0)
            active_budget = max(self.MIN_PAIR_NOMINATION_ANCHORS, len(admitted) - pressure_drop)
            active_admitted = frozenset(sorted(
                admitted,
                key=lambda uid: (-len(full_posting[uid]),
                                 -self._anchor_nomination_value(uid), uid),
            )[:active_budget])
            witnessed = set()

            # The local lattice is an interval DAG: edges go only to later,
            # non-overlapping Occurrences. For the same unit-prefix, an earlier
            # finishing witness dominates a later one because it admits every
            # continuation the later witness can admit. The cached per-unit interval
            # postings are also the exact recurrence oracle above.
            frontier = {}
            for uid in active_admitted:
                suffix_min = suffix_min_end_by_uid.get(uid, ())
                if suffix_min:
                    frontier[(uid,)] = suffix_min[0]
            ordered_uids = tuple(sorted(active_admitted))
            for _depth in range(1, self.MAX_ANCHORS):
                next_frontier = {}
                for anchors, end in frontier.items():
                    for uid in ordered_uids:
                        if pair_support(anchors[-1], uid) < self.MIN_SUPPORT:
                            continue
                        starts = starts_by_uid.get(uid, ())
                        pos = bisect_left(starts, end)
                        if pos >= len(starts):
                            continue
                        row_end = suffix_min_end_by_uid[uid][pos]
                        extended = anchors + (uid,)
                        if len(extended) > self.MIN_ANCHORS and len(set(extended)) >= self.MIN_ANCHORS:
                            witnessed.add(extended)
                        if len(extended) < self.MAX_ANCHORS:
                            previous = next_frontier.get(extended)
                            if previous is None or row_end < previous:
                                next_frontier[extended] = row_end
                        if len(witnessed) > self.MAX_PAIR_CANDIDATES:
                            raise ChunkRelationRefuse("chunk_recipe:candidate_bound")
                frontier = next_frontier
                if not frontier:
                    break
            for anchors in witnessed:
                candidate_votes[anchors] += 1
                if len(candidate_votes) > self.MAX_PAIR_CANDIDATES:
                    raise ChunkRelationRefuse("chunk_recipe:candidate_bound")
        self.pair_candidates_examined=len(candidate_votes)

        by_support = sorted(
            candidate_votes,
            key=lambda anchors: (-candidate_votes[anchors], anchors),
        )
        information_candidates = [
            anchors for anchors in candidate_votes
            if candidate_votes[anchors] >= self.MIN_SUPPORT
        ]
        by_information = sorted(
            information_candidates,
            key=lambda anchors: (
                -(sum(self._anchor_nomination_value(uid) for uid in anchors)
                  * math.log2(1 + candidate_votes[anchors])),
                -candidate_votes[anchors], anchors,
            ),
        )
        support_budget = (self.MAX_RECIPE_CANDIDATES + 1) // 2
        nominated = list(by_support[:support_budget])
        nominated_set = set(nominated)
        for anchors in by_information:
            if len(nominated) >= self.MAX_RECIPE_CANDIDATES:
                break
            if anchors in nominated_set:
                continue
            nominated.append(anchors); nominated_set.add(anchors)
        posting_sets = {uid: set(ids) for uid, ids in full_posting.items()}
        recipes = []
        for anchors in nominated:
            ordered_anchors = sorted(set(anchors), key=lambda uid: (len(full_posting[uid]), uid))
            candidate_ids = set(posting_sets[ordered_anchors[0]])
            for uid in ordered_anchors[1:]:
                candidate_ids.intersection_update(posting_sets[uid])
                if len(candidate_ids) < self.MIN_SUPPORT:
                    break
            if len(candidate_ids) < self.MIN_SUPPORT:
                continue

            span_lengths = [[] for _ in range(len(anchors) + 1)]
            span_values = [set() for _ in range(len(anchors) + 1)]
            adjacent_equality = [set() for _ in range(len(anchors))]
            lengths = []
            support = 0
            for index in sorted(candidate_ids, key=lambda item: roots[item]):
                raw = self._windows[index]
                aligned = self._align(raw, anchors)
                if aligned is None:
                    continue
                intervals = self._port_intervals(len(raw), aligned)
                if len(intervals) != len(span_lengths):
                    continue
                spans = tuple(raw[begin:end] for begin, end in intervals)
                support += 1
                lengths.append(len(raw))
                for pos, span in enumerate(spans):
                    span_lengths[pos].append(len(span))
                    span_values[pos].add(span)
                for pos in range(len(spans) - 1):
                    adjacent_equality[pos].add(spans[pos] == spans[pos + 1])
            if support < self.MIN_SUPPORT:
                continue
            # Mature Recipes preserve structural boundary law, not the exact byte
            # extents happened to occur in their construction witnesses.  Exact
            # historical min/max ranges made the compact Recipe a disguised cache
            # and blocked productive held-out rebinding.  A port is therefore only
            # constrained as always-empty, optional, or non-empty; the current
            # ephemeral Network carries its actual extent.
            def structural_guard(values):
                if not any(values):
                    return PortGuardV1(0, 0, 1)
                may_empty = int(0 in values)
                return PortGuardV1(0 if may_empty else 1, 0xFFFF, may_empty)
            guards = list(structural_guard(values) for values in span_lengths)
            # Search-window exterior is context aperture, not construction body.
            # Keep it bindable for comprehension but always allow it to disappear
            # when this same Recipe is unfolded outward from shared cognition.
            guards[0] = PortGuardV1(0, 0xFFFF, 1)
            guards[-1] = PortGuardV1(0, 0xFFFF, 1)
            guards = tuple(guards)
            if not any(guard.max_bytes > 0 for guard in guards[1:-1]):
                continue
            pair_guards = []
            for pos, seen in enumerate(adjacent_equality):
                # Relations to exterior aperture are transport-context accidents.
                # Only relations among internal bound ports belong to the Recipe.
                if pos == 0 or pos + 1 == len(span_values) - 1:
                    continue
                if len(span_values[pos]) < 2 or len(span_values[pos + 1]) < 2:
                    continue
                if seen == {True}:
                    pair_guards.append(PortPairGuardV1(pos, pos + 1, 1))
                elif seen == {False}:
                    pair_guards.append(PortPairGuardV1(pos, pos + 1, 2))
            pair_guards = tuple(pair_guards)
            identity = self._identity(b"chunk-recipe-v3", [
                list(map(int, anchors)),
                [[g.min_bytes, g.max_bytes, g.may_be_empty] for g in guards],
                [[g.left_port, g.right_port, g.relation] for g in pair_guards],
            ])
            recipes.append(ChunkRecipeV1(
                identity, tuple(map(int, anchors)), support, guards, pair_guards,
                min(lengths), max(lengths),
            ))
        def rank_key(row):
            return (row.support * row.anchor_count,
                    row.support, row.anchor_count, -row.identity)

        def information_key(row):
            information = sum(self._anchor_nomination_value(uid) for uid in row.anchors)
            return (information * row.anchor_count,
                    information, row.support, -row.identity)

        recipes.sort(key=rank_key, reverse=True)
        # Do not let one structural rank or one frequency regime consume the
        # bounded resident pool. Within each rank, half the matter follows broad
        # recurrence and half follows recurrent information/extent. This keeps
        # glue and content-bearing relations co-resident without increasing N.
        selected = []
        selected_ids = set()
        pair_budget = 12
        higher_budget = max(1, (self.MAX_RECIPES - pair_budget) // 2)
        for width in range(self.MIN_ANCHORS, self.MAX_ANCHORS + 1):
            rank_budget = pair_budget if width == self.MIN_ANCHORS else higher_budget
            rows = [row for row in recipes if row.anchor_count == width]
            support_budget = (rank_budget + 1) // 2
            information_budget = rank_budget - support_budget
            for row in rows[:support_budget]:
                if row.identity not in selected_ids:
                    selected.append(row); selected_ids.add(row.identity)
            for row in sorted(rows, key=information_key, reverse=True):
                if information_budget == 0:
                    break
                if row.identity in selected_ids:
                    continue
                selected.append(row); selected_ids.add(row.identity)
                information_budget -= 1
        selected.extend(row for row in recipes if row.identity not in selected_ids)
        self.recipes = {row.identity: row for row in selected[:self.MAX_RECIPES]}
        self._rebuild_index()

    def _unfold(self, raw, limit) -> tuple[ChunkNetworkOccurrenceV1, ...]:
        data = bytes(int(x) & 0xFF for x in raw)
        lattice = self.lattice(data)
        units = {row.unit for row in lattice}
        # Rebuildable two-anchor incidence is evaluated as a bitset reduction.
        # Each compact Recipe contributes one bit to each of its two designated
        # recruitment anchors. A Recipe is dereferenced only after both anchors
        # are physically present in this current lattice.
        matched_once = 0
        matched_twice = 0
        for uid in units:
            mask = self._anchor_recipe_bits.get(int(uid), 0)
            if not mask:
                continue
            matched_twice |= matched_once & mask
            matched_once |= mask
        nominated = []
        bits = matched_twice
        while bits and len(nominated) < self.MAX_NOMINATED_CURRENT:
            low = bits & -bits
            pos = low.bit_length() - 1
            if 0 <= pos < len(self._recipe_bit_order):
                nominated.append(self._recipe_bit_order[pos])
            bits ^= low
        touches = 0

        def recipe_numerator(recipe):
            information=sum(1/self._anchor_support(uid) for uid in recipe.anchors)
            return recipe.anchor_count*math.log2(1+recipe.support)*(1+information)

        def static_key(recipe_id):
            recipe=self.recipes[recipe_id]
            return (recipe_numerator(recipe),
                    recipe.support,recipe.anchor_count,-recipe.identity)

        matches = []
        for recipe_id in sorted(nominated,key=static_key,reverse=True):
            recipe = self.recipes[recipe_id]
            touches += 1
            aligned = self._align(data, recipe.anchors)
            if aligned is None:
                continue
            ports = self._port_intervals(len(data), aligned)
            if len(ports) != len(recipe.port_guards):
                continue
            structural_ok = True
            for (begin, end), guard in zip(ports, recipe.port_guards):
                extent = end - begin
                if not guard.min_bytes <= extent <= guard.max_bytes or (
                    extent == 0 and not guard.may_be_empty
                ):
                    structural_ok = False
                    break
            if not structural_ok:
                continue
            # Cross-port relational guards are part of current Network validity,
            # not a late motor/realization check.  An invalid current binding must
            # not win competition and only then fail during realization.
            for guard in recipe.pair_guards:
                l0,l1=ports[guard.left_port];r0,r1=ports[guard.right_port]
                equal=data[l0:l1]==data[r0:r1]
                if (guard.relation==1 and not equal) or (guard.relation==2 and equal):
                    structural_ok=False;break
            if not structural_ok:
                continue
            score = recipe_numerator(recipe) / max(1, len(lattice))
            key = (score, recipe.support, recipe.anchor_count, -recipe.identity)
            matches.append((key, recipe, aligned, ports))
            if limit is not None and len(matches) >= limit:
                break
        self.last_candidate_touches = touches
        occurrences = []
        for _key, recipe, aligned, ports in matches:
            anchor_occurrences = tuple((start, end, uid) for start, end, uid in aligned)
            identity = self._identity(
                b"chunk-network-occurrence-v1",
                [recipe.identity, hashlib.sha256(data).hexdigest(), anchor_occurrences, ports],
            )
            occurrences.append(ChunkNetworkOccurrenceV1(
                identity, recipe.identity, anchor_occurrences, ports, touches, len(lattice),
            ))
        return tuple(occurrences)

    def unfold(self, raw) -> ChunkNetworkOccurrenceV1 | None:
        rows = self._unfold(raw, 1)
        return rows[0] if rows else None

    def unfold_all(self, raw) -> tuple[ChunkNetworkOccurrenceV1, ...]:
        """Return every currently valid coactive Recipe Network occurrence."""
        return self._unfold(raw, None)

    def score(self, raw) -> ChunkRelationScoreV1:
        data = bytes(int(x) & 0xFF for x in raw)
        occurrence = self.unfold(data)
        coverage = self.last_multi_byte_coverage
        if occurrence is None:
            return ChunkRelationScoreV1(
                0, 0, 0, 0.0, self.last_candidate_touches,
                self.last_lattice_occurrences, coverage,
            )
        recipe = self.recipes[occurrence.recipe_identity]
        information = sum(1 / self._anchor_support(uid) for uid in recipe.anchors)
        structure = (
            recipe.anchor_count / max(1, occurrence.lattice_occurrences)
            * math.log2(1 + recipe.support)
            * (1 + information)
        )
        return ChunkRelationScoreV1(
            recipe.identity, recipe.support, recipe.anchor_count, structure,
            occurrence.candidate_touches, occurrence.lattice_occurrences,
            coverage,
        )

    def current_bindings(self, raw, occurrence: ChunkNetworkOccurrenceV1) -> tuple[bytes, ...]:
        recipe = self.recipes.get(int(occurrence.recipe_identity))
        if recipe is None:
            raise ChunkRelationRefuse("chunk_recipe:stale_occurrence")
        data = bytes(int(x) & 0xFF for x in raw)
        aligned=self._align(data,recipe.anchors)
        ports=() if aligned is None else self._port_intervals(len(data),aligned)
        expected=self._identity(b"chunk-network-occurrence-v1",[
            recipe.identity,hashlib.sha256(data).hexdigest(),
            tuple((start,end,uid) for start,end,uid in aligned or ()),ports,
        ])
        if (occurrence.identity!=expected
                or occurrence.anchor_occurrences!=tuple(aligned or ())
                or occurrence.port_intervals!=ports):
            raise ChunkRelationRefuse("chunk_recipe:binding_occurrence")
        if len(occurrence.port_intervals) != len(recipe.port_guards):
            raise ChunkRelationRefuse("chunk_recipe:binding_arity")
        bindings = []
        for begin, end in occurrence.port_intervals:
            if not 0 <= begin <= end <= len(data):
                raise ChunkRelationRefuse("chunk_recipe:binding_interval")
            bindings.append(data[begin:end])
        return tuple(bindings)

    def current_core_bindings(
        self, raw, occurrence: ChunkNetworkOccurrenceV1,
    ) -> tuple[bytes, ...]:
        """Return current internal bindings with search-aperture context removed."""
        bindings = list(self.current_bindings(raw, occurrence))
        if len(bindings) < 3:
            raise ChunkRelationRefuse("chunk_recipe:core_arity")
        bindings[0] = b""
        bindings[-1] = b""
        return tuple(bindings)

    def network_morphology_features(
        self, raw, occurrence: ChunkNetworkOccurrenceV1,
    ) -> tuple[int, ...]:
        """Lower one actual current Network to its stable opaque Recipe morphology.

        Current bindings are validated but never copied into the persistent
        morphology.  Thus a later changed contact can recruit the same morphology
        only by independently unfolding a valid occurrence of the same Recipe.
        """
        self.current_bindings(raw, occurrence)
        recipe = self.recipes[int(occurrence.recipe_identity)]
        return (
            self.NETWORK_MORPHOLOGY_TAG,
            recipe.identity & 0xFFFFFFFF,
            (recipe.identity >> 32) & 0xFFFFFFFF,
            recipe.anchor_count,
        )

    def boundary_network_morphology_features(self, raw, cap: int = 8) -> tuple[int, ...]:
        """Lower current span edges to opaque recurrent lower-Network identities.

        The returned unit identities are learned raw-surface morphology, not byte
        classes or authored grammar. They are transient compatibility evidence for
        a parent port; no surface bytes are persisted in the descriptor.
        """
        data = bytes(int(x) & 0xFF for x in raw)
        cap = max(1, min(int(cap), self.MAX_PORT_ALTERNATIVES))
        if not data:
            return (0, 0)
        lattice = self.lattice(data)
        left = tuple(sorted({int(row.unit) for row in lattice if row.start == 0}))[:cap]
        right = tuple(sorted({int(row.unit) for row in lattice if row.end == len(data)}))[:cap]
        return (len(left), *left, len(right), *right)

    def current_network_closure_features(self, raw, minimum: int = 2) -> tuple[int, ...]:
        """Lower all coactive learned Recipes on one current surface to one closure.

        This is a transient higher-order Network descriptor.  It contains no bytes,
        segmentation, winner rank, or expected output: only the set of learned
        Recipe morphologies that independently validate on the same current surface.
        A single Recipe is not promoted to a closure because it adds no constraint
        beyond ordinary parent re-entry.
        """
        minimum = max(2, int(minimum))
        occurrences = self.unfold_all(raw)
        recipes = tuple(sorted({int(row.recipe_identity) for row in occurrences}))
        if len(recipes) < minimum:
            return ()
        body = []
        for recipe_identity in recipes:
            recipe = self.recipes.get(recipe_identity)
            if recipe is None:
                raise ChunkRelationRefuse("chunk_recipe:closure_recipe")
            body.extend((
                recipe.identity & 0xFFFFFFFF,
                (recipe.identity >> 32) & 0xFFFFFFFF,
                recipe.anchor_count,
            ))
        return (self.CLOSURE_MORPHOLOGY_TAG, len(recipes), *body)

    def recursive_network_morphology_features(
        self, raw, occurrence: ChunkNetworkOccurrenceV1, depth: int = 2,
    ) -> tuple[int, ...]:
        """Lower nested current computation to opaque Recipe geometry, never bytes.

        This is a transient morphology descriptor for generic higher-order Network
        competition.  It contains only learned Recipe identities/arity and the
        presence or absence of recursively unfolded child Networks.  Current port
        payloads are validated and discarded.
        """
        depth = int(depth)
        if not 1 <= depth <= 3:
            raise ChunkRelationRefuse("chunk_recipe:morphology_depth")

        def lower(data: bytes, current: ChunkNetworkOccurrenceV1, remaining: int):
            bindings = self.current_core_bindings(data, current)
            recipe = self.recipes[int(current.recipe_identity)]
            node = [
                recipe.identity & 0xFFFFFFFF,
                (recipe.identity >> 32) & 0xFFFFFFFF,
                recipe.anchor_count,
                max(0, len(bindings) - 2),
            ]
            if remaining <= 1:
                return tuple(node)
            for span in bindings[1:-1]:
                child = self.unfold(span) if span else None
                if child is None:
                    node.extend((0, 0))
                    continue
                child_features = lower(bytes(span), child, remaining - 1)
                node.extend((1, len(child_features), *child_features))
            return tuple(node)

        data = bytes(int(x) & 0xFF for x in raw)
        body = lower(data, occurrence, depth)
        return (self.RECURSIVE_MORPHOLOGY_TAG, depth, len(body), *body)

    def realize(self, recipe_identity: int, bindings) -> bytes:
        recipe = self.recipes.get(int(recipe_identity))
        if recipe is None:
            raise ChunkRelationRefuse("chunk_recipe:missing_recipe")
        bindings = tuple(bytes(span) for span in bindings)
        if len(bindings) != len(recipe.port_guards):
            raise ChunkRelationRefuse("chunk_recipe:binding_arity")
        for span, guard in zip(bindings, recipe.port_guards):
            if not guard.min_bytes <= len(span) <= guard.max_bytes:
                raise ChunkRelationRefuse("chunk_recipe:port_guard")
            if not span and not guard.may_be_empty:
                raise ChunkRelationRefuse("chunk_recipe:port_guard")
        for guard in recipe.pair_guards:
            equal = bindings[guard.left_port] == bindings[guard.right_port]
            if (guard.relation == 1 and not equal) or (guard.relation == 2 and equal):
                raise ChunkRelationRefuse("chunk_recipe:pair_guard")
        out = bytearray(bindings[0])
        for pos, anchor in enumerate(recipe.anchors):
            unit = self.surface.surface_units.get(int(anchor))
            if unit is None:
                raise ChunkRelationRefuse("chunk_recipe:missing_anchor")
            out.extend(bytes(unit.raw))
            out.extend(bindings[pos + 1])
        return bytes(out)

    def packed_state(self) -> bytes:
        if len(self.construction_root) != 32:
            raise ChunkRelationRefuse("chunk_recipe:construction_root")
        out = bytearray(b"CRP3")
        out.extend(self.construction_root)
        out.extend(struct.pack("<III", self.contact_count, self.contact_bytes, len(self.recipes)))
        for recipe in sorted(self.recipes.values(), key=lambda row: row.identity):
            out.extend(struct.pack(
                "<QIII4B", recipe.identity, recipe.support, recipe.min_bytes, recipe.max_bytes,
                len(recipe.anchors), len(recipe.port_guards), len(recipe.pair_guards), 0,
            ))
            for anchor in recipe.anchors:
                out.extend(struct.pack("<Q", anchor))
            for guard in recipe.port_guards:
                out.extend(struct.pack("<HHB", guard.min_bytes, guard.max_bytes, guard.may_be_empty))
            for guard in recipe.pair_guards:
                out.extend(struct.pack("<BBB", guard.left_port, guard.right_port, guard.relation))
        out.extend(hashlib.sha256(out).digest())
        return bytes(out)

    @classmethod
    def restore_packed(cls, surface: LanguageDevelopmentV1, blob: bytes) -> "ChunkRelationInductionV1":
        data = bytes(blob)
        header_size = 4 + 32 + 12
        if len(data) < header_size + 32 or data[:4] != b"CRP3":
            raise ChunkRelationRefuse("chunk_recipe:checkpoint_header")
        body, digest = data[:-32], data[-32:]
        if hashlib.sha256(body).digest() != digest:
            raise ChunkRelationRefuse("chunk_recipe:checkpoint_digest")
        offset = 4
        construction_root = body[offset:offset + 32]; offset += 32
        contact_count, contact_bytes, recipe_count = struct.unpack_from("<III", body, offset); offset += 12
        if contact_count < cls.MIN_SUPPORT or contact_bytes == 0 or recipe_count > cls.MAX_RECIPES:
            raise ChunkRelationRefuse("chunk_recipe:checkpoint_counts")
        out = cls(surface)
        out.contact_count = int(contact_count)
        out.contact_bytes = int(contact_bytes)
        out.construction_root = bytes(construction_root)
        recipes = {}
        try:
            for _ in range(recipe_count):
                identity, support, min_bytes, max_bytes, anchor_count, guard_count, pair_count, reserved = struct.unpack_from("<QIII4B", body, offset); offset += 24
                if reserved or not cls.MIN_ANCHORS <= anchor_count <= cls.MAX_ANCHORS or guard_count != anchor_count + 1 or pair_count > anchor_count:
                    raise ChunkRelationRefuse("chunk_recipe:checkpoint_shape")
                anchors = tuple(struct.unpack_from("<Q", body, offset + 8 * i)[0] for i in range(anchor_count)); offset += 8 * anchor_count
                guards = []
                for _guard in range(guard_count):
                    minimum, maximum, may_empty = struct.unpack_from("<HHB", body, offset); offset += 5
                    if minimum > maximum or may_empty not in (0, 1):
                        raise ChunkRelationRefuse("chunk_recipe:checkpoint_guard")
                    guards.append(PortGuardV1(int(minimum), int(maximum), int(may_empty)))
                pair_guards = []
                for _guard in range(pair_count):
                    left, right, relation = struct.unpack_from("<BBB", body, offset); offset += 3
                    if left >= guard_count or right >= guard_count or left == right or relation not in (1, 2):
                        raise ChunkRelationRefuse("chunk_recipe:checkpoint_pair_guard")
                    pair_guards.append(PortPairGuardV1(int(left), int(right), int(relation)))
                guards = tuple(guards); pair_guards = tuple(pair_guards)
                expected = cls._identity(b"chunk-recipe-v3", [
                    list(map(int, anchors)),
                    [[g.min_bytes, g.max_bytes, g.may_be_empty] for g in guards],
                    [[g.left_port, g.right_port, g.relation] for g in pair_guards],
                ])
                if identity != expected or identity in recipes or support < cls.MIN_SUPPORT or min_bytes <= 0 or max_bytes < min_bytes:
                    raise ChunkRelationRefuse("chunk_recipe:checkpoint_recipe")
                recipes[int(identity)] = ChunkRecipeV1(
                    int(identity), tuple(map(int, anchors)), int(support), guards, pair_guards,
                    int(min_bytes), int(max_bytes),
                )
        except struct.error as exc:
            raise ChunkRelationRefuse("chunk_recipe:checkpoint_truncated") from exc
        if offset != len(body):
            raise ChunkRelationRefuse("chunk_recipe:checkpoint_trailing")
        out.recipes = recipes
        out._build_active_surface()
        out._rebuild_index()
        if out.packed_state() != data:
            raise ChunkRelationRefuse("chunk_recipe:checkpoint_roundtrip")
        return out

    def persistent_state(self) -> dict:
        return {
            "schema": 3,
            "construction_root": self.construction_root.hex(),
            "contact_count": self.contact_count,
            "contact_bytes": self.contact_bytes,
            "recipes": [
                {
                    "identity": row.identity,
                    "anchors": list(row.anchors),
                    "support": row.support,
                    "port_guards": [[g.min_bytes, g.max_bytes, g.may_be_empty] for g in row.port_guards],
                    "pair_guards": [[g.left_port, g.right_port, g.relation] for g in row.pair_guards],
                    "min_bytes": row.min_bytes,
                    "max_bytes": row.max_bytes,
                }
                for row in sorted(self.recipes.values(), key=lambda item: item.identity)
            ],
        }

    def quantity(self) -> dict:
        return {
            "contacts": self.contact_count,
            "hot_training_bytes": len(self._stream),
            "contact_bytes_observed": self.contact_bytes,
            "recipes": len(self.recipes),
            "pair_candidates": self.pair_candidates_examined,
            "last_candidate_touches": self.last_candidate_touches,
            "surface_units": len(self.surface.surface_units),
            "persistent_bytes_packed": len(self.packed_state()),
        }
