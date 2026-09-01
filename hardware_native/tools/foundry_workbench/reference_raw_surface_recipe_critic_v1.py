#!/usr/bin/env python3
"""Reference bridge: raw-corpus Recipe geometry ranks current resident surfaces.

The critic receives no semantic label, prompt, expected bytes, or host-selected
shortlist. It enumerates every currently realizable surface from the resident
language ecology for one current Scene, then evaluates those bytes only through
an independently learned ChunkRelationInductionV1 ecology. It is a reference
hypothesis for surface well-formedness pressure, not an LLM decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from reference_chunk_relation_induction_v1 import ChunkRelationInductionV1


class RawSurfaceCriticRefuse(ValueError):
    pass


@dataclass(frozen=True)
class RawSurfaceCandidateV1:
    surface: bytes
    template_identity: int
    lexical_identities: tuple[int, ...]
    developmental_support: int


@dataclass(frozen=True)
class RawSurfaceSelectionV1:
    candidate: RawSurfaceCandidateV1
    higher_closures: int
    rank_mass: int
    structure_score_q20: int
    candidate_touches: int
    alternatives: int


@dataclass(frozen=True)
class RawSurfaceProposalV1:
    scene_identity: int
    template_identity: int
    lexical_identities: tuple[int, ...]
    alternatives: int
    world_occurrence: int = 0


class RawSurfaceRecipeCriticV1:
    """Use only learned raw-surface structure to break resident surface ties."""

    MAX_CANDIDATES = 4096

    def __init__(self, raw_ecology: ChunkRelationInductionV1):
        self.raw = raw_ecology
        self.last_raw_touches = 0
        self.last_surface_candidates = 0

    def current_candidates(self, organism, scene) -> tuple[RawSurfaceCandidateV1, ...]:
        if scene is None:
            return ()
        active_atoms, used_context, _binding, _relation_occurrences = organism._surface_view(scene)
        surface_context, conditions = organism._surface_context(scene, used_context, active_atoms)
        out = []
        if any(conditions):
            templates = organism.language.template_candidates(int(surface_context), len(active_atoms))
            if not templates and int(surface_context) != int(used_context):
                templates = organism.language.template_candidates(int(used_context), len(active_atoms))
            if not templates or len(templates) > self.MAX_CANDIDATES:
                return ()
            surfaces=[];lexical_ids=[]
            for atom,cond in zip(active_atoms,conditions):
                if cond:
                    found=organism._conditioned_form(int(atom),cond)
                    if found is None:return ()
                    donor,units,required=found
                    lexical_ids.append(organism.language.form_identity(int(donor),required,units))
                else:
                    found=organism._realized_lexeme(int(atom))
                    if found is None:return ()
                    donor,units=found
                    lexical_ids.append(organism.language.lexeme_identity(int(donor),units))
                surfaces.append(bytes(units))
            for template in templates:
                tid=int(template.identity[:15],16)
                rendered=organism.language.render_template(template,tuple(surfaces))
                out.append(RawSurfaceCandidateV1(bytes(rendered),tid,tuple(lexical_ids),int(template.support)))
        else:
            templates = organism.language.template_candidates(int(used_context), len(active_atoms))
            if not templates:
                return ()
            lexical_rows = []
            for atom in active_atoms:
                rows = organism._lexeme_rows(int(atom))
                if not rows:
                    return ()
                lexical_rows.append(rows)
            combinations = len(templates)
            for rows in lexical_rows:
                combinations *= len(rows)
                if combinations > self.MAX_CANDIDATES:
                    raise RawSurfaceCriticRefuse("raw_surface_critic:candidate_bound")
            for template in templates:
                tid = int(template.identity[:15], 16)
                for lexical_choice in product(*lexical_rows):
                    surfaces = tuple(bytes(row[1]) for row in lexical_choice)
                    lexical_ids = tuple(
                        organism.language.lexeme_identity(
                            organism._lexeme_owner(atom, row[1]), row[1]
                        )
                        for atom, row in zip(active_atoms, lexical_choice)
                    )
                    rendered = organism.language.render_template(template, surfaces)
                    support = int(template.support) + sum(int(row[0]) for row in lexical_choice)
                    out.append(RawSurfaceCandidateV1(bytes(rendered), tid, lexical_ids, support))
        # Identical bytes reached through equivalent resident configurations are one
        # outward alternative; retain the strongest deterministic resident witness.
        unique = {}
        for row in out:
            key = row.surface
            prior = unique.get(key)
            if prior is None or (row.developmental_support, -row.template_identity, row.lexical_identities) > (
                prior.developmental_support, -prior.template_identity, prior.lexical_identities
            ):
                unique[key] = row
        rows = tuple(sorted(unique.values(), key=lambda x: (x.surface, x.template_identity, x.lexical_identities)))
        self.last_surface_candidates = len(rows)
        return rows

    def _raw_score(self, surface: bytes):
        occurrences = self.raw.unfold_all(surface)
        touches = int(self.raw.last_candidate_touches)
        rank_mass = 0
        for occurrence in occurrences:
            recipe = self.raw.recipes.get(int(occurrence.recipe_identity))
            if recipe is not None:
                rank_mass += int(recipe.anchor_count) * int(recipe.support)
        winner = self.raw.score(surface)
        touches += int(winner.candidate_touches)
        q20 = int(round(float(winner.structure_score) * (1 << 20)))
        return len(occurrences), rank_mass, q20, touches

    def select(self, organism, scene) -> RawSurfaceSelectionV1 | None:
        candidates = self.current_candidates(organism, scene)
        if len(candidates) < 2:
            return None
        scored = []
        touches = 0
        for candidate in candidates:
            higher, rank_mass, structure_q20, row_touches = self._raw_score(candidate.surface)
            touches += row_touches
            # Developmental support is deliberately not a tie-break. This bridge is
            # useful only where resident developmental geometry is unresolved and
            # independent raw-surface structure contributes new evidence.
            # Structure-score density is diagnostic only: its lattice denominator
            # can reward a thinner surface that recruits the same learned Recipe.
            # Surface matter may break a resident tie only when the higher raw
            # closure evidence itself differs; semantic/content pressure remains
            # owned by the resident organism and later consequence.
            scored.append(((higher, rank_mass), candidate, structure_q20, row_touches))
        self.last_raw_touches = touches
        peak = max(row[0] for row in scored)
        if peak == (0, 0):
            return None
        winners = [row for row in scored if row[0] == peak]
        if len(winners) != 1:
            return None
        key, candidate, structure_q20, _ = winners[0]
        return RawSurfaceSelectionV1(
            candidate, key[0], key[1], structure_q20, touches, len(candidates)
        )

    def propose(self, organism, scene) -> RawSurfaceProposalV1 | None:
        selection = self.select(organism, scene)
        if selection is None:
            return None
        world_occurrence=(int(organism.world_state_occurrence)
                          if int(getattr(scene,'population_occurrence',0))==int(organism.world_state_occurrence)
                          and int(organism.world_state_occurrence)>0 else 0)
        return RawSurfaceProposalV1(
            int(scene.identity),
            int(selection.candidate.template_identity),
            tuple(map(int, selection.candidate.lexical_identities)),
            int(selection.alternatives),
            world_occurrence,
        )
