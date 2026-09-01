#!/usr/bin/env python3
"""Consequence-qualified bridge between raw world features and learned lexical concepts."""
from __future__ import annotations
from reference_population_v1 import PopulationBankV1,PopulationRecruitmentEcologyV1,PopulationSpecV1

RAW_TAG=0xC0A001
LANG_TAG=0xC0A002
MIN_BRIDGE_EVIDENCE=2


class CrossmodalConceptGroundingV1:
    """Own only cross-modal morphology relations; modalities remain separately owned."""
    def __init__(self,site_count=8192):
        self.population=PopulationBankV1(PopulationSpecV1(int(site_count),2,3,8,8))
        self.ecology=PopulationRecruitmentEcologyV1()
        self._current_world=()
        self._current_language=None
        self.last_touches=0

    @staticmethod
    def _unique_surface_feature(adult,surface):
        units=tuple(surface if not isinstance(surface,(bytes,bytearray)) else bytes(surface))
        features=[]
        for feature,candidates in adult.language._lexeme_index.items():
            if units in candidates and adult.language.lexeme(int(feature))==units:
                features.append(int(feature))
        return features[0] if len(features)==1 else 0

    def observe_world(self,organism):
        if (getattr(organism,'world_state',None) is None
                or int(getattr(organism,'world_source',0))<=0
                or int(getattr(organism,'world_state_occurrence',0))<=0):
            self._current_world=();return ()
        self._current_world=tuple(
            self.population.recruit((RAW_TAG,int(raw))) for raw in tuple(organism.world_state))
        return self._current_world

    def observe_language_surface(self,adult,surface):
        feature=self._unique_surface_feature(adult,surface)
        self._current_language=(None if not feature else self.population.recruit((LANG_TAG,feature)))
        return 0 if self._current_language is None else feature

    def settle_current_pair(self,source:int,effect:int,independent:bool=True):
        if len(self._current_world)!=1 or self._current_language is None:
            raise RuntimeError('crossmodal_grounding:current_pair')
        members=(self._current_world[0],self._current_language)
        network=self.population.recruit(self.ecology.network_occurrence_features(members))
        return self.ecology.record_qualified_network(
            self.population,network,members,int(source),int(effect),bool(independent))

    def _candidate_features(self,adult,raw_feature:int):
        cue=self.population.recruit((RAW_TAG,int(raw_feature)))
        rows=self.ecology.unfold_candidate_rows(cue);self.last_touches=self.ecology.last_touches
        winners=[]
        for strength,relation_identity,siblings,_sources in rows:
            relation=self.ecology.relations.get(int(relation_identity))
            if relation is None or relation.evidence_count<MIN_BRIDGE_EVIDENCE or strength<=0:continue
            for sibling in siblings:
                signature=self.ecology.morphologies.get(int(sibling))
                if signature is None:continue
                for feature in adult.language._lexeme_index:
                    if tuple(sorted(set(self.population.signature((LANG_TAG,int(feature))))))==tuple(signature):
                        winners.append((int(strength),int(feature)))
        if not winners:return ()
        best=max(x[0] for x in winners)
        return tuple(sorted(set(feature for strength,feature in winners if strength==best)))

    def resolve_raw_feature(self,adult,raw_feature:int):
        winners=self._candidate_features(adult,int(raw_feature))
        return winners[0] if len(winners)==1 else 0

    def resolve_world_atoms(self,adult,organism,raw_feature:int):
        """Return the sparse resident closure for one current world constituent.

        A tracked entity is an individual coordinate, not an alias for whichever
        category happens to overlap its features.  Keep that coordinate available
        to relation construction while also exposing independently learned lexical
        concepts.  Ordinary raw features have no such numerical-identity claim.
        """
        raw=int(raw_feature);direct=self.resolve_raw_feature(adult,raw)
        if raw not in getattr(organism,'entity_features',{}):
            return (() if not direct else (int(direct),))
        concepts={raw}
        if direct:concepts.add(int(direct))
        for other in organism._overlapping_entities(raw):
            if int(other)==raw:continue
            resolved=self.resolve_raw_feature(adult,int(other))
            if resolved:concepts.add(int(resolved))
        return tuple(sorted(concepts))

    def resolve_world_atom(self,adult,organism,raw_feature:int):
        """Compatibility scalar for callers that require one unambiguous concept."""
        raw=int(raw_feature);resolved=self.resolve_world_atoms(adult,organism,raw)
        if raw in getattr(organism,'entity_features',{}):
            resolved=tuple(x for x in resolved if int(x)!=raw)
        return resolved[0] if len(resolved)==1 else 0

    def resolve_current_world(self,adult,organism):
        if (getattr(organism,'world_state',None) is None
                or int(getattr(organism,'world_source',0))<=0
                or int(getattr(organism,'world_state_occurrence',0))<=0):return ()
        out=[]
        for raw in tuple(organism.world_state):
            out.extend(self.resolve_world_atoms(adult,organism,int(raw)))
        return tuple(dict.fromkeys(map(int,out)))

    def lesion_raw_feature(self,raw_feature:int):
        morphology=self.ecology.morphology_identity(
            self.population.signature((RAW_TAG,int(raw_feature))))
        if morphology in self.ecology.morphologies:self.ecology.lesion_morphology(morphology)

    def checkpoint(self):
        return {'schema':1,'population':self.population.checkpoint(),'ecology':self.ecology.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('crossmodal_grounding:checkpoint')
        out=cls();out.population=PopulationBankV1.restore(data['population'])
        out.ecology=PopulationRecruitmentEcologyV1.restore(data['ecology'])
        out._current_world=();out._current_language=None;return out
