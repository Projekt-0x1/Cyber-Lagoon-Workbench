#!/usr/bin/env python3
"""Strict population-backed continuing reference organism, post-legacy.

The organism is not an LLM and has no prompt/goal API. It receives authenticated
numeric world/surface/consequence contact, maintains distributed population state,
stores episodic signatures, learns a numeric surface ecology, may reinstate missing
scene constituents, and may emit an action endogenously when learned resident state
supports one unique realization.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import heapq
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))

from reference_population_v1 import (PopulationBankV1,PopulationOccurrenceV1,
    PopulationRecruitmentEcologyV1,PopulationSpecV1,ResidentEventRecruitmentV1,mix64)
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1
from reference_language_learning_v1 import LearnedSurfaceEcologyV1,MIN_SOURCE_SUPPORT
from reference_hierarchical_composition_v1 import HierarchicalRefuse,TransientConstructionPlanV1,rematerialize_transient_plan,rematerialize_transient_sequence_plan
from reference_cognition_v1 import MAX_PLAN_DEPTH,MAX_STATE_FEATURES,PlanV1,TransitionEcologyV1
from reference_cognitive_tick_v1 import cognitive_tick
from reference_packed_selection_revision_v1 import PackedSelectionRevisionV1
from reference_organism_surface_state_v1 import control_values as surface_control_values,surface_conditions,surface_context
from reference_organism_scene_competition_v1 import (realize_conditioned_selected,
    realize_explicit_selected,
    select_explicit_lexeme,select_explicit_template,select_pending_scene,
    selection_configuration,selection_configuration_evidence,
    selection_construction_evidence,selection_member_evidence,
    selection_network_identity)
from reference_organism_relation_hypothesis_v1 import (index_pair_episode_features,
    recruit_entity_geometry,recruit_relation_hypothesis,reindex_entity_pair_features)
from reference_organism_utterance_boundary_v1 import (
    UtteranceBoundaryBankV1,apply_learned_discourse_surface,current_expression_plan)

CONTACT_SCENE=1
CONTACT_SURFACE=2
CONTACT_CONSEQUENCE=3
CONTACT_WITHDRAW_SOURCE=4
CONTACT_WORLD_STATE=5
DISCOVERED_VISUAL_ENTITY_TAG=0xD15C0E11
CONTACT_BODY_TARGET=6
CONTACT_AFFORDANCES=7
CONTACT_MOTOR_CONSEQUENCE=8
CONTACT_COMM_CHANNEL=9
CONTACT_PARTNER_CONTEXT=10
CONTACT_SCENE_LINK=11
CONTACT_DISCOURSE_SURFACE=12
CONTACT_SOURCE_ASSERTION=13
CONTACT_SOURCE_RETRACT=14
CONTACT_SURFACE_STREAM=15
CONTACT_ENTITY_FEATURES=16
CONTACT_BODY_STATE=17
CONTACT_SOURCE_UTTERANCE=18
CONTACT_CHANNEL_SAMPLE=19
CONTACT_EPISODE_BOUNDARY=20
INQUIRY_CONTEXT=0x1A11
COND_REINSTATED=0x1C011A
PREF_TEMPLATE=1
PREF_LEXEME=2
PREF_FORM=3
PREF_SPAN=4
PREF_BINDING=5
PREF_SOMA=6
PREF_VIEW=7
SOMA_CONTEXT=0x50BA7E
WORLD_SOMA_CONTEXT=0x50B0
WORLD_VIEW_CONTEXT=0x50B1E
MAX_SCENE_ATOMS=16
MAX_SELECTION_NETWORK_CANDIDATES=4096
MAX_ACTIONS=256
MAX_EPISODES=4096
MAX_SOURCE_ASSERTIONS=8192
MAX_SOURCE_CALIBRATIONS=4096
MAX_SELECTION_CONFIGURATION_REVISIONS=4096
SOURCE_SELECTION_TAG=0x50A7CE
SOURCE_LEXICAL_ELIGIBILITY_TAG=0x1E71CA1
RELATION_LINK_TAG=0x1E1A710
BODY_STATE_TAG=0xB0D157A7
EVENT_STATE_TAG=0xE7E1757A
EVENT_SCENE_CONTEXT=0xE7E17ACE
LANGUAGE_ACTION_CUE_TAG=0x1A6AC710
ACTION_RECIPE_TAG=0xAC710ACE
PROSPECTIVE_STATE_TAG=0xA11051A7
PROSPECTIVE_EDGE_TAG=0xA110ED6E
PROSPECTIVE_NETWORK_TAG=0xA1100E7


def _digest(tag,obj):return hashlib.sha256(tag.encode()+b'\0'+json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass
class SceneStateV2:
    identity:int
    channel:int
    context:int
    atoms:tuple[int,...]
    source:int
    population_occurrence:int
    demonstrated:bool=False
    acted:bool=False
    completed_from_episode:int=0
    binding_identity:int=0
    relation_occurrences:tuple[int,...]=()

@dataclass(frozen=True)
class EpisodeV2:
    identity:int
    scene_identity:int
    context:int
    atoms:tuple[int,...]
    source:int
    signature:tuple[int,...]
    tick:int

@dataclass
class SceneLinkV2:
    left_scene:int
    right_scene:int
    relation:int
    source:int
    active:bool=True
    population_occurrence:int=0

@dataclass
class SelectionConfigurationRevisionV2:
    context:int
    configuration:tuple[tuple[int,int,int],...]
    source:int
    support:int=0
    counter:int=0

class SelectionRevisionView:
    """List-shaped read/lesion surface over the packed revision store."""
    def __init__(self,organism):
        self._organism=organism
    def __iter__(self):
        for context,configuration,source,support,counter in self._organism._selection_revisions.iter_revisions():
            yield SelectionConfigurationRevisionV2(context,configuration,source,support,counter)
    def __len__(self):
        return self._organism._selection_revisions.row_count
    def __bool__(self):
        return self._organism._selection_revisions.row_count>0
    def remove(self,row):
        self._organism._selection_revisions.drop(row.context,row.configuration,row.source)

@dataclass(frozen=True)
class SharedEpisodeRelationV2:
    partner:int
    episode_identity:int
    action_ticket:int
    closure_identity:int
    source_roots:tuple[int,...]=()

@dataclass
class ActionV2:
    ticket:int
    tick:int
    channel:int
    source:int
    payload:tuple[int,...]
    population_occurrence:int
    scene_identity:int
    template_identity:int
    contributors:tuple[int,...]
    settled:bool=False
    effect:int=0
    planned_payload:tuple[int,...]=()
    repair:bool=False
    selection_context:int=0
    lexical_identities:tuple[int,...]=()
    selection_occurrences:tuple[tuple[int,int,int,int],...]=()
    closure_identity:int=0
    selection_network_identity:int=0
    form_slots:tuple[int,...]=()
    span_identity:int=0
    binding_identity:int=0
    relation_occurrences:tuple[int,...]=()
    somatic_occurrences:tuple[int,...]=()
    body_occurrence:int=0
    body_signature:int=0
    body_source:int=0

@dataclass
class SourceAssertionV2:
    identity:int
    source:int
    context:int
    action_id:int
    state:tuple[int,...]
    repetitions:int=1
    active:bool=True
    language_binding:int=0
    binding_atoms:tuple[int,...]=()
    predicted_state:tuple[int,...]=()
    prospective_closure:int=0
    prospective_step:int=0
    lexical_eligibilities:tuple[int,...]=()
    lexical_occurrence:int=0
    lexical_tick:int=0

@dataclass(frozen=True)
class ResidentProspectiveHypothesisV1:
    action_id:int
    binding_identity:int
    binding_atoms:tuple[int,...]=()

@dataclass
class ResidentProspectiveClosureV1:
    identity:int
    source:int
    recipe_identity:int
    hypotheses:tuple[ResidentProspectiveHypothesisV1,...]
    cursor:int=0

@dataclass(frozen=True)
class ProspectiveExpressionOpportunityV1:
    """Byte-free current communication opportunity over endogenous cognition."""
    identity:int
    recipe_identity:int
    surface_context:int
    span_relation:int
    actions:tuple[int,...]
    start:tuple[int,...]
    goal:tuple[int,...]
    world_occurrence:int
    body_occurrence:int
    partner_source:int
    partner_channel:int

@dataclass
class SourceCalibrationV2:
    source:int
    context:int
    support:int=0
    counter:int=0
    revision:int=0
    active:bool=True

@dataclass
class MotorActionV2:
    ticket:int
    tick:int
    action_id:int
    source:int
    state_before:tuple[int,...]
    population_occurrence:int
    settled:bool=False
    effect:int=0
    state_after:tuple[int,...]=()
    source_assertion_ids:tuple[int,...]=()
    source_context:int=0
    source_occurrences:tuple[tuple[int,int],...]=()
    source_counterfactual_action:int=0
    event_ticket:int=0
    event_relation:int=0
    world_occurrences:tuple[int,...]=()
    somatic_occurrences:tuple[int,...]=()
    prospective_recipe:int=0
    prospective_snapshot:tuple[tuple[int,...],...]=()
    prospective_context_signature:int=0
    independent_consequence:bool=False
    lexical_occurrences:tuple[tuple[int,...],...]=()

class ReferenceOrganismV2:
    def __init__(self,population_spec:PopulationSpecV1|None=None):
        self.population=PopulationBankV1(population_spec or PopulationSpecV1(131072,fanout=2,sites_per_feature=4,eligibility_horizon=8))
        self.__cognition_authority=object()
        self.language=LearnedSurfaceEcologyV1();self.utterances=UtteranceBoundaryBankV1();self.cognition=TransitionEcologyV1(self.__cognition_authority)
        self.recruitment=PopulationRecruitmentEcologyV1();self.event_recruitment=ResidentEventRecruitmentV1(self.recruitment)
        self.tick_count=0;self.next_scene=1;self.next_episode=1;self.next_ticket=1
        self.current_scene:SceneStateV2|None=None;self.pending_scenes:list[SceneStateV2]=[];self.scene_links:list[SceneLinkV2]=[];self.episodes:list[EpisodeV2]=[];self.actions:list[ActionV2]=[];self.motor_actions:list[MotorActionV2]=[];self.withdrawn_sources:set[int]=set();self.last_retrieval={'status':0,'winner':0,'score':0,'alternatives':0}
        self.world_state:tuple[int,...]|None=None;self.world_source=0;self.world_state_occurrence=0;self.body_target:tuple[int,...]=();self.body_state:tuple[int,...]=();self.body_state_source=0;self.body_state_occurrence=0;self.affordances:set[int]=set();self.exploration_trials:dict[tuple[tuple[int,...],int],int]={};self.information_need:tuple[int,...]=();self.information_need_asked=False;self.communication_channel=0;self.prospective_expression_announced=0
        self.partner_present=False;self.partner_source=0;self.partner_channel=0;self.shared_episode_relations:list[SharedEpisodeRelationV2]=[];self.last_shared_episode_by_partner:dict[int,int]={};self.last_shared_closure_by_partner:dict[int,int]={}
        self.output_fault_offset=-1;self.output_fault_value=0;self.pending_repair=None
        self._selection_revisions=PackedSelectionRevisionV1();self._somatic_revisions=PackedSelectionRevisionV1();self._world_revisions=PackedSelectionRevisionV1();self.selection_configuration_revisions=SelectionRevisionView(self);self.last_selection_network_touches=0;self.last_selection_candidate_touches=0;self.last_somatic_touches=0;self._somatic_occurrence_ids={};self._world_occurrence_ids={}
        self._action_commitments:dict[int,str]={}
        self.source_assertions:list[SourceAssertionV2]=[];self.source_calibrations:list[SourceCalibrationV2]=[];self.last_source_touches=0
        self.entity_features:dict[int,tuple[int,...]]={};self.entity_feature_sources:dict[int,set[int]]={};self._active_feature_cache:dict[int,tuple[int,...]]={}
        self.entity_conditions:dict[int,tuple[int,...]]={}
        self._pending_event_ticket=0;self._pending_event_relation=0
        self._prospective_source_closures:dict[int,ResidentProspectiveClosureV1]={}
        self.last_prospective_recipe=0;self.last_prospective_touches=0;self.last_prospective_occurrences=()
        # Rebuildable touched-work indexes. Durable episodes/scenes/links remain
        # authoritative; these structures only nominate exact local candidates.
        self._episode_by_id={};self._episode_by_scene={};self._episode_incidence={};self._pair_episode_index={};self._pair_episode_by_atom={};self._pair_feature_episode_index={};self._pair_feature_keys_by_episode={};self._shared_episode_incidence={};self._shared_feature_incidence={};self._shared_site_incidence={};self._entity_feature_index={};self._entity_site_index={};self._scene_by_id={};self._pending_heap=[];self._pending_structure_index={};self._links_from={};self._links_to={};self._link_by_occurrence={};self._action_by_ticket={};self._source_index={};self._source_recipe_index={};self._selection_construction_index={}
        self.last_episode_lookup_touches=0;self.last_pending_lookup_touches=0;self.last_relation_hypothesis_touches=0;self.last_relation_feature_reindex_touches=0;self.last_shared_site_touches=0;self.last_entity_candidate_touches=0;self.last_selection_construction_touches=0;self.last_relation_binding_touches=0;self.last_language_recipe_touches=0;self.last_world_scene_touches=0;self.last_world_scene_feature_touches=0;self.last_world_scene_candidate_touches=0

    def _index_episode(self,ep:EpisodeV2):
        self._episode_by_id[int(ep.identity)]=ep;self._episode_by_scene[int(ep.scene_identity)]=ep
        if len(ep.atoms)==2:
            self._pair_episode_index.setdefault(tuple(ep.atoms),set()).add(int(ep.identity))
            for atom in set(int(x) for x in ep.atoms):
                self._pair_episode_by_atom.setdefault(atom,set()).add(int(ep.identity))
            index_pair_episode_features(self,ep)
        for atom in set(int(x) for x in ep.atoms if int(x)!=0):
            self._episode_incidence.setdefault((int(ep.context),atom),set()).add(int(ep.identity))

    def _atom_sites(self,atom:int):
        atom=int(atom);features=self._active_entity_features(atom)
        if not features:
            return () if atom in self.entity_features else self.population.feature_sites(atom)
        sites=[]
        for feature in features:sites.extend(self.population.feature_sites(int(feature)))
        return tuple(dict.fromkeys(sites))

    def _active_entity_features(self,entity:int):
        entity=int(entity)
        if entity not in self._active_feature_cache:
            sources=self.entity_feature_sources.get(entity,())
            self._active_feature_cache[entity]=(self.entity_features.get(entity,())
                if any(source not in self.withdrawn_sources for source in sources)
                else ())
        return self._active_feature_cache[entity]

    def _set_entity_features(self,entity:int,features,source:int):
        entity=int(entity);source=int(source);features=tuple(dict.fromkeys(
            int(x) for x in features if int(x)!=0))
        if entity<=0 or source<=0 or not features:raise ValueError('organism:entity_features')
        old=self._active_entity_features(entity)
        if self.entity_features.get(entity)==features:
            self.entity_feature_sources.setdefault(entity,set()).add(source)
            self._active_feature_cache.pop(entity,None)
            if not old:
                for feature in features:self._entity_feature_index.setdefault(feature,set()).add(entity)
                for site in self._atom_sites(entity):self._entity_site_index.setdefault(int(site),set()).add(entity)
                reindex_entity_pair_features(self,entity)
            return
        old_sites=self._atom_sites(entity)
        if old:
            for feature in old:
                bucket=self._entity_feature_index.get(int(feature))
                if bucket:
                    bucket.discard(entity)
                    if not bucket:self._entity_feature_index.pop(int(feature),None)
            for site in old_sites:
                bucket=self._entity_site_index.get(int(site))
                if bucket:
                    bucket.discard(entity)
                    if not bucket:self._entity_site_index.pop(int(site),None)
        self.entity_features[entity]=features
        self.entity_feature_sources[entity]={source}
        self._active_feature_cache.pop(entity,None)
        for feature in features:self._entity_feature_index.setdefault(int(feature),set()).add(entity)
        for site in self._atom_sites(entity):self._entity_site_index.setdefault(int(site),set()).add(entity)
        reindex_entity_pair_features(self,entity)

    def mint_visual_entity(self,features,source:int,sequence:int):
        source=int(source);sequence=int(sequence);features=tuple(dict.fromkeys(int(x) for x in features if int(x)>0))
        if source<=0 or sequence<=0 or not features:raise ValueError('organism:visual_entity_mint')
        identity=mix64(DISCOVERED_VISUAL_ENTITY_TAG^mix64(source)^mix64(sequence))&((1<<63)-1)
        if identity<=0:identity=1
        while identity in self.entity_features:
            identity=mix64(identity^DISCOVERED_VISUAL_ENTITY_TAG)&((1<<63)-1) or 1
        self._set_entity_features(identity,features,source);self.population.recruit(features)
        return int(identity)

    def update_visual_entity(self,entity:int,features,source:int):
        entity=int(entity)
        if entity not in self.entity_features:raise ValueError('organism:visual_entity_unknown')
        self._set_entity_features(entity,features,int(source));self.population.recruit(tuple(int(x) for x in features))
        return entity

    def _index_shared_episode_relation(self,row:SharedEpisodeRelationV2):
        if any(int(source) in self.withdrawn_sources
               for source in row.source_roots):return
        ep=self._episode_by_id.get(int(row.episode_identity))
        if ep is None:return
        partner=int(row.partner)
        self.last_shared_episode_by_partner[partner]=int(row.episode_identity)
        self.last_shared_closure_by_partner[partner]=int(row.closure_identity)
        for atom in set(int(x) for x in ep.atoms if int(x)!=0):
            self._shared_episode_incidence.setdefault((partner,atom),set()).add(int(row.episode_identity))
            for feature in self._active_entity_features(int(atom)):self._shared_feature_incidence.setdefault((partner,int(feature)),set()).add(int(row.episode_identity))
            for site in self._atom_sites(atom):self._shared_site_incidence.setdefault((partner,int(site)),set()).add(int(row.episode_identity))

    def _shared_reinstated(self,partner:int,atom:int):
        partner=int(partner);atom=int(atom)
        if self._shared_episode_incidence.get((partner,atom)):return True
        features=self._active_entity_features(atom)
        if features:
            self.last_shared_site_touches=len(features)
            hits=sum(1 for feature in features if self._shared_feature_incidence.get((partner,int(feature))))
            return hits>0 and hits*4>=len(features)*3
        if atom in self.entity_features:return False
        sites=self._atom_sites(atom);self.last_shared_site_touches=len(sites)
        if not sites:return False
        hits=sum(1 for site in sites if self._shared_site_incidence.get((partner,int(site))))
        return hits>0 and hits*4>=len(sites)*3

    def _command_form_aliases(self,source,surface):
        extra={};source=int(source)
        if source<=0:return extra
        conditions={(COND_REINSTATED,),*(surface_conditions(self,int(target),BODY_STATE_TAG) for target in self.body_target)}
        for required in conditions:
            for feature,_conditions,units,_sources in self.language.invert_form_candidates(surface,required,True):
                if feature in self.body_target or self._shared_reinstated(source,int(feature)):extra.setdefault(units,set()).add(int(feature))
        for target in self.body_target:
            for other in self._overlapping_entities(int(target)):
                for _support,units,sources in self.language.lexeme_candidates(int(other)):
                    if source in sources or len(sources)>=MIN_SOURCE_SUPPORT:extra.setdefault(units,set()).add(int(target))
        return extra

    def _overlapping_entities(self,atom:int):
        atom=int(atom);features=self._active_entity_features(atom)
        if features:
            feature_set=set(features);nominated={atom};touches=0
            for feature in features:
                rows=self._entity_feature_index.get(int(feature),());touches+=len(rows);nominated.update(rows)
            self.last_entity_candidate_touches=touches;out=[]
            for other in nominated:
                other_features=self._active_entity_features(int(other))
                if not other_features:continue
                hits=sum(1 for feature in other_features if feature in feature_set)
                if hits>0 and hits*4>=len(features)*3:out.append(int(other))
            out.sort();return tuple(out)
        if atom in self.entity_features:return ()
        sites=self._atom_sites(atom);site_set=set(sites);nominated={atom} if atom else set();touches=0
        for site in sites:
            rows=self._entity_site_index.get(int(site),());touches+=len(rows);nominated.update(rows)
        self.last_entity_candidate_touches=touches;out=[]
        for other in nominated:
            other_sites=self._atom_sites(int(other))
            if not other_sites:continue
            hits=sum(1 for site in other_sites if site in site_set)
            if hits>0 and hits*4>=len(sites)*3:out.append(int(other))
        out.sort();return tuple(out)

    def _unique_form_row(self,rows,cond):
        if COND_REINSTATED in cond:
            compact=[row for row in rows if COND_REINSTATED in row[2]]
            if compact:rows=compact
        if not rows:return None
        best=rows[0];ties=[row for row in rows if row[:2]==best[:2]];surfaces={row[3] for row in ties}
        if len(surfaces)!=1:return None
        units=next(iter(surfaces));return next(row for row in ties if row[3]==units)

    def _lexeme_entities(self,atom:int):
        return recruit_entity_geometry(self,atom) if self._active_entity_features(int(atom)) else self._overlapping_entities(int(atom))

    def _conditioned_form(self,atom:int,conditions):
        cond=tuple(int(x) for x in conditions if int(x)!=0)
        partner=int(self.partner_source) if self.partner_present else 0
        others=self._overlapping_entities(int(atom)) or (int(atom),)
        found=[]
        if partner:
            for other in others:
                ohit=[row for row in self.language.form_candidates(int(other),cond,True) if partner in row[4]]
                if not ohit:continue
                row=self._unique_form_row(ohit,cond)
                if row is None:return None
                found.append((int(other),row[3],row[2]))
            if found:
                surfaces={row[1] for row in found};required={row[2] for row in found}
                if len(surfaces)!=1 or len(required)!=1:return None
                return min(found,key=lambda row:row[0])
            if any(partner in row[2] for other in others for row in self.language.lexeme_observations(int(other))):return None
        for other in others:
            row=self._unique_form_row(self.language.form_candidates(int(other),cond,True),cond)
            if row is not None:found.append((int(other),row[3],row[2]))
        if not found:return None
        surfaces={row[1] for row in found};required={row[2] for row in found}
        if len(surfaces)!=1 or len(required)!=1:return None
        return min(found,key=lambda row:row[0])

    def _lexeme_rows(self,atom:int):
        rows=self.language.lexeme_candidates(int(atom))
        if rows:return rows
        by_units={};confirmed=set()
        for other in self._lexeme_entities(int(atom)):
            confirmed.update(row[1] for row in self.language.lexeme_candidates(int(other)))
            for _support,units,sources in self.language.lexeme_observations(int(other)):by_units.setdefault(units,set()).update(sources)
        out=[(len(sources),units,tuple(sorted(sources))) for units,sources in by_units.items() if len(sources)>=MIN_SOURCE_SUPPORT or units in confirmed]
        out.sort(key=lambda x:(-x[0],x[1]));return out

    def _lexeme_owner(self,atom,units):
        atom=int(atom);units=tuple(units)
        found=[int(other) for other in self._lexeme_entities(atom) if any(row[1]==units for row in self.language.lexeme_candidates(int(other)))]
        return min(found) if found else atom

    def _realized_lexeme(self,atom:int):
        partner=int(self.partner_source) if self.partner_present else 0
        if partner:
            hit=[row for row in self._lexeme_rows(int(atom)) if partner in row[2]]
            surfaces={row[1] for row in hit}
            if len(surfaces)==1:
                units=next(iter(surfaces));return self._lexeme_owner(atom,units),units
            if hit:return None
            if any(partner in row[2] for other in (self._lexeme_entities(int(atom)) or (int(atom),)) for row in self.language.lexeme_observations(int(other))):return None
        units=self.language.lexeme(int(atom))
        if units is not None:return self._lexeme_owner(atom,units),units
        found=[]
        for other in self._lexeme_entities(int(atom)):
            units=self.language.lexeme(int(other))
            if units is not None:found.append((int(other),units))
        if not found:return None
        surfaces={row[1] for row in found}
        if len(surfaces)!=1:return None
        return min(found,key=lambda row:row[0])

    def _index_scene(self,scene:SceneStateV2):
        self._scene_by_id[int(scene.identity)]=scene
        if self._scene_available(scene):
            heapq.heappush(self._pending_heap,int(scene.identity))
            if scene.binding_identity:self._pending_structure_index.setdefault((int(scene.channel),int(scene.context),tuple(scene.atoms)),set()).add(int(scene.identity))

    def _index_link(self,link:SceneLinkV2):
        self._links_from.setdefault(int(link.left_scene),[]).append(link);self._links_to.setdefault(int(link.right_scene),[]).append(link)
        if int(link.population_occurrence):self._link_by_occurrence[int(link.population_occurrence)]=link

    def _relation_link_occurrence(self,link:SceneLinkV2):
        left=self._scene_by_id.get(int(link.left_scene));right=self._scene_by_id.get(int(link.right_scene))
        if left is None or right is None or not int(link.population_occurrence):return None
        occ=next((row for row in self.population.occurrences if row.identity==int(link.population_occurrence)),None)
        expected=self.population.signature((RELATION_LINK_TAG,int(link.left_scene),int(left.population_occurrence),int(link.relation),int(link.right_scene),int(right.population_occurrence)))
        return occ if occ is not None and occ.sites==expected else None

    def _relation_binding(self,scene:SceneStateV2):
        incoming=[row for row in self._links_to.get(int(scene.identity),()) if row.active and row.source not in self.withdrawn_sources]
        outgoing=[row for row in self._links_from.get(int(scene.identity),()) if row.active and row.source not in self.withdrawn_sources]
        self.last_relation_binding_touches=len(incoming)+len(outgoing)
        candidates=[]
        for left_link in incoming:
            for right_link in outgoing:
                if left_link.relation!=right_link.relation or left_link.source!=right_link.source:continue
                left=self._scene_by_id.get(int(left_link.left_scene));right=self._scene_by_id.get(int(right_link.right_scene))
                if left is None or right is None or len(left.atoms)!=1 or len(right.atoms)!=1:continue
                ordered=(int(left.atoms[0]),int(right.atoms[0]))
                if ordered[0]==0 or ordered[0]==ordered[1] or sorted(ordered)!=sorted(map(int,scene.atoms)):continue
                left_occ=self._relation_link_occurrence(left_link);right_occ=self._relation_link_occurrence(right_link)
                if left_occ is None or right_occ is None:continue
                binding=int(_digest('relation-binding-v1',[int(scene.context),int(left_link.relation),2])[:15],16) or 1
                context=int(_digest('relation-surface-context-v1',[int(scene.context),binding])[:15],16) or 1
                candidates.append((ordered,context,binding,(left_occ.identity,right_occ.identity)))
        return candidates[0] if len(candidates)==1 else None

    def _surface_view(self,scene:SceneStateV2):
        role=self._relation_role_view(scene)
        if role is not None and self.language.role_context_supported(role[1],len(role[0])):
            return role[:4]
        bound=self._relation_binding(scene)
        if bound is not None:return bound
        context=int(scene.context);binding=int(scene.binding_identity)
        occurrences=tuple(scene.relation_occurrences)
        if binding and occurrences:
            context=int(_digest('relation-surface-context-v1',[context,binding])[:15],16) or 1
        return tuple(scene.atoms),context,binding,occurrences

    def _relation_role_view(self,scene:SceneStateV2):
        """Project opaque leaf-relation participation independent of linear placement."""
        def walk(row):
            if row is None:return None
            if len(row.atoms)==1:return ((None,int(row.atoms[0])),)
            if not row.relation_occurrences:return None
            link=self._link_by_occurrence.get(int(row.relation_occurrences[-1]))
            if link is None or not link.active or link.source in self.withdrawn_sources:return None
            left=self._scene_by_id.get(int(link.left_scene));right=self._scene_by_id.get(int(link.right_scene))
            if left is None or right is None:return None
            if len(left.atoms)==len(right.atoms)==1:
                return (((int(link.relation),0),int(left.atoms[0])),
                        ((int(link.relation),1),int(right.atoms[0])))
            lrows=walk(left);rrows=walk(right)
            if lrows is None or rrows is None:return None
            return (*lrows,*rrows)
        rows=walk(scene)
        if not rows or any(role is None for role,_atom in rows):return None
        roles=[tuple(map(int,role)) for role,_atom in rows]
        if len(set(roles))!=len(roles):return None
        ordered=sorted(zip(roles,(int(atom) for _role,atom in rows)))
        canonical_roles=tuple(role for role,_atom in ordered)
        canonical_atoms=tuple(atom for _role,atom in ordered)
        context=int(_digest('relation-role-surface-context-v1',
            [int(scene.context),[list(role) for role in canonical_roles]])[:15],16) or 1
        return canonical_atoms,context,int(scene.binding_identity),tuple(scene.relation_occurrences),canonical_roles

    def _resident_relation_hypothesis(self,scene:SceneStateV2):
        return recruit_relation_hypothesis(self,scene,_digest)

    def _synthesize_relation_scene(self,link:SceneLinkV2):
        """Materialize one current relation computation from actual endpoint scenes.

        Rank-1 binds two singleton entity scenes.  Rank n+1 is allowed only when
        both endpoints are themselves relation scenes; their binding identities
        preserve grouping while their current relation Occurrences become actual
        participants of the new computation.  No semantic role or ToM opcode is
        introduced.
        """
        left=self._scene_by_id.get(int(link.left_scene));right=self._scene_by_id.get(int(link.right_scene))
        if left is None or right is None or left.channel!=right.channel:return 0
        context=int(link.relation)
        singleton=len(left.atoms)==1 and len(right.atoms)==1
        relational=bool(left.binding_identity and right.binding_identity and left.relation_occurrences and right.relation_occurrences)
        if singleton:
            if left.atoms[0]==0 or right.atoms[0]==0:return 0
            atoms=(int(left.atoms[0]),int(right.atoms[0]));feats=[context]
            for atom in atoms:feats.extend(self._active_entity_features(atom) or (atom,))
            binding=int(_digest('relation-binding-v1',[context])[:15],16) or 1
            inherited=()
        elif relational:
            atoms=tuple(int(x) for x in (*left.atoms,*right.atoms) if int(x)!=0)
            if not atoms or len(atoms)>MAX_SCENE_ATOMS:return 0
            binding=int(_digest('relation-binding-v2',[context,int(left.binding_identity),int(right.binding_identity)])[:15],16) or 1
            inherited=tuple(dict.fromkeys((*left.relation_occurrences,*right.relation_occurrences)))
            feats=[context,int(left.binding_identity),int(right.binding_identity),int(left.population_occurrence),int(right.population_occurrence),*inherited]
        else:return 0
        relation_occurrences=tuple(dict.fromkeys((*inherited,int(link.population_occurrence)))) if int(link.population_occurrence) else inherited
        occ=self.population.recruit(tuple(feats))
        scene=SceneStateV2(self.next_scene,int(left.channel),context,atoms,int(link.source),occ.identity,False,False,0,binding,relation_occurrences);self.next_scene+=1
        self.current_scene=scene;self.pending_scenes.append(scene);self._index_scene(scene);self._store_episode(scene)
        left.demonstrated=True;right.demonstrated=True
        return scene.identity

    def _index_action(self,action):
        self._action_by_ticket[int(action.ticket)]=action

    def _retire_settled_language_occurrences(self,action:ActionV2):
        """Remove action-private computations after their only return settles."""
        retired={int(action.population_occurrence)}
        retired.update(int(row[3]) for row in action.selection_occurrences)
        self.population.occurrences=[
            row for row in self.population.occurrences
            if int(row.identity) not in retired
        ]

    def _retire_settled_motor_occurrence(self,action:MotorActionV2):
        """Keep continuation metadata, not its consumed credit computation."""
        identity=int(action.population_occurrence)
        self.population.occurrences=[
            row for row in self.population.occurrences
            if int(row.identity)!=identity
        ]

    def _ensure_action_capacity(self):
        """Page settled public-action work out of the hot runtime set.

        Durable conversational consequence already lives in Episodes, shared
        episode relations, selection revisions, population credit, and source
        calibration. Keeping every settled ActionV2 hot therefore turns an
        observer/audit object into a lifetime context window. Preserve all
        unsettled work plus a bounded recent audit tail; old settled actions are
        rebuildable history, not current cognition.
        """
        if len(self.actions)<MAX_ACTIONS:return
        repair_origin=(int(self.pending_repair[6])
                       if self.pending_repair is not None and len(self.pending_repair)>6
                       else 0)
        target=max(1,MAX_ACTIONS//2)
        retained=[]
        for action in reversed(self.actions):
            keep=(not action.settled or int(action.ticket)==repair_origin or
                  len(retained)<target)
            if keep:
                retained.append(action)
            else:
                self._action_by_ticket.pop(int(action.ticket),None)
                self._action_commitments.pop(int(action.ticket),None)
        self.actions=list(reversed(retained))
        if len(self.actions)>=MAX_ACTIONS and all(a.settled for a in self.actions):
            raise ValueError('organism:action_hot_bound')

    def _action_commitment(self,action:ActionV2):
        return _digest('resident-action-closure-v2',[action.ticket,action.tick,action.channel,action.source,list(action.payload),action.population_occurrence,action.scene_identity,action.template_identity,list(action.contributors),list(action.planned_payload),action.repair,action.selection_context,list(action.lexical_identities),[list(row) for row in action.selection_occurrences],action.closure_identity,action.selection_network_identity,action.binding_identity,list(action.relation_occurrences),list(action.somatic_occurrences),action.body_occurrence,action.body_signature,action.body_source])

    def _index_source_assertion(self,row:SourceAssertionV2):
        self._source_index.setdefault((int(row.context),tuple(row.state)),set()).add(int(row.identity))
        # A grounded predictive utterance is also a defeasible prospective
        # Recipe for the same body-target context. This derived index is rebuilt
        # from source receipts; it never becomes world evidence or credit.
        if row.language_binding and row.predicted_state and not row.prospective_closure:
            self._source_recipe_index.setdefault(int(row.context),set()).add(int(row.identity))

    def _index_selection_construction(self,context:int,configuration):
        configuration=tuple(tuple(int(y) for y in member) for member in configuration)
        if configuration and configuration[0][0]==PREF_TEMPLATE and all(member[0] in (PREF_TEMPLATE,PREF_BINDING,PREF_LEXEME) for member in configuration):
            binding=next((int(member[2]) for member in configuration if member[0]==PREF_BINDING),0)
            bucket=self._selection_construction_index.setdefault((int(context),int(configuration[0][2]),binding),[])
            if configuration not in bucket:bucket.append(configuration)

    def _index_selection_configuration_revision(self,row:SelectionConfigurationRevisionV2):
        self._index_selection_construction(int(row.context),row.configuration)

    def _rebuild_runtime_indices(self):
        self._episode_by_id.clear();self._episode_by_scene.clear();self._episode_incidence.clear();self._pair_episode_index.clear();self._pair_episode_by_atom.clear();self._pair_feature_episode_index.clear();self._pair_feature_keys_by_episode.clear();self._shared_episode_incidence.clear();self._shared_feature_incidence.clear();self._shared_site_incidence.clear();self._entity_feature_index.clear();self._entity_site_index.clear();self.last_shared_episode_by_partner.clear();self.last_shared_closure_by_partner.clear();self._scene_by_id.clear();self._pending_heap=[];self._pending_structure_index.clear();self._links_from.clear();self._links_to.clear();self._link_by_occurrence.clear();self._action_by_ticket.clear();self._source_index.clear();self._source_recipe_index.clear();self._selection_construction_index.clear()
        self._active_feature_cache.clear()
        for entity in self.entity_features:
            features=self._active_entity_features(entity)
            for feature in features:self._entity_feature_index.setdefault(int(feature),set()).add(int(entity))
            for site in self._atom_sites(int(entity)):self._entity_site_index.setdefault(int(site),set()).add(int(entity))
        for ep in self.episodes:self._index_episode(ep)
        for row in self.shared_episode_relations:self._index_shared_episode_relation(row)
        for scene in self.pending_scenes:self._index_scene(scene)
        for link in self.scene_links:self._index_link(link)
        for action in self.actions:self._index_action(action)
        for row in self.source_assertions:self._index_source_assertion(row)
        for row in self.selection_configuration_revisions:self._index_selection_configuration_revision(row)

    def _scene_signature(self,context,atoms):
        return self.population.signature((int(context),*(int(x) for x in atoms if int(x)!=0)))

    def _store_episode(self,scene:SceneStateV2):
        if len(self.episodes)>=MAX_EPISODES:raise ValueError('organism:episode_bound')
        if any(x==0 for x in scene.atoms):return 0
        ep=EpisodeV2(self.next_episode,scene.identity,scene.context,scene.atoms,scene.source,self._scene_signature(scene.context,scene.atoms),self.tick_count)
        self.next_episode+=1;self.episodes.append(ep);self._index_episode(ep);return ep.identity

    def _control_values(self,atoms):
        return surface_control_values(self,atoms,BODY_STATE_TAG)

    def _port_compatibility_ok(self,contexts,atoms,conditions=()):
        values=(tuple(next((int(x) for x in row if int(x)!=COND_REINSTATED),0) for row in conditions)
                if conditions else self._control_values(atoms))
        for context in dict.fromkeys(int(x) for x in contexts):
            if context in self.language._compat_index and not self.language.compatible(context,values):
                return False
        return True

    def _surface_context(self,scene:SceneStateV2,context=0,atoms=()):
        return surface_context(self,scene,COND_REINSTATED,BODY_STATE_TAG,mix64,context,atoms)

    def _selection_preference_context(self,scene:SceneStateV2):
        if not self.partner_present or self.partner_source<=0:return 0
        derived=mix64(int(scene.context)^mix64(int(self.partner_source)^0x5E1EC710)^mix64(int(self.partner_channel))) & ((1<<63)-1)
        return int(derived or 1)

    def _source_context_signature(self):
        if not self.body_target:return 0
        return int(_digest('source-context-v2',list(self.body_target))[:15],16) or 1

    def _language_action_cue_occurrence(self,lexeme_identity:int):
        lexeme_identity=int(lexeme_identity)
        if lexeme_identity<=0:raise ValueError('organism:language_action_cue')
        return self.population.recruit((LANGUAGE_ACTION_CUE_TAG,lexeme_identity))

    def _action_recipe_occurrence(self,action_id:int):
        action_id=int(action_id)
        if action_id<=0:raise ValueError('organism:action_recipe')
        return self.population.recruit((ACTION_RECIPE_TAG,action_id))

    def _action_recipe_morphology(self,action_id:int):
        return self.recruitment.morphology_identity(self.population.signature((ACTION_RECIPE_TAG,int(action_id))))

    def _ground_language_action_recruitment(self,lexeme_identity:int,action_id:int,source:int,effect:int=1,independent:bool=True):
        """Reference grounding through one actual joint Network occurrence.

        The language cue and motor Recipe have distinct identities. Persistent
        recruitment is earned only from their actual co-participation plus an
        independent consequence; callers cannot install a surface->motor table.
        """
        cue=self._language_action_cue_occurrence(lexeme_identity);motor=self._action_recipe_occurrence(action_id)
        network=self.population.recruit(self.recruitment.network_occurrence_features((cue,motor)))
        return self.recruitment.record_qualified_network(self.population,network,(cue,motor),int(source),int(effect),bool(independent))

    def _language_action_candidate(self,binding):
        nominated=set();touches=0
        lexeme_identities=tuple(int(x) for x in getattr(binding,'lexical_identities',()) if int(x)>0)
        if not lexeme_identities:
            rows=[]
            for atom in tuple(int(x) for x in binding.atoms):
                units=self.language.lexeme(atom)
                if units is not None:rows.append(self.language.lexeme_identity(atom,units))
            lexeme_identities=tuple(rows)
        action_morphologies={int(action):self._action_recipe_morphology(action) for action in self.affordances}
        for lexeme_identity in lexeme_identities:
            cue=self._language_action_cue_occurrence(lexeme_identity);rows=self.recruitment.unfold_candidate_rows(cue)
            touches+=self.recruitment.last_touches
            matched=[]
            for credit,_relation,siblings,_sources in rows:
                for action,morphology in action_morphologies.items():
                    if morphology in siblings:matched.append((int(credit),int(action)))
            if not matched:continue
            best=max(row[0] for row in matched);winners=sorted(set(action for credit,action in matched if credit==best))
            if len(winners)==1:nominated.add(winners[0])
        self.last_language_recipe_touches+=touches
        return next(iter(nominated)) if len(nominated)==1 else 0

    def _record_source_assertion(self,action_id:int,source:int,language_binding:int=0,binding_atoms=(),predicted_state=(),prospective_closure:int=0,prospective_step:int=0,lexical_eligibilities=()):
        if self.world_state is None or not self.body_target or not self.affordances:raise ValueError('organism:source_assertion_context')
        action_id=int(action_id);source=int(source);language_binding=int(language_binding);context=self._source_context_signature();state=tuple(self.world_state)
        prospective_closure=int(prospective_closure);prospective_step=int(prospective_step)
        if action_id<=0 or action_id not in self.affordances or source<=0 or source in self.withdrawn_sources:raise ValueError('organism:source_assertion_value')
        atoms=tuple(int(x) for x in binding_atoms);lexical=tuple(sorted(set(int(x) for x in lexical_eligibilities if int(x)>0)))
        predicted=tuple(sorted(set(int(x) for x in predicted_state if int(x)!=0)))
        identity=int(_digest('source-assertion-v5',[source,context,action_id,language_binding,list(predicted),prospective_closure,prospective_step,*state])[:15],16);rows=[r for r in self.source_assertions if r.identity==identity]
        if len(rows)>1:raise ValueError('organism:source_assertion_duplicate')
        if rows:row=rows[0];row.repetitions+=1;row.active=True
        else:
            if len(self.source_assertions)>=MAX_SOURCE_ASSERTIONS:raise ValueError('organism:source_assertion_bound')
            row=SourceAssertionV2(identity,source,context,action_id,state,1,True,language_binding,atoms,predicted,prospective_closure,prospective_step);self.source_assertions.append(row);self._index_source_assertion(row)
        occ=self.population.recruit((context,action_id,source,language_binding,len(predicted),*predicted,prospective_closure,prospective_step,len(lexical),*lexical,*state))
        row.lexical_eligibilities=lexical;row.lexical_occurrence=occ.identity;row.lexical_tick=self.tick_count;return identity

    def _collapse_language_bindings(self,bindings,predict_target:bool=True):
        qualified=[]
        for binding in tuple(bindings):
            action_id=self._language_action_candidate(binding)
            if not action_id:continue
            predicted=tuple(sorted(set(int(atom) for atom in binding.atoms if predict_target and int(atom) in self.body_target)))
            qualified.append((binding,int(action_id),predicted))
        if not qualified:return None
        if predict_target and any(row[2] for row in qualified):qualified=[row for row in qualified if row[2]]
        meanings={(action,predicted) for _binding,action,predicted in qualified}
        if len(meanings)!=1:return ()
        action_id,predicted=next(iter(meanings));common=set(int(x) for x in qualified[0][0].atoms)
        for binding,_action,_predicted in qualified[1:]:common.intersection_update(int(x) for x in binding.atoms)
        binding_atoms=tuple(int(x) for x in qualified[0][0].atoms if int(x) in common)
        descriptors=sorted((int(binding.context),binding.template_identity,list(map(int,binding.atoms)),list(map(int,binding.lexical_identities))) for binding,_action,_predicted in qualified)
        binding_id=int(_digest('source-language-binding-set-v2',descriptors)[:15],16) or 1
        return action_id,predicted,binding_atoms,binding_id

    def _build_prospective_source_closure(self,span,source:int):
        hypotheses=[]
        for child in span.children:
            collapsed=self._collapse_language_bindings(self.language.invert_surface(child,aliases=self._command_form_aliases(source,child)),False)
            if collapsed is None:return None
            if collapsed==():return ()
            action_id,_predicted,binding_atoms,binding_id=collapsed
            hypotheses.append(ResidentProspectiveHypothesisV1(int(action_id),int(binding_id),tuple(binding_atoms)))
        if len(hypotheses)<2:return None
        recipe_identity=int(span.template_identity[:15],16) or 1
        identity=int(_digest('resident-prospective-closure-v1',[int(source),recipe_identity,[[h.action_id,h.binding_identity,list(h.binding_atoms)] for h in hypotheses]])[:15],16) or 1
        return ResidentProspectiveClosureV1(identity,int(source),recipe_identity,tuple(hypotheses),0)

    def _activate_prospective_source_step(self,closure:ResidentProspectiveClosureV1):
        if closure.cursor<0 or closure.cursor>=len(closure.hypotheses):return 0
        h=closure.hypotheses[closure.cursor]
        return self._record_source_assertion(h.action_id,closure.source,h.binding_identity,h.binding_atoms,(),closure.identity,closure.cursor)

    def _source_calibration(self,source:int,context:int):
        rows=[r for r in self.source_calibrations if r.active and r.source==int(source) and r.context==int(context)]
        if len(rows)>1:raise ValueError('organism:source_calibration_duplicate')
        return 0 if not rows else rows[0].support-rows[0].counter

    def _revise_source_calibration(self,source:int,context:int,effect:int):
        rows=[r for r in self.source_calibrations if r.source==int(source) and r.context==int(context)]
        if len(rows)>1:raise ValueError('organism:source_calibration_duplicate')
        if rows:row=rows[0];row.active=True
        else:
            if len(self.source_calibrations)>=MAX_SOURCE_CALIBRATIONS:raise ValueError('organism:source_calibration_bound')
            row=SourceCalibrationV2(int(source),int(context));self.source_calibrations.append(row)
        if effect>0:row.support+=1
        elif effect<0:row.counter+=1
        else:return 0
        row.revision+=1;return row.revision

    def _source_nomination(self,materialize=False):
        if self.world_state is None or not self.body_target:return None,(),(),()
        context=self._source_context_signature()
        ids=set(self._source_index.get((context,tuple(self.world_state)),()))
        ids.update(self._source_recipe_index.get(context,()))
        self.last_source_touches=len(ids)
        rows=[r for r in self.source_assertions if r.identity in ids and r.active and r.source not in self.withdrawn_sources and r.action_id in self.affordances]
        if not rows:return None,(),(),()
        by_source={}
        for row in rows:by_source.setdefault(row.source,set()).add(row.action_id)
        usable=[row for row in rows if len(by_source[row.source])==1]
        if not usable:return None,(),(),tuple(sorted(set(r.action_id for r in rows)))
        by_action={}
        for row in usable:by_action.setdefault(row.action_id,[]).append(row)
        alternatives=tuple(sorted(by_action));scored=[]
        for action,claims in by_action.items():
            unique={r.source:r for r in claims};score=sum(self._source_calibration(src,context) for src in unique)
            scored.append(((score,len(unique)),int(action),tuple(sorted(r.identity for r in unique.values()))))
        scored.sort(key=lambda x:(-x[0][0],-x[0][1],x[1]));peak=scored[0][0];w=[r for r in scored if r[0]==peak]
        if len(w)!=1 or peak[0]<0:return None,(),(),alternatives
        action=w[0][1];assertions=w[0][2];occurrences=[]
        if materialize:
            lookup={r.identity:r for r in rows}
            for aid in assertions:
                row=lookup[aid];occ=self.population.recruit((SOURCE_SELECTION_TAG,context,action,row.source,row.identity,row.language_binding,len(row.predicted_state),*row.predicted_state,*self.world_state));occurrences.append((row.identity,occ.identity))
        return action,assertions,tuple(occurrences),alternatives

    def _source_world_occurrences(self,assertions):
        lookup={r.identity:r for r in self.source_assertions};atoms=[]
        for aid in assertions:
            row=lookup.get(int(aid))
            if row is None or not row.language_binding:continue
            atoms.extend(int(x) for x in row.binding_atoms)
        return tuple(int(row[3]) for row in self._world_state_occurrences(0,atoms)) if atoms else ()

    def _source_somatic_occurrences(self,assertions):
        lookup={r.identity:r for r in self.source_assertions};atoms=[]
        for aid in assertions:
            row=lookup.get(int(aid))
            if row is None or not row.language_binding:continue
            atoms.extend(int(x) for x in row.binding_atoms)
        return tuple(int(row[3]) for row in self._somatic_state_occurrences(0,atoms)) if atoms else ()

    def _exploration_candidate(self):
        if not self.affordances or self.world_state is None:return 0
        scored=[]
        for action in self.affordances:
            trials=self.exploration_trials.get((self.world_state,int(action)),0);signature=self.population.signature((int(action),*self.world_state));physical=(len(signature),sum(signature),sum((i+1)*site for i,site in enumerate(signature)));scored.append(((trials,*physical),int(action)))
        scored.sort(key=lambda x:x[0]);peak=scored[0][0];w=[r for r in scored if r[0]==peak]
        return w[0][1] if len(w)==1 else 0

    def _validate_consequence(self,payload,source:int):
        if len(payload)!=2:raise ValueError('organism:consequence_shape')
        ticket,_effect=payload;action=self._action_by_ticket.get(int(ticket))
        if action is None or action.settled:raise ValueError('organism:consequence_ticket')
        if int(source)!=action.source:raise ValueError('organism:consequence_source')
        if self._action_commitments.get(int(ticket))!=self._action_commitment(action):raise ValueError('organism:consequence_action_commitment')
        selections=[];configuration=()
        if action.selection_occurrences:
            scene=self._scene_by_id.get(int(action.scene_identity));configuration=tuple(row[:3] for row in action.selection_occurrences)
            if scene is None:raise ValueError('organism:consequence_selection_configuration')
            active_atoms,used_context,binding_identity,relation_occurrences=self._surface_view(scene)
            surface_context,conditions=self._surface_context(scene,used_context,active_atoms)
            if any(conditions) and int(surface_context)!=int(used_context) and action.form_slots:
                conditioned_templates=self.language.template_candidates(int(surface_context),len(active_atoms))
                if any(int(row.identity[:15],16)==int(action.template_identity) for row in conditioned_templates):
                    used_context=int(surface_context)
            if binding_identity!=int(action.binding_identity) or tuple(relation_occurrences)!=tuple(action.relation_occurrences):raise ValueError('organism:consequence_relation_binding')
            if any(next((row for row in self.population.occurrences if row.identity==int(oid)),None) is None or int(oid) not in action.contributors for oid in relation_occurrences):raise ValueError('organism:consequence_relation_occurrence')
            if any(next((row for row in self.population.occurrences if row.identity==int(oid)),None) is None or int(oid) not in action.contributors for oid in action.somatic_occurrences):raise ValueError('organism:consequence_somatic_occurrence')
            if configuration!=self._selection_configuration(action.template_identity,action.lexical_identities,action.form_slots,action.span_identity,action.binding_identity):raise ValueError('organism:consequence_selection_configuration')
            for member_kind,slot,candidate,oid in action.selection_occurrences:
                sel=next((x for x in self.population.occurrences if x.identity==oid),None)
                qualifier=used_context if member_kind==PREF_TEMPLATE else active_atoms[slot-1] if member_kind in (PREF_LEXEME,PREF_FORM) and 0<slot<=len(active_atoms) else 0
                features=(0x51EC7,member_kind,slot,candidate,action.selection_context,scene.population_occurrence,*relation_occurrences) if member_kind==PREF_BINDING else (0x51EC7,member_kind,slot,candidate,action.selection_context,int(qualifier))
                expected=self.population.signature(features)
                if sel is None or oid not in action.contributors or sel.sites!=expected:raise ValueError('organism:consequence_selection_occurrence')
                selections.append(sel)
            expected_network=self._selection_network_identity(action.selection_context,action.population_occurrence,action.closure_identity,action.selection_occurrences)
            if action.selection_network_identity!=expected_network:raise ValueError('organism:consequence_selection_network')
        occ=next((o for o in self.population.occurrences if o.identity==action.population_occurrence),None)
        if occ is None:raise ValueError('organism:consequence_occurrence')
        if action.body_occurrence:
            body=next((o for o in self.population.occurrences if o.identity==action.body_occurrence),None)
            signature=0 if body is None else self._body_occurrence_signature(body)
            if (body is None or action.body_occurrence not in action.contributors or
                    signature!=action.body_signature or action.body_source<=0):
                raise ValueError('organism:consequence_body_occurrence')
        return action,tuple(selections),configuration,occ

    def _validate_motor_consequence(self,payload,source:int):
        if len(payload)<4:raise ValueError('organism:motor_consequence_shape')
        ticket,_effect,count=payload[:3];next_state=tuple(sorted(set(x for x in payload[3:] if x!=0)))
        if count!=len(next_state) or not next_state or len(next_state)>MAX_STATE_FEATURES:raise ValueError('organism:motor_consequence_shape')
        action=next((row for row in self.motor_actions if row.ticket==ticket),None)
        if action is None or action.settled:raise ValueError('organism:motor_consequence_ticket')
        if int(source)!=action.source:raise ValueError('organism:motor_consequence_source')
        occurrence=next((row for row in self.population.occurrences
                         if row.identity==action.population_occurrence),None)
        if occurrence is None:raise ValueError('organism:motor_action_occurrence')
        if not action.event_ticket:
            try:
                features=self._motor_action_features(action.action_id,action.state_before,
                    action.prospective_recipe,action.prospective_snapshot,
                    action.prospective_context_signature,action.lexical_occurrences)
            except ValueError as exc:
                raise ValueError('organism:motor_action_commitment') from exc
            if occurrence.feature_count!=len(features) or occurrence.sites!=self.population.signature(features):
                raise ValueError('organism:motor_action_commitment')
        return action,occurrence,next_state

    def contact(self,kind:int,payload,source:int,authenticated=True,independent=True):
        if not authenticated:raise ValueError('organism:unauthenticated')
        kind=int(kind);source=int(source);payload=tuple(int(x) for x in payload)
        consequence=self._validate_consequence(payload,source) if kind==CONTACT_CONSEQUENCE else None
        motor_consequence=self._validate_motor_consequence(payload,source) if kind==CONTACT_MOTOR_CONSEQUENCE else None
        if kind==CONTACT_SOURCE_UTTERANCE and (self.world_state is None or not self.body_target or not self.affordances):raise ValueError('organism:source_utterance_context')
        self.tick_count+=1
        if kind==CONTACT_SCENE:
            if int(source) in self.withdrawn_sources:raise ValueError('organism:scene_withdrawn')
            if len(payload)<4:raise ValueError('organism:scene_shape')
            channel,context,arity=payload[:3];atoms=tuple(payload[3:])
            if arity!=len(atoms) or not 1<=arity<=MAX_SCENE_ATOMS or channel<=0:raise ValueError('organism:scene_shape')
            # A newer actual scene supersedes a prior world's eligibility to
            # originate outward description; durable world state/markers remain.
            self.world_state_occurrence=0
            feats=[int(context)]
            for atom in atoms:
                if int(atom)==0:continue
                feats.extend(self._active_entity_features(int(atom)) or (int(atom),))
            occ=self.population.recruit(tuple(feats))
            scene=SceneStateV2(self.next_scene,channel,context,atoms,source,occ.identity);self.next_scene+=1;self.current_scene=scene;self.pending_scenes.append(scene);self._index_scene(scene);self._store_episode(scene);return scene.identity
        if kind==CONTACT_ENTITY_FEATURES:
            if source<=0 or source in self.withdrawn_sources:raise ValueError('organism:entity_features_source')
            if len(payload)<2:raise ValueError('organism:entity_features_shape')
            entity,count=int(payload[0]),int(payload[1])
            if count<0 or len(payload)<2+count:raise ValueError('organism:entity_features_shape')
            features=tuple(payload[2:2+count]);tail=tuple(int(x) for x in payload[2+count:] if int(x)!=0)
            self._set_entity_features(int(entity),features,source);self.population.recruit(features)
            if tail:self.entity_conditions[int(entity)]=tail
            return int(entity)
        if kind==CONTACT_CHANNEL_SAMPLE:
            if len(payload)<4:raise ValueError('organism:channel_sample_shape')
            episode,channel,count=payload[:3];features=payload[3:]
            if count!=len(features):raise ValueError('organism:channel_sample_shape')
            occurrence=self.event_recruitment.contact(self.population,episode,channel,source,features)
            relations=self.event_recruitment.recall_relations(
                self.population,episode,source,
            )
            if len(relations)==1:
                relation=relations[0]
                self.world_state=(EVENT_STATE_TAG,relation);self.world_source=source
                self.world_state_occurrence=occurrence.identity
                self.information_need=();self.information_need_asked=False
            else:
                if self.world_state and self.world_state[0]==EVENT_STATE_TAG:
                    self.world_state=None;self.world_source=0;self.world_state_occurrence=0
                self.information_need=(6,*relations);self.information_need_asked=False
            return occurrence.identity
        if kind==CONTACT_EPISODE_BOUNDARY:
            if len(payload)!=1:raise ValueError('organism:event_boundary_shape')
            closure=self.event_recruitment.close(self.population,payload[0],source)
            if closure is None:return 0
            retained={row.identity:row for row in self.population.occurrences}
            members=tuple(retained[oid] for oid in closure.member_occurrences)
            morphologies=tuple(self.recruitment.morphology_identity(row.sites) for row in members)
            relation=self.recruitment.relation_identity(morphologies)
            self.world_state=(EVENT_STATE_TAG,relation);self.world_source=source
            self.world_state_occurrence=closure.network_occurrence
            self._pending_event_ticket=closure.ticket;self._pending_event_relation=relation
            self.information_need=();self.information_need_asked=False
            return closure.ticket
        if kind==CONTACT_SURFACE_STREAM:
            if self.current_scene is None or self.current_scene.source in self.withdrawn_sources or len(self.current_scene.atoms)!=1:raise ValueError('organism:surface_stream_context')
            if not independent:raise ValueError('organism:surface_stream_independence')
            scene=self.current_scene;learned=self.language.observe_stream_naming(scene.atoms[0],payload,source);scene.demonstrated=True
            while self._pending_heap:
                head=self._scene_by_id.get(int(self._pending_heap[0]))
                if self._scene_available(head):break
                heapq.heappop(self._pending_heap)
            return 0 if learned is None else self.language.lexeme_identity(scene.atoms[0],learned)
        if kind==CONTACT_SURFACE:
            if self.current_scene is None or self.current_scene.source in self.withdrawn_sources:raise ValueError('organism:surface_without_scene')
            if not independent:raise ValueError('organism:surface_independence')
            scene=self.current_scene;hypothesis=self._resident_relation_hypothesis(scene)
            if hypothesis is not None:scene=hypothesis
            surface_atoms,surface_context,_binding,_relation_occurrences=self._surface_view(scene)
            surface_context,conditions=self._surface_context(scene,surface_context,surface_atoms)
            if len(scene.atoms)==1:
                if conditions[0]:self.language.observe_form(scene.atoms[0],conditions[0],payload,source)
                else:self.language.observe_naming(scene.atoms[0],payload,source)
            else:
                if any(conditions):
                    if not self.language.observe_conditioned_contact(surface_context,surface_atoms,conditions,payload,source):self.language.observe_conditioned_construction(surface_context,surface_atoms,conditions,payload,source)
                else:
                    self.language.observe_construction(surface_context,surface_atoms,payload,source)
                    role=self._relation_role_view(scene)
                    if role is not None:
                        self.language.observe_role_construction(
                            role[1],role[0],payload,source,scene.binding_identity)
                controls=self._control_values(surface_atoms)
                directed=self.language.complete_dependencies(int(surface_context),controls) if any(controls) else None
                if any(controls) and directed is None and not self.language.last_remote_dependency:self.language.observe_compatibility(int(surface_context),controls,source)
            scene.demonstrated=True
            while self._pending_heap:
                head=self._scene_by_id.get(int(self._pending_heap[0]))
                if self._scene_available(head):break
                heapq.heappop(self._pending_heap)
            return scene.identity
        if kind==CONTACT_CONSEQUENCE:
            ticket,effect=payload;action,selections,configuration,occ=consequence
            action.settled=True;action.effect=int(effect)
            learned=self.population.settle(occ,int(effect),bool(independent));selection_credit=0;selection_revisions=0
            selection_qualified=True
            for sel in selections:
                part=self.population.settle(sel,int(effect),bool(independent));selection_credit+=part.get('credit',0);selection_revisions+=part.get('revisions',0)
                selection_qualified=selection_qualified and part.get('credit',0)>0
            if action.selection_occurrences:
                learned=dict(learned);learned['selection_credit']=selection_credit;learned['selection_revisions']=selection_revisions
            if independent and int(effect)!=0 and learned.get('credit',0)>0 and selection_qualified and action.selection_context and configuration and (not action.planned_payload or tuple(action.payload)==tuple(action.planned_payload)):
                direction=1 if int(effect)>0 else -1
                network_updates=self._record_selection_configuration_revision(action.selection_context,configuration,action.source,direction)
                learned=dict(learned);learned['selection_network_updates']=network_updates
                scene=self._scene_by_id.get(int(action.scene_identity))
                if scene is not None and int(scene.context)!=INQUIRY_CONTEXT:
                    learned['somatic_updates']=self._record_somatic_markers(
                        scene.context,scene.atoms,action.source,direction,
                        action.body_signature,action.body_source)
            if independent and int(effect)>0 and (action.repair or not action.planned_payload or tuple(action.payload)==tuple(action.planned_payload)):
                ep=self._episode_by_scene.get(int(action.scene_identity))
                if ep is not None:
                    partner=int(action.source)
                    # Historical shared evidence remains in episodes/tickets, but
                    # only the latest partner-local utterance closure is needed to
                    # seed the next discourse computation. Retire older frozen
                    # surface leaves rather than retaining a transcript-shaped
                    # closure per successful turn.
                    for index,prior in enumerate(tuple(self.shared_episode_relations)):
                        if int(prior.partner)!=partner or not int(prior.closure_identity):continue
                        self.utterances.retire(int(prior.closure_identity))
                        self.shared_episode_relations[index]=SharedEpisodeRelationV2(
                            int(prior.partner),int(prior.episode_identity),
                            int(prior.action_ticket),0,tuple(prior.source_roots))
                    roots=tuple(dict.fromkeys((int(ep.source),int(action.source))))
                    row=SharedEpisodeRelationV2(partner,int(ep.identity),
                        int(action.ticket),int(action.closure_identity),roots)
                    if row not in self.shared_episode_relations:
                        if len(self.shared_episode_relations)>=MAX_EPISODES:raise ValueError('organism:shared_episode_relation_bound')
                        self.shared_episode_relations.append(row);self._index_shared_episode_relation(row)
            # The selection Network is an Occurrence-level computation, not durable
            # language state.  Once its consequence has settled, retain the exact
            # member/configuration witness and earned resident relation, but retire
            # the transient Network identity itself.  A future realization must
            # reconstruct a fresh Network from current Occurrences.
            action.selection_network_identity=0
            self._action_commitments[int(action.ticket)]=self._action_commitment(action)
            self._retire_settled_language_occurrences(action)
            return learned
        if kind==CONTACT_WORLD_STATE:
            state=tuple(sorted(set(x for x in payload if x!=0)))
            if not state:raise ValueError('organism:world_state')
            if int(source) in self.withdrawn_sources:return self.world_state
            world_occurrence=self.population.recruit(state)
            self.world_state=state;self.world_source=source;self.world_state_occurrence=world_occurrence.identity;self.information_need=();self.information_need_asked=False;self.prospective_expression_announced=0
            if independent:
                referents=tuple(int(x) for x in state if int(x) in self.entity_features)
                if referents:self._record_world_markers(WORLD_VIEW_CONTEXT,referents,source,1)
            return state
        if kind==CONTACT_BODY_STATE:
            state=tuple(int(x) for x in payload if int(x)!=0)
            if not state or source<=0:raise ValueError('organism:body_state')
            if int(source) in self.withdrawn_sources:return int(self.body_state_occurrence)
            occ=self.population.recruit((BODY_STATE_TAG,*state))
            self.body_state=state;self.body_state_source=source;self.body_state_occurrence=occ.identity;self.prospective_expression_announced=0
            return occ.identity
        if kind==CONTACT_BODY_TARGET:
            target=tuple(sorted(set(x for x in payload if x!=0)))
            if not target:raise ValueError('organism:body_target')
            if int(source) in self.withdrawn_sources:return self.body_target
            self.body_target=target;self.information_need=();self.information_need_asked=False;self.prospective_expression_announced=0;return target
        if kind==CONTACT_AFFORDANCES:
            rows=set(x for x in payload if x>0)
            if not rows:raise ValueError('organism:affordances')
            if int(source) in self.withdrawn_sources:return tuple(sorted(self.affordances))
            self.affordances=rows;self.prospective_expression_announced=0;return tuple(sorted(rows))
        if kind==CONTACT_COMM_CHANNEL:
            if len(payload)!=1 or payload[0]<=0:raise ValueError('organism:communication_channel')
            if int(source) in self.withdrawn_sources:return self.communication_channel
            self.communication_channel=payload[0];return self.communication_channel
        if kind==CONTACT_SCENE_LINK:
            if int(source) in self.withdrawn_sources:raise ValueError('organism:scene_link_withdrawn')
            if len(payload)!=3:raise ValueError('organism:scene_link_shape')
            left,right,relation=payload
            if left==right or int(left) not in self._scene_by_id or int(right) not in self._scene_by_id:raise ValueError('organism:scene_link')
            lscene=self._scene_by_id[int(left)];rscene=self._scene_by_id[int(right)]
            relational_singletons=len(lscene.atoms)==1 and len(rscene.atoms)==1
            higher_rank=bool(lscene.binding_identity and rscene.binding_identity and lscene.relation_occurrences and rscene.relation_occurrences)
            if (relational_singletons or higher_rank) and not independent:raise ValueError('organism:scene_relation_independence')
            link_occ=self.population.recruit((RELATION_LINK_TAG,int(left),int(lscene.population_occurrence),int(relation),int(right),int(rscene.population_occurrence)))
            link=SceneLinkV2(int(left),int(right),int(relation),source,True,link_occ.identity);self.scene_links.append(link);self._index_link(link)
            if relational_singletons or higher_rank:self._synthesize_relation_scene(link)
            return (left,right,relation)
        if kind==CONTACT_DISCOURSE_SURFACE:
            if self.current_scene is None or self.current_scene.source in self.withdrawn_sources:raise ValueError('organism:discourse_surface_without_scene')
            incoming=[l for l in self._links_to.get(int(self.current_scene.identity),()) if l.active and l.source==source and l.source not in self.withdrawn_sources]
            if len(incoming)!=1:raise ValueError('organism:discourse_surface_link')
            link=incoming[0];left=self._scene_by_id.get(int(link.left_scene));right=self.current_scene
            if left is None:raise ValueError('organism:discourse_surface_left')
            left_surface=self._realize_scene_surface(left);right_surface=self._realize_scene_surface(right)
            if left_surface is None or right_surface is None:raise ValueError('organism:discourse_surface_children')
            if not self.language.observe_span(link.relation,(left_surface,right_surface),payload,source):raise ValueError('organism:discourse_surface_factorization')
            # A raw discourse surface is a demonstration over both child scenes.
            # Demonstrated scenes are evidence, not pending unsaid experiences.
            left.demonstrated=True;right.demonstrated=True
            while self._pending_heap:
                head=self._scene_by_id.get(int(self._pending_heap[0]))
                if self._scene_available(head):break
                heapq.heappop(self._pending_heap)
            return right.identity
        if kind==CONTACT_PARTNER_CONTEXT:
            if len(payload)!=3:raise ValueError('organism:partner_context_shape')
            present,channel,partner=payload
            if present not in (0,1) or (present and (channel<=0 or partner<=0)):raise ValueError('organism:partner_context')
            self.partner_present=bool(present);self.partner_channel=int(channel) if present else 0;self.partner_source=int(partner) if present else 0;self.prospective_expression_announced=0
            return (int(self.partner_present),self.partner_channel,self.partner_source)
        if kind==CONTACT_SOURCE_ASSERTION:
            if len(payload)!=1:raise ValueError('organism:source_assertion_shape')
            return self._record_source_assertion(int(payload[0]),source,0)
        if kind==CONTACT_SOURCE_UTTERANCE:
            if int(source) in self.withdrawn_sources:raise ValueError('organism:source_withdrawn')
            self.last_language_recipe_touches=0
            spans=self.language.invert_span(payload)
            if spans:
                if len(spans)!=1:raise ValueError('organism:source_span_ambiguous')
                closure=self._build_prospective_source_closure(spans[0],source)
                if closure is None:raise ValueError('organism:source_span_unsupported')
                if closure==():raise ValueError('organism:source_span_ambiguous')
                self._prospective_source_closures[closure.identity]=closure
                assertion=self._activate_prospective_source_step(closure)
                if not assertion:self._prospective_source_closures.pop(closure.identity,None);raise ValueError('organism:source_span_step')
                return assertion
            bindings=self.language.invert_surface(payload,aliases=self._command_form_aliases(source,payload))
            if not bindings:raise ValueError('organism:source_utterance_unsupported')
            collapsed=self._collapse_language_bindings(bindings,True)
            if collapsed is None:raise ValueError('organism:source_utterance_ambiguous')
            if collapsed==():
                alternatives=tuple(sorted(set(self._language_action_candidate(binding) for binding in bindings if self._language_action_candidate(binding))))
                self.information_need=(3,*alternatives) if alternatives else (5,);self.information_need_asked=False;return 0
            action_id,predicted,binding_atoms,binding_id=collapsed
            lexical=set(bindings[0].lexical_identities)
            for binding in bindings[1:]:lexical.intersection_update(binding.lexical_identities)
            return self._record_source_assertion(action_id,source,binding_id,binding_atoms,predicted,lexical_eligibilities=lexical)
        if kind==CONTACT_SOURCE_RETRACT:
            if len(payload)!=1 or self.world_state is None or not self.body_target:raise ValueError('organism:source_retract_context')
            action_id=int(payload[0]);context=self._source_context_signature();state=tuple(self.world_state);rows=[r for r in self.source_assertions if r.source==source and r.context==context and r.action_id==action_id and r.state==state and r.active]
            if len(rows)!=1:raise ValueError('organism:source_retract')
            rows[0].active=False;return rows[0].identity
        if kind==CONTACT_MOTOR_CONSEQUENCE:
            action,action_occurrence,next_state=motor_consequence
            _ticket,effect,_count=payload[:3]
            source_causal=bool(independent and int(effect)!=0 and action.source_occurrences and action.source_context and action.source_counterfactual_action!=action.action_id)
            amap={r.identity:r for r in self.source_assertions};prospective_rows=[]
            for aid in action.source_assertion_ids:
                row=amap.get(int(aid))
                if row is not None and row.prospective_closure:prospective_rows.append(row)
            source_rows=[]
            if source_causal:
                if tuple(aid for aid,_ in action.source_occurrences)!=action.source_assertion_ids:raise ValueError('organism:source_closure')
                for aid,oid in action.source_occurrences:
                    row=amap.get(aid);socc=next((o for o in self.population.occurrences if o.identity==oid),None)
                    if row is None or socc is None or row.context!=action.source_context:raise ValueError('organism:source_occurrence')
                    source_rows.append((row,socc))
            event_relation=0
            if action.event_ticket:
                event_relation,_=self.event_recruitment.reafference(
                    self.population,action.event_ticket,int(effect),source,bool(independent))
                if event_relation and event_relation!=action.event_relation:raise ValueError('organism:event_relation_identity')
            action.settled=True;action.effect=int(effect);action.state_after=next_state
            action.independent_consequence=bool(independent)
            expert_context=(int(action.prospective_context_signature)
                            if action.prospective_snapshot else self._prospective_expert_context_signature())
            self.cognition.observe(action.state_before,action.action_id,next_state,int(effect),source,bool(independent))
            expert_nomination=None
            if bool(independent) and int(effect)>0 and action.prospective_snapshot:
                identity,_shadow,start,goal,actions,_effects,_sources,states=(
                    self._prospective_snapshot_parts(action.prospective_snapshot))
                matches=[index for index in range(len(actions))
                         if states[index]==action.state_before and actions[index]==action.action_id
                         and states[index+1]==next_state]
                if len(matches)==1 and matches[0]==len(actions)-1 and next_state==goal:
                    expert_nomination=self.cognition.record_expert_completion(
                        start,goal,actions,states,source,expert_context)
            learned=self.population.settle(action_occurrence,int(effect),bool(independent));source_updates=source_credit=source_revisions=0
            if expert_nomination is not None:
                learned=dict(learned);learned['prospective_expert_nomination']=expert_nomination.identity
            if action.prospective_recipe:
                learned=dict(learned);learned['host_signed_prospective_settlement']=learned.get('credit',0)
            for row,socc in source_rows:
                part=self.population.settle(socc,int(effect),True);source_credit+=part.get('credit',0);source_revisions+=part.get('revisions',0)
                if part.get('credit',0)>0:
                    calibration_direction=int(effect)
                    if row.predicted_state:
                        calibration_direction=1 if set(row.predicted_state).issubset(set(next_state)) else -1
                    source_updates+=int(bool(self._revise_source_calibration(row.source,action.source_context,calibration_direction)))
            world_occurrence=self.population.recruit(next_state)
            self.world_state=next_state;self.world_source=source;self.world_state_occurrence=world_occurrence.identity
            body_occ=self.population.recruit((BODY_STATE_TAG,*next_state))
            self.body_state=next_state;self.body_state_source=source;self.body_state_occurrence=body_occ.identity
            body_signature=self._body_occurrence_signature(body_occ)
            self.information_need=();self.information_need_asked=False
            learned=dict(learned);learned['source_updates']=source_updates;learned['source_credit']=source_credit;learned['source_revisions']=source_revisions
            if source_causal and source_credit>0 and action.lexical_occurrences:
                settlements=[]
                for oid,lexical_source,*lexemes in action.lexical_occurrences:
                    locc=next((o for o in self.population.occurrences if o.identity==oid),None)
                    if locc is None:raise ValueError('organism:lexical_eligibility_occurrence')
                    part=self.population.settle(locc,int(effect),True)
                    if part.get('credit',0)>0:
                        for lexeme in lexemes:
                            handled,result=self.language.settle_lexeme_identity(lexeme,lexical_source,action.ticket,int(effect))
                            if handled:settlements.append(result)
                if len(settlements)==1:learned['lexeme_settlement']=settlements[0]
            for row in prospective_rows:
                closure=self._prospective_source_closures.get(int(row.prospective_closure))
                if closure is None or closure.cursor!=int(row.prospective_step):continue
                if source_causal and int(effect)>0:
                    closure.cursor+=1
                    if closure.cursor>=len(closure.hypotheses):self._prospective_source_closures.pop(closure.identity,None)
                    else:
                        next_assertion=self._activate_prospective_source_step(closure)
                        if next_assertion:learned['prospective_next_assertion']=next_assertion
                else:self._prospective_source_closures.pop(closure.identity,None)
            if independent and int(effect)!=0 and learned.get('credit',0)>0:
                referents=tuple(dict.fromkeys(int(x) for x in (*action.state_before,*next_state) if int(x) in self.entity_features))
                if referents:
                    learned['somatic_updates']=self._record_somatic_markers(
                        WORLD_SOMA_CONTEXT,referents,action.source,
                        1 if int(effect)>0 else -1,body_signature,source)
            if independent and int(effect)!=0:
                prior=set(int(x) for x in action.state_before)
                appeared=tuple(int(x) for x in next_state if int(x) in self.entity_features and int(x) not in prior)
                if appeared:learned['world_updates']=self._record_world_markers(WORLD_VIEW_CONTEXT,appeared,source,1)
            if event_relation and independent and int(effect)>0:
                scene_occ=self.population.recruit((EVENT_SCENE_CONTEXT,event_relation,action.action_id))
                scene=SceneStateV2(self.next_scene,1,EVENT_SCENE_CONTEXT,
                                   (event_relation,action.action_id),source,scene_occ.identity)
                self.next_scene+=1;self.current_scene=scene;self.pending_scenes.append(scene)
                self._index_scene(scene);self._store_episode(scene)
                learned['event_relation']=event_relation;learned['event_scene']=scene.identity
            self._retire_settled_motor_occurrence(action)
            return learned
        if kind==CONTACT_WITHDRAW_SOURCE:
            if len(payload)!=1:raise ValueError('organism:withdraw_shape')
            target=payload[0];self.withdrawn_sources.add(target);self.language.withdraw_source(target);self.cognition.withdraw_source(target);self.recruitment.withdraw_source(target);self.event_recruitment.withdraw_source(target)
            if self._pending_event_ticket not in self.event_recruitment.closures:self._pending_event_ticket=0;self._pending_event_relation=0
            if self.current_scene is not None and self.current_scene.source==target:self.current_scene=None
            for link in self.scene_links:
                if link.source==target:link.active=False
            for row in self.source_assertions:
                if row.source==target:row.active=False
            self._prospective_source_closures={identity:row for identity,row in self._prospective_source_closures.items() if row.source!=target}
            for action in self.actions:
                if action.source==target and not action.settled:action.settled=True;action.effect=0
            for action in self.motor_actions:
                if action.source==target and not action.settled:action.settled=True;action.effect=0
            if self.body_state_source==target:
                self.body_state=();self.body_state_source=0;self.body_state_occurrence=0
            if self.world_source==target:
                self.world_state=None;self.world_source=0;self.world_state_occurrence=0
            if self.partner_source==target:
                self.partner_present=False;self.partner_channel=0;self.partner_source=0
            if self.pending_repair is not None and int(self.pending_repair[2])==int(target):
                self.pending_repair=None
            self._rebuild_runtime_indices()
            return target
        raise ValueError('organism:contact_kind')

    def _complete_scene(self,scene:SceneStateV2|None):
        if scene is None or not any(x==0 for x in scene.atoms):return scene
        known=tuple(int(x) for x in scene.atoms if int(x)!=0)
        postings=[]
        for atom in set(known):
            ids=self._episode_incidence.get((int(scene.context),atom),())
            if ids:postings.append((len(ids),atom,ids))
        if not postings:
            self.last_episode_lookup_touches=0;self.last_retrieval={'status':0,'winner':0,'score':0,'alternatives':0};return scene
        postings.sort(key=lambda x:(x[0],x[1]));ids=postings[0][2];self.last_episode_lookup_touches=len(ids)
        candidates=[]
        for eid in ids:
            ep=self._episode_by_id.get(int(eid))
            if ep is None or ep.source in self.withdrawn_sources or ep.context!=scene.context:continue
            if len(ep.atoms)!=len(scene.atoms) or any(a and a!=b for a,b in zip(scene.atoms,ep.atoms)):continue
            candidates.append((ep.identity,ep.signature))
        cue=(scene.context,*known);self.last_retrieval=self.population.retrieve(cue,candidates)
        if self.last_retrieval['status']!=1:return scene
        ep=self._episode_by_id.get(int(self.last_retrieval['winner']))
        if ep is None:return scene
        scene.atoms=ep.atoms;scene.completed_from_episode=ep.identity;return scene

    def _scene_available(self,scene):
        return scene is not None and not scene.acted and not scene.demonstrated and scene.source not in self.withdrawn_sources

    def _select_pending_scene(self):
        return select_pending_scene(self,COND_REINSTATED)

    def _resident_world_scene(self):
        """Reinstate one consequence-qualified event from actual world contact.

        Shared episodes nominate an ordered construction, but the current world
        Occurrence and partner are required to unfold it.  Ties remain unresolved;
        this path creates neither an episode nor credit.
        """
        self.last_world_scene_touches=0;self.last_world_scene_feature_touches=0;self.last_world_scene_candidate_touches=0
        if (not self.partner_present or self.partner_source<=0
                or self.world_state is None or self.world_source<=0
                or not self.world_state_occurrence
                or self.world_source in self.withdrawn_sources):return None
        prior=[scene for scene in self.pending_scenes
               if scene.population_occurrence==self.world_state_occurrence]
        if prior:return next((scene for scene in prior if self._scene_available(scene)),None)
        present=set(map(int,self.world_state));grouped={}
        for relation in self.shared_episode_relations:
            if relation.partner!=self.partner_source:continue
            self.last_world_scene_touches+=1
            if any(int(source) in self.withdrawn_sources
                   for source in relation.source_roots):continue
            episode=self._episode_by_id.get(int(relation.episode_identity))
            if episode is None or episode.source in self.withdrawn_sources:continue
            key=(int(episode.context),tuple(map(int,episode.atoms)))
            grouped.setdefault(key,set()).add(int(relation.action_ticket))
        candidates=[];exact=[(key,tickets) for key,tickets in grouped.items()
                            if set(key[1])==present]
        for (context,atoms),tickets in exact:
            score=(len(tickets),)
            identity=int(_digest('resident-world-scene-v1',[context,*atoms])[:15],16) or 1
            candidates.append((score,identity,context,atoms))
        if not candidates:
            arity=len(present);position_support={};position_feature_support={};contexts=set()
            for (context,atoms),tickets in grouped.items():
                if len(atoms)!=arity:continue
                contexts.add(context)
                for position,atom in enumerate(atoms):
                    position_support.setdefault((context,position,atom),set()).update(tickets)
                    for feature in self._active_entity_features(int(atom)):
                        position_feature_support.setdefault(
                            (context,position,int(feature)),set(),
                        ).update(tickets)
            for context in sorted(contexts):
                if not self.language.template_candidates(context,arity):continue
                choices=[];combination_bound=1
                for position in range(arity):
                    rows=[]
                    for atom in sorted(present):
                        tickets=position_support.get((context,position,atom))
                        if tickets:
                            rows.append((atom,len(tickets),1));continue
                        features=self._active_entity_features(int(atom))
                        if not features:continue
                        self.last_world_scene_feature_touches+=len(features)
                        feature_tickets=[position_feature_support.get(
                            (context,position,int(feature)),set(),
                        ) for feature in features]
                        supported=[row for row in feature_tickets if row]
                        if not supported or len(supported)*4<len(features)*3:continue
                        inherited=set().union(*supported)
                        if inherited:rows.append((atom,len(inherited),0))
                    if not rows:choices=[];break
                    choices.append(rows);combination_bound*=len(rows)
                    if combination_bound>MAX_SELECTION_NETWORK_CANDIDATES:
                        self.information_need=(7,);self.information_need_asked=False
                        return None
                if not choices:continue
                def assign(position,used,atoms,supports,exact):
                    if position==arity:
                        self.last_world_scene_candidate_touches+=1
                        score=(min(supports),sum(supports),sum(exact))
                        identity=int(_digest('resident-world-scene-v1',[context,*atoms])[:15],16) or 1
                        candidates.append((score,identity,context,tuple(atoms)));return
                    for atom,support,is_exact in choices[position]:
                        if atom in used:continue
                        assign(position+1,used|{atom},(*atoms,atom),
                               (*supports,support),(*exact,is_exact))
                assign(0,set(),(),(),())
        if not candidates:return None
        peak=max(row[0] for row in candidates);winners=[row for row in candidates if row[0]==peak]
        if len(winners)!=1:
            self.information_need=(7,*sorted(row[1] for row in winners));self.information_need_asked=False
            return None
        _score,_identity,context,atoms=winners[0]
        scene=SceneStateV2(self.next_scene,self.partner_channel,context,atoms,
                           self.world_source,self.world_state_occurrence)
        self.next_scene+=1;self.current_scene=scene;self.pending_scenes.append(scene);self._index_scene(scene)
        return scene

    def _prospective_recipe_by_identity(self,identity:int):
        rows=[row for row in self.cognition._prospective_recipes.values()
              if row.identity==int(identity)]
        if len(rows)!=1:raise ValueError('organism:prospective_action_identity')
        return rows[0]

    @staticmethod
    def _prospective_snapshot(recipe):
        return ((int(recipe.identity),int(recipe.shadow_credit)),tuple(recipe.start),
                tuple(recipe.goal),tuple(recipe.actions),tuple(recipe.effects),
                tuple(recipe.sources),*(tuple(row) for row in recipe.states))

    @staticmethod
    def _prospective_snapshot_from_intention(intention):
        """Rematerialize action-local validation from a recalled intention."""
        if (intention.identity<=0 or intention.recipe_identity<=0
                or not intention.actions or len(intention.states)!=len(intention.actions)+1
                or len(intention.effects)!=len(intention.actions) or not intention.sources):
            raise ValueError('organism:prospective_intention_snapshot')
        return ((int(intention.recipe_identity),max(1,len(intention.sources))),
                tuple(intention.states[0]),tuple(intention.states[-1]),tuple(intention.actions),
                tuple(intention.effects),tuple(intention.sources),
                *(tuple(row) for row in intention.states))

    @staticmethod
    def _prospective_snapshot_parts(snapshot):
        snapshot=tuple(tuple(int(x) for x in row) for row in snapshot)
        if not 7<=len(snapshot)<=MAX_PLAN_DEPTH+7 or len(snapshot[0])!=2:
            raise ValueError('organism:prospective_action_snapshot')
        identity,shadow_credit=snapshot[0];start,goal,actions,effects,sources=snapshot[1:6]
        states=snapshot[6:]
        if (not 1<=len(actions)<=MAX_PLAN_DEPTH or len(states)!=len(actions)+1
                or len(effects)!=len(actions) or len(sources)>64
                or states[0]!=start or states[-1]!=goal
                or any(not row or len(row)>MAX_STATE_FEATURES for row in states)):
            raise ValueError('organism:prospective_action_snapshot')
        expected=int(_digest('prospective-recipe-v1',[list(start),list(goal),list(actions),
            [list(row) for row in states],list(effects),list(sources)])[:15],16) or 1
        if identity!=expected or shadow_credit<=0:
            raise ValueError('organism:prospective_action_snapshot')
        return identity,shadow_credit,start,goal,actions,effects,sources,states

    def _motor_action_features(self,action_id:int,state,prospective_recipe:int=0,
                               prospective_snapshot=(),prospective_context_signature:int=0,
                               lexical_occurrences=()):
        state=tuple(int(x) for x in state);prospective_recipe=int(prospective_recipe);prospective_context_signature=int(prospective_context_signature)
        if prospective_recipe<=0:
            if prospective_snapshot or prospective_context_signature:raise ValueError('organism:prospective_action_identity')
            lexical=tuple(tuple(int(y) for y in row) for row in lexical_occurrences)
            flat=tuple(y for row in lexical for y in (len(row),*row))
            return (int(action_id),*state,SOURCE_LEXICAL_ELIGIBILITY_TAG,len(lexical),*flat) if lexical else (int(action_id),*state)
        if lexical_occurrences or prospective_context_signature<=0:raise ValueError('organism:prospective_lexical_eligibility')
        identity,shadow_credit,start,_goal,actions,_effects,_sources,states=(
            self._prospective_snapshot_parts(prospective_snapshot))
        participates=any(states[index]==state and actions[index]==int(action_id)
                         for index in range(len(actions)))
        if identity!=prospective_recipe or not participates:
            raise ValueError('organism:prospective_action_identity')
        witnesses=[]
        for index,current in enumerate(states):
            witnesses.append((PROSPECTIVE_STATE_TAG,index,*current))
            if index<len(actions):
                witnesses.append((PROSPECTIVE_EDGE_TAG,index,actions[index],
                                  *current,*states[index+1]))
        witnesses.append((PROSPECTIVE_NETWORK_TAG,identity,len(actions),shadow_credit,prospective_context_signature))
        signatures=tuple(self.population.signature(row) for row in witnesses)
        flattened=tuple(value for row in signatures for value in (len(row),*row))
        return (int(action_id),*state,PROSPECTIVE_NETWORK_TAG,identity,
                prospective_context_signature,len(signatures),*flattened)

    def _issue_motor(self,action_id:int,source_assertions=(),source_context:int=0,source_occurrences=(),source_counterfactual_action=0,world_occurrences=(),somatic_occurrences=(),prospective_recipe:int=0,prospective_snapshot=(),prospective_context_signature:int=0):
        if self.world_state is None or self.world_source<=0:return None
        event_ticket=int(self._pending_event_ticket);event_relation=int(self._pending_event_relation)
        if event_ticket and int(prospective_recipe):raise ValueError('organism:prospective_event_unimplemented')
        prospective_snapshot=(tuple(prospective_snapshot) if prospective_snapshot else
            (() if not prospective_recipe else self._prospective_snapshot(
                self._prospective_recipe_by_identity(prospective_recipe))))
        prospective_context_signature=int(prospective_context_signature)
        if prospective_recipe:
            if prospective_context_signature<=0:prospective_context_signature=self._prospective_expert_context_signature()
            if prospective_context_signature<=0:raise ValueError('organism:prospective_context')
        elif prospective_context_signature:
            raise ValueError('organism:prospective_context')
        amap={row.identity:row for row in self.source_assertions};lexical_occurrences=tuple((row.lexical_occurrence,row.source,*row.lexical_eligibilities) for aid in source_assertions if (row:=amap.get(int(aid))) is not None and row.lexical_eligibilities and row.lexical_tick==self.tick_count-1)
        if event_ticket and lexical_occurrences:raise ValueError('organism:event_lexical_eligibility_unimplemented')
        occ=(self.event_recruitment.issue_action(self.population,event_ticket,int(action_id))
             if event_ticket else self.population.recruit(self._motor_action_features(
                 action_id,self.world_state,prospective_recipe,prospective_snapshot,
                 prospective_context_signature,lexical_occurrences)))
        key=(self.world_state,int(action_id));self.exploration_trials[key]=self.exploration_trials.get(key,0)+1
        a=MotorActionV2(ticket=self.next_ticket,tick=self.tick_count,action_id=int(action_id),
            source=self.world_source,state_before=self.world_state,population_occurrence=occ.identity,
            source_assertion_ids=tuple(source_assertions),source_context=int(source_context),
            source_occurrences=tuple(source_occurrences),source_counterfactual_action=int(source_counterfactual_action),
            event_ticket=event_ticket,event_relation=event_relation,
            world_occurrences=tuple(int(x) for x in world_occurrences),
            somatic_occurrences=tuple(int(x) for x in somatic_occurrences),
            prospective_recipe=int(prospective_recipe),
            prospective_snapshot=prospective_snapshot,
            prospective_context_signature=prospective_context_signature,
            lexical_occurrences=lexical_occurrences)
        self.next_ticket+=1;self.motor_actions.append(a);self._pending_event_ticket=0;self._pending_event_relation=0;return a

    def _emit_information_request(self,alternatives):
        alternatives=tuple(int(x) for x in alternatives)
        if self.communication_channel<=0 or self.world_source<=0 or int(self.world_source) in self.withdrawn_sources or not alternatives:return None
        preference_context=mix64(INQUIRY_CONTEXT^mix64(int(self.world_source)^0x1A11)^mix64(int(self.communication_channel))) & ((1<<63)-1)
        preference_context=int(preference_context or 1)
        if self.partner_present and self.partner_source>0:
            preference_context=mix64(int(INQUIRY_CONTEXT)^mix64(int(self.partner_source)^0x5E1EC710)^mix64(int(self.partner_channel))) & ((1<<63)-1)
            preference_context=int(preference_context or 1)
            surface,template,ids,_alts=self._realize_explicit_selected(INQUIRY_CONTEXT,alternatives,preference_context)
            if surface is None or template is None:return None
            ids=tuple(ids);tid=int(template.identity[:15],16)
        else:
            # Inquiry wording is ordinary resident expression selection too.  This
            # lets independently earned consequence on one clarification form
            # reopen the learned alternatives and select a different wording on a
            # later ambiguity instead of collapsing to permanent silence.
            surface,template,ids,_alts=self._realize_explicit_selected(
                INQUIRY_CONTEXT,alternatives,preference_context)
            if surface is None or template is None:return None
            ids=tuple(ids);tid=int(template.identity[:15],16)
        occ=self.population.recruit((INQUIRY_CONTEXT,*alternatives));sid=self.next_scene;self.next_scene+=1
        scene=SceneStateV2(sid,self.communication_channel,INQUIRY_CONTEXT,alternatives,self.world_source,occ.identity,True,True)
        self.pending_scenes.append(scene);self._scene_by_id[int(sid)]=scene
        selection_rows=[(PREF_TEMPLATE,0,tid,self.population.recruit((0x51EC7,PREF_TEMPLATE,0,tid,preference_context,INQUIRY_CONTEXT)).identity)]
        for slot,(atom,cid) in enumerate(zip(alternatives,ids),1):
            selection_rows.append((PREF_LEXEME,slot,cid,self.population.recruit((0x51EC7,PREF_LEXEME,slot,cid,preference_context,int(atom))).identity))
        selection_rows=tuple(selection_rows);network_identity=self._selection_network_identity(preference_context,occ.identity,0,selection_rows)
        world_states=self._world_state_occurrences(INQUIRY_CONTEXT,alternatives);world_oids=tuple(int(row[3]) for row in world_states)
        somatic_occurrences=tuple(int(row[3]) for row in self._somatic_state_occurrences(INQUIRY_CONTEXT,alternatives))
        body=None
        if self.body_state_source>0 and self.body_state_source not in self.withdrawn_sources:
            body=next((row for row in self.population.occurrences
                       if row.identity==self.body_state_occurrence),None)
        body_occurrence=0 if body is None else int(body.identity)
        body_signature=0 if body is None else self._body_occurrence_signature(body)
        body_source=0 if body is None else int(self.body_state_source)
        contributors=tuple(dict.fromkeys(x for x in (sid,occ.identity,tid,*(r[3] for r in selection_rows),network_identity,*world_oids,*somatic_occurrences,body_occurrence) if x))
        planned=tuple(surface);actual=list(planned)
        if 0<=self.output_fault_offset<len(actual):
            actual[self.output_fault_offset]=self.output_fault_value;self.output_fault_offset=-1
        actual=tuple(actual)
        a=ActionV2(self.next_ticket,self.tick_count,self.communication_channel,self.world_source,actual,occ.identity,sid,tid,contributors,False,0,planned,False,preference_context,ids,selection_rows,0,network_identity,somatic_occurrences=somatic_occurrences,body_occurrence=body_occurrence,body_signature=body_signature,body_source=body_source)
        self._ensure_action_capacity();self.next_ticket+=1;self.actions.append(a);self._index_action(a);self._action_commitments[a.ticket]=self._action_commitment(a);self.information_need_asked=True
        if actual!=planned:self.pending_repair=(sid,self.communication_channel,self.world_source,planned,tid,contributors,a.ticket)
        return a

    def inject_output_fault(self,offset:int,value:int):
        # Explicit experimental lesion/fault injection. It is not a cognitive API.
        self.output_fault_offset=int(offset);self.output_fault_value=int(value)&255

    def _emit_pending_repair(self):
        row=self.pending_repair
        if row is None:return None
        scene_identity,channel,source,planned,template_identity,contributors=row[:6]
        origin=self._action_by_ticket.get(int(row[6])) if len(row)>6 else None
        # Repair is local to the exact failed motor trajectory.  Other unsettled
        # public actions neither authorize nor globally inhibit it.
        if origin is None or not origin.settled:return None
        if int(origin.effect)>=0:self.pending_repair=None;return None
        occ=self.population.recruit((0x5EFA17,int(scene_identity),len(planned)))
        preference_context=int(getattr(origin,'selection_context',0) or 0);lexical_ids=tuple(getattr(origin,'lexical_identities',()));form_slots=tuple(getattr(origin,'form_slots',()));span_identity=int(getattr(origin,'span_identity',0));closure_identity=int(getattr(origin,'closure_identity',0));binding_identity=int(getattr(origin,'binding_identity',0));relation_occurrences=tuple(getattr(origin,'relation_occurrences',()))
        body=None
        if self.body_state_source>0 and self.body_state_source not in self.withdrawn_sources:
            body=next((row for row in self.population.occurrences
                       if row.identity==self.body_state_occurrence),None)
        body_occurrence=0 if body is None else int(body.identity)
        body_signature=0 if body is None else self._body_occurrence_signature(body)
        body_source=0 if body is None else int(self.body_state_source)
        selection_rows=();network_identity=0
        scene=self._scene_by_id.get(int(scene_identity));active_atoms=() if scene is None else tuple(scene.atoms);used_context=0 if scene is None else int(scene.context)
        if origin is not None and origin.selection_occurrences:
            rows=[]
            if scene is None:current_binding=0;current_relation_occurrences=()
            else:active_atoms,used_context,current_binding,current_relation_occurrences=self._surface_view(scene)
            if current_binding!=binding_identity or current_relation_occurrences!=relation_occurrences:return None
            for kind,slot,cid,_oid in origin.selection_occurrences:
                qualifier=used_context if kind==PREF_TEMPLATE else active_atoms[slot-1] if kind in (PREF_LEXEME,PREF_FORM) and 0<slot<=len(active_atoms) else 0
                features=(0x51EC7,kind,slot,cid,preference_context,scene.population_occurrence,*relation_occurrences) if kind==PREF_BINDING else (0x51EC7,kind,slot,cid,preference_context,int(qualifier))
                rows.append((kind,slot,cid,self.population.recruit(features).identity))
            selection_rows=tuple(rows);network_identity=self._selection_network_identity(preference_context,occ.identity,closure_identity,selection_rows)
        world_states=self._world_state_occurrences(used_context,active_atoms)
        somatic_states=self._somatic_state_occurrences(used_context,active_atoms)
        if span_identity:
            prior_ep=self._episode_by_id.get(int(self.last_shared_episode_by_partner.get(int(source),0)))
            if prior_ep is not None:
                world_states=world_states+self._world_state_occurrences(int(prior_ep.context),prior_ep.atoms)
                somatic_states=somatic_states+self._somatic_state_occurrences(int(prior_ep.context),prior_ep.atoms)
        world_oids=tuple(int(row[3]) for row in world_states)
        somatic_occurrences=tuple(int(row[3]) for row in somatic_states)
        stale=set(int(x) for x in getattr(origin,'somatic_occurrences',()) if origin is not None)
        stale.update(int(x) for x in self._world_occurrence_ids.values())
        origin_body=int(getattr(origin,'body_occurrence',0) or 0)
        if origin_body and origin_body!=body_occurrence:stale.add(origin_body)
        base=tuple(x for x in contributors if int(x) not in stale)
        merged=tuple(dict.fromkeys((*base,*relation_occurrences,occ.identity,*(r[3] for r in selection_rows),network_identity,*world_oids,*somatic_occurrences,body_occurrence)))
        action=ActionV2(self.next_ticket,self.tick_count,int(channel),int(source),tuple(planned),occ.identity,int(scene_identity),int(template_identity),merged,False,0,tuple(planned),True,preference_context,lexical_ids,selection_rows,closure_identity,network_identity,form_slots,span_identity,binding_identity,relation_occurrences,somatic_occurrences,body_occurrence,body_signature,body_source)
        self._ensure_action_capacity();self.next_ticket+=1;self.actions.append(action);self._index_action(action);self._action_commitments[action.ticket]=self._action_commitment(action);self.pending_repair=None;return action

    def _cognitive_tick(self):
        return cognitive_tick(self,self.__cognition_authority,PROSPECTIVE_STATE_TAG,
                              PROSPECTIVE_EDGE_TAG,PROSPECTIVE_NETWORK_TAG)

    def _reconcile_low_pressure(self,budget:int=1):
        """Experimental target-free recurrence schedule; no action or evidence."""
        return self.cognition._reconcile_local(
            self.__cognition_authority,self.tick_count,budget)

    def _selection_configuration(self,template_identity:int,lexical_identities,form_slots=(),span_identity=0,binding_identity=0):
        return selection_configuration(template_identity,lexical_identities,form_slots,span_identity,binding_identity,(PREF_TEMPLATE,PREF_BINDING,PREF_FORM,PREF_LEXEME,PREF_SPAN))

    def _conditioned_surface_ports(self,atoms,conditions):
        atoms=tuple(int(x) for x in atoms);conditions=tuple(tuple(int(y) for y in row) for row in conditions)
        if len(atoms)!=len(conditions):return None
        surfaces=[];ids=[];form_slots=[]
        for slot,(atom,cond) in enumerate(zip(atoms,conditions),1):
            if cond:
                found=self._conditioned_form(int(atom),cond)
                if found is None:return None
                donor,units,required=found;ids.append(self.language.form_identity(int(donor),required,units));form_slots.append(slot)
            else:
                found=self._realized_lexeme(int(atom))
                if found is None:return None
                donor,units=found;ids.append(self.language.lexeme_identity(int(donor),units))
            surfaces.append(units)
        return tuple(surfaces),tuple(ids),tuple(form_slots)

    def _realize_conditioned_selected(self,context:int,atoms,conditions,preference_context:int,binding_identity=0,surface_proposal=None):
        return realize_conditioned_selected(self,context,atoms,conditions,
            preference_context,binding_identity,surface_proposal,
            ((COND_REINSTATED,),),MAX_SELECTION_NETWORK_CANDIDATES,
            PREF_TEMPLATE,PREF_BINDING,
            PREF_FORM,PREF_LEXEME)

    def _selection_network_identity(self,context:int,action_occurrence:int,closure_identity:int,occurrences):
        return selection_network_identity(_digest,context,action_occurrence,closure_identity,occurrences)

    def _selection_configuration_evidence(self,context:int,configuration):
        return selection_configuration_evidence(self,context,configuration)

    def _selection_member_evidence(self,context:int,member):
        return selection_member_evidence(self,context,member)
    def _selection_construction_evidence(self,context:int,template_identity:int,binding_identity=0):
        return selection_construction_evidence(self,context,template_identity,binding_identity,PREF_BINDING,PREF_LEXEME)

    def _record_selection_configuration_revision(self,context:int,configuration,source:int,direction:int):
        configuration=tuple(tuple(int(y) for y in member) for member in configuration)
        if not context or not configuration or direction==0:return 0
        if not self._selection_revisions.contains(context,configuration,source):
            if self._selection_revisions.row_count>=MAX_SELECTION_CONFIGURATION_REVISIONS:raise ValueError('organism:selection_configuration_revision_bound')
            self._index_selection_construction(int(context),configuration)
        return self._selection_revisions.record(context,configuration,source,direction)

    def _body_occurrence_signature(self,occurrence:PopulationOccurrenceV1):
        return int(_digest('body-occurrence-v1',list(occurrence.sites))[:15],16) or 1

    def _prospective_expert_context_signature(self):
        body=next((row for row in self.population.occurrences
                   if row.identity==int(self.body_state_occurrence)),None)
        if body is None or self.body_state_source<=0:return 0
        return int(_digest('prospective-expert-context-v1',[
            self._body_occurrence_signature(body),int(self.body_state_source)
        ])[:15],16) or 1

    def _soma_origin(self,context:int,atoms):
        return int(_digest('soma-origin-v1',[int(context),*(int(x) for x in atoms if int(x)!=0)])[:15],16) or 1

    def _soma_configuration(self,entity:int,origin:int,body_signature:int,body_source:int):
        return ((PREF_SOMA,0,int(entity)),(PREF_SOMA,1,int(origin) or 1),
                (PREF_SOMA,2,int(body_signature)),(PREF_SOMA,3,int(body_source)))

    def _record_somatic_markers(self,context:int,atoms,source:int,direction:int,
                                body_signature:int,body_source:int):
        if body_signature<=0 or body_source<=0:return 0
        if int(body_source) in self.withdrawn_sources or int(source) in self.withdrawn_sources:return 0
        origin=self._soma_origin(int(context),atoms)
        updates=0
        for atom in atoms:
            if int(atom)<=0:continue
            configuration=self._soma_configuration(int(atom),origin,body_signature,body_source)
            if not self._somatic_revisions.contains(SOMA_CONTEXT,configuration,source):
                if self._somatic_revisions.row_count>=MAX_SELECTION_CONFIGURATION_REVISIONS:raise ValueError('organism:somatic_revision_bound')
            updates+=self._somatic_revisions.record(SOMA_CONTEXT,configuration,source,direction)
        return updates

    def _somatic_marked_entities(self):
        out=[]
        for context,configuration,source,support,counter in self._somatic_revisions.iter_revisions():
            if context!=SOMA_CONTEXT or source in self.withdrawn_sources or not configuration:continue
            if configuration[0][0]==PREF_SOMA:out.append(int(configuration[0][2]))
        return tuple(dict.fromkeys(out))

    def _somatic_state_occurrences(self,context:int,atoms):
        self.last_somatic_touches=0
        current=self._soma_origin(context,atoms)
        marked={}
        for ctx,configuration,source,support,counter in self._somatic_revisions.iter_revisions():
            self.last_somatic_touches+=1
            if ctx!=SOMA_CONTEXT or source in self.withdrawn_sources or len(configuration)<4:continue
            if configuration[0][0]!=PREF_SOMA:continue
            entity,origin=int(configuration[0][2]),int(configuration[1][2])
            body_signature,body_source=int(configuration[2][2]),int(configuration[3][2])
            if body_source in self.withdrawn_sources:continue
            row=marked.setdefault((entity,origin,body_signature,body_source),[0,0]);row[0]+=int(support)-int(counter);row[1]+=int(support)+int(counter)
        states=[];seen=set()
        for atom in atoms:
            if int(atom)<=0:continue
            for other in self._overlapping_entities(int(atom)) or (int(atom),):
                for (entity,origin,body_signature,body_source),(value,evidence) in marked.items():
                    key=(int(entity),int(origin),int(body_signature),int(body_source))
                    if entity!=int(other) or not evidence or value==0 or origin==current or key in seen:continue
                    seen.add(key);direction=1 if value>0 else -1
                    occ_key=(int(entity),int(origin),int(body_signature),int(body_source),direction,int(evidence))
                    oid=self._somatic_occurrence_ids.get(occ_key)
                    if oid is None:
                        occ=self.population.recruit((SOMA_CONTEXT,int(entity),int(origin),int(body_signature),int(body_source),direction,int(evidence)))
                        oid=int(occ.identity);self._somatic_occurrence_ids[occ_key]=oid
                    states.append((int(entity),direction,int(evidence),oid))
        return tuple(states)

    def _world_origin(self,context:int,atoms):
        return int(_digest('world-origin-v1',[int(context),*(int(x) for x in atoms if int(x)!=0)])[:15],16) or 1

    def _world_configuration(self,entity:int,origin:int):
        return ((PREF_VIEW,0,int(entity)),(PREF_VIEW,1,int(origin) or 1))

    def _record_world_markers(self,context:int,atoms,source:int,direction:int):
        if int(source) in self.withdrawn_sources:return 0
        origin=self._world_origin(int(context),atoms)
        updates=0
        for atom in atoms:
            if int(atom)<=0:continue
            configuration=self._world_configuration(int(atom),origin)
            if not self._world_revisions.contains(WORLD_VIEW_CONTEXT,configuration,source):
                if self._world_revisions.row_count>=MAX_SELECTION_CONFIGURATION_REVISIONS:raise ValueError('organism:world_revision_bound')
            updates+=self._world_revisions.record(WORLD_VIEW_CONTEXT,configuration,source,direction)
        return updates

    def _world_marked_entities(self):
        out=[]
        for context,configuration,source,support,counter in self._world_revisions.iter_revisions():
            if context!=WORLD_VIEW_CONTEXT or source in self.withdrawn_sources or not configuration:continue
            if configuration[0][0]==PREF_VIEW:out.append(int(configuration[0][2]))
        return tuple(dict.fromkeys(out))

    def _world_state_occurrences(self,context:int,atoms):
        current=self._world_origin(context,atoms)
        marked={}
        for ctx,configuration,source,support,counter in self._world_revisions.iter_revisions():
            if ctx!=WORLD_VIEW_CONTEXT or source in self.withdrawn_sources or len(configuration)<2:continue
            if configuration[0][0]!=PREF_VIEW:continue
            entity,origin=int(configuration[0][2]),int(configuration[1][2])
            row=marked.setdefault((entity,origin),[0,0]);row[0]+=int(support)-int(counter);row[1]+=int(support)+int(counter)
        states=[];seen=set()
        for atom in atoms:
            if int(atom)<=0:continue
            for other in self._overlapping_entities(int(atom)) or (int(atom),):
                for (entity,origin),(value,evidence) in marked.items():
                    key=(int(entity),int(origin))
                    if entity!=int(other) or not evidence or value==0 or origin==current or key in seen:continue
                    seen.add(key);direction=1 if value>0 else -1
                    occ_key=(int(entity),int(origin),direction,int(evidence))
                    oid=self._world_occurrence_ids.get(occ_key)
                    if oid is None:
                        occ=self.population.recruit((WORLD_VIEW_CONTEXT,int(entity),int(origin),direction,int(evidence)))
                        oid=int(occ.identity);self._world_occurrence_ids[occ_key]=oid
                    states.append((int(entity),direction,int(evidence),oid))
        return tuple(states)

    def _select_explicit_template(self,context:int,arity:int,preference_context:int):
        return select_explicit_template(self,context,arity,preference_context)

    def _select_explicit_lexeme(self,feature:int,preference_context:int):
        return select_explicit_lexeme(self,feature,preference_context)

    def _realize_explicit_selected(self,context:int,atoms,preference_context:int,binding_identity=0,surface_proposal=None):
        return realize_explicit_selected(self,context,atoms,preference_context,binding_identity,surface_proposal,MAX_SELECTION_NETWORK_CANDIDATES,PREF_TEMPLATE,PREF_BINDING,PREF_LEXEME)

    def _realize_scene_surface(self,scene):
        if scene is None:return None
        if self.partner_present and self.partner_source>0:
            surface,template,_,_=self._realize_explicit_selected(int(scene.context),scene.atoms,self._selection_preference_context(scene))
            if surface is not None:return surface
        return self.language.realize(int(scene.context),scene.atoms)

    def _span_template_for_partner(self,relation:int,arity:int=2):
        partner=int(self.partner_source) if self.partner_present else 0
        rows=self.language.span_candidates(int(relation),int(arity))
        if partner:
            hit=[row for row in rows if partner in row.sources]
            if len(hit)==1:return hit[0]
            if hit:return None
        return self.language.span_template(int(relation),int(arity))

    def _apply_learned_discourse_surface(self,scene:SceneStateV2,surface,current_closure):
        return apply_learned_discourse_surface(self,scene,surface,current_closure)

    def current_expression_plan(self,action:ActionV2):
        """Reconstruct the exact transient discourse plan selected for this action."""
        if not isinstance(action,ActionV2):return None
        return current_expression_plan(self,action)

    def _realize_prospective_expression_candidate(self,plan:PlanV1,surface_context:int,span_relation:int):
        if plan.status!=1 or not plan.recipe_identity or not plan.actions:return None
        surface_context=int(surface_context);span_relation=int(span_relation)
        if surface_context<=0 or (len(plan.actions)>1 and span_relation<=0):return None
        preference_context=(mix64(surface_context^mix64(int(self.partner_source)^0x5E1EC710)
                                  ^mix64(int(self.partner_channel)))&((1<<63)-1)) or 1
        children=[]
        for step,(action_id,next_state) in enumerate(zip(plan.actions,plan.states[1:])):
            clause_atoms=(int(action_id),*(int(x) for x in next_state))
            surface,template,_lexical_ids,_alternatives=self._realize_explicit_selected(
                surface_context,clause_atoms,int(preference_context))
            if (surface is None or template is None or self.partner_source<=0
                    or int(self.partner_source) not in template.sources):return None
            resident_identity=int(_digest('prospective-expression-clause-v1',[
                int(plan.recipe_identity),step,list(clause_atoms)
            ])[:15],16) or 1
            children.append(self.utterances.remember(
                surface_context,resident_identity,surface,persist=False))
        try:
            if len(children)==1:
                child=children[0]
                expression_plan=TransientConstructionPlanV1(
                    int(_digest('prospective-expression-single-v1',[
                        int(plan.recipe_identity),int(child.identity),int(template.identity[:15],16)
                    ])[:15],16) or 1,
                    surface_context,int(template.identity[:15],16),(int(child.identity),),0)
            else:
                span=self._span_template_for_partner(span_relation,2)
                if (span is None or self.partner_source<=0
                        or int(self.partner_source) not in span.sources):return None
            if len(children)==2:
                expression_plan,_surface=rematerialize_transient_plan(
                    self.language,span_relation,tuple(children),span)
            elif len(children)>2:
                expression_plan,_surface=rematerialize_transient_sequence_plan(
                    self.language,span_relation,tuple(children),span)
        except HierarchicalRefuse:return None
        identity=int(_digest('prospective-expression-opportunity-v1',[
            int(plan.recipe_identity),surface_context,span_relation,list(map(int,plan.actions)),
            [list(map(int,state)) for state in plan.states],int(self.world_state_occurrence),
            int(self.body_state_occurrence),int(self.partner_source),int(self.partner_channel)
        ])[:15],16) or 1
        opportunity=ProspectiveExpressionOpportunityV1(
            identity,int(plan.recipe_identity),surface_context,span_relation,
            tuple(map(int,plan.actions)),tuple(map(int,plan.states[0])),
            tuple(map(int,plan.states[-1])),int(self.world_state_occurrence),
            int(self.body_state_occurrence),int(self.partner_source),int(self.partner_channel))
        return opportunity,expression_plan

    def _prospective_somatic_bias(self,plan:PlanV1):
        """Score predicted states from learned markers matching the current body."""
        body=next((row for row in self.population.occurrences
                   if row.identity==int(self.body_state_occurrence)),None)
        if body is None or self.body_state_source<=0:return 0
        signature=self._body_occurrence_signature(body);marked={}
        for ctx,configuration,source,support,counter in self._somatic_revisions.iter_revisions():
            if (ctx!=SOMA_CONTEXT or source in self.withdrawn_sources
                    or len(configuration)<4 or configuration[0][0]!=PREF_SOMA):continue
            entity=int(configuration[0][2]);body_signature=int(configuration[2][2]);body_source=int(configuration[3][2])
            if body_signature!=signature or body_source!=int(self.body_state_source):continue
            marked[entity]=marked.get(entity,0)+int(support)-int(counter)
        bias=0;seen=set()
        for state in plan.states[1:]:
            for atom in state:
                for entity in self._overlapping_entities(int(atom)) or (int(atom),):
                    if entity in marked and entity not in seen:
                        seen.add(entity);bias+=marked[entity]
        return bias

    def current_prospective_expression_frontier(self,surface_context:int,span_relation:int,max_candidates:int=8):
        """Realize a bounded multi-depth endogenous candidate population."""
        if (self.world_state is None or not self.body_target or not self.partner_present
                or self.partner_source<=0 or self.body_state_occurrence<=0):return ()
        plans=self.cognition.prospective_frontier(
            tuple(self.world_state),tuple(self.body_target),max_candidates=max_candidates,
            current_tick=self.tick_count,depth_slack=2)
        out=[]
        for plan in plans:
            realized=self._realize_prospective_expression_candidate(plan,surface_context,span_relation)
            if realized is not None:out.append((plan,*realized))
        return tuple(out)

    def _resident_selected_prospective_plan(self):
        """Select one current prospective route from resident history/body state."""
        if (self.world_state is None or not self.body_target or self.body_state_occurrence<=0):return None
        start=tuple(self.world_state);goal=tuple(self.body_target)
        expert_context=self._prospective_expert_context_signature()
        fast=self.cognition.active_expert(start,goal,expert_context)
        if fast is not None:return fast
        frontier=self.cognition.prospective_frontier(
            start,goal,current_tick=self.tick_count,depth_slack=2)
        if not frontier:return None
        def rank(plan):
            depth=max(1,len(plan.actions));return (
                self._prospective_somatic_bias(plan),
                (int(plan.score)*1024)//depth,
                -depth,
            )
        peak=max(rank(plan) for plan in frontier)
        winners=[plan for plan in frontier if rank(plan)==peak]
        if len(winners)!=1:return None
        selected=winners[0]
        self.cognition.probation_expert(start,goal,expert_context,selected,frontier)
        return selected

    def current_prospective_expression_plan(self,surface_context:int,span_relation:int):
        """Observer probe over the same resident route selection used by live speech."""
        if (not self.partner_present or self.partner_source<=0):return None
        selected=self._resident_selected_prospective_plan()
        if selected is None:return None
        return self._realize_prospective_expression_candidate(
            selected,surface_context,span_relation)

    def _resident_prospective_expression_candidates(self,plan:PlanV1,max_candidates:int=32):
        """Discover learned expression ecologies without an observer naming one.

        Candidate contexts and span relations come only from the resident learned
        surface ecology.  If several distinct learned ecologies can realize the
        same prospective Network, the organism refuses rather than letting host
        iteration order become semantic or discourse policy.
        """
        if (plan.status!=1 or not plan.recipe_identity or not plan.actions
                or not self.partner_present or self.partner_source<=0):return ()
        arities=tuple(sorted(set(1+len(state) for state in plan.states[1:])))
        contexts=sorted(set(int(context) for context,_arity in self.language._template_index
                            if all(self.language.template_candidates(int(context),arity)
                                   for arity in arities)))
        relations=((0,) if len(plan.actions)==1 else
            tuple(sorted(set(int(context) for context,arity in self.language._span_index
                             if int(arity)==2))))
        out=[];seen=set()
        for context in contexts:
            for relation in relations:
                realized=self._realize_prospective_expression_candidate(plan,context,relation)
                if realized is None:continue
                opportunity,expression_plan=realized
                if opportunity.identity in seen:continue
                seen.add(opportunity.identity);out.append((opportunity,expression_plan))
                if len(out)>int(max_candidates):return ()
        return tuple(out)

    def _prospective_expression_surface(self,expression_plan):
        child_ids=tuple(int(x) for x in getattr(expression_plan,'child_identities',()))
        if len(child_ids)==1:
            child=self.utterances.lookup(child_ids[0])
            template_row=self.language._template_identity_rows.get(
                int(expression_plan.template_identity))
            if (child is None or not child.surface or template_row is None
                    or int(template_row[0])!=int(expression_plan.context)):
                return None
            candidates=self.language.template_candidates(
                int(expression_plan.context),int(template_row[1]))
            if (not candidates
                    or not any(int(row.identity[:15],16)==int(expression_plan.template_identity)
                               for row in candidates)):return None
            return tuple(child.surface),child_ids
        if len(child_ids)<2:return None
        children=tuple(self.utterances.lookup(identity) for identity in child_ids)
        if any(child is None for child in children):return None
        template=self._span_template_for_partner(int(expression_plan.context),2)
        if template is None or int(template.identity[:15],16)!=int(expression_plan.template_identity):return None
        try:
            if len(children)==2:
                rebuilt,surface=rematerialize_transient_plan(
                    self.language,int(expression_plan.context),children,template)
            else:
                rebuilt,surface=rematerialize_transient_sequence_plan(
                    self.language,int(expression_plan.context),children,template)
        except HierarchicalRefuse:return None
        if int(rebuilt.identity)!=int(expression_plan.identity) or not surface:return None
        return tuple(surface),child_ids

    def _emit_resident_prospective_expression(self,plan:PlanV1):
        candidates=self._resident_prospective_expression_candidates(plan)
        if len(candidates)!=1:return None
        opportunity,expression_plan=candidates[0]
        if int(self.prospective_expression_announced)==int(opportunity.identity):return None
        rendered=self._prospective_expression_surface(expression_plan)
        if rendered is None:return None
        surface,child_ids=rendered
        body=next((row for row in self.population.occurrences
                   if row.identity==int(self.body_state_occurrence)),None)
        body_occurrence=0 if body is None else int(body.identity)
        body_signature=0 if body is None else self._body_occurrence_signature(body)
        body_source=0 if body is None else int(self.body_state_source)
        features=(PROSPECTIVE_NETWORK_TAG,int(opportunity.identity),int(opportunity.recipe_identity),
                  len(opportunity.actions),*opportunity.actions,int(opportunity.world_occurrence),
                  int(opportunity.body_occurrence),int(opportunity.partner_source),
                  int(opportunity.partner_channel))
        occ=self.population.recruit(features)
        contributors=tuple(dict.fromkeys(x for x in (
            int(opportunity.identity),int(opportunity.recipe_identity),int(expression_plan.identity),
            *child_ids,int(opportunity.world_occurrence),int(opportunity.body_occurrence),
            int(occ.identity),body_occurrence) if x))
        action=ActionV2(self.next_ticket,self.tick_count,int(self.partner_channel),int(self.partner_source),
            tuple(surface),int(occ.identity),int(opportunity.identity),int(expression_plan.template_identity),
            contributors,False,0,tuple(surface),False,0,(),(),int(expression_plan.identity),0,(),
            int(expression_plan.template_identity),0,(),(),body_occurrence,body_signature,body_source)
        self._ensure_action_capacity();self.next_ticket+=1;self.actions.append(action);self._index_action(action)
        self._action_commitments[action.ticket]=self._action_commitment(action)
        self.prospective_expression_announced=int(opportunity.identity)
        for identity in child_ids:self.utterances.retire(int(identity))
        return action

    def tick(self,surface_proposal=None):
        self.tick_count+=1;self.population.decay()
        # A completed public trajectory may await an independent social/world
        # consequence while a later occurrence becomes actionable.  Physical
        # motor actions still exclude one another; public consequence tickets do not.
        if any(not a.settled for a in self.motor_actions):return None
        repair=self._emit_pending_repair()
        if repair is not None:return repair
        cognitive=self._cognitive_tick()
        if cognitive is not None or self.information_need:return cognitive
        scene=self._select_pending_scene()
        if scene is None:scene=self._resident_world_scene()
        scene=self._complete_scene(scene)
        if scene is None or scene.acted or scene.demonstrated or scene.source in self.withdrawn_sources:return None
        if any(x==0 for x in scene.atoms) or not self.partner_present:return None
        hypothesis=self._resident_relation_hypothesis(scene)
        if hypothesis is not None:scene=hypothesis
        if surface_proposal is not None:
            try:
                proposal_scene=int(surface_proposal.scene_identity)
                proposal_world=int(getattr(surface_proposal,'world_occurrence',0))
            except (AttributeError,TypeError,ValueError):return None
            if proposal_scene!=int(scene.identity):return None
            if proposal_world and (proposal_world!=int(self.world_state_occurrence)
                                   or proposal_world!=int(scene.population_occurrence)):
                return None
        active_atoms,used_context,binding_identity,relation_occurrences=self._surface_view(scene)
        surface_context,conditions=self._surface_context(scene,used_context,active_atoms)
        preference_context=self._selection_preference_context(scene)
        if not self._port_compatibility_ok((surface_context,scene.context),active_atoms,conditions):return None
        somatic_states=self._somatic_state_occurrences(int(scene.context),active_atoms)
        world_states=self._world_state_occurrences(int(scene.context),active_atoms)
        template=None;lexical_ids=();form_slots=();surface=None;conditioned_veto=False
        if any(conditions):
            surface,template,lexical_ids,form_slots,conditioned_veto=self._realize_conditioned_selected(surface_context,active_atoms,conditions,preference_context,binding_identity,surface_proposal)
            if surface is not None:used_context=surface_context
            elif int(surface_context)!=int(used_context) and not self.language.template_candidates(int(surface_context),len(active_atoms)):
                surface,template,lexical_ids,form_slots,conditioned_veto=self._realize_conditioned_selected(int(used_context),active_atoms,conditions,preference_context,binding_identity,surface_proposal)
        if surface is None:
            conditioned_rows=self.language.template_candidates(int(surface_context),len(active_atoms)) if any(conditions) else ()
            if conditioned_rows and self._conditioned_surface_ports(active_atoms,conditions) is not None and not conditioned_veto:return None
            if any(int(x) and int(x)!=COND_REINSTATED for row in conditions for x in row):return None
            surface,template,lexical_ids,_alternatives=self._realize_explicit_selected(used_context,active_atoms,preference_context,binding_identity,surface_proposal);form_slots=()
        if surface is None or template is None:return None
        leaf_closure=self.utterances.remember(used_context,scene.identity,surface)
        surface,discourse_contributors,action_closure,span_identity=self._apply_learned_discourse_surface(scene,surface,leaf_closure)
        tid=int(template.identity[:15],16)
        if span_identity:
            value,evidence=self._selection_configuration_evidence(preference_context,self._selection_configuration(tid,lexical_ids,form_slots,span_identity,binding_identity))
            if evidence and value<=0:
                surface=tuple(leaf_closure.surface);discourse_contributors=();action_closure=leaf_closure;span_identity=0
            else:
                prior_ep=self._episode_by_id.get(int(self.last_shared_episode_by_partner.get(int(self.partner_source),0)))
                if prior_ep is not None:
                    world_states=world_states+self._world_state_occurrences(int(prior_ep.context),prior_ep.atoms)
                    somatic_states=somatic_states+self._somatic_state_occurrences(int(prior_ep.context),prior_ep.atoms)
        self._ensure_action_capacity()
        body=None
        if self.body_state_source>0 and self.body_state_source not in self.withdrawn_sources:
            body=next((row for row in self.population.occurrences
                       if row.identity==self.body_state_occurrence),None)
        body_occurrence=0 if body is None else int(body.identity)
        body_signature=0 if body is None else self._body_occurrence_signature(body)
        body_source=0 if body is None else int(self.body_state_source)
        feats=[int(used_context)];form_set=set(int(x) for x in form_slots)
        if body_signature:feats.extend((BODY_STATE_TAG,body_signature,body_source))
        for atom in active_atoms:
            if int(atom)==0:continue
            feats.extend(self._active_entity_features(int(atom)) or (int(atom),))
        occ=self.population.recruit(tuple(feats));selection_rows=[]
        tsel=self.population.recruit((0x51EC7,PREF_TEMPLATE,0,tid,preference_context,used_context));selection_rows.append((PREF_TEMPLATE,0,tid,tsel.identity))
        if binding_identity:
            bsel=self.population.recruit((0x51EC7,PREF_BINDING,0,binding_identity,preference_context,scene.population_occurrence,*relation_occurrences));selection_rows.append((PREF_BINDING,0,binding_identity,bsel.identity))
        for slot,(atom,cid) in enumerate(zip(active_atoms,lexical_ids),1):
            kind=PREF_FORM if slot in form_set else PREF_LEXEME
            lsel=self.population.recruit((0x51EC7,kind,slot,cid,preference_context,int(atom)));selection_rows.append((kind,slot,cid,lsel.identity))
        if span_identity:
            ssel=self.population.recruit((0x51EC7,PREF_SPAN,0,span_identity,preference_context,0));selection_rows.append((PREF_SPAN,0,span_identity,ssel.identity))
        selection_rows=tuple(selection_rows);network_identity=self._selection_network_identity(preference_context,occ.identity,action_closure.identity,selection_rows) if selection_rows else 0;somatic_occurrences=tuple(int(row[3]) for row in somatic_states);world_occurrences=tuple(int(row[3]) for row in world_states);contributors=tuple(dict.fromkeys(x for x in (scene.identity,scene.population_occurrence,*relation_occurrences,*somatic_occurrences,*world_occurrences,body_occurrence,occ.identity,tid,scene.completed_from_episode,action_closure.identity,*lexical_ids,*(r[3] for r in selection_rows),network_identity,*discourse_contributors) if x))
        planned=tuple(surface);actual=list(planned)
        if 0<=self.output_fault_offset<len(actual):
            actual[self.output_fault_offset]=self.output_fault_value;self.output_fault_offset=-1
        actual=tuple(actual)
        action=ActionV2(self.next_ticket,self.tick_count,self.partner_channel,self.partner_source,actual,occ.identity,scene.identity,tid,contributors,False,0,planned,False,preference_context,lexical_ids,selection_rows,action_closure.identity,network_identity,form_slots,span_identity,binding_identity,relation_occurrences,somatic_occurrences,body_occurrence,body_signature,body_source);self.next_ticket+=1;self.actions.append(action);self._index_action(action);self._action_commitments[action.ticket]=self._action_commitment(action);scene.acted=True
        if actual!=planned:self.pending_repair=(scene.identity,self.partner_channel,self.partner_source,planned,tid,contributors,action.ticket)
        return action

    def checkpoint(self):
        repair_origin=int(self.pending_repair[6]) if self.pending_repair is not None and len(self.pending_repair)>6 else 0
        checkpoint_actions=[a for a in self.actions if not a.settled or (repair_origin and a.ticket==repair_origin)]
        checkpoint_tickets={int(a.ticket) for a in checkpoint_actions}
        checkpoint_commitments={int(k):v for k,v in self._action_commitments.items() if int(k) in checkpoint_tickets}
        checkpoint_episode_rows=[]
        for episode in self.episodes:
            row=dict(episode.__dict__)
            # Episode signature is an exact function of context/atoms plus the fixed
            # population topology. Persist the evidence, not its rebuildable index key.
            row.pop('signature',None);checkpoint_episode_rows.append(row)
        checkpoint_action_rows=[]
        for action in checkpoint_actions:
            row=dict(action.__dict__)
            # When the emitted bytes equal the plan, planned_payload is exactly
            # reconstructable from payload. Fault/repair divergence keeps both.
            if tuple(action.planned_payload)==tuple(action.payload):row.pop('planned_payload',None)
            checkpoint_action_rows.append(row)
        # last_retrieval is an observer diagnostic of the most recent retrieval.
        # It never participates in a later transition and is intentionally not z_t.
        return copy.deepcopy({'schema':8,'tick_count':self.tick_count,'next_scene':self.next_scene,'next_episode':self.next_episode,'next_ticket':self.next_ticket,
                'population':self.population.checkpoint(),'language':self.language.checkpoint(),'utterance_leaves':self.utterances.checkpoint(),'cognition':self.cognition.checkpoint(),
                'recruitment':self.recruitment.checkpoint(),'event_recruitment':self.event_recruitment.checkpoint(),
                'pending_event_ticket':self._pending_event_ticket,'pending_event_relation':self._pending_event_relation,
                'prospective_source_closures':[{'identity':row.identity,'source':row.source,'recipe_identity':row.recipe_identity,'cursor':row.cursor,'hypotheses':[{'action_id':h.action_id,'binding_identity':h.binding_identity,'binding_atoms':list(h.binding_atoms)} for h in row.hypotheses]} for row in sorted(self._prospective_source_closures.values(),key=lambda x:x.identity)],
                'current_scene_identity':0 if self.current_scene is None else self.current_scene.identity,
                'pending_scenes':[x.__dict__ for x in self.pending_scenes],'scene_links':[x.__dict__ for x in self.scene_links],
                'episodes':checkpoint_episode_rows,'actions':checkpoint_action_rows,'motor_actions':[x.__dict__ for x in self.motor_actions],
                'withdrawn_sources':sorted(self.withdrawn_sources),
                'world_state':None if self.world_state is None else list(self.world_state),'world_source':self.world_source,'world_state_occurrence':self.world_state_occurrence,'body_target':list(self.body_target),'body_state':list(self.body_state),'body_state_source':self.body_state_source,'body_state_occurrence':self.body_state_occurrence,'affordances':sorted(self.affordances),
                'exploration_trials':[{'state':list(k[0]),'action':k[1],'count':v} for k,v in sorted(self.exploration_trials.items())],'information_need':list(self.information_need),'information_need_asked':self.information_need_asked,'communication_channel':self.communication_channel,'prospective_expression_announced':self.prospective_expression_announced,
                'partner_present':self.partner_present,'partner_source':self.partner_source,'partner_channel':self.partner_channel,'shared_episode_relations':[x.__dict__ for x in self.shared_episode_relations],
                'output_fault_offset':self.output_fault_offset,'output_fault_value':self.output_fault_value,'pending_repair':None if self.pending_repair is None else [self.pending_repair[0],self.pending_repair[1],self.pending_repair[2],list(self.pending_repair[3]),self.pending_repair[4],list(self.pending_repair[5]),int(self.pending_repair[6]) if len(self.pending_repair)>6 else 0],
                'selection_revisions_packed':self._selection_revisions.checkpoint().hex(),'somatic_revisions_packed':self._somatic_revisions.checkpoint().hex(),'world_revisions_packed':self._world_revisions.checkpoint().hex(),'action_commitments':checkpoint_commitments,'entity_features':[[k,*v] for k,v in sorted(self.entity_features.items())],'entity_feature_sources':[[entity,*sorted(sources)] for entity,sources in sorted(self.entity_feature_sources.items())],'entity_conditions':[[k,*v] for k,v in sorted(self.entity_conditions.items())],'source_assertions':[x.__dict__ for x in self.source_assertions],'source_calibrations':[x.__dict__ for x in self.source_calibrations]})

    @classmethod
    def restore(cls,d):
        if d.get('schema')!=8:raise ValueError('organism:checkpoint')
        x=cls(PopulationSpecV1(**d['population']['spec']));x.population=PopulationBankV1.restore(d['population']);x.language=LearnedSurfaceEcologyV1.restore(d['language']);x.utterances=UtteranceBoundaryBankV1();x.utterances.restore(d.get('utterance_leaves'));x.cognition=TransitionEcologyV1.restore(d.get('cognition',{'schema':3,'evidence':[],'withdrawn':[]}),x.__cognition_authority);x.recruitment=PopulationRecruitmentEcologyV1.restore(d.get('recruitment',{'schema':1,'morphologies':[],'relations':[],'withdrawn_sources':[]}));x.event_recruitment=ResidentEventRecruitmentV1.restore(x.recruitment,d.get('event_recruitment',{'schema':3,'max_lag':8,'next_ticket':1,'common_cause_support':[],'common_cause_keys':[],'common_cause_control':PredictiveCreditBankV1().checkpoint(),'pending':[],'closures':[]}));x._pending_event_ticket=int(d.get('pending_event_ticket',0));x._pending_event_relation=int(d.get('pending_event_relation',0));x._prospective_source_closures={}
        for raw in d.get('prospective_source_closures',()):
            hs=tuple(ResidentProspectiveHypothesisV1(int(h['action_id']),int(h['binding_identity']),tuple(map(int,h.get('binding_atoms',())))) for h in raw.get('hypotheses',()))
            row=ResidentProspectiveClosureV1(int(raw['identity']),int(raw['source']),int(raw['recipe_identity']),hs,int(raw.get('cursor',0)));x._prospective_source_closures[row.identity]=row
        x.tick_count=int(d['tick_count']);x.next_scene=int(d['next_scene']);x.next_episode=int(d['next_episode']);x.next_ticket=int(d['next_ticket'])
        rows=d.get('pending_scenes')
        if rows is None:
            cs=d.get('current_scene');rows=[] if cs is None else [cs]
        x.pending_scenes=[SceneStateV2(int(cs['identity']),int(cs['channel']),int(cs['context']),tuple(cs['atoms']),int(cs['source']),int(cs['population_occurrence']),bool(cs['demonstrated']),bool(cs['acted']),int(cs['completed_from_episode']),int(cs.get('binding_identity',0)),tuple(int(y) for y in cs.get('relation_occurrences',()))) for cs in rows]
        current_id=int(d.get('current_scene_identity',0));x.current_scene=next((cs for cs in x.pending_scenes if cs.identity==current_id),None)
        if x.current_scene is None and x.pending_scenes:x.current_scene=x.pending_scenes[-1]
        x.scene_links=[SceneLinkV2(int(l['left_scene']),int(l['right_scene']),int(l['relation']),int(l['source']),bool(l.get('active',True)),int(l.get('population_occurrence',0))) for l in d.get('scene_links',[])]
        x.episodes=[EpisodeV2(int(e['identity']),int(e.get('scene_identity',0)),int(e['context']),tuple(e['atoms']),int(e['source']),x._scene_signature(int(e['context']),tuple(e['atoms'])),int(e['tick'])) for e in d['episodes']]
        x.entity_features={int(row[0]):tuple(int(y) for y in row[1:]) for row in d.get('entity_features',[])}
        x.entity_feature_sources={}
        for row in d.get('entity_feature_sources',[]):
            if len(row)<2:raise ValueError('organism:checkpoint_entity_feature_sources')
            entity=int(row[0]);sources=set(map(int,row[1:]))
            if entity in x.entity_feature_sources or not sources or min(sources)<=0:raise ValueError('organism:checkpoint_entity_feature_sources')
            x.entity_feature_sources[entity]=sources
        if set(x.entity_feature_sources)!=set(x.entity_features):raise ValueError('organism:checkpoint_entity_feature_sources')
        x.entity_conditions={int(row[0]):tuple(int(y) for y in row[1:]) for row in d.get('entity_conditions',[])}
        x.actions=[ActionV2(int(a['ticket']),int(a['tick']),int(a['channel']),int(a['source']),tuple(a['payload']),int(a['population_occurrence']),int(a['scene_identity']),int(a['template_identity']),tuple(a['contributors']),bool(a['settled']),int(a['effect']),tuple(a.get('planned_payload',a['payload'])),bool(a.get('repair',False)),int(a.get('selection_context',0)),tuple(a.get('lexical_identities',())),tuple(tuple(int(y) for y in row) for row in a.get('selection_occurrences',())),int(a.get('closure_identity',0)),int(a.get('selection_network_identity',0)),tuple(int(y) for y in a.get('form_slots',())),int(a.get('span_identity',0)),int(a.get('binding_identity',0)),tuple(int(y) for y in a.get('relation_occurrences',())),tuple(int(y) for y in a.get('somatic_occurrences',())),int(a.get('body_occurrence',0)),int(a.get('body_signature',0)),int(a.get('body_source',0))) for a in d['actions']]
        x.motor_actions=[MotorActionV2(ticket=int(a['ticket']),tick=int(a['tick']),action_id=int(a['action_id']),source=int(a['source']),state_before=tuple(a['state_before']),population_occurrence=int(a['population_occurrence']),settled=bool(a['settled']),effect=int(a['effect']),state_after=tuple(a['state_after']),source_assertion_ids=tuple(a.get('source_assertion_ids',())),source_context=int(a.get('source_context',0)),source_occurrences=tuple(tuple(int(y) for y in row) for row in a.get('source_occurrences',())),source_counterfactual_action=int(a.get('source_counterfactual_action',0)),event_ticket=int(a.get('event_ticket',0)),event_relation=int(a.get('event_relation',0)),world_occurrences=tuple(int(y) for y in a.get('world_occurrences',())),somatic_occurrences=tuple(int(y) for y in a.get('somatic_occurrences',())),prospective_recipe=int(a.get('prospective_recipe',0)),prospective_snapshot=tuple(tuple(int(y) for y in row) for row in a.get('prospective_snapshot',())),prospective_context_signature=int(a.get('prospective_context_signature',0)),independent_consequence=bool(a.get('independent_consequence',False)),lexical_occurrences=tuple(tuple(int(y) for y in row) for row in a.get('lexical_occurrences',()))) for a in d.get('motor_actions',[])]
        x.withdrawn_sources=set(map(int,d['withdrawn_sources']));x.last_retrieval={'status':0,'winner':0,'score':0,'alternatives':0};x.world_state=None if d.get('world_state') is None else tuple(d['world_state']);x.world_source=int(d.get('world_source',0));x.world_state_occurrence=int(d.get('world_state_occurrence',0));x.body_target=tuple(d.get('body_target',[]));x.body_state=tuple(d.get('body_state',[]));x.body_state_source=int(d.get('body_state_source',0));x.body_state_occurrence=int(d.get('body_state_occurrence',0));x.affordances=set(map(int,d.get('affordances',[])));x.exploration_trials={(tuple(row['state']),int(row['action'])):int(row['count']) for row in d.get('exploration_trials',[])};x.information_need=tuple(d.get('information_need',[]));x.information_need_asked=bool(d.get('information_need_asked',False));x.communication_channel=int(d.get('communication_channel',0));x.prospective_expression_announced=int(d.get('prospective_expression_announced',0));x.partner_present=bool(d.get('partner_present',False));x.partner_source=int(d.get('partner_source',0));x.partner_channel=int(d.get('partner_channel',0));x.shared_episode_relations=[SharedEpisodeRelationV2(int(r['partner']),int(r['episode_identity']),int(r['action_ticket']),int(r['closure_identity']),tuple(int(y) for y in r.get('source_roots',()))) for r in d.get('shared_episode_relations',[])];x.output_fault_offset=int(d.get('output_fault_offset',-1));x.output_fault_value=int(d.get('output_fault_value',0));pr=d.get('pending_repair');x.pending_repair=None if pr is None else (int(pr[0]),int(pr[1]),int(pr[2]),tuple(pr[3]),int(pr[4]),tuple(pr[5]),int(pr[6]) if len(pr)>6 else 0);x._load_selection_revisions(d);x._action_commitments={int(k):str(v) for k,v in d.get('action_commitments',{}).items()};x.source_assertions=[SourceAssertionV2(int(r['identity']),int(r['source']),int(r['context']),int(r['action_id']),tuple(r['state']),int(r.get('repetitions',1)),bool(r.get('active',True)),int(r.get('language_binding',0)),tuple(int(y) for y in r.get('binding_atoms',())),tuple(int(y) for y in r.get('predicted_state',())),int(r.get('prospective_closure',0)),int(r.get('prospective_step',0)),tuple(int(y) for y in r.get('lexical_eligibilities',())),int(r.get('lexical_occurrence',0)),int(r.get('lexical_tick',0))) for r in d.get('source_assertions',[])];x.source_calibrations=[SourceCalibrationV2(int(r['source']),int(r['context']),int(r.get('support',0)),int(r.get('counter',0)),int(r.get('revision',0)),bool(r.get('active',True))) for r in d.get('source_calibrations',[])];x._rebuild_runtime_indices()
        for action in x.motor_actions:
            if action.event_ticket:
                if action.prospective_recipe or action.prospective_snapshot or action.prospective_context_signature or action.lexical_occurrences:raise ValueError('organism:checkpoint_motor_action_commitment')
                continue
            if action.settled:continue
            occurrence=next((row for row in x.population.occurrences if row.identity==action.population_occurrence),None)
            try:
                features=x._motor_action_features(action.action_id,action.state_before,
                                                  action.prospective_recipe,action.prospective_snapshot,action.prospective_context_signature,action.lexical_occurrences)
            except ValueError as exc:
                raise ValueError('organism:checkpoint_motor_action_commitment') from exc
            if occurrence is None or occurrence.feature_count!=len(features) or occurrence.sites!=x.population.signature(features):raise ValueError('organism:checkpoint_motor_action_commitment')
        if ((x.world_state is None and x.world_state_occurrence)
                or (x.world_state_occurrence and not any(
                    row.identity==x.world_state_occurrence
                    for row in x.population.occurrences))):
            raise ValueError('organism:checkpoint_world_occurrence')
        if set(x._action_commitments)!=set(x._action_by_ticket) or any(x._action_commitments[t]!=x._action_commitment(a) for t,a in x._action_by_ticket.items()):raise ValueError('organism:checkpoint_action_commitment')
        return x

    def _load_selection_revisions(self,d):
        packed=d.get('selection_revisions_packed')
        if packed is not None:
            blob=bytes.fromhex(packed) if isinstance(packed,str) else bytes(packed)
            self._selection_revisions=PackedSelectionRevisionV1.restore(blob)
        else:
            self._selection_revisions=PackedSelectionRevisionV1()
            for row in d.get('selection_configuration_revisions',[]):
                configuration=tuple(tuple(int(y) for y in member) for member in row.get('configuration',()))
                if not configuration:continue
                self._selection_revisions._write(int(row['context']),configuration,int(row['source']),int(row.get('support',0)),int(row.get('counter',0)))
        soma=d.get('somatic_revisions_packed')
        if soma is not None:
            blob=bytes.fromhex(soma) if isinstance(soma,str) else bytes(soma)
            self._somatic_revisions=PackedSelectionRevisionV1.restore(blob)
        else:
            self._somatic_revisions=PackedSelectionRevisionV1()
        world=d.get('world_revisions_packed')
        if world is not None:
            blob=bytes.fromhex(world) if isinstance(world,str) else bytes(world)
            self._world_revisions=PackedSelectionRevisionV1.restore(blob)
        else:
            self._world_revisions=PackedSelectionRevisionV1()

    def digest(self):return _digest('reference-organism-v2',self.checkpoint())
