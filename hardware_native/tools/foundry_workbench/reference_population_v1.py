#!/usr/bin/env python3
"""Strict numeric population substrate for the graph-neutral reference organism.

This is architecture-workshop code, not a neuron simulator and not a language engine.
It tests whether very many individually stateful simple sites can be represented and
updated with touched-work semantics. All resident content is numeric.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from reference_population_physical_plane_v1 import PlaneSpecV1, PopulationPhysicalPlaneV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q

MASK64=(1<<64)-1
RECRUITMENT_NETWORK_TAG=0x4E455457
RESIDENT_PORT_TAG=0x504F5254
RESIDENT_ACTION_TAG=0x4143544E
RESIDENT_REAFFERENCE_TAG=0x52454146


def mix64(x:int)->int:
    x=(int(x)+0x9E3779B97F4A7C15)&MASK64
    x=((x^(x>>30))*0xBF58476D1CE4E5B9)&MASK64
    x=((x^(x>>27))*0x94D049BB133111EB)&MASK64
    return (x^(x>>31))&MASK64


@dataclass(frozen=True)
class PopulationSpecV1:
    site_count:int
    fanout:int=2
    sites_per_feature:int=2
    territory_count:int=42
    eligibility_horizon:int=8

    def validate(self):
        if not 32<=self.site_count<=2_000_000: raise ValueError('population:site_count')
        if not 1<=self.fanout<=8: raise ValueError('population:fanout')
        if not 1<=self.sites_per_feature<=16: raise ValueError('population:sites_per_feature')
        if not 1<=self.territory_count<=64: raise ValueError('population:territory_count')
        if not 1<=self.eligibility_horizon<=255: raise ValueError('population:eligibility_horizon')


@dataclass(frozen=True)
class PopulationOccurrenceV1:
    identity:int
    tick:int
    sites:tuple[int,...]
    feature_count:int
    fanout:int

    @property
    def edges(self)->tuple[int,...]:
        return tuple(site*self.fanout+lane for site in self.sites for lane in range(self.fanout))


@dataclass
class PopulationRecruitmentRelationV1:
    identity:int
    morphologies:tuple[int,...]
    credit:int=0
    evidence_count:int=0
    source_evidence:tuple[int,...]=()


@dataclass(frozen=True)
class ResidentPortParticipationV1:
    episode:int
    channel:int
    source:int
    occurrence:int
    tick:int


@dataclass
class ResidentEventClosureV1:
    ticket:int
    episode:int
    source:int
    network_occurrence:int
    member_occurrences:tuple[int,...]
    channels:tuple[int,...]
    opened_tick:int
    action_occurrence:int=0
    action_id:int=0


class PopulationRecruitmentEcologyV1:
    """Persistent relations among population morphologies, never sensory payloads.

    Current modality-specific states remain fresh PopulationOccurrenceV1 instances.
    This ecology stores only which prepared distributed morphologies have earned
    future co-recruitment authority through actual joint participation plus an
    independent consequence. No modality name, word, pixel, waveform, concept,
    reward label, or raw surface is represented here.
    """
    def __init__(self):
        self.morphologies:dict[int,tuple[int,...]]={}
        self.relations:dict[int,PopulationRecruitmentRelationV1]={}
        self._relations_by_morphology:dict[int,set[int]]={}
        self.withdrawn_sources:set[int]=set()
        self.disabled_morphologies:set[int]=set()
        self.last_touches=0

    @staticmethod
    def morphology_identity(signature)->int:
        h=hashlib.sha256(b'population-morphology-v1\0')
        for site in tuple(map(int,signature)):
            h.update(site.to_bytes(4,'little',signed=False))
        value=int.from_bytes(h.digest()[:8],'little')&((1<<63)-1)
        return value or 1

    @staticmethod
    def relation_identity(morphologies)->int:
        children=tuple(sorted(set(map(int,morphologies))))
        if len(children)<2:raise ValueError('population:relation_children')
        h=hashlib.sha256(b'population-recruitment-relation-v1\0')
        for child in children:h.update(child.to_bytes(8,'little',signed=False))
        value=int.from_bytes(h.digest()[:8],'little')&((1<<63)-1)
        return value or 1

    def prepare_morphology(self,signature)->int:
        signature=tuple(sorted(set(map(int,signature))))
        if not signature:raise ValueError('population:morphology_empty')
        identity=self.morphology_identity(signature)
        prior=self.morphologies.get(identity)
        if prior is not None and prior!=signature:raise ValueError('population:morphology_collision')
        self.morphologies[identity]=signature
        return identity

    def _index_relation(self,row:PopulationRecruitmentRelationV1)->None:
        for morphology in row.morphologies:
            self._relations_by_morphology.setdefault(morphology,set()).add(row.identity)

    def relation_rows_touching(self,morphologies):
        identities=set()
        for morphology in set(map(int,morphologies)):
            identities.update(self._relations_by_morphology.get(morphology,()))
        return tuple(self.relations[identity] for identity in sorted(identities)
                     if identity in self.relations)

    @staticmethod
    def network_occurrence_features(occurrences)->tuple[int,...]:
        identities=tuple(int(row.identity) for row in occurrences)
        if len(identities)<2 or len(set(identities))!=len(identities):
            raise ValueError('population:relation_members')
        return (RECRUITMENT_NETWORK_TAG,len(identities),*identities)

    def record_qualified_network(self,bank:'PopulationBankV1',network_occurrence,
                                 occurrences,source:int,effect:int,
                                 independent:bool)->int:
        occurrences=tuple(occurrences);source=int(source)
        if source<=0 or len(occurrences)<2 or not independent or effect==0:
            return 0
        retained={row.identity:row for row in bank.occurrences}
        actual_network=retained.get(int(network_occurrence.identity))
        if actual_network is None or actual_network!=network_occurrence:
            raise ValueError('population:relation_network_occurrence')
        expected_features=self.network_occurrence_features(occurrences)
        if (actual_network.feature_count!=len(expected_features) or
                actual_network.sites!=bank.signature(expected_features)):
            raise ValueError('population:relation_network_signature')
        signatures=[];morphologies=[]
        for occurrence in occurrences:
            actual=retained.get(int(occurrence.identity))
            if actual is None or actual!=occurrence:raise ValueError('population:relation_occurrence')
            signature=tuple(sorted(set(map(int,actual.sites))))
            morphology=self.morphology_identity(signature)
            prior=self.morphologies.get(morphology)
            if prior is not None and prior!=signature:
                raise ValueError('population:morphology_collision')
            signatures.append(signature)
            morphologies.append(morphology)
        if len(set(morphologies))<2:return 0
        identity=self.relation_identity(morphologies)
        # Validation is complete before mutation. Credit belongs to the one actual
        # joint Network Occurrence; channel members establish provenance and must
        # not each receive a duplicate marginal update.
        settled=bank.settle(actual_network,effect,independent)
        if settled.get('credit',0)<=0:return 0
        for morphology,signature in zip(morphologies,signatures):
            self.morphologies[morphology]=signature
            # Reacquisition is itself gated by a fresh actual joint Occurrence and
            # independent consequence; merely activating a cue cannot heal a lesion.
            self.disabled_morphologies.discard(morphology)
        row=self.relations.get(identity)
        if row is None:
            row=PopulationRecruitmentRelationV1(identity,tuple(sorted(set(morphologies))))
            self.relations[identity]=row
            self._index_relation(row)
        if source not in row.source_evidence:
            row.source_evidence=tuple(sorted((*row.source_evidence,source)))
        row.evidence_count+=1
        row.credit=max(-127,min(127,row.credit+(1 if effect>0 else -1)))
        return identity

    def withdraw_source(self,source:int)->None:
        self.withdrawn_sources.add(int(source))

    def lesion_morphology(self,morphology:int)->None:
        morphology=int(morphology)
        if morphology not in self.morphologies:raise ValueError('population:morphology_unknown')
        self.disabled_morphologies.add(morphology)

    def unfold_candidate_rows(self,cue:PopulationOccurrenceV1,causal_recall:bool=False):
        """Return every live consequence-qualified relation touching this cue.

        This is a rebuildable nomination view, not selection authority. Callers may
        intersect it with their own current resident constraints; exact ambiguity
        must remain unresolved after that intersection.  Ordinary nomination
        requires positive signed credit.  Event-state recall instead ranks actual
        independent source support, so a reliably harmful relation remains
        recallable without becoming a positively nominated action.
        """
        cue_morphology=self.morphology_identity(cue.sites)
        rows=self.relation_rows_touching((cue_morphology,))
        candidates=[]
        for row in rows:
            if any(m in self.disabled_morphologies for m in row.morphologies):continue
            live_sources=tuple(s for s in row.source_evidence if s not in self.withdrawn_sources)
            if not live_sources or (row.credit<=0 and not causal_recall):continue
            strength=len(live_sources) if causal_recall else row.credit
            candidates.append((strength,row.identity,tuple(m for m in row.morphologies if m!=cue_morphology),live_sources))
        self.last_touches=len(rows)
        return tuple(sorted(candidates,key=lambda row:(-row[0],row[1],row[2])))

    def unfold_candidates(self,cue:PopulationOccurrenceV1,causal_recall:bool=False):
        candidates=self.unfold_candidate_rows(cue,causal_recall)
        if not candidates:return ()
        best=candidates[0][0]
        winners=[row for row in candidates if row[0]==best]
        if len(winners)!=1:return ()
        return winners[0][2]

    def activate_morphology(self,bank:'PopulationBankV1',morphology:int,retain:bool=False)->PopulationOccurrenceV1:
        signature=self.morphologies.get(int(morphology))
        if signature is None:raise ValueError('population:morphology_unknown')
        return bank.activate_signature(signature,retain=retain)

    def checkpoint(self):
        return {'schema':1,'morphologies':[[k,list(v)] for k,v in sorted(self.morphologies.items())],
                'relations':[r.__dict__ for r in sorted(self.relations.values(),key=lambda r:r.identity)],
                'withdrawn_sources':sorted(self.withdrawn_sources),
                'disabled_morphologies':sorted(self.disabled_morphologies)}

    @classmethod
    def restore(cls,data):
        if data.get('schema')!=1:raise ValueError('population:relation_checkpoint')
        out=cls()
        for identity,signature in data.get('morphologies',()):
            got=out.prepare_morphology(tuple(map(int,signature)))
            if got!=int(identity):raise ValueError('population:morphology_checkpoint')
        for raw in data.get('relations',()):
            row=PopulationRecruitmentRelationV1(int(raw['identity']),tuple(map(int,raw['morphologies'])),int(raw.get('credit',0)),int(raw.get('evidence_count',0)),tuple(map(int,raw.get('source_evidence',()))))
            if out.relation_identity(row.morphologies)!=row.identity or any(m not in out.morphologies for m in row.morphologies):raise ValueError('population:relation_checkpoint')
            out.relations[row.identity]=row
            out._index_relation(row)
        out.withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        out.disabled_morphologies=set(map(int,data.get('disabled_morphologies',())))
        if any(m not in out.morphologies for m in out.disabled_morphologies):raise ValueError('population:morphology_checkpoint')
        return out


class ResidentEventRecruitmentV1:
    """Transient physical closure that can earn a persistent morphology relation.

    Episode and channel numbers are transport coordinates, not referent labels.
    Closing an event creates one actual joint Network Occurrence but no credit.
    Only a later participating action plus an actual independent reafference can
    revise PopulationRecruitmentEcologyV1.
    """
    def __init__(self,ecology:PopulationRecruitmentEcologyV1,max_lag:int=8):
        if not 1<=int(max_lag)<=255:raise ValueError('population:event_lag')
        self.ecology=ecology;self.max_lag=int(max_lag);self.next_ticket=1
        self.pending:list[ResidentPortParticipationV1]=[]
        self.closures:dict[int,ResidentEventClosureV1]={}
        self.common_cause_support:dict[int,int]={}
        self.common_cause_keys:dict[int,int]={}
        self.common_cause_control=PredictiveCreditBankV1(128)
        self.last_touches=0

    @staticmethod
    def _retained(bank:'PopulationBankV1',identity:int):
        return next((row for row in bank.occurrences if row.identity==int(identity)),None)

    def contact(self,bank:'PopulationBankV1',episode:int,channel:int,source:int,features):
        episode=int(episode);channel=int(channel);source=int(source);features=tuple(map(int,features))
        if episode<=0 or channel<=0 or source<=0 or not features:raise ValueError('population:event_contact')
        occurrence=bank.recruit((RESIDENT_PORT_TAG,channel,len(features),*features))
        self.pending.append(ResidentPortParticipationV1(episode,channel,source,occurrence.identity,occurrence.tick))
        return occurrence

    def recall_relations(self,bank:'PopulationBankV1',episode:int,source:int):
        """Return co-maximal learned relations supported by this actual event.

        Current Occurrences supply coverage; prior independently settled source
        evidence supplies reliability.  This is resident competition, not credit:
        recall cannot create or revise a relation.
        """
        episode=int(episode);source=int(source)
        rows=[row for row in self.pending
              if row.episode==episode and row.source==source]
        if not rows:self.last_touches=0;return ()
        newest=max(row.tick for row in rows)
        rows=[row for row in rows if newest-row.tick<=self.max_lag]
        occurrences=tuple(self._retained(bank,row.occurrence) for row in rows)
        if any(row is None for row in occurrences):raise ValueError('population:event_occurrence')
        present={self.ecology.morphology_identity(row.sites) for row in occurrences}
        relation_rows=self.ecology.relation_rows_touching(present)
        candidates=[]
        for relation in relation_rows:
            overlap=present.intersection(relation.morphologies)
            if any(m in self.ecology.disabled_morphologies
                   for m in relation.morphologies):continue
            live_sources=tuple(s for s in relation.source_evidence
                               if s not in self.ecology.withdrawn_sources)
            if not live_sources:continue
            score=(int(len(overlap)==len(relation.morphologies)),
                   len(overlap),len(live_sources))
            candidates.append((score,relation.identity))
        self.last_touches=len(relation_rows)
        if not candidates:return ()
        best=max(score for score,_identity in candidates)
        return tuple(sorted(identity for score,identity in candidates if score==best))

    def close(self,bank:'PopulationBankV1',episode:int,source:int):
        episode=int(episode);source=int(source)
        rows=[row for row in self.pending if row.episode==episode and row.source==source]
        self.last_touches=len(rows)
        # Ambiguous duplicate channels are refused instead of host-resolved.
        channels=tuple(row.channel for row in rows)
        if len(rows)<2 or len(set(channels))!=len(channels):return None
        if max(row.tick for row in rows)-min(row.tick for row in rows)>self.max_lag:return None
        occurrences=tuple(self._retained(bank,row.occurrence) for row in rows)
        if any(row is None for row in occurrences):raise ValueError('population:event_occurrence')
        occurrences=tuple(sorted(occurrences,key=lambda row:row.identity))
        network=bank.recruit(self.ecology.network_occurrence_features(occurrences))
        closure=ResidentEventClosureV1(self.next_ticket,episode,source,network.identity,
                                       tuple(row.identity for row in occurrences),
                                       tuple(sorted(channels)),network.tick)
        self.next_ticket+=1;self.closures[closure.ticket]=closure
        self.pending=[row for row in self.pending if not (row.episode==episode and row.source==source)]
        return closure

    @staticmethod
    def _physical_lineage(rows)->int:
        """Content-free provenance for a resident join; never a referent identity."""
        h=hashlib.sha256(b'resident-common-cause-lineage-v1\0')
        for channel,source in sorted(set((int(row.channel),int(row.source)) for row in rows)):
            h.update(channel.to_bytes(4,'little'));h.update(source.to_bytes(8,'little'))
        return (int.from_bytes(h.digest()[:8],'little')&((1<<63)-1)) or 1

    def close_common_cause(self,bank:'PopulationBankV1',episode:int):
        """Close only cross-port traces with matching nonconstant change timing.

        Transport provenance remains separate per port.  The resident comparison
        therefore receives no caller-selected common source or expected pair.
        """
        episode=int(episode);rows=[row for row in self.pending if row.episode==episode]
        by_channel={}
        for row in rows:by_channel.setdefault(int(row.channel),[]).append(row)
        groups=[sorted(group,key=lambda row:row.tick) for _,group in sorted(by_channel.items())]
        self.last_touches=len(rows)
        if len(groups)<2 or min(map(len,groups))<3 or len({len(group) for group in groups})!=1:return None
        if max(row.tick for row in rows)-min(row.tick for row in rows)>self.max_lag:return None
        traces=[];occurrences=[];key_rows=[]
        for channel,group in zip(sorted(by_channel),groups):
            current=[]
            for row in group:
                occurrence=self._retained(bank,row.occurrence)
                if occurrence is None:raise ValueError('population:event_occurrence')
                current.append(occurrence);occurrences.append(occurrence)
            morphologies=tuple(self.ecology.morphology_identity(row.sites) for row in current)
            traces.append(tuple(morphologies[i]!=morphologies[i-1] for i in range(1,len(morphologies))))
            key_rows.append((int(channel),morphologies[0],morphologies[-1]))
        self.pending=[row for row in self.pending if row.episode!=episode]
        # Constant activity contributes no causal evidence. Correlated versus
        # deranged change timing revises one bounded common/separate competition.
        if not any(any(trace) for trace in traces):return None
        h=hashlib.sha256(b'resident-common-cause-prior-v1\0')
        for channel,first,last in key_rows:
            h.update(channel.to_bytes(4,'little'));h.update(first.to_bytes(8,'little'));h.update(last.to_bytes(8,'little'))
        key=(int.from_bytes(h.digest()[:8],'little')&((1<<63)-1)) or 1
        if key not in self.common_cause_support and len(self.common_cause_support)>=128:
            victim=min(self.common_cause_support,key=lambda k:(abs(self.common_cause_support[k]),k))
            self.common_cause_support.pop(victim)
        common=len(set(traces))==1 and any(traces[0]) and not all(traces[0])
        support=max(-2,min(2,self.common_cause_support.get(key,0)+(1 if common else -1)))
        self.common_cause_support[key]=support
        if support<2:return None
        occurrences=tuple(sorted(occurrences,key=lambda row:row.identity))
        network=bank.recruit(self.ecology.network_occurrence_features(occurrences))
        lineage=self._physical_lineage(rows)
        closure=ResidentEventClosureV1(self.next_ticket,episode,lineage,network.identity,
                                       tuple(row.identity for row in occurrences),
                                       tuple(sorted(by_channel)),network.tick)
        self.next_ticket+=1;self.closures[closure.ticket]=closure;self.common_cause_keys[closure.ticket]=key
        self.common_cause_control.observe_use(key,min(row.tick for row in occurrences),network.tick,0,key)
        return closure

    def issue_action(self,bank:'PopulationBankV1',ticket:int,action_id:int):
        closure=self.closures.get(int(ticket));action_id=int(action_id)
        if closure is None or closure.action_occurrence or action_id<=0:raise ValueError('population:event_action')
        action=bank.recruit((RESIDENT_ACTION_TAG,closure.ticket,action_id,
                             closure.network_occurrence,*closure.member_occurrences))
        closure.action_occurrence=action.identity;closure.action_id=action_id
        return action

    def reafference(self,bank:'PopulationBankV1',ticket:int,effect:int,source:int,
                    independent:bool=True):
        closure=self.closures.get(int(ticket));effect=int(effect);source=int(source)
        if closure is None or not closure.action_occurrence:raise ValueError('population:event_reafference_action')
        if source!=closure.source:raise ValueError('population:event_reafference_source')
        network=self._retained(bank,closure.network_occurrence)
        action=self._retained(bank,closure.action_occurrence)
        members=tuple(self._retained(bank,oid) for oid in closure.member_occurrences)
        if network is None or action is None or any(row is None for row in members):raise ValueError('population:event_reafference_ancestry')
        consequence=bank.recruit((RESIDENT_REAFFERENCE_TAG,closure.ticket,closure.action_id,
                                  effect,network.identity,action.identity))
        if consequence.tick<=action.tick:raise ValueError('population:event_reafference_order')
        identity=self.ecology.record_qualified_network(bank,network,members,source,effect,bool(independent))
        self.closures.pop(closure.ticket,None)
        return identity,consequence

    def reafference_common_cause(self,bank:'PopulationBankV1',ticket:int,effect:int):
        """Settle action evidence, but bind only above its background rate."""
        closure=self.closures.get(int(ticket));key=self.common_cause_keys.get(int(ticket))
        if closure is None or key is None:raise ValueError('population:event_common_cause')
        if not closure.action_occurrence:raise ValueError('population:event_reafference_action')
        network=self._retained(bank,closure.network_occurrence)
        action=self._retained(bank,closure.action_occurrence)
        members=tuple(self._retained(bank,oid) for oid in closure.member_occurrences)
        if network is None or action is None or any(row is None for row in members):raise ValueError('population:event_reafference_ancestry')
        consequence=bank.recruit((RESIDENT_REAFFERENCE_TAG,closure.ticket,closure.action_id,
                                  int(effect),network.identity,action.identity))
        if consequence.tick<=action.tick:raise ValueError('population:event_reafference_order')
        control=self.common_cause_control.observe_control(key,True,True,key)
        context=control.contexts[key]
        identity=0
        if context.background_attempts and self.common_cause_control.contextual_control_supported(key,key):
            identity=self.ecology.record_qualified_network(bank,network,members,closure.source,int(effect),True)
        self.closures.pop(closure.ticket,None);self.common_cause_keys.pop(closure.ticket,None)
        return identity,consequence

    def background_reafference_common_cause(self,bank:'PopulationBankV1',ticket:int,effect:int):
        """Retain an actually occurring consequence during a no-action opportunity."""
        closure=self.closures.get(int(ticket));key=self.common_cause_keys.get(int(ticket))
        if closure is None or key is None or closure.action_occurrence:raise ValueError('population:event_common_cause_background')
        network=self._retained(bank,closure.network_occurrence)
        if network is None:raise ValueError('population:event_reafference_ancestry')
        consequence=bank.recruit((RESIDENT_REAFFERENCE_TAG,closure.ticket,0,int(effect),network.identity,0))
        self.common_cause_control.observe_control(key,False,True,key)
        self.closures.pop(closure.ticket,None);self.common_cause_keys.pop(closure.ticket,None)
        return consequence

    def expire_common_cause(self,bank:'PopulationBankV1',ticket:int):
        """Settle elapsed action or background opportunity with no consequence."""
        closure=self.closures.get(int(ticket));key=self.common_cause_keys.get(int(ticket))
        if closure is None or key is None:raise ValueError('population:event_common_cause')
        if bank.tick-closure.opened_tick<=self.max_lag:raise ValueError('population:event_common_cause_live')
        row=self.common_cause_control.observe_control(key,bool(closure.action_occurrence),False,key)
        self.closures.pop(closure.ticket,None);self.common_cause_keys.pop(closure.ticket,None)
        return row

    def withdraw_source(self,source:int):
        source=int(source)
        self.pending=[row for row in self.pending if row.source!=source]
        removed={ticket for ticket,row in self.closures.items() if row.source==source}
        self.closures={ticket:row for ticket,row in self.closures.items() if row.source!=source}
        for ticket in removed:self.common_cause_keys.pop(ticket,None)

    def checkpoint(self):
        return {'schema':3,'max_lag':self.max_lag,'next_ticket':self.next_ticket,
                'common_cause_support':sorted(self.common_cause_support.items()),
                'common_cause_keys':sorted(self.common_cause_keys.items()),
                'common_cause_control':self.common_cause_control.checkpoint(),
                'pending':[row.__dict__ for row in self.pending],
                'closures':[row.__dict__ for row in sorted(self.closures.values(),key=lambda x:x.ticket)]}

    @classmethod
    def restore(cls,ecology:PopulationRecruitmentEcologyV1,data):
        if data.get('schema')!=3:raise ValueError('population:event_checkpoint')
        out=cls(ecology,int(data['max_lag']));out.next_ticket=int(data['next_ticket'])
        for raw in data.get('common_cause_support',()):
            if len(raw)!=2:raise ValueError('population:event_common_cause_checkpoint')
            key,support=map(int,raw)
            if key<=0 or key in out.common_cause_support or support < -2 or support > 2:
                raise ValueError('population:event_common_cause_checkpoint')
            out.common_cause_support[key]=support
        if len(out.common_cause_support)>128:raise ValueError('population:event_common_cause_checkpoint')
        out.pending=[ResidentPortParticipationV1(**{k:int(v) for k,v in row.items()}) for row in data.get('pending',())]
        for raw in data.get('closures',()):
            row=dict(raw);row['member_occurrences']=tuple(map(int,row['member_occurrences']));row['channels']=tuple(map(int,row['channels']))
            closure=ResidentEventClosureV1(**{k:(v if k in ('member_occurrences','channels') else int(v)) for k,v in row.items()})
            out.closures[closure.ticket]=closure
        for raw in data.get('common_cause_keys',()):
            if len(raw)!=2:raise ValueError('population:event_common_cause_checkpoint')
            ticket,key=map(int,raw)
            if ticket not in out.closures or ticket in out.common_cause_keys or key not in out.common_cause_support:
                raise ValueError('population:event_common_cause_checkpoint')
            out.common_cause_keys[ticket]=key
        out.common_cause_control=PredictiveCreditBankV1.restore(data.get('common_cause_control',{}))
        return out


class _EligibilityView:
    """Fixed-domain compatibility view over sparse absolute decay-epoch expiries."""
    def __init__(self,owner,edge:bool):self.owner=owner;self.edge=bool(edge)
    def __len__(self):return self.owner.allocated_edge_count if self.edge else self.owner.spec.site_count
    def _index(self,index):
        n=len(self);index=int(index)
        if index<0:index+=n
        if index<0 or index>=n:raise IndexError(index)
        return index
    def __getitem__(self,index):
        if isinstance(index,slice):return bytearray(self[i] for i in range(*index.indices(len(self))))
        return self.owner._eligibility_value(self._index(index),self.edge)
    def __setitem__(self,index,value):
        if isinstance(index,slice):raise TypeError('population:eligibility_slice')
        self.owner._set_eligibility(self._index(index),int(value),self.edge)
    def __iter__(self):
        for index in range(len(self)):yield self.owner._eligibility_value(index,self.edge)


class _SparseDefaultView:
    """Fixed-domain scalar view backed only by non-default causal values."""
    def __init__(self,owner,edge:bool,default:int,minimum:int,maximum:int,itemsize:int):
        self.owner=owner;self.edge=bool(edge);self.default=int(default);self.minimum=int(minimum);self.maximum=int(maximum);self.itemsize=int(itemsize)
    def __len__(self):return self.owner.allocated_edge_count if self.edge else self.owner.spec.site_count
    def _index(self,index):
        n=len(self);index=int(index)
        if index<0:index+=n
        if index<0 or index>=n:raise IndexError(index)
        return index
    def _store(self):return self.owner._edge_weight_values if self.edge else self.owner._support_values
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        return int(self._store().get(self._index(index),self.default))
    def __setitem__(self,index,value):
        if isinstance(index,slice):raise TypeError('population:sparse_scalar_slice')
        index=self._index(index);value=int(value)
        if value<self.minimum or value>self.maximum:raise ValueError('population:sparse_scalar')
        store=self._store()
        if value==self.default:store.pop(index,None)
        else:store[index]=value
    def __iter__(self):
        for index in range(len(self)):yield int(self._store().get(index,self.default))
    def tolist(self):return list(self)
    def nondefault_count(self):return len(self._store())


class _CompressedSupportStore:
    """Dict-compatible view used only after support leaves the raw-map fast path."""
    def __init__(self,plane):self.plane=plane
    def __len__(self):return len(self.plane.snapshot())
    def __iter__(self):return iter(self.plane.snapshot())
    def items(self):return self.plane.snapshot().items()
    def get(self,key,default=None):
        value=self.plane.get(int(key));return default if value==self.plane.spec.default else value
    def __getitem__(self,key):
        value=self.plane.get(int(key))
        if value==self.plane.spec.default:raise KeyError(key)
        return value
    def __setitem__(self,key,value):self.plane.set(int(key),int(value))
    def pop(self,key,default=None):
        key=int(key);value=self.plane.get(key)
        if value==self.plane.spec.default:return default
        self.plane.set(key,self.plane.spec.default);return value


class PopulationBankV1:
    """Independently stateful simple sites + fixed sparse incidence.

    Cold allocation is reported separately from materialized/touched state. Capability
    receipts count only sites with nonzero resident support and edges sourced from them.
    """
    def __init__(self,spec:PopulationSpecV1):
        spec.validate();self.spec=spec;self.tick=0;self.next_occurrence=1
        # Species defaults are procedural: support=0, edge_weight=1. Individual
        # causal deviations alone are materialized while fixed-domain indexing stays exact.
        self._support_values={};self._support_backing='sparse_map';self._support_plane=None
        self._edge_weight_values={}
        self.support=_SparseDefaultView(self,False,0,0,65535,2)
        self.edge_weight=_SparseDefaultView(self,True,1,-128,127,1)
        # Eligibility is pending causal state represented by sparse absolute expiry
        # in a decay-only epoch. Ordinary contacts may advance `tick` without aging it.
        self._decay_epoch=0
        self._site_eligibility_expiry={};self._edge_eligibility_expiry={}
        self._site_expiry_buckets={};self._edge_expiry_buckets={}
        self.eligibility=_EligibilityView(self,False);self.edge_eligibility=_EligibilityView(self,True)
        self.last_decay_touches=0
        # Canonical topology/territory are Species-derived and procedural.
        # Only individual deviations are materialized as sparse overrides.
        self._edge_target_overrides={};self._territory_overrides={}
        self.occurrences:list[PopulationOccurrenceV1]=[]
        self.credit_events=0;self.revision_events=0

    @staticmethod
    def _canonical_edge_target(spec:PopulationSpecV1,site:int,lane:int)->int:
        n=spec.site_count;site=int(site);lane=int(lane)
        # Fixed generic sparse topology. Most links are local-ish in identity;
        # a second mixed term supplies deterministic long-range incidence.
        delta=1+(mix64(site*0xD6E8FEB86659FD93+lane)%max(2,min(n-1,257)))
        target=(site+delta+mix64((lane+1)*0xA0761D6478BD642F+site)%(n))%n
        return (site+1)%n if target==site else target

    def set_edge_target(self,edge:int,target:int):
        edge=int(edge);target=int(target)
        if edge<0 or edge>=self.allocated_edge_count or target<0 or target>=self.spec.site_count:raise ValueError('population:edge_target')
        site,lane=divmod(edge,self.spec.fanout);canonical=self._canonical_edge_target(self.spec,site,lane)
        if target==canonical:self._edge_target_overrides.pop(edge,None)
        else:self._edge_target_overrides[edge]=target

    def edge_target_value(self,edge:int)->int:
        edge=int(edge)
        if edge<0 or edge>=self.allocated_edge_count:raise ValueError('population:edge_target')
        override=self._edge_target_overrides.get(edge)
        if override is not None:return int(override)
        site,lane=divmod(edge,self.spec.fanout)
        return self._canonical_edge_target(self.spec,site,lane)

    def set_territory(self,site:int,territory:int):
        site=int(site);territory=int(territory)
        if site<0 or site>=self.spec.site_count or territory<0 or territory>=self.spec.territory_count:raise ValueError('population:territory')
        canonical=site%self.spec.territory_count
        if territory==canonical:self._territory_overrides.pop(site,None)
        else:self._territory_overrides[site]=territory

    def territory_value(self,site:int)->int:
        site=int(site)
        if site<0 or site>=self.spec.site_count:raise ValueError('population:territory')
        return int(self._territory_overrides.get(site,site%self.spec.territory_count))

    @property
    def allocated_edge_count(self):return self.spec.site_count*self.spec.fanout

    def _eligibility_value(self,index:int,edge:bool)->int:
        expiry=(self._edge_eligibility_expiry if edge else self._site_eligibility_expiry).get(int(index),0)
        return max(0,int(expiry)-self._decay_epoch)

    def _set_eligibility(self,index:int,value:int,edge:bool):
        index=int(index);value=int(value)
        if value<0 or value>255:raise ValueError('population:eligibility')
        expiry_map=self._edge_eligibility_expiry if edge else self._site_eligibility_expiry
        buckets=self._edge_expiry_buckets if edge else self._site_expiry_buckets
        prior=expiry_map.pop(index,None)
        if prior is not None:
            bucket=buckets.get(int(prior))
            if bucket is not None:
                bucket.discard(index)
                if not bucket:buckets.pop(int(prior),None)
        if not value:return
        expiry=self._decay_epoch+value;expiry_map[index]=expiry
        buckets.setdefault(expiry,set()).add(index)

    def sparse_site_eligibility_count(self):return len(self._site_eligibility_expiry)
    def sparse_edge_eligibility_count(self):return len(self._edge_eligibility_expiry)
    def sparse_support_count(self):return len(self._support_values)
    @property
    def support_backing(self):return self._support_backing
    def migrate_support_backing(self,target:str,fail_after:int|None=None):
        allowed=('sparse_map','sorted_sparse_pages','bitmap_pages','dense_pages')
        if target not in allowed:raise ValueError('population:support_backing')
        if target==self._support_backing:return {'changed':False,'backing':target}
        before_digest=self.digest();before_checkpoint=self.checkpoint()
        old_store=self._support_values;old_plane=self._support_plane;old_backing=self._support_backing
        rows=dict(old_store.items())
        try:
            if target=='sparse_map':
                candidate={}
                if fail_after is not None and int(fail_after)==0:raise RuntimeError('population:injected_support_migration_fault')
                for n,(index,value) in enumerate(sorted(rows.items()),1):
                    candidate[int(index)]=int(value)
                    if fail_after is not None and n==int(fail_after):raise RuntimeError('population:injected_support_migration_fault')
                self._support_values=candidate;self._support_plane=None;self._support_backing='sparse_map'
            else:
                plane=PopulationPhysicalPlaneV1(PlaneSpecV1(self.spec.site_count,0,0,65535,2),target)
                if fail_after is not None and int(fail_after)==0:raise RuntimeError('population:injected_support_migration_fault')
                for n,(index,value) in enumerate(sorted(rows.items()),1):
                    plane.set(index,value)
                    if fail_after is not None and n==int(fail_after):raise RuntimeError('population:injected_support_migration_fault')
                self._support_values=_CompressedSupportStore(plane);self._support_plane=plane;self._support_backing=target
            if self.digest()!=before_digest or self.checkpoint()!=before_checkpoint:raise ValueError('population:support_migration_changed_state')
        except Exception:
            self._support_values=old_store;self._support_plane=old_plane;self._support_backing=old_backing
            if self.digest()!=before_digest or self.checkpoint()!=before_checkpoint:raise AssertionError('population:support_migration_not_atomic')
            raise
        return {'changed':True,'backing':self._support_backing,'rows':len(rows)}
    def sparse_edge_weight_count(self):return len(self._edge_weight_values)
    def sparse_support_items(self):return tuple(sorted(self._support_values.items()))
    def sparse_edge_weight_items(self):return tuple(sorted(self._edge_weight_values.items()))
    def numeric_allocation_bytes(self):return 0

    def feature_sites(self,feature:int)->tuple[int,...]:
        n=self.spec.site_count;out=[]
        for lane in range(self.spec.sites_per_feature):
            site=mix64(int(feature)^mix64(lane+1))%n
            if site not in out:out.append(site)
        return tuple(out)

    def signature(self,features)->tuple[int,...]:
        sites=[]
        for feature in tuple(features):
            sites.extend(self.feature_sites(int(feature)))
        # One sparse propagation step makes topology causally relevant.
        seed=tuple(dict.fromkeys(sites)); propagated=[]
        for site in seed:
            edge=site*self.spec.fanout
            propagated.append(self.edge_target_value(edge))
        return tuple(sorted(set((*seed,*propagated))))

    def prepare(self,features)->tuple[int,...]:
        """Materialize distributed resident support without minting an Occurrence.

        Preparation is capacity/matter, not current computation and not credit. It
        therefore changes resident support but creates no eligibility trace and no
        lifetime Occurrence row.
        """
        features=tuple(features);self.tick+=1;sites=self.signature(features)
        for site in sites:
            if self.support[site]<65535:self.support[site]+=1
        return sites

    def activate_signature(self,signature,retain:bool=False)->PopulationOccurrenceV1:
        """Fresh computation over already-prepared resident sites."""
        sites=tuple(sorted(set(map(int,signature))))
        if not sites or any(site<0 or site>=self.spec.site_count for site in sites):
            raise ValueError('population:signature')
        self.tick+=1;edges=[]
        for site in sites:
            if self.support[site]==0:raise ValueError('population:unprepared_signature')
            self.eligibility[site]=self.spec.eligibility_horizon
            base=site*self.spec.fanout
            for lane in range(self.spec.fanout):
                eid=base+lane;self.edge_eligibility[eid]=self.spec.eligibility_horizon;edges.append(eid)
        o=PopulationOccurrenceV1(self.next_occurrence,self.tick,sites,len(sites),self.spec.fanout)
        self.next_occurrence+=1
        if retain:self.occurrences.append(o)
        return o

    def activate(self,features,retain:bool=False)->PopulationOccurrenceV1:
        """Create one fresh current-computation Occurrence.

        `retain=False` is the ephemeral Network path: sites/eligibility become live
        for this computation but the Occurrence is not written into lifetime history.
        A caller that needs later causal settlement uses `retain=True`.
        """
        features=tuple(features);self.tick+=1;sites=self.signature(features);edges=[]
        for site in sites:
            self.eligibility[site]=self.spec.eligibility_horizon
            base=site*self.spec.fanout
            for lane in range(self.spec.fanout):
                eid=base+lane;self.edge_eligibility[eid]=self.spec.eligibility_horizon;edges.append(eid)
        o=PopulationOccurrenceV1(self.next_occurrence,self.tick,sites,len(features),self.spec.fanout)
        self.next_occurrence+=1
        if retain:self.occurrences.append(o)
        return o

    def recruit(self,features)->PopulationOccurrenceV1:
        features=tuple(features);sites=self.prepare(features)
        # `prepare` already advanced the developmental clock. Reuse that tick while
        # making the actual retained Occurrence and its live eligibility.
        edges=[]
        for site in sites:
            self.eligibility[site]=self.spec.eligibility_horizon
            base=site*self.spec.fanout
            for lane in range(self.spec.fanout):
                eid=base+lane;self.edge_eligibility[eid]=self.spec.eligibility_horizon;edges.append(eid)
        o=PopulationOccurrenceV1(self.next_occurrence,self.tick,sites,len(features),self.spec.fanout)
        self.next_occurrence+=1;self.occurrences.append(o);return o

    def decay(self):
        self.tick+=1;self._decay_epoch+=1
        site_ids=self._site_expiry_buckets.pop(self._decay_epoch,())
        edge_ids=self._edge_expiry_buckets.pop(self._decay_epoch,())
        for site in site_ids:
            if self._site_eligibility_expiry.get(site)==self._decay_epoch:self._site_eligibility_expiry.pop(site,None)
        for edge in edge_ids:
            if self._edge_eligibility_expiry.get(edge)==self._decay_epoch:self._edge_eligibility_expiry.pop(edge,None)
        self.last_decay_touches=len(site_ids)+len(edge_ids)

    def settle(self,occurrence:PopulationOccurrenceV1,effect:int,independent:bool):
        if occurrence not in self.occurrences: raise ValueError('population:occurrence')
        if not independent or effect==0:return {'credit':0,'revisions':0}
        direction=1 if effect>0 else -1;credit=0;revisions=0
        for site in occurrence.sites:
            if self.eligibility[site]:credit+=1
        for eid in occurrence.edges:
            if not self.edge_eligibility[eid]:continue
            prior=int(self.edge_weight[eid]);new=max(-127,min(127,prior+direction))
            if new!=prior:self.edge_weight[eid]=new;revisions+=1
        self.credit_events+=credit;self.revision_events+=revisions
        return {'credit':credit,'revisions':revisions}

    @staticmethod
    def overlap(a:tuple[int,...],b:tuple[int,...])->int:
        # Small sparse codes; set conversion is bounded by the active signature, not N.
        bs=set(b);return sum(x in bs for x in a)

    def retrieve(self,cue_features,memories):
        cue=self.signature(cue_features);scored=[]
        for mid,signature in memories:
            score=self.overlap(cue,signature)
            if score:scored.append((score,int(mid)))
        if not scored:return {'status':0,'winner':0,'score':0,'alternatives':0}
        scored.sort(key=lambda x:(-x[0],x[1]));peak=scored[0][0];winners=[x for x in scored if x[0]==peak]
        if len(winners)!=1:return {'status':2,'winner':0,'score':peak,'alternatives':len(winners)}
        return {'status':1,'winner':winners[0][1],'score':peak,'alternatives':1}

    def materialized_site_count(self):return len(self._support_values)
    def live_eligibility_count(self):return len(self._site_eligibility_expiry)
    def touched_incidence_count(self):return self.materialized_site_count()*self.spec.fanout

    def quantity_vector(self,current:PopulationOccurrenceV1|None=None,alternatives=0,horizon=0,trajectory=0):
        p=len(current.sites) if current else 0;edges=len(current.edges) if current else 0
        experienced=self.materialized_site_count()
        # R/I are physically resident prepared matter, analogous to silent as well
        # as currently firing neurons/synapses. P/F are the current/experienced hot subsets.
        return {'R':self.spec.site_count,'I':self.allocated_edge_count,'O':len(self.occurrences),
                'P':p,'E':self.live_eligibility_count(),'G':edges,'A':int(alternatives),
                'H':int(horizon),'T':edges,'F':experienced,'C':self.spec.site_count-experienced,
                'Y':int(trajectory),'resident_sites':self.spec.site_count,
                'resident_edges':self.allocated_edge_count}

    def checkpoint(self):
        # The Species/birth substrate is deterministic from spec. Persist only this
        # individual's deviations plus current/past causal participation. Lesions of
        # topology/territory remain causal state as sparse overrides.
        sites=sorted(set(self._support_values)|set(self._site_eligibility_expiry))
        edges=sorted(set(self._edge_weight_values)|set(self._edge_eligibility_expiry))
        target_overrides=sorted(self._edge_target_overrides.items())
        territory_overrides=sorted(self._territory_overrides.items())
        return {'schema':2,'spec':self.spec.__dict__,'tick':self.tick,'next_occurrence':self.next_occurrence,
                'sites':sites,'site_support':[int(self.support[i]) for i in sites],
                'site_eligibility':[int(self.eligibility[i]) for i in sites],
                'edges':edges,'edge_weight':[int(self.edge_weight[i]) for i in edges],
                'edge_eligibility':[int(self.edge_eligibility[i]) for i in edges],
                'edge_target_overrides':target_overrides,'territory_overrides':territory_overrides,
                'occurrences':[{'identity':o.identity,'tick':o.tick,'sites':o.sites,
                                'feature_count':o.feature_count} for o in self.occurrences],
                'credit_events':self.credit_events,'revision_events':self.revision_events}

    @classmethod
    def restore(cls,d):
        schema=int(d.get('schema',0))
        if schema not in (1,2):raise ValueError('population:checkpoint_schema')
        b=cls(PopulationSpecV1(**d['spec']));b.tick=int(d['tick']);b.next_occurrence=int(d['next_occurrence'])
        if schema==1:
            raw_support=tuple(map(int,d['support']));raw_weights=tuple(map(int,d['edge_weight']))
            raw_site_elig=tuple(map(int,d['eligibility']));raw_edge_elig=tuple(map(int,d['edge_eligibility']))
            if len(raw_support)!=b.spec.site_count or len(raw_site_elig)!=b.spec.site_count or len(raw_weights)!=b.allocated_edge_count or len(raw_edge_elig)!=b.allocated_edge_count:raise ValueError('population:checkpoint_schema1_shape')
            for site,value in enumerate(raw_support):
                if value:b.support[site]=value
            for edge,value in enumerate(raw_weights):
                if value!=1:b.edge_weight[edge]=value
            for site,value in enumerate(raw_site_elig):
                if value:b.eligibility[site]=value
            for edge,value in enumerate(raw_edge_elig):
                if value:b.edge_eligibility[edge]=value
            for edge,target in enumerate(map(int,d['edge_target'])):
                site,lane=divmod(edge,b.spec.fanout)
                if int(target)!=b._canonical_edge_target(b.spec,site,lane):b._edge_target_overrides[edge]=int(target)
            b._territory_overrides={site:int(value) for site,value in enumerate(map(int,d['territory'])) if int(value)!=site%b.spec.territory_count}
        else:
            sites=tuple(map(int,d.get('sites',())));support=tuple(map(int,d.get('site_support',())));elig=tuple(map(int,d.get('site_eligibility',())))
            if len(sites)!=len(support) or len(sites)!=len(elig) or len(set(sites))!=len(sites):raise ValueError('population:checkpoint_sites')
            for site,sup,eligible in zip(sites,support,elig):
                if site<0 or site>=b.spec.site_count or sup<0 or sup>65535 or eligible<0 or eligible>255:raise ValueError('population:checkpoint_sites')
                b.support[site]=sup;b.eligibility[site]=eligible
            edges=tuple(map(int,d.get('edges',())));weights=tuple(map(int,d.get('edge_weight',())));edge_elig=tuple(map(int,d.get('edge_eligibility',())))
            if len(edges)!=len(weights) or len(edges)!=len(edge_elig) or len(set(edges))!=len(edges):raise ValueError('population:checkpoint_edges')
            for edge,weight,eligible in zip(edges,weights,edge_elig):
                if edge<0 or edge>=b.allocated_edge_count or weight<-128 or weight>127 or eligible<0 or eligible>255:raise ValueError('population:checkpoint_edges')
                b.edge_weight[edge]=weight;b.edge_eligibility[edge]=eligible
            seen=set()
            for raw in d.get('edge_target_overrides',()):
                if len(raw)!=2:raise ValueError('population:checkpoint_topology')
                edge,target=map(int,raw)
                if edge in seen or edge<0 or edge>=b.allocated_edge_count or target<0 or target>=b.spec.site_count:raise ValueError('population:checkpoint_topology')
                seen.add(edge);b.set_edge_target(edge,target)
            seen.clear()
            for raw in d.get('territory_overrides',()):
                if len(raw)!=2:raise ValueError('population:checkpoint_territory')
                site,territory=map(int,raw)
                if site in seen or site<0 or site>=b.spec.site_count or territory<0 or territory>=b.spec.territory_count:raise ValueError('population:checkpoint_territory')
                seen.add(site);b.set_territory(site,territory)
        b.occurrences=[PopulationOccurrenceV1(int(o['identity']),int(o['tick']),tuple(o['sites']),int(o['feature_count']),b.spec.fanout) for o in d['occurrences']]
        b.credit_events=int(d['credit_events']);b.revision_events=int(d['revision_events']);return b

    def digest(self):
        h=hashlib.sha256();h.update(b'population-v1\0')
        site_eligibility=[(i,self._eligibility_value(i,False)) for i in sorted(self._site_eligibility_expiry)]
        edge_eligibility=[(i,self._eligibility_value(i,True)) for i in sorted(self._edge_eligibility_expiry)]
        h.update(json.dumps({'spec':self.spec.__dict__,'tick':self.tick,'next':self.next_occurrence,
                             'credit':self.credit_events,'revision':self.revision_events,
                             'support':sorted(self._support_values.items()),'edge_weight':sorted(self._edge_weight_values.items()),
                             'site_eligibility':site_eligibility,'edge_eligibility':edge_eligibility,
                             'edge_target_overrides':sorted(self._edge_target_overrides.items()),
                             'territory_overrides':sorted(self._territory_overrides.items())},sort_keys=True,separators=(',',':')).encode())
        return h.hexdigest()
