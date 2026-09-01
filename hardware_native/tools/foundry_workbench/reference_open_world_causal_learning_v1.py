#!/usr/bin/env python3
"""Bounded Adult-owned bridge from matched interventions to opaque causal relation evidence."""
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass
from reference_causal_attribution_ecology_v1 import CausalAttributionEcologyV1,Refuse
from reference_hierarchical_composition_v1 import _identity
from reference_grounded_causal_operator_v1 import GroundedCausalOperatorV1

MAX_CAUSAL_BINDINGS=64
MAX_LOCAL_SLOTS=256
MAX_TESTIMONY_SOURCES=64
MAX_TESTIMONY_CLAIMS=128
MAX_PENDING_TESTIMONY_GUIDANCE=256

@dataclass(frozen=True)
class CausalBindingV1:
    receipt:int
    causes:tuple[int,int]
    effect:int
    slots:tuple[int,int]

class OpenWorldCausalLearningV1:
    """Maps full Adult proposition identities onto the existing matched-intervention ecology."""
    def __init__(self):
        self.ecology=CausalAttributionEcologyV1()
        self.identity_to_slot={}
        self.slot_to_identity={}
        self.bindings={}
        # Sparse source-qualified claims are hypotheses, not intervention truth.
        # Accuracy is learned only where this Adult already has a resolved world
        # relation; a source may then provisionally teach a new relation.
        self.testimony_accuracy={}
        self.testimony_claims={}
        self.testimony_withdrawn=set()
        # A testimony-guided intervention must remember which authenticated claim
        # actually changed the attempted coalition until independent world evidence
        # resolves the relation.  This is causal provenance, not a transcript.
        self.pending_testimony_guidance={}
        self.grounding=GroundedCausalOperatorV1()

    @staticmethod
    def testimony_source(body_source):
        return _identity('adult-authenticated-body-source-v1',(str(body_source),))

    def _resolved_world_causes(self,effect):
        effect=int(effect)
        return tuple(sorted(set(
            int(cause) for _participants,row_effect,cause,resolved_effect,
            _receipt,_opened in self.current_resolutions()
            if int(row_effect)==effect and int(resolved_effect)==effect)))

    def testimony_reliable(self,source):
        source=int(source);correct,wrong=self.testimony_accuracy.get(source,(0,0))
        return source not in self.testimony_withdrawn and correct>wrong and correct>0

    def observe_testimony(self,cause,effect,source):
        """Assimilate one authenticated speaker claim without calling it world truth."""
        cause=int(cause);effect=int(effect);source=int(source)
        if min(cause,effect,source)<=0 or cause==effect:return False
        world=self._resolved_world_causes(effect)
        if world:
            if (source not in self.testimony_accuracy
                    and len(self.testimony_accuracy)>=MAX_TESTIMONY_SOURCES):return False
            correct,wrong=self.testimony_accuracy.get(source,(0,0))
            if len(world)==1 and world[0]==cause:correct+=1
            else:wrong+=1
            self.testimony_accuracy[source]=(correct,wrong)
            return True
        if not self.testimony_reliable(source):return False
        # One current claim per source/effect makes correction a revision rather
        # than an ever-growing transcript or vote history.
        key=(source,effect)
        if key not in self.testimony_claims and len(self.testimony_claims)>=MAX_TESTIMONY_CLAIMS:return False
        self.testimony_claims[key]=cause
        return True

    def observe_language_relation(self,adult,factor,left_binding,right_binding,
                                  body_credentials,claim_source=0):
        """Ground or assimilate one authenticated learned relation occurrence."""
        if len(tuple(body_credentials))!=3:return False
        body_source,sequence,commitment=body_credentials
        adult._authenticated_body_occurrence(body_source,sequence,commitment)
        # The route credentials authenticate the physical occurrence; the contact
        # source identifies who supplied its claim. Conflating them makes every
        # speaker on one terminal/body route share one truth history.
        claim_source=int(claim_source)
        source=self.testimony_source(claim_source if claim_source>0 else body_source)
        try:
            left=adult.leaf(left_binding.context,left_binding.atoms)
            right=adult.leaf(right_binding.context,right_binding.atoms)
        except (KeyError,RuntimeError):return False
        matches=[]
        for _causes,_effect,cause,effect,receipt,_opened in self.current_resolutions():
            if ((adult.leaf_equivalent(cause,left.identity) and adult.leaf_equivalent(effect,right.identity))
                    or (adult.leaf_equivalent(cause,right.identity) and adult.leaf_equivalent(effect,left.identity))):
                matches.append(int(receipt))
        if len(matches)==1:
            grounded=self.grounding.observe(
                factor,left.identity,right.identity,self,matches[0],source,adult)
            resolved=self.resolve(matches[0])
            orientation=self.grounding.orientation(int(factor))
            claimed=((right.identity,left.identity) if orientation>0
                     else (left.identity,right.identity) if orientation<0 else ())
            if grounded and resolved is not None and claimed==tuple(resolved):
                self.observe_testimony(resolved[0],resolved[1],source)
            return grounded
        orientation=self.grounding.orientation(int(factor))
        if not orientation:return False
        cause,effect=((right.identity,left.identity) if orientation>0
                      else (left.identity,right.identity))
        return self.observe_testimony(cause,effect,source)

    def language_relation_certificate(self,adult,operator_factor,left,right):
        """Rematerialize intervention or testimony provenance for public speech."""
        factor=int(operator_factor);left=int(left);right=int(right)
        orientation=self.grounding.orientation(factor)
        if not orientation:return ()
        for _causes,_effect,cause,effect,receipt,opened in self.current_resolutions():
            expected=((int(effect),int(cause)) if orientation>0
                      else (int(cause),int(effect)))
            if adult.leaf_equivalent(expected[0],left) and adult.leaf_equivalent(expected[1],right):
                return (1,factor,left,right,int(receipt),int(opened))
        for cause,effect,evidence,sources in self.current_testimony_resolutions():
            expected=((int(effect),int(cause)) if orientation>0
                      else (int(cause),int(effect)))
            if adult.leaf_equivalent(expected[0],left) and adult.leaf_equivalent(expected[1],right):
                return (4,factor,left,right,int(evidence),len(sources))
        return ()

    def current_testimony_resolutions(self):
        """Unique active provisional relation per effect, with exact provenance."""
        world_effects={int(effect) for _p,effect,_c,_e,_r,_o in self.current_resolutions()}
        by_effect={}
        for (source,effect),cause in self.testimony_claims.items():
            if effect in world_effects or not self.testimony_reliable(source):continue
            by_effect.setdefault(int(effect),{}).setdefault(int(cause),[]).append(int(source))
        out=[]
        for effect,causes in sorted(by_effect.items()):
            if len(causes)!=1:continue
            cause,sources=next(iter(causes.items()));sources=tuple(sorted(sources))
            digest=hashlib.sha256(b'CYBER_LAGOON_TESTIMONY_RELATION_V1\0')
            for value in (cause,effect,*sources):digest.update(int(value).to_bytes(16,'big'))
            evidence=int.from_bytes(digest.digest()[:8],'big') or 1
            out.append((cause,effect,evidence,sources))
        return tuple(out)

    def _slot(self,identity:int):
        identity=int(identity)
        if identity<=0:raise Refuse('causal proposition identity')
        prior=self.identity_to_slot.get(identity)
        if prior is not None:return prior
        if len(self.identity_to_slot)>=MAX_LOCAL_SLOTS:raise Refuse('causal local slot capacity')
        slot=len(self.identity_to_slot)+1
        self.identity_to_slot[identity]=slot;self.slot_to_identity[slot]=identity
        return slot

    def _prune_expired_bindings(self):
        """Retire evidence windows and every ownerless wrapper in one transaction."""
        self.ecology._prune_expired()
        live_receipts=set(self.ecology.pending)
        self.bindings={receipt:row for receipt,row in self.bindings.items()
                       if int(receipt) in live_receipts}
        live_nominations=set(self.ecology.nominations)
        self.pending_testimony_guidance={identity:row
            for identity,row in self.pending_testimony_guidance.items()
            if int(identity) in live_nominations}

    def participate(self,causes,effect,horizon=32):
        causes=tuple(map(int,causes));effect=int(effect)
        if len(causes)!=2 or causes[0]==causes[1] or effect<=0 or effect in causes:
            raise Refuse('causal binding shape')
        self._prune_expired_bindings()
        if len(self.bindings)>=MAX_CAUSAL_BINDINGS:raise Refuse('causal binding capacity')
        slots=tuple(self._slot(x) for x in causes)
        receipt=self.ecology.participate(slots,int(horizon))
        self.bindings[receipt]=CausalBindingV1(receipt,causes,effect,slots)
        return receipt

    def _guided_testimony(self,receipt):
        binding=self.bindings.get(int(receipt))
        if binding is None:return ()
        claims=[row for row in self.current_testimony_resolutions()
                if int(row[1])==binding.effect and int(row[0]) in binding.causes]
        if len(claims)!=1:return ()
        claimed=int(claims[0][0]);index=binding.causes.index(claimed)
        # Language changes which experiment is attempted, not its result: isolate
        # the claimed cause without the rival and let the world consequence decide.
        return ((int(binding.slots[index]),),claimed,tuple(map(int,claims[0][3])))

    def _guided_coalition(self,receipt):
        guided=self._guided_testimony(receipt)
        return () if not guided else guided[0]

    def nominate_intervention(self,receipt,horizon=32,source=0):
        receipt=int(receipt);source=int(source);row=self.ecology.pending.get(receipt)
        if row is None:raise Refuse('causal binding receipt')
        coalitions=((),(row.participants[0],),(row.participants[1],),tuple(row.participants))
        used={tuple(e.coalition) for e in row.evidence if e.active and source>0 and int(e.source)==source}
        guided=self._guided_testimony(receipt)
        if guided and guided[0] not in used:
            if len(self.pending_testimony_guidance)>=MAX_PENDING_TESTIMONY_GUIDANCE:
                live=set(self.ecology.nominations)
                self.pending_testimony_guidance={key:value for key,value in self.pending_testimony_guidance.items() if key in live}
            if len(self.pending_testimony_guidance)>=MAX_PENDING_TESTIMONY_GUIDANCE:
                raise Refuse('causal testimony guidance capacity')
            nomination,occurrence=self.ecology.nominate_intervention_at(receipt,guided[0],int(horizon))
            self.pending_testimony_guidance[int(nomination.identity)]={
                'receipt':receipt,'cause':int(guided[1]),'sources':tuple(guided[2]),'settled':0}
            return nomination,occurrence
        available=[coalition for coalition in coalitions if coalition not in used]
        if not available:raise Refuse('causal source coalition complete')
        counts={coalition:0 for coalition in coalitions}
        for evidence in row.evidence:
            if evidence.active:counts[tuple(evidence.coalition)]+=1
        coalition=min(available,key=lambda value:(counts[value],coalitions.index(value)))
        return self.ecology.nominate_intervention_at(receipt,coalition,int(horizon))

    def _reconcile_testimony_guidance(self,receipt,resolved):
        if resolved is None or self.complete_source_blocks(receipt)<3:return 0
        actual=int(resolved[0]);tested=set();settled=[]
        for identity,row in self.pending_testimony_guidance.items():
            if int(row['receipt'])==int(receipt) and int(row['settled']):
                claimed=int(row['cause']);settled.append(int(identity))
                tested.update((claimed,int(source)) for source in row['sources'])
        updates=0
        for claimed,source in sorted(tested):
            if source not in self.testimony_accuracy and len(self.testimony_accuracy)>=MAX_TESTIMONY_SOURCES:
                continue
            correct,wrong=self.testimony_accuracy.get(source,(0,0))
            if claimed==actual:correct+=1
            else:wrong+=1
            self.testimony_accuracy[source]=(correct,wrong);updates+=1
        for identity in settled:self.pending_testimony_guidance.pop(identity,None)
        return updates

    def settle_intervention(self,nomination,occurrence,source,effect,independent=True):
        identity=int(nomination.identity if hasattr(nomination,'identity') else nomination)
        result=self.ecology.settle_intervention(nomination,occurrence,int(source),int(effect),bool(independent))
        guidance=self.pending_testimony_guidance.get(identity)
        if guidance is not None:
            guidance['settled']=1
            receipt=int(guidance['receipt'])
            if self.complete_source_blocks(receipt)>=3:self.resolve(receipt)
        return result

    def resolve(self,receipt):
        receipt=int(receipt);binding=self.bindings.get(receipt)
        if binding is None:raise Refuse('causal binding receipt')
        result=self.ecology.resolve(receipt)
        if result is None:return None
        credit={int(slot):(int(n),int(d)) for slot,n,d in result.participant_credit}
        positive=[]
        for slot,cause in zip(binding.slots,binding.causes):
            n,d=credit.get(slot,(0,1))
            if d<=0:raise Refuse('causal credit denominator')
            if n>0:positive.append(cause)
        if len(positive)!=1:return None
        resolved=(int(positive[0]),int(binding.effect))
        self._reconcile_testimony_guidance(receipt,resolved)
        return resolved

    def complete_source_blocks(self,receipt):
        row=self.ecology.pending.get(int(receipt))
        if row is None:return 0
        required={(),(row.participants[0],),(row.participants[1],),tuple(row.participants)}
        by_source={}
        for evidence in row.evidence:
            if evidence.active:by_source.setdefault(int(evidence.source),set()).add(tuple(evidence.coalition))
        return sum(1 for values in by_source.values() if required<=values)

    def current_resolutions(self):
        """Latest complete resolved relation per opaque binding key; no new state."""
        latest={}
        for receipt,binding in self.bindings.items():
            row=self.ecology.pending.get(int(receipt))
            if row is None or row.result is None or self.complete_source_blocks(receipt)<3:
                continue
            credit={int(slot):(int(n),int(d)) for slot,n,d in row.result.participant_credit}
            positive=[]
            for slot,cause in zip(binding.slots,binding.causes):
                n,d=credit.get(int(slot),(0,1))
                if d>0 and n>0:positive.append(int(cause))
            if len(positive)!=1:continue
            key=(tuple(sorted(map(int,binding.causes))),int(binding.effect))
            candidate=(int(row.opened_tick),int(receipt),positive[0],int(binding.effect))
            prior=latest.get(key)
            if prior is None or candidate[:2]>prior[:2]:latest[key]=candidate
        return tuple((key[0],key[1],row[2],row[3],row[1],row[0])
                     for key,row in sorted(latest.items()))

    def current_open_fields(self):
        """Rematerialize evidence-bearing fields that still lack one current cause."""
        self._prune_expired_bindings()
        resolved={int(row[4]) for row in self.current_resolutions()}
        rows=[]
        for receipt,binding in self.bindings.items():
            occurrence=self.ecology.pending.get(int(receipt))
            if (occurrence is None or int(receipt) in resolved
                    or not any(evidence.active for evidence in occurrence.evidence)):
                continue
            rows.append((tuple(map(int,binding.causes)),int(binding.effect),
                         int(receipt),int(occurrence.opened_tick)))
        return tuple(sorted(rows,key=lambda row:(row[3],row[2])))

    def common_cause_certificate(self,left_effect,right_effect):
        """Exact current witness that two effects share one independently earned cause."""
        left_effect=int(left_effect);right_effect=int(right_effect)
        if min(left_effect,right_effect)<=0 or left_effect==right_effect:return ()
        rows=self.current_resolutions()
        left=tuple(row for row in rows if int(row[3])==left_effect)
        right=tuple(row for row in rows if int(row[3])==right_effect)
        left_by_cause={int(row[2]):row for row in left}
        right_by_cause={int(row[2]):row for row in right}
        shared=set(left_by_cause)&set(right_by_cause)
        if len(shared)!=1:return ()
        cause=next(iter(shared));left_row=left_by_cause[cause];right_row=right_by_cause[cause]
        return (1,cause,int(left_row[4]),int(right_row[4]),
                int(left_row[5]),int(right_row[5]))

    def preferred_factor(self,adult):
        candidates=[]
        for factor in self.grounding.rows:
            if not self.grounding.orientation(factor):continue
            support=0
            for (context,arity,pieces),sources in adult.language._span_sources.items():
                if adult.language.span_factor_identity(context,arity,pieces)==int(factor):
                    support=max(support,len(adult.language._active_sources(sources)))
            candidates.append((-support,int(factor)))
        return 0 if not candidates else min(candidates)[1]

    def materialize_program(self,adult,receipt,factor=0):
        factor=int(factor) or self.preferred_factor(adult)
        return None if factor<=0 else self.grounding.materialize(adult,self,int(receipt),factor)

    def withdraw_source(self,source):
        source=int(source);self.ecology.withdraw_source(source)
        self.testimony_withdrawn.add(source);self.grounding.withdraw_source(source)

    def checkpoint(self):
        self._prune_expired_bindings()
        live=set(self.ecology.nominations)
        guidance=[{'nomination':identity,'receipt':int(row['receipt']),'cause':int(row['cause']),
                   'sources':list(map(int,row['sources'])),'settled':int(row['settled'])}
                  for identity,row in sorted(self.pending_testimony_guidance.items()) if identity in live]
        return {'schema':4,'ecology':self.ecology.checkpoint(),
                'identity_to_slot':[[identity,slot] for identity,slot in sorted(self.identity_to_slot.items())],
                'bindings':[{'receipt':r,'causes':list(b.causes),'effect':b.effect,'slots':list(b.slots)} for r,b in sorted(self.bindings.items())],
                'testimony_accuracy':[{'source':source,'correct':row[0],'wrong':row[1]}
                                      for source,row in sorted(self.testimony_accuracy.items())],
                'testimony_claims':[{'source':source,'effect':effect,'cause':cause}
                                    for (source,effect),cause in sorted(self.testimony_claims.items())],
                'testimony_withdrawn':sorted(self.testimony_withdrawn),
                'pending_testimony_guidance':guidance,'grounding':self.grounding.checkpoint()}

    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2,3,4):raise Refuse('causal learning checkpoint')
        out=cls();out.ecology=CausalAttributionEcologyV1.restore(data.get('ecology',''))
        ids={};slots={}
        for row in data.get('identity_to_slot',()):
            identity,slot=map(int,row)
            if identity<=0 or not 0<slot<=MAX_LOCAL_SLOTS or identity in ids or slot in slots:raise Refuse('causal slot checkpoint')
            ids[identity]=slot;slots[slot]=identity
        bindings={}
        for row in data.get('bindings',()):
            receipt=int(row.get('receipt',0));causes=tuple(map(int,row.get('causes',())));effect=int(row.get('effect',0));local=tuple(map(int,row.get('slots',())))
            if (receipt<=0 or receipt in bindings or len(causes)!=2 or len(local)!=2 or causes[0]==causes[1]
                    or effect<=0 or effect in causes or any(ids.get(c)!=s for c,s in zip(causes,local))
                    or receipt not in out.ecology.pending):raise Refuse('causal binding checkpoint')
            bindings[receipt]=CausalBindingV1(receipt,causes,effect,local)
        out.identity_to_slot=ids;out.slot_to_identity=slots;out.bindings=bindings
        if schema>=2:
            for row in data.get('testimony_accuracy',()):
                source=int(row.get('source',0));correct=int(row.get('correct',0));wrong=int(row.get('wrong',0))
                if source<=0 or min(correct,wrong)<0 or source in out.testimony_accuracy:raise Refuse('causal testimony accuracy checkpoint')
                out.testimony_accuracy[source]=(correct,wrong)
            if len(out.testimony_accuracy)>MAX_TESTIMONY_SOURCES:raise Refuse('causal testimony accuracy capacity')
            for row in data.get('testimony_claims',()):
                source=int(row.get('source',0));effect=int(row.get('effect',0));cause=int(row.get('cause',0));key=(source,effect)
                if min(source,effect,cause)<=0 or cause==effect or key in out.testimony_claims:raise Refuse('causal testimony claim checkpoint')
                out.testimony_claims[key]=cause
            if len(out.testimony_claims)>MAX_TESTIMONY_CLAIMS:raise Refuse('causal testimony claim capacity')
            out.testimony_withdrawn=set(map(int,data.get('testimony_withdrawn',())))
            if any(source<=0 for source in out.testimony_withdrawn):raise Refuse('causal testimony withdrawn checkpoint')
        if schema>=3:out.grounding=GroundedCausalOperatorV1.restore(data.get('grounding',{'schema':1}))
        if schema>=4:
            for row in data.get('pending_testimony_guidance',()):
                identity=int(row.get('nomination',0));receipt=int(row.get('receipt',0));cause=int(row.get('cause',0))
                sources=tuple(map(int,row.get('sources',())));settled=int(row.get('settled',-1))
                nomination=out.ecology.nominations.get(identity);binding=out.bindings.get(receipt)
                if (identity<=0 or identity in out.pending_testimony_guidance or nomination is None
                        or nomination.receipt!=receipt or binding is None or cause not in binding.causes
                        or not sources or len(sources)!=len(set(sources)) or any(source<=0 for source in sources)
                        or settled not in (0,1) or int(nomination.settled)!=settled):
                    raise Refuse('causal testimony guidance checkpoint')
                out.pending_testimony_guidance[identity]={
                    'receipt':receipt,'cause':cause,'sources':sources,'settled':settled}
            if len(out.pending_testimony_guidance)>MAX_PENDING_TESTIMONY_GUIDANCE:
                raise Refuse('causal testimony guidance capacity')
        return out
