#!/usr/bin/env python3
"""Reference-only language nomination into ordinary resident action arbitration.

Language contributes a prospective candidate identity. Ordinary lived transition/
consequence evidence determines whether that candidate, another action, or no action
is selected. Nomination is never evidence, credit, or motor publication.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json


def _identity(tag,values):
    raw=json.dumps([tag,*map(int,values)],separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


@dataclass(frozen=True)
class LanguageActionNominationV1:
    identity:int
    action_identity:int
    teaching_occurrence_identity:int
    current_parent_occurrence_identity:int
    current_parent_revision_identity:int
    current_participation_identity:int
    context_signature:int
    tick:int
    lineage:int=1       # endogenous
    authority:int=0     # none


@dataclass(frozen=True)
class ResidentActionDecisionV1:
    status:int          # 0 none/refused, 1 unique commit, 2 unresolved
    action_identity:int
    nomination_identity:int
    evidence_support:int
    effect:int
    alternatives:int
    resource_veto:int=0


class LanguageActionNominationBankV1:
    def __init__(self):
        self._current:dict[int,LanguageActionNominationV1]={}
        self.nominations=0
        self.commits=0
        self.refusals=0
        self.unresolved=0

    def nominate(self,action_identity,teaching_occurrence_identity,current_parent_occurrence_identity,
                 current_parent_revision_identity,current_participation_identity,context_signature,tick):
        vals=tuple(map(int,(action_identity,teaching_occurrence_identity,current_parent_occurrence_identity,
                            current_parent_revision_identity,current_participation_identity,context_signature,tick)))
        if min(vals)<=0:raise ValueError('language_nomination:identity')
        action,teaching,parent,revision,participation,context,tick=vals
        ident=_identity('language-action-nomination-v2',vals)
        row=LanguageActionNominationV1(ident,action,teaching,parent,revision,participation,context,tick)
        self._current[action]=row;self.nominations+=1
        return row

    def clear(self):
        self._current.clear()

    def arbitrate(self,ecology,current_state,goal,ordinary_candidates=(),available_resource=1,
                  action_costs=None,control_bank=None):
        """Select from language + ordinary candidate identities using ecology only.

        Candidate generation does not mutate ecology. Scores are derived only from
        already-lived transition evidence. Equal best candidates fail closed.
        """
        state=tuple(current_state);goal=tuple(goal)
        candidates=set(map(int,ordinary_candidates));candidates.update(self._current)
        action_costs={} if action_costs is None else {int(k):int(v) for k,v in action_costs.items()}
        viable=[]
        for action in sorted(candidates):
            edge=ecology.transition(state,action,1)
            if edge is None or tuple(edge.next_state)!=goal:continue
            cost=max(0,action_costs.get(action,0))
            if cost>int(available_resource):continue
            # No language bonus. Current consequence/support stay primary. A slower
            # program-local controllability history may separate otherwise matched
            # candidates, but it is read without creating a row/evidence. This lets
            # lived history affect control while current semantic identity stays fixed.
            control_row=None if control_bank is None else control_bank.rows.get(action)
            control_history=0 if control_row is None else int(control_row.control_history_q16)
            viable.append((int(edge.effect),int(edge.support),control_history,action,edge))
        if not viable:
            self.refusals+=1;return ResidentActionDecisionV1(0,0,0,0,0,0,
                int(any(action_costs.get(a,0)>int(available_resource) for a in candidates)))
        best_key=max((effect,support,history) for effect,support,history,_a,_e in viable)
        best=[row for row in viable if row[:3]==best_key]
        if len(best)!=1:
            self.unresolved+=1;return ResidentActionDecisionV1(2,0,0,best_key[1],best_key[0],len(best))
        effect,support,_history,action,_edge=best[0]
        nomination=self._current.get(action)
        self.commits+=1
        return ResidentActionDecisionV1(1,action,0 if nomination is None else nomination.identity,
                                        support,effect,len(viable))

    def checkpoint(self):
        # Current nominations are prospective occurrences, not durable learned state.
        # Persist only aggregate diagnostics; restored Adult rematerializes candidates
        # from learned state + new current situation.
        return {'schema':1,'nominations':self.nominations,'commits':self.commits,
                'refusals':self.refusals,'unresolved':self.unresolved}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('language_nomination:checkpoint')
        x=cls();x.nominations=int(data.get('nominations',0));x.commits=int(data.get('commits',0));x.refusals=int(data.get('refusals',0));x.unresolved=int(data.get('unresolved',0))
        if min(x.nominations,x.commits,x.refusals,x.unresolved)<0:raise ValueError('language_nomination:checkpoint_counter')
        return x
