#!/usr/bin/env python3
"""Canonical embodied causal bridge over the post-legacy organism core.

This file used to sit on top of four historical developmental wrapper layers
(`frontier -> structure -> amplifier -> e2e`).  Their useful language/cognition
mechanisms have since moved into the one-Life MathematicalWorkbenchAdult and its
explicit curriculum tails.  Keeping that wrapper chain alive would create a second
training authority and make deletion of obsolete representations impossible.

The surviving organism responsibility is narrower and causal: one continuing body
may form a question from prediction mismatch, nominate an intervention, issue an
ordinary motor action, and let the ordinary motor-consequence path settle it.  The
small developmental clock below is history state used by the recursive self/culture
owner; it contains no language stages, parser state, or semantic authority.
"""
from __future__ import annotations

import hashlib

from reference_causal_attribution_ecology_v1 import Refuse
import reference_organism_v2_core as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
CONSOLIDATION_SCHEMA=3


class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    """One body/history owner with resident experiment origination and consequence credit."""

    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self._resident_causal_motor=None
        self._resident_causal_question_formation=None
        self._developmental_curriculum_tick=0
        self._developmental_stage=0
        self._developmental_history_digest=hashlib.sha256(b'canonical-embodied-life-v3').hexdigest()

    def _developmental_advance(self,stage,event_tag):
        """Advance one monotone life/history coordinate; never select semantic content."""
        self._developmental_curriculum_tick+=1
        self._developmental_stage=max(self._developmental_stage,max(0,int(stage)))
        raw=(self._developmental_history_digest+'|'+str(self._developmental_curriculum_tick)
             +'|'+str(self._developmental_stage)+'|'+str(event_tag)).encode()
        self._developmental_history_digest=hashlib.sha256(raw).hexdigest()
        return self._developmental_curriculum_tick

    def attach_resident_causal_motor(self,owner):
        """Attach one resident experiment owner without moving evidence authority."""
        required=('try_nominate_endogenous_motor_intervention','bind_motor_action',
                  'settle_motor_intervention','pending')
        if owner is not None and any(not hasattr(owner,name) for name in required):
            raise ValueError('organism:resident-causal-motor-owner')
        self._resident_causal_motor=owner
        return self

    @property
    def resident_causal_motor(self):
        return self._resident_causal_motor

    def attach_resident_causal_question_formation(self,owner):
        """Attach prediction-mismatch question formation to ordinary cognition."""
        required=('observe','select_question','bridge_for','field_status')
        if owner is not None and any(not hasattr(owner,name) for name in required):
            raise ValueError('organism:resident-causal-question-owner')
        self._resident_causal_question_formation=owner
        self._refresh_resident_causal_motor()
        return self

    def _refresh_resident_causal_motor(self):
        formation=self._resident_causal_question_formation
        if formation is None:return 0
        owner=self._resident_causal_motor
        if owner is not None:
            if owner.pending:return next(iter(owner.bindings),0)
            live=tuple(receipt for receipt in map(int,owner.bindings)
                       if formation.field_status(receipt)=='open')
            if live:return live[0] if len(live)==1 else 0
            self._resident_causal_motor=None
        receipt,_bid=formation.select_question(self)
        if receipt<=0:return 0
        self.attach_resident_causal_motor(formation.bridge_for(receipt,self))
        return int(receipt)

    def _background_exploration_bid(self):
        action=int(self._exploration_candidate())
        if action<=0 or self.world_state is None:return 0
        trials=int(self.exploration_trials.get((self.world_state,action),0))
        return max(1,(1<<16)//(2+trials))

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload);resident_match=None
        if int(kind)==CONTACT_MOTOR_CONSEQUENCE and payload and self._resident_causal_motor is not None:
            ticket=int(payload[0])
            matches=tuple(row for row in self._resident_causal_motor.pending.values()
                          if int(row.motor_ticket)==ticket)
            if len(matches)>1:raise ValueError('organism:resident-causal-motor-pending-ambiguity')
            resident_match=matches[0] if matches else None
        result=super().contact(kind,payload,source,authenticated,independent)
        if int(kind)==CONTACT_MOTOR_CONSEQUENCE and payload:
            ticket=int(payload[0]);motor=next((row for row in self.motor_actions
                                               if int(row.ticket)==ticket),None)
            if resident_match is not None:
                if motor is None:raise ValueError('organism:resident-causal-motor-missing-action')
                self._resident_causal_motor.settle_motor_intervention(resident_match.nomination,motor,self)
                result=dict(result);result['resident_causal_settlement']=int(resident_match.receipt)
            formation=self._resident_causal_question_formation
            if formation is not None and resident_match is None and motor is not None:
                try:
                    opened=formation.observe(motor,self)
                except Refuse as exc:
                    # Wondering is a selective causal-field observer. Ordinary motor
                    # episodes outside its four-action field are valid Life contact,
                    # not a fatal error and not evidence for the wondering owner.
                    if str(exc)!='wondering action field':raise
                    opened=()
                if opened:
                    result=dict(result)
                    result['resident_causal_questions_opened']=tuple(int(row.receipt) for row in opened)
            if formation is not None:self._refresh_resident_causal_motor()
        return result

    def _cognitive_tick(self):
        """Let a resident unresolved causal field outbid undirected exploration."""
        self._refresh_resident_causal_motor()
        owner=self._resident_causal_motor;formation=self._resident_causal_question_formation
        if owner is not None and not owner.pending:
            proposal=owner.try_nominate_endogenous_motor_intervention(self)
            if proposal is not None:
                receipt,nomination,expected=proposal
                if formation is not None:
                    selected,bid=formation.select_question(self)
                    if int(selected)!=int(receipt) or int(bid)<=self._background_exploration_bid():
                        proposal=None
                if proposal is not None:
                    action=self._issue_motor(int(expected))
                    if action is None:raise ValueError('organism:resident-causal-motor-issue')
                    owner.bind_motor_action(nomination,action,self)
                    return action
        return super()._cognitive_tick()

    def checkpoint(self):
        data=super().checkpoint()
        data['embodied_causal_bridge_v3']={
            'schema':CONSOLIDATION_SCHEMA,
            'curriculum_tick':int(self._developmental_curriculum_tick),
            'stage':int(self._developmental_stage),
            'history_digest':str(self._developmental_history_digest),
        }
        return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('embodied_causal_bridge_v3')
        if row is None:return out
        if int(row.get('schema',0))!=CONSOLIDATION_SCHEMA:
            raise ValueError('organism:embodied-causal-bridge-checkpoint')
        out._developmental_curriculum_tick=max(0,int(row.get('curriculum_tick',0)))
        out._developmental_stage=max(0,int(row.get('stage',0)))
        digest=str(row.get('history_digest',''))
        if len(digest)!=64:raise ValueError('organism:embodied-causal-history-digest')
        out._developmental_history_digest=digest
        out._resident_causal_motor=None;out._resident_causal_question_formation=None
        return out
