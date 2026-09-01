#!/usr/bin/env python3
"""Execution-evidence-qualified authority over the sequence-cultural Adult.

V2 requires one contiguous active-program execution for cultural confirmation. V3 tightens
that boundary further: every step in the execution must carry independent consequence
evidence from one consistent consequence source, and any unrelated/mismatched motor action
invalidates the in-flight execution certificate. The V2 authority is preserved verbatim in
reference_organism_v2_selfculture_v2.py.
"""
from __future__ import annotations

import reference_organism_v2_selfculture_v2 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
SELF_CULTURE_V3_SCHEMA=1


class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self._culture_execution_all_independent=True
        self._culture_execution_source=0

    def _culture_reset_execution_evidence(self):
        self._culture_execution_all_independent=True
        self._culture_execution_source=0

    def _culture_accumulate_execution_evidence(self,source,independent):
        source=int(source)
        if self._culture_execution_source==0:
            self._culture_execution_source=source
        elif self._culture_execution_source!=source:
            self._culture_execution_all_independent=False
        self._culture_execution_all_independent=(
            bool(self._culture_execution_all_independent) and bool(independent))
        return bool(self._culture_execution_all_independent)

    def _culture_program_nomination(self):
        before=tuple(self._culture_active_program)
        action=super()._culture_program_nomination()
        after=tuple(self._culture_active_program)
        if after and (not before or after[0]!=before[0]):
            self._culture_reset_execution_evidence()
        elif not after:
            self._culture_reset_execution_evidence()
        return action

    def _culture_episode_from_motor(self,motor,source):
        """Admit self-memory; certify culture only from one fully independent execution."""
        if motor is None or not getattr(motor,'settled',False):
            return 0
        state_before=tuple(getattr(motor,'state_before',()) or ())
        context=self._culture_context(state_before)
        action=int(getattr(motor,'action_id',0));effect=int(getattr(motor,'effect',0))
        outcome=max(1,min(3,effect+2));resource,control=self._culture_body_envelope(source,action)
        social=int(self.partner_source if self.partner_present else 0)
        independent=bool(getattr(motor,'independent_consequence',False))
        tick=self._developmental_advance(max(1,int(self._developmental_stage)),'self-episode-v3')
        identity=self.recursive_self_culture.observe_episode(
            context,action,outcome,int(source),tick,resource,control,social,independent)

        active=tuple(self._culture_active_program)
        if not active:
            self._culture_reset_execution_evidence()
            return identity
        pid,cursor=map(int,active)
        program=next((x for x in self.recursive_self_culture._programs
            if int(x.identity)==pid),None)
        if program is None or cursor>=len(program.actions):
            self._culture_active_program=();self._culture_reset_execution_evidence();return identity
        if int(program.actions[cursor])!=action:
            # An unrelated or out-of-order motor breaks execution continuity.
            self._culture_active_program=();self._culture_reset_execution_evidence();return identity

        self._culture_accumulate_execution_evidence(int(source),independent)
        if effect<=0:
            self._culture_active_program=();self._culture_reset_execution_evidence();return identity
        if cursor+1<len(program.actions):
            self._culture_active_program=(pid,cursor+1)
            return identity

        source_id=int(self._culture_execution_source)
        qualified=bool(self._culture_execution_all_independent and source_id>0)
        self.recursive_self_culture.confirm_program_execution(pid,source_id,tick,qualified)
        self._culture_active_program=();self._culture_reset_execution_evidence()
        return identity

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        result=super().contact(kind,payload,source,authenticated,independent)
        if not self._culture_active_program:
            self._culture_reset_execution_evidence()
        return result

    def checkpoint(self):
        data=super().checkpoint()
        data['recursive_self_culture_v3']={
            'schema':SELF_CULTURE_V3_SCHEMA,
            'execution_all_independent':bool(self._culture_execution_all_independent),
            'execution_source':int(self._culture_execution_source),
        }
        return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data)
        row=data.get('recursive_self_culture_v3')
        if row is None:
            out._culture_reset_execution_evidence();return out
        if int(row.get('schema',0))!=SELF_CULTURE_V3_SCHEMA:
            raise ValueError('organism:self-culture-v3-checkpoint')
        out._culture_execution_all_independent=bool(row.get('execution_all_independent',True))
        out._culture_execution_source=int(row.get('execution_source',0))
        if not out._culture_active_program:
            out._culture_reset_execution_evidence()
        return out

    def _run_recursive_self_culture_ratchet(self):
        checks,local_failed,future_count=super()._run_recursive_self_culture_ratchet()
        checks=dict(checks)
        self._culture_reset_execution_evidence()
        checks['execution_evidence_requires_every_step_independent']=(
            self._culture_accumulate_execution_evidence(9101,True)
            and not self._culture_accumulate_execution_evidence(9101,False))
        self._culture_reset_execution_evidence()
        first=self._culture_accumulate_execution_evidence(9101,True)
        second=self._culture_accumulate_execution_evidence(9102,True)
        checks['execution_evidence_requires_consistent_consequence_source']=(first and not second)
        self._culture_reset_execution_evidence()
        cp=self.checkpoint();restored=type(self).restore(cp)
        checks['execution_evidence_checkpoint_contract']=(
            restored._culture_execution_all_independent
            and restored._culture_execution_source==0)
        failed=sorted(k for k,v in checks.items() if not v)
        return checks,failed,future_count
