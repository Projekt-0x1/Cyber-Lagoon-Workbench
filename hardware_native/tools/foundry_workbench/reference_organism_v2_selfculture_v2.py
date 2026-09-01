#!/usr/bin/env python3
"""Sequence-qualified self-in-culture authority over the V1 integration.

The prior authority is preserved in reference_organism_v2_selfculture_v1.py. This layer
upgrades cultural persistence and transmission without changing the incumbent motor/world
firewalls: full procedures must earn one continuous lived execution, teacher withdrawal
removes unsupported hearsay, culturally repeated reasons are attributed to the actual
retransmitter, and missing embodied causal state fails closed rather than defaulting to
full resource/control.
"""
from __future__ import annotations

import reference_organism_v2_selfculture_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_self_culture_control_v2 import RecursiveSelfCultureControlV2, Q

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
SELF_CULTURE_V2_SCHEMA=1


class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_self_culture=RecursiveSelfCultureControlV2.restore(
            self.recursive_self_culture.checkpoint())

    def _culture_body_envelope(self,source=0,action=0):
        source=int(source or self.partner_source or 1)
        action=int(action or (min(self.affordances) if self.affordances else 1))
        if not hasattr(self,'developmental_causal_envelope'):
            raise RuntimeError('organism:self-culture-missing-causal-envelope')
        env=self.developmental_causal_envelope(source,action,0)
        if not isinstance(env,dict) or 'available_resource_q16' not in env:
            raise RuntimeError('organism:self-culture-invalid-causal-envelope')
        resource=max(0,min(Q,int(env['available_resource_q16'])))
        control=max(0,min(Q,int(env.get('effective_control_q16',env.get('control_q16',0)))))
        return resource,control

    def _culture_episode_from_motor(self,motor,source):
        """Admit causal self-memory and certify only one completed active execution."""
        if motor is None or not getattr(motor,'settled',False):
            return 0
        state_before=tuple(getattr(motor,'state_before',()) or ())
        context=self._culture_context(state_before)
        action=int(getattr(motor,'action_id',0));effect=int(getattr(motor,'effect',0))
        outcome=max(1,min(3,effect+2));resource,control=self._culture_body_envelope(source,action)
        social=int(self.partner_source if self.partner_present else 0)
        independent=bool(getattr(motor,'independent_consequence',False))
        tick=self._developmental_advance(max(1,int(self._developmental_stage)),'self-episode-v2')
        identity=self.recursive_self_culture.observe_episode(
            context,action,outcome,int(source),tick,resource,control,social,independent)
        active=self._culture_active_program
        if active and action>0:
            pid,cursor=map(int,active)
            program=next((x for x in self.recursive_self_culture._programs
                if int(x.identity)==pid),None)
            if program is not None and cursor<len(program.actions) and int(program.actions[cursor])==action:
                if effect>0 and cursor+1<len(program.actions):
                    self._culture_active_program=(pid,cursor+1)
                elif effect>0 and cursor+1==len(program.actions):
                    self.recursive_self_culture.confirm_program_execution(
                        pid,int(source),tick,independent)
                    self._culture_active_program=()
                else:
                    self._culture_active_program=()
        return identity

    def checkpoint(self):
        data=super().checkpoint()
        data['recursive_self_culture_v2']={'schema':SELF_CULTURE_V2_SCHEMA}
        return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data)
        marker=data.get('recursive_self_culture_v2')
        if marker is not None and int(marker.get('schema',0))!=SELF_CULTURE_V2_SCHEMA:
            raise ValueError('organism:self-culture-v2-checkpoint')
        out.recursive_self_culture=RecursiveSelfCultureControlV2.restore(
            out.recursive_self_culture.checkpoint())
        return out

    def _run_recursive_self_culture_ratchet(self):
        """Same-Adult falsifiers for simulation, metacognition and sequence-qualified culture."""
        checks={};owner=self.recursive_self_culture
        ctx=_culture_id('curriculum-context-v2',[23]);A=0xA101;B=0xB202;C=0xC303
        for source,action,outcome in ((8101,A,3),(8102,B,1),(8103,A,3),(8104,C,3)):
            tick=self._developmental_advance(10,'culture-v2-lived-episode')
            owner.observe_episode(ctx,action,outcome,source,tick,Q,Q//2,77,True)
        before=owner.episode_count;futures=owner.constructive_futures(ctx,64)
        checks['constructive_memory_recombines_without_evidence_creation']=(
            bool(futures) and any(len(x.actions)>=2 for x in futures)
            and all(int(x.authority)==0 for x in futures) and owner.episode_count==before)

        reason=owner.observe_shared_reason(0x5151,8201,ctx,Q//2,
            self._developmental_advance(10,'culture-v2-shared-reason'),True)
        decision=owner.begin_decision(ctx,(A,B),((A,Q//8),(B,Q//4)),Q,Q//2,(reason,),
            self._developmental_advance(10,'culture-v2-decision'))
        episodes_before=owner.episode_count
        review=owner.review_decision(decision,((A,Q//2),),Q,Q//2,0)
        checks['metacognition_changes_mind_without_external_feedback']=(
            bool(review.get('changed')) and int(review.get('selected',0))==A
            and owner.episode_count==episodes_before)

        known=(A,B,C)
        program=owner.compose_instruction(ctx,(A,C),(reason,),8301,ctx,
            self._developmental_advance(10,'culture-v2-first-trial'),known)
        checks['rapid_instruction_composes_novel_known_primitive_sequence']=program>0
        rejected=owner.compose_instruction(ctx,(A,0xDEAD,B),(reason,),8302,ctx,
            self._developmental_advance(10,'culture-v2-unknown-primitive'),known)
        checks['instruction_cannot_create_unknown_primitive']=rejected==0

        # Matching positive fragments alone cannot certify a cultural procedure.
        tick=self._developmental_advance(10,'culture-v2-fragment-a')
        owner.observe_episode(ctx,A,3,8311,tick,Q,Q,77,True)
        tick=self._developmental_advance(10,'culture-v2-fragment-c')
        owner.observe_episode(ctx,C,3,8311,tick,Q,Q,77,True)
        row=next((x for x in owner._programs if int(x.identity)==int(program)),None)
        checks['separate_matching_fragments_cannot_confirm_program']=(
            row is not None and 8311 not in row.confirmations)
        confirmed=owner.confirm_program_execution(program,8311,
            self._developmental_advance(10,'culture-v2-resident-execution-complete'),True)
        row=next((x for x in owner._programs if int(x.identity)==int(program)),None)
        checks['resident_completed_execution_confirms_program']=(
            bool(confirmed) and row is not None and 8311 in row.confirmations)

        high=owner.program_candidates(ctx,Q,Q,lambda _s:Q//2)
        low=owner.program_candidates(ctx,Q//8,Q//8,lambda _s:Q//2)
        checks['same_cultural_reason_is_body_state_sensitive']=(
            bool(high and low) and int(high[0]['score_q16'])>int(low[0]['score_q16']))
        reason_good=owner.program_candidates(ctx,Q,Q,lambda s:Q if int(s)==8201 else 0)
        reason_bad=owner.program_candidates(ctx,Q,Q,lambda s:-Q if int(s)==8201 else 0)
        checks['reason_weight_uses_reason_source_not_teacher_source']=(
            bool(reason_good and reason_bad)
            and int(reason_good[0]['score_q16'])>int(reason_bad[0]['score_q16']))

        packet=owner.transmit_program(program);peer=RecursiveSelfCultureControlV2()
        received=peer.receive_program(packet,8401,ctx,1,known)
        checks['cultural_reason_is_resourced_to_retransmitter']=(
            bool(packet) and int(packet[-1])==0 and received>0
            and peer.reason_count>0 and all(int(r.source)==8401 for r in peer._reasons))
        peer_packet=peer.transmit_program(received);third=RecursiveSelfCultureControlV2()
        received2=third.receive_program(peer_packet,8402,ctx,1,known)
        checks['cumulative_culture_is_retransmissible']=(received2>0 and int(peer_packet[-1])==0)

        owner.withdraw_source(8301)
        surviving=next((x for x in owner._programs if int(x.identity)==int(program)),None)
        checks['lived_confirmed_program_survives_teacher_withdrawal']=(
            surviving is not None and not surviving.teachers and 8311 in surviving.confirmations)
        post=owner.program_candidates(ctx,Q,Q,lambda _s:Q//2)
        checks['confirmed_program_remains_actionable_without_teacher']=(
            any(int(x['identity'])==int(program) for x in post))

        cp=self.checkpoint();restored=type(self).restore(cp)
        checks['recursive_self_culture_survives_checkpoint']=(
            restored.recursive_self_culture.checkpoint()==owner.checkpoint())
        checks['self_culture_uses_strict_embodied_envelope']=(
            type(self)._culture_body_envelope is ReferenceOrganismV2._culture_body_envelope)
        failed=sorted(k for k,v in checks.items() if not v)
        return checks,failed,len(futures)
