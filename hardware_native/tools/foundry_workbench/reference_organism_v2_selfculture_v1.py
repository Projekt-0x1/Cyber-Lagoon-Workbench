#!/usr/bin/env python3
"""Recursive self-in-culture authority over the consolidated developmental Adult.

The prior Adult is preserved in reference_organism_v2_consolidation_v1.py. This layer
integrates constructive episodic futures, metacognitive revision, rapid instruction
composition, shared reasons and cultural procedure transmission into ordinary motor life.
There is no executive/homunculus: all new state is revisable, source-qualified and
body/allostasis-conditioned; simulation and culture packets have epistemic authority zero.
"""
from __future__ import annotations
import hashlib, json

import reference_organism_v2_consolidation_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_self_culture_control_v1 import RecursiveSelfCultureControlV1, Q

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
SELF_CULTURE_SCHEMA=1


def _culture_id(tag,payload):
    raw=json.dumps([tag,payload],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    """Same Adult plus recursive self-monitoring and culturally compositional control."""
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_self_culture=RecursiveSelfCultureControlV1()
        self._culture_active_program=()  # (program identity,cursor)
        self._culture_last_decision=0

    def _culture_context(self,state=None):
        state=tuple(self.world_state if state is None and self.world_state is not None else (state or ()))
        target=tuple(self.body_target or ())
        return _culture_id('adult-context-v1',[list(state),list(target)])

    def _culture_body_envelope(self,source=0,action=0):
        source=int(source or self.partner_source or 1);action=int(action or (min(self.affordances) if self.affordances else 1))
        try:env=self.developmental_causal_envelope(source,action,0)
        except Exception:env={}
        return (int(env.get('available_resource_q16',Q)),int(env.get('effective_control_q16',env.get('control_q16',Q//2))))

    def observe_shared_reason(self,proposition,source,joint_context=None,support_q16=Q//2,independent=True):
        context=int(joint_context or self._culture_context())
        tick=self._developmental_advance(max(1,int(self._developmental_stage)),'shared-reason')
        return self.recursive_self_culture.observe_shared_reason(int(proposition),int(source),context,int(support_q16),tick,bool(independent))

    def compose_cultural_instruction(self,actions,source,reason_ids=(),joint_context=None):
        """Compose a first-trial program only from currently known/afforded primitives."""
        context=self._culture_context();joint=int(joint_context or context)
        tick=self._developmental_advance(max(1,int(self._developmental_stage)),'cultural-instruction')
        return self.recursive_self_culture.compose_instruction(context,tuple(map(int,actions)),tuple(map(int,reason_ids)),int(source),joint,tick,tuple(sorted(self.affordances)))

    def export_cultural_program(self,identity):
        return self.recursive_self_culture.transmit_program(int(identity))

    def import_cultural_program(self,packet,source,joint_context=None):
        joint=int(joint_context or self._culture_context());tick=self._developmental_advance(max(1,int(self._developmental_stage)),'cultural-transmission')
        return self.recursive_self_culture.receive_program(packet,int(source),joint,tick,tuple(sorted(self.affordances)))

    def constructive_self_futures(self,max_futures=64):
        return self.recursive_self_culture.constructive_futures(self._culture_context(),int(max_futures))

    def _culture_episode_from_motor(self,motor,source):
        if motor is None or not getattr(motor,'settled',False):return 0
        state_before=tuple(getattr(motor,'state_before',()) or ())
        context=self._culture_context(state_before)
        action=int(getattr(motor,'action_id',0));effect=int(getattr(motor,'effect',0));outcome=max(1,min(3,effect+2))
        resource,control=self._culture_body_envelope(source,action)
        social=int(self.partner_source if self.partner_present else 0)
        tick=self._developmental_advance(max(1,int(self._developmental_stage)),'self-episode')
        identity=self.recursive_self_culture.observe_episode(context,action,outcome,int(source),tick,resource,control,social,True)
        active=self._culture_active_program
        if active and action>0:
            pid,cursor=map(int,active);program=next((x for x in self.recursive_self_culture._programs if x.identity==pid),None)
            if program is not None and cursor<len(program.actions) and int(program.actions[cursor])==action:
                if effect>0 and cursor+1<len(program.actions):self._culture_active_program=(pid,cursor+1)
                else:self._culture_active_program=()
        return identity

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(int(x) for x in payload);result=super().contact(kind,payload,source,authenticated,independent)
        if int(kind)==CONTACT_MOTOR_CONSEQUENCE and payload:
            ticket=int(payload[0]);motor=next((x for x in self.motor_actions if int(x.ticket)==ticket),None)
            if motor is not None:self._culture_episode_from_motor(motor,int(source))
        elif int(kind)==CONTACT_WITHDRAW_SOURCE and len(payload)==1:
            self.recursive_self_culture.withdraw_source(int(payload[0]))
            if self._culture_active_program:
                pid=int(self._culture_active_program[0]);
                if not any(x.identity==pid for x in self.recursive_self_culture._programs):self._culture_active_program=()
        return result

    def _culture_internal_evidence(self,alternatives):
        alternatives=set(map(int,alternatives));score={a:0 for a in alternatives}
        for future in self.constructive_self_futures(128):
            if not future.actions or int(future.actions[0]) not in score:continue
            # Constructive memory supplies defeasible internal evidence, not truth.
            for outcome in future.predicted_outcomes[:1]:
                if int(outcome)==3:score[int(future.actions[0])]+=Q//8
                elif int(outcome)==1:score[int(future.actions[0])]-=Q//8
        return tuple(sorted(score.items()))

    def _culture_program_nomination(self):
        if not self.affordances:return None
        context=self._culture_context();resource,control=self._culture_body_envelope()
        rows=self.recursive_self_culture.program_candidates(context,resource,control,self.developmental_source_credibility_q16)
        if not rows:return None
        # Continue a currently instructed sequence before selecting a competing program.
        if self._culture_active_program:
            pid,cursor=map(int,self._culture_active_program);row=next((x for x in rows if int(x['identity'])==pid),None)
            if row is not None and cursor<len(row['actions']) and int(row['actions'][cursor]) in self.affordances:return int(row['actions'][cursor])
            self._culture_active_program=()
        peak=int(rows[0]['score_q16']);winners=[x for x in rows if int(x['score_q16'])==peak]
        if peak<=0:return None
        actions=tuple(sorted(set(int(x['actions'][0]) for x in winners if x['actions'] and int(x['actions'][0]) in self.affordances)))
        if not actions:return None
        evidence=[]
        for action in actions:
            evidence.append((action,max(int(x['score_q16']) for x in winners if x['actions'] and int(x['actions'][0])==action)))
        decision=self.recursive_self_culture.begin_decision(context,actions,evidence,resource,control,tuple(r for x in winners for r in x.get('reason_ids',())),int(self._developmental_curriculum_tick))
        self._culture_last_decision=int(decision)
        selected=actions[0] if len(actions)==1 else 0
        if decision:
            review=self.recursive_self_culture.review_decision(decision,self._culture_internal_evidence(actions),resource,control,0)
            if int(review.get('selected',0)) in actions:selected=int(review['selected'])
        if selected<=0:return None
        chosen=next((x for x in winners if x['actions'] and int(x['actions'][0])==selected),None)
        if chosen is not None:self._culture_active_program=(int(chosen['identity']),0)
        return selected

    def _cognitive_tick(self):
        """Incumbent lived cognition first; culture may compose known primitives only if idle."""
        action=super()._cognitive_tick()
        if action is not None or self.information_need:return action
        nominated=self._culture_program_nomination()
        if nominated is None:return None
        # RITL-like first-trial execution: instruction composes known motor primitives;
        # actual motor issuance still passes the organism's incumbent affordance path.
        return self._issue_motor(int(nominated))

    def checkpoint(self):
        data=super().checkpoint();data['recursive_self_culture_v1']={'schema':SELF_CULTURE_SCHEMA,'state':self.recursive_self_culture.checkpoint(),'active_program':list(map(int,self._culture_active_program)),'last_decision':int(self._culture_last_decision)};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('recursive_self_culture_v1')
        if row is None:return out
        if int(row.get('schema',0))!=SELF_CULTURE_SCHEMA:raise ValueError('organism:self-culture-checkpoint')
        out.recursive_self_culture=RecursiveSelfCultureControlV1.restore(row['state']);out._culture_active_program=tuple(map(int,row.get('active_program',())));out._culture_last_decision=int(row.get('last_decision',0));return out

    def _run_recursive_self_culture_ratchet(self):
        """Shared curriculum ratchet over self-simulation, metacognition, instruction and culture."""
        checks={};owner=self.recursive_self_culture;ctx=_culture_id('curriculum-context',[17]);A=0xA101;B=0xB202;C=0xC303
        # Use a separate observer context so the assay never depends on current world bytes.
        for source,action,outcome in ((8101,A,3),(8102,B,1),(8103,A,3),(8104,C,3)):
            tick=self._developmental_advance(10,'culture-lived-episode');owner.observe_episode(ctx,action,outcome,source,tick,Q,Q//2,77,True)
        before=owner.episode_count;futures=owner.constructive_futures(ctx,64)
        checks['constructive_memory_recombines_without_evidence_creation']=bool(futures) and any(len(x.actions)>=2 for x in futures) and all(int(x.authority)==0 for x in futures) and owner.episode_count==before
        reason=owner.observe_shared_reason(0x5151,8201,ctx,Q//2,self._developmental_advance(10,'culture-shared-reason'),True)
        decision=owner.begin_decision(ctx,(A,B),((A,Q//8),(B,Q//4)),Q,Q//2,(reason,),self._developmental_advance(10,'culture-decision'))
        episodes_before=owner.episode_count;review=owner.review_decision(decision,((A,Q//2),),Q,Q//2,0)
        checks['metacognition_changes_mind_without_external_feedback']=bool(review.get('changed')) and int(review.get('selected',0))==A and owner.episode_count==episodes_before
        known=(A,B,C);program=owner.compose_instruction(ctx,(A,C,B),(reason,),8301,ctx,self._developmental_advance(10,'culture-first-trial-instruction'),known)
        checks['rapid_instruction_composes_novel_known_primitive_sequence']=program>0
        rejected=owner.compose_instruction(ctx,(A,0xDEAD,B),(reason,),8302,ctx,self._developmental_advance(10,'culture-unknown-primitive'),known)
        checks['instruction_cannot_create_unknown_primitive']=rejected==0
        high=owner.program_candidates(ctx,Q,Q,lambda _s:Q//2);low=owner.program_candidates(ctx,Q//8,Q//8,lambda _s:Q//2)
        checks['same_cultural_reason_is_body_state_sensitive']=bool(high and low) and int(high[0]['score_q16'])>int(low[0]['score_q16'])
        packet=owner.transmit_program(program);peer=RecursiveSelfCultureControlV1();received=peer.receive_program(packet,8401,ctx,1,known)
        checks['cultural_procedure_transmits_without_authority']=bool(packet) and int(packet[-1])==0 and received>0
        peer_packet=peer.transmit_program(received);third=RecursiveSelfCultureControlV1();received2=third.receive_program(peer_packet,8402,ctx,1,known)
        checks['cumulative_culture_is_retransmissible']=received2>0 and int(peer_packet[-1])==0
        cp=self.checkpoint();restored=type(self).restore(cp);checks['recursive_self_culture_survives_checkpoint']=restored.recursive_self_culture.checkpoint()==owner.checkpoint()
        failed=sorted(k for k,v in checks.items() if not v);return checks,failed,len(futures)

    def run_frontier_multilingual_curriculum(self):
        receipt=super().run_frontier_multilingual_curriculum();checks,local_failed,future_count=self._run_recursive_self_culture_ratchet();merged=dict(receipt.get('checks',{}));merged.update(checks);failed=sorted(k for k,v in merged.items() if not v);receipt=dict(receipt);receipt['checks']=merged;receipt['failed']=failed;receipt['status']='GREEN' if not failed else 'RED';receipt['recursive_self_culture_future_hypotheses']=future_count;receipt['recursive_self_culture_programs']=self.recursive_self_culture.program_count;receipt['recursive_self_culture_decisions']=self.recursive_self_culture.decision_count;receipt['constructive_future_authority']=0;receipt['cultural_packet_authority']=0;receipt['self_culture_runtime_verification_required']=True;return receipt
