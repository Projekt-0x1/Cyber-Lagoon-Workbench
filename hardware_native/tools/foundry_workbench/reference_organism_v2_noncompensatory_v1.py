#!/usr/bin/env python3
"""Category-safe public Adult over typed world/control/access/recommendation relations.

The full typed-context implementation is preserved in reference_organism_v2_category_split_v1.py.
Current control removes cross-category scalar compensation: program support, simulation,
source quality, self competence, risk, epistemic gap, resource and controllability retain
separate causal roles through commitment and experiment selection.
"""
from __future__ import annotations

import reference_organism_v2_category_split_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

import reference_organism_v2_metacontrol_v1 as _metacontrol_lifecycle
from reference_recursive_context_partition_v2 import RecursiveContextPartitionV2,WORLD_CONTEXT,CONTROL_CONTEXT,Q as CTXQ
from reference_recursive_contextual_owners_v1 import ContextualMetacontrolV1,ContextualCausalExperimentV1
from reference_recursive_partner_access_v2 import RecursivePartnerAccessV2
from reference_reason_action_recommendation_v1 import ReasonActionRecommendationV1

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def infer_causal_regime(self,*_args,**_kwargs):
        raise RuntimeError('organism:category-error-use-world-context-or-control-context')

    def _experiment_reason_map(self,rows):
        """Map cultural reasons to source-qualified action recommendations, never truth forecasts."""
        result={};withdrawn=set(getattr(self.recursive_self_culture,'_withdrawn_sources',()))
        for row in rows:
            if not row.get('actions'):continue
            action=int(row['actions'][0])
            program=next((p for p in self.recursive_self_culture._programs if int(p.identity)==int(row['identity'])),None)
            if program is None:continue
            bucket=result.setdefault(action,[])
            for reason_id in program.reason_ids:
                reason=next((r for r in self.recursive_self_culture._reasons if int(r.identity)==int(reason_id) and int(r.source) not in withdrawn),None)
                if reason is None:continue
                recommendation=ReasonActionRecommendationV1(int(reason.identity),int(reason.source),action,0)
                if recommendation not in bucket:bucket.append(recommendation)
        return {action:tuple(items) for action,items in result.items()}

    def recommendation_outcome_reliability_q16(self,reason,source,action,world_context=0):
        return self.recursive_causal_experiment.recommendation_reliability_world_q16(int(reason),int(source),int(action),int(world_context))
    def contextual_recommendation_outcome_reliability_q16(self,reason,source,action,world_context=None):
        resolved=int(self._world_context_current if world_context is None else world_context)
        return self.recommendation_outcome_reliability_q16(reason,source,action,resolved)
    # Historical API aliases: numeric storage compatibility only, not proposition prediction.
    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):return self.recommendation_outcome_reliability_q16(reason,source,action,int(regime))
    def contextual_reason_predictive_reliability_q16(self,reason,source,action,world_context=None):return self.contextual_recommendation_outcome_reliability_q16(reason,source,action,world_context)

    def _culture_episode_from_motor(self,motor,source):
        if motor is None or not getattr(motor,'settled',False):return _metacontrol_lifecycle.ReferenceOrganismV2._culture_episode_from_motor(self,motor,source)
        pending=int(self._metacontrol_pending_intervention);intervention=self.recursive_causal_experiment.intervention(pending) if pending else None
        ticket=int(getattr(motor,'ticket',0));action=int(getattr(motor,'action_id',0));success=int(getattr(motor,'effect',0))>0
        independent=bool(getattr(motor,'independent_consequence',False));state_before=tuple(getattr(motor,'state_before',()) or ())
        receipt=self._pre_outcome_transition_receipts.get(ticket)
        if receipt is None:
            world_context,control_context,world_cues,_cc,_r,_c=self._infer_contexts(state_before,source);predicted_state=();support=0
        else:world_context,control_context,world_cues,predicted_state,support=receipt
        decision=self.recursive_metacontrol.decision(self._metacontrol_pending_decision)
        parent_writes_self=bool(decision is not None and int(decision.selected)==action)
        identity=_metacontrol_lifecycle.ReferenceOrganismV2._culture_episode_from_motor(self,motor,source)
        if intervention is not None:
            if not parent_writes_self:self.recursive_metacontrol.record_outcome(action,success,independent)
            settle_tick=max(int(self.recursive_causal_experiment._tick),int(getattr(self,'_developmental_curriculum_tick',0)))
            self.recursive_causal_experiment.settle(pending,success,independent,settle_tick)
            self._metacontrol_pending_intervention=0;self._metacontrol_pending_decision=0
        resolved_world=self.recursive_context_partition.resolve_world_after_transition(
            world_context,world_cues,predicted_state,tuple(getattr(motor,'state_after',()) or ()),support,independent,self._context_tick())
        self._world_context_current=int(resolved_world);self._context_cache=None
        _wc,resolved_control,_wcu,_cc,_r,_c=self._infer_contexts(state_before,source);self._control_context_current=int(resolved_control)
        self.recursive_metacontrol.record_self_outcome(action,success,independent,resolved_control)
        if intervention is not None:
            self.recursive_causal_experiment.settle_world_context_evidence(pending,resolved_world)
            after_certainty=self.recursive_causal_experiment.recommendation_calibration_world(action,intervention.reasons,resolved_world)[1]
            comparable=int(resolved_world)==int(world_context)
            self.recursive_experiment_strategy.settle_control(pending,after_certainty,independent,comparable,control_context)
            self.recursive_experiment_policy.settle_control_policy(pending,after_certainty,independent,comparable,control_context)
        self._pre_outcome_transition_receipts.pop(ticket,None);return identity

    def _run_recursive_policy_metacontrol_ratchet(self):
        owner=ContextualMetacontrolV1();A=0xC101;B=0xC202;tick=1;Q=CTXQ
        # Crossed strengths: neither may be converted into an authored weighted winner.
        a=PolicyBranchV1(A,3*Q//4,Q//2,Q//3,Q//8,Q//4,0)
        b=PolicyBranchV1(B,Q//2,3*Q//4,3*Q//4,Q//8,Q//4,0)
        unresolved=owner.choose_contextual((a,b),Q,Q,Q,tick,0,0);tick+=1
        # Genuine dominance: A is no worse on every coordinate and better on several.
        d1=PolicyBranchV1(A,3*Q//4,3*Q//4,3*Q//4,Q//8,Q//4,0)
        d2=PolicyBranchV1(B,Q//2,Q//2,Q//2,Q//8,Q//3,0)
        dominated=owner.choose_contextual((d1,d2),Q,Q,0,tick,0,0);tick+=1
        # Risk veto cannot be bought off by maximal support/source/simulation.
        risky=PolicyBranchV1(A,Q,Q,Q,3*Q//4,0,0)
        risk_result=owner.choose_contextual((risky,),Q,Q,Q,tick,0,0);tick+=1
        # Low embodied control is a hard defer regardless of all evidence-like control axes.
        low_resource=owner.choose_contextual((d1,),Q//8,Q//8,Q,tick,0,0);tick+=1
        # High epistemic gap remains inquiry even after self competence becomes strong.
        control_context=71
        for _ in range(4):owner.record_self_outcome(A,True,True,control_context)
        high_gap=PolicyBranchV1(A,Q,Q,Q,0,3*Q//4,0)
        epistemic=owner.choose_contextual((high_gap,),Q,Q,Q,tick,0,control_context);tick+=1
        clean=PolicyBranchV1(A,Q,Q,Q,0,Q//8,0)
        clean_result=owner.choose_contextual((clean,),Q,Q,0,tick,B,control_context)
        checks={
          'cross_category_tradeoffs_remain_non_dominated_information_need':int(unresolved.get('mode',0))==MODE_ASK and int(unresolved.get('selected',0))==0 and set(unresolved.get('pareto_frontier',()))=={A,B},
          'genuine_multiaxis_dominance_can_commit_without_weights':int(dominated.get('mode',0))==MODE_ACT and int(dominated.get('selected',0))==A,
          'source_simulation_support_cannot_rescue_risk_veto':int(risk_result.get('selected',0))==0,
          'support_cannot_rescue_low_resource_or_control':int(low_resource.get('mode',0))==MODE_DEFER and int(low_resource.get('selected',0))==0,
          'self_confidence_cannot_erase_epistemic_gap':int(epistemic.get('mode',0))==MODE_ASK and int(epistemic.get('selected',0))==0,
          'confidence_field_is_self_reliability_not_branch_utility':int(clean_result.get('confidence_q16',-1))==owner.self_reliability_q16(A,control_context),
          'uncertainty_field_is_epistemic_gap_not_inverse_confidence':int(clean_result.get('uncertainty_q16',-1))==Q//8,
          'recalibration_can_revise_only_after_vetoes_clear':int(clean_result.get('mode',0))==MODE_REVISE and int(clean_result.get('selected',0))==A,
        }
        cp=owner.checkpoint();restored=ContextualMetacontrolV1.restore(cp)
        checks['multiaxis_metacontrol_survives_checkpoint']=restored.checkpoint()==cp
        return checks,sorted(name for name,value in checks.items() if not value)

    def _run_latent_causal_regime_ratchet(self):
        partition=RecursiveContextPartitionV2();meta=ContextualMetacontrolV1();causal=ContextualCausalExperimentV1();access=RecursivePartnerAccessV2()
        A=0xAB11;R=0xAB21;SRC=9601;world_a=(101,102,103,104);world_b=(101,102,103,105)
        w=partition.infer(WORLD_CONTEXT,world_a,1,0);control_a=(9001,9002,9003,9004,9005,9006);control_b=(9001,9102,9103,9104,9105,9106)
        c1=partition.infer(CONTROL_CONTEXT,control_a,2,0);world_same=partition.infer(WORLD_CONTEXT,world_a,3,w);c2=partition.infer(CONTROL_CONTEXT,control_b,4,c1)
        meta.record_self_outcome(A,True,True,c1);meta.record_self_outcome(A,True,True,c1);world_count_before=partition.context_count;meta.record_self_outcome(A,False,True,c2);world_count_after_self=partition.context_count
        recommendation=ReasonActionRecommendationV1(R,SRC,A,0);iv=causal.begin_world_probe(A,0,(recommendation,),(A,),CTXQ//2,1,w,True);causal.settle(iv,True,True,2);causal.settle_world_context_evidence(iv,w);recommendation_w=causal.recommendation_reliability_world_q16(R,SRC,A,w)
        split=partition.resolve_world_after_transition(w,world_b,(201,),(202,),2,True,5)
        identical_owner=RecursiveContextPartitionV2();wi=identical_owner.infer(WORLD_CONTEXT,world_a,1,0);before_identical=identical_owner.context_count;same_after_mismatch=identical_owner.resolve_world_after_transition(wi,world_a,(201,),(202,),2,True,2)
        access_before=access.evidence_count;access.observe_shared(7001,8001,100,world_a,w,1,False);qualified=access.observe_shared(7001,8001,100,world_a,w,2,True);stale=access.shared_access_staleness_q16(7001,world_b,split)
        old_api_refused=False
        try:access.perspective_gap_q16(7001,world_b,split)
        except RuntimeError:old_api_refused=True
        checks={
          'same_external_world_reuses_world_context':world_same==w,
          'body_control_history_can_change_control_context_without_world_split':c2!=c1 and world_same==w,
          'self_reliability_is_control_context_local':meta.self_reliability_q16(A,c1)>CTXQ//2 and meta.self_reliability_q16(A,c2)<CTXQ//2,
          'self_confidence_changes_cannot_split_world_context':world_count_after_self==world_count_before,
          'recommendation_outcome_reliability_is_world_context_local':recommendation_w>CTXQ//2,
          'recommendation_object_has_zero_proposition_truth_authority':recommendation.authority==0,
          'supported_transition_mismatch_with_world_cue_novelty_can_split_world_context':split!=w and partition.row(split).parent==w,
          'identical_world_cue_mismatch_revises_same_context':same_after_mismatch==wi and identical_owner.context_count==before_identical,
          'unqualified_partner_observation_creates_no_access':access.evidence_count==access_before+1 and qualified>0,
          'partner_access_staleness_is_not_belief_truth':stale>0 and old_api_refused,
          'typed_context_rows_have_zero_world_truth_authority':all(int(row.authority)==0 for row in partition._rows),
        }
        cp=(partition.checkpoint(),meta.checkpoint(),causal.checkpoint(),access.checkpoint())
        checks['category_safe_context_stack_survives_checkpoint']=(RecursiveContextPartitionV2.restore(cp[0]).checkpoint()==cp[0] and ContextualMetacontrolV1.restore(cp[1]).checkpoint()==cp[1] and ContextualCausalExperimentV1.restore(cp[2]).checkpoint()==cp[2] and RecursivePartnerAccessV2.restore(cp[3]).checkpoint()==cp[3])
        return checks,sorted(name for name,value in checks.items() if not value)
