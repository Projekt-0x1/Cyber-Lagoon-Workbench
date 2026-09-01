#!/usr/bin/env python3
"""Live Adult repairing prospective/source sequence gates that incorrectly depended on reward.

The typed source-prediction Adult is preserved in `reference_organism_v2_source_prediction_v1.py`.
Ordinary core settlement runs first. This layer then restores only category-valid execution or
prediction evidence suppressed by the historical `effect > 0` / `effect != 0` gates.
"""
from __future__ import annotations

import reference_organism_v2_source_prediction_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload);kind=int(kind);source=int(source)
        repair=None
        if kind==CONTACT_MOTOR_CONSEQUENCE and payload:
            ticket=int(payload[0]);effect=int(payload[1]);next_state=tuple(sorted(set(int(x) for x in payload[3:] if int(x)!=0)))
            motor=next((row for row in self.motor_actions if int(row.ticket)==ticket and not row.settled),None)
            if motor is not None:
                lookup={int(row.identity):row for row in self.source_assertions}
                source_rows=tuple(lookup[aid] for aid in tuple(motor.source_assertion_ids) if int(aid) in lookup)
                source_selected=bool(independent and motor.source_occurrences and int(motor.source_context)>0
                    and int(motor.source_counterfactual_action)!=int(motor.action_id))
                closures=[]
                if source_selected and effect<=0:
                    for row in source_rows:
                        if not int(row.prospective_closure) or int(row.action_id)!=int(motor.action_id):continue
                        closure=self._prospective_source_closures.get(int(row.prospective_closure))
                        if closure is None or int(closure.cursor)!=int(row.prospective_step):continue
                        closures.append((closure,int(row.prospective_step)))
                neutral_predictions=tuple(row for row in source_rows if source_selected and effect==0 and tuple(row.predicted_state))
                expert=None
                if bool(independent) and effect<=0 and tuple(motor.prospective_snapshot):
                    identity,_shadow,start,goal,actions,_effects,_sources,states=self._prospective_snapshot_parts(motor.prospective_snapshot)
                    matches=[index for index in range(len(actions))
                        if states[index]==tuple(motor.state_before) and int(actions[index])==int(motor.action_id)
                        and states[index+1]==next_state]
                    if len(matches)==1 and matches[0]==len(actions)-1 and next_state==goal:
                        expert=(start,goal,actions,states,source,int(motor.prospective_context_signature))
                repair=(motor,effect,next_state,source_selected,tuple(closures),neutral_predictions,expert)

        result=super().contact(kind,payload,source,authenticated,independent)
        if repair is None:return result
        motor,effect,next_state,source_selected,closures,neutral_predictions,expert=repair
        learned=dict(result) if isinstance(result,dict) else {'result':result}

        # Goal/path completion is independent of whether the consequence was pleasant.
        if expert is not None:
            start,goal,actions,states,completion_source,context=expert
            nomination=self.cognition.record_expert_completion(start,goal,actions,states,completion_source,context)
            learned['prospective_completion_observed']=1
            if nomination is not None:learned['prospective_expert_nomination']=int(nomination.identity)

        # Source-provided sequence execution advances on causally selected independent action
        # settlement. Negative/neutral valence remains available to pragmatic owners separately.
        for closure,step in closures:
            if int(closure.cursor)!=int(step):continue
            closure.cursor+=1
            if closure.cursor>=len(closure.hypotheses):
                self._prospective_source_closures.pop(int(closure.identity),None)
            else:
                self._prospective_source_closures[int(closure.identity)]=closure
                next_assertion=self._activate_prospective_source_step(closure)
                if next_assertion:learned['prospective_next_assertion']=int(next_assertion)

        # Neutral reward must not hide an independently observable explicit prediction match.
        neutral_updates=0
        for row in neutral_predictions:
            matched=set(map(int,row.predicted_state)).issubset(set(next_state))
            neutral_updates+=int(self.source_prediction_calibration.observe(
                int(row.source),int(motor.source_context),matched,True))
        if neutral_updates:
            learned['source_updates']=int(learned.get('source_updates',0))+neutral_updates
            learned['typed_neutral_prediction_updates']=neutral_updates
        return learned
