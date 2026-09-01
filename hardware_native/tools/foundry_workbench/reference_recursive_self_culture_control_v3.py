#!/usr/bin/env python3
"""Category-safe cultural candidate surface over V2 sequence-qualified culture.

V2 remains checkpoint/history authority. V3 removes the multiplicative cultural candidate
score and body-scaled evidence review. Candidate rows expose independent causal coordinates;
resource/control are returned as state, never multiplied into social/reason/lived evidence.
"""
from __future__ import annotations
from reference_recursive_self_culture_control_v2 import RecursiveSelfCultureControlV2
from reference_recursive_self_culture_control_v1 import DecisionStateV1,Q

MAX_CULTURAL_CANDIDATES=64

class RecursiveSelfCultureControlV3(RecursiveSelfCultureControlV2):
    def _program_reason_support_raw(self,program):
        rows=[]
        for reason_id in set(map(int,program.reason_ids)):
            row=next((x for x in self._reasons
                if int(x.identity)==reason_id and int(x.source) not in self._withdrawn_sources),None)
            if row is not None:rows.append(max(-Q,min(Q,int(row.support_q16))))
        return 0 if not rows else max(-Q,min(Q,sum(rows)//len(rows)))

    def program_candidates(self,context,body_resource_q16,controllability_q16,source_credibility):
        self._refresh_program_confirmations();context=int(context)
        resource=max(0,min(Q,int(body_resource_q16)));control=max(0,min(Q,int(controllability_q16)))
        rows=[]
        for program in self._programs:
            if int(program.context)!=context:continue
            teachers=tuple(s for s in program.teachers if int(s) not in self._withdrawn_sources)
            confirmations=tuple(s for s in program.confirmations if int(s) not in self._withdrawn_sources)
            supporters=tuple(sorted(set((*teachers,*confirmations))))
            if not supporters:continue
            cred=sum(max(-Q,min(Q,int(source_credibility(int(source))))) for source in supporters)//len(supporters)
            source_quality=max(0,min(Q,(Q+cred)//2))
            raw_reason=self._program_reason_support_raw(program)
            recommendation_support=max(0,min(Q,(Q+raw_reason)//2)) if program.reason_ids else Q//2
            lived_support=min(Q,(len(confirmations)*Q)//3)
            rows.append({
                'identity':int(program.identity),'actions':tuple(program.actions),
                'teachers':teachers,'confirmations':confirmations,'reason_ids':tuple(program.reason_ids),
                'lived_program_support_q16':int(lived_support),
                'recommendation_support_q16':int(recommendation_support),
                'source_quality_q16':int(source_quality),
                'resource_q16':int(resource),'controllability_q16':int(control),
                'authority':0,
            })
        if len(rows)>MAX_CULTURAL_CANDIDATES:raise RuntimeError('self-culture-v3:candidate-capacity')
        # Identity order is a capacity-stable enumeration, not a preference ranking.
        rows.sort(key=lambda row:int(row['identity']))
        return tuple(rows)

    def review_decision(self,identity,new_evidence_q16,body_resource_q16,
                        controllability_q16,source_credibility_q16=0):
        """Re-evaluate resident evidence without converting body state into evidence magnitude.

        This owner no longer claims metacognitive confidence authority; current confidence is
        owned by RecursivePolicyMetacontrolV3. Body/control can defer selection, but the stored
        evidence values remain unchanged.
        """
        row=next((x for x in self._decisions if int(x.identity)==int(identity)),None)
        if row is None:return {'status':0,'selected':0,'changed':False,'confidence_q16':0}
        evidence=dict(row.evidence_q16)
        for action,value in new_evidence_q16:
            if int(action) in evidence:evidence[int(action)]=max(-Q,min(Q,evidence[int(action)]+int(value)))
        resource=max(0,min(Q,int(body_resource_q16)));control=max(0,min(Q,int(controllability_q16)))
        selected=0
        if resource>=Q//5 and control>=Q//6 and evidence:
            peak=max(evidence.values());winners=sorted(action for action,value in evidence.items() if value==peak)
            selected=winners[0] if len(winners)==1 else 0
        revised=DecisionStateV1(row.identity,row.context,row.alternatives,selected,
            tuple(sorted(evidence.items())),0,resource,control,row.reason_ids,row.revision+1,self._tick)
        self._decisions[self._decisions.index(row)]=revised
        return {'status':1,'selected':selected,'changed':bool(selected and selected!=row.selected),
            'confidence_q16':0,'revision':revised.revision}

    @classmethod
    def restore(cls,data):
        old=RecursiveSelfCultureControlV2.restore(data);out=cls();out.__dict__.update(old.__dict__);return out
