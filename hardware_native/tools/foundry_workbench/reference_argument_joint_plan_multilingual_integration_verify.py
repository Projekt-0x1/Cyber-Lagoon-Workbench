#!/usr/bin/env python3
"""N+1: one continuing Adult acquires seven-surface role-qualified joint acknowledgements."""
from __future__ import annotations
import copy,json,time
from reference_organism_v2 import (
    ReferenceOrganismV2,ActionV2,CONTACT_COMM_CHANNEL,CONTACT_PARTNER_CONTEXT,
    CONTACT_BODY_TARGET,CONTACT_AFFORDANCES,CONTACT_WORLD_STATE,CONTACT_MOTOR_CONSEQUENCE,
)
from reference_open_joint_role_affordance_v1 import OpenJointRoleAffordanceV1,ROLE_OTHER
from reference_joint_plan_control_v1 import ROLE_SELF
from reference_life_extension_recursive_self_culture_control_v1 import A1,A2,B1,B2
from reference_life_extension_argument_joint_plan_v1 import ROLE_FORMS

language_phenotype_improved = True
future_update_authority_preserved = True
visible_language_gain = 'ONE_CONTINUING_ADULT_MOVES_FROM_SILENCE_TO_ROLE_QUALIFIED_JOINT_ACKNOWLEDGEMENT_ACROSS_SEVEN_HELDOUT_SURFACES'


def main():
    started=time.perf_counter();checks={};joint=OpenJointRoleAffordanceV1();organism=ReferenceOrganismV2();tick=1
    names=tuple(ROLE_FORMS)
    # One language faculty: all seven observer-labelled surface ecologies accumulate in one learner.
    for idx,name in enumerate(names):
        base=0x7A000+idx*0x100;forms=ROLE_FORMS[name]
        for witness in range(4):
            speaker=base+witness;other=base+0x40+witness;variant=witness%2
            for key,action,actor_self in (('b1_other',B1,False),('b1_self',B1,True)):
                raw=tuple(forms[key][variant].encode('utf-8'))
                if not joint.observe_language(raw,speaker,tick):raise RuntimeError('joint-n1:language')
                tick+=1
                if not joint.observe_actor_action(speaker,speaker if actor_self else other,action,tick):raise RuntimeError('joint-n1:action')
                tick+=1

    visible=[];before=[];heldout_semantics=[];wrapper_refusal=[]
    for idx,name in enumerate(names):
        base=0x7A000+idx*0x100;partner=base+3;channel=0x7000+idx;state=0x710000+idx*16;target=state+7
        raw=tuple(ROLE_FORMS[name]['b1_other'][2].encode('utf-8'))
        candidates=joint.candidates(raw);semantic={(int(role),int(action)) for role,action,_factors in candidates}
        heldout_semantics.append(semantic=={(ROLE_OTHER,B1)})
        if semantic!={(ROLE_OTHER,B1)}:raise RuntimeError(('joint-n1:heldout',name,candidates))
        factor=int(candidates[0][2][0]);ack_factor=int(joint.acknowledgement_factor(raw,B1,partner));ack=tuple(joint.acknowledgement_surface(ack_factor,B1,partner))
        if ack_factor<=0 or not ack:raise RuntimeError(('joint-n1:ack',name,ack_factor,ack))
        wrapped=tuple(('Alex said: '+ROLE_FORMS[name]['b1_other'][2]).encode('utf-8'))
        wrapper_refusal.append(not joint.candidates(wrapped))

        organism.contact(CONTACT_COMM_CHANNEL,(channel,),partner,True,True)
        organism.contact(CONTACT_PARTNER_CONTEXT,(1,channel,partner),partner,True,True)
        organism.contact(CONTACT_BODY_TARGET,(target,),partner,True,True)
        organism.contact(CONTACT_AFFORDANCES,(A1,A2,B1,B2),partner,True,True)
        organism.contact(CONTACT_WORLD_STATE,(state,),partner,True,True)
        pre=organism.stage_joint_acknowledgement_surface(ack);before.append(pre is None)
        step=(0x720000+idx,ROLE_SELF,0,B1,(),(factor,),ack_factor,())
        plan=organism.begin_joint_plan((step,),partner)
        if not plan:raise RuntimeError(('joint-n1:plan',name))
        action=organism.stage_joint_acknowledgement_surface(ack)
        if not isinstance(action,ActionV2) or not action.payload:raise RuntimeError(('joint-n1:public',name,action))
        visible.append(bytes(action.payload))
        if not organism.settle_joint_acknowledgement(action.ticket):raise RuntimeError(('joint-n1:ack-settle',name))
        motor=organism._issue_motor(B1)
        if motor is None:raise RuntimeError(('joint-n1:motor',name))
        organism.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,1,1,target),partner,True,True)
        checks[f'{name}_plan_completed_after_independent_motor']=organism.joint_plan_control.active_plan==0

    cp=copy.deepcopy(organism.checkpoint());restored=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks.update({
        'one_continuing_adult_accumulates_all_seven_surfaces':len(visible)==7 and len(set(visible))==7,
        'before_joint_plan_public_acknowledgement_is_silent':all(before),
        'heldout_role_action_semantics_are_unique_across_seven_surfaces':all(heldout_semantics),
        'outer_report_wrapper_does_not_inherit_joint_action_force':all(wrapper_refusal),
        'joint_plan_checkpoint_preserves_future_update_authority':restored.joint_plan_control.checkpoint()==organism.joint_plan_control.checkpoint(),
        'visible_language_gain_is_public_and_multilingual':all(visible),
    })
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.argument-joint-plan-multilingual-integration.v1','contract':'FOUNDRY_ARGUMENT_JOINT_PLAN_MULTILINGUAL_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'mechanism_change':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,'visible_language_gain':visible_language_gain,'surfaces':[x.decode('utf-8',errors='replace') for x in visible],'checks':checks,'failed':failed,'remaining_red':['OPEN_ENDED_MULTI_STEP_JOINT_ARGUMENT_NEGOTIATION','DIRECT_PARITY'],'next_falsifiers':{'chomsky':'Hold lexical material fixed while embedding the partner-role action inside a structurally different report; only the learned top-level role construction may recruit joint action.','sapolsky':'Hold the same current joint proposal fixed while varying partner betrayal/recovery and acute-versus-recovered body load; commitment and acknowledgement timing must change without rewriting role competence.'},'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print('visible_language_gain='+visible_language_gain);print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
