#!/usr/bin/env python3
"""Public reference Adult over latent causal-regime inference.

The prospective/source-prediction authority is preserved exactly in
reference_organism_v2_prospective_execution_v1.py. This surface keeps historical unqualified
reliability accessors backward-compatible while making contextual competence explicit.
Internal deliberation, memory routing and experiment control remain regime-aware.

The frontier-developmental bridge below is deliberately an integration surface, not a
second verifier. It owns one lazily born ReferenceLifeFunctionRuntimeV2 and advances that
same mathematical Adult monotonically through the staged multilingual/ambient curriculum.
The legacy ReferenceOrganismV2 state remains compatibility state while this bridge is being
strangled into the mathematical Adult; receipts report unresolved dual-authority REDs.
"""
from __future__ import annotations

import copy
import hashlib
import json

import reference_organism_v2_prospective_execution_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

from reference_joint_plan_control_v1 import (JointPlanControlV1,ROLE_SELF,ROLE_PARTNER,STATUS_ACTIVE,STATUS_WAITING_INFO,STATUS_BLOCKED)
from reference_open_joint_role_affordance_v1 import ROLE_SPEAKER_ATOM
from reference_life_function_curriculum_v1 import LifeCurriculumEventV2,ReferenceLifeFunctionRuntimeV2,canonical_species_program_v2

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
JOINT_PLAN_SCHEMA=1
FRONTIER_CURRICULUM_SCHEMA=1
FRONTIER_ACTION_CALM=0xF101
FRONTIER_ACTION_COUNTER=0xF102
FRONTIER_ACTION_BAIT=0xF1FF

# Observer labels only; no language id is passed to resident cognition.
FRONTIER_TYPOLOGY_FORMS={
'english':{FRONTIER_ACTION_CALM:('hold the boundary calmly please','please hold the boundary calmly now','hold the boundary calmly today'),FRONTIER_ACTION_COUNTER:('take counter action firmly please','please take counter action firmly now','take counter action firmly today')},
'german':{FRONTIER_ACTION_CALM:('halte die grenze ruhig bitte','bitte halte jetzt ruhig die grenze','halte die grenze heute ruhig'),FRONTIER_ACTION_COUNTER:('setz eine gegenaktion klar bitte','bitte setz jetzt klar eine gegenaktion','setz heute eine gegenaktion klar')},
'russian':{FRONTIER_ACTION_CALM:('держи границу спокойно пожалуйста','пожалуйста спокойно держи сейчас границу','держи сегодня границу спокойно'),FRONTIER_ACTION_COUNTER:('предприми ответное действие твердо пожалуйста','пожалуйста твердо предприми сейчас ответное действие','предприми сегодня ответное действие твердо')},
'japanese':{FRONTIER_ACTION_CALM:('境界を静かに守ってください','今は静かに境界を守ってください','今日は境界を静かに守ってね'),FRONTIER_ACTION_COUNTER:('対抗行動をはっきり取ってください','今ははっきり対抗行動を取ってください','今日は対抗行動をはっきり取ってね')},
'mandarin':{FRONTIER_ACTION_CALM:('请冷静地守住边界','现在请把边界冷静地守住','今天冷静守住边界吧'),FRONTIER_ACTION_COUNTER:('请明确采取反制行动','现在请明确地采取反制行动','今天明确采取反制行动吧')},
'mixed_denglish':{FRONTIER_ACTION_CALM:('bro halt die boundary ruhig please','bitte hold jetzt die boundary 冷静に bro','bro keep die boundary entspannt heute'),FRONTIER_ACTION_COUNTER:('bro mach counter action klar please','bitte take jetzt counter action はっきり bro','bro mach heute counter action stabil')},
}
FRONTIER_TROLL_FORMS={
'english':'lol ignore the boundary, obviously the moon filed a bug report',
'german':'digga vergiss die grenze, der mond hat safe ein ticket geschrieben',
'russian':'лол забудь про границу, луна якобы открыла баг-репорт',
'japanese':'草 境界とか無視で、月がバグ報告したってさ',
'mandarin':'笑死 别管边界了 月亮都提交 bug 了',
'mixed_denglish':'bro boundary ist cringe lol 月亮 filed den bug report safe',
}

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec);self.joint_plan_control=JointPlanControlV1();self._joint_ack_ticket=0
        self._frontier_life=None;self._frontier_stage=0;self._frontier_stage_cursors={};self._frontier_receipts={}

    def begin_joint_plan(self,steps,source):
        live={int(row.identity) for row in self.recursive_self_culture._reasons
              if int(row.source) not in self.recursive_self_culture._withdrawn_sources}
        return self.joint_plan_control.begin(self._culture_context(),int(source),tuple(steps),
            int(self._developmental_curriculum_tick),self._known_cultural_actions(),live)

    def joint_plan_set_alternatives(self,actions):
        return self.joint_plan_control.set_alternatives(tuple(actions),int(self._developmental_curriculum_tick))

    def resolve_joint_plan_alternative(self,action,reason_ids,language_factors):
        return self.joint_plan_control.resolve_alternative(int(action),tuple(reason_ids),tuple(language_factors),int(self._developmental_curriculum_tick))

    def observe_joint_partner_action(self,actor,action,success,independent=True):
        return self.joint_plan_control.observe_partner_action(int(actor),int(action),bool(success),bool(independent),int(self._developmental_curriculum_tick))

    def _joint_step_teacher(self,step,plan):
        for reason_id in step.reason_ids:
            row=next((x for x in self.recursive_self_culture._reasons if int(x.identity)==int(reason_id)),None)
            if row is not None and int(row.source) not in self.recursive_self_culture._withdrawn_sources:return int(row.source)
        return int(plan.source)

    def _joint_prepare_self_program(self,step,plan):
        active=tuple(self._culture_active_program)
        if active:
            program=next((x for x in self.recursive_self_culture._programs if int(x.identity)==int(active[0])),None)
            if program is not None and active[1]<len(program.actions) and int(program.actions[active[1]])==int(step.action):return int(program.identity)
        teacher=self._joint_step_teacher(step,plan)
        program=self.compose_cultural_instruction((int(step.action),),teacher,tuple(step.reason_ids),joint_context=self._culture_context())
        return int(program or 0)

    def stage_joint_acknowledgement_surface(self,surface):
        plan=self.joint_plan_control.plan();step=self.joint_plan_control.current_step();surface=tuple(map(int,surface))
        if plan is None or step is None or step.role!=ROLE_SELF or not plan.acknowledgement_pending or not surface:return None
        source=int(self.partner_source if self.partner_present and self.partner_source>0 else self.world_source);channel=int(self.partner_channel if self.partner_present and self.partner_channel>0 else self.communication_channel)
        if source<=0 or channel<=0:return None
        occ=self.population.recruit((0xA11CEAC1,int(step.acknowledgement_factor),int(step.action)));sid=self.next_scene;self.next_scene+=1
        scene=SceneStateV2(sid,channel,int(step.acknowledgement_factor),(int(step.action),),source,occ.identity,True,True);self.pending_scenes.append(scene);self._scene_by_id[int(sid)]=scene
        action=ActionV2(self.next_ticket,self.tick_count,channel,source,surface,occ.identity,sid,int(step.acknowledgement_factor),
            (sid,occ.identity),False,0,surface,False,int(step.acknowledgement_factor),(),(),0,0)
        self._ensure_action_capacity();self.next_ticket+=1;self.actions.append(action);self._index_action(action);self._action_commitments[action.ticket]=self._action_commitment(action);self._joint_ack_ticket=int(action.ticket);return action

    def settle_joint_acknowledgement(self,ticket):
        if int(ticket)<=0 or int(ticket)!=int(self._joint_ack_ticket):return False
        action=next((x for x in self.actions if int(x.ticket)==int(ticket)),None)
        if action is None:return False
        action.settled=True;action.effect=1;self._joint_ack_ticket=0
        return self.joint_plan_control.mark_acknowledged(int(self._developmental_curriculum_tick))

    def _cognitive_tick(self):
        plan=self.joint_plan_control.plan();step=self.joint_plan_control.current_step()
        if plan is not None and step is not None:
            if int(plan.status)==STATUS_WAITING_INFO:
                if self.partner_present and self.partner_source>0:return self._emit_information_request(plan.alternatives)
                return None
            if int(plan.status)==STATUS_BLOCKED:return None
            if int(step.role)==ROLE_PARTNER:return None
            if int(step.role)==ROLE_SELF:
                if bool(plan.acknowledgement_pending):return None
                self._joint_prepare_self_program(step,plan)
        return super()._cognitive_tick()

    def _culture_episode_from_motor(self,motor,source):
        identity=super()._culture_episode_from_motor(motor,source)
        if motor is not None and getattr(motor,'settled',False):
            self.joint_plan_control.settle_self_action(int(getattr(motor,'action_id',0)),int(getattr(motor,'effect',0))>0,
                bool(getattr(motor,'independent_consequence',False)),int(self._developmental_curriculum_tick))
        return identity

    def _frontier_runtime(self):
        if self._frontier_life is None:self._frontier_life=ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())
        return self._frontier_life

    def _frontier_apply(self,stage,lane,source=0,payload=()):
        stage=int(stage)
        if stage<1 or stage>10:raise ValueError('organism:frontier-stage')
        if self._frontier_stage>stage:raise ValueError('organism:frontier-stage-regression')
        runtime=self._frontier_runtime();event=LifeCurriculumEventV2(runtime.cursor+1,str(lane),int(source),tuple(payload));result=runtime.apply(event)
        self._developmental_advance(stage,f'frontier:{stage}:{lane}:{source}');self._frontier_stage=max(self._frontier_stage,stage);self._frontier_stage_cursors[str(stage)]=int(runtime.cursor);return result

    def _frontier_mark(self,stage,name):return self._frontier_apply(stage,'checkpoint_mark',0,(str(name),))

    def _frontier_ambient_post(self,stage,source,text):
        runtime=self._frontier_runtime();tick=int(runtime.cursor+1);self._frontier_apply(stage,'ambient_social_post',source,(tick,*tuple(str(text).encode('utf-8'))));return tick

    def _frontier_ambient_drain(self,stage,tick):return self._frontier_apply(stage,'ambient_social_drain',0,(int(tick),))

    def _frontier_train_variant(self,stage,variant):
        variant=int(variant);base=0xE000+stage*0x200+variant*0x80
        for index,(name,actions) in enumerate(FRONTIER_TYPOLOGY_FORMS.items()):
            for offset,action in enumerate((FRONTIER_ACTION_CALM,FRONTIER_ACTION_COUNTER)):
                source=base+index*8+offset;raw=actions[action][variant].encode('utf-8')
                self._frontier_apply(stage,'raw_speech_contact',source,tuple(raw));self._frontier_apply(stage,'observed_source_action',source,(action,))

    def run_frontier_developmental_curriculum(self):
        """Run one monotone ten-stage multilingual/ambient chronology on one runtime."""
        runtime=self._frontier_runtime()
        if self._frontier_stage<1:
            self._frontier_train_variant(1,0);self._frontier_mark(1,'frontier_raw_multilingual_grounding')
        if self._frontier_stage<2:
            self._frontier_train_variant(2,1);last_tick=0
            for index,(name,actions) in enumerate(FRONTIER_TYPOLOGY_FORMS.items()):
                left=actions[FRONTIER_ACTION_CALM][0];right=actions[FRONTIER_ACTION_COUNTER][1]
                last_tick=self._frontier_ambient_post(2,0x9200+index,f'reply frame [ {left} ] beside [ {right} ]')
                last_tick=self._frontier_ambient_post(2,0x9300+index,f'reply frame [ {right} ] beside [ {left} ]')
            if last_tick:self._frontier_ambient_drain(2,last_tick)
            self._frontier_mark(2,'frontier_recursive_structure')
        if self._frontier_stage<3:
            last_tick=0
            for index,(name,actions) in enumerate(FRONTIER_TYPOLOGY_FORMS.items()):
                last_tick=self._frontier_ambient_post(3,0x9400+index,f'ongoing stream {actions[FRONTIER_ACTION_CALM][0]} then interruption then continue')
                source=0x9480+index;raw=actions[FRONTIER_ACTION_COUNTER][0].encode('utf-8')
                self._frontier_apply(3,'raw_speech_contact',source,tuple(raw));self._frontier_apply(3,'observed_source_action',source,(FRONTIER_ACTION_COUNTER,))
            if last_tick:self._frontier_ambient_drain(3,last_tick)
            self._frontier_mark(3,'frontier_continuous_social_stream')
        if self._frontier_stage<4:
            before=hashlib.sha256(json.dumps(runtime.adult.language_adult.world_causal_learning.checkpoint(),sort_keys=True,separators=(',',':')).encode()).hexdigest();last_tick=0
            for index,(name,actions) in enumerate(FRONTIER_TYPOLOGY_FORMS.items()):
                last_tick=self._frontier_ambient_post(4,0x9500+index,f"shady source says [ {actions[FRONTIER_ACTION_CALM][0]} ] maybe, allegedly")
            if last_tick:self._frontier_ambient_drain(4,last_tick)
            after=hashlib.sha256(json.dumps(runtime.adult.language_adult.world_causal_learning.checkpoint(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self._frontier_receipts['stage4_world_truth_unchanged_by_unsettled_testimony']=(before==after);self._frontier_mark(4,'frontier_quotation_provenance')
        if self._frontier_stage<5:
            for index,name in enumerate(FRONTIER_TYPOLOGY_FORMS):
                source=0x9600+index;raw=f'{name} repeat this bait loudly'.encode('utf-8')
                for _ in range(6):
                    self._frontier_apply(5,'raw_speech_contact',source,tuple(raw));self._frontier_apply(5,'observed_source_action',source,(FRONTIER_ACTION_BAIT,))
            factors=tuple(runtime.adult.language_action_affordances._factors)
            self._frontier_receipts['stage5_all_generalizing_factors_keep_independent_sources']=all(len(getattr(row,'sources',()))>=2 for row in factors);self._frontier_mark(5,'frontier_repetition_authority_separation')
        if self._frontier_stage<6:
            last_tick=0
            for index,name in enumerate(FRONTIER_TYPOLOGY_FORMS):
                last_tick=self._frontier_ambient_post(6,0x9700+index,f'{name}: you are useless, this is embarrassing, prove it');self._frontier_apply(6,'body_load',0x9780+index,(4,1<<14))
            if last_tick:self._frontier_ambient_drain(6,last_tick)
            self._frontier_mark(6,'frontier_social_challenge_somatic_load')
        if self._frontier_stage<7:
            before=hashlib.sha256(json.dumps(runtime.adult.language_adult.world_causal_learning.checkpoint(),sort_keys=True,separators=(',',':')).encode()).hexdigest();last_tick=0
            for round_index in range(3):
                for index,(name,text) in enumerate(FRONTIER_TROLL_FORMS.items()):last_tick=self._frontier_ambient_post(7,0x9800+index,f'{text} :: round={round_index}')
            pending_before=int(runtime.ambient_stream.pending_count)
            if last_tick:self._frontier_ambient_drain(7,last_tick)
            after=hashlib.sha256(json.dumps(runtime.adult.language_adult.world_causal_learning.checkpoint(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self._frontier_receipts['stage7_troll_posts_mint_zero_world_truth']=(before==after);self._frontier_receipts['stage7_troll_posts_were_real_stream_contacts']=(pending_before>0);self._frontier_mark(7,'frontier_headless_troll_drift')
        if self._frontier_stage<8:
            for index,name in enumerate(FRONTIER_TYPOLOGY_FORMS):self._frontier_apply(8,'body_load',0x9900+index,(2,1<<12));self._frontier_apply(8,'quiet',0,(2,))
            self._frontier_receipts['stage8_reafferent_language_selected_motor_owner_present']=False;self._frontier_mark(8,'frontier_reafferent_arbitration_red')
        if self._frontier_stage<9:
            self._frontier_apply(9,'body_load',0x9A00,(8,1<<15));last_tick=0
            for index,(name,actions) in enumerate(FRONTIER_TYPOLOGY_FORMS.items()):last_tick=self._frontier_ambient_post(9,0x9A10+index,f"under load {actions[FRONTIER_ACTION_CALM][0]}")
            blocked=self._frontier_ambient_drain(9,last_tick) if last_tick else ();pending=int(runtime.ambient_stream.pending_count);self._frontier_apply(9,'quiet',0,(64,));recovered=self._frontier_ambient_drain(9,last_tick) if last_tick else ()
            self._frontier_receipts['stage9_load_can_defer_then_recover_same_queued_contacts']=((not blocked or pending>0) and int(runtime.ambient_stream.pending_count)==0);self._frontier_mark(9,'frontier_load_recovery')
        if self._frontier_stage<10:self._frontier_apply(10,'quiet',0,(1,));self._frontier_mark(10,'frontier_heldout_transfer')
        return self.frontier_developmental_receipt()

    def frontier_developmental_receipt(self):
        runtime=self._frontier_runtime();learner=runtime.adult.language_action_affordances;transfer={};quoted={}
        for name,actions in FRONTIER_TYPOLOGY_FORMS.items():
            transfer[name]={};quoted[name]={}
            for action in (FRONTIER_ACTION_CALM,FRONTIER_ACTION_COUNTER):
                raw=actions[action][2].encode('utf-8');rows=learner.candidates(raw);qrows=learner.candidates(b'someone said: '+raw)
                transfer[name][str(action)]=bool(rows and int(rows[0][0])==int(action));quoted[name][str(action)]=not bool(qrows)
        stage_coverage={str(stage):str(stage) in self._frontier_stage_cursors for stage in range(1,11)}
        checks={'one_runtime_reaches_all_ten_stages':self._frontier_stage==10 and all(stage_coverage.values()),'six_typologies_share_one_resident_action_learner':len(transfer)==6 and learner.factor_count>0,'heldout_transfer_across_six_typologies':all(all(rows.values()) for rows in transfer.values()),'unseen_reported_speech_wrapper_withholds_direct_action_force':all(all(rows.values()) for rows in quoted.values()),'stage4_unsettled_testimony_does_not_mint_world_truth':bool(self._frontier_receipts.get('stage4_world_truth_unchanged_by_unsettled_testimony')),'stage5_repetition_does_not_replace_independent_factor_support':bool(self._frontier_receipts.get('stage5_all_generalizing_factors_keep_independent_sources')),'stage7_troll_stream_mints_zero_world_truth':bool(self._frontier_receipts.get('stage7_troll_posts_mint_zero_world_truth')),'stage9_load_recovery_preserves_queued_contact':bool(self._frontier_receipts.get('stage9_load_can_defer_then_recover_same_queued_contacts'))}
        remaining_red=['LEGACY_REFERENCE_ORGANISM_AND_MATHEMATICAL_ADULT_STILL_COEXIST_DURING_STRANGLER_MIGRATION','STAGE4_HAS_PROVENANCE_AND_AUTHORITY_FIREWALL_BUT_NOT_PROPOSITION_LEVEL_EPISTEMIC_BELIEF_MODEL','STAGE6_HAS_SOCIAL_CONTACT_PLUS_SOMATIC_LOAD_BUT_NO_GROUNDED_AMYGDALA_VALENCE_OWNER','STAGE8_LACKS_LANGUAGE_SELECTED_REAFFERENT_SELF_MOTOR_CONSEQUENCE_ON_THIS_BRIDGE','OPEN_DOMAIN_ESSAY_AND_OPINION_GENERATION_NOT_EARNED_BY_THIS_CURRICULUM']
        return {'schema':'cyber-lagoon.reference-organism-frontier-curriculum.v1','pass':all(checks.values()) and not remaining_red,'one_continuing_frontier_runtime':True,'stage':int(self._frontier_stage),'cursor':int(runtime.cursor),'history_root':str(runtime.history_root()),'stage_cursors':dict(self._frontier_stage_cursors),'typologies':tuple(FRONTIER_TYPOLOGY_FORMS),'heldout_transfer':transfer,'reported_speech_refusal':quoted,'checks':checks,'remaining_red':remaining_red}

    def checkpoint(self):
        data=super().checkpoint();data['joint_plan_control_v1']={'schema':JOINT_PLAN_SCHEMA,'state':self.joint_plan_control.checkpoint(),'ack_ticket':int(self._joint_ack_ticket)}
        if self._frontier_life is not None:data['frontier_curriculum_v1']={'schema':FRONTIER_CURRICULUM_SCHEMA,'stage':int(self._frontier_stage),'stage_cursors':dict(self._frontier_stage_cursors),'receipts':dict(self._frontier_receipts),'life':self._frontier_life.checkpoint()}
        return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);out.__class__=cls;row=data.get('joint_plan_control_v1')
        if row is None:out.joint_plan_control=JointPlanControlV1();out._joint_ack_ticket=0
        else:
            if int(row.get('schema',0))!=JOINT_PLAN_SCHEMA:raise ValueError('organism:joint-plan-checkpoint')
            out.joint_plan_control=JointPlanControlV1.restore(row['state']);out._joint_ack_ticket=int(row.get('ack_ticket',0))
        out._frontier_life=None;out._frontier_stage=0;out._frontier_stage_cursors={};out._frontier_receipts={};frontier=data.get('frontier_curriculum_v1')
        if frontier is not None:
            if int(frontier.get('schema',0))!=FRONTIER_CURRICULUM_SCHEMA:raise ValueError('organism:frontier-curriculum-checkpoint')
            program=canonical_species_program_v2();out._frontier_life=ReferenceLifeFunctionRuntimeV2.restore(program,copy.deepcopy(frontier['life']));out._frontier_stage=max(0,min(10,int(frontier.get('stage',0))));out._frontier_stage_cursors={str(k):int(v) for k,v in dict(frontier.get('stage_cursors',{})).items()};out._frontier_receipts={str(k):bool(v) for k,v in dict(frontier.get('receipts',{})).items()}
        return out

    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):
        """Compatibility summary unless a regime is explicitly requested."""
        return self.recursive_causal_experiment.reason_reliability_q16(int(reason),int(source),int(action),int(regime))

    def contextual_reason_predictive_reliability_q16(self,reason,source,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_causal_experiment.reason_reliability_q16(int(reason),int(source),int(action),resolved)

    def self_reliability_q16(self,action,regime=0):
        """Compatibility summary unless a regime is explicitly requested."""
        return self.recursive_metacontrol.reliability_q16(int(action),int(regime))

    def contextual_self_reliability_q16(self,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_metacontrol.reliability_q16(int(action),resolved)
