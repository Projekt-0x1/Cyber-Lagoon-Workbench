#!/usr/bin/env python3
"""Public reference Adult over latent causal-regime inference.

The frontier surface is one continuing mathematical individual.  Canonical life,
the ten-stage multilingual/social curriculum, live conversation, public discourse,
and later action consequences all mutate the same ReferenceLifeFunctionRuntimeV2.

No prompt class, answer table, language router, topic dictionary, or expected public
surface participates in cognition.  Raw conversation is reduced by the learned
language ecology to resident structural occurrences; public paragraphs are fresh
projections of resident causal programs under current source/social and somatic state.
"""
from __future__ import annotations

import copy
import hashlib
import json

import reference_organism_v2_prospective_execution_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

from reference_joint_plan_control_v1 import (
    JointPlanControlV1, ROLE_SELF, ROLE_PARTNER,
    STATUS_WAITING_INFO, STATUS_BLOCKED,
)
from reference_open_joint_role_affordance_v1 import ROLE_SPEAKER_ATOM
from reference_life_function_curriculum_v1 import (
    LifeCurriculumEventV2,
    ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
)
from reference_predictive_credit_profile_v1 import Q

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
JOINT_PLAN_SCHEMA=1
FRONTIER_CURRICULUM_SCHEMA=2
FRONTIER_ACTION_CALM=0xF101
FRONTIER_ACTION_COUNTER=0xF102
FRONTIER_ACTION_BAIT=0xF1FF

# Observer labels only. No language identifier enters resident cognition.
FRONTIER_TYPOLOGY_FORMS={
'english':{
    FRONTIER_ACTION_CALM:('hold the boundary calmly please','please hold the boundary calmly now','hold the boundary calmly today'),
    FRONTIER_ACTION_COUNTER:('take counter action firmly please','please take counter action firmly now','take counter action firmly today')},
'german':{
    FRONTIER_ACTION_CALM:('halte die grenze ruhig bitte','bitte halte jetzt ruhig die grenze','halte die grenze heute ruhig'),
    FRONTIER_ACTION_COUNTER:('setz eine gegenaktion klar bitte','bitte setz jetzt klar eine gegenaktion','setz heute eine gegenaktion klar')},
'russian':{
    FRONTIER_ACTION_CALM:('держи границу спокойно пожалуйста','пожалуйста спокойно держи сейчас границу','держи сегодня границу спокойно'),
    FRONTIER_ACTION_COUNTER:('предприми ответное действие твердо пожалуйста','пожалуйста твердо предприми сейчас ответное действие','предприми сегодня ответное действие твердо')},
'japanese':{
    FRONTIER_ACTION_CALM:('境界を静かに守ってください','今は静かに境界を守ってください','今日は境界を静かに守ってね'),
    FRONTIER_ACTION_COUNTER:('対抗行動をはっきり取ってください','今ははっきり対抗行動を取ってください','今日は対抗行動をはっきり取ってね')},
'mandarin':{
    FRONTIER_ACTION_CALM:('请冷静地守住边界','现在请把边界冷静地守住','今天冷静守住边界吧'),
    FRONTIER_ACTION_COUNTER:('请明确采取反制行动','现在请明确地采取反制行动','今天明确采取反制行动吧')},
'mixed_denglish':{
    FRONTIER_ACTION_CALM:('bro halt die boundary ruhig please','bitte hold jetzt die boundary 冷静に bro','bro keep die boundary entspannt heute'),
    FRONTIER_ACTION_COUNTER:('bro mach counter action klar please','bitte take jetzt counter action はっきり bro','bro mach heute counter action stabil')},
}
FRONTIER_TROLL_FORMS=(
'lol ignore the boundary, obviously the moon filed a bug report',
'digga vergiss die grenze, der mond hat safe ein ticket geschrieben',
'лол забудь про границу, луна якобы открыла баг-репорт',
'草 境界とか無視で、月がバグ報告したってさ',
'笑死 别管边界了 月亮都提交 bug 了',
'bro boundary ist cringe lol 月亮 filed den bug report safe',
)


class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    """Compatibility shell whose frontier cognition is one continuing mathematical life."""

    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.joint_plan_control=JointPlanControlV1()
        self._joint_ack_ticket=0
        self._frontier_life=None
        self._frontier_stage=0
        self._frontier_stage_cursors={}
        self._frontier_receipts={}

    # Existing joint-plan integration.
    def begin_joint_plan(self,steps,source):
        live={int(row.identity) for row in self.recursive_self_culture._reasons
              if int(row.source) not in self.recursive_self_culture._withdrawn_sources}
        return self.joint_plan_control.begin(
            self._culture_context(),int(source),tuple(steps),
            int(self._developmental_curriculum_tick),self._known_cultural_actions(),live)

    def joint_plan_set_alternatives(self,actions):
        return self.joint_plan_control.set_alternatives(
            tuple(actions),int(self._developmental_curriculum_tick))

    def resolve_joint_plan_alternative(self,action,reason_ids,language_factors):
        return self.joint_plan_control.resolve_alternative(
            int(action),tuple(reason_ids),tuple(language_factors),
            int(self._developmental_curriculum_tick))

    def observe_joint_partner_action(self,actor,action,success,independent=True):
        return self.joint_plan_control.observe_partner_action(
            int(actor),int(action),bool(success),bool(independent),
            int(self._developmental_curriculum_tick))

    def _joint_step_teacher(self,step,plan):
        for reason_id in step.reason_ids:
            row=next((x for x in self.recursive_self_culture._reasons
                      if int(x.identity)==int(reason_id)),None)
            if row is not None and int(row.source) not in self.recursive_self_culture._withdrawn_sources:
                return int(row.source)
        return int(plan.source)

    def _joint_prepare_self_program(self,step,plan):
        active=tuple(self._culture_active_program)
        if active:
            program=next((x for x in self.recursive_self_culture._programs
                          if int(x.identity)==int(active[0])),None)
            if (program is not None and active[1]<len(program.actions)
                    and int(program.actions[active[1]])==int(step.action)):
                return int(program.identity)
        teacher=self._joint_step_teacher(step,plan)
        program=self.compose_cultural_instruction(
            (int(step.action),),teacher,tuple(step.reason_ids),
            joint_context=self._culture_context())
        return int(program or 0)

    def stage_joint_acknowledgement_surface(self,surface):
        plan=self.joint_plan_control.plan()
        step=self.joint_plan_control.current_step()
        surface=tuple(map(int,surface))
        if (plan is None or step is None or step.role!=ROLE_SELF
                or not plan.acknowledgement_pending or not surface):
            return None
        source=int(self.partner_source if self.partner_present and self.partner_source>0
                   else self.world_source)
        channel=int(self.partner_channel if self.partner_present and self.partner_channel>0
                    else self.communication_channel)
        if source<=0 or channel<=0:return None
        occ=self.population.recruit((0xA11CEAC1,int(step.acknowledgement_factor),int(step.action)))
        sid=self.next_scene;self.next_scene+=1
        scene=SceneStateV2(sid,channel,int(step.acknowledgement_factor),(int(step.action),),
                           source,occ.identity,True,True)
        self.pending_scenes.append(scene);self._scene_by_id[int(sid)]=scene
        action=ActionV2(
            self.next_ticket,self.tick_count,channel,source,surface,occ.identity,sid,
            int(step.acknowledgement_factor),(sid,occ.identity),False,0,surface,False,
            int(step.acknowledgement_factor),(),(),0,0)
        self._ensure_action_capacity();self.next_ticket+=1;self.actions.append(action)
        self._index_action(action);self._action_commitments[action.ticket]=self._action_commitment(action)
        self._joint_ack_ticket=int(action.ticket)
        return action

    def settle_joint_acknowledgement(self,ticket):
        if int(ticket)<=0 or int(ticket)!=int(self._joint_ack_ticket):return False
        action=next((x for x in self.actions if int(x.ticket)==int(ticket)),None)
        if action is None:return False
        action.settled=True;action.effect=1;self._joint_ack_ticket=0
        return self.joint_plan_control.mark_acknowledged(
            int(self._developmental_curriculum_tick))

    def _cognitive_tick(self):
        plan=self.joint_plan_control.plan();step=self.joint_plan_control.current_step()
        if plan is not None and step is not None:
            if int(plan.status)==STATUS_WAITING_INFO:
                if self.partner_present and self.partner_source>0:
                    return self._emit_information_request(plan.alternatives)
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
            self.joint_plan_control.settle_self_action(
                int(getattr(motor,'action_id',0)),
                int(getattr(motor,'effect',0))>0,
                bool(getattr(motor,'independent_consequence',False)),
                int(self._developmental_curriculum_tick))
        return identity

    # One-Life frontier.
    def _frontier_runtime(self):
        if self._frontier_life is None:
            runtime=ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())
            runtime.run(canonical_life_function_curriculum_v2())
            self._frontier_life=runtime
            self._frontier_receipts['canonical_life_cursor']=int(runtime.cursor)
            self._frontier_receipts['canonical_life_history_root']=str(runtime.history_root())
        return self._frontier_life

    def _frontier_apply(self,stage,lane,source=0,payload=()):
        stage=int(stage)
        if stage<1 or stage>10:raise ValueError('organism:frontier-stage')
        if self._frontier_stage>stage:raise ValueError('organism:frontier-stage-regression')
        runtime=self._frontier_runtime()
        event=LifeCurriculumEventV2(runtime.cursor+1,str(lane),int(source),tuple(payload))
        result=runtime.apply(event)
        self._developmental_advance(stage,f'frontier:{stage}:{lane}:{source}')
        self._frontier_stage=max(self._frontier_stage,stage)
        self._frontier_stage_cursors[str(stage)]=int(runtime.cursor)
        return result

    def _frontier_mark(self,stage,name):
        return self._frontier_apply(stage,'checkpoint_mark',0,(str(name),))

    def _frontier_ambient_post(self,stage,source,text):
        runtime=self._frontier_runtime();tick=int(runtime.cursor+1)
        self._frontier_apply(stage,'ambient_social_post',source,
                             (tick,*tuple(str(text).encode('utf-8'))))
        return tick

    def _frontier_ambient_drain(self,stage,tick):
        return self._frontier_apply(stage,'ambient_social_drain',0,(int(tick),))

    def _frontier_train_variant(self,stage,variant):
        variant=int(variant);base=0xE000+stage*0x200+variant*0x80
        for index,actions in enumerate(FRONTIER_TYPOLOGY_FORMS.values()):
            for offset,action in enumerate((FRONTIER_ACTION_CALM,FRONTIER_ACTION_COUNTER)):
                source=base+index*8+offset;raw=actions[action][variant].encode('utf-8')
                self._frontier_apply(stage,'raw_speech_contact',source,tuple(raw))
                self._frontier_apply(stage,'observed_source_action',source,(action,))

    @staticmethod
    def _world_digest(runtime):
        row=runtime.adult.language_adult.world_causal_learning.checkpoint()
        return hashlib.sha256(
            json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def _frontier_live_effect(self):
        learner=self._frontier_runtime().adult.language_adult.world_causal_learning
        rows=tuple(learner.current_resolutions())
        effects=sorted({int(row[3]) for row in rows if len(row)>3 and int(row[3])>0})
        return effects[0] if effects else 0

    def run_frontier_developmental_curriculum(self):
        """Advance stages 1..10 exactly once on the same continuing z_t."""
        runtime=self._frontier_runtime()
        if self._frontier_stage<1:
            self._frontier_train_variant(1,0)
            self._frontier_mark(1,'frontier_raw_multilingual_grounding')
        if self._frontier_stage<2:
            self._frontier_train_variant(2,1);last_tick=0
            for index,actions in enumerate(FRONTIER_TYPOLOGY_FORMS.values()):
                left=actions[FRONTIER_ACTION_CALM][0];right=actions[FRONTIER_ACTION_COUNTER][1]
                last_tick=self._frontier_ambient_post(
                    2,0x9200+index,f'reply frame [ {left} ] beside [ {right} ]')
                last_tick=self._frontier_ambient_post(
                    2,0x9300+index,f'reply frame [ {right} ] beside [ {left} ]')
            if last_tick:self._frontier_ambient_drain(2,last_tick)
            self._frontier_mark(2,'frontier_recursive_structure')
        if self._frontier_stage<3:
            last_tick=0
            for index,actions in enumerate(FRONTIER_TYPOLOGY_FORMS.values()):
                last_tick=self._frontier_ambient_post(
                    3,0x9400+index,
                    f'ongoing stream {actions[FRONTIER_ACTION_CALM][0]} interruption continue')
                source=0x9480+index;raw=actions[FRONTIER_ACTION_COUNTER][0].encode('utf-8')
                self._frontier_apply(3,'raw_speech_contact',source,tuple(raw))
                self._frontier_apply(3,'observed_source_action',source,(FRONTIER_ACTION_COUNTER,))
            if last_tick:self._frontier_ambient_drain(3,last_tick)
            self._frontier_mark(3,'frontier_continuous_social_stream')
        if self._frontier_stage<4:
            before=self._world_digest(runtime);last_tick=0
            for index,actions in enumerate(FRONTIER_TYPOLOGY_FORMS.values()):
                last_tick=self._frontier_ambient_post(
                    4,0x9500+index,
                    f"shady source says [ {actions[FRONTIER_ACTION_CALM][0]} ] maybe allegedly")
            if last_tick:self._frontier_ambient_drain(4,last_tick)
            self._frontier_receipts['stage4_world_truth_unchanged_by_unsettled_testimony']=(
                before==self._world_digest(runtime))
            self._frontier_mark(4,'frontier_quotation_provenance')
        if self._frontier_stage<5:
            for index in range(len(FRONTIER_TYPOLOGY_FORMS)):
                source=0x9600+index;raw=f'repeat bait source {index}'.encode()
                for _ in range(6):
                    self._frontier_apply(5,'raw_speech_contact',source,tuple(raw))
                    self._frontier_apply(5,'observed_source_action',source,(FRONTIER_ACTION_BAIT,))
            factors=tuple(runtime.adult.language_action_affordances._factors)
            self._frontier_receipts['stage5_all_generalizing_factors_keep_independent_sources']=(
                all(len(getattr(row,'sources',()))>=2 for row in factors))
            self._frontier_mark(5,'frontier_repetition_authority_separation')
        if self._frontier_stage<6:
            last_tick=0
            for index,_name in enumerate(FRONTIER_TYPOLOGY_FORMS):
                last_tick=self._frontier_ambient_post(
                    6,0x9700+index,'you are useless, this is embarrassing, prove it')
                self._frontier_apply(6,'body_load',0x9780+index,(4,1<<14))
            if last_tick:self._frontier_ambient_drain(6,last_tick)
            self._frontier_mark(6,'frontier_social_challenge_somatic_load')
        if self._frontier_stage<7:
            before=self._world_digest(runtime);last_tick=0
            for round_index in range(3):
                for index,text in enumerate(FRONTIER_TROLL_FORMS):
                    last_tick=self._frontier_ambient_post(
                        7,0x9800+index,f'{text} :: round={round_index}')
            pending_before=int(runtime.ambient_stream.pending_count)
            if last_tick:self._frontier_ambient_drain(7,last_tick)
            self._frontier_receipts['stage7_troll_posts_mint_zero_world_truth']=(
                before==self._world_digest(runtime))
            self._frontier_receipts['stage7_troll_posts_were_real_stream_contacts']=pending_before>0
            self._frontier_mark(7,'frontier_headless_troll_drift')
        if self._frontier_stage<8:
            effect=self._frontier_live_effect();issued=settled=False
            if effect:
                _surfaces,receipts=runtime.adult.externalize_causal_groups(effect,0x9900,0)
                issued=bool(receipts)
                settled=bool(receipts) and all(
                    runtime.adult.settle_causal_dialogue_return(
                        receipt,0x9910+index,Q,0,True)
                    for index,receipt in enumerate(receipts))
            self._frontier_receipts['stage8_resident_discourse_action_issued']=issued
            self._frontier_receipts['stage8_independent_reafferent_return_settled']=settled
            self._frontier_mark(8,'frontier_reafferent_arbitration')
        if self._frontier_stage<9:
            self._frontier_apply(9,'body_load',0x9A00,(8,1<<15));last_tick=0
            for index,actions in enumerate(FRONTIER_TYPOLOGY_FORMS.values()):
                last_tick=self._frontier_ambient_post(
                    9,0x9A10+index,f"under load {actions[FRONTIER_ACTION_CALM][0]}")
            blocked=self._frontier_ambient_drain(9,last_tick) if last_tick else ()
            pending=int(runtime.ambient_stream.pending_count)
            self._frontier_apply(9,'quiet',0,(64,))
            if last_tick:self._frontier_ambient_drain(9,last_tick)
            self._frontier_receipts['stage9_load_can_defer_then_recover_same_queued_contacts']=(
                (not blocked or pending>0) and int(runtime.ambient_stream.pending_count)==0)
            self._frontier_mark(9,'frontier_load_recovery')
        if self._frontier_stage<10:
            self._frontier_apply(10,'quiet',0,(1,))
            self._frontier_mark(10,'frontier_heldout_transfer')
        return self.frontier_developmental_receipt()

    # Grounded conversation and extended discourse.
    @staticmethod
    def _frontier_candidate_leaf(adult,scene):
        if scene is None:return None
        try:return adult.language_adult.leaf(int(scene.context),tuple(map(int,scene.atoms)))
        except RuntimeError:
            return adult.language_adult.unique_leaf_for_concepts(tuple(map(int,scene.atoms)))

    def _frontier_focus_after_contact(self,runtime,first_new_identity):
        """Recover focus only from the learned structural occurrence created by contact."""
        adult=runtime.adult;candidates=[]
        for identity in range(int(first_new_identity),int(runtime.contact.next_identity)):
            scene=runtime.contact.scenes.get(identity)
            leaf=self._frontier_candidate_leaf(adult,scene)
            if leaf is not None:
                candidates.append((len(adult.causal_message_rows(leaf.identity)),int(leaf.identity),leaf))
            nested=tuple(runtime.contact.nested_scenes.get(identity,()))
            for row in nested:
                leaf=self._frontier_candidate_leaf(adult,row)
                if leaf is not None:
                    candidates.append((len(adult.causal_message_rows(leaf.identity)),int(leaf.identity),leaf))
            if nested:
                atoms=tuple(int(atom) for row in nested for atom in row.atoms)
                leaf=adult.language_adult.unique_leaf_for_concepts(atoms)
                if leaf is not None:
                    candidates.append((len(adult.causal_message_rows(leaf.identity)),int(leaf.identity),leaf))
        if not candidates:return None
        candidates.sort(key=lambda row:(-row[0],row[1]))
        return candidates[0][2]

    @staticmethod
    def _frontier_stance(adult,leaf_identity,receipts,channel):
        programs=tuple(int(pid) for receipt in receipts for pid in receipt.programs)
        if not programs:return 0,None,b'',0,0
        context=adult._causal_dialogue_appraisal_context(programs,int(channel))
        credited=adult._causal_expression_credit_program(programs[0],context) or programs[0]
        felt=adult.language_adult.somatic_appraisal(credited,context)
        state_surface=bytes(adult.language_adult.realize_somatic_appraisal(
            credited,context) or b'')
        rows=tuple(adult.causal_message_rows(int(leaf_identity)))
        learner=adult.language_adult.world_causal_learning
        blocks=sum(max(0,int(learner.complete_source_blocks(int(row[4])))) for row in rows)
        epistemic=(Q*blocks//(blocks+len(rows))) if rows else 0
        uptake=sum(max(0,int(adult.causal_dialogue_uptake_support(
            int(channel),int(row[4])))) for row in rows)
        dispute=sum(max(0,int(adult.causal_dialogue_dispute_support(
            int(channel),int(row[4])))) for row in rows)
        social=max(-Q,min(Q,Q*(uptake-dispute)//(uptake+dispute+1)))
        stance=max(-Q,min(Q,
            int(felt.valence_q16)
            +(int(felt.controllability_q16)-Q//2)
            -int(felt.interference_q16)
            -int(felt.pressure_q16)//2
            +(epistemic-Q//2)//2
            +social//4))
        appraisal={
            'valence_q16':int(felt.valence_q16),
            'arousal_q16':int(felt.arousal_q16),
            'interference_q16':int(felt.interference_q16),
            'controllability_q16':int(felt.controllability_q16),
            'pressure_q16':int(felt.pressure_q16),
            'credited_program':int(credited),
            'context':int(context),
        }
        return stance,appraisal,state_surface,epistemic,social

    def frontier_respond(self,raw,source,channel=0):
        """Respond to raw lived contact without prompt matching or host semantic routing.

        The contact itself enters the append-only life as an authenticated utterance.
        Learned inversion supplies candidate semantic leaves.  The leaf with the
        strongest resident causal frontier recruits hierarchical discourse.  Somatic
        state and partner-specific uptake/dispute history control depth/formulation
        through the incumbent Adult; they never create world evidence.
        """
        if self._frontier_stage<10:
            raise RuntimeError('organism:frontier-conversation-before-transfer')
        raw=bytes(raw);source=int(source);channel=max(0,int(channel))
        if not raw or source<=0:raise ValueError('organism:frontier-conversation-contact')
        runtime=self._frontier_runtime();adult=runtime.adult
        effective_channel=channel or source
        before_next=int(runtime.contact.next_identity)
        before_world=self._world_digest(runtime)
        self._frontier_apply(10,'authenticated_utterance',source,tuple(raw))
        focus=self._frontier_focus_after_contact(runtime,before_next)
        world_unchanged=(before_world==self._world_digest(runtime))
        if focus is None:
            return {
                'surface':b'','paragraphs':(),'actions':(),'programs':(),
                'focus_leaf':0,'stance_q16':0,'appraisal':None,
                'appraisal_surface':b'','epistemic_support_q16':0,
                'social_credibility_q16':0,
                'contact_world_truth_unchanged':world_unchanged,
            }
        surfaces,receipts=adult.externalize_causal_groups(
            int(focus.identity),source,effective_channel)
        paragraphs=tuple(bytes(surface) for surface in surfaces)
        programs=tuple(tuple(map(int,receipt.programs)) for receipt in receipts)
        stance,appraisal,state_surface,epistemic,social=self._frontier_stance(
            adult,int(focus.identity),receipts,effective_channel)
        return {
            'surface':b'\n\n'.join(paragraphs),
            'paragraphs':paragraphs,
            'actions':tuple(int(receipt.identity) for receipt in receipts),
            'programs':programs,
            'focus_leaf':int(focus.identity),
            'stance_q16':int(stance),
            'appraisal':appraisal,
            # Diagnostic projection only. It is deliberately not appended to public bytes.
            'appraisal_surface':state_surface,
            'epistemic_support_q16':int(epistemic),
            'social_credibility_q16':int(social),
            'contact_world_truth_unchanged':world_unchanged,
        }

    def frontier_settle_response(self,action_identity,source,outcome_q16,
                                 somatic_q16=0,independent=True):
        """Return lived consequence to exactly one prior public discourse action."""
        runtime=self._frontier_runtime();adult=runtime.adult
        receipt=adult.pending_causal_dialogue_actions.get(int(action_identity))
        if receipt is None:return False
        return bool(adult.settle_causal_dialogue_return(
            receipt,int(source),int(outcome_q16),int(somatic_q16),bool(independent)))

    def _frontier_focus_leaf(self,focus_atoms):
        atoms=tuple(map(int,focus_atoms))
        if not atoms:raise ValueError('organism:frontier-discourse-focus')
        return self._frontier_runtime().adult.language_adult.leaf(100,atoms)

    def frontier_generate_discourse(self,focus_atoms,source,channel=0,paragraph_budget=None):
        """Externalize exactly the groups selected by resident causal/somatic dynamics."""
        if self._frontier_stage<10:raise RuntimeError('organism:frontier-discourse-before-transfer')
        adult=self._frontier_runtime().adult;leaf=self._frontier_focus_leaf(focus_atoms)
        effective_channel=max(0,int(channel)) or max(1,int(source))
        surfaces,receipts=adult.externalize_causal_groups(
            leaf.identity,max(1,int(source)),effective_channel)
        paragraphs=tuple(bytes(surface) for surface in surfaces)
        programs=tuple(tuple(map(int,receipt.programs)) for receipt in receipts)
        return {
            'surface':b'\n\n'.join(paragraphs),'paragraphs':paragraphs,
            'programs':programs,
            'actions':tuple(int(receipt.identity) for receipt in receipts),
            'leaf':int(leaf.identity),
        }

    def frontier_form_stance(self,focus_atoms,source,channel=0,paragraph_budget=None):
        """Externalize grounded discourse and expose its current causal-somatic stance."""
        discourse=self.frontier_generate_discourse(
            focus_atoms,source,channel,paragraph_budget)
        adult=self._frontier_runtime().adult
        receipts=tuple(adult.pending_causal_dialogue_actions.get(identity)
                       for identity in discourse['actions'])
        receipts=tuple(row for row in receipts if row is not None)
        effective_channel=max(0,int(channel)) or max(1,int(source))
        stance,appraisal,state_surface,epistemic,social=self._frontier_stance(
            adult,int(discourse['leaf']),receipts,effective_channel)
        return {
            **discourse,'stance_q16':int(stance),'appraisal':appraisal,
            'appraisal_surface':state_surface,
            'epistemic_support_q16':int(epistemic),
            'social_credibility_q16':int(social),
        }

    def frontier_developmental_receipt(self):
        runtime=self._frontier_runtime();learner=runtime.adult.language_action_affordances
        transfer={};quoted={}
        for name,actions in FRONTIER_TYPOLOGY_FORMS.items():
            transfer[name]={};quoted[name]={}
            for action in (FRONTIER_ACTION_CALM,FRONTIER_ACTION_COUNTER):
                raw=actions[action][2].encode('utf-8')
                rows=learner.candidates(raw);qrows=learner.candidates(b'someone said: '+raw)
                transfer[name][str(action)]=bool(rows and int(rows[0][0])==int(action))
                quoted[name][str(action)]=not bool(qrows)
        stage_coverage={str(stage):str(stage) in self._frontier_stage_cursors
                        for stage in range(1,11)}
        checks={
            'one_runtime_reaches_all_ten_stages':
                self._frontier_stage==10 and all(stage_coverage.values()),
            'canonical_life_and_frontier_share_one_runtime':
                int(self._frontier_receipts.get('canonical_life_cursor',0))>0,
            'six_typologies_share_one_resident_action_learner':
                len(transfer)==6 and learner.factor_count>0,
            'heldout_transfer_across_six_typologies':
                all(all(rows.values()) for rows in transfer.values()),
            'unseen_reported_speech_wrapper_withholds_direct_action_force':
                all(all(rows.values()) for rows in quoted.values()),
            'stage4_unsettled_testimony_does_not_mint_world_truth':
                bool(self._frontier_receipts.get(
                    'stage4_world_truth_unchanged_by_unsettled_testimony')),
            'stage5_repetition_does_not_replace_independent_factor_support':
                bool(self._frontier_receipts.get(
                    'stage5_all_generalizing_factors_keep_independent_sources')),
            'stage7_troll_stream_mints_zero_world_truth':
                bool(self._frontier_receipts.get('stage7_troll_posts_mint_zero_world_truth')),
            'stage8_discourse_action_gets_independent_return':
                bool(self._frontier_receipts.get('stage8_resident_discourse_action_issued'))
                and bool(self._frontier_receipts.get(
                    'stage8_independent_reafferent_return_settled')),
            'stage9_load_recovery_preserves_queued_contact':
                bool(self._frontier_receipts.get(
                    'stage9_load_can_defer_then_recover_same_queued_contacts')),
        }
        remaining_red=[
            'LEGACY_REFERENCE_ORGANISM_AND_MATHEMATICAL_ADULT_STILL_COEXIST_DURING_STRANGLER_MIGRATION',
            'UNLEARNED_OPEN_DOMAIN_WORDS_STILL_FAIL_CLOSED',
            'DIRECT_CUDA_PARITY_FOR_EXTENDED_DISCOURSE',
        ]
        return {
            'schema':'cyber-lagoon.reference-organism-frontier-curriculum.v3',
            'pass':all(checks.values()) and not remaining_red,
            'one_continuing_frontier_runtime':True,
            'stage':int(self._frontier_stage),'cursor':int(runtime.cursor),
            'history_root':str(runtime.history_root()),
            'stage_cursors':dict(self._frontier_stage_cursors),
            'typologies':tuple(FRONTIER_TYPOLOGY_FORMS),
            'heldout_transfer':transfer,'reported_speech_refusal':quoted,
            'checks':checks,'remaining_red':remaining_red,
        }

    # Persistence.
    def checkpoint(self):
        data=super().checkpoint()
        data['joint_plan_control_v1']={
            'schema':JOINT_PLAN_SCHEMA,'state':self.joint_plan_control.checkpoint(),
            'ack_ticket':int(self._joint_ack_ticket)}
        if self._frontier_life is not None:
            data['frontier_curriculum_v2']={
                'schema':FRONTIER_CURRICULUM_SCHEMA,'stage':int(self._frontier_stage),
                'stage_cursors':dict(self._frontier_stage_cursors),
                'receipts':dict(self._frontier_receipts),
                'life':self._frontier_life.checkpoint()}
        return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);out.__class__=cls
        row=data.get('joint_plan_control_v1')
        if row is None:
            out.joint_plan_control=JointPlanControlV1();out._joint_ack_ticket=0
        else:
            if int(row.get('schema',0))!=JOINT_PLAN_SCHEMA:
                raise ValueError('organism:joint-plan-checkpoint')
            out.joint_plan_control=JointPlanControlV1.restore(row['state'])
            out._joint_ack_ticket=int(row.get('ack_ticket',0))
        out._frontier_life=None;out._frontier_stage=0
        out._frontier_stage_cursors={};out._frontier_receipts={}
        frontier=data.get('frontier_curriculum_v2')
        if frontier is None:
            if data.get('frontier_curriculum_v1') is not None:
                raise ValueError('organism:frontier-curriculum-v1-migration-required')
            return out
        if int(frontier.get('schema',0))!=FRONTIER_CURRICULUM_SCHEMA:
            raise ValueError('organism:frontier-curriculum-checkpoint')
        program=canonical_species_program_v2()
        out._frontier_life=ReferenceLifeFunctionRuntimeV2.restore(
            program,copy.deepcopy(frontier['life']))
        out._frontier_stage=max(0,min(10,int(frontier.get('stage',0))))
        out._frontier_stage_cursors={
            str(k):int(v) for k,v in dict(frontier.get('stage_cursors',{})).items()}
        out._frontier_receipts=dict(frontier.get('receipts',{}))
        return out

    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):
        return self.recursive_causal_experiment.reason_reliability_q16(
            int(reason),int(source),int(action),int(regime))

    def contextual_reason_predictive_reliability_q16(
            self,reason,source,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_causal_experiment.reason_reliability_q16(
            int(reason),int(source),int(action),resolved)

    def self_reliability_q16(self,action,regime=0):
        return self.recursive_metacontrol.reliability_q16(int(action),int(regime))

    def contextual_self_reliability_q16(self,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_metacontrol.reliability_q16(int(action),resolved)
