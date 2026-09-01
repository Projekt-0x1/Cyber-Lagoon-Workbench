#!/usr/bin/env python3
"""Fast graph-neutral Adult adapter for language-mastery mechanism tournaments.

The adapter owns the *behavioral* interface only.  Language surface ecology is a
transducer over resident construction identities; causal programs own selection,
consequence history, controllability and hierarchical reuse.  No expected answer,
token objective, transformer, or transcript is resident state.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

from reference_language_learning_v1 import LearnedSurfaceEcologyV1, PIECE_LITERAL, PIECE_PORT
from reference_hierarchical_composition_v1 import HierarchicalRefuse, _identity
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q
from reference_causal_program_chunk_v1 import CausalChunkBankV1
from reference_social_latent_prediction_v1 import SocialLatentPredictionV1
from reference_social_prospection_v1 import SocialProspectionV1
from reference_slow_resource_history_v1 import SlowResourceHistoryV1
from reference_open_world_causal_learning_v1 import OpenWorldCausalLearningV1

MAX_COMPLETED_PARTNER_CONTEXTS=16
MAX_PENDING_PARTNER_ACTIONS=16
PARTNER_ACTION_TTL=256
MAX_COMPLETED_CONTEXT_HISTORY=4
MAX_SOCIAL_TOPIC_RELATIONS=256
SOCIAL_TOPIC_MIN_SOURCES=2
DISCOURSE_COLD_RETRY_TICKS=8

@dataclass(frozen=True)
class AdultStateV1:
    urgency_q16: int = 0
    pressure_q16: int = 0
    relief_q16: int = 0
    relief_authenticated: bool = False


@dataclass(frozen=True)
class SomaticAppraisalV1:
    """Disposable current body/candidate appraisal; never world-truth authority."""
    pressure_q16:int=0
    somatic_q16:int=0
    outcome_q16:int=0
    controllability_q16:int=0
    uncertainty_q16:int=0
    arousal_q16:int=0
    valence_q16:int=0
    interference_q16:int=0


@dataclass(frozen=True)
class ConstructionLeafV1:
    identity:int
    context:int
    surface:tuple[int,...]
    depth:int=0


@dataclass(frozen=True)
class CompositionWitnessV1:
    identity:int
    context:int
    template_identity:int
    child_identities:tuple[int,...]
    depth:int
    surface:tuple[int,...]
    pieces:tuple=()


@dataclass(frozen=True)
class ProgramExpressionStepV1:
    value: int
    ordinal: int


class WitnessSurfaceExpressionV1:
    """Rematerialized motor cursor over one Adult-owned pending commitment."""
    def __init__(self, adult, witness):
        self._adult=adult;self.root_identity=int(witness.identity)
        self._context=int(witness.context);self._channel=int(getattr(adult,'_current_language_channel',0));self._surface=tuple(witness.surface)
        if self._context<=0:raise RuntimeError('adult:expression_context')
        row=adult._pending_public_expression.get((self._context,self.root_identity))
        self._ordinal=0 if row is None else int(row[0]);self._pending=None
        self._validated_epoch=-1
        self._prefix=hashlib.sha256(bytes(self._surface[:self._ordinal]))
        if (self._ordinal<0 or self._ordinal>=len(self._surface)
                or (row is not None and self._prefix.hexdigest()!=row[1])):
            raise RuntimeError('adult:expression_prefix_commitment')
    def emit(self):
        if self._pending is not None:return self._pending
        if self._adult._current_selection_context!=self._context:
            self._validated_epoch=-1;return None
        status=('active' if self._validated_epoch==self._adult._select_epoch
                else self._adult._public_expression_status(
                    self._context,self.root_identity,len(self._surface)))
        if status!='active':
            if status=='superseded':self._adult._discard_public_expression(
                self._context,self.root_identity)
            return None
        self._validated_epoch=self._adult._select_epoch
        if self._ordinal>=len(self._surface):return None
        if not self._adult._can_begin_public_expression(self._context,self.root_identity):return None
        self._pending=ProgramExpressionStepV1(self._surface[self._ordinal],self._ordinal)
        return self._pending
    def reafference(self, step, observed):
        if step!=self._pending or int(observed)!=step.value:return False
        self._prefix.update(bytes((int(observed),)))
        self._ordinal+=1;self._pending=None
        self._adult._commit_public_expression(
            self._context,self.root_identity,self._ordinal,
            self._prefix.hexdigest(),len(self._surface),self._channel)
        return True


class ProgramSurfaceExpressionV1:
    """Ephemeral one-byte stack cursor over the Adult's causal-program factorization."""
    def __init__(self, adult, root_identity):
        self._adult=adult;self.root_identity=int(root_identity);self._context=int(adult._current_selection_context);self._channel=int(getattr(adult,'_current_language_channel',0))
        if adult.programs.factor(self.root_identity) is None:raise KeyError(root_identity)
        self._stack=[[self.root_identity,0]];self._segment=();self._seg_i=0;self._have_seg=False
        self._pending=None;self._ordinal=0;self.repairs=0;self._settled=False
        self._leaf_frames={};self._span_pieces={};self._lexeme_units={}

    def _leaf_frame(self, identity):
        if identity in self._leaf_frames:return self._leaf_frames[identity]
        factor=self._adult.programs.factor(identity)
        if factor is not None and int(factor)<0:lid=-int(factor)
        elif identity in self._adult.programs.chunks:
            self._leaf_frames[identity]=None;return None
        else:lid=int(identity)
        raw=self._adult._surface_leaf_surfaces.get(lid)
        if raw is not None:
            held=('raw',tuple(raw));self._leaf_frames[identity]=held;return held
        tid=self._adult._surface_leaf_family_index.get(lid)
        if tid is None:raise RuntimeError('adult:program_surface_missing_leaf')
        lexeme_ids=self._adult._surface_leaf_families.get(int(tid),{}).get(lid)
        pieces=self._adult.language.historical_template_pieces(int(tid))
        if lexeme_ids is None or pieces is None:raise RuntimeError('adult:program_surface_missing_leaf_factor')
        held=('family',lexeme_ids,pieces);self._leaf_frames[identity]=held;return held

    def _ensure(self):
        while not self._have_seg:
            if not self._stack:return
            ident,cur=self._stack[-1]
            leaf=self._leaf_frame(ident)
            if leaf is not None:
                if leaf[0]=='raw':
                    if cur!=0:self._stack.pop();continue
                    self._stack[-1][1]=1;self._segment=leaf[1];self._seg_i=0;self._have_seg=True;return
                lexeme_ids,pieces=leaf[1],leaf[2]
                if cur>=len(pieces):self._stack.pop();continue
                self._stack[-1][1]=cur+1;piece=pieces[cur]
                if piece.kind==PIECE_LITERAL:
                    if not piece.literal:continue
                    self._segment=tuple(piece.literal);self._seg_i=0;self._have_seg=True;return
                if piece.kind==PIECE_PORT and 0<=piece.port<len(lexeme_ids):
                    lid=int(lexeme_ids[piece.port])
                    units=self._lexeme_units.get(lid)
                    if units is None:
                        units=self._adult.language.historical_lexeme_units(lid)
                        if units is None:raise RuntimeError('adult:program_surface_missing_lexeme')
                        self._lexeme_units[lid]=units
                    if not units:continue
                    self._segment=tuple(units);self._seg_i=0;self._have_seg=True;return
                raise RuntimeError('adult:program_surface_missing_leaf_factor')
            chunk=self._adult.programs.chunks.get(ident)
            tid=self._adult.programs.factor(ident)
            if chunk is None or tid is None or int(tid)<=0:raise RuntimeError('adult:program_surface_missing_program')
            pieces=self._span_pieces.get(ident)
            if pieces is None:
                pieces=self._adult.language.historical_span_pieces(int(tid))
                if pieces is None:raise RuntimeError('adult:program_surface_missing_template')
                self._span_pieces[ident]=pieces
            if cur>=len(pieces):self._stack.pop();continue
            self._stack[-1][1]=cur+1;piece=pieces[cur]
            if piece.kind==PIECE_LITERAL:
                if not piece.literal:continue
                self._segment=tuple(piece.literal);self._seg_i=0;self._have_seg=True;return
            if piece.kind==PIECE_PORT and 0<=piece.port<len(chunk.members):
                child=int(chunk.members[piece.port])
                if any(frame[0]==child for frame in self._stack):raise RuntimeError('adult:program_surface_cycle')
                self._stack.append([child,0]);continue
            raise RuntimeError('adult:program_surface_template_piece')

    def emit(self):
        if self._pending is not None:return self._pending
        self._ensure()
        if not self._have_seg:
            if self._ordinal and not self._settled:
                self._adult._settle_public_expression(self.root_identity)
                if self._context>0:
                    self._adult._commit_public_expression(
                        self._context,self.root_identity,self._ordinal,'',self._ordinal,self._channel)
                self._settled=True
            return None
        self._pending=ProgramExpressionStepV1(int(self._segment[self._seg_i]),self._ordinal)
        return self._pending

    def reafference(self,plan,actual_value):
        if self._pending is None or plan!=self._pending:return False
        if (int(actual_value)&255)!=plan.value:
            self.repairs+=1
            return False
        self._pending=None;self._ordinal+=1;self._seg_i+=1
        if self._seg_i>=len(self._segment):self._have_seg=False;self._segment=()
        return True


class LanguageMasteryAdultV1:
    """One continuing simulated Adult behind a stable adapter boundary."""
    def __init__(self):
        self.language = LearnedSurfaceEcologyV1()
        self.programs = CausalChunkBankV1()
        # Program-selection credit is contextual; chunk-internal credit remains
        # the generic temporal/controllability evidence in CausalChunkBankV1.
        self.credit = PredictiveCreditBankV1(256)
        # Generic consequence-qualified selection among learned opaque relation
        # operators. This is acquired Adult state, not a paragraph/topic ontology.
        self.organization_credit = PredictiveCreditBankV1(64)
        # Global discourse selection uses the same causal-credit law on proposition
        # identities; it stores no topic label, semantic score, paragraph or outline.
        self.discourse_credit = PredictiveCreditBankV1(256)
        # Weak source-qualified conversational salience. It gates only proposition
        # matter already supplied by the current lived-world frontier.
        self._social_topic_sources = {}
        # Partner-local pragmatic consequence is a second factor beside transferable
        # social relation credit, never a replacement partner-specific context/answer map.
        self.partner_credit = PredictiveCreditBankV1(256)
        # Opaque sensory sources must earn a unique resident social-identity anchor.
        # The caller's agent integer never addresses partner-local memory directly.
        self.social_identity_credit = PredictiveCreditBankV1(128)
        # Content-free organism-wide load history; language/program identities never enter it.
        self.slow_resource_history = SlowResourceHistoryV1()
        self.world_causal_learning = OpenWorldCausalLearningV1()
        self.social = SocialLatentPredictionV1(min_support=2)
        self.prospection = SocialProspectionV1()
        self._current_selection_context = 0
        # One current body/world settlement over the resident-selected program.
        # It is occurrence state: learned role/dependency relations persist, while
        # the next authenticated situation rematerializes these opaque conditions.
        self._current_program_role_occurrence = ()
        # One uniquely reconstructed partner return may participate in the
        # current response construction. It is disposable occurrence matter,
        # never checkpointed answer/transcript state.
        self._current_language_reply_binding = ()
        # Unsettled structural-role eligibility: template, port, answer source, partner context, selected Program.
        self._pending_span_reply_role = ()
        self._pending_span_reply_actions = {}
        self._current_partner_action_ticket = 0
        self._current_partner_action = ()
        self._next_partner_action_ticket = 1
        self._pending_episode_retrieval = ()
        self._current_partner_context = 0
        self._current_social_source = 0
        self._current_social_subject = 0
        self._current_social_agent = 0
        self._current_social_tick = 0
        # One recent opaque source occurrence is transient coincidence matter, not identity truth.
        self._recent_social_source = 0
        self._recent_social_subject = 0
        self._recent_social_tick = 0
        self._surface_leaf_surfaces = {}
        self._surface_leaf_families = {}
        self._surface_leaf_family_index = {}
        self._surface_leaf_concept_index = {}
        self._leaf_surface_hold = {}
        self._public_surface_hold = {}
        self._surface_hold_epoch = self.language._support_epoch
        self._template_exemplars = {}
        # An unresolved learned structural competition can keep settling across
        # quiet organism ticks. Keep alternatives and decision state, not raw
        # wording, a transcript, a host-authored label, or a fixed turn ladder.
        self._pending_language_competition = ()
        self._language_competition_evidence_q16 = 0
        self._language_competition_leader = 0
        self._language_competition_active = False
        self._language_competition_expressed = False
        self._language_competition_channel = 0
        # A body source retains only its acknowledged event boundary and a
        # commitment to this Adult's last reafferent action. Raw contacts and
        # transcript-shaped transport history are never resident state.
        self._body_ingress = {}
        # A few genuinely pending public commitments survive body interruption.
        # The surface and cursor object do not: they are rematerialized from learned
        # relations and checked against this compact prefix commitment.
        self._pending_public_expression = {}
        # One opaque context of the most recently fully reafferent public action.
        # It is future-relevant event relation, never a surface/transcript cache.
        self._last_completed_public_context = 0
        self._last_completed_public_plan = 0
        self._last_completed_public_context_by_channel = {}
        self._last_completed_public_plan_by_channel = {}
        self._completed_public_context_history_by_channel = {}
        self._completed_public_episode_history_by_channel = {}
        self._completed_public_episode_retention = {}
        self._current_language_channel = 0
        self._conversation_move_episodes = {}
        self.last_public_expression_revalidations = 0
        self.last_public_expression_revision_hash_bytes = 0
        self._select_epoch=0;self._cached_select_key=None;self._cached_select=0;self.last_select_touches=0
        self._productive_leaves={}
        self._tick = 1

    def _advance(self, amount=1):
        before=self.slow_resource_history.pressure_q16()
        self._tick += max(1, int(amount))
        after=self.slow_resource_history.advance(self._tick)
        if after!=before:self._select_epoch+=1
        return self._tick

    def _effective_pressure_q16(self, state: AdultStateV1):
        return min(Q,max(0,int(state.pressure_q16))+self.slow_resource_history.pressure_q16())

    def structural_reply_port(self, template_identity, state=AdultStateV1()):
        """Resource-modulate use of learned structural evidence without rewriting it."""
        port=self.language.span_reply_port(template_identity)
        if port is None:return None
        pressure=self._effective_pressure_q16(state)
        if pressure<=0:return int(port)
        support=self.language.span_reply_support(template_identity,port)
        if support<self.language.minimum_source_support:return None
        confidence=(Q*support)//(support+1)
        return int(port) if pressure<=confidence else None

    @staticmethod
    def _somatic_appraisal_language_context():
        return _identity('adult-somatic-appraisal-language-context-v1',(3,))

    @staticmethod
    def _somatic_appraisal_feature(axis,band):
        return _identity('adult-somatic-appraisal-feature-v1',(int(axis),int(band)))

    def somatic_appraisal_atoms(self, appraisal):
        """Opaque learned-language access to current appraisal geometry.

        Axes are numeric mechanism coordinates, not emotion categories.  Their surface
        meaning exists only if ordinary language experience has named the resulting
        feature identities.
        """
        row=appraisal
        return (
            self._somatic_appraisal_feature(1,1 if int(row.valence_q16)>=0 else 0),
            self._somatic_appraisal_feature(2,1 if int(row.controllability_q16)>=Q//2 else 0),
            self._somatic_appraisal_feature(3,1 if int(row.pressure_q16)>=Q//2 else 0),
        )

    def realize_somatic_appraisal(self, program_identity, context=0, state=AdultStateV1()):
        appraisal=self.somatic_appraisal(program_identity,context,state)
        atoms=self.somatic_appraisal_atoms(appraisal)
        return self.language.realize(self._somatic_appraisal_language_context(),atoms)

    def somatic_appraisal(self, program_identity, context=0, state=AdultStateV1()):
        """Rematerialize a multi-coordinate felt-state closure for one current candidate.

        This is a transient Network-of-Networks view over body/resource history,
        candidate-local somatic/outcome history, controllability and uncertainty.
        Valence/arousal are derived coordinates for arbitration; neither can certify
        world truth and neither replaces the contributing dimensions.
        """
        pid=int(program_identity);context=int(context);pressure=self._effective_pressure_q16(state)
        row=self.credit.rows.get(pid)
        if row is None:
            relief=max(0,min(Q,int(state.relief_q16))) if state.relief_authenticated else 0
            return SomaticAppraisalV1(pressure_q16=pressure,arousal_q16=pressure,
                                      valence_q16=max(-Q,min(Q,relief-pressure//2)),
                                      interference_q16=pressure)
        if context<=0:context=int(row.last_context)
        local=row.contexts.get(context)
        if local is not None:
            # Value may borrow the globally learned consequence when this partner/context
            # has not yet returned an outcome. Controllability may not: local action/background
            # history is the causal authority for whether this context has ever been controllable.
            outcome=int(local.outcome_mean_q16) if local.outcome_samples>0 else int(row.outcome_mean_q16)
            somatic=int(local.somatic_mean_q16) if local.outcome_samples>0 else int(row.somatic_mean_q16)
            current_control=int(local.controllability_q16) if local.control_ready else 0
            control=max(current_control,int(local.control_history_q16))
        else:
            outcome=int(row.outcome_mean_q16);somatic=int(row.somatic_mean_q16)
            current_control=int(row.controllability_q16) if row.control_ready else 0
            control=max(current_control,int(row.control_history_q16))
        control=max(0,min(Q,control))
        uncertainty=max(0,min(Q,int(row.uncertainty_q16)))
        relief=max(0,min(Q,int(state.relief_q16))) if state.relief_authenticated else 0
        # Learned action control does not delete organism-wide load. It changes how
        # much of that same load can interfere with this already-learned candidate.
        # This interaction is rematerialized, not persisted as another stress register.
        interference=(pressure*(Q-control)+Q-1)//Q if pressure else 0
        arousal=min(Q,interference+min(Q,abs(somatic)//2+uncertainty//4))
        valence=max(-Q,min(Q,outcome+somatic+relief-interference//2))
        return SomaticAppraisalV1(pressure,somatic,outcome,control,uncertainty,arousal,valence,interference)

    def checkpoint(self):
        """Future-relevant learned state, excluding the active contact occurrence.

        Lookup indices and the current selection context are rematerialized from
        durable learned relations and the next authenticated body contact.
        """
        return {'schema':3,'tick':self._tick,
                'language':self.language.checkpoint(),
                'programs':self.programs.checkpoint(),
                'program_surfaces':self.program_surface_checkpoint(),
                'selection_credit':self.credit.checkpoint(),
                'organization_credit':self.organization_credit.checkpoint(),
                'discourse_credit':self.discourse_credit.checkpoint(),
                'social_topic_sources':[{'context':c,'proposition':p,'sources':sorted(srcs)}
                                        for (c,p),srcs in sorted(self._social_topic_sources.items())],
                'partner_credit':self.partner_credit.checkpoint(),
                'social_identity_credit':self.social_identity_credit.checkpoint(),
                'slow_resource_history':self.slow_resource_history.checkpoint(),
                'world_causal_learning':self.world_causal_learning.checkpoint(),
                'social_latent':self.social.checkpoint(),
                'prospection':self.prospection.checkpoint(),
                'pending_language_competition':[
                    {'context':context,'atoms':list(atoms)}
                    for context,atoms in self._pending_language_competition],
                'pending_language_evidence':self._language_competition_evidence_q16,
                'pending_language_leader':self._language_competition_leader,
                'pending_language_active':self._language_competition_active,
                'pending_language_expressed':self._language_competition_expressed,
                'pending_language_channel':self._language_competition_channel,
                'pending_span_reply_role':(
                    {'roles':[{'template':tid,'port':port,'source':source}
                              for tid,port,source in self._pending_span_reply_role[0]],
                     'partner_context':self._pending_span_reply_role[1],
                     'selected_program':self._pending_span_reply_role[2]}
                    if self._pending_span_reply_role else None),
                'current_partner_action_ticket':int(self._current_partner_action_ticket),
                'current_partner_action':({'partner_context':self._current_partner_action[0],'selected_program':self._current_partner_action[1]} if self._current_partner_action else None),
                'next_partner_action_ticket':int(self._next_partner_action_ticket),
                'pending_span_reply_actions':[
                    {'ticket':ticket,'roles':[{'template':tid,'port':port,'source':source} for tid,port,source in roles],
                     'partner_context':partner_context,'selected_program':selected,'born_tick':born}
                    for ticket,(roles,partner_context,selected,born) in sorted(self._pending_span_reply_actions.items())],
                'body_ingress':[{'source':source,'sequence':sequence,'reafference':reafference}
                                for source,(sequence,reafference) in sorted(self._body_ingress.items())],
                'pending_public_expression':[
                    {'context':context,'plan':plan,'ordinal':ordinal,'prefix':prefix}
                    for (context,plan),(ordinal,prefix) in sorted(
                        self._pending_public_expression.items())],
                'last_completed_public_context':int(self._last_completed_public_context),
                'last_completed_public_plan':int(self._last_completed_public_plan),
                'last_completed_public_context_by_channel':[{'channel':c,'context':x} for c,x in sorted(self._last_completed_public_context_by_channel.items())],
                'last_completed_public_plan_by_channel':[{'channel':c,'plan':x} for c,x in sorted(self._last_completed_public_plan_by_channel.items())],
                'completed_public_context_history_by_channel':[{'channel':c,'contexts':list(rows)} for c,rows in sorted(self._completed_public_context_history_by_channel.items())],
                'completed_public_episode_history_by_channel':[{'channel':c,'episodes':[{'context':ctx,'plan':plan} for ctx,plan in rows]} for c,rows in sorted(self._completed_public_episode_history_by_channel.items())],
                'completed_public_episode_retention':[{'channel':c,'context':ctx,'plan':plan,'score':score} for (c,ctx,plan),score in sorted(self._completed_public_episode_retention.items()) if score],
                'conversation_move_episodes':[{'factors':list(f),'plan':p,'source':src,'evidence':ev} for (f,p,src),ev in sorted(self._conversation_move_episodes.items())],
                'template_exemplars':[{'template':tid,'examples':[list(x) for x in sorted(rows)]}
                                      for tid,rows in sorted(self._template_exemplars.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=3:raise RuntimeError('adult:checkpoint_schema')
        adult=cls();adult.language=LearnedSurfaceEcologyV1.restore(data.get('language',{}))
        adult.programs=CausalChunkBankV1.restore(data.get('programs',{}))
        adult.restore_program_surface_checkpoint(data.get('program_surfaces',{}))
        adult.credit=PredictiveCreditBankV1.restore(data.get('selection_credit',{}))
        organization=data.get('organization_credit')
        adult.organization_credit=(PredictiveCreditBankV1(64) if organization is None
                                  else PredictiveCreditBankV1.restore(organization))
        discourse=data.get('discourse_credit')
        adult.discourse_credit=(PredictiveCreditBankV1(256) if discourse is None
                               else PredictiveCreditBankV1.restore(discourse))
        social_topics={}
        for row in data.get('social_topic_sources',()):
            context=int(row.get('context',0));proposition=int(row.get('proposition',0));sources=set(map(int,row.get('sources',())))
            key=(context,proposition)
            if context<=0 or proposition<=0 or not sources or any(source<=0 for source in sources) or key in social_topics or not adult._has_leaf(proposition):raise RuntimeError('adult:checkpoint_social_topic')
            social_topics[key]=sources
        if len(social_topics)>MAX_SOCIAL_TOPIC_RELATIONS:raise RuntimeError('adult:checkpoint_social_topic_capacity')
        adult._social_topic_sources=social_topics
        partner=data.get('partner_credit')
        adult.partner_credit=(PredictiveCreditBankV1(256) if partner is None
                             else PredictiveCreditBankV1.restore(partner))
        adult.social_identity_credit=PredictiveCreditBankV1.restore(
            data.get('social_identity_credit',{}))
        slow=data.get('slow_resource_history')
        adult.slow_resource_history=(SlowResourceHistoryV1() if slow is None
                                    else SlowResourceHistoryV1.restore(slow))
        world=data.get('world_causal_learning')
        adult.world_causal_learning=(OpenWorldCausalLearningV1() if world is None
                                    else OpenWorldCausalLearningV1.restore(world))
        adult.social=SocialLatentPredictionV1.restore(data.get('social_latent',{}))
        adult.prospection=SocialProspectionV1.restore(data.get('prospection',{}))
        adult._tick=int(data.get('tick',0))
        if adult._tick<1:raise RuntimeError('adult:checkpoint_tick')
        ingress={}
        for row in data.get('body_ingress',()):
            source=row.get('source');sequence=int(row.get('sequence',0))
            reafference=row.get('reafference')
            if (not isinstance(source,str) or not source or len(source)>128
                    or source in ingress or sequence<=0
                    or not isinstance(reafference,str) or len(reafference)!=64
                    or any(unit not in '0123456789abcdef' for unit in reafference)):
                raise RuntimeError('adult:checkpoint_body_ingress')
            ingress[source]=(sequence,reafference)
        if len(ingress)>64:raise RuntimeError('adult:checkpoint_body_ingress_capacity')
        adult._body_ingress=ingress
        public={}
        for row in data.get('pending_public_expression',()):
            context=int(row.get('context',0));plan=int(row.get('plan',0))
            ordinal=int(row.get('ordinal',0));prefix=row.get('prefix')
            key=(context,plan)
            if (min(context,plan,ordinal)<=0 or key in public
                    or not isinstance(prefix,str) or len(prefix)!=64
                    or any(unit not in '0123456789abcdef' for unit in prefix)):
                raise RuntimeError('adult:checkpoint_public_expression')
            public[key]=(ordinal,prefix)
        if len(public)>4:raise RuntimeError('adult:checkpoint_public_expression_capacity')
        adult._pending_public_expression=public
        completed=int(data.get('last_completed_public_context',0))
        if completed<0:raise RuntimeError('adult:checkpoint_completed_public_context')
        adult._last_completed_public_context=completed
        completed_plan=int(data.get('last_completed_public_plan',0))
        if completed_plan<0:raise RuntimeError('adult:checkpoint_completed_public_plan')
        adult._last_completed_public_plan=completed_plan
        completed_by_channel={}
        for row in data.get('last_completed_public_context_by_channel',()):
            channel=int(row.get('channel',0));context=int(row.get('context',0))
            if channel<=0 or context<=0 or channel in completed_by_channel:raise RuntimeError('adult:checkpoint_completed_public_partner_context')
            completed_by_channel[channel]=context
        if len(completed_by_channel)>MAX_COMPLETED_PARTNER_CONTEXTS:raise RuntimeError('adult:checkpoint_completed_public_partner_capacity')
        adult._last_completed_public_context_by_channel=completed_by_channel
        completed_plan_by_channel={}
        for row in data.get('last_completed_public_plan_by_channel',()):
            channel=int(row.get('channel',0));plan=int(row.get('plan',0))
            if channel<=0 or plan<=0 or channel in completed_plan_by_channel:
                raise RuntimeError('adult:checkpoint_completed_public_partner_plan')
            completed_plan_by_channel[channel]=plan
        if len(completed_plan_by_channel)>MAX_COMPLETED_PARTNER_CONTEXTS:raise RuntimeError('adult:checkpoint_completed_public_partner_plan_capacity')
        adult._last_completed_public_plan_by_channel=completed_plan_by_channel
        completed_history={}
        for row in data.get('completed_public_context_history_by_channel',()):
            channel=int(row.get('channel',0));contexts=tuple(map(int,row.get('contexts',())))
            if (channel<=0 or channel in completed_history or not contexts or len(contexts)>MAX_COMPLETED_CONTEXT_HISTORY
                    or any(context<=0 for context in contexts) or (channel in completed_by_channel and contexts[-1]!=completed_by_channel[channel])):
                raise RuntimeError('adult:checkpoint_completed_public_history')
            completed_history[channel]=contexts
        if len(completed_history)>MAX_COMPLETED_PARTNER_CONTEXTS:raise RuntimeError('adult:checkpoint_completed_public_history_capacity')
        adult._completed_public_context_history_by_channel=completed_history
        episode_history={}
        for row in data.get('completed_public_episode_history_by_channel',()):
            channel=int(row.get('channel',0));episodes=tuple((int(ep.get('context',0)),int(ep.get('plan',0))) for ep in row.get('episodes',()))
            if (channel<=0 or channel in episode_history or not episodes or len(episodes)>MAX_COMPLETED_CONTEXT_HISTORY
                    or any(ctx<=0 or plan<=0 or (adult.programs.factor(plan) is None and not adult._has_leaf(plan)) for ctx,plan in episodes)):
                raise RuntimeError('adult:checkpoint_completed_public_episode_history')
            episode_history[channel]=episodes
        if len(episode_history)>MAX_COMPLETED_PARTNER_CONTEXTS:raise RuntimeError('adult:checkpoint_completed_public_episode_history_capacity')
        adult._completed_public_episode_history_by_channel=episode_history
        retention={}
        live_episode_keys={(channel,ctx,plan) for channel,episodes in episode_history.items() for ctx,plan in episodes}
        for row in data.get('completed_public_episode_retention',()):
            key=(int(row.get('channel',0)),int(row.get('context',0)),int(row.get('plan',0)));score=int(row.get('score',0))
            if key not in live_episode_keys or key in retention or score==0 or not -8<=score<=8:raise RuntimeError('adult:checkpoint_completed_public_episode_retention')
            retention[key]=score
        adult._completed_public_episode_retention=retention
        moves={}
        for row in data.get('conversation_move_episodes',()):
            factors=tuple(map(int,row.get('factors',())));plan=int(row.get('plan',0));source=int(row.get('source',0));ev=int(row.get('evidence',0));key=(factors,plan,source)
            if not 4<=len(factors)<=14 or any(v<0 for v in factors) or plan<=0 or source<=0 or not -8<=ev<=8 or ev==0 or key in moves or (adult.programs.factor(plan) is None and not adult._has_leaf(plan)):raise RuntimeError('adult:checkpoint_conversation_move_episode')
            moves[key]=ev
        if len(moves)>256:raise RuntimeError('adult:checkpoint_conversation_move_capacity')
        adult._conversation_move_episodes=moves
        pending=tuple(sorted((int(row.get('context',0)),tuple(map(int,row.get('atoms',()))))
                             for row in data.get('pending_language_competition',())))
        if (pending and (len(pending)<2 or pending!=tuple(sorted(set(pending)))
                         or any(context<=0 or not atoms
                                or adult.language.template(context,len(atoms)) is None
                                or any(not adult.language.lexeme_candidates(atom) for atom in atoms)
                                for context,atoms in pending))):
            raise RuntimeError('adult:checkpoint_language_competition')
        adult._pending_language_competition=pending
        evidence=int(data.get('pending_language_evidence',0))
        leader=int(data.get('pending_language_leader',0))
        active=bool(data.get('pending_language_active',False))
        expressed=bool(data.get('pending_language_expressed',False))
        channel=int(data.get('pending_language_channel',0))
        if (evidence<0 or evidence>Q or (not pending and (evidence or leader or active))
                or (leader and adult.programs.factor(leader) is None)
                or channel<0 or (not pending and channel)
                or (expressed and (not pending or active or not leader))):
            raise RuntimeError('adult:checkpoint_language_competition_state')
        adult._language_competition_evidence_q16=evidence
        adult._language_competition_leader=leader
        adult._language_competition_active=active
        adult._language_competition_expressed=expressed
        adult._language_competition_channel=channel
        role=data.get('pending_span_reply_role')
        pending_role=()
        if role is not None:
            partner_context=int(role.get('partner_context',0));selected=int(role.get('selected_program',0))
            raw_roles=role.get('roles')
            if raw_roles is None:
                raw_roles=({'template':role.get('template',0),'port':role.get('port',-1),'source':role.get('source',0)},)
            rows=[]
            for item in raw_roles:
                tid=int(item.get('template',0));port=int(item.get('port',-1));source=int(item.get('source',0))
                pieces=adult.language._span_identity_pieces.get(tid)
                if (tid<=0 or source<=0 or pieces is None
                        or port not in {int(piece.port) for piece in pieces if piece.kind==PIECE_PORT}):
                    raise RuntimeError('adult:checkpoint_pending_span_reply_role')
                rows.append((tid,port,source))
            rows=tuple(sorted(set(rows)))
            if (not 1<=len(rows)<=4 or len(rows)!=len(raw_roles) or partner_context<=0 or selected<=0
                    or (adult.programs.factor(selected) is None and not adult._has_leaf(selected))):
                raise RuntimeError('adult:checkpoint_pending_span_reply_role')
            pending_role=(rows,partner_context,selected)
        adult._pending_span_reply_role=pending_role
        next_ticket=int(data.get('next_partner_action_ticket',1));current_ticket=int(data.get('current_partner_action_ticket',0));current_action=()
        raw_current_action=data.get('current_partner_action')
        if raw_current_action is not None:
            ca_context=int(raw_current_action.get('partner_context',0));ca_selected=int(raw_current_action.get('selected_program',0))
            if ca_context<=0 or ca_selected<=0 or (adult.programs.factor(ca_selected) is None and not adult._has_leaf(ca_selected)):raise RuntimeError('adult:checkpoint_partner_action')
            current_action=(ca_context,ca_selected)
        if next_ticket<=0 or current_ticket<0 or bool(current_ticket)!=bool(current_action):raise RuntimeError('adult:checkpoint_partner_action_ticket')
        if current_ticket and current_ticket>=next_ticket:raise RuntimeError('adult:checkpoint_partner_action_ticket')
        if pending_role and current_action and (pending_role[1],pending_role[2])!=current_action:raise RuntimeError('adult:checkpoint_partner_action_role_mismatch')
        bank={}
        for item in data.get('pending_span_reply_actions',()):
            ticket=int(item.get('ticket',0));partner_context=int(item.get('partner_context',0));selected=int(item.get('selected_program',0));born=int(item.get('born_tick',0));raw_roles=item.get('roles',());rows=[]
            for role_item in raw_roles:
                tid=int(role_item.get('template',0));port=int(role_item.get('port',-1));source=int(role_item.get('source',0));pieces=adult.language._span_identity_pieces.get(tid)
                if tid<=0 or source<=0 or pieces is None or port not in {int(piece.port) for piece in pieces if piece.kind==PIECE_PORT}:raise RuntimeError('adult:checkpoint_pending_partner_action')
                rows.append((tid,port,source))
            rows=tuple(sorted(set(rows)))
            if (ticket<=0 or ticket>=next_ticket or ticket==current_ticket or ticket in bank or not 0<=len(rows)<=4 or len(rows)!=len(raw_roles) or partner_context<=0 or selected<=0 or born<=0 or (adult.programs.factor(selected) is None and not adult._has_leaf(selected))):raise RuntimeError('adult:checkpoint_pending_partner_action')
            bank[ticket]=(rows,partner_context,selected,born)
        if len(bank)>MAX_PENDING_PARTNER_ACTIONS:raise RuntimeError('adult:checkpoint_pending_partner_action_capacity')
        adult._pending_span_reply_actions=bank;adult._current_partner_action_ticket=current_ticket;adult._current_partner_action=current_action;adult._next_partner_action_ticket=next_ticket
        exemplars={}
        for row in data.get('template_exemplars',()):
            tid=int(row.get('template',0));examples=set(tuple(map(int,x)) for x in row.get('examples',()))
            bindings=[(c,a,p) for c,a,p in adult.language._template_sources
                      if adult.language.construction_factor_identity(c,a,p)==tid]
            binding=bindings[0] if len(bindings)==1 else None
            span=adult.language._span_identity_pieces.get(tid)
            arity=(int(binding[1]) if binding is not None else
                   len({int(piece.port) for piece in span if piece.kind==PIECE_PORT})
                   if span is not None else 0)
            if (tid<=0 or tid in exemplars or arity<=0 or not examples or
                    any(len(x)!=arity for x in examples)):
                raise RuntimeError('adult:checkpoint_exemplar')
            exemplars[tid]=examples
        adult._template_exemplars=exemplars
        # An active situation is an occurrence, not learned connectivity. The
        # resumed organism must receive contact before selecting public action.
        adult._current_selection_context=(
            adult._language_competition_context(pending) if pending else 0)
        if pending_role:
            if pending:raise RuntimeError('adult:checkpoint_competing_pending_causes')
            adult._current_partner_context=pending_role[1]
        elif current_action:
            if pending:raise RuntimeError('adult:checkpoint_competing_pending_causes')
            adult._current_partner_context=current_action[0]
        return adult

    def body_ingress_cursor(self, source):
        if not isinstance(source,str) or not source or len(source)>128:
            raise ValueError('adult:body_source')
        return self._body_ingress.get(source,(0,''))

    def settle_body_ingress(self, source, sequence, reafference, resource_load_q16):
        previous,_=self.body_ingress_cursor(source);sequence=int(sequence)
        try:load=int(resource_load_q16)
        except (TypeError,ValueError):raise ValueError('adult:body_resource')
        if (sequence<=previous or not isinstance(reafference,str) or len(reafference)!=64
                or any(unit not in '0123456789abcdef' for unit in reafference)):
            raise ValueError('adult:body_sequence')
        if load<0:raise ValueError('adult:body_resource')
        if source not in self._body_ingress and len(self._body_ingress)>=64:
            raise ValueError('adult:body_source_capacity')
        self._advance();before=self.slow_resource_history.pressure_q16()
        after=(self.slow_resource_history.observe(self._tick,load)
               if load>0 else self.slow_resource_history.pressure_q16())
        if after!=before:self._select_epoch+=1
        self._body_ingress[source]=(sequence,reafference)
        return after

    def _authenticated_body_occurrence(self, source, sequence, contact_commitment):
        """Resolve an exact current ingress cursor; it carries no truth semantics."""
        if self.body_ingress_cursor(source)!=(int(sequence),str(contact_commitment)):
            raise HierarchicalRefuse('adult:unauthenticated_body_occurrence')
        return _identity('adult-authenticated-body-occurrence-v1',
                         (str(source),int(sequence),str(contact_commitment)))

    def _register_surface_template(self, template_identity, pieces=None):
        tid=int(template_identity);historical=self.language.historical_span_pieces(tid)
        if not tid or historical is None:raise RuntimeError('adult:program_surface_template')
        if pieces is not None and tuple(pieces)!=tuple(historical):
            raise RuntimeError('adult:program_surface_template_collision')

    def _has_leaf(self, identity):
        identity=int(identity)
        return identity in self._surface_leaf_surfaces or identity in self._surface_leaf_family_index

    def leaf_signature(self, identity):
        identity=int(identity);tid=self._surface_leaf_family_index.get(identity)
        if tid is None:return None
        lexemes=self._surface_leaf_families.get(int(tid),{}).get(identity);template=self.language.historical_template_binding(int(tid))
        if lexemes is None or template is None:return None
        concepts=[]
        for lexeme in lexemes:
            binding=self.language.historical_lexeme_binding(int(lexeme))
            if binding is None:return None
            concepts.append(int(binding[0]))
        return int(template[0]),tuple(concepts)
    def unique_leaf_for_concepts(self,concepts):
        """Recover one current productive proposition from ordered resident participants."""
        concepts=tuple(map(int,concepts))
        if not concepts:return None
        banks=tuple(self._surface_leaf_concept_index.get(concept,set()) for concept in concepts)
        if any(not bank for bank in banks):return None
        candidates=set.intersection(*(set(bank) for bank in banks));rows=[]
        for identity in candidates:
            signature=self.leaf_signature(int(identity))
            if signature is None or tuple(signature[1])!=concepts:continue
            try:rows.append(self.leaf(signature[0],signature[1]))
            except (RuntimeError,HierarchicalRefuse):continue
        return rows[0] if len(rows)==1 else None
    def leaf_equivalent(self,left,right):
        left=int(left);right=int(right)
        if left==right:return True
        a=self.leaf_signature(left);b=self.leaf_signature(right)
        return a is not None and a==b
    def current_leaf_for_historical(self,identity):
        signature=self.leaf_signature(int(identity))
        if signature is None:return None
        try:return self.leaf(signature[0],signature[1])
        except (RuntimeError,HierarchicalRefuse):return None

    def refresh_lexeme_identity(self,lexeme_identity):
        """Locally rematerialize stable propositions touched by one lexeme revision."""
        binding=self.language.historical_lexeme_binding(int(lexeme_identity))
        if binding is None:return 0
        concept=int(binding[0]);units=self.language.lexeme(concept)
        if units is None:return 0
        current=self.language.lexeme_identity(concept,units);changed=0
        for identity in tuple(self._surface_leaf_concept_index.get(concept,())):
            tid=self._surface_leaf_family_index.get(int(identity))
            lexemes=self._surface_leaf_families.get(int(tid),{}).get(int(identity))
            if lexemes is None:continue
            revised=list(lexemes)
            for slot,held in enumerate(lexemes):
                row=self.language.historical_lexeme_binding(int(held))
                if row is not None and int(row[0])==concept:revised[slot]=current
            revised=tuple(revised)
            if revised==lexemes:continue
            self._surface_leaf_families[int(tid)][int(identity)]=revised
            self._leaf_surface_hold.pop(int(identity),None);changed+=1
        if changed:self._public_surface_hold.clear()
        return changed

    def _sync_surface_holds(self):
        """Historical-factor holds survive unrelated current-support revision."""
        epoch=self.language._support_epoch
        if self._surface_hold_epoch!=epoch:
            self._surface_hold_epoch=epoch

    def _leaf_surface(self, identity):
        self._sync_surface_holds()
        identity=int(identity);raw=self._surface_leaf_surfaces.get(identity)
        if raw is not None:return tuple(raw)
        held=self._leaf_surface_hold.get(identity)
        if held is not None:return held
        tid=self._surface_leaf_family_index.get(identity)
        if tid is None:raise RuntimeError('adult:program_surface_missing_leaf')
        lexeme_ids=self._surface_leaf_families.get(int(tid),{}).get(identity)
        pieces=self.language.historical_template_pieces(int(tid))
        if lexeme_ids is None or pieces is None:raise RuntimeError('adult:program_surface_missing_leaf_factor')
        surfaces=[]
        for lexeme_identity in lexeme_ids:
            units=self.language.historical_lexeme_units(int(lexeme_identity))
            if units is None:raise RuntimeError('adult:program_surface_missing_lexeme')
            surfaces.append(tuple(units))
        rendered=self._render_pieces(pieces,tuple(surfaces))
        if not rendered:raise RuntimeError('adult:program_surface_missing_leaf')
        self._leaf_surface_hold[identity]=rendered;return rendered

    def _leaf_extent(self, identity):
        identity=int(identity);raw=self._surface_leaf_surfaces.get(identity)
        if raw is not None:return len(raw)
        tid=self._surface_leaf_family_index.get(identity)
        if tid is None:raise RuntimeError('adult:program_surface_missing_leaf')
        lexeme_ids=self._surface_leaf_families.get(int(tid),{}).get(identity)
        pieces=self.language.historical_template_pieces(int(tid))
        if lexeme_ids is None or pieces is None:raise RuntimeError('adult:program_surface_missing_leaf_factor')
        n=0
        for piece in pieces:
            if piece.kind==PIECE_LITERAL:n+=len(piece.literal)
            elif piece.kind==PIECE_PORT and 0<=piece.port<len(lexeme_ids):
                units=self.language.historical_lexeme_units(int(lexeme_ids[piece.port]))
                if units is None:raise RuntimeError('adult:program_surface_missing_lexeme')
                n+=len(units)
            else:raise RuntimeError('adult:program_surface_missing_leaf_factor')
        if not n:raise RuntimeError('adult:program_surface_missing_leaf')
        return n

    def _register_program_surface(self, chunk, root):
        if int(getattr(root,'depth',0))<=0 or not getattr(root,'template_identity',0):
            raise RuntimeError('adult:program_surface_root_shape')
        self._register_surface_template(root.template_identity, getattr(root,'pieces',None))
        for member in chunk.members:
            identity=int(member)
            if identity in self.programs.chunks:
                if self.programs.factor(identity) is None:raise RuntimeError('adult:program_surface_child')
                continue
            if self._has_leaf(identity):continue
            raise RuntimeError('adult:program_surface_leaf')
        tid=int(root.template_identity)
        try:self.programs.bind_factor(chunk.identity,tid)
        except RuntimeError as exc:raise RuntimeError('adult:program_surface_collision') from exc

    def _register_atomic_surface(self, program_identity, root):
        pid=int(program_identity)
        if int(getattr(root,'depth',-1))!=0:
            raise RuntimeError('adult:atomic_surface_root')
        rendered=tuple(getattr(root,'surface',()) or getattr(root,'_leaf_surface',()))
        if not rendered:raise RuntimeError('adult:atomic_surface_empty')
        lid=int(root.identity)
        if self._has_leaf(lid):
            if self._leaf_surface(lid)!=rendered:raise RuntimeError('adult:atomic_surface_collision')
        else:
            self._surface_leaf_surfaces[lid]=rendered
        factor=-lid
        try:self.programs.bind_factor(pid,factor)
        except RuntimeError as exc:raise RuntimeError('adult:atomic_program_collision') from exc

    def _program_surface_frame(self, identity, stack):
        identity=int(identity)
        if identity in stack:raise RuntimeError('adult:program_surface_cycle')
        chunk=self.programs.chunks.get(identity)
        if chunk is None:
            factor=self.programs.factor(identity)
            if factor is not None:
                factor=int(factor)
                if factor>=0:raise RuntimeError('adult:program_surface_missing_program')
                return -factor,None,None
            return identity,None,None
        tid=self.programs.factor(identity)
        if tid is None or int(tid)<=0:raise RuntimeError('adult:program_surface_missing_program')
        pieces=self.language.historical_span_pieces(int(tid))
        if pieces is None:raise RuntimeError('adult:program_surface_missing_template')
        return identity,tuple(map(int,chunk.members)),pieces

    def _program_literal_initial(self, identity):
        chunk=self.programs.chunks.get(int(identity))
        if chunk is None:return False
        factor=self.programs.factor(int(identity))
        pieces=(() if factor is None else self.language.historical_span_pieces(int(factor))) or ()
        for piece in pieces:
            if piece.kind==PIECE_LITERAL and piece.literal:return True
            if piece.kind==PIECE_PORT:return False
        return False

    def _rematerialize_program_surface(self, identity, stack=None):
        self._sync_surface_holds()
        identity=int(identity)
        stack=set() if stack is None else stack
        if identity not in stack:
            held=self._public_surface_hold.get(identity)
            if held is not None:return held
        ident,members,pieces=self._program_surface_frame(identity,stack)
        if pieces is None:
            surface=self._leaf_surface(ident)
            self._public_surface_hold[identity]=surface
            return surface
        stack.add(ident)
        try:
            children=tuple(self._rematerialize_program_surface(member,stack) for member in members)
            embedded=self._embedded_port_surfaces(
                pieces,children,tuple(self.program_depth(member) for member in members),
                tuple(self._program_literal_initial(member) for member in members))
            surface=tuple(self._render_pieces(pieces,embedded))
        finally:
            stack.remove(ident)
        self._public_surface_hold[identity]=surface
        return surface

    def _program_surface_extent(self, identity, stack=None):
        return len(self._rematerialize_program_surface(int(identity),stack))

    def program_surface_checkpoint(self):
        return {
            'schema':9,
            'raw_leaf_surfaces':[
                {'identity':k,'surface':list(v)} for k,v in sorted(self._surface_leaf_surfaces.items())],
            'leaf_families':[
                {'template':tid,'leaves':[
                    {'identity':leaf,'lexemes':list(lexemes)} for leaf,lexemes in sorted(rows.items())]}
                for tid,rows in sorted(self._surface_leaf_families.items())],
            'program_factor_state':self.programs.factor_checkpoint(),
        }

    def restore_program_surface_checkpoint(self, data):
        if int(data.get('schema',0))!=9:raise RuntimeError('adult:program_surface_checkpoint_schema')
        raw_leaves={};families={};family_index={};concept_index={};factors={}
        for row in data.get('raw_leaf_surfaces',()):
            identity=int(row['identity']);rendered=tuple(map(int,row.get('surface',())))
            if identity in raw_leaves or not rendered or any(x<0 or x>255 for x in rendered):
                raise RuntimeError('adult:program_surface_checkpoint_leaf')
            raw_leaves[identity]=rendered
        for family_row in data.get('leaf_families',()):
            tid=int(family_row.get('template',0));template_binding=self.language.historical_template_binding(tid)
            if not tid or template_binding is None or tid in families:raise RuntimeError('adult:program_surface_checkpoint_leaf_family')
            context,arity,pieces=template_binding;rows={}
            for row in family_row.get('leaves',()):
                identity=int(row['identity']);lexemes=tuple(map(int,row.get('lexemes',())))
                if identity in raw_leaves or identity in family_index or len(lexemes)!=arity:
                    raise RuntimeError('adult:program_surface_checkpoint_leaf_family')
                bindings=tuple(self.language.historical_lexeme_binding(x) for x in lexemes)
                if any(binding is None for binding in bindings):raise RuntimeError('adult:program_surface_checkpoint_leaf_lexeme')
                concepts=tuple(int(binding[0]) for binding in bindings);surfaces=tuple(tuple(binding[1]) for binding in bindings)
                rendered=self._render_pieces(pieces,surfaces)
                expected=_identity('hierarchy-leaf-v2',(context,concepts))
                if not rendered or expected!=identity:raise RuntimeError('adult:program_surface_checkpoint_leaf_identity')
                rows[identity]=lexemes;family_index[identity]=tid
                for concept in concepts:concept_index.setdefault(int(concept),set()).add(identity)
            if not rows:raise RuntimeError('adult:program_surface_checkpoint_leaf_family')
            families[tid]=rows
        factor_state=data.get('program_factor_state',{})
        if int(factor_state.get('schema',0))!=1:raise RuntimeError('adult:program_surface_checkpoint_program')
        for row in factor_state.get('factors',()):
            pid=int(row['program']);factor=int(row.get('factor',0))
            if pid in factors or factor==0:raise RuntimeError('adult:program_surface_checkpoint_program')
            if factor>0:
                if self.language.historical_span_pieces(factor) is None or pid not in self.programs.chunks:
                    raise RuntimeError('adult:program_surface_checkpoint_program')
            elif -factor not in raw_leaves and -factor not in family_index:
                raise RuntimeError('adult:program_surface_checkpoint_atomic')
            factors[pid]=factor
        for pid,factor in factors.items():
            if factor<0:continue
            for member in self.programs.chunks[pid].members:
                member=int(member)
                if member in self.programs.chunks:continue
                member_factor=factors.get(member)
                leaf_identity=-member_factor if member_factor is not None and member_factor<0 else member
                if leaf_identity not in raw_leaves and leaf_identity not in family_index:
                    raise RuntimeError('adult:program_surface_checkpoint_member')
        self._surface_leaf_surfaces=raw_leaves;self._surface_leaf_families=families;self._surface_leaf_family_index=family_index;self._surface_leaf_concept_index=concept_index
        self._leaf_surface_hold.clear();self._public_surface_hold.clear();self._productive_leaves.clear()
        self._surface_hold_epoch=self.language._support_epoch
        self.programs.restore_factor_checkpoint(factor_state)

    def _observe_surface_item(self, concept, raw, source):
        self.language.observe_naming(int(concept), tuple(map(int,raw)), int(source))
        return True

    def observe_surface_item(self, concept, raw, source):
        """Behavioral adapter: one externally heard/seen surface-item contact."""
        return self._observe_surface_item(concept, raw, source)

    def _observe_scene_surface(self, context, atoms, raw, source):
        """Bind raw language to one lived scene without observer-owned port order."""
        factor,ordered,_settled=self.language.observe_scene_surface(
            int(context),tuple(map(int,atoms)),tuple(map(int,raw)),int(source))
        if factor:self._template_exemplars.setdefault(int(factor),set()).add(tuple(ordered))
        return tuple(ordered)

    def _observe_surface_construction(self, context, concepts, raw, source):
        context = int(context); concepts = tuple(map(int, concepts)); raw = tuple(map(int,raw)); source = int(source)
        factor = self.language.observe_construction_factor(context, concepts, raw, source)
        if factor:self._template_exemplars.setdefault(int(factor),set()).add(concepts)
        return bool(factor)

    def observe_surface_construction(self, context, concepts, raw, source):
        """Behavioral adapter: one externally observed multi-item construction."""
        return self._observe_surface_construction(context, concepts, raw, source)

    def construction_productivity(self, context, concepts):
        context = int(context); concepts = tuple(map(int, concepts))
        template = self.language.template(context, len(concepts))
        if template is None:
            return 0
        tid = int(template.identity[:15], 16)
        examples = tuple(self._template_exemplars.get(tid, ()))
        if tuple(concepts) in examples:
            return 1  # concrete witnessed instance remains usable before abstraction
        if len(examples) < 2:
            return 0
        diversity = min((len({row[slot] for row in examples}) for slot in range(len(concepts))), default=0)
        return diversity if diversity >= 2 else 0

    def leaf(self, context, concepts):
        context = int(context); concepts = tuple(map(int, concepts)); key=(context,concepts)
        epoch=(int(self.language._template_epoch),int(self.language._support_epoch))
        held=self._productive_leaves.get(key)
        if held is not None and held[1]==epoch:return held[0]
        if self.construction_productivity(context, concepts) == 0:
            self._productive_leaves.pop(key,None)
            raise RuntimeError('adult:construction_not_productive')
        # A support revision can change the currently live lexical realization even
        # when the reusable proposition is unchanged. Identity therefore belongs to
        # resident context/participants; rendering is transiently rematerialized.
        template=self.language.template(context,len(concepts))
        if template is None:raise HierarchicalRefuse('hierarchy:leaf_unrealized')
        surfaces=[];lexeme_ids=[]
        for concept in concepts:
            units=self.language.lexeme(concept)
            if units is None:raise HierarchicalRefuse('hierarchy:leaf_unrealized')
            surfaces.append(tuple(units));lexeme_ids.append(self.language.lexeme_identity(concept,units))
        surface=self.language.render_template(template,tuple(surfaces))
        if surface is None:raise HierarchicalRefuse('hierarchy:leaf_unrealized')
        tid=int(template.identity[:15],16);identity=_identity('hierarchy-leaf-v2',(context,concepts))
        row=self._remember_productive_leaf(identity,context,tid,tuple(lexeme_ids),surface)
        self._productive_leaves[key]=(row,epoch)
        return row

    def leaf_surface(self, context, resident_identity, surface):
        surface=tuple(int(x) for x in surface)
        if not surface or any(x<0 or x>255 for x in surface):raise HierarchicalRefuse('hierarchy:surface')
        return self._remember_raw_leaf(
            _identity('hierarchy-resident-leaf-v1',(int(context),int(resident_identity),surface)),
            int(context), surface)

    def _remember_productive_leaf(self, identity, context, template_identity, lexeme_ids, surface):
        identity=int(identity);context=int(context);template_identity=int(template_identity);lexeme_ids=tuple(map(int,lexeme_ids));surface=tuple(surface)
        template_binding=self.language.historical_template_binding(template_identity)
        lexeme_bindings=tuple(self.language.historical_lexeme_binding(x) for x in lexeme_ids)
        if template_binding is None or any(row is None for row in lexeme_bindings):raise RuntimeError('adult:leaf_factor_missing')
        template_context,arity,pieces=template_binding
        if template_context!=context or arity!=len(lexeme_ids):raise RuntimeError('adult:leaf_factor_shape')
        concepts=tuple(int(row[0]) for row in lexeme_bindings);surfaces=tuple(tuple(row[1]) for row in lexeme_bindings)
        rendered=self._render_pieces(pieces,surfaces)
        expected=_identity('hierarchy-leaf-v2',(context,concepts))
        if rendered!=surface or expected!=identity:raise RuntimeError('adult:leaf_factor_surface')
        family=self._surface_leaf_families.setdefault(template_identity,{})
        prior=family.get(identity)
        prior_family=self._surface_leaf_family_index.get(identity)
        changed=prior is not None and prior!=lexeme_ids
        if prior_family is not None and prior_family!=template_identity:
            self._surface_leaf_families[prior_family].pop(identity,None)
            if not self._surface_leaf_families[prior_family]:
                self._surface_leaf_families.pop(prior_family,None)
            changed=True
        family[identity]=lexeme_ids;self._surface_leaf_family_index[identity]=template_identity
        for concept in concepts:self._surface_leaf_concept_index.setdefault(concept,set()).add(identity)
        # Only a changed factor invalidates composed rendering holds. Unrelated
        # language evidence must not turn local rematerialization into a scan.
        if changed:self._public_surface_hold.clear()
        self._leaf_surface_hold[identity]=surface
        return ConstructionLeafV1(identity,context,surface)

    def _remember_raw_leaf(self, identity, context, surface):
        identity=int(identity);surface=tuple(surface)
        if self._has_leaf(identity):
            if self._leaf_surface(identity)!=surface:raise RuntimeError('adult:leaf_surface_collision')
        else:self._surface_leaf_surfaces[identity]=surface
        return ConstructionLeafV1(identity,int(context),surface)

    @staticmethod
    def _render_pieces(pieces, child_surfaces):
        out=[]
        for piece in pieces:
            if piece.kind==PIECE_LITERAL:out.extend(piece.literal)
            elif piece.kind==PIECE_PORT and 0<=piece.port<len(child_surfaces):out.extend(child_surfaces[piece.port])
            else:raise HierarchicalRefuse('hierarchy:template_witness_invalid')
        return tuple(out)

    def _observe_relation_span(self, context, children, raw, source):
        children=tuple(children);arity=len(children);context=int(context)
        if arity<2:return False
        context=int(context);observed=bool(self.language.observe_span(
            context,tuple(tuple(child.surface) for child in children),tuple(map(int,raw)),int(source)))
        rows=[pieces for row_context,arity,pieces in self.language._span_sources
              if int(row_context)==context and int(arity)==len(children)]
        if observed and len(rows)==1:
            factor=self.language.span_factor_identity(context,len(children),rows[0])
            self._template_exemplars.setdefault(int(factor),set()).add(
                tuple(int(child.identity) for child in children))
        return observed

    def common_cause_span_expression(self,*effects):
        """Rematerialize one exact-arity learned span licensed by common-cause history."""
        effects=tuple(map(int,effects));arity=len(effects)
        if arity<2 or len(set(effects))!=arity:return ()
        current=tuple(self.world_causal_learning.common_cause_certificate(effects[0],effect)
                      for effect in effects[1:])
        if not all(current) or len({int(row[1]) for row in current})!=1:return ()
        candidates=[]
        for factor,examples in self._template_exemplars.items():
            pieces=self.language.historical_span_pieces(int(factor))
            if pieces is None or not examples:continue
            ports=tuple(int(piece.port) for piece in pieces if piece.kind==PIECE_PORT)
            if sorted(ports)!=list(range(arity)) or not any(
                    piece.kind==PIECE_LITERAL and piece.literal for piece in pieces):continue
            support=0
            for (context,row_arity,row_pieces),sources in self.language._span_sources.items():
                if (int(row_arity)==arity and row_pieces==pieces and
                        self.language.span_factor_identity(context,row_arity,row_pieces)==int(factor)):
                    support=max(support,len(self.language._active_sources(sources)))
            if support<self.language.minimum_source_support:continue
            example_causes=[]
            for example in examples:
                if len(example)!=arity:continue
                certificates=tuple(self.world_causal_learning.common_cause_certificate(
                    int(example[0]),int(effect)) for effect in example[1:])
                if all(certificates) and len({int(row[1]) for row in certificates})==1:
                    example_causes.append(int(certificates[0][1]))
            example_causes=set(example_causes);current_cause=int(current[0][1])
            if not example_causes or (len(example_causes)<2 and current_cause in example_causes):continue
            try:
                surface=tuple(self._render_pieces(
                    pieces,tuple(tuple(self._leaf_surface(effect)) for effect in effects)))
            except (KeyError,RuntimeError):continue
            if surface:candidates.append((int(factor),surface,current))
        return candidates[0] if len(candidates)==1 else ()

    def _observe_open_span_from_known_surfaces(
            self, raw, source, max_matches=64, body_credentials=()):
        """Induce one two-child span; authenticated occurrence may ground causal use."""
        raw=tuple(map(int,raw));source=int(source);candidates=[];seen=set()
        if source<=0 or not raw:return False
        # A fully acquired relation can be re-lived after its lexical children
        # develop overlapping longer surfaces. Prefer the one learned oriented
        # factor whose two children each reconstruct one resident leaf; arbitrary
        # substring partitions have no causal certificate and cannot compete.
        if body_credentials:
            grounded=[]
            try:spans=self.language.invert_span(raw,max_candidates=max_matches)
            except ValueError:spans=()
            for span in spans:
                if len(span.children)!=2:continue
                factor=int(self.language._span_tid(span.template_identity));orientation=int(self.world_causal_learning.grounding.orientation(factor))
                if not orientation:continue
                bindings=tuple(self.language.invert_surface(child) for child in span.children)
                if any(len(rows)!=1 for rows in bindings):continue
                grounded.append((factor,bindings[0][0],bindings[1][0]))
            unique={(factor,left.context,left.atoms,right.context,right.atoms):(factor,left,right) for factor,left,right in grounded}
            if len(unique)==1:
                factor,left,right=next(iter(unique.values()))
                return self.world_causal_learning.observe_language_relation(
                    self,factor,left,right,body_credentials,source)
        for start,end,binding in self.language.embedded_surface_bindings(raw,max_candidates=max_matches):
            # A recognized whole construction is the contact being analysed,
            # not one of its own candidate children.
            if int(start)==0 and int(end)==len(raw):continue
            surface=tuple(raw[int(start):int(end)])
            key=(int(start),int(end),surface,int(binding.context),tuple(map(int,binding.atoms)))
            if key in seen:continue
            seen.add(key);candidates.append((int(start),int(end),surface,binding))
            if len(candidates)>int(max_matches):return False
        # A single already learned occurrence inside new boundary matter can
        # teach a unary receptive relation.  The literal bytes have no innate
        # punctuation/quote/bracket meaning: their only earned function is the
        # repeatable structural transformation witnessed by independent contact.
        grounded=tuple(row for row in candidates if row[3] is not None)
        if len(grounded)==1:
            child=grounded[0]
            prefix=tuple(raw[:child[0]]);suffix=tuple(raw[child[1]:])
            if prefix or suffix:
                context=_identity('adult-receptive-wrapper-v1',
                                  (1,bool(prefix),prefix,bool(suffix),suffix))
                return bool(self.language.observe_span(
                    context,(child[2],),raw,source))
        # Close the current contact over learned binary span relations.  A span
        # discovered on one pass may itself become a child on the next pass; no
        # depth-specific wrapper law or persistent parse object is introduced.
        # Nominate only learned span patterns whose literals occur in this raw
        # contact, so dormant structure does not create whole-Adult scan work.
        nominated=set(self.language._inverse_span_no_literal)
        for cursor in range(len(raw)):
            rows,_used=self.language._trie_matches(self.language._inverse_span_trie,raw,cursor)
            nominated.update(rows)
        binary_contexts=tuple(sorted({int(context) for context,arity,_pieces in nominated
                                     if int(arity)==2}))
        for _depth in range(4):
            prior=tuple(candidates);added=False
            for left in prior:
                for right in prior:
                    if left==right or left[1]>right[0]:continue
                    for context in binary_contexts:
                        template=self.language.span_template(context,2)
                        if template is None:continue
                        nested=self.language.realize_span(context,(left[2],right[2]))
                        if nested is None:continue
                        for start in self.language._positions(raw,nested):
                            end=start+len(nested)
                            if start==0 and end==len(raw):continue
                            if not (start<=left[0] and left[1]<=end and start<=right[0] and right[1]<=end):continue
                            key=(start,end,tuple(nested),'span',str(template.identity))
                            if key in seen:continue
                            seen.add(key);candidates.append((start,end,tuple(nested),None));added=True
                            if len(candidates)>int(max_matches):return False
            if not added:break
        pairs=[];best=0
        for left in candidates:
            for right in candidates:
                if left[1]>right[0]:continue
                score=len(left[2])+len(right[2])
                if score>best:best=score;pairs=[(left,right)]
                elif score==best:pairs.append((left,right))
        if best<=0 or len(pairs)!=1:return False
        left,right=pairs[0]
        for row in candidates:
            if row==left or row==right:continue
            if not (left[0]<=row[0] and row[1]<=left[1]
                    or right[0]<=row[0] and row[1]<=right[1]):return False
        literals=(tuple(raw[:left[0]]),tuple(raw[left[1]:right[0]]),tuple(raw[right[1]:]))
        if not any(literals):return False
        context=_identity('adult-open-span-pattern-v1',(2,literals))
        observed=bool(self.language.observe_span(context,(left[2],right[2]),raw,source))
        rows=[pieces for (row_context,arity,pieces) in self.language._span_sources
              if int(row_context)==int(context) and int(arity)==2]
        factor=(self.language.span_factor_identity(context,2,rows[0])
                if len(rows)==1 else 0)
        grounded=(self.world_causal_learning.observe_language_relation(
            self,factor,left[3],right[3],body_credentials,source)
            if factor and body_credentials and left[3] is not None and right[3] is not None else False)
        return bool(observed or grounded)

    def _reconstruct_unique_nested_bindings(
            self, raw, max_depth=4, max_closures=64):
        """Rematerialize one fully learned closure; retain no parse or raw child."""
        raw=tuple(map(int,raw));overflow=[False];visits=[0]
        if not raw:return ()

        def spans(surface):
            try:rows=self.language.invert_span(surface,max_candidates=16)
            except ValueError:
                overflow[0]=True;return ()
            if not rows:return ()
            scored=[]
            for row in rows:
                pieces=self.language.historical_span_pieces(
                    self.language._span_tid(row.template_identity)) or ()
                specificity=sum(len(piece.literal) for piece in pieces
                                if piece.kind==PIECE_LITERAL)
                scored.append((specificity,row))
            peak=max(score for score,_row in scored)
            # Exact learned boundary matter constrains the current closure more
            # than a zero-literal span. Equal evidence remains genuinely ambiguous.
            candidates=tuple(row for score,row in scored if score==peak)
            settled=[]
            for row in candidates:
                coverage=0
                for child in row.children:
                    try:direct=self.language.invert_surface(child,max_candidates=16)
                    except ValueError:direct=()
                    if direct:coverage+=len(child)
                settled.append((coverage,row))
            direct_peak=max(score for score,_row in settled)
            # Prefer the relation whose children already settle the greatest
            # amount of current matter; this is evidence coverage, not arity.
            return tuple(row for score,row in settled
                         if not direct_peak or score==direct_peak)

        def close(surface,depth,ancestors):
            surface=tuple(map(int,surface));visits[0]+=1
            if visits[0]>int(max_closures)*int(max_depth):
                overflow[0]=True;return ()
            try:direct=self.language.invert_surface(surface,max_candidates=16)
            except ValueError:
                overflow[0]=True;return ()
            found=[(("leaf",int(row.context),tuple(map(int,row.atoms)),
                     str(row.template_identity)),(row,))
                   for row in direct if row.atoms and int(row.context)>0]
            # A directly bound learned Occurrence is already sufficient for
            # lawful future cognition. Do not let deeper donor derivations
            # re-enter as rival minds beneath that settlement.
            if found or depth>=int(max_depth):return tuple(found)
            for span in spans(surface):
                partial=(((),()),)
                for child in span.children:
                    child=tuple(map(int,child))
                    if not child or child in ancestors:
                        partial=();break
                    alternatives=close(child,depth+1,ancestors|{child})
                    if not alternatives:
                        partial=();break
                    joined=[]
                    for signatures,bindings in partial:
                        for signature,rows in alternatives:
                            joined.append((signatures+(signature,),bindings+rows))
                            if len(joined)>int(max_closures):
                                overflow[0]=True;return ()
                    partial=tuple(joined)
                for signatures,bindings in partial:
                    found.append((("span",str(span.template_identity),signatures),bindings))
            unique={repr(signature):(signature,bindings) for signature,bindings in found}
            return tuple(unique[key] for key in sorted(unique))

        closures=[bindings for signature,bindings in close(raw,0,{raw})
                  if signature[0]=="span"]
        # Several disposable derivations are not several perceptions when they
        # settle to the exact same bound resident Occurrences.  Preserve real
        # binding ambiguity, not transient parse-path multiplicity.
        settlements={}
        for bindings in closures:
            key=tuple((int(row.context),tuple(map(int,row.atoms)),
                       str(row.template_identity),tuple(map(int,row.lexical_identities)))
                      for row in bindings)
            settlements.setdefault(key,bindings)
        return (next(iter(settlements.values()))
                if not overflow[0] and len(settlements)==1 else ())

    def observe_join(self, context, left, right, source, separator=b" "):
        raw=tuple(left.surface)+tuple(separator)+tuple(right.surface)
        return self._observe_relation_span(context,(left,right),raw,source)

    def _as_compose_child(self, child):
        if isinstance(child, ConstructionLeafV1):
            return int(child.identity), 0, tuple(child.surface)
        if isinstance(child, CompositionWitnessV1):
            return int(child.identity), int(child.depth), tuple(child.surface)
        pid=int(getattr(child, 'identity', child))
        if self.programs.factor(pid) is not None:
            return pid, max(0, int(self.program_depth(pid))), ()
        if self._has_leaf(pid):return pid,0,self._leaf_surface(pid)
        raise HierarchicalRefuse('hierarchy:unknown_child')

    def compose(self, context, left, right):
        left_id,left_depth,left_surface=self._as_compose_child(left)
        right_id,right_depth,right_surface=self._as_compose_child(right)
        template=self.language.span_template(int(context), 2)
        if template is None or int(getattr(template, 'arity', 2))!=2:
            raise HierarchicalRefuse('hierarchy:template_missing_or_ambiguous')
        tid=int(template.identity[:15], 16)
        pieces=tuple(template.pieces)
        child_ids=(left_id, right_id)
        depth=1+max(left_depth, right_depth)
        ident=_identity('adult-compose-witness-v1', (int(context), tid, child_ids, depth))
        if left_surface and right_surface:
            embedded=self._embedded_port_surfaces(
                pieces,(left_surface,right_surface),(left_depth,right_depth),
                (self._composite_literal_initial(left),self._composite_literal_initial(right)))
            surface=self._render_pieces(pieces,embedded)
        else:surface=()
        return CompositionWitnessV1(
            ident, int(context), tid, child_ids, depth, surface, pieces)

    def program_role_view(self, program_identity, base_context=1):
        """Rematerialize terminal local roles and recursive topology from program factors."""
        root=int(program_identity);base=int(base_context)
        if root not in self.programs.chunks or base<=0:
            raise HierarchicalRefuse('adult:program_role_root')
        terminals=[];stack=set()
        def leaf_concept(identity):
            identity=int(identity)
            factor=self.programs.factor(identity)
            leaf=-int(factor) if factor is not None and int(factor)<0 else identity
            family=self._surface_leaf_family_index.get(leaf)
            if family is None:raise HierarchicalRefuse('adult:program_role_productive_leaf')
            lexemes=self._surface_leaf_families.get(int(family),{}).get(leaf)
            if lexemes is None or len(lexemes)!=1:
                raise HierarchicalRefuse('adult:program_role_terminal_arity')
            binding=self.language.historical_lexeme_binding(int(lexemes[0]))
            if binding is None:raise HierarchicalRefuse('adult:program_role_lexeme')
            return leaf,int(binding[0])
        def walk(identity,parent_operator=0,parent_branch=-1):
            identity=int(identity)
            if identity in stack:raise HierarchicalRefuse('adult:program_role_cycle')
            chunk=self.programs.chunks.get(identity)
            if chunk is None:
                if parent_operator<=0 or parent_branch<0:
                    raise HierarchicalRefuse('adult:program_role_terminal_parent')
                leaf,concept=leaf_concept(identity);role=(int(parent_operator),int(parent_branch))
                terminals.append((role,leaf,concept));return ('leaf',role)
            factor=self.programs.factor(identity)
            if factor is None or int(factor)<=0 or len(chunk.members)!=2:
                raise HierarchicalRefuse('adult:program_role_operator')
            operator=int(factor);stack.add(identity)
            try:
                children=tuple(walk(member,operator,branch) for branch,member in enumerate(chunk.members))
            finally:stack.remove(identity)
            return ('operator',operator,children)
        topology_shape=walk(root)
        roles=[row[0] for row in terminals]
        if len(set(roles))!=len(roles):raise HierarchicalRefuse('adult:program_role_ambiguous')
        ordered=tuple(sorted(terminals,key=lambda row:row[0]))
        canonical_roles=tuple(row[0] for row in ordered)
        context=_identity('adult-program-role-context-v1',(base,canonical_roles))
        topology=_identity('adult-program-role-topology-v1',topology_shape)
        return {'context':int(context),'topology':int(topology),
                'roles':canonical_roles,'leaves':tuple(row[1] for row in ordered),
                'atoms':tuple(row[2] for row in ordered)}

    @staticmethod
    def _program_role_condition_rows(view, leaf_conditions):
        raw={int(k):tuple(int(x) for x in v if int(x)) for k,v in dict(leaf_conditions or {}).items()}
        if any(len(v)>1 for v in raw.values()):raise HierarchicalRefuse('adult:program_role_condition_width')
        return tuple(raw.get(int(leaf),()) for leaf in view['leaves'])

    def observe_program_role_construction(self, program_identity, raw, source, base_context=1):
        view=self.program_role_view(program_identity,base_context)
        return self.language.observe_role_construction(
            view['context'],view['atoms'],tuple(map(int,raw)),int(source),view['topology'])

    def observe_program_role_conditioned_contact(self, program_identity, leaf_conditions, raw, source, base_context=1):
        view=self.program_role_view(program_identity,base_context)
        conditions=self._program_role_condition_rows(view,leaf_conditions)
        return self.language.observe_conditioned_contact(
            view['context'],view['atoms'],conditions,tuple(map(int,raw)),int(source))

    def realize_program_role_conditioned(self, program_identity, leaf_conditions=(), base_context=1):
        view=self.program_role_view(program_identity,base_context)
        conditions=self._program_role_condition_rows(view,leaf_conditions)
        seeds=tuple(int(row[0]) if row else 0 for row in conditions)
        completed=self.language.complete_dependencies(view['context'],seeds)
        values=seeds if completed is None else tuple(map(int,completed))
        expanded=tuple((value,) if value else () for value in values)
        return self.language.realize_conditioned(view['context'],view['atoms'],expanded)

    @staticmethod
    def _language_followup_context(program_identity, binding_context):
        return _identity('adult-language-followup-construction-v2',
            (int(program_identity),int(binding_context)))

    def observe_language_followup_construction(self, program_identity, binding_context,
                                               atoms, raw, source):
        """Learn one program-qualified productive surface over a returned binding."""
        pid=int(program_identity)
        if pid<=0 or self.programs.factor(pid) is None:return False
        context=self._language_followup_context(pid,binding_context)
        return self.language.observe_construction(
            context,tuple(map(int,atoms)),tuple(map(int,raw)),int(source))

    def _current_language_followup_plan(self, program_identity):
        binding=self._current_language_reply_binding;pid=int(program_identity)
        if not binding or pid<=0 or self.programs.factor(pid) is None:return None
        binding_context,atoms=binding
        context=self._language_followup_context(pid,binding_context)
        template=self.language.template(context,len(atoms))
        surface=self.language.realize(context,atoms)
        if template is None or not surface:return None
        identity=_identity('adult-language-followup-plan-v1',
            (pid,int(binding_context),tuple(atoms),tuple(surface)))
        return CompositionWitnessV1(
            identity,int(self._current_selection_context),
            int(template.identity[:15],16),(pid,),self.program_depth(pid),
            tuple(surface),tuple(template.pieces))

    def clear_current_program_role_conditions(self):
        if self._current_program_role_occurrence:
            self._current_program_role_occurrence=();self._select_epoch+=1

    def settle_current_program_role_conditions(self, program_identity, leaf_conditions,
                                               base_context=1):
        """Bind live opaque conditions only to the Adult's current winner."""
        pid=int(program_identity);base=int(base_context)
        if pid<=0 or pid!=int(self.choose()):return False
        view=self.program_role_view(pid,base)
        rows=self._program_role_condition_rows(view,leaf_conditions)
        occurrence=(int(self._current_selection_context),pid,base,
                    tuple(zip(view['leaves'],rows)))
        if occurrence!=self._current_program_role_occurrence:
            self._current_program_role_occurrence=occurrence;self._select_epoch+=1
        return True

    def _current_program_role_plan(self, program_identity):
        occurrence=self._current_program_role_occurrence;pid=int(program_identity)
        if not occurrence or occurrence[0]!=int(self._current_selection_context) or occurrence[1]!=pid:
            return None
        _context,_pid,base,bindings=occurrence
        view=self.program_role_view(pid,base);leaf_conditions=dict(bindings)
        surface=self.realize_program_role_conditioned(pid,leaf_conditions,base)
        if not surface:return None
        identity=_identity(
            'adult-program-role-public-plan-v1',
            (pid,int(base),view['topology'],tuple(bindings),tuple(surface)))
        return CompositionWitnessV1(
            identity,int(self._current_selection_context),int(self.programs.factor(pid)),
            tuple(view['leaves']),int(self.program_depth(pid)),tuple(surface),())

    def _resource_compete_public_realization(self, program_identity, plan, state):
        """Let resident body load price the actual transient motor plan, not its root."""
        pid=int(program_identity)
        if not isinstance(plan,CompositionWitnessV1) or pid<=0 or not plan.surface:return plan
        pressure=self._effective_pressure_q16(state)
        relief=max(0,min(Q,int(state.relief_q16))) if state.relief_authenticated else 0
        effective_pressure=max(0,pressure-relief)
        if not effective_pressure:return plan
        try:base=self.public_surface(pid)
        except (KeyError,RuntimeError):return plan
        if not base or len(plan.surface)<=len(base):return plan
        row=self.credit.row(pid);base_effort=max(1,int(row.effort_mean_q16))
        planned_effort=min(Q,(base_effort*len(plan.surface)+len(base)-1)//len(base))
        capacity_margin=max(Q//16,Q-effective_pressure)
        base_surcharge=(base_effort*effective_pressure)//capacity_margin
        planned_surcharge=(planned_effort*effective_pressure)//capacity_margin
        return pid if planned_surcharge>base_surcharge else plan

    @staticmethod
    def _composite_literal_initial(child):
        if not isinstance(child,CompositionWitnessV1):return False
        for piece in tuple(child.pieces):
            if piece.kind==PIECE_LITERAL and piece.literal:return True
            if piece.kind==PIECE_PORT:return False
        return False

    @staticmethod
    def _embedded_port_surfaces(pieces, child_surfaces, child_depths=(), child_literal_initials=()):
        """Externalize child boundaries in the parent factor, before public bytes exist.

        A child may be a complete utterance when emitted alone and therefore carry a
        terminal `.?!`.  If the learned parent factor itself supplies structural
        punctuation immediately after that port, the child boundary is not the public
        utterance boundary anymore.  Suppress only that redundant terminal marker;
        never rewrite lexical bytes or infer structure from the child text.
        """
        rows=[];pieces=tuple(pieces)
        for port,surface in enumerate(child_surfaces):
            units=list(map(int,surface));indices=[i for i,piece in enumerate(pieces)
                if piece.kind==PIECE_PORT and int(piece.port)==port]
            if len(indices)!=1:
                rows.append(tuple(units));continue
            index=indices[0];following=()
            if index+1<len(pieces) and pieces[index+1].kind==PIECE_LITERAL:
                following=tuple(map(int,pieces[index+1].literal))
            first=next((value for value in following if value not in (9,10,13,32)),0)
            if first:
                while units and units[-1] in (9,10,13,32):units.pop()
                if units and units[-1] in (33,46,63):units.pop()
            prior_content=any(
                (piece.kind==PIECE_LITERAL and bool(piece.literal)) or piece.kind==PIECE_PORT
                for piece in pieces[:index])
            literal_initial=port<len(child_literal_initials) and bool(child_literal_initials[port])
            depth=int(child_depths[port]) if port<len(child_depths) else 0
            if depth>0 and prior_content and literal_initial and units and 65<=units[0]<=90:
                units[0]+=32
            rows.append(tuple(units))
        return tuple(rows)

    def _compose_factor(self, operator_factor, left, right):
        """Transiently compose with one already-learned opaque binary operator."""
        factor=int(operator_factor);pieces=self.language.historical_span_pieces(factor)
        if pieces is None:
            raise HierarchicalRefuse('adult:organization_operator_missing')
        ports=tuple(piece.port for piece in pieces if piece.kind==PIECE_PORT)
        if sorted(ports)!=[0,1]:
            raise HierarchicalRefuse('adult:organization_operator_arity')
        left_id,left_depth,left_surface=self._as_compose_child(left)
        right_id,right_depth,right_surface=self._as_compose_child(right)
        if not left_surface or not right_surface:
            raise HierarchicalRefuse('adult:organization_child_unmaterialized')
        child_ids=(left_id,right_id);depth=1+max(left_depth,right_depth)
        ident=_identity('adult-organize-witness-v1',(factor,child_ids,depth))
        embedded=self._embedded_port_surfaces(
            pieces,(left_surface,right_surface),(left_depth,right_depth),
            (self._composite_literal_initial(left),self._composite_literal_initial(right)))
        surface=self._render_pieces(tuple(pieces),embedded)
        return CompositionWitnessV1(ident,0,factor,child_ids,depth,surface,tuple(pieces))

    def experience_program(self, children, root,
                           outcome_q16, somatic_q16=0, context=0,
                           effort_q16=Q//8, controllable=True,
                           retire_closure=True):
        context=self._experience_context(context);children=tuple(map(int,children))
        semantic_depth=1+max((self.programs.chunks[x].depth if x in self.programs.chunks else 0) for x in children)
        start = self._advance()
        end = start + max(1, semantic_depth + 1)
        chunk = self.programs.observe(
            children, start, end, int(effort_q16), int(outcome_q16), int(somatic_q16),
            operator=int(root.template_identity))
        self._tick = end + 1
        if chunk is None:
            return None
        self._register_program_surface(chunk,root)
        self.credit.observe_use(chunk.identity, start, end, int(effort_q16), context)
        self.credit.observe_return(chunk.identity, int(outcome_q16), int(somatic_q16),
                                   end + 1, True, context)
        self.credit.observe_control(chunk.identity, bool(controllable), bool(controllable), context)
        for other in tuple(self.credit.candidates(context)):
            other=int(other)
            if other!=chunk.identity:self.credit.observe_control(other,False,False,context)
        self._select_epoch+=1
        return chunk

    def experience_atomic_program(self, program_identity, root: ConstructionClosureV1,
                                  outcome_q16, somatic_q16=0, context=0,
                                  effort_q16=Q//16, controllable=True):
        """Give an already-resident atomic public behavior the same selection ecology."""
        context=self._experience_context(context);pid = int(program_identity)
        self._register_atomic_surface(pid,root)
        start = self._advance(); end = start + 1
        self.credit.observe_use(pid, start, end, int(effort_q16), context)
        self.credit.observe_return(pid, int(outcome_q16), int(somatic_q16), end + 1, True, context)
        self.credit.observe_control(pid, bool(controllable), bool(controllable), context)
        for other in tuple(self.credit.candidates(context)):
            other=int(other)
            if other!=pid:self.credit.observe_control(other,False,False,context)
        self._select_epoch+=1
        self._tick = end + 1
        return pid

    def experience_choice(self, program_identity, outcome_q16, somatic_q16=0,
                          context=0, effort_q16=Q//8, duration=2,
                          controllable=True, independent_return=True):
        """Settle one public use of an already learned program in a lived context."""
        context=self._experience_context(context);pid = int(program_identity)
        if self.programs.factor(pid) is None:
            raise KeyError(pid)
        start = self._advance(); end = start + max(1, int(duration))
        self.credit.observe_use(pid, start, end, int(effort_q16), context)
        self.credit.observe_return(pid, int(outcome_q16), int(somatic_q16), end + 1,
                                   bool(independent_return), context)
        self.credit.observe_control(
            pid,bool(controllable),bool(controllable and independent_return),context)
        for other in tuple(self.credit.candidates(context)):
            other=int(other)
            if other!=pid:self.credit.observe_control(other,False,False,context)
        self._select_epoch+=1
        self._tick = end + 1

    def experience_program_background(self, program_identity, outcome_occurs, context=None):
        """Observe one matched no-program opportunity without rewriting program value."""
        pid=int(program_identity)
        if self.programs.factor(pid) is None:raise KeyError(pid)
        context=self._current_selection_context if context is None else int(context)
        if not context:raise RuntimeError('adult:no_current_situation')
        local=self.credit.row(pid).contexts.get(context)
        if local is None or not local.participated:raise RuntimeError('adult:program_background_without_participation')
        self.credit.observe_control(pid,False,bool(outcome_occurs),context);self._select_epoch+=1

    def experience_operator_choice(self, operator_factor, outcome_q16, somatic_q16=0,
                                   context=0, effort_q16=Q//16, duration=1,
                                   independent_return=True):
        """Settle one used opaque operator from independently returned consequence."""
        factor=int(operator_factor)
        if self.language.historical_span_pieces(factor) is None:
            raise KeyError(factor)
        context=self._experience_context(context)
        start=self._advance();end=start+max(1,int(duration))
        self.organization_credit.observe_use(factor,start,end,int(effort_q16),context)
        independent=bool(independent_return)
        self.organization_credit.observe_return(
            factor,int(outcome_q16),int(somatic_q16),end+1,independent,context)
        self.organization_credit.observe_control(factor,True,independent,context)
        for other in tuple(self.organization_credit.candidates(context)):
            other=int(other)
            if other!=factor:self.organization_credit.observe_control(other,False,False,context)
        self._select_epoch+=1;self._tick=end+1

    def experience_operator_background(self, operator_factor, outcome_occurs):
        """Observe a matched no-action opportunity without rewriting action value."""
        factor=int(operator_factor)
        if self.language.historical_span_pieces(factor) is None:
            raise KeyError(factor)
        context=self._current_selection_context
        if not context:raise RuntimeError('adult:no_current_situation')
        self.organization_credit.observe_control(factor,False,bool(outcome_occurs),context)
        self._select_epoch+=1

    def observe_social_topic(self, context, proposition_identity, source):
        context=int(context);proposition=int(proposition_identity);source=int(source);key=(context,proposition)
        if context<=0 or source<=0 or not self._has_leaf(proposition):return False
        if key not in self._social_topic_sources:
            if len(self._social_topic_sources)>=MAX_SOCIAL_TOPIC_RELATIONS:return False
            self._social_topic_sources[key]=set()
        before=len(self._social_topic_sources[key]);self._social_topic_sources[key].add(source)
        if len(self._social_topic_sources[key])!=before:self._select_epoch+=1
        return True

    def social_topic_supported(self, context, proposition_identity):
        sources=self._social_topic_sources.get((int(context),int(proposition_identity)),())
        return sum(1 for source in sources if source not in self.language._withdrawn)>=SOCIAL_TOPIC_MIN_SOURCES

    def experience_discourse_candidate(self, proposition_identity, outcome_q16, somatic_q16=0,
                                       context=0, effort_q16=Q//16, duration=1,
                                       independent_return=True):
        """Settle one expressed proposition from independently returned consequence."""
        proposition=int(proposition_identity)
        if not self._has_leaf(proposition):raise KeyError(proposition)
        context=self._experience_context(context)
        start=self._advance();end=start+max(1,int(duration))
        self.discourse_credit.observe_use(proposition,start,end,int(effort_q16),context)
        independent=bool(independent_return)
        self.discourse_credit.observe_return(
            proposition,int(outcome_q16),int(somatic_q16),end+1,independent,context)
        self.discourse_credit.observe_control(proposition,True,independent,context)
        for other in tuple(self.discourse_credit.candidates(context)):
            other=int(other)
            if other!=proposition:self.discourse_credit.observe_control(other,False,False,context)
        self._select_epoch+=1;self._tick=end+1

    def experience_discourse_background(self, proposition_identity, outcome_occurs):
        """Observe matched no-expression opportunity for one known proposition."""
        proposition=int(proposition_identity)
        if not self._has_leaf(proposition):raise KeyError(proposition)
        context=self._current_selection_context
        if not context:raise RuntimeError('adult:no_current_situation')
        self.discourse_credit.observe_control(proposition,False,bool(outcome_occurs),context)
        self._select_epoch+=1

    def experience_discourse_transition(self, predecessor, successor, context=None,
                                        gap=1, independent_return=True):
        """Learn one context-local proposition transition from independent return."""
        context=self._experience_context(context);left=int(predecessor);right=int(successor)
        if left==right or not self._has_leaf(left) or not self._has_leaf(right):
            raise ValueError('adult:discourse_transition')
        if not (self.discourse_credit.contextual_control_supported(left,context)
                and self.discourse_credit.contextual_control_supported(right,context)):
            raise RuntimeError('adult:discourse_transition_unearned')
        if not independent_return:return False
        self.discourse_credit.observe_successor(left,right,int(gap),context)
        self._select_epoch+=1;return True

    def experience_partner_choice(self, program_identity, outcome_q16, somatic_q16=0,
                                  effort_q16=Q//16, duration=1,
                                  independent_return=True,action_ticket=None):
        """Settle an attached or exact-ticket delayed authenticated partner consequence."""
        pid=int(program_identity);ticket=0 if action_ticket is None else int(action_ticket);role_row=None
        if ticket:
            self._expire_partner_action_tickets();banked=self._pending_span_reply_actions.get(ticket)
            if banked is None:raise RuntimeError('adult:partner_action_ticket')
            roles,context,selected,_born=banked
            if int(selected)!=pid:raise RuntimeError('adult:partner_action_ticket_program')
            role_row=(roles,int(context),int(selected));context=int(context)
        else:
            context=int(self._current_partner_context)
            if self._pending_span_reply_actions and not context:raise RuntimeError('adult:partner_action_ticket_required')
            if self._pending_span_reply_role and self._pending_span_reply_role[2]==pid:role_row=self._pending_span_reply_role
        if not context or (self.programs.factor(pid) is None and not self._has_leaf(pid)):
            raise RuntimeError('adult:no_current_partner')
        start=self._advance();end=start+max(1,int(duration));independent=bool(independent_return)
        self.partner_credit.observe_use(pid,start,end,int(effort_q16),context)
        self.partner_credit.observe_return(pid,int(outcome_q16),int(somatic_q16),end+1,independent,context)
        self.partner_credit.observe_control(pid,True,independent,context)
        for other in tuple(self.partner_credit.candidates(context)):
            other=int(other)
            if other!=pid:self.partner_credit.observe_control(other,False,False,context)
        if role_row:
            roles,partner_context,selected=role_row
            if partner_context==context and selected==pid:
                if independent:
                    effect=1 if int(outcome_q16)>0 else -1 if int(outcome_q16)<0 else 0
                    if effect:
                        for tid,port,source in roles:
                            pieces=self.language._span_identity_pieces.get(tid)
                            if (pieces is None or source<=0
                                    or port not in {int(piece.port) for piece in pieces if piece.kind==PIECE_PORT}):
                                raise RuntimeError('adult:pending_span_reply_role_invalid')
                        for tid,port,source in roles:
                            if not self.language.observe_span_reply_role(tid,port,source,effect):
                                raise RuntimeError('adult:pending_span_reply_role_settlement')
                if ticket:self._pending_span_reply_actions.pop(ticket,None)
                else:
                    self._pending_span_reply_role=()
                    self._current_partner_action_ticket=0
        if self._pending_episode_retrieval:
            channel,episode_context,episode_plan,selected,partner_context=self._pending_episode_retrieval
            if int(partner_context)==context and int(selected)==pid:
                if independent and int(outcome_q16)!=0:
                    source=max(1,int(channel))
                    self.reinforce_completed_public_episode(
                        channel,episode_context,episode_plan,source,int(outcome_q16),True)
                self._pending_episode_retrieval=()
        if not ticket and self._current_partner_action==(context,pid):
            self._current_partner_action_ticket=0;self._current_partner_action=()
        self._select_epoch+=1;self._tick=end+1

    def experience_partner_background(self, program_identity, outcome_occurs):
        pid=int(program_identity);context=int(self._current_partner_context)
        if not context or (self.programs.factor(pid) is None and not self._has_leaf(pid)):
            raise RuntimeError('adult:no_current_partner')
        self.partner_credit.observe_control(pid,False,bool(outcome_occurs),context);self._select_epoch+=1

    def _clear_social_identity_occurrence(self):
        self._current_partner_context=0
        self._current_social_source=0
        self._current_social_subject=0
        self._current_social_agent=0
        self._current_social_tick=0
        self._recent_social_source=0
        self._recent_social_subject=0
        self._recent_social_tick=0

    def _experience_context(self, context):
        if context is None:
            context=self._current_selection_context
            if not context:raise RuntimeError('adult:no_current_situation')
        context=int(context)
        if context!=self._current_selection_context:
            self.clear_current_program_role_conditions()
            self._current_language_reply_binding=()
            self._clear_social_identity_occurrence()
            self._pending_language_competition=()
            self._language_competition_evidence_q16=0
            self._language_competition_leader=0
            self._language_competition_active=False
            self._language_competition_expressed=False
            self._language_competition_channel=0
        self._current_selection_context=context;return context

    def _clear_current_occurrence(self, preserve_language_reply=False):
        """End the transient situation before accepting a new body contact."""
        if not preserve_language_reply:
            self._expire_partner_action_tickets()
            if self._current_partner_action_ticket>0 and self._current_partner_action:
                if len(self._pending_span_reply_actions)<MAX_PENDING_PARTNER_ACTIONS:
                    partner_context,selected=self._current_partner_action
                    roles=self._pending_span_reply_role[0] if self._pending_span_reply_role and self._pending_span_reply_role[1:]==(partner_context,selected) else ()
                    self._pending_span_reply_actions[int(self._current_partner_action_ticket)]=(roles,partner_context,selected,int(self._tick))
            self._pending_span_reply_role=()
            self._current_partner_action_ticket=0
            self._current_partner_action=()
            self._pending_episode_retrieval=()
        self.clear_current_program_role_conditions()
        self._current_language_reply_binding=()
        self._current_selection_context=0
        self._current_language_channel=0
        self.last_discourse_selected=()
        self._clear_social_identity_occurrence()
        if preserve_language_reply and self._language_competition_expressed:
            return
        self._pending_language_competition=()
        self._language_competition_evidence_q16=0
        self._language_competition_leader=0
        self._language_competition_active=False
        self._language_competition_expressed=False
        self._language_competition_channel=0

    def _activate_language_occurrence(self, context, concepts, preserve_language_reply=False):
        """Let a residently reconstructed construction become the current situation."""
        self.clear_current_program_role_conditions()
        self._current_language_reply_binding=()
        self._clear_social_identity_occurrence()
        if not (preserve_language_reply and self._language_competition_expressed):
            self._pending_language_competition=()
            self._language_competition_evidence_q16=0
            self._language_competition_leader=0
            self._language_competition_active=False
            self._language_competition_expressed=False
            self._language_competition_channel=0
        leaf=self.leaf(int(context),tuple(map(int,concepts)))
        self._current_selection_context=int(leaf.identity)
        return self._current_selection_context

    @staticmethod
    def _language_competition_context(alternatives):
        return _identity('adult-language-competition-v1',alternatives)

    @staticmethod
    def _language_repair_context(alternatives, matched):
        return _identity('adult-language-repair-v1',(tuple(alternatives),tuple(matched)))

    @staticmethod
    def _repair_partner_context(channel, repair_context):
        """Factor authenticated source history by the resident repair relation."""
        return _identity(
            'adult-repair-source-context-v1',(max(0,int(channel)),int(repair_context)))

    def _settle_public_expression(self, program_identity):
        """Reafferent completion, not selection, opens the pending social return."""
        if (self._pending_language_competition
                and not self._language_competition_active
                and int(program_identity)==self._language_competition_leader):
            self._language_competition_expressed=True

    def _expire_partner_action_tickets(self):
        cutoff=int(self._tick)-PARTNER_ACTION_TTL
        if cutoff<=0:return 0
        expired=[ticket for ticket,row in self._pending_span_reply_actions.items() if int(row[3])<=cutoff]
        for ticket in expired:self._pending_span_reply_actions.pop(ticket,None)
        return len(expired)

    def _mint_partner_action_ticket(self):
        self._expire_partner_action_tickets()
        ticket=int(self._next_partner_action_ticket);self._next_partner_action_ticket+=1
        self._current_partner_action_ticket=ticket
        return ticket

    def current_partner_action_ticket(self):
        return int(self._current_partner_action_ticket)

    def _stage_span_reply_roles(self, roles, source):
        source=int(source);rows=[]
        if source<=0 or self._current_partner_context<=0 or self._pending_span_reply_role:return False
        for template_identity,port in roles:
            tid=self.language._span_tid(template_identity);port=int(port)
            pieces=self.language._span_identity_pieces.get(tid)
            if pieces is None or port not in {int(p.port) for p in pieces if p.kind==PIECE_PORT}:return False
            rows.append((tid,port,source))
        rows=tuple(sorted(set(rows)))
        if not 1<=len(rows)<=4:return False
        self._pending_span_reply_role=(rows,int(self._current_partner_context),0);return True

    def _stage_span_reply_role(self, template_identity, port, source):
        return self._stage_span_reply_roles(((template_identity,port),),source)

    def _language_reply_open(self, channel=0):
        return bool(self._language_competition_expressed
                    and self._language_competition_channel==int(channel))

    def _settle_language_reply(self, bindings, channel=0):
        """Close one expressed competition by sparse learned-structure intersection."""
        if (not self._pending_language_competition
                or self._language_competition_active
                or not self._language_competition_expressed
                or self._language_competition_channel!=int(channel)):
            return None
        pending=self._pending_language_competition
        reconstructed={(int(row.context),tuple(map(int,row.atoms))) for row in bindings}
        matched=tuple(row for row in pending if row in reconstructed)
        if len(matched)!=1:return None
        context=self._language_repair_context(pending,matched[0])
        self._pending_language_competition=()
        self._language_competition_evidence_q16=0
        self._language_competition_leader=0
        self._language_competition_active=False
        self._language_competition_expressed=False
        self._language_competition_channel=0
        self.clear_current_program_role_conditions()
        self._current_language_reply_binding=matched[0]
        self._clear_social_identity_occurrence()
        self._current_partner_context=self._repair_partner_context(channel,context)
        self._current_selection_context=context
        return context,matched[0]

    def _activate_language_competition(self, bindings, channel=0):
        """Retain one unresolved structural occurrence without choosing its meaning."""
        alternatives=[]
        for binding in bindings:
            if not binding.atoms or int(binding.context)==0:continue
            alternatives.append((int(binding.context),tuple(map(int,binding.atoms))))
        alternatives=tuple(sorted(set(alternatives)))
        if len(alternatives)<2:return 0
        self._pending_language_competition=alternatives
        self._language_competition_evidence_q16=0
        self._language_competition_leader=0
        self._language_competition_active=True
        self._language_competition_expressed=False
        self._language_competition_channel=max(0,int(channel))
        self.clear_current_program_role_conditions()
        self._current_language_reply_binding=()
        self._clear_social_identity_occurrence()
        self._current_selection_context=self._language_competition_context(alternatives)
        return self._current_selection_context

    @staticmethod
    def _partner_interaction_context(source_anchor,subject_identity):
        return _identity('adult-partner-interaction-v2',(int(source_anchor),int(subject_identity)))

    def _resolved_social_source(self, source):
        source=int(source);supported=[]
        for anchor in self.social_identity_credit.candidates(source):
            anchor=int(anchor);local=self.social_identity_credit.row(anchor).contexts.get(source)
            if local is None or not local.participated:continue
            # Base source recognition is repeated authenticated relation evidence;
            # cross-source continuity remains action-vs-background contingency.
            ready=(local.outcome_samples>=2 and local.control_successes>=2) if anchor==source else local.control_supported
            if ready:supported.append(anchor)
        return supported[0] if len(supported)==1 else 0

    def _refresh_partner_context(self):
        source=self._resolved_social_source(self._current_social_source)
        self._current_partner_context=(0 if not source or not self._current_social_subject else
            self._partner_interaction_context(source,self._current_social_subject))

    def experience_social_source_continuity(self, anchor_source, outcome_q16=Q,
                                            effort_q16=Q//64, duration=1,
                                            independent_return=True):
        """Earn current-source continuity with one previously lived opaque source."""
        source=int(self._current_social_source);anchor=int(anchor_source)
        if min(source,anchor)<=0 or source==anchor:raise RuntimeError('adult:no_social_source_pair')
        start=self._advance();end=start+max(1,int(duration));independent=bool(independent_return)
        self.social_identity_credit.observe_use(anchor,start,end,int(effort_q16),source)
        self.social_identity_credit.observe_return(
            anchor,int(outcome_q16),0,end+1,independent,source)
        self.social_identity_credit.observe_control(anchor,True,independent,source)
        self._select_epoch+=1;self._tick=end+1;self._refresh_partner_context()

    def experience_social_source_background(self, anchor_source, outcome_occurs):
        source=int(self._current_social_source);anchor=int(anchor_source)
        if min(source,anchor)<=0 or source==anchor:raise RuntimeError('adult:no_social_source_pair')
        self.social_identity_credit.observe_control(anchor,False,bool(outcome_occurs),source)
        self._select_epoch+=1;self._refresh_partner_context()

    def _recent_social_anchor(self, max_gap=8):
        source=int(self._current_social_source);recent=int(self._recent_social_source)
        if (min(source,recent)<=0 or source==recent
                or self._current_social_subject!=self._recent_social_subject
                or self._tick-self._recent_social_tick>max(0,int(max_gap))):return 0
        return int(self._resolved_social_source(recent))

    def experience_recent_social_source_continuity(self, outcome_q16=Q,
                                                    effort_q16=Q//64, duration=1):
        """Earn cross-source continuity from the resident recent-source candidate only."""
        anchor=self._recent_social_anchor()
        if not anchor:raise RuntimeError('adult:no_recent_social_anchor')
        return self.experience_social_source_continuity(
            anchor,outcome_q16,effort_q16,duration,True)

    def experience_recent_social_source_background(self, outcome_occurs):
        anchor=self._recent_social_anchor()
        if not anchor:raise RuntimeError('adult:no_recent_social_anchor')
        return self.experience_social_source_background(anchor,outcome_occurs)

    def observe_social_contact(self, agent_identity, subject_identity,
                               observed_state_identity, current_action_identity, source):
        """Authenticated contact activates generic relation plus partner-local occurrence."""
        prior_source=int(self._current_social_source);prior_subject=int(self._current_social_subject);prior_tick=int(self._current_social_tick)
        self._clear_current_occurrence()
        self.prospection.observe_contact(
            agent_identity,subject_identity,observed_state_identity,source)
        self._current_social_source=int(source)
        self._current_social_subject=int(subject_identity)
        self._current_social_agent=int(agent_identity)
        self._current_social_tick=int(self._tick)
        if prior_source>0 and prior_source!=self._current_social_source and prior_subject==self._current_social_subject:
            self._recent_social_source=prior_source;self._recent_social_subject=prior_subject;self._recent_social_tick=prior_tick
        self._current_selection_context=self.prospection.pragmatic_context(
            agent_identity,subject_identity,current_action_identity)
        self._refresh_partner_context()

    def observe_social_behavior(self, agent_identity, subject_identity,
                                action_identity, independent=True):
        observed=self.prospection.observe_behavior(
            agent_identity,subject_identity,action_identity,independent)
        if (observed and int(agent_identity)==self._current_social_agent
                and int(subject_identity)==self._current_social_subject):
            source=int(self._current_social_source);start=self._advance();end=start+1
            self.social_identity_credit.observe_use(source,start,end,Q//64,source)
            self.social_identity_credit.observe_return(source,Q,0,end+1,True,source)
            self.social_identity_credit.observe_control(source,True,True,source)
            self._select_epoch+=1;self._tick=end+1;self._refresh_partner_context()
        return observed

    def predict_social_action(self, agent_identity, subject_identity):
        return self.prospection.predict_action(agent_identity,subject_identity)

    def social_prediction_error(self, agent_identity, subject_identity, action_identity):
        return self.prospection.prediction_error(agent_identity,subject_identity,action_identity)

    def _score(self, program_identity, context, state: AdultStateV1):
        row = self.credit.row(int(program_identity))
        # Consequence and somatic history determine preference.  Accessibility is
        # confidence/entrenchment; effort/duration become state-dependent costs.
        value = (self.credit.contextual_causal_value(program_identity, int(context)) +
                 row.accessibility_q16 // 4 - row.uncertainty_q16 // 8 -
                 row.effort_mean_q16 // 8)
        if not self.credit.contextual_action_supported(program_identity,context):
            return -10**30
        urgency = max(0, min(Q, int(state.urgency_q16)))
        pressure = self._effective_pressure_q16(state)
        relief = max(0, min(Q, int(state.relief_q16))) if state.relief_authenticated else 0
        effective_pressure = max(0, pressure - relief)
        value -= (row.duration_mean_q16 * urgency) // (4 * Q)
        # Ordinary effort has an ordinary opportunity cost. Load adds a nonlinear
        # surcharge as the remaining allostatic margin collapses; at zero load the
        # surcharge is exactly zero rather than punishing useful deep cognition.
        if effective_pressure:
            capacity_margin = max(Q // 16, Q - effective_pressure)
            value -= (row.effort_mean_q16 * effective_pressure) // capacity_margin
        return value

    def _action_null_score(self, state: AdultStateV1):
        urgency=max(0,min(Q,int(state.urgency_q16)))
        return -self._effective_pressure_q16(state)-2*urgency

    def _score_organization(self, operator_factor, context, state: AdultStateV1):
        factor=int(operator_factor);row=self.organization_credit.row(factor)
        value=(self.organization_credit.contextual_causal_value(factor,int(context)) +
               row.accessibility_q16//4-row.uncertainty_q16//8-row.effort_mean_q16//8)
        if not self.organization_credit.contextual_action_supported(factor,context):return -10**30
        urgency=max(0,min(Q,int(state.urgency_q16)))
        pressure=self._effective_pressure_q16(state)
        relief=max(0,min(Q,int(state.relief_q16))) if state.relief_authenticated else 0
        effective_pressure=max(0,pressure-relief)
        value-=(row.duration_mean_q16*urgency)//(4*Q)
        if effective_pressure:
            capacity_margin=max(Q//16,Q-effective_pressure)
            value-=(row.effort_mean_q16*effective_pressure)//capacity_margin
        return value

    def _select_organization_operator(self, context, state=AdultStateV1()):
        winner=0;best=None;ties=0;self.last_organization_touches=0
        for factor in self.organization_credit.candidates(int(context)):
            if self.language.historical_span_pieces(int(factor)) is None:continue
            self.last_organization_touches+=1
            score=self._score_organization(factor,context,state)
            if score<=-10**30:continue
            if best is None or score>best:best=score;winner=int(factor);ties=1
            elif score==best:ties+=1
        return 0 if best is None or ties!=1 else winner

    def organize_current_frontier(self, children, state=AdultStateV1()):
        """Transiently organize current proposition matter; persist no paragraph plan."""
        if not self._current_selection_context or self._pending_language_competition:return None
        level=list(children)
        if len(level)<2:return None
        factor=self._select_organization_operator(self._current_selection_context,state)
        if not factor:return None
        while len(level)>1:
            next_level=[]
            for i in range(0,len(level),2):
                if i+1>=len(level):next_level.append(level[i]);continue
                next_level.append(self._compose_factor(factor,level[i],level[i+1]))
            level=next_level
        return level[0]

    def _score_discourse_candidate(self, proposition_identity, context, state: AdultStateV1):
        proposition=int(proposition_identity);row=self.discourse_credit.row(proposition)
        value=(self.discourse_credit.contextual_causal_value(proposition,int(context)) +
               row.accessibility_q16//4-row.uncertainty_q16//8-row.effort_mean_q16//8)
        if not self.discourse_credit.contextual_action_supported(proposition,context):return -10**30
        urgency=max(0,min(Q,int(state.urgency_q16)))
        pressure=self._effective_pressure_q16(state)
        relief=max(0,min(Q,int(state.relief_q16))) if state.relief_authenticated else 0
        effective_pressure=max(0,pressure-relief)
        value-=(row.duration_mean_q16*urgency)//(4*Q)
        if effective_pressure:
            capacity_margin=max(Q//16,Q-effective_pressure)
            value-=(row.effort_mean_q16*effective_pressure)//capacity_margin
        # Partner-local history may bias only matter that already passed generic
        # discourse support; it cannot introduce a proposition absent from the
        # generic current-situation incidence set.
        value+=self._partner_bias(proposition,state)
        return value

    def _cold_discourse_probe_ready(self, proposition_identity, context, state: AdultStateV1):
        row=self.discourse_credit.row(int(proposition_identity));local=row.contexts.get(int(context))
        return bool(local is not None and local.outcome_samples>=4
                    and abs(local.outcome_mean_q16)<=Q//16
                    and row.control_history_q16>=Q//2
                    and row.prediction_error_q16>0
                    and row.last_context==int(context)
                    and self._tick-row.last_tick>=DISCOURSE_COLD_RETRY_TICKS
                    and self._effective_pressure_q16(state)==0
                    and int(state.urgency_q16)<=0)

    def select_discourse_frontier(self, children, state=AdultStateV1()):
        """Select current globally supported proposition matter against silence/null."""
        if not self._current_selection_context or self._pending_language_competition:return ()
        context=int(self._current_selection_context);participants=set(self.discourse_credit.candidates(context))
        selected=[];cold=[];self.last_discourse_touches=0
        for child in children:
            identity=int(getattr(child,'identity',child))
            if not self._has_leaf(identity):continue
            if identity in participants:
                self.last_discourse_touches+=1
                score=self._score_discourse_candidate(identity,context,state)
                if score>0:selected.append(child)
                elif self._cold_discourse_probe_ready(identity,context,state):cold.append(child)
                continue
            if self.social_topic_supported(context,identity):
                self.last_discourse_touches+=1;selected.append(child)
        if selected:return tuple(selected)
        return tuple(cold) if len(cold)==1 else ()

    def organize_relevant_frontier(self, children, state=AdultStateV1()):
        """Select globally relevant propositions, then organize each local boundary."""
        selected=self.select_discourse_frontier(children,state)
        self.last_discourse_selected=tuple(int(getattr(child,'identity',child)) for child in selected)
        if len(selected)==1:
            leaf=selected[0]
            return CompositionWitnessV1(
                int(leaf.identity),int(self._current_selection_context),0,(),
                int(getattr(leaf,'depth',0)),tuple(leaf.surface),())
        if len(selected)<2:return None
        root=selected[0]
        for i in range(1,len(selected)):
            local_context=int(getattr(selected[i-1],'identity',selected[i-1]))
            factor=self._select_organization_operator(local_context,state)
            if not factor:return None
            root=self._compose_factor(factor,root,selected[i])
        return CompositionWitnessV1(
            root.identity,int(self._current_selection_context),root.template_identity,
            root.child_identities,root.depth,root.surface,root.pieces)

    def current_discourse_frontier(self, state=AdultStateV1(),min_transition_samples=2):
        """Rematerialize one unique learned order without caller-supplied members/order."""
        context=int(self._current_selection_context);self.last_discourse_order_touches=0
        if not context or self._pending_language_competition:return ()
        candidates=tuple(self.discourse_credit.candidates(context))
        selected={pid for pid in candidates
                  if self._has_leaf(pid) and self._score_discourse_candidate(pid,context,state)>0}
        if not selected:
            social=tuple(sorted(proposition for (row_context,proposition) in self._social_topic_sources
                                if row_context==context and self._has_leaf(proposition) and self.social_topic_supported(context,proposition)))
            self.last_discourse_order_touches=len(social)
            return tuple(ConstructionLeafV1(pid,0,self._leaf_surface(pid)) for pid in social) if social else ()
        self.last_discourse_order_touches=len(selected)
        if len(selected)<2:return ()
        successors={pid:self.discourse_credit.contextual_expected_successor(
                        pid,context,min_transition_samples) for pid in selected}
        if any(nxt and nxt not in selected for nxt in successors.values()):return ()
        incoming={nxt for nxt in successors.values() if nxt}
        starts=selected-incoming
        if len(starts)!=1:return ()
        ordered=[];seen=set();current=next(iter(starts))
        while current and current not in seen:
            ordered.append(current);seen.add(current);current=successors[current]
        if current or seen!=selected:return ()
        return tuple(ConstructionLeafV1(pid,0,self._leaf_surface(pid)) for pid in ordered)

    def organize_current_discourse(self, state=AdultStateV1()):
        """Select, linearize and transiently compose current resident proposition matter."""
        ordered=self.current_discourse_frontier(state)
        return self.organize_relevant_frontier(ordered,state) if ordered else None

    def choose_public_plan(self, state=AdultStateV1()):
        """Choose one Adult-owned public program or transient multi-proposition plan."""
        chosen=self.choose(state)
        if self._pending_language_competition:return chosen
        if self._current_partner_context:
            plan=(self._current_language_followup_plan(chosen)
                  or self._current_program_role_plan(chosen) or chosen)
        else:
            plan=(self.organize_current_discourse(state)
                  or self._current_program_role_plan(chosen) or chosen)
        return self._resource_compete_public_realization(chosen,plan,state)

    @staticmethod
    def _public_plan_identity(plan):
        return int(getattr(plan,'identity',plan)) if plan else 0

    def _public_expression_status(self, context, plan, extent):
        """Recompete after state/history change; changed contact merely suspends."""
        context=int(context);plan=int(plan);extent=int(extent)
        if self._current_selection_context!=context:return 'suspended'
        self.last_public_expression_revalidations+=1
        if (tuple(getattr(self,'last_discourse_selected',()))==(plan,)
                and plan in self.discourse_credit.candidates(context)
                and (self._score_discourse_candidate(plan,context,AdultStateV1())>0
                     or self._cold_discourse_probe_ready(plan,context,AdultStateV1()))):
            return 'active'
        current=self.choose_public_plan();current_identity=self._public_plan_identity(current)
        if not current_identity:return 'suspended'
        if current_identity==plan:return 'active'
        old_key=(context,plan);row=self._pending_public_expression.get(old_key)
        surface=getattr(current,'surface',())
        if row is not None:
            ordinal,digest=row;ordinal=int(ordinal);surface=tuple(surface)
            self.last_public_expression_revision_hash_bytes=ordinal
            new_key=(context,current_identity)
            if (0<ordinal<len(surface) and len(surface)>=extent
                    and hashlib.sha256(bytes(surface[:ordinal])).hexdigest()==digest
                    and (new_key not in self._pending_public_expression
                         or self._pending_public_expression[new_key]==row)):
                self._pending_public_expression.pop(old_key)
                self._pending_public_expression[new_key]=row
                return 'revised'
        return 'superseded'

    def _can_begin_public_expression(self, context, plan):
        key=(int(context),int(plan))
        return key in self._pending_public_expression or len(self._pending_public_expression)<4

    def _commit_public_expression(self, context, plan, ordinal, prefix, extent, channel=0):
        key=(int(context),int(plan));ordinal=int(ordinal);extent=int(extent);channel=max(0,int(channel))
        if ordinal>=extent:
            self._pending_public_expression.pop(key,None)
            self._last_completed_public_context=int(context)
            self._last_completed_public_plan=int(plan)
            if channel>0 and (channel in self._last_completed_public_context_by_channel or len(self._last_completed_public_context_by_channel)<MAX_COMPLETED_PARTNER_CONTEXTS):
                self._last_completed_public_context_by_channel[channel]=int(context)
                self._last_completed_public_plan_by_channel[channel]=int(plan)
                history=list(self._completed_public_context_history_by_channel.get(channel,()))
                if not history or history[-1]!=int(context):history.append(int(context))
                self._completed_public_context_history_by_channel[channel]=tuple(history[-MAX_COMPLETED_CONTEXT_HISTORY:])
                if self.programs.factor(int(plan)) is not None or self._has_leaf(int(plan)):
                    episodes=list(self._completed_public_episode_history_by_channel.get(channel,()))
                    episode=(int(context),int(plan))
                    if episode in episodes:episodes.remove(episode)
                    episodes.append(episode)
                    if len(episodes)>MAX_COMPLETED_CONTEXT_HISTORY:
                        indexed=list(enumerate(episodes))
                        pinned=None
                        if self._pending_episode_retrieval and int(self._pending_episode_retrieval[0])==channel:
                            pinned=(int(self._pending_episode_retrieval[1]),int(self._pending_episode_retrieval[2]))
                        candidates=[row for row in indexed if row[1]!=pinned] or indexed
                        evict_index,_evict=min(candidates,key=lambda row:(self._completed_public_episode_retention.get((channel,row[1][0],row[1][1]),0),row[0]))
                        evicted=episodes.pop(evict_index)
                        self._completed_public_episode_retention.pop((channel,evicted[0],evicted[1]),None)
                    self._completed_public_episode_history_by_channel[channel]=tuple(episodes)
            return
        if key not in self._pending_public_expression and len(self._pending_public_expression)>=4:
            raise RuntimeError('adult:pending_public_expression_capacity')
        self._pending_public_expression[key]=(ordinal,str(prefix))

    def _discard_public_expression(self, context, plan):
        self._pending_public_expression.pop((int(context),int(plan)),None)

    def last_completed_public_context(self, channel=0):
        channel=max(0,int(channel))
        return int(self._last_completed_public_context if channel==0 else self._last_completed_public_context_by_channel.get(channel,0))

    def last_completed_public_plan(self, channel=0):
        channel=max(0,int(channel))
        return int(self._last_completed_public_plan if channel==0 else self._last_completed_public_plan_by_channel.get(channel,0))

    def completed_public_context_history(self, channel):
        return tuple(self._completed_public_context_history_by_channel.get(max(0,int(channel)),()))

    def completed_public_episode_history(self, channel):
        return tuple(self._completed_public_episode_history_by_channel.get(max(0,int(channel)),()))

    def reinforce_completed_public_episode(self, channel, context, plan, source, outcome, independent=True):
        channel=max(0,int(channel));context=int(context);plan=int(plan);source=int(source);outcome=int(outcome)
        episode=(context,plan)
        if (not independent or channel<=0 or source<=0 or outcome==0
                or episode not in self._completed_public_episode_history_by_channel.get(channel,())):return False
        key=(channel,context,plan);prior=int(self._completed_public_episode_retention.get(key,0))
        score=max(-8,min(8,prior+(1 if outcome>0 else -1)))
        if score:self._completed_public_episode_retention[key]=score
        else:self._completed_public_episode_retention.pop(key,None)
        return True

    def completed_public_episode_retention(self, channel, context, plan):
        return int(self._completed_public_episode_retention.get((max(0,int(channel)),int(context),int(plan)),0))

    def _partner_bias(self, program_identity, state=AdultStateV1()):
        context=int(self._current_partner_context);pid=int(program_identity)
        if not context or pid not in self.partner_credit.candidates(context):return 0
        if not self.partner_credit.contextual_action_supported(pid,context):return 0
        row=self.partner_credit.row(pid)
        value=(self.partner_credit.contextual_causal_value(pid,context)-
               row.uncertainty_q16//8-row.effort_mean_q16//8)
        urgency=max(0,min(Q,int(state.urgency_q16)));pressure=self._effective_pressure_q16(state)
        relief=max(0,min(Q,int(state.relief_q16))) if state.relief_authenticated else 0
        effective_pressure=max(0,pressure-relief)
        value-=(row.duration_mean_q16*urgency)//(4*Q)
        if effective_pressure:
            capacity_margin=max(Q//16,Q-effective_pressure)
            value-=(row.effort_mean_q16*effective_pressure)//capacity_margin
        return value

    def _select_partner_factored(self, state=AdultStateV1()):
        generic=int(self._current_selection_context);partner=int(self._current_partner_context)
        if not generic or not partner:return self._select(generic,state) if generic else 0
        generic_candidates=set(self.credit.candidates(generic));local_candidates=set(self.partner_credit.candidates(partner))
        winner=0;best=self._action_null_score(state);ties=0;self.last_select_touches=0
        for pid in sorted(generic_candidates|local_candidates):
            if self.programs.factor(pid) is None or pid not in generic_candidates:continue
            generic_score=self._score(pid,generic,state)
            if generic_score<=-10**30:continue
            self.last_select_touches+=1;score=generic_score+self._partner_bias(pid,state)
            if best is None or score>best:best=score;winner=pid;ties=1
            elif score==best:ties+=1
        return 0 if best is None or ties!=1 else winner

    def _select(self, context, state):
        key=(int(context),int(state.urgency_q16),int(state.pressure_q16),int(state.relief_q16),bool(state.relief_authenticated),self._select_epoch)
        if key==self._cached_select_key:
            self.last_select_touches=0;return self._cached_select
        winner=0;best=self._action_null_score(state);ties=0;self.last_select_touches=0
        for pid in self.credit.candidates(context):
            if self.programs.factor(pid) is None:continue
            self.last_select_touches+=1
            score=self._score(pid,context,state)
            if score<=-10**30:continue
            if best is None or score>best:best=score;winner=pid;ties=1
            elif score==best:ties+=1
        out=0 if best is None or ties!=1 else winner
        self._cached_select_key=key;self._cached_select=out
        return out

    def choose(self, state=AdultStateV1()):
        """Select from generic situation plus any current authenticated partner factor."""
        if not self._current_selection_context:return 0
        if (self._pending_language_competition
                and self._current_selection_context==self._language_competition_context(
                    self._pending_language_competition)):
            return 0
        if self._current_partner_context:
            chosen=self._select_partner_factored(state)
            if chosen:
                if self._pending_span_reply_role:
                    roles,partner_context,selected=self._pending_span_reply_role
                    if partner_context==int(self._current_partner_context) and selected==0:self._pending_span_reply_role=(roles,partner_context,int(chosen))
                if not self._current_partner_action_ticket:self._mint_partner_action_ticket()
                self._current_partner_action=(int(self._current_partner_context),int(chosen))
            return chosen
        return self._select(self._current_selection_context,state)

    def internal_tick(self, state=AdultStateV1()):
        """Advance resident organism time and any pending learned settling."""
        self._advance();self.language.consolidate_scene_streams()
        alternatives=self._pending_language_competition
        if not alternatives or not self._language_competition_active:return 0
        context=self._language_competition_context(alternatives)
        if self._current_selection_context!=context:
            self._language_competition_active=False;return 0
        winner=self._select(context,state)
        if not winner:
            self._language_competition_evidence_q16=0
            self._language_competition_leader=0
            self._language_competition_active=False
            return 0
        contextual_credit=self.credit.row(winner).contexts.get(context)
        if contextual_credit is None or contextual_credit.outcome_samples<=0:
            self._language_competition_evidence_q16=0
            self._language_competition_leader=winner
            self._language_competition_active=False
            return 0
        score=self._score(winner,context,state)
        if score<=0:
            self._language_competition_evidence_q16=0
            self._language_competition_leader=winner
            self._language_competition_active=False
            return 0
        if winner!=self._language_competition_leader:
            self._language_competition_evidence_q16=0
            self._language_competition_leader=winner
        # Fixed-point evidence accumulation: stronger learned support reaches the
        # action bound sooner. The quantization floor bounds silicon work; it is
        # not an urgency clock and contributes no evidence when support <= 0.
        # Repeated independent returns increase precision without changing the
        # learned mean into a reward counter. This is a smooth evidence law,
        # not a named cognitive phase or host-authored timeout ladder.
        samples=int(contextual_credit.outcome_samples)
        precision_q16=(samples*Q)//(samples+1) if samples else Q//16
        drive=max(Q//32,min(Q,(int(score)*precision_q16)//Q))
        self._language_competition_evidence_q16=min(
            Q,self._language_competition_evidence_q16+drive)
        if self._language_competition_evidence_q16<Q:return 0
        self._language_competition_active=False
        return winner

    def internal_work_pending(self):
        """Content-free scheduler boundary; exposes no context or candidate."""
        return bool(self._pending_language_competition and self._language_competition_active)

    def _probe_choice(self, context, state=AdultStateV1()):
        """Workbench intervention for contrasts; never a body or production surface."""
        return self._select(int(context),state)

    def public_surface(self, program_identity):
        identity=int(program_identity)
        if self.programs.factor(identity) is None:return None
        return self._rematerialize_program_surface(identity)

    def expression(self, program_identity):
        if isinstance(program_identity,CompositionWitnessV1):
            return WitnessSurfaceExpressionV1(self,program_identity)
        return ProgramSurfaceExpressionV1(self,int(program_identity))

    def program_depth(self, program_identity):
        identity=int(program_identity)
        if self.programs.factor(identity) is None:return -1
        chunk=self.programs.chunks.get(identity)
        return 0 if chunk is None else int(chunk.depth)

    def public_bytes(self, program_identity):
        identity=int(program_identity)
        if self.programs.factor(identity) is None:return 0
        return self._program_surface_extent(identity)

    def current_width(self, program_identity):
        # CausalChunkBank's decision depth is one after a consequence-backed chunk
        # is executable: the higher controller handles one opaque constituent.
        return self.programs.decision_depth(int(program_identity)) if int(program_identity) in self.programs.chunks else 1

    def observe_social_history(self, features, source):
        """Induce/update one opaque partner-latent hypothesis from observed behavior."""
        return self.social.observe_history(features, source)

    def settle_social_action(self, hypothesis_identity, program_identity, outcome,
                             source, independent=True):
        """Learn how one public program fares under one inferred social hypothesis."""
        if self.programs.factor(int(program_identity)) is None:
            raise KeyError(program_identity)
        return self.social.observe_action_return(
            int(hypothesis_identity), int(program_identity), int(outcome),
            int(source), bool(independent))

    def choose_social(self, observed_features, candidates, unresolved_program=0):
        """Choose public language from inferred partner state; unresolved stays explicit."""
        hypothesis = self.social.infer(observed_features)
        if hypothesis == 0:
            return int(unresolved_program), 0
        chosen = self.social.choose(hypothesis, tuple(map(int, candidates)))
        return chosen, hypothesis
