#!/usr/bin/env python3
"""Ordinary contact teaches a productive inquiry construction, never an ask policy."""
from __future__ import annotations

LIFE_AFTER=('reference_life_extension_open_state_prompt_v1',)
INQUIRY_CONTEXT=0xB221
RELATION=0xB222
CAUSAL_RELATION_CONTEXT=0xB223
CAUSAL_INQUIRY_CONTEXT=0xB224
CAUSAL_RELATION=0xB225
CAUSAL_FANIN_CONTEXT=0xB226
CAUSAL_FANIN_INQUIRY_CONTEXT=0xB227
CAUSAL_FANIN_RELATION=0xB228
CAUSAL_INQUIRY_SOURCES=(0xDF60,0xDF61,0xDF62)
CAUSAL_FANIN_SOURCES=(0xDF70,0xDF71,0xDF72)
TESTIMONY_PARTNER=0xEE01
TESTIMONY_CONTROL_PARTNER=0xEE02
TESTIMONY_MOTOR_SOURCE=0xEE11
TESTIMONY_WORLD_SOURCE=0xEE21
TESTIMONY_ALARM=0xA120
# Reuse the already authenticated resource-body route; a new semantic body source
# would add no causal distinction and would only consume provenance capacity.
TESTIMONY_BODY_SOURCE=0xDFA0


def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
    rows=[]
    def add(lane,source=0,payload=()):
        rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))

    state_context=LanguageMasteryAdultV1._somatic_appraisal_language_context()
    f=LanguageMasteryAdultV1._somatic_appraisal_feature
    # Two crossed construction examples: neither is the later partial-control/loaded
    # state.  The Adult must reuse the learned construction on that held-out tuple.
    examples=(
        ((f(1,1),f(2,1),f(3,0)),
         b'Does my current state feel manageable: I can influence what happens, and my body is settled?'),
        ((f(1,0),f(2,0),f(3,1)),
         b'Does my current state feel aversive: I have little control over what happens, and my body is under strain?'),
    )
    # These contacts teach only state-construction -> inquiry-construction incidence.
    # Whether an inquiry is worth expressing is decided later by resident causal
    # competition, not by these examples or their source identities.
    for index,(atoms,surface) in enumerate(examples):
        left=int(start)+len(rows)+1;add('scene',0xDF10+index,(state_context,*atoms))
        right=int(start)+len(rows)+1;add('scene',0xDF20+index,(INQUIRY_CONTEXT,*atoms))
        add('surface',0xDF30+index,tuple(surface))
        add('relation',0xDF40+index,(RELATION,left,right))
    # Ordinary examples relate two witnessed causal bindings to a productive
    # verification construction.  The later heater relation is held out, and the
    # Adult—not this curriculum—must decide whether a partner contact warrants use.
    causal_examples=(
        ((0xA104,0xA107),b'Because warm air dries the soil, dry soil cracks the surface.',b'Is it true that dry soil cracks the surface because warm air dries the soil?'),
        ((0xA107,0xA108),b'Because dry soil cracks the surface, plant roots lose water.',b'Is it true that plant roots lose water because dry soil cracks the surface?'),
        ((0xA108,0xA109),b'Because plant roots lose water, plant stomata close.',b'Is it true that plant stomata close because plant roots lose water?'),
    )
    for index,(atoms,statement,question) in enumerate(causal_examples):
        source=CAUSAL_INQUIRY_SOURCES[index]
        left=int(start)+len(rows)+1;add('scene',source,(CAUSAL_RELATION_CONTEXT,*atoms))
        add('surface',source+0x08,tuple(statement))
        right=int(start)+len(rows)+1;add('scene',source+0x10,(CAUSAL_INQUIRY_CONTEXT,*atoms))
        add('surface',source+0x20,tuple(question))
        add('relation',source,(CAUSAL_RELATION,left,right))
    # Alternative questions are learned as a three-part construction.  Surface
    # "or" has no causal authority: the source scene binds two witnessed
    # alternatives and one effect, and later field topology must supply all three.
    fanin_examples=(
        ((0xA104,0xA107,0xA108),
         b'warm air dries the soil and dry soil cracks the surface before plant roots lose water.',
         b'Is it true that plant roots lose water because warm air dries the soil, or because dry soil cracks the surface?'),
        ((0xA107,0xA108,0xA109),
         b'dry soil cracks the surface and plant roots lose water before plant stomata close.',
         b'Is it true that plant stomata close because dry soil cracks the surface, or because plant roots lose water?'),
        ((0xA108,0xA109,0xA10A),
         b'plant roots lose water and plant stomata close before plant leaves wilt.',
         b'Is it true that plant leaves wilt because plant roots lose water, or because plant stomata close?'),
    )
    for index,(atoms,statement,question) in enumerate(fanin_examples):
        source=CAUSAL_FANIN_SOURCES[index]
        left=int(start)+len(rows)+1;add('scene',source,(CAUSAL_FANIN_CONTEXT,*atoms))
        add('surface',source+0x08,tuple(statement))
        right=int(start)+len(rows)+1;add('scene',source+0x10,(CAUSAL_FANIN_INQUIRY_CONTEXT,*atoms))
        add('surface',source+0x20,tuple(question))
        add('relation',source,(CAUSAL_FANIN_RELATION,left,right))
    add('checkpoint_mark',0,('endogenous_state_inquiry_language',))
    # Recreate a common current resource challenge only after the inquiry surface is
    # learned.  The historical partner differences are untouched; quiet recovery is
    # the predeclared reversal arm.
    for offset in range(6):add('body_load',0xDFA0+offset,(96+offset,1<<15))
    add('checkpoint_mark',0,('endogenous_state_inquiry_loaded',))
    add('quiet',0,(64,))
    add('checkpoint_mark',0,('endogenous_state_inquiry_recovered',))

    # One continuing-Life causal conflict joins source monitoring, productive
    # inquiry, resident intervention and later discourse.  No event supplies the
    # expected question or corrected relation: those are rematerialized from the
    # Adult's acquired structures and independently returned field consequences.
    root,heater,stomata,roots,leaves,growth,alarm=(
        0xA104,0xA106,0xA109,0xA108,0xA10A,0xA10B,TESTIMONY_ALARM)
    scenes={}
    for atom,surface in (
            (root,b''),(heater,b''),(stomata,b''),(roots,b''),(leaves,b''),
            (growth,b''),(alarm,b'the irrigation alarm sounds')):
        for witness in range(2):
            scenes[atom]=int(start)+len(rows)+1;source=0xEC00+atom+witness
            add('scene',source,(100,atom))
            if surface:add('surface',source,tuple(surface))
    old_sources=(0xED10,0xED11,0xED12);old_field=int(start)+len(rows)+1
    add('causal_field',0xED01,(scenes[heater],scenes[stomata],scenes[alarm],256))
    for source in old_sources:
        for _ in range(4):add('resident_world_step',source,(old_field,scenes[heater],1))
    add('authenticated_utterance',TESTIMONY_PARTNER,tuple(
        b'Because heavy rain soaks the garden, dark soil holds the water.'))
    add('checkpoint_mark',0,('testimony_revision_old_world',))
    old_action=int(start)+len(rows)+1
    add('partner_causal_dialogue_opportunity',TESTIMONY_PARTNER,(scenes[alarm],))
    add('causal_dialogue_return',TESTIMONY_WORLD_SOURCE,(old_action,1<<16,0,1))
    old_growth_action=int(start)+len(rows)+1
    add('partner_causal_dialogue_opportunity',TESTIMONY_PARTNER,(scenes[growth],))
    add('causal_dialogue_return',TESTIMONY_WORLD_SOURCE+2,(old_growth_action,1<<16,0,1))
    # A second attempt receives no independent consequence.  Both relations
    # remain learned and acted, but their lived action efficacy now differs.
    old_growth_failure=int(start)+len(rows)+1
    add('partner_causal_dialogue_opportunity',TESTIMONY_PARTNER,(scenes[growth],))
    add('causal_dialogue_return',TESTIMONY_WORLD_SOURCE+3,(old_growth_failure,0,0,0))
    add('checkpoint_mark',0,('testimony_revision_committed',))
    for source in old_sources:add('source_withdrawal',source)
    # The earlier leaf->growth field came from these independent canonical-Life
    # witnesses. Withdraw its present authority after the Adult has acted it so
    # the partner's incompatible root->growth claim remains a live hypothesis.
    for source in (0xF610,0xF611,0xF612,0xF690,0xF691,0xF692):
        add('source_withdrawal',source)
    new_field=int(start)+len(rows)+1
    add('causal_field',0xEE31,(scenes[stomata],scenes[roots],scenes[alarm],256))
    growth_field=int(start)+len(rows)+1
    add('causal_field',0xEE32,(scenes[roots],scenes[leaves],scenes[growth],256))
    add('authenticated_utterance',TESTIMONY_PARTNER,tuple(
        b'Because plant stomata close, the irrigation alarm sounds.'))
    add('authenticated_utterance',TESTIMONY_PARTNER,tuple(
        b'Because plant roots lose water, plant growth slows.'))
    # One guided sample is insufficient to resolve the alternative field.
    add('resident_world_step',0xEE40,(new_field,scenes[roots],1))
    add('resident_world_step',0xEE43,(growth_field,scenes[roots],1))
    # An independently returned low-load action supplies a lived comparison for
    # the same causal closure. The later strained Adult can therefore discover a
    # real capability gap rather than consulting an authored ideal state.
    control_action=int(start)+len(rows)+1
    add('partner_causal_dialogue_opportunity',TESTIMONY_CONTROL_PARTNER,(scenes[root],))
    add('causal_dialogue_return',TESTIMONY_WORLD_SOURCE+1,(control_action,1<<16,0,1))
    add('checkpoint_mark',0,('testimony_revision_need',))
    # The same Adult now has two lawful information needs. Acute load makes the
    # learned self-state/capability gap win transient competition; it does not
    # erase the source-causal dispute. After authenticated contact and quiet
    # recovery, the deferred causal question wins without sentence replay.
    for offset in range(6):add('body_load',TESTIMONY_BODY_SOURCE,(96+offset,1<<15))
    add('checkpoint_mark',0,('testimony_revision_competing_loaded',))
    somatic_inquiry_action=int(start)+len(rows)+1
    add('endogenous_inquiry_opportunity',TESTIMONY_MOTOR_SOURCE,(TESTIMONY_PARTNER,))
    add('endogenous_inquiry_motor_return',TESTIMONY_PARTNER,(somatic_inquiry_action,))
    add('quiet',0,(64,))
    add('endogenous_inquiry_resolution',TESTIMONY_WORLD_SOURCE,(somatic_inquiry_action,))
    add('checkpoint_mark',0,('testimony_revision_competing_recovered',))
    inquiry_action=int(start)+len(rows)+1
    add('endogenous_inquiry_opportunity',TESTIMONY_MOTOR_SOURCE+1,(TESTIMONY_PARTNER,))
    add('endogenous_inquiry_motor_return',TESTIMONY_PARTNER,(inquiry_action,))
    add('checkpoint_mark',0,('testimony_revision_conflict',))
    for _ in range(3):add('resident_world_step',0xEE40,(new_field,scenes[roots],1))
    for source in (0xEE41,0xEE42):
        for _ in range(4):add('resident_world_step',source,(new_field,scenes[roots],1))
    add('endogenous_inquiry_resolution',TESTIMONY_WORLD_SOURCE,(inquiry_action,))
    # Settlement of the first dispute exposes the second rather than ending the
    # Adult's discourse horizon. Its next public question is then answered by a
    # separate independently sampled field, not by the partner's wording.
    growth_inquiry_action=int(start)+len(rows)+1
    add('endogenous_inquiry_opportunity',TESTIMONY_MOTOR_SOURCE+2,(TESTIMONY_PARTNER,))
    add('endogenous_inquiry_motor_return',TESTIMONY_PARTNER,(growth_inquiry_action,))
    add('checkpoint_mark',0,('testimony_revision_second_conflict',))
    for _ in range(3):add('resident_world_step',0xEE43,(growth_field,scenes[roots],1))
    for source in (0xEE44,0xEE45):
        for _ in range(4):add('resident_world_step',source,(growth_field,scenes[roots],1))
    add('endogenous_inquiry_resolution',TESTIMONY_WORLD_SOURCE+4,(growth_inquiry_action,))
    # A final opportunity is deliberately silent after both lawful needs settle.
    add('endogenous_inquiry_opportunity',TESTIMONY_MOTOR_SOURCE+3,(TESTIMONY_PARTNER,))
    add('checkpoint_mark',0,('testimony_revision_settled',))
    return tuple(rows)
