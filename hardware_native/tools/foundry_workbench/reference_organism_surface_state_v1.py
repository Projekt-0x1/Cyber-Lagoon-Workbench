#!/usr/bin/env python3
"""Stateless projection from live resident/body state into surface conditions."""
from __future__ import annotations

WORLD_NUMEROSITY_TAG=0x4E554D45524F5349


def body_condition(owner,body_state_tag:int,contact_source:int=0):
    if (not owner.body_state or owner.body_state_occurrence<=0 or owner.body_state_source<=0
            or owner.body_state_source in owner.withdrawn_sources):return 0
    signature=owner.population.signature((int(body_state_tag),*owner.body_state))
    condition=int(owner.recruitment.morphology_identity(signature))
    if int(contact_source)!=int(owner.body_state_source) and not owner.language.condition_supported(condition):return 0
    return condition


def world_numerosity_condition(owner,atom:int):
    """Project current shared-feature population size into opaque morphology matter.

    The feature must have been learned on at least two independently sourced entity
    identities.  The current world supplies which of those identities participate;
    neither language bytes nor a semantic number label enters this computation.
    """
    atom=int(atom)
    if (not owner.world_state or owner.world_state_occurrence<=0 or owner.world_source<=0
            or owner.world_source in owner.withdrawn_sources):return 0
    known=[]
    for entity in owner._entity_feature_index.get(atom,()):
        entity=int(entity)
        if atom not in owner._active_entity_features(entity):continue
        active={int(source) for source in owner.entity_feature_sources.get(entity,())
                if int(source) not in owner.withdrawn_sources}
        if not active:continue
        known.append((entity,active))
    if len(known)<2:return 0
    independent=any(left_source!=right_source
        for left,(_left_entity,left_sources) in enumerate(known)
        for _right_entity,right_sources in known[left+1:]
        for left_source in left_sources for right_source in right_sources)
    if not independent:return 0
    current=set(map(int,owner.world_state))
    count=sum(entity in current for entity,_sources in known)
    if count<=0:return 0
    # Small exact numerosities are the bounded Workbench lowering of an
    # overlapping population code.  Saturation prevents unbounded cardinal state.
    signature=owner.population.signature((WORLD_NUMEROSITY_TAG,min(count,4)))
    return int(owner.recruitment.morphology_identity(signature))


def surface_conditions(owner,atom:int,body_state_tag:int,contact_source:int=0):
    conditions=list(owner.entity_conditions.get(int(atom),()))
    condition=body_condition(owner,body_state_tag,contact_source) if int(atom) in owner.body_target else 0
    if condition and condition not in conditions:conditions.append(condition)
    condition=world_numerosity_condition(owner,atom)
    if condition and condition not in conditions:conditions.append(condition)
    return tuple(conditions)


def control_values(owner,atoms,body_state_tag:int):
    values=[]
    for atom in atoms:
        cond=surface_conditions(owner,int(atom),body_state_tag)
        if not cond:
            found={surface_conditions(owner,int(other),body_state_tag) for other in owner._overlapping_entities(int(atom))}
            found={row for row in found if row}
            if len(found)==1:cond=next(iter(found))
        values.append(int(cond[0]) if cond else 0)
    return tuple(values)


def surface_context(owner,scene,cond_reinstated:int,body_state_tag:int,mix64,context=0,atoms=()):
    context=int(context or scene.context);atoms=tuple(int(x) for x in (atoms or scene.atoms))
    control=tuple(surface_conditions(owner,atom,body_state_tag,int(scene.source)) for atom in atoms)
    overlap=control_values(owner,atoms,body_state_tag)
    values=tuple(int(row[0]) if row else int(value) for row,value in zip(control,overlap))
    control=tuple(row if row else ((int(value),) if value else ()) for row,value in zip(control,values))
    directed=owner.language.complete_dependencies(context,values) if any(values) else None
    if directed is not None:
        values=directed;control=tuple(row if row else ((int(value),) if value else ()) for row,value in zip(control,values))
    completed=owner.language.complete_compatibility(context,values) if any(values) else None
    if completed is not None:
        control=tuple(row if row else ((int(value),) if value else ()) for row,value in zip(control,completed))
    reinst=tuple(() for _ in atoms)
    if owner.partner_present and owner.partner_source>0:
        partner=int(owner.partner_source)
        reinst=tuple(((int(cond_reinstated),) if atom and owner._shared_reinstated(partner,atom) else ()) for atom in atoms)
    conditions=tuple(r+c for r,c in zip(reinst,control))
    if not any(reinst):return context,conditions
    mask=sum((1<<i) for i,c in enumerate(reinst) if c)
    derived=mix64(context^mix64(mask^0xC0A57E17))&((1<<63)-1)
    return int(derived or 1),conditions
