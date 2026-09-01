#!/usr/bin/env python3
"""Sparse nomination among live scenes; consequence history breaks only real ties."""
from __future__ import annotations

import heapq
import itertools


def _attachment_score(owner,scene):
    atoms,context,binding,_occurrences=owner._surface_view(scene)
    template=owner.language.template(context,len(atoms))
    if template is None:return 0
    return owner._selection_construction_evidence(
        owner._selection_preference_context(scene),int(template.identity[:15],16),binding)


def _expression_score(owner,scene,reinstated_condition):
    """Exact learned consequence for one currently realizable expression."""
    atoms,context,binding,_occurrences=owner._surface_view(scene)
    surface_context,conditions=owner._surface_context(scene,context,atoms)
    preference=owner._selection_preference_context(scene)
    if any(conditions):
        surface,template,identities,form_slots,_veto=(
            owner._realize_conditioned_selected(surface_context,atoms,conditions,
                preference,binding))
        if surface is None and int(surface_context)!=int(context) and not owner.language.template_candidates(int(surface_context),len(atoms)):
            surface,template,identities,form_slots,_veto=(
                owner._realize_conditioned_selected(context,atoms,conditions,
                    preference,binding))
        if surface is None and not any(int(x) and int(x)!=int(reinstated_condition)
                                       for row in conditions for x in row):
            surface,template,identities,_alternatives=owner._realize_explicit_selected(
                context,atoms,preference,binding)
            form_slots=()
    else:
        surface,template,identities,_alternatives=owner._realize_explicit_selected(
            surface_context,atoms,preference,binding)
        form_slots=()
    if surface is None or template is None:return 0
    configuration=owner._selection_configuration(
        int(template.identity[:15],16),identities,form_slots,
        binding_identity=binding)
    value,evidence=owner._selection_configuration_evidence(
        preference,configuration)
    return int(value) if evidence else 0


def select_pending_scene(owner,reinstated_condition):
    owner.last_pending_lookup_touches=0
    if owner.partner_present and owner.partner_source>0:
        prior_id=int(owner.last_shared_episode_by_partner.get(int(owner.partner_source),0))
        prior=owner._episode_by_id.get(prior_id) if prior_id else None
        if prior is not None:
            links=owner._links_from.get(int(prior.scene_identity),())
            owner.last_pending_lookup_touches+=len(links);linked=[]
            for link in links:
                if not link.active or link.source in owner.withdrawn_sources:continue
                scene=owner._scene_by_id.get(int(link.right_scene));owner.last_pending_lookup_touches+=1
                if owner._scene_available(scene):linked.append(scene)
            unique={scene.identity:scene for scene in linked}
            if len(unique)==1:return next(iter(unique.values()))
            if len(unique)>1:return None
    while owner._pending_heap:
        scene=owner._scene_by_id.get(int(owner._pending_heap[0]));owner.last_pending_lookup_touches+=1
        if owner._scene_available(scene):break
        heapq.heappop(owner._pending_heap)
    else:return None
    current=[scene]
    for identity in owner._pending_heap[1:]:
        candidate=owner._scene_by_id.get(int(identity));owner.last_pending_lookup_touches+=1
        if owner._scene_available(candidate):current.append(candidate)
    if len(current)==1:return current[0]

    # A source-authenticated resident link may identify one current root.  This
    # is ordering evidence, not proof that a rendered connective is semantic.
    current_ids={int(candidate.identity) for candidate in current}
    outgoing=set();incoming=set()
    for candidate in current:
        links=owner._links_from.get(int(candidate.identity),())
        owner.last_pending_lookup_touches+=len(links)
        for link in links:
            if (link.active and link.source not in owner.withdrawn_sources
                    and int(link.right_scene) in current_ids):
                outgoing.add(int(candidate.identity));incoming.add(int(link.right_scene))
    roots=outgoing-incoming
    if len(roots)==1:return owner._scene_by_id[next(iter(roots))]
    if len(roots)>1:return None

    scored=[(_expression_score(owner,candidate,reinstated_condition),candidate)
            for candidate in current]
    peak=max(score for score,_candidate in scored)
    winners=[candidate for score,candidate in scored if score==peak]
    if peak>0:return winners[0] if len(winners)==1 else None

    if not scene.binding_identity:return None
    key=(int(scene.channel),int(scene.context),tuple(scene.atoms))
    ids=owner._pending_structure_index.get(key,());owner.last_pending_lookup_touches+=len(ids)
    alternatives={}
    for identity in ids:
        candidate=owner._scene_by_id.get(int(identity))
        if owner._scene_available(candidate) and candidate.binding_identity:
            alternatives.setdefault(int(candidate.binding_identity),candidate)
    if len(alternatives)<=1:return None
    scored=[(_attachment_score(owner,candidate),candidate) for candidate in alternatives.values()]
    peak=max(score for score,_candidate in scored);winners=[candidate for score,candidate in scored if score==peak]
    return winners[0] if peak>0 and len(winners)==1 else None


def select_explicit_template(owner,context,arity,preference_context):
    rows=owner.language.template_candidates(int(context),int(arity));scored=[]
    for row in rows:
        cid=int(row.identity[:15],16);scored.append((row.support,cid,row))
    if not scored:return None,0
    scored.sort(key=lambda x:(-x[0],x[1]));peak=scored[0][0]
    winners=[row for row in scored if row[0]==peak]
    return (winners[0][2],len(scored)) if len(winners)==1 else (None,len(scored))


def select_explicit_lexeme(owner,feature,preference_context):
    rows=owner.language.lexeme_candidates(int(feature));scored=[]
    for support,units,_sources in rows:
        cid=owner.language.lexeme_identity(int(feature),units)
        scored.append((support,cid,units))
    if not scored:return None,0,0
    scored.sort(key=lambda x:(-x[0],x[1]));peak=scored[0][0]
    winners=[row for row in scored if row[0]==peak]
    return ((winners[0][2],winners[0][1],len(scored)) if len(winners)==1
            else (None,0,len(scored)))


def _credited_rows(rows,score,support):
    scored=[(*score(row),row) for row in rows]
    positive=[row for row in scored if row[0]>0]
    if not positive:return rows
    peak=max(row[0] for row in positive)
    credited=[row[-1] for row in positive if row[0]==peak]
    credited_support=max(support(row) for row in credited)
    developmental_peak=max(support(row) for row in rows)
    if developmental_peak>credited_support:
        return [row for row in rows if support(row)==developmental_peak]
    return credited


def _reopened_configuration(owner,context,atoms,template_rows,lexical_rows,
                            binding_identity,prefs):
    pref_template,pref_binding,pref_lexeme=prefs
    available={(pref_template,0,int(row.identity[:15],16))
               for row in template_rows}
    if int(binding_identity):available.add((pref_binding,0,int(binding_identity)))
    for slot,(atom,rows) in enumerate(zip(atoms,lexical_rows),1):
        for _support,units,_sources in rows:
            available.add((pref_lexeme,slot,owner.language.lexeme_identity(
                owner._lexeme_owner(atom,units),units)))
    for template in template_rows:
        key=(int(context),int(template.identity[:15],16),int(binding_identity))
        for configuration in owner._selection_construction_index.get(key,()):
            if not set(configuration).issubset(available):continue
            value,evidence=owner._selection_revisions.evidence(
                context,configuration,owner.withdrawn_sources)
            owner.last_selection_network_touches+=owner._selection_revisions.last_revision_rows
            if evidence and value<=0:return True
    return False


def selection_configuration_evidence(owner,context,configuration):
    value,evidence=owner._selection_revisions.evidence(
        context,configuration,owner.withdrawn_sources)
    owner.last_selection_network_touches+=owner._selection_revisions.last_revision_rows
    return value,evidence


def selection_configuration(template_identity,lexical_identities,form_slots,
                            span_identity,binding_identity,prefs):
    pref_template,pref_binding,pref_form,pref_lexeme,pref_span=prefs
    members=[(pref_template,0,int(template_identity))]
    forms=set(int(x) for x in form_slots)
    if int(binding_identity):members.append((pref_binding,0,int(binding_identity)))
    members.extend((pref_form if i+1 in forms else pref_lexeme,i+1,
                    int(candidate))
                   for i,candidate in enumerate(lexical_identities))
    if int(span_identity):members.append((pref_span,0,int(span_identity)))
    return tuple(members)


def selection_network_identity(digest,context,action_occurrence,
                               closure_identity,occurrences):
    value=int(digest('selection-network-v2',[int(context),
        int(action_occurrence),int(closure_identity),
        [list(row) for row in occurrences]])[:15],16)
    return value or 1


def selection_member_evidence(owner,context,member):
    value,evidence=owner._selection_revisions.member_evidence(
        context,member,owner.withdrawn_sources)
    owner.last_selection_network_touches+=owner._selection_revisions.last_revision_rows
    return value,evidence


def selection_construction_evidence(owner,context,template_identity,
                                    binding_identity,pref_binding,pref_lexeme):
    configurations=owner._selection_construction_index.get(
        (int(context),int(template_identity),int(binding_identity)),())
    values={}
    for configuration in configurations:
        value,_=owner._selection_revisions.evidence(
            context,configuration,owner.withdrawn_sources)
        owner.last_selection_construction_touches+=owner._selection_revisions.last_revision_rows
        values[configuration]=value
    if any(value<0 for value in values.values()):return 0
    positive=[configuration for configuration,value in values.items() if value>0]
    if len(positive)<2:return 0
    ports={}
    for configuration in positive:
        for kind,slot,candidate in configuration[1:]:
            if kind==pref_binding:continue
            if kind!=pref_lexeme:return 0
            ports.setdefault(int(slot),set()).add(int(candidate))
    diversity=min((len(candidates) for candidates in ports.values()),default=0)
    return diversity if diversity>=2 else 0


def realize_explicit_selected(owner,context,atoms,preference_context,
                              binding_identity,surface_proposal,max_candidates,
                              pref_template,pref_binding,pref_lexeme):
    owner.last_selection_network_touches=0
    owner.last_selection_construction_touches=0
    atoms=tuple(int(x) for x in atoms)
    template_rows=owner.language.template_candidates(int(context),len(atoms))
    if not template_rows:return None,None,(),0
    reported_combinations=len(template_rows)
    raw_lexical_rows=[];partner=int(owner.partner_source) if owner.partner_present else 0
    for atom in atoms:
        rows=owner._lexeme_rows(int(atom))
        if not rows:return None,None,(),0
        if partner:
            partner_rows=[row for row in rows if partner in row[2]]
            if partner_rows:rows=partner_rows
            elif any(partner in sources for _support,_units,sources in
                     owner.language.lexeme_observations(int(atom))):
                return None,None,(),0
        raw_lexical_rows.append(rows);reported_combinations*=len(rows)
    reopened=_reopened_configuration(owner,preference_context,atoms,template_rows,
        raw_lexical_rows,binding_identity,
        (pref_template,pref_binding,pref_lexeme))
    if not reopened:
        template_rows=_credited_rows(template_rows,lambda row:
            owner._selection_member_evidence(preference_context,
                (pref_template,0,int(row.identity[:15],16))),lambda row:row.support)
    lexical_rows=[]
    for slot,(atom,rows) in enumerate(zip(atoms,raw_lexical_rows),1):
        if not reopened:
            rows=_credited_rows(rows,lambda row:
                owner._selection_member_evidence(preference_context,
                    (pref_lexeme,slot,owner.language.lexeme_identity(
                        owner._lexeme_owner(atom,row[1]),row[1]))),lambda row:row[0])
        lexical_rows.append(rows)
    combinations=len(template_rows)
    for rows in lexical_rows:
        combinations*=len(rows)
    if combinations>max_candidates:
        template_peak=max(row.support for row in template_rows)
        template_rows=[row for row in template_rows
                       if row.support==template_peak]
        lexical_rows=[[row for row in rows
                       if row[0]==max(candidate[0] for candidate in rows)]
                      for rows in lexical_rows]
        combinations=len(template_rows)
        for rows in lexical_rows:combinations*=len(rows)
        if combinations>max_candidates:
            return None,None,(),reported_combinations
    candidates=[];unambiguous_lexical=all(len(rows)==1 for rows in lexical_rows)
    for template in template_rows:
        stack=[([],[],0,0)]
        for atom,rows in zip(atoms,lexical_rows):
            next_stack=[]
            for surfaces,ids,support,hits in stack:
                for row_support,units,sources in rows:
                    cid=owner.language.lexeme_identity(
                        owner._lexeme_owner(atom,units),units)
                    next_stack.append((surfaces+[units],ids+[cid],
                        support+int(row_support),
                        hits+(1 if partner and partner in sources else 0)))
            stack=next_stack
        tid=int(template.identity[:15],16)
        template_hit=1 if partner and partner in template.sources else 0
        for surfaces,ids,lexical_support,hits in stack:
            configuration=owner._selection_configuration(
                tid,ids,binding_identity=binding_identity)
            value,evidence=owner._selection_configuration_evidence(
                preference_context,configuration)
            construction=(owner._selection_construction_evidence(
                preference_context,tid,binding_identity)
                if unambiguous_lexical else 0)
            candidates.append((value,evidence,
                int(template.support)+lexical_support,tid,tuple(ids),
                tuple(surfaces),template,configuration,construction,
                hits+template_hit))
    if not candidates:return None,None,(),0
    owner.last_selection_candidate_touches=len(candidates)
    earned=[row for row in candidates if row[0]>0]
    if earned:
        causal_peak=max(row[0] for row in earned)
        causal_winners=[row for row in earned if row[0]==causal_peak]
        causal_developmental=max(row[2] for row in causal_winners)
        causal_winners=[row for row in causal_winners
                        if row[2]==causal_developmental]
        admissible=[row for row in candidates if row[0]>=0]
        developmental_peak=max(row[2] for row in admissible)
        winners=([row for row in admissible if row[2]==developmental_peak]
                 if developmental_peak>causal_developmental
                 else causal_winners)
    else:
        structural=[row for row in candidates
                    if row[0]==0 and row[1]==0 and row[8]>=2]
        if structural:
            peak=max(row[8] for row in structural)
            winners=[row for row in structural if row[8]==peak]
        else:
            least=min(row[1] for row in candidates)
            novel=[row for row in candidates if row[1]==least]
            peak=max(row[2] for row in novel)
            winners=[row for row in novel if row[2]==peak]
    if len(winners)!=1:
        peak=max(row[9] for row in winners)
        if peak>0:winners=[row for row in winners if row[9]==peak]
    if len(winners)!=1 and surface_proposal is not None:
        try:
            proposal_tid=int(surface_proposal.template_identity)
            proposal_ids=tuple(map(int,surface_proposal.lexical_identities))
            proposal_alternatives=int(surface_proposal.alternatives)
        except (AttributeError,TypeError,ValueError):
            return None,None,(),len(candidates)
        if proposal_alternatives!=len(candidates):return None,None,(),len(candidates)
        proposed=[row for row in winners
                  if row[3]==proposal_tid and row[4]==proposal_ids]
        if len(proposed)==1:winners=proposed
    if len(winners)!=1:return None,None,(),len(candidates)
    (value,evidence,_developmental,_tid,ids,surfaces,template,
     _configuration,_construction,_hits)=winners[0]
    if evidence and value<=0:return None,None,(),len(candidates)
    return (owner.language.render_template(template,surfaces),template,ids,
            reported_combinations)


def _conditioned_candidate_rows(owner,atom,conditions,pref_form,pref_lexeme,
                                generalized_conditions):
    conditions=tuple(int(x) for x in conditions if int(x))
    if not conditions:
        rows=[]
        for support,units,sources in owner._lexeme_rows(int(atom)):
            donor=owner._lexeme_owner(atom,units)
            rows.append((int(support),tuple(units),
                owner.language.lexeme_identity(donor,units),pref_lexeme,
                tuple(sources)))
        return rows
    generalized=(owner.language.condition_form_candidates(conditions)
                 if tuple(conditions) in generalized_conditions else ())
    if generalized:
        return [(int(support)+int(specificity),tuple(units),
                 owner.language.condition_form_identity(required,units),
                 pref_form,tuple(sources))
                for specificity,support,required,units,sources,_donors,_bases
                in generalized]
    rows={}
    for donor in owner._overlapping_entities(int(atom)) or (int(atom),):
        for specificity,support,required,units,sources in (
                owner.language.form_candidates(int(donor),conditions,True)):
            cid=owner.language.form_identity(int(donor),required,units)
            key=(cid,tuple(units),tuple(required))
            prior=rows.get(key)
            candidate=(int(support)+int(specificity),tuple(units),cid,
                       pref_form,tuple(sources))
            if prior is None or candidate[0]>prior[0]:rows[key]=candidate
    return list(rows.values())


def realize_conditioned_selected(owner,context,atoms,conditions,
                                 preference_context,binding_identity,
                                 surface_proposal,generalized_conditions,
                                 max_candidates,
                                 pref_template,pref_binding,pref_form,
                                 pref_lexeme):
    """Select learned compact forms and report an earned negative veto."""
    owner.last_selection_network_touches=0
    owner.last_selection_construction_touches=0
    atoms=tuple(int(x) for x in atoms)
    conditions=tuple(tuple(int(y) for y in row) for row in conditions)
    if len(atoms)!=len(conditions) or not any(conditions):return None,None,(),(),False
    templates=owner.language.template_candidates(int(context),len(atoms))
    if not templates:return None,None,(),(),False
    partner=int(owner.partner_source) if owner.partner_present else 0
    slots=[];reported=len(templates)
    for atom,condition in zip(atoms,conditions):
        rows=_conditioned_candidate_rows(owner,atom,condition,pref_form,
                                         pref_lexeme,generalized_conditions)
        if not rows:return None,None,(),(),False
        if partner and condition:
            hits=[row for row in rows if partner in row[4]]
            if hits:rows=hits
            elif any(partner in row[2]
                     for donor in (owner._overlapping_entities(int(atom))
                                   or (int(atom),))
                     for row in owner.language.lexeme_observations(int(donor))):
                return None,None,(),(),False
        slots.append(rows);reported*=len(rows)
    templates=_credited_rows(templates,lambda row:
        owner._selection_member_evidence(preference_context,
            (pref_template,0,int(row.identity[:15],16))),lambda row:row.support)
    credited_slots=[]
    for slot,rows in enumerate(slots,1):
        kind=rows[0][3]
        credited_slots.append(_credited_rows(rows,lambda row:
            owner._selection_member_evidence(preference_context,
                (kind,slot,int(row[2]))),lambda row:row[0]))
    combinations=len(templates)
    for rows in credited_slots:combinations*=len(rows)
    if combinations>max_candidates:
        peak=max(row.support for row in templates)
        templates=[row for row in templates if row.support==peak]
        credited_slots=[[row for row in rows
                         if row[0]==max(candidate[0] for candidate in rows)]
                        for rows in credited_slots]
        combinations=len(templates)
        for rows in credited_slots:combinations*=len(rows)
        if combinations>max_candidates:return None,None,(),(),False
    candidates=[]
    for template in templates:
        tid=int(template.identity[:15],16)
        for chosen in itertools.product(*credited_slots):
            surfaces=tuple(row[1] for row in chosen)
            ids=tuple(int(row[2]) for row in chosen)
            form_slots=tuple(i for i,row in enumerate(chosen,1)
                             if row[3]==pref_form)
            configuration=owner._selection_configuration(
                tid,ids,form_slots,binding_identity=binding_identity)
            value,evidence=owner._selection_configuration_evidence(
                preference_context,configuration)
            developmental=int(template.support)+sum(row[0] for row in chosen)
            hits=(1 if partner and partner in template.sources else 0)+sum(
                1 for row in chosen if partner and partner in row[4])
            candidates.append((value,evidence,developmental,tid,ids,surfaces,
                               template,form_slots,hits))
    owner.last_selection_candidate_touches=len(candidates)
    if not candidates:return None,None,(),(),False
    positive=[row for row in candidates if row[0]>0]
    if positive:
        peak=max(row[0] for row in positive)
        winners=[row for row in positive if row[0]==peak]
    else:
        admissible=[row for row in candidates if not row[1] or row[0]>=0]
        if not admissible:return None,None,(),(),True
        if partner:
            hit_peak=max(row[8] for row in admissible)
            if hit_peak:
                admissible=[row for row in admissible if row[8]==hit_peak]
        peak=max(row[2] for row in admissible)
        winners=[row for row in admissible if row[2]==peak]
    if len(winners)!=1:
        hit_peak=max(row[8] for row in winners)
        if hit_peak:winners=[row for row in winners if row[8]==hit_peak]
    if len(winners)!=1 and surface_proposal is not None:
        try:
            proposal_tid=int(surface_proposal.template_identity)
            proposal_ids=tuple(map(int,surface_proposal.lexical_identities))
            proposal_alternatives=int(surface_proposal.alternatives)
        except (AttributeError,TypeError,ValueError):return None,None,(),(),False
        if proposal_alternatives!=reported:return None,None,(),(),False
        winners=[row for row in winners
                 if row[3]==proposal_tid and row[4]==proposal_ids]
    if len(winners)!=1:return None,None,(),(),False
    value,evidence,_support,_tid,ids,surfaces,template,form_slots,_hits=winners[0]
    if evidence and value<=0:return None,None,(),(),True
    surface=owner.language.render_template(template,surfaces)
    return surface,template,ids,form_slots,False
