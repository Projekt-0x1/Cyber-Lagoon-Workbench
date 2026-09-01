#!/usr/bin/env python3
"""Joint-attention episode memory promoted only through Adult partner-return consequence."""
from __future__ import annotations
from reference_joint_attention_episode_memory_v1 import JointAttentionEpisodeMemoryV1,JointAttentionEpisodeV1,MAX_JOINT_EPISODES,MIN_JOINT_SOURCES,MAX_EPISODE_SOURCES
from reference_language_mastery_adult_v1 import AdultStateV1

MAX_PENDING_JOINT_EPISODES=16

class ConsequenceQualifiedJointAttentionMemoryV1:
    def __init__(self):
        self.evidence={};self.pending={}

    def stage(self,adult,organism,tracker,grounding,raw_marker,marker_feature,
              point_y2,point_x2,channel,source):
        source=int(source);episode=JointAttentionEpisodeMemoryV1.candidate(
            adult,organism,tracker,grounding,raw_marker,marker_feature,
            point_y2,point_x2,channel)
        if episode is None or source<=0 or len(self.pending)>=MAX_PENDING_JOINT_EPISODES:return None
        partner_context=int(adult._repair_partner_context(episode.channel,episode.event))
        adult._clear_current_occurrence();adult._current_selection_context=int(episode.event)
        adult._current_partner_context=partner_context;adult._current_language_channel=int(episode.channel)
        response=int(adult.choose(AdultStateV1()))
        ticket=int(adult.current_partner_action_ticket())
        if not response or ticket<=0:return None
        self.pending[ticket]=(episode,source,response,partner_context)
        return ticket,response,episode

    def settle_partner_return(self,adult,ticket,response,outcome_q16,independent=True):
        ticket=int(ticket);response=int(response);row=self.pending.get(ticket)
        if row is None:return False
        episode,source,selected,partner_context=row
        if response!=int(selected):return False
        if ticket in getattr(adult,'_pending_span_reply_actions',{}):
            adult.experience_partner_choice(response,int(outcome_q16),independent_return=bool(independent),action_ticket=ticket)
        else:
            if int(adult.current_partner_action_ticket())!=ticket or int(adult._current_partner_context)!=int(partner_context):
                # If the Adult no longer owns this occurrence (expiry/eviction), the
                # memory metadata has no causal authority and must be discarded.
                if ticket not in getattr(adult,'_pending_span_reply_actions',{}) and int(adult.current_partner_action_ticket())!=ticket:self.pending.pop(ticket,None)
                return False
            adult.experience_partner_choice(response,int(outcome_q16),independent_return=bool(independent))
        self.pending.pop(ticket,None)
        if not independent or int(outcome_q16)==0:return False
        key=(episode.channel,episode.marker,episode.entity,episode.event);rows=self.evidence.setdefault(key,{})
        prior=int(rows.get(source,0));value=max(-8,min(8,prior+(1 if int(outcome_q16)>0 else -1)))
        if value:rows[source]=value
        else:rows.pop(source,None)
        if not rows:self.evidence.pop(key,None)
        return True

    def resolve(self,adult,organism,raw,channel):
        atoms=JointAttentionEpisodeMemoryV1._contents(adult,raw);channel=int(channel)
        if not atoms or channel<=0:return None
        candidates=[]
        for (row_channel,marker,entity,event),rows in self.evidence.items():
            if row_channel!=channel or marker not in atoms:continue
            active=sum(1 for source,value in rows.items()
                       if value>0 and int(source) not in adult.language._withdrawn)
            if active<MIN_JOINT_SOURCES:continue
            if not organism._active_entity_features(entity) or not adult._has_leaf(event):continue
            candidates.append(JointAttentionEpisodeV1(row_channel,marker,entity,event))
        if len(candidates)==1:return candidates[0]
        if len(candidates)<2:return None
        scores=[]
        for candidate in candidates:
            event_atoms=set(JointAttentionEpisodeMemoryV1._event_atoms(adult,candidate.event))
            cue=set(atoms)-{candidate.marker};scores.append((len(event_atoms.intersection(cue)),candidate))
        best=max(score for score,_ in scores);winners=[row for score,row in scores if score==best]
        return winners[0] if best>0 and len(winners)==1 else None

    def synchronize_pending(self,adult):
        live=set(getattr(adult,'_pending_span_reply_actions',{}))
        current=int(adult.current_partner_action_ticket())
        if current>0:live.add(current)
        stale=[ticket for ticket in self.pending if int(ticket) not in live]
        for ticket in stale:self.pending.pop(ticket,None)
        return len(stale)

    def checkpoint(self):
        return {'schema':2,
                'evidence':[{'channel':c,'marker':m,'entity':e,'event':v,
                             'rows':[[s,x] for s,x in sorted(rows.items())]}
                    for (c,m,e,v),rows in sorted(self.evidence.items())],
                'pending':[{'ticket':ticket,'channel':episode.channel,'marker':episode.marker,'entity':episode.entity,'event':episode.event,
                            'source':source,'response':response,'partner_context':partner_context}
                    for ticket,(episode,source,response,partner_context) in sorted(self.pending.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0)) not in (1,2):raise ValueError('consequence_joint:checkpoint')
        out=cls()
        for row in data.get('evidence',()):
            key=(int(row.get('channel',0)),int(row.get('marker',0)),int(row.get('entity',0)),int(row.get('event',0)))
            rows={int(s):int(v) for s,v in row.get('rows',())}
            if min(key)<=0 or key in out.evidence or len(rows)>MAX_EPISODE_SOURCES or any(s<=0 or v==0 or not -8<=v<=8 for s,v in rows.items()):raise ValueError('consequence_joint:evidence')
            out.evidence[key]=rows
        if len(out.evidence)>MAX_JOINT_EPISODES:raise ValueError('consequence_joint:capacity')
        pending={}
        for row in data.get('pending',()):
            ticket=int(row.get('ticket',0));episode=JointAttentionEpisodeV1(int(row.get('channel',0)),int(row.get('marker',0)),int(row.get('entity',0)),int(row.get('event',0)));source=int(row.get('source',0));response=int(row.get('response',0));partner_context=int(row.get('partner_context',0))
            if ticket<=0 or ticket in pending or min(episode.channel,episode.marker,episode.entity,episode.event,source,response,partner_context)<=0:raise ValueError('consequence_joint:pending')
            pending[ticket]=(episode,source,response,partner_context)
        if len(pending)>MAX_PENDING_JOINT_EPISODES:raise ValueError('consequence_joint:pending_capacity')
        out.pending=pending;return out
