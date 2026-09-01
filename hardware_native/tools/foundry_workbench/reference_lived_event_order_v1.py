#!/usr/bin/env python3
"""Source-qualified learned event succession from authenticated lived occurrence order."""
from __future__ import annotations
MAX_EVENT_ORDER_EDGES=256
MIN_EVENT_ORDER_SOURCES=2

class LivedEventOrderV1:
    def __init__(self):self.last={};self.edges={};self.withdrawn=set()
    def observe(self,source,occurrence,event):
        source=int(source);occurrence=int(occurrence);event=int(event)
        if source<=0 or occurrence<=0 or event<=0:return False
        prior=self.last.get(source);changed=False
        if prior is not None:
            po,pe=prior
            if occurrence<=po:return False
            if pe!=event:
                key=(int(pe),event)
                if key not in self.edges:
                    if len(self.edges)>=MAX_EVENT_ORDER_EDGES:return False
                    self.edges[key]=set()
                before=len(self.edges[key]);self.edges[key].add(source);changed=len(self.edges[key])!=before
        self.last[source]=(occurrence,event);return changed
    def supports(self,left,right):
        return sum(1 for s in self.edges.get((int(left),int(right)),()) if s not in self.withdrawn)>=MIN_EVENT_ORDER_SOURCES
    def successor(self,left):
        left=int(left);rows=[]
        for (a,b),sources in self.edges.items():
            if a==left and sum(1 for s in sources if s not in self.withdrawn)>=MIN_EVENT_ORDER_SOURCES:rows.append(int(b))
        rows=sorted(set(rows));return rows[0] if len(rows)==1 else 0
    def withdraw_source(self,source):self.withdrawn.add(int(source))
    def checkpoint(self):
        return {'schema':1,'edges':[{'left':l,'right':r,'sources':sorted(src)} for (l,r),src in sorted(self.edges.items())],'withdrawn':sorted(self.withdrawn)}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('lived_event_order:checkpoint')
        out=cls()
        for row in data.get('edges',()):
            l=int(row.get('left',0));r=int(row.get('right',0));src=set(map(int,row.get('sources',())))
            if min(l,r)<=0 or l==r or (l,r) in out.edges or any(s<=0 for s in src):raise ValueError('lived_event_order:row')
            out.edges[(l,r)]=src
        out.withdrawn=set(map(int,data.get('withdrawn',())));return out
