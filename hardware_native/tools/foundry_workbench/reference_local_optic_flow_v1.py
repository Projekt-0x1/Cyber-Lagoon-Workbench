#!/usr/bin/env python3
"""Content-free multi-component local motion field over raw grayscale frames."""
from __future__ import annotations
from reference_population_v1 import mix64
from reference_raw_visual_elements_v1 import RawVisualElementsV1
from reference_unsupervised_perceptual_features_v1 import UnsupervisedPerceptualFeaturesV1

FLOW_TAG=0xF10F10
SCALE=256


class LocalOpticFlowV1:
    """Track connected contrast components and learn simultaneous local-vector patterns."""
    def __init__(self,learner=None):
        self.learner=learner or UnsupervisedPerceptualFeaturesV1()
        self.previous=()

    @staticmethod
    def _components(frame):
        rows=RawVisualElementsV1._matrix(frame);active={}
        for y in range(len(rows)-1):
            for x in range(len(rows[0])-1):
                events=RawVisualElementsV1._events(rows[y][x],rows[y][x+1],rows[y+1][x],rows[y+1][x+1])
                if events:active[(x,y)]=events
        unseen=set(active);components=[]
        while unseen:
            seed=unseen.pop();stack=[seed];points=[];events=set()
            while stack:
                x,y=stack.pop();points.append((x,y));events.update(active[(x,y)])
                for nxt in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if nxt in unseen:unseen.remove(nxt);stack.append(nxt)
            n=len(points);cx=sum(x for x,_ in points)*SCALE//n;cy=sum(y for _,y in points)*SCALE//n
            components.append((cx,cy,tuple(sorted(events)),n))
        return tuple(sorted(components,key=lambda row:(row[0],row[1],row[2])))

    @staticmethod
    def _direction(dx:int,dy:int):
        if dx==0 and dy==0:return 0
        ax,ay=abs(dx),abs(dy)
        if ax>=ay*2:return 1 if dx>0 else 5
        if ay>=ax*2:return 3 if dy>0 else 7
        if dx>0 and dy>0:return 2
        if dx<0 and dy>0:return 4
        if dx<0 and dy<0:return 6
        return 8

    @staticmethod
    def _magnitude(dx:int,dy:int):
        pixels=max(abs(int(dx)),abs(int(dy)))//SCALE
        return 0 if pixels<=0 else min(7,int(pixels))

    @staticmethod
    def _origin_sector(cx:int,cy:int,center_x:int,center_y:int):
        dx=cx-center_x;dy=cy-center_y;ax,ay=abs(dx),abs(dy)
        if dx==0 and dy==0:return 5
        if ax>=ay:return 1 if dx<0 else 2
        return 3 if dy<0 else 4

    @staticmethod
    def _token(sector:int,direction:int,magnitude:int)->int:
        value=mix64(FLOW_TAG^mix64(int(sector)+3)^mix64(int(direction)+17)^mix64(int(magnitude)+31))
        return int(value&((1<<63)-1) or 1)

    @staticmethod
    def _match(previous,current):
        if not previous or not current:return ()
        candidates=[]
        for i,p in enumerate(previous):
            for j,c in enumerate(current):
                distance=(p[0]-c[0])**2+(p[1]-c[1])**2
                # Similar local contrast signature wins ties before geometry.
                overlap=len(set(p[2])&set(c[2]));candidates.append((distance,-overlap,i,j))
        used_p=set();used_c=set();pairs=[]
        for _distance,_overlap,i,j in sorted(candidates):
            if i in used_p or j in used_c:continue
            used_p.add(i);used_c.add(j);pairs.append((previous[i],current[j]))
        return tuple(pairs)

    def observe_frame(self,frame,contiguous:bool):
        current=self._components(frame)
        if not contiguous:
            self.previous=current;return ()
        if not self.previous or not current:
            self.previous=current;return ()
        center_x=sum(row[0] for row in self.previous)//len(self.previous)
        center_y=sum(row[1] for row in self.previous)//len(self.previous)
        tokens=[]
        for prior,now in self._match(self.previous,current):
            dx=now[0]-prior[0];dy=now[1]-prior[1];direction=self._direction(dx,dy);magnitude=self._magnitude(dx,dy)
            if not direction or not magnitude:continue
            sector=self._origin_sector(prior[0],prior[1],center_x,center_y)
            if sector:tokens.append(self._token(sector,direction,magnitude))
        field=tuple(sorted(set(tokens)))
        if len(field)>=2:self.learner.observe_scene(field)
        self.previous=current;return field

    def flow_feature(self,field)->int:
        row=tuple(sorted(set(int(x) for x in field if int(x)>0)))
        if len(row)!=2:return 0
        return int(self.learner.feature(row[0],row[1]))

    def gap(self):self.previous=()

    def checkpoint(self):return {'schema':1,'learner':self.learner.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('local_flow:checkpoint')
        return cls(UnsupervisedPerceptualFeaturesV1.restore(data['learner']))
