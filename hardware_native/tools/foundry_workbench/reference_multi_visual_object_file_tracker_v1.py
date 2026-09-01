#!/usr/bin/env python3
"""Transient multiple-object files from raw spatial candidates and learned continuity."""
from __future__ import annotations
from reference_raw_visual_elements_v1 import RawVisualElementsV1
from reference_raw_visual_object_candidates_v1 import RawVisualObjectCandidatesV1

MAX_ACTIVE_OBJECT_FILES=4
MAX_OBJECT_FILE_MISSES=1

class MultiVisualObjectFileTrackerV1:
    """Bounded object files: entity -> (y2,x2,vy2,vx2,feature,misses)."""
    def __init__(self):self.active={};self.motion_stability={}
    @staticmethod
    def _feature(level1,crop):
        patches=RawVisualElementsV1.pair_patches(crop)
        if len(patches)!=1:return 0
        return int(level1.feature(*patches[0]))
    @staticmethod
    def _predicted(row):
        y2,x2,vy2,vx2,_feature,misses=map(int,row)
        horizon=1+max(0,misses)
        return y2+vy2*horizon,x2+vx2*horizon
    def observe(self,organism,sensor,temporal,level1,source,sequence,frame,digest):
        rows,contiguous=sensor.ingest(source,sequence,frame,digest);candidates=RawVisualObjectCandidatesV1.extract(rows)
        if len(candidates)>MAX_ACTIVE_OBJECT_FILES:return ()
        if not contiguous:self.active={}
        prior=dict(self.active)
        if not candidates:
            held={}
            for entity,row in prior.items():
                y2,x2,vy2,vx2,feature,misses=map(int,row)
                stability=int(self.motion_stability.get(int(entity),0));budget=MAX_OBJECT_FILE_MISSES+min(2,stability//2)
                if misses<budget:held[int(entity)]=(y2,x2,vy2,vx2,feature,misses+1)
            self.active=held;self.motion_stability={e:self.motion_stability.get(e,0) for e in held};return ()
        derived=[]
        for c in candidates:
            f=self._feature(level1,c.crop)
            if not f:return ()
            derived.append((c,f))
        options_by_candidate=[]
        for ci,(c,f) in enumerate(derived):
            options=[]
            for entity,row in prior.items():
                py2,px2,vy2,vx2,pf,misses=map(int,row)
                relation=int(temporal.relation(pf,f)) if pf!=f else pf
                if relation<=0:continue
                ey,ex=self._predicted(row);distance=abs(int(c.y2)-ey)+abs(int(c.x2)-ex)
                options.append((distance,int(entity),relation,row))
            options_by_candidate.append(tuple(options))
        # Bounded global one-to-one assignment. Preserve as many existing files as
        # possible, then minimize total predicted distance. Equal optima refuse.
        assignments=[]
        def rec(ci,used,chosen,matches,distance):
            if ci==len(derived):
                assignments.append((int(matches),int(distance),tuple(chosen)));return
            rec(ci+1,used,(*chosen,None),matches,distance)
            for option in options_by_candidate[ci]:
                entity=int(option[1])
                if entity in used:continue
                rec(ci+1,used|{entity},(*chosen,option),matches+1,distance+int(option[0]))
        rec(0,set(),(),0,0)
        if not assignments:return ()
        peak=max(row[0] for row in assignments);best_distance=min(row[1] for row in assignments if row[0]==peak)
        best=[row for row in assignments if row[0]==peak and row[1]==best_distance]
        if len(best)!=1:return ()
        chosen=best[0][2];edges={ci:row for ci,row in enumerate(chosen) if row is not None}
        new_active={};result=[]
        for ci,(c,f) in enumerate(derived):
            if ci in edges:
                _distance,entity,relation,prior_row=edges[ci]
                py2,px2,old_vy2,old_vx2,_pf,misses=map(int,prior_row);gap=max(1,misses+1)
                vy2=(int(c.y2)-py2)//gap;vx2=(int(c.x2)-px2)//gap
                prior_stability=int(self.motion_stability.get(int(entity),0))
                self.motion_stability[int(entity)]=(prior_stability+1 if (old_vy2 or old_vx2) and (vy2,vx2)==(old_vy2,old_vx2) else 0)
                organism.update_visual_entity(entity,(relation,),source);same=True
            else:
                entity=organism.mint_visual_entity((f,),source,sequence*8+ci+1);relation=0;same=False;vy2=vx2=0;self.motion_stability[int(entity)]=0
            new_active[int(entity)]=(int(c.y2),int(c.x2),int(vy2),int(vx2),int(f),0)
            result.append((int(entity),int(c.y2),int(c.x2),int(relation),same))
        # Unmatched prior tracks remain dormant for one frame if not already stale.
        used={row[1] for row in edges.values()}
        for entity,row in prior.items():
            if entity in used:continue
            y2,x2,vy2,vx2,feature,misses=map(int,row)
            stability=int(self.motion_stability.get(int(entity),0));budget=MAX_OBJECT_FILE_MISSES+min(2,stability//2)
            if misses<budget and len(new_active)<MAX_ACTIVE_OBJECT_FILES:
                new_active[int(entity)]=(y2,x2,vy2,vx2,feature,misses+1)
        self.active=new_active;self.motion_stability={e:self.motion_stability.get(e,0) for e in new_active};return tuple(result)
    def checkpoint(self):return {'schema':1}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('multi_object_file:checkpoint')
        return cls()
