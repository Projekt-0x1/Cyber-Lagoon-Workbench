#!/usr/bin/env python3
"""Content-neutral directional ray extraction from a short raw hand trajectory."""
from __future__ import annotations

class RawPointingMotionV1:
    @staticmethod
    def ray(samples):
        rows=tuple((int(y),int(x)) for y,x in samples)
        if len(rows)<3:return None
        oy,ox=rows[0];fy,fx=rows[-1];vy,vx=fy-oy,fx-ox;vnorm=vy*vy+vx*vx
        if vnorm<=0:return None
        prior=0
        for y,x in rows[1:]:
            dy,dx=y-oy,x-ox;norm=dy*dy+dx*dx
            if norm<=prior:return None
            prior=norm;cross=dy*vx-dx*vy
            if cross*cross*16>vnorm*norm:return None
        return oy,ox,vy,vx

    @staticmethod
    def scores(tracker,samples):
        ray=RawPointingMotionV1.ray(samples)
        if ray is None:return ()
        oy,ox,vy,vx=ray;vnorm=vy*vy+vx*vx;ranked=[]
        for entity,row in getattr(tracker,'active',{}).items():
            if not row or int(row[-1])!=0:continue
            ty,tx=int(row[0])-oy,int(row[1])-ox;dot=ty*vy+tx*vx
            if dot<=0:continue
            tnorm=ty*ty+tx*tx
            if tnorm<=0:continue
            cross=ty*vx-tx*vy
            if cross*cross*16>vnorm*tnorm:continue
            score=(cross*cross*1_000_000)//tnorm
            ranked.append((score,int(entity)))
        return tuple(sorted(ranked))

    @staticmethod
    def target(tracker,samples):
        ranked=RawPointingMotionV1.scores(tracker,samples)
        if not ranked:return 0
        best=ranked[0][0];wins=[entity for score,entity in ranked if score==best]
        return wins[0] if len(wins)==1 else 0
