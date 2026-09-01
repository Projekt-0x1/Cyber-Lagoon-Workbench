#!/usr/bin/env python3
"""Content-neutral connected visual components from a zero-background raw frame."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RawVisualObjectCandidateV1:
    y2:int;x2:int;crop:tuple[tuple[int,...],...]

class RawVisualObjectCandidatesV1:
    @staticmethod
    def extract(frame):
        rows=tuple(tuple(int(v) for v in row) for row in frame)
        if not rows or not rows[0] or any(len(row)!=len(rows[0]) for row in rows):raise ValueError('visual_candidates:shape')
        h=len(rows);w=len(rows[0]);live={(y,x) for y in range(h) for x in range(w) if rows[y][x]!=0};out=[]
        while live:
            seed=min(live);stack=[seed];component=set();live.remove(seed)
            while stack:
                y,x=stack.pop();component.add((y,x))
                for p in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if p in live:live.remove(p);stack.append(p)
            ys=[p[0] for p in component];xs=[p[1] for p in component];y0,y1=min(ys),max(ys);x0,x1=min(xs),max(xs)
            crop=tuple(tuple(rows[y][x0:x1+1]) for y in range(y0,y1+1))
            out.append(RawVisualObjectCandidateV1(y0+y1,x0+x1,crop))
        return tuple(sorted(out,key=lambda row:(row.y2,row.x2)))
