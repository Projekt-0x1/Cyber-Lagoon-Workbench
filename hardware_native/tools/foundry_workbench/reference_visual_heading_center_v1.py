#!/usr/bin/env python3
"""Stateless center-of-flow estimate from preserved local optic-flow geometry."""
from __future__ import annotations
from reference_local_optic_flow_v1 import LocalOpticFlowV1
from reference_population_v1 import mix64

HEADING_TAG=0xF10C0F
SCALE=256

class VisualHeadingCenterV1:
    @staticmethod
    def vector_rows(previous_frame,current_frame):
        previous=LocalOpticFlowV1._components(previous_frame);current=LocalOpticFlowV1._components(current_frame)
        rows=[]
        for prior,now in LocalOpticFlowV1._match(previous,current):
            dx=now[0]-prior[0];dy=now[1]-prior[1]
            if dx or dy:rows.append((int(prior[0]),int(prior[1]),int(dx),int(dy)))
        return tuple(rows)

    @classmethod
    def estimate_rows(cls,rows,width:int,height:int):
        rows=tuple(tuple(int(v) for v in row) for row in rows)
        width=int(width);height=int(height)
        if len(rows)<3 or width<=0 or height<=0:return None
        a00=a01=a11=b0=b1=0
        for px,py,dx,dy in rows:
            nx=-dy;ny=dx;rhs=nx*px+ny*py
            a00+=nx*nx;a01+=nx*ny;a11+=ny*ny;b0+=nx*rhs;b1+=ny*rhs
        det=a00*a11-a01*a01
        if det==0:return None
        cx_num=b0*a11-b1*a01;cy_num=a00*b1-a01*b0
        cx=cx_num/det;cy=cy_num/det
        signs=[]
        for px,py,dx,dy in rows:
            rx=px-cx;ry=py-cy
            cross=abs(rx*dy-ry*dx);scale=max(1,abs(rx*dx)+abs(ry*dy))
            if cross*8>scale:return None
            dot=rx*dx+ry*dy
            if abs(dot)<1:return None
            signs.append(1 if dot>0 else -1)
        if len(set(signs))!=1:return None
        if not (0<=cx<width*SCALE and 0<=cy<height*SCALE):return None
        return (int(round(cx)),int(round(cy)),int(signs[0]))

    @classmethod
    def estimate(cls,previous_frame,current_frame):
        return cls.estimate_rows(cls.vector_rows(previous_frame,current_frame),len(previous_frame[0]),len(previous_frame))

    @staticmethod
    def token(center,width:int,height:int):
        if center is None:return 0
        cx,cy,sign=center;width=int(width);height=int(height)
        if width<=0 or height<=0:return 0
        # Coarse population-like retinal sector; no heading class name.
        bx=min(7,max(0,int(cx)*8//(width*SCALE)));by=min(7,max(0,int(cy)*8//(height*SCALE)))
        value=mix64(HEADING_TAG^mix64(bx+1)^mix64((by+1)*17)^mix64((int(sign)+2)*31))
        return int(value&((1<<63)-1) or 1)
