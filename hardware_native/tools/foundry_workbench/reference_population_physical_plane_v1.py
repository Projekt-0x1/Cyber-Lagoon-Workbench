#!/usr/bin/env python3
"""Backend-neutral scalar causal plane for PopulationBank lowering experiments.

Workbench-only representation adapter. Logical scalar values are authority; backing
format and migration telemetry are non-causal. No language, semantic labels, source
truth, controllability flag, reward head, or host answer path exists here.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json

PAGE = 256


@dataclass(frozen=True)
class PlaneSpecV1:
    domain: int
    default: int
    minimum: int
    maximum: int
    payload_bytes: int

    def validate(self):
        if self.domain <= 0:
            raise ValueError("plane:domain")
        if self.minimum > self.default or self.default > self.maximum:
            raise ValueError("plane:default")
        if self.payload_bytes not in (1, 2, 4, 8):
            raise ValueError("plane:payload")
        return self


class _MapBacking:
    name = "sparse_map"
    def __init__(self, spec): self.spec=spec; self.rows={}
    def get(self, i): return int(self.rows.get(i,self.spec.default))
    def set(self, i, v):
        if v==self.spec.default:self.rows.pop(i,None)
        else:self.rows[i]=v
    def snapshot(self):return dict(self.rows)


class _SortedPageBacking:
    name = "sorted_sparse_pages"
    def __init__(self,spec):self.spec=spec;self.pages={}
    def get(self,i):
        page,off=divmod(i,PAGE); offsets,values=self.pages.get(page,([],[])); p=bisect_left(offsets,off)
        return int(values[p]) if p<len(offsets) and offsets[p]==off else self.spec.default
    def set(self,i,v):
        page,off=divmod(i,PAGE); offsets,values=self.pages.get(page,([],[])); p=bisect_left(offsets,off); present=p<len(offsets) and offsets[p]==off
        if v==self.spec.default:
            if present:
                offsets.pop(p);values.pop(p)
                if offsets:self.pages[page]=(offsets,values)
                else:self.pages.pop(page,None)
            return
        if present:values[p]=v
        else:offsets.insert(p,off);values.insert(p,v)
        self.pages[page]=(offsets,values)
    def snapshot(self):return {page*PAGE+off:int(v) for page,(offs,vals) in self.pages.items() for off,v in zip(offs,vals)}


class _BitmapPageBacking:
    name = "bitmap_pages"
    def __init__(self,spec):self.spec=spec;self.pages={}
    def get(self,i):
        page,off=divmod(i,PAGE); bits,values=self.pages.get(page,(0,[])); mask=1<<off
        if not bits&mask:return self.spec.default
        return int(values[(bits&(mask-1)).bit_count()])
    def set(self,i,v):
        page,off=divmod(i,PAGE); bits,values=self.pages.get(page,(0,[])); mask=1<<off; p=(bits&(mask-1)).bit_count(); present=bool(bits&mask)
        if v==self.spec.default:
            if present:
                values.pop(p);bits&=~mask
                if bits:self.pages[page]=(bits,values)
                else:self.pages.pop(page,None)
            return
        if present:values[p]=v
        else:values.insert(p,v);bits|=mask
        self.pages[page]=(bits,values)
    def snapshot(self):
        out={}
        for page,(bits,values) in self.pages.items():
            b=bits;p=0
            while b:
                lsb=b&-b;off=lsb.bit_length()-1;out[page*PAGE+off]=int(values[p]);p+=1;b^=lsb
        return out


class _DensePageBacking:
    name = "dense_pages"
    def __init__(self,spec):self.spec=spec;self.pages={}
    def get(self,i):
        page,off=divmod(i,PAGE); vals=self.pages.get(page)
        return self.spec.default if vals is None else int(vals[off])
    def set(self,i,v):
        page,off=divmod(i,PAGE); vals=self.pages.get(page)
        if vals is None:
            if v==self.spec.default:return
            vals=[self.spec.default]*PAGE;self.pages[page]=vals
        vals[off]=v
        if v==self.spec.default and all(x==self.spec.default for x in vals):self.pages.pop(page,None)
    def snapshot(self):return {page*PAGE+off:int(v) for page,vals in self.pages.items() for off,v in enumerate(vals) if v!=self.spec.default}


BACKINGS={c.name:c for c in (_MapBacking,_SortedPageBacking,_BitmapPageBacking,_DensePageBacking)}


class PopulationPhysicalPlaneV1:
    def __init__(self,spec:PlaneSpecV1,backing="sparse_map"):
        self.spec=spec.validate()
        if backing not in BACKINGS:raise ValueError("plane:backing")
        self._backing=BACKINGS[backing](self.spec)
        self.migration_count=0

    @property
    def backing(self):return self._backing.name

    def _index(self,i):
        i=int(i)
        if i<0 or i>=self.spec.domain:raise IndexError(i)
        return i

    def _value(self,v):
        v=int(v)
        if v<self.spec.minimum or v>self.spec.maximum:raise ValueError("plane:value")
        return v

    def get(self,i):return self._backing.get(self._index(i))
    def set(self,i,v):self._backing.set(self._index(i),self._value(v))
    def snapshot(self):return dict(sorted(self._backing.snapshot().items()))

    def digest(self):
        h=hashlib.sha256(b"population-physical-plane-v1\0")
        h.update(json.dumps({"spec":self.spec.__dict__,"rows":self.snapshot()},sort_keys=True,separators=(",",":")).encode())
        return h.hexdigest()

    def migrate(self,target:str,fail_after:int|None=None):
        if target not in BACKINGS:raise ValueError("plane:backing")
        if target==self.backing:return {"changed":False,"rows":len(self.snapshot())}
        source_digest=self.digest(); source_backing=self._backing; rows=self.snapshot(); candidate=BACKINGS[target](self.spec)
        try:
            if fail_after is not None and int(fail_after)==0:
                raise RuntimeError("plane:injected_migration_fault")
            for n,(i,v) in enumerate(rows.items(),1):
                candidate.set(i,v)
                if fail_after is not None and n==int(fail_after):
                    raise RuntimeError("plane:injected_migration_fault")
            old=self._backing; self._backing=candidate
            if self.digest()!=source_digest:
                self._backing=old
                raise ValueError("plane:migration_digest")
        except Exception:
            self._backing=source_backing
            if self.digest()!=source_digest:
                raise AssertionError("plane:migration_not_atomic")
            raise
        self.migration_count+=1
        return {"changed":True,"rows":len(rows),"backing":target}
