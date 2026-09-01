#!/usr/bin/env python3
"""Content-addressed packed persistence for language selection-Network revisions."""
from __future__ import annotations
from array import array
import hashlib,struct

_CONFIG_HEADER=struct.Struct('<QH')
_MEMBER=struct.Struct('<IIQ')
_ROW=struct.Struct('<IQii')
_BODY_HDR=struct.Struct('<QH')
_MAGIC=b'0X1SEL3\0'
_HEADER=struct.Struct('<8sIII32s')
_ROOT_BYTES=32

class PackedSelectionRevisionV1:
    """Persistent selection-Network revision authority.

    Lookup is content-addressed: Root256 is reconstructed from the live candidate.
    Bodies persist the same member triples already earned on the Network so
    construction/lesion can iterate without a second Python list. This store is
    not a grammar.
    """
    def __init__(self):
        self.roots=bytearray()
        self.rows=bytearray()
        self.bodies=bytearray()
        self._body_off=array('I')
        self._root_buckets={}
        self._row_heads=array('i')
        self._row_next=array('i')
        self.last_config_candidates=0
        self.last_revision_rows=0

    @staticmethod
    def _normalize(context,configuration):
        context=int(context);cfg=tuple(tuple(map(int,row)) for row in configuration)
        if context<=0 or not cfg or len(cfg)>65535:raise ValueError('packed-selection:configuration')
        for row in cfg:
            if len(row)!=3 or row[0]<0 or row[0]>0xffffffff or row[1]<0 or row[1]>0xffffffff or row[2]<0 or row[2]>0xffffffffffffffff:
                raise ValueError('packed-selection:member')
        return context,cfg

    @classmethod
    def configuration_root(cls,context,configuration):
        context,cfg=cls._normalize(context,configuration)
        h=hashlib.sha256();h.update(b'0x1-selection-configuration-v2\0');h.update(_CONFIG_HEADER.pack(context,len(cfg)))
        for kind,slot,candidate in cfg:h.update(_MEMBER.pack(kind,slot,candidate))
        return h.digest()

    def _root(self,index):
        off=index*_ROOT_BYTES;return bytes(self.roots[off:off+_ROOT_BYTES])

    @staticmethod
    def _bucket(root):return int.from_bytes(root[:2],'little')

    def _find_config(self,context,configuration,create=False):
        context,configuration=self._normalize(context,configuration)
        root=self.configuration_root(context,configuration);bucket=self._bucket(root);candidates=self._root_buckets.get(bucket,())
        self.last_config_candidates=len(candidates)
        for index in candidates:
            if self._root(index)==root:return int(index)
        if not create:return None
        index=self.config_count
        if index>0xffffffff:raise ValueError('packed-selection:config_index')
        self.roots.extend(root);self._row_heads.append(-1)
        self._body_off.append(len(self.bodies));self.bodies.extend(_BODY_HDR.pack(context,len(configuration)))
        for kind,slot,candidate in configuration:self.bodies.extend(_MEMBER.pack(kind,slot,candidate))
        rows=self._root_buckets.get(bucket)
        if rows is None:rows=array('I');self._root_buckets[bucket]=rows
        rows.append(index)
        return index

    def configuration_at(self,index):
        off=int(self._body_off[int(index)]);context,n=_BODY_HDR.unpack_from(self.bodies,off);members=[];p=off+_BODY_HDR.size
        for _ in range(n):
            members.append(_MEMBER.unpack_from(self.bodies,p));p+=_MEMBER.size
        return int(context),tuple(members)

    def iter_revisions(self):
        for ri in range(self.row_count):
            ci,source,support,counter=_ROW.unpack_from(self.rows,ri*_ROW.size)
            context,cfg=self.configuration_at(ci)
            yield context,cfg,int(source),int(support),int(counter)

    def drop(self,context,configuration,source):
        context,cfg=self._normalize(context,configuration);source=int(source);keep=[]
        for ctx,row_cfg,src,support,counter in self.iter_revisions():
            if not (ctx==context and row_cfg==cfg and src==source):keep.append((ctx,row_cfg,src,support,counter))
        self.__init__()
        for ctx,row_cfg,src,support,counter in keep:self._write(ctx,row_cfg,src,support,counter)
        return 1

    def _write(self,context,configuration,source,support,counter):
        ci=self._find_config(context,configuration,True);ri=self.row_count
        self.rows.extend(_ROW.pack(ci,int(source),int(support),int(counter)));self._row_next.append(int(self._row_heads[ci]));self._row_heads[ci]=ri;return 1

    def record(self,context,configuration,source,direction):
        source=int(source);direction=int(direction)
        if source<=0 or direction==0:return 0
        ci=self._find_config(context,configuration,True);ri=int(self._row_heads[ci])
        while ri>=0:
            off=ri*_ROW.size;stored_ci,stored_source,support,counter=_ROW.unpack_from(self.rows,off)
            if stored_source==source:
                if direction>0:support+=1
                else:counter+=1
                if support>0x7fffffff or counter>0x7fffffff:raise ValueError('packed-selection:counter')
                _ROW.pack_into(self.rows,off,stored_ci,stored_source,support,counter);return 1
            ri=int(self._row_next[ri])
        ri=self.row_count;self.rows.extend(_ROW.pack(ci,source,1 if direction>0 else 0,1 if direction<0 else 0));self._row_next.append(int(self._row_heads[ci]));self._row_heads[ci]=ri;return 1

    def contains(self,context,configuration,source):
        ci=self._find_config(context,configuration,False)
        if ci is None:return False
        source=int(source);ri=int(self._row_heads[ci])
        while ri>=0:
            stored_ci,stored_source,support,counter=_ROW.unpack_from(self.rows,ri*_ROW.size)
            if stored_source==source:return True
            ri=int(self._row_next[ri])
        return False

    def evidence(self,context,configuration,withdrawn=()):
        ci=self._find_config(context,configuration,False)
        if ci is None:self.last_revision_rows=0;return 0,0
        withdrawn=set(map(int,withdrawn));value=evidence=0;ri=int(self._row_heads[ci]);touches=0
        while ri>=0:
            stored_ci,source,support,counter=_ROW.unpack_from(self.rows,ri*_ROW.size);touches+=1
            if stored_ci!=ci:raise ValueError('packed-selection:index')
            if source not in withdrawn:value+=support-counter;evidence+=support+counter
            ri=int(self._row_next[ri])
        self.last_revision_rows=touches
        return value,evidence

    def member_evidence(self,context,member,withdrawn=()):
        """Aggregate earned credit for one constituent of prior Networks."""
        context=int(context);member=tuple(map(int,member))
        if context<=0 or len(member)!=3:raise ValueError('packed-selection:member')
        withdrawn=set(map(int,withdrawn));value=evidence=touches=0
        for ci in range(self.config_count):
            stored_context,configuration=self.configuration_at(ci)
            if stored_context!=context or member not in configuration:continue
            ri=int(self._row_heads[ci])
            while ri>=0:
                stored_ci,source,support,counter=_ROW.unpack_from(
                    self.rows,ri*_ROW.size);touches+=1
                if stored_ci!=ci:raise ValueError('packed-selection:index')
                if source not in withdrawn:
                    # Configuration failure does not identify which reusable
                    # constituent caused it.  Only positive configuration
                    # evidence may generalize to a member; counter-evidence
                    # remains exact to the complete Network root.
                    value+=support;evidence+=support
                ri=int(self._row_next[ri])
        self.last_revision_rows=touches
        return value,evidence

    @property
    def config_count(self):return len(self.roots)//_ROOT_BYTES
    @property
    def row_count(self):return len(self.rows)//_ROW.size
    @property
    def persistent_bytes(self):return _HEADER.size+len(self.roots)+len(self.rows)+len(self.bodies)
    @property
    def runtime_index_payload_bytes(self):
        # Silicon-lowering payload: fixed 16-bit bucket directory plus packed
        # config indices and row adjacency. Python container overhead is not authority.
        return 65536*8 + 4*sum(len(v) for v in self._root_buckets.values()) + 4*len(self._row_heads) + 4*len(self._row_next)
    @property
    def max_bucket_occupancy(self):return max((len(v) for v in self._root_buckets.values()),default=0)

    def checkpoint(self):
        body=bytes(self.roots)+bytes(self.rows)+bytes(self.bodies);digest=hashlib.sha256(body).digest()
        return _HEADER.pack(_MAGIC,self.config_count,self.row_count,len(self.bodies),digest)+body

    @classmethod
    def restore(cls,blob):
        blob=bytes(blob)
        if len(blob)<_HEADER.size:raise ValueError('packed-selection:checkpoint')
        magic,nc,nr,nb,digest=_HEADER.unpack_from(blob,0);root_n=nc*_ROOT_BYTES;row_n=nr*_ROW.size;body=blob[_HEADER.size:]
        if magic!=_MAGIC or len(body)!=root_n+row_n+nb or hashlib.sha256(body).digest()!=digest:raise ValueError('packed-selection:checkpoint')
        out=cls();out.roots=bytearray(body[:root_n]);out.rows=bytearray(body[root_n:root_n+row_n]);out.bodies=bytearray(body[root_n+row_n:]);seen=set();cursor=0
        for i in range(nc):
            root=out._root(i)
            if root==b'\0'*_ROOT_BYTES or root in seen or cursor+ _BODY_HDR.size>len(out.bodies):raise ValueError('packed-selection:checkpoint_root')
            context,n=_BODY_HDR.unpack_from(out.bodies,cursor);need=_BODY_HDR.size+n*_MEMBER.size
            if context<=0 or cursor+need>len(out.bodies):raise ValueError('packed-selection:checkpoint_body')
            stored=out.configuration_root(context,tuple(_MEMBER.unpack_from(out.bodies,cursor+_BODY_HDR.size+k*_MEMBER.size) for k in range(n)))
            if stored!=root:raise ValueError('packed-selection:checkpoint_body')
            out._body_off.append(cursor);cursor+=need
            seen.add(root);bucket=out._bucket(root);rows=out._root_buckets.get(bucket)
            if rows is None:rows=array('I');out._root_buckets[bucket]=rows
            rows.append(i);out._row_heads.append(-1)
        if cursor!=len(out.bodies):raise ValueError('packed-selection:checkpoint_body')
        for ri in range(nr):
            ci,source,support,counter=_ROW.unpack_from(out.rows,ri*_ROW.size)
            if ci>=nc or source==0 or support<0 or counter<0:raise ValueError('packed-selection:checkpoint_row')
            out._row_next.append(int(out._row_heads[ci]));out._row_heads[ci]=ri
        if out.checkpoint()!=blob:raise ValueError('packed-selection:checkpoint_roundtrip')
        return out
