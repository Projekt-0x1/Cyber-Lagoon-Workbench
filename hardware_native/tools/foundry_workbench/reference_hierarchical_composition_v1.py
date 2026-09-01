#!/usr/bin/env python3
"""Compact generic hierarchical closure ecology for Foundry reference experiments.

Persistent closures retain structural child references. Span pieces live on
language. Public ``surface`` and ``ancestry`` are derived. A small rebuildable
hot cache bounds repeated materialization cost without becoming checkpoint authority.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass,field
import hashlib,json

from reference_language_learning_v1 import LearnedSurfaceEcologyV1,PIECE_LITERAL,PIECE_PORT

MAX_CLOSURES=65536
MAX_DEPTH=64
MAX_ANCESTRY=4096
MAX_HOT_SURFACE_BYTES=65536
MAX_CONDENSED_RECIPES=4096
MAX_CONDENSED_OPS=4096
MIN_CONDENSE_DEPTH=4
MIN_CONDENSE_COLD_USES=3
MIN_CONDENSE_TOUCHES=9
CONDENSE_PROBATION_PASSES=2


class HierarchicalRefuse(RuntimeError):pass


def _identity(tag,obj):
    raw=json.dumps(obj,sort_keys=True,separators=(',',':'),default=list).encode()
    return int.from_bytes(hashlib.sha256(tag.encode()+b'\0'+raw).digest()[:8],'little') or 1


@dataclass(eq=True)
class ConstructionClosureV1:
    identity:int
    context:int
    template_identity:int
    child_identities:tuple[int,...]
    depth:int
    _leaf_surface:tuple[int,...]=()
    _owner:object=field(default=None,repr=False,compare=False)

    @property
    def surface(self):
        if self._owner is None:raise HierarchicalRefuse('hierarchy:unbound_closure')
        return self._owner._surface(self.identity)

    @property
    def ancestry(self):
        if self._owner is None:raise HierarchicalRefuse('hierarchy:unbound_closure')
        return self._owner._ancestry_ids(self.identity)


@dataclass(frozen=True)
class TransientConstructionPlanV1:
    identity:int
    context:int
    template_identity:int
    child_identities:tuple[int,...]
    depth:int
    piece_start:int=0


@dataclass(frozen=True)
class TransientSequencePlanV1:
    """Recurrent left-fold over one learned append-compatible binary Recipe."""
    identity:int
    context:int
    template_identity:int
    child_identities:tuple[int,...]
    depth:int


def render_template_pieces(pieces, child_surfaces, start=0):
    out=[]
    for piece in pieces[int(start):]:
        if piece.kind==PIECE_LITERAL:out.extend(piece.literal)
        elif piece.kind==PIECE_PORT and 0<=piece.port<len(child_surfaces):out.extend(child_surfaces[piece.port])
        else:raise HierarchicalRefuse('hierarchy:template_witness_invalid')
    return tuple(out)


def rematerialize_transient_plan(language, context, children, template=None, completed_child=None, render=True):
    """Build one current span plan from language + child identities. Writes no hierarchy state.

    Full surface render is optional: the motor externalizes incrementally from pieces.
    """
    children=tuple(children)
    if not children:raise HierarchicalRefuse('hierarchy:no_children')
    if any(getattr(c,'identity',None) is None for c in children):
        raise HierarchicalRefuse('hierarchy:unknown_child')
    if render and any(getattr(c,'surface',None) is None for c in children):
        raise HierarchicalRefuse('hierarchy:unknown_child')
    t=template if template is not None else language.span_template(int(context),len(children))
    if t is None or int(getattr(t,'arity',len(children)))!=len(children):raise HierarchicalRefuse('hierarchy:template_missing_or_ambiguous')
    tid=int(t.identity[:15],16);pieces=tuple(t.pieces);child_ids=tuple(c.identity for c in children)
    depth=1+max(int(getattr(c,'depth',0)) for c in children)
    piece_start=0
    if completed_child is not None:
        completed_child=int(completed_child)
        hits=[index for index,piece in enumerate(pieces)
              if piece.kind==PIECE_PORT and int(piece.port)==completed_child]
        if len(hits)!=1:raise HierarchicalRefuse('hierarchy:transient_completed_child')
        piece_start=hits[0]+1
    ident=_identity('hierarchy-transient-closure-v1',(
        (int(context),tid,child_ids,depth) if piece_start==0
        else (int(context),tid,child_ids,depth,piece_start)
    ))
    plan=TransientConstructionPlanV1(ident,int(context),tid,child_ids,depth,piece_start)
    if not render:return plan,()
    return plan,render_template_pieces(pieces,tuple(tuple(c.surface) for c in children),piece_start)


def rematerialize_transient_sequence_plan(language, context, children, template=None, render=True):
    children=tuple(children)
    if len(children)<2:raise HierarchicalRefuse('hierarchy:sequence_children')
    if any(getattr(c,'identity',None) is None for c in children):
        raise HierarchicalRefuse('hierarchy:unknown_child')
    if render and any(getattr(c,'surface',None) is None for c in children):
        raise HierarchicalRefuse('hierarchy:unknown_child')
    t=template if template is not None else language.span_template(int(context),2)
    if t is None or int(getattr(t,'arity',0))!=2:raise HierarchicalRefuse('hierarchy:sequence_template')
    tid=int(t.identity[:15],16);pieces=tuple(t.pieces)
    if (not pieces or pieces[0].kind!=PIECE_PORT or int(pieces[0].port)!=0
            or sum(1 for piece in pieces if piece.kind==PIECE_PORT and int(piece.port)==0)!=1
            or sum(1 for piece in pieces if piece.kind==PIECE_PORT and int(piece.port)==1)!=1):
        raise HierarchicalRefuse('hierarchy:sequence_not_append_compatible')
    child_ids=tuple(c.identity for c in children);depth=max(c.depth for c in children)+len(children)-1
    plan=TransientSequencePlanV1(
        _identity('hierarchy-transient-sequence-v1',(int(context),tid,child_ids,depth)),
        int(context),tid,child_ids,depth)
    if not render:return plan,()
    out=list(children[0].surface)
    for left,right in zip(children,children[1:]):
        _stage,suffix=rematerialize_transient_plan(language,int(context),(left,right),t,completed_child=0)
        out.extend(suffix)
    return plan,tuple(out)


@dataclass(frozen=True)
class CondensedOpV1:
    kind:int
    reference_identity:int
    piece_index:int=0


@dataclass
class CondensedClosureRecipeV1:
    identity:int
    closure_identity:int
    ops:tuple[CondensedOpV1,...]
    dependencies:tuple[tuple[int,int,int],...]
    ancestry_digest:str
    probation_passes:int=0
    active:int=0
    uses:int=0
    deoptimizations:int=0


class HierarchicalConstructionV1:
    def __init__(self,language:LearnedSurfaceEcologyV1):
        self.language=language
        self._closures={}
        self._hot_surfaces=OrderedDict()
        self._hot_surface_bytes=0
        self._condensed={}
        self._condense_evidence={}
        self.last_child_touches=0
        self.last_materialization_touches=0
        self.last_condensed_recipe=0
        self.max_depth_seen=0

    @property
    def closure_count(self):return len(self._closures)

    def _cache_surface(self,identity,surface):
        identity=int(identity);surface=tuple(surface)
        prior=self._hot_surfaces.pop(identity,None)
        if prior is not None:self._hot_surface_bytes-=len(prior)
        if len(surface)>MAX_HOT_SURFACE_BYTES:return surface
        self._hot_surfaces[identity]=surface;self._hot_surface_bytes+=len(surface)
        while self._hot_surface_bytes>MAX_HOT_SURFACE_BYTES and self._hot_surfaces:
            _key,row=self._hot_surfaces.popitem(last=False);self._hot_surface_bytes-=len(row)
        return surface

    def _admit(self,c,require_live_template=True):
        if c.depth>MAX_DEPTH:raise HierarchicalRefuse('hierarchy:depth')
        if c.depth<0:raise HierarchicalRefuse('hierarchy:depth')
        if c.depth==0:
            if c.child_identities or c.template_identity or not c._leaf_surface:raise HierarchicalRefuse('hierarchy:leaf_shape')
        else:
            if not c.child_identities or c._leaf_surface:raise HierarchicalRefuse('hierarchy:closure_shape')
            if require_live_template and self.span_pieces(c.template_identity,c.context,len(c.child_identities)) is None:
                raise HierarchicalRefuse('hierarchy:closure_shape')
        old=self._closures.get(c.identity)
        if old is not None:
            if old!=c:raise HierarchicalRefuse('hierarchy:identity_collision')
            return old
        if len(self._closures)>=MAX_CLOSURES:raise HierarchicalRefuse('hierarchy:closure_bound')
        c._owner=self;self._closures[c.identity]=c;self.max_depth_seen=max(self.max_depth_seen,c.depth)
        if c.depth==0:self._cache_surface(c.identity,c._leaf_surface)
        return c

    def leaf(self,context,atoms):
        atoms=tuple(map(int,atoms));surface=self.language.realize(int(context),atoms)
        if surface is None:raise HierarchicalRefuse('hierarchy:leaf_unrealized')
        ident=_identity('hierarchy-leaf-v1',(int(context),atoms,surface))
        return self._admit(ConstructionClosureV1(ident,int(context),0,(),0,tuple(surface)))

    def leaf_surface(self,context,resident_identity,surface):
        surface=tuple(int(x) for x in surface)
        if not surface or any(x<0 or x>255 for x in surface):raise HierarchicalRefuse('hierarchy:surface')
        ident=_identity('hierarchy-resident-leaf-v1',(int(context),int(resident_identity),surface))
        return self._admit(ConstructionClosureV1(ident,int(context),0,(),0,surface))

    def observe(self,context,children,raw_surface,source):
        children=tuple(children)
        if not children:return False
        if any(not isinstance(c,ConstructionClosureV1) or c.identity not in self._closures for c in children):raise HierarchicalRefuse('hierarchy:unknown_child')
        self.last_child_touches=len(children)
        return bool(self.language.observe_span(int(context),tuple(c.surface for c in children),raw_surface,int(source)))

    def _composition_witness(self,context,children,template=None,persist=True):
        children=tuple(children)
        if not children:raise HierarchicalRefuse('hierarchy:no_children')
        if persist:
            if any(not isinstance(c,ConstructionClosureV1) or c.identity not in self._closures for c in children):raise HierarchicalRefuse('hierarchy:unknown_child')
        elif any(getattr(c,'identity',None) is None or getattr(c,'surface',None) is None for c in children):
            raise HierarchicalRefuse('hierarchy:unknown_child')
        self.last_child_touches=len(children)
        t=template if template is not None else self.language.span_template(int(context),len(children))
        if t is None or int(getattr(t,'arity',len(children)))!=len(children):raise HierarchicalRefuse('hierarchy:template_missing_or_ambiguous')
        tid=int(t.identity[:15],16)
        if persist and self.span_pieces(tid,int(context),len(children)) is None:
            raise HierarchicalRefuse('hierarchy:template_missing_or_ambiguous')
        depth=1+max(int(getattr(c,'depth',0)) for c in children)
        return children,tid,t,depth

    def compose(self,context,children,template=None):
        children,tid,_t,depth=self._composition_witness(context,children,template)
        ident=_identity('hierarchy-closure-v1',(int(context),tid,tuple(c.identity for c in children)))
        c=self._admit(ConstructionClosureV1(ident,int(context),tid,tuple(ch.identity for ch in children),depth,()))
        # Materialize once now to prove the exact current witness can realize the
        # closure. The bytes are a bounded derived cache, never persistent state.
        self._surface(c.identity)
        return c

    def compose_transient_plan(self,context,children,template=None,completed_child=None):
        children=tuple(children);self.last_child_touches=len(children)
        return rematerialize_transient_plan(self.language,context,children,template,completed_child)

    def compose_transient(self,context,children,template=None):
        """Unfold one current compatible composition without admitting a closure.

        This is the Network-like current-computation path: learned template
        mathematics plus resident child closures produce an ephemeral identity
        and surface, but no frozen combined computation enters persistent state.
        """
        plan,surface=self.compose_transient_plan(context,children,template)
        return plan.identity,surface

    def compose_transient_sequence_plan(self,context,children,template=None):
        children=tuple(children);self.last_child_touches=len(children)
        return rematerialize_transient_sequence_plan(self.language,context,children,template)

    def retire_unreferenced(self,identity):
        """Retire settled derived closure state once no persistent closure depends on it."""
        identity=int(identity);node=self._closures.get(identity)
        if node is None:return False
        if any(identity in row.child_identities for row in self._closures.values()):return False
        cached=self._hot_surfaces.pop(identity,None)
        if cached is not None:self._hot_surface_bytes-=len(cached)
        self._condensed.pop(identity,None);self._condense_evidence.pop(identity,None)
        del self._closures[identity]
        return True

    def retire_unreferenced_leaf(self,identity):
        """Compatibility wrapper: retire only an unreferenced leaf closure."""
        identity=int(identity);node=self._closures.get(identity)
        return bool(node is not None and node.depth==0 and self.retire_unreferenced(identity))

    def retire_unreferenced_composites(self):
        retired=0;changed=True
        while changed:
            changed=False
            for identity,node in list(self._closures.items()):
                if node.depth>0 and self.retire_unreferenced(identity):
                    retired+=1;changed=True
        return retired

    def span_pieces(self,template_identity,context,arity):
        t=self.language.span_template(int(context),int(arity))
        if t is None or int(t.identity[:15],16)!=int(template_identity):return None
        return tuple(t.pieces)

    def historical_supported_span_pieces(self,template_identity,context,arity):
        """Frozen lived witness, usable only while some owning source remains current."""
        template_identity=int(template_identity);context=int(context);arity=int(arity)
        pieces=self.language.historical_span_pieces(template_identity)
        if pieces is None:return None
        pieces=tuple(pieces)
        if self.language.span_factor_identity(context,arity,pieces)!=template_identity:return None
        sources=self.language._span_sources.get((context,arity,pieces),())
        return pieces if self.language._active_sources(sources) else None

    @staticmethod
    def _render_pieces(pieces,child_surfaces,start=0):
        return render_template_pieces(pieces,child_surfaces,start)

    def _render_from_witness(self,template_identity,child_surfaces,context=0):
        pieces=self.span_pieces(int(template_identity),int(context),len(child_surfaces))
        if pieces is None:raise HierarchicalRefuse('hierarchy:template_witness_missing')
        return self._render_pieces(pieces,child_surfaces)

    @staticmethod
    def _ancestry_digest(values):
        raw=json.dumps(tuple(map(int,values)),separators=(',',':')).encode()
        return hashlib.sha256(b'hierarchy-ancestry-v1\0'+raw).hexdigest()

    def _compile_condensed_ops(self,identity):
        ops=[];dependencies=set();stack=set()
        def walk(node_id):
            node_id=int(node_id)
            if node_id in stack:raise HierarchicalRefuse('hierarchy:cycle')
            node=self._closures.get(node_id)
            if node is None:raise HierarchicalRefuse('hierarchy:missing_child')
            if node.depth==0:
                ops.append(CondensedOpV1(1,node.identity,0));return
            pieces=self.span_pieces(node.template_identity,node.context,len(node.child_identities))
            if pieces is None:raise HierarchicalRefuse('hierarchy:template_witness_missing')
            dependencies.add((node.context,len(node.child_identities),node.template_identity));stack.add(node_id)
            for index,piece in enumerate(pieces):
                if piece.kind==PIECE_LITERAL:
                    if piece.literal:ops.append(CondensedOpV1(2,node.template_identity,index))
                elif piece.kind==PIECE_PORT and 0<=piece.port<len(node.child_identities):
                    walk(node.child_identities[piece.port])
                else:raise HierarchicalRefuse('hierarchy:template_witness_invalid')
            stack.remove(node_id)
        walk(identity)
        if not ops or len(ops)>MAX_CONDENSED_OPS:raise HierarchicalRefuse('hierarchy:condensed_ops')
        return tuple(ops),tuple(sorted(dependencies))

    def _execute_condensed(self,recipe):
        out=[];touches=1
        for op in recipe.ops:
            touches+=1
            if op.kind==1:
                leaf=self._closures.get(op.reference_identity)
                if leaf is None or leaf.depth!=0 or not leaf._leaf_surface:raise HierarchicalRefuse('hierarchy:condensed_leaf')
                out.extend(leaf._leaf_surface)
            elif op.kind==2:
                pieces=None
                for context,arity,tid in recipe.dependencies:
                    if int(tid)==int(op.reference_identity):
                        pieces=self.span_pieces(tid,context,arity);break
                if pieces is None or not 0<=op.piece_index<len(pieces):raise HierarchicalRefuse('hierarchy:condensed_literal')
                piece=pieces[op.piece_index]
                if piece.kind!=PIECE_LITERAL:raise HierarchicalRefuse('hierarchy:condensed_literal')
                out.extend(piece.literal)
            else:raise HierarchicalRefuse('hierarchy:condensed_op')
        self.last_materialization_touches=touches;self.last_condensed_recipe=recipe.identity;recipe.uses+=1
        return tuple(out)

    def _condensed_dependencies_live(self,recipe):
        for context,arity,template_identity in recipe.dependencies:
            current=self.language.span_template(context,arity)
            if current is None or int(current.identity[:15],16)!=template_identity:return False
        return True

    def _nominate_condensed(self,closure):
        if closure.depth<MIN_CONDENSE_DEPTH or closure.identity in self._condensed:return None
        if len(self._condensed)>=MAX_CONDENSED_RECIPES:return None
        ops,deps=self._compile_condensed_ops(closure.identity)
        ancestry_digest=self._ancestry_digest(self._ancestry_ids(closure.identity))
        identity=_identity('hierarchy-condensed-recipe-v1',(closure.identity,[(o.kind,o.reference_identity,o.piece_index) for o in ops],deps,ancestry_digest))
        recipe=CondensedClosureRecipeV1(identity,closure.identity,ops,deps,ancestry_digest)
        self._condensed[closure.identity]=recipe
        return recipe

    def _record_condense_pressure(self,closure,touches):
        if closure.depth<MIN_CONDENSE_DEPTH or touches<MIN_CONDENSE_TOUCHES:return
        uses,total=self._condense_evidence.get(closure.identity,(0,0));uses+=1;total+=int(touches);self._condense_evidence[closure.identity]=(uses,total)
        if uses>=MIN_CONDENSE_COLD_USES and closure.identity not in self._condensed:self._nominate_condensed(closure)

    def _surface_original(self,identity,_stack=None):
        identity=int(identity);cached=self._hot_surfaces.get(identity)
        if cached is not None:
            self._hot_surfaces.move_to_end(identity);self.last_materialization_touches=1;return cached
        c=self._closures.get(identity)
        if c is None:raise HierarchicalRefuse('hierarchy:unknown_closure')
        if c.depth==0:self.last_materialization_touches=1;return self._cache_surface(identity,c._leaf_surface)
        pieces=self.span_pieces(c.template_identity,c.context,len(c.child_identities))
        if pieces is None:pieces=self.historical_supported_span_pieces(c.template_identity,c.context,len(c.child_identities))
        if pieces is None:
            cached=self._hot_surfaces.get(identity)
            if cached is not None:
                self._hot_surfaces.move_to_end(identity);self.last_materialization_touches=1;return cached
            raise HierarchicalRefuse('hierarchy:template_witness_missing')
        stack=set() if _stack is None else _stack
        if identity in stack:raise HierarchicalRefuse('hierarchy:cycle')
        stack.add(identity);touches=1;children=[]
        for child_id in c.child_identities:
            child=self._closures.get(int(child_id))
            if child is None:raise HierarchicalRefuse('hierarchy:missing_child')
            children.append(self._surface_original(child.identity,stack));touches+=self.last_materialization_touches
        stack.remove(identity);surface=self._render_pieces(pieces,tuple(children))
        touches+=len(pieces)
        self.last_materialization_touches=touches
        return self._cache_surface(identity,surface)

    def _surface(self,identity,_stack=None):
        if _stack is not None:return self._surface_original(identity,_stack)
        identity=int(identity);self.last_condensed_recipe=0
        cached=self._hot_surfaces.get(identity)
        if cached is not None:
            self._hot_surfaces.move_to_end(identity);self.last_materialization_touches=1;return cached
        closure=self._closures.get(identity)
        if closure is None:raise HierarchicalRefuse('hierarchy:unknown_closure')
        recipe=self._condensed.get(identity)
        if recipe is not None and recipe.active:
            if not self._condensed_dependencies_live(recipe):
                recipe.active=0;recipe.probation_passes=0;recipe.deoptimizations+=1
            else:
                return self._cache_surface(identity,self._execute_condensed(recipe))
        surface=self._surface_original(identity)
        original_touches=self.last_materialization_touches
        if recipe is not None and not recipe.active and self._condensed_dependencies_live(recipe):
            shadow=self._execute_condensed(recipe)
            ancestry_digest=self._ancestry_digest(self._ancestry_ids(identity))
            if shadow!=surface or ancestry_digest!=recipe.ancestry_digest:
                recipe.probation_passes=0;recipe.deoptimizations+=1
            else:
                recipe.probation_passes+=1
                if recipe.probation_passes>=CONDENSE_PROBATION_PASSES:recipe.active=1
            self.last_materialization_touches=original_touches+len(recipe.ops)+1;self.last_condensed_recipe=0
        self._record_condense_pressure(closure,original_touches)
        return surface

    def _ancestry_ids(self,identity):
        identity=int(identity);c=self._closures.get(identity)
        if c is None:raise HierarchicalRefuse('hierarchy:unknown_closure')
        rows=[];stack=set()
        def walk(node):
            if node.identity in stack:raise HierarchicalRefuse('hierarchy:cycle')
            stack.add(node.identity)
            for child_id in node.child_identities:
                child=self._closures.get(int(child_id))
                if child is None:raise HierarchicalRefuse('hierarchy:missing_child')
                rows.append(child.identity);walk(child)
                if len(rows)>MAX_ANCESTRY:raise HierarchicalRefuse('hierarchy:ancestry')
            stack.remove(node.identity)
        walk(c)
        return tuple(dict.fromkeys(rows))

    def closure(self,identity):return self._closures.get(int(identity))

    def drop_hot_cache(self):
        self._hot_surfaces.clear();self._hot_surface_bytes=0;self.last_materialization_touches=0

    def persistent_quantity(self):
        leaf_bytes=sum(len(c._leaf_surface) for c in self._closures.values())
        child_refs=sum(len(c.child_identities) for c in self._closures.values())
        condensed_ops=sum(len(r.ops) for r in self._condensed.values())
        active_condensed=sum(int(bool(r.active)) for r in self._condensed.values())
        return {'retained_leaf_surface_bytes':leaf_bytes,'retained_child_refs':child_refs,'retained_template_literal_bytes':0,'retained_template_pieces':0,'retained_condensed_ops':condensed_ops,'condensed_recipes':len(self._condensed),'active_condensed_recipes':active_condensed,'hot_surface_cache_bytes':self._hot_surface_bytes,'hot_surface_cache_entries':len(self._hot_surfaces)}

    def quantity_vector(self,current=None):
        q=self.persistent_quantity();q.update({'closure_count':len(self._closures),'max_depth':self.max_depth_seen,'current_depth':0 if current is None else current.depth,'current_child_width':0 if current is None else len(current.child_identities),'current_ancestry':0 if current is None else len(current.ancestry),'current_surface_bytes':0 if current is None else len(current.surface),'last_child_touches':self.last_child_touches,'last_materialization_touches':self.last_materialization_touches,'last_condensed_recipe':self.last_condensed_recipe});return q

    def checkpoint(self):
        closures=[{'identity':c.identity,'context':c.context,'template_identity':c.template_identity,'child_identities':list(c.child_identities),'depth':c.depth,'leaf_surface':list(c._leaf_surface)} for c in sorted(self._closures.values(),key=lambda x:x.identity)]
        condensed=[{'identity':r.identity,'closure_identity':r.closure_identity,'ops':[{'kind':o.kind,'reference_identity':o.reference_identity,'piece_index':o.piece_index} for o in r.ops],'dependencies':[list(x) for x in r.dependencies],'ancestry_digest':r.ancestry_digest,'probation_passes':r.probation_passes,'active':r.active,'uses':r.uses,'deoptimizations':r.deoptimizations} for r in sorted(self._condensed.values(),key=lambda x:x.identity)]
        evidence=[{'closure_identity':identity,'cold_uses':row[0],'touches':row[1]} for identity,row in sorted(self._condense_evidence.items())]
        return {'schema':4,'closures':closures,'condensed':condensed,'condense_evidence':evidence,'max_depth_seen':self.max_depth_seen}

    @classmethod
    def restore(cls,language,data):
        schema=int(data.get('schema',0))
        if schema not in (2,3,4):raise HierarchicalRefuse('hierarchy:checkpoint_schema')
        out=cls(language)
        rows=data.get('closures',())
        if len(rows)>MAX_CLOSURES:raise HierarchicalRefuse('hierarchy:closure_bound')
        for row in rows:
            c=ConstructionClosureV1(int(row['identity']),int(row['context']),int(row['template_identity']),tuple(map(int,row['child_identities'])),int(row['depth']),tuple(map(int,row.get('leaf_surface',()))))
            out._admit(c,require_live_template=False)
        visiting=set();memo={}
        def depth(identity):
            identity=int(identity)
            if identity in memo:return memo[identity]
            if identity in visiting:raise HierarchicalRefuse('hierarchy:cycle')
            c=out._closures.get(identity)
            if c is None:raise HierarchicalRefuse('hierarchy:missing_child')
            visiting.add(identity)
            got=0 if not c.child_identities else 1+max(depth(x) for x in c.child_identities)
            visiting.remove(identity)
            if got!=c.depth:raise HierarchicalRefuse('hierarchy:checkpoint_depth')
            memo[identity]=got;return got
        for identity in out._closures:depth(identity)
        if out.max_depth_seen!=int(data.get('max_depth_seen',out.max_depth_seen)):raise HierarchicalRefuse('hierarchy:checkpoint_depth')
        if schema>=3:
            for row in data.get('condensed',()):
                closure_identity=int(row['closure_identity'])
                if closure_identity not in out._closures or closure_identity in out._condensed:raise HierarchicalRefuse('hierarchy:checkpoint_condensed')
                ops=tuple(CondensedOpV1(int(o['kind']),int(o['reference_identity']),int(o.get('piece_index',0))) for o in row.get('ops',()))
                deps=tuple(tuple(map(int,x)) for x in row.get('dependencies',()))
                if not ops or len(ops)>MAX_CONDENSED_OPS:raise HierarchicalRefuse('hierarchy:checkpoint_condensed')
                recipe=CondensedClosureRecipeV1(int(row['identity']),closure_identity,ops,deps,str(row['ancestry_digest']),int(row.get('probation_passes',0)),int(row.get('active',0)),int(row.get('uses',0)),int(row.get('deoptimizations',0)))
                expected_ops,expected_deps=out._compile_condensed_ops(closure_identity)
                if recipe.ops!=expected_ops or recipe.dependencies!=expected_deps or recipe.ancestry_digest!=out._ancestry_digest(out._ancestry_ids(closure_identity)):raise HierarchicalRefuse('hierarchy:checkpoint_condensed_mismatch')
                expected_id=_identity('hierarchy-condensed-recipe-v1',(closure_identity,[(o.kind,o.reference_identity,o.piece_index) for o in recipe.ops],recipe.dependencies,recipe.ancestry_digest))
                if recipe.identity!=expected_id:raise HierarchicalRefuse('hierarchy:checkpoint_condensed_identity')
                out._condensed[closure_identity]=recipe
            if len(out._condensed)>MAX_CONDENSED_RECIPES:raise HierarchicalRefuse('hierarchy:checkpoint_condensed')
            for row in data.get('condense_evidence',()):
                identity=int(row['closure_identity']);uses=int(row['cold_uses']);touches=int(row['touches'])
                if identity not in out._closures or uses<0 or touches<0:raise HierarchicalRefuse('hierarchy:checkpoint_condense_evidence')
                out._condense_evidence[identity]=(uses,touches)
        return out

    def digest(self):return hashlib.sha256(b'hierarchy-v3\0'+json.dumps(self.checkpoint(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
