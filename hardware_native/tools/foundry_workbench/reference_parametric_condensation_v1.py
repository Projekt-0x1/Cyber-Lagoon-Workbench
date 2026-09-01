#!/usr/bin/env python3
"""Foundry hypothesis: recurrent closure shape condenses into parameterized N+1 math."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from types import SimpleNamespace
import hashlib,json
from reference_hierarchical_composition_v1 import ConstructionClosureV1,HierarchicalConstructionV1,HierarchicalRefuse,render_template_pieces
from reference_incremental_expression_v1 import language_span_pieces
from reference_language_learning_v1 import PIECE_LITERAL,PIECE_PORT

MIN_DISTINCT_WITNESSES=3
SCHEMA_USE_LOAD=2
PROBATION_PASSES=2
MAX_RECIPES=4096
MAX_OPS=4096
OP_SLOT=1
OP_LITERAL=2
OP_CALL=3

def _digest(tag,value):
    return hashlib.sha256(tag.encode()+b'\0'+json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _id(tag,value):return int(_digest(tag,value)[:16],16) or 1

@dataclass(frozen=True)
class ParametricOpV1:
    kind:int
    path:tuple[int,...]=()
    reference_identity:int=0
    piece_index:int=0

@dataclass
class ParametricCondensedRecipeV1:
    identity:int
    shape_digest:str
    rank:int
    ops:tuple[ParametricOpV1,...]
    dependencies:tuple[tuple[int,int,int],...]
    witness_closures:tuple[int,...]
    witness_digest:str
    probation_passes:int=0
    active:int=0
    uses:int=0
    deoptimizations:int=0

class ParametricCondensationV1:
    def __init__(self,hierarchy:HierarchicalConstructionV1):
        self.h=hierarchy
        self.recipes:dict[int,ParametricCondensedRecipeV1]={}
        self.by_shape:dict[str,int]={}
        self.pressure:dict[str,list[int]]={}
        self.last_touches=0
        self.last_recipe=0
        self.last_rank=0
        self.nodes={}

    def _node(self,closure_or_id):
        if isinstance(closure_or_id,ConstructionClosureV1):return closure_or_id
        if getattr(closure_or_id,'identity',None) is not None and getattr(closure_or_id,'child_identities',None) is not None:
            return closure_or_id
        ident=int(closure_or_id)
        if ident in self.nodes:return self.nodes[ident]
        row=self.h.closure(ident)
        if row is None:raise HierarchicalRefuse('parametric:unknown_closure')
        return row

    def _remember(self,node):
        if self.h.closure(int(node.identity)) is None:
            self.nodes[int(node.identity)]=node
        return node

    def _pieces(self,node):
        tid=int(getattr(node,'template_identity',0) or 0)
        kids=getattr(node,'child_identities',())
        if tid and kids:
            return language_span_pieces(self.h.language,tid,node.context,len(kids))
        return None

    def _render_original(self,node):
        node=self._node(node)
        if int(getattr(node,'depth',0))==0:
            surface=getattr(node,'_leaf_surface',()) or getattr(node,'surface',())
            if not surface:raise HierarchicalRefuse('parametric:leaf')
            return tuple(surface),1
        pieces=self._pieces(node)
        if pieces is None:
            lived=tuple(getattr(node,'surface',()) or ())
            if lived:return lived,1
            raise HierarchicalRefuse('parametric:template_witness')
        children=[];touches=1
        load=sum(len(v) for v in self.pressure.values())
        for child_id in node.child_identities:
            child=self._node(child_id)
            shape=self.shape_digest(child)
            recipe=self.recipes.get(self.by_shape.get(shape,0))
            if load<SCHEMA_USE_LOAD and recipe is not None and recipe.active and self._dependencies_live(recipe):
                try:
                    surface,t=self._execute(recipe,child,True);children.append(surface);touches+=t;continue
                except HierarchicalRefuse:pass
            surface,t=self._render_original(child);children.append(surface);touches+=t
        return render_template_pieces(pieces,children),touches+len(pieces)

    def _witness_digest(self,witnesses):
        rows=[]
        for item in witnesses:
            ident=int(item)
            if self.h.closure(ident) is not None:
                rows.append((ident,self.h._ancestry_digest(self.h._ancestry_ids(ident))))
                continue
            node=self._node(ident)
            rows.append((ident,_digest('parametric-transient-ancestry-v1',[ident,int(node.context),int(getattr(node,'template_identity',0) or 0),list(map(int,getattr(node,'child_identities',())))])))
        return _digest('parametric-witness-v1',rows)

    def _shape(self,closure_or_id):
        node=self._node(closure_or_id)
        if node.depth==0:return ('leaf',int(node.context))
        return ('node',int(node.context),int(node.template_identity),tuple(self._shape(x) for x in node.child_identities))
    def shape_digest(self,closure):return _digest('parametric-shape-v1',self._shape(closure))

    def _subtree(self,root,path):
        node=self._node(root)
        for index in path:
            if node.depth==0 or not 0<=int(index)<len(node.child_identities):raise HierarchicalRefuse('parametric:path')
            node=self._node(node.child_identities[int(index)])
        return node

    def _dependencies_live(self,recipe):
        for context,arity,tid in recipe.dependencies:
            current=self.h.language.span_template(context,arity)
            if current is None or int(current.identity[:15],16)!=tid:return False
        return True

    def _compile(self,source,witnesses):
        source=self._node(source);ops=[];deps=set();called=[];root_shape=self.shape_digest(source)
        def walk(node,path,is_root=False):
            shape=self.shape_digest(node)
            lower_id=self.by_shape.get(shape,0)
            lower=self.recipes.get(lower_id)
            if not is_root and lower is not None and lower.active and lower.rank>=1:
                ops.append(ParametricOpV1(OP_CALL,tuple(path),lower.identity,0));called.append(lower.rank);return
            if node.depth==0:
                ops.append(ParametricOpV1(OP_SLOT,tuple(path),0,0));return
            pieces=self._pieces(node)
            if pieces is None:raise HierarchicalRefuse('parametric:template_witness')
            deps.add((int(node.context),len(node.child_identities),int(node.template_identity)))
            for i,piece in enumerate(pieces):
                if piece.kind==PIECE_LITERAL:
                    if piece.literal:ops.append(ParametricOpV1(OP_LITERAL,tuple(path),int(node.template_identity),i))
                elif piece.kind==PIECE_PORT and 0<=piece.port<len(node.child_identities):
                    walk(self._node(node.child_identities[piece.port]),(*path,int(piece.port)),False)
                else:raise HierarchicalRefuse('parametric:template_piece')
        walk(source,(),True)
        if not ops or len(ops)>MAX_OPS:raise HierarchicalRefuse('parametric:ops')
        rank=1+(max(called) if called else 0)
        wd=self._witness_digest(witnesses)
        identity=_id('parametric-condensed-recipe-v1',[root_shape,rank,[asdict(x) for x in ops],sorted(deps),list(map(int,witnesses)),wd])
        return ParametricCondensedRecipeV1(identity,root_shape,rank,tuple(ops),tuple(sorted(deps)),tuple(map(int,witnesses)),wd)

    def _original(self,closure):
        node=self._node(closure)
        if self.h.closure(int(node.identity)) is None:return self._render_original(node)
        if (int(getattr(node,'depth',0))>0
                and self.h.span_pieces(int(node.template_identity),int(node.context),len(getattr(node,'child_identities',()))) is None):
            cached=self.h._hot_surfaces.get(int(node.identity))
            if cached is not None:return tuple(cached),1
        self.h.drop_hot_cache();surface=self.h._surface_original(node.identity);touches=self.h.last_materialization_touches
        return tuple(surface),int(touches)

    def _execute(self,recipe,target,count_use=True):
        target=self._node(target)
        if self.shape_digest(target)!=recipe.shape_digest:raise HierarchicalRefuse('parametric:shape')
        if not self._dependencies_live(recipe):raise HierarchicalRefuse('parametric:dependency')
        out=[];touches=1
        for op in recipe.ops:
            touches+=1
            if op.kind==OP_SLOT:
                leaf=self._subtree(target,op.path)
                if leaf.depth!=0 or not leaf._leaf_surface:raise HierarchicalRefuse('parametric:slot')
                # Formal slot values are already bound at the compact Recipe boundary.
                # Reading that boundary value is one op; do not recursively rematerialize
                # the leaf through the source hierarchy.
                out.extend(leaf._leaf_surface)
            elif op.kind==OP_LITERAL:
                pieces=None
                for context,arity,dep_tid in recipe.dependencies:
                    if int(dep_tid)==int(op.reference_identity):
                        pieces=language_span_pieces(self.h.language,dep_tid,context,arity)
                        if pieces is not None:break
                if pieces is None or not 0<=op.piece_index<len(pieces):raise HierarchicalRefuse('parametric:literal')
                piece=pieces[op.piece_index]
                if piece.kind!=PIECE_LITERAL:raise HierarchicalRefuse('parametric:literal')
                out.extend(piece.literal)
            elif op.kind==OP_CALL:
                child=self._subtree(target,op.path);lower=self.recipes.get(op.reference_identity)
                if lower is None or not lower.active:raise HierarchicalRefuse('parametric:subrecipe')
                surface,t=self._execute(lower,child,count_use);out.extend(surface);touches+=t
            else:raise HierarchicalRefuse('parametric:op')
        if count_use:recipe.uses+=1
        return tuple(out),touches

    def _owning_relation_live(self,node):
        node=self._node(node)
        if int(getattr(node,'depth',0))==0:return True
        kids=getattr(node,'child_identities',())
        current=self.h.language.span_template(int(node.context),len(kids))
        tid=int(getattr(node,'template_identity',0) or 0)
        return current is not None and tid!=0 and int(current.identity[:15],16)==tid

    def _nominate(self,shape,source):
        if shape in self.by_shape:return self.recipes[self.by_shape[shape]]
        if len(self.recipes)>=MAX_RECIPES:raise HierarchicalRefuse('parametric:recipe_bound')
        if not self._owning_relation_live(source):return None
        witnesses=tuple(self.pressure.get(shape,())[:MIN_DISTINCT_WITNESSES])
        if len(witnesses)<MIN_DISTINCT_WITNESSES:return None
        recipe=self._compile(source,witnesses);self.recipes[recipe.identity]=recipe;self.by_shape[shape]=recipe.identity;self.pressure.pop(shape,None);return recipe

    def materialize(self,closure):
        closure=self._remember(self._node(closure));shape=self.shape_digest(closure);self.last_recipe=0;self.last_rank=0
        recipe=self.recipes.get(self.by_shape.get(shape,0))
        if recipe is not None and recipe.active:
            if not self._dependencies_live(recipe):
                recipe.active=0;recipe.probation_passes=0;recipe.deoptimizations+=1;self.pressure.pop(shape,None)
            else:
                surface,t=self._execute(recipe,closure,True);self.last_touches=t;self.last_recipe=recipe.identity;self.last_rank=recipe.rank;return surface
        original,touches=self._original(closure);self.last_touches=touches
        if recipe is not None and self._dependencies_live(recipe):
            try:shadow,shadow_touches=self._execute(recipe,closure,False)
            except HierarchicalRefuse:shadow=();shadow_touches=0
            if shadow==original:
                recipe.probation_passes+=1
                if recipe.probation_passes>=PROBATION_PASSES:recipe.active=1
            else:
                recipe.probation_passes=0;recipe.deoptimizations+=1
            self.last_touches+=shadow_touches
        else:
            rows=self.pressure.setdefault(shape,[])
            if closure.identity not in rows:rows.append(closure.identity)
            if len(rows)>=MIN_DISTINCT_WITNESSES:self._nominate(shape,closure)
        return original

    def _collect_rematerialized(self,ident,out):
        ident=int(ident)
        if ident in out or self.h.closure(ident) is not None:return
        node=self.nodes.get(ident)
        if node is None:return
        out[ident]=node
        for child in getattr(node,'child_identities',()):self._collect_rematerialized(int(child),out)

    def checkpoint(self):
        data={'schema':2,'recipes':[{'identity':r.identity,'shape_digest':r.shape_digest,'rank':r.rank,'ops':[asdict(o) for o in r.ops],'dependencies':[list(x) for x in r.dependencies],'witness_closures':list(r.witness_closures),'witness_digest':r.witness_digest,'probation_passes':r.probation_passes,'active':r.active,'uses':r.uses,'deoptimizations':r.deoptimizations} for r in sorted(self.recipes.values(),key=lambda x:(x.rank,x.identity))]}
        residual={k:list(v) for k,v in sorted(self.pressure.items()) if k not in self.by_shape}
        if residual:data['pressure']=residual
        needed={}
        for recipe in self.recipes.values():
            for ident in recipe.witness_closures:self._collect_rematerialized(ident,needed)
        for rows in residual.values():
            for ident in rows:self._collect_rematerialized(ident,needed)
        if needed:
            data['nodes']=[{'identity':int(n.identity),'context':int(n.context),'template_identity':int(getattr(n,'template_identity',0) or 0),'child_identities':list(map(int,getattr(n,'child_identities',()))),'depth':int(getattr(n,'depth',0))} for n in (needed[i] for i in sorted(needed))]
        return data

    @classmethod
    def restore(cls,hierarchy,data):
        if data.get('schema')!=2:raise HierarchicalRefuse('parametric:checkpoint_schema')
        out=cls(hierarchy);out.pressure={str(k):list(map(int,v)) for k,v in data.get('pressure',{}).items()}
        for row in data.get('nodes',()):
            node=SimpleNamespace(identity=int(row['identity']),context=int(row['context']),template_identity=int(row.get('template_identity',0) or 0),child_identities=tuple(map(int,row.get('child_identities',()))),depth=int(row.get('depth',0)))
            out.nodes[node.identity]=node
        for row in data.get('recipes',()):
            ops=tuple(ParametricOpV1(int(o['kind']),tuple(map(int,o.get('path',()))),int(o.get('reference_identity',0)),int(o.get('piece_index',0))) for o in row['ops'])
            r=ParametricCondensedRecipeV1(int(row['identity']),str(row['shape_digest']),int(row['rank']),ops,tuple(tuple(map(int,x)) for x in row['dependencies']),tuple(map(int,row['witness_closures'])),str(row['witness_digest']),int(row['probation_passes']),int(row['active']),int(row['uses']),int(row['deoptimizations']))
            if r.identity in out.recipes or r.shape_digest in out.by_shape or not r.ops or r.rank<1:raise HierarchicalRefuse('parametric:checkpoint_recipe')
            witness=out._node(r.witness_closures[-1])
            if out._owning_relation_live(witness):
                expected=out._compile(witness,r.witness_closures)
                if (r.identity,r.shape_digest,r.rank,r.ops,r.dependencies,r.witness_digest)!=(expected.identity,expected.shape_digest,expected.rank,expected.ops,expected.dependencies,expected.witness_digest):raise HierarchicalRefuse('parametric:checkpoint_mismatch')
            elif r.active:
                raise HierarchicalRefuse('parametric:checkpoint_active_without_relation')
            out.recipes[r.identity]=r;out.by_shape[r.shape_digest]=r.identity
        return out
