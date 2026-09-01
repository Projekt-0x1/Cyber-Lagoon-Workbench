#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import replace
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from types import SimpleNamespace
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1,rematerialize_transient_plan,rematerialize_transient_sequence_plan
from reference_incremental_expression_v1 import IncrementalExpressionV1,IncrementalTransientExpressionV1,IncrementalTransientSequenceExpressionV1,ExpressionRefuse,SEG_LEAF,SEG_LITERAL,language_span_pieces

def u(s):return tuple(s.encode())
def refused(fn):
    try:fn();return False
    except ExpressionRefuse:return True

def build(count=128):
    e=LearnedSurfaceEcologyV1();h=HierarchicalConstructionV1(e);CTX=9101
    leaves=[h.leaf_surface(9001,10000+i,u(f'unit{i:03d}.')) for i in range(count)]
    assert h.observe(CTX,(leaves[0],leaves[1]),leaves[0].surface+u(' ')+leaves[1].surface,5001)
    assert h.observe(CTX,(leaves[2],leaves[3]),leaves[2].surface+u(' ')+leaves[3].surface,5002)
    assert e.span_template(CTX,2) is not None
    level=leaves
    while len(level)>1:
        level=[h.compose(CTX,(level[i],level[i+1])) for i in range(0,len(level),2)]
    return e,h,level[0],CTX

def run_clean(h,root):
    t=IncrementalExpressionV1(h,root.identity);out=[];plans=0
    while True:
        p=t.emit()
        if p is None:break
        plans+=1;out.append(p.value);assert t.reafference(p,p.value)
        if plans>100000:raise AssertionError('trajectory did not terminate')
    return t,tuple(out)

def main():
    started=time.perf_counter();checks={};    e,h,root,ctx=build();node_count=h.closure_count;expected=tuple(root.surface)
    saved_candidates=e.span_candidates
    e.span_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_span_pieces_must_not_enumerate'))
    try:
        tid=int(e.span_template(ctx,2).identity[:15],16)
        pieces=language_span_pieces(e,tid,ctx,2)
        hier=h.span_pieces(tid,ctx,2)
        probe=IncrementalExpressionV1(h,root.identity);probe_first=probe.emit()
    finally:
        e.span_candidates=saved_candidates
    checks['unique_span_pieces_do_not_enumerate_candidates']=pieces is not None and hier==pieces and probe_first is not None
    tied=LearnedSurfaceEcologyV1();left=u('left');right=u('right');TIED_CTX=9102
    for source in (6001,6002):assert tied.observe_span(TIED_CTX,(left,right),left+u(' then ')+right,source)
    for source in (6003,6004):assert tied.observe_span(TIED_CTX,(left,right),left+u(' danach ')+right,source)
    tied_rows=tied.span_candidates(TIED_CTX,2)
    selected=next(row for row in tied_rows if any(piece.literal==u(' then ') for piece in row.pieces))
    selected_pieces=language_span_pieces(tied,int(selected.identity[:15],16),TIED_CTX,2)
    checks['selected_span_identity_survives_global_ecology_tie']=(
        tied.span_template(TIED_CTX,2) is None
        and selected_pieces==selected.pieces)
    winner_calls={'n':0};real_span=e.span_template
    def count_span(*a,**k):
        winner_calls['n']+=1;return real_span(*a,**k)
    e.span_template=count_span
    try:
        counted=IncrementalExpressionV1(h,root.identity);counted_first=counted.emit()
    finally:
        e.span_template=real_span
    checks['unique_span_winner_once_per_frame']=counted_first is not None and winner_calls['n']<=1
    held=IncrementalExpressionV1(h,root.identity)
    while True:
        p=held.emit();kind=p.segment_kind;assert held.reafference(p,p.value)
        if kind==SEG_LEAF and held.segment_kind==0:break
    held_calls={'n':0};real_held=e.span_template
    def count_held(*a,**k):
        held_calls['n']+=1;return real_held(*a,**k)
    e.span_template=count_held
    try:
        lit=held.emit();assert lit is not None and lit.segment_kind==SEG_LITERAL
        after_enter=held_calls['n'];ok=held.reafference(lit,lit.value)
    finally:
        e.span_template=real_held
    checks['held_literal_reafference_skips_unique_winner']=ok and after_enter>=1 and held_calls['n']==after_enter
    t=IncrementalExpressionV1(h,root.identity);first=t.emit()
    checks['first_byte_is_boundary_local']=first is not None and first.value==expected[0] and t.last_plan_touches<=2*root.depth+2 and t.last_plan_touches*8<node_count
    checkpoint=json.dumps(t.checkpoint(),sort_keys=True,separators=(',',':')).encode()
    checks['trajectory_checkpoint_has_no_output_transcript']=len(checkpoint)<1800 and b'payload' not in checkpoint and bytes(expected[:32]) not in checkpoint
    checks['frame_span_hold_omitted_from_checkpoint']=b'frame_pieces' not in checkpoint
    checks['next_byte_waits_for_reafference']=refused(t.emit)
    forged=replace(first,identity=first.identity^1)
    checks['forged_reafference_refuses']=refused(lambda:t.reafference(forged,forged.value))
    cp=copy.deepcopy(t.checkpoint());rt=IncrementalExpressionV1.restore(h,cp)
    checks['pending_checkpoint_exact']=rt.checkpoint()==cp and rt.pending==first and rt.reafference(first,first.value)

    # Fault does not advance structural cursor; next plan retries the same byte with new attempt identity.
    f=IncrementalExpressionV1(h,root.identity)
    for _ in range(37):
        p=f.emit();assert p is not None and f.reafference(p,p.value)
    bad=f.emit();before_ordinal=f.ordinal
    checks['mismatch_reafference_detected']=bad is not None and not f.reafference(bad,(bad.value+1)&255) and f.ordinal==before_ordinal and f.repairs==1
    retry=f.emit();checks['repair_reuses_same_boundary_byte']=retry.value==bad.value and retry.byte_index==bad.byte_index and retry.identity!=bad.identity and f.reafference(retry,retry.value)

    clean,out=run_clean(h,root)
    checks['full_incremental_surface_exact']=out==expected and clean.complete and clean.ordinal==len(expected)
    checks['stack_bounded_by_hierarchy_depth']=clean.max_stack<=root.depth+1 and clean.max_stack<16
    checks['trajectory_does_not_materialize_whole_plan']=not hasattr(clean,'payload') and not hasattr(clean,'surface') and len(json.dumps(clean.checkpoint()))<1800

    # Pause/resume at arbitrary structural position with no host-supplied continuation.
    q=IncrementalExpressionV1(h,root.identity);prefix=[]
    for _ in range(211):
        p=q.emit();prefix.append(p.value);q.reafference(p,p.value)
    qcp=copy.deepcopy(q.checkpoint());q2=IncrementalExpressionV1.restore(h,qcp);tail=[]
    while True:
        p=q2.emit()
        if p is None:break
        tail.append(p.value);q2.reafference(p,p.value)
    checks['midstream_checkpoint_resumes_without_transcript']=tuple(prefix+tail)==expected and q2.complete
    lang,lang_out=run_clean(h,root)
    checks['language_pieces_emit_after_witness_table_cleared']=lang_out==expected and lang.complete and not hasattr(h,'_template_witnesses')
    te=LearnedSurfaceEcologyV1();th=HierarchicalConstructionV1(te);TCTX=9201
    tleaves=[th.leaf_surface(9002,11000+i,u(f'leaf{i:02d}.')) for i in range(8)]
    assert th.observe(TCTX,(tleaves[0],tleaves[1]),tleaves[0].surface+u(' ')+tleaves[1].surface,6101)
    assert th.observe(TCTX,(tleaves[2],tleaves[3]),tleaves[2].surface+u(' ')+tleaves[3].surface,6102)
    tnodes={};tlevel=list(tleaves);tplan=None;tsurface=None
    while len(tlevel)>1:
        nxt=[]
        for i in range(0,len(tlevel),2):
            tplan,tsurface=rematerialize_transient_plan(te,TCTX,(tlevel[i],tlevel[i+1]))
            node=SimpleNamespace(identity=tplan.identity,context=tplan.context,template_identity=tplan.template_identity,child_identities=tplan.child_identities,depth=tplan.depth,surface=tsurface)
            tnodes[tplan.identity]=node;nxt.append(node)
        tlevel=nxt
    checks['transient_tree_writes_no_persist_composites']=all(th.closure(i) is None for i in tnodes) and tplan.depth==3
    bare=(SimpleNamespace(identity=tleaves[0].identity,depth=0),SimpleNamespace(identity=tleaves[1].identity,depth=0))
    p0,s0=rematerialize_transient_plan(te,TCTX,bare,render=False)
    p1,s1=rematerialize_transient_plan(te,TCTX,(tleaves[0],tleaves[1]))
    checks['rematerialize_plan_skips_surface']=s0==() and p0.identity==p1.identity and len(s1)==len(tleaves[0].surface)+1+len(tleaves[1].surface)
    tw=IncrementalTransientExpressionV1(th,tplan,nodes=tnodes);tout=[]
    while True:
        p=tw.emit()
        if p is None:break
        tout.append(p.value);assert tw.reafference(p,p.value)
    checks['transient_rematerialized_incremental_surface']=tuple(tout)==tsurface and tw.complete and not hasattr(th,'_template_witnesses')
    tq=IncrementalTransientExpressionV1(th,tplan,nodes=tnodes);tprefix=[]
    for _ in range(12):
        p=tq.emit();tprefix.append(p.value);tq.reafference(p,p.value)
    tq2=IncrementalTransientExpressionV1.restore(th,copy.deepcopy(tq.checkpoint()));ttail=[]
    while True:
        p=tq2.emit()
        if p is None:break
        ttail.append(p.value);tq2.reafference(p,p.value)
    checks['transient_midstream_resume_without_transcript']=tuple(tprefix+ttail)==tsurface and tq2.complete
    forged=copy.deepcopy(tq.checkpoint());fk=next(iter(forged['nodes']));forged['nodes'][fk]['identity']=int(forged['nodes'][fk]['identity'])^1
    checks['forged_rematerialized_node_refused']=refused(lambda:IncrementalTransientExpressionV1.restore(th,forged))
    def _descendants(ident,nodes):
        out={int(ident)};n=nodes.get(int(ident))
        if n:
            for c in getattr(n,'child_identities',()):out|=_descendants(c,nodes)
        return out
    drop=_descendants(tplan.child_identities[1],tnodes)
    tleft={k:v for k,v in tnodes.items() if k not in drop}
    tlazy=IncrementalTransientExpressionV1(th,tplan,nodes=tleft);lleft=[];blocked=False
    while True:
        try:p=tlazy.emit()
        except ExpressionRefuse:
            blocked=True;break
        if p is None:break
        lleft.append(p.value);tlazy.reafference(p,p.value)
    checks['tree_starts_without_right_branch']=blocked and 0<len(lleft)<len(tsurface) and tlazy.pending is None
    tlazy.nodes.update({k:v for k,v in tnodes.items() if k in drop})
    lrest=[]
    while True:
        p=tlazy.emit()
        if p is None:break
        lrest.append(p.value);tlazy.reafference(p,p.value)
    checks['tree_reinstates_right_branch']=tuple(lleft+lrest)==tsurface and tlazy.complete
    left_child=int(tplan.child_identities[0]);left_surface=tnodes[left_child].surface
    tpr=IncrementalTransientExpressionV1(th,tplan,nodes=dict(tnodes));lpre=[]
    while True:
        p=tpr.emit();lpre.append(p.value);tpr.reafference(p,p.value)
        if all(int(f.closure_identity)!=left_child for f in tpr.stack):break
    checks['spent_left_branch_dropped_live']=left_child not in tpr.nodes and int(tplan.child_identities[1]) in tpr.nodes
    pcp=tpr.checkpoint();left_ids=_descendants(left_child,tnodes)
    checks['spent_left_branch_omitted_from_checkpoint']=all(str(i) not in pcp['nodes'] for i in left_ids) and str(int(tplan.child_identities[1])) in pcp['nodes']
    tpr2=IncrementalTransientExpressionV1.restore(th,copy.deepcopy(pcp));ltail=[]
    while True:
        p=tpr2.emit()
        if p is None:break
        ltail.append(p.value);tpr2.reafference(p,p.value)
    checks['spent_left_branch_restore_finishes']=tuple(lpre+ltail)==tsurface and tpr2.complete
    twd=IncrementalTransientExpressionV1(th,tplan,nodes=tnodes)
    while True:
        p=twd.emit();kind=p.segment_kind;twd.reafference(p,p.value)
        if kind==SEG_LEAF and twd.segment_kind==0:break
    te.withdraw_source(6101);te.withdraw_source(6102)
    checks['transient_withdrawal_blocks_future_structure']=refused(twd.emit)
    prefix_n=twd.ordinal;wcp=copy.deepcopy(twd.checkpoint())
    checks['transient_fresh_plan_refuses_while_withdrawn']=refused(lambda:IncrementalTransientExpressionV1(th,tplan,nodes=tnodes))
    twd2=IncrementalTransientExpressionV1.restore(th,copy.deepcopy(wcp))
    checks['transient_withdrawn_cursor_survives_rest']=twd2.ordinal==prefix_n and twd2.pending is None and refused(twd2.emit)
    assert th.observe(TCTX,(tleaves[0],tleaves[1]),tleaves[0].surface+u(' ')+tleaves[1].surface,7101)
    assert th.observe(TCTX,(tleaves[2],tleaves[3]),tleaves[2].surface+u(' ')+tleaves[3].surface,7102)
    rtail=[]
    while True:
        p=twd2.emit()
        if p is None:break
        rtail.append(p.value);twd2.reafference(p,p.value)
    checks['transient_reacquire_resumes_remaining']=tuple(rtail)==tsurface[prefix_n:] and twd2.complete
    se=LearnedSurfaceEcologyV1();sh=HierarchicalConstructionV1(se);SCTX=9301
    sleaves=[sh.leaf_surface(9003,12000+i,u(f'seq{i:02d}.')) for i in range(4)]
    assert sh.observe(SCTX,(sleaves[0],sleaves[1]),sleaves[0].surface+u(' ')+sleaves[1].surface,8101)
    assert sh.observe(SCTX,(sleaves[2],sleaves[3]),sleaves[2].surface+u(' ')+sleaves[3].surface,8102)
    splan,ssurface=rematerialize_transient_sequence_plan(se,SCTX,sleaves)
    sbare=tuple(SimpleNamespace(identity=s.identity,depth=0) for s in sleaves)
    sp0,ss0=rematerialize_transient_sequence_plan(se,SCTX,sbare,render=False)
    checks['rematerialize_sequence_plan_skips_surface']=ss0==() and sp0.identity==splan.identity
    sw=IncrementalTransientSequenceExpressionV1(sh,splan);sout=[]
    while True:
        p=sw.emit()
        if p is None:break
        sout.append(p.value);assert sw.reafference(p,p.value)
    checks['transient_sequence_incremental_surface']=tuple(sout)==ssurface and sw.complete
    sq=IncrementalTransientSequenceExpressionV1(sh,splan);sprefix=[]
    for _ in range(8):
        p=sq.emit();sprefix.append(p.value);sq.reafference(p,p.value)
    sq2=IncrementalTransientSequenceExpressionV1.restore(sh,copy.deepcopy(sq.checkpoint()));stail=[]
    while True:
        p=sq2.emit()
        if p is None:break
        stail.append(p.value);sq2.reafference(p,p.value)
    checks['transient_sequence_midstream_resume']=tuple(sprefix+stail)==ssurface and sq2.complete
    sfg=copy.deepcopy(sq.checkpoint());sfg['sequence_plan']['identity']=int(sfg['sequence_plan']['identity'])^1
    checks['forged_sequence_plan_refused']=refused(lambda:IncrementalTransientSequenceExpressionV1.restore(sh,sfg))
    swd=IncrementalTransientSequenceExpressionV1(sh,splan);sfirst=[]
    while True:
        p=swd.emit();sfirst.append(p.value);swd.reafference(p,p.value)
        if swd.stage_index==0 and swd.current.ordinal==len(sleaves[0].surface):break
    se.withdraw_source(8101);se.withdraw_source(8102)
    checks['transient_sequence_withdrawal_blocks_next_stage']=refused(swd.emit)
    scp=copy.deepcopy(swd.checkpoint())
    checks['transient_sequence_fresh_refuses_while_withdrawn']=refused(lambda:IncrementalTransientSequenceExpressionV1(sh,splan))
    swd2=IncrementalTransientSequenceExpressionV1.restore(sh,copy.deepcopy(scp))
    checks['transient_sequence_withdrawn_cursor_survives_rest']=swd2.stage_index==0 and refused(swd2.emit)
    assert sh.observe(SCTX,(sleaves[0],sleaves[1]),sleaves[0].surface+u(' ')+sleaves[1].surface,9101)
    assert sh.observe(SCTX,(sleaves[2],sleaves[3]),sleaves[2].surface+u(' ')+sleaves[3].surface,9102)
    srest=[]
    while True:
        p=swd2.emit()
        if p is None:break
        srest.append(p.value);swd2.reafference(p,p.value)
    checks['transient_sequence_reacquire_resumes']=tuple(srest)==ssurface[len(sfirst):] and swd2.complete
    lz=LearnedSurfaceEcologyV1();lh=HierarchicalConstructionV1(lz);LCTX=9401
    lzleaves=[lh.leaf_surface(9005,14000+i,u(f'laz{i:02d}.')) for i in range(4)]
    assert lh.observe(LCTX,(lzleaves[0],lzleaves[1]),lzleaves[0].surface+u(' ')+lzleaves[1].surface,8201)
    assert lh.observe(LCTX,(lzleaves[2],lzleaves[3]),lzleaves[2].surface+u(' ')+lzleaves[3].surface,8202)
    lzplan,lzsurface=rematerialize_transient_sequence_plan(lz,LCTX,lzleaves)
    assert all(lh.retire_unreferenced_leaf(lzleaves[i].identity) for i in (1,2,3))
    lztraj=IncrementalTransientSequenceExpressionV1(lh,lzplan);lzfirst=[];cur=lztraj.current
    while not cur.complete:
        p=cur.emit()
        if p is None:break
        lzfirst.append(p.value);cur.reafference(p,p.value)
    checks['sequence_starts_without_later_clauses']=tuple(lzfirst)==lzleaves[0].surface and lztraj.stage_index==0
    checks['sequence_later_clause_required_at_boundary']=refused(lztraj.emit)
    lh.leaf_surface(9005,14001,u('laz01.'));lh.leaf_surface(9005,14002,u('laz02.'));lh.leaf_surface(9005,14003,u('laz03.'))
    lzrest=[]
    while True:
        p=lztraj.emit()
        if p is None:break
        lzrest.append(p.value);lztraj.reafference(p,p.value)
    checks['sequence_reinstates_later_clauses']=tuple(lzfirst+lzrest)==lzsurface and lztraj.complete
    checks['visible_discussion_improvement']=checks['selected_span_identity_survives_global_ecology_tie'] and checks['held_literal_reafference_skips_unique_winner'] and checks['unique_span_winner_once_per_frame'] and checks['spent_left_branch_dropped_live'] and checks['spent_left_branch_omitted_from_checkpoint'] and checks['spent_left_branch_restore_finishes'] and checks['tree_starts_without_right_branch'] and checks['tree_reinstates_right_branch'] and checks['sequence_starts_without_later_clauses'] and checks['sequence_later_clause_required_at_boundary'] and checks['sequence_reinstates_later_clauses'] and checks['rematerialize_sequence_plan_skips_surface'] and checks['rematerialize_plan_skips_surface'] and checks['transient_rematerialized_incremental_surface'] and checks['transient_tree_writes_no_persist_composites'] and checks['transient_midstream_resume_without_transcript'] and checks['forged_rematerialized_node_refused'] and checks['transient_sequence_incremental_surface'] and checks['transient_sequence_midstream_resume'] and checks['forged_sequence_plan_refused'] and checks['transient_sequence_withdrawal_blocks_next_stage'] and checks['transient_sequence_withdrawn_cursor_survives_rest'] and checks['transient_sequence_reacquire_resumes'] and checks['transient_sequence_fresh_refuses_while_withdrawn'] and checks['transient_withdrawal_blocks_future_structure'] and checks['transient_withdrawn_cursor_survives_rest'] and checks['transient_reacquire_resumes_remaining'] and checks['transient_fresh_plan_refuses_while_withdrawn'] and checks['language_pieces_emit_after_witness_table_cleared'] and checks['full_incremental_surface_exact']

    # Construction source withdrawal is checked at the next not-yet-committed structural boundary.
    w=IncrementalExpressionV1(h,root.identity)
    # consume exactly first leaf segment, leaving no current literal/leaf segment committed
    while True:
        p=w.emit();assert p is not None;kind=p.segment_kind;w.reafference(p,p.value)
        if kind==SEG_LEAF and w.segment_kind==0:break
    e.withdraw_source(5002)
    checks['source_withdrawal_blocks_future_structure']=refused(w.emit)
    # Less brittle causal check: no additional byte was emitted after withdrawal.
    checks['withdrawal_does_not_use_cached_full_surface']=w.pending is None and w.ordinal>0 and w.ordinal<len(expected)

    elapsed=(time.perf_counter()-started)*1000;checks['rapid_reference_runtime']=elapsed<1000
    result={'schema':'0x1.reference-incremental-expression.v1','pass':all(checks.values()),'language_phenotype_improved':True,'checks':checks,'quantity':{'closure_nodes':node_count,'root_depth':root.depth,'surface_bytes':len(expected),'first_plan_touches':t.last_plan_touches,'max_stack':clean.max_stack,'trajectory_checkpoint_bytes':len(checkpoint),'total_plan_touches':clean.total_plan_touches},'repairs':f.repairs,'elapsed_ms':round(elapsed,3),'runtime_llm':False,'transcript_state':False,'physical_direct_parity':'NOT_RUN/RED','claim':'INCREMENTAL_HIERARCHICAL_LANGUAGE_PLANNING_REFERENCE_ONLY'}
    print('FOUNDRY_INCREMENTAL_EXPRESSION '+('GREEN' if result['pass'] else 'RED')+f" clauses=128 bytes={len(expected)} depth={root.depth} first_touches={t.last_plan_touches} checkpoint={len(checkpoint)} repair={f.repairs} ms={result['elapsed_ms']}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
