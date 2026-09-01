#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1,HierarchicalRefuse,MAX_HOT_SURFACE_BYTES


def u(s):return tuple(s.encode())


def fixture(depth=48):
    e=LearnedSurfaceEcologyV1();A,B=1,2;CLAUSE,JOIN=10,11
    for feature,text in ((A,'a'),(B,'b')):
        e.observe_naming(feature,u(text),100+feature);e.observe_naming(feature,u(text),200+feature)
    e.observe_construction(CLAUSE,(A,),u('a'),301);e.observe_construction(CLAUSE,(B,),u('b'),302)
    h=HierarchicalConstructionV1(e);a=h.leaf(CLAUSE,(A,));b=h.leaf(CLAUSE,(B,))
    h.observe(JOIN,(a,b),u('a b'),401);h.observe(JOIN,(b,a),u('b a'),402)
    cur=a
    legacy_surface=2;legacy_ancestry=0
    for _ in range(depth):
        cur=h.compose(JOIN,(cur,b))
        # What the previous representation physically retained at every closure:
        # full cumulative surface plus one flattened ancestry integer per ancestor.
        legacy_surface+=len(cur.surface)
        legacy_ancestry+=len(cur.ancestry)
    return e,h,cur,b,JOIN,legacy_surface,legacy_ancestry


def main():
    started=time.perf_counter();checks={};e,h,deep,b,JOIN,legacy_surface,legacy_ancestry=fixture(48)
    expected=u('a'+' b'*48)
    checks['depth48_exact_surface']=deep.depth==48 and deep.surface==expected
    q=h.persistent_quantity();structural_units=q['retained_leaf_surface_bytes']+q['retained_child_refs']+q['retained_template_literal_bytes']+q['retained_template_pieces']
    legacy_units=legacy_surface+legacy_ancestry
    checks['persistent_structure_linear']=q['retained_child_refs']==96 and q['retained_leaf_surface_bytes']==2 and not hasattr(h,'_template_witnesses') and q['retained_template_pieces']==0 and 'templates' not in h.checkpoint()
    checks['legacy_superlinear_gap']=legacy_units>20*structural_units
    cp=h.checkpoint();serialized=json.dumps(cp,sort_keys=True,separators=(',',':')).encode()
    checks['checkpoint_has_no_cumulative_surface_or_flat_ancestry']=all('surface' not in row and 'ancestry' not in row for row in cp['closures'] if row['depth']>0)
    checks['bounded_hot_cache']=q['hot_surface_cache_bytes']<=MAX_HOT_SURFACE_BYTES

    # Cold rematerialization must reconstruct exact bytes solely from compact child
    # references plus the frozen structural template witness.
    h.drop_hot_cache();cold=deep.surface
    checks['cold_rematerialization_exact']=cold==expected and h.last_materialization_touches>=49
    checks['hot_reuse_touched_one']=deep.surface==expected and h.last_materialization_touches==1

    # Exact checkpoint restore has no hot cache authority and rematerializes the same
    # structure/output. The serialized state remains compact.
    e2=LearnedSurfaceEcologyV1.restore(e.checkpoint());h2=HierarchicalConstructionV1.restore(e2,copy.deepcopy(cp));restored=h2.closure(deep.identity)
    checkpoint_exact=h2.checkpoint()==cp
    checks['checkpoint_exact']=checkpoint_exact and restored is not None and restored.surface==expected and h2.persistent_quantity()['hot_surface_cache_bytes']<=MAX_HOT_SURFACE_BYTES

    e.withdraw_source(402)
    checks['withdrawal_preserves_lived_hot_surface']=deep.surface==expected
    h.drop_hot_cache()
    try:_=deep.surface;cold_refused=False
    except HierarchicalRefuse:cold_refused=True
    checks['single_source_withdrawal_preserves_lived_cold_rematerialization']=not cold_refused and deep.surface==expected
    try:h.compose(JOIN,(deep,b));new_refused=False
    except HierarchicalRefuse:new_refused=True
    checks['withdrawal_reopens_future_composition']=new_refused
    e.restore_source(402)

    lesion_e=LearnedSurfaceEcologyV1.restore(e.checkpoint());lesion_e.withdraw_source(401);lesion_e.withdraw_source(402)
    lesion=HierarchicalConstructionV1.restore(lesion_e,copy.deepcopy(cp));leaf_surface=lesion.closure(b.identity).surface
    lesion.drop_hot_cache()
    try:_=lesion.closure(deep.identity).surface;lesion_refused=False
    except HierarchicalRefuse:lesion_refused=True
    checks['language_lesion_breaks_cold_recursive_closure']=lesion_refused and lesion.closure(b.identity).surface==leaf_surface
    checks['visible_discussion_improvement']=checks['single_source_withdrawal_preserves_lived_cold_rematerialization'] and checks['withdrawal_reopens_future_composition'] and checks['language_lesion_breaks_cold_recursive_closure'] and checks['depth48_exact_surface']

    out={'schema':'0x1.reference-hierarchy-storage-pressure.v1','pass':all(checks.values()),'checks':checks,'depth':deep.depth,'closures':h.closure_count,'current_surface_bytes':len(expected),'legacy_retained_surface_bytes':legacy_surface,'legacy_retained_ancestry_ints':legacy_ancestry,'legacy_units':legacy_units,'compact_structural_units':structural_units,'checkpoint_bytes':len(serialized),'persistent_quantity':q,'runtime_llm':False,'semantic_summary_cache':False,'physical_direct_parity':'NOT_RUN/RED','claim':'COMPACT_RECURSIVE_CLOSURE_STORAGE_REFERENCE_PROPERTY_ONLY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_HIERARCHY_STORAGE '+('GREEN' if out['pass'] else 'RED')+f" depth={deep.depth} legacy_units={legacy_units} compact_units={structural_units} direct_parity=RED")
    print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)


if __name__=='__main__':main()
