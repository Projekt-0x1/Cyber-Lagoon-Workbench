#!/usr/bin/env python3
from __future__ import annotations
import copy,hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from language_development_v1 import LanguageDevelopmentV1
from reference_population_v1 import PopulationRecruitmentEcologyV1

PKG=Path('hardware_native/data/organism_packages/alice-prose-v1')
MANIFEST=PKG/'package.json';CONTACTS=PKG/'contacts.ndjson'
SOURCE=1001

def load():
    manifest=json.loads(MANIFEST.read_text())
    rows=[json.loads(line) for line in CONTACTS.read_text().splitlines() if line.strip()]
    payloads=[bytes.fromhex(row['payload_hex']) for row in rows]
    return manifest,rows,payloads

def state_digest(machine):
    rows=[]
    for (unit,source),ev in machine.raw_chunk_evidence.items():
        rows.append((unit,source,ev.count,ev.left_diversity,ev.right_diversity,ev.utility,tuple(ev.resident_signature),bytes(machine.surface_units[unit].raw).hex()))
    return hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()

def repartition(raw,size):
    return [raw[i:i+size] for i in range(0,len(raw),size)]

def longest_profile(machine,raw):
    lengths=[]
    data=bytes(raw)
    machine.last_raw_chunk_touches=0
    for i,first in enumerate(data):
        best=1
        for uid in machine._raw_chunk_first.get(first,()):
            unit=machine.surface_units.get(uid)
            if unit is None:continue
            chunk=bytes(unit.raw)
            if len(chunk)>best and data.startswith(chunk,i) and machine._raw_unit_utility(uid) is not None:
                best=len(chunk)
        lengths.append(best)
    return {
      'mean_longest':sum(lengths)/len(lengths),
      'ge5_fraction':sum(x>=5 for x in lengths)/len(lengths),
      'ge8_fraction':sum(x>=8 for x in lengths)/len(lengths),
    }

def train(packets,cap):
    m=LanguageDevelopmentV1(32768)
    ids=m.consolidate_raw_stream(packets,SOURCE,max_chunk=12,min_count=8,min_contexts=3,max_units=cap)
    return m,ids

def main():
    t=time.perf_counter();checks={};manifest,rows,payloads=load()
    sequences=[int(row['sequence']) for row in rows]
    checks['candidate_package_not_promoted']=manifest.get('admissibility')=='candidate_only' and manifest['contact_stream'].get('canonical_ingress') is False
    checks['opaque_payload_only']=manifest['contact_stream'].get('ingress_projection')=='payload_hex_only' and manifest['contact_stream'].get('metadata_is_not_ingress') is True
    checks['contiguous_contact_chronology']=sequences==list(range(1,44))
    checks['declared_split']=manifest['curriculum']['training_ranges']==[[1,32]] and manifest['curriculum']['heldout_range']==[33,43]
    training=payloads[:32];heldout=b''.join(payloads[32:]);train_raw=b''.join(training)
    checks['nontoy_raw_extent']=len(train_raw)==131072 and len(heldout)==43242

    full,ids=train(training,768);full_digest=state_digest(full)
    one,_=train([train_raw],768)
    repacket,_=train(repartition(train_raw,997),768)
    checks['transport_packetization_invariant']=state_digest(one)==full_digest==state_digest(repacket)

    shuffled=list(training);random.Random(0x1A11CE).shuffle(shuffled)
    scrambled,_=train(shuffled,768)
    checks['chronology_scramble_changes_lived_chunk_evidence']=state_digest(scrambled)!=full_digest

    lattice=full.raw_chunk_lattice_stats(heldout)
    retained=sum(len(full.surface_units[uid].raw) for uid in ids)
    checks['bounded_no_transcript_retention']=retained<len(train_raw)//20 and max(len(full.surface_units[uid].raw) for uid in ids)<=12
    checks['heldout_reusable_chunk_lattice']=lattice['coverage_fraction']>0.85 and lattice['max_candidates_per_position']<=12 and lattice['positions_with_candidates']>0.99*len(heldout)
    checks['ambiguity_preserved_not_tokenized']=lattice['ambiguous_positions']>len(heldout)//2

    # Persist only compact reusable ecology, then unfold the same computation again.
    checkpoint=full.raw_ecology_checkpoint()
    persistent_wire_bytes=len(json.dumps(checkpoint,sort_keys=True,separators=(',',':')).encode())
    restored=LanguageDevelopmentV1.restore_raw_ecology(copy.deepcopy(checkpoint))
    checks['compact_checkpoint_restores_same_lattice']=(restored.raw_chunk_lattice_stats(heldout[:12000])==full.raw_chunk_lattice_stats(heldout[:12000]) and len(restored.population.occurrences)==0)
    corrupt=copy.deepcopy(checkpoint);corrupt['population']['prepared'][0][1]=0
    try:LanguageDevelopmentV1.restore_raw_ecology(corrupt)
    except Exception:checks['corrupt_compact_checkpoint_refuses']=True
    else:checks['corrupt_compact_checkpoint_refuses']=False

    network=full.raw_chunk_occurrence_network_stats(heldout[:12000])
    restored_network=restored.raw_chunk_occurrence_network_stats(heldout[:12000])
    checks['compact_recipe_unfolds_same_ephemeral_network']=network==restored_network
    checks['expanded_state_not_checkpointed']=(network['ephemeral_network'] and network['retained_occurrences_before']==network['retained_occurrences_after']==0 and network['unfolded_total_wire_bytes']>40*persistent_wire_bytes)
    q=full.quantity()
    checks['raw_chunks_are_distributed_population_matter']=(q['raw_chunk_prepared_signatures']==len(ids) and q['materialized_sites']>len(ids) and q['materialized_sites']<q['population_sites'])
    checks['heldout_chunk_occurrences_form_sparse_local_networks']=(network['active_chunk_occurrences']>20000 and network['networked_chunk_occurrences']>0.99*network['active_chunk_occurrences'] and network['span_edges']>network['active_chunk_occurrences'] and network['max_local_degree']<=16 and network['ephemeral_network'] and network['retained_occurrences_before']==network['retained_occurrences_after'])

    # Quantity aperture: same raw experience, only bounded retained resident matter changes.
    sweep=[]
    for cap in (128,512,1536):
        m,cap_ids=train(training,cap);stats=m.raw_chunk_lattice_stats(heldout);profile=longest_profile(m,heldout)
        sweep.append({'capacity':cap,'rows':len(cap_ids),'coverage':stats['coverage_fraction'],'max_candidates':stats['max_candidates_per_position'],**profile})
    checks['resident_quantity_increases_heldout_coverage']=sweep[0]['coverage']<sweep[1]['coverage']<sweep[2]['coverage']
    checks['resident_quantity_increases_reusable_span']=sweep[0]['mean_longest']<sweep[1]['mean_longest']<sweep[2]['mean_longest'] and sweep[0]['ge8_fraction']<sweep[1]['ge8_fraction']<sweep[2]['ge8_fraction']
    checks['quantity_work_remains_sparse']=max(row['max_candidates'] for row in sweep)<=12

    # Language is an outer modality into shared cognition, not the owner of other
    # representations. Use one actual Alice-derived prepared morphology plus three
    # opaque observer-side modality families. Persistent cross-Network state stores
    # only distributed morphology identities/site signatures; no bytes/pixels/audio
    # or valence payload can cross this boundary.
    relation_ecology=PopulationRecruitmentEcologyV1()
    language_evidence=next(row for row in full.raw_chunk_evidence.values()
                           if len(full.surface_units[row.unit].raw)>=3)
    language_signature=language_evidence.resident_signature
    # Observer-side families only: the runtime relation below never stores these
    # modality meanings. For the falsifier we treat them as visual-like,
    # auditory-like and interoceptive/somatic-like prepared morphologies.
    other_signatures=[full.population.prepare(features) for features in (
        (0x710001,0x710002,0x710003,0x710004),
        (0x720001,0x720002,0x720003),
        (0x730001,0x730002,0x730003,0x730004,0x730005))]
    actual=[full.population.activate_signature(signature,retain=True)
            for signature in (language_signature,*other_signatures)]
    joint=full.population.recruit(relation_ecology.network_occurrence_features(actual))
    credit_events_before=full.population.credit_events
    checks['yoked_cross_network_return_cannot_revise']=(
        relation_ecology.record_qualified_network(
            full.population,joint,actual,0x740001,1,False)==0 and
        full.population.credit_events==credit_events_before)
    counterfeit_features=list(relation_ecology.network_occurrence_features(actual))
    counterfeit_features[-1]=0xBAD
    counterfeit=full.population.recruit(tuple(counterfeit_features))
    try:
        relation_ecology.record_qualified_network(
            full.population,counterfeit,actual,0x740001,1,True)
        counterfeit_refused=False
    except ValueError as exc:
        counterfeit_refused=str(exc)=='population:relation_network_signature'
    checks['counterfeit_cross_network_occurrence_refuses']=counterfeit_refused
    relation_identity=relation_ecology.record_qualified_network(
        full.population,joint,actual,0x740001,1,True)
    checks['cross_network_credit_is_joint_not_marginal']=(
        relation_identity!=0 and
        full.population.credit_events-credit_events_before==len(joint.sites))
    cue=full.population.activate_signature(language_signature,retain=False)
    retained_before_unfold=len(full.population.occurrences)
    cross_candidates=relation_ecology.unfold_candidates(cue)
    unfolded=[relation_ecology.activate_morphology(full.population,m,False)
              for m in cross_candidates]
    # The same learned relation is bidirectional across opaque morphologies. These
    # synthetic signatures have no raw modality or body provenance, so this is a
    # generic recruitment result, not sensory grounding or a somatic-marker claim.
    # Candidate recruitment cannot manufacture world evidence or credit.
    other_cue=full.population.activate_signature(other_signatures[2],retain=False)
    other_candidates=relation_ecology.unfold_candidates(other_cue)
    language_morphology=relation_ecology.morphology_identity(language_signature)
    retained_after_unfold=len(full.population.occurrences)
    relation_checkpoint=relation_ecology.checkpoint()
    relation_wire=json.dumps(relation_checkpoint,sort_keys=True,separators=(',',':')).encode()
    restored_relation=PopulationRecruitmentEcologyV1.restore(copy.deepcopy(relation_checkpoint))
    checks['language_outer_cross_network_recruitment']=(relation_identity!=0 and len(cross_candidates)==3 and restored_relation.unfold_candidates(cue)==cross_candidates)
    checks['opaque_nonlinguistic_cue_recruits_language_morphology']=(language_morphology in other_candidates and len(other_candidates)==3)
    checks['cross_network_unfold_is_ephemeral']=(retained_before_unfold==retained_after_unfold and len({o.identity for o in unfolded})==len(unfolded))
    serialized_relation=relation_wire.decode()
    language_hex=bytes(full.surface_units[language_evidence.unit].raw).hex()
    checks['cross_network_relation_has_no_surface_payload']=language_hex not in serialized_relation and len(relation_wire)<4096
    relation_ecology.withdraw_source(0x740001)
    checks['cross_network_source_withdrawal_removes_authority']=relation_ecology.unfold_candidates(cue)==()

    full.withdraw_raw_source(SOURCE);withdrawn=full.raw_chunk_lattice_stats(heldout[:4096])
    checks['source_withdrawal_removes_chunk_authority']=withdrawn['positions_with_candidates']==0 and withdrawn['coverage_fraction']==0
    checks['no_tokenizer_expected_output_or_llm']=all(not hasattr(full,n) for n in ('tokenizer','encode_tokens','decode_tokens','expected_output','prompt','answer','think','speak'))

    result={'schema':'0x1.raw-corpus-surface-quantity.v1','pass':all(checks.values()),'checks':checks,
      'package':'alice-prose-v1','package_state':manifest['admissibility'],'training_bytes':len(train_raw),'heldout_bytes':len(heldout),
      'retained_chunk_bytes':retained,'retained_units':len(ids),'persistent_checkpoint_bytes':persistent_wire_bytes,'unfolded_to_persistent_ratio':network['unfolded_total_wire_bytes']/persistent_wire_bytes,'cross_network_relation_bytes':len(relation_wire),'cross_network_candidate_families':len(cross_candidates),'heldout_lattice':lattice,'heldout_network':network,'population_quantity':q,'quantity_sweep':sweep,
      'claim':'RAW_CORPUS_SURFACE_LATTICE_QUANTITY_REFERENCE_ONLY_NOT_LANGUAGE_MASTERY_OR_DIRECT_PARITY',
      'runtime_llm':False,'direct_parity':'NOT_RUN/RED','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_RAW_CORPUS_SURFACE_QUANTITY '+('GREEN' if result['pass'] else 'RED')+' nontoy=1 tokenizer=0 ambiguity=1 quantity=1 direct_parity=RED')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
