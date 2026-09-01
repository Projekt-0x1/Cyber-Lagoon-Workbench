#!/usr/bin/env python3
"""One append-only #1185-style external chronology for the fast mathematical Adult."""
from __future__ import annotations
from dataclasses import dataclass
import ast,copy,functools,hashlib,importlib,json
from pathlib import Path
from autotrans_species_ir_v0 import FoundrySpeciesProgramV0,SpeciesLawV0
from reference_language_mastery_contact_adapter_v1 import CONTACT_DISCOURSE_SURFACE,CONTACT_RELATION,CONTACT_SCENE,CONTACT_SURFACE,CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1,RelationContactV1,SceneContactV1
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_mathematical_adult_workbench_v1 import MathematicalWorkbenchAdultV1
from reference_resident_variable_depth_endogenous_unfolding_v1 import (
    AFFINE,BISIMULATION,POLYNOMIAL,Q,SCHUR,EndogenousNeedV1,ReductionTermV1,
    ResidentProgramStateV1,ResidentReductionNodeV1,
)
from reference_species_language_life_function import birth_language_mastery_adult,source_support_from_species

SCHEMA='cyber-lagoon.life-function-curriculum.v2';CHECKPOINT_SCHEMA='cyber-lagoon.life-function-runtime.v4';MAX_OCCURRENCES=65536
HISTORY_SEED=hashlib.sha256(b'CYBER_LAGOON_LIFE_HISTORY_V3').hexdigest()
ALLOWED_LANES={
    'scene','surface','relation','discourse_surface','utterance','authenticated_utterance','raw_speech_contact','observed_source_action',
    'quiet','body_load','causal_field','resident_world_step','source_withdrawal','language_source_withdrawal','action_consequence',
    'operator_node','operator_need','operator_binding','operator_join','operator_source_withdrawal',
    'public_opportunity','operator_return','relation_basis_edge','relation_basis_opportunity',
    'causal_dialogue_opportunity','partner_causal_dialogue_opportunity','causal_dialogue_return','causal_dialogue_background',
    'endogenous_inquiry_opportunity','endogenous_inquiry_motor_return','endogenous_inquiry_resolution',
    'checkpoint_mark',
}
CANONICAL_TAIL_MODULES_V2=(
    'reference_life_extension_causal_depth_v1',
    'reference_life_extension_causal_discourse_v1',
    'reference_life_extension_causal_depth_plus_v1',
    'reference_life_extension_controllability_resilience_v1',
    'reference_life_extension_somatic_appraisal_language_v1',
    'reference_life_extension_causal_discourse_forms_v2',
    'reference_life_extension_history_matrix_v1',
    'reference_life_extension_open_state_prompt_v1',
    'reference_life_extension_endogenous_state_inquiry_v1',
    'reference_life_extension_relational_productive_surplus_v1',
)
SOURCE_SEMANTIC_ROOTS=(
    'reference_life_function_curriculum_v1.py',
    'reference_mathematical_adult_workbench_v1.py',
    'reference_resident_variable_depth_endogenous_unfolding_v1.py',
    'reference_species_language_life_function.py',
)
C=9001;JOIN=9101

@dataclass(frozen=True)
class LifeCurriculumEventV2:
    sequence:int;lane:str;source:int=0;payload:tuple=()
    def validate(self):
        if self.sequence<=0 or self.lane not in ALLOWED_LANES:raise ValueError('life-curriculum:event')
        if self.lane not in {'quiet','checkpoint_mark'} and self.source<=0:raise ValueError('life-curriculum:source')
        if self.lane in {'surface','discourse_surface','utterance','authenticated_utterance'} and not self.payload:raise ValueError('life-curriculum:surface')
    def document(self):
        def norm(value):
            if isinstance(value,ResidentReductionNodeV1):return {'operator_node':value.row()}
            if isinstance(value,dict):return {str(k):norm(v) for k,v in sorted(value.items())}
            if isinstance(value,(tuple,list)):return [norm(v) for v in value]
            return value
        return {'sequence':self.sequence,'lane':self.lane,'source':self.source,'payload':norm(self.payload)}

@dataclass(frozen=True)
class LifeFunctionCurriculumV2:
    events:tuple[LifeCurriculumEventV2,...]
    schema:str=SCHEMA
    def __post_init__(self):
        if tuple(e.sequence for e in self.events)!=tuple(range(1,len(self.events)+1)):raise ValueError('life-curriculum:chronology')
        for event in self.events:event.validate()
    def canonical_document(self):return {'schema':self.schema,'events':[event.document() for event in self.events]}
    def root(self):return hashlib.sha256(json.dumps(self.canonical_document(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def history_root(self,cursor=None):
        root=HISTORY_SEED
        for event in self.events[:len(self.events) if cursor is None else int(cursor)]:
            root=_extend_history(root,event.document())
        return root
    def mark_cursor(self,name):
        rows=[event.sequence for event in self.events if event.lane=='checkpoint_mark' and event.payload==(str(name),)]
        if len(rows)!=1:raise KeyError(name)
        return rows[0]
    def prefix(self,cursor):return LifeFunctionCurriculumV2(self.events[:int(cursor)])
    def prefix_at_mark(self,name):return self.prefix(self.mark_cursor(name))
    def append(self,events):
        rows=list(self.events);seq=len(rows)
        for event in events:
            seq+=1;rows.append(LifeCurriculumEventV2(seq,event.lane,event.source,event.payload))
        return LifeFunctionCurriculumV2(tuple(rows))

def canonical_developmental_probe_v2(curriculum):
    """Derive the assay from contact incidence; transport tuple order has no authority."""
    features=set();examples=[];lexical={};current=None;before_grounding=True;collect_constructions=False
    grounding=LearnedSurfaceEcologyV1()
    for event in curriculum.events:
        if event.lane=='scene':
            context=int(event.payload[0]);atoms=tuple(map(int,event.payload[1:]))
            current=(context,atoms,int(event.source))
            if before_grounding and context==100:features.update(atoms)
        elif event.lane=='surface' and current is not None and int(event.source)==current[2]:
            context,atoms,source=current;raw=tuple(map(int,event.payload));current=None
            if before_grounding and context==100:
                grounding.observe_scene_surface(0,atoms,raw,source)
            elif collect_constructions and len(atoms)==4:
                located=[]
                for atom in atoms:
                    rows=lexical.get(atom,())
                    if len(rows)!=1:raise ValueError('life-curriculum:probe-lexical')
                    units=next(iter(rows));positions=tuple(i for i in range(len(raw)-len(units)+1) if raw[i:i+len(units)]==units)
                    if len(positions)!=1:raise ValueError('life-curriculum:probe-incidence')
                    located.append((positions[0],positions[0]+len(units),atom))
                located.sort()
                if any(located[i][1]>located[i+1][0] for i in range(len(located)-1)):raise ValueError('life-curriculum:probe-overlap')
                examples.append((context,tuple(row[2] for row in located),source))
        elif event.lane=='checkpoint_mark':
            mark=str(event.payload[0])
            if mark=='grounded':
                before_grounding=False;collect_constructions=True
                lexical={feature:{tuple(grounding.lexeme(feature) or ())} for feature in features}
                if any(not next(iter(rows)) for rows in lexical.values()):raise ValueError('life-curriculum:probe-grounding')
            elif mark=='productive':break
    if len(examples)<2 or len({row[0] for row in examples})!=1:raise ValueError('life-curriculum:construction-probe')
    atoms=tuple(row[1] for row in examples)
    if any(len(row)!=4 for row in atoms):raise ValueError('life-curriculum:construction-arity')
    heldout=(atoms[0][0],atoms[1][1],atoms[0][2],atoms[1][3])
    return examples[0][0],atoms,heldout,tuple(sorted(features))

def _extend_history(root,document):
    h=hashlib.sha256(bytes.fromhex(str(root)))
    h.update(json.dumps(document,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()

def _local_source_semantic_files_v2():
    root=Path(__file__).resolve().parent;seen=set();pending=list(SOURCE_SEMANTIC_ROOTS)
    # Only explicitly admitted canonical tails can change this Adult's source
    # semantics. An experimental filename appearing in the Workbench is not lived
    # contact and must not silently change the canonical individual.
    pending.extend(module+'.py' for module in CANONICAL_TAIL_MODULES_V2)
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=root/name
        if not path.is_file():raise FileNotFoundError(path)
        seen.add(name);tree=ast.parse(path.read_text())
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level==0 and node.module:modules.append(node.module.split('.')[0])
            elif isinstance(node,ast.Import):modules.extend(alias.name.split('.')[0] for alias in node.names)
        for module in modules:
            candidate=module+'.py'
            if candidate not in seen and (root/candidate).is_file():pending.append(candidate)
    return tuple(sorted(seen))


def canonical_species_program_v2(minimum_distinct_sources=2):
    """One content-free Species law; structural reach is resident relation matter."""
    return FoundrySpeciesProgramV0.build((
        SpeciesLawV0('authenticated_external_contact'),
        SpeciesLawV0('source_conditioned_access_evidence',parameters=(('minimum_distinct_sources',int(minimum_distinct_sources)),)),
        SpeciesLawV0('independent_consequence_revision'),
        SpeciesLawV0('contradiction_reopens_access'),
    ),resource_bounds=(('plastic_work_budget',24),))
def _surface_payload(text):return tuple(str(text).encode())
def canonical_tail_builders_v2():
    """Load only explicitly admitted tails, preserving their declared order constraints."""
    found={}
    for module in CANONICAL_TAIL_MODULES_V2:
        loaded=importlib.import_module(module);fn=getattr(loaded,'build',None) or getattr(loaded,'build_extension',None)
        if not callable(fn):raise ValueError(f'life-curriculum:extension-builder:{module}')
        deps=tuple(map(str,getattr(loaded,'LIFE_AFTER',())))
        if module in found or any(not dep or dep==module for dep in deps):raise ValueError(f'life-curriculum:extension-dependency:{module}')
        found[module]=(fn,deps)
    missing=sorted({dep for _fn,deps in found.values() for dep in deps if dep not in found})
    if missing:raise ValueError('life-curriculum:extension-missing:'+','.join(missing))
    ordered=[];pending=set(found)
    while pending:
        ready=sorted(module for module in pending if all(dep in ordered for dep in found[module][1]))
        if not ready:raise ValueError('life-curriculum:extension-cycle')
        for module in ready:ordered.append(module);pending.remove(module)
    return tuple((module,found[module][0]) for module in ordered)
@functools.lru_cache(maxsize=1)
def source_semantics_root_v2():
    root=Path(__file__).resolve().parent;h=hashlib.sha256(b'cyber-lagoon-source-semantics-v2\0')
    for name in _local_source_semantic_files_v2():
        h.update(name.encode());h.update(b'\0');h.update(hashlib.sha256((root/name).read_bytes()).digest())
    return h.hexdigest()
def _operator_nodes_v2(rewired=False):
    """Cross-family exact operator ecology; topology is evidence, never an active path list."""
    lit=ReductionTermV1.literal;q=ReductionTermV1.q16;u=ReductionTermV1.u32_q16
    n1=ResidentReductionNodeV1(0xD101,AFFINE,(lit(2*Q),lit(Q),lit(Q),lit(0)))
    n2=ResidentReductionNodeV1(0xD102,POLYNOMIAL,(q(n1.identity,0),q(n1.identity,1),lit(Q//2),lit(Q)))
    n3=ResidentReductionNodeV1(0xD103,SCHUR,(q(n2.identity,0),lit(Q),lit(Q),lit(0)))
    n4=ResidentReductionNodeV1(0xD104,AFFINE,(q(n3.identity,0),q(n2.identity,1),lit(Q),lit(0)))
    n7=ResidentReductionNodeV1(0xD107,BISIMULATION,(),(1,1),(0,Q))
    n5=ResidentReductionNodeV1(0xD105,POLYNOMIAL,(q(n4.identity,0),q(n4.identity,1),lit(Q),u(n7.identity,0) if rewired else lit(0)))
    n6=ResidentReductionNodeV1(0xD106,SCHUR,(q(n5.identity,2),lit(Q),lit(Q),lit(0)))
    n8=ResidentReductionNodeV1(0xD108,AFFINE,(q(n6.identity,0),u(n7.identity,0),lit(Q),lit(0)))
    n9=ResidentReductionNodeV1(0xD109,POLYNOMIAL,(q(n8.identity,0),q(n8.identity,1),lit(Q//2),lit(Q)))
    alternative=ResidentReductionNodeV1(0xD10A,AFFINE,(lit(Q),lit(Q),lit(Q),lit(0)))
    return (n1,n2,n3,n4,n5,n6,n7,n8,n9),alternative

def operator_curriculum_extension_v2(atoms_by_node,alternative_atoms,rewired=False,include_depth7=False):
    """Ordinary lived evidence up to the currently reachable constructor frontier."""
    events=[]
    def add(lane,source=0,payload=()):events.append(LifeCurriculumEventV2(len(events)+1,lane,int(source),tuple(payload)))
    deep,alternative=_operator_nodes_v2(bool(rewired));active=deep if bool(include_depth7) else deep[:-1]
    for index,node in enumerate((*active,alternative)):
        for witness in range(2):add('operator_node',810000+index*10+witness,(node,))
    for witness in range(2):
        add('operator_need',820000+witness,(active[-1].identity,8,4))
        add('operator_need',820010+witness,(alternative.identity,7,1))
    for index,(node,atoms) in enumerate(zip(active,atoms_by_node)):
        for witness in range(2):add('operator_binding',830000+index*10+witness,(node.identity,C,*atoms))
    for witness in range(2):add('operator_binding',830100+witness,(alternative.identity,C,*alternative_atoms))
    for witness in range(2):add('operator_join',840000+witness,(JOIN,))
    add('quiet',0,(3,))
    return tuple(events)

def operator_depth8_extension_v2(atoms):
    """Probe-only next frontier: one depth-8 relation built on the canonical depth-7 root."""
    events=[]
    def add(lane,source=0,payload=()):events.append(LifeCurriculumEventV2(len(events)+1,lane,int(source),tuple(payload)))
    deep,_alternative=_operator_nodes_v2(False);prior=deep[-1];lit=ReductionTermV1.literal;q=ReductionTermV1.q16
    node=ResidentReductionNodeV1(0xD10B,AFFINE,(q(prior.identity,0),q(prior.identity,1),lit(Q),lit(0)))
    for witness in range(2):add('operator_node',860000+witness,(node,))
    for witness in range(2):add('operator_need',860010+witness,(node.identity,12,4))
    for witness in range(2):add('operator_binding',860020+witness,(node.identity,C,*atoms))
    add('quiet',0,(2,))
    return tuple(events)

def language_guided_world_extension_v2(start):
    """One lived language-to-experiment-to-world-revision history."""
    events=[]
    def add(lane,source=0,payload=()):
        events.append(LifeCurriculumEventV2(
            int(start)+len(events)+1,lane,int(source),tuple(payload)))

    # The propositions arrive independently.  No composed target answer or
    # intervention choice is supplied by the curriculum.
    propositions=(
        (0xA101,'heavy rain soaks the garden'),
        (0xA102,'dark soil holds the water'),
        (0xA103,'bright sunlight warms the greenhouse'),
        (0xA104,'warm air dries the soil'),
        (0xA105,'steady wind closes the vent'),
    )
    scenes={}
    for identity,surface in propositions:
        for source_base in (0xB000,0xB800):
            scenes[identity]=int(start)+len(events)+1
            add('scene',source_base+identity,(100,identity))
            add('surface',source_base+identity,_surface_payload(surface))

    # A first independently settled world relation calibrates testimony.
    first_field=int(start)+len(events)+1
    add('causal_field',0xC101,(
        scenes[0xA101],scenes[0xA105],scenes[0xA102],256))
    for source in (0xC111,0xC112,0xC113):
        for _ in range(4):
            add('resident_world_step',source,
                (first_field,scenes[0xA101],1))
    calibration='Because heavy rain soaks the garden, dark soil holds the water.'
    alternative='dark soil holds the water since heavy rain soaks the garden'
    for source in (0xC121,0xC122,0xC123):
        add('relation',source,(0xCB11,scenes[0xA102],scenes[0xA101]))
        add('discourse_surface',source,_surface_payload(calibration))
    for source in (0xD101,0xD102,0xD101,0xD105,0xD106):
        add('authenticated_utterance',source,_surface_payload(calibration))
    for source in (0xC131,0xC132):
        add('relation',source,(0xCB12,scenes[0xA102],scenes[0xA101]))
        add('discourse_surface',source,_surface_payload(alternative))
    for source in (0xD103,0xD104):
        add('authenticated_utterance',source,_surface_payload(alternative))

    # Balanced prehistory makes the unaided next intervention non-diagnostic.
    # The learned testimony changes only prospective sampling; actual returns
    # remain the sole authority that settles causal credit.
    target_field=int(start)+len(events)+1
    add('causal_field',0xC201,(
        scenes[0xA103],scenes[0xA105],scenes[0xA104],256))
    for block_source,filler_source in (
            (0xC211,0xC2F1),(0xC212,0xC2F2),(0xC213,0xC2F3)):
        for return_source in (block_source,block_source,filler_source,block_source):
            add('resident_world_step',return_source,
                (target_field,scenes[0xA103],1))
    add('authenticated_utterance',0xD101,_surface_payload(
        'warm air dries the soil since bright sunlight warms the greenhouse'))
    for source in (0xC211,0xC212,0xC213):
        add('resident_world_step',source,
            (target_field,scenes[0xA103],1))
    add('source_withdrawal',0xD101)
    add('checkpoint_mark',0,('language_guided_world_learning',))
    return tuple(events)

def correction_repair_extension_v2(start,sunlight_scene,effect_scene):
    """Retract stale evidence, test a spoken alternative, then revise after consequence."""
    events=[]
    def add(lane,source=0,payload=()):
        events.append(LifeCurriculumEventV2(
            int(start)+len(events)+1,lane,int(source),tuple(payload)))

    # Changed source authority removes the old relation; it does not install an
    # expected replacement or activate a privileged correction transition.
    for source in (0xC211,0xC212,0xC213):add('source_withdrawal',source)
    heater_scene=0
    for source_base in (0xB000,0xB800):
        heater_scene=int(start)+len(events)+1
        add('scene',source_base+0xA106,(100,0xA106))
        add('surface',source_base+0xA106,_surface_payload(
            'the heater warms the greenhouse'))

    replacement_field=int(start)+len(events)+1
    add('causal_field',0xC301,(
        heater_scene,int(sunlight_scene),int(effect_scene),256))
    for block_source,filler_source in (
            (0xC311,0xC3F1),(0xC312,0xC3F2),(0xC313,0xC3F3)):
        for return_source in (block_source,block_source,filler_source,block_source):
            add('resident_world_step',return_source,
                (replacement_field,heater_scene,1))

    # A calibrated source changes prospective sampling only. At this mark the
    # alternative is heard, but it is not yet licensed as a public world fact.
    add('authenticated_utterance',0xD102,_surface_payload(
        'warm air dries the soil since the heater warms the greenhouse'))
    add('checkpoint_mark',0,('correction_contact',))
    for source in (0xC311,0xC312,0xC313):
        add('resident_world_step',source,
            (replacement_field,heater_scene,1))
    add('checkpoint_mark',0,('correction_repair',))
    return tuple(events)

def unseen_causal_topic_extension_v1(start):
    """Let the developed Adult acquire one new topology from world consequence."""
    events=[]
    def add(lane,source=0,payload=()):
        events.append(LifeCurriculumEventV2(
            int(start)+len(events)+1,lane,int(source),tuple(payload)))
        return events[-1].sequence

    add('checkpoint_mark',0,('pre_unseen_causal_topic',))
    sunlight=add('scene',0xE101,(100,0xA103))
    drying=add('scene',0xE102,(100,0xA104))
    vent=add('scene',0xE103,(100,0xA105))
    field=add('causal_field',0xE110,(sunlight,drying,vent,256))
    # Three independently sourced intervention blocks are the only authority.
    # The curriculum names the actual world return, never a language answer.
    for source in (0xE111,0xE112,0xE113):
        for _ in range(4):
            add('resident_world_step',source,(field,sunlight,1))
    add('checkpoint_mark',0,('unseen_causal_topic',))
    return tuple(events)

def post_life_lexical_causal_integration_v1(start):
    """Let lived lexical consequences revise both ends of resident world structure."""
    events=[]
    def add(lane,source=0,payload=()):
        events.append(LifeCurriculumEventV2(
            int(start)+len(events)+1,lane,int(source),tuple(payload)))
        return events[-1].sequence

    add('checkpoint_mark',0,('pre_lexical_causal_integration',))
    # These alternate forms first earn independent grounding. Two different
    # referents then expose the same two-port carrier, so no literal separator
    # or observer-authored grammar role has lexical authority.
    for atom,surface,source in (
            (0xA103,'morning sunlight warms the greenhouse',0xE205),
            (0xA105,'steady wind closes the vent',0xE206),
            (0xA103,'sunlight heats the glasshouse',0xE201),
            (0xA103,'sunlight heats the glasshouse',0xE202),
            (0xA105,'wind shuts the vent',0xE203),
            (0xA105,'wind shuts the vent',0xE204)):
        add('scene',source,(100,atom));add('surface',source,_surface_payload(surface))
    for atom,surface,source in (
            (0xA103,'morning sunlight warms the greenhouse aka sunlight heats the glasshouse',0xE211),
            (0xA105,'steady wind closes the vent aka wind shuts the vent',0xE212)):
        add('scene',source,(0xD3F,atom));add('surface',source,_surface_payload(surface))

    # Questions arrive as ordinary authenticated social contact. Their unary
    # construction is learned around the already acquired carrier relation;
    # neither a question opcode nor an expected response enters the Genome.
    add('checkpoint_mark',0,('pre_lexical_clarification_language',))
    add('authenticated_utterance',0xE213,_surface_payload(
        'do you mean morning sunlight warms the greenhouse?'))
    add('authenticated_utterance',0xE214,_surface_payload(
        'do you mean steady wind closes the vent?'))
    add('authenticated_utterance',0xE215,_surface_payload(
        'do you mean steady wind closes the vent?'))
    add('authenticated_utterance',0xE216,_surface_payload(
        'do you mean steady wind closes the vent?'))
    add('checkpoint_mark',0,('lexical_carrier_ready',))

    # Raw social contact nominates two provisional forms. Their occurrence
    # identities, not the strings below, bind later independent consequences.
    sunlight_hypothesis=add('authenticated_utterance',0xE221,
                            _surface_payload('morning sunlight warms the greenhouse aka sunbeams heat the glasshouse'))
    vent_hypothesis=add('authenticated_utterance',0xE222,
                        _surface_payload('steady wind closes the vent aka airflow seals the vent'))
    add('checkpoint_mark',0,('lexical_causal_hypotheses',))
    add('action_consequence',0xE231,(sunlight_hypothesis,1,1))
    add('action_consequence',0xE241,(vent_hypothesis,1,1))
    add('checkpoint_mark',0,('lexical_causal_integration',))
    return tuple(events)

def canonical_life_function_curriculum_v2():
    """Small canonical life: language, quiet/body history and operator evidence."""
    events=[]
    def add(lane,source=0,payload=()):events.append(LifeCurriculumEventV2(len(events)+1,lane,int(source),tuple(payload)))
    def mark(name):add('checkpoint_mark',0,(str(name),))
    def transport_atoms(atoms):return (atoms[2],atoms[0],atoms[3],atoms[1])
    # Authenticated contacts are the curriculum. No exported word list or answer
    # dictionary owns lexical boundaries. Each surface accompanies four ambiguous
    # world participants; cross-situational incidence must discover the spans.
    x=(101,201,301,401);partial=(102,201,301,401);y=(102,202,302,402);z=(101,202,302,401)
    heldout=(x[0],y[1],x[2],y[3])
    surfaces={x:'bright sunlight warms the greenhouse steadily.',
              partial:'warm air warms the greenhouse steadily.',
              y:'warm air dries the soil quickly.',z:'bright sunlight dries the soil steadily.'}
    # Ordinary sparse contacts, not a complete host-enumerated language grid.
    # Every participant has independent positive and negative history, but half
    # of the possible co-occurrences never arrive during development.
    grounding_contacts=(
        (x,'bright sunlight warms the greenhouse steadily.'),
        (y,'quickly, warm air dries the soil.'),
        (z,'dries; the soil; steadily; bright sunlight.'),
        (partial,'warm air warms, steadily, the greenhouse.'),
        ((101,201,302,402),'bright sunlight warms the soil quickly.'),
        ((102,202,301,401),'steadily, warm air dries the greenhouse.'),
        ((101,202,301,402),'the greenhouse dries quickly in bright sunlight.'),
        ((102,201,301,402),'warms; the greenhouse; quickly; warm air.'),
        ((102,201,302,402),'warm air warms the soil quickly.'),
        ((102,202,302,401),'the soil dries steadily in warm air.'),
    )
    for index,(atoms,surface) in enumerate(grounding_contacts):
        source=4100+index
        add('scene',source,(100,*transport_atoms(atoms)))
        add('surface',source,_surface_payload(surface))
    mark('grounded');add('quiet',0,(3,))
    # Distinct constructions are ordinary sequential experience. Their incidence,
    # not observer-authored tuple order or numbered checkpoints, determines acquisition.
    for atoms,source in ((x,5001),(y,5002),(z,5003),(partial,5004)):
        add('scene',source,(C,*transport_atoms(atoms)));add('surface',source,_surface_payload(surfaces[atoms]))
    add('body_load',7001,(4,1<<12));mark('productive')
    # Two independently sourced discourse observations establish one reusable join relation.
    dx=len(events)+1;add('scene',6101,(C,*transport_atoms(x)));add('surface',6101,_surface_payload(surfaces[x]));dy=len(events)+1;add('scene',6102,(C,*transport_atoms(y)));add('surface',6102,_surface_payload(surfaces[y]));add('relation',6201,(JOIN,dx,dy));add('discourse_surface',6201,_surface_payload(surfaces[x]+' '+surfaces[y]))
    dy2=len(events)+1;add('scene',6103,(C,*transport_atoms(y)));add('surface',6103,_surface_payload(surfaces[y]));dz=len(events)+1;add('scene',6104,(C,*transport_atoms(z)));add('surface',6104,_surface_payload(surfaces[z]));add('relation',6202,(JOIN,dy2,dz));add('discourse_surface',6202,_surface_payload(surfaces[y]+' '+surfaces[z]));mark('discourse')
    # Cross-family nonlinguistic operator evidence lives in the same Adult checkpoint.
    # Supported structure is admissible directly; no scalar constructor horizon exists.
    operator_atoms=(x,y,z,heldout,x,y,z,heldout,x)
    for event in operator_curriculum_extension_v2(operator_atoms,y,False,True):add(event.lane,event.source,event.payload)
    mark('operator_ready')
    action=len(events)+1;add('public_opportunity',850000,())
    add('operator_return',850001,(action,Q,1,1));mark('operator_public')
    for event in operator_depth8_extension_v2(y):add(event.lane,event.source,event.payload)
    mark('operator_extended')

    # Primitive exact relations connect nine already-constructible proposition spaces.
    # No new word, clause surface or discourse surface is taught here. Quiet recurrence
    # compiles relation-of-relations and later public language projects that resident path.
    basis_atoms=(x,y,z,heldout,(102,201,302,401),(102,202,301,401),
                 (101,201,302,402),(102,201,301,402),(101,202,302,402))
    basis_scenes=[]
    for index,atoms in enumerate(basis_atoms):
        seq=len(events)+1;add('scene',870000+index,(C,*atoms));basis_scenes.append(seq)
    for index,(left,right) in enumerate(zip(basis_scenes,basis_scenes[1:])):
        add('relation_basis_edge',871000+index,(left,right,Q,(index+1)*(Q//8)))
    mark('relation_primitives')
    add('quiet',0,(7,));mark('self_derived_basis')
    scrambled=(basis_scenes[4],basis_scenes[0],basis_scenes[8],basis_scenes[2],basis_scenes[6],basis_scenes[1],basis_scenes[7],basis_scenes[3],basis_scenes[5])
    add('relation_basis_opportunity',872000,scrambled);mark('self_derived_public')
    for event in language_guided_world_extension_v2(len(events)):
        add(event.lane,event.source,event.payload)

    # The same life keeps growing. Later lexical revision must change only language
    # realization; retained nonlinguistic relation structure remains on stable concept spaces.
    old_subject_sources={int(event.source) for event in events
                         if event.lane=='scene' and x[0] in tuple(map(int,event.payload[1:]))}
    for source in (7601,7602):
        add('scene',source,(100,x[0]));add('surface',source,_surface_payload('morning sunlight'))
    for source in sorted(old_subject_sources):add('language_source_withdrawal',source,())
    mark('lexical_revision')
    add('relation_basis_opportunity',873000,scrambled);mark('lexical_revision_public')

    # Later ordinary contact revises the wording of an already world-causal
    # proposition. World evidence and causal topology stay resident and unchanged.
    world_cause=0xA103
    for source in (7701,7702):
        add('scene',source,(100,world_cause));add('surface',source,_surface_payload('morning sunlight warms the greenhouse'))
    for source in (0xB000+world_cause,0xB800+world_cause):add('language_source_withdrawal',source,())
    mark('world_lexical_revision')
    def latest_scene(atom):
        return next(event.sequence for event in reversed(events)
                    if event.lane=='scene' and event.payload==(100,int(atom)))
    for event in correction_repair_extension_v2(
            len(events),latest_scene(0xA103),latest_scene(0xA104)):
        add(event.lane,event.source,event.payload)
    for _module,builder in canonical_tail_builders_v2():
        for event in tuple(builder(len(events))):add(event.lane,event.source,event.payload)
    for event in unseen_causal_topic_extension_v1(len(events)):
        add(event.lane,event.source,event.payload)
    for event in post_life_lexical_causal_integration_v1(len(events)):
        add(event.lane,event.source,event.payload)
    return LifeFunctionCurriculumV2(tuple(events))

class ReferenceLifeFunctionRuntimeV2:
    """One birth and one continuing heterogeneous mathematical Adult chronology."""
    def __init__(self,program:FoundrySpeciesProgramV0,adult=None,cursor=0,occurrences=None,history_root=None,marks=None,transport=None):
        program.validate();self.program=program
        if adult is None:
            language_adult=birth_language_mastery_adult(program);operators=ResidentProgramStateV1(source_support_from_species(program));adult=MathematicalWorkbenchAdultV1(language_adult,operators)
        self.adult=adult;self.contact=LanguageMasteryContactAdapterV1(self.adult.language_adult);self.cursor=int(cursor);self.occurrences=dict(occurrences or {});self._history_root=str(history_root or HISTORY_SEED);self.marks={str(k):int(v) for k,v in dict(marks or {}).items()}
        if transport is not None:self._restore_transport(transport)
    def _transport_checkpoint(self):
        return {'scenes':[{'identity':row.identity,'context':row.context,'atoms':list(row.atoms),'source':row.source} for row in sorted(self.contact.scenes.values(),key=lambda row:row.identity)],'relations':[{'identity':row.identity,'context':row.context,'scenes':list(row.scenes),'source':row.source} for row in sorted(self.contact.relations.values(),key=lambda row:row.identity)],'current_scene':self.contact.current_scene,'current_relation':self.contact.current_relation,'next_identity':self.contact.next_identity}
    def _restore_transport(self,data):
        self.contact.scenes={int(row['identity']):SceneContactV1(int(row['identity']),int(row['context']),tuple(map(int,row['atoms'])),int(row['source'])) for row in data.get('scenes',())}
        self.contact.relations={int(row['identity']):RelationContactV1(int(row['identity']),int(row['context']),tuple(map(int,row['scenes'])),int(row['source'])) for row in data.get('relations',())}
        self.contact.current_scene=int(data.get('current_scene',0));self.contact.current_relation=int(data.get('current_relation',0));self.contact.next_identity=int(data.get('next_identity',1))
        if self.contact.next_identity<=max((0,*self.contact.scenes,*self.contact.relations)):raise ValueError('life-curriculum:transport')
    def _language_leaf(self,scene_sequence):
        identity=int(self.occurrences.get(int(scene_sequence),0))
        scene=self.contact.scenes.get(identity)
        if scene is None:raise ValueError('life-curriculum:scene-reference')
        return self.adult.language_adult.leaf(scene.context,scene.atoms)
    def _remember(self,sequence,identity):
        identity=int(identity)
        if identity<=0:raise ValueError('life-curriculum:occurrence')
        if int(sequence) not in self.occurrences and len(self.occurrences)>=MAX_OCCURRENCES:raise RuntimeError('life-curriculum:occurrence-capacity')
        self.occurrences[int(sequence)]=identity
    def _bound_contact_scene(self,sequence):
        """Rematerialize one unique scene from either flat or nested ingress."""
        identity=int(self.occurrences.get(int(sequence),0));scene=self.contact.scenes.get(identity)
        if scene is not None:return scene
        nested=tuple(self.contact.nested_scenes.get(identity,()))
        bindings={(int(row.context),tuple(map(int,row.atoms))) for row in nested}
        if len(bindings)!=1:return None
        context,atoms=next(iter(bindings));return SceneContactV1(identity,context,atoms,0)
    def apply(self,event:LifeCurriculumEventV2):
        event.validate()
        if event.sequence!=self.cursor+1:raise ValueError('life-curriculum:sequence')
        lane=event.lane;source=event.source;payload=event.payload;result=None
        if lane=='scene':
            result=self.contact.contact(CONTACT_SCENE,tuple(map(int,payload)),source)
            if result:self._remember(event.sequence,result)
        elif lane=='surface':result=self.contact.contact(CONTACT_SURFACE,tuple(map(int,payload)),source)
        elif lane=='relation':
            if len(payload)<3:raise ValueError('life-curriculum:relation-reference')
            context,*sequences=map(int,payload);scenes=tuple(int(self.occurrences.get(sequence,0)) for sequence in sequences)
            if context<=0 or min(scenes)<=0:raise ValueError('life-curriculum:relation-reference')
            result=self.contact.contact(CONTACT_RELATION,(context,*scenes),source)
            if result:
                self._remember(event.sequence,result)
                relation_scenes=tuple(self.contact.scenes.get(int(identity)) for identity in scenes)
                if len(relation_scenes)==2 and all(relation_scenes):
                    left_scene,right_scene=relation_scenes
                    self.adult.observe_context_affordance(left_scene.context,left_scene.atoms,right_scene.context,source)
        elif lane=='discourse_surface':result=self.contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(map(int,payload)),source)
        elif lane=='utterance':result=self.contact.contact(CONTACT_UTTERANCE,tuple(map(int,payload)),source)
        elif lane=='raw_speech_contact':
            result=self.adult.language_action_affordances.observe_language(payload,source,event.sequence)
            if not result:raise ValueError('life-curriculum:raw-speech-contact-refused')
        elif lane=='authenticated_utterance':
            # Body ingress authenticates the transport occurrence. Testimony/source identity
            # remains the separate CONTACT_UTTERANCE provenance coordinate below.
            body_source='life-curriculum-authenticated-utterance'
            commitment=hashlib.sha256(json.dumps(event.document(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.adult.language_adult.settle_body_ingress(body_source,event.sequence,commitment,0)
            # Competing interpretations must be inspected before this contact
            # changes their support.  Otherwise perception settles the very
            # ambiguity that should recruit an information-seeking action.
            surface,receipt=self.adult.externalize_lexeme_clarification(
                payload,source,source)
            result=self.contact.contact(
                CONTACT_UTTERANCE,tuple(map(int,payload)),source,
                body_credentials=(body_source,event.sequence,commitment))
            self.adult.observe_authenticated_causal_dialogue_contact(payload,source)
            if receipt is None:
                surface,receipt=self.adult.externalize_lexeme_hypothesis(
                    result,source,source)
            if receipt is not None:result=receipt.identity
            if result:self._remember(event.sequence,result)
        elif lane=='quiet':
            ticks=max(0,int(payload[0] if payload else 1))
            for _ in range(ticks):self.adult.language_adult.internal_tick();self.adult.resident_silent_wave()
            result=ticks
        elif lane=='body_load':
            work,load=map(int,payload);reaff=hashlib.sha256(json.dumps(event.document(),sort_keys=True).encode()).hexdigest()
            self.adult.language_adult.settle_body_ingress(f'life-curriculum-{source}',event.sequence,reaff,load);result=self.adult.operators.body_contact(work,event.sequence,True);self.adult.reset_operator_transient()
        elif lane=='observed_source_action':
            if len(payload)!=1:raise ValueError('life-curriculum:observed-source-action')
            result=self.adult.language_action_affordances.observe_action(int(payload[0]),source,event.sequence)
            if not result:raise ValueError('life-curriculum:observed-source-action-refused')
        elif lane=='causal_field':
            cause_sequence,rival_sequence,effect_sequence,horizon=map(int,payload)
            cause=self._language_leaf(cause_sequence);rival=self._language_leaf(rival_sequence);effect=self._language_leaf(effect_sequence)
            result=self.adult.language_adult.world_causal_learning.participate(
                (cause.identity,rival.identity),effect.identity,horizon)
            self._remember(event.sequence,result)
        elif lane=='resident_world_step':
            field_sequence,true_cause_sequence,independent=map(int,payload)
            receipt=int(self.occurrences.get(field_sequence,0));learner=self.adult.language_adult.world_causal_learning
            true_cause=int(self._language_leaf(true_cause_sequence).identity)
            binding=learner.bindings.get(receipt)
            if binding is None or true_cause not in binding.causes:raise ValueError('life-curriculum:causal-field-reference')
            true_slot=int(binding.slots[binding.causes.index(true_cause)])
            nomination,occurrence=learner.nominate_intervention(receipt,source=source)
            effect=1 if true_slot in set(map(int,nomination.coalition)) else 0
            result=learner.settle_intervention(
                nomination,occurrence,source,effect,bool(independent))
            if learner.complete_source_blocks(receipt)>=3:
                result=learner.resolve(receipt)
                factor=int(learner.preferred_factor(self.adult.language_adult))
                if result is not None and factor:
                    learner.materialize_program(self.adult.language_adult,receipt,factor)
        elif lane=='source_withdrawal':
            learner=self.adult.language_adult.world_causal_learning
            # The source is the causal authority named by the lived event. Its
            # raw consequence evidence and authenticated language projection
            # lose authority together; neither is retained as compatibility state.
            learner.withdraw_source(source)
            result=learner.withdraw_source(
                learner.testimony_source(source))
        elif lane=='language_source_withdrawal':
            self.adult.language_adult.language.withdraw_source(source);result=True
        elif lane=='action_consequence':
            if len(payload)!=3:raise ValueError('life-curriculum:action-consequence')
            action_sequence,outcome,independent=map(int,payload)
            identity=int(self.occurrences.get(action_sequence,0))
            if identity<=0:raise ValueError('life-curriculum:action-reference')
            result=self.settle_contact_consequence(
                identity,source,outcome,0,bool(independent))
            if not result:raise ValueError('life-curriculum:action-consequence-refused')
        elif lane=='operator_node':
            node=payload[0] if isinstance(payload[0],ResidentReductionNodeV1) else ResidentReductionNodeV1.from_row(payload[0]);self.adult.operators.observe_node(node,source);self.adult.reset_operator_transient();result=node.identity
        elif lane=='operator_need':
            root,priority,minimum_work=map(int,payload);self.adult.operators.observe_need(EndogenousNeedV1(root,priority,minimum_work),source);self.adult.reset_operator_transient();result=root
        elif lane=='operator_binding':
            node,context,*atoms=map(int,payload);result=self.adult.observe_operator_binding(node,context,atoms,source)
        elif lane=='operator_join':result=self.adult.observe_operator_join_context(int(payload[0]),source)
        elif lane=='operator_source_withdrawal':
            self.adult.operators.withdraw_source(source);self.adult.reset_operator_transient();result=True
        elif lane=='public_opportunity':
            winner=self.adult.operator_run_until_settled();surface,receipt=self.adult.externalize_operator() if winner is not None else (None,None)
            if receipt is not None:self._remember(event.sequence,receipt.identity)
            result=(surface,0 if receipt is None else int(receipt.identity))
        elif lane=='operator_return':
            if len(payload)!=4:raise ValueError('life-curriculum:operator-return')
            action_sequence,effect,independent,controllable=map(int,payload);action_identity=int(self.occurrences.get(action_sequence,0));receipt=self.adult.pending_operator_actions.get(action_identity)
            if receipt is None:raise ValueError('life-curriculum:operator-action-reference')
            result=self.adult.settle_operator_consequence(receipt,source,effect,bool(independent),bool(controllable),False)
        elif lane=='relation_basis_edge':
            if len(payload) not in (4,5):raise ValueError('life-curriculum:relation-basis-edge')
            left_sequence,right_sequence,gain,offset=map(int,payload[:4]);relation_sequence=int(payload[4]) if len(payload)==5 else 0
            left_id=int(self.occurrences.get(left_sequence,0));right_id=int(self.occurrences.get(right_sequence,0))
            left_scene=self.contact.scenes.get(left_id);right_scene=self.contact.scenes.get(right_id)
            if left_scene is None or right_scene is None:raise ValueError('life-curriculum:relation-basis-scene')
            language_context=0
            if relation_sequence:
                relation=self.contact.relations.get(int(self.occurrences.get(relation_sequence,0)))
                if relation is None or relation.scenes!=(left_id,right_id):raise ValueError('life-curriculum:relation-basis-relation-reference')
                language_context=int(relation.context)
            left_space=self.adult.observe_relation_language_space(left_scene.context,left_scene.atoms);right_space=self.adult.observe_relation_language_space(right_scene.context,right_scene.atoms)
            if min(left_space,right_space)<=0:raise ValueError('life-curriculum:relation-basis-language-space')
            result=self.adult.observe_relation_basis_edge(left_space,right_space,(gain,offset),source,language_context)
        elif lane=='relation_basis_opportunity':
            if payload:
                spaces=[]
                for sequence in map(int,payload):
                    scene=self.contact.scenes.get(int(self.occurrences.get(sequence,0)))
                    if scene is None:raise ValueError('life-curriculum:relation-basis-opportunity-scene')
                    sid=self.adult.observe_relation_language_space(scene.context,scene.atoms)
                    if sid<=0:raise ValueError('life-curriculum:relation-basis-opportunity-space')
                    spaces.append(sid)
                surface,receipt=self.adult.externalize_relation_frontier(tuple(spaces))
            else:surface,receipt=self.adult.externalize_relation_basis()
            if receipt is not None:self._remember(event.sequence,receipt.identity)
            result=(surface,0 if receipt is None else int(receipt.identity))
        elif lane=='causal_dialogue_opportunity':
            if len(payload)!=1:raise ValueError('life-curriculum:causal-dialogue-scene-reference')
            scene=self._bound_contact_scene(payload[0])
            if scene is None:raise ValueError('life-curriculum:causal-dialogue-scene')
            leaf=self.adult.language_adult.leaf(scene.context,scene.atoms)
            surface,receipt=self.adult.externalize_causal_component(leaf.identity,source,0)
            if receipt is None:raise ValueError('life-curriculum:causal-dialogue-action')
            self._remember(event.sequence,receipt.identity);result=(surface,receipt.identity)
        elif lane=='partner_causal_dialogue_opportunity':
            if len(payload)!=1:raise ValueError('life-curriculum:partner-causal-dialogue-scene-reference')
            scene=self._bound_contact_scene(payload[0])
            if scene is None:raise ValueError('life-curriculum:partner-causal-dialogue-scene')
            leaf=self.adult.language_adult.leaf(scene.context,scene.atoms)
            surface,receipt=self.adult.externalize_causal_component(leaf.identity,source,source)
            if receipt is None:raise ValueError('life-curriculum:partner-causal-dialogue-action')
            self._remember(event.sequence,receipt.identity);result=(surface,receipt.identity)
        elif lane=='endogenous_inquiry_opportunity':
            if len(payload)!=1:raise ValueError('life-curriculum:endogenous-inquiry-channel')
            surface,receipt=self.adult.externalize_endogenous_inquiry(source,int(payload[0]))
            if receipt is not None:self._remember(event.sequence,receipt.identity)
            result=(surface,0 if receipt is None else int(receipt.identity))
        elif lane=='endogenous_inquiry_motor_return':
            if len(payload)!=1:raise ValueError('life-curriculum:endogenous-inquiry-action-reference')
            identity=int(self.occurrences.get(int(payload[0]),0));receipt=self.adult.pending_endogenous_inquiry_actions.get(identity)
            if receipt is None:raise ValueError('life-curriculum:endogenous-inquiry-action')
            result=self.adult.settle_endogenous_inquiry_motor_return(receipt,source,True)
            if not result:raise ValueError('life-curriculum:endogenous-inquiry-motor-return-refused')
        elif lane=='endogenous_inquiry_resolution':
            if len(payload)!=1:raise ValueError('life-curriculum:endogenous-inquiry-action-reference')
            identity=int(self.occurrences.get(int(payload[0]),0));receipt=self.adult.pending_endogenous_inquiry_actions.get(identity)
            if receipt is None:raise ValueError('life-curriculum:endogenous-inquiry-action')
            result=self.adult.settle_endogenous_inquiry_resolution(receipt,source)
            if not result:raise ValueError('life-curriculum:endogenous-inquiry-resolution-refused')
        elif lane=='causal_dialogue_return':
            if len(payload)!=4:raise ValueError('life-curriculum:causal-dialogue-return')
            action_sequence,outcome,somatic,independent=map(int,payload);identity=int(self.occurrences.get(action_sequence,0));receipt=self.adult.pending_causal_dialogue_actions.get(identity)
            if receipt is None:raise ValueError('life-curriculum:causal-dialogue-action-reference')
            result=self.adult.settle_causal_dialogue_return(receipt,source,outcome,somatic,bool(independent))
            if not result:raise ValueError('life-curriculum:causal-dialogue-return-refused')
        elif lane=='causal_dialogue_background':
            if len(payload)!=2:raise ValueError('life-curriculum:causal-dialogue-background')
            action_sequence,outcome_occurs=map(int,payload);identity=int(self.occurrences.get(action_sequence,0))
            result=self.adult.observe_causal_dialogue_background(identity,source,bool(outcome_occurs))
            if not result:raise ValueError('life-curriculum:causal-dialogue-background-refused')
        elif lane=='checkpoint_mark':
            if len(payload)!=1 or not isinstance(payload[0],str) or not payload[0] or payload[0] in self.marks:raise ValueError('life-curriculum:checkpoint-mark')
            self.marks[payload[0]]=event.sequence;result=event.sequence
        self.cursor=event.sequence;self._history_root=_extend_history(self._history_root,event.document());return result
    def run(self,curriculum:LifeFunctionCurriculumV2):
        if self.cursor>len(curriculum.events):raise ValueError('life-curriculum:cursor')
        if curriculum.history_root(self.cursor)!=self._history_root:raise ValueError('life-curriculum:history-prefix')
        for event in curriculum.events[self.cursor:]:self.apply(event)
        return self
    def history_root(self):return self._history_root
    def fork_for_probe(self):
        """Fast isolated observer branch; durable equivalence is still proved by checkpoint restore."""
        out=copy.deepcopy(self)
        if out.adult is self.adult or out.contact is self.contact or out.contact.adult is not out.adult.language_adult:
            raise RuntimeError('life-curriculum:probe-fork-alias')
        return out
    @staticmethod
    def _emit_language_choice(adult,chosen):
        """Complete one resident language motor trajectory with exact reafference."""
        if not chosen:return b''
        expression=adult.expression(chosen);out=bytearray()
        while (step:=expression.emit()) is not None:
            out.append(step.value)
            if not expression.reafference(step,step.value):
                raise RuntimeError('life-curriculum:motor-reafference')
        return bytes(out)
    def contact_utterance(self,raw,source,channel=0):
        """Let current contact open one resident, evidence-certified public opportunity.

        The caller supplies bytes and physical provenance only.  Learned inversion
        chooses the bound scene; exact current causal evidence either generates a
        public trajectory or the Adult remains silent.  There is no host question,
        answer type, topic route, or expected surface.
        """
        raw=bytes(raw);source=int(source);channel=max(0,int(channel))
        if not raw or source<=0:return b'',0
        current_tick=int(self.adult.language_adult._tick)
        continuation=tuple((receipt,effect) for receipt,effect,tick in
                           self.adult.last_causal_dialogue_contact_continuations
                           if int(tick)==current_tick)
        continuation_support=tuple(self.adult.last_causal_dialogue_continuation_support)
        self.adult.last_causal_dialogue_contact_continuations=()
        self.adult.last_causal_dialogue_continuation_support=()
        self.adult.language_adult._advance()
        self.adult.settle_reafferenced_lexeme_hypothesis_contact(
            raw,source,channel)
        surface,receipt=self.adult.externalize_lexeme_clarification(
            raw,source,channel)
        identity=int(self.contact.contact(CONTACT_UTTERANCE,tuple(raw),source,channel))
        if receipt is None:
            surface,receipt=self.adult.externalize_lexeme_hypothesis(
                identity,source,channel)
        if receipt is not None:return bytes(surface),int(receipt.identity)
        frontier=self.adult.causal_continuation_frontier(continuation)
        if frontier:
            effect,resolved_receipts=frontier
            surface,receipt=self.adult.externalize_causal_component(
                effect,source,channel,resolved_receipts)
            if receipt is not None and continuation_support:
                self.adult.begin_causal_continuation(receipt,continuation_support)
            return bytes(surface or b''),0 if receipt is None else int(receipt.identity)
        scene=self.contact.scenes.get(identity)
        if scene is None:
            nested=tuple(self.contact.nested_scenes.get(identity,()))
            bindings={(int(row.context),tuple(map(int,row.atoms))) for row in nested}
            if len(bindings)==1:
                context,atoms=next(iter(bindings))
            else:
                atoms=tuple(int(atom) for row in nested for atom in row.atoms)
                if (len(nested)<2 or len(bindings)!=len(nested)
                        or not atoms):return b'',0
                combined=self.adult.language_adult.unique_leaf_for_concepts(atoms)
                if combined is None:return b'',0
                signature=self.adult.language_adult.leaf_signature(combined.identity)
                if signature is None:return b'',0
                context,atoms=signature
            scene=SceneContactV1(identity,context,atoms,source)
        try:leaf=self.adult.language_adult.leaf(scene.context,scene.atoms)
        except RuntimeError:return b'',0
        surface,receipt=self.adult.externalize_contact_affordance(
            scene.context,scene.atoms,leaf.identity,source,channel)
        return bytes(surface or b''),0 if receipt is None else int(receipt.identity)
    def settle_contact_consequence(self,action_identity,source,outcome_q16,somatic_q16=0,independent=True):
        """Bind an external return to one exact resident public-contact action."""
        identity=int(action_identity)
        receipt=self.adult.pending_causal_dialogue_actions.get(identity)
        if receipt is not None:
            return self.adult.settle_causal_dialogue_return(
                receipt,source,outcome_q16,somatic_q16,independent)
        receipt=self.adult.pending_endogenous_inquiry_actions.get(identity)
        if receipt is not None and self.adult.lexical_hypothesis_for_inquiry(receipt) is not None:
            return self.adult.settle_lexeme_hypothesis_return(
                receipt,source,outcome_q16,independent)
        receipt=self.adult.pending_context_affordance_actions.get(identity)
        return False if receipt is None else self.adult.settle_context_affordance_return(
            receipt,source)
    def observe_contact_background(self,action_identity,source,outcome_occurs=False):
        """Record the matched no-action opportunity for one settled public act."""
        return self.adult.observe_causal_dialogue_background(
            int(action_identity),source,outcome_occurs)
    def quiet_public_opportunity(self,source,channel=0):
        """Advance the same organism without contact and transport any chosen action."""
        source=int(source);channel=max(0,int(channel))
        surface,_receipt=self.adult.externalize_pending_causal_continuation(
            source,channel)
        if surface:return bytes(surface)
        chosen=self.adult.language_adult.internal_tick()
        self.adult.resident_silent_wave()
        if chosen:return self._emit_language_choice(self.adult.language_adult,chosen)
        surface,_receipt=self.adult.externalize_endogenous_inquiry(source,channel)
        return bytes(surface or b'')
    def internal_work_pending(self,channel=0):
        return (self.adult.causal_continuation_work_pending(channel)
                or self.adult.endogenous_inquiry_work_pending(channel)
                or self.adult.language_adult.internal_work_pending())
    def checkpoint(self):return {'schema':CHECKPOINT_SCHEMA,'species_root':self.program.root(),'cursor':self.cursor,'history_root':self.history_root(),'transport':self._transport_checkpoint(),'occurrences':[[k,v] for k,v in sorted(self.occurrences.items())],'marks':[[k,v] for k,v in sorted(self.marks.items())],'adult':self.adult.checkpoint()}
    @classmethod
    def restore(cls,program,data):
        if data.get('schema')!=CHECKPOINT_SCHEMA or data.get('species_root')!=program.root():raise ValueError('life-curriculum:checkpoint')
        root=str(data.get('history_root',''))
        if len(root)!=64:raise ValueError('life-curriculum:history-root')
        adult=MathematicalWorkbenchAdultV1.restore(copy.deepcopy(data['adult']));occ={int(k):int(v) for k,v in data.get('occurrences',())}
        return cls(program,adult,int(data.get('cursor',0)),occ,root,{str(k):int(v) for k,v in data.get('marks',())},copy.deepcopy(data.get('transport',{})))
