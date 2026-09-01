#include <cstdio>
#include "hardware_native/direct_causal_program.cuh"
using namespace substrate::direct_causal_program;
static Program make_program(std::uint64_t id,unsigned depth,unsigned steps){Program p{};p.identity=id;p.depth=depth;p.step_count=steps;p.initiation_participation_identity=id+0x1000;p.initiation_expiry_tick=999;return p;}
int main(){
 ProgramBank b{};std::uint64_t ev=0;
 auto* old=admit_program(&b,make_program(0x79,1,5),&ev);for(int i=0;i<3;++i){observe_use(&old->profile,7,kQ16One/2);observe_return(&old->profile,1,kQ16One,0,true);observe_control(&old->profile,true,true);}
 auto* avoid=admit_program(&b,make_program(0xBAD,0,2),&ev);for(int i=0;i<2;++i){observe_use(&avoid->profile,2,kQ16One/8);observe_return(&avoid->profile,2,-kQ16One,0,true);observe_control(&avoid->profile,true,true);}
 for(unsigned i=0;i<14;++i){auto* f=admit_program(&b,make_program(0x1000+i,0,1),&ev);observe_use(&f->profile,1,kQ16One/32);}
 auto* newer=admit_program(&b,make_program(0x118,2,7),&ev);for(int i=0;i<2;++i){observe_use(&newer->profile,10,3*kQ16One/4);observe_return(&newer->profile,1,3*kQ16One/2,0,true);observe_control(&newer->profile,true,true);}
 const bool old_ok=find_program(&b,0x79)!=nullptr;const bool avoid_ok=find_program(&b,0xBAD)!=nullptr;const bool new_ok=find_program(&b,0x118)!=nullptr;const bool victim_ok=ev>=0x1000&&ev<0x100e;const bool green=b.count==16&&b.evictions==1&&old_ok&&avoid_ok&&new_ok&&victim_ok;
 std::printf("FOUNDRY_DIRECT_CAUSAL_PROGRAM_CAPACITY_%s capacity=%u evictions=%u victim=%llu old_bytes=79 old_depth=1 new_bytes=118 new_depth=2 avoid=%u\n",green?"GREEN":"RED",b.count,b.evictions,(unsigned long long)ev,avoid_ok?1u:0u);return green?0:1;
}
