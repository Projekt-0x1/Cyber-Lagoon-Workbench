#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>

#include "hardware_native/direct_adult_core.cuh"
namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_resident_relational_network.cuh"
}  // namespace substrate::direct_adult_core

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

constexpr std::uint64_t T0=1001u,T1=1002u,L0=2001u,L1=2002u,X0=3001u,X1=3002u;

struct Config { std::uint64_t t,l,x; };

bool make_member(std::uint64_t logical, std::uint32_t rank,
                 std::uint32_t input, std::uint32_t output,
                 std::uint64_t episode,
                 ResidentRecipeCell* recipe,
                 ResidentRecipeDerivation* derivation,
                 ResidentRecipeOccurrence* occurrence) {
  *recipe={}; recipe->logical_recipe_id=logical; recipe->revision_identity=logical+0x100000u;
  *derivation={}; derivation->logical_recipe_id=recipe->logical_recipe_id;
  derivation->revision_identity=recipe->revision_identity; derivation->generation=rank;
  derivation->port_count=2u;
  derivation->ports[0]={input,ResidentRecipePortDomain::q16_scalar,ResidentRecipePortDirection::input,1u};
  derivation->ports[1]={output,ResidentRecipePortDomain::q16_scalar,ResidentRecipePortDirection::output,1u};
  const std::uint32_t bindings[2]={input,output};
  const bool ok=bind_resident_recipe_occurrence(
      *recipe,*derivation,bindings,2u,episode+logical,episode+logical+100u,
      episode+logical+200u,static_cast<std::uint32_t>(episode&0xffffffffu),
      ResidentOccurrenceLineageKind::actual,
      DirectParticipationAuthority::independent_external,77u,1u,100u,0,occurrence);
  if(ok) occurrence->eligibility_q16=10;
  return ok;
}

bool make_config(const Config& cfg,std::uint64_t episode,
                 ResidentRelationalNetworkClosure* out) {
  ResidentRecipeCell recipes[3]{}; ResidentRecipeDerivation derivations[3]{};
  ResidentRecipeOccurrence occurrences[3]{};
  if(!make_member(cfg.t,1u,10u,20u,episode,&recipes[0],&derivations[0],&occurrences[0]) ||
     !make_member(cfg.l,2u,20u,30u,episode,&recipes[1],&derivations[1],&occurrences[1]) ||
     !make_member(cfg.x,3u,30u,40u,episode,&recipes[2],&derivations[2],&occurrences[2])) return false;
  ResidentOccurrenceCoupling edges[2]{};
  if(!bind_resident_occurrence_coupling(occurrences[0],derivations[0],1u,
                                        occurrences[1],derivations[1],0u,&edges[0]) ||
     !bind_resident_occurrence_coupling(occurrences[1],derivations[1],1u,
                                        occurrences[2],derivations[2],0u,&edges[1])) return false;
  return bind_resident_relational_network_closure(recipes,derivations,occurrences,3u,edges,2u,out);
}

bool recruit_credit(ResidentDevelopmentState* development,
                    const ResidentRelationalNetworkClosure& closure,
                    std::int64_t credit,std::uint32_t tick) {
  ResidentRelationalNetworkSet set{}; set.network_count=1u; set.networked_occurrence_count=3u;
  set.coupling_count=2u; set.networks[0]=closure;
  if(!recruit_resident_relational_network_set(development,set,tick)) return false;
  const auto rid=resident_relational_network_recruitment_identity(closure);
  ResidentRecruitedNetworkCreditPlan plan{};
  if(!plan_resident_recruited_network_credit(development,rid,closure.identity,credit,&plan) ||
     plan.valid==0u) return false;
  return apply_resident_recruited_network_credit(development,plan,tick+1u);
}

Config marginal_winner(const Config (&configs)[4],const std::int64_t (&weights)[4]) {
  std::map<std::pair<int,std::uint64_t>,std::int64_t> score;
  for(int i=0;i<4;++i){score[{0,configs[i].t}]+=weights[i];score[{1,configs[i].l}]+=weights[i];score[{2,configs[i].x}]+=weights[i];}
  auto pick=[&](int slot){std::uint64_t id=0;std::int64_t best=INT64_MIN;bool tied=false;for(const auto& kv:score){if(kv.first.first!=slot)continue;if(kv.second>best){best=kv.second;id=kv.first.second;tied=false;}else if(kv.second==best){tied=true;}}return tied?0u:id;};
  return Config{pick(0),pick(1),pick(2)};
}

bool same(const Config&a,const Config&b){return a.t==b.t&&a.l==b.l&&a.x==b.x;}

} // namespace

int main(){
  const Config W{T0,L0,X1},D1{T1,L1,X0},D2{T1,L1,X1},D3{T1,L0,X0},H{T1,L0,X1};
  const Config lived[4]={W,D1,D2,D3}; const std::int64_t credits[4]={2000,1000,1000,1000};
  ResidentRelationalNetworkClosure cw{},c1{},c2{},c3{},ch{};
  if(!make_config(W,10000,&cw)||!make_config(D1,20000,&c1)||!make_config(D2,30000,&c2)||!make_config(D3,40000,&c3)||!make_config(H,50000,&ch)) return 2;
  const Config marginal=marginal_winner(lived,credits);
  const bool marginal_hybrid=same(marginal,H) && !same(H,W)&&!same(H,D1)&&!same(H,D2)&&!same(H,D3);

  ResidentDevelopmentState development{};
  if(!recruit_credit(&development,cw,credits[0],10u)||!recruit_credit(&development,c1,credits[1],20u)||
     !recruit_credit(&development,c2,credits[2],30u)||!recruit_credit(&development,c3,credits[3],40u)) return 3;
  const auto hybrid_rid=resident_relational_network_recruitment_identity(ch);
  bool hybrid_absent=true;for(std::uint32_t i=0;i<development.recruited_networks.incidence_count;++i)hybrid_absent&=development.recruited_networks.incidences[i].recruitment_identity!=hybrid_rid;

  ResidentRelationalNetworkSet hybrid_only{};hybrid_only.network_count=1u;hybrid_only.networked_occurrence_count=3u;hybrid_only.coupling_count=2u;hybrid_only.networks[0]=ch;
  std::uint64_t selected_n=0,selected_r=0;std::int64_t selected_credit=0;
  const bool hybrid_refused=select_resident_recruited_network(development.recruited_networks,hybrid_only,&selected_n,&selected_r,&selected_credit)&&selected_n==0u&&selected_r==0u&&selected_credit==0;

  ResidentRelationalNetworkSet coactive{};coactive.network_count=2u;coactive.networked_occurrence_count=6u;coactive.coupling_count=4u;coactive.networks[0]=cw;coactive.networks[1]=ch;
  selected_n=selected_r=0u;selected_credit=0;
  const auto wrid=resident_relational_network_recruitment_identity(cw);
  const bool exact_winner=select_resident_recruited_network(development.recruited_networks,coactive,&selected_n,&selected_r,&selected_credit)&&selected_n==cw.identity&&selected_r==wrid&&selected_credit==2000;

  ResidentRecruitedNetworkCreditPlan down{};bool down_ready=plan_resident_recruited_network_credit(&development,wrid,cw.identity,-1000,&down)&&down.valid!=0u;
  if(down_ready)down_ready=apply_resident_recruited_network_credit(&development,down,60u);
  ResidentRelationalNetworkSet tied{};tied.network_count=4u;tied.networked_occurrence_count=12u;tied.coupling_count=8u;tied.networks[0]=cw;tied.networks[1]=c1;tied.networks[2]=c2;tied.networks[3]=c3;
  selected_n=selected_r=0u;selected_credit=0;
  const bool tie_refuses=down_ready&&select_resident_recruited_network(development.recruited_networks,tied,&selected_n,&selected_r,&selected_credit)&&selected_n==0u&&selected_r==0u&&selected_credit==0;

  const bool pass=marginal_hybrid&&hybrid_absent&&hybrid_refused&&exact_winner&&tie_refuses&&development.recruited_networks.incidence_count==4u;
  std::printf("DIRECT_SELECTION_NETWORK_HYBRID %s marginal_hybrid=%u hybrid_absent=%u hybrid_refuse=%u exact_winner=%u negative_reopen=%u incidence=%u adult_runtime=0 direct_structs=1\n",pass?"GREEN":"RED",marginal_hybrid,hybrid_absent,hybrid_refused,exact_winner,tie_refuses,development.recruited_networks.incidence_count);
  return pass?0:1;
}
