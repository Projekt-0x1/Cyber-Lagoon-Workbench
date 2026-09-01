#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hardware_native/direct_adult_core.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

static void ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA %s: %s\n", where, cudaGetErrorString(e));
    std::exit(20);
  }
}

static bool seed_rank(ResidentRecipeCell* cell, ResidentRecipeDerivation* der,
                      std::uint64_t logical, std::uint64_t rev, std::uint64_t generation) {
  *cell = {};
  cell->logical_recipe_id = logical;
  cell->revision_identity = rev;
  *der = {};
  der->logical_recipe_id = logical;
  der->revision_identity = rev;
  der->generation = generation;
  der->port_count = 2u;
  der->ports[0] = ResidentRecipePort{
      10u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::input, 1u};
  der->ports[1] = ResidentRecipePort{
      20u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::output, 1u};
  return true;
}

static bool bind_live(const ResidentRecipeCell& cell, const ResidentRecipeDerivation& der,
                      const std::uint32_t* vars, std::uint64_t oid,
                      ResidentRecipeOccurrence* out) {
  return bind_resident_recipe_occurrence(
      cell, der, vars, 2u, oid, oid + 1000u, 77u, 1u,
      ResidentOccurrenceLineageKind::actual,
      DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, out);
}

static bool boundary_has(const ResidentRelationalNetworkClosure& closure, std::uint32_t var,
                         ResidentRecipePortDirection direction) {
  for (std::uint16_t i = 0; i < closure.boundary_count; ++i)
    if (closure.boundary[i].variable_identity == var &&
        closure.boundary[i].direction == direction)
      return true;
  return false;
}

static bool same_boundary(const ResidentRelationalNetworkClosure& a,
                          const ResidentRelationalNetworkClosure& b) {
  if (a.boundary_count != b.boundary_count) return false;
  for (std::uint16_t i = 0; i < a.boundary_count; ++i)
    if (a.boundary[i].variable_identity != b.boundary[i].variable_identity ||
        a.boundary[i].direction != b.boundary[i].direction ||
        a.boundary[i].occurrence_identity != b.boundary[i].occurrence_identity)
      return false;
  return true;
}

__global__ void bind_n(const ResidentRecipeCell* rec, const ResidentRecipeDerivation* der,
                       const ResidentRecipeOccurrence* occ, std::uint32_t n,
                       const ResidentOccurrenceCoupling* edges, std::uint32_t en,
                       ResidentRelationalNetworkClosure* out, int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  *ok = bind_resident_relational_network_closure(rec, der, occ, n, edges, en, out) ? 1 : 0;
}

struct GpuN {
  ResidentRecipeCell* rec;
  ResidentRecipeDerivation* der;
  ResidentRecipeOccurrence* occ;
  ResidentOccurrenceCoupling* edge;
  ResidentRelationalNetworkClosure* cl;
  int* ok;
};

static bool gpu_bind(GpuN& g, const ResidentRecipeCell* rec, std::uint32_t n,
                     const ResidentRecipeDerivation* der,
                     const ResidentRecipeOccurrence* occ,
                     const ResidentOccurrenceCoupling* edges, std::uint32_t en,
                     ResidentRelationalNetworkClosure* out) {
  ck(cudaMemcpy(g.rec, rec, n * sizeof(*rec), cudaMemcpyHostToDevice), "h2d rec");
  ck(cudaMemcpy(g.der, der, n * sizeof(*der), cudaMemcpyHostToDevice), "h2d der");
  ck(cudaMemcpy(g.occ, occ, n * sizeof(*occ), cudaMemcpyHostToDevice), "h2d occ");
  ck(cudaMemcpy(g.edge, edges, en * sizeof(*edges), cudaMemcpyHostToDevice), "h2d edge");
  bind_n<<<1, 1>>>(g.rec, g.der, g.occ, n, g.edge, en, g.cl, g.ok);
  ck(cudaGetLastError(), "launch");
  ck(cudaDeviceSynchronize(), "sync");
  int ok = 0;
  ck(cudaMemcpy(&ok, g.ok, sizeof(ok), cudaMemcpyDeviceToHost), "ok");
  if (out != nullptr)
    ck(cudaMemcpy(out, g.cl, sizeof(*out), cudaMemcpyDeviceToHost), "cl");
  return ok != 0;
}

int main() {
  ResidentRecipeCell rec[4]{};
  ResidentRecipeDerivation der[4]{};
  if (!seed_rank(&rec[0], &der[0], 0xA0, 0xB0, 1u) ||
      !seed_rank(&rec[1], &der[1], 0xA1, 0xB1, 2u) ||
      !seed_rank(&rec[2], &der[2], 0xA2, 0xB2, 3u))
    return 2;
  const std::uint32_t v0[] = {10u, 20u};
  const std::uint32_t v1[] = {20u, 30u};
  const std::uint32_t v2[] = {30u, 40u};
  ResidentRecipeOccurrence occ[4]{};
  if (!bind_live(rec[0], der[0], v0, 0xD10, &occ[0]) ||
      !bind_live(rec[1], der[1], v1, 0xD11, &occ[1]) ||
      !bind_live(rec[2], der[2], v2, 0xD12, &occ[2]))
    return 3;
  ResidentOccurrenceCoupling edges[3]{};
  if (!bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[1], der[1], 0u, &edges[0]) ||
      !bind_resident_occurrence_coupling(occ[1], der[1], 1u, occ[2], der[2], 0u, &edges[1]))
    return 4;
  ResidentRelationalNetworkClosure host{};
  if (!bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &host) ||
      host.identity == 0u || host.boundary_count != 2u ||
      !boundary_has(host, 10u, ResidentRecipePortDirection::input) ||
      !boundary_has(host, 40u, ResidentRecipePortDirection::output) ||
      boundary_has(host, 20u, ResidentRecipePortDirection::output) ||
      boundary_has(host, 30u, ResidentRecipePortDirection::input))
    return 5;

  GpuN g{};
  ck(cudaMalloc(&g.rec, 4u * sizeof(ResidentRecipeCell)), "rec");
  ck(cudaMalloc(&g.der, 4u * sizeof(ResidentRecipeDerivation)), "der");
  ck(cudaMalloc(&g.occ, 4u * sizeof(ResidentRecipeOccurrence)), "occ");
  ck(cudaMalloc(&g.edge, 3u * sizeof(ResidentOccurrenceCoupling)), "edge");
  ck(cudaMalloc(&g.cl, sizeof(ResidentRelationalNetworkClosure)), "cl");
  ck(cudaMalloc(&g.ok, sizeof(int)), "ok");

  ResidentRelationalNetworkClosure dev{};
  if (!gpu_bind(g, rec, 3u, der, occ, edges, 2u, &dev) ||
      dev.identity != host.identity || !same_boundary(dev, host))
    return 6;

  ResidentRecipeCell rec_rev[3] = {rec[2], rec[0], rec[1]};
  ResidentRecipeDerivation der_rev[3] = {der[2], der[0], der[1]};
  ResidentRecipeOccurrence occ_rev[3] = {occ[2], occ[0], occ[1]};
  ResidentRelationalNetworkClosure rev{};
  if (!gpu_bind(g, rec_rev, 3u, der_rev, occ_rev, edges, 2u, &rev) ||
      rev.identity != host.identity || !same_boundary(rev, host))
    return 7;

  if (gpu_bind(g, rec, 3u, der, occ, edges, 1u, nullptr)) return 8;

  rec[3] = rec[0];
  der[3] = der[0];
  const std::uint32_t v_share[] = {20u, 40u};
  if (!bind_live(rec[3], der[3], v_share, 0xD13, &occ[3])) return 9;
  if (!bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[3], der[3], 0u, &edges[2]))
    return 9;
  if (gpu_bind(g, rec, 4u, der, occ, edges, 3u, nullptr)) return 9;

  rec[3] = rec[2];
  der[3] = der[2];
  const std::uint32_t v3[] = {40u, 50u};
  if (!bind_live(rec[3], der[3], v3, 0xD14, &occ[3])) return 11;
  if (!bind_resident_occurrence_coupling(occ[2], der[2], 1u, occ[3], der[3], 0u, &edges[2]))
    return 11;
  ResidentRelationalNetworkClosure lesion_host{};
  if (!bind_resident_relational_network_closure(rec, der, occ, 4u, edges, 3u, &lesion_host) ||
      lesion_host.identity == host.identity || lesion_host.boundary_count != 2u ||
      !boundary_has(lesion_host, 10u, ResidentRecipePortDirection::input) ||
      !boundary_has(lesion_host, 50u, ResidentRecipePortDirection::output) ||
      boundary_has(lesion_host, 40u, ResidentRecipePortDirection::output))
    return 11;
  ResidentRelationalNetworkClosure lesion_dev{};
  if (!gpu_bind(g, rec, 4u, der, occ, edges, 3u, &lesion_dev) ||
      lesion_dev.identity != lesion_host.identity || !same_boundary(lesion_dev, lesion_host))
    return 12;

  occ[1].state = kResidentRecipeOccurrenceSettled;
  if (gpu_bind(g, rec, 3u, der, occ, edges, 2u, nullptr)) return 10;

  std::printf(
      "FOUNDRY_MIXED_RANK_NETWORK CUDA GREEN host_device_match=1 order_canon=1 "
      "short_coupling_refuse=1 settled_member_refuse=1 "
      "boundary_bn=1 boundary_lesion=1 uncoupled_shared_refuse=1 "
      "identity=%llu boundary=%u lesion_identity=%llu graph_flip=0\n",
      static_cast<unsigned long long>(host.identity), host.boundary_count,
      static_cast<unsigned long long>(lesion_host.identity));
  return 0;
}
