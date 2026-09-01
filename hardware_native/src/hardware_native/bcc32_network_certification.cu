// Patch 0006: real GPU implementation of the t0 (construction fronts)
// certification pass. See bcc32_network_certification.cuh for the
// documented scope of this landing (t0 only; t1-t6 are honest RED, not
// faked).

#include "hardware_native/bcc32_network_certification.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "hardware_native/bcc32_network_construction_kernels.cuh"

namespace substrate::bcc32::network_recipe {

namespace {

#define BCC32_CERT_CUDA_CHECK(expr)                                                     \
  do {                                                                                   \
    cudaError_t _status = (expr);                                                        \
    if (_status != cudaSuccess) {                                                        \
      std::fprintf(stderr, "RED: CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(_status));                                          \
      std::exit(1);                                                                       \
    }                                                                                      \
  } while (0)

constexpr std::uint32_t kSeedCount = 3;
constexpr std::uint32_t kNodeCapacity = 8192;
constexpr std::uint32_t kExtendExtent = 70;  // > kPageSiteExtent(64): every
                                              // extension is guaranteed to
                                              // land in a page none of the
                                              // three seeds occupy, so the
                                              // page-directory's own slot
                                              // assignment can never alias a
                                              // directly host-placed seed
                                              // index. See the header
                                              // comment for why this matters.

// One committed node's observable content, captured for the
// allocation-permutation-invariance comparison. Deliberately does NOT
// include anything about *when* (which tick, which thread) it committed --
// only what a later reader of the grown network could itself observe.
struct CommittedNodeSummary {
  std::uint32_t global_index;
  std::uint32_t parent_index;
  std::uint32_t lineage;
  SiteWord chemistry;

  friend bool operator<(const CommittedNodeSummary& a, const CommittedNodeSummary& b) {
    return a.global_index < b.global_index;
  }
  friend bool operator==(const CommittedNodeSummary& a, const CommittedNodeSummary& b) {
    return a.global_index == b.global_index && a.parent_index == b.parent_index &&
           a.lineage == b.lineage && a.chemistry == b.chemistry;
  }
};

Genome build_t0_genome() {
  Genome genome{};
  genome.header.abi_version = kCurrentAbiVersion;
  genome.header.life_function_version = 1;
  genome.header.development_end_tick = 8;
  genome.header.matter_budget = kNodeCapacity;
  genome.header.seed_count = kSeedCount;
  genome.header.field_count = 0;
  genome.header.rule_count = 1;

  // Three independent genesis lineages, deliberately far enough apart
  // (200 sites, > 3x kExtendExtent) that no pair of their single-step
  // extensions can ever target the same coordinate -- t0 is about
  // construction-front geometry and matter closure, not conflict
  // resolution, which patch 0004's own contract already certified.
  genome.seeds[0] = SeedBlock{{0, 0, 0}, /*chemistry=*/1, /*lineage=*/10, 0, 0};
  genome.seeds[1] = SeedBlock{{200, 0, 0}, /*chemistry=*/1, /*lineage=*/20, 0, 0};
  genome.seeds[2] = SeedBlock{{400, 0, 0}, /*chemistry=*/1, /*lineage=*/30, 0, 0};

  ConstructionRule rule{};
  rule.opcode = RuleOpcode::extend;
  rule.direction_mode = 0;  // +x
  rule.begin_tick = 0;
  rule.end_tick = 100;
  rule.require_mask = 0xffffffffu;
  rule.require_value = 1;  // matches only the original seeds' chemistry
  rule.write_value = 100;  // constructed children never match this rule
                            // again, so the front quiesces after tick 0
  rule.extent = kExtendExtent;
  rule.field = 0xffffffffu;
  rule.child_slot = 0xffffffffu;
  genome.rules[0] = rule;

  return genome;
}

struct ScenarioResult {
  std::vector<CommittedNodeSummary> committed;
  bool matter_closure_held;
};

ScenarioResult run_t0_scenario(const Genome& host_genome, std::uint32_t block_size) {
  ScenarioResult result{};
  result.matter_closure_held = true;

  Genome* device_genome = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_genome, sizeof(Genome)));
  BCC32_CERT_CUDA_CHECK(
      cudaMemcpy(device_genome, &host_genome, sizeof(Genome), cudaMemcpyHostToDevice));

  NetworkNode* device_nodes = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_nodes, sizeof(NetworkNode) * kNodeCapacity));
  BCC32_CERT_CUDA_CHECK(cudaMemset(device_nodes, 0, sizeof(NetworkNode) * kNodeCapacity));

  std::vector<NetworkNode> host_seed_nodes(kSeedCount);
  for (std::uint32_t i = 0; i < kSeedCount; ++i) {
    host_seed_nodes[i] = empty_network_node(host_genome.seeds[i].chemistry,
                                              host_genome.seeds[i].lineage, /*birth_tick=*/0);
  }
  BCC32_CERT_CUDA_CHECK(cudaMemcpy(device_nodes, host_seed_nodes.data(),
                                    sizeof(NetworkNode) * kSeedCount, cudaMemcpyHostToDevice));

  PageDirectoryEntry* device_pages = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_pages, sizeof(PageDirectoryEntry) * kPageCount));
  BCC32_CERT_CUDA_CHECK(cudaMemset(device_pages, 0, sizeof(PageDirectoryEntry) * kPageCount));
  for (std::uint32_t i = 0; i < kSeedCount; ++i) {
    const std::uint32_t page_id = page_index_for_coordinate(host_genome.seeds[i].coordinate);
    PageDirectoryEntry host_page{};
    BCC32_CERT_CUDA_CHECK(cudaMemcpy(&host_page, &device_pages[page_id],
                                      sizeof(PageDirectoryEntry), cudaMemcpyDeviceToHost));
    ++host_page.node_count;
    BCC32_CERT_CUDA_CHECK(cudaMemcpy(&device_pages[page_id], &host_page,
                                      sizeof(PageDirectoryEntry), cudaMemcpyHostToDevice));
  }

  NetworkMatterAccount host_account{};
  host_account.initial = kNodeCapacity;
  host_account.live_nodes = kSeedCount;
  NetworkMatterAccount* device_account = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_account, sizeof(NetworkMatterAccount)));
  BCC32_CERT_CUDA_CHECK(cudaMemcpy(device_account, &host_account, sizeof(NetworkMatterAccount),
                                    cudaMemcpyHostToDevice));

  TargetClaim* device_claims = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_claims, sizeof(TargetClaim) * kClaimTableSize));
  TickReport* device_report = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_report, sizeof(TickReport)));

  FrontierEntry* device_frontier = nullptr;
  FrontierEntry* device_next_frontier = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_frontier, sizeof(FrontierEntry) * kNodeCapacity));
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_next_frontier, sizeof(FrontierEntry) * kNodeCapacity));

  std::uint32_t* device_frontier_count = nullptr;
  std::uint32_t* device_next_frontier_count = nullptr;
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_frontier_count, sizeof(std::uint32_t)));
  BCC32_CERT_CUDA_CHECK(cudaMalloc(&device_next_frontier_count, sizeof(std::uint32_t)));

  std::vector<FrontierEntry> host_initial_frontier(kSeedCount);
  for (std::uint32_t i = 0; i < kSeedCount; ++i) {
    host_initial_frontier[i].node_index = i;
    host_initial_frontier[i].coordinate[0] = host_genome.seeds[i].coordinate[0];
    host_initial_frontier[i].coordinate[1] = host_genome.seeds[i].coordinate[1];
    host_initial_frontier[i].coordinate[2] = host_genome.seeds[i].coordinate[2];
  }
  BCC32_CERT_CUDA_CHECK(cudaMemcpy(device_frontier, host_initial_frontier.data(),
                                    sizeof(FrontierEntry) * kSeedCount, cudaMemcpyHostToDevice));
  BCC32_CERT_CUDA_CHECK(
      cudaMemcpy(device_frontier_count, &kSeedCount, sizeof(std::uint32_t), cudaMemcpyHostToDevice));

  LifeFunctionDeviceState state{};
  state.genome = device_genome;
  state.nodes = device_nodes;
  state.node_capacity = kNodeCapacity;
  state.pages = device_pages;
  state.account = device_account;
  state.claims = device_claims;
  state.frontier = device_frontier;
  state.frontier_count = device_frontier_count;
  state.frontier_size = kSeedCount;
  state.next_frontier = device_next_frontier;
  state.next_frontier_count = device_next_frontier_count;
  state.tick = 0;

  for (int t = 0; t < 3; ++t) {
    run_one_tick(state, device_report, block_size);
    BCC32_CERT_CUDA_CHECK(cudaGetLastError());

    NetworkMatterAccount snapshot{};
    BCC32_CERT_CUDA_CHECK(
        cudaMemcpy(&snapshot, device_account, sizeof(NetworkMatterAccount), cudaMemcpyDeviceToHost));
    if (committed_matter(snapshot) > snapshot.initial) result.matter_closure_held = false;
  }

  std::vector<NetworkNode> host_nodes(kNodeCapacity);
  BCC32_CERT_CUDA_CHECK(cudaMemcpy(host_nodes.data(), device_nodes,
                                    sizeof(NetworkNode) * kNodeCapacity, cudaMemcpyDeviceToHost));
  for (std::uint32_t i = kSeedCount; i < kNodeCapacity; ++i) {
    if (host_nodes[i].chemistry == 0) continue;  // never committed into
    CommittedNodeSummary summary{};
    summary.global_index = i;
    summary.parent_index = host_nodes[i].parent[0];
    summary.lineage = host_nodes[i].lineage;
    summary.chemistry = host_nodes[i].chemistry;
    result.committed.push_back(summary);
  }
  std::sort(result.committed.begin(), result.committed.end());

  cudaFree(device_genome);
  cudaFree(device_nodes);
  cudaFree(device_pages);
  cudaFree(device_account);
  cudaFree(device_claims);
  cudaFree(device_report);
  cudaFree(device_frontier);
  cudaFree(device_next_frontier);
  cudaFree(device_frontier_count);
  cudaFree(device_next_frontier_count);

  return result;
}

}  // namespace

T0CertificationReport run_t0_construction_front_certification() {
  T0CertificationReport report{};

  const Genome host_genome = build_t0_genome();

  // Independently predicted from the genome's own geometry, before any
  // device execution -- not read back from a first run and then compared to
  // a second. Each seed's single-step extension lands in a page none of the
  // three seeds occupy (kExtendExtent > kPageSiteExtent), so that page's
  // very first reservation always gets slot_in_page==0, making the
  // resulting global_index fully predictable from page arithmetic alone.
  struct Prediction {
    std::uint32_t global_index;
    std::uint32_t parent_index;
    std::uint32_t lineage;
  };
  std::vector<Prediction> predicted;
  for (std::uint32_t i = 0; i < kSeedCount; ++i) {
    std::uint32_t target[3] = {host_genome.seeds[i].coordinate[0],
                                 host_genome.seeds[i].coordinate[1],
                                 host_genome.seeds[i].coordinate[2]};
    step_coordinate(target, host_genome.rules[0].direction_mode, host_genome.rules[0].extent);
    const std::uint32_t page_id = page_index_for_coordinate(target);
    predicted.push_back(Prediction{global_node_index(page_id, /*slot_in_page=*/0), i,
                                     host_genome.seeds[i].lineage});
  }
  std::sort(predicted.begin(), predicted.end(),
            [](const Prediction& a, const Prediction& b) { return a.global_index < b.global_index; });

  const ScenarioResult run_narrow = run_t0_scenario(host_genome, /*block_size=*/32);
  const ScenarioResult run_wide = run_t0_scenario(host_genome, /*block_size=*/256);

  report.nodes_constructed = static_cast<std::uint32_t>(run_narrow.committed.size());
  report.matter_closure_held = run_narrow.matter_closure_held && run_wide.matter_closure_held;

  std::uint32_t distinct_lineages = 0;
  {
    std::vector<std::uint32_t> lineages;
    for (const auto& c : run_narrow.committed) lineages.push_back(c.lineage);
    std::sort(lineages.begin(), lineages.end());
    lineages.erase(std::unique(lineages.begin(), lineages.end()), lineages.end());
    distinct_lineages = static_cast<std::uint32_t>(lineages.size());
  }
  report.distinct_lineages_constructed = distinct_lineages;

  report.front_geometry_exact =
      run_narrow.committed.size() == predicted.size() &&
      std::equal(run_narrow.committed.begin(), run_narrow.committed.end(), predicted.begin(),
                 [](const CommittedNodeSummary& observed, const Prediction& expected) {
                   return observed.global_index == expected.global_index &&
                          observed.parent_index == expected.parent_index &&
                          observed.lineage == expected.lineage;
                 });

  report.allocation_permutation_invariant =
      run_narrow.committed.size() == run_wide.committed.size() &&
      std::equal(run_narrow.committed.begin(), run_narrow.committed.end(),
                 run_wide.committed.begin());

  report.certified = report.distinct_lineages_constructed >= 2 && report.matter_closure_held &&
                      report.front_geometry_exact && report.allocation_permutation_invariant;

  return report;
}

}  // namespace substrate::bcc32::network_recipe
