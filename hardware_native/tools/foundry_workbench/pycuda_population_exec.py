#!/usr/bin/env python3
"""Generic PyCUDA lowering for very large simple resident populations.

This is a physical execution backend for the workbench population law. It has no
language, task, Recipe, or semantic opcodes. One generic module owns deterministic
feature->site projection, procedural sparse incidence, touched-site recruitment,
eligibility, causal settlement, and plastic edge revision.

The large dormant population is represented as one independently addressable bit
per resident site. Life-changed state lives in bounded sparse GPU tables. This is
intentional: baseline matter is huge; irreducible lived deltas stay sparse.
"""
from __future__ import annotations

from dataclasses import dataclass

MASK64 = (1 << 64) - 1
EMPTY64 = MASK64


def mix64(x: int) -> int:
    x = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


@dataclass(frozen=True)
class GpuPopulationSpecV1:
    site_count: int = 80_000_000_000
    fanout: int = 2
    sites_per_feature: int = 3
    eligibility_horizon: int = 8
    site_delta_capacity: int = 1 << 20
    edge_delta_capacity: int = 1 << 21

    def validate(self) -> None:
        if not 32 <= self.site_count < (1 << 63):
            raise ValueError("gpu_population:site_count")
        if not 1 <= self.fanout <= 8:
            raise ValueError("gpu_population:fanout")
        if not 1 <= self.sites_per_feature <= 16:
            raise ValueError("gpu_population:sites_per_feature")
        if not 1 <= self.eligibility_horizon <= 65535:
            raise ValueError("gpu_population:eligibility_horizon")
        for name, value in (("site_delta_capacity", self.site_delta_capacity),
                            ("edge_delta_capacity", self.edge_delta_capacity)):
            if value < 1024 or value & (value - 1):
                raise ValueError(f"gpu_population:{name}_power2")


@dataclass(frozen=True)
class GpuPopulationOccurrenceV1:
    identity: int
    tick: int
    sites: tuple[int, ...]
    edges: tuple[int, ...]
    feature_count: int
    recruit_kernel_ms: float


CUDA_SOURCE = r"""
#include <stdint.h>

#define EMPTY64 0xffffffffffffffffULL

extern "C" {

__device__ __forceinline__ unsigned long long mix64_dev(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL);
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EBULL);
    return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned long long topology_target(
    unsigned long long site,
    unsigned int lane,
    unsigned long long n) {
    unsigned long long span = (n - 1ULL < 257ULL) ? (n - 1ULL) : 257ULL;
    if (span < 2ULL) span = 2ULL;
    unsigned long long delta = 1ULL +
        (mix64_dev(site * 0xD6E8FEB86659FD93ULL + (unsigned long long)lane) % span);
    unsigned long long mixed = mix64_dev(
        ((unsigned long long)lane + 1ULL) * 0xA0761D6478BD642FULL + site) % n;
    unsigned long long target = (site + delta + mixed) % n;
    if (target == site) target = (site + 1ULL) % n;
    return target;
}

__global__ void make_signature(
    const unsigned long long* features,
    unsigned int feature_count,
    unsigned int sites_per_feature,
    unsigned long long site_count,
    unsigned long long* out_pairs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = feature_count * sites_per_feature;
    if (i >= total) return;
    unsigned int feature_i = i / sites_per_feature;
    unsigned int lane = i - feature_i * sites_per_feature;
    unsigned long long site = mix64_dev(
        features[feature_i] ^ mix64_dev((unsigned long long)lane + 1ULL)) % site_count;
    out_pairs[(unsigned long long)i * 2ULL] = site;
    out_pairs[(unsigned long long)i * 2ULL + 1ULL] = topology_target(site, 0U, site_count);
}

__device__ __forceinline__ long long find_or_insert(
    unsigned long long* keys,
    unsigned int capacity,
    unsigned long long key,
    unsigned int* inserted,
    unsigned int* overflow) {
    unsigned int slot = (unsigned int)(mix64_dev(key) & (unsigned long long)(capacity - 1U));
    for (unsigned int probe = 0; probe < 128U; ++probe) {
        unsigned long long prior = atomicCAS(keys + slot, EMPTY64, key);
        if (prior == EMPTY64) {
            atomicAdd(inserted, 1U);
            return (long long)slot;
        }
        if (prior == key) return (long long)slot;
        slot = (slot + 1U) & (capacity - 1U);
    }
    atomicAdd(overflow, 1U);
    return -1LL;
}

__device__ __forceinline__ long long find_existing(
    const unsigned long long* keys,
    unsigned int capacity,
    unsigned long long key) {
    unsigned int slot = (unsigned int)(mix64_dev(key) & (unsigned long long)(capacity - 1U));
    for (unsigned int probe = 0; probe < 128U; ++probe) {
        unsigned long long prior = keys[slot];
        if (prior == key) return (long long)slot;
        if (prior == EMPTY64) return -1LL;
        slot = (slot + 1U) & (capacity - 1U);
    }
    return -1LL;
}

__global__ void recruit_sites(
    const unsigned long long* sites,
    unsigned int count,
    unsigned long long site_count,
    unsigned int fanout,
    unsigned int tick,
    unsigned int horizon,
    unsigned long long* site_keys,
    unsigned int* site_support,
    unsigned int* site_last_tick,
    unsigned int* site_expiry,
    unsigned int site_capacity,
    unsigned long long* edge_keys,
    int* edge_weight,
    unsigned int* edge_expiry,
    unsigned int edge_capacity,
    unsigned int* counters) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long site = sites[i];
    if (site >= site_count) return;

    long long ss = find_or_insert(site_keys, site_capacity, site,
                                  counters + 0, counters + 2);
    if (ss >= 0) {
        atomicAdd(site_support + ss, 1U);
        atomicMax(site_last_tick + ss, tick);
        atomicMax(site_expiry + ss, tick + horizon);
    }

    for (unsigned int lane = 0; lane < fanout; ++lane) {
        unsigned long long edge = site * (unsigned long long)fanout + (unsigned long long)lane;
        long long es = find_or_insert(edge_keys, edge_capacity, edge,
                                      counters + 1, counters + 3);
        if (es >= 0) {
            if (edge_weight[es] == 0) edge_weight[es] = 1;
            atomicMax(edge_expiry + es, tick + horizon);
        }
    }
}

__global__ void settle_sites(
    const unsigned long long* sites,
    unsigned int count,
    unsigned int fanout,
    unsigned int tick,
    int effect,
    unsigned int independent,
    const unsigned long long* site_keys,
    const unsigned int* site_expiry,
    unsigned int site_capacity,
    const unsigned long long* edge_keys,
    int* edge_weight,
    const unsigned int* edge_expiry,
    unsigned int edge_capacity,
    unsigned int* result) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || independent == 0U || effect == 0) return;
    unsigned long long site = sites[i];
    long long ss = find_existing(site_keys, site_capacity, site);
    if (ss >= 0 && site_expiry[ss] >= tick) atomicAdd(result + 0, 1U);

    int direction = effect > 0 ? 1 : -1;
    for (unsigned int lane = 0; lane < fanout; ++lane) {
        unsigned long long edge = site * (unsigned long long)fanout + (unsigned long long)lane;
        long long es = find_existing(edge_keys, edge_capacity, edge);
        if (es < 0 || edge_expiry[es] < tick) continue;
        int prior = edge_weight[es];
        int next = prior + direction;
        if (next > 127) next = 127;
        if (next < -127) next = -127;
        if (next != prior) {
            edge_weight[es] = next;
            atomicAdd(result + 1, 1U);
        }
    }
}

__global__ void gather_resident_bits(
    const unsigned long long* sites,
    unsigned int count,
    const unsigned long long* site_keys,
    unsigned int site_capacity,
    unsigned char* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = (unsigned char)(find_existing(site_keys, site_capacity, sites[i]) >= 0 ? 1U : 0U);
}

__global__ void gather_edge_weights(
    const unsigned long long* edge_ids,
    unsigned int count,
    const unsigned long long* edge_keys,
    const int* edge_weight,
    unsigned int edge_capacity,
    int* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long slot = find_existing(edge_keys, edge_capacity, edge_ids[i]);
    out[i] = slot >= 0 ? edge_weight[slot] : 0;
}

__global__ void gather_site_state(
    const unsigned long long* site_ids,
    unsigned int count,
    const unsigned long long* site_keys,
    const unsigned int* site_support,
    const unsigned int* site_last_tick,
    const unsigned int* site_expiry,
    unsigned int site_capacity,
    unsigned int* out_support,
    unsigned int* out_last_tick,
    unsigned int* out_expiry) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long slot = find_existing(site_keys, site_capacity, site_ids[i]);
    if (slot < 0) {
        out_support[i] = 0U; out_last_tick[i] = 0U; out_expiry[i] = 0U;
    } else {
        out_support[i] = site_support[slot];
        out_last_tick[i] = site_last_tick[slot];
        out_expiry[i] = site_expiry[slot];
    }
}

__global__ void gather_edge_state(
    const unsigned long long* edge_ids,
    unsigned int count,
    const unsigned long long* edge_keys,
    const int* edge_weight,
    const unsigned int* edge_expiry,
    unsigned int edge_capacity,
    int* out_weight,
    unsigned int* out_expiry) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long slot = find_existing(edge_keys, edge_capacity, edge_ids[i]);
    if (slot < 0) {
        out_weight[i] = 0; out_expiry[i] = 0U;
    } else {
        out_weight[i] = edge_weight[slot]; out_expiry[i] = edge_expiry[slot];
    }
}

__global__ void restore_site_state(
    const unsigned long long* site_ids,
    const unsigned int* support,
    const unsigned int* last_tick,
    const unsigned int* expiry,
    unsigned int count,
    unsigned long long* site_keys,
    unsigned int* site_support,
    unsigned int* site_last_tick,
    unsigned int* site_expiry,
    unsigned int site_capacity,
    unsigned int* counters) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long site = site_ids[i];
    long long slot = find_or_insert(site_keys, site_capacity, site, counters + 0, counters + 2);
    if (slot < 0) return;
    site_support[slot] = support[i];
    site_last_tick[slot] = last_tick[i];
    site_expiry[slot] = expiry[i];
}

__global__ void restore_edge_state(
    const unsigned long long* edge_ids,
    const int* weight,
    const unsigned int* expiry,
    unsigned int count,
    unsigned long long* edge_keys,
    int* edge_weight,
    unsigned int* edge_expiry,
    unsigned int edge_capacity,
    unsigned int* counters) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long edge = edge_ids[i];
    long long slot = find_or_insert(edge_keys, edge_capacity, edge, counters + 1, counters + 3);
    if (slot < 0) return;
    edge_weight[slot] = weight[i];
    edge_expiry[slot] = expiry[i];
}

}
"""


class PyCudaPopulationExecV1:
    """One generic GPU executor instance for a resident population."""

    _module = None
    _module_context_handle = None
    _module_compile_count = 0

    def __init__(self, spec: GpuPopulationSpecV1):
        spec.validate()
        self.spec = spec
        self.tick = 0
        self.next_occurrence = 1
        self.occurrences: list[GpuPopulationOccurrenceV1] = []
        self.credit_events = 0
        self.revision_events = 0
        self.signature_kernel_ms_total = 0.0
        self.recruit_kernel_ms_total = 0.0
        self.settle_kernel_ms_total = 0.0
        self._life_sites: set[int] = set()
        self._life_edges: set[int] = set()
        self._allocations = []

        import numpy as np
        import pycuda.autoinit  # noqa: F401 - owns the process CUDA context
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self.np = np
        self.cuda = cuda
        context_handle = int(cuda.Context.get_current().handle)
        if (self.__class__._module is None or self.__class__._module_context_handle != context_handle):
            self.__class__._module = SourceModule(
                CUDA_SOURCE,
                no_extern_c=True,
                options=["-O3", "-std=c++14"],
            )
            self.__class__._module_context_handle = context_handle
            self.__class__._module_compile_count += 1
        self.module = self.__class__._module
        self.make_signature_kernel = self.module.get_function("make_signature")
        self.recruit_kernel = self.module.get_function("recruit_sites")
        self.settle_kernel = self.module.get_function("settle_sites")
        self.gather_bits_kernel = self.module.get_function("gather_resident_bits")
        self.gather_weights_kernel = self.module.get_function("gather_edge_weights")
        self.gather_site_state_kernel = self.module.get_function("gather_site_state")
        self.gather_edge_state_kernel = self.module.get_function("gather_edge_state")
        self.restore_site_state_kernel = self.module.get_function("restore_site_state")
        self.restore_edge_state_kernel = self.module.get_function("restore_edge_state")

        # Baseline species matter is procedural and therefore costs O(1) writes at
        # birth. Only irreducible life deltas are explicitly materialized. This is
        # the same logical population namespace at every scale; dormant size must
        # not create an initialization sweep.
        self.resident_bytes = 0
        self.procedural_baseline_bits = int(spec.site_count)
        self.site_delta_bytes = spec.site_delta_capacity * (8 + 4 + 4 + 4)
        self.edge_delta_bytes = spec.edge_delta_capacity * (8 + 4 + 4)
        self.counter_bytes = 6 * 4
        self.required_device_bytes = self.site_delta_bytes + self.edge_delta_bytes + self.counter_bytes
        free_bytes, total_bytes = cuda.mem_get_info()
        self.free_before = int(free_bytes)
        self.total_device_bytes = int(total_bytes)
        if self.required_device_bytes > self.free_before:
            raise MemoryError(
                f"gpu_population:need={self.required_device_bytes} free={self.free_before}"
            )

        sc = spec.site_delta_capacity
        ec = spec.edge_delta_capacity
        self.d_site_keys = self._alloc(sc * 8)
        self.d_site_support = self._alloc(sc * 4)
        self.d_site_last_tick = self._alloc(sc * 4)
        self.d_site_expiry = self._alloc(sc * 4)
        self.d_edge_keys = self._alloc(ec * 8)
        self.d_edge_weight = self._alloc(ec * 4)
        self.d_edge_expiry = self._alloc(ec * 4)
        self.d_counters = self._alloc(4 * 4)
        self.d_settle_result = self._alloc(2 * 4)

        cuda.memset_d8(self.d_site_keys, 0xFF, sc * 8)
        cuda.memset_d32(self.d_site_support, 0, sc)
        cuda.memset_d32(self.d_site_last_tick, 0, sc)
        cuda.memset_d32(self.d_site_expiry, 0, sc)
        cuda.memset_d8(self.d_edge_keys, 0xFF, ec * 8)
        cuda.memset_d32(self.d_edge_weight, 0, ec)
        cuda.memset_d32(self.d_edge_expiry, 0, ec)
        cuda.memset_d32(self.d_counters, 0, 4)
        cuda.memset_d32(self.d_settle_result, 0, 2)
        cuda.Context.synchronize()
        self.free_after_init = int(cuda.mem_get_info()[0])

    @property
    def module_compile_count(self) -> int:
        return self.__class__._module_compile_count

    def _alloc(self, size: int):
        allocation = self.cuda.mem_alloc(int(size))
        self._allocations.append(allocation)
        return allocation

    @staticmethod
    def _grid(count: int, block: int = 256):
        return (max(1, (int(count) + block - 1) // block), 1, 1), (block, 1, 1)

    def _timed_kernel(self, func, count: int, *args) -> float:
        start = self.cuda.Event()
        end = self.cuda.Event()
        grid, block = self._grid(count)
        start.record()
        func(*args, block=block, grid=grid)
        end.record()
        end.synchronize()
        return float(start.time_till(end))

    def signature(self, features) -> tuple[int, ...]:
        values = tuple(int(x) & MASK64 for x in features)
        if not values:
            return ()
        np = self.np
        host_features = np.asarray(values, dtype=np.uint64)
        pair_count = len(values) * self.spec.sites_per_feature
        host_pairs = np.empty(pair_count * 2, dtype=np.uint64)
        d_features = self._alloc(host_features.nbytes)
        d_pairs = self._alloc(host_pairs.nbytes)
        try:
            self.cuda.memcpy_htod(d_features, host_features)
            kernel_ms = self._timed_kernel(
                self.make_signature_kernel,
                pair_count,
                d_features,
                np.uint32(len(values)),
                np.uint32(self.spec.sites_per_feature),
                np.uint64(self.spec.site_count),
                d_pairs,
            )
            self.signature_kernel_ms_total += kernel_ms
            self.cuda.memcpy_dtoh(host_pairs, d_pairs)
        finally:
            d_features.free()
            d_pairs.free()
            self._allocations.remove(d_features)
            self._allocations.remove(d_pairs)
        return tuple(sorted(set(int(x) for x in host_pairs.tolist())))

    def topology_target_cpu(self, site: int, lane: int = 0) -> int:
        n = self.spec.site_count
        span = max(2, min(n - 1, 257))
        delta = 1 + (mix64(site * 0xD6E8FEB86659FD93 + lane) % span)
        target = (site + delta + (mix64((lane + 1) * 0xA0761D6478BD642F + site) % n)) % n
        return (site + 1) % n if target == site else target

    def feature_sites_cpu(self, feature: int) -> tuple[int, ...]:
        out = []
        for lane in range(self.spec.sites_per_feature):
            site = mix64(int(feature) ^ mix64(lane + 1)) % self.spec.site_count
            if site not in out:
                out.append(site)
        return tuple(out)

    def signature_cpu(self, features) -> tuple[int, ...]:
        seeds = []
        for feature in features:
            seeds.extend(self.feature_sites_cpu(int(feature)))
        seeds = tuple(dict.fromkeys(seeds))
        return tuple(sorted(set((*seeds, *(self.topology_target_cpu(s, 0) for s in seeds)))))

    def _upload_sites(self, sites: tuple[int, ...]):
        host = self.np.asarray(sites, dtype=self.np.uint64)
        d = self._alloc(max(8, host.nbytes))
        if host.nbytes:
            self.cuda.memcpy_htod(d, host)
        return host, d

    def recruit(self, features) -> GpuPopulationOccurrenceV1:
        values = tuple(int(x) for x in features)
        sites = self.signature(values)
        self.tick += 1
        host_sites, d_sites = self._upload_sites(sites)
        try:
            kernel_ms = self._timed_kernel(
                self.recruit_kernel,
                len(sites),
                d_sites,
                self.np.uint32(len(sites)),
                self.np.uint64(self.spec.site_count),
                self.np.uint32(self.spec.fanout),
                self.np.uint32(self.tick),
                self.np.uint32(self.spec.eligibility_horizon),
                self.d_site_keys,
                self.d_site_support,
                self.d_site_last_tick,
                self.d_site_expiry,
                self.np.uint32(self.spec.site_delta_capacity),
                self.d_edge_keys,
                self.d_edge_weight,
                self.d_edge_expiry,
                self.np.uint32(self.spec.edge_delta_capacity),
                self.d_counters,
            )
        finally:
            d_sites.free()
            self._allocations.remove(d_sites)
        edges = tuple(
            site * self.spec.fanout + lane
            for site in sites
            for lane in range(self.spec.fanout)
        )
        self.recruit_kernel_ms_total += kernel_ms
        occurrence = GpuPopulationOccurrenceV1(
            self.next_occurrence,
            self.tick,
            sites,
            edges,
            len(values),
            kernel_ms,
        )
        self._life_sites.update(sites)
        self._life_edges.update(edges)
        self.next_occurrence += 1
        self.occurrences.append(occurrence)
        return occurrence

    def settle(self, occurrence: GpuPopulationOccurrenceV1, effect: int, independent: bool):
        if occurrence not in self.occurrences:
            raise ValueError("gpu_population:occurrence")
        if not independent or int(effect) == 0:
            return {"credit": 0, "revisions": 0, "kernel_ms": 0.0}
        host_sites, d_sites = self._upload_sites(occurrence.sites)
        self.cuda.memset_d32(self.d_settle_result, 0, 2)
        try:
            kernel_ms = self._timed_kernel(
                self.settle_kernel,
                len(occurrence.sites),
                d_sites,
                self.np.uint32(len(occurrence.sites)),
                self.np.uint32(self.spec.fanout),
                self.np.uint32(self.tick),
                self.np.int32(int(effect)),
                self.np.uint32(1 if independent else 0),
                self.d_site_keys,
                self.d_site_expiry,
                self.np.uint32(self.spec.site_delta_capacity),
                self.d_edge_keys,
                self.d_edge_weight,
                self.d_edge_expiry,
                self.np.uint32(self.spec.edge_delta_capacity),
                self.d_settle_result,
            )
            result = self.np.empty(2, dtype=self.np.uint32)
            self.cuda.memcpy_dtoh(result, self.d_settle_result)
        finally:
            d_sites.free()
            self._allocations.remove(d_sites)
        credit, revisions = (int(result[0]), int(result[1]))
        self.settle_kernel_ms_total += kernel_ms
        self.credit_events += credit
        self.revision_events += revisions
        return {"credit": credit, "revisions": revisions, "kernel_ms": kernel_ms}

    def resident_bits(self, sites: tuple[int, ...]) -> tuple[int, ...]:
        if not sites:
            return ()
        host_sites, d_sites = self._upload_sites(sites)
        out = self.np.empty(len(sites), dtype=self.np.uint8)
        d_out = self._alloc(out.nbytes)
        try:
            grid, block = self._grid(len(sites))
            self.gather_bits_kernel(
                d_sites,
                self.np.uint32(len(sites)),
                self.d_site_keys,
                self.np.uint32(self.spec.site_delta_capacity),
                d_out,
                block=block,
                grid=grid,
            )
            self.cuda.memcpy_dtoh(out, d_out)
        finally:
            d_sites.free()
            d_out.free()
            self._allocations.remove(d_sites)
            self._allocations.remove(d_out)
        return tuple(int(x) for x in out.tolist())

    def edge_weights(self, edge_ids: tuple[int, ...]) -> tuple[int, ...]:
        if not edge_ids:
            return ()
        host = self.np.asarray(edge_ids, dtype=self.np.uint64)
        out = self.np.empty(len(edge_ids), dtype=self.np.int32)
        d_edges = self._alloc(host.nbytes)
        d_out = self._alloc(out.nbytes)
        try:
            self.cuda.memcpy_htod(d_edges, host)
            grid, block = self._grid(len(edge_ids))
            self.gather_weights_kernel(
                d_edges,
                self.np.uint32(len(edge_ids)),
                self.d_edge_keys,
                self.d_edge_weight,
                self.np.uint32(self.spec.edge_delta_capacity),
                d_out,
                block=block,
                grid=grid,
            )
            self.cuda.memcpy_dtoh(out, d_out)
        finally:
            d_edges.free()
            d_out.free()
            self._allocations.remove(d_edges)
            self._allocations.remove(d_out)
        return tuple(int(x) for x in out.tolist())

    @staticmethod
    def overlap(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        bs = set(b)
        return sum(x in bs for x in a)

    @property
    def allocated_edge_count(self) -> int:
        return int(self.spec.site_count * self.spec.fanout)

    def materialized_site_count(self) -> int:
        return len(self._life_sites)

    def _site_state(self, site_ids: tuple[int, ...]):
        if not site_ids:
            return (), (), ()
        np = self.np
        host = np.asarray(site_ids, dtype=np.uint64)
        support = np.empty(len(site_ids), dtype=np.uint32)
        last_tick = np.empty(len(site_ids), dtype=np.uint32)
        expiry = np.empty(len(site_ids), dtype=np.uint32)
        d_ids = self._alloc(host.nbytes)
        d_support = self._alloc(support.nbytes)
        d_last = self._alloc(last_tick.nbytes)
        d_expiry = self._alloc(expiry.nbytes)
        try:
            self.cuda.memcpy_htod(d_ids, host)
            grid, block = self._grid(len(site_ids))
            self.gather_site_state_kernel(
                d_ids, np.uint32(len(site_ids)), self.d_site_keys,
                self.d_site_support, self.d_site_last_tick, self.d_site_expiry,
                np.uint32(self.spec.site_delta_capacity), d_support, d_last, d_expiry,
                block=block, grid=grid)
            self.cuda.memcpy_dtoh(support, d_support)
            self.cuda.memcpy_dtoh(last_tick, d_last)
            self.cuda.memcpy_dtoh(expiry, d_expiry)
        finally:
            for d in (d_ids, d_support, d_last, d_expiry):
                d.free(); self._allocations.remove(d)
        return (tuple(int(x) for x in support.tolist()),
                tuple(int(x) for x in last_tick.tolist()),
                tuple(int(x) for x in expiry.tolist()))

    def _edge_state(self, edge_ids: tuple[int, ...]):
        if not edge_ids:
            return (), ()
        np = self.np
        host = np.asarray(edge_ids, dtype=np.uint64)
        weight = np.empty(len(edge_ids), dtype=np.int32)
        expiry = np.empty(len(edge_ids), dtype=np.uint32)
        d_ids = self._alloc(host.nbytes)
        d_weight = self._alloc(weight.nbytes)
        d_expiry = self._alloc(expiry.nbytes)
        try:
            self.cuda.memcpy_htod(d_ids, host)
            grid, block = self._grid(len(edge_ids))
            self.gather_edge_state_kernel(
                d_ids, np.uint32(len(edge_ids)), self.d_edge_keys,
                self.d_edge_weight, self.d_edge_expiry,
                np.uint32(self.spec.edge_delta_capacity), d_weight, d_expiry,
                block=block, grid=grid)
            self.cuda.memcpy_dtoh(weight, d_weight)
            self.cuda.memcpy_dtoh(expiry, d_expiry)
        finally:
            for d in (d_ids, d_weight, d_expiry):
                d.free(); self._allocations.remove(d)
        return (tuple(int(x) for x in weight.tolist()),
                tuple(int(x) for x in expiry.tolist()))

    def live_eligibility_count(self) -> int:
        ids = tuple(sorted(self._life_sites))
        _, _, expiry = self._site_state(ids)
        return sum(x >= self.tick for x in expiry)

    def sparse_counts(self) -> dict[str, int]:
        out = self.np.empty(4, dtype=self.np.uint32)
        self.cuda.memcpy_dtoh(out, self.d_counters)
        return {
            "site_deltas": int(out[0]),
            "edge_deltas": int(out[1]),
            "site_overflow": int(out[2]),
            "edge_overflow": int(out[3]),
        }

    def checkpoint(self) -> dict:
        sites = tuple(sorted(self._life_sites))
        edges = tuple(sorted(self._life_edges))
        support, last_tick, site_expiry = self._site_state(sites)
        edge_weight, edge_expiry = self._edge_state(edges)
        return {
            "schema": 2,
            "backend": "pycuda-procedural-population-v1",
            "spec": self.spec.__dict__,
            "tick": self.tick,
            "next_occurrence": self.next_occurrence,
            "sites": sites,
            "site_support": support,
            "site_last_tick": last_tick,
            "site_expiry": site_expiry,
            "edges": edges,
            "edge_weight": edge_weight,
            "edge_expiry": edge_expiry,
            "occurrences": [o.__dict__ for o in self.occurrences],
            "credit_events": self.credit_events,
            "revision_events": self.revision_events,
        }

    @classmethod
    def restore(cls, data: dict):
        if data.get("schema") != 2 or data.get("backend") != "pycuda-procedural-population-v1":
            raise ValueError("gpu_population:checkpoint_schema")
        obj = cls(GpuPopulationSpecV1(**data["spec"]))
        np = obj.np
        sites = tuple(int(x) for x in data.get("sites", ()))
        edges = tuple(int(x) for x in data.get("edges", ()))
        if sites:
            arrays = (
                np.asarray(sites, dtype=np.uint64),
                np.asarray(data["site_support"], dtype=np.uint32),
                np.asarray(data["site_last_tick"], dtype=np.uint32),
                np.asarray(data["site_expiry"], dtype=np.uint32),
            )
            dev = [obj._alloc(a.nbytes) for a in arrays]
            try:
                for d, a in zip(dev, arrays): obj.cuda.memcpy_htod(d, a)
                grid, block = obj._grid(len(sites))
                obj.restore_site_state_kernel(
                    dev[0], dev[1], dev[2], dev[3], np.uint32(len(sites)),
                    obj.d_site_keys, obj.d_site_support, obj.d_site_last_tick,
                    obj.d_site_expiry, np.uint32(obj.spec.site_delta_capacity),
                    obj.d_counters, block=block, grid=grid)
            finally:
                for d in dev: d.free(); obj._allocations.remove(d)
        if edges:
            arrays = (
                np.asarray(edges, dtype=np.uint64),
                np.asarray(data["edge_weight"], dtype=np.int32),
                np.asarray(data["edge_expiry"], dtype=np.uint32),
            )
            dev = [obj._alloc(a.nbytes) for a in arrays]
            try:
                for d, a in zip(dev, arrays): obj.cuda.memcpy_htod(d, a)
                grid, block = obj._grid(len(edges))
                obj.restore_edge_state_kernel(
                    dev[0], dev[1], dev[2], np.uint32(len(edges)),
                    obj.d_edge_keys, obj.d_edge_weight, obj.d_edge_expiry,
                    np.uint32(obj.spec.edge_delta_capacity), obj.d_counters,
                    block=block, grid=grid)
            finally:
                for d in dev: d.free(); obj._allocations.remove(d)
        obj.cuda.Context.synchronize()
        obj.tick = int(data["tick"])
        obj.next_occurrence = int(data["next_occurrence"])
        obj.occurrences = [GpuPopulationOccurrenceV1(
            int(o["identity"]), int(o["tick"]), tuple(o["sites"]), tuple(o["edges"]),
            int(o["feature_count"]), float(o.get("recruit_kernel_ms", 0.0)))
            for o in data.get("occurrences", ())]
        obj.credit_events = int(data.get("credit_events", 0))
        obj.revision_events = int(data.get("revision_events", 0))
        obj._life_sites = set(sites)
        obj._life_edges = set(edges)
        return obj

    def quantity_vector(self, occurrence: GpuPopulationOccurrenceV1 | None = None,
                        alternatives: int = 0, horizon: int = 0,
                        trajectory: int = 0) -> dict[str, int]:
        counts = self.sparse_counts()
        hot_sites = len(occurrence.sites) if occurrence else 0
        hot_edges = len(occurrence.edges) if occurrence else 0
        live_eligibility = self.live_eligibility_count()
        # Preserve the compact R/I/O/P/E/G/A/H/T/F/C/Y ABI used by the CPU
        # reference organism while exposing the physical lowering receipt too.
        return {
            "R": int(self.spec.site_count),
            "I": int(self.spec.site_count * self.spec.fanout),
            "O": len(self.occurrences),
            "P": hot_sites,
            "E": live_eligibility,
            "G": hot_edges,
            "A": int(alternatives),
            "H": int(horizon),
            "T": hot_edges,
            "F": counts["site_deltas"],
            "C": int(self.spec.site_count - counts["site_deltas"]),
            "Y": int(trajectory),
            "resident_sites": int(self.spec.site_count),
            "resident_state_bits": int(self.procedural_baseline_bits),
            "procedural_edges": int(self.spec.site_count * self.spec.fanout),
            "life_changed_sites": counts["site_deltas"],
            "life_changed_edges": counts["edge_deltas"],
            "hot_sites": hot_sites,
            "hot_edges": hot_edges,
            "device_bytes": int(self.required_device_bytes),
            "procedural_baseline_bytes": 0,
        }

    def close(self) -> None:
        for allocation in reversed(self._allocations):
            try:
                allocation.free()
            except Exception:
                pass
        self._allocations.clear()
        self.cuda.Context.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
