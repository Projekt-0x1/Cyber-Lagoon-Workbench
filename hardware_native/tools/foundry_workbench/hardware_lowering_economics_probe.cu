// Fast engineering assay for earned physical lowerings.  This is not an Adult
// capability contract: it compares equivalent mechanics and charges setup work.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include <mma.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Point {
  float x, y, z;
};

struct BvhNode {
  Point lo{}, hi{};
  int begin = 0, end = 0, left = -1, right = -1;
};

struct Bvh {
  const std::vector<Point>* points = nullptr;
  std::vector<int> order;
  std::vector<BvhNode> nodes;

  int build(int begin, int end) {
    BvhNode node;
    node.begin = begin;
    node.end = end;
    node.lo = node.hi = (*points)[order[begin]];
    for (int i = begin + 1; i < end; ++i) {
      const Point& p = (*points)[order[i]];
      node.lo.x = std::min(node.lo.x, p.x);
      node.lo.y = std::min(node.lo.y, p.y);
      node.lo.z = std::min(node.lo.z, p.z);
      node.hi.x = std::max(node.hi.x, p.x);
      node.hi.y = std::max(node.hi.y, p.y);
      node.hi.z = std::max(node.hi.z, p.z);
    }
    const int here = static_cast<int>(nodes.size());
    nodes.push_back(node);
    if (end - begin <= 16)
      return here;
    const float extent[3] = {node.hi.x - node.lo.x, node.hi.y - node.lo.y, node.hi.z - node.lo.z};
    const int axis =
        extent[1] > extent[0] ? (extent[2] > extent[1] ? 2 : 1) : (extent[2] > extent[0] ? 2 : 0);
    const int middle = (begin + end) / 2;
    std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                     [&](int a, int b) {
                       const Point& pa = (*points)[a];
                       const Point& pb = (*points)[b];
                       return axis == 0 ? pa.x < pb.x : (axis == 1 ? pa.y < pb.y : pa.z < pb.z);
                     });
    nodes[here].left = build(begin, middle);
    nodes[here].right = build(middle, end);
    return here;
  }

  void reset(const std::vector<Point>& ps) {
    points = &ps;
    order.resize(ps.size());
    std::iota(order.begin(), order.end(), 0);
    nodes.clear();
    nodes.reserve(ps.size() / 8);
    build(0, static_cast<int>(ps.size()));
  }

  static float box_distance2(const Point& q, const BvhNode& n) {
    const float dx = q.x < n.lo.x ? n.lo.x - q.x : (q.x > n.hi.x ? q.x - n.hi.x : 0.0f);
    const float dy = q.y < n.lo.y ? n.lo.y - q.y : (q.y > n.hi.y ? q.y - n.hi.y : 0.0f);
    const float dz = q.z < n.lo.z ? n.lo.z - q.z : (q.z > n.hi.z ? q.z - n.hi.z : 0.0f);
    return dx * dx + dy * dy + dz * dz;
  }

  std::uint64_t query(const Point& q, float radius2, std::uint64_t& touched) const {
    std::uint64_t hits = 0;
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top) {
      const BvhNode& n = nodes[stack[--top]];
      if (box_distance2(q, n) > radius2)
        continue;
      if (n.left >= 0) {
        stack[top++] = n.left;
        stack[top++] = n.right;
        continue;
      }
      for (int i = n.begin; i < n.end; ++i) {
        const Point& p = (*points)[order[i]];
        const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
        ++touched;
        hits += dx * dx + dy * dy + dz * dz <= radius2;
      }
    }
    return hits;
  }
};

std::uint32_t lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

float unit(std::uint32_t& state) {
  return static_cast<float>(lcg(state) >> 8) / 16777216.0f;
}

std::uint64_t direct_queries(const std::vector<Point>& points, const std::vector<Point>& queries,
                             float radius2, std::uint64_t& touched) {
  std::uint64_t hits = 0;
  for (const Point& q : queries)
    for (const Point& p : points) {
      const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
      ++touched;
      hits += dx * dx + dy * dy + dz * dz <= radius2;
    }
  return hits;
}

bool spatial_assay() {
  constexpr int kPoints = 4096, kQueries = 512, kFrames = 6;
  constexpr float kRadius = 0.045f;
  std::uint32_t random = 0x51A7E123u;
  std::vector<std::vector<Point>> frames(kFrames, std::vector<Point>(kPoints));
  for (Point& p : frames[0])
    p = {unit(random), unit(random), unit(random)};
  for (int frame = 1; frame < kFrames; ++frame) {
    frames[frame] = frames[frame - 1];
    for (int i = 0; i < kPoints; ++i) {
      const float drift = 0.0015f * static_cast<float>((i + frame) % 7 - 3);
      frames[frame][i].x = std::fmod(frames[frame][i].x + drift + 1.0f, 1.0f);
    }
  }
  std::vector<Point> queries(kQueries);
  for (Point& q : queries)
    q = {unit(random), unit(random), unit(random)};

  std::uint64_t direct_hits = 0, bvh_hits = 0;
  std::uint64_t direct_touched = 0, bvh_touched = 0;
  const auto direct_begin = Clock::now();
  for (const auto& frame : frames)
    direct_hits += direct_queries(frame, queries, kRadius * kRadius, direct_touched);
  const auto direct_end = Clock::now();

  double build_ms = 0.0, query_ms = 0.0;
  for (const auto& frame : frames) {
    Bvh bvh;
    const auto build_begin = Clock::now();
    bvh.reset(frame);
    const auto build_end = Clock::now();
    build_ms += std::chrono::duration<double, std::milli>(build_end - build_begin).count();
    const auto query_begin = Clock::now();
    for (const Point& q : queries)
      bvh_hits += bvh.query(q, kRadius * kRadius, bvh_touched);
    const auto query_end = Clock::now();
    query_ms += std::chrono::duration<double, std::milli>(query_end - query_begin).count();
  }
  const double direct_ms =
      std::chrono::duration<double, std::milli>(direct_end - direct_begin).count();
  const bool equivalent = direct_hits == bvh_hits;
  const bool prunes = bvh_touched < direct_touched;
  const double bvh_total_ms = build_ms + query_ms;
  const bool nominated = equivalent && prunes && bvh_total_ms < direct_ms;
  std::printf(
      "SPATIAL_BVH_WORKLOAD equivalence=%d direct_hits=%llu bvh_hits=%llu "
      "direct_touched=%llu bvh_touched=%llu direct_ms=%.3f build_ms=%.3f "
      "query_ms=%.3f total_speedup=%.3f rt_candidate_nominated=%d "
      "rt_hardware_proof=0\n",
      equivalent, static_cast<unsigned long long>(direct_hits),
      static_cast<unsigned long long>(bvh_hits), static_cast<unsigned long long>(direct_touched),
      static_cast<unsigned long long>(bvh_touched), direct_ms, build_ms, query_ms,
      direct_ms / bvh_total_ms, nominated);
  return equivalent && prunes;
}

__global__ void scalar_dense_block(const half* weights, const half* activations, float* out,
                                   int blocks, int nodes) {
  const int block = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (block >= blocks)
    return;
  const std::size_t weight_offset = static_cast<std::size_t>(block) * nodes * nodes;
  const std::size_t activation_offset = static_cast<std::size_t>(block) * nodes;
  for (int row = lane; row < nodes; row += 32) {
    float sum = 0.0f;
    for (int column = 0; column < nodes; ++column) {
      const float activation = __half2float(activations[activation_offset + column]);
      if (activation > 0.0001f)
        sum += __half2float(weights[weight_offset + row * nodes + column]) * activation;
    }
    out[activation_offset + row] = sum;
  }
}

__global__ void wmma_dense_block(const half* weights, const half* activations, float* out,
                                 int blocks, int nodes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const int block = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (block >= blocks)
    return;
  __shared__ half shared_weights[256];
  __shared__ half shared_activation[256];
  __shared__ float shared_output[256];
  const std::size_t weight_offset = static_cast<std::size_t>(block) * nodes * nodes;
  const std::size_t activation_offset = static_cast<std::size_t>(block) * nodes;
  const int tiles = nodes / 16;
  for (int row_tile = 0; row_tile < tiles; ++row_tile) {
    float accumulated = 0.0f;
    for (int column_tile = 0; column_tile < tiles; ++column_tile) {
      for (int element = 0; element < 8; ++element) {
        const int flat = lane * 8 + element;
        const int row = flat / 16;
        const int column = flat % 16;
        shared_weights[flat] =
            weights[weight_offset + (row_tile * 16 + row) * nodes + column_tile * 16 + column];
      }
      if (lane < 16) {
        const half activation = activations[activation_offset + column_tile * 16 + lane];
        for (int column = 0; column < 16; ++column)
          shared_activation[lane * 16 + column] = activation;
      }
      __syncwarp();
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
      nvcuda::wmma::fill_fragment(cf, 0.0f);
      nvcuda::wmma::load_matrix_sync(af, shared_weights, 16);
      nvcuda::wmma::load_matrix_sync(bf, shared_activation, 16);
      nvcuda::wmma::mma_sync(cf, af, bf, cf);
      nvcuda::wmma::store_matrix_sync(shared_output, cf, 16, nvcuda::wmma::mem_row_major);
      __syncwarp();
      if (lane < 16)
        accumulated += shared_output[lane * 16];
      __syncwarp();
    }
    if (lane < 16)
      out[activation_offset + row_tile * 16 + lane] = accumulated;
  }
#endif
}

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess)
    return true;
  std::fprintf(stderr, "CUDA_ERROR operation=%s detail=%s\n", operation,
               cudaGetErrorString(status));
  return false;
}

bool tensor_case(int blocks, int nodes, int active) {
  const std::size_t weight_elements = static_cast<std::size_t>(blocks) * nodes * nodes;
  const std::size_t activation_elements = static_cast<std::size_t>(blocks) * nodes;
  std::vector<half> host_weights(weight_elements);
  std::vector<half> host_activations(activation_elements, __float2half(0.0f));
  for (std::size_t i = 0; i < weight_elements; ++i)
    host_weights[i] = __float2half(static_cast<float>((i * 3 + 1) % 5) * 0.25f);
  for (int block = 0; block < blocks; ++block)
    for (int column = 0; column < active; ++column)
      host_activations[block * nodes + column] =
          __float2half(static_cast<float>((column * 7 + 2) % 5) * 0.25f);
  half *weights = nullptr, *activations = nullptr;
  float *scalar = nullptr, *tensor = nullptr;
  if (!cuda_ok(cudaMalloc(&weights, weight_elements * sizeof(half)), "malloc_weights") ||
      !cuda_ok(cudaMalloc(&activations, activation_elements * sizeof(half)),
               "malloc_activations") ||
      !cuda_ok(cudaMalloc(&scalar, activation_elements * sizeof(float)), "malloc_scalar") ||
      !cuda_ok(cudaMalloc(&tensor, activation_elements * sizeof(float)), "malloc_tensor"))
    return false;
  bool ok = cuda_ok(cudaMemcpy(weights, host_weights.data(), weight_elements * sizeof(half),
                               cudaMemcpyHostToDevice),
                    "copy_weights") &&
            cuda_ok(cudaMemcpy(activations, host_activations.data(),
                               activation_elements * sizeof(half), cudaMemcpyHostToDevice),
                    "copy_activations");
  scalar_dense_block<<<blocks, 32>>>(weights, activations, scalar, blocks, nodes);
  wmma_dense_block<<<blocks, 32>>>(weights, activations, tensor, blocks, nodes);
  ok = ok && cuda_ok(cudaDeviceSynchronize(), "warmup");

  const int work = blocks * nodes * nodes;
  const int repeats = std::max(20, std::min(200, 262144 / work));
  cudaEvent_t start{}, stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto measure_scalar = [&]() {
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i)
      scalar_dense_block<<<blocks, 32>>>(weights, activations, scalar, blocks, nodes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  };
  auto measure_tensor = [&]() {
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i)
      wmma_dense_block<<<blocks, 32>>>(weights, activations, tensor, blocks, nodes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  };
  std::vector<float> scalar_samples, tensor_samples;
  for (int trial = 0; trial < 7; ++trial) {
    if ((trial & 1) == 0) {
      scalar_samples.push_back(measure_scalar());
      tensor_samples.push_back(measure_tensor());
    } else {
      tensor_samples.push_back(measure_tensor());
      scalar_samples.push_back(measure_scalar());
    }
  }
  std::sort(scalar_samples.begin(), scalar_samples.end());
  std::sort(tensor_samples.begin(), tensor_samples.end());
  const float scalar_ms = scalar_samples[3];
  const float tensor_ms = tensor_samples[3];

  std::vector<float> host_scalar(activation_elements), host_tensor(activation_elements);
  ok = ok &&
       cuda_ok(cudaMemcpy(host_scalar.data(), scalar, activation_elements * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy_scalar") &&
       cuda_ok(cudaMemcpy(host_tensor.data(), tensor, activation_elements * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy_tensor");
  float max_error = 0.0f;
  for (std::size_t i = 0; i < activation_elements; ++i)
    max_error = std::max(max_error, std::fabs(host_scalar[i] - host_tensor[i]));
  const bool equivalent = ok && max_error <= 1.0e-3f;
  const float speedup = scalar_ms / tensor_ms;
  std::printf(
      "TENSOR_WMMA_WORKLOAD blocks=%d nodes=%d active=%d repeats=%d equivalence=%d "
      "max_abs_error=%.6g scalar_ms=%.4f wmma_ms=%.4f speedup=%.3f "
      "tensor_candidate_nominated=%d\n",
      blocks, nodes, active, repeats, equivalent, max_error, scalar_ms, tensor_ms, speedup,
      equivalent && speedup > 1.05f);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(weights);
  cudaFree(activations);
  cudaFree(scalar);
  cudaFree(tensor);
  return equivalent;
}

}  // namespace

int main() {
  if (!spatial_assay()) {
    std::puts("LOWERING_ECONOMICS RED spatial_equivalence=0");
    return 1;
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::puts(
        "LOWERING_ECONOMICS SKIP gpu=0 spatial=GREEN "
        "rt_hardware_proof=0 direct_parity=NOT_RUN");
    return 77;
  }
  cudaDeviceProp properties{};
  cudaGetDeviceProperties(&properties, 0);
  if (properties.major < 7) {
    std::printf(
        "LOWERING_ECONOMICS SKIP gpu=%s compute=%d.%d wmma=unsupported "
        "spatial=GREEN direct_parity=NOT_RUN\n",
        properties.name, properties.major, properties.minor);
    return 77;
  }
  bool equivalent = true;
  equivalent &= tensor_case(1, 16, 16);
  equivalent &= tensor_case(1, 32, 32);
  equivalent &= tensor_case(1, 64, 64);
  equivalent &= tensor_case(1, 128, 128);
  equivalent &= tensor_case(16, 16, 16);
  equivalent &= tensor_case(16, 32, 32);
  equivalent &= tensor_case(16, 64, 64);
  equivalent &= tensor_case(16, 128, 128);
  equivalent &= tensor_case(16, 64, 4);
  std::printf(
      "LOWERING_ECONOMICS %s gpu=%s compute=%d.%d tensor_equivalence=%d "
      "spatial=GREEN rt_hardware_proof=0 adult_attached=0 graph_flip=0\n",
      equivalent ? "GREEN" : "RED", properties.name, properties.major, properties.minor,
      equivalent);
  return equivalent ? 0 : 1;
}
