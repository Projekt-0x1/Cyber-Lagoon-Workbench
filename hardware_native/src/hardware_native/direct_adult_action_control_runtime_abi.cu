#include <cuda_runtime.h>
#include <stdexcept>
#include "hardware_native/direct_adult_action_control_runtime.cuh"
#include "hardware_native/direct_adult_action_control_runtime_abi.cuh"
namespace substrate::direct_adult_core {
std::size_t direct_adult_action_control_runtime_storage_bytes() noexcept { return sizeof(DirectAdultActionControlRuntimeBlock); }
DirectAdultActionControlRuntimeBlock* create_direct_adult_action_control_runtime(){
  DirectAdultActionControlRuntimeBlock* p=nullptr;
  if(cudaMalloc(&p,sizeof(*p))!=cudaSuccess) throw std::runtime_error("create_direct_adult_action_control_runtime: cudaMalloc");
  if(cudaMemset(p,0,sizeof(*p))!=cudaSuccess){cudaFree(p);throw std::runtime_error("create_direct_adult_action_control_runtime: cudaMemset");}
  return p;
}
void destroy_direct_adult_action_control_runtime(DirectAdultActionControlRuntimeBlock* p) noexcept { if(p) (void)cudaFree(p); }
}
