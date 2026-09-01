BCC32_PA_GLOBAL void initialize_predictive_assembly_kernel(DeviceState* state) {
#if defined(__CUDACC__)
  if (blockIdx.x != 0u) return;
  auto* bytes = reinterpret_cast<std::uint8_t*>(state);
  for (std::size_t index = threadIdx.x; index < sizeof(DeviceState);
       index += blockDim.x) {
    bytes[index] = 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0u) initialize_device_state(state);
#else
  *state = DeviceState{};
  initialize_device_state(state);
#endif
}

BCC32_PA_GLOBAL void predictive_assembly_kernel(DeviceState* state, RunCommand command,
                                                DeviceReceipt* receipt) {
#if defined(__CUDACC__)
  if (blockIdx.x == 0u && threadIdx.x == 0u) run_tick(state, command, receipt);
#else
  run_tick(state, command, receipt);
#endif
}

// Stable names for the later adult integration.
using PredictiveAssemblyState = DeviceState;
using PredictiveAssemblyReceipt = DeviceReceipt;
using PredictiveAssemblyCommand = RunCommand;
