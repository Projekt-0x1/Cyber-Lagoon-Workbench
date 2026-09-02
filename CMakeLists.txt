cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "Cyber Lagoon Direct CUDA architecture")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

project(CyberLagoonWorkbench LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "89")
  message(FATAL_ERROR "Direct public Workbench currently requires CMAKE_CUDA_ARCHITECTURES=89")
endif()

find_package(CUDAToolkit 12.0 REQUIRED COMPONENTS cudart cuda_driver)
set(DIRECT_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/hardware_native/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hardware_native/src
  ${CMAKE_CURRENT_SOURCE_DIR}/hardware_native/src/hardware_native)

add_library(direct_network_life_function STATIC
  hardware_native/src/hardware_native/direct_network_genome.cu
  hardware_native/src/hardware_native/direct_network_life_function.cu
  hardware_native/src/hardware_native/direct_network_resident_development.cu
  hardware_native/src/hardware_native/direct_network_certification.cu
  hardware_native/src/hardware_native/direct_seed_atlas.cu
  hardware_native/src/hardware_native/direct_seed_atlas_net03.cu
  hardware_native/src/hardware_native/direct_seed_atlas_species.cu)
target_include_directories(direct_network_life_function PUBLIC ${DIRECT_INCLUDE_DIRS})
target_compile_options(direct_network_life_function PRIVATE -O3 --compiler-options=-Wall,-Wextra,-Werror)
set_target_properties(direct_network_life_function PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

add_library(direct_exact_history_cold_archive_core OBJECT
  hardware_native/src/hardware_native/direct_exact_history_cold_archive.cu)
target_include_directories(direct_exact_history_cold_archive_core PUBLIC ${DIRECT_INCLUDE_DIRS})
target_compile_options(direct_exact_history_cold_archive_core PRIVATE -O3 --compiler-options=-Wall,-Wextra,-Werror)
set_target_properties(direct_exact_history_cold_archive_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(direct_adult_language_device STATIC
  hardware_native/src/hardware_native/direct_adult_language_one_shot_binding.cu)
target_include_directories(direct_adult_language_device PUBLIC ${DIRECT_INCLUDE_DIRS})
target_compile_options(direct_adult_language_device PRIVATE -O3 --compiler-options=-Wall,-Wextra,-Werror)
set_target_properties(direct_adult_language_device PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

add_library(direct_adult_core STATIC
  hardware_native/src/hardware_native/direct_adult_core.cu
  hardware_native/src/hardware_native/direct_adult_actual_frontier_condensation.cu
  hardware_native/src/hardware_native/direct_adult_core_actual_advance_kernels.cu
  hardware_native/src/hardware_native/direct_adult_core_actual_condensation_kernels.cu
  hardware_native/src/hardware_native/direct_adult_core_actual_publication_kernels.cu
  hardware_native/src/hardware_native/direct_adult_core_dynamics_kernels.cu
  hardware_native/src/hardware_native/direct_adult_core_ingress_kernels.cu
  hardware_native/src/hardware_native/direct_adult_core_route_kernels.cu
  hardware_native/src/hardware_native/direct_adult_core_settlement_kernels.cu
  hardware_native/src/hardware_native/direct_adult_device_ops.cu
  hardware_native/src/hardware_native/direct_adult_observer.cu
  hardware_native/src/hardware_native/direct_adult_action_control_runtime_abi.cu
  hardware_native/src/hardware_native/direct_adult_language_phase_wiring.cu
  hardware_native/src/hardware_native/direct_adult_language_candidate_phase.cu
  hardware_native/src/hardware_native/direct_adult_language_action_candidate_device.cu
  hardware_native/src/hardware_native/direct_adult_action_commit_phase.cu
  hardware_native/src/hardware_native/direct_adult_action_commit_device.cu
  hardware_native/src/hardware_native/direct_adult_resident_language_runtime_abi.cu
  hardware_native/src/hardware_native/direct_adult_language_motor_finalize.cu
  hardware_native/src/hardware_native/direct_adult_persistent.cu
  hardware_native/src/hardware_native/direct_adult_persistent_development_phase.cu
  hardware_native/src/hardware_native/direct_adult_persistent_execution_phase.cu
  hardware_native/src/hardware_native/direct_adult_persistent_ingress_phase.cu
  hardware_native/src/hardware_native/direct_adult_persistent_motor_phase.cu
  hardware_native/src/hardware_native/direct_adult_persistent_settlement_phase.cu
  $<TARGET_OBJECTS:direct_exact_history_cold_archive_core>
  hardware_native/src/hardware_native/direct_arrival_gate.cu)
target_include_directories(direct_adult_core PUBLIC ${DIRECT_INCLUDE_DIRS})
target_compile_definitions(direct_adult_core PUBLIC DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY=1)
target_compile_options(direct_adult_core PRIVATE -O3 --compiler-options=-Wall,-Wextra,-Werror)
target_link_libraries(direct_adult_core PRIVATE direct_adult_language_device)
set_target_properties(direct_adult_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

add_executable(direct_adult_sitdown hardware_native/tools/direct_adult_sitdown.cu)
target_include_directories(direct_adult_sitdown PRIVATE ${DIRECT_INCLUDE_DIRS})
target_link_libraries(direct_adult_sitdown PRIVATE
  direct_adult_core direct_network_life_function CUDA::cudart_static CUDA::cuda_driver)
set_target_properties(direct_adult_sitdown PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(direct_recipe_ir_lab hardware_native/tools/direct_recipe_ir_lab.cu)
target_include_directories(direct_recipe_ir_lab PRIVATE ${DIRECT_INCLUDE_DIRS})
target_link_libraries(direct_recipe_ir_lab PRIVATE CUDA::cudart_static CUDA::cuda_driver)
set_target_properties(direct_recipe_ir_lab PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
