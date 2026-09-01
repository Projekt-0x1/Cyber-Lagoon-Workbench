#pragma once

#include <utility>

#include "substrate/circuits/built_connectome.cuh"

namespace substrate::circuits {

static constexpr uint32_t kOrganoidCellDishPoolWords = 134217728u;
static constexpr uint32_t kOrganoidCellDishRawContactWords = 524288u;
static constexpr uint32_t kOrganoidCellDishRawContactBase = 1048576u;
static constexpr uint32_t kOrganoidCellDishPublicSurfaceWords = 16388u;
static constexpr uint32_t kOrganoidCellDishHardwarePressureWords = 1024u;
static constexpr uint32_t kOrganoidCellDishMotorSurfaceWords =
    kOrganoidCellDishPublicSurfaceWords;
static constexpr uint32_t kOrganoidCellDishActionSurfaceWords =
    kOrganoidCellDishPublicSurfaceWords;
static constexpr uint32_t kOrganoidCellDishFieldWords = 4194304u;
static constexpr uint32_t kOrganoidCellDishSourceGroundingWords =
    kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishActionReadinessWords =
    kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishAutonomicSurfaceWords =
    kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishBodyContactWords =
    kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishBodyContactBase =
    kOrganoidCellDishRawContactBase + kOrganoidCellDishRawContactWords;
static constexpr uint32_t kOrganoidCellDishHardwarePressureBase =
    kOrganoidCellDishBodyContactBase + kOrganoidCellDishBodyContactWords;
static constexpr uint32_t kOrganoidCellDishReafferenceWords =
    kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishReafferenceBase =
    kOrganoidCellDishHardwarePressureBase + kOrganoidCellDishHardwarePressureWords;
static constexpr uint32_t kOrganoidCellDishSourceGroundingBase =
    kOrganoidCellDishReafferenceBase + kOrganoidCellDishReafferenceWords;
static constexpr uint32_t kOrganoidCellDishActionReadinessBase =
    kOrganoidCellDishSourceGroundingBase + kOrganoidCellDishSourceGroundingWords;
static constexpr uint32_t kOrganoidCellDishMotorSurfaceBase =
    kOrganoidCellDishActionReadinessBase + kOrganoidCellDishActionReadinessWords;
static constexpr uint32_t kOrganoidCellDishActionSurfaceBase =
    kOrganoidCellDishMotorSurfaceBase + kOrganoidCellDishMotorSurfaceWords;
static constexpr uint32_t kOrganoidCellDishPublicSurfaceBase =
    kOrganoidCellDishActionSurfaceBase + kOrganoidCellDishActionSurfaceWords;
static constexpr uint32_t kOrganoidCellDishAutonomicSurfaceBase =
    kOrganoidCellDishPublicSurfaceBase + kOrganoidCellDishPublicSurfaceWords;
static constexpr uint32_t kOrganoidCellDishSurvivorBase =
    kOrganoidCellDishAutonomicSurfaceBase + kOrganoidCellDishAutonomicSurfaceWords;
static constexpr uint32_t kOrganoidCellDishColonyBase =
    kOrganoidCellDishSurvivorBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishGenericBase =
    kOrganoidCellDishColonyBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishGeometryBase =
    kOrganoidCellDishGenericBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishSheetBase =
    kOrganoidCellDishGeometryBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishPlasticBase =
    kOrganoidCellDishSheetBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishEnergyBase =
    kOrganoidCellDishPlasticBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishGainBase =
    kOrganoidCellDishEnergyBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishBridgeBase =
    kOrganoidCellDishGainBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishMotorPressureBase =
    kOrganoidCellDishBridgeBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishRecurrenceBase =
    kOrganoidCellDishMotorPressureBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishInhibitionBase =
    kOrganoidCellDishRecurrenceBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishContactTraceBase =
    kOrganoidCellDishInhibitionBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishRecipePopulationBase =
    kOrganoidCellDishContactTraceBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishRuleParameterBase =
    kOrganoidCellDishRecipePopulationBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishMutationTraceBase =
    kOrganoidCellDishRuleParameterBase + kOrganoidCellDishFieldWords;
static constexpr uint32_t kOrganoidCellDishActionTraceBase =
    kOrganoidCellDishMutationTraceBase + kOrganoidCellDishFieldWords;

inline void circuit_organoid_cell_dish(BuiltConnectome* dish) {
    if (!dish) return;
    dish->pool_words = kOrganoidCellDishPoolWords;
    dish->exact_memory = true;

    auto surface = [&](const char* id,
                       uint32_t words,
                       uint32_t base,
                       const char* domain,
                       const char* owner,
                       const char* purpose) {
        BuiltStream s{};
        s.id = id;
        s.domain = domain;
        s.owner = owner;
        s.purpose = purpose;
        s.base = base;
        s.words = words;
        s.init_fill = 0u;
        s.init_fill_set = true;
        dish->streams.push_back(std::move(s));
    };

    surface(
        "raw_contact_stream",
        kOrganoidCellDishRawContactWords,
        kOrganoidCellDishRawContactBase,
        "source_contact",
        "body_contact",
        "continuous raw contact pressure");
    surface(
        "body_contact_stream",
        kOrganoidCellDishBodyContactWords,
        kOrganoidCellDishBodyContactBase,
        "body_contact",
        "body_contact",
        "map-shaped multibody ingress and body-loop consequence pressure");
    surface(
        "hardware_pressure_stream",
        kOrganoidCellDishHardwarePressureWords,
        kOrganoidCellDishHardwarePressureBase,
        "viability",
        "substrate",
        "device-resident timing, contention, and collision residue");
    surface(
        "body_reafference_stream",
        kOrganoidCellDishReafferenceWords,
        kOrganoidCellDishReafferenceBase,
        "body_contact",
        "body",
        "map-shaped previous body-action phenotype re-entering as body contact");
    surface(
        "source_grounding_stream",
        kOrganoidCellDishSourceGroundingWords,
        kOrganoidCellDishSourceGroundingBase,
        "source_contact",
        "substrate",
        "field-local source/body grounding residue");
    surface(
        "action_readiness_potential_stream",
        kOrganoidCellDishActionReadinessWords,
        kOrganoidCellDishActionReadinessBase,
        "body_output",
        "body",
        "field-local body-derived action readiness pressure");
    surface(
        "body_motor_surface_stream",
        kOrganoidCellDishMotorSurfaceWords,
        kOrganoidCellDishMotorSurfaceBase,
        "body_output",
        "body",
        "passive phenotype motor surface");
    surface(
        "body_action_surface_stream",
        kOrganoidCellDishActionSurfaceWords,
        kOrganoidCellDishActionSurfaceBase,
        "body_output",
        "body",
        "packed generic body-action phenotype surface");
    surface(
        "body_public_glyph_surface_stream",
        kOrganoidCellDishPublicSurfaceWords,
        kOrganoidCellDishPublicSurfaceBase,
        "body_output",
        "body",
        "device-native public glyph/render pressure surface downstream of body action");
    surface(
        "body_autonomic_surface_stream",
        kOrganoidCellDishAutonomicSurfaceWords,
        kOrganoidCellDishAutonomicSurfaceBase,
        "body_output",
        "body",
        "field-local hardware/body viability pressure surface");
    surface(
        "survivor_field_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishSurvivorBase,
        "survivor_geometry",
        "observer",
        "survivor geometry snapshot surface");
    surface(
        "organoid_colony_field_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishColonyBase,
        "survivor_geometry",
        "observer",
        "grown colony pressure field surface");
    surface(
        "generic_cell_colony_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishGenericBase,
        "survivor_geometry",
        "observer",
        "generic cell colony residue surface");
    surface(
        "cell_colony_survivor_geometry_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishGeometryBase,
        "survivor_geometry",
        "observer",
        "surviving colony geometry export surface");
    surface(
        "cell_sheet_geometry_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishSheetBase,
        "development",
        "substrate",
        "generic sheet scaffold and local coordinate residue");
    surface(
        "cell_plastic_trace_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishPlasticBase,
        "development",
        "substrate",
        "local plastic trace from pressure mismatch");
    surface(
        "cell_energy_budget_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishEnergyBase,
        "viability",
        "substrate",
        "hardware pressure budget residue from contention and occupancy");
    surface(
        "cell_gain_field_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishGainBase,
        "viability",
        "substrate",
        "generic gain pressure from local viability state");
    surface(
        "cell_long_bridge_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishBridgeBase,
        "development",
        "substrate",
        "sparse long-range survivor bridge residue");
    surface(
        "cell_motor_pressure_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishMotorPressureBase,
        "body_output",
        "body",
        "passive motor pressure projected from survivor geometry");
    surface(
        "cell_recurrent_trace_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishRecurrenceBase,
        "development",
        "substrate",
        "local recurrent wave residue");
    surface(
        "cell_inhibition_trace_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishInhibitionBase,
        "development",
        "substrate",
        "local competition and pruning residue");
    surface(
        "cell_contact_trace_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishContactTraceBase,
        "development",
        "substrate",
        "persistent generic trace written only by body/contact pressure");
    surface(
        "cell_recipe_population_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishRecipePopulationBase,
        "development",
        "substrate",
        "device-discovered starting recipe variants from local pressure");
    surface(
        "cell_rule_parameter_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishRuleParameterBase,
        "development",
        "substrate",
        "mutable local update parameters inherited by active cells");
    surface(
        "cell_mutation_trace_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishMutationTraceBase,
        "development",
        "substrate",
        "stress-triggered local rule-parameter mutation residue");
    surface(
        "cell_action_consequence_trace_stream",
        kOrganoidCellDishFieldWords,
        kOrganoidCellDishActionTraceBase,
        "body_output",
        "body",
        "persistent generic trace written only by emitted body-action phenotype");
    dish->drone_surfaces.push_back(
        BuiltDroneSurface{"sensory", "body_contact", kOrganoidCellDishBodyContactBase,
                          kOrganoidCellDishBodyContactWords});
    dish->drone_surfaces.push_back(
        BuiltDroneSurface{"motor", "body_output", kOrganoidCellDishMotorSurfaceBase,
                          kOrganoidCellDishMotorSurfaceWords});
    dish->drone_surfaces.push_back(
        BuiltDroneSurface{"motor", "body_action", kOrganoidCellDishActionSurfaceBase,
                          kOrganoidCellDishActionSurfaceWords});
    dish->drone_surfaces.push_back(
        BuiltDroneSurface{"motor", "body_public_glyph", kOrganoidCellDishPublicSurfaceBase,
                          kOrganoidCellDishPublicSurfaceWords});
}

}  // namespace substrate::circuits
