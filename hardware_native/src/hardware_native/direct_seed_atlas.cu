#include "hardware_native/direct_seed_atlas.cuh"

// The species seed atlas is extracted one atlas family per unit below;
// this translation unit is the catalog composition point -- which families
// exist, in atlas order. Each unit owns its constants, authoring helpers,
// and public genome builders outright.
#include "direct_seed_atlas_net00_terminal_surface.cuh"
#include "direct_seed_atlas_net01_dorsal_stream.cuh"
#include "direct_seed_atlas_net02_ventral_stream.cuh"
#include "direct_seed_atlas_net12_modulatory_channels.cuh"
#include "direct_seed_atlas_net04_binding_index.cuh"
#include "direct_seed_atlas_net08_thalamic_relays.cuh"
#include "direct_seed_atlas_net10_replicated_microzones.cuh"
#include "direct_seed_atlas_net05_association_gradient.cuh"
#include "direct_seed_atlas_net13_backbone_tracts.cuh"
#include "direct_seed_atlas_net15_grounding_streams.cuh"
#include "direct_seed_atlas_net14_bilateral_homologues.cuh"
#include "direct_seed_atlas_net06_net07_control.cuh"
#include "direct_seed_atlas_net09_selection_loops.cuh"
#include "direct_seed_atlas_net17_homeostatic_loops.cuh"
#include "direct_seed_atlas_net11_salience_switch_ecology.cuh"
#include "direct_seed_atlas_net16_limbic_loops.cuh"
