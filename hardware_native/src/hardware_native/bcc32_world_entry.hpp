#pragma once

#include "bcc32_checkpoint.hpp"
#include "bcc32_genesis.hpp"

#include <filesystem>
#include <string>

namespace substrate::bcc32 {

// G0, G1, and G2 material all enter through this one function. Their
// provenance claims differ, but the resulting commit is consumed by the same
// BCC32 transition engine.
bool establish_world(const Genesis& genesis,
                     const SiteCoord& origin,
                     const std::filesystem::path& repository,
                     WorldCommit* commit,
                     std::string* error,
                     const std::filesystem::path& lineage_repository = {});

}  // namespace substrate::bcc32
