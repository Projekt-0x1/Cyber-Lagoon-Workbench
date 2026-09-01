#include <cuda_runtime.h>

#include "bcc32_checkpoint.hpp"
#include "bcc32_genesis.hpp"
#include "bcc32_law_identity.hpp"
#include "bcc32_transition.cuh"
#include "bcc32_types.cuh"
#include "bcc32_world_entry.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace substrate::bcc32;

enum class Command {
    help,
    preflight_80b,
    inspect,
    init_g0,
    init_capsule,
    advance,
};

struct Arguments {
    Command command = Command::help;
    std::filesystem::path world;
    std::filesystem::path capsule;
    std::filesystem::path lineage_repository;
    SiteCoord origin{};
    std::uint64_t steps = 1u;
    std::uint32_t testing_aperture_chunks = 0u;
    std::uint32_t maximum_core_chunks = 0u;
    bool inverse = false;
    bool reverse_page_order = false;
};

void usage() {
    std::puts(
        "substrate_pure_runtime - canonical BCC32 lineage runtime\n"
        "\n"
        "Commands:\n"
        "  preflight-80b\n"
        "      Allocate and touch the exact 80,000,000,000-bit CUDA aperture.\n"
        "  inspect --world PATH\n"
        "      Read the latest immutable adult commit without running physics.\n"
        "  init-g0 --world PATH\n"
        "      Establish canonical generic-Q genesis under the frozen law.\n"
        "  init-capsule --world PATH --capsule FILE [--lineage-repository PATH]\n"
        "      Establish a sealed G1 or inherited G2 material genesis.\n"
        "  advance --world PATH [--steps N] [--inverse]\n"
        "      Continue the latest lineage through complete F or F^-1 ticks.\n"
        "\n"
        "Advance and genesis options:\n"
        "  --lineage-repository PATH   immutable parent store required for G2\n"
        "  --origin-x INTEGER --origin-y INTEGER --origin-z INTEGER\n"
        "  --maximum-core-chunks N\n"
        "  --reverse-page-order\n"
        "  --testing-aperture-chunks N   explicit contract-only aperture; production\n"
        "                                advance otherwise uses all 2,500 slots\n"
        "  --help, -h\n"
        "\n"
        "Page size and order are non-causal execution choices. All capability remains\n"
        "ordinary material state under the same law; this host never installs behavior.");
}

bool parse_u64(std::string_view text, std::uint64_t* value) {
    // from_chars avoids the C23 __isoc23_strtoull redirect that newer glibc
    // headers emit for stoull, keeping binaries loadable on older hosts.
    std::uint64_t parsed = 0u;
    const auto [end, ec] = std::from_chars(text.begin(), text.end(), parsed, 10);
    if (ec != std::errc() || end != text.end()) return false;
    *value = parsed;
    return true;
}

bool parse_u32(std::string_view text, std::uint32_t* value) {
    std::uint64_t parsed = 0u;
    if (!parse_u64(text, &parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    *value = static_cast<std::uint32_t>(parsed);
    return true;
}

bool parse_coordinate(std::string_view text, CoordinateComponent* value) {
    try {
        *value = CoordinateComponent(std::string(text));
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<Command> parse_command(std::string_view text) {
    if (text == "preflight-80b") return Command::preflight_80b;
    if (text == "inspect") return Command::inspect;
    if (text == "init-g0") return Command::init_g0;
    if (text == "init-capsule") return Command::init_capsule;
    if (text == "advance") return Command::advance;
    if (text == "--help" || text == "-h" || text == "help") return Command::help;
    return std::nullopt;
}

bool parse_arguments(int argc, char** argv, Arguments* arguments) {
    if (arguments == nullptr || argc < 2 || argv[1] == nullptr) return false;
    const std::optional<Command> command = parse_command(argv[1]);
    if (!command.has_value()) return false;
    arguments->command = *command;
    for (int index = 2; index < argc; ++index) {
        const std::string_view option(argv[index] == nullptr ? "" : argv[index]);
        auto next = [&]() -> std::optional<std::string_view> {
            if (++index >= argc || argv[index] == nullptr) return std::nullopt;
            return std::string_view(argv[index]);
        };
        if (option == "--help" || option == "-h") {
            arguments->command = Command::help;
        } else if (option == "--world") {
            const auto value = next();
            if (!value.has_value() || value->empty()) return false;
            arguments->world = *value;
        } else if (option == "--capsule") {
            const auto value = next();
            if (!value.has_value() || value->empty()) return false;
            arguments->capsule = *value;
        } else if (option == "--lineage-repository") {
            const auto value = next();
            if (!value.has_value() || value->empty()) return false;
            arguments->lineage_repository = *value;
        } else if (option == "--origin-x" || option == "--origin-y" ||
                   option == "--origin-z") {
            const auto value = next();
            if (!value.has_value()) return false;
            CoordinateComponent* component = option == "--origin-x"
                                                 ? &arguments->origin.x
                                             : option == "--origin-y"
                                                 ? &arguments->origin.y
                                                 : &arguments->origin.z;
            if (!parse_coordinate(*value, component)) return false;
        } else if (option == "--steps") {
            const auto value = next();
            if (!value.has_value() || !parse_u64(*value, &arguments->steps) ||
                arguments->steps == 0u) return false;
        } else if (option == "--testing-aperture-chunks") {
            const auto value = next();
            if (!value.has_value() ||
                !parse_u32(*value, &arguments->testing_aperture_chunks) ||
                arguments->testing_aperture_chunks == 0u) return false;
        } else if (option == "--maximum-core-chunks") {
            const auto value = next();
            if (!value.has_value() ||
                !parse_u32(*value, &arguments->maximum_core_chunks) ||
                arguments->maximum_core_chunks == 0u) return false;
        } else if (option == "--inverse") {
            arguments->inverse = true;
        } else if (option == "--reverse-page-order") {
            arguments->reverse_page_order = true;
        } else {
            return false;
        }
    }
    if (arguments->command == Command::help ||
        arguments->command == Command::preflight_80b) return true;
    if (arguments->world.empty()) return false;
    if (arguments->command == Command::init_capsule && arguments->capsule.empty()) {
        return false;
    }
    if (arguments->command != Command::init_capsule &&
        (!arguments->capsule.empty() || !arguments->lineage_repository.empty())) {
        return false;
    }
    if (arguments->command != Command::advance &&
        (arguments->steps != 1u || arguments->inverse ||
         arguments->testing_aperture_chunks != 0u ||
         arguments->maximum_core_chunks != 0u ||
         arguments->reverse_page_order)) {
        return false;
    }
    return true;
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open genesis capsule: " + path.string());
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) >
                        std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("genesis capsule size is invalid");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (input.gcount() != size) throw std::runtime_error("genesis capsule is truncated");
    return bytes;
}

void print_commit(const WorldCommit& commit) {
    const ContentAddress material = material_state_identity(commit.chunks);
    std::printf("world_identity=sha256:%s:%llu\n",
                hash_hex(commit.identity.digest).c_str(),
                static_cast<unsigned long long>(commit.identity.byte_count));
    std::printf("material_state_identity=sha256:%s:%llu\n",
                hash_hex(material.digest).c_str(),
                static_cast<unsigned long long>(material.byte_count));
    std::printf("genesis_class=G%u artifact_kind=%u completed_supersteps=%llu\n",
                static_cast<unsigned>(commit.metadata.provenance.genesis_class),
                static_cast<unsigned>(commit.metadata.kind),
                static_cast<unsigned long long>(
                    commit.metadata.replay_boundary.completed_supersteps));
    std::printf("materialized_chunks=%llu non_q_sites=%llu direct_chunk_bytes=%llu\n",
                static_cast<unsigned long long>(commit.chunks.size()),
                static_cast<unsigned long long>(
                    commit.metadata.world_support.non_quiescent_sites),
                static_cast<unsigned long long>(
                    commit.metadata.world_support.direct_word_bytes));
    std::puts("runtime_claim=material_continuation_only");
}

void preflight_80b() {
    std::size_t free_bytes = 0u;
    std::size_t total_bytes = 0u;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        throw std::runtime_error("cannot query CUDA memory for the 80B aperture");
    }
    constexpr std::uint64_t required =
        kProductionBytes + kCarrierSnapshotBytes +
        kProductionExecutorHeadroomBytes;
    PagedWorldExecutor executor = PagedWorldExecutor::production();
    executor.verify_quiescent_aperture();
    const std::uint32_t seeded_words =
        executor.verify_nontrivial_forward_inverse_aperture();
    std::printf(
        "capacity_sites=%llu capacity_bits=%llu aperture_bytes=%llu chunk_slots=%u\n",
        static_cast<unsigned long long>(kProductionSites),
        static_cast<unsigned long long>(kProductionBits),
        static_cast<unsigned long long>(executor.aperture_bytes()),
        executor.aperture_chunks());
    std::printf("production_preflight=passed required_bytes=%llu free_bytes=%llu total_bytes=%llu\n",
                static_cast<unsigned long long>(required),
                static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(total_bytes));
    std::puts("aperture_q_touch=passed");
    std::printf("full_aperture_forward_inverse=passed non_q_seed_words=%u supersteps=2\n",
                seeded_words);
    std::puts("runtime_claim=capacity_only");
}

WorldCommit establish(const Arguments& arguments) {
    const ContentAddress law = canonical_law_identity();
    Genesis genesis{};
    if (arguments.command == Command::init_g0) {
        const auto compiled = compile_g0(law);
        if (!compiled.has_value()) throw std::runtime_error("canonical G0 compilation failed");
        genesis = *compiled;
    } else {
        const std::vector<std::byte> bytes = read_bytes(arguments.capsule);
        if (const auto capsule = open_g1_capsule(bytes, law); capsule.has_value()) {
            genesis = compile_g1(*capsule);
        } else if (const auto capsule = open_g2_capsule(bytes, law); capsule.has_value()) {
            genesis = compile_g2(*capsule);
        } else {
            throw std::runtime_error("capsule is neither canonical G1 nor eligible G2");
        }
    }
    WorldCommit commit{};
    std::string error;
    if (!establish_world(genesis,
                         arguments.origin,
                         arguments.world,
                         &commit,
                         &error,
                         arguments.lineage_repository)) {
        throw std::runtime_error(error);
    }
    return commit;
}

WorldCommit inspect(const std::filesystem::path& world) {
    WorldCommit commit{};
    std::string error;
    if (!load_world_commit(world,
                           ArtifactKind::adult_continuity_checkpoint,
                           &commit,
                           &error)) {
        throw std::runtime_error(error);
    }
    return commit;
}

WorldCommit advance(const Arguments& arguments) {
    PagedWorldExecutor executor = arguments.testing_aperture_chunks == 0u
                                      ? PagedWorldExecutor::production()
                                      : PagedWorldExecutor::testing(
                                            arguments.testing_aperture_chunks);
    TransitionReceipt receipt{};
    std::string error;
    for (std::uint64_t step = 0u; step < arguments.steps; ++step) {
        if (!executor.advance(
                arguments.world,
                ArtifactKind::adult_continuity_checkpoint,
                arguments.inverse,
                {.maximum_core_chunks = arguments.maximum_core_chunks,
                 .reverse_core_order = arguments.reverse_page_order},
                &receipt,
                &error)) {
            throw std::runtime_error(error);
        }
    }
    std::printf("transition=passed inverse=%s steps=%llu pages_last_step=%llu\n",
                arguments.inverse ? "true" : "false",
                static_cast<unsigned long long>(arguments.steps),
                static_cast<unsigned long long>(receipt.pages));
    std::printf("execution_profile=sha256:%s:%llu aperture_chunks=%u\n",
                hash_hex(receipt.execution_profile.digest).c_str(),
                static_cast<unsigned long long>(
                    receipt.execution_profile.byte_count),
                executor.aperture_chunks());
    return inspect(arguments.world);
}

}  // namespace

int main(int argc, char** argv) {
    Arguments arguments{};
    if (!parse_arguments(argc, argv, &arguments)) {
        usage();
        return 2;
    }
    try {
        switch (arguments.command) {
            case Command::help:
                usage();
                return 0;
            case Command::preflight_80b:
                preflight_80b();
                return 0;
            case Command::inspect:
                print_commit(inspect(arguments.world));
                return 0;
            case Command::init_g0:
            case Command::init_capsule:
                print_commit(establish(arguments));
                return 0;
            case Command::advance:
                print_commit(advance(arguments));
                return 0;
        }
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "substrate_pure_runtime: %s\n", exception.what());
        return 1;
    }
    return 2;
}
