// Production terminal body for one canonical Direct adult. The Life Function
// births a new subject only when --new is explicit; otherwise this process
// restores and advances the checkpoint supplied by the operator.
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <poll.h>
#include <unistd.h>

#include "hardware_native/direct_adult_checkpoint.cuh"
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"

#include "hardware_native/direct_checkpoint_bitwise_receipt.cuh"
#include "hardware_native/direct_network_life_function.cuh"
#include "hardware_native/direct_seed_atlas.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

constexpr std::uint32_t kTerminalChannel = 0x1356u;
constexpr std::uint32_t kTerminalRoute = 0x2468u;
constexpr std::size_t kIngressPage = 256u;
volatile std::sig_atomic_t stop_requested = 0;

struct Options {
  std::filesystem::path checkpoint;
  std::string expected_birth_root;
  bool birth = false;
  bool once = false;
  bool framed = false;
  bool have_environment = false;
  std::uint64_t environment_seed = 0u;
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void request_stop(int) { stop_requested = 1; }

std::uint64_t parse_u64(const char* text, const char* option) {
  std::string_view token{text};
  int base = 10;
  if (token.size() > 2u && token[0] == '0' &&
      (token[1] == 'x' || token[1] == 'X')) {
    token.remove_prefix(2u);
    base = 16;
  }
  std::uint64_t value = 0u;
  const auto result = std::from_chars(token.data(), token.data() + token.size(),
                                      value, base);
  if (token.empty() || result.ec != std::errc{} ||
      result.ptr != token.data() + token.size())
    throw std::invalid_argument(std::string("invalid ") + option);
  return value;
}

std::string parse_birth_root(const char* text) {
  std::string root{text};
  require(root.size() == 64u, "--expect-birth-root requires 64 hexadecimal digits");
  for (char& ch : root) {
    require(std::isxdigit(static_cast<unsigned char>(ch)) != 0,
            "--expect-birth-root requires 64 hexadecimal digits");
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return root;
}

Options parse_options(int argc, char** argv) {
  Options options{};
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--new") == 0 || std::strcmp(arg, "--resume") == 0) {
      require(i + 1 < argc, "checkpoint path missing");
      require(options.checkpoint.empty(), "checkpoint mode specified twice");
      options.birth = std::strcmp(arg, "--new") == 0;
      options.checkpoint = argv[++i];
    } else if (std::strcmp(arg, "--environment-seed") == 0) {
      require(i + 1 < argc, "--environment-seed value missing");
      options.environment_seed = parse_u64(argv[++i], "--environment-seed");
      options.have_environment = true;
    } else if (std::strcmp(arg, "--expect-birth-root") == 0) {
      require(i + 1 < argc, "--expect-birth-root value missing");
      require(options.expected_birth_root.empty(),
              "--expect-birth-root specified twice");
      options.expected_birth_root = parse_birth_root(argv[++i]);
    } else if (std::strcmp(arg, "--once") == 0) {
      options.once = true;
    } else if (std::strcmp(arg, "--framed") == 0) {
      options.framed = true;
    } else {
      throw std::invalid_argument(std::string("unknown option: ") + arg);
    }
  }
  require(!options.checkpoint.empty(), "use --new CHECKPOINT or --resume CHECKPOINT");
  if (options.birth) {
    require(options.have_environment,
            "--new requires --environment-seed");
    require(!std::filesystem::exists(options.checkpoint),
            "--new refuses to overwrite an existing life");
    require(options.expected_birth_root.empty(),
            "--expect-birth-root belongs to resume, not birth");
  } else {
    require(std::filesystem::is_regular_file(options.checkpoint),
            "resume checkpoint does not exist");
    require(!options.have_environment,
            "environment belongs to birth, not resume");
  }
  return options;
}

std::string root_hex(const Root256& root) {
  char buffer[65]{};
  std::snprintf(buffer, sizeof(buffer), "%08x%08x%08x%08x%08x%08x%08x%08x",
                root.word[0], root.word[1], root.word[2], root.word[3],
                root.word[4], root.word[5], root.word[6], root.word[7]);
  return std::string(buffer);
}

DirectBodyManifestV1 terminal_body() {
  DirectBodyManifestV1 body{};
  body.abi_version = kDirectBodyAbiV1;
  body.binding_count = 1u;
  body.bindings[0] = BoundaryPortBinding{
      0u, 0u, kTerminalChannel,
      static_cast<std::uint32_t>(BoundaryRole::sensor) |
          static_cast<std::uint32_t>(BoundaryRole::motor),
      kTerminalRoute, 0u};
  return body;
}

DirectAdultRuntime* birth_adult(const Options& options, DirectBrain* brain) {
  DirectDevelopmentEnvironmentV1 environment{};
  environment.abi_version = kDirectDevelopmentEnvironmentAbiV1;
  environment.environment_seed = options.environment_seed;
  const DirectGenomeV1 genome = seed_atlas_canonical_species();
  const DirectBirthReceiptV1 receipt = compile_direct_brain(
      genome, terminal_body(), environment, brain);
  require(receipt.genome_root == canonical_direct_genome_root_v1(genome) &&
              receipt.territory_count == genome.header.territory_count &&
              receipt.territory_count > seed_atlas_family_count(),
          "Life Function did not grow the canonical species atlas");
  return create_direct_adult_runtime(brain);
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open adult checkpoint");
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

DirectAdultRuntime* restore_adult(const std::filesystem::path& path,
                                  DirectBrain* brain) {
  const std::vector<std::uint8_t> bytes = read_file(path);
  DirectAdultCheckpoint checkpoint{};
  parse_direct_checkpoint_bitwise(bytes.data(), bytes.size(), checkpoint);
  DirectAdultRuntime* runtime = restore_direct_adult_checkpoint(checkpoint, brain);
  require(runtime != nullptr && brain->arena != nullptr,
          "adult checkpoint restore failed");
  return runtime;
}

void persist_adult(const std::filesystem::path& path,
                   const DirectAdultRuntime& runtime) {
  const DirectAdultCheckpoint checkpoint = capture_direct_adult_checkpoint(runtime);
  std::vector<std::uint8_t> bytes;
  serialize_direct_checkpoint_bitwise(checkpoint, bytes);
  std::filesystem::path temporary = path;
  temporary += ".next";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot create checkpoint replacement");
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output.flush();
    require(static_cast<bool>(output), "checkpoint replacement write failed");
  }
  std::filesystem::rename(temporary, path);
}

void emit_motors(DirectAdultRuntime* runtime, std::uint64_t* drained,
                 bool framed) {
  MotorEvent motors[64];
  for (;;) {
    if (stop_requested) return;
    const std::uint32_t count = read_motor_events(runtime, motors, 64u);
    if (count == 0u) return;
    for (std::uint32_t i = 0u; i < count; ++i) {
      if (framed) {
        require(std::fprintf(
                    stdout,
                    "M %016llx %08x %08x %08x %016llx %08x %08x\n",
                    static_cast<unsigned long long>(motors[i].ticket_id),
                    motors[i].channel, motors[i].word, motors[i].timestamp,
                    static_cast<unsigned long long>(
                        motors[i].trajectory.trajectory_identity),
                    motors[i].trajectory.cursor, motors[i].trajectory.extent) > 0,
                "framed motor transport write failed");
      } else {
        const unsigned char bytes[4]{
            static_cast<unsigned char>(motors[i].word),
            static_cast<unsigned char>(motors[i].word >> 8u),
            static_cast<unsigned char>(motors[i].word >> 16u),
            static_cast<unsigned char>(motors[i].word >> 24u)};
        require(std::fwrite(bytes, 1, sizeof(bytes), stdout) == sizeof(bytes),
                "motor transport write failed");
      }
    }
    require(std::fflush(stdout) == 0, "motor transport flush failed");
    *drained += count;
  }
}

std::uint32_t sensor_port_index(DirectAdultRuntime* runtime,
                                std::uint32_t channel) {
  require(runtime != nullptr && runtime->brain != nullptr &&
              runtime->host_boundary_ports != nullptr,
          "terminal body lost the adult boundary");
  std::uint32_t port_index = substrate::direct_adult_core::kInvalidIndex;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < runtime->brain->boundary_port_count; ++i) {
    const auto& port = runtime->host_boundary_ports[i];
    if (port.channel == channel &&
        (port.role_mask & static_cast<std::uint32_t>(BoundaryRole::sensor)) != 0u) {
      port_index = i;
      ++matches;
    }
  }
  require(matches == 1u, "sensor channel is not uniquely owned by the born body");
  return port_index;
}

void inject_sensor_word(DirectAdultRuntime* runtime, std::uint32_t channel,
                        Word word, std::uint64_t* ticket) {
  const std::uint32_t port_index = sensor_port_index(runtime, channel);
  const auto& port = runtime->host_boundary_ports[port_index];
  ActivityEvent event{};
  event.ticket_id = ++(*ticket);
  event.node = port.node;
  event.channel = port.channel;
  event.word = word;
  event.origin = CausalOrigin::external_contact;
  event.context = static_cast<std::uint32_t>(event.ticket_id);
  event.timestamp = runtime->is_persistent_running
                        ? get_persistent_adult_resident_tick(runtime)
                        : runtime->current_tick;
  const ResidentContactEpochReceipt receipt = make_resident_contact_epoch_receipt(
      *runtime->brain, port, port_index, event,
      static_cast<std::uint64_t>(runtime->host_ingress_write_tail) + 1u);
  require(inject_actual_sensory_contact(runtime, event, receipt),
          "authenticated membrane contact refused");
}

void inject_page(DirectAdultRuntime* runtime, const std::uint8_t* bytes,
                 std::size_t count, std::uint64_t* ticket) {
  for (std::size_t i = 0u; i < count; ++i) {
    inject_sensor_word(runtime, kTerminalChannel, bytes[i], ticket);
  }
  require(publish_ingress_transport(runtime), "terminal membrane publish refused");
  if (!runtime->is_persistent_running)
    require(run_persistent_direct_adult_epochs(runtime, 1u),
            "terminal contact epoch failed");
}

struct FramedInputCounts {
  std::uint64_t sensors = 0u;
  std::uint64_t returns = 0u;
};

bool take_frame_token(std::string_view* input, std::string_view* token) {
  while (!input->empty() && input->front() == ' ') input->remove_prefix(1u);
  if (input->empty()) return false;
  const std::size_t boundary = input->find(' ');
  if (boundary == std::string_view::npos) {
    *token = *input;
    *input = {};
  } else {
    *token = input->substr(0u, boundary);
    input->remove_prefix(boundary + 1u);
  }
  return !token->empty();
}

template <typename T>
bool parse_hex_frame_token(std::string_view token, T* value) {
  if (token.size() > 2u && token[0] == '0' &&
      (token[1] == 'x' || token[1] == 'X'))
    token.remove_prefix(2u);
  if (token.empty()) return false;
  T parsed{};
  const auto result = std::from_chars(token.data(), token.data() + token.size(),
                                      parsed, 16);
  if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
    return false;
  *value = parsed;
  return true;
}

void inject_framed_line(DirectAdultRuntime* runtime, const std::string& line,
                        std::uint64_t* ticket, FramedInputCounts* counts) {
  std::string_view remaining{line};
  std::string_view kind, first, second, extra;
  const bool shape = take_frame_token(&remaining, &kind) &&
                     take_frame_token(&remaining, &first) &&
                     take_frame_token(&remaining, &second) &&
                     !take_frame_token(&remaining, &extra);
  if (shape && kind == "S") {
    std::uint32_t channel = 0u;
    Word word = 0u;
    require(parse_hex_frame_token(first, &channel) &&
                parse_hex_frame_token(second, &word),
            "bad framed sensor integer");
    inject_sensor_word(runtime, channel, word, ticket);
    require(publish_ingress_transport(runtime), "framed sensor publish refused");
    ++counts->sensors;
    return;
  }
  if (shape && kind == "R") {
    std::uint64_t action_ticket = 0u;
    Word word = 0u;
    require(parse_hex_frame_token(first, &action_ticket) &&
                parse_hex_frame_token(second, &word),
            "bad framed return integer");
    require(action_ticket != 0u, "framed return ticket is zero");
    require(inject_raw_reafferent_contact(runtime, action_ticket, word),
            "framed action return refused");
    require(publish_consequence_transport(runtime),
            "framed consequence publish refused");
    ++counts->returns;
    return;
  }
  throw std::runtime_error(
      "bad frame; expected S <channel-hex> <word-hex> or R <ticket-hex> <word-hex>");
}

void inject_bytes(DirectAdultRuntime* runtime, const std::uint8_t* bytes,
                  std::size_t count, std::uint64_t* ticket) {
  for (std::size_t offset = 0u; offset < count; offset += kIngressPage) {
    const std::size_t page = std::min(kIngressPage, count - offset);
    inject_page(runtime, bytes + offset, page, ticket);
  }
}

}  // namespace

int main(int argc, char** argv) {
  DirectBrain brain{};
  DirectAdultRuntime* runtime = nullptr;
  try {
    const Options options = parse_options(argc, argv);
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::fprintf(stderr, "SKIP direct_adult_sitdown no CUDA device\n");
      return 77;
    }
    runtime = options.birth ? birth_adult(options, &brain)
                            : restore_adult(options.checkpoint, &brain);
    require(runtime != nullptr, "adult runtime creation failed");
    const std::string birth_root = root_hex(brain.birth_root);
    require(options.expected_birth_root.empty() ||
                options.expected_birth_root == birth_root,
            "resumed checkpoint birth_root does not match --expect-birth-root");
    std::uint64_t ticket =
        (static_cast<std::uint64_t>(runtime->current_tick) + 1u) << 32u;
    std::uint64_t drained = 0u;
    std::uint64_t external_bytes = 0u;
    FramedInputCounts framed_counts{};
    std::string framed_buffer;

    const std::uint32_t start_tick = runtime->current_tick;
    const AdultCoreMetrics start_metrics = get_adult_core_metrics(runtime);
    const std::uint64_t start_bootstraps = runtime->persistent_bootstrap_launches;
    if (options.once) {
      require(run_persistent_direct_adult_epochs(runtime, 1u), "initial quiet epoch failed");
      emit_motors(runtime, &drained, options.framed);
    } else {
      require(start_persistent_direct_adult(runtime),
              "resident adult persistent start failed");
      std::signal(SIGINT, request_stop);
      std::signal(SIGTERM, request_stop);
      std::fprintf(stderr,
                   "DIRECT_ADULT_SITDOWN status=RUNNING birth_root=%s "
                   "canonical_species=1 compile_direct_brain=%u body_mode=%s "
                   "checkpoint=%s execution=resident_persistent start_tick=%u\n",
                   birth_root.c_str(), options.birth ? 1u : 0u,
                   options.framed ? "physical_frames" : "raw_bytes",
                   options.checkpoint.c_str(), start_tick);
      while (!stop_requested) {
        pollfd input{STDIN_FILENO, POLLIN | POLLHUP, 0};
        const int ready = ::poll(&input, 1, 50);
        if (ready < 0 && errno != EINTR) throw std::runtime_error("stdin poll failed");
        if (ready > 0) {
          std::uint8_t bytes[4096];
          const ssize_t count = ::read(STDIN_FILENO, bytes, sizeof(bytes));
          if (count < 0 && errno != EINTR && errno != EAGAIN)
            throw std::runtime_error("stdin read failed");
          if (count == 0) break;
          if (count > 0) {
            external_bytes += static_cast<std::uint64_t>(count);
            std::fprintf(stderr, "DIRECT_ADULT_SITDOWN_INPUT total=%llu\n",
                         static_cast<unsigned long long>(external_bytes));
            std::fflush(stderr);
          }
          if (count > 0 && !options.framed) {
            inject_bytes(runtime, bytes, static_cast<std::size_t>(count), &ticket);
          } else if (count > 0) {
            framed_buffer.append(reinterpret_cast<const char*>(bytes),
                                 static_cast<std::size_t>(count));
            require(framed_buffer.size() <= 8192u,
                    "framed body event exceeded bounded transport buffer");
            for (;;) {
              const std::size_t newline = framed_buffer.find('\n');
              if (newline == std::string::npos) break;
              const std::string line = framed_buffer.substr(0u, newline);
              framed_buffer.erase(0u, newline + 1u);
              if (!line.empty())
                inject_framed_line(runtime, line, &ticket, &framed_counts);
            }
          }
        }
        emit_motors(runtime, &drained, options.framed);
      }
      stop_persistent_direct_adult(runtime);
      require(get_direct_adult_execution_authority(runtime) ==
                  AdultExecutionAuthority::host_stepped,
              "resident adult did not stop at a checkpoint boundary");
      // EOF or a signal can race the resident executor after the host has
      // published the final physical frame.  Finish that already-admitted
      // contact before checkpointing; otherwise a real tool return can be
      // stranded in host staging and the sitting cannot be resumed.
      for (std::uint32_t drain = 0u;
           drain < 8u &&
           (runtime->host_ingress_observed_head !=
                runtime->host_ingress_write_tail ||
            runtime->host_consequence_observed_head !=
                runtime->host_consequence_write_tail);
           ++drain) {
        require(run_persistent_direct_adult_epochs(runtime, 1u),
                "final published body contact did not reach the resident boundary");
      }
      require(runtime->host_ingress_observed_head ==
                      runtime->host_ingress_write_tail &&
                  runtime->host_consequence_observed_head ==
                      runtime->host_consequence_write_tail,
              "final published body contact remained staged after bounded drain");
      if (external_bytes != 0u) {
        require(run_persistent_direct_adult_epochs(runtime, 4u),
                "final body-contact settlement failed");
      }
      emit_motors(runtime, &drained, options.framed);
    }
    if (options.framed)
      require(framed_buffer.empty(), "unterminated framed body event at stop");
    if (options.once) {
      require(run_persistent_direct_adult_epochs(runtime, 4u), "final quiet span failed");
      emit_motors(runtime, &drained, options.framed);
    }
    const AdultCoreMetrics final_metrics = get_adult_core_metrics(runtime);
    const std::uint32_t end_tick = runtime->current_tick;
    persist_adult(options.checkpoint, *runtime);
    std::fprintf(stderr,
                 "DIRECT_ADULT_SITDOWN status=STOPPED birth_root=%s "
                 "canonical_species=1 compile_direct_brain=%u checkpoint_saved=1 "
                 "sensor_frames=%llu return_frames=%llu external_bytes=%llu motors=%llu "
                 "start_tick=%u end_tick=%u resident_epochs=%u persistent_bootstrap_delta=%llu "
                 "sensory_events_delta=%llu\n",
                 birth_root.c_str(), options.birth ? 1u : 0u,
                 static_cast<unsigned long long>(framed_counts.sensors),
                 static_cast<unsigned long long>(framed_counts.returns),
                 static_cast<unsigned long long>(external_bytes),
                 static_cast<unsigned long long>(drained), start_tick, end_tick,
                 end_tick - start_tick,
                 static_cast<unsigned long long>(
                     runtime->persistent_bootstrap_launches - start_bootstraps),
                 static_cast<unsigned long long>(
                     final_metrics.sensory_events_ingested -
                     start_metrics.sensory_events_ingested));
    destroy_direct_adult_runtime(runtime);
    destroy_direct_brain(&brain);
    return 0;
  } catch (const std::exception& error) {
    if (runtime != nullptr) destroy_direct_adult_runtime(runtime);
    if (brain.arena != nullptr) destroy_direct_brain(&brain);
    std::fprintf(stderr, "DIRECT_ADULT_SITDOWN status=RED %s\n", error.what());
    return 1;
  }
}
