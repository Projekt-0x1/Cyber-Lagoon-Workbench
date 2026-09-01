// github #1303 -- life.canonical_sit_body.
//
// This contract exercises the production hardware_native/tools/
// direct_adult_sitdown.cu binary itself as a subprocess, not an in-process
// fixture genome. The law names direct_adult_sitdown as the live membrane
// body for one designated Life-Function adult. The operator supplies that
// Adult's retained checkpoint; this contract clones it and only resumes the
// clone. It must never compile a substitute Adult. Driving the real client/
// transport binary is the only way to measure that property honestly -- an
// in-process helper that re-implements a sitting would be testing the helper,
// not the sit-down body.
//
// Proves, against the real binary:
//   1. --new refuses an existing checkpoint before compilation (birth-once
//      guard), without ever creating another canonical Adult.
//   2. --resume never recompiles (compile_direct_brain=0) and reproduces the
//      retained birth_root across three resumptions.
//   3. A live (non---once) --resume session reaches status=RUNNING before
//      touching stdin, accepts real raw sensor contact through the terminal
//      membrane, and stops cleanly on EOF with status=STOPPED.
//   4. The checkpoint file is replaced atomically (no dangling ".next"
//      temporary survives any sitting) and its persisted bytes actually
//      advance once live contact has been processed.
//   5. --resume rejects environment flags, which belong to birth only, and
//      the operator-supplied retained checkpoint remains byte-identical.
//
// No helper mind, answer bank, or host-selected semantic route: every check
// below is a structural read of the binary's own stderr status protocol and
// checkpoint file, never a judgment about what the organism "meant".

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& what) {
  if (!condition)
    throw std::runtime_error(what);
}

std::string read_text(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open " + path.string());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

fs::path own_executable_directory() {
  std::error_code error;
  const fs::path resolved = fs::canonical("/proc/self/exe", error);
  require(!error, "cannot resolve own executable path");
  return resolved.parent_path();
}

fs::path locate_sitdown_binary(const fs::path& directory) {
  // The GPU contract lock wrapper renames the compiled binary to
  // "<name>.real" and installs a thin lock-acquiring script in its place
  // (cmake/bcc32_gpu_contract_lock_enforcement.cmake). This contract already
  // owns exclusive GPU access for its own run, so it drives the real
  // sit-down binary directly rather than nesting a second lock acquisition
  // inside the one it is already holding.
  const fs::path real_path = directory / "direct_adult_sitdown.real";
  if (fs::exists(real_path))
    return real_path;
  const fs::path plain_path = directory / "direct_adult_sitdown";
  require(fs::exists(plain_path), "direct_adult_sitdown was not built next to this contract");
  return plain_path;
}

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

struct RunResult {
  int exit_code = -1;
  std::string stdout_text;
  std::string stderr_text;
};

struct Options {
  fs::path checkpoint;
  fs::path sitdown;
};

Options parse_options(int argc, char** argv) {
  Options options{};
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    require(index + 1 < argc, "missing value for " + arg);
    if (arg == "--checkpoint") {
      options.checkpoint = argv[++index];
    } else if (arg == "--sitdown") {
      options.sitdown = argv[++index];
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

RunResult run_sitdown(const fs::path& binary, const std::vector<std::string>& args,
                      const std::string& stdin_bytes, const fs::path& scratch,
                      const std::string& label) {
  const fs::path stdin_path = scratch / (label + ".stdin");
  const fs::path stdout_path = scratch / (label + ".stdout");
  const fs::path stderr_path = scratch / (label + ".stderr");
  {
    std::ofstream stdin_file(stdin_path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(stdin_file), "cannot stage stdin fixture for " + label);
    stdin_file.write(stdin_bytes.data(), static_cast<std::streamsize>(stdin_bytes.size()));
  }
  std::string command = shell_quote(binary.string());
  for (const std::string& arg : args)
    command += " " + shell_quote(arg);
  command += " < " + shell_quote(stdin_path.string());
  command += " > " + shell_quote(stdout_path.string());
  command += " 2> " + shell_quote(stderr_path.string());
  const int status = std::system(command.c_str());
  RunResult result{};
  result.exit_code = status;
  result.stdout_text = read_text(stdout_path);
  result.stderr_text = read_text(stderr_path);
  return result;
}

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

std::vector<std::string> extract_all(const std::string& text, const std::string& key) {
  std::vector<std::string> matches;
  std::size_t cursor = 0u;
  for (;;) {
    const std::size_t pos = text.find(key, cursor);
    if (pos == std::string::npos)
      break;
    const std::size_t start = pos + key.size();
    std::size_t end = text.find_first_of(" \n", start);
    if (end == std::string::npos)
      end = text.size();
    matches.push_back(text.substr(start, end - start));
    cursor = end;
  }
  return matches;
}

std::string extract_one(const std::string& text, const std::string& key) {
  const std::vector<std::string> matches = extract_all(text, key);
  require(matches.size() == 1u, "expected exactly one '" + key + "' in: " + text);
  return matches.front();
}

}  // namespace

int main(int argc, char** argv) {
  Options options{};
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "DIRECT_LIFE_CANONICAL_SIT_BODY status=RED %s\n", error.what());
    return 1;
  }
  if (options.checkpoint.empty()) {
    std::printf(
        "SKIP cuda_direct_life_canonical_sit_body_contract requires "
        "--checkpoint for the designated retained Adult\n");
    return 77;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP cuda_direct_life_canonical_sit_body_contract no CUDA device\n");
    return 77;
  }

  const fs::path own_dir = own_executable_directory();
  const fs::path scratch =
      fs::temp_directory_path() / fs::path("life_canonical_sit_body_" + std::to_string(::getpid()));
  try {
    fs::create_directories(scratch);
    const fs::path sitdown =
        options.sitdown.empty() ? locate_sitdown_binary(own_dir) : options.sitdown;
    require(fs::is_regular_file(sitdown), "sit-down binary does not exist: " + sitdown.string());
    require(fs::is_regular_file(options.checkpoint),
            "designated checkpoint does not exist: " + options.checkpoint.string());
    const std::vector<std::uint8_t> designated_bytes = read_bytes(options.checkpoint);
    require(!designated_bytes.empty(), "designated checkpoint is empty");
    const fs::path checkpoint = scratch / "adult.checkpoint";
    const fs::path checkpoint_next = scratch / "adult.checkpoint.next";
    fs::copy_file(options.checkpoint, checkpoint, fs::copy_options::overwrite_existing);

    // Establish the retained identity by resuming a disposable clone. No
    // --new path is ever allowed to create a substitute life here.
    const RunResult initial_resume = run_sitdown(
        sitdown, {"--resume", checkpoint.string(), "--once"}, "", scratch, "resume_initial");
    require(initial_resume.exit_code == 0, "initial resume failed: " + initial_resume.stderr_text);
    require(contains(initial_resume.stderr_text, "status=STOPPED") &&
                contains(initial_resume.stderr_text, "compile_direct_brain=0") &&
                contains(initial_resume.stderr_text, "canonical_species=1"),
            "initial resume did not continue the retained canonical Adult: " +
                initial_resume.stderr_text);
    const std::string birth_root = extract_one(initial_resume.stderr_text, "birth_root=");
    require(!fs::exists(checkpoint_next),
            "initial resume left a dangling atomic-rename .next temporary");
    const std::vector<std::uint8_t> bytes_after_initial_resume = read_bytes(checkpoint);

    // Birth-once guard: --new refuses to run compile_direct_brain a second
    // time over an existing checkpoint path.
    const RunResult rebirth_guard = run_sitdown(sitdown,
                                                {"--new", checkpoint.string(),
                                                 "--environment-seed", "0x3"},
                                                "", scratch, "rebirth_guard");
    require(rebirth_guard.exit_code != 0,
            "--new accepted an existing checkpoint path instead of refusing to "
            "overwrite an existing life");
    require(contains(rebirth_guard.stderr_text, "status=RED"),
            "birth-once guard did not fail closed: " + rebirth_guard.stderr_text);

    // 2 & 3. Live resume: reaches status=RUNNING before consuming stdin,
    // accepts real raw sensor contact, stops cleanly on EOF. compile_direct_
    // brain=0 and the same birth_root prove no recompile happened.
    const std::string raw_contact = "SIT-DOWN-BODY-CONTACT";
    const RunResult live_resume = run_sitdown(sitdown, {"--resume", checkpoint.string()},
                                              raw_contact, scratch, "resume_live");
    require(live_resume.exit_code == 0,
            "live resume did not stop cleanly on stdin EOF: " + live_resume.stderr_text);
    require(contains(live_resume.stderr_text, "status=RUNNING") &&
                contains(live_resume.stderr_text, "compile_direct_brain=0"),
            "live resume never reached RUNNING continuity without recompiling: " +
                live_resume.stderr_text);
    require(contains(live_resume.stderr_text, "status=STOPPED") &&
                contains(live_resume.stderr_text, "compile_direct_brain=0") &&
                contains(live_resume.stderr_text, "checkpoint_saved=1"),
            "live resume did not persist a clean stop: " + live_resume.stderr_text);
    const std::vector<std::string> live_roots = extract_all(live_resume.stderr_text, "birth_root=");
    require(live_roots.size() == 2u, "expected one RUNNING and one STOPPED birth_root line, saw " +
                                         std::to_string(live_roots.size()));
    require(live_roots[0] == birth_root && live_roots[1] == birth_root,
            "resume changed the canonical birth_root instead of continuing it");
    require(!fs::exists(checkpoint_next),
            "live resume left a dangling atomic-rename .next temporary");
    const std::vector<std::uint8_t> bytes_after_live_resume = read_bytes(checkpoint);
    require(bytes_after_live_resume != bytes_after_initial_resume,
            "checkpoint bytes did not advance after live membrane contact");

    // A second, independent resumption still never recompiles and still
    // reproduces the identical birth_root.
    const RunResult resume_again = run_sitdown(sitdown, {"--resume", checkpoint.string(), "--once"},
                                               "", scratch, "resume_again");
    require(resume_again.exit_code == 0, "second resume failed: " + resume_again.stderr_text);
    require(contains(resume_again.stderr_text, "status=STOPPED") &&
                contains(resume_again.stderr_text, "compile_direct_brain=0"),
            "second resume recompiled instead of continuing: " + resume_again.stderr_text);
    const std::string root_after_second_resume =
        extract_one(resume_again.stderr_text, "birth_root=");
    require(root_after_second_resume == birth_root,
            "second independent resumption diverged from the canonical "
            "birth_root");
    require(!fs::exists(checkpoint_next),
            "second resume left a dangling atomic-rename .next temporary");

    // 5. Environment belongs to birth only; --resume must reject it.
    const RunResult resume_environment_guard = run_sitdown(
        sitdown, {"--resume", checkpoint.string(), "--environment-seed", "0x1"},
        "", scratch, "resume_environment_guard");
    require(resume_environment_guard.exit_code != 0,
            "--resume accepted a birth-only environment flag");
    require(contains(resume_environment_guard.stderr_text, "status=RED"),
            "resume environment guard did not fail closed: " +
                resume_environment_guard.stderr_text);

    require(read_bytes(options.checkpoint) == designated_bytes,
            "contract mutated the operator-supplied designated checkpoint");

    std::printf(
        "DIRECT_LIFE_CANONICAL_SIT_BODY status=GREEN birth_root=%s "
        "births=0 resumes=3 recompiles=0 checkpoint_advanced=1 "
        "designated_checkpoint_unchanged=1\n",
        birth_root.c_str());
    fs::remove_all(scratch);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "DIRECT_LIFE_CANONICAL_SIT_BODY status=RED %s\n", error.what());
    std::error_code ignored;
    fs::remove_all(scratch, ignored);
    return 1;
  }
}
