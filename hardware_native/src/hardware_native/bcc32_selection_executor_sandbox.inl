// Selection-executor sandbox implementation.
//
// Included inside the anonymous namespace before the public evaluator entrypoint.
// This unit owns limits, Landlock/seccomp setup, and child-process execution.

bool set_limit(int resource, std::uint64_t value) {
    const rlim_t bounded = value > static_cast<std::uint64_t>(RLIM_INFINITY)
        ? RLIM_INFINITY
        : static_cast<rlim_t>(value);
    const struct rlimit limit {bounded, bounded};
    return ::setrlimit(resource, &limit) == 0;
}

int create_landlock_ruleset() {
    struct landlock_ruleset_attr ruleset {};
    ruleset.handled_access_fs =
        LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_MAKE_SYM;
    return static_cast<int>(::syscall(SYS_landlock_create_ruleset,
                                      &ruleset, sizeof(ruleset), 0u));
}

bool add_landlock_path(int ruleset,
                       const std::filesystem::path& path,
                       std::uint64_t allowed_access) {
    const int descriptor = ::open(path.c_str(), O_PATH | O_CLOEXEC);
    if (descriptor < 0) return errno == ENOENT;
    struct landlock_path_beneath_attr rule {};
    rule.allowed_access = allowed_access;
    rule.parent_fd = descriptor;
    const int result = static_cast<int>(::syscall(
        SYS_landlock_add_rule, ruleset, LANDLOCK_RULE_PATH_BENEATH, &rule, 0u));
    const int saved_errno = errno;
    ::close(descriptor);
    errno = saved_errno;
    return result == 0;
}

bool install_landlock(const std::filesystem::path& sandbox_directory) {
    const int abi = static_cast<int>(::syscall(
        SYS_landlock_create_ruleset, nullptr, 0u, LANDLOCK_CREATE_RULESET_VERSION));
    if (abi < 1) return false;
    const int ruleset = create_landlock_ruleset();
    if (ruleset < 0) return false;
    constexpr std::uint64_t read_execute =
        LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR;
    constexpr std::uint64_t read_only =
        LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;
    constexpr std::uint64_t read_file_only = LANDLOCK_ACCESS_FS_READ_FILE;
    const bool rules_added =
        add_landlock_path(ruleset, sandbox_directory, read_execute) &&
        add_landlock_path(ruleset, "/lib", read_only) &&
        add_landlock_path(ruleset, "/lib64", read_only) &&
        add_landlock_path(ruleset, "/usr/lib", read_only) &&
        add_landlock_path(ruleset, "/usr/lib64", read_only) &&
        add_landlock_path(ruleset, "/etc/ld.so.cache", read_file_only);
    if (!rules_added || ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        ::close(ruleset);
        return false;
    }
    const int result = static_cast<int>(
        ::syscall(SYS_landlock_restrict_self, ruleset, 0u));
    ::close(ruleset);
    return result == 0;
}

bool install_seccomp() {
    std::vector<sock_filter> filters;
    filters.reserve(80u);
    filters.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                               offsetof(struct seccomp_data, arch)));
    filters.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                               AUDIT_ARCH_X86_64, 1u, 0u));
    filters.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    filters.push_back(BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                               offsetof(struct seccomp_data, nr)));
    // An exact allowlist also rejects x32 syscall numbers: their high ABI bit
    // prevents them from matching any native x86-64 number below.
    const std::array<int, 30> allowed = {
        __NR_read,
        __NR_write,
        __NR_close,
        __NR_lseek,
        __NR_fstat,
        __NR_newfstatat,
        __NR_mmap,
        __NR_mprotect,
        __NR_munmap,
        __NR_brk,
        __NR_madvise,
        __NR_arch_prctl,
        __NR_set_tid_address,
        __NR_set_robust_list,
        __NR_rseq,
        __NR_futex,
        __NR_uname,
        __NR_readlink,
        __NR_fcntl,
        __NR_getppid,
        __NR_clock_nanosleep,
        __NR_sched_yield,
        __NR_rt_sigaction,
        __NR_rt_sigprocmask,
        __NR_sigaltstack,
        __NR_restart_syscall,
        __NR_rt_sigreturn,
        __NR_execve,
        __NR_exit,
        __NR_exit_group,
    };
    for (const int syscall_number : allowed) {
        filters.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                                    static_cast<std::uint32_t>(syscall_number),
                                    0u, 1u));
        filters.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    }
    filters.push_back(BPF_STMT(BPF_RET | BPF_K,
                               SECCOMP_RET_ERRNO |
                                   static_cast<std::uint32_t>(EPERM)));
    struct sock_fprog program {
        static_cast<unsigned short>(filters.size()), filters.data()
    };
    return ::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
           ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

bool set_nonblocking(int descriptor) {
    const int flags = ::fcntl(descriptor, F_GETFL, 0);
    return flags >= 0 && ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0;
}

void close_descriptor(int* descriptor) {
    if (descriptor != nullptr && *descriptor >= 0) {
        ::close(*descriptor);
        *descriptor = -1;
    }
}

class ScopedSigpipeBlock {
  public:
    ScopedSigpipeBlock() {
        ::sigemptyset(&set_);
        ::sigaddset(&set_, SIGPIPE);
        sigset_t pending{};
        prior_pending_ = ::sigpending(&pending) == 0 &&
                         ::sigismember(&pending, SIGPIPE) == 1;
        active_ = ::pthread_sigmask(SIG_BLOCK, &set_, &prior_mask_) == 0;
    }

    ~ScopedSigpipeBlock() {
        if (!active_) return;
        sigset_t pending{};
        if (!prior_pending_ && ::sigpending(&pending) == 0 &&
            ::sigismember(&pending, SIGPIPE) == 1) {
            struct timespec immediate {};
            while (::sigtimedwait(&set_, nullptr, &immediate) < 0 && errno == EINTR) {
            }
        }
        ::pthread_sigmask(SIG_SETMASK, &prior_mask_, nullptr);
    }

    [[nodiscard]] bool active() const { return active_; }

  private:
    sigset_t set_{};
    sigset_t prior_mask_{};
    bool prior_pending_ = false;
    bool active_ = false;
};

bool write_executable(const std::filesystem::path& path,
                      std::span<const std::byte> bytes,
                      std::string* error) {
    const int descriptor = ::open(path.c_str(),
                                  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                                  0700);
    if (descriptor < 0 || !write_all(descriptor, bytes) ||
        ::fchmod(descriptor, 0500) != 0 || ::fsync(descriptor) != 0) {
        const int saved_errno = errno;
        if (descriptor >= 0) ::close(descriptor);
        return fail(error, "cannot materialize BCC-32 evaluator artifact: " +
                               std::string(std::strerror(saved_errno)));
    }
    ::close(descriptor);
    return sync_directory(path.parent_path(), error);
}

bool append_from_fd(int* descriptor,
                    std::size_t maximum,
                    Bytes* output,
                    bool* open,
                    std::string* error) {
    if (descriptor == nullptr || *descriptor < 0) {
        return fail(error, "BCC-32 evaluator pipe descriptor is invalid");
    }
    std::array<std::byte, 8192> buffer{};
    for (;;) {
        const ssize_t count = ::read(*descriptor, buffer.data(), buffer.size());
        if (count > 0) {
            const std::size_t received = static_cast<std::size_t>(count);
            if (received > maximum || output->size() > maximum - received) {
                close_descriptor(descriptor);
                *open = false;
                return fail(error, "BCC-32 evaluator output exceeded its bound");
            }
            output->insert(output->end(), buffer.begin(),
                           buffer.begin() + static_cast<std::ptrdiff_t>(count));
            continue;
        }
        if (count == 0) {
            close_descriptor(descriptor);
            *open = false;
            return true;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        close_descriptor(descriptor);
        *open = false;
        return fail(error, "cannot read BCC-32 evaluator pipe");
    }
}

bool run_sandboxed_evaluator(const std::filesystem::path& executable,
                             const SelectionEvaluationLimits& limits,
                             std::span<const std::byte> input,
                             Bytes* output,
                             std::uint32_t* exit_status,
                             std::string* error) {
    if (output == nullptr || exit_status == nullptr) {
        return fail(error, "BCC-32 evaluator runner requires outputs");
    }
    int input_pipe[2] = {-1, -1};
    int output_pipe[2] = {-1, -1};
    int error_pipe[2] = {-1, -1};
    const auto close_all = [&] {
        close_descriptor(&input_pipe[0]);
        close_descriptor(&input_pipe[1]);
        close_descriptor(&output_pipe[0]);
        close_descriptor(&output_pipe[1]);
        close_descriptor(&error_pipe[0]);
        close_descriptor(&error_pipe[1]);
    };
    if (::pipe2(input_pipe, O_CLOEXEC) != 0 ||
        ::pipe2(output_pipe, O_CLOEXEC) != 0 ||
        ::pipe2(error_pipe, O_CLOEXEC) != 0) {
        close_all();
        return fail(error, "cannot create BCC-32 evaluator pipes");
    }
    const pid_t child = ::fork();
    if (child < 0) {
        close_all();
        return fail(error, "cannot fork BCC-32 evaluator process");
    }
    if (child == 0) {
        const auto sandbox_abort = [](const char* message, std::size_t size) {
            const ssize_t ignored = ::write(STDERR_FILENO, message, size);
            (void)ignored;
            _exit(126);
        };
        ::setpgid(0, 0);
        if (::dup2(input_pipe[0], STDIN_FILENO) < 0 ||
            ::dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
            ::dup2(error_pipe[1], STDERR_FILENO) < 0) {
            _exit(126);
        }
        ::close(input_pipe[0]);
        ::close(input_pipe[1]);
        ::close(output_pipe[0]);
        ::close(output_pipe[1]);
        ::close(error_pipe[0]);
        ::close(error_pipe[1]);
        if (::syscall(SYS_close_range, 3u, ~0u, 0u) != 0) {
            static constexpr char message[] = "sandbox close-range failed\n";
            sandbox_abort(message, sizeof(message) - 1u);
        }
        const std::filesystem::path directory = executable.parent_path();
        const bool limits_ok =
            set_limit(RLIMIT_CPU, limits.maximum_cpu_seconds) &&
            set_limit(RLIMIT_AS, limits.maximum_address_space_bytes) &&
            set_limit(RLIMIT_FSIZE, limits.maximum_transcript_bytes + 65'536u) &&
            set_limit(RLIMIT_NOFILE, 16u) && set_limit(RLIMIT_NPROC, 1u) &&
            set_limit(RLIMIT_CORE, 0u);
        if (!limits_ok) {
            static constexpr char message[] = "sandbox limits failed\n";
            sandbox_abort(message, sizeof(message) - 1u);
        }
        if (::chdir(directory.c_str()) != 0) {
            static constexpr char message[] = "sandbox chdir failed\n";
            sandbox_abort(message, sizeof(message) - 1u);
        }
        if (!install_landlock(directory)) {
            static constexpr char message[] = "sandbox landlock failed\n";
            sandbox_abort(message, sizeof(message) - 1u);
        }
        if (!install_seccomp()) {
            static constexpr char message[] = "sandbox seccomp failed\n";
            sandbox_abort(message, sizeof(message) - 1u);
        }
        char* const arguments[] = {const_cast<char*>("bcc32-selection-evaluator"), nullptr};
        char* const environment[] = {const_cast<char*>("LANG=C"),
                                     const_cast<char*>("LC_ALL=C"),
                                     const_cast<char*>("TZ=UTC"), nullptr};
        ::execve(executable.c_str(), arguments, environment);
        static constexpr char message[] = "evaluator exec failed\n";
        const ssize_t ignored =
            ::write(STDERR_FILENO, message, sizeof(message) - 1u);
        (void)ignored;
        _exit(127);
    }

    ::setpgid(child, child);
    close_descriptor(&input_pipe[0]);
    close_descriptor(&output_pipe[1]);
    close_descriptor(&error_pipe[1]);
    const auto terminate_child = [&](int* status) {
        if (::kill(-child, SIGKILL) != 0) ::kill(child, SIGKILL);
        while (::waitpid(child, status, 0) < 0 && errno == EINTR) {
        }
    };
    ScopedSigpipeBlock sigpipe;
    if (!sigpipe.active()) {
        terminate_child(nullptr);
        close_all();
        return fail(error, "cannot block SIGPIPE for BCC-32 evaluator transport");
    }
    if (!set_nonblocking(input_pipe[1]) || !set_nonblocking(output_pipe[0]) ||
        !set_nonblocking(error_pipe[0])) {
        terminate_child(nullptr);
        close_all();
        return fail(error, "cannot configure BCC-32 evaluator pipes");
    }

    output->clear();
    Bytes stderr_bytes;
    std::size_t input_offset = 0u;
    bool input_open = true;
    bool output_open = true;
    bool error_open = true;
    bool child_exited = false;
    int child_status = 0;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(limits.maximum_wall_milliseconds);

    while (!child_exited || output_open || error_open) {
        if (std::chrono::steady_clock::now() >= deadline) {
            terminate_child(&child_status);
            close_all();
            return fail(error, "BCC-32 evaluator exceeded its precommitted wall time");
        }
        std::array<struct pollfd, 3> descriptors{{
            {input_open ? input_pipe[1] : -1, POLLOUT, 0},
            {output_open ? output_pipe[0] : -1, POLLIN | POLLHUP, 0},
            {error_open ? error_pipe[0] : -1, POLLIN | POLLHUP, 0},
        }};
        const int polled = ::poll(descriptors.data(), descriptors.size(), 25);
        if (polled < 0 && errno != EINTR) {
            terminate_child(&child_status);
            close_all();
            return fail(error, "cannot poll BCC-32 evaluator process");
        }
        if (input_open && (descriptors[0].revents & (POLLERR | POLLHUP))) {
            close_descriptor(&input_pipe[1]);
            input_open = false;
        } else if (input_open && (descriptors[0].revents & POLLOUT)) {
            const ssize_t count = ::write(input_pipe[1], input.data() + input_offset,
                                          input.size() - input_offset);
            if (count > 0) input_offset += static_cast<std::size_t>(count);
            if (input_offset == input.size() ||
                (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) {
                close_descriptor(&input_pipe[1]);
                input_open = false;
            }
        }
        if (output_open &&
            !append_from_fd(&output_pipe[0], limits.maximum_transcript_bytes,
                            output, &output_open, error)) {
            terminate_child(&child_status);
            close_all();
            return false;
        }
        if (error_open &&
            !append_from_fd(&error_pipe[0], 65'536u, &stderr_bytes,
                            &error_open, error)) {
            terminate_child(&child_status);
            close_all();
            return false;
        }
        if (!child_exited) {
            const pid_t waited = ::waitpid(child, &child_status, WNOHANG);
            child_exited = waited == child;
            if (waited < 0 && errno != EINTR) {
                terminate_child(&child_status);
                close_all();
                return fail(error, "cannot reap BCC-32 evaluator process");
            }
        }
    }
    close_all();
    if (!child_exited) {
        while (::waitpid(child, &child_status, 0) < 0 && errno == EINTR) {
        }
    }
    if (!WIFEXITED(child_status)) {
        return fail(error, "BCC-32 evaluator terminated outside the protocol");
    }
    *exit_status = static_cast<std::uint32_t>(WEXITSTATUS(child_status));
    if (*exit_status != 0u) {
        const std::string stderr_text(
            reinterpret_cast<const char*>(stderr_bytes.data()), stderr_bytes.size());
        return fail(error, "BCC-32 evaluator exited unsuccessfully with status " +
                               std::to_string(*exit_status) + ": " + stderr_text);
    }
    return true;
}
