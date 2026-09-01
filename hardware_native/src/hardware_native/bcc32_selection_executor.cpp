#include "bcc32_selection_executor.hpp"
#include "bcc32_transition.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <elf.h>
#include <fcntl.h>
#include <limits>
#include <map>
#include <poll.h>
#include <pthread.h>
#include <set>
#include <string_view>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>

namespace substrate::bcc32 {
namespace {

using Bytes = std::vector<std::byte>;

constexpr std::size_t kMaximumSelectionObjects = 1u << 20u;
constexpr std::uint64_t kMaximumObjectBytes = 1ull << 30u;
constexpr std::string_view kOperationDomain =
    "0x1/bcc32/selection-operation/v1";
constexpr std::string_view kEvaluatorEnvelopeDomain =
    "0x1/bcc32/selection-evaluator-envelope/v1";
constexpr std::string_view kEvaluatorReceiptDomain =
    "0x1/bcc32/selection-evaluator-receipt/v1";
constexpr std::string_view kOperationResolutionDomain =
    "0x1/bcc32/selection-operation-resolution/v1";
constexpr std::string_view kExecutionPlanDomain =
    "0x1/bcc32/selection-execution-plan/v1";
constexpr std::string_view kExecutionAdmissionDomain =
    "0x1/bcc32/selection-execution-admission/v1";
constexpr std::string_view kOperationRequestPointerDomain =
    "0x1/bcc32/selection-operation-request-pointer/v1";
constexpr std::string_view kOperationResolutionPointerDomain =
    "0x1/bcc32/selection-operation-resolution-pointer/v1";
constexpr std::string_view kOperationAdmissionPointerDomain =
    "0x1/bcc32/selection-operation-admission-pointer/v1";
constexpr std::string_view kSandboxProfileDomain =
    "0x1/bcc32/evaluator-sandbox/static-elf-open-deny-landlock-seccomp-v2";

bool fail(std::string* error, std::string message) {
    if (error != nullptr) *error = std::move(message);
    return false;
}

void append_u32(Bytes* bytes, std::uint32_t value) {
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
        bytes->push_back(static_cast<std::byte>((value >> shift) & 0xffu));
    }
}

void append_u64(Bytes* bytes, std::uint64_t value) {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
        bytes->push_back(static_cast<std::byte>((value >> shift) & 0xffu));
    }
}

void append_raw(Bytes* bytes, std::span<const std::byte> value) {
    bytes->insert(bytes->end(), value.begin(), value.end());
}

void append_domain(Bytes* bytes, std::string_view domain) {
    append_u64(bytes, domain.size());
    append_raw(bytes,
               {reinterpret_cast<const std::byte*>(domain.data()), domain.size()});
}

void append_address(Bytes* bytes, const ContentAddress& address) {
    append_raw(bytes, {reinterpret_cast<const std::byte*>(address.digest.data()),
                       address.digest.size()});
    append_u64(bytes, address.byte_count);
}

class Reader {
  public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    bool take_u32(std::uint32_t* value) {
        if (value == nullptr || cursor_ + 4u > bytes_.size()) return false;
        *value = 0u;
        for (std::uint32_t index = 0u; index < 4u; ++index) {
            *value |= static_cast<std::uint32_t>(bytes_[cursor_ + index]) <<
                      (index * 8u);
        }
        cursor_ += 4u;
        return true;
    }

    bool take_u64(std::uint64_t* value) {
        if (value == nullptr || cursor_ + 8u > bytes_.size()) return false;
        *value = 0u;
        for (std::uint32_t index = 0u; index < 8u; ++index) {
            *value |= static_cast<std::uint64_t>(bytes_[cursor_ + index]) <<
                      (index * 8u);
        }
        cursor_ += 8u;
        return true;
    }

    bool take_domain(std::string_view expected) {
        std::uint64_t size = 0u;
        if (!take_u64(&size) || size != expected.size() ||
            cursor_ + size > bytes_.size()) {
            return false;
        }
        const auto expected_bytes = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(expected.data()), expected.size());
        if (!std::equal(expected_bytes.begin(), expected_bytes.end(),
                        bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_))) {
            return false;
        }
        cursor_ += static_cast<std::size_t>(size);
        return true;
    }

    bool take_address(ContentAddress* address) {
        if (address == nullptr || cursor_ + address->digest.size() + 8u > bytes_.size()) {
            return false;
        }
        std::memcpy(address->digest.data(), bytes_.data() + cursor_,
                    address->digest.size());
        cursor_ += address->digest.size();
        return take_u64(&address->byte_count);
    }

    bool take_bytes(std::size_t size, Bytes* bytes) {
        if (bytes == nullptr || cursor_ + size > bytes_.size()) return false;
        bytes->assign(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                      bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_ + size));
        cursor_ += size;
        return true;
    }

    [[nodiscard]] bool done() const { return cursor_ == bytes_.size(); }

  private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_ = 0u;
};

bool take_count(Reader* reader,
                std::size_t maximum,
                std::size_t element_bytes,
                std::size_t* count) {
    std::uint64_t encoded = 0u;
    if (reader == nullptr || count == nullptr || !reader->take_u64(&encoded) ||
        encoded > maximum ||
        (element_bytes != 0u && encoded > kMaximumObjectBytes / element_bytes)) {
        return false;
    }
    *count = static_cast<std::size_t>(encoded);
    return true;
}

bool canonical_equal(std::span<const std::byte> left,
                     std::span<const std::byte> right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

bool canonical_text_equal(std::span<const std::byte> bytes,
                          std::string_view expected) {
    return canonical_equal(
        bytes,
        {reinterpret_cast<const std::byte*>(expected.data()), expected.size()});
}

bool validate_static_evaluator_artifact(std::span<const std::byte> bytes,
                                        std::string* error) {
    if (bytes.size() < sizeof(Elf64_Ehdr)) {
        return fail(error, "BCC-32 evaluator artifact is not an ELF64 executable");
    }
    Elf64_Ehdr header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 ||
        header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB ||
        header.e_machine != EM_X86_64 ||
        (header.e_type != ET_EXEC && header.e_type != ET_DYN) ||
        header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phnum == 0u ||
        header.e_phnum > 4096u ||
        header.e_phoff > bytes.size() ||
        header.e_phnum >
            (bytes.size() - static_cast<std::size_t>(header.e_phoff)) /
                sizeof(Elf64_Phdr)) {
        return fail(error, "BCC-32 evaluator artifact has an unsupported ELF contract");
    }
    bool has_executable_load = false;
    for (std::size_t index = 0u; index < header.e_phnum; ++index) {
        Elf64_Phdr program{};
        const std::size_t offset = static_cast<std::size_t>(header.e_phoff) +
                                   index * sizeof(Elf64_Phdr);
        std::memcpy(&program, bytes.data() + offset, sizeof(program));
        if (program.p_type == PT_INTERP) {
            return fail(error,
                        "BCC-32 evaluator artifact must be statically linked");
        }
        if (program.p_type == PT_LOAD) {
            has_executable_load = has_executable_load ||
                                  (program.p_flags & PF_X) != 0u;
            if ((program.p_flags & (PF_W | PF_X)) == (PF_W | PF_X)) {
                return fail(error,
                            "BCC-32 evaluator ELF cannot contain a writable executable segment");
            }
            if (program.p_filesz > program.p_memsz) {
                return fail(error,
                            "BCC-32 evaluator ELF load segment has invalid extents");
            }
        }
        if (program.p_filesz != 0u &&
            (program.p_offset > bytes.size() ||
             program.p_filesz > bytes.size() -
                                      static_cast<std::size_t>(program.p_offset))) {
            return fail(error, "BCC-32 evaluator ELF segment exceeds the artifact");
        }
    }
    if (!has_executable_load) {
        return fail(error,
                    "BCC-32 evaluator ELF has no executable load segment");
    }
    return true;
}

bool valid_continuation_action(SelectionAction action) {
    return action == SelectionAction::preserve_candidate ||
           action == SelectionAction::migrate_intact_candidate;
}

bool validate_receipt_shape(const SelectionEvaluatorReceipt& receipt,
                            std::string* error) {
    if (receipt.schema_version != SelectionEvaluatorReceipt::kSchemaVersion ||
        receipt.exit_status != 0u ||
        !is_valid_content_address(receipt.evaluation_request) ||
        !is_valid_content_address(receipt.evaluator_artifact) ||
        !is_valid_content_address(receipt.input_envelope) ||
        !is_valid_content_address(receipt.output_bytes) ||
        !is_valid_content_address(receipt.sandbox_profile) ||
        !is_valid_content_address(receipt.transcript)) {
        return fail(error, "BCC-32 evaluator receipt is malformed");
    }
    return true;
}

bool validate_resolution_shape(const SelectionOperationResolution& resolution,
                               std::string* error) {
    if (resolution.schema_version != SelectionOperationResolution::kSchemaVersion ||
        !is_valid_content_address(resolution.evaluation_request) ||
        !is_valid_content_address(resolution.transcript) ||
        !is_valid_content_address(resolution.evaluator_receipt)) {
        return fail(error, "BCC-32 selection operation resolution is malformed");
    }
    return true;
}

bool valid_artifact_kind(ArtifactKind kind) {
    return kind == ArtifactKind::propagule_capsule ||
           kind == ArtifactKind::adult_continuity_checkpoint ||
           kind == ArtifactKind::cultural_capsule;
}

bool validate_plan_shape(const SelectionExecutionPlan& plan, std::string* error) {
    if (plan.schema_version != SelectionExecutionPlan::kSchemaVersion ||
        !is_valid_content_address(plan.evaluation_request) ||
        !is_valid_content_address(plan.operation_resolution) ||
        !is_valid_content_address(plan.transcript) ||
        !is_valid_content_address(plan.scheduler_protocol) ||
        !valid_artifact_kind(plan.artifact_kind) || plan.tasks.empty() ||
        plan.tasks.size() > kMaximumSelectionObjects ||
        plan.mating_assignments.size() > kMaximumSelectionObjects) {
        return fail(error, "BCC-32 selection execution plan header is malformed");
    }
    std::uint64_t compute = 0u;
    for (std::size_t index = 0u; index < plan.tasks.size(); ++index) {
        const SelectionExecutionTask& task = plan.tasks[index];
        if (task.population_slot != index || !valid_continuation_action(task.action) ||
            !is_valid_content_address(task.candidate_root) ||
            !is_valid_content_address(task.task_head) ||
            compute > std::numeric_limits<std::uint64_t>::max() -
                          task.compute_supersteps) {
            return fail(error, "BCC-32 selection execution task is malformed");
        }
        compute += task.compute_supersteps;
    }
    if (compute != plan.total_compute_supersteps) {
        return fail(error, "BCC-32 selection execution plan compute sum disagrees");
    }
    SelectionMatingAssignment previous{};
    bool have_previous = false;
    for (const SelectionMatingAssignment& assignment : plan.mating_assignments) {
        if (assignment.first_population_slot >= plan.tasks.size() ||
            assignment.second_population_slot >= plan.tasks.size() ||
            assignment.first_population_slot >= assignment.second_population_slot ||
            plan.tasks[assignment.first_population_slot].candidate_index !=
                assignment.first_candidate_index ||
            plan.tasks[assignment.second_population_slot].candidate_index !=
                assignment.second_candidate_index ||
            (have_previous &&
             (assignment.first_population_slot < previous.first_population_slot ||
              (assignment.first_population_slot == previous.first_population_slot &&
               assignment.second_population_slot <= previous.second_population_slot)))) {
            return fail(error, "BCC-32 selection mating assignment is malformed");
        }
        previous = assignment;
        have_previous = true;
    }
    return true;
}

bool validate_admission_shape(const SelectionExecutionAdmission& admission,
                              std::string* error) {
    if (admission.schema_version != SelectionExecutionAdmission::kSchemaVersion ||
        !is_valid_content_address(admission.evaluation_request) ||
        !is_valid_content_address(admission.operation_resolution) ||
        !is_valid_content_address(admission.execution_plan)) {
        return fail(error, "BCC-32 selection execution admission is malformed");
    }
    return true;
}

Bytes canonical_pointer(std::string_view domain, const ContentAddress& identity) {
    Bytes bytes;
    append_domain(&bytes, domain);
    append_address(&bytes, identity);
    return bytes;
}

bool decode_pointer(std::span<const std::byte> bytes,
                    std::string_view domain,
                    ContentAddress* identity,
                    std::string* error) {
    Reader reader(bytes);
    ContentAddress decoded{};
    if (identity == nullptr || !reader.take_domain(domain) ||
        !reader.take_address(&decoded) || !reader.done() ||
        !is_valid_content_address(decoded) ||
        !canonical_equal(bytes, canonical_pointer(domain, decoded))) {
        return fail(error, "BCC-32 selection operation pointer is malformed");
    }
    *identity = decoded;
    return true;
}

std::filesystem::path object_path(const std::filesystem::path& repository,
                                  std::string_view directory,
                                  const ContentAddress& identity) {
    return repository / "objects" / directory / hash_hex(identity.digest);
}

std::filesystem::path operation_path(const std::filesystem::path& repository,
                                     const ContentAddress& operation) {
    return repository / "selection-operations" / hash_hex(operation.digest);
}

bool sync_directory(const std::filesystem::path& path, std::string* error) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor < 0) {
        return fail(error, "cannot open BCC-32 selection directory for sync: " +
                               std::string(std::strerror(errno)));
    }
    const int result = ::fsync(descriptor);
    const int saved_errno = errno;
    ::close(descriptor);
    if (result != 0) {
        return fail(error, "cannot sync BCC-32 selection directory: " +
                               std::string(std::strerror(saved_errno)));
    }
    return true;
}

bool read_file(const std::filesystem::path& path,
               std::uint64_t maximum_bytes,
               Bytes* bytes,
               std::string* error) {
    if (bytes == nullptr) return fail(error, "BCC-32 selection read requires output");
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        return fail(error, "cannot open BCC-32 selection object: " +
                               std::string(std::strerror(errno)));
    }
    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) > maximum_bytes) {
        ::close(descriptor);
        return fail(error, "BCC-32 selection object has an invalid file shape");
    }
    bytes->assign(static_cast<std::size_t>(status.st_size), std::byte{});
    std::size_t offset = 0u;
    while (offset < bytes->size()) {
        const ssize_t count = ::read(descriptor, bytes->data() + offset,
                                     bytes->size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            ::close(descriptor);
            return fail(error, "cannot read complete BCC-32 selection object");
        }
        offset += static_cast<std::size_t>(count);
    }
    ::close(descriptor);
    return true;
}

bool write_all(int descriptor, std::span<const std::byte> bytes) {
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset,
                                      bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

bool write_immutable(const std::filesystem::path& path,
                     std::span<const std::byte> bytes,
                     std::string* error) {
    Bytes existing;
    if (std::filesystem::exists(path)) {
        if (!read_file(path, kMaximumObjectBytes, &existing, error)) return false;
        return canonical_equal(existing, bytes) ||
               fail(error, "BCC-32 immutable selection object collision");
    }
    static std::atomic<std::uint64_t> serial{0u};
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(static_cast<unsigned long>(::getpid())) +
        "." + std::to_string(serial.fetch_add(1u));
    const int descriptor = ::open(temporary.c_str(),
                                  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                                  0600);
    if (descriptor < 0) {
        return fail(error, "cannot create BCC-32 immutable selection temporary: " +
                               std::string(std::strerror(errno)));
    }
    const bool wrote = write_all(descriptor, bytes) && ::fsync(descriptor) == 0;
    const int saved_errno = errno;
    ::close(descriptor);
    if (!wrote) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return fail(error, "cannot write BCC-32 immutable selection object: " +
                               std::string(std::strerror(saved_errno)));
    }
    if (::link(temporary.c_str(), path.c_str()) != 0) {
        const int link_errno = errno;
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        if (link_errno == EEXIST) {
            if (!read_file(path, kMaximumObjectBytes, &existing, error)) return false;
            return canonical_equal(existing, bytes) ||
                   fail(error, "BCC-32 immutable selection object collision");
        }
        return fail(error, "cannot publish BCC-32 immutable selection object: " +
                               std::string(std::strerror(link_errno)));
    }
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    if (remove_error) {
        return fail(error, "cannot remove BCC-32 selection temporary");
    }
    return sync_directory(path.parent_path(), error);
}

bool ensure_selection_repository(const std::filesystem::path& repository,
                                 std::string* error) {
    static constexpr std::array<std::string_view, 6> directories = {
        "selection-blobs",
        "selection-evaluator-receipts",
        "selection-operation-resolutions",
        "selection-execution-plans",
        "selection-execution-admissions",
        "selection-evaluation-requests",
    };
    std::error_code filesystem_error;
    std::filesystem::create_directories(repository / "objects", filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 selection object store");
    }
    for (const std::string_view directory : directories) {
        std::filesystem::create_directories(repository / "objects" / directory,
                                            filesystem_error);
        if (filesystem_error) {
            return fail(error, "cannot create BCC-32 selection object directory");
        }
    }
    std::filesystem::create_directories(repository / "selection-operations",
                                        filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 selection operation store");
    }
    for (const std::string_view directory : directories) {
        if (!sync_directory(repository / "objects" / directory, error)) return false;
    }
    return sync_directory(repository / "objects", error) &&
           sync_directory(repository / "selection-operations", error) &&
           sync_directory(repository, error);
}

bool ensure_operation_directory(const std::filesystem::path& repository,
                                const ContentAddress& operation,
                                std::string* error) {
    if (!ensure_selection_repository(repository, error)) return false;
    std::error_code filesystem_error;
    const std::filesystem::path path = operation_path(repository, operation);
    std::filesystem::create_directories(path, filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 selection operation directory");
    }
    return sync_directory(path, error) &&
           sync_directory(repository / "selection-operations", error);
}

template <typename Value, typename Decode>
bool read_typed_object(const std::filesystem::path& path,
                       const ContentAddress& identity,
                       Value* value,
                       Decode decode,
                       std::string* error) {
    if (value == nullptr || !is_valid_content_address(identity)) {
        return fail(error, "BCC-32 typed selection-object lookup is invalid");
    }
    Bytes bytes;
    if (!read_file(path, kMaximumObjectBytes, &bytes, error) ||
        content_address(bytes) != identity || !decode(bytes, value, error)) {
        return fail(error, "BCC-32 typed selection object is missing or corrupt");
    }
    return true;
}

}  // namespace

ContentAddress selection_operation_identity(
    const SelectionEvaluationRequest& request) {
    Bytes bytes;
    append_domain(&bytes, kOperationDomain);
    append_address(&bytes, request.operation_predecessor);
    append_u64(&bytes, request.selection_round);
    return content_address(bytes);
}

ContentAddress canonical_selection_sandbox_profile() {
    return content_address({reinterpret_cast<const std::byte*>(kSandboxProfileDomain.data()),
                            kSandboxProfileDomain.size()});
}

std::vector<std::byte> canonical_selection_evaluator_receipt(
    const SelectionEvaluatorReceipt& receipt) {
    Bytes bytes;
    append_domain(&bytes, kEvaluatorReceiptDomain);
    append_u32(&bytes, receipt.schema_version);
    append_address(&bytes, receipt.evaluation_request);
    append_address(&bytes, receipt.evaluator_artifact);
    append_address(&bytes, receipt.input_envelope);
    append_address(&bytes, receipt.output_bytes);
    append_address(&bytes, receipt.sandbox_profile);
    append_address(&bytes, receipt.transcript);
    append_u32(&bytes, receipt.exit_status);
    return bytes;
}

ContentAddress selection_evaluator_receipt_address(
    const SelectionEvaluatorReceipt& receipt) {
    return content_address(canonical_selection_evaluator_receipt(receipt));
}

bool decode_selection_evaluator_receipt(std::span<const std::byte> bytes,
                                        SelectionEvaluatorReceipt* receipt,
                                        std::string* error) {
    Reader reader(bytes);
    SelectionEvaluatorReceipt decoded{};
    if (receipt == nullptr || !reader.take_domain(kEvaluatorReceiptDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !reader.take_address(&decoded.evaluation_request) ||
        !reader.take_address(&decoded.evaluator_artifact) ||
        !reader.take_address(&decoded.input_envelope) ||
        !reader.take_address(&decoded.output_bytes) ||
        !reader.take_address(&decoded.sandbox_profile) ||
        !reader.take_address(&decoded.transcript) ||
        !reader.take_u32(&decoded.exit_status) || !reader.done() ||
        !validate_receipt_shape(decoded, error) ||
        !canonical_equal(bytes, canonical_selection_evaluator_receipt(decoded))) {
        return fail(error, "BCC-32 evaluator receipt is not canonical");
    }
    *receipt = decoded;
    return true;
}

std::vector<std::byte> canonical_selection_operation_resolution(
    const SelectionOperationResolution& resolution) {
    Bytes bytes;
    append_domain(&bytes, kOperationResolutionDomain);
    append_u32(&bytes, resolution.schema_version);
    append_address(&bytes, resolution.evaluation_request);
    append_address(&bytes, resolution.transcript);
    append_address(&bytes, resolution.evaluator_receipt);
    return bytes;
}

ContentAddress selection_operation_resolution_address(
    const SelectionOperationResolution& resolution) {
    return content_address(canonical_selection_operation_resolution(resolution));
}

bool decode_selection_operation_resolution(
    std::span<const std::byte> bytes,
    SelectionOperationResolution* resolution,
    std::string* error) {
    Reader reader(bytes);
    SelectionOperationResolution decoded{};
    if (resolution == nullptr || !reader.take_domain(kOperationResolutionDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !reader.take_address(&decoded.evaluation_request) ||
        !reader.take_address(&decoded.transcript) ||
        !reader.take_address(&decoded.evaluator_receipt) || !reader.done() ||
        !validate_resolution_shape(decoded, error) ||
        !canonical_equal(bytes, canonical_selection_operation_resolution(decoded))) {
        return fail(error, "BCC-32 selection resolution is not canonical");
    }
    *resolution = decoded;
    return true;
}

std::vector<std::byte> canonical_selection_execution_plan(
    const SelectionExecutionPlan& plan) {
    Bytes bytes;
    append_domain(&bytes, kExecutionPlanDomain);
    append_u32(&bytes, plan.schema_version);
    append_address(&bytes, plan.evaluation_request);
    append_address(&bytes, plan.operation_resolution);
    append_address(&bytes, plan.transcript);
    append_address(&bytes, plan.scheduler_protocol);
    append_u32(&bytes, static_cast<std::uint32_t>(plan.artifact_kind));
    append_u64(&bytes, plan.tasks.size());
    for (const SelectionExecutionTask& task : plan.tasks) {
        append_u64(&bytes, task.population_slot);
        append_u32(&bytes, static_cast<std::uint32_t>(task.action));
        append_u32(&bytes, task.candidate_index);
        append_u64(&bytes, task.branch_ordinal);
        append_address(&bytes, task.candidate_root);
        append_address(&bytes, task.task_head);
        append_u64(&bytes, task.compute_supersteps);
    }
    append_u64(&bytes, plan.mating_assignments.size());
    for (const SelectionMatingAssignment& assignment : plan.mating_assignments) {
        append_u64(&bytes, assignment.first_population_slot);
        append_u64(&bytes, assignment.second_population_slot);
        append_u32(&bytes, assignment.first_candidate_index);
        append_u32(&bytes, assignment.second_candidate_index);
    }
    append_u64(&bytes, plan.total_compute_supersteps);
    return bytes;
}

ContentAddress selection_execution_plan_address(const SelectionExecutionPlan& plan) {
    return content_address(canonical_selection_execution_plan(plan));
}

bool decode_selection_execution_plan(std::span<const std::byte> bytes,
                                     SelectionExecutionPlan* plan,
                                     std::string* error) {
    Reader reader(bytes);
    SelectionExecutionPlan decoded{};
    std::uint32_t artifact_kind = 0u;
    if (plan == nullptr || !reader.take_domain(kExecutionPlanDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !reader.take_address(&decoded.evaluation_request) ||
        !reader.take_address(&decoded.operation_resolution) ||
        !reader.take_address(&decoded.transcript) ||
        !reader.take_address(&decoded.scheduler_protocol) ||
        !reader.take_u32(&artifact_kind)) {
        return fail(error, "BCC-32 selection execution plan header is truncated");
    }
    decoded.artifact_kind = static_cast<ArtifactKind>(artifact_kind);
    std::size_t task_count = 0u;
    if (!take_count(&reader, kMaximumSelectionObjects, 96u, &task_count)) {
        return fail(error, "BCC-32 selection execution plan task count is invalid");
    }
    decoded.tasks.reserve(task_count);
    for (std::size_t index = 0u; index < task_count; ++index) {
        SelectionExecutionTask task{};
        std::uint32_t action = 0u;
        if (!reader.take_u64(&task.population_slot) || !reader.take_u32(&action) ||
            !reader.take_u32(&task.candidate_index) ||
            !reader.take_u64(&task.branch_ordinal) ||
            !reader.take_address(&task.candidate_root) ||
            !reader.take_address(&task.task_head) ||
            !reader.take_u64(&task.compute_supersteps)) {
            return fail(error, "BCC-32 selection execution task is truncated");
        }
        task.action = static_cast<SelectionAction>(action);
        decoded.tasks.push_back(task);
    }
    std::size_t mating_count = 0u;
    if (!take_count(&reader, kMaximumSelectionObjects, 24u, &mating_count)) {
        return fail(error, "BCC-32 selection execution mating count is invalid");
    }
    decoded.mating_assignments.reserve(mating_count);
    for (std::size_t index = 0u; index < mating_count; ++index) {
        SelectionMatingAssignment assignment{};
        if (!reader.take_u64(&assignment.first_population_slot) ||
            !reader.take_u64(&assignment.second_population_slot) ||
            !reader.take_u32(&assignment.first_candidate_index) ||
            !reader.take_u32(&assignment.second_candidate_index)) {
            return fail(error, "BCC-32 selection mating assignment is truncated");
        }
        decoded.mating_assignments.push_back(assignment);
    }
    if (!reader.take_u64(&decoded.total_compute_supersteps) || !reader.done() ||
        !validate_plan_shape(decoded, error) ||
        !canonical_equal(bytes, canonical_selection_execution_plan(decoded))) {
        return fail(error, "BCC-32 selection execution plan is not canonical");
    }
    *plan = std::move(decoded);
    return true;
}

std::vector<std::byte> canonical_selection_execution_admission(
    const SelectionExecutionAdmission& admission) {
    Bytes bytes;
    append_domain(&bytes, kExecutionAdmissionDomain);
    append_u32(&bytes, admission.schema_version);
    append_address(&bytes, admission.evaluation_request);
    append_address(&bytes, admission.operation_resolution);
    append_address(&bytes, admission.execution_plan);
    return bytes;
}

ContentAddress selection_execution_admission_address(
    const SelectionExecutionAdmission& admission) {
    return content_address(canonical_selection_execution_admission(admission));
}

bool decode_selection_execution_admission(
    std::span<const std::byte> bytes,
    SelectionExecutionAdmission* admission,
    std::string* error) {
    Reader reader(bytes);
    SelectionExecutionAdmission decoded{};
    if (admission == nullptr || !reader.take_domain(kExecutionAdmissionDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !reader.take_address(&decoded.evaluation_request) ||
        !reader.take_address(&decoded.operation_resolution) ||
        !reader.take_address(&decoded.execution_plan) || !reader.done() ||
        !validate_admission_shape(decoded, error) ||
        !canonical_equal(bytes, canonical_selection_execution_admission(decoded))) {
        return fail(error, "BCC-32 selection execution admission is not canonical");
    }
    *admission = decoded;
    return true;
}

namespace {

Bytes evaluator_envelope(const SelectionEvaluationRequest& request,
                         const SelectionCandidateSet& candidate_set,
                         const std::vector<Bytes>& inputs) {
    const Bytes request_bytes = canonical_selection_evaluation_request(request);
    const Bytes candidate_bytes = canonical_selection_candidate_set(candidate_set);
    Bytes bytes;
    append_domain(&bytes, kEvaluatorEnvelopeDomain);
    append_u32(&bytes, 1u);
    append_u64(&bytes, request_bytes.size());
    append_raw(&bytes, request_bytes);
    append_u64(&bytes, candidate_bytes.size());
    append_raw(&bytes, candidate_bytes);
    append_u64(&bytes, inputs.size());
    for (std::size_t index = 0u; index < inputs.size(); ++index) {
        append_address(&bytes, request.evaluation_inputs[index]);
        append_u64(&bytes, inputs[index].size());
        append_raw(&bytes, inputs[index]);
    }
    return bytes;
}

bool validate_selection_request_closure(
    const std::filesystem::path& repository,
    const SelectionEvaluationRequest& request,
    const SelectionCandidateSet& candidate_set,
    std::string* error);

bool load_request_inputs(const std::filesystem::path& repository,
                         const SelectionEvaluationRequest& request,
                         SelectionCandidateSet* candidate_set,
                         std::vector<Bytes>* inputs,
                         Bytes* envelope,
                         std::string* error) {
    if (candidate_set == nullptr || inputs == nullptr || envelope == nullptr ||
        !read_selection_candidate_set_object(repository,
                                             request.candidate_set_commitment,
                                             candidate_set,
                                             error) ||
        !validate_selection_evaluation_request(*candidate_set, request, error)) {
        return false;
    }
    if (!validate_selection_request_closure(repository, request,
                                            *candidate_set, error)) {
        return false;
    }
    inputs->clear();
    inputs->reserve(request.evaluation_inputs.size());
    for (const ContentAddress& input : request.evaluation_inputs) {
        inputs->emplace_back();
        if (!read_selection_blob_object(repository, input, &inputs->back(), error)) {
            return false;
        }
    }
    *envelope = evaluator_envelope(request, *candidate_set, *inputs);
    if (envelope->size() > request.limits.maximum_input_bytes) {
        return fail(error, "BCC-32 evaluator envelope exceeds its precommitted byte bound");
    }
    return true;
}

bool persist_typed(const std::filesystem::path& repository,
                   std::string_view directory,
                   std::span<const std::byte> bytes,
                   ContentAddress* identity,
                   std::string* error) {
    if (identity == nullptr || !ensure_selection_repository(repository, error)) {
        return fail(error, "BCC-32 typed selection publication requires identity output");
    }
    const ContentAddress address = content_address(bytes);
    if (!write_immutable(object_path(repository, directory, address), bytes, error)) {
        return false;
    }
    *identity = address;
    return true;
}

bool read_operation_pointer(const std::filesystem::path& repository,
                            const ContentAddress& operation,
                            std::string_view name,
                            std::string_view domain,
                            ContentAddress* identity,
                            std::string* error) {
    Bytes bytes;
    if (!read_file(operation_path(repository, operation) / name,
                   256u, &bytes, error)) {
        return false;
    }
    return decode_pointer(bytes, domain, identity, error);
}

bool publish_operation_pointer(const std::filesystem::path& repository,
                               const ContentAddress& operation,
                               std::string_view name,
                               std::string_view domain,
                               const ContentAddress& identity,
                               std::string* error) {
    if (!ensure_operation_directory(repository, operation, error)) return false;
    const Bytes bytes = canonical_pointer(domain, identity);
    return write_immutable(operation_path(repository, operation) / name, bytes, error);
}

bool load_precommitted_request_by_identity(
    const std::filesystem::path& repository,
    const ContentAddress& request_identity,
    SelectionEvaluationRequest* request,
    ContentAddress* operation,
    std::string* error) {
    SelectionEvaluationRequest decoded{};
    if (!read_selection_evaluation_request_object(repository,
                                                  request_identity,
                                                  &decoded,
                                                  error)) {
        return false;
    }
    const ContentAddress decoded_operation = selection_operation_identity(decoded);
    ContentAddress admitted_request{};
    if (!read_operation_pointer(repository, decoded_operation, "REQUEST",
                                kOperationRequestPointerDomain,
                                &admitted_request, error) ||
        admitted_request != request_identity) {
        return fail(error, "BCC-32 evaluator request is not the admitted operation request");
    }
    if (request != nullptr) *request = decoded;
    if (operation != nullptr) *operation = decoded_operation;
    return true;
}

bool read_receipt_object(const std::filesystem::path& repository,
                         const ContentAddress& identity,
                         SelectionEvaluatorReceipt* receipt,
                         std::string* error) {
    return read_typed_object(
        object_path(repository, "selection-evaluator-receipts", identity),
        identity, receipt, decode_selection_evaluator_receipt, error);
}

bool read_resolution_object(const std::filesystem::path& repository,
                            const ContentAddress& identity,
                            SelectionOperationResolution* resolution,
                            std::string* error) {
    return read_typed_object(
        object_path(repository, "selection-operation-resolutions", identity),
        identity, resolution, decode_selection_operation_resolution, error);
}

bool read_admission_object(const std::filesystem::path& repository,
                           const ContentAddress& identity,
                           SelectionExecutionAdmission* admission,
                           std::string* error) {
    return read_typed_object(
        object_path(repository, "selection-execution-admissions", identity),
        identity, admission, decode_selection_execution_admission, error);
}

bool validate_selection_request_closure(
    const std::filesystem::path& repository,
    const SelectionEvaluationRequest& request,
    const SelectionCandidateSet& candidate_set,
    std::string* error) {
    for (const SelectionCandidate& candidate : candidate_set.candidates) {
        WorldCommit frozen{};
        if (!load_world_commit_object(repository, request.candidate_artifact_kind,
                                      candidate.root, &frozen, error)) {
            return fail(error,
                        "BCC-32 selection candidate has the wrong precommitted artifact kind");
        }
    }

    if (request.selection_round == 0u) {
        return fail(error, "BCC-32 selection round zero has no admitted predecessor");
    }
    if (request.selection_round == 1u) {
        WorldCommit world{};
        SelectionCandidateSet population{};
        std::string ignored;
        if (!load_world_commit_object(repository, request.candidate_artifact_kind,
                                      request.operation_predecessor, &world,
                                      &ignored) &&
            !read_selection_candidate_set_object(repository,
                                                 request.operation_predecessor,
                                                 &population, &ignored)) {
            return fail(error,
                        "BCC-32 first selection predecessor is not a frozen world or population");
        }
    } else {
        SelectionExecutionAdmission prior{};
        SelectionEvaluationRequest prior_request{};
        ContentAddress committed_prior_request{};
        ContentAddress committed_prior_admission{};
        if (!read_admission_object(repository, request.operation_predecessor,
                                   &prior, error) ||
            !read_selection_evaluation_request_object(repository,
                                                      prior.evaluation_request,
                                                      &prior_request, error)) {
            return fail(error,
                        "BCC-32 selection predecessor is not an admitted operation");
        }
        const ContentAddress prior_operation =
            selection_operation_identity(prior_request);
        if (!read_operation_pointer(repository, prior_operation, "REQUEST",
                                    kOperationRequestPointerDomain,
                                    &committed_prior_request, error) ||
            committed_prior_request != prior.evaluation_request ||
            !read_operation_pointer(repository, prior_operation, "ADMISSION",
                                    kOperationAdmissionPointerDomain,
                                    &committed_prior_admission, error) ||
            committed_prior_admission != request.operation_predecessor ||
            prior_request.selection_round + 1u != request.selection_round ||
            prior_request.candidate_artifact_kind !=
                request.candidate_artifact_kind) {
            return fail(error,
                        "BCC-32 selection predecessor is not the immediately prior admission");
        }
    }

    const std::array<ContentAddress, 8> protocol_inputs = {
        request.evaluator_identity,
        request.evaluator_version,
        request.evaluator_artifact,
        request.evaluator_target_provenance,
        request.environment_contact_input,
        request.replay_semantics,
        request.slot_materialization_semantics,
        request.scheduler_protocol,
    };
    Bytes bytes;
    for (const ContentAddress& input : protocol_inputs) {
        if (!read_selection_blob_object(repository, input, &bytes, error) ||
            bytes.empty()) {
            return fail(error,
                        "BCC-32 selection protocol dependency is missing or empty");
        }
    }
    for (const ContentAddress& input : request.evaluation_inputs) {
        if (!read_selection_blob_object(repository, input, &bytes, error)) {
            return fail(error,
                        "BCC-32 selection evaluation input is missing or corrupt");
        }
    }
    if (!read_selection_blob_object(repository, request.replay_semantics,
                                    &bytes, error) ||
        !canonical_text_equal(bytes, kSelectionReplaySemanticsV1) ||
        !read_selection_blob_object(repository,
                                    request.slot_materialization_semantics,
                                    &bytes, error) ||
        !canonical_text_equal(bytes,
                              kSelectionSlotMaterializationSemanticsV1) ||
        !read_selection_blob_object(repository, request.scheduler_protocol,
                                    &bytes, error) ||
        !canonical_text_equal(bytes, kSelectionSchedulerProtocolV1)) {
        return fail(error,
                    "BCC-32 selection request names unsupported execution semantics");
    }
    if (request.mating_contact_manifest != ContentAddress{} &&
        (!read_selection_blob_object(repository, request.mating_contact_manifest,
                                     &bytes, error) ||
         bytes.empty())) {
        return fail(error, "BCC-32 selection mating manifest is missing or empty");
    }
    if (!read_selection_blob_object(repository, request.evaluator_artifact,
                                    &bytes, error) ||
        !validate_static_evaluator_artifact(bytes, error)) {
        return false;
    }
    return true;
}

}  // namespace

bool precommit_selection_operation(
    const std::filesystem::path& repository,
    const SelectionEvaluationRequest& request,
    ContentAddress* request_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure) {
    if (request_identity == nullptr) {
        return fail(error, "BCC-32 selection precommit requires identity output");
    }
    SelectionCandidateSet candidate_set{};
    std::vector<Bytes> inputs;
    Bytes envelope;
    if (!load_request_inputs(repository, request, &candidate_set, &inputs,
                             &envelope, error)) {
        return false;
    }
    ContentAddress identity{};
    if (!put_selection_evaluation_request_object(repository, request, &identity, error)) {
        return false;
    }
    *request_identity = identity;
    if (failure == SelectionExecutorFailurePoint::after_request_object) {
        return fail(error, "injected crash after BCC-32 selection request object");
    }
    const ContentAddress operation = selection_operation_identity(request);
    if (!publish_operation_pointer(repository, operation, "REQUEST",
                                   kOperationRequestPointerDomain, identity, error)) {
        return false;
    }
    if (failure == SelectionExecutorFailurePoint::after_request_admission) {
        return fail(error, "injected crash after BCC-32 selection request admission");
    }
    return true;
}

bool read_precommitted_selection_request(
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    ContentAddress* request_identity,
    SelectionEvaluationRequest* request,
    std::string* error) {
    if (request_identity == nullptr || request == nullptr ||
        !is_valid_content_address(operation_identity) ||
        !read_operation_pointer(repository, operation_identity, "REQUEST",
                                kOperationRequestPointerDomain,
                                request_identity, error) ||
        !read_selection_evaluation_request_object(repository, *request_identity,
                                                  request, error) ||
        selection_operation_identity(*request) != operation_identity) {
        return fail(error, "BCC-32 selection operation request is missing or inconsistent");
    }
    return true;
}

namespace {

bool accept_selection_evaluator_output(
    const std::filesystem::path& repository,
    const ContentAddress& request_identity,
    std::span<const std::byte> input_envelope,
    std::span<const std::byte> output_bytes,
    std::uint32_t exit_status,
    ContentAddress* resolution_identity,
    ContentAddress* transcript_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure) {
    if (resolution_identity == nullptr || transcript_identity == nullptr) {
        return fail(error, "BCC-32 evaluator acceptance requires identity outputs");
    }
    SelectionEvaluationRequest request{};
    ContentAddress operation{};
    if (!load_precommitted_request_by_identity(repository, request_identity,
                                               &request, &operation, error)) {
        return false;
    }
    if (exit_status != 0u || input_envelope.empty() ||
        input_envelope.size() > request.limits.maximum_input_bytes ||
        output_bytes.empty() ||
        output_bytes.size() > request.limits.maximum_transcript_bytes) {
        return fail(error, "BCC-32 evaluator did not return a bounded successful transcript");
    }
    SelectionCandidateSet candidate_set{};
    std::vector<Bytes> inputs;
    Bytes expected_envelope;
    if (!load_request_inputs(repository, request, &candidate_set, &inputs,
                             &expected_envelope, error) ||
        !canonical_equal(input_envelope, expected_envelope)) {
        return fail(error, "BCC-32 evaluator input is not the precommitted envelope");
    }
    SelectionTranscript transcript{};
    if (!decode_selection_transcript(output_bytes, &transcript, error) ||
        !validate_selection_transcript(candidate_set, request, transcript, error)) {
        return fail(error, "BCC-32 evaluator output is not the precommitted canonical transcript");
    }
    ContentAddress persisted_transcript{};
    if (!put_selection_transcript_object(repository, transcript,
                                         &persisted_transcript, error)) {
        return false;
    }
    *transcript_identity = persisted_transcript;
    if (failure == SelectionExecutorFailurePoint::after_transcript_object) {
        return fail(error, "injected crash after BCC-32 evaluator transcript object");
    }

    const SelectionEvaluatorReceipt receipt{
        .evaluation_request = request_identity,
        .evaluator_artifact = request.evaluator_artifact,
        .input_envelope = content_address(input_envelope),
        .output_bytes = content_address(output_bytes),
        .sandbox_profile = canonical_selection_sandbox_profile(),
        .transcript = persisted_transcript,
        .exit_status = exit_status,
    };
    if (!validate_receipt_shape(receipt, error)) return false;
    const Bytes receipt_bytes = canonical_selection_evaluator_receipt(receipt);
    ContentAddress receipt_identity{};
    if (!persist_typed(repository, "selection-evaluator-receipts",
                       receipt_bytes, &receipt_identity, error)) {
        return false;
    }
    const SelectionOperationResolution resolution{
        .evaluation_request = request_identity,
        .transcript = persisted_transcript,
        .evaluator_receipt = receipt_identity,
    };
    const Bytes resolution_bytes = canonical_selection_operation_resolution(resolution);
    ContentAddress persisted_resolution{};
    if (!persist_typed(repository, "selection-operation-resolutions",
                       resolution_bytes, &persisted_resolution, error)) {
        return false;
    }
    *resolution_identity = persisted_resolution;
    if (failure == SelectionExecutorFailurePoint::after_resolution_object) {
        return fail(error, "injected crash after BCC-32 selection resolution object");
    }
    if (!publish_operation_pointer(repository, operation, "RESOLUTION",
                                   kOperationResolutionPointerDomain,
                                   persisted_resolution, error)) {
        return false;
    }
    if (failure == SelectionExecutorFailurePoint::after_resolution_admission) {
        return fail(error, "injected crash after BCC-32 selection resolution admission");
    }
    return true;
}

}  // namespace

bool read_selection_operation_resolution(
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    ContentAddress* resolution_identity,
    SelectionOperationResolution* resolution,
    std::string* error) {
    if (resolution_identity == nullptr || resolution == nullptr ||
        !read_operation_pointer(repository, operation_identity, "RESOLUTION",
                                kOperationResolutionPointerDomain,
                                resolution_identity, error) ||
        !read_resolution_object(repository, *resolution_identity, resolution, error)) {
        return fail(error, "BCC-32 selection operation resolution is missing");
    }
    SelectionEvaluationRequest request{};
    ContentAddress request_identity{};
    SelectionEvaluatorReceipt receipt{};
    SelectionTranscript transcript{};
    SelectionCandidateSet candidate_set{};
    std::vector<Bytes> inputs;
    Bytes expected_envelope;
    if (!read_precommitted_selection_request(repository, operation_identity,
                                             &request_identity, &request, error) ||
        resolution->evaluation_request != request_identity ||
        !read_receipt_object(repository, resolution->evaluator_receipt,
                             &receipt, error) ||
        receipt.evaluation_request != request_identity ||
        receipt.evaluator_artifact != request.evaluator_artifact ||
        receipt.transcript != resolution->transcript ||
        receipt.output_bytes != resolution->transcript ||
        receipt.sandbox_profile != canonical_selection_sandbox_profile() ||
        !load_request_inputs(repository, request, &candidate_set, &inputs,
                             &expected_envelope, error) ||
        receipt.input_envelope != content_address(expected_envelope) ||
        !read_selection_transcript_object(repository, resolution->transcript,
                                          &transcript, error) ||
        selection_transcript_address(transcript) != receipt.output_bytes) {
        return fail(error, "BCC-32 selection operation resolution closure is invalid");
    }
    return true;
}

namespace {

#include "bcc32_selection_executor_sandbox.inl"

}  // namespace

bool run_precommitted_selection_evaluator(
    const std::filesystem::path& repository,
    const ContentAddress& request_identity,
    ContentAddress* resolution_identity,
    ContentAddress* transcript_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure) {
    SelectionEvaluationRequest request{};
    ContentAddress operation{};
    if (!load_precommitted_request_by_identity(repository, request_identity,
                                               &request, &operation, error)) {
        return false;
    }
    SelectionCandidateSet candidate_set{};
    std::vector<Bytes> inputs;
    Bytes envelope;
    if (!load_request_inputs(repository, request, &candidate_set, &inputs,
                             &envelope, error)) {
        return false;
    }
    Bytes artifact;
    if (!read_selection_blob_object(repository, request.evaluator_artifact,
                                    &artifact, error)) {
        return false;
    }
    std::array<char, 96> directory_template{};
    const std::string prefix =
        (std::filesystem::temp_directory_path() / "bcc32_selection_eval_XXXXXX").string();
    if (prefix.size() + 1u > directory_template.size()) {
        return fail(error, "BCC-32 evaluator temporary path is too long");
    }
    std::memcpy(directory_template.data(), prefix.c_str(), prefix.size() + 1u);
    char* created = ::mkdtemp(directory_template.data());
    if (created == nullptr) {
        return fail(error, "cannot create BCC-32 evaluator sandbox directory");
    }
    const std::filesystem::path directory(created);
    const std::filesystem::path executable = directory / "evaluator";
    Bytes output;
    std::uint32_t exit_status = 0u;
    const bool ran = write_executable(executable, artifact, error) &&
        run_sandboxed_evaluator(executable, request.limits, envelope,
                                &output, &exit_status, error);
    std::error_code cleanup_error;
    std::filesystem::remove_all(directory, cleanup_error);
    if (!ran) return false;
    return accept_selection_evaluator_output(
        repository, request_identity, envelope, output, exit_status,
        resolution_identity, transcript_identity, error, failure);
}

namespace {

bool assign_compute(const SelectionReplay& replay,
                    std::vector<SelectionExecutionTask>* tasks,
                    std::string* error) {
    std::map<std::uint32_t, std::vector<std::size_t>> by_candidate;
    for (std::size_t index = 0u; index < tasks->size(); ++index) {
        by_candidate[(*tasks)[index].candidate_index].push_back(index);
    }
    for (const SelectionComputeAllocation& allocation : replay.compute_allocations) {
        const auto found = by_candidate.find(allocation.candidate_index);
        if (found == by_candidate.end() || found->second.empty()) {
            return fail(error, "BCC-32 compute allocation has no admitted population task");
        }
        const std::uint64_t quotient =
            allocation.compute_units / found->second.size();
        const std::uint64_t remainder =
            allocation.compute_units % found->second.size();
        for (std::size_t ordinal = 0u; ordinal < found->second.size(); ++ordinal) {
            (*tasks)[found->second[ordinal]].compute_supersteps =
                quotient + (ordinal < remainder ? 1u : 0u);
        }
    }
    return true;
}

bool assign_mating(const SelectionReplay& replay,
                   const std::vector<SelectionExecutionTask>& tasks,
                   std::vector<SelectionMatingAssignment>* assignments,
                   std::string* error) {
    std::map<std::uint32_t, std::uint64_t> first_slot;
    for (const SelectionExecutionTask& task : tasks) {
        first_slot.try_emplace(task.candidate_index, task.population_slot);
    }
    assignments->clear();
    assignments->reserve(replay.mating_graph.size());
    for (const SelectionMatingEdge& edge : replay.mating_graph) {
        const auto first = first_slot.find(edge.parent_candidate_index);
        const auto second = first_slot.find(edge.mate_candidate_index);
        if (first == first_slot.end() || second == first_slot.end()) {
            return fail(error, "BCC-32 mating edge has no admitted population task");
        }
        SelectionMatingAssignment assignment{
            .first_population_slot = first->second,
            .second_population_slot = second->second,
            .first_candidate_index = edge.parent_candidate_index,
            .second_candidate_index = edge.mate_candidate_index,
        };
        if (assignment.first_population_slot > assignment.second_population_slot) {
            std::swap(assignment.first_population_slot,
                      assignment.second_population_slot);
            std::swap(assignment.first_candidate_index,
                      assignment.second_candidate_index);
        }
        assignments->push_back(assignment);
    }
    std::sort(assignments->begin(), assignments->end(),
              [](const auto& left, const auto& right) {
                  return std::pair{left.first_population_slot,
                                   left.second_population_slot} <
                         std::pair{right.first_population_slot,
                                   right.second_population_slot};
              });
    return std::adjacent_find(assignments->begin(), assignments->end()) ==
               assignments->end() ||
           fail(error, "BCC-32 mating task assignment is duplicated");
}

bool load_replay_closure(const std::filesystem::path& repository,
                         const ContentAddress& request_identity,
                         SelectionEvaluationRequest* request,
                         ContentAddress* operation,
                         ContentAddress* resolution_identity,
                         SelectionOperationResolution* resolution,
                         SelectionCandidateSet* candidate_set,
                         SelectionTranscript* transcript,
                         SelectionReplay* replay,
                         std::string* error) {
    if (!load_precommitted_request_by_identity(repository, request_identity,
                                               request, operation, error) ||
        !read_selection_operation_resolution(repository, *operation,
                                             resolution_identity, resolution, error) ||
        resolution->evaluation_request != request_identity ||
        !read_selection_candidate_set_object(repository,
                                             request->candidate_set_commitment,
                                             candidate_set, error) ||
        !read_selection_transcript_object(repository, resolution->transcript,
                                          transcript, error) ||
        !replay_selection_transcript(*candidate_set, *request, *transcript,
                                     replay, error)) {
        return false;
    }
    return true;
}

bool validate_plan_context(const std::filesystem::path& repository,
                           const SelectionEvaluationRequest& request,
                           const SelectionOperationResolution& resolution,
                           const SelectionCandidateSet& candidate_set,
                           const SelectionTranscript& transcript,
                           const SelectionReplay& replay,
                           const SelectionExecutionPlan& plan,
                           ArtifactKind expected_kind,
                           std::string* error) {
    if (!validate_plan_shape(plan, error) ||
        plan.evaluation_request != resolution.evaluation_request ||
        plan.transcript != resolution.transcript ||
        plan.scheduler_protocol != request.scheduler_protocol ||
        plan.artifact_kind != expected_kind ||
        plan.tasks.size() != replay.next_population_slot_count ||
        plan.total_compute_supersteps != replay.total_compute_units) {
        return fail(error, "BCC-32 selection execution plan disagrees with replay");
    }
    std::vector<SelectionExecutionTask> expected_tasks;
    expected_tasks.reserve(plan.tasks.size());
    for (std::uint64_t slot = 0u; slot < replay.next_population_slot_count; ++slot) {
        SelectionPopulationSlot resolved{};
        if (!resolve_selection_population_slot(replay, slot, &resolved, error)) {
            return false;
        }
        const SelectionExecutionTask& actual = plan.tasks[slot];
        if (actual.population_slot != slot || actual.action != resolved.action ||
            actual.candidate_index != resolved.candidate_index ||
            actual.branch_ordinal != resolved.branch_ordinal ||
            actual.candidate_root != resolved.root) {
            return fail(error, "BCC-32 execution task does not resolve from the transcript");
        }
        WorldCommit task_head{};
        WorldCommit candidate{};
        if (!load_world_commit_object(repository, expected_kind, actual.task_head,
                                      &task_head, error) ||
            !load_world_commit_object(repository, expected_kind, actual.candidate_root,
                                      &candidate, error) ||
            task_head.metadata.provenance.entry_event.kind !=
                EntryEventKind::evaluator_selection ||
            task_head.metadata.provenance.entry_event.evaluator_transcript !=
                resolution.transcript ||
            task_head.metadata.provenance.entry_event.next_population_slot != slot ||
            task_head.metadata.replay_boundary.predecessor_commit !=
                actual.candidate_root ||
            material_state_identity(task_head.chunks) !=
                material_state_identity(candidate.chunks)) {
            return fail(error, "BCC-32 execution task head is not the selected material");
        }
        expected_tasks.push_back({
            .population_slot = slot,
            .action = resolved.action,
            .candidate_index = resolved.candidate_index,
            .branch_ordinal = resolved.branch_ordinal,
            .candidate_root = resolved.root,
            .task_head = actual.task_head,
        });
    }
    if (!assign_compute(replay, &expected_tasks, error)) return false;
    for (std::size_t index = 0u; index < plan.tasks.size(); ++index) {
        if (plan.tasks[index].compute_supersteps !=
            expected_tasks[index].compute_supersteps) {
            return fail(error, "BCC-32 execution task compute split is not deterministic");
        }
    }
    std::vector<SelectionMatingAssignment> expected_mating;
    if (!assign_mating(replay, plan.tasks, &expected_mating, error) ||
        expected_mating != plan.mating_assignments) {
        return fail(error, "BCC-32 execution task mating assignment is not deterministic");
    }
    (void)candidate_set;
    (void)transcript;
    return true;
}

}  // namespace

#include "bcc32_selection_executor_tail.inl"
