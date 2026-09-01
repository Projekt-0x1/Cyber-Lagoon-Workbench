#pragma once

#if !defined(_WIN32)

#include "bcc32_external_body_command_outbox.cuh"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

namespace bcc32 {

class ExternalBodyPosixWriter {
public:
    ExternalBodyPosixWriter() = default;
    explicit ExternalBodyPosixWriter(int borrowed_fd) : fd_(borrowed_fd) {}

    bool valid() const { return fd_ >= 0; }
    int borrowed_fd() const { return fd_; }

    bool require_nonblocking() const {
        if (!valid()) {
            return false;
        }
        int flags = -1;
        do {
            flags = ::fcntl(fd_, F_GETFL, 0);
        } while (flags < 0 && errno == EINTR);
        return flags >= 0 && (flags & O_NONBLOCK) != 0;
    }

    ExternalBodyWriteResult write(const std::uint8_t* bytes,
                                  std::size_t byte_count) const {
        if (!valid() || bytes == nullptr || byte_count == 0 ||
            !require_nonblocking()) {
            return {0, false, true};
        }
        ssize_t result = -1;
        do {
            result = ::write(fd_, bytes, byte_count);
        } while (result < 0 && errno == EINTR);
        if (result > 0) {
            return {static_cast<std::size_t>(result), false, false};
        }
        if (result == 0) {
            return {0, false, true};
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return {0, true, false};
        }
        return {0, false, true};
    }

private:
    int fd_ = -1;
};

}  // namespace bcc32

#endif
