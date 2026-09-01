#pragma once

#if !defined(_WIN32)

#include "bcc32_external_body_stream_pump.cuh"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

namespace bcc32 {

class ExternalBodyPosixReader {
public:
    ExternalBodyPosixReader() = default;
    explicit ExternalBodyPosixReader(int borrowed_fd) : fd_(borrowed_fd) {}

    bool valid() const { return fd_ >= 0; }
    int borrowed_fd() const { return fd_; }

    bool require_nonblocking() const {
        if (!valid()) {
            return false;
        }
        const int flags = ::fcntl(fd_, F_GETFL, 0);
        return flags >= 0 && (flags & O_NONBLOCK) != 0;
    }

    ExternalBodyReadResult read(std::uint8_t* out, std::size_t capacity) const {
        if (!valid() || out == nullptr || capacity == 0 || !require_nonblocking()) {
            return {0, false, false, true};
        }
        for (;;) {
            const ssize_t result = ::read(fd_, out, capacity);
            if (result > 0) {
                return {static_cast<std::size_t>(result), false, false, false};
            }
            if (result == 0) {
                return {0, false, true, false};
            }
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return {0, true, false, false};
            }
            return {0, false, false, true};
        }
    }

private:
    int fd_ = -1;
};

}  // namespace bcc32

#endif
