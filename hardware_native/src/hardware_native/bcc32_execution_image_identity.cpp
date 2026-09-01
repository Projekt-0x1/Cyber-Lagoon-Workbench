#include "bcc32_execution_image_identity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "bcc32_law_identity.hpp"

namespace substrate::bcc32 {
namespace {

// The running image is read through the kernel's own view of it rather than
// through argv[0], which a caller controls and can point at a different file.
constexpr const char* kSelfImagePath = "/proc/self/exe";

[[nodiscard]] std::vector<std::byte> read_running_image() {
  std::ifstream image(kSelfImagePath, std::ios::binary);
  if (!image) {
    throw std::runtime_error(
        "BCC32 execution image identity: cannot open the running image at "
        "/proc/self/exe -- an unreadable image is a failed seal, not a pass");
  }

  std::vector<std::byte> bytes;
  std::array<char, 1u << 16> chunk{};
  while (image.read(chunk.data(), static_cast<std::streamsize>(chunk.size())) ||
         image.gcount() > 0) {
    const std::size_t got = static_cast<std::size_t>(image.gcount());
    const std::byte* first = reinterpret_cast<const std::byte*>(chunk.data());
    bytes.insert(bytes.end(), first, first + got);
    if (got < chunk.size()) break;
  }
  if (bytes.empty()) {
    throw std::runtime_error(
        "BCC32 execution image identity: the running image read as zero bytes");
  }
  return bytes;
}

}  // namespace

ContentAddress running_image_identity() {
  // Read once per process. The image cannot change under a running process on
  // this platform, and re-reading it per call would make every receipt pay for
  // a multi-megabyte hash.
  static const ContentAddress cached = [] {
    const std::vector<std::byte> bytes = read_running_image();
    return content_address(std::span<const std::byte>(bytes));
  }();
  return cached;
}

ContentAddress sealed_execution_identity() {
  const ContentAddress law = canonical_law_identity();
  const ContentAddress image = running_image_identity();

  // Length-prefixed concatenation, so no pair of different (law, image) inputs
  // can produce the same byte string and collide by construction.
  std::vector<std::byte> combined;
  combined.reserve(2u * (sizeof(Hash256) + sizeof(std::uint64_t)));
  auto append_address = [&combined](const ContentAddress& address) {
    for (const std::uint8_t byte : address.digest)
      combined.push_back(static_cast<std::byte>(byte));
    for (std::uint32_t index = 0u; index < 8u; ++index) {
      combined.push_back(
          static_cast<std::byte>((address.byte_count >> (index * 8u)) & 0xffu));
    }
  };
  append_address(law);
  append_address(image);
  return content_address(std::span<const std::byte>(combined));
}

}  // namespace substrate::bcc32
