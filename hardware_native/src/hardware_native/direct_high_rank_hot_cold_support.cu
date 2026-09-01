#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "hardware_native/direct_high_rank_hot_cold_support.cuh"

namespace substrate::direct_adult_core {
namespace {
std::string support_path(const char* directory, const direct_network::recipe::Root256& address) {
  std::ostringstream name;
  name << directory << "/support-" << std::hex << std::setfill('0');
  for (std::uint32_t word : address.word)
    name << std::setw(8) << word;
  return name.str() + ".bin";
}
bool read_page(const std::string& path, DirectHighRankSupportPageV1* page) {
  if (page == nullptr)
    return false;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != static_cast<std::streamoff>(sizeof(*page)))
    return false;
  input.seekg(0);
  input.read(reinterpret_cast<char*>(page), sizeof(*page));
  return input.good();
}
}  // namespace

HighRankColdStatus store_high_rank_support_page(const char* directory,
                                                const DirectHighRankHotColdStateV1& request,
                                                const DirectHighRankSupportPageV1& page,
                                                std::string* object_path) {
  if (directory == nullptr || object_path == nullptr ||
      request.request != HighRankSupportRequest::evict || request.support_present != 1u ||
      !high_rank_support_page_valid(page) ||
      high_rank_support_address(page) != request.requested_address)
    return HighRankColdStatus::invalid_request;
  if (request.support_capacity_bytes < sizeof(page))
    return HighRankColdStatus::capacity_exhausted;
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return HighRankColdStatus::io_error;
  const std::string path = support_path(directory, request.requested_address);
  DirectHighRankSupportPageV1 present{};
  if (std::filesystem::exists(path) &&
      (!read_page(path, &present) || std::memcmp(&present, &page, sizeof(page)) != 0))
    return HighRankColdStatus::content_mismatch;
  if (!std::filesystem::exists(path)) {
    const std::string temporary = path + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&page), sizeof(page));
    output.flush();
    if (!output.good())
      return HighRankColdStatus::io_error;
    output.close();
    std::filesystem::rename(temporary, path, error);
    if (error)
      return HighRankColdStatus::io_error;
  }
  *object_path = path;
  return HighRankColdStatus::stored;
}

HighRankColdStatus load_high_rank_support_page(const char* directory,
                                               const DirectHighRankHotColdStateV1& request,
                                               DirectHighRankSupportPageV1* page,
                                               std::string* object_path) {
  if (directory == nullptr || page == nullptr || object_path == nullptr ||
      request.request != HighRankSupportRequest::restore || request.support_present != 0u ||
      high_rank_root_zero(request.requested_address))
    return HighRankColdStatus::invalid_request;
  const std::string path = support_path(directory, request.requested_address);
  DirectHighRankSupportPageV1 candidate{};
  if (!read_page(path, &candidate))
    return HighRankColdStatus::io_error;
  if (!high_rank_support_page_valid(candidate) ||
      high_rank_support_address(candidate) != request.requested_address ||
      candidate.logical_recipe_id != request.logical_recipe_id ||
      candidate.revision_identity != request.revision_identity)
    return HighRankColdStatus::content_mismatch;
  *page = candidate;
  *object_path = path;
  return HighRankColdStatus::loaded;
}

}  // namespace substrate::direct_adult_core
