#include "bcc32_coordinate.hpp"

#include <stdexcept>

namespace substrate::bcc32 {

bool CoordinateLess::operator()(const ExactCoordinate& left,
                                const ExactCoordinate& right) const {
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    return left.z < right.z;
}

ExactCoordinate operator+(const ExactCoordinate& left,
                          const ExactCoordinate& right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

ExactCoordinate operator-(const ExactCoordinate& left,
                          const ExactCoordinate& right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

CoordinateComponent floor_divide(const CoordinateComponent& value,
                                 std::uint32_t positive_divisor) {
    if (positive_divisor == 0u) {
        throw std::invalid_argument("BCC-32 coordinate divisor must be positive");
    }
    const CoordinateComponent divisor = positive_divisor;
    CoordinateComponent quotient = value / divisor;
    if (value % divisor < 0) --quotient;
    return quotient;
}

std::uint32_t floor_modulo(const CoordinateComponent& value,
                           std::uint32_t positive_modulus) {
    if (positive_modulus == 0u) {
        throw std::invalid_argument("BCC-32 coordinate modulus must be positive");
    }
    const CoordinateComponent modulus = positive_modulus;
    CoordinateComponent remainder = value % modulus;
    if (remainder < 0) remainder += modulus;
    return remainder.convert_to<std::uint32_t>();
}

std::string canonical_coordinate_component(const CoordinateComponent& value) {
    const CoordinateComponent magnitude = value < 0 ? -value : value;
    return (value < 0 ? "n" : "p") + magnitude.str();
}

}  // namespace substrate::bcc32
