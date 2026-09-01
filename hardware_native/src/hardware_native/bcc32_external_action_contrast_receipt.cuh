#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_ACTION_CONTRAST_HD __host__ __device__
#else
#define BCC32_ACTION_CONTRAST_HD
#endif

namespace bcc32 {

enum class ExternalActionArm : std::uint8_t {
    kExecuted = 1,
    kWithheld = 2,
};

struct ExternalActionObservation {
    std::uint64_t experiment_epoch = 0;
    std::uint64_t contrast_id = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t capture_id = 0;
    std::uint64_t body_session_epoch = 0;
    std::uint64_t device_tick = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t raw_consequence_digest = 0;
    ExternalActionArm arm = ExternalActionArm::kExecuted;
    bool command_applied = false;
};

struct ExternalActionContrastReceipt {
    bool complete = false;
    bool raw_bytes_diverged = false;
    bool transport_only = true;
    std::uint64_t experiment_epoch = 0;
    std::uint64_t contrast_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t body_session_epoch = 0;
    std::uint64_t executed_attempt_id = 0;
    std::uint64_t withheld_attempt_id = 0;
    std::uint64_t executed_capture_id = 0;
    std::uint64_t withheld_capture_id = 0;
    std::uint64_t executed_device_tick = 0;
    std::uint64_t withheld_device_tick = 0;
    std::uint64_t executed_raw_digest = 0;
    std::uint64_t withheld_raw_digest = 0;
    std::uint64_t structural_commitment = 0;
};

class ExternalActionContrastAccumulator {
public:
    BCC32_ACTION_CONTRAST_HD ExternalActionContrastAccumulator(
        std::uint64_t experiment_epoch,
        std::uint64_t contrast_id,
        std::uint64_t body_session_epoch,
        std::uint64_t action_envelope,
        std::uint64_t command_digest,
        std::uint64_t route_digest)
        : experiment_epoch_(experiment_epoch),
          contrast_id_(contrast_id),
          body_session_epoch_(body_session_epoch),
          action_envelope_(action_envelope),
          command_digest_(command_digest),
          route_digest_(route_digest),
          configured_(experiment_epoch != 0 && contrast_id != 0 &&
                      body_session_epoch != 0 && action_envelope != 0 &&
                      command_digest != 0 && route_digest != 0) {}

    BCC32_ACTION_CONTRAST_HD bool observe(
        const ExternalActionObservation& observation) {
        if (!configured_ || !matches_fixed_identity(observation) ||
            observation.attempt_id == 0 || observation.capture_id == 0 ||
            observation.device_tick == 0 ||
            observation.raw_consequence_digest == 0 ||
            observation.device_tick <= last_device_tick_ ||
            observation.attempt_id == executed_.attempt_id ||
            observation.attempt_id == withheld_.attempt_id ||
            observation.capture_id == executed_.capture_id ||
            observation.capture_id == withheld_.capture_id) {
            return false;
        }

        Slot* target = nullptr;
        if (observation.arm == ExternalActionArm::kExecuted) {
            if (executed_.present || !observation.command_applied ||
                observation.command_digest != command_digest_) {
                return false;
            }
            target = &executed_;
        } else if (observation.arm == ExternalActionArm::kWithheld) {
            if (withheld_.present || observation.command_applied ||
                observation.command_digest != 0) {
                return false;
            }
            target = &withheld_;
        } else {
            return false;
        }

        target->present = true;
        target->attempt_id = observation.attempt_id;
        target->capture_id = observation.capture_id;
        target->device_tick = observation.device_tick;
        target->raw_digest = observation.raw_consequence_digest;
        last_device_tick_ = observation.device_tick;
        return true;
    }

    BCC32_ACTION_CONTRAST_HD ExternalActionContrastReceipt receipt() const {
        ExternalActionContrastReceipt result{};
        result.experiment_epoch = experiment_epoch_;
        result.contrast_id = contrast_id_;
        result.action_envelope = action_envelope_;
        result.command_digest = command_digest_;
        result.route_digest = route_digest_;
        result.body_session_epoch = body_session_epoch_;
        if (!configured_ || !executed_.present || !withheld_.present) {
            return result;
        }

        result.complete = true;
        result.raw_bytes_diverged = executed_.raw_digest != withheld_.raw_digest;
        result.executed_attempt_id = executed_.attempt_id;
        result.withheld_attempt_id = withheld_.attempt_id;
        result.executed_capture_id = executed_.capture_id;
        result.withheld_capture_id = withheld_.capture_id;
        result.executed_device_tick = executed_.device_tick;
        result.withheld_device_tick = withheld_.device_tick;
        result.executed_raw_digest = executed_.raw_digest;
        result.withheld_raw_digest = withheld_.raw_digest;
        result.structural_commitment = commitment(result);
        return result;
    }

private:
    struct Slot {
        bool present = false;
        std::uint64_t attempt_id = 0;
        std::uint64_t capture_id = 0;
        std::uint64_t device_tick = 0;
        std::uint64_t raw_digest = 0;
    };

    BCC32_ACTION_CONTRAST_HD bool matches_fixed_identity(
        const ExternalActionObservation& observation) const {
        return observation.experiment_epoch == experiment_epoch_ &&
               observation.contrast_id == contrast_id_ &&
               observation.body_session_epoch == body_session_epoch_ &&
               observation.action_envelope == action_envelope_ &&
               observation.route_digest == route_digest_;
    }

    BCC32_ACTION_CONTRAST_HD static std::uint64_t mix(
        std::uint64_t state, std::uint64_t value) {
        state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
        return state;
    }

    BCC32_ACTION_CONTRAST_HD static std::uint64_t commitment(
        const ExternalActionContrastReceipt& receipt) {
        std::uint64_t state = 0x434f4e5452415354ULL;
        const std::uint64_t fields[] = {
            receipt.experiment_epoch, receipt.contrast_id,
            receipt.body_session_epoch, receipt.action_envelope,
            receipt.command_digest, receipt.route_digest,
            receipt.executed_attempt_id, receipt.withheld_attempt_id,
            receipt.executed_capture_id, receipt.withheld_capture_id,
            receipt.executed_device_tick, receipt.withheld_device_tick,
            receipt.executed_raw_digest, receipt.withheld_raw_digest,
        };
        for (const std::uint64_t field : fields) {
            state = mix(state, field);
        }
        return state == 0 ? 1 : state;
    }

    std::uint64_t experiment_epoch_ = 0;
    std::uint64_t contrast_id_ = 0;
    std::uint64_t body_session_epoch_ = 0;
    std::uint64_t action_envelope_ = 0;
    std::uint64_t command_digest_ = 0;
    std::uint64_t route_digest_ = 0;
    std::uint64_t last_device_tick_ = 0;
    bool configured_ = false;
    Slot executed_{};
    Slot withheld_{};
};

}  // namespace bcc32

#undef BCC32_ACTION_CONTRAST_HD
