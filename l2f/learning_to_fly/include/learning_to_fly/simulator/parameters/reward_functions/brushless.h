#include "squared.h"

// Same weights as reward_squared_position_only_torque(_curriculum_target) in
// default.h; only the action baseline changes. The baseline is the normalized
// hover action: brushed 14.5k/21.7k RPM -> 0.334, brushless 15,460/24,000 RPM
// -> 2 * 0.644 - 1 = 0.288.
#define L2F_CF21BL_ACTION_BASELINE (0.288)
// Penalty on angular acceleration (rad/s^2)^2 to discourage motor chatter
// under realistic gyro noise; 0 = the paper's reward.
#ifndef L2F_ANG_ACC_WEIGHT
#define L2F_ANG_ACC_WEIGHT 0
#endif
// Penalty on the change of the normalized action between steps,
// |a_t - a_{t-1}|^2, to smooth motor commands directly; 0 = off.
#ifndef L2F_ACTION_RATE_WEIGHT
#define L2F_ACTION_RATE_WEIGHT 0
#endif
// Penalty on angular velocity (rad/s)^2 against wobble (the brushless paper
// rewards low rotation speed); 0 = off.
#ifndef L2F_ANG_VEL_WEIGHT
#define L2F_ANG_VEL_WEIGHT 0
#endif
// Crash penalty: the reward of the step on which the episode terminates is
// -L2F_CRASH_PENALTY instead of 0. Termination means a crash: more than 0.6 m
// from the target on any axis, or touching the floor tilted more than 60 deg.
// A landing never terminates (the drone rests on the floor upright until the
// episode ends), so it never gets this penalty; how well it lands is the
// separate landing-speed cost below. Without a penalty, a crash scores 0, which
// beats the negative rewards of flying far (> ~32 cm) off target at the final
// position weight. 0 = the paper's reward. With L2F_MOO the penalty is in part 0
// (survival), so the parts still add up to the reward.
#ifndef L2F_CRASH_PENALTY
#define L2F_CRASH_PENALTY 0
#endif
// Penalty on (downward speed)^2 within 10 cm of the floor: gentle touchdowns
// (mission training); 0 = off.
#ifndef L2F_LANDING_SPEED_WEIGHT
#define L2F_LANDING_SPEED_WEIGHT 0
#endif

namespace rl_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    constexpr Squared<T> reward_squared_position_only_torque_brushless = {
            false, // non-negative
            0.5, // scale
            2, // constant
            (T)(-(L2F_CRASH_PENALTY)), // termination penalty (crash)
            5, // position
            5, // orientation
            0.01, // linear_velocity
            L2F_ANG_VEL_WEIGHT, // angular_velocity
            0, // linear_acceleration
            L2F_ANG_ACC_WEIGHT, // angular_acceleration
            L2F_CF21BL_ACTION_BASELINE, // action baseline
            0.01, // action
            L2F_ACTION_RATE_WEIGHT, // action rate
            L2F_LANDING_SPEED_WEIGHT, // landing speed
    };
    template<typename T>
    constexpr Squared<T> reward_squared_position_only_torque_curriculum_target_brushless = {
            false, // non-negative
            0.5, // scale
            2, // constant
            (T)(-(L2F_CRASH_PENALTY)), // termination penalty (crash)
            40, // position
            5, // orientation
            1.00, // linear_velocity
            L2F_ANG_VEL_WEIGHT, // angular_velocity
            0, // linear_acceleration
            L2F_ANG_ACC_WEIGHT, // angular_acceleration
            L2F_CF21BL_ACTION_BASELINE, // action baseline
            1.00, // action
            L2F_ACTION_RATE_WEIGHT, // action rate
            L2F_LANDING_SPEED_WEIGHT, // landing speed
    };
    // Untuned reward (L2F_REWARD_UNIFORM, multi-objective experiment): the same seven cost
    // terms with every weight 1 and no weight curriculum, i.e. no hand tuning. Not used by
    // any flown policy; the tuned weights above stay the default.
    template<typename T>
    constexpr Squared<T> reward_uniform_brushless = {
            false, // non-negative
            0.5, // scale
            2, // constant
            (T)(-(L2F_CRASH_PENALTY)), // termination penalty (crash)
            1, // position
            1, // orientation
            1, // linear_velocity
            0, // angular_velocity
            0, // linear_acceleration
            1, // angular_acceleration
            L2F_CF21BL_ACTION_BASELINE, // action baseline
            1, // action
            1, // action rate
            1, // landing speed
    };
}
