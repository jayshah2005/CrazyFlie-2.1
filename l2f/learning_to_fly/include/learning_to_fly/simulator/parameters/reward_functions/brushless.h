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
            0, // termination penalty
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
            0, // termination penalty
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
}
