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
// Crash penalty: the reward of the step on which a crash ends the episode is
// -L2F_CRASH_PENALTY instead of 0. A crash is more than 0.6 m from the target on
// any axis, touching the floor tilted more than 60 deg, rotating faster than
// L2F_OMEGA_CRASH, or a failed landing (timeout cut too high). A landing never
// gets it: without L2F_LANDING_CUT the drone rests on the floor until the episode
// ends; with it, the motor cut ends the episode with the landing payout below.
// Without a penalty, a crash scores 0, which beats the negative rewards of flying
// far (> ~32 cm) off target at the final position weight. 0 = the paper's reward.
#ifndef L2F_CRASH_PENALTY
#define L2F_CRASH_PENALTY 0
#endif
// Survival bonus: the constant reward of every step the drone is flying (the
// paper's 2). Losing it by crashing costs about bonus / (1 - gamma) = 200, so it
// works like a crash penalty of that size (train_mission.sh SURVIVAL_BONUS=0 to
// test the crash penalty alone).
#ifndef L2F_SURVIVAL_BONUS
#define L2F_SURVIVAL_BONUS 2
#endif
// TD3 discount, needed for the landing payout; src/config/config.h checks that it
// matches the one training uses.
#ifndef L2F_GAMMA
#define L2F_GAMMA 0.99
#endif
// Landing payout (with L2F_LANDING_CUT): the reward of the step on which the
// motor cut ends the episode is the survival bonus it gives up,
// bonus / (1 - gamma), so ending the episode by landing is as good as hovering on.
#define L2F_LANDING_PAYOUT ((L2F_SURVIVAL_BONUS) / (1 - (L2F_GAMMA)))
// Zig-zag penalty |a_t - 2 a_{t-1} + a_{t-2}|^2 (train_mission.sh SMOOTHING=zigzag):
// large for step-to-step chatter, zero for a steady ramp. Passing L2F_ZIGZAG_WEIGHT
// also keeps 2 past actions in the state (src/config/parameters.h); the actor's
// inputs do not change. Unset = off.
#ifdef L2F_ZIGZAG_WEIGHT
#define L2F_ZIGZAG_WEIGHT_VALUE (L2F_ZIGZAG_WEIGHT)
#else
#define L2F_ZIGZAG_WEIGHT_VALUE 0
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
            L2F_SURVIVAL_BONUS, // constant (survival bonus)
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
            L2F_ZIGZAG_WEIGHT_VALUE, // zigzag
            L2F_LANDING_PAYOUT, // landing payout
    };
    template<typename T>
    constexpr Squared<T> reward_squared_position_only_torque_curriculum_target_brushless = {
            false, // non-negative
            0.5, // scale
            L2F_SURVIVAL_BONUS, // constant (survival bonus)
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
            L2F_ZIGZAG_WEIGHT_VALUE, // zigzag
            L2F_LANDING_PAYOUT, // landing payout
    };
    // Untuned reward (L2F_REWARD_UNIFORM, multi-objective experiment): the same seven cost
    // terms with every weight 1 and no weight curriculum, i.e. no hand tuning. Not used by
    // any flown policy; the tuned weights above stay the default.
    template<typename T>
    constexpr Squared<T> reward_uniform_brushless = {
            false, // non-negative
            0.5, // scale
            L2F_SURVIVAL_BONUS, // constant (survival bonus)
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
            0, // zigzag
            L2F_LANDING_PAYOUT, // landing payout
    };
}
