#ifndef RL_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H
#define RL_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H

#include <learning_to_fly/simulator/parameters/reward_functions/abs_exp.h>
#include <learning_to_fly/simulator/parameters/reward_functions/squared.h>
#include <learning_to_fly/simulator/parameters/reward_functions/absolute.h>
#include <learning_to_fly/simulator/parameters/reward_functions/default.h>
#include <learning_to_fly/simulator/parameters/dynamics/crazy_flie.h>
#include <learning_to_fly/simulator/parameters/dynamics/crazyflie_brushless.h>
#include <learning_to_fly/simulator/parameters/reward_functions/brushless.h>
#include <learning_to_fly/simulator/parameters/init/default.h>
#include <learning_to_fly/simulator/parameters/termination/default.h>

#include <rl_tools/utils/generic/typing.h>

// Observation noise std (Gaussian). Defaults are the paper's; L2F_REAL_NOISE uses
// values measured on our Crazyflie 2.1 Brushless (PID hover + still tests):
// gyro ~0.069 rad/s, attitude ~0.006 rad, velocity ~0.010 m/s, position 3-10 mm.
// L2F_NOISE_SCALE multiplies the measured values (training with more noise than
// measured makes the policy ignore high-frequency sensor jitter).
#ifndef L2F_NOISE_SCALE
#define L2F_NOISE_SCALE 1
#endif
#ifdef L2F_REAL_NOISE
#define L2F_NOISE_POSITION (0.005 * L2F_NOISE_SCALE)
#define L2F_NOISE_ORIENTATION (0.006 * L2F_NOISE_SCALE)
#define L2F_NOISE_LINEAR_VELOCITY (0.010 * L2F_NOISE_SCALE)
#define L2F_NOISE_ANGULAR_VELOCITY (0.069 * L2F_NOISE_SCALE)
#else
#define L2F_NOISE_POSITION 0.001
#define L2F_NOISE_ORIENTATION 0.001
#define L2F_NOISE_LINEAR_VELOCITY 0.002
#define L2F_NOISE_ANGULAR_VELOCITY 0.002
#endif

namespace parameters{
    namespace rlt = rl_tools;
    struct DefaultAblationSpec{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = true;
        static constexpr bool ENABLE_CURRICULUM = true;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
        static constexpr bool INIT_NORMAL = true;
    };
    namespace builder {
        namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
        using namespace rlt::rl::environments::multirotor;
        template<typename T, typename TI, typename T_ABLATION_SPEC>
        struct environment {
            using ABLATION_SPEC = T_ABLATION_SPEC;
#ifdef L2F_BRUSHLESS
            static constexpr auto initial_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_brushless<T>;
            static constexpr auto target_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_curriculum_target_brushless<T>;
            static constexpr T MASS = 0.0413;
#else
            static constexpr auto initial_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
            static constexpr auto target_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_curriculum_target<T>;
            static constexpr T MASS = 0.027;
#endif
            static constexpr auto reward_function = ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? initial_reward_function : target_reward_function;


            using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
            using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

            using PARAMETERS_TYPE = rl_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, rl_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>>;

            static_assert(ABLATION_SPEC::INIT_NORMAL);
            static constexpr auto init_params = ABLATION_SPEC::INIT_NORMAL ?
                                                rl_tools::rl::environments::multirotor::parameters::init::orientation_biggest_angle<T, TI, 4, REWARD_FUNCTION>
                                                                           :
                                                rl_tools::rl::environments::multirotor::parameters::init::all_positions<T, TI, 4, REWARD_FUNCTION>;

            static constexpr PARAMETERS_TYPE parameters = {
#ifdef L2F_BRUSHLESS
                    rl_tools::rl::environments::multirotor::parameters::dynamics::crazyflie_brushless<T, TI, REWARD_FUNCTION>,
#else
                    rl_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION>,
#endif
                    {0.01}, // integration dt
                    {
                            init_params,
                            reward_function,
                            {   // Observation noise
                                    L2F_NOISE_POSITION * ABLATION_SPEC::OBSERVATION_NOISE, // position
                                    L2F_NOISE_ORIENTATION * ABLATION_SPEC::OBSERVATION_NOISE, // orientation
                                    L2F_NOISE_LINEAR_VELOCITY * ABLATION_SPEC::OBSERVATION_NOISE, // linear_velocity
                                    L2F_NOISE_ANGULAR_VELOCITY * ABLATION_SPEC::OBSERVATION_NOISE, // angular_velocity
                            },
                            {   // Action noise
                                    0, // std of additive gaussian noise onto the normalized action (-1, 1)
                            },
                            rl_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                    },
                    typename PARAMETERS_TYPE::Disturbances{
                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, MASS * 9.81 / 20 * ABLATION_SPEC::DISTURBANCE}, // random_force;
                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, MASS * 9.81 / 10000 * ABLATION_SPEC::DISTURBANCE} // random_torque;
                    }

            };

            using PARAMETERS = typename rl_tools::utils::typing::remove_cv_t<decltype(parameters)>;

            struct ENVIRONMENT_STATIC_PARAMETERS{
                // L2F_HISTORY: number of past actions in the actor's observation (paper: 32).
                // 0 = no action history input; the state still keeps the last action
                // (length 1), which the action-rate reward needs.
#ifdef L2F_HISTORY
                static constexpr TI ACTOR_HISTORY_LENGTH = L2F_HISTORY;
#else
                static constexpr TI ACTOR_HISTORY_LENGTH = 32;
#endif
                static constexpr TI ACTION_HISTORY_LENGTH = ACTOR_HISTORY_LENGTH > 0 ? ACTOR_HISTORY_LENGTH : 1;
                // L2F_POS_INTEGRAL: append the leaky position-error integral (3 values)
                // to the actor's and the critic's observation.
#ifdef L2F_POS_INTEGRAL
                using OBS_TAIL = observation::PositionIntegral<observation::PositionIntegralSpecification<T, TI>>;
#else
                using OBS_TAIL = observation::LastComponent<TI>;
#endif
#if defined(L2F_HISTORY) && L2F_HISTORY == 0
                using ACTOR_HISTORY = OBS_TAIL;
#else
                using ACTOR_HISTORY = rlt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, OBS_TAIL>>, OBS_TAIL>;
#endif
                // L2F_RPM_OBS: the actor also gets the 4 measured rotor speeds (noisy,
                // sometimes repeated; see Mission::rpm_noise), between the angular
                // velocity and the action history. The critic keeps the true speeds.
#ifdef L2F_RPM_OBS
                using ACTOR_ROTORS = observation::RotorSpeedsMeasured<observation::RotorSpeedsMeasuredSpecification<T, TI, ACTOR_HISTORY>>;
#else
                using ACTOR_ROTORS = ACTOR_HISTORY;
#endif
                using STATE_TYPE = rlt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                    rlt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY,
                        StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>,
                        StateRotors<T, TI, rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>>,
                    rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>;
                using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                        observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                                observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                        observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                                ACTOR_ROTORS>>>>>>>>;
                using OBSERVATION_TYPE_PRIVILEGED = rlt::utils::typing::conditional_t<ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC,
                    observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                        observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                            observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                    rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE,
                                        observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                            rlt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI, OBS_TAIL>>,
                                                OBS_TAIL
                                            >
                                        >>,
                                        rlt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI, OBS_TAIL>>,
                                                OBS_TAIL
                                        >
                                    >
                                >>
                            >>
                        >>
                    >>,
                    observation::NONE<TI>
                >;
                static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            };

            using ENVIRONMENT_SPEC = rlt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
            using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
        };
    }
    template<typename T, typename TI, typename ABLATION_SPEC>
    using environment = builder::environment<T, TI, ABLATION_SPEC>;
}

#endif
