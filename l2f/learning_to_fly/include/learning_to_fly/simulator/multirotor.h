#ifndef LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_MULTIROTOR_H
#define LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_MULTIROTOR_H

#include <rl_tools/utils/generic/typing.h>
#include <cstdlib>
#ifdef L2F_SENSOR_SIM
#include "sensor_sim.h"  // simulated sensors + the flown Kalman estimator (l2f/estimator)
#endif

// Position-error integral observation (must match the firmware): leak time
// constant (s) and clip (m s).
#define L2F_POS_INTEGRAL_TAU 5.0
#define L2F_POS_INTEGRAL_MAX 1.0

namespace rl_tools::rl::environments::multirotor{
    template <typename T, typename TI, TI N, typename T_REWARD_FUNCTION>
    struct ParametersBase{
        struct Dynamics{
            struct ActionLimit{
                T min;
                T max;
            };
            T rotor_positions[N][3];
            T rotor_thrust_directions[N][3];
            T rotor_torque_directions[N][3];
            T thrust_constants[3];
            T torque_constant;
            T mass;
            T gravity[3];
            T J[3][3];
            T J_inv[3][3];
            T rpm_time_constant;
            ActionLimit action_limit;
            T drag = 0;  // linear air drag (N s/m): F = -drag * v; Mission::drag_max randomizes it
        };
        struct Integration{
            T dt;
        };
        struct MDP{
            using REWARD_FUNCTION = T_REWARD_FUNCTION;
            struct Initialization{
                T guidance;
                T max_position;
                T max_angle;
                T max_linear_velocity;
                T max_angular_velocity;
                bool relative_rpm; //(specification from -1 to 1)
                T min_rpm; // -1 for default limit when relative_rpm is true, -1 if relative_rpm is false
                T max_rpm; //  1 for default limit when relative_rpm is true, -1 if relative_rpm is false
            };
            struct Termination{
                bool enabled = false;
                T position_threshold;
                T linear_velocity_threshold;
                T angular_velocity_threshold;
            };
            struct ObservationNoise{
                T position;
                T orientation;
                T linear_velocity;
                T angular_velocity;
            };
            struct ActionNoise{
                T normalized_rpm; // std of additive gaussian noise onto the normalized action (-1, 1)
            };
            Initialization init;
            REWARD_FUNCTION reward;
            ObservationNoise observation_noise;
            ActionNoise action_noise;
            Termination termination;
        };
        Dynamics dynamics;
        Integration integration;
        MDP mdp;
    };
    template <typename T, typename TI, typename T_NEXT_COMPONENT>
    struct ParametersDisturbances: T_NEXT_COMPONENT{
        struct Disturbances{
            struct UnivariateGaussian{
                T mean;
                T std;
            };
            UnivariateGaussian random_force;
            UnivariateGaussian random_torque;
        };
        Disturbances disturbances;
    };

    template <typename T, typename TI, typename T_NEXT_COMPONENT>
    struct ParametersDomainRandomization: T_NEXT_COMPONENT{
        struct DomainRandomization{
            struct UnivariateGaussian{
                T mean;
                T std;
            };
            UnivariateGaussian J_factor;
            UnivariateGaussian mass_factor;
        };
        DomainRandomization domain_randomization;
    };


//    enum class LatentStateType{
//        Empty,
//        RandomForce
//    };
//    enum class StateType{
//        Base,
//        BaseRotors,
//        BaseRotorsHistory,
//    };
//    enum class ObservationType{
//        Normal,
//        DoubleQuaternion,
//        RotationMatrix
//    };
    namespace observation{
        template <typename T_TI>
        struct LastComponent{
            static constexpr T_TI DIM = 0;
        };
        template <typename T_TI>
        struct NONE{
            static constexpr T_TI DIM = 0;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionSpecificationPrivileged: PositionSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct Position{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationQuaternionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationQuaternionSpecificationPrivileged: OrientationQuaternionSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct OrientationQuaternion{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationRotationMatrixSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationRotationMatrixSpecificationPrivileged: OrientationRotationMatrixSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct OrientationRotationMatrix{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 9;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct LinearVelocitySpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct LinearVelocitySpecificationPrivileged: LinearVelocitySpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct LinearVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocitySpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocitySpecificationPrivileged: AngularVelocitySpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct AngularVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RotorSpeedsSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RotorSpeeds{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        // Measured rotor speeds for the actor (L2F_RPM_OBS): StateBase::rpm_meas,
        // the true speeds with telemetry-like noise and repeated values.
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RotorSpeedsMeasuredSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RotorSpeedsMeasured{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ActionHistorySpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        };
        template <typename SPEC>
        struct ActionHistory{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI HISTORY_LENGTH = SPEC::HISTORY_LENGTH;
            static constexpr TI ACTION_DIM = 4;
            static constexpr TI CURRENT_DIM = ACTION_DIM * HISTORY_LENGTH;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        // Leaky position-error integral (see StateBase::pos_integral).
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionIntegralSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct PositionIntegral{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RandomForceSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RandomForce{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 6;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
    }


    template <typename T_T, typename T_TI>
    struct StateBase{
        using T = T_T;
        using TI = T_TI;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI DIM = 13;
        T position[3];
        T orientation[4];
        T linear_velocity[3];
        T angular_velocity[3];
        // Mission extension (all zero / disabled unless Multirotor::mission is
        // enabled). Not integrated: post_integration carries them forward.
        // position is relative to the (possibly moving) target; the target moves
        // along a trajectory and the frame is shifted after every step.
        T target_velocity[3];  // velocity of the target (observations and reward use v - target_velocity)
        T floor_z;             // floor height (drone resting height) in the target-relative frame
        T traj_t;              // time since episode start (s)
        TI traj_type;          // 0 static, 1 ramp (constant velocity for traj[3] s), 2 horizontal circle,
                               // 3 figure-eight, 4 static target with a gust, 5 target step
        TI task;               // episode type (label only, no effect on dynamics or reward):
                               // 0 free flight, 1 take-off, 2 landing, 3 circle, 4 line,
                               // 5 figure-eight, 6 fast circle, 7 point-to-point, 8 gust
        T traj[6];             // ramp: vx, vy, vz, duration | circle: radius, omega, phase |
                               // eight: size, omega, phase, cos/sin heading | gust: accel xyz,
                               // start, duration | step: offset xyz, time
        T delay_steps;         // state-estimate delay in simulation steps (0..2, fractional)
        T ground_effect;       // ground-effect strength k (0 = none), see Mission
        T glitch_offset[3];    // position-estimate glitch added to the actor's observation (m)
        T glitch_vel[3];       // velocity-estimate spike that comes with the glitch (m/s, decays)
        TI glitch_steps;       // steps the current glitch lasts (0 = none)
        // Leaky integral of the position error (m s), an observation so the policy
        // can remove steady offsets itself: I <- clip((1 - dt/tau) I + e dt, +-max).
        // The firmware computes the same (rl_tools_controller.c, L2F_POS_INTEGRAL).
        T pos_integral[3];
        T prev[2][13];         // true base state 1 and 2 steps ago (position in the current frame)
        // Per-motor command -> speed mismatch for this episode: the motor settles at
        // motor_gain * commanded rpm + motor_offset (see Mission::motor_gain_dr).
        T motor_gain[4];
        T motor_offset[4];
        // Measured rotor speeds as the actor sees them (normalized like the actions,
        // -1..1); only used as an observation with L2F_RPM_OBS.
        T rpm_meas[4];
        // This episode's position-noise std per axis (m); 0 = the fixed
        // observation_noise.position (see Mission::pos_noise_min / pos_noise_max).
        T pos_noise_sigma[3];
        // Landing like the flight script (Mission::landing_cut): this episode's cut
        // height (m), the outcome (0 flying, 1 landed: motors cut, 2 failed landing:
        // timeout cut too high above the floor), and the touchdown speed after the
        // cut, sqrt(v_down^2 + 2 g h) (m/s), for evaluation.
        T cut_height;
        TI landed;
        T touchdown_speed;
        // Actuator dead time (Mission::act_delay_p): with act_delay = 1 the motors get the
        // previous step's command (10 ms, measured 2026-11-22); action_pending holds it (rpm).
        T action_pending[4];
        TI act_delay;
        // Slowly wandering common motor gain (Mission::gain_walk_sigma): the motors settle
        // at (1 + gain_walk) x the command, an AR(1) process (battery sag, 2026-11-22).
        T gain_walk;
#ifdef L2F_SENSOR_SIM
        // Sensor path (sensor_sim.h): the actor observes the estimator's output, not the
        // truth. world_offset turns the target-relative frame into the Lighthouse (room)
        // frame: world = position + world_offset (floor at world z = 0). est_position is in
        // the target-relative frame like position; est_prev holds the estimate 1 and 2 steps
        // ago in the world frame (position, quaternion, velocity, gyro), for the delay.
        T world_offset[3];
        T est_position[3];
        T est_orientation[4];
        T est_velocity[3];
        T gyro_obs[3];         // the actor's gyro sample (rad/s)
        T est_prev[2][13];
        TI est_failed;         // the estimator failed (a firmware assert): terminated as a crash
#endif
    };
    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct StateRotors: T_NEXT_COMPONENT{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI PARENT_DIM = NEXT_COMPONENT::DIM;
        static constexpr TI DIM = PARENT_DIM + 4;
        T rpm[4];
    };
    template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, typename T_NEXT_COMPONENT>
    struct StateRotorsHistory: StateRotors<T_T, T_TI, T_NEXT_COMPONENT>{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = StateRotors<T, TI, T_NEXT_COMPONENT>;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        static constexpr TI PARENT_DIM = StateRotors<T, TI, NEXT_COMPONENT>::DIM;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI DIM = PARENT_DIM + HISTORY_LENGTH * ACTION_DIM;
        T action_history[HISTORY_LENGTH][4];
    };
    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct StateRandomForce: T_NEXT_COMPONENT{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 6 + NEXT_COMPONENT::DIM;
        T force[3];
        T torque[3];
    };
    
    template <typename T, typename TI>
    struct StaticParametersDefault{
//        static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
//        static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
//        static constexpr LatentStateType LATENT_STATE_TYPE = LatentStateType::Empty;
//        static constexpr StateType STATE_TYPE = StateType::Base;
//        static constexpr ObservationType OBSERVATION_TYPE = ObservationType::Normal;
        using STATE_TYPE = StateBase<T, TI>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                                 observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                                 observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                 observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = observation::NONE<TI>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS, typename T_STATIC_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
        using STATIC_PARAMETERS = T_STATIC_PARAMETERS;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments{
    template <typename SPEC>
    struct Multirotor{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMETERS = typename SPEC::PARAMETERS;
        using REWARD_FUNCTION = typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION;
//        static constexpr TI STATE_DIM = 13;
        static constexpr TI ACTION_DIM = 4;

        static constexpr TI ACTION_HISTORY_LENGTH = SPEC::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH;

//        static constexpr multirotor::LatentStateType LATENT_STATE_TYPE = SPEC::STATIC_PARAMETERS::LATENT_STATE_TYPE;
//        static constexpr multirotor::StateType STATE_TYPE = SPEC::STATIC_PARAMETERS::STATE_TYPE;
        using State = typename SPEC::STATIC_PARAMETERS::STATE_TYPE;
        using Observation = typename SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE;
        using ObservationPrivileged = typename SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;

//        using LatentState = utils::typing::conditional_t<
//            LATENT_STATE_TYPE == multirotor::LatentStateType::Empty,
//            multirotor::StateLatentEmpty<T, TI>,
//            multirotor::StateLatentRandomForce<T, TI>
//        >;
//        using State = utils::typing::conditional_t<
//            STATE_TYPE == multirotor::StateType::Base,
//            multirotor::StateBase<T, TI, LatentState>,
//            utils::typing::conditional_t<
//                STATE_TYPE == multirotor::StateType::BaseRotors,
//                multirotor::StateBaseRotors<T, TI, LatentState>,
//                multirotor::StateBaseRotorsHistory<T, TI, SPEC::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH, LatentState>
//            >
//        >;

//        static constexpr TI OBSERVATION_DIM_BASE = 3 + 3 + 3;
//        static constexpr TI OBSERVATION_DIM_ORIENTATION = OBSERVATION_TYPE == multirotor::ObservationType::Normal ? 4 : (OBSERVATION_TYPE == multirotor::ObservationType::DoubleQuaternion ? (2*4) : (9));
//        static constexpr TI OBSERVATION_DIM_ACTION_HISTORY = (STATE_TYPE == multirotor::StateType::BaseRotorsHistory) * ACTION_DIM * ACTION_HISTORY_LENGTH;
//        static constexpr TI OBSERVATION_DIM = OBSERVATION_DIM_BASE + OBSERVATION_DIM_ORIENTATION + OBSERVATION_DIM_ACTION_HISTORY;
//        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = STATE_TYPE == multirotor::StateType::Base || LATENT_STATE_TYPE == multirotor::LatentStateType::RandomForce;
//        static constexpr TI OBSERVATION_DIM_PRIVILEGED_LATENT_STATE = (LATENT_STATE_TYPE == multirotor::LatentStateType::RandomForce ? LatentState::DIM : 0);
//        static constexpr TI OBSERVATION_DIM_PRIVILEGED = PRIVILEGED_OBSERVATION_AVAILABLE ? (
//                OBSERVATION_DIM_BASE + OBSERVATION_DIM_ORIENTATION
//                + (STATE_TYPE == multirotor::StateType::BaseRotors || STATE_TYPE == multirotor::StateType::BaseRotorsHistory ? ACTION_DIM : 0)
//                + OBSERVATION_DIM_PRIVILEGED_LATENT_STATE
//        ) : 0;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = !utils::typing::is_same_v<typename SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED, multirotor::observation::NONE<TI>>;
        using STATIC_PARAMETERS = typename SPEC::STATIC_PARAMETERS;
        typename SPEC::PARAMETERS parameters;
        typename SPEC::PARAMETERS::Dynamics current_dynamics;
        // Mission training (take-off from the floor, moving targets, landing),
        // state-estimate delay and per-episode domain randomization. Defaults
        // come from compile flags (L2F_MISSION) so training needs no code
        // changes; evaluators can switch parts off at runtime.
        struct Mission{
#ifdef L2F_MISSION
            bool enabled = true;
#else
            bool enabled = false;
#endif
            // episode type probabilities (rest: the original free-flight start);
            // every value can be overridden at runtime with the L2F_* variable named
            // next to it
            T p_takeoff = env_or("L2F_P_TAKEOFF", 0.2);
            T p_land = env_or("L2F_P_LAND", 0.2);
            T p_circle = env_or("L2F_P_CIRCLE", 0.2);
            T p_line = env_or("L2F_P_LINE", 0.1);
            // harder episode types (MOO framing B, phase 2), off by default: figure-eight,
            // fast circle, point-to-point (target step), hover in a gust
            T p_eight = env_or("L2F_P_EIGHT", 0);
            T p_fast_circle = env_or("L2F_P_FAST_CIRCLE", 0);
            T p_p2p = env_or("L2F_P_P2P", 0);
            T p_gust = env_or("L2F_P_GUST", 0);
            // Waypoint episodes (time-to-target study), off by default: the target jumps
            // waypoint_min..max m (scaled by the curriculum), then moves to a second point on
            // the same line; the time objective counts the steps until arrival
            T p_waypoint = env_or("L2F_P_WAYPOINT", 0);
            T waypoint_min = env_or("L2F_WAYPOINT_MIN", 0.3), waypoint_max = env_or("L2F_WAYPOINT_MAX", 1.5);
            // Arrival at a static target (reward part "time"): within arrive_pos m and slower
            // than arrive_speed m/s
            T arrive_pos = env_or("L2F_ARRIVE_POS", 0.05), arrive_speed = env_or("L2F_ARRIVE_SPEED", 0.2);
            // L2F_POS_CAP > 0: for a static target, the position cost is quadratic within pos_cap and
            // linear beyond it (Huber), and the velocity cost applies only within pos_cap of it (far away
            // the time objective and the linear pull drive, near it precision); 0 = off (plain squared)
            T pos_cap = env_or("L2F_POS_CAP", 0);
            // speed_cap: reward part "speed_limit" = (speed - speed_cap)^2 above it (safety);
            // speed_crash > 0: faster than this ends the episode like a crash
            T speed_cap = env_or("L2F_SPEED_CAP", 1.5);
            T speed_crash = env_or("L2F_SPEED_CRASH", 0);
            // brake_acc > 0: for a static target the allowed speed is the one that can still stop
            // at the target, max(brake_min_speed, sqrt(2 brake_acc distance)), at most speed_cap
            // (a fixed cap let the policies dash into the speed crash; 2026-11-23)
            T brake_acc = env_or("L2F_BRAKE_ACC", 0), brake_min_speed = env_or("L2F_BRAKE_MIN_SPEED", 0.3);
            // the firmware's input clips (rl_tools_controller.c: rlt.pdlp 0.5 m, rlt.vdlp 2 m/s per
            // axis; its integral adds up the clipped error); 0 = off
            T obs_pos_clip = env_or("L2F_OBS_POS_CLIP", 0), obs_vel_clip = env_or("L2F_OBS_VEL_CLIP", 0);
            // waypoint jump curriculum: >= 0 replaces progress for the jump size (set per step by
            // train.py from L2F_WAYPOINT_RAMP); -1 = use progress
            T waypoint_progress = -1;
            // Sim-to-real options (2026-11-22 flights), all off by default:
            // linear air drag U(0, drag_max) N s/m per episode
            T drag_max = env_or("L2F_DRAG_MAX", 0);
            // probability that an episode has the 1-step (10 ms) actuator dead time
            T act_delay_p = env_or("L2F_ACT_DELAY_P", 0);
            // absolute motor time-constant range (s); when tau_max > 0 it replaces the
            // relative +-dr_motor_tau (low battery: the rotors followed at 0.09-0.19 s)
            T tau_min = env_or("L2F_MOTOR_TAU_MIN", 0), tau_max = env_or("L2F_MOTOR_TAU_MAX", 0);
            // common motor gain wander: std gain_walk_sigma, correlation time gain_walk_tau s
            T gain_walk_sigma = env_or("L2F_GAIN_WALK", 0), gain_walk_tau = env_or("L2F_GAIN_WALK_TAU", 0.3);
            T delay_max_steps = env_or("L2F_DELAY_MAX_MS", 12) / 10;  // uniform 0..max (dt = 10 ms)
            bool domain_randomization = env_or("L2F_DR", 1) != 0;
            T dr_scale = env_or("L2F_DR_SCALE", 1);  // 1 = the paper's ranges
            // L2F_DR_MASS: absolute mass range (e.g. 0.2 = +-20%), overrides 0.1 * dr_scale
            T dr_mass = std::getenv("L2F_DR_MASS") ? env_or("L2F_DR_MASS", 0) : 0.1 * dr_scale;
            T dr_inertia = 0.2 * dr_scale;
            // L2F_DR_MOTOR_TAU: absolute motor time-constant range (e.g. 0.25 = +-25%), overrides
            // 0.2 * dr_scale. Identified 2026-11-22: 0.045 s at 3.66 V, 0.057 s at 3.84 V (model 0.060)
            T dr_motor_tau = std::getenv("L2F_DR_MOTOR_TAU") ? env_or("L2F_DR_MOTOR_TAU", 0) : 0.2 * dr_scale;
            T dr_thrust = 0.2 * dr_scale, dr_torque = 0.2 * dr_scale, dr_cog = 0.01 * dr_scale;
            // Curriculum: training ramps this from 0 to 1 over L2F_MISSION_RAMP steps
            // (src/steps/curriculum.h); episode-type probabilities and randomization
            // ranges are scaled by it. Evaluation keeps 1.
            T progress = 1;
            T ramp_steps = env_or("L2F_MISSION_RAMP", 500000);
            // Circle episodes: radius up to circle_r_max (m), speed up to circle_v_max (m/s).
            T circle_r_max = env_or("L2F_CIRCLE_R_MAX", 1.2);
            T circle_v_max = env_or("L2F_CIRCLE_V_MAX", 1.0);
            // Ground effect (Cheeseman-Bennett): thrust x 1 / (1 - k (R / 4h)^2), h =
            // rotor height above the floor = (z - floor_z) + ge_h0, k ~ U(0, ge_k_max)
            // per episode (not identified on our drone), capped at ge_max_ratio.
            T ge_k_max = env_or("L2F_GE_K_MAX", 2.0);
            T ge_rotor_radius = 0.0275;  // 55 mm propellers
            T ge_h0 = 0.02;              // rotor height above the floor at rest (m)
            T ge_max_ratio = 1.4;
            // Position-estimate glitches (seen on the drone: a 36 cm Lighthouse jump
            // lasting ~0.3 s): per step, with probability glitch_p, the actor's
            // position observation jumps by U(glitch_min, glitch_max) m in a random
            // direction for U(0.05, glitch_t_max) s. The critic sees the true state.
            T glitch_p = env_or("L2F_GLITCH_P", 0.0);
            T glitch_min = env_or("L2F_GLITCH_MIN", 0.05);
            T glitch_max = env_or("L2F_GLITCH_MAX", 0.4);
            T glitch_t_max = env_or("L2F_GLITCH_T_MAX", 0.4);
            T glitch_vel_max = env_or("L2F_GLITCH_VEL", 3.0);  // velocity spike up to this x the jump size (1/s)
            // Per-motor mismatch (with domain randomization): each motor settles at
            // gain * commanded rpm + offset, gain ~ 1 + U(+-motor_gain_dr), offset ~
            // U(+-motor_offset_dr) rpm, per episode. Measured on the drone: 300-700 rpm
            // between the motor model and each motor's telemetry (rpm_quality.py).
            T motor_gain_dr = env_or("L2F_MOTOR_GAIN", 0.0);
            T motor_offset_dr = env_or("L2F_MOTOR_OFFSET", 0.0);
            // One gain for all four motors on top (L2F_MOTOR_GAIN_ALL, with domain randomization):
            // the firmware's voltage-compensated command-to-speed map was 18-23% off with a fresh
            // battery on 2026-11-22 (2% at 3.4-3.6 V). 0 = off (no random draws, as before).
            T motor_gain_all_dr = env_or("L2F_MOTOR_GAIN_ALL", 0.0);
            // ESC idle (L2F_MOTOR_IDLE_RPM): an armed motor never turns slower than this, whatever the
            // command (the firmware clips the command at 0 and the ESC holds idle; its arming check
            // expects 1000-3500 rpm). 0 = off: the simulated motor can spin down to 0.
            T motor_idle_rpm = env_or("L2F_MOTOR_IDLE_RPM", 0.0);
            // Measured rotor speeds (L2F_RPM_OBS): Gaussian noise (normalized units,
            // 1 = 12000 rpm) and the chance per step that the previous value repeats.
            // Measured: ~100 rpm noise, ~14% repeats at 100 Hz (rpm_quality.py).
            T rpm_noise = env_or("L2F_RPM_NOISE", 0.01);
            T rpm_hold_p = env_or("L2F_RPM_HOLD", 0.15);
            // Gyro range: the actor's angular-velocity reading is clipped at
            // +-gyro_clip rad/s, like the BMI088 at the firmware's +-2000 deg/s
            // (34.9 rad/s). 0 = off.
            T gyro_clip = env_or("L2F_GYRO_CLIP", 0);
            // Rotation crash: an episode ends as a crash when any axis of the true
            // angular velocity exceeds omega_crash rad/s. 0 = the termination
            // preset's limit (1000 rad/s in fast_learning, i.e. never).
            T omega_crash = env_or("L2F_OMEGA_CRASH", 0);
            // Per-axis position noise: per episode, the std of the actor's position
            // noise on axis i is drawn from U(pos_noise_min[i], pos_noise_max[i]) m.
            // pos_noise_max[i] = 0 keeps the fixed observation_noise.position.
            T pos_noise_min[3] = {env_or("L2F_POS_NOISE_MIN_X", 0), env_or("L2F_POS_NOISE_MIN_Y", 0), env_or("L2F_POS_NOISE_MIN_Z", 0)};
            T pos_noise_max[3] = {env_or("L2F_POS_NOISE_MAX_X", 0), env_or("L2F_POS_NOISE_MAX_Y", 0), env_or("L2F_POS_NOISE_MAX_Z", 0)};
            // Landing like the flight script (rl_trigger.py --mode mission, values in
            // config/landing.yaml). With landing_cut, landing episodes descend at
            // U(descent_min, descent_max) m/s to cut_below m under the floor, and the
            // motors are cut, ending the episode as a landing, when the target is less
            // than cut_window m above the floor and the estimated height is below the
            // episode's cut height h_cut ~ U(cut_h_min, cut_h_max); or floor_timeout s
            // after the target reached its lowest point. A timeout cut more than
            // cut_max_height above the floor counts as a crash.
            bool landing_cut = env_or("L2F_LANDING_CUT", 0) != 0;
            T cut_h_min = env_or("L2F_CUT_H_MIN", 0.05);
            T cut_h_max = env_or("L2F_CUT_H_MAX", 0.07);
            T cut_window = env_or("L2F_CUT_WINDOW", 0.1);
            T cut_below = env_or("L2F_CUT_BELOW", 0.02);
            T descent_min = env_or("L2F_DESCENT_MIN", 0.1);
            T descent_max = env_or("L2F_DESCENT_MAX", 0.25);
            T floor_timeout = env_or("L2F_FLOOR_TIMEOUT", 2.0);
            T cut_max_height = 0.1;
            // Scale of the per-episode random external force (ABLATION_SPEC::DISTURBANCE,
            // sigma m g / 20 per axis). The flown estimator reads a constant sideways force as
            // tilt (its flight model has thrust along body z only), so on the sensor path this
            // sets part of the estimate's attitude error. 1 = the recipe.
            T force_scale = env_or("L2F_FORCE_SCALE", 1);
            static T env_or(const char* name, double fallback){
                const char* s = std::getenv(name);
                return (T)(s ? std::atof(s) : fallback);
            }
        };
        Mission mission;
#ifdef L2F_SENSOR_SIM
        // This environment's estimator and sensor draws, created on its first episode
        // (mission::sensor_runtime). mutable: the rl_tools step/observe take the env const.
        mutable learning_to_fly::sensor_sim::Runtime* sensor_rt = nullptr;
#endif
        typename SPEC::PARAMETERS::Dynamics nominal_dynamics;
        bool nominal_dynamics_valid = false;
    };
}

#endif