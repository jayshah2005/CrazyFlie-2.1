#ifndef LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_PARAMETERS_REWARD_FUNCTIONS_SQUARED_H
#define LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_PARAMETERS_REWARD_FUNCTIONS_SQUARED_H

#include "../../multirotor.h"
#include <rl_tools/utils/generic/typing.h>
#include <rl_tools/utils/generic/vector_operations.h>

namespace rl_tools::rl::environments::multirotor::parameters::reward_functions{
    template<typename T>
    struct Squared{
        bool non_negative;
        T scale;
        T constant;
        T termination_penalty;
        T position;
        T orientation;
        T linear_velocity;
        T angular_velocity;
        T linear_acceleration;
        T angular_acceleration;
        T action_baseline;
        T action;
        T action_rate = 0; // weight on |a_t - a_{t-1}|^2 (needs the action history state)
        T landing_speed = 0; // weight on (downward speed)^2 within 10 cm of the floor (gentle touchdown)
        T zigzag = 0; // weight on |a_t - 2 a_{t-1} + a_{t-2}|^2 (needs 2 past actions in the state)
        T landing_payout = 0; // reward of the step on which a landing ends the episode (Mission::landing_cut)
        // time-to-target study (all 0 = off; set at runtime by the Python binding, L2F_W_*):
        T time = 0;        // weight on 1 per step while a static target is not reached (Mission::arrive_*)
        T power = 0;       // weight on rotor power above hover, max(0, sum (rpm / rpm_max)^3 minus its hover value)
        T speed_limit = 0; // weight on (speed - allowed speed)^2 above it (Mission::speed_cap, brake_acc)
        T energy = 0;      // weight on sum (rpm / rpm_max)^3 (hover included) while a static target is not reached
        T progress = 0;    // K, reward per metre closed (Mission::progress_cap; set by the Python binding, L2F_PROGRESS_K)
        struct Components{
            T orientation_cost;
            T position_cost;
            T linear_vel_cost;
            T angular_vel_cost;
            T linear_acc_cost;
            T angular_acc_cost;
            T action_cost;
            T action_rate_cost;
            T zigzag_cost;  // |a_t - 2 a_{t-1} + a_{t-2}|^2
            T landing_cost; // (downward speed)^2 near the floor
            T time_cost = 0;   // 1 while a static target is not reached
            T power_cost = 0;  // max(0, sum (rpm / rpm_max)^3 - its hover value)
            T speed_cost = 0;  // (speed - allowed speed)^2 above it
            T energy_cost = 0; // sum (rpm / rpm_max)^3 while a static target is not reached (energy per leg)
            T progress_gain = 0;     // d_t - gamma d_t+1 (an unweighted reward, not a cost)
            T progress_distance = 0; // d_t, charged on a crash step
            bool terminated;
            bool landed;    // terminated by a landing (motor cut), not a crash
            T weighted_cost;
            T scaled_weighted_cost;
            T reward;
        };
    };
    // Squared change of the action since the previous step. Only states that keep
    // an action history know the previous action; for the others it is 0.
    template<typename STATE, typename ACTION_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename STATE::T action_rate_cost(const STATE& state, const Matrix<ACTION_SPEC>& action){
        return 0;
    }
    template<typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename NEXT_COMPONENT, typename ACTION_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT T_S action_rate_cost(const rl::environments::multirotor::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action){
        T_S cost = 0;
        if constexpr(HISTORY_LENGTH > 0){
            for(TI_S i = 0; i < 4; i++){
                T_S d = get(action, 0, i) - state.action_history[HISTORY_LENGTH-1][i];
                cost += d * d;
            }
        }
        return cost;
    }
    // Squared second difference of the action, a_t - 2 a_{t-1} + a_{t-2}: zero for a
    // steady ramp, large for step-to-step zig-zag (chatter). Needs at least 2 past
    // actions in the state; otherwise 0.
    template<typename STATE, typename ACTION_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename STATE::T zigzag_cost(const STATE& state, const Matrix<ACTION_SPEC>& action){
        return 0;
    }
    template<typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename NEXT_COMPONENT, typename ACTION_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT T_S zigzag_cost(const rl::environments::multirotor::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action){
        T_S cost = 0;
        if constexpr(HISTORY_LENGTH > 1){
            for(TI_S i = 0; i < 4; i++){
                T_S d = get(action, 0, i) - 2 * state.action_history[HISTORY_LENGTH-1][i] + state.action_history[HISTORY_LENGTH-2][i];
                cost += d * d;
            }
        }
        return cost;
    }
    // Progress reward (Squared::progress = K, Mission::progress_cap / progress_gamma): flat position
    // cost and no velocity cost beyond the cap, and the distance gain d_t - gamma d_t+1 (any target,
    // moving or not; d = |position|, the error to the target). Out of line, like the terms below.
    template<typename DEVICE, typename SPEC, typename STATE, typename COMPONENTS>
    __attribute__((noinline)) void progress_terms(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, const STATE& next_state, COMPONENTS& components){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        const auto& m = env.mission;
        T d2 = 0, d2_next = 0;
        for(TI i = 0; i < 3; i++){
            d2 += state.position[i] * state.position[i];
            d2_next += next_state.position[i] * next_state.position[i];
        }
        const T d = math::sqrt(device.math, d2), d_next = math::sqrt(device.math, d2_next);
        if(d > m.progress_cap){
            components.position_cost = m.progress_cap * m.progress_cap;
            components.linear_vel_cost = 0;
        }
        components.progress_gain = d - m.progress_gamma * d_next;
        components.progress_distance = d;
    }
    // Reward terms of the time-to-target study (Mission::pos_cap, arrive_*, speed_cap; reward
    // parts time, power, speed_limit). Static target = no target velocity. Kept out of line so
    // the other reward terms compile exactly as before.
    template<typename DEVICE, typename SPEC, typename STATE, typename COMPONENTS>
    __attribute__((noinline)) void time_to_target_terms(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, COMPONENTS& components){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        const auto& m = env.mission;
        T tv2 = 0, v2 = 0;
        for(TI i = 0; i < 3; i++){
            tv2 += state.target_velocity[i] * state.target_velocity[i];
            v2 += state.linear_velocity[i] * state.linear_velocity[i];
        }
        const bool static_target = tv2 == 0;
        if(m.pos_cap > 0 && static_target){
            const T cap2 = m.pos_cap * m.pos_cap;
            if(components.position_cost > cap2){
                // Huber: linear beyond the cap, so a far target still pulls (a flat cap left the
                // policies hovering ~30 cm off) without the large quadratic cost of a 1.5 m jump
                const T d = math::sqrt(device.math, components.position_cost);
                components.position_cost = cap2 + 2 * m.pos_cap * (d - m.pos_cap);
                components.linear_vel_cost = 0;  // far from it: no speed penalty
            }
        }
        const bool arrived = components.position_cost < m.arrive_pos * m.arrive_pos && v2 < m.arrive_speed * m.arrive_speed;
        components.time_cost = static_target && !arrived ? (T)1 : (T)0;
        const auto& dyn = env.parameters.dynamics;
        const T hover = math::sqrt(device.math, dyn.mass * -dyn.gravity[2] / 4 / dyn.thrust_constants[2]) / dyn.action_limit.max;
        T power = 0;
        for(TI i = 0; i < 4; i++){
            const T rho = state.rpm[i] / dyn.action_limit.max;
            power += rho * rho * rho;
        }
        // only thrust above hover costs: a negative cost would pay the policy for idling (falling),
        // which outweighed the crash penalty and made every run of the first study drop (2026-11-23)
        components.power_cost = power > 4 * hover * hover * hover ? power - 4 * hover * hover * hover : (T)0;
        // energy per leg: all rotor power (hover included) until the drone has arrived, so a slow
        // leg pays hover power for longer and a fast one pays for accelerating and braking
        components.energy_cost = static_target && !arrived ? power : (T)0;
        const T speed = math::sqrt(device.math, v2);
        T allowed = m.speed_cap;
        if(m.brake_acc > 0 && static_target){
            // the speed that can still stop at the target with brake_acc
            T d2 = 0;
            for(TI i = 0; i < 3; i++){ d2 += state.position[i] * state.position[i]; }
            T stop = math::sqrt(device.math, 2 * m.brake_acc * math::sqrt(device.math, d2));
            stop = stop > m.brake_min_speed ? stop : m.brake_min_speed;
            allowed = stop < m.speed_cap ? stop : m.speed_cap;
        }
        components.speed_cost = speed > allowed ? (speed - allowed) * (speed - allowed) : (T)0;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename rl::environments::multirotor::parameters::reward_functions::Squared<T>::Components reward_components(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Squared<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng){
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        typename rl::environments::multirotor::parameters::reward_functions::Squared<T>::Components components;
        components.orientation_cost = 1 - state.orientation[0] * state.orientation[0]; //math::abs(device.math, 2 * math::acos(device.math, quaternion_w));
        components.position_cost = state.position[0] * state.position[0] + state.position[1] * state.position[1] + state.position[2] * state.position[2];
        // velocity relative to the (possibly moving) target; target_velocity is 0 for a static target
        T rel_v[3];
        for(TI i = 0; i < 3; i++){
            rel_v[i] = state.linear_velocity[i] - state.target_velocity[i];
        }
        components.linear_vel_cost = rel_v[0] * rel_v[0] + rel_v[1] * rel_v[1] + rel_v[2] * rel_v[2];
        components.angular_vel_cost = state.angular_velocity[0] * state.angular_velocity[0] + state.angular_velocity[1] * state.angular_velocity[1] + state.angular_velocity[2] * state.angular_velocity[2];
        T linear_acc[3];
        T angular_acc[3];
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.linear_velocity, state.linear_velocity, linear_acc);
        components.linear_acc_cost = (linear_acc[0] * linear_acc[0] + linear_acc[1] * linear_acc[1] + linear_acc[2] * linear_acc[2]) / (env.parameters.integration.dt * env.parameters.integration.dt);
        utils::vector_operations::sub<DEVICE, T, 3>(next_state.angular_velocity, state.angular_velocity, angular_acc);
        components.angular_acc_cost = (angular_acc[0] * angular_acc[0] + angular_acc[1] * angular_acc[1] + angular_acc[2] * angular_acc[2]) / (env.parameters.integration.dt * env.parameters.integration.dt);

        T action_diff[ACTION_DIM];
        for(TI i = 0; i < ACTION_DIM; i++){
            action_diff[i] = get(action, 0, i) - params.action_baseline;
        }
        components.action_cost = utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        components.action_cost *= components.action_cost;
        components.action_rate_cost = action_rate_cost(state, action);
        // time-to-target study: computed only when one of its options is on, in a function
        // of its own, so recipes without it keep exactly the same arithmetic (2026-11-22)
        if(params.time != 0 || params.power != 0 || params.speed_limit != 0 || params.energy != 0 || env.mission.pos_cap > 0){
            time_to_target_terms(device, env, state, components);
        }
        // progress reward: after the time terms, so their arrival test sees the true distance
        if(params.progress != 0){
            progress_terms(device, env, state, next_state, components);
        }
        // gentle touchdown: penalize sinking fast close to the floor (mission episodes only)
        T landing_cost = 0;
        if(state.floor_z > -50 && state.position[2] - state.floor_z < (T)0.1 && state.linear_velocity[2] < 0){
            landing_cost = state.linear_velocity[2] * state.linear_velocity[2];
        }
        components.weighted_cost = params.position * components.position_cost + params.orientation * components.orientation_cost + params.linear_velocity * components.linear_vel_cost + params.angular_velocity * components.angular_vel_cost + params.linear_acceleration * components.linear_acc_cost + params.angular_acceleration * components.angular_acc_cost + params.action * components.action_cost + params.action_rate * components.action_rate_cost + params.landing_speed * landing_cost;
        // time-to-target terms, only added when weighted (older recipes reproduce exactly)
        if(params.time != 0) components.weighted_cost += params.time * components.time_cost;
        if(params.power != 0) components.weighted_cost += params.power * components.power_cost;
        if(params.speed_limit != 0) components.weighted_cost += params.speed_limit * components.speed_cost;
        if(params.energy != 0) components.weighted_cost += params.energy * components.energy_cost;
        // zig-zag (only added when weighted, so recipes without it reproduce exactly)
        components.zigzag_cost = zigzag_cost(state, action);
        if(params.zigzag != 0){
            components.weighted_cost += params.zigzag * components.zigzag_cost;
        }
        bool terminated_flag = terminated(device, env, next_state, rng);
        components.landing_cost = landing_cost;
        components.terminated = terminated_flag;
        components.landed = terminated_flag && next_state.landed == 1;
        components.scaled_weighted_cost = params.scale * components.weighted_cost;

        if(terminated_flag){
            // a landing (motor cut) is paid the bonus it gives up; a crash or a failed
            // landing gets the termination (crash) penalty
            components.reward = components.landed ? params.landing_payout : params.termination_penalty;
            if(params.progress != 0 && !components.landed){
                // a crash does not refund the distance still to fly
                components.reward -= params.progress * components.progress_distance;
            }
        }
        else{
            components.reward = -components.scaled_weighted_cost + params.constant;
            if(params.progress != 0){
                components.reward += params.progress * components.progress_gain;
            }
            components.reward = (components.reward > 0 || !params.non_negative) ? components.reward : 0;
        }

        return components;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Squared<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        constexpr typename SPEC::TI cadence = 1;
        auto components = reward_components(device, env, params, state, action, next_state, rng);
        add_scalar(device, device.logger, "reward/orientation_cost", components.orientation_cost, cadence);
        add_scalar(device, device.logger, "reward/position_cost",    components.position_cost, cadence);
        add_scalar(device, device.logger, "reward/linear_vel_cost",  components.linear_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_vel_cost", components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/linear_acc_cost",  components.linear_acc_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_acc_cost", components.angular_acc_cost, cadence);
        add_scalar(device, device.logger, "reward/action_cost",      components.action_cost, cadence);
        add_scalar(device, device.logger, "reward/action_rate_cost", components.action_rate_cost, cadence);
        add_scalar(device, device.logger, "reward/zigzag_cost", components.zigzag_cost, cadence);
        add_scalar(device, device.logger, "reward/pre_exp",         -components.weighted_cost, cadence);

        add_scalar(device, device.logger, "reward_weighted/orientation_cost", params.orientation          * components.orientation_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/position_cost",    params.position             * components.position_cost,    cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_vel_cost",  params.linear_velocity      * components.linear_vel_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_vel_cost", params.angular_velocity     * components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_acc_cost" , params.linear_acceleration  * components.linear_acc_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_acc_cost", params.angular_acceleration * components.angular_acc_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/action_cost",      params.action               * components.action_cost,      cadence);
        // log share of the weighted abs cost
        add_scalar(device, device.logger, "reward_share/orientation", params.orientation          * components.orientation_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/position",    params.position             * components.position_cost    / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/linear_vel",  params.linear_velocity      * components.linear_vel_cost  / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/angular_vel", params.angular_velocity     * components.angular_vel_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/linear_acc",  params.linear_acceleration  * components.linear_acc_cost  / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/angular_acc", params.angular_acceleration * components.angular_acc_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/action",      params.action               * components.action_cost      / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/const",       components.reward/params.constant, cadence);

        add_scalar(device, device.logger, "reward/weighted_cost",        components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/scaled_weighted_cost", components.scaled_weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/reward",               components.reward, cadence);
        add_scalar(device, device.logger, "reward/reward_zero",          components.reward == 0, cadence);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::parameters::reward_functions::Squared<T>& params, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action,  const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng){
        auto components = reward_components(device, env, params, state, action, next_state, rng);
        return components.reward;
    }
}

#endif