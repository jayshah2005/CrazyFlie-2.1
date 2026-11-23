#ifndef LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_OPERATIONS_GENERIC_H
#define LEARNING_TO_FLY_IN_SECONDS_SIMULATOR_OPERATIONS_GENERIC_H

#include "multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

namespace rl_tools{
    // State arithmetic for RK4 integration
    // scalar multiply
    template<typename DEVICE, typename T, typename TI, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI>& state, T2 scalar, typename rl::environments::multirotor::StateBase<T, TI>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         = scalar * state.position[i]        ;
            out.orientation[i]      = scalar * state.orientation[i]     ;
            out.linear_velocity[i]  = scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] = scalar * state.angular_velocity[i];
        }
        out.orientation[3] = scalar * state.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& state, T2 scalar, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& out){
        scalar_multiply(device, (const NEXT_COMPONENT&)state, scalar, (NEXT_COMPONENT&)out);
        for(int i = 0; i < 4; ++i){
            out.rpm[i] = scalar * state.rpm[i];
        }
    }
    template<typename DEVICE, typename STATE, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const STATE& state, T2 scalar, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply(device, (const typename STATE::NEXT_COMPONENT&)state, scalar, (typename STATE::NEXT_COMPONENT&)out);
    }
    // scalar multiply in place
    template<typename DEVICE, typename T, typename TI, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::multirotor::StateBase<T, TI>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename STATE, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, STATE& state, T2 scalar, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply(device, (typename STATE::NEXT_COMPONENT&)state, scalar);
    }

    template<typename DEVICE, typename T, typename TI, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI>& state, T2 scalar, typename rl::environments::multirotor::StateBase<T, TI>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         += scalar * state.position[i]        ;
            out.orientation[i]      += scalar * state.orientation[i]     ;
            out.linear_velocity[i]  += scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] += scalar * state.angular_velocity[i];
        }
        out.orientation[3] += scalar * state.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& state, T2 scalar, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& out){
        scalar_multiply_accumulate(device, static_cast<const NEXT_COMPONENT&>(state), scalar, static_cast<NEXT_COMPONENT&>(out));
        for(int i = 0; i < 4; ++i){
            out.rpm[i] += scalar * state.rpm[i];
        }
    }
    template<typename DEVICE, typename STATE, typename T2>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const STATE& state, T2 scalar, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }

    template<typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI>& s1, const typename rl::environments::multirotor::StateBase<T, TI>& s2, typename rl::environments::multirotor::StateBase<T, TI>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         = s1.position[i] + s2.position[i];
            out.orientation[i]      = s1.orientation[i] + s2.orientation[i];
            out.linear_velocity[i]  = s1.linear_velocity[i] + s2.linear_velocity[i];
            out.angular_velocity[i] = s1.angular_velocity[i] + s2.angular_velocity[i];
        }
        out.orientation[3] = s1.orientation[3] + s2.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& s1, const typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& s2, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& out){
        add_accumulate(device, static_cast<const NEXT_COMPONENT&>(s1), static_cast<const NEXT_COMPONENT&>(s2), static_cast<NEXT_COMPONENT&>(out));
        for(int i = 0; i < 4; ++i){
            out.rpm[i] = s1.rpm[i] + s2.rpm[i];
        }
    }
    template<typename DEVICE, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const STATE& s1, const STATE& s2, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        add_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(s1), static_cast<const typename STATE::NEXT_COMPONENT&>(s2), static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
    template<typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI>& s, typename rl::environments::multirotor::StateBase<T, TI>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& s, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const STATE& s, STATE& out, utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        add_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(s), static_cast<const typename STATE::NEXT_COMPONENT&>(out), static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
}

#include <rl_tools/utils/generic/integrators.h>


namespace rl_tools::rl::environments::multirotor {
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateBase<T, TI>& state, const T* action, StateBase<T, TI>& state_change) {
        using STATE = StateBase<T, TI>;

        T thrust[3];
        T torque[3];
        thrust[0] = 0;
        thrust[1] = 0;
        thrust[2] = 0;
        torque[0] = 0;
        torque[1] = 0;
        torque[2] = 0;
        // flops: N*23 => 4 * 23 = 92
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
            // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
            T rpm = action[i_rotor];
            T thrust_magnitude = params.dynamics.thrust_constants[0] + params.dynamics.thrust_constants[1] * rpm + params.dynamics.thrust_constants[2] * rpm * rpm;
            T rotor_thrust[3];
            utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
            utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);

            utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[i_rotor], thrust_magnitude * params.dynamics.torque_constant, torque);
            utils::vector_operations::cross_product_accumulate<DEVICE, T>(params.dynamics.rotor_positions[i_rotor], rotor_thrust, torque);
        }

        // linear_velocity_global
        state_change.position[0] = state.linear_velocity[0];
        state_change.position[1] = state.linear_velocity[1];
        state_change.position[2] = state.linear_velocity[2];

        // angular_velocity_global
        // flops: 16
        quaternion_derivative<DEVICE, T>(state.orientation, state.angular_velocity, state_change.orientation);

        // linear_acceleration_global
        // flops: 21
        rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust, state_change.linear_velocity);
        // flops: 4
        utils::vector_operations::scalar_multiply<DEVICE, T, 3>(state_change.linear_velocity, 1 / params.dynamics.mass);
        utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, state_change.linear_velocity);
        if(params.dynamics.drag > 0){  // linear air drag (Dynamics::drag)
            for(TI i = 0; i < 3; i++){
                state_change.linear_velocity[i] -= params.dynamics.drag / params.dynamics.mass * state.linear_velocity[i];
            }
        }

        T vector[3];
        T vector2[3];

        // angular_acceleration_local
        // flops: 9
        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, state.angular_velocity, vector);
        // flops: 6
        utils::vector_operations::cross_product<DEVICE, T>(state.angular_velocity, vector, vector2);
        utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        // flops: 9
        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, vector, state_change.angular_velocity);
        // total flops: (quadrotor): 92 + 16 + 21 + 4 + 9 + 6 + 9 = 157
//        multirotor_dynamics<DEVICE, T, TI, PARAMETERS>(device, params, (const typename STATE::LATENT_STATE&)state, action, state_change);
//        multirotor_dynamics(device, params, (const typename STATE::LATENT_STATE&)state, action, state_change);
    }
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS, typename NEXT_COMPONENT>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRandomForce<T, TI, NEXT_COMPONENT>& state, const T* action, StateRandomForce<T, TI, NEXT_COMPONENT>& state_change){
        multirotor_dynamics(device, params, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(state_change));

        state_change.linear_velocity[0] += state.force[0] / params.dynamics.mass;
        state_change.linear_velocity[1] += state.force[1] / params.dynamics.mass;
        state_change.linear_velocity[2] += state.force[2] / params.dynamics.mass;

        T angular_acceleration[3];

        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, state.torque, angular_acceleration);
        utils::vector_operations::add_accumulate<DEVICE, T, 3>(angular_acceleration, state_change.angular_velocity);
    }
    template<typename DEVICE, typename T, typename TI, typename NEXT_COMPONENT, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(DEVICE& device, const PARAMETERS& params, const StateRotors<T, TI, NEXT_COMPONENT>& state, const T* action, StateRotors<T, TI, NEXT_COMPONENT>& state_change) {
        multirotor_dynamics(device, params, static_cast<const NEXT_COMPONENT&>(state), state.rpm, static_cast<NEXT_COMPONENT&>(state_change));
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
            state_change.rpm[i_rotor] = (action[i_rotor] - state.rpm[i_rotor]) * 1/params.dynamics.rpm_time_constant;
        }

    }
    template<typename DEVICE, typename T, typename PARAMETERS, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics_dispatch(DEVICE& device, const PARAMETERS& params, const STATE& state, const T* action, STATE& state_change) {
        // this dispatch function is required to pass the multirotor dynamics function to the integrator (euler, rk4) as a template parameter (so that it can be inlined/optimized at compile time)
        // If we would try to pass the multirotor_dynamics function directly the state type-based overloading would make the inference of the auto template parameter for the dynamics function in the integrator function impossible
//        multirotor_dynamics<DEVICE, T, typename DEVICE::index_t, typename STATE::LATENT_STATE, PARAMETERS>(device, params, state, action, state_change);
        multirotor_dynamics(device, params, state, action, state_change);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE&, rl::environments::Multirotor<SPEC>){

    }
    // ---- Mission extension helpers (see Multirotor::Mission) ----
    namespace rl::environments::multirotor::mission{
        template<typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void reset_fields(StateBase<T, TI>& state){
            for(TI i = 0; i < 3; i++){
                state.target_velocity[i] = 0;
            }
            state.floor_z = -100;  // no floor
            state.traj_t = 0;
            for(TI i = 0; i < 4; i++){
                state.action_pending[i] = 0;
            }
            state.act_delay = 0;
            state.gain_walk = 0;
            state.traj_type = 0;
            state.task = 0;
            for(TI i = 0; i < 6; i++){
                state.traj[i] = 0;
            }
            state.delay_steps = 0;
            state.ground_effect = 0;
            state.glitch_steps = 0;
            for(TI i = 0; i < 3; i++){
                state.glitch_offset[i] = 0;
                state.glitch_vel[i] = 0;
                state.pos_integral[i] = 0;
                state.pos_noise_sigma[i] = 0;
            }
            state.cut_height = 0;
            state.landed = 0;
            state.touchdown_speed = 0;
#ifdef L2F_SENSOR_SIM
            for(TI i = 0; i < 3; i++){
                state.world_offset[i] = 0;
                state.est_position[i] = state.position[i];
                state.est_velocity[i] = state.linear_velocity[i];
                state.gyro_obs[i] = state.angular_velocity[i];
            }
            for(TI i = 0; i < 4; i++){
                state.est_orientation[i] = state.orientation[i];
            }
            for(TI k = 0; k < 2; k++){
                for(TI i = 0; i < 13; i++){
                    state.est_prev[k][i] = 0;
                }
            }
            state.est_failed = 0;
#endif
            for(TI k = 0; k < 2; k++){
                for(TI i = 0; i < 3; i++){
                    state.prev[k][0 + i] = state.position[i];
                    state.prev[k][7 + i] = state.linear_velocity[i];
                    state.prev[k][10 + i] = state.angular_velocity[i];
                }
                for(TI i = 0; i < 4; i++){
                    state.prev[k][3 + i] = state.orientation[i];
                }
            }
        }
        // Target position (relative to its start) and velocity at time t.
        template<typename DEVICE, typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void target_at(DEVICE& device, const StateBase<T, TI>& state, T t, T* p, T* v){
            for(TI i = 0; i < 3; i++){ p[i] = 0; v[i] = 0; }
            if(state.traj_type == 1){
                // ramp: constant velocity traj[0..2] between traj[4] and traj[4] + traj[3]
                T start = state.traj[4], duration = state.traj[3];
                T tau = t - start;
                tau = tau < 0 ? 0 : (tau > duration ? duration : tau);
                bool moving = t >= start && t < start + duration;
                for(TI i = 0; i < 3; i++){
                    p[i] = state.traj[i] * tau;
                    v[i] = moving ? state.traj[i] : 0;
                }
            }
            else if(state.traj_type == 2){
                // horizontal circle: radius traj[0], angular rate traj[1], phase traj[2]
                T r = state.traj[0], w = state.traj[1], a = w * t + state.traj[2];
                p[0] = r * math::cos(device.math, a);
                p[1] = r * math::sin(device.math, a);
                v[0] = -r * w * math::sin(device.math, a);
                v[1] =  r * w * math::cos(device.math, a);
            }
            else if(state.traj_type == 3){
                // horizontal figure-eight (lemniscate of Gerono) of size traj[0], angular
                // rate traj[1], phase traj[2], rotated to heading (cos traj[3], sin traj[4]);
                // positions relative to the start point
                T A = state.traj[0], w = state.traj[1], a = w * t + state.traj[2], a0 = state.traj[2];
                T x = A * (math::sin(device.math, a) - math::sin(device.math, a0));
                T y = A / 2 * (math::sin(device.math, 2 * a) - math::sin(device.math, 2 * a0));
                T vx = A * w * math::cos(device.math, a), vy = A * w * math::cos(device.math, 2 * a);
                T c = state.traj[3], s = state.traj[4];
                p[0] = c * x - s * y;  p[1] = s * x + c * y;
                v[0] = c * vx - s * vy; v[1] = s * vx + c * vy;
            }
            else if(state.traj_type == 5){
                // target step: jumps by traj[0..2] at time traj[3], static otherwise
                if(t >= state.traj[3]){
                    for(TI i = 0; i < 3; i++){ p[i] = state.traj[i]; }
                }
            }
            else if(state.traj_type == 6){
                // waypoints: jumps by traj[0..2] at time traj[3], then to traj[5] x traj[0..2]
                // at time traj[4] (on the same line), static otherwise
                T s = t >= state.traj[4] ? state.traj[5] : (t >= state.traj[3] ? (T)1 : (T)0);
                for(TI i = 0; i < 3; i++){ p[i] = s * state.traj[i]; }
            }
            // traj_type 4 (gust): static target, the gust acts in post_integration
        }
        // Nominal motors: no per-motor mismatch.
        template<typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void reset_motors(StateBase<T, TI>& state){
            for(TI i = 0; i < 4; i++){
                state.motor_gain[i] = 1;
                state.motor_offset[i] = 0;
            }
        }
        // Measured rotor speeds start equal to the true ones (call after setting rpm).
        template<typename SPEC, typename T, typename TI, typename NEXT_COMPONENT>
        RL_TOOLS_FUNCTION_PLACEMENT void sync_rpm_meas(const Multirotor<SPEC>& env, StateRotors<T, TI, NEXT_COMPONENT>& state){
            const auto& lim = env.parameters.dynamics.action_limit;
            for(TI i = 0; i < 4; i++){
                state.rpm_meas[i] = (state.rpm[i] - lim.min) / (lim.max - lim.min) * 2 - 1;
            }
        }
#ifdef L2F_SENSOR_SIM
        // ---- sensor path (L2F_SENSOR_SIM, sensor_sim.h) ----
        // uniform / normal draws from the simulator's RNG for sensor_sim.h
        template<typename DEVICE, typename T, typename RNG>
        struct SensorDraw{
            RNG& rng;
            double uniform(double lo, double hi){ return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)lo, (T)hi, rng); }
            double normal(double sigma){ return sigma > 0 ? random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, (T)sigma, rng) : 0.0; }
        };
        // This environment object's sensor runtime (created on first use; a copied
        // environment gets its own).
        template<typename SPEC>
        learning_to_fly::sensor_sim::Runtime& sensor_runtime(const Multirotor<SPEC>& env){
            if(!env.sensor_rt || env.sensor_rt->owner != &env){
                env.sensor_rt = new learning_to_fly::sensor_sim::Runtime();
                env.sensor_rt->owner = &env;
            }
            return *env.sensor_rt;
        }
        // the estimate as a 13-vector (world position, quaternion, velocity, gyro)
        template<typename T, typename TI>
        void estimate_world(const StateBase<T, TI>& state, T out[13]){
            for(TI i = 0; i < 3; i++){
                out[0 + i] = state.est_position[i] + state.world_offset[i];
                out[7 + i] = state.est_velocity[i];
                out[10 + i] = state.gyro_obs[i];
            }
            for(TI i = 0; i < 4; i++){
                out[3 + i] = state.est_orientation[i];
            }
        }
        // Episode start: place the episode in the room (Lighthouse frame: the start target
        // within +-0.3 m of the room centre, the floor at z = 0; free flight without a floor
        // at 0.6..1.5 m), draw the sensors and start the estimator converged at the truth.
        template<typename DEVICE, typename SPEC, typename T, typename TI, typename RNG>
        void sensor_begin(DEVICE& device, const Multirotor<SPEC>& env, StateBase<T, TI>& state, RNG& rng){
            SensorDraw<DEVICE, T, RNG> d{rng};
            state.world_offset[0] = (T)d.uniform(-0.3, 0.3);
            state.world_offset[1] = (T)d.uniform(-0.3, 0.3);
            state.world_offset[2] = state.floor_z > -10 ? -state.floor_z : (T)d.uniform(0.6, 1.5);
            float p[3], v[3], q[4];
            for(TI i = 0; i < 3; i++){
                p[i] = (float)(state.position[i] + state.world_offset[i]);
                v[i] = (float)state.linear_velocity[i];
                state.pos_noise_sigma[i] = 0;  // the noise comes from the sensors now
            }
            for(TI i = 0; i < 4; i++){
                q[i] = (float)state.orientation[i];
            }
            auto& rt = sensor_runtime(env);
            learning_to_fly::sensor_sim::begin_episode(rt, p, v, q, d);
            for(TI i = 0; i < 3; i++){  // the converged start estimate
                state.est_position[i] = (T)rt.out.position[i] - state.world_offset[i];
                state.est_velocity[i] = (T)rt.out.velocity[i];
                state.gyro_obs[i] = state.angular_velocity[i];
            }
            for(TI i = 0; i < 4; i++){
                state.est_orientation[i] = (T)rt.out.quat[i];
            }
            state.est_failed = 0;
            T now[13];
            estimate_world(state, now);
            for(TI k = 0; k < 2; k++){
                for(TI i = 0; i < 13; i++){
                    state.est_prev[k][i] = now[i];
                }
            }
        }
        // One step of the sensor path (post_integration, after the frame shift and floor
        // contact): sensors from the truth between state and next_state, 10 estimator ticks.
        template<typename DEVICE, typename SPEC, typename T, typename TI, typename ACTION_SPEC, typename RNG>
        void sensor_step(DEVICE& device, const Multirotor<SPEC>& env, const StateBase<T, TI>& state, const Matrix<ACTION_SPEC>& action, StateBase<T, TI>& next_state, RNG& rng){
            static_assert(learning_to_fly::sensor_sim::TICKS_PER_STEP == 10, "the sensor path assumes 10 ms steps");
            SensorDraw<DEVICE, T, RNG> d{rng};
            learning_to_fly::sensor_sim::Truth x;
            for(TI i = 0; i < 3; i++){
                x.p0[i] = (float)(state.position[i] + state.world_offset[i]);
                x.p1[i] = (float)(next_state.position[i] + next_state.world_offset[i]);
                x.v0[i] = (float)state.linear_velocity[i];
                x.v1[i] = (float)next_state.linear_velocity[i];
                x.w0[i] = (float)state.angular_velocity[i];
                x.w1[i] = (float)next_state.angular_velocity[i];
            }
            for(TI i = 0; i < 4; i++){
                x.q0[i] = (float)state.orientation[i];
                x.q1[i] = (float)next_state.orientation[i];
            }
            bool thrust = false;  // supervisor.c: a motor above idle (action -1 = idle)
            for(TI i = 0; i < 4; i++){
                thrust = thrust || get(action, 0, i) > (T)-0.99;
            }
            auto& rt = sensor_runtime(env);
            if(!rt.est){
                // the episode start (mission::sample, from the rotor-state sample_initial_state)
                // never ran: no estimator to step
                throw std::runtime_error("L2F_SENSOR_SIM: no estimator; the sensor path needs the mission episode setup with rotor state (ROTOR_DELAY)");
            }
            float gyro[3];
            learning_to_fly::sensor_sim::simulate_step(rt, x, thrust, d, gyro);
            T prev_now[13];
            estimate_world(state, prev_now);
            for(TI i = 0; i < 13; i++){
                next_state.est_prev[1][i] = state.est_prev[0][i];
                next_state.est_prev[0][i] = prev_now[i];
            }
            for(TI i = 0; i < 3; i++){
                next_state.est_position[i] = (T)rt.out.position[i] - next_state.world_offset[i];
                next_state.est_velocity[i] = (T)rt.out.velocity[i];
                next_state.gyro_obs[i] = (T)gyro[i];
            }
            for(TI i = 0; i < 4; i++){
                next_state.est_orientation[i] = (T)rt.out.quat[i];
            }
            next_state.est_failed = rt.failed ? 1 : 0;
        }
#endif
        // Per-episode setup: domain randomization, episode type, delay.
        template<typename DEVICE, typename SPEC, typename T, typename TI_S, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void sample(DEVICE& device, Multirotor<SPEC>& env, StateBase<T, TI_S>& state, T* rpm, RNG& rng){
            using TI = typename DEVICE::index_t;
            auto& m = env.mission;
            auto U = [&](T a, T b){ return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), a, b, rng); };
            if(!env.nominal_dynamics_valid){
                env.nominal_dynamics = env.parameters.dynamics;
                env.nominal_dynamics_valid = true;
            }
            const T k = m.progress;  // curriculum: 0 = plain hover training, 1 = full
            if(m.domain_randomization){
                auto d = env.nominal_dynamics;
                d.mass *= U(1 - k * m.dr_mass, 1 + k * m.dr_mass);
                for(TI i = 0; i < 3; i++){
                    T f = U(1 - k * m.dr_inertia, 1 + k * m.dr_inertia);
                    d.J[i][i] *= f;
                    d.J_inv[i][i] /= f;
                }
                d.rpm_time_constant *= U(1 - k * m.dr_motor_tau, 1 + k * m.dr_motor_tau);
                if(m.tau_max > 0){  // absolute range (Mission::tau_min / tau_max), faded in by k
                    const T tau0 = env.nominal_dynamics.rpm_time_constant;
                    d.rpm_time_constant = tau0 + k * (U(m.tau_min, m.tau_max) - tau0);
                }
                if(m.drag_max > 0){
                    d.drag = U(0, k * m.drag_max);
                }
                T kt = U(1 - k * m.dr_thrust, 1 + k * m.dr_thrust);
                for(TI i = 0; i < 3; i++){
                    d.thrust_constants[i] *= kt;
                }
                d.torque_constant *= U(1 - k * m.dr_torque, 1 + k * m.dr_torque);
                T cx = U(-k * m.dr_cog, k * m.dr_cog), cy = U(-k * m.dr_cog, k * m.dr_cog);
                for(TI r = 0; r < 4; r++){
                    d.rotor_positions[r][0] -= cx;
                    d.rotor_positions[r][1] -= cy;
                }
                env.parameters.dynamics = d;
            }
            const auto& d = env.parameters.dynamics;
            T hover_rpm = math::sqrt(device.math, d.mass * -d.gravity[2] / 4 / d.thrust_constants[2]);
            auto clamp_rpm = [&](T x){ return x < d.action_limit.min ? d.action_limit.min : (x > d.action_limit.max ? d.action_limit.max : x); };
            auto set_yaw_tilt = [&](T max_tilt, T max_yaw){
                // random yaw, then a tilt about a random horizontal axis
                T yaw = U(-max_yaw, max_yaw), tilt = U(0, max_tilt), axis = U(0, 2 * math::PI<T>);
                T cy = math::cos(device.math, yaw / 2), sy = math::sin(device.math, yaw / 2);
                T ct = math::cos(device.math, tilt / 2), st = math::sin(device.math, tilt / 2);
                T ax = math::cos(device.math, axis), ay = math::sin(device.math, axis);
                // q = q_tilt * q_yaw, q_tilt = (ct, st*ax, st*ay, 0), q_yaw = (cy, 0, 0, sy)
                state.orientation[0] = ct * cy;
                state.orientation[1] = st * ax * cy + st * ay * sy;
                state.orientation[2] = st * ay * cy - st * ax * sy;
                state.orientation[3] = ct * sy;
            };
            auto near_hover = [&](){
                for(TI i = 0; i < 3; i++){
                    state.position[i] = U(-0.1, 0.1);
                    state.linear_velocity[i] = U(-0.2, 0.2);
                    state.angular_velocity[i] = U(-0.5, 0.5);
                }
                set_yaw_tilt(10.0 / 180.0 * math::PI<T>, 30.0 / 180.0 * math::PI<T>);
                for(TI i = 0; i < 4; i++){
                    rpm[i] = clamp_rpm(hover_rpm * U(0.9, 1.1));
                }
            };
            reset_fields(state);
            reset_motors(state);
            if(m.domain_randomization && m.motor_gain_dr > 0){
                for(TI i = 0; i < 4; i++){
                    state.motor_gain[i] = 1 + U(-k * m.motor_gain_dr, k * m.motor_gain_dr);
                }
            }
            if(m.domain_randomization && m.motor_offset_dr > 0){
                for(TI i = 0; i < 4; i++){
                    state.motor_offset[i] = U(-k * m.motor_offset_dr, k * m.motor_offset_dr);
                }
            }
            if(m.domain_randomization && m.motor_gain_all_dr > 0){
                const T all = 1 + U(-k * m.motor_gain_all_dr, k * m.motor_gain_all_dr);
                for(TI i = 0; i < 4; i++){
                    state.motor_gain[i] *= all;
                }
            }
            state.delay_steps = U(0, k * m.delay_max_steps);
            if(m.act_delay_p > 0){
                state.act_delay = U(0, 1) < k * m.act_delay_p ? 1 : 0;
            }
            if(m.gain_walk_sigma > 0){
                state.gain_walk = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, k * m.gain_walk_sigma, rng);
            }
            state.ground_effect = U(0, k * m.ge_k_max);
            // per-axis position noise (no random numbers drawn when off, so older
            // recipes reproduce exactly)
            for(TI i = 0; i < 3; i++){
                if(m.pos_noise_max[i] > 0){
                    state.pos_noise_sigma[i] = U(m.pos_noise_min[i], m.pos_noise_max[i]);
                }
            }
            // dividing by k grows the mission episode types from 0 to their full share
            T r = k > 0 ? U(0, 1) / k : 2;
            if(r < m.p_takeoff){
                // resting on the floor; the target starts at the drone and climbs
                state.position[0] = U(-0.05, 0.05);
                state.position[1] = U(-0.05, 0.05);
                state.position[2] = 0;
                for(TI i = 0; i < 3; i++){
                    state.linear_velocity[i] = 0;
                    state.angular_velocity[i] = 0;
                }
                set_yaw_tilt(0, 20.0 / 180.0 * math::PI<T>);
                T idle = U(0, 0.4) * hover_rpm;
                for(TI i = 0; i < 4; i++){
                    rpm[i] = clamp_rpm(idle);
                }
                state.floor_z = 0;
                state.task = 1;
                T vz = U(0.15, 0.8);
                T height = U(0.2, 0.8);
                T start = U(0, 1.0);
                if(start + height / vz > 3.5){
                    height = vz * (3.5 - start);
                }
                state.traj_type = 1;
                state.traj[2] = vz;
                state.traj[3] = height / vz;
                state.traj[4] = start;
            }
            else if(r < m.p_takeoff + m.p_land){
                // hovering; the target descends to (slightly below) the floor
                near_hover();
                state.task = 2;
                T height = U(0.1, 0.8);
                T vz, below;
                if(m.landing_cut){
                    // the flight script's descent (config/landing.yaml)
                    vz = U(m.descent_min, m.descent_max);
                    below = m.cut_below;
                }
                else{
                    vz = U(0.1, 0.5);
                    below = U(0, 0.03);
                }
                T start = U(0, 1.0);
                if(m.landing_cut){
                    state.cut_height = U(m.cut_h_min, m.cut_h_max);
                    // reach the floor by 3.5 s so the cut happens inside the 5 s episode
                    if(start + (height + below) / vz > 3.5){
                        height = vz * (3.5 - start) - below;
                        height = height > (T)0.05 ? height : (T)0.05;
                    }
                }
                state.floor_z = -height;
                state.traj_type = 1;
                state.traj[2] = -vz;
                state.traj[3] = (height + below) / vz;
                state.traj[4] = start;
            }
            else if(r < m.p_takeoff + m.p_land + m.p_circle){
                // horizontal circle
                near_hover();
                state.task = 3;
                state.floor_z = -U(0.3, 1.0);
                T radius = U(0.15, m.circle_r_max);
                T speed = U(0.1, m.circle_v_max);
                T dir = U(0, 1) < 0.5 ? -1 : 1;
                state.traj_type = 2;
                state.traj[0] = radius;
                state.traj[1] = dir * speed / radius;
                state.traj[2] = U(0, 2 * math::PI<T>);
                // the drone starts near the target; optionally already moving with it
                T p[3], v[3];
                target_at(device, state, (T)0, p, v);
                bool matched = U(0, 1) < 0.5;
                for(TI i = 0; i < 3; i++){
                    state.target_velocity[i] = v[i];
                    if(matched){
                        state.linear_velocity[i] += v[i];
                    }
                }
            }
            else if(r < m.p_takeoff + m.p_land + m.p_circle + m.p_line){
                // straight line at constant speed
                near_hover();
                state.task = 4;
                state.floor_z = -U(0.3, 1.0);
                T speed = U(0.05, 0.5);
                T heading = U(0, 2 * math::PI<T>), climb = U(-0.3, 0.3);
                state.traj_type = 1;
                state.traj[0] = speed * math::cos(device.math, heading);
                state.traj[1] = speed * math::sin(device.math, heading);
                state.traj[2] = speed * climb;
                state.traj[3] = U(1, 5);
                state.traj[4] = 0;
                for(TI i = 0; i < 3; i++){
                    state.target_velocity[i] = state.traj[i];
                }
            }
            // Harder episode types (off unless L2F_P_EIGHT / _FAST_CIRCLE / _P2P / _GUST are
            // set; with all four 0 no branch is taken and no random numbers are drawn)
            else if(r < m.p_takeoff + m.p_land + m.p_circle + m.p_line + m.p_eight){
                // figure-eight, peak speed up to 1.5 m/s, peak acceleration <= 6 m/s^2
                near_hover();
                state.task = 5;
                state.floor_z = -U(0.3, 1.0);
                T A = U(0.3, 0.8);
                T v_max = math::sqrt(device.math, 6 * A);
                T speed = U(0.3, v_max < 1.5 ? v_max : (T)1.5);
                T dir = U(0, 1) < 0.5 ? -1 : 1;
                T heading = U(0, 2 * math::PI<T>);
                state.traj_type = 3;
                state.traj[0] = A;
                state.traj[1] = dir * speed / (A * math::sqrt(device.math, (T)2));
                state.traj[2] = U(0, 2 * math::PI<T>);
                state.traj[3] = math::cos(device.math, heading);
                state.traj[4] = math::sin(device.math, heading);
                T p[3], v[3];
                target_at(device, state, (T)0, p, v);
                bool matched = U(0, 1) < 0.5;
                for(TI i = 0; i < 3; i++){
                    state.target_velocity[i] = v[i];
                    if(matched){
                        state.linear_velocity[i] += v[i];
                    }
                }
            }
            else if(r < m.p_takeoff + m.p_land + m.p_circle + m.p_line + m.p_eight + m.p_fast_circle){
                // fast circle: 1-2 m/s, centripetal acceleration <= 6 m/s^2
                near_hover();
                state.task = 6;
                state.floor_z = -U(0.3, 1.0);
                T radius = U(0.4, 1.0);
                T v_max = math::sqrt(device.math, 6 * radius);
                T speed = U(1.0, v_max < 2.0 ? v_max : (T)2.0);
                T dir = U(0, 1) < 0.5 ? -1 : 1;
                state.traj_type = 2;
                state.traj[0] = radius;
                state.traj[1] = dir * speed / radius;
                state.traj[2] = U(0, 2 * math::PI<T>);
                T p[3], v[3];
                target_at(device, state, (T)0, p, v);
                bool matched = U(0, 1) < 0.5;
                for(TI i = 0; i < 3; i++){
                    state.target_velocity[i] = v[i];
                    if(matched){
                        state.linear_velocity[i] += v[i];
                    }
                }
            }
            else if(r < m.p_takeoff + m.p_land + m.p_circle + m.p_line + m.p_eight + m.p_fast_circle + m.p_p2p){
                // point-to-point: the target jumps 0.2-0.45 m (the position termination is
                // 0.6 m) in a random, mostly horizontal direction after 0.3-2 s
                near_hover();
                state.task = 7;
                state.floor_z = -U(0.5, 1.0);
                T dist = U(0.2, 0.45);
                T heading = U(0, 2 * math::PI<T>), climb = U(-0.3, 0.3);
                T n = math::sqrt(device.math, 1 + climb * climb);
                state.traj_type = 5;
                state.traj[0] = dist * math::cos(device.math, heading) / n;
                state.traj[1] = dist * math::sin(device.math, heading) / n;
                state.traj[2] = dist * climb / n;
                state.traj[3] = U(0.3, 2.0);
            }
            else if(r < m.p_takeoff + m.p_land + m.p_circle + m.p_line + m.p_eight + m.p_fast_circle + m.p_p2p + m.p_gust){
                // hover in a gust: a constant push of 1-4 m/s^2 (mostly horizontal) for
                // 0.2-1.5 s, starting after 0.3-3 s; the policy does not observe it
                near_hover();
                state.task = 8;
                state.floor_z = -U(0.3, 1.0);
                T a = U(1.0, 4.0);
                T heading = U(0, 2 * math::PI<T>), climb = U(-0.3, 0.3);
                T n = math::sqrt(device.math, 1 + climb * climb);
                state.traj_type = 4;
                state.traj[0] = a * math::cos(device.math, heading) / n;
                state.traj[1] = a * math::sin(device.math, heading) / n;
                state.traj[2] = a * climb / n;
                state.traj[3] = U(0.3, 3.0);
                state.traj[4] = U(0.2, 1.5);
            }
            else if(r < m.p_takeoff + m.p_land + m.p_circle + m.p_line + m.p_eight + m.p_fast_circle + m.p_p2p + m.p_gust + m.p_waypoint){
                // waypoints (time-to-target study): hover, then the target jumps
                // waypoint_min..max m (the far end faded in by the curriculum) in a mostly
                // horizontal direction after 0.3-1 s, and 1.5-2.5 s later moves on to
                // s x the jump, s in [-1, 0.5] (back past the start, or part way back)
                near_hover();
                state.task = 9;
                state.floor_z = -U(0.6, 1.0);
                const T kw = m.waypoint_progress >= 0 ? m.waypoint_progress : k;
                T dist = U(m.waypoint_min, m.waypoint_min + kw * (m.waypoint_max - m.waypoint_min));
                T heading = U(0, 2 * math::PI<T>), climb = U(-0.25, 0.25);
                T n = math::sqrt(device.math, 1 + climb * climb);
                state.traj_type = 6;
                state.traj[0] = dist * math::cos(device.math, heading) / n;
                state.traj[1] = dist * math::sin(device.math, heading) / n;
                state.traj[2] = dist * climb / n;
                state.traj[3] = U(0.3, 1.0);
                state.traj[4] = state.traj[3] + U(1.5, 2.5);
                state.traj[5] = U(-1, 0.5);
            }
            // else: the original free-flight start, static target, no floor
#ifdef L2F_TASK_REGIME
            // MOO framing B with hidden-dynamics tasks: the label is the episode's dynamics
            // regime instead of its type, 3 x delay (thirds of 0..delay_max) + mass (thirds of
            // the randomized range); the policy observes neither
            {
                T fd = m.delay_max_steps > 0 ? state.delay_steps / m.delay_max_steps : 0;
                T fm = m.domain_randomization && m.dr_mass > 0 ? (env.parameters.dynamics.mass / env.nominal_dynamics.mass - 1) / m.dr_mass : 0;
                TI bd = fd < (T)1 / 3 ? 0 : (fd < (T)2 / 3 ? 1 : 2);
                TI bm = fm < -(T)1 / 3 ? 0 : (fm < (T)1 / 3 ? 1 : 2);
                state.task = 3 * bd + bm;
            }
#endif
            for(TI k = 0; k < 2; k++){
                for(TI i = 0; i < 3; i++){
                    state.prev[k][0 + i] = state.position[i];
                    state.prev[k][7 + i] = state.linear_velocity[i];
                    state.prev[k][10 + i] = state.angular_velocity[i];
                }
                for(TI i = 0; i < 4; i++){
                    state.prev[k][3 + i] = state.orientation[i];
                }
            }
#ifdef L2F_SENSOR_SIM
            sensor_begin(device, env, state, rng);
#endif
        }
        template<typename T, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool on_floor(const StateBase<T, TI>& state){
            return state.position[2] <= state.floor_z + (T)0.005;
        }
        // Height above the floor as the drone's estimator would report it: the true
        // height plus the current position glitch and this episode's position noise.
        // The observation delay is left out (at most 12 ms of descent, a few mm).
        template<typename DEVICE, typename SPEC, typename T, typename TI, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT T estimated_height(DEVICE& device, const Multirotor<SPEC>& env, const StateBase<T, TI>& state, RNG& rng){
#ifdef L2F_SENSOR_SIM
            return state.est_position[2] - state.floor_z;  // the estimator's height, as rl_trigger.py reads it
#endif
            T sigma = state.pos_noise_sigma[2] > 0 ? state.pos_noise_sigma[2] : (T)env.parameters.mdp.observation_noise.position;
            T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, sigma, rng);
            return state.position[2] + state.glitch_offset[2] + noise - state.floor_z;
        }
        // Landing like the flight script (Mission::landing_cut), landing episodes only:
        // cut the motors when the target is within cut_window of the floor and the
        // estimated height is below the episode's cut height, or floor_timeout s after
        // the target reached its lowest point. Sets state.landed and the touchdown
        // speed of the free fall that follows, v = sqrt(v_down^2 + 2 g h).
        template<typename DEVICE, typename SPEC, typename T, typename TI, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void landing_cut(DEVICE& device, const Multirotor<SPEC>& env, StateBase<T, TI>& state, RNG& rng){
            const auto& m = env.mission;
            if(!m.landing_cut || state.task != 2 || state.landed != 0){
                return;
            }
            T target_h = -state.floor_z;                  // target height above the floor
            T t_low = state.traj[4] + state.traj[3];      // the target reaches its lowest point
            bool cut = target_h < m.cut_window && estimated_height(device, env, state, rng) < state.cut_height;
            bool timeout = state.traj_t >= t_low + m.floor_timeout;
            if(cut || timeout){
                T h = state.position[2] - state.floor_z;
                h = h > 0 ? h : (T)0;
                T v_down = state.linear_velocity[2] < 0 ? -state.linear_velocity[2] : (T)0;
                state.touchdown_speed = math::sqrt(device.math, v_down * v_down + 2 * (T)9.81 * h);
                state.landed = (cut || h < m.cut_max_height) ? 1 : 2;
            }
        }
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC>
    static void initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBase<T, TI>& state){
//        T J_factor = random::uniform_real_distribution(random_dev, (T)0.5, (T)2, rng);
//        env.current_dynamics.J[0][0] *= J_factor;
//        env.current_dynamics.J[1][1] *= J_factor;
//        env.current_dynamics.J[2][2] *= J_factor;
//        env.current_dynamics.J_inv[0][0] /= J_factor;
//        env.current_dynamics.J_inv[1][1] /= J_factor;
//        env.current_dynamics.J_inv[2][2] /= J_factor;
//        T mass_factor = random::uniform_real_distribution(random_dev, (T)0.5, (T)1.5, rng);
//        env.current_dynamics.mass *= mass_factor;
//        printf("initial state: %f %f %f %f %f %f %f %f %f %f %f %f %f\n", state.state[0], state.state[1], state.state[2], state.state[3], state.state[4], state.state[5], state.state[6], state.state[7], state.state[8], state.state[9], state.state[10], state.state[11], state.state[12]);
        env.current_dynamics = env.parameters.dynamics;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBase<T, TI>& state){
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            state.position[i] = 0;
        }
        state.orientation[0] = 1;
        for(typename DEVICE::index_t i = 1; i < 4; i++){
            state.orientation[i] = 0;
        }
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            state.linear_velocity[i] = 0;
        }
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            state.angular_velocity[i] = 0;
        }
        rl::environments::multirotor::mission::reset_fields(state);
        rl::environments::multirotor::mission::reset_motors(state);
        initial_parameters(device, env, state);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename NEXT_COMPONENT>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateRandomForce<T, TI, NEXT_COMPONENT>& state){
        initial_state(device, env, static_cast<NEXT_COMPONENT&>(state));
        state.force[0] = 0;
        state.force[1] = 0;
        state.force[2] = 0;
        state.torque[0] = 0;
        state.torque[1] = 0;
        state.torque[2] = 0;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename NEXT_COMPONENT>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& state){
        initial_state(device, env, static_cast<NEXT_COMPONENT&>(state));
        for(typename DEVICE::index_t i = 0; i < 4; i++){
            state.rpm[i] = (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) / 2 + env.parameters.dynamics.action_limit.min;
        }
        rl::environments::multirotor::mission::sync_rpm_meas(env, state);
    }
    template<typename DEVICE, typename T, typename TI_H, TI_H HISTORY_LENGTH, typename SPEC, typename NEXT_COMPONENT>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateRotorsHistory<T, TI_H, HISTORY_LENGTH, NEXT_COMPONENT>& state){
        using TI = typename DEVICE::index_t;
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        initial_state(device, env, static_cast<rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>&>(state));
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                state.action_history[step_i][action_i] = (state.rpm[action_i] - env.parameters.dynamics.action_limit.min) / (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) * 2 - 1;
            }
        }
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename RNG, bool INHERIT_GUIDANCE = false>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBase<T, TI>& state, RNG& rng, bool inherited_guidance = false){
        typename DEVICE::SPEC::MATH math_dev;
        typename DEVICE::SPEC::RANDOM random_dev;
        using STATE = typename rl::environments::multirotor::StateBase<T, TI>;
        bool guidance;
        guidance = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng) < env.parameters.mdp.init.guidance;
        if(!guidance){
            for(TI i = 0; i < 3; i++){
                state.position[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_position, env.parameters.mdp.init.max_position, rng);
            }
            // https://web.archive.org/web/20181126051029/http://planning.cs.uiuc.edu/node198.html
        }
        else{
            for(TI i = 0; i < 3; i++){
                state.position[i] = 0;
            }
        }
        if(env.parameters.mdp.init.max_angle > 0 && !guidance){
            do{
                T u[3];
                for(TI i = 0; i < 3; i++){
                    u[i] = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng);
                }
                state.orientation[0] = math::sqrt(math_dev, 1-u[0]) * math::sin(math_dev, 2*math::PI<T>*u[1]);
                state.orientation[1] = math::sqrt(math_dev, 1-u[0]) * math::cos(math_dev, 2*math::PI<T>*u[1]);
                state.orientation[2] = math::sqrt(math_dev,   u[0]) * math::sin(math_dev, 2*math::PI<T>*u[2]);
                state.orientation[3] = math::sqrt(math_dev,   u[0]) * math::cos(math_dev, 2*math::PI<T>*u[2]);
            } while(math::abs(math_dev, 2*math::acos(math_dev, state.orientation[0])) > env.parameters.mdp.init.max_angle);
        }
        else{
            state.orientation[0] = 1;
            state.orientation[1] = 0;
            state.orientation[2] = 0;
            state.orientation[3] = 0;
        }
        for(TI i = 0; i < 3; i++){
            state.linear_velocity[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_linear_velocity, env.parameters.mdp.init.max_linear_velocity, rng);
        }
        for(TI i = 0; i < 3; i++){
            state.angular_velocity[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_angular_velocity, env.parameters.mdp.init.max_angular_velocity, rng);
        }
        rl::environments::multirotor::mission::reset_fields(state);
        rl::environments::multirotor::mission::reset_motors(state);
        initial_parameters(device, env, state);
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateRandomForce<T_S, TI_S, NEXT_COMPONENT>& state, RNG& rng){
        typename DEVICE::SPEC::RANDOM random_dev;
        using T = typename SPEC::T;
//        bool guidance = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng) < env.parameters.mdp.init.guidance;
        sample_initial_state(device, env, static_cast<NEXT_COMPONENT&>(state), rng);
//        if(!guidance){
        {
            auto distribution = env.parameters.disturbances.random_force;
            state.force[0] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.force[1] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.force[2] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            // L2F_FORCE_SCALE (default 1: unchanged, same draws) scales the force's deviation
            for(int i = 0; i < 3; i++){
                state.force[i] = (T)distribution.mean + env.mission.force_scale * (state.force[i] - (T)distribution.mean);
            }
        }
        {
            auto distribution = env.parameters.disturbances.random_torque;
            state.torque[0] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.torque[1] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
            state.torque[2] = random::normal_distribution::sample(random_dev, (T)distribution.mean, (T)distribution.std, rng);
        }
//        }
//        else{
//            state.force[0] = 0;
//            state.force[1] = 0;
//            state.force[2] = 0;
//            state.torque[0] = 0;
//            state.torque[1] = 0;
//            state.torque[2] = 0;
//        }

    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& state, RNG& rng){
        sample_initial_state(device, env, static_cast<NEXT_COMPONENT&>(state), rng);
        T min_rpm, max_rpm;
        if(env.parameters.mdp.init.relative_rpm){
            min_rpm = (env.parameters.mdp.init.min_rpm + 1)/2 * (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) + env.parameters.dynamics.action_limit.min;
            max_rpm = (env.parameters.mdp.init.max_rpm + 1)/2 * (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) + env.parameters.dynamics.action_limit.min;
        }
        else{
            min_rpm = env.parameters.mdp.init.min_rpm < 0 ? env.parameters.dynamics.action_limit.min : env.parameters.mdp.init.min_rpm;
            max_rpm = env.parameters.mdp.init.max_rpm < 0 ? env.parameters.dynamics.action_limit.max : env.parameters.mdp.init.max_rpm;
            if(max_rpm > env.parameters.dynamics.action_limit.max){
                max_rpm = env.parameters.dynamics.action_limit.max;
            }
            if(min_rpm > max_rpm){
                min_rpm = max_rpm;
            }
        }
        for(TI i = 0; i < 4; i++){
            state.rpm[i] = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), min_rpm, max_rpm, rng);
        }
        rl::environments::multirotor::mission::sync_rpm_meas(env, state);
    }
    template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename NEXT_COMPONENT, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, NEXT_COMPONENT>& state, RNG& rng){
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        sample_initial_state(device, env, static_cast<typename rl::environments::multirotor::StateRotors<T_S, TI_S, NEXT_COMPONENT>&>(state), rng);
        if(env.mission.enabled){
            rl::environments::multirotor::mission::sample(device, env, state, state.rpm, rng);
            rl::environments::multirotor::mission::sync_rpm_meas(env, state);
        }
        for(TI i = 0; i < 4; i++){
            state.action_pending[i] = state.rpm[i];  // actuator dead time: the first step keeps the current speeds
        }
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                state.action_history[step_i][action_i] = (state.rpm[action_i] - env.parameters.dynamics.action_limit.min) / (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) * 2 - 1;
            }
        }
    }
    namespace rl::environments::multirotor{
        template<typename DEVICE, typename SPEC, typename STATE, typename OBSERVATION_TI, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, rl::environments::multirotor::observation::LastComponent<OBSERVATION_TI>, Matrix<OBS_SPEC>& observation, RNG& rng){
            static_assert(OBS_SPEC::COLS == 0);
            static_assert(OBS_SPEC::ROWS == 1);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::Position<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using OBSERVATION = rl::environments::multirotor::observation::Position<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;

            for(TI i = 0; i < 3; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.position[i]);
                }
                else{
                    // this episode's per-axis std if set (Mission::pos_noise_min/max), else the fixed one
                    T sigma = state.pos_noise_sigma[i] > 0 ? state.pos_noise_sigma[i] : (T)env.parameters.mdp.observation_noise.position;
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, sigma, rng);
                    const T lim = env.mission.obs_pos_clip;
                    if(lim > 0){
                        T p = state.position[i] + noise;
                        set(observation, 0, i, p > lim ? lim : (p < -lim ? -lim : p));
                    }
                    else{
                        set(observation, 0, i, state.position[i] + noise);
                    }
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::OrientationQuaternion<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::OrientationQuaternion<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.orientation[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, env.parameters.mdp.observation_noise.orientation, rng);
                    set(observation, 0, i, state.orientation[i] + noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::OrientationRotationMatrix<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::OrientationRotationMatrix<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            const typename SPEC::T* q = state.orientation;
            set(observation, 0, 0, (1 - 2*q[2]*q[2] - 2*q[3]*q[3]));
            set(observation, 0, 1, (    2*q[1]*q[2] - 2*q[0]*q[3]));
            set(observation, 0, 2, (    2*q[1]*q[3] + 2*q[0]*q[2]));
            set(observation, 0, 3, (    2*q[1]*q[2] + 2*q[0]*q[3]));
            set(observation, 0, 4, (1 - 2*q[1]*q[1] - 2*q[3]*q[3]));
            set(observation, 0, 5, (    2*q[2]*q[3] - 2*q[0]*q[1]));
            set(observation, 0, 6, (    2*q[1]*q[3] - 2*q[0]*q[2]));
            set(observation, 0, 7, (    2*q[2]*q[3] + 2*q[0]*q[1]));
            set(observation, 0, 8, (1 - 2*q[1]*q[1] - 2*q[2]*q[2]));
            if constexpr(!OBSERVATION_SPEC::PRIVILEGED || SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                    T noise;
                    noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng);
                    increment(observation, 0, i, noise);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::LinearVelocity<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::LinearVelocity<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                // velocity relative to the (possibly moving) target, as on the drone
                T v = state.linear_velocity[i] - state.target_velocity[i];
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, v);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, env.parameters.mdp.observation_noise.linear_velocity, rng);
                    const T lim = env.mission.obs_vel_clip;
                    if(lim > 0){
                        T u = v + noise;
                        set(observation, 0, i, u > lim ? lim : (u < -lim ? -lim : u));
                    }
                    else{
                        set(observation, 0, i, v + noise);
                    }
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::AngularVelocity<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::AngularVelocity<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                if constexpr(OBSERVATION_SPEC::PRIVILEGED && !SPEC::STATIC_PARAMETERS::PRIVILEGED_OBSERVATION_NOISE){
                    set(observation, 0, i, state.angular_velocity[i]);
                }
                else{
                    T noise = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM{}, (T)0, env.parameters.mdp.observation_noise.angular_velocity, rng);
                    T w = state.angular_velocity[i] + noise;
                    // the gyro's range: readings beyond it stay at the limit (Mission::gyro_clip)
                    const T lim = env.mission.gyro_clip;
                    if(lim > 0){
                        w = w > lim ? lim : (w < -lim ? -lim : w);
                    }
                    set(observation, 0, i, w);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::RotorSpeeds<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::RotorSpeeds<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI action_i = 0; action_i < OBSERVATION::CURRENT_DIM; action_i++){
                T action_value = (state.rpm[action_i] - env.parameters.dynamics.action_limit.min)/(env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) * 2 - 1;
                set(observation, 0, action_i, action_value);
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::RotorSpeedsMeasured<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::RotorSpeedsMeasured<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < OBSERVATION::CURRENT_DIM; i++){
                set(observation, 0, i, state.rpm_meas[i]);
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::PositionIntegral<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::PositionIntegral<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < 3; i++){
                set(observation, 0, i, state.pos_integral[i]);
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::ActionHistory<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::ActionHistory<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            static_assert(rl::environments::Multirotor<SPEC>::State::HISTORY_LENGTH == OBSERVATION::HISTORY_LENGTH);
            static_assert(rl::environments::Multirotor<SPEC>::State::ACTION_DIM == OBSERVATION::ACTION_DIM);
            static_assert(rl::environments::Multirotor<SPEC>::ACTION_DIM == OBSERVATION::ACTION_DIM);
            for(TI step_i = 0; step_i < OBSERVATION::HISTORY_LENGTH; step_i++){
                for(TI action_i = 0; action_i < OBSERVATION::ACTION_DIM; action_i++){
                    set(observation, 0, step_i*OBSERVATION::ACTION_DIM + action_i, state.action_history[step_i][action_i]);
                }
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::multirotor::observation::RandomForce<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::multirotor::observation::RandomForce<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            for(TI i = 0; i < 3; i++){
                set(observation, 0, i, state.force[i]);
                set(observation, 0, 3 + i, state.torque[i]);
            }
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
    }
    template<typename DEVICE, typename SPEC, typename STATE, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        static_assert(OBS_SPEC::COLS == ENVIRONMENT::OBSERVATION_DIM);
        static_assert(OBS_SPEC::ROWS == 1);
#ifdef L2F_SENSOR_SIM
        if(env.mission.enabled){
            // Sensor path: position, orientation and velocity from the estimator, angular
            // velocity from the gyro sample, delay_steps old (interpolated, as below); no
            // glitch (the estimator makes its own errors). Noise is in the sensors.
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            STATE est = state;
            T cur[13], mixed[13];
            rl::environments::multirotor::mission::estimate_world(state, cur);
            T d = state.delay_steps > 2 ? (T)2 : (state.delay_steps < 0 ? (T)0 : state.delay_steps);
            TI k = (TI)d;
            T f = d - (T)k;
            const T* a = k == 0 ? cur : state.est_prev[k - 1];
            const T* b = k == 0 ? state.est_prev[0] : state.est_prev[k < 2 ? k : 1];
            T dot = 0;
            for(TI i = 3; i < 7; i++){
                dot += a[i] * b[i];
            }
            T sign = dot < 0 ? -1 : 1;
            for(TI i = 0; i < 13; i++){
                T bi = (i >= 3 && i < 7) ? sign * b[i] : b[i];
                mixed[i] = (1 - f) * a[i] + f * bi;
            }
            T qn = 0;
            for(TI i = 3; i < 7; i++){
                qn += mixed[i] * mixed[i];
            }
            qn = math::sqrt(device.math, qn);
            for(TI i = 0; i < 3; i++){
                est.position[i] = mixed[0 + i] - state.world_offset[i];
                est.linear_velocity[i] = mixed[7 + i];
                est.angular_velocity[i] = mixed[10 + i];
            }
            for(TI i = 0; i < 4; i++){
                est.orientation[i] = mixed[3 + i] / qn;
            }
            rl::environments::multirotor::observe(device, env, est, typename ENVIRONMENT::Observation{}, observation, rng);
            return;
        }
#endif
        if(env.mission.enabled && (state.delay_steps > 0 || state.glitch_steps > 0)){
            // State-estimate delay: the actor sees position, orientation and
            // velocities delay_steps old (interpolated between steps); the action
            // history and the target velocity stay current. A position glitch is
            // added on top.
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            STATE delayed = state;
            if(state.glitch_steps > 0){
                for(TI i = 0; i < 3; i++){
                    delayed.position[i] += state.glitch_offset[i];
                    delayed.linear_velocity[i] += state.glitch_vel[i];
                }
            }
            if(state.delay_steps <= 0){
                rl::environments::multirotor::observe(device, env, delayed, typename ENVIRONMENT::Observation{}, observation, rng);
                return;
            }
            T d = state.delay_steps > 2 ? (T)2 : state.delay_steps;
            TI k = (TI)d;
            T f = d - (T)k;
            T cur[13];
            for(TI i = 0; i < 3; i++){
                cur[0 + i] = state.position[i];
                cur[7 + i] = state.linear_velocity[i];
                cur[10 + i] = state.angular_velocity[i];
            }
            for(TI i = 0; i < 4; i++){
                cur[3 + i] = state.orientation[i];
            }
            const T* a = k == 0 ? cur : state.prev[k - 1];
            const T* b = k == 0 ? state.prev[0] : state.prev[k < 2 ? k : 1];
            T dot = 0;
            for(TI i = 3; i < 7; i++){
                dot += a[i] * b[i];
            }
            T sign = dot < 0 ? -1 : 1;
            T mixed[13];
            for(TI i = 0; i < 13; i++){
                T bi = (i >= 3 && i < 7) ? sign * b[i] : b[i];
                mixed[i] = (1 - f) * a[i] + f * bi;
            }
            T qn = 0;
            for(TI i = 3; i < 7; i++){
                qn += mixed[i] * mixed[i];
            }
            qn = math::sqrt(device.math, qn);
            for(TI i = 0; i < 3; i++){
                delayed.position[i] = mixed[0 + i] + (state.glitch_steps > 0 ? state.glitch_offset[i] : (T)0);
                delayed.linear_velocity[i] = mixed[7 + i] + (state.glitch_steps > 0 ? state.glitch_vel[i] : (T)0);
                delayed.angular_velocity[i] = mixed[10 + i];
            }
            for(TI i = 0; i < 4; i++){
                delayed.orientation[i] = mixed[3 + i] / qn;
            }
            rl::environments::multirotor::observe(device, env, delayed, typename ENVIRONMENT::Observation{}, observation, rng);
            return;
        }
        rl::environments::multirotor::observe(device, env, state, typename ENVIRONMENT::Observation{}, observation, rng);
    }
    template<typename DEVICE, typename SPEC, typename STATE, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        static_assert(OBS_SPEC::COLS == ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED);
        static_assert(OBS_SPEC::ROWS == 1);
        rl::environments::multirotor::observe(device, env, state, typename ENVIRONMENT::ObservationPrivileged{}, observation, rng);
    }
//    template<typename DEVICE, typename T, typename TI, typename SPEC, typename OBS_SPEC, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateLatentEmpty<T, TI>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
//        static_assert(OBS_SPEC::COLS == 0);
//    }
//    template<typename DEVICE, typename T, typename TI, typename SPEC, typename OBS_SPEC, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateLatentRandomForce<T, TI>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
//        static_assert(OBS_SPEC::COLS == 6);
//        for(TI i = 0; i < 3; i++){
//            set(observation, 0, 0 + i, state.force[i]);
//            set(observation, 0, 3 + i, state.torque[i]);
//        }
//    }
//    template<typename DEVICE, typename T, typename TI, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
//        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
//        static_assert(OBS_SPEC::COLS == MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION + MULTIROTOR::OBSERVATION_DIM_PRIVILEGED_LATENT_STATE);
//        auto base_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION>{}, 0, 0);
//        observe(device, env, state, base_observation, rng, true);
//        auto latent_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_PRIVILEGED_LATENT_STATE>{}, 0, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION);
//        observe_privileged(device, env, (const LATENT_STATE&)state, latent_observation, rng);
//    }
//    template<typename DEVICE, typename T_H, typename TI_H, TI_H HISTORY_LENGTH, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBaseRotorsHistory<T_H, TI_H, HISTORY_LENGTH, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
//        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
//        static_assert(OBS_SPEC::ROWS == 1);
//        static_assert(OBS_SPEC::COLS == MULTIROTOR::OBSERVATION_DIM);
//        using TI = typename DEVICE::index_t;
//        auto base_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION>{}, 0, 0);
//        observe(device, env, (rl::environments::multirotor::StateBaseRotors<T_H, TI_H, LATENT_STATE>&)state, base_observation, rng);
//        auto action_history_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_ACTION_HISTORY>{}, 0, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION);
//        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
//            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
//                set(action_history_observation, 0, step_i*MULTIROTOR::ACTION_DIM + action_i, state.action_history[step_i][action_i]);
//            }
//        }
//    }
//    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
//        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
//        using T = typename SPEC::T;
//        using TI = typename DEVICE::index_t;
//        static_assert(OBS_SPEC::ROWS == 1);
//        static_assert(OBS_SPEC::COLS == MULTIROTOR::OBSERVATION_DIM_PRIVILEGED);
//        auto base_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_PRIVILEGED - MULTIROTOR::ACTION_DIM>{}, 0, 0);
//        observe_privileged(device, env, (const rl::environments::multirotor::StateBase<T_S, TI_S, LATENT_STATE>&) state, base_observation, rng);
//        auto rpm_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::ACTION_DIM>{}, 0, MULTIROTOR::OBSERVATION_DIM_PRIVILEGED - MULTIROTOR::ACTION_DIM);
//        for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
//            T action_value = (state.rpm[action_i] - env.parameters.dynamics.action_limit.min)/(env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) * 2 - 1;
//            set(rpm_observation, 0, action_i, action_value);
//        }
//    }
//    template<typename DEVICE, typename SPEC, typename T, typename TI>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, rl::environments::multirotor::StateBase<T, TI>& state) {
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateBase<T_S, TI_S>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateBase<T_S, TI_S>& next_state, RNG& rng) {
        using T = T_S;
        using TI = TI_S;
        T quaternion_norm = 0;
        for(TI state_i = 0; state_i < 4; state_i++){
            quaternion_norm += next_state.orientation[state_i] * next_state.orientation[state_i];
        }
        quaternion_norm = math::sqrt(device.math, quaternion_norm);
        for(TI state_i = 0; state_i < 4; state_i++){
            next_state.orientation[state_i] /= quaternion_norm;
        }
        // Mission extension: carry the non-integrated fields forward, remember the
        // previous true state (for the observation delay), move the target and
        // apply floor contact.
        for(TI i = 0; i < 3; i++){
            next_state.target_velocity[i] = state.target_velocity[i];
        }
        next_state.floor_z = state.floor_z;
        next_state.traj_type = state.traj_type;
        next_state.task = state.task;
        for(TI i = 0; i < 6; i++){
            next_state.traj[i] = state.traj[i];
        }
        next_state.delay_steps = state.delay_steps;
        next_state.ground_effect = state.ground_effect;
        for(TI i = 0; i < 3; i++){
            next_state.pos_noise_sigma[i] = state.pos_noise_sigma[i];
        }
        next_state.cut_height = state.cut_height;
        next_state.landed = state.landed;
        next_state.touchdown_speed = state.touchdown_speed;
        for(TI i = 0; i < 4; i++){
            next_state.action_pending[i] = state.action_pending[i];  // step() sets the new one
        }
        next_state.act_delay = state.act_delay;
        next_state.gain_walk = state.gain_walk;
        if(env.mission.enabled && env.mission.gain_walk_sigma > 0){
            // AR(1): g <- a g + sigma sqrt(1 - a^2) N(0, 1), a = exp(-dt / tau)
            const T a = math::exp(device.math, -env.parameters.integration.dt / env.mission.gain_walk_tau);
            const T s = env.mission.progress * env.mission.gain_walk_sigma * math::sqrt(device.math, 1 - a * a);
            next_state.gain_walk = a * state.gain_walk + random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, s, rng);
        }
        for(TI i = 0; i < 4; i++){
            next_state.motor_gain[i] = state.motor_gain[i];
            next_state.motor_offset[i] = state.motor_offset[i];
            next_state.rpm_meas[i] = state.rpm_meas[i];
        }
        // position-estimate glitches: count down the current one, maybe start a new one
        next_state.glitch_steps = state.glitch_steps > 0 ? state.glitch_steps - 1 : 0;
        for(TI i = 0; i < 3; i++){
            next_state.glitch_offset[i] = next_state.glitch_steps > 0 ? state.glitch_offset[i] : (T)0;
            next_state.glitch_vel[i] = next_state.glitch_steps > 0 ? state.glitch_vel[i] * (T)0.8 : (T)0;
        }
        // leaky position-error integral (observation), from the true relative position
        {
            const T dt = env.parameters.integration.dt;
            const T leak = 1 - dt / (T)L2F_POS_INTEGRAL_TAU;
            for(TI i = 0; i < 3; i++){
#ifdef L2F_SENSOR_SIM
                T v = leak * state.pos_integral[i] + state.est_position[i] * dt;  // the drone integrates its estimate
#else
                const T lim = env.mission.obs_pos_clip;
                T v = lim > 0 ? leak * state.pos_integral[i] + (state.position[i] > lim ? lim : (state.position[i] < -lim ? -lim : state.position[i])) * dt
                              : leak * state.pos_integral[i] + state.position[i] * dt;
#endif
                next_state.pos_integral[i] = v > (T)L2F_POS_INTEGRAL_MAX ? (T)L2F_POS_INTEGRAL_MAX : (v < -(T)L2F_POS_INTEGRAL_MAX ? -(T)L2F_POS_INTEGRAL_MAX : v);
            }
        }
        if(env.mission.enabled && env.mission.glitch_p > 0 && next_state.glitch_steps == 0){
            typename DEVICE::SPEC::RANDOM rd;
            if(random::uniform_real_distribution(rd, (T)0, (T)1, rng) < env.mission.glitch_p * env.mission.progress){
                T dir[3], n = 0;
                for(TI i = 0; i < 3; i++){
                    dir[i] = random::normal_distribution::sample(rd, (T)0, (T)1, rng);
                    n += dir[i] * dir[i];
                }
                n = math::sqrt(device.math, n) + (T)1e-6;
                T size = random::uniform_real_distribution(rd, env.mission.glitch_min, env.mission.glitch_max, rng);
                T vel = random::uniform_real_distribution(rd, (T)0, env.mission.glitch_vel_max, rng) * size;
                for(TI i = 0; i < 3; i++){
                    next_state.glitch_offset[i] = size * dir[i] / n;
                    next_state.glitch_vel[i] = vel * dir[i] / n;  // the estimate "moves" toward the jump
                }
                T duration = random::uniform_real_distribution(rd, (T)0.05, env.mission.glitch_t_max, rng);
                next_state.glitch_steps = (TI)(duration / env.parameters.integration.dt) + 1;
            }
        }
        for(TI i = 0; i < 13; i++){
            next_state.prev[1][i] = state.prev[0][i];
        }
        for(TI i = 0; i < 3; i++){
            next_state.prev[0][0 + i] = state.position[i];
            next_state.prev[0][7 + i] = state.linear_velocity[i];
            next_state.prev[0][10 + i] = state.angular_velocity[i];
        }
        for(TI i = 0; i < 4; i++){
            next_state.prev[0][3 + i] = state.orientation[i];
        }
        T t0 = state.traj_t, t1 = state.traj_t + env.parameters.integration.dt;
        next_state.traj_t = t1;
        if(state.traj_type == 4 && t0 >= state.traj[3] && t0 < state.traj[3] + state.traj[4]){
            // gust: external acceleration traj[0..2] (applied over the step, after RK4)
            for(TI i = 0; i < 3; i++){
                next_state.linear_velocity[i] += state.traj[i] * env.parameters.integration.dt;
            }
        }
#ifdef L2F_SENSOR_SIM
        for(TI i = 0; i < 3; i++){
            next_state.world_offset[i] = state.world_offset[i];
        }
#endif
        if(state.traj_type != 0){
            T p0[3], v0[3], p1[3], v1[3];
            rl::environments::multirotor::mission::target_at(device, state, t0, p0, v0);
            rl::environments::multirotor::mission::target_at(device, state, t1, p1, v1);
            for(TI i = 0; i < 3; i++){
                T shift = p1[i] - p0[i];
#ifdef L2F_SENSOR_SIM
                next_state.world_offset[i] += shift;  // world = position + offset is unchanged
#endif
                next_state.position[i] -= shift;
                next_state.prev[0][i] -= shift;
                next_state.prev[1][i] -= shift;
                next_state.target_velocity[i] = v1[i];
            }
            next_state.floor_z -= p1[2] - p0[2];
        }
        if(next_state.position[2] < next_state.floor_z){
            // Floor contact: the landing feet stop downward motion, friction and
            // the feet damp sliding and rotation.
            next_state.position[2] = next_state.floor_z;
            if(next_state.linear_velocity[2] < 0){
                next_state.linear_velocity[2] = 0;
            }
            for(TI i = 0; i < 2; i++){
                next_state.linear_velocity[i] *= (T)0.5;
            }
            for(TI i = 0; i < 3; i++){
                next_state.angular_velocity[i] *= (T)0.5;
            }
        }
#ifdef L2F_SENSOR_SIM
        if(env.mission.enabled){
            rl::environments::multirotor::mission::sensor_step(device, env, state, action, next_state, rng);
        }
#endif
        // landing like the flight script: maybe cut the motors (ends the episode)
        rl::environments::multirotor::mission::landing_cut(device, env, next_state, rng);
    }
//    template<typename DEVICE, typename SPEC, typename T, typename TI, typename NEXT_COMPONENT>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, rl::environments::multirotor::StateRotors<T, TI, NEXT_COMPONENT>& state) {
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateRotors<T_S, TI_S, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateRotors<T_S, TI_S, NEXT_COMPONENT>& next_state, RNG& rng) {
        post_integration(device, env, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(next_state), rng);
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        for(typename DEVICE::index_t rpm_i = 0; rpm_i < MULTIROTOR::ACTION_DIM; rpm_i++){
            next_state.rpm[rpm_i] = math::clamp(typename DEVICE::SPEC::MATH{}, next_state.rpm[rpm_i], env.parameters.dynamics.action_limit.min, env.parameters.dynamics.action_limit.max);
        }
        // Measured rotor speeds: like the ESC telemetry, sometimes the previous value
        // repeats, otherwise the true speed plus noise. Without L2F_RPM_OBS no random
        // numbers are drawn, so older training recipes reproduce exactly.
        const auto& lim = env.parameters.dynamics.action_limit;
        for(typename DEVICE::index_t i = 0; i < MULTIROTOR::ACTION_DIM; i++){
            T_S v = (next_state.rpm[i] - lim.min) / (lim.max - lim.min) * 2 - 1;
#ifdef L2F_RPM_OBS
            typename DEVICE::SPEC::RANDOM rd;
            const auto& m = env.mission;
            if(m.rpm_hold_p > 0 && random::uniform_real_distribution(rd, (T_S)0, (T_S)1, rng) < m.rpm_hold_p){
                v = state.rpm_meas[i];
            }
            else if(m.rpm_noise > 0){
                v += random::normal_distribution::sample(rd, (T_S)0, m.rpm_noise, rng);
            }
#endif
            next_state.rpm_meas[i] = v;
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename NEXT_COMPONENT, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateRandomForce<T_S, TI_S, NEXT_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateRandomForce<T_S, TI_S, NEXT_COMPONENT>& next_state, RNG& rng) {
        post_integration(device, env, static_cast<const NEXT_COMPONENT&>(state), action, static_cast<NEXT_COMPONENT&>(next_state), rng);
        next_state.force[0] = state.force[0];
        next_state.force[1] = state.force[1];
        next_state.force[2] = state.force[2];
        next_state.torque[0] = state.torque[0];
        next_state.torque[1] = state.torque[1];
        next_state.torque[2] = state.torque[2];
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename NEXT_STATE_COMPONENT, TI_S HISTORY_LENGTH, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, NEXT_STATE_COMPONENT>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateRotorsHistory<T_S, TI_S, HISTORY_LENGTH, NEXT_STATE_COMPONENT>& next_state, RNG& rng) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        post_integration(device, env, static_cast<const rl::environments::multirotor::StateRotors<T_S, TI_S, NEXT_STATE_COMPONENT>&>(state), action, static_cast<rl::environments::multirotor::StateRotors<T_S, TI_S, NEXT_STATE_COMPONENT>&>(next_state), rng);
        if constexpr(HISTORY_LENGTH > 0){
            for(TI step_i = 0; step_i < HISTORY_LENGTH-1; step_i++){
                for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                    next_state.action_history[step_i][action_i] = state.action_history[step_i+1][action_i];
                }
            }
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                next_state.action_history[HISTORY_LENGTH-1][action_i] = get(action, 0, action_i);
            }
        }
    }
//    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename STATE, typename RNG>
//    RL_TOOLS_FUNCTION_PLACEMENT void post_integration(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const STATE& state, const Matrix<ACTION_SPEC>& action, STATE& next_state, RNG& rng) {
//        static_assert(!STATE::REQUIRES_INTEGRATION);
//        post_integration(device, env, static_cast<typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(next_state), rng);
//    }
    // todo: make state const again
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr auto STATE_DIM = STATE::DIM;
        constexpr auto ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);
        T action_scaled[ACTION_DIM];

        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T half_range = (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) / 2;
            T action_noisy = get(action, 0, action_i);
            action_noisy += random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.action_noise.normalized_rpm, rng);
            action_noisy = math::clamp(device.math, action_noisy, -(T)1, (T)1);
            action_scaled[action_i] = action_noisy * half_range + env.parameters.dynamics.action_limit.min + half_range;
//            state.rpm[action_i] = action_scaled[action_i];
        }
        // Per-motor command -> speed mismatch (StateBase::motor_gain / motor_offset).
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T v = action_scaled[action_i] * state.motor_gain[action_i] + state.motor_offset[action_i];
            if(state.gain_walk != 0){  // common gain wander (Mission::gain_walk_sigma); same arithmetic as before when off
                v = action_scaled[action_i] * state.motor_gain[action_i] * (1 + state.gain_walk) + state.motor_offset[action_i];
            }
            action_scaled[action_i] = math::clamp(device.math, v, env.parameters.dynamics.action_limit.min, env.parameters.dynamics.action_limit.max);
        }
        if(env.mission.motor_idle_rpm > 0){  // ESC idle (Mission::motor_idle_rpm)
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                action_scaled[action_i] = math::max(device.math, action_scaled[action_i], env.mission.motor_idle_rpm);
            }
        }
        // Actuator dead time (StateBase::act_delay): the motors get the previous step's
        // command; this step's waits in action_pending (set after post_integration).
        T action_now[ACTION_DIM];
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            action_now[action_i] = action_scaled[action_i];
            if(state.act_delay){
                action_scaled[action_i] = state.action_pending[action_i];
            }
        }
        if(env.mission.enabled && state.ground_effect > 0 && state.floor_z > -50){
            // Ground effect: more thrust close to the floor (constant over the step).
            const auto& m = env.mission;
            T h = state.position[2] - state.floor_z + m.ge_h0;
            h = h < m.ge_rotor_radius / 2 ? m.ge_rotor_radius / 2 : h;
            T q = m.ge_rotor_radius / (4 * h);
            T denom = 1 - state.ground_effect * q * q;
            T ratio = denom > 1 / m.ge_max_ratio ? 1 / denom : m.ge_max_ratio;
            auto params = env.parameters;
            for(TI i = 0; i < 3; i++){
                params.dynamics.thrust_constants[i] *= ratio;
            }
            utils::integrators::rk4  <DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, params, state, action_scaled, env.parameters.integration.dt, next_state);
        }
        else{
            utils::integrators::rk4  <DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, env.parameters, state, action_scaled, env.parameters.integration.dt, next_state);
        }
//        utils::integrators::euler<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, env.parameters, state, action_scaled, env.parameters.integration.dt, next_state);

        post_integration(device, env, state, action, next_state, rng);
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            next_state.action_pending[action_i] = action_now[action_i];
        }

        return env.parameters.integration.dt;
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        // landing like the flight script: the motors were cut (a landing, or a failed
        // landing; the reward tells them apart)
        if(env.mission.enabled && state.landed != 0){
            return true;
        }
#ifdef L2F_SENSOR_SIM
        // the estimator failed (a firmware assert would halt the drone): a crash
        if(env.mission.enabled && state.est_failed){
            return true;
        }
#endif
        if(env.parameters.mdp.termination.enabled){
            // rotation crash (Mission::omega_crash), else the preset's limit
            const T w_lim = env.mission.omega_crash > 0 ? env.mission.omega_crash : (T)env.parameters.mdp.termination.angular_velocity_threshold;
            if(env.mission.enabled && env.mission.speed_crash > 0){
                T v2 = 0;
                for(TI i = 0; i < 3; i++){
                    v2 += state.linear_velocity[i] * state.linear_velocity[i];
                }
                if(v2 > env.mission.speed_crash * env.mission.speed_crash){
                    return true;
                }
            }
            for(TI i = 0; i < 3; i++){
                // waypoint episodes: the target is up to 2 jumps away (traj_type 6)
                const T p_lim = env.parameters.mdp.termination.position_threshold + (state.traj_type == 6 ? 2 * math::abs(device.math, state.traj[i]) : (T)0);
                if(
                    math::abs(device.math, state.position[i]) > p_lim ||
                    math::abs(device.math, state.linear_velocity[i]) > env.parameters.mdp.termination.linear_velocity_threshold ||
                    math::abs(device.math, state.angular_velocity[i]) > w_lim
                ){
                    return true;
                }
            }
            // Mission: hitting the floor tilted more than 60 deg is a crash.
            if(env.mission.enabled && rl::environments::multirotor::mission::on_floor(state)){
                T up = 1 - 2 * (state.orientation[1] * state.orientation[1] + state.orientation[2] * state.orientation[2]);
                if(up < (T)0.5){
                    return true;
                }
            }
        }
        return false;
    }
}
#include "parameters/reward_functions/reward_functions.h"
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        return rl::environments::multirotor::parameters::reward_functions::reward(device, env, env.parameters.mdp.reward, state, action, next_state, rng);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        rl::environments::multirotor::parameters::reward_functions::log_reward(device, env, env.parameters.mdp.reward, state, action, next_state, rng);
    }
}

//template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
//static void deserialize(DEVICE& device, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
//    using STATE = typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>;
//    static_assert(SPEC::ROWS == 1);
//    static_assert(SPEC::COLS == STATE::DIM);
//    state.position[0] = get(flat_state, 0, 0);
//    state.position[1] = get(flat_state, 0, 1);
//    state.position[2] = get(flat_state, 0, 2);
//    state.orientation[0] = get(flat_state, 0, 3);
//    state.orientation[1] = get(flat_state, 0, 4);
//    state.orientation[2] = get(flat_state, 0, 5);
//    state.orientation[3] = get(flat_state, 0, 6);
//    state.linear_velocity[0] = get(flat_state, 0, 7);
//    state.linear_velocity[1] = get(flat_state, 0, 8);
//    state.linear_velocity[2] = get(flat_state, 0, 9);
//    state.angular_velocity[0] = get(flat_state, 0, 10);
//    state.angular_velocity[1] = get(flat_state, 0, 11);
//    state.angular_velocity[2] = get(flat_state, 0, 12);
//}
//template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
//static void serialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
//    using STATE = typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>;
//    static_assert(SPEC::ROWS == 1);
//    static_assert(SPEC::COLS == STATE::DIM);
//    auto state_base_flat = view(device, flat_state, matrix::ViewSpec<1, 13>{}, 0, 0);
//    serialize(device, (rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state, state_base_flat);
//    constexpr TI offset = rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>::DIM;
//    set(flat_state, 0, offset + 0, state.rpm[0]);
//    set(flat_state, 0, offset + 1, state.rpm[1]);
//    set(flat_state, 0, offset + 2, state.rpm[2]);
//    set(flat_state, 0, offset + 3, state.rpm[3]);
//}
//template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
//static void deserialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
//    using STATE = typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>;
//    static_assert(SPEC::ROWS == 1);
//    static_assert(SPEC::COLS == STATE::DIM);
//    constexpr TI OFFSET = rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>::DIM;
//    auto state_base_rotors_flat = view(device, flat_state, matrix::ViewSpec<1, OFFSET>{}, 0, 0);
//    deserialize(device, (rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state, state_base_rotors_flat);
//    state.rpm[0] = get(flat_state, 0, OFFSET + 0);
//    state.rpm[1] = get(flat_state, 0, OFFSET + 1);
//    state.rpm[2] = get(flat_state, 0, OFFSET + 2);
//    state.rpm[3] = get(flat_state, 0, OFFSET + 3);
//}
//template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename SPEC, typename LATENT_STATE>
//static void serialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
//    using STATE = typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>;
//    using TI = typename DEVICE::index_t;
//    static_assert(SPEC::ROWS == 1);
//    static_assert(SPEC::COLS == STATE::DIM);
//    constexpr TI OFFSET = rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>::DIM;
//    auto state_base_flat = view(device, flat_state, matrix::ViewSpec<1, OFFSET>{}, 0, 0);
//    serialize(device, (rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>&)state, state_base_flat);
//    for(TI step_i = 0; step_i < HISTORY_LENGTH; ++step_i){
//        set(flat_state, 0, OFFSET + step_i * 4 + 0, state.action_history[step_i][0]);
//        set(flat_state, 0, OFFSET + step_i * 4 + 1, state.action_history[step_i][1]);
//        set(flat_state, 0, OFFSET + step_i * 4 + 2, state.action_history[step_i][2]);
//        set(flat_state, 0, OFFSET + step_i * 4 + 3, state.action_history[step_i][3]);
//    }
//}
//template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename SPEC, typename LATENT_STATE>
//static void deserialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
//    using STATE = typename rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>;
//    using TI = typename DEVICE::index_t;
//    static_assert(SPEC::ROWS == 1);
//    static_assert(SPEC::COLS == STATE::DIM);
//    auto state_base_flat = view(device, flat_state, matrix::ViewSpec<1, 13>{}, 0, 0);
//    deserialize(device, (rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>&)state, state_base_flat);
//    constexpr TI offset = rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>::DIM;
//    for(TI step_i = 0; step_i < HISTORY_LENGTH; ++step_i){
//        state.rpm_history[step_i][0] = get(flat_state, 0, offset + step_i * 4 + 0);
//        state.rpm_history[step_i][1] = get(flat_state, 0, offset + step_i * 4 + 1);
//        state.rpm_history[step_i][2] = get(flat_state, 0, offset + step_i * 4 + 2);
//        state.rpm_history[step_i][3] = get(flat_state, 0, offset + step_i * 4 + 3);
//    }
//}

#endif