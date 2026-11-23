// Headless robustness check of a trained actor (.h5 checkpoint).
//
// Runs many 10 s episodes from the training initial-state distribution (up to
// 90 deg tilt) and from near-hover starts, under the nominal dynamics and with
// the dynamics parameters perturbed (mass, thrust, motor time constant,
// inertia). Reports the fraction of episodes that never terminate and the
// position error over the last 5 s. Observation noise and disturbances stay as
// in training.
//
// Usage: evaluate_robustness <actor.h5> [episodes]
#include "training.h"

#include <highfive/H5File.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using CONFIG = learning_to_fly::config::Config<parameters::DefaultAblationSpec>;
using DEVICE = typename CONFIG::DEVICE;
using T = typename CONFIG::T;
using TI = typename CONFIG::TI;
using ENVIRONMENT = typename CONFIG::ENVIRONMENT;
using PENV = parameters::environment<T, TI, parameters::DefaultAblationSpec>;

constexpr TI EPISODE_STEPS = 1000;        // 10 s at dt = 0.01
constexpr TI SETTLED_FROM_STEP = 500;     // error measured over the last 5 s

struct Variation{
    const char* name;
    T mass, thrust, tau, inertia;
    T disturbance = 1;
    // Handover test: if >= -1, near-hover starts get motors at hover speed and an
    // action history filled with this value (hover = normalized hover action).
    T history = -2;
    T history_rear = -3;  // if >= -1: rotors 1 and 2 (rear) get this instead
};

struct Result{
    TI ok = 0, total = 0;
    T err_sum = 0;
    TI err_n = 0;
};

// L2F_EVAL_NOISE=<scale> (environment variable) evaluates every policy under
// the same observation noise: the measured values times <scale>, regardless of
// the noise this binary was trained/compiled with.
template <typename ENV>
void apply_eval_noise(ENV& env){
    const char* s = std::getenv("L2F_EVAL_NOISE");
    if(!s) return;
    T k = std::atof(s);
    env.parameters.mdp.observation_noise.position = 0.005 * k;
    env.parameters.mdp.observation_noise.orientation = 0.006 * k;
    env.parameters.mdp.observation_noise.linear_velocity = 0.010 * k;
    env.parameters.mdp.observation_noise.angular_velocity = 0.069 * k;
}

// L2F_EVAL_DELAY_MS=<ms> (environment variable) feeds the actor a state estimate
// (position, orientation, velocities: the first 18 observation entries) that is
// <ms> old, linearly interpolated between 10 ms steps. The action history stays
// current, as on the drone. Models estimator / sensor-filter latency.
struct ObservationDelay{
    static constexpr TI STATE_DIM = 18;
    static constexpr TI MAX_STEPS = 16;
    T buf[MAX_STEPS][STATE_DIM];
    TI head = 0, filled = 0;
    T delay_steps = 0;
    ObservationDelay(){
        const char* s = std::getenv("L2F_EVAL_DELAY_MS");
        delay_steps = s ? std::atof(s) / 10.0 : 0;
        if(delay_steps > MAX_STEPS - 2) delay_steps = MAX_STEPS - 2;
    }
    void reset(){ filled = 0; }
    template <typename OBS>
    void apply(OBS& observation){
        head = (head + 1) % MAX_STEPS;
        for(TI i = 0; i < STATE_DIM; i++) buf[head][i] = rlt::get(observation, 0, i);
        if(filled < MAX_STEPS) filled++;
        if(delay_steps <= 0) return;
        TI k = (TI)delay_steps;
        T f = delay_steps - k;
        auto at = [&](TI back){
            if(back >= filled) back = filled - 1;   // not enough history yet: oldest available
            return (head + MAX_STEPS - back) % MAX_STEPS;
        };
        TI a = at(k), b = at(k + 1);
        for(TI i = 0; i < STATE_DIM; i++){
            rlt::set(observation, 0, i, (1 - f) * buf[a][i] + f * buf[b][i]);
        }
    }
};

// Normalized action -> motor command u in [0,1] as the firmware maps it
// (rpm = 5400 * u * vbat + 3750), at a typical in-flight vbat of 3.9 V.
inline T action_to_command(T a){
    T rpm = (a + 1) / 2 * 24000;
    return (rpm - 3750) / (5400 * 3.9);
}

// Mission profile (same as rl_trigger.py --mode mission): sit on the floor,
// climb to height H, hover, move to the circle start, one circle of radius R in
// period P, back to the centre, hover, descend to 2 cm below the floor, sit.
struct MissionProfile{
    T H = 0.4, R = 0.3, P = 8.0, laps = 1, v_up = 0.25, v_down = 0.15;
    T t_idle = 0.3, t_hover1 = 2.0, t_move = 2.0, t_back = 2.0, t_hover2 = 1.0, t_ground = 1.5;
    T below = 0.02;
    T duration() const {
        return t_idle + H / v_up + t_hover1 + t_move + laps * P + t_back + t_hover2 + (H + below) / v_down + t_ground;
    }
    // returns the phase: 0 idle, 1 take-off, 2 hover, 3 move, 4 circle, 5 back, 6 hover, 7 landing, 8 ground
    int at(T t, T* p, T* v) const {
        for(int i = 0; i < 3; i++){ p[i] = 0; v[i] = 0; }
        T t_up = H / v_up, t_circle = laps * P, t_down = (H + below) / v_down;
        if(t < t_idle) return 0;
        t -= t_idle;
        if(t < t_up){ p[2] = v_up * t; v[2] = v_up; return 1; }
        t -= t_up; p[2] = H;
        if(t < t_hover1) return 2;
        t -= t_hover1;
        if(t < t_move){ p[0] = R * t / t_move; v[0] = R / t_move; return 3; }
        t -= t_move;
        if(t < t_circle){
            T w = 2 * M_PI / P, a = w * t;
            p[0] = R * std::cos(a); p[1] = R * std::sin(a);
            v[0] = -R * w * std::sin(a); v[1] = R * w * std::cos(a);
            return 4;
        }
        t -= t_circle;
        if(t < t_back){ p[0] = R * (1 - t / t_back); v[0] = -R / t_back; return 5; }
        t -= t_back;
        if(t < t_hover2) return 6;
        t -= t_hover2;
        if(t < t_down){ p[2] = H - v_down * t; v[2] = -v_down; return 7; }
        p[2] = -below;
        return 8;
    }
};

// Full mission in simulation from the floor. Env vars: L2F_EVAL_DR=0 turns the
// per-episode domain randomization off; L2F_EVAL_DELAY_MS sets the
// state-estimate delay (default 6 ms); L2F_MISSION_CSV=<path> writes the first
// episode's trajectory.
template <typename ACTOR, typename BUFFER, typename RNG>
int run_mission(DEVICE& dev, ACTOR& actor, BUFFER& buffer, TI episodes, RNG& rng){
    MissionProfile mp;
    if(const char* s = std::getenv("L2F_MISSION_H")) mp.H = std::atof(s);
    if(const char* s = std::getenv("L2F_MISSION_R")) mp.R = std::atof(s);
    if(const char* s = std::getenv("L2F_MISSION_V_DOWN")) mp.v_down = std::atof(s);
    if(const char* s = std::getenv("L2F_MISSION_P")) mp.P = std::atof(s);
    // ground-effect strength in the mission test (training samples U(0, 2))
    const char* s_ge = std::getenv("L2F_EVAL_GE");
    T ge = s_ge ? std::atof(s_ge) : 1.0;
    // L2F_EVAL_GLITCH=<m>: one position-estimate jump of that size (random
    // direction) lasting 0.3 s, three quarters into the circle (as on 2026-11-04).
    const char* s_gl = std::getenv("L2F_EVAL_GLITCH");
    T glitch = s_gl ? std::atof(s_gl) : 0;
    const char* s_dr = std::getenv("L2F_EVAL_DR");
    const char* s_delay = std::getenv("L2F_EVAL_DELAY_MS");
    const char* s_csv = std::getenv("L2F_MISSION_CSV");
    bool dr = !s_dr || std::atoi(s_dr) != 0;
    T delay_steps = (s_delay ? std::atof(s_delay) : 6.0) / 10.0;
    // L2F_EVAL_KI=<1/s>: outer integrator (as in rl_trigger.py --mode mission)
    // that shifts the commanded target to cancel the policy's steady offset.
    const char* s_ki = std::getenv("L2F_EVAL_KI");
    T ki = s_ki ? std::atof(s_ki) : 0;
    const char* s_ki_z = std::getenv("L2F_EVAL_KI_Z");  // height part, off by default as in rl_trigger.py
    T ki_z = s_ki_z ? std::atof(s_ki_z) : 0;
    std::printf("mission: H %.2f m, circle R %.2f m in %.1f s, %.1f s total, domain randomization %s, delay %.0f ms\n",
                (double)mp.H, (double)mp.R, (double)mp.P, (double)mp.duration(), dr ? "on" : "off", (double)(delay_steps * 10));
    const char* names[] = {"idle", "take-off", "hover", "move", "circle", "back", "hover", "landing", "ground"};
    T err_sum[9] = {}, err_max[9] = {};
    TI err_n[9] = {};
    T jitter_sum = 0, touchdown_sum = 0, land_xy_sum = 0, max_tilt_all = 0, bounce_sum = 0, bounce_max = 0;
    TI crashes = 0, landed = 0, jitter_n = 0;
    T cut_h_sum = 0;
    TI cuts = 0, touched_before_cut = 0;
    std::FILE* csv = s_csv ? std::fopen(s_csv, "w") : nullptr;
    if(csv) std::fprintf(csv, "t,phase,tx,ty,tz,x,y,z,tilt_deg,u1,u2,u3,u4\n");
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, observation);
    rlt::malloc(dev, action);
    for(TI e = 0; e < episodes; e++){
        ENVIRONMENT env;
        env.parameters = PENV::parameters;
        env.parameters.mdp.reward = PENV::target_reward_function;
        apply_eval_noise(env);
        env.mission.enabled = true;
        env.mission.p_takeoff = env.mission.p_land = env.mission.p_circle = env.mission.p_line = 0;
        env.mission.p_eight = env.mission.p_fast_circle = env.mission.p_p2p = env.mission.p_gust = 0;
        env.mission.domain_randomization = dr;
        env.mission.delay_max_steps = 0;
        rlt::malloc(dev, env);
        typename ENVIRONMENT::State s, next;
        rlt::sample_initial_state(dev, env, s, rng);  // applies domain randomization
        // resting on the floor, motors stopped
        for(TI i = 0; i < 3; i++){ s.position[i] = 0; s.linear_velocity[i] = 0; s.angular_velocity[i] = 0; }
        s.orientation[0] = 1; s.orientation[1] = 0; s.orientation[2] = 0; s.orientation[3] = 0;
        for(TI m = 0; m < 4; m++){
            s.rpm[m] = env.parameters.dynamics.action_limit.min;
            for(TI h = 0; h < ENVIRONMENT::State::HISTORY_LENGTH; h++) s.action_history[h][m] = -1;
        }
        rlt::rl::environments::multirotor::mission::sync_rpm_meas(env, s);
        rlt::rl::environments::multirotor::mission::reset_fields(s);
        s.floor_z = 0;
        s.delay_steps = delay_steps;
        s.ground_effect = ge;
#ifdef L2F_SENSOR_SIM
        // sample_initial_state started the estimator at the sampled state: start it again
        // at this resting state (room frame with the floor at world z = 0)
        rlt::rl::environments::multirotor::mission::sensor_begin(dev, env, s, rng);
#endif
        T p_prev[3] = {0, 0, 0};
        T comp[3] = {0, 0, 0};  // outer-integrator target correction
        T last_err[3] = {0, 0, 0};
        TI steps = (TI)(mp.duration() / env.parameters.integration.dt);
        T usum[4] = {}, usq[4] = {};
        TI un = 0;
        bool crashed = false, touched = false;
        T touchdown_v = 0, max_tilt = 0, bounce = 0;
        // the flight script's motor cut (Mission::landing_cut): one fixed cut height, the
        // middle of the range training draws from
        bool cut_done = false;
        T low_since = -1;
        const T cut_h_eval = (env.mission.cut_h_min + env.mission.cut_h_max) / 2;
        for(TI step = 0; step < steps; step++){
            T t = step * env.parameters.integration.dt, p[3], v[3];
            int phase = mp.at(t, p, v);
            // reference (where the drone should be), before the correction
            T p_ref[3] = {p[0], p[1], p[2]};
            if(ki > 0){
                if(phase >= 2 && phase <= 6){
                    // drone error w.r.t. the reference: s.position is relative to
                    // the commanded target p_prev = p_ref_prev + comp
                    for(TI i = 0; i < 3; i++){
                        T e = -(s.position[i] + comp[i]);
                        comp[i] += (i < 2 ? ki : ki_z) * e * env.parameters.integration.dt;
                        comp[i] = std::max((T)-0.2, std::min((T)0.2, comp[i]));
                    }
                }
                T zs = std::max((T)0, std::min((T)1, p_ref[2] / mp.H));  // fade the height correction out near the floor
                p[0] += comp[0]; p[1] += comp[1]; p[2] += comp[2] * zs;
            }
            for(TI i = 0; i < 3; i++){
                T shift = p[i] - p_prev[i];
                s.position[i] -= shift;
                s.prev[0][i] -= shift;
                s.prev[1][i] -= shift;
#ifdef L2F_SENSOR_SIM
                // as training's frame shift (post_integration): the drone does not move in
                // the room, and the estimate is target-relative like position
                s.world_offset[i] += shift;
                s.est_position[i] -= shift;
#endif
                s.target_velocity[i] = v[i];
                p_prev[i] = p[i];
            }
            s.floor_z = -p[2];  // the floor is at absolute height 0
            if(glitch > 0){
                T t_glitch = mp.t_idle + mp.H / mp.v_up + mp.t_hover1 + mp.t_move + 0.75 * mp.laps * mp.P;
                if(std::abs(t - t_glitch) < env.parameters.integration.dt / 2){
                    T a = 2 * M_PI * (T)(e % 8) / 8;  // spread directions over the episodes
                    s.glitch_offset[0] = glitch * std::cos(a) * 0.95;
                    s.glitch_offset[1] = glitch * std::sin(a) * 0.95;
                    s.glitch_offset[2] = glitch * 0.3;
                    s.glitch_steps = 30;
                }
            }
            rlt::observe(dev, env, s, observation, rng);
            rlt::evaluate(dev, actor, observation, action, buffer);
            rlt::clamp(dev, action, (T)-1, (T)1);
            T vz_before = s.linear_velocity[2];
            bool was_airborne = !rlt::rl::environments::multirotor::mission::on_floor(s);
            rlt::step(dev, env, s, action, next, rng);
            s = next;
            T up = 1 - 2 * (s.orientation[1] * s.orientation[1] + s.orientation[2] * s.orientation[2]);
            T tilt = std::acos(std::max((T)-1, std::min((T)1, up))) * 180 / M_PI;
            max_tilt = std::max(max_tilt, tilt);
            // error w.r.t. the reference (the commanded target may include the correction)
            for(TI i = 0; i < 3; i++){
                last_err[i] = s.position[i] + p[i] - p_ref[i];
            }
            T err = std::sqrt(last_err[0] * last_err[0] + last_err[1] * last_err[1] + last_err[2] * last_err[2]);
            T w_max = std::max(std::abs(s.angular_velocity[0]), std::max(std::abs(s.angular_velocity[1]), std::abs(s.angular_velocity[2])));
            if(tilt > 60 || err > 0.6 || (env.mission.omega_crash > 0 && w_max > env.mission.omega_crash)){
                crashed = true;
            }
#ifdef L2F_SENSOR_SIM
            if(s.est_failed){
                crashed = true;  // a firmware assert halts the drone (training ends such episodes as crashes)
                break;
            }
#endif
            err_sum[phase] += err; err_n[phase]++;
            err_max[phase] = std::max(err_max[phase], err);
            if(phase == 2 || phase == 6){
                for(TI m = 0; m < 4; m++){
                    T u = action_to_command(rlt::get(action, 0, m));
                    usum[m] += u; usq[m] += u * u;
                }
                un++;
            }
            if(phase >= 7 && !touched && was_airborne && rlt::rl::environments::multirotor::mission::on_floor(s)){
                touched = true;
                touchdown_v = vz_before;
            }
            if(touched){
                bounce = std::max(bounce, s.position[2] - s.floor_z);
            }
            if(env.mission.landing_cut && !crashed && phase >= 7){
                // rl_trigger.py --mode mission: cut when the target is in its last
                // cut_window above the floor (or on it) and the estimated height is below
                // the cut height, or floor_timeout s after the target reached the floor
                if(phase == 8 && low_since < 0){
                    low_since = t;
                }
                bool in_window = phase == 8 || p[2] < env.mission.cut_window;
                bool cut = in_window && rlt::rl::environments::multirotor::mission::estimated_height(dev, env, s, rng) < cut_h_eval;
                bool timeout = low_since >= 0 && t - low_since > env.mission.floor_timeout;
                if(cut || timeout){
                    T h = std::max((T)0, s.position[2] - s.floor_z);
                    if(touched){
                        touched_before_cut++;  // already on the floor with motors on
                    }
                    else{
                        // free fall from the cut: v = sqrt(v_down^2 + 2 g h), downward
                        T vd = std::max((T)0, -s.linear_velocity[2]);
                        touchdown_v = -std::sqrt(vd * vd + 2 * (T)9.81 * h);
                        touched = true;
                    }
                    if(!cut && h >= env.mission.cut_max_height){
                        crashed = true;  // timeout cut far above the floor: the drone drops
                    }
                    cut_done = true;
                    cut_h_sum += h;
                    cuts++;
                    break;
                }
            }
            if(csv && e == 0){
                std::fprintf(csv, "%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.3f,%.3f,%.3f,%.3f\n", (double)t, phase,
                             (double)p_ref[0], (double)p_ref[1], (double)p_ref[2],
                             (double)(s.position[0] + p[0]), (double)(s.position[1] + p[1]), (double)(s.position[2] + p[2]), (double)tilt,
                             (double)action_to_command(rlt::get(action, 0, 0)), (double)action_to_command(rlt::get(action, 0, 1)),
                             (double)action_to_command(rlt::get(action, 0, 2)), (double)action_to_command(rlt::get(action, 0, 3)));
            }
        }
        if(un > 0){  // 0 only if the episode ended before the first hover (an estimator failure)
            T jit = 0;
            for(TI m = 0; m < 4; m++){
                T mean = usum[m] / un;
                jit += std::sqrt(std::max((T)0, usq[m] / un - mean * mean)) / 4;
            }
            jitter_sum += jit; jitter_n++;
        }
        max_tilt_all = std::max(max_tilt_all, max_tilt);
        crashes += crashed;
        if(!crashed && (cut_done || rlt::rl::environments::multirotor::mission::on_floor(s))){
            landed++;
            touchdown_sum += touched ? -touchdown_v : 0;
            bounce_sum += bounce;
            bounce_max = std::max(bounce_max, bounce);
            land_xy_sum += std::sqrt(last_err[0] * last_err[0] + last_err[1] * last_err[1]);
        }
    }
    if(csv) std::fclose(csv);
    rlt::free(dev, observation);
    rlt::free(dev, action);
    std::printf("episodes %d: crashed %d, landed on the floor %d\n", (int)episodes, (int)crashes, (int)landed);
    for(int ph = 0; ph < 9; ph++){
        if(err_n[ph]) std::printf("  %-9s tracking error mean %5.1f cm, max %5.1f cm\n", names[ph],
                                  (double)(100 * err_sum[ph] / err_n[ph]), (double)(100 * err_max[ph]));
    }
    std::printf("  hover jitter (paper metric) %.4f, max tilt %.1f deg\n", (double)(jitter_sum / jitter_n), (double)max_tilt_all);
    if(landed) std::printf("  landing: touchdown speed %.2f m/s, final horizontal error %.1f cm, bounce after touchdown mean %.1f cm / max %.1f cm\n",
                           (double)(touchdown_sum / landed), (double)(100 * land_xy_sum / landed),
                           (double)(100 * bounce_sum / landed), (double)(100 * bounce_max));
    if(cuts) std::printf("  motor cut: %d episodes, mean height %.1f cm above the floor, touched down before the cut %d\n",
                         (int)cuts, (double)(100 * cut_h_sum / cuts), (int)touched_before_cut);
    return 0;
}

template <typename ACTOR, typename BUFFER, typename RNG>
Result run(DEVICE& dev, ACTOR& actor, BUFFER& buffer, const Variation& v, bool hover_start, TI episodes, RNG& rng){
    ENVIRONMENT env;
    env.parameters = PENV::parameters;
    env.parameters.mdp.reward = PENV::target_reward_function;
    env.mission.enabled = false;  // fixed variations below; delay via L2F_EVAL_DELAY_MS
    apply_eval_noise(env);
    if(hover_start){
        env.parameters.mdp.init = rlt::rl::environments::multirotor::parameters::init::orientation_small_angle<T, TI, 4, typename PENV::REWARD_FUNCTION>;
    }
    auto& d = env.parameters.dynamics;
    d.mass *= v.mass;
    d.thrust_constants[2] *= v.thrust;
    d.rpm_time_constant *= v.tau;
    env.parameters.disturbances.random_force.std *= v.disturbance;
    env.parameters.disturbances.random_torque.std *= v.disturbance;
    for(TI i = 0; i < 3; i++){
        d.J[i][i] *= v.inertia;
        d.J_inv[i][i] /= v.inertia;
    }
    rlt::malloc(dev, env);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, observation);
    rlt::malloc(dev, action);

    Result r;
    ObservationDelay delay;
    typename ENVIRONMENT::State state, next_state;
    for(TI e = 0; e < episodes; e++){
        delay.reset();
        rlt::sample_initial_state(dev, env, state, rng);
        if(hover_start && v.history >= -1){
            T hover_rpm = std::sqrt(d.mass * 9.81 / 4 / d.thrust_constants[2]);
            T hover_action = (hover_rpm - d.action_limit.min) / (d.action_limit.max - d.action_limit.min) * 2 - 1;
            T fill = v.history > 1.5 ? hover_action : v.history;
            for(TI m = 0; m < 4; m++){
                state.rpm[m] = hover_rpm;
                for(TI h = 0; h < ENVIRONMENT::State::HISTORY_LENGTH; h++){
                    state.action_history[h][m] = (v.history_rear >= -1 && (m == 1 || m == 2)) ? v.history_rear : fill;
                }
            }
            rlt::rl::environments::multirotor::mission::sync_rpm_meas(env, state);
        }
        bool alive = true;
        for(TI step = 0; step < EPISODE_STEPS; step++){
            rlt::observe(dev, env, state, observation, rng);
            delay.apply(observation);
            rlt::evaluate(dev, actor, observation, action, buffer);
            rlt::clamp(dev, action, (T)-1, (T)1);
            rlt::step(dev, env, state, action, next_state, rng);
            state = next_state;
            if(rlt::terminated(dev, env, state, rng)){
                alive = false;
                break;
            }
            if(step >= SETTLED_FROM_STEP){
                r.err_sum += std::sqrt(state.position[0] * state.position[0] + state.position[1] * state.position[1] + state.position[2] * state.position[2]);
                r.err_n++;
            }
        }
        r.ok += alive;
        r.total++;
    }
    rlt::free(dev, observation);
    rlt::free(dev, action);
    return r;
}

// Training's own landing episodes (Mission episode type 2, sampled exactly as in
// training) flown by the actor, with this run's L2F_* settings: how each one ends.
// Shows whether the landing cut (L2F_LANDING_CUT) happens in the training
// environment, not only in the mission test.
template <typename ACTOR, typename BUFFER, typename RNG>
int run_landing(DEVICE& dev, ACTOR& actor, BUFFER& buffer, TI episodes, RNG& rng){
    ENVIRONMENT env;
    env.parameters = PENV::parameters;
    env.parameters.mdp.reward = PENV::target_reward_function;
    apply_eval_noise(env);
    env.mission.enabled = true;
    env.mission.p_takeoff = env.mission.p_circle = env.mission.p_line = 0;
    env.mission.p_eight = env.mission.p_fast_circle = env.mission.p_p2p = env.mission.p_gust = 0;
    env.mission.p_land = 1;
    rlt::malloc(dev, env);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, observation);
    rlt::malloc(dev, action);
    TI landed = 0, failed = 0, crashed = 0, flying = 0;
    T cut_h_sum = 0, touchdown_sum = 0, return_sum = 0;
    for(TI e = 0; e < episodes; e++){
        typename ENVIRONMENT::State s, next;
        rlt::sample_initial_state(dev, env, s, rng);
        bool ended = false;
        for(TI step = 0; step < CONFIG::ENVIRONMENT_STEP_LIMIT; step++){
            rlt::observe(dev, env, s, observation, rng);
            rlt::evaluate(dev, actor, observation, action, buffer);
            rlt::clamp(dev, action, (T)-1, (T)1);
            rlt::step(dev, env, s, action, next, rng);
            return_sum += rlt::reward(dev, env, s, action, next, rng);
            s = next;
            if(rlt::terminated(dev, env, s, rng)){
                ended = true;
                break;
            }
        }
        if(!ended){
            flying++;
        }
        else if(s.landed == 1){
            landed++;
            cut_h_sum += s.position[2] - s.floor_z;
            touchdown_sum += s.touchdown_speed;
        }
        else if(s.landed == 2){
            failed++;
        }
        else{
            crashed++;
        }
    }
    rlt::free(dev, observation);
    rlt::free(dev, action);
    std::printf("randomization %s (scale %.2f: mass +-%.0f%%, inertia +-%.0f%%), motor mismatch +-%.0f%% / +-%.0f rpm\n",
                env.mission.domain_randomization ? "on" : "off", (double)env.mission.dr_scale, (double)(100 * env.mission.dr_mass),
                (double)(100 * env.mission.dr_inertia), (double)(100 * env.mission.motor_gain_dr), (double)env.mission.motor_offset_dr);
    std::printf("training landing episodes %d (landing cut %s): landed %d, failed landing %d, crashed %d, "
                "still flying at the %d-step limit %d; mean return %.1f\n", (int)episodes,
                env.mission.landing_cut ? "on" : "off", (int)landed, (int)failed, (int)crashed,
                (int)CONFIG::ENVIRONMENT_STEP_LIMIT, (int)flying, (double)(return_sum / episodes));
    if(landed) std::printf("  at the cut: mean height %.1f cm above the floor, touchdown speed %.2f m/s\n",
                           (double)(100 * cut_h_sum / landed), (double)(touchdown_sum / landed));
    return 0;
}


// Waypoint episodes (Mission episode type 9, the time-to-target study, sampled as in training
// with this run's L2F_* settings, forced to 100%): per leg (the jump, then the second move)
// the time from the target's move to arrival (within L2F_ARRIVE_POS m and slower than
// L2F_ARRIVE_SPEED m/s), the overshoot past the target along the leg, the error over the
// last 0.3 s of the leg, and over the episode the peak speed and the mean rotor power above
// hover (sum (rpm / rpm_max)^3 - hover). Also arrival within 10 cm (and slower than
// L2F_ARRIVE_SPEED) and its time, on a line of its own.
// L2F_EVAL_LEG_S=<s> (test only, 0 = off): both legs last this long (the second move comes
// L2F_EVAL_LEG_S after the jump, and the episode runs on past the training step limit), so a
// slow policy can settle before a leg ends; training's legs end after 1.5-2.5 s (the first)
// and at the 5 s step limit (the second).
template <typename ACTOR, typename BUFFER, typename RNG>
int run_waypoint(DEVICE& dev, ACTOR& actor, BUFFER& buffer, TI episodes, RNG& rng){
    ENVIRONMENT env;
    env.parameters = PENV::parameters;
    env.parameters.mdp.reward = PENV::target_reward_function;
    apply_eval_noise(env);
    env.mission.enabled = true;
    env.mission.p_takeoff = env.mission.p_land = env.mission.p_circle = env.mission.p_line = 0;
    env.mission.p_eight = env.mission.p_fast_circle = env.mission.p_p2p = env.mission.p_gust = 0;
    env.mission.p_waypoint = 1;
    rlt::malloc(dev, env);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, observation);
    rlt::malloc(dev, action);
    const T dt = env.parameters.integration.dt;
    const auto& m = env.mission;
    const char* s_leg = std::getenv("L2F_EVAL_LEG_S");
    const T leg_s = s_leg ? (T)std::atof(s_leg) : 0;
    constexpr T ARRIVE_POS2 = 0.10;
    // L2F_EVAL_RESET_INTEGRAL=1 (test only, diagnostic): zero the position-error integral
    // observation whenever the target moves (it builds up while the drone flies a leg)
    const char* s_reset = std::getenv("L2F_EVAL_RESET_INTEGRAL");
    const bool reset_integral = s_reset && std::atoi(s_reset) != 0;
    T along_sum = 0, along_n = 0;  // signed final error along the leg (> 0: past the target)
    TI crashed = 0, legs = 0, arrived = 0, arrived2 = 0;
    T t2_sum = 0;
    T t_sum = 0, over_sum = 0, final_sum = 0, final_n = 0, vmax_sum = 0, power_sum = 0, power_n = 0, dist_sum = 0;
    for(TI e = 0; e < episodes; e++){
        typename ENVIRONMENT::State s, next;
        rlt::sample_initial_state(dev, env, s, rng);
        if(leg_s > 0){
            s.traj[4] = s.traj[3] + leg_s;  // the second move comes leg_s after the jump
        }
        const T t_leg[2] = {s.traj[3], s.traj[4]};
        // the episode: the training step limit, or both legs in full
        const TI n_steps = leg_s > 0 ? (TI)std::ceil((s.traj[4] + leg_s) / dt) : CONFIG::ENVIRONMENT_STEP_LIMIT;
        // leg directions (unit) and lengths: 0 -> d, then d -> traj[5] d
        T d[3], len0 = 0;
        for(TI i = 0; i < 3; i++){ d[i] = s.traj[i]; len0 += d[i] * d[i]; }
        len0 = std::sqrt(len0);
        const T len1 = std::abs(1 - s.traj[5]) * len0;
        T dir[2][3];
        for(TI i = 0; i < 3; i++){
            dir[0][i] = d[i] / (len0 + (T)1e-9);
            dir[1][i] = (s.traj[5] - 1) * d[i] / (len1 + (T)1e-9);
        }
        T t_arrive[2] = {-1, -1}, t_arrive2[2] = {-1, -1}, over[2] = {0, 0}, fin[2] = {0, 0}, fin_n[2] = {0, 0}, vmax = 0;
        bool alive = true;
        TI last_leg = 2;
        for(TI step = 0; step < n_steps; step++){
            rlt::observe(dev, env, s, observation, rng);
            rlt::evaluate(dev, actor, observation, action, buffer);
            rlt::clamp(dev, action, (T)-1, (T)1);
            rlt::step(dev, env, s, action, next, rng);
            s = next;
            if(rlt::terminated(dev, env, s, rng)){ alive = false; break; }
            const T t = s.traj_t;
            const TI leg = t >= t_leg[1] ? 1 : (t >= t_leg[0] ? 0 : 2);
            if(reset_integral && leg != last_leg){
                for(TI i = 0; i < 3; i++){ s.pos_integral[i] = 0; }
            }
            last_leg = leg;
            T p2 = 0, v2 = 0, along = 0;
            for(TI i = 0; i < 3; i++){
                p2 += s.position[i] * s.position[i];
                v2 += s.linear_velocity[i] * s.linear_velocity[i];
            }
            vmax = std::max(vmax, std::sqrt(v2));
            const T hover = std::sqrt(env.parameters.dynamics.mass * 9.81 / 4 / env.parameters.dynamics.thrust_constants[2]) / env.parameters.dynamics.action_limit.max;
            T pw = 0;
            for(TI r = 0; r < 4; r++){ const T rho = s.rpm[r] / env.parameters.dynamics.action_limit.max; pw += rho * rho * rho; }
            power_sum += pw - 4 * hover * hover * hover; power_n++;
            if(leg < 2){
                for(TI i = 0; i < 3; i++){ along += s.position[i] * dir[leg][i]; }  // > 0: past the target
                if(t_arrive[leg] < 0 && p2 < m.arrive_pos * m.arrive_pos && v2 < m.arrive_speed * m.arrive_speed){
                    t_arrive[leg] = t - t_leg[leg];
                }
                if(t_arrive2[leg] < 0 && p2 < ARRIVE_POS2 * ARRIVE_POS2 && v2 < m.arrive_speed * m.arrive_speed){
                    t_arrive2[leg] = t - t_leg[leg];
                }
                over[leg] = std::max(over[leg], along);
                const T t_end = leg == 0 ? t_leg[1] : n_steps * dt;
                if(t >= t_end - (T)0.3){ fin[leg] += std::sqrt(p2); fin_n[leg]++; along_sum += along; along_n++; }
            }
        }
        if(!alive){ crashed++; continue; }
        vmax_sum += vmax;
        for(TI l = 0; l < 2; l++){
            legs++;
            dist_sum += l == 0 ? len0 : len1;
            if(t_arrive[l] >= 0){ arrived++; t_sum += t_arrive[l]; }
            if(t_arrive2[l] >= 0){ arrived2++; t2_sum += t_arrive2[l]; }
            over_sum += over[l];
            if(fin_n[l] > 0){ final_sum += fin[l] / fin_n[l]; final_n++; }
        }
    }
    rlt::free(dev, observation);
    rlt::free(dev, action);
    const TI ok = episodes - crashed;
    std::printf("waypoint episodes %d: crashed %d; legs %d (mean %.2f m), arrived %d (%.0f%%)\n", (int)episodes, (int)crashed,
                (int)legs, (double)(legs ? dist_sum / legs : 0), (int)arrived, (double)(legs ? 100.0 * arrived / legs : 0));
    std::printf("waypoint time to target %.2f s, overshoot %.1f cm, final error %.1f cm, peak speed %.2f m/s, power %.4f\n",
                (double)(arrived ? t_sum / arrived : -1), (double)(legs ? 100 * over_sum / legs : 0),
                (double)(final_n ? 100 * final_sum / final_n : 0), (double)(ok ? vmax_sum / ok : 0),
                (double)(power_n ? power_sum / power_n : 0));
    std::printf("waypoint final error along the leg %.1f cm (> 0: past the target)%s\n",
                (double)(along_n ? 100 * along_sum / along_n : 0), reset_integral ? "; integral reset at every target move" : "");
    std::printf("waypoint 10 cm: arrived %d (%.0f%%), time to target %.2f s; legs %s\n", (int)arrived2,
                (double)(legs ? 100.0 * arrived2 / legs : 0), (double)(arrived2 ? t2_sum / arrived2 : -1),
                leg_s > 0 ? (std::to_string(leg_s).substr(0, 4) + " s each").c_str() : "as in training");
    return 0;
}

int main(int argc, char** argv){
    if(argc < 2){
        std::printf("usage: %s <actor.h5> [episodes] [mission|landing|waypoint|probe]\n", argv[0]);
        return 1;
    }
    TI episodes = argc > 2 ? std::atoi(argv[2]) : 200;
    DEVICE dev;
    typename CONFIG::ACTOR_TYPE actor;
    typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> buffer;
    rlt::malloc(dev, actor);
    rlt::malloc(dev, buffer);
    {
        auto file = HighFive::File(argv[1], HighFive::File::ReadOnly);
        rlt::load(dev, actor, file.getGroup("actor"));
    }
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 12345);

    if(argc > 3 && std::string(argv[3]) == "mission"){
        return run_mission(dev, actor, buffer, episodes, rng);
    }
    if(argc > 3 && std::string(argv[3]) == "landing"){
        return run_landing(dev, actor, buffer, episodes, rng);
    }
    if(argc > 3 && std::string(argv[3]) == "waypoint"){
        return run_waypoint(dev, actor, buffer, episodes, rng);
    }
    if(argc > 3 && std::string(argv[3]) == "probe"){
        // Start from a perfectly hovering drone and fly 5 s. Observation noise
        // is this build's (paper or L2F_REAL_NOISE); "clean" turns it off.
        ENVIRONMENT env;
        env.parameters = PENV::parameters;
        env.mission.enabled = false;
        apply_eval_noise(env);
        if(argc > 4 && std::string(argv[4]) == "clean"){
            env.parameters.mdp.observation_noise.position = 0;
            env.parameters.mdp.observation_noise.orientation = 0;
            env.parameters.mdp.observation_noise.linear_velocity = 0;
            env.parameters.mdp.observation_noise.angular_velocity = 0;
        }
        std::printf("observation noise: pos %.3f, orient %.3f, vel %.3f, gyro %.3f\n",
                    (double)env.parameters.mdp.observation_noise.position, (double)env.parameters.mdp.observation_noise.orientation,
                    (double)env.parameters.mdp.observation_noise.linear_velocity, (double)env.parameters.mdp.observation_noise.angular_velocity);
        rlt::malloc(dev, env);
        typename ENVIRONMENT::State s;
        rlt::sample_initial_state(dev, env, s, rng);
        auto& d = env.parameters.dynamics;
        T hover_rpm = std::sqrt(d.mass * 9.81 / 4 / d.thrust_constants[2]);
        T hover_action = (hover_rpm - d.action_limit.min) / (d.action_limit.max - d.action_limit.min) * 2 - 1;
        for(TI i = 0; i < 3; i++){ s.position[i] = 0; s.linear_velocity[i] = 0; s.angular_velocity[i] = 0; }
        s.orientation[0] = 1; s.orientation[1] = 0; s.orientation[2] = 0; s.orientation[3] = 0;
        for(TI m = 0; m < 4; m++){
            s.rpm[m] = hover_rpm;
            for(TI h = 0; h < ENVIRONMENT::State::HISTORY_LENGTH; h++) s.action_history[h][m] = hover_action;
        }
        rlt::rl::environments::multirotor::mission::sync_rpm_meas(env, s);
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        rlt::malloc(dev, observation);
        rlt::malloc(dev, action);
        rlt::observe(dev, env, s, observation, rng);
        rlt::evaluate(dev, actor, observation, action, buffer);
        std::printf("hover rpm %.0f, hover action %.3f, obs dim %d\nobs[0..21]:", (double)hover_rpm, (double)hover_action, (int)ENVIRONMENT::OBSERVATION_DIM);
        for(TI i = 0; i < 22; i++) std::printf(" %.3f", (double)rlt::get(observation, 0, i));
        std::printf("\naction: %.3f %.3f %.3f %.3f\n", (double)rlt::get(action, 0, 0), (double)rlt::get(action, 0, 1), (double)rlt::get(action, 0, 2), (double)rlt::get(action, 0, 3));

        // Fly 5 s from this hover state and summarize how jumpy the actions are.
        T sum[4] = {}, sq[4] = {}, usum[4] = {}, usq[4] = {}, jump = 0, sat = 0, max_tilt = 0;
        // measured-rotor-speed input (L2F_RPM_OBS, observation entries 18..21)
        T rpm_prev[4] = {-9, -9, -9, -9}, rpm_err_sq = 0;
        TI rpm_repeats = 0, rpm_fresh = 0;
        TI n = 0;
        ObservationDelay delay;
        std::printf("state-estimate delay %.0f ms\n", (double)(delay.delay_steps * 10));
        typename ENVIRONMENT::State next;
        for(TI step = 0; step < 2000; step++){
            rlt::observe(dev, env, s, observation, rng);
            delay.apply(observation);
            rlt::evaluate(dev, actor, observation, action, buffer);
            rlt::clamp(dev, action, (T)-1, (T)1);
            for(TI m = 0; m < 4; m++){
                T a = rlt::get(action, 0, m);
                sum[m] += a; sq[m] += a * a;
#ifdef L2F_RPM_OBS
                {
                    T meas = rlt::get(observation, 0, 18 + m);
                    const auto& lim = env.parameters.dynamics.action_limit;
                    T truth = (s.rpm[m] - lim.min) / (lim.max - lim.min) * 2 - 1;
                    if(meas == rpm_prev[m]) rpm_repeats++;
                    else{ rpm_err_sq += (meas - truth) * (meas - truth); rpm_fresh++; }
                    rpm_prev[m] = meas;
                }
#endif
                T u = action_to_command(a);
                usum[m] += u; usq[m] += u * u;
                sat += std::abs(a) > 0.95;
                jump += std::abs(a - s.action_history[ENVIRONMENT::State::HISTORY_LENGTH - 1][m]);
            }
            rlt::step(dev, env, s, action, next, rng);
            s = next;
            T qw = s.orientation[0];
            T tilt = 2 * std::acos(std::min((T)1, std::sqrt(qw * qw + s.orientation[3] * s.orientation[3])));
            max_tilt = std::max(max_tilt, tilt);
            n++;
        }
        T ustd = 0;
        for(TI m = 0; m < 4; m++){
            T mean = sum[m] / n;
            std::printf("motor %d: action mean %.3f std %.3f\n", (int)m, (double)mean, (double)std::sqrt(sq[m] / n - mean * mean));
            T umean = usum[m] / n;
            ustd += std::sqrt(std::max((T)0, usq[m] / n - umean * umean)) / 4;
        }
        std::printf("motor command std (paper metric, u in [0,1]) %.4f\n", (double)ustd);
#ifdef L2F_RPM_OBS
        {
            T rms = std::sqrt(rpm_err_sq / std::max<TI>(rpm_fresh, 1));
            const auto& lim = env.parameters.dynamics.action_limit;
            std::printf("rpm input: repeated %.1f%% of steps, error of fresh values std %.4f (= %.0f rpm)\n",
                        (double)(100.0 * rpm_repeats / (4 * n)), (double)rms, (double)(rms * (lim.max - lim.min) / 2));
        }
#endif
        std::printf("mean |action change| per step %.3f, saturated %.1f%%, max tilt %.1f deg, final pos (%.3f %.3f %.3f)\n",
                    (double)(jump / (4 * n)), (double)(100 * sat / (4 * n)), (double)(max_tilt * 180 / M_PI),
                    (double)s.position[0], (double)s.position[1], (double)s.position[2]);
        return 0;
    }

    const Variation variations[] = {
        {"nominal",          1.00, 1.00, 1.00, 1.0},
        {"no disturbance",   1.00, 1.00, 1.00, 1.0, 0.0},
        {"handover: history=hover",  1.00, 1.00, 1.00, 1.0, 1.0, 2.0},
        {"handover: history=-0.9",   1.00, 1.00, 1.00, 1.0, 1.0, -0.9},
        {"handover: history=-1.0",   1.00, 1.00, 1.00, 1.0, 1.0, -1.0},
        {"handover: front -1, rear -0.6", 1.00, 1.00, 1.00, 1.0, 1.0, -1.0, -0.6},
        {"handover: front -1, rear 0",    1.00, 1.00, 1.00, 1.0, 1.0, -1.0, 0.0},
        {"mass +10%",        1.10, 1.00, 1.00, 1.0},
        {"mass -10%",        0.90, 1.00, 1.00, 1.0},
        {"thrust +10%",      1.00, 1.10, 1.00, 1.0},
        {"thrust -10%",      1.00, 0.90, 1.00, 1.0},
        {"motor tau 0.04 s", 1.00, 1.00, 0.04 / 0.06, 1.0},
        {"motor tau 0.08 s", 1.00, 1.00, 0.08 / 0.06, 1.0},
        {"inertia +50%",     1.00, 1.00, 1.00, 1.5},
        {"inertia -30%",     1.00, 1.00, 1.00, 0.7},
    };
    std::printf("dynamics: mass %.4f kg, thrust c2 %.3e, tau %.3f s, max rpm %.0f\n",
                (double)PENV::parameters.dynamics.mass, (double)PENV::parameters.dynamics.thrust_constants[2],
                (double)PENV::parameters.dynamics.rpm_time_constant, (double)PENV::parameters.dynamics.action_limit.max);
    std::printf("%-26s | %-26s | %-26s\n", "variation", "training starts (<=90deg)", "near-hover starts (<=10deg)");
    for(const auto& v : variations){
        Result a = run(dev, actor, buffer, v, false, episodes, rng);
        Result b = run(dev, actor, buffer, v, true, episodes, rng);
        std::printf("%-26s | %5.1f%% ok, err %6.3f m    | %5.1f%% ok, err %6.3f m\n", v.name,
                    100.0 * a.ok / a.total, a.err_n ? (double)(a.err_sum / a.err_n) : NAN,
                    100.0 * b.ok / b.total, b.err_n ? (double)(b.err_sum / b.err_n) : NAN);
    }
    return 0;
}
