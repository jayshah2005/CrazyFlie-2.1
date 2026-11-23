// Python module of the l2f simulator (plan A2): the training's C++ environment,
// stepped from Python. Same headers and the same -D flags as l2f/sim/train_mission.sh
// (one module per build), so the physics, noise, randomization, mission episodes and
// rewards are exactly the ones C++ training uses.
//
// One Sim holds n independent environments and one random-number generator, and
// follows the loop of rl_tools' off-policy runner (off_policy_runner/
// operations_generic_per_env.h): observe (actor and critic), step, reward,
// observe the next state, terminated. Exploration noise, truncation at
// info()["step_limit"] and resets are the caller's.
//
// The reward comes split into unweighted parts u_k (see PART_NAMES), so a replay
// buffer can store them and apply the curriculum's current weights when it samples:
//   r = sum_k w_k * u_k,   w = weights()   (w_0 = 1; u_0 = the survival bonus, or on
//   the terminating step the crash penalty / landing payout, with all other u_k = 0)
//
// Built by learning_to_fly/src/CMakeLists.txt with -DL2F_BUILD_PYTHON=ON.
#include "../learning_to_fly/src/training.h"

#include <highfive/H5File.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace py = pybind11;

using CONFIG = learning_to_fly::config::Config<parameters::DefaultAblationSpec>;
using DEVICE = typename CONFIG::DEVICE;
using T = typename CONFIG::T;
using TI = typename CONFIG::TI;
using ENVIRONMENT = typename CONFIG::ENVIRONMENT;
using STATE = typename ENVIRONMENT::State;
using PENV = parameters::environment<T, TI, parameters::DefaultAblationSpec>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0));
constexpr TI OBS_DIM = ENVIRONMENT::OBSERVATION_DIM;
constexpr TI OBS_DIM_PRIVILEGED = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
constexpr TI N_PARTS = 14;
const std::vector<std::string> PART_NAMES = {
    "survival", "position", "orientation", "linear_velocity", "angular_velocity", "linear_acceleration",
    "angular_acceleration", "action", "action_rate", "zigzag", "landing_speed",
    "time", "power", "speed_limit"};

// Unweighted reward parts of one transition (the reward is sum_k w_k u_k with w from weights_of).
template <typename P, typename C>
void parts_of(const P& p, const C& c, T* u){
    if(c.terminated){
        u[0] = c.reward;  // crash penalty or landing payout
        for(TI k = 1; k < N_PARTS; k++) u[k] = 0;
        return;
    }
    u[0] = p.constant;
    u[1] = -p.scale * c.position_cost;
    u[2] = -p.scale * c.orientation_cost;
    u[3] = -p.scale * c.linear_vel_cost;
    u[4] = -p.scale * c.angular_vel_cost;
    u[5] = -p.scale * c.linear_acc_cost;
    u[6] = -p.scale * c.angular_acc_cost;
    u[7] = -p.scale * c.action_cost;
    u[8] = -p.scale * c.action_rate_cost;
    u[9] = -p.scale * c.zigzag_cost;
    u[10] = -p.scale * c.landing_cost;
    u[11] = -p.scale * c.time_cost;
    u[12] = -p.scale * c.power_cost;
    u[13] = -p.scale * c.speed_cost;
}
template <typename P>
void weights_of(const P& p, T* w){
    T v[N_PARTS] = {1, p.position, p.orientation, p.linear_velocity, p.angular_velocity, p.linear_acceleration,
                    p.angular_acceleration, p.action, p.action_rate, p.zigzag, p.landing_speed,
                    p.time, p.power, p.speed_limit};
    for(TI k = 0; k < N_PARTS; k++) w[k] = v[k];
}

template <TI COLS>
using Row = rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, COLS>>;

class Sim{
public:
    Sim(TI n_envs, std::uint64_t seed): n(n_envs), envs(n_envs), states(n_envs),
        rng(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), (TI)seed)){
        // weights of the time-to-target terms (0 = off), runtime options like the L2F_* ones
        auto env_or = [](const char* name){ const char* v = std::getenv(name); return v ? (T)std::atof(v) : (T)0; };
        const T w_time = env_or("L2F_W_TIME"), w_power = env_or("L2F_W_POWER"), w_speed = env_or("L2F_W_SPEED");
        for(auto& env: envs){
            env.parameters = PENV::parameters;
            env.parameters.mdp.reward.time = w_time;
            env.parameters.mdp.reward.power = w_power;
            env.parameters.mdp.reward.speed_limit = w_speed;
            rlt::malloc(dev, env);
        }
        rlt::malloc(dev, obs);
        rlt::malloc(dev, obs_p);
        rlt::malloc(dev, action);
    }
    ~Sim(){
        rlt::free(dev, obs);
        rlt::free(dev, obs_p);
        rlt::free(dev, action);
    }
    // New episodes (rlt::sample_initial_state) for the environments where mask is true (all if None).
    void reset(py::object mask){
        std::vector<bool> m(n, true);
        if(!mask.is_none()){
            auto a = mask.cast<py::array_t<bool>>();
            if((TI)a.size() != n) throw std::runtime_error("reset: mask must have n_envs entries");
            for(TI i = 0; i < n; i++) m[i] = a.data()[i];
        }
        for(TI i = 0; i < n; i++){
            if(m[i]) rlt::sample_initial_state(dev, envs[i], states[i], rng);
        }
    }
    // Actor and critic observations of the current states: (n, OBS_DIM), (n, OBS_DIM_PRIVILEGED).
    py::tuple observe(){
        py::array_t<T> a({(py::ssize_t)n, (py::ssize_t)OBS_DIM}), p({(py::ssize_t)n, (py::ssize_t)OBS_DIM_PRIVILEGED});
        for(TI i = 0; i < n; i++) observe_one(i, a.mutable_data() + i * OBS_DIM, p.mutable_data() + i * OBS_DIM_PRIVILEGED);
        return py::make_tuple(a, p);
    }
    // Advance every environment by 10 ms with actions (n, ACTION_DIM) in [-1, 1].
    py::dict step(py::array_t<T, py::array::c_style | py::array::forcecast> actions){
        if(actions.ndim() != 2 || (TI)actions.shape(0) != n || (TI)actions.shape(1) != ACTION_DIM)
            throw std::runtime_error("step: actions must be (n_envs, action_dim)");
        py::array_t<T> next_obs({(py::ssize_t)n, (py::ssize_t)OBS_DIM}), next_obs_p({(py::ssize_t)n, (py::ssize_t)OBS_DIM_PRIVILEGED});
        py::array_t<T> parts({(py::ssize_t)n, (py::ssize_t)N_PARTS}), reward({(py::ssize_t)n});
        py::array_t<bool> terminated({(py::ssize_t)n}), landed({(py::ssize_t)n});
        py::array_t<std::int32_t> task({(py::ssize_t)n});
        for(TI i = 0; i < n; i++){
            for(TI k = 0; k < ACTION_DIM; k++) rlt::set(action, 0, k, actions.data()[i * ACTION_DIM + k]);
            STATE next;
            auto& env = envs[i];
            rlt::step(dev, env, states[i], action, next, rng);
            auto c = rlt::rl::environments::multirotor::parameters::reward_functions::reward_components(
                dev, env, env.parameters.mdp.reward, states[i], action, next, rng);
            parts_of(env.parameters.mdp.reward, c, parts.mutable_data() + i * N_PARTS);
            reward.mutable_data()[i] = c.reward;
            states[i] = next;
            observe_one(i, next_obs.mutable_data() + i * OBS_DIM, next_obs_p.mutable_data() + i * OBS_DIM_PRIVILEGED);
            terminated.mutable_data()[i] = rlt::terminated(dev, env, states[i], rng);
            landed.mutable_data()[i] = states[i].landed == 1;
            task.mutable_data()[i] = (std::int32_t)states[i].task;
        }
        py::dict out;
        out["next_obs"] = next_obs;
        out["next_obs_privileged"] = next_obs_p;
        out["parts"] = parts;
        out["reward"] = reward;
        out["terminated"] = terminated;
        out["landed"] = landed;
        out["task"] = task;
        return out;
    }
    // Mission curriculum (src/steps/curriculum.h): 0 = plain hover training, 1 = full.
    void set_progress(T p){
        for(auto& env: envs) env.mission.progress = p;
    }
    // Current part weights w_k (the same in every environment).
    py::array_t<T> weights() const{
        py::array_t<T> w({(py::ssize_t)N_PARTS});
        weights_of(envs[0].parameters.mdp.reward, w.mutable_data());
        return w;
    }
    // Set the weights the curriculum changes (keys: position, linear_velocity, action).
    void set_weights(const py::dict& w){
        for(auto& env: envs){
            auto& r = env.parameters.mdp.reward;
            for(auto item: w){
                auto key = item.first.cast<std::string>();
                T v = item.second.cast<T>();
                if(key == "position") r.position = v;
                else if(key == "linear_velocity") r.linear_velocity = v;
                else if(key == "action") r.action = v;
                else if(key == "time") r.time = v;
                else if(key == "power") r.power = v;
                else if(key == "speed_limit") r.speed_limit = v;
                else throw std::runtime_error("set_weights: unknown key " + key);
            }
        }
    }
    // True state of every environment (for tests and logging).
    py::dict states_now() const{
        py::array_t<T> pos({(py::ssize_t)n, (py::ssize_t)3}), vel({(py::ssize_t)n, (py::ssize_t)3});
        py::array_t<T> floor({(py::ssize_t)n}), touchdown({(py::ssize_t)n});
        py::array_t<std::int32_t> landed({(py::ssize_t)n}), task({(py::ssize_t)n});
        for(TI i = 0; i < n; i++){
            for(TI k = 0; k < 3; k++){
                pos.mutable_data()[i * 3 + k] = states[i].position[k];
                vel.mutable_data()[i * 3 + k] = states[i].linear_velocity[k];
            }
            floor.mutable_data()[i] = states[i].floor_z;
            touchdown.mutable_data()[i] = states[i].touchdown_speed;
            landed.mutable_data()[i] = (std::int32_t)states[i].landed;
            task.mutable_data()[i] = (std::int32_t)states[i].task;
        }
        py::dict out;
        out["position"] = pos;
        out["linear_velocity"] = vel;
        out["floor_z"] = floor;
        out["landed"] = landed;
        out["touchdown_speed"] = touchdown;
        out["task"] = task;
        py::array_t<T> q({(py::ssize_t)n, (py::ssize_t)4}), w({(py::ssize_t)n, (py::ssize_t)3});
        for(TI i = 0; i < n; i++){
            for(TI k = 0; k < 4; k++) q.mutable_data()[i * 4 + k] = states[i].orientation[k];
            for(TI k = 0; k < 3; k++) w.mutable_data()[i * 3 + k] = states[i].angular_velocity[k];
        }
        out["orientation"] = q;
        out["angular_velocity"] = w;
#ifdef L2F_SENSOR_SIM
        // the sensor path's estimate (same frames as position / orientation / velocity)
        py::array_t<T> ep({(py::ssize_t)n, (py::ssize_t)3}), eq({(py::ssize_t)n, (py::ssize_t)4});
        py::array_t<T> ev({(py::ssize_t)n, (py::ssize_t)3}), eg({(py::ssize_t)n, (py::ssize_t)3});
        for(TI i = 0; i < n; i++){
            for(TI k = 0; k < 3; k++){
                ep.mutable_data()[i * 3 + k] = states[i].est_position[k];
                ev.mutable_data()[i * 3 + k] = states[i].est_velocity[k];
                eg.mutable_data()[i * 3 + k] = states[i].gyro_obs[k];
            }
            for(TI k = 0; k < 4; k++) eq.mutable_data()[i * 4 + k] = states[i].est_orientation[k];
        }
        out["est_position"] = ep;
        out["est_orientation"] = eq;
        out["est_velocity"] = ev;
        out["gyro_obs"] = eg;
        py::array_t<std::int32_t> ef({(py::ssize_t)n});
        for(TI i = 0; i < n; i++) ef.mutable_data()[i] = (std::int32_t)states[i].est_failed;
        out["est_failed"] = ef;
#endif
        return out;
    }
    TI n;

private:
    void observe_one(TI i, T* a, T* p){
        rlt::observe(dev, envs[i], states[i], obs, rng);
        rlt::observe_privileged(dev, envs[i], states[i], obs_p, rng);
        for(TI k = 0; k < OBS_DIM; k++) a[k] = rlt::get(obs, 0, k);
        for(TI k = 0; k < OBS_DIM_PRIVILEGED; k++) p[k] = rlt::get(obs_p, 0, k);
    }
    DEVICE dev;
    std::vector<ENVIRONMENT> envs;
    std::vector<STATE> states;
    RNG rng;
    Row<OBS_DIM> obs;
    Row<OBS_DIM_PRIVILEGED> obs_p;
    Row<ACTION_DIM> action;
};

// Reference rollout written directly against the simulator, in the runner's call
// order, for the test that the Sim API reproduces it (l2f/py/test_sim.py). Actions:
// a_k(t) = clip(hover + 0.3 sin(0.05 t + k)); resets at a termination or step_limit.
py::dict reference_rollout(std::uint64_t seed, TI steps){
    DEVICE dev;
    ENVIRONMENT env;
    env.parameters = PENV::parameters;
    rlt::malloc(dev, env);
    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), (TI)seed);
    Row<OBS_DIM> obs;
    Row<OBS_DIM_PRIVILEGED> obs_p;
    Row<ACTION_DIM> action;
    rlt::malloc(dev, obs);
    rlt::malloc(dev, obs_p);
    rlt::malloc(dev, action);
    py::array_t<T> o({(py::ssize_t)steps, (py::ssize_t)OBS_DIM}), r({(py::ssize_t)steps});
    py::array_t<T> acts({(py::ssize_t)steps, (py::ssize_t)ACTION_DIM});
    py::array_t<bool> term({(py::ssize_t)steps});
    STATE s, next;
    rlt::sample_initial_state(dev, env, s, rng);
    TI episode_step = 0;
    for(TI t = 0; t < steps; t++){
        rlt::observe(dev, env, s, obs, rng);
        rlt::observe_privileged(dev, env, s, obs_p, rng);
        for(TI k = 0; k < OBS_DIM; k++) o.mutable_data()[t * OBS_DIM + k] = rlt::get(obs, 0, k);
        for(TI k = 0; k < ACTION_DIM; k++){
            T a = (T)0.288 + (T)0.3 * std::sin((T)0.05 * t + k);
            a = a > 1 ? 1 : (a < -1 ? -1 : a);
            rlt::set(action, 0, k, a);
            acts.mutable_data()[t * ACTION_DIM + k] = a;
        }
        rlt::step(dev, env, s, action, next, rng);
        r.mutable_data()[t] = rlt::reward(dev, env, s, action, next, rng);
        s = next;
        rlt::observe(dev, env, s, obs, rng);              // the runner's next observation
        rlt::observe_privileged(dev, env, s, obs_p, rng);
        bool done = rlt::terminated(dev, env, s, rng);
        term.mutable_data()[t] = done;
        episode_step++;
        if(done || episode_step == CONFIG::ENVIRONMENT_STEP_LIMIT){
            rlt::sample_initial_state(dev, env, s, rng);
            episode_step = 0;
        }
    }
    rlt::free(dev, obs);
    rlt::free(dev, obs_p);
    rlt::free(dev, action);
    py::dict out;
    out["obs"] = o;
    out["reward"] = r;
    out["terminated"] = term;
    out["actions"] = acts;
    return out;
}

// rl_tools' own TD3 updates on data given from Python (plan A5 check, l2f/py/test_td3.py):
// actor-critic initialized by rl_tools (init_weights with `seed`), then train_critic /
// train_actor / target updates exactly as rl_tools' TD3 loop calls them, on the batches and
// target-smoothing noise passed in. The PyTorch TD3 run on the same data must match it.
using TS = learning_to_fly::TrainingState<CONFIG>;
template <typename M>
using member_t = std::remove_reference_t<M>;
class TD3Reference{
public:
    using AC = member_t<decltype(std::declval<TS&>().actor_critic)>;
    using OPT = member_t<decltype(std::declval<TS&>().actor_optimizer)>;
    using CRITIC_BATCH = member_t<decltype(std::declval<TS&>().critic_batch)>;
    using ACTOR_BATCH = member_t<decltype(std::declval<TS&>().actor_batch)>;
    using CTB = member_t<decltype(std::declval<TS&>().critic_training_buffers)>;
    using ATB = member_t<decltype(std::declval<TS&>().actor_training_buffers)>;
    using CB = member_t<decltype(std::declval<TS&>().critic_buffers[0])>;
    using AB = member_t<decltype(std::declval<TS&>().actor_buffers[0])>;
    static constexpr TI B = CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
    TD3Reference(std::uint64_t seed): rng(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), (TI)seed)){
        rlt::malloc(dev, ac);
        rlt::malloc(dev, critic_batch);
        rlt::malloc(dev, actor_batch);
        rlt::malloc(dev, ctb);
        rlt::malloc(dev, atb);
        for(TI i = 0; i < 2; i++){
            rlt::malloc(dev, critic_buffers[i]);
            rlt::malloc(dev, actor_buffers[i]);
        }
        rlt::init(dev, ac, rng);
    }
    ~TD3Reference(){
        rlt::free(dev, ac);
        rlt::free(dev, critic_batch);
        rlt::free(dev, actor_batch);
        rlt::free(dev, ctb);
        rlt::free(dev, atb);
        for(TI i = 0; i < 2; i++){
            rlt::free(dev, critic_buffers[i]);
            rlt::free(dev, actor_buffers[i]);
        }
    }
    template <typename LAYER>
    static py::tuple layer_arrays(const LAYER& l){
        using WS = typename std::remove_reference_t<decltype(l.weights.parameters)>::SPEC;
        py::array_t<T> w({(py::ssize_t)WS::ROWS, (py::ssize_t)WS::COLS}), b({(py::ssize_t)WS::ROWS});
        for(TI i = 0; i < WS::ROWS; i++){
            b.mutable_data()[i] = rlt::get(l.biases.parameters, 0, i);
            for(TI j = 0; j < WS::COLS; j++) w.mutable_data()[i * WS::COLS + j] = rlt::get(l.weights.parameters, i, j);
        }
        return py::make_tuple(w, b);
    }
    template <typename MODEL>
    static py::list model_arrays(const MODEL& m){
        py::list out;
        out.append(layer_arrays(m.content));
        out.append(layer_arrays(m.next_module.content));
        out.append(layer_arrays(m.next_module.next_module.content));
        return out;
    }
    // All six networks as {name: [(W, b) per layer]}, W (out, in) as torch.nn.Linear.
    py::dict weights() const{
        py::dict d;
        d["actor"] = model_arrays(ac.actor);
        d["actor_target"] = model_arrays(ac.actor_target);
        d["critic_1"] = model_arrays(ac.critic_1);
        d["critic_2"] = model_arrays(ac.critic_2);
        d["critic_target_1"] = model_arrays(ac.critic_target_1);
        d["critic_target_2"] = model_arrays(ac.critic_target_2);
        d["gamma"] = (double)ac.gamma;
        return d;
    }
    template <typename VIEW>
    static void fill(VIEW& v, const py::array_t<T, py::array::c_style | py::array::forcecast>& a, TI cols){
        if(a.ndim() != 2 || (TI)a.shape(0) != B || (TI)a.shape(1) != cols) throw std::runtime_error("TD3Reference: wrong batch shape");
        for(TI i = 0; i < B; i++)
            for(TI j = 0; j < cols; j++) rlt::set(v, i, j, a.data()[i * cols + j]);
    }
    using A = py::array_t<T, py::array::c_style | py::array::forcecast>;
    // rl_tools train_critic for critic i (0 or 1) on this batch with this target noise (B, 4).
    void train_critic(int i, A obs, A obs_p, A act, A reward, A next_obs, A next_obs_p, py::array_t<bool> terminated, A noise){
        fill(critic_batch.observations, obs, OBS_DIM);
        fill(critic_batch.observations_privileged, obs_p, OBS_DIM_PRIVILEGED);
        fill(critic_batch.actions, act, ACTION_DIM);
        fill(critic_batch.next_observations, next_obs, OBS_DIM);
        fill(critic_batch.next_observations_privileged, next_obs_p, OBS_DIM_PRIVILEGED);
        fill(ctb.target_next_action_noise, noise, ACTION_DIM);
        for(TI b = 0; b < B; b++){
            rlt::set(critic_batch.rewards, 0, b, reward.data()[b]);
            rlt::set(critic_batch.terminated, 0, b, terminated.data()[b]);
            rlt::set(critic_batch.truncated, 0, b, false);
        }
        rlt::train_critic(dev, ac, i == 0 ? ac.critic_1 : ac.critic_2, critic_batch, critic_optimizers[i], actor_buffers[i], critic_buffers[i], ctb);
    }
    // rl_tools train_actor on these observations (actor and critic inputs).
    void train_actor(A obs, A obs_p){
        fill(actor_batch.observations, obs, OBS_DIM);
        fill(actor_batch.observations_privileged, obs_p, OBS_DIM_PRIVILEGED);
        rlt::train_actor(dev, ac, actor_batch, actor_optimizer, actor_buffers[0], critic_buffers[0], atb);
    }
    void update_critic_targets(){ rlt::update_critic_targets(dev, ac); }
    void update_actor_target(){ rlt::update_actor_target(dev, ac); }
private:
    DEVICE dev;
    RNG rng;
    AC ac;
    OPT actor_optimizer, critic_optimizers[2];
    CRITIC_BATCH critic_batch;
    ACTOR_BATCH actor_batch;
    CTB ctb;
    ATB atb;
    CB critic_buffers[2];
    AB actor_buffers[2];
};

// Actor of a checkpoint (.h5) evaluated by rl_tools on observations (n, OBS_DIM): the
// C++ reference for the PyTorch actor (l2f/py/test_networks.py).
py::array_t<T> evaluate_actor(const std::string& path, py::array_t<T, py::array::c_style | py::array::forcecast> observations){
    DEVICE dev;
    typename CONFIG::ACTOR_TYPE actor;
    typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> buffer;
    rlt::malloc(dev, actor);
    rlt::malloc(dev, buffer);
    {
        auto file = HighFive::File(path, HighFive::File::ReadOnly);
        rlt::load(dev, actor, file.getGroup("actor"));
    }
    TI n = (TI)observations.shape(0);
    if(observations.ndim() != 2 || (TI)observations.shape(1) != OBS_DIM) throw std::runtime_error("evaluate_actor: observations must be (n, obs_dim)");
    py::array_t<T> out({(py::ssize_t)n, (py::ssize_t)ACTION_DIM});
    Row<OBS_DIM> o;
    Row<ACTION_DIM> a;
    rlt::malloc(dev, o);
    rlt::malloc(dev, a);
    for(TI i = 0; i < n; i++){
        for(TI k = 0; k < OBS_DIM; k++) rlt::set(o, 0, k, observations.data()[i * OBS_DIM + k]);
        rlt::evaluate(dev, actor, o, a, buffer);
        for(TI k = 0; k < ACTION_DIM; k++) out.mutable_data()[i * ACTION_DIM + k] = rlt::get(a, 0, k);
    }
    rlt::free(dev, o);
    rlt::free(dev, a);
    rlt::free(dev, actor);
    rlt::free(dev, buffer);
    return out;
}

py::dict info(){
    py::dict d;
    d["obs_dim"] = (int)OBS_DIM;
    d["obs_dim_privileged"] = (int)OBS_DIM_PRIVILEGED;
    d["action_dim"] = (int)ACTION_DIM;
    d["part_names"] = PART_NAMES;
    d["step_limit"] = (int)CONFIG::ENVIRONMENT_STEP_LIMIT;
    d["dt"] = (double)PENV::parameters.integration.dt;
    d["hover_action"] = (double)L2F_CF21BL_ACTION_BASELINE;
#ifdef L2F_MISSION
    d["mission"] = true;
#else
    d["mission"] = false;
#endif
#ifdef L2F_RPM_OBS
    d["rpm_obs"] = true;
#else
    d["rpm_obs"] = false;
#endif
    d["state_action_history"] = (int)STATE::HISTORY_LENGTH;  // past actions kept in the state (not the actor input)
#ifdef L2F_SENSOR_SIM
    d["sensor_sim"] = true;
#else
    d["sensor_sim"] = false;
#endif
    // the reward-weight curriculum (curriculum.h) is off in the untuned-reward build; train.py
    // applies the curriculum itself, so it must know
#ifdef L2F_REWARD_UNIFORM
    d["weight_curriculum"] = false;
#else
    d["weight_curriculum"] = true;
#endif
    return d;
}

PYBIND11_MODULE(l2f_sim, m){
    m.doc() = "The l2f training simulator (C++), stepped from Python; see sim_binding.cpp.";
    py::class_<Sim>(m, "Sim")
        .def(py::init<TI, std::uint64_t>(), py::arg("n_envs"), py::arg("seed"))
        .def("reset", &Sim::reset, py::arg("mask") = py::none())
        .def("observe", &Sim::observe)
        .def("step", &Sim::step, py::arg("actions"))
        .def("set_progress", &Sim::set_progress, py::arg("progress"))
        .def("weights", &Sim::weights)
        .def("set_weights", &Sim::set_weights, py::arg("weights"))
        .def("states", &Sim::states_now)
        .def_readonly("n_envs", &Sim::n);
    m.def("reference_rollout", &reference_rollout, py::arg("seed"), py::arg("steps"));
    m.def("evaluate_actor", &evaluate_actor, py::arg("path"), py::arg("observations"));
    py::class_<TD3Reference>(m, "TD3Reference")
        .def(py::init<std::uint64_t>(), py::arg("seed"))
        .def("weights", &TD3Reference::weights)
        .def("train_critic", &TD3Reference::train_critic)
        .def("train_actor", &TD3Reference::train_actor)
        .def("update_critic_targets", &TD3Reference::update_critic_targets)
        .def("update_actor_target", &TD3Reference::update_actor_target);
    m.def("info", &info);
}
