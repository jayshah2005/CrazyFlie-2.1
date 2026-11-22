// Multi-objective TD3 step (L2F_MOO). Replaces rl_tools' td3::loop::step.
//
// The per-step reward is split into K = 8 parts that add up to the scalar reward:
//   0 survival (the constant), 1 position, 2 orientation, 3 linear velocity,
//   4 action (effort), 5 smoothing (action rate or zig-zag), 6 angular acceleration, 7 landing speed
// (cost parts carry their current weight and the -scale factor; on a terminated step
// part 0 holds the whole reward, the crash penalty or the landing payout, and the
// others are 0). The critics have K outputs, one per
// part, trained on that part's TD target. The twin-critic minimum is taken on the
// summed value, so the summed target equals the scalar TD3 target exactly.
// The actor gets one gradient per output and combines them with weights from their
// Gram matrix (aggregators.h, chosen at compile time with -DL2F_MOO_AGG=<name>,
// default sum); with "sum" this is the gradient of the summed value, i.e. plain TD3.
// Optional logs, every 50 actor updates:
//   L2F_MOO_LOG=path       per-objective gradient norms, pairwise cosines, the
//                          aggregator weights and the two built-in checks
//   L2F_MOO_GRAM_LOG=path  the full Gram matrix and weights (for check_aggregators.py)
//   L2F_MOO_NOISE_LOG=path per-objective gradient agreement between two independent
//                          minibatches, and cross-batch conflict cosines (noise check)
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include "aggregators.h"
#include "aggregators_pga.h"

namespace learning_to_fly::moo {
#ifndef L2F_MOO_PARTS
#define L2F_MOO_PARTS 8
#endif
    // L2F_MOO_PARTS=1 merges all parts into one output: the same code path as K=8 but
    // learning the scalar reward, i.e. it must reproduce ordinary TD3 (diagnostic).
    constexpr int K = L2F_MOO_PARTS;
    static_assert(K == 8 || K == 1);
    inline const char* NAMES8[8] = {"survival", "position", "orientation", "linear_velocity",
                                    "action", "action_rate", "angular_acceleration", "landing_speed"};
    inline const char* name(int k) { return K == 1 ? "total" : NAMES8[k]; }
#ifdef L2F_MOO_TASKS
    // Framing B: the actor's objectives are the mission episode types (state.task), each
    // the mean critic value over a minibatch of that task's transitions; the critic is the
    // ordinary scalar one. The policy does not see the task (same inputs as the firmware).
    constexpr bool TASKS = true;
#ifndef L2F_MOO_NTASKS
#define L2F_MOO_NTASKS 5  // 9 = with the harder episode types (figure-eight .. gust)
#endif
    constexpr int NOBJ = L2F_MOO_NTASKS;
    static_assert(NOBJ == 5 || NOBJ == 9, "L2F_MOO_NTASKS: 5 or 9");
    static_assert(K == 1, "L2F_MOO_TASKS uses the scalar critic: build with L2F_MOO_PARTS=1");
#else
    constexpr bool TASKS = false;
    constexpr int NOBJ = K;  // framing A: the objectives are the reward parts
#endif
    inline const char* TASK_NAMES[9] = {"free_flight", "takeoff", "landing", "circle", "line",
                                        "eight", "fast_circle", "p2p", "gust"};
#ifdef L2F_TASK_REGIME
    static_assert(!TASKS || NOBJ == 9, "L2F_TASK_REGIME labels 9 regimes: L2F_MOO_NTASKS=9");
    // hidden-dynamics regimes: delay (short/mid/long) x mass (light/nominal/heavy)
    inline const char* REGIME_NAMES[9] = {"dS_mL", "dS_mN", "dS_mH", "dM_mL", "dM_mN", "dM_mH", "dL_mL", "dL_mN", "dL_mH"};
    inline const char* obj_name(int k) { return TASKS ? REGIME_NAMES[k] : name(k); }
#else
    inline const char* obj_name(int k) { return TASKS ? TASK_NAMES[k] : name(k); }
#endif
#ifdef L2F_MOO_NORM
    // Each critic output learns its UNWEIGHTED part divided by a fixed scale s_k (the part's
    // RMS reward / (1 - gamma), measured once from the replay buffer when critic training
    // starts), so every output is O(1) and the curriculum weights w_k(t) act outside the
    // network: value = sum_k w_k s_k out_k. Against Adam giving tiny parts the same step size.
    constexpr bool NORM = true;
    static_assert(K == 8, "L2F_MOO_NORM needs the 8 parts");
#else
    constexpr bool NORM = false;
#endif

    template <typename DEVICE, typename ENV, typename P, typename ACTION, typename RNG, typename T>
    void reward_vector_parts(DEVICE& device, const ENV& env, const P& p, const typename ENV::State& state, const ACTION& action,
                             const typename ENV::State& next_state, RNG& rng, T* out);
    template <typename P>
    void part_weights(const P& p, double* w);
    // Reward parts for one transition (K of them); they add up to the scalar reward.
    template <typename DEVICE, typename ENV, typename ACTION, typename RNG, typename T>
    void reward_vector(DEVICE& device, const ENV& env, const typename ENV::State& state, const ACTION& action,
                       const typename ENV::State& next_state, RNG& rng, T* out) {
        const auto& p = env.parameters.mdp.reward;
        T u[8];
        double w[8];
        reward_vector_parts(device, env, p, state, action, next_state, rng, u);
        part_weights(p, w);
        if constexpr (K == 1) {
            out[0] = 0;
            for (int k = 0; k < 8; k++) out[0] += w[k] * u[k];
        } else {
            for (int k = 0; k < 8; k++) out[k] = NORM ? u[k] : (T)(w[k] * u[k]);
        }
    }
    template <typename DEVICE, typename ENV, typename P, typename ACTION, typename RNG, typename T>
    void reward_vector_parts(DEVICE& device, const ENV& env, const P& p, const typename ENV::State& state, const ACTION& action,
                             const typename ENV::State& next_state, RNG& rng, T* out) {
        auto c = rlt::rl::environments::multirotor::parameters::reward_functions::reward_components(device, env, p, state, action, next_state, rng);
        if (c.terminated) {
            // the crash penalty, or the landing payout for a landing (both in part 0)
            for (int k = 0; k < 8; k++) out[k] = k == 0 ? c.reward : 0;
            return;
        }
        // unweighted parts u_k; the reward is sum_k w_k u_k with w from part_weights
        out[0] = p.constant;
        out[1] = -p.scale * c.position_cost;
        out[2] = -p.scale * c.orientation_cost;
        out[3] = -p.scale * c.linear_vel_cost;
        out[4] = -p.scale * c.action_cost;
        // smoothing: action rate, zig-zag, or (both weighted) their weighted sum with weight 1
        out[5] = -p.scale * (p.zigzag == 0 ? c.action_rate_cost
                             : (p.action_rate == 0 ? c.zigzag_cost : p.action_rate * c.action_rate_cost + p.zigzag * c.zigzag_cost));
        out[6] = -p.scale * c.angular_acc_cost;
        out[7] = -p.scale * c.landing_cost;
    }
    // Current weight of each part (the curriculum changes them during training).
    template <typename P>
    void part_weights(const P& p, double* w) {
        double smoothing = p.zigzag == 0 ? p.action_rate : (p.action_rate == 0 ? p.zigzag : 1.0);
        double w8[8] = {1, p.position, p.orientation, p.linear_velocity, p.action, smoothing, p.angular_acceleration, p.landing_speed};
        for (int k = 0; k < 8; k++) w[k] = w8[k];
    }

    // Flatten / write back the parameter gradients of a sequential model.
    template <typename MODULE, typename F>
    void for_each_layer(MODULE& m, F&& f) {
        f(m.content);
        if constexpr (!std::is_same_v<typename MODULE::NEXT_MODULE, rlt::nn_models::sequential::OutputModule>) {
            for_each_layer(m.next_module, f);
        }
    }
    template <typename MAT, typename F>
    void for_each_entry(MAT& mat, F&& f) {
        using S = typename std::remove_reference_t<MAT>::SPEC;
        for (typename S::TI r = 0; r < S::ROWS; r++)
            for (typename S::TI c = 0; c < S::COLS; c++) f(mat, r, c);
    }
    template <typename MODULE>
    void read_gradient(MODULE& model, std::vector<double>& g) {
        g.clear();
        for_each_layer(model, [&](auto& layer) {
            for_each_entry(layer.weights.gradient, [&](auto& m, auto r, auto c) { g.push_back(rlt::get(m, r, c)); });
            for_each_entry(layer.biases.gradient, [&](auto& m, auto r, auto c) { g.push_back(rlt::get(m, r, c)); });
        });
    }
    template <typename MODULE>
    void write_gradient(MODULE& model, const std::vector<double>& g) {
        size_t i = 0;
        for_each_layer(model, [&](auto& layer) {
            for_each_entry(layer.weights.gradient, [&](auto& m, auto r, auto c) { rlt::set(m, r, c, g[i++]); });
            for_each_entry(layer.biases.gradient, [&](auto& m, auto r, auto c) { rlt::set(m, r, c, g[i++]); });
        });
    }

#ifndef L2F_MOO_AGG
#define L2F_MOO_AGG sum
#endif
#define L2F_MOO_STR2(x) #x
#define L2F_MOO_STR(x) L2F_MOO_STR2(x)
    // Aggregator: combination weights from the NOBJ x NOBJ Gram matrix (double precision).
    namespace agg {
        // Tasks only, not a Gram-matrix aggregator: weight each task by its share of the
        // replay buffer (times NOBJ), i.e. ordinary TD3 on the mixed buffer, split per task.
        // The fair "no aggregation" baseline for framing B.
        struct task_share {};
    }
    using AGG = agg::L2F_MOO_AGG;
    static_assert(!std::is_same_v<AGG, agg::task_share> || TASKS, "AGG=task_share needs L2F_MOO_TASKS");
    inline const char* AGG_NAME = L2F_MOO_STR(L2F_MOO_AGG);
    template <typename A = AGG>  // template: only instantiated for Gram-matrix aggregators
    void aggregate_weights(const double gram[NOBJ][NOBJ], double w[NOBJ]) { agg::weights<NOBJ>(A{}, gram, w); }

    template <typename CONFIG>
    struct State {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        static constexpr TI B = CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, B, K>> rewards, next_value_1, next_value_2, target, d_output, value;
        bool ready = false;
        FILE* log = nullptr;
        FILE* gram_log = nullptr;
        FILE* noise_log = nullptr;
        TI actor_updates = 0;
        double reward_check_max = 0;  // max |sum of parts - stored scalar reward| since the last log line
        double scale[K];              // L2F_MOO_NORM: s_k (1 otherwise)
        bool scales_ready = false;
    };

    // Same sampling as rl_tools' gather_batch (same random draws), plus the reward parts.
    // task >= 0: only transitions of that episode type (rejection sampling); returns false
    // if none is found (the task is not in the buffer yet).
    template <typename TS, typename BATCH, typename REWARDS, typename RNG>
    bool gather_batch(TS& ts, BATCH& batch, REWARDS& rewards, double& reward_check_max, RNG& sample_rng, int task = -1) {
        using RUNNER = std::remove_reference_t<decltype(ts.off_policy_runner)>;
        using RSPEC = typename RUNNER::SPEC;
        using DEVICE = std::remove_reference_t<decltype(ts.device)>;
        using TI = typename DEVICE::index_t;
        using T = typename RSPEC::T;
        auto& device = ts.device;
        auto& runner = ts.off_policy_runner;
        constexpr TI BATCH_SIZE = std::remove_reference_t<BATCH>::BATCH_SIZE;
        static auto rng_reward = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 0);
        for (TI b = 0; b < BATCH_SIZE; b++) {
            TI env_i = 0, i = 0;
            bool found = false;
            for (int attempt = 0; attempt < (task < 0 ? 1 : 5000) && !found; attempt++) {
                env_i = rlt::random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (TI)0, RSPEC::N_ENVIRONMENTS - 1, sample_rng);
                auto& rb = runner.replay_buffers[env_i];
                using RB_SPEC = typename std::remove_reference_t<decltype(rb)>::SPEC;
                TI index_max = (rb.full ? RB_SPEC::CAPACITY : rb.position) - 1;
                i = rlt::random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (TI)0, index_max, sample_rng);
                found = task < 0 || (int)rlt::get(rb.states, i, 0).task == task;
            }
            if (!found) return false;
            auto& rb = runner.replay_buffers[env_i];

            auto obs_t = rlt::row(device, batch.observations, b);
            auto obs_s = rlt::row(device, rb.observations, i);
            rlt::copy(device, device, obs_s, obs_t);
            if constexpr (RSPEC::ASYMMETRIC_OBSERVATIONS) {
                auto t = rlt::row(device, batch.observations_privileged, b);
                auto s = rlt::row(device, rb.observations_privileged, i);
                rlt::copy(device, device, s, t);
            }
            auto act_t = rlt::row(device, batch.actions, b);
            auto act_s = rlt::row(device, rb.actions, i);
            rlt::copy(device, device, act_s, act_t);
            auto nobs_t = rlt::row(device, batch.next_observations, b);
            auto nobs_s = rlt::row(device, rb.next_observations, i);
            rlt::copy(device, device, nobs_s, nobs_t);
            if constexpr (RSPEC::ASYMMETRIC_OBSERVATIONS) {
                auto t = rlt::row(device, batch.next_observations_privileged, b);
                auto s = rlt::row(device, rb.next_observations_privileged, i);
                rlt::copy(device, device, s, t);
            }
            rlt::set(batch.rewards, 0, b, rlt::get(rb.rewards, i, 0));
            rlt::set(batch.terminated, 0, b, rlt::get(rb.terminated, i, 0));
            rlt::set(batch.truncated, 0, b, rlt::get(rb.truncated, i, 0));

            T parts[K];
            reward_vector(device, runner.envs[env_i], rlt::get(rb.states, i, 0), act_s, rlt::get(rb.next_states, i, 0), rng_reward, parts);
            double sum = 0, w[8];
            part_weights(runner.envs[env_i].parameters.mdp.reward, w);
            for (int k = 0; k < K; k++) {
                rlt::set(rewards, b, k, parts[k]);
                sum += NORM ? w[k] * parts[k] : parts[k];
            }
            double diff = std::abs(sum - (double)rlt::get(rb.rewards, i, 0));
            if (diff > reward_check_max) reward_check_max = diff;
        }
        return true;
    }

    // L2F_MOO_NORM: fix s_k = RMS(u_k) / (1 - gamma) from 20k replay-buffer samples.
    template <typename TS>
    void estimate_scales(TS& ts, State<typename TS::CONFIG>& st) {
        using DEVICE = std::remove_reference_t<decltype(ts.device)>;
        using TI = typename DEVICE::index_t;
        using T = typename TS::CONFIG::T;
        for (int k = 0; k < K; k++) st.scale[k] = 1;
        st.scales_ready = true;
        if (!NORM) return;
        auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 1);
        auto& rb = ts.off_policy_runner.replay_buffers[0];
        using RB_SPEC = typename std::remove_reference_t<decltype(rb)>::SPEC;
        TI index_max = (rb.full ? RB_SPEC::CAPACITY : rb.position) - 1;
        double sq[K] = {0};
        const int N = 20000;
        for (int n = 0; n < N; n++) {
            TI i = rlt::random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (TI)0, index_max, rng);
            auto act = rlt::row(ts.device, rb.actions, i);
            T u[K];
            reward_vector(ts.device, ts.off_policy_runner.envs[0], rlt::get(rb.states, i, 0), act, rlt::get(rb.next_states, i, 0), rng, u);
            for (int k = 0; k < K; k++) sq[k] += (double)u[k] * u[k];
        }
        double rms_max = 0, rms[K];
        for (int k = 0; k < K; k++) { rms[k] = std::sqrt(sq[k] / N); rms_max = std::max(rms_max, rms[k]); }
        std::printf("L2F_MOO_NORM scales at step %lu:", (unsigned long)ts.step);
        for (int k = 0; k < K; k++) {
            // a part that never occurred yet (e.g. landing before the first touchdown) gets 1
            st.scale[k] = rms[k] > 1e-6 ? rms[k] / (1 - ts.actor_critic.gamma) : 1.0;
            std::printf(" %s %.3g", name(k), st.scale[k]);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    // Factor turning output k into its share of the (weighted) value: w_k s_k, or 1.
    template <typename TS>
    void value_factors(TS& ts, State<typename TS::CONFIG>& st, double* f) {
        double w[8];
        part_weights(ts.off_policy_runner.envs[0].parameters.mdp.reward, w);
        for (int k = 0; k < K; k++) f[k] = NORM ? w[k] * st.scale[k] : 1;
    }

    template <typename TS>
    void train_critic(TS& ts, State<typename TS::CONFIG>& st, int critic_i) {
        using CONFIG = typename TS::CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        constexpr TI B = State<CONFIG>::B;
        auto& device = ts.device;
        auto& ac = ts.actor_critic;
        auto& tb = ts.critic_training_buffers;
        auto& batch = ts.critic_batch;
        auto& critic = critic_i == 0 ? ac.critic_1 : ac.critic_2;

        rlt::target_action_noise(device, ac, tb.target_next_action_noise, ts.rng);
        gather_batch(ts, batch, st.rewards, st.reward_check_max, ts.rng);
        rlt::zero_gradient(device, critic);
        rlt::evaluate(device, ac.actor_target, batch.next_observations, tb.next_actions, ts.actor_buffers[critic_i]);
        rlt::noisy_next_actions(device, tb);
        rlt::copy(device, device, batch.next_observations_privileged, tb.next_observations);
        rlt::evaluate(device, ac.critic_target_1, tb.next_state_action_value_input, st.next_value_1, ts.critic_buffers[critic_i]);
        rlt::evaluate(device, ac.critic_target_2, tb.next_state_action_value_input, st.next_value_2, ts.critic_buffers[critic_i]);
        if (!st.scales_ready) estimate_scales(ts, st);
        double f[K];
        value_factors(ts, st, f);
        for (TI b = 0; b < B; b++) {
            T total_1 = 0, total_2 = 0;
            for (int k = 0; k < K; k++) {
                total_1 += f[k] * rlt::get(st.next_value_1, b, k);
                total_2 += f[k] * rlt::get(st.next_value_2, b, k);
            }
            auto& next = total_1 < total_2 ? st.next_value_1 : st.next_value_2;  // twin minimum on the sum
            bool terminated = rlt::get(batch.terminated, 0, b);
            bool bootstrap = CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::IGNORE_TERMINATION || !terminated;
            for (int k = 0; k < K; k++) {
                T future = bootstrap ? ac.gamma * rlt::get(next, b, k) : 0;
                rlt::set(st.target, b, k, rlt::get(st.rewards, b, k) / (T)st.scale[k] + future);
            }
        }
        rlt::forward(device, critic, batch.observations_and_actions);
#ifdef L2F_MOO_TOTAL_LOSS
        // Hybrid loss: (error of the weighted total)^2 + L2F_MOO_TOTAL_LOSS * sum_k (error of part k)^2
        // (part errors in their own units). The total keeps the single-output TD3 loss.
        {
            auto out = rlt::output(critic);
            for (TI b = 0; b < B; b++) {
                double e_total = 0;
                for (int k = 0; k < K; k++) e_total += f[k] * (rlt::get(out, b, k) - rlt::get(st.target, b, k));
                for (int k = 0; k < K; k++) {
                    double e_k = rlt::get(out, b, k) - rlt::get(st.target, b, k);
                    rlt::set(st.d_output, b, k, (T)(2.0 / B * (f[k] * e_total + (L2F_MOO_TOTAL_LOSS) * e_k)));
                }
            }
        }
#else
        // loss weight K keeps each output's gradient at the scale of the single-output loss
        rlt::nn::loss_functions::mse::gradient(device, rlt::output(critic), st.target, st.d_output, (T)K);
#endif
        rlt::backward(device, critic, batch.observations_and_actions, st.d_output, ts.critic_buffers[critic_i]);
        rlt::step(device, ts.critic_optimizers[critic_i], critic);
    }

    template <typename TS>
    void train_actor(TS& ts, State<typename TS::CONFIG>& st) {
        using CONFIG = typename TS::CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        constexpr TI B = State<CONFIG>::B;
        constexpr TI ACTION_DIM = CONFIG::ENVIRONMENT::ACTION_DIM;
        constexpr TI CRITIC_INPUT_DIM = CONFIG::CRITIC_TYPE::INPUT_DIM;
        auto& device = ts.device;
        auto& ac = ts.actor_critic;
        auto& tb = ts.actor_training_buffers;
        auto& batch = ts.actor_batch;
        auto& critic = ac.critic_1;

        // sample a minibatch (only episodes of `task` if task >= 0) and run actor and critic
        // forward on it; false if the task has no transitions in the buffer yet
        auto prepare = [&](auto& rng, int task) {
            if (!gather_batch(ts, batch, st.rewards, st.reward_check_max, rng, task)) return false;
            rlt::forward(device, ac.actor, batch.observations, tb.actions);
            rlt::copy(device, device, batch.observations_privileged, tb.observations);
            rlt::forward(device, critic, tb.state_action_value_input, st.value);
            return true;
        };
        auto d_actor_output = rlt::view(device, tb.d_critic_input, rlt::matrix::ViewSpec<B, ACTION_DIM>{}, 0, CRITIC_INPUT_DIM - ACTION_DIM);
        double f[K];
        value_factors(ts, st, f);

        // gradient of sum_j c_j * (-mean_b Q_j) w.r.t. the actor parameters (current batch)
        auto weighted_gradient = [&](const double* c, std::vector<double>& g) {
            rlt::zero_gradient(device, ac.actor);
            for (TI b = 0; b < B; b++)
                for (int j = 0; j < K; j++) rlt::set(st.d_output, b, j, (T)(-c[j] * f[j] / B));
            rlt::backward_input(device, critic, st.d_output, tb.d_critic_input, ts.critic_buffers[0]);
            rlt::backward(device, ac.actor, batch.observations, d_actor_output, ts.actor_buffers[0]);
            read_gradient(ac.actor, g);
        };
        // The NOBJ objective gradients. Reward parts (framing A): one batch, one critic output
        // each. Tasks (framing B, L2F_MOO_TASKS): one batch per task, the scalar critic value;
        // a task without transitions yet gets a zero gradient.
        auto collect = [&](auto& rng, std::vector<double>* out) {
            if constexpr (TASKS) {
                for (int t = 0; t < NOBJ; t++) {
                    double c[K] = {prepare(rng, t) ? 1.0 : 0.0};
                    weighted_gradient(c, out[t]);
                }
            } else {
                prepare(rng, -1);
                for (int k = 0; k < K; k++) {
                    double c[K];
                    for (int j = 0; j < K; j++) c[j] = j == k;
                    weighted_gradient(c, out[k]);
                }
            }
        };
        static std::vector<double> G[NOBJ];
        collect(ts.rng, G);
        const size_t P = G[0].size();
        double gram[NOBJ][NOBJ];
        for (int a = 0; a < NOBJ; a++)
            for (int b = a; b < NOBJ; b++) {
                double s = 0;
                for (size_t i = 0; i < P; i++) s += G[a][i] * G[b][i];
                gram[a][b] = gram[b][a] = s;
            }
        double w[NOBJ];
        if constexpr (std::is_same_v<AGG, agg::task_share>) {
            // task shares from 4096 random replay-buffer entries (own RNG)
            static auto rng_share = rlt::random::default_engine(typename std::remove_reference_t<decltype(device)>::SPEC::RANDOM(), 54321);
            using RSPEC = typename std::remove_reference_t<decltype(ts.off_policy_runner)>::SPEC;
            constexpr int SAMPLES = 4096;
            double count[NOBJ] = {};
            for (int s = 0; s < SAMPLES; s++) {
                TI env_i = rlt::random::uniform_int_distribution(typename std::remove_reference_t<decltype(device)>::SPEC::RANDOM(), (TI)0, RSPEC::N_ENVIRONMENTS - 1, rng_share);
                auto& rb = ts.off_policy_runner.replay_buffers[env_i];
                using RB_SPEC = typename std::remove_reference_t<decltype(rb)>::SPEC;
                TI index_max = (rb.full ? RB_SPEC::CAPACITY : rb.position) - 1;
                TI i = rlt::random::uniform_int_distribution(typename std::remove_reference_t<decltype(device)>::SPEC::RANDOM(), (TI)0, index_max, rng_share);
                int t = (int)rlt::get(rb.states, i, 0).task;
                if (t >= 0 && t < NOBJ) count[t] += 1;
            }
            for (int k = 0; k < NOBJ; k++) w[k] = NOBJ * count[k] / SAMPLES;
        } else {
            aggregate_weights(gram, w);
        }
        static std::vector<double> combined;
        combined.assign(P, 0);
        for (int k = 0; k < NOBJ; k++)
            for (size_t i = 0; i < P; i++) combined[i] += w[k] * G[k][i];

        st.actor_updates++;
        bool log_now = (st.log || st.gram_log || st.noise_log) && st.actor_updates % 50 == 0;
        if (log_now && st.log) {
            // check: combined per-objective gradients == one backward pass on the weighted value
            // (reward parts only: the tasks use different batches, grad_check is then 0)
            double check = 0;
            if constexpr (!TASKS) {
                static std::vector<double> reference;
                weighted_gradient(w, reference);
                double diff = 0, scale = 0;
                for (size_t i = 0; i < P; i++) {
                    diff = std::max(diff, std::abs(combined[i] - reference[i]));
                    scale = std::max(scale, std::abs(reference[i]));
                }
                check = scale > 0 ? diff / scale : 0.0;
            }
            std::fprintf(st.log, "%lu,%.3e,%.3e", (unsigned long)ts.step, st.reward_check_max, check);
            for (int k = 0; k < NOBJ; k++) std::fprintf(st.log, ",%.4e", std::sqrt(gram[k][k]));
            for (int a = 0; a < NOBJ; a++)
                for (int b = a + 1; b < NOBJ; b++) {
                    double d = std::sqrt(gram[a][a] * gram[b][b]);
                    std::fprintf(st.log, ",%.4f", d > 0 ? gram[a][b] / d : 0.0);
                }
            for (int k = 0; k < NOBJ; k++) std::fprintf(st.log, ",%.4e", w[k]);
            std::fprintf(st.log, "\n");
            std::fflush(st.log);
            st.reward_check_max = 0;
        }
        if (log_now && st.gram_log) {
            std::fprintf(st.gram_log, "%lu", (unsigned long)ts.step);
            for (int a = 0; a < NOBJ; a++)
                for (int b = a; b < NOBJ; b++) std::fprintf(st.gram_log, ",%.17g", gram[a][b]);
            for (int k = 0; k < NOBJ; k++) std::fprintf(st.gram_log, ",%.17g", w[k]);
            std::fprintf(st.gram_log, "\n");
            std::fflush(st.gram_log);
        }
        if (log_now && st.noise_log) {
            // Noise check: the same objectives on second, independent minibatches (own RNG, so
            // training is unchanged). same_k = cos(g_k batch 1, g_k batch 2): is objective k's
            // direction reproducible? cross_ab = mean of cos(g_a batch 1, g_b batch 2) and
            // cos(g_b batch 1, g_a batch 2): the a-b conflict without shared-sample noise.
            static auto rng_noise = rlt::random::default_engine(typename std::remove_reference_t<decltype(device)>::SPEC::RANDOM(), 12345);
            static std::vector<double> H[NOBJ];
            collect(rng_noise, H);
            auto cosine = [&](const std::vector<double>& x, const std::vector<double>& y) {
                double xy = 0, xx = 0, yy = 0;
                for (size_t i = 0; i < P; i++) { xy += x[i] * y[i]; xx += x[i] * x[i]; yy += y[i] * y[i]; }
                return xx > 0 && yy > 0 ? xy / std::sqrt(xx * yy) : 0.0;
            };
            std::fprintf(st.noise_log, "%lu", (unsigned long)ts.step);
            for (int k = 0; k < NOBJ; k++) std::fprintf(st.noise_log, ",%.4f", cosine(G[k], H[k]));
            for (int a = 0; a < NOBJ; a++)
                for (int b = a + 1; b < NOBJ; b++)
                    std::fprintf(st.noise_log, ",%.4f", 0.5 * (cosine(G[a], H[b]) + cosine(H[a], G[b])));
            std::fprintf(st.noise_log, "\n");
            std::fflush(st.noise_log);
        }
        write_gradient(ac.actor, combined);
        rlt::step(device, ts.actor_optimizer, ac.actor);
    }

    // Drop-in replacement for rlt::rl::algorithms::td3::loop::step.
    template <typename TS>
    bool step(TS& ts) {
        using CONFIG = typename TS::CONFIG;
        using SPEC = CONFIG;
        using TI = typename CONFIG::TI;
        static_assert(CONFIG::CRITIC_TYPE::OUTPUT_DIM == K, "build with L2F_MOO so the critic has K outputs");
        static State<CONFIG> st;
        auto& device = ts.device;
        if (!st.ready) {
            const auto& p = ts.off_policy_runner.envs[0].parameters.mdp.reward;
            if (p.non_negative || p.angular_velocity != 0 || p.linear_acceleration != 0) {
                std::fprintf(stderr, "L2F_MOO: reward has terms outside the K=%d parts (non_negative, angular_velocity or linear_acceleration)\n", K);
                std::exit(1);
            }
            rlt::malloc(device, st.rewards); rlt::malloc(device, st.next_value_1); rlt::malloc(device, st.next_value_2);
            rlt::malloc(device, st.target); rlt::malloc(device, st.d_output); rlt::malloc(device, st.value);
            if (const char* path = std::getenv("L2F_MOO_LOG")) {
                st.log = std::fopen(path, "w");
                std::fprintf(st.log, "step,reward_check,grad_check");
                for (int k = 0; k < NOBJ; k++) std::fprintf(st.log, ",norm_%s", obj_name(k));
                for (int a = 0; a < NOBJ; a++)
                    for (int b = a + 1; b < NOBJ; b++) std::fprintf(st.log, ",cos_%s__%s", obj_name(a), obj_name(b));
                for (int k = 0; k < NOBJ; k++) std::fprintf(st.log, ",w_%s", obj_name(k));
                std::fprintf(st.log, "\n");
            }
            if (const char* path = std::getenv("L2F_MOO_GRAM_LOG")) {
                st.gram_log = std::fopen(path, "w");
                std::fprintf(st.gram_log, "step");
                for (int a = 0; a < NOBJ; a++)
                    for (int b = a; b < NOBJ; b++) std::fprintf(st.gram_log, ",g_%d_%d", a, b);
                for (int k = 0; k < NOBJ; k++) std::fprintf(st.gram_log, ",w_%s", obj_name(k));
                std::fprintf(st.gram_log, "\n");
            }
            if (const char* path = std::getenv("L2F_MOO_NOISE_LOG")) {
                st.noise_log = std::fopen(path, "w");
                std::fprintf(st.noise_log, "step");
                for (int k = 0; k < NOBJ; k++) std::fprintf(st.noise_log, ",same_%s", obj_name(k));
                for (int a = 0; a < NOBJ; a++)
                    for (int b = a + 1; b < NOBJ; b++) std::fprintf(st.noise_log, ",cross_%s__%s", obj_name(a), obj_name(b));
                std::fprintf(st.noise_log, "\n");
            }
            std::printf("L2F_MOO aggregator: %s, objectives: %s (%d)\n", AGG_NAME, TASKS ? "tasks" : "reward parts", NOBJ);
            st.ready = true;
        }
        rlt::set_step(device, device.logger, ts.step);
        if constexpr (SPEC::DETERMINISTIC_EVALUATION == true) {
            if (ts.step % SPEC::EVALUATION_INTERVAL == 0) {
                auto result = rlt::evaluate(device, ts.env_eval, ts.ui, ts.actor_critic.actor, rlt::rl::utils::evaluation::Specification<SPEC::NUM_EVALUATION_EPISODES, SPEC::ENVIRONMENT_STEP_LIMIT_EVALUATION>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng_eval, false);
                TI current_evaluation_i = ts.step / SPEC::EVALUATION_INTERVAL;
                ts.evaluation_results[current_evaluation_i] = result;
                rlt::add_scalar(device, device.logger, "evaluation/returns_mean", result.returns_mean);
            }
        }
        rlt::step(device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if (ts.step > SPEC::N_WARMUP_STEPS_CRITIC && ts.step % SPEC::TD3_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
            for (int critic_i = 0; critic_i < 2; critic_i++) train_critic(ts, st, critic_i);
        }
        if (ts.step > SPEC::N_WARMUP_STEPS_ACTOR && ts.step % SPEC::TD3_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0) {
            train_actor(ts, st);
        }
        if (ts.step > SPEC::N_WARMUP_STEPS_CRITIC && ts.step % SPEC::TD3_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0) {
            rlt::update_critic_targets(device, ts.actor_critic);
        }
        if (ts.step > SPEC::N_WARMUP_STEPS_ACTOR && ts.step % SPEC::TD3_PARAMETERS::ACTOR_TARGET_UPDATE_INTERVAL == 0) {
            rlt::update_actor_target(device, ts.actor_critic);
        }
        ts.step++;
        ts.finished = ts.step >= SPEC::STEP_LIMIT;
        return ts.finished;
    }
}
