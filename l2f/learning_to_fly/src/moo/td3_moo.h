// Multi-objective TD3 step (L2F_MOO). Replaces rl_tools' td3::loop::step.
//
// The per-step reward is split into K = 8 parts that add up to the scalar reward:
//   0 survival (the constant), 1 position, 2 orientation, 3 linear velocity,
//   4 action (effort), 5 action rate, 6 angular acceleration, 7 landing speed
// (cost parts carry their current weight and the -scale factor; all parts are 0 on
// a terminated step, like the scalar reward). The critics have K outputs, one per
// part, trained on that part's TD target. The twin-critic minimum is taken on the
// summed value, so the summed target equals the scalar TD3 target exactly.
// The actor gets one gradient per output and combines them; with the "sum"
// aggregator this is the gradient of the summed value, i.e. plain TD3.
// Optional log (L2F_MOO_LOG=path): per logged actor update the per-objective
// gradient norms and pairwise cosines, plus the two built-in checks.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

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
            for (int k = 0; k < 8; k++) out[k] = p.termination_penalty * (k == 0);
            return;
        }
        // unweighted parts u_k; the reward is sum_k w_k u_k with w from part_weights
        out[0] = p.constant;
        out[1] = -p.scale * c.position_cost;
        out[2] = -p.scale * c.orientation_cost;
        out[3] = -p.scale * c.linear_vel_cost;
        out[4] = -p.scale * c.action_cost;
        out[5] = -p.scale * c.action_rate_cost;
        out[6] = -p.scale * c.angular_acc_cost;
        out[7] = -p.scale * c.landing_cost;
    }
    // Current weight of each part (the curriculum changes them during training).
    template <typename P>
    void part_weights(const P& p, double* w) {
        double w8[8] = {1, p.position, p.orientation, p.linear_velocity, p.action, p.action_rate, p.angular_acceleration, p.landing_speed};
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

    // Aggregator: combination weights from the K x K Gram matrix (double precision).
    // Only the plain sum for now (= TD3 on the summed value).
    inline void aggregate_weights(const double gram[K][K], double w[K]) {
        (void)gram;
        for (int k = 0; k < K; k++) w[k] = 1;
    }

    template <typename CONFIG>
    struct State {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        static constexpr TI B = CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, B, K>> rewards, next_value_1, next_value_2, target, d_output, value;
        bool ready = false;
        FILE* log = nullptr;
        TI actor_updates = 0;
        double reward_check_max = 0;  // max |sum of parts - stored scalar reward| since the last log line
        double scale[K];              // L2F_MOO_NORM: s_k (1 otherwise)
        bool scales_ready = false;
    };

    // Same sampling as rl_tools' gather_batch (same random draws), plus the reward parts.
    template <typename TS, typename BATCH, typename REWARDS>
    void gather_batch(TS& ts, BATCH& batch, REWARDS& rewards, double& reward_check_max) {
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
            TI env_i = rlt::random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (TI)0, RSPEC::N_ENVIRONMENTS - 1, ts.rng);
            auto& rb = runner.replay_buffers[env_i];
            using RB_SPEC = typename std::remove_reference_t<decltype(rb)>::SPEC;
            TI index_max = (rb.full ? RB_SPEC::CAPACITY : rb.position) - 1;
            TI i = rlt::random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (TI)0, index_max, ts.rng);

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
        gather_batch(ts, batch, st.rewards, st.reward_check_max);
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

        gather_batch(ts, batch, st.rewards, st.reward_check_max);
        rlt::forward(device, ac.actor, batch.observations, tb.actions);
        rlt::copy(device, device, batch.observations_privileged, tb.observations);
        rlt::forward(device, critic, tb.state_action_value_input, st.value);
        auto d_actor_output = rlt::view(device, tb.d_critic_input, rlt::matrix::ViewSpec<B, ACTION_DIM>{}, 0, CRITIC_INPUT_DIM - ACTION_DIM);
        double f[K];
        value_factors(ts, st, f);

        // gradient of objective k: d(-mean_b Q_k)/d(actor parameters)
        auto objective_gradient = [&](int k, std::vector<double>& g) {
            rlt::zero_gradient(device, ac.actor);
            for (TI b = 0; b < B; b++)
                for (int j = 0; j < K; j++) rlt::set(st.d_output, b, j, (k < 0 || j == k) ? (T)(-f[j] / B) : 0);
            rlt::backward_input(device, critic, st.d_output, tb.d_critic_input, ts.critic_buffers[0]);
            rlt::backward(device, ac.actor, batch.observations, d_actor_output, ts.actor_buffers[0]);
            read_gradient(ac.actor, g);
        };
        static std::vector<double> G[K];
        for (int k = 0; k < K; k++) objective_gradient(k, G[k]);
        const size_t P = G[0].size();
        double gram[K][K];
        for (int a = 0; a < K; a++)
            for (int b = a; b < K; b++) {
                double s = 0;
                for (size_t i = 0; i < P; i++) s += G[a][i] * G[b][i];
                gram[a][b] = gram[b][a] = s;
            }
        double w[K];
        aggregate_weights(gram, w);
        static std::vector<double> combined;
        combined.assign(P, 0);
        for (int k = 0; k < K; k++)
            for (size_t i = 0; i < P; i++) combined[i] += w[k] * G[k][i];

        st.actor_updates++;
        bool log_now = st.log && st.actor_updates % 50 == 0;
        if (log_now) {
            // check: sum of per-objective gradients == one backward pass on the summed value
            static std::vector<double> reference;
            objective_gradient(-1, reference);
            double diff = 0, scale = 0;
            for (size_t i = 0; i < P; i++) {
                diff = std::max(diff, std::abs(combined[i] - reference[i]));
                scale = std::max(scale, std::abs(reference[i]));
            }
            std::fprintf(st.log, "%lu,%.3e,%.3e", (unsigned long)ts.step, st.reward_check_max, scale > 0 ? diff / scale : 0.0);
            for (int k = 0; k < K; k++) std::fprintf(st.log, ",%.4e", std::sqrt(gram[k][k]));
            for (int a = 0; a < K; a++)
                for (int b = a + 1; b < K; b++) {
                    double d = std::sqrt(gram[a][a] * gram[b][b]);
                    std::fprintf(st.log, ",%.4f", d > 0 ? gram[a][b] / d : 0.0);
                }
            std::fprintf(st.log, "\n");
            std::fflush(st.log);
            st.reward_check_max = 0;
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
                for (int k = 0; k < K; k++) std::fprintf(st.log, ",norm_%s", name(k));
                for (int a = 0; a < K; a++)
                    for (int b = a + 1; b < K; b++) std::fprintf(st.log, ",cos_%s__%s", name(a), name(b));
                std::fprintf(st.log, "\n");
            }
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
