// Convert an actor checkpoint (.h5) into the firmware's actor header (.h), the same
// way src/steps/checkpoint.h writes it during training: the weights as code, an
// example observation and the actor's action for it (the firmware adapter's
// self-test), and a name. Works for checkpoints from C++ training and from the
// PyTorch training (l2f/py/export.py), as long as the build's flags match the
// checkpoint's input layout (variant).
//
// Usage: h5_to_header <actor.h5> <actor.h> [name]
#include "training.h"

#include <highfive/H5File.hpp>
#include <cstdio>
#include <fstream>
#include <string>

using CONFIG = learning_to_fly::config::Config<parameters::DefaultAblationSpec>;
using DEVICE = typename CONFIG::DEVICE;
using T = typename CONFIG::T;
using TI = typename CONFIG::TI;
using ENVIRONMENT = typename CONFIG::ENVIRONMENT;
using PENV = parameters::environment<T, TI, parameters::DefaultAblationSpec>;

int main(int argc, char** argv){
    if(argc < 3){
        std::printf("usage: %s <actor.h5> <actor.h> [name]\n", argv[0]);
        return 1;
    }
    std::string name = argc > 3 ? argv[3] : argv[1];
    DEVICE dev;
    typename CONFIG::ACTOR_TYPE actor;
    rlt::malloc(dev, actor);
    {
        auto file = HighFive::File(argv[1], HighFive::File::ReadOnly);
        rlt::load(dev, actor, file.getGroup("actor"));
    }
    typename CONFIG::ACTOR_CHECKPOINT_TYPE checkpoint;
    typename decltype(checkpoint)::template DoubleBuffer<1> buffer;
    rlt::malloc(dev, checkpoint);
    rlt::malloc(dev, buffer);
    rlt::copy(dev, dev, actor, checkpoint);
    std::ofstream out(argv[2]);
    out << rlt::save_code(dev, checkpoint, std::string("rl_tools::checkpoint::actor"), true);
    {
        ENVIRONMENT env;
        env.parameters = PENV::parameters;
        rlt::malloc(dev, env);
        auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 0);
        typename ENVIRONMENT::State state;
        rlt::sample_initial_state(dev, env, state, rng);
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        rlt::malloc(dev, observation);
        rlt::malloc(dev, action);
        rlt::observe(dev, env, state, observation, rng);
        rlt::evaluate(dev, checkpoint, observation, action, buffer);
        out << "\n" << rlt::save_code(dev, observation, std::string("rl_tools::checkpoint::observation"), true);
        out << "\n" << rlt::save_code(dev, action, std::string("rl_tools::checkpoint::action"), true);
        out << "\n" << "namespace rl_tools::checkpoint::meta{";
        out << "\n" << "   " << "char name[] = \"" << name << "\";";
        out << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
        out << "\n" << "}";
        rlt::free(dev, observation);
        rlt::free(dev, action);
    }
    rlt::free(dev, checkpoint);
    rlt::free(dev, buffer);
    rlt::free(dev, actor);
    std::printf("wrote %s (%d inputs, %d outputs)\n", argv[2], (int)ENVIRONMENT::OBSERVATION_DIM, (int)ENVIRONMENT::ACTION_DIM);
    return 0;
}
