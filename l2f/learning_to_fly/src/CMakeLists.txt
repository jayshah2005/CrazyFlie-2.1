
add_executable(training_headless training.cpp)
target_link_libraries(
        training_headless
        PRIVATE
        rl_tools
        learning_to_fly
)
if(RL_TOOLS_ENABLE_HDF5)
add_executable(evaluate_robustness evaluate_robustness.cpp)
target_link_libraries(
        evaluate_robustness
        PRIVATE
        rl_tools
        learning_to_fly
)
endif()
# actor checkpoint .h5 -> firmware actor header (for PyTorch-trained checkpoints, plan A3)
if(RL_TOOLS_ENABLE_HDF5)
add_executable(h5_to_header h5_to_header.cpp)
target_link_libraries(
        h5_to_header
        PRIVATE
        rl_tools
        learning_to_fly
)
endif()

# Python module of the simulator (l2f/py/sim_binding.cpp, plan A2), off by default:
#   cmake ... -DL2F_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)   (needs pybind11)
option(L2F_BUILD_PYTHON "Build the l2f_sim Python module" OFF)
if(L2F_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(l2f_sim ${CMAKE_CURRENT_SOURCE_DIR}/../../py/sim_binding.cpp)
    target_link_libraries(l2f_sim PRIVATE rl_tools learning_to_fly)
endif()
