cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0003 NEW)


project(learning_to_fly)

set(CMAKE_CXX_STANDARD 17)

find_package(HDF5)
if(HDF5_FOUND)
    set(RL_TOOLS_ENABLE_HDF5 ON)
endif()
if(APPLE)
find_package(Protobuf CONFIG)
else()
find_package(Protobuf)
endif()
if(Protobuf_FOUND)
    set(RL_TOOLS_ENABLE_TENSORBOARD ON)
endif()
set(RL_TOOLS_ENABLE_JSON ON)
set(RL_TOOLS_ENABLE_BOOST_BEAST ON)

if(APPLE)
    set(RL_TOOLS_BACKEND_ENABLE_ACCELERATE ON)
    # Homebrew Boost >= 1.89 has no boost_system library, which HighFive's optional
    # Boost containers ask for; rl_tools does not use them.
    set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
endif()

# CMake >= 4 rejects the old cmake_minimum_required of bundled deps (HighFive)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

add_library(learning_to_fly INTERFACE)
target_include_directories(learning_to_fly INTERFACE include)

#set(RL_TOOLS_ENABLE_TESTS ON)
add_subdirectory(external/rl_tools)
if(APPLE)
    # rl_tools adds -Ofast (fast-math) on Apple; it lets the compiler drop NaN checks
    # (crash detection, evaluate_robustness). Keep plain -O3 like the Linux builds.
    set_property(TARGET rl_tools_optimizations PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()
add_subdirectory(src)

option(LEARNING_TO_FLY_IN_SECONDS_ENABLE_TESTS "Enable tests" OFF)

if(LEARNING_TO_FLY_IN_SECONDS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()