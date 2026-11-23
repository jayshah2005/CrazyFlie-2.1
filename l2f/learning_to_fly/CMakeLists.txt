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
endif()

add_library(learning_to_fly INTERFACE)
target_include_directories(learning_to_fly INTERFACE include)
# The sensor path (-DL2F_SENSOR_SIM, plan Step 2) needs the flown Kalman estimator, compiled
# from the firmware tree by l2f/estimator (FW=, default ~/crazyflie-firmware-2026.08).
if(CMAKE_CXX_FLAGS MATCHES "L2F_SENSOR_SIM")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../estimator ${CMAKE_BINARY_DIR}/cf_estimator)
    target_link_libraries(learning_to_fly INTERFACE cf_estimator)
endif()

#set(RL_TOOLS_ENABLE_TESTS ON)
add_subdirectory(external/rl_tools)
add_subdirectory(src)
