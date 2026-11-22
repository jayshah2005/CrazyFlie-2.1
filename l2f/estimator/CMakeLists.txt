# The flown Kalman estimator as a library for the simulator (plan Step 2, E1); see cf_estimator.h.
#   cmake -S l2f/estimator -B l2f/estimator/build && cmake --build l2f/estimator/build
cmake_minimum_required(VERSION 3.16)
project(cf_estimator C CXX)

set(FW "$ENV{HOME}/crazyflie-firmware-2026.08" CACHE PATH "crazyflie-firmware 2026.08 (l2f/firmware/build_firmware.sh clones it there)")
set(FW_DEFCONFIG cf21bl_defconfig CACHE STRING "firmware configuration, as build_firmware.sh DEFCONFIG")
if(NOT EXISTS ${FW}/src/modules/src/kalman_core/kalman_core.c)
  message(FATAL_ERROR "no firmware at FW=${FW}: run l2f/firmware/build_firmware.sh once, or pass -DFW=<path>")
endif()

# the firmware's generated autoconf.h for that configuration, made outside the firmware tree
set(FWCONF ${CMAKE_CURRENT_BINARY_DIR}/fwconfig)
if(NOT EXISTS ${FWCONF}/include/generated/autoconf.h)
  file(MAKE_DIRECTORY ${FWCONF})
  execute_process(COMMAND make KBUILD_OUTPUT=${FWCONF} -C ${FW} ${FW_DEFCONFIG} OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE r1)
  execute_process(COMMAND make KBUILD_OUTPUT=${FWCONF} -C ${FW} silentoldconfig OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE r2)
  if(NOT EXISTS ${FWCONF}/include/generated/autoconf.h)
    message(FATAL_ERROR "could not generate autoconf.h for ${FW_DEFCONFIG} (make: ${r1} ${r2})")
  endif()
endif()

set(FW_SOURCES
  vendor/CMSIS/CMSIS/DSP/Source/BasicMathFunctions/arm_add_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/CommonTables/arm_common_tables.c
  vendor/CMSIS/CMSIS/DSP/Source/FastMathFunctions/arm_cos_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/FastMathFunctions/arm_sin_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/StatisticsFunctions/arm_power_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/MatrixFunctions/arm_mat_scale_f32.c
  vendor/CMSIS/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_f32.c
  src/modules/src/kalman_core/kalman_core.c
  src/modules/src/kalman_core/mm_tof.c
  src/modules/src/kalman_core/mm_flow.c
  src/modules/src/kalman_core/mm_sweep_angles.c
  src/modules/src/kalman_core/mm_yaw_error.c
  src/modules/src/kalman_core/mm_position.c
  src/modules/src/kalman_core/mm_pose.c
  src/modules/src/kalman_supervisor.c
  src/modules/src/outlierfilter/outlierFilterLighthouse.c
  src/modules/src/axis3fSubSampler.c
  src/modules/src/lighthouse/lighthouse_position_est.c
  src/utils/src/lighthouse/pulse_processor.c
  src/utils/src/lighthouse/pulse_processor_v1.c
  src/utils/src/lighthouse/pulse_processor_v2.c
  src/utils/src/lighthouse/lighthouse_calibration.c
  src/utils/src/lighthouse/lighthouse_geometry.c
)
list(TRANSFORM FW_SOURCES PREPEND ${FW}/)

add_library(cf_estimator STATIC cf_estimator.c firmware_stubs.c ${FW_SOURCES})
target_include_directories(cf_estimator
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
  ${FW}/src/modules/interface ${FW}/src/modules/interface/estimator ${FW}/src/modules/interface/kalman_core
  ${FW}/src/modules/interface/outlierfilter ${FW}/src/modules/interface/lighthouse
  ${FW}/src/hal/interface ${FW}/src/utils/interface ${FW}/src/utils/interface/lighthouse
  ${FW}/src/config ${FW}/src/drivers/interface ${FW}/src/platform/interface ${FW}/src/deck/interface
  ${FWCONF}/include/generated
  ${FW}/vendor/CMSIS/CMSIS/DSP/Include ${FW}/vendor/CMSIS/CMSIS/Core/Include
)
# as the firmware build (tools/kbuild: -include autoconf.h) and its Python bindings (UNIT_TEST_MODE)
target_compile_options(cf_estimator PRIVATE -O2 -fno-strict-aliasing -Wno-address-of-packed-member
  -include ${FWCONF}/include/generated/autoconf.h)
target_compile_definitions(cf_estimator PRIVATE UNIT_TEST_MODE)
set_target_properties(cf_estimator PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 11)
target_link_libraries(cf_estimator PUBLIC m)

add_executable(check_estimator check_estimator.cpp)
target_link_libraries(check_estimator PRIVATE cf_estimator)

# E1 check 2: replay a sensor-survey log (noise_survey.py) through the library
add_executable(replay_survey replay_survey.cpp)
target_link_libraries(replay_survey PRIVATE cf_estimator)
target_compile_features(replay_survey PRIVATE cxx_std_17)
