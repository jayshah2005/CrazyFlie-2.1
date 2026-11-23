// The Crazyflie's Kalman estimator, as flown, for the simulator (plan Step 2, E1).
//
// The estimator itself is the firmware's code, compiled unchanged from crazyflie-firmware
// 2026.08 (the tree l2f/firmware/build_firmware.sh flashes): kalman_core.c, mm_tof.c, mm_flow.c,
// mm_sweep_angles.c, mm_yaw_error.c, kalman_supervisor.c, outlierFilterLighthouse.c,
// axis3fSubSampler.c, the Lighthouse preprocessing (lighthouse_position_est.c, pulse_processor*.c,
// lighthouse_calibration.c, lighthouse_geometry.c) and their CMSIS-DSP functions.
//
// cf_estimator.c re-implements only the parts that are tied to FreeRTOS tasks or keep global
// state, line for line from the firmware and without changing what they compute:
//   estimator_kalman.c  kalmanTask (one call of cfEstimatorTick = one pass of the task loop,
//                       1 kHz on the drone), updateQueuedMeasurements, estimatorKalmanInit
//   estimator.c         the measurement queue (estimatorEnqueue / estimatorDequeue)
//   supervisor.c        the "is flying" flag: thrust over idle within the last 2 s
//   lighthouse_core.c   usePulseResult for V2 base stations (calibration, V1 angles, throttle)
//   lighthouse_throttle.c  at most lighthouse.lh2maxRate (50) LH2 samples / s, random discard
//   zranger2.c          range measurement: kept if < 5 m, sigma from the distance
//   flowdeck_v1v2.c     flow measurement: outlier limit 100 px, motion bit, motion.disable
// Each cfEstimator_t holds the state the firmware keeps in globals, so several simulated
// drones can run side by side.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cfEstimator_s cfEstimator_t;

// What the stabilizer gets from the estimator (state_t fields the controller reads).
typedef struct {
  float position[3];  // world frame [m]
  float velocity[3];  // world frame [m/s]
  float quat[4];      // attitude quaternion w, x, y, z
  float var_position[3];  // the filter's own position variance (P diagonal) [m^2]
  float attitude_deg[3];  // roll, pitch, yaw as the firmware logs them (stateEstimate.*, legacy pitch sign)
} cfEstimate_t;

// One Lighthouse V2 base station as the drone has it stored (geometry and calibration,
// which may differ from the true ones: that difference is the geometry error).
typedef struct {
  float origin[3];
  float rotation[3][3];  // rotation matrix, base station to world
  // lighthouseCalibrationSweep_t per sweep: phase, tilt, curve, gibmag, gibphase, ogeemag, ogeephase
  float calibration[2][7];
  bool calibration_valid;
} cfBaseStation_t;

// Firmware tuning that the simulator may change (defaults = firmware defaults).
typedef struct {
  float flow_std;          // flowdeck flowStdFixed [px] (2.0)
  float lh2_max_rate;      // lighthouse.lh2maxRate [samples / s] (50)
  float tof_min_distance;  // E4 patch: below this sensor distance [m] the range reading's sigma is
                           // tof_low_std (0 = firmware as released: no special case)
  float tof_low_std;       // [m]; a value <= 0 skips such readings instead
} cfEstimatorConfig_t;

void cfEstimatorDefaultConfig(cfEstimatorConfig_t* config);
cfEstimator_t* cfEstimatorCreate(const cfEstimatorConfig_t* config);
void cfEstimatorDestroy(cfEstimator_t* est);

// estimatorKalmanInit at time now_ms with the kalman.initialX/Y/Z/Yaw parameters. seed: the
// random discard of the LH2 throttle (the firmware uses rand()).
void cfEstimatorReset(cfEstimator_t* est, uint32_t now_ms, const float initial_position[3], float initial_yaw,
                      uint32_t seed);
// A converged filter at a known state: estimatorKalmanInit, then the state set to it and the
// covariance to the diagonal p_diag (KC_STATE_DIM = 9: x y z px py pz d0 d1 d2; velocities
// px..pz in the body frame, as the filter keeps them). The first prediction is at
// now_ms + 10: feed the IMU before it.
void cfEstimatorResetTo(cfEstimator_t* est, uint32_t now_ms, const float position[3], const float velocity_body[3],
                        const float quat[4], const float p_diag[9], uint32_t seed);

void cfEstimatorSetBaseStation(cfEstimator_t* est, int bs, const cfBaseStation_t* geometry);

// Measurements, queued as the firmware's tasks queue them; processed by the next tick.
// IMU sample (sensors task): accelerometer [g], gyroscope [deg/s], body frame.
void cfEstimatorImu(cfEstimator_t* est, const float acc_g[3], const float gyro_dps[3]);
// zranger2 reading [mm] (the sensor's integer output) at tick time time_ms.
void cfEstimatorRange(cfEstimator_t* est, uint16_t range_mm, uint32_t time_ms);
// PMW3901 motion burst: deltaX, deltaY (sensor frame, as read), motion byte (0xB0 = motion),
// dt since the previous reading [s].
void cfEstimatorFlow(cfEstimator_t* est, int16_t delta_x, int16_t delta_y, uint8_t motion, float dt);
// The motion.disable parameter (flow not used by the filter).
void cfEstimatorSetFlowDisabled(cfEstimator_t* est, bool disabled);
// One complete LH2 rotation of base station bs (lighthouse_core: second sweep received):
// raw angles [rad] per sensor (4) and sweep (2); valid[s] = both sweeps seen by sensor s.
void cfEstimatorLighthouse(cfEstimator_t* est, int bs, const float angles[4][2], const bool valid[4], uint32_t now_ms);

// One pass of the Kalman task loop at time now_ms. thrust_over_idle: any motor above idle
// thrust (supervisor.c isFlyingCheck input). Writes the externalized state.
void cfEstimatorTick(cfEstimator_t* est, uint32_t now_ms, bool thrust_over_idle, cfEstimate_t* out);

// Forward model for the simulator: the raw angles base station bs produces for the deck's 4
// sensors (lighthouse_position_est.c sensorDeckPositions) with the drone at position p and
// rotation R (body to world), from lighthouseCalibrationMeasurementModelLh2 in the rotor frame,
// i.e. the angle mm_sweep_angles.c predicts. Pass the true geometry and calibration here; the
// filter gets the stored ones (cfEstimatorSetBaseStation). visible[s]: the sensor faces the
// base station and is inside its field of view (150 x 110 degrees).
void cfLighthouseModel(const cfBaseStation_t* bs, const float p[3], const float R[3][3], float angles[4][2],
                       bool visible[4]);
// The Flow deck camera position in the body frame (platform FLOWDECK_POS_X/Y/Z) [m].
void cfFlowdeckPosition(float pos[3]);

// True once a firmware ASSERT failed or the filter state became non-finite (on the drone the
// assert halts the system): ticks then keep returning the last finite estimate until a reset.
// why: the first failure's description.
bool cfEstimatorFailed(const cfEstimator_t* est, const char** why);
// firmware_stubs.c's assertFail calls this for the estimator that is running.
void cfEstimatorAssertFailed(const char* exp, const char* file, int line);

// Diagnostics: number of resets by the bounds supervisor, of LH2 samples used / offered, of
// queued measurements dropped because the queue was full.
void cfEstimatorCounters(const cfEstimator_t* est, uint32_t* resets, uint32_t* lh_used, uint32_t* lh_offered,
                         uint32_t* dropped);

#ifdef __cplusplus
}
#endif
