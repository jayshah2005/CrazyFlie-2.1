// The flown Kalman estimator for the simulator; see cf_estimator.h. Every function here names
// the firmware function (crazyflie-firmware 2026.08) it reproduces; the estimator mathematics
// is the firmware's own compiled code.
#include "cf_estimator.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "estimator.h"
#include "kalman_core.h"
#include "kalman_supervisor.h"
#include "mm_flow.h"
#include "mm_sweep_angles.h"
#include "mm_pose.h"
#include "mm_position.h"
#include "mm_tof.h"
#include "mm_yaw_error.h"
#include "outlierFilterLighthouse.h"
#include "axis3fSubSampler.h"
#include "physicalConstants.h"
#include "pulse_processor.h"
#include "pulse_processor_v2.h"
#include "lighthouse_calibration.h"
#include "lighthouse_position_est.h"
#include "platform_defaults.h"

#define QUEUE_SIZE 20                    // estimator.c MEASUREMENTS_QUEUE_SIZE
#define PREDICTION_UPDATE_INTERVAL_MS 10 // estimator_kalman.c: 1000 / PREDICT_RATE (100 Hz)
#define IS_FLYING_HYSTERESIS_MS 2000     // supervisor.c IS_FLYING_HYSTERESIS_THRESHOLD
#define RANGE_OUTLIER_LIMIT 5000         // zranger2.c [mm]
#define OULIER_LIMIT 100                 // flowdeck_v1v2.c [px]
#define LH2_EVALUATION_INTERVAL_MS 100   // lighthouse_throttle.c evaluationIntervalMs

struct cfEstimator_s {
  cfEstimatorConfig_t config;
  // estimator_kalman.c module state
  kalmanCoreData_t coreData;  // holds Pm, a pointer into itself: the struct never moves
  kalmanCoreParams_t coreParams;
  Axis3fSubSampler_t accSubSampler;
  Axis3fSubSampler_t gyroSubSampler;
  Axis3f accLatest;
  Axis3f gyroLatest;
  OutlierFilterLhState_t sweepOutlierFilterState;
  uint32_t nextPredictionMs;
  bool resetEstimation;
  state_t taskEstimatorState;
  // estimator.c queue
  measurement_t queue[QUEUE_SIZE];
  int queueHead, queueCount;
  // supervisor.c
  uint32_t latestThrustTick;
  // lighthouse_core.c / pulse processor state
  pulseProcessor_t lh;
  pulseProcessorResult_t angles;
  // lighthouse_throttle.c (function statics there)
  uint32_t previousEvaluationTime, nextEvaluationTime, eventCounter;
  float discardProbability;
  uint32_t rng;
  // flowdeck_v1v2.c
  bool useFlowDisabled;
  // diagnostics
  uint32_t resets, lhUsed, lhOffered, dropped;
  // a firmware ASSERT failed or the state became non-finite (the drone would halt): the
  // estimate stays at the last finite one until the next reset
  bool failed;
  char failure[200];
  cfEstimate_t lastOut;
};

// The instance the firmware's global entry points (estimatorEnqueue, estimatorKalmanGet*) act on.
static cfEstimator_t* current = NULL;
// the measurement being applied (for the failure report)
static const measurement_t* currentMeasurement = NULL;

void cfEstimatorDefaultConfig(cfEstimatorConfig_t* c) {
  c->flow_std = 2.0f;
  c->lh2_max_rate = 50.0f;
  c->tof_min_distance = 0.0f;
  c->tof_low_std = 0.0f;
}

cfEstimator_t* cfEstimatorCreate(const cfEstimatorConfig_t* config) {
  cfEstimator_t* est = (cfEstimator_t*)calloc(1, sizeof(cfEstimator_t));
  if (config) {
    est->config = *config;
  } else {
    cfEstimatorDefaultConfig(&est->config);
  }
  return est;
}

void cfEstimatorDestroy(cfEstimator_t* est) {
  if (current == est) current = NULL;
  free(est);
}

// ---- estimator.c: the measurement queue (xQueueSend with no wait: dropped when full) ----
void estimatorEnqueue(const measurement_t* measurement) {
  cfEstimator_t* est = current;
  if (!est) return;
  if (est->queueCount == QUEUE_SIZE) {
    est->dropped++;
    return;
  }
  est->queue[(est->queueHead + est->queueCount) % QUEUE_SIZE] = *measurement;
  est->queueCount++;
}

bool estimatorDequeue(measurement_t* measurement) {
  cfEstimator_t* est = current;
  if (!est || est->queueCount == 0) return false;
  *measurement = est->queue[est->queueHead];
  est->queueHead = (est->queueHead + 1) % QUEUE_SIZE;
  est->queueCount--;
  return true;
}

// ---- estimator_kalman.c accessors used by lighthouse_position_est.c (estimateYaw) ----
void estimatorKalmanGetEstimatedPos(point_t* pos) {
  pos->x = current->coreData.S[KC_STATE_X];
  pos->y = current->coreData.S[KC_STATE_Y];
  pos->z = current->coreData.S[KC_STATE_Z];
}

void estimatorKalmanGetEstimatedRot(float* rotationMatrix) {
  memcpy(rotationMatrix, current->coreData.R, 9 * sizeof(float));
}

// ---- estimator_kalman.c estimatorKalmanInit (no deck needs attitude reversion off: Lighthouse
// and Flow decks keep the default) ----
static void kalmanInit(cfEstimator_t* est, uint32_t nowMs) {
  axis3fSubSamplerInit(&est->accSubSampler, GRAVITY_MAGNITUDE);
  axis3fSubSamplerInit(&est->gyroSubSampler, DEG_TO_RAD);
  outlierFilterLighthouseReset(&est->sweepOutlierFilterState, 0);
  kalmanCoreInit(&est->coreData, &est->coreParams, nowMs);
}

static void resetCommon(cfEstimator_t* est, uint32_t now_ms, uint32_t seed) {
  current = est;
  kalmanCoreDefaultParams(&est->coreParams);
  est->queueHead = est->queueCount = 0;
  est->nextPredictionMs = now_ms;  // kalmanTask: nextPredictionMs = nowMs at task start
  est->resetEstimation = false;
  est->latestThrustTick = 0;
  memset(&est->angles, 0, sizeof(est->angles));
  est->angles.measurementType = lighthouseBsTypeV2;
  memset(est->lh.receivedBsSweep, 0, sizeof(est->lh.receivedBsSweep));
  est->previousEvaluationTime = est->nextEvaluationTime = est->eventCounter = 0;
  est->discardProbability = 0;
  est->rng = seed ? seed : 1;
  est->resets = est->lhUsed = est->lhOffered = est->dropped = 0;
  est->failed = false;
  est->failure[0] = 0;
  memset(&est->accLatest, 0, sizeof(Axis3f));
  memset(&est->gyroLatest, 0, sizeof(Axis3f));
  memset(&est->taskEstimatorState, 0, sizeof(state_t));
}

void cfEstimatorReset(cfEstimator_t* est, uint32_t now_ms, const float p[3], float yaw, uint32_t seed) {
  resetCommon(est, now_ms, seed);
  est->coreParams.initialX = p[0];
  est->coreParams.initialY = p[1];
  est->coreParams.initialZ = p[2];
  est->coreParams.initialYaw = yaw;
  kalmanInit(est, now_ms);
  memset(&est->lastOut, 0, sizeof(cfEstimate_t));
  for (int i = 0; i < 3; i++) est->lastOut.position[i] = p[i];
  est->lastOut.quat[0] = cosf(yaw / 2);
  est->lastOut.quat[3] = sinf(yaw / 2);
}

void cfEstimatorResetTo(cfEstimator_t* est, uint32_t now_ms, const float p[3], const float v_body[3],
                        const float q[4], const float p_diag[9], uint32_t seed) {
  resetCommon(est, now_ms, seed);
  // first prediction one period after the reset, so it has 10 ms of IMU samples (with none,
  // axis3fSubSamplerFinalize returns the zero vector: zero thrust in the flight model)
  est->nextPredictionMs = now_ms + PREDICTION_UPDATE_INTERVAL_MS;
  est->coreParams.initialX = p[0];
  est->coreParams.initialY = p[1];
  est->coreParams.initialZ = p[2];
  kalmanInit(est, now_ms);
  kalmanCoreData_t* d = &est->coreData;
  for (int i = 0; i < 3; i++) {
    d->S[KC_STATE_X + i] = p[i];
    d->S[KC_STATE_PX + i] = v_body[i];
    d->S[KC_STATE_D0 + i] = 0;
  }
  // q normalized (the simulator's may be off by rounding), R from it with diagonals
  // 1 - 2 (..): mm_tof.c takes acosf(R[2][2]), NaN if R[2][2] exceeds 1 by a rounding error
  const float qn = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (int i = 0; i < 4; i++) d->q[i] = q[i] / qn;
  const float w = d->q[0], x = d->q[1], y = d->q[2], z = d->q[3];
  d->R[0][0] = 1 - 2 * (y * y + z * z);
  d->R[0][1] = 2 * x * y - 2 * w * z;
  d->R[0][2] = 2 * x * z + 2 * w * y;
  d->R[1][0] = 2 * x * y + 2 * w * z;
  d->R[1][1] = 1 - 2 * (x * x + z * z);
  d->R[1][2] = 2 * y * z - 2 * w * x;
  d->R[2][0] = 2 * x * z - 2 * w * y;
  d->R[2][1] = 2 * y * z + 2 * w * x;
  d->R[2][2] = 1 - 2 * (x * x + y * y);
  for (int i = 0; i < KC_STATE_DIM; i++)
    for (int j = 0; j < KC_STATE_DIM; j++) d->P[i][j] = i == j ? p_diag[i] : 0;
  for (int i = 0; i < 3; i++) {
    est->lastOut.position[i] = p[i];
    est->lastOut.var_position[i] = p_diag[i];
  }
  for (int i = 0; i < 4; i++) est->lastOut.quat[i] = d->q[i];
  for (int i = 0; i < 3; i++)  // world velocity R v_body
    est->lastOut.velocity[i] = d->R[i][0] * v_body[0] + d->R[i][1] * v_body[1] + d->R[i][2] * v_body[2];
}

// lighthouse_position_est.c lighthousePositionGeometryDataUpdated / preProcessGeometryData for
// one base station, on this instance's pulse processor (the firmware's is a global).
void cfEstimatorSetBaseStation(cfEstimator_t* est, int bs, const cfBaseStation_t* g) {
  baseStationGeometry_t* geo = &est->lh.bsGeometry[bs];
  for (int i = 0; i < 3; i++) {
    geo->origin[i] = g->origin[i];
    for (int j = 0; j < 3; j++) {
      geo->mat[i][j] = g->rotation[i][j];
      est->lh.bsGeoCache[bs].baseStationInvertedRotationMatrixes[j][i] = g->rotation[i][j];  // transpose
    }
  }
  geo->valid = true;
  lighthouseCalibration_t* cal = &est->lh.bsCalibration[bs];
  for (int s = 0; s < 2; s++) {
    const float* c = g->calibration[s];
    cal->sweep[s].phase = c[0];
    cal->sweep[s].tilt = c[1];
    cal->sweep[s].curve = c[2];
    cal->sweep[s].gibmag = c[3];
    cal->sweep[s].gibphase = c[4];
    cal->sweep[s].ogeemag = c[5];
    cal->sweep[s].ogeephase = c[6];
  }
  cal->uid = 0;
  cal->valid = g->calibration_valid;
}

// ---- sensors: IMU samples as sensors_bmi088 queues them ----
void cfEstimatorImu(cfEstimator_t* est, const float acc_g[3], const float gyro_dps[3]) {
  current = est;
  measurement_t m;
  m.type = MeasurementTypeAcceleration;
  m.data.acceleration.acc.x = acc_g[0];
  m.data.acceleration.acc.y = acc_g[1];
  m.data.acceleration.acc.z = acc_g[2];
  estimatorEnqueue(&m);
  m.type = MeasurementTypeGyroscope;
  m.data.gyroscope.gyro.x = gyro_dps[0];
  m.data.gyroscope.gyro.y = gyro_dps[1];
  m.data.gyroscope.gyro.z = gyro_dps[2];
  estimatorEnqueue(&m);
}

// ---- zranger2.c zRanger2Task + range.c rangeEnqueueDownRangeInEstimator ----
void cfEstimatorRange(cfEstimator_t* est, uint16_t range_last, uint32_t time_ms) {
  current = est;
  static const float expPointA = 2.5f, expStdA = 0.0025f, expPointB = 4.0f, expStdB = 0.2f;
  const float expCoeff = logf(expStdB / expStdA) / (expPointB - expPointA);
  if (range_last < RANGE_OUTLIER_LIMIT) {
    float distance = (float)range_last * 0.001f;
    float stdDev = expStdA * (1.0f + expf(expCoeff * (distance - expPointA)));
    // E4 patch (patch_firmware.py): below tof_min_distance the reading is not accurate (VL53L1X
    // datasheet: from 4 cm), so its sigma is raised, or the reading is skipped
    if (est->config.tof_min_distance > 0 && distance < est->config.tof_min_distance) {
      if (est->config.tof_low_std <= 0) return;
      stdDev = est->config.tof_low_std;
    }
    tofMeasurement_t tofData;
    tofData.timestamp = time_ms;
    tofData.distance = distance;
    tofData.stdDev = stdDev;
    estimatorEnqueueTOF(&tofData);
  }
}

// ---- flowdeck_v1v2.c flowdeckTask (no smoothing, fixed sigma) ----
void cfEstimatorFlow(cfEstimator_t* est, int16_t delta_x, int16_t delta_y, uint8_t motion, float dt) {
  current = est;
  // flip motion information to comply with sensor mounting
  int16_t accpx = -delta_y;
  int16_t accpy = -delta_x;
  if (abs(accpx) < OULIER_LIMIT && abs(accpy) < OULIER_LIMIT) {
    flowMeasurement_t flowData;
    flowData.stdDevX = est->config.flow_std;
    flowData.stdDevY = est->config.flow_std;
    flowData.dt = dt;
    flowData.dpixelx = (float)accpx;
    flowData.dpixely = (float)accpy;
    if (!est->useFlowDisabled && motion == 0xB0) {
      estimatorEnqueueFlow(&flowData);
    }
  }
}

void cfEstimatorSetFlowDisabled(cfEstimator_t* est, bool disabled) { est->useFlowDisabled = disabled; }

// ---- lighthouse_throttle.c throttleLh2Samples, per instance (rand() -> own generator) ----
static uint32_t nextRandom(cfEstimator_t* est) {  // xorshift32, in [0, 2^31)
  uint32_t x = est->rng;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  est->rng = x;
  return x >> 1;
}

static bool throttleLh2Samples(cfEstimator_t* est, const uint32_t nowMs) {
  const uint32_t RANDMAX = 0x7fffffff;
  est->eventCounter++;
  if (nowMs > est->nextEvaluationTime) {
    const float currentRate = 1000.0f * (float)est->eventCounter / (float)(nowMs - est->previousEvaluationTime);
    if (currentRate < est->config.lh2_max_rate) {
      est->discardProbability = 0.0f;
    } else {
      est->discardProbability = 1.0f - est->config.lh2_max_rate / currentRate;
    }
    est->previousEvaluationTime = nowMs;
    est->eventCounter = 0;
    est->nextEvaluationTime = nowMs + LH2_EVALUATION_INTERVAL_MS;
  }
  const int discardThreshold = (int)(RANDMAX * est->discardProbability);
  return (int)nextRandom(est) > discardThreshold;
}

// ---- lighthouse_core.c usePulseResult (second sweep of a V2 rotation, estimation method 1) ----
void cfEstimatorLighthouse(cfEstimator_t* est, int bs, const float angles[4][2], const bool valid[4], uint32_t now_ms) {
  current = est;
  pulseProcessorResult_t* a = &est->angles;
  a->measurementType = lighthouseBsTypeV2;
  for (int s = 0; s < PULSE_PROCESSOR_N_SENSORS; s++) {
    pulseProcessorSensorMeasurement_t* m = &a->baseStationMeasurementsLh2[bs].sensorMeasurements[s];
    m->angles[0] = angles[s][0];
    m->angles[1] = angles[s][1];
    m->validCount = valid[s] ? PULSE_PROCESSOR_N_SWEEPS : 0;
  }
  const bool hasCalibrationData = pulseProcessorApplyCalibration(&est->lh, a, bs);
  if (!hasCalibrationData) return;
  // convertV2AnglesToV1Angles
  for (int s = 0; s < PULSE_PROCESSOR_N_SENSORS; s++) {
    for (int b = 0; b < CONFIG_DECK_LIGHTHOUSE_MAX_N_BS; b++) {
      pulseProcessorSensorMeasurement_t* from = &a->baseStationMeasurementsLh2[b].sensorMeasurements[s];
      pulseProcessorSensorMeasurement_t* to = &a->baseStationMeasurementsLh1[b].sensorMeasurements[s];
      if (2 == from->validCount) {
        pulseProcessorV2ConvertToV1Angles(from->correctedAngles[0], from->correctedAngles[1], to->correctedAngles);
        to->validCount = from->validCount;
      } else {
        to->validCount = 0;
      }
    }
  }
  if (!est->lh.bsGeometry[bs].valid) return;
  est->lhOffered++;
  if (!throttleLh2Samples(est, now_ms)) return;
  est->lhUsed++;
  // usePulseResultSweeps
  pulseProcessorClearOutdated(&est->lh, a, bs);
  lighthousePositionEstimatePoseSweeps(&est->lh, a, bs);
  pulseProcessorProcessed(a, bs);
}

// ---- estimator_kalman.c updateQueuedMeasurements (the measurement types the CF has here) ----
static void updateQueuedMeasurements(cfEstimator_t* est, const uint32_t nowMs) {
  measurement_t m;
  while (estimatorDequeue(&m)) {
    currentMeasurement = &m;
    switch (m.type) {
      case MeasurementTypePosition:
        kalmanCoreUpdateWithPosition(&est->coreData, &m.data.position);
        break;
      case MeasurementTypePose:
        kalmanCoreUpdateWithPose(&est->coreData, &m.data.pose);
        break;
      case MeasurementTypeTOF:
        kalmanCoreUpdateWithTof(&est->coreData, &m.data.tof);
        break;
      case MeasurementTypeFlow:
        kalmanCoreUpdateWithFlow(&est->coreData, &m.data.flow, &est->gyroLatest);
        break;
      case MeasurementTypeYawError:
        kalmanCoreUpdateWithYawError(&est->coreData, &m.data.yawError);
        break;
      case MeasurementTypeSweepAngle:
        kalmanCoreUpdateWithSweepAngles(&est->coreData, &m.data.sweepAngle, nowMs, &est->sweepOutlierFilterState);
        break;
      case MeasurementTypeGyroscope:
        axis3fSubSamplerAccumulate(&est->gyroSubSampler, &m.data.gyroscope.gyro);
        est->gyroLatest = m.data.gyroscope.gyro;
        break;
      case MeasurementTypeAcceleration:
        axis3fSubSamplerAccumulate(&est->accSubSampler, &m.data.acceleration.acc);
        est->accLatest = m.data.acceleration.acc;
        break;
      default:  // TDOA, distance, height, barometer: no such sensor on this drone
        break;
    }
  }
  currentMeasurement = NULL;
}

// ---- estimator_kalman.c kalmanTask, one pass; supervisor.c isFlyingCheck ----
// firmware_stubs.c assertFail: mark the current estimator failed instead of halting
void cfEstimatorAssertFailed(const char* exp, const char* file, int line) {
  static int reported = 0;
  if (current && !current->failed) {
    current->failed = true;
    const kalmanCoreData_t* d = &current->coreData;
    const measurement_t* m = currentMeasurement;
    char what[96] = "no measurement";
    if (m && m->type == MeasurementTypeTOF) snprintf(what, sizeof(what), "range %.4f m sd %.4f", m->data.tof.distance, m->data.tof.stdDev);
    else if (m && m->type == MeasurementTypeFlow) snprintf(what, sizeof(what), "flow %.1f %.1f px dt %.4f", m->data.flow.dpixelx, m->data.flow.dpixely, m->data.flow.dt);
    else if (m && m->type == MeasurementTypeSweepAngle) snprintf(what, sizeof(what), "sweep bs %d sensor %d angle %.4f", m->data.sweepAngle.baseStationId, m->data.sweepAngle.sensorId, m->data.sweepAngle.measuredSweepAngle);
    else if (m) snprintf(what, sizeof(what), "measurement type %d", (int)m->type);
    snprintf(current->failure, sizeof(current->failure), "firmware assert failed: %s (%s:%d) during %s; z %.4f R22 %.9f Pzz %.3g",
             exp, file, line, what, d->S[KC_STATE_Z], d->R[2][2], d->P[KC_STATE_Z][KC_STATE_Z]);
    if (reported++ < 5) fprintf(stderr, "cf_estimator: %s; episode ends\n", current->failure);
  }
}

bool cfEstimatorFailed(const cfEstimator_t* est, const char** why) {
  if (why) *why = est->failure;
  return est->failed;
}

void cfEstimatorTick(cfEstimator_t* est, uint32_t nowMs, bool thrust_over_idle, cfEstimate_t* out) {
  current = est;
  if (est->failed) {
    if (out) *out = est->lastOut;
    return;
  }
  if (est->resetEstimation) {
    kalmanInit(est, nowMs);
    est->resetEstimation = false;
  }
  if (thrust_over_idle) est->latestThrustTick = nowMs;
  const bool quadIsFlying = est->latestThrustTick != 0 && (nowMs - est->latestThrustTick) < IS_FLYING_HYSTERESIS_MS;

  if (nowMs >= est->nextPredictionMs) {
    axis3fSubSamplerFinalize(&est->accSubSampler);
    axis3fSubSamplerFinalize(&est->gyroSubSampler);
    kalmanCorePredict(&est->coreData, &est->coreParams, &est->accSubSampler.subSample, &est->gyroSubSampler.subSample,
                      nowMs, quadIsFlying);
    est->nextPredictionMs = nowMs + PREDICTION_UPDATE_INTERVAL_MS;
  }
  kalmanCoreAddProcessNoise(&est->coreData, &est->coreParams, nowMs);
  updateQueuedMeasurements(est, nowMs);
  kalmanCoreFinalize(&est->coreData);
  if (!kalmanSupervisorIsStateWithinBounds(&est->coreData)) {
    est->resetEstimation = true;
    est->resets++;
  }
  kalmanCoreExternalizeState(&est->coreData, &est->taskEstimatorState, &est->accLatest);
  bool finite = true;
  for (int i = 0; i < KC_STATE_DIM; i++) finite = finite && isfinite(est->coreData.S[i]) && isfinite(est->coreData.P[i][i]);
  for (int i = 0; i < 4; i++) finite = finite && isfinite(est->coreData.q[i]);
  if (!finite || est->failed) {
    if (!est->failed) {
      est->failed = true;
      snprintf(est->failure, sizeof(est->failure), "non-finite state at %u ms", (unsigned)nowMs);
    }
    if (out) *out = est->lastOut;
    return;
  }
  if (out) {
    const state_t* s = &est->taskEstimatorState;
    out->position[0] = s->position.x;
    out->position[1] = s->position.y;
    out->position[2] = s->position.z;
    out->velocity[0] = s->velocity.x;
    out->velocity[1] = s->velocity.y;
    out->velocity[2] = s->velocity.z;
    out->quat[0] = s->attitudeQuaternion.w;
    out->quat[1] = s->attitudeQuaternion.x;
    out->quat[2] = s->attitudeQuaternion.y;
    out->quat[3] = s->attitudeQuaternion.z;
    for (int i = 0; i < 3; i++) out->var_position[i] = est->coreData.P[KC_STATE_X + i][KC_STATE_X + i];
    out->attitude_deg[0] = s->attitude.roll;
    out->attitude_deg[1] = s->attitude.pitch;
    out->attitude_deg[2] = s->attitude.yaw;
    est->lastOut = *out;
  }
}

// lighthouse_position_est.c sensorDeckPositions
#define SENSOR_POS_W (0.015f / 2.0f)
#define SENSOR_POS_L (0.030f / 2.0f)
static const float deckSensors[4][3] = {
    {-SENSOR_POS_L, SENSOR_POS_W, 0.0f},
    {-SENSOR_POS_L, -SENSOR_POS_W, 0.0f},
    {SENSOR_POS_L, SENSOR_POS_W, 0.0f},
    {SENSOR_POS_L, -SENSOR_POS_W, 0.0f},
};

void cfLighthouseModel(const cfBaseStation_t* bs, const float p[3], const float R[3][3], float angles[4][2],
                       bool visible[4]) {
  const float t30 = (float)M_PI / 6;  // lighthouse_position_est.c: sweep 0 at -30 deg, sweep 1 at +30 deg
  lighthouseCalibrationSweep_t calib[2];
  for (int k = 0; k < 2; k++) {
    const float* c = bs->calibration[k];
    calib[k].phase = c[0];
    calib[k].tilt = c[1];
    calib[k].curve = c[2];
    calib[k].gibmag = c[3];
    calib[k].gibphase = c[4];
    calib[k].ogeemag = c[5];
    calib[k].ogeephase = c[6];
  }
  for (int s = 0; s < 4; s++) {
    float w[3], d[3], r[3];
    for (int i = 0; i < 3; i++)
      w[i] = p[i] + R[i][0] * deckSensors[s][0] + R[i][1] * deckSensors[s][1] + R[i][2] * deckSensors[s][2];
    for (int i = 0; i < 3; i++) d[i] = w[i] - bs->origin[i];
    for (int i = 0; i < 3; i++)  // rotor frame: R_bs^T (sensor - origin)
      r[i] = bs->rotation[0][i] * d[0] + bs->rotation[1][i] * d[1] + bs->rotation[2][i] * d[2];
    // the deck's normal (body z) must point towards the base station
    const float facing = -(R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]);
    const float az = atan2f(r[1], r[0]), el = atan2f(r[2], sqrtf(r[0] * r[0] + r[1] * r[1]));
    visible[s] = r[0] > 0 && facing > 0 && fabsf(az) < 75.0f * (float)M_PI / 180 && fabsf(el) < 55.0f * (float)M_PI / 180;
    angles[s][0] = lighthouseCalibrationMeasurementModelLh2(r[0], r[1], r[2], -t30, &calib[0]);
    angles[s][1] = lighthouseCalibrationMeasurementModelLh2(r[0], r[1], r[2], t30, &calib[1]);
  }
}

void cfFlowdeckPosition(float pos[3]) {
  pos[0] = FLOWDECK_POS_X;
  pos[1] = FLOWDECK_POS_Y;
  pos[2] = FLOWDECK_POS_Z;
}

void cfEstimatorCounters(const cfEstimator_t* est, uint32_t* resets, uint32_t* lh_used, uint32_t* lh_offered,
                         uint32_t* dropped) {
  if (resets) *resets = est->resets;
  if (lh_used) *lh_used = est->lhUsed;
  if (lh_offered) *lh_offered = est->lhOffered;
  if (dropped) *dropped = est->dropped;
}
