// Checks of the estimator library (plan E1) that need no measured data. Ideal sensors, made
// with the filter's own measurement models (so any error comes from the loop, not the models):
//   1 converge   filter reset at the origin, drone hovering elsewhere with 2 base stations,
//                range and flow: the estimate converges to the truth; about half the LH2
//                samples pass the throttle (2 base stations x 50 Hz against 50 / s)
//   2 track      filter started converged, drone flying a 0.5 m circle at 0.8 m/s (twice the
//                flown circle's speed), tilted into the turn as a quadrotor must be (in flight
//                the filter uses only the accelerometer's thrust axis): the estimate follows it.
//                Also printed at 1.6 m/s: the error grows with the dynamics (100 Hz first-order
//                prediction, LH2 samples 20 ms apart after the throttle), as on the drone
//   3 instances  two filters interleaved tick by tick give exactly what each gives alone (no
//                state shared through the firmware's globals)
//   4 low flow   flow and range only, 5 cm above the floor: the filter's speed is v * 0.1 / h,
//                the firmware's 0.1 m height clamp (why the plan turns the flow off below 10 cm)
// Usage: check_estimator   (exit 1 if a check fails)
#include "cf_estimator.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>

namespace {
constexpr float G = 9.81f;
constexpr float PI = 3.14159265f;

// A base station at `origin` looking at `target` (rotor x towards the target, z up).
cfBaseStation_t base_station(float ox, float oy, float oz, float tx, float ty, float tz) {
  cfBaseStation_t b{};
  float f[3] = {tx - ox, ty - oy, tz - oz};
  float n = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (float& v : f) v /= n;
  float y[3] = {-f[1], f[0], 0};  // world z x f
  n = std::sqrt(y[0] * y[0] + y[1] * y[1]);
  for (float& v : y) v /= n;
  float z[3] = {f[1] * y[2] - f[2] * y[1], f[2] * y[0] - f[0] * y[2], f[0] * y[1] - f[1] * y[0]};
  float o[3] = {ox, oy, oz};
  for (int i = 0; i < 3; i++) {
    b.origin[i] = o[i];
    b.rotation[i][0] = f[i];
    b.rotation[i][1] = y[i];
    b.rotation[i][2] = z[i];
  }
  b.calibration_valid = true;  // all-zero calibration: an ideal base station
  return b;
}

struct Truth {
  float p[3], v[3], a[3];  // world
  float yaw;               // level attitude
};

struct Scenario {
  bool lighthouse = true, range = true, flow = true;
  Truth (*truth)(float t);
};

// Attitude of a quadrotor producing acceleration a with heading yaw: body z along a + g z.
void rotation(const Truth& x, float R[3][3]) {
  float zb[3] = {x.a[0], x.a[1], x.a[2] + G};
  float n = std::sqrt(zb[0] * zb[0] + zb[1] * zb[1] + zb[2] * zb[2]);
  for (float& v : zb) v /= n;
  const float xc[3] = {std::cos(x.yaw), std::sin(x.yaw), 0};
  float yb[3] = {zb[1] * xc[2] - zb[2] * xc[1], zb[2] * xc[0] - zb[0] * xc[2], zb[0] * xc[1] - zb[1] * xc[0]};
  n = std::sqrt(yb[0] * yb[0] + yb[1] * yb[1] + yb[2] * yb[2]);
  for (float& v : yb) v /= n;
  const float xb[3] = {yb[1] * zb[2] - yb[2] * zb[1], yb[2] * zb[0] - yb[0] * zb[2], yb[0] * zb[1] - yb[1] * zb[0]};
  for (int i = 0; i < 3; i++) {
    R[i][0] = xb[i];
    R[i][1] = yb[i];
    R[i][2] = zb[i];
  }
}

// Body angular velocity from the attitude's change: omega^ = R^T dR/dt (central difference).
void angular_velocity(Truth (*truth)(float), float t, float w[3]) {
  const float h = 1e-4f;
  float R0[3][3], R1[3][3], R[3][3];
  rotation(truth(t - h), R0);
  rotation(truth(t + h), R1);
  rotation(truth(t), R);
  float D[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      D[i][j] = 0;
      for (int k = 0; k < 3; k++) D[i][j] += R[k][i] * (R1[k][j] - R0[k][j]) / (2 * h);
    }
  w[0] = 0.5f * (D[2][1] - D[1][2]);
  w[1] = 0.5f * (D[0][2] - D[2][0]);
  w[2] = 0.5f * (D[1][0] - D[0][1]);
}

// Feed one millisecond of ideal measurements for time t_ms, then run one tick.
void step(cfEstimator_t* est, const Scenario& sc, const std::vector<cfBaseStation_t>& bs, uint32_t t_ms, cfEstimate_t* out) {
  const float t = t_ms * 1e-3f;
  Truth x = sc.truth(t);
  float R[3][3], w[3];
  rotation(x, R);
  angular_velocity(sc.truth, t, w);
  if (t_ms % 10 == 9) {  // the simulator's 10 ms IMU sample, one tick before the prediction
    float aw[3] = {x.a[0], x.a[1], x.a[2] + G}, acc[3], gyro[3];
    for (int i = 0; i < 3; i++) acc[i] = (R[0][i] * aw[0] + R[1][i] * aw[1] + R[2][i] * aw[2]) / G;
    for (int i = 0; i < 3; i++) gyro[i] = w[i] * 180 / PI;
    cfEstimatorImu(est, acc, gyro);
  }
  if (sc.range && t_ms % 25 == 0) {  // mm_tof.c: closest point of the 15 deg cone, z / cos(max(0, tilt - 7.5 deg))
    const float cone = std::max(0.0f, std::acos(R[2][2]) - 7.5f * PI / 180);
    cfEstimatorRange(est, (uint16_t)std::lround(x.p[2] / std::cos(cone) * 1000), t_ms);
  }
  if (sc.flow && t_ms % 10 == 5) {
    // inverse of mm_flow.c: dpixel = 10 dt Npix / thetapix * (v_cam * R22 / h -+ omega), v_cam with the lever arm
    float fp[3];
    cfFlowdeckPosition(fp);
    float vb[3];
    for (int i = 0; i < 3; i++) vb[i] = R[0][i] * x.v[0] + R[1][i] * x.v[1] + R[2][i] * x.v[2];
    const float vcx = vb[0] + w[1] * fp[2] - w[2] * fp[1], vcy = vb[1] + w[2] * fp[0] - w[0] * fp[2];
    const float k = 10.0f * 0.01f * 35.0f / 0.71674f;
    const float npx = k * (vcx * R[2][2] / x.p[2] - w[1]), npy = k * (vcy * R[2][2] / x.p[2] + w[0]);
    cfEstimatorFlow(est, (int16_t)std::lround(-npy), (int16_t)std::lround(-npx), 0xB0, 0.01f);
  }
  if (sc.lighthouse) {
    for (size_t b = 0; b < bs.size(); b++) {
      if (t_ms % 20 == 10 * b) {  // one rotation every 20 ms per base station, interleaved
        float ang[4][2];
        bool vis[4];
        cfLighthouseModel(&bs[b], x.p, R, ang, vis);
        cfEstimatorLighthouse(est, (int)b, ang, vis, t_ms);
      }
    }
  }
  cfEstimatorTick(est, t_ms, true, out);
}

Truth hover(float) { return {{0.3f, -0.2f, 0.75f}, {0, 0, 0}, {0, 0, 0}, 20 * PI / 180}; }
float circle_w = 1.6f;  // [rad/s] on the 0.5 m circle
Truth circle(float t) {
  const float r = 0.5f, w = circle_w;
  return {{r * std::cos(w * t), r * std::sin(w * t), 0.75f},
          {-r * w * std::sin(w * t), r * w * std::cos(w * t), 0},
          {-r * w * w * std::cos(w * t), -r * w * w * std::sin(w * t), 0},
          0.3f};
}
Truth low_slide(float) { return {{0, 0, 0.05f}, {0.2f, 0, 0}, {0, 0, 0}, 0}; }

float yaw_of(const cfEstimate_t& e) {
  return std::atan2(2 * (e.quat[0] * e.quat[3] + e.quat[1] * e.quat[2]), 1 - 2 * (e.quat[2] * e.quat[2] + e.quat[3] * e.quat[3]));
}

void start_converged(cfEstimator_t* est, Truth x, uint32_t now) {
  float R[3][3];
  rotation(x, R);
  float vb[3];
  for (int i = 0; i < 3; i++) vb[i] = R[0][i] * x.v[0] + R[1][i] * x.v[1] + R[2][i] * x.v[2];
  // quaternion of R (w largest here: tilts are small)
  const float qw = 0.5f * std::sqrt(1 + R[0][0] + R[1][1] + R[2][2]);
  const float q[4] = {qw, (R[2][1] - R[1][2]) / (4 * qw), (R[0][2] - R[2][0]) / (4 * qw), (R[1][0] - R[0][1]) / (4 * qw)};
  const float p_diag[9] = {1e-6f, 1e-6f, 1e-6f, 1e-4f, 1e-4f, 1e-4f, 1e-5f, 1e-5f, 1e-5f};
  cfEstimatorResetTo(est, now, x.p, vb, q, p_diag, 7);
}
}  // namespace

int main() {
  bool ok = true;
  std::vector<cfBaseStation_t> bs = {base_station(-1.8f, -1.8f, 2.4f, 0, 0, 0.5f), base_station(1.8f, 1.8f, 2.4f, 0, 0, 0.5f)};
  auto make = [&]() {
    cfEstimator_t* e = cfEstimatorCreate(nullptr);
    for (size_t b = 0; b < bs.size(); b++) cfEstimatorSetBaseStation(e, (int)b, &bs[b]);
    return e;
  };

  {  // 1 converge
    cfEstimator_t* e = make();
    const float zero[3] = {0, 0, 0};
    cfEstimatorReset(e, 1, zero, 0, 7);
    Scenario sc;
    sc.truth = hover;
    cfEstimate_t out{};
    for (uint32_t t = 1; t <= 10000; t++) step(e, sc, bs, t, &out);
    Truth x = hover(10);
    float err = std::sqrt(std::pow(out.position[0] - x.p[0], 2) + std::pow(out.position[1] - x.p[1], 2) + std::pow(out.position[2] - x.p[2], 2));
    float verr = std::sqrt(out.velocity[0] * out.velocity[0] + out.velocity[1] * out.velocity[1] + out.velocity[2] * out.velocity[2]);
    float yerr = std::fabs(yaw_of(out) - x.yaw) * 180 / PI;
    uint32_t resets, used, offered, dropped;
    cfEstimatorCounters(e, &resets, &used, &offered, &dropped);
    bool pass = err < 0.005f && verr < 0.01f && yerr < 2 && resets == 0 && dropped == 0 && used > 0.4 * offered && used < 0.6 * offered;
    std::printf("1 converge   after 10 s: position error %.1f mm, speed %.1f mm/s, yaw error %.2f deg, sigma_xyz %.1f/%.1f/%.1f mm; "
                "LH2 samples used %u of %u (%.0f%%), resets %u, queue drops %u: %s\n",
                err * 1e3, verr * 1e3, yerr, std::sqrt(out.var_position[0]) * 1e3, std::sqrt(out.var_position[1]) * 1e3,
                std::sqrt(out.var_position[2]) * 1e3, used, offered, 100.0 * used / offered, resets, dropped, pass ? "ok" : "FAIL");
    ok &= pass;
    cfEstimatorDestroy(e);
  }
  for (float w : {1.6f, 3.2f}) {  // 2 track
    circle_w = w;
    cfEstimator_t* e = make();
    start_converged(e, circle(0), 0);
    Scenario sc;
    sc.truth = circle;
    cfEstimate_t out{};
    double se = 0, sv = 0, emax = 0;
    int n = 0;
    for (uint32_t t = 1; t <= 10000; t++) {
      step(e, sc, bs, t, &out);
      Truth x = circle(t * 1e-3f);
      double err2 = 0, verr2 = 0;
      for (int i = 0; i < 3; i++) {
        err2 += std::pow(out.position[i] - x.p[i], 2);
        verr2 += std::pow(out.velocity[i] - x.v[i], 2);
      }
      se += err2;
      sv += verr2;
      emax = std::max(emax, std::sqrt(err2));
      n++;
    }
    double rms = std::sqrt(se / n), vrms = std::sqrt(sv / n);
    const bool graded = w == 1.6f;
    bool pass = rms < 0.006 && emax < 0.012 && vrms < 0.015;
    std::printf("2 track      0.5 m circle at %.1f m/s, 10 s: position error RMS %.2f mm (max %.2f), velocity error RMS %.1f mm/s: %s\n",
                0.5 * w, rms * 1e3, emax * 1e3, vrms * 1e3, graded ? (pass ? "ok" : "FAIL") : "(information)");
    if (graded) ok &= pass;
    cfEstimatorDestroy(e);
  }
  {  // 3 instances
    Scenario a, b;
    a.truth = hover;
    b.truth = circle;
    circle_w = 3.2f;
    std::vector<cfEstimate_t> alone_a, alone_b;
    cfEstimate_t out{};
    const float zero[3] = {0, 0, 0};
    cfEstimator_t* e = make();
    cfEstimatorReset(e, 1, zero, 0, 3);
    for (uint32_t t = 1; t <= 3000; t++) { step(e, a, bs, t, &out); alone_a.push_back(out); }
    cfEstimatorDestroy(e);
    e = make();
    start_converged(e, circle(0), 0);
    for (uint32_t t = 1; t <= 3000; t++) { step(e, b, bs, t, &out); alone_b.push_back(out); }
    cfEstimatorDestroy(e);
    cfEstimator_t* ea = make();
    cfEstimator_t* eb = make();
    cfEstimatorReset(ea, 1, zero, 0, 3);
    start_converged(eb, circle(0), 0);
    size_t mismatches = 0;
    for (uint32_t t = 1; t <= 3000; t++) {
      cfEstimate_t oa{}, ob{};
      step(ea, a, bs, t, &oa);
      step(eb, b, bs, t, &ob);
      mismatches += std::memcmp(&oa, &alone_a[t - 1], sizeof(oa)) != 0;
      mismatches += std::memcmp(&ob, &alone_b[t - 1], sizeof(ob)) != 0;
    }
    std::printf("3 instances  two filters interleaved for 3 s vs each alone: %zu differing ticks: %s\n", mismatches,
                mismatches == 0 ? "ok" : "FAIL");
    ok &= mismatches == 0;
    cfEstimatorDestroy(ea);
    cfEstimatorDestroy(eb);
  }
  {  // 4 low flow
    cfEstimator_t* e = make();
    start_converged(e, low_slide(0), 0);
    Scenario sc;
    sc.truth = low_slide;
    sc.lighthouse = false;
    cfEstimate_t out{};
    double vx = 0;
    int n = 0;
    for (uint32_t t = 1; t <= 3000; t++) {
      step(e, sc, bs, t, &out);
      if (t > 2000) { vx += out.velocity[0]; n++; }
    }
    double ratio = vx / n / 0.2;
    bool pass = std::fabs(ratio - 0.1 / 0.05) < 0.15;
    std::printf("4 low flow   flow + range only at 5 cm, true speed 0.2 m/s: estimated %.3f m/s, ratio %.2f (0.1 m clamp predicts %.2f): %s\n",
                vx / n, ratio, 0.1 / 0.05, pass ? "ok" : "FAIL");
    ok &= pass;
    cfEstimatorDestroy(e);
  }
  std::printf(ok ? "all checks passed\n" : "checks FAILED\n");
  return ok ? 0 : 1;
}
