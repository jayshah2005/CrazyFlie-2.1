// Sensor path of the simulator (L2F_SENSOR_SIM; plan Step 2, E2 + E3): the drone's sensors are
// simulated from the true motion and fed to the flown Kalman estimator (l2f/estimator, the
// firmware's own code), and the actor observes the estimate, as on the drone.
//
// Per 10 ms simulation step (start state 0, end state 1, true world frame):
//   IMU (filter)  one sample: the step's average, gyro (w0 + w1) / 2 + b_g + n, specific force
//                 R_mid^T ((v1 - v0) / dt + g z) + b_a + n; gyro clipped at +-gyro_clip
//   gyro (actor)  w1 + b_g + n_sample, clipped: the filtered sample rl_tools_controller.c reads
//   range         every 25 ms: (z_sensor + rest_gap) / cos(max(0, tilt - cone)) + bias + n, in mm
//                 (below min_distance: below_min bias and noise)
//   flow          every ~10 ms: raw counts 10 dt Npix / thetapix (v_cam R22 / h - w) + n, rounded,
//                 the inverse of mm_flow.c with the true camera height (gain out_of_spec_gain when
//                 h < min_height or |v / h| > max_rate)
//   Lighthouse    one V2 rotation per base station at its own rate and phase: raw angles from the
//                 firmware's measurement model with the true geometry (stored + geometry error),
//                 + angle noise; drop-outs as a random on/off process
// Event times inside the step use the state interpolated in time (positions linear, attitude
// nlerp). The estimator runs 10 ticks (1 kHz) per step; each event goes to the tick it falls in.
// The 100 Hz prediction runs on the step's last tick with that step's IMU average, so the
// estimate the actor reads is predicted to the step's end; the unknown phase between the
// drone's prediction and its controller is left to the 0-12 ms delay randomization.
// Parameters: config/sensors.yaml (path L2F_SENSORS), [lo, hi] values drawn per episode.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cf_estimator.h"

namespace learning_to_fly::sensor_sim {

struct Interval {
  double lo = 0, hi = 0;
};

// config/sensors.yaml: "section:" lines and indented "key: value" / "key: [lo, hi]" lines.
struct Config {
  std::map<std::string, Interval> values;
  std::map<std::string, std::string> strings;
  std::vector<cfBaseStation_t> stations;  // stored geometry and calibration (lighthouse.system)
  std::string path;

  Interval get(const std::string& key) const {
    auto it = values.find(key);
    if (it == values.end()) throw std::runtime_error("sensors config " + path + ": missing " + key);
    return it->second;
  }
  double fixed(const std::string& key) const { return get(key).lo; }

  static std::string strip(const std::string& s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
  }

  void read(const std::string& file) {
    path = file;
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot read sensors config " + file + " (set L2F_SENSORS)");
    std::string line, section;
    while (std::getline(in, line)) {
      size_t hash = line.find('#');
      if (hash != std::string::npos) line = line.substr(0, hash);
      if (strip(line).empty()) continue;
      const bool indented = line[0] == ' ';
      const std::string s = strip(line);
      const size_t colon = s.find(':');
      if (colon == std::string::npos) continue;
      const std::string key = strip(s.substr(0, colon)), value = strip(s.substr(colon + 1));
      if (!indented) {
        section = key;
        continue;
      }
      const std::string full = section + "." + key;
      if (!value.empty() && value[0] == '[') {
        Interval r;
        if (std::sscanf(value.c_str(), "[%lf , %lf]", &r.lo, &r.hi) != 2)
          throw std::runtime_error("sensors config: bad interval for " + full);
        values[full] = r;
      } else {
        char* end = nullptr;
        double v = std::strtod(value.c_str(), &end);
        if (end && *end == 0 && !value.empty()) {
          values[full] = {v, v};
        } else {
          strings[full] = value;
        }
      }
    }
  }

  // cfclient Lighthouse system file (calibs: <bs>: sweeps: - curve: ..., geos: <bs>: origin:, rotation:)
  void read_lighthouse(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot read Lighthouse system file " + file);
    std::map<int, cfBaseStation_t> bs;
    std::string line, top, sub;
    int id = -1, sweep = -1, row = -1, col = 0;
    static const char* calib_keys[7] = {"phase", "tilt", "curve", "gibmag", "gibphase", "ogeemag", "ogeephase"};
    while (std::getline(in, line)) {
      if (strip(line).empty()) continue;
      size_t indent = line.find_first_not_of(' ');
      std::string s = strip(line);
      if (indent == 0) {
        top = s.substr(0, s.find(':'));
        continue;
      }
      if (indent == 2 && s.back() == ':') {
        id = std::atoi(s.c_str());
        bs[id];  // value-initialized: all zero
        sweep = -1;
        row = -1;
        continue;
      }
      if (indent == 4 && s.back() == ':') {
        sub = s.substr(0, s.size() - 1);
        sweep = -1;
        row = -1;
        continue;
      }
      if (id < 0) continue;
      cfBaseStation_t& b = bs[id];
      if (top == "calibs" && sub == "sweeps") {
        if (s.rfind("- ", 0) == 0) {
          sweep++;
          s = strip(s.substr(2));
        }
        size_t c = s.find(':');
        if (c == std::string::npos || sweep < 0 || sweep > 1) continue;
        std::string k = strip(s.substr(0, c));
        double v = std::atof(s.substr(c + 1).c_str());
        for (int i = 0; i < 7; i++)
          if (k == calib_keys[i]) b.calibration[sweep][i] = (float)v;
        b.calibration_valid = true;
      } else if (top == "geos" && sub == "rotation") {
        if (s.rfind("- - ", 0) == 0) {
          row++;
          col = 0;
          b.rotation[row][col++] = (float)std::atof(s.substr(4).c_str());
        } else if (s.rfind("- ", 0) == 0 && row >= 0 && row < 3 && col < 3) {
          b.rotation[row][col++] = (float)std::atof(s.substr(2).c_str());
        }
      }
    }
    // origins: a second pass (a plain list of 3 numbers under "origin:")
    std::ifstream in2(file);
    top.clear();
    sub.clear();
    id = -1;
    int n = 0;
    while (std::getline(in2, line)) {
      if (strip(line).empty()) continue;
      size_t indent = line.find_first_not_of(' ');
      std::string s = strip(line);
      if (indent == 0) {
        top = s.substr(0, s.find(':'));
        continue;
      }
      if (indent == 2 && s.back() == ':') {
        id = std::atoi(s.c_str());
        continue;
      }
      if (indent == 4 && s.back() == ':') {
        sub = s.substr(0, s.size() - 1);
        n = 0;
        continue;
      }
      if (top == "geos" && sub == "origin" && id >= 0 && s.rfind("- ", 0) == 0 && n < 3)
        bs[id].origin[n++] = (float)std::atof(s.substr(2).c_str());
    }
    stations.clear();
    for (auto& kv : bs) {
      if (kv.first != (int)stations.size()) throw std::runtime_error("Lighthouse system file: base stations must be 0, 1, ...");
      stations.push_back(kv.second);
    }
    if (stations.empty()) throw std::runtime_error("Lighthouse system file " + file + ": no base stations");
  }

  // Loaded once: L2F_SENSORS, else config/sensors.yaml searched from the working directory up.
  static const Config& instance() {
    static Config c = [] {
      Config cfg;
      std::string file;
      if (const char* e = std::getenv("L2F_SENSORS")) {
        file = e;
      } else {
        std::string up;
        for (int i = 0; i < 6 && file.empty(); i++, up += "../")
          if (std::ifstream(up + "config/sensors.yaml")) file = up + "config/sensors.yaml";
        if (file.empty()) throw std::runtime_error("config/sensors.yaml not found: set L2F_SENSORS");
      }
      cfg.read(file);
      std::string lh = cfg.strings.count("lighthouse.system") ? cfg.strings["lighthouse.system"] : "";
      if (const char* e = std::getenv("L2F_LIGHTHOUSE_SYSTEM")) lh = e;
      if (!lh.empty() && lh[0] != '/') {  // relative to the repository root (the folder holding config/)
        std::string root = file.substr(0, file.rfind("config/"));
        lh = root + lh;
      }
      cfg.read_lighthouse(lh);
      return cfg;
    }();
    return c;
  }
};

// Per-episode draws of everything given as [lo, hi].
struct Episode {
  float gyro_bias[3], acc_bias[3], gyro_noise_avg, gyro_noise_sample, acc_noise_avg;
  float range_gap, range_bias, range_noise, range_below_bias, range_below_noise, range_phase;
  float flow_gap, flow_noise, flow_period, flow_gain_out, flow_phase;
  std::vector<cfBaseStation_t> true_stations;  // stored + geometry error
  std::vector<float> lh_period, lh_phase, dropout_rate;
  std::vector<double> dropout_until;  // episode time a drop-out ends [s]
  float lh_angle_noise;
};

// One environment's sensor path: its estimator and this episode's draws. Owned by one
// Multirotor object (owner); created on its first episode.
struct Runtime {
  const void* owner = nullptr;
  cfEstimator_t* est = nullptr;
  Episode ep;
  uint32_t now_ms = 0;       // estimator clock
  double t = 0;              // episode time of the step start [s]
  bool flow_disabled = false;
  bool failed = false;       // the estimator failed (cfEstimatorFailed): the episode ends as a crash
  cfEstimate_t out{};
  ~Runtime() {
    if (est) cfEstimatorDestroy(est);
  }
};

// DRAW (begin_episode, simulate_step): uniform(lo, hi) and normal(sigma) from the simulator's RNG.

// small vector helpers (float)
inline void quat_to_R(const float q[4], float R[3][3]) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  R[0][0] = 1 - 2 * (y * y + z * z);
  R[0][1] = 2 * (x * y - w * z);
  R[0][2] = 2 * (x * z + w * y);
  R[1][0] = 2 * (x * y + w * z);
  R[1][1] = 1 - 2 * (x * x + z * z);
  R[1][2] = 2 * (y * z - w * x);
  R[2][0] = 2 * (x * z - w * y);
  R[2][1] = 2 * (y * z + w * x);
  R[2][2] = 1 - 2 * (x * x + y * y);
}

// Truth at the step's start and end (world frame) for interpolation.
struct Truth {
  float p0[3], p1[3], v0[3], v1[3], q0[4], q1[4], w0[3], w1[3];
  void at(float f, float p[3], float v[3], float R[3][3], float w[3]) const {
    float q[4], dot = 0, n = 0;
    for (int i = 0; i < 4; i++) dot += q0[i] * q1[i];
    const float s = dot < 0 ? -1.0f : 1.0f;
    for (int i = 0; i < 4; i++) {
      q[i] = (1 - f) * q0[i] + f * s * q1[i];
      n += q[i] * q[i];
    }
    n = std::sqrt(n);
    for (int i = 0; i < 4; i++) q[i] /= n;
    quat_to_R(q, R);
    for (int i = 0; i < 3; i++) {
      p[i] = (1 - f) * p0[i] + f * p1[i];
      v[i] = (1 - f) * v0[i] + f * v1[i];
      w[i] = (1 - f) * w0[i] + f * w1[i];
    }
  }
};

// The step size the estimator clock advances by; the simulator's dt must be a multiple of 1 ms.
constexpr int TICKS_PER_STEP = 10;

// Start an episode: draw the sensor parameters and the true base station geometry, and start
// the estimator converged at the true state (as after reset_estimator on the drone).
template <typename DRAW>
void begin_episode(Runtime& rt, const float p[3], const float v_world[3], const float q[4], DRAW& d) {
  const Config& c = Config::instance();
  Episode& e = rt.ep;
  auto u = [&](const char* key) {
    Interval r = c.get(key);
    return r.lo == r.hi ? (float)r.lo : (float)d.uniform(r.lo, r.hi);
  };
  for (int i = 0; i < 3; i++) {
    e.gyro_bias[i] = u("imu.gyro_bias");
    e.acc_bias[i] = u("imu.acc_bias");
  }
  e.gyro_noise_avg = u("imu.gyro_noise_avg");
  e.gyro_noise_sample = u("imu.gyro_noise_sample");
  e.acc_noise_avg = u("imu.acc_noise_avg");
  e.range_gap = u("range.rest_gap");
  e.range_bias = u("range.bias");
  e.range_noise = u("range.noise");
  e.range_below_bias = u("range.below_min_bias");
  e.range_below_noise = u("range.below_min_noise");
  e.range_phase = (float)d.uniform(0, c.fixed("range.period"));
  e.flow_gap = u("flow.rest_gap");
  e.flow_noise = u("flow.noise");
  e.flow_period = u("flow.period");
  e.flow_gain_out = u("flow.out_of_spec_gain");
  e.flow_phase = (float)d.uniform(0, e.flow_period);
  e.lh_angle_noise = u("lighthouse.angle_noise");
  const size_t n = c.stations.size();
  e.true_stations = c.stations;
  e.lh_period.resize(n);
  e.lh_phase.resize(n);
  e.dropout_rate.resize(n);
  e.dropout_until.assign(n, -1);
  const float gp = (float)c.fixed("lighthouse.geometry_error_position");
  const float gr = (float)c.fixed("lighthouse.geometry_error_rotation");
  for (size_t b = 0; b < n; b++) {
    cfBaseStation_t& s = e.true_stations[b];
    for (int i = 0; i < 3; i++) s.origin[i] += (float)d.normal(gp);
    // small rotation error: R_true = R_stored (I + [r]x), re-orthonormalized
    float r[3] = {(float)d.normal(gr), (float)d.normal(gr), (float)d.normal(gr)};
    float K[3][3] = {{1, -r[2], r[1]}, {r[2], 1, -r[0]}, {-r[1], r[0], 1}}, M[3][3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        M[i][j] = 0;
        for (int k = 0; k < 3; k++) M[i][j] += s.rotation[i][k] * K[k][j];
      }
    for (int j = 0; j < 3; j++) {  // Gram-Schmidt on the columns
      for (int k = 0; k < j; k++) {
        float dot = 0;
        for (int i = 0; i < 3; i++) dot += M[i][j] * M[i][k];
        for (int i = 0; i < 3; i++) M[i][j] -= dot * M[i][k];
      }
      float nn = 0;
      for (int i = 0; i < 3; i++) nn += M[i][j] * M[i][j];
      nn = std::sqrt(nn);
      for (int i = 0; i < 3; i++) M[i][j] /= nn;
    }
    std::memcpy(s.rotation, M, sizeof(M));
    e.lh_period[b] = 1.0f / u("lighthouse.rotation_rate");
    e.lh_phase[b] = (float)d.uniform(0, e.lh_period[b]);
    e.dropout_rate[b] = u("lighthouse.dropout_rate");
  }
  if (!rt.est) {
    cfEstimatorConfig_t ec;
    cfEstimatorDefaultConfig(&ec);
    ec.tof_min_distance = (float)c.fixed("estimator.range_min_distance");
    ec.tof_low_std = (float)c.fixed("estimator.range_low_std");
    rt.est = cfEstimatorCreate(&ec);
    for (size_t b = 0; b < n; b++) cfEstimatorSetBaseStation(rt.est, (int)b, &c.stations[b]);
  }
  rt.now_ms = 1000;
  rt.t = 0;
  rt.flow_disabled = false;
  rt.failed = false;
  cfEstimatorSetFlowDisabled(rt.est, false);
  float R[3][3], vb[3];
  quat_to_R(q, R);
  for (int i = 0; i < 3; i++) vb[i] = R[0][i] * v_world[0] + R[1][i] * v_world[1] + R[2][i] * v_world[2];
  // converged covariance: a few mm, cm/s and tenths of a degree (the filter settles there in < 1 s)
  const float p_diag[9] = {4e-6f, 4e-6f, 4e-6f, 1e-4f, 1e-4f, 1e-4f, 1e-5f, 1e-5f, 1e-5f};
  // A converged filter is off the truth by errors of that covariance; start it there. (Starting
  // exactly at a level truth also hits a firmware rounding edge: kalmanCoreFinalize can give
  // R[2][2] = 1 + 1 ulp when roll and pitch are exactly 0, and mm_tof.c's acosf of it is NaN.)
  float pe[3], vbe[3], qe[4], r[3];
  for (int i = 0; i < 3; i++) {
    pe[i] = p[i] + (float)d.normal(std::sqrt(p_diag[i]));
    vbe[i] = vb[i] + (float)d.normal(std::sqrt(p_diag[3 + i]));
    r[i] = (float)d.normal(std::sqrt(p_diag[6 + i]));
  }
  // qe = q * (1, r / 2), normalized
  qe[0] = q[0] - 0.5f * (q[1] * r[0] + q[2] * r[1] + q[3] * r[2]);
  qe[1] = q[1] + 0.5f * (q[0] * r[0] + q[2] * r[2] - q[3] * r[1]);
  qe[2] = q[2] + 0.5f * (q[0] * r[1] + q[3] * r[0] - q[1] * r[2]);
  qe[3] = q[3] + 0.5f * (q[0] * r[2] + q[1] * r[1] - q[2] * r[0]);
  const float qn = std::sqrt(qe[0] * qe[0] + qe[1] * qe[1] + qe[2] * qe[2] + qe[3] * qe[3]);
  for (int i = 0; i < 4; i++) qe[i] /= qn;
  cfEstimatorResetTo(rt.est, rt.now_ms, pe, vbe, qe, p_diag, (uint32_t)(d.uniform(1, 4e9)));
  float Re[3][3];
  quat_to_R(qe, Re);
  for (int i = 0; i < 3; i++) {
    rt.out.position[i] = pe[i];
    rt.out.velocity[i] = Re[i][0] * vbe[0] + Re[i][1] * vbe[1] + Re[i][2] * vbe[2];
  }
  for (int i = 0; i < 4; i++) rt.out.quat[i] = qe[i];
}

// Simulate one step of length TICKS_PER_STEP ms: sensors from the truth, 10 estimator ticks.
// thrust_over_idle: any motor commanded above idle. gyro_actor: the actor's gyro sample [rad/s].
template <typename DRAW>
void simulate_step(Runtime& rt, const Truth& x, bool thrust_over_idle, DRAW& d, float gyro_actor[3]) {
  const Config& c = Config::instance();
  Episode& e = rt.ep;
  const float G = 9.81f, dt = TICKS_PER_STEP * 1e-3f, PI = 3.14159265f;
  const float clip = (float)c.fixed("imu.gyro_clip");
  auto clipf = [](float v, float lim) { return v > lim ? lim : (v < -lim ? -lim : v); };
  const double t0 = rt.t, t1 = rt.t + dt;
  // events of this step, as (time, kind, index)
  struct Event {
    double t;
    int kind, index;
  };
  std::vector<Event> events;
  const double range_period = c.fixed("range.period");
  for (double te = e.range_phase + std::ceil((t0 - e.range_phase) / range_period) * range_period; te <= t1; te += range_period)
    if (te > t0) events.push_back({te, 0, 0});
  for (double te = e.flow_phase + std::ceil((t0 - e.flow_phase) / e.flow_period) * e.flow_period; te <= t1; te += e.flow_period)
    if (te > t0) events.push_back({te, 1, 0});
  for (size_t b = 0; b < e.true_stations.size(); b++) {
    const double T = e.lh_period[b];
    for (double te = e.lh_phase[b] + std::ceil((t0 - e.lh_phase[b]) / T) * T; te <= t1; te += T)
      if (te > t0) events.push_back({te, 2, (int)b});
  }
  const float max_rate = (float)c.fixed("flow.max_rate"), min_h = (float)c.fixed("flow.min_height");
  const float cone = (float)(c.fixed("range.cone_half_angle") * PI / 180);
  const float min_range = (float)c.fixed("range.min_distance");
  const float flow_off_below = (float)c.fixed("estimator.flow_off_below");
  float fp[3];
  cfFlowdeckPosition(fp);
  cfEstimate_t out = rt.out;
  for (int k = 1; k <= TICKS_PER_STEP; k++) {
    const uint32_t ms = rt.now_ms + k;
    const double tk0 = t0 + (k - 1) * 1e-3, tk1 = t0 + k * 1e-3;
    for (const Event& ev : events) {
      if (!(ev.t > tk0 && ev.t <= tk1)) continue;
      float p[3], v[3], R[3][3], w[3];
      x.at((float)((ev.t - t0) / dt), p, v, R, w);
      if (ev.kind == 0) {  // range
        const float tilt = std::acos(clipf(R[2][2], 1.0f));
        const float a = tilt > cone ? tilt - cone : 0.0f;
        float dist = (p[2] + e.range_gap) / std::cos(a);
        if (dist < min_range) {
          dist += e.range_below_bias + (float)d.normal(e.range_below_noise);
        } else {
          dist += e.range_bias + (float)d.normal(e.range_noise);
        }
        if (R[2][2] > 0.1f) {
          long mm = std::lround(dist * 1000);
          cfEstimatorRange(rt.est, (uint16_t)(mm < 0 ? 0 : (mm > 8190 ? 8190 : mm)), ms);
        }
      } else if (ev.kind == 1) {  // flow
        float vb[3];
        for (int i = 0; i < 3; i++) vb[i] = R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2];
        const float h = (p[2] + e.flow_gap) > 0.005f ? (p[2] + e.flow_gap) : 0.005f;
        const float vcx = vb[0] + w[1] * fp[2] - w[2] * fp[1], vcy = vb[1] + w[2] * fp[0] - w[0] * fp[2];
        const float rate = std::sqrt(vcx * vcx + vcy * vcy) / h;
        const float gain = (h < min_h || rate > max_rate) ? e.flow_gain_out : 1.0f;
        const float kpx = 10.0f * e.flow_period * 35.0f / 0.71674f;
        const float nx = gain * kpx * (vcx * R[2][2] / h - w[1]) + (float)d.normal(e.flow_noise);
        const float ny = gain * kpx * (vcy * R[2][2] / h + w[0]) + (float)d.normal(e.flow_noise);
        if (R[2][2] > 0) {
          long ix = std::lround(nx), iy = std::lround(ny);
          auto sat = [](long v) { return (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v)); };
          cfEstimatorFlow(rt.est, sat(-iy), sat(-ix), 0xB0, e.flow_period);
        }
      } else {  // Lighthouse rotation of base station ev.index
        const int b = ev.index;
        if (ev.t < e.dropout_until[b]) continue;
        if (e.dropout_rate[b] > 0 && d.uniform(0, 1) < e.dropout_rate[b] * e.lh_period[b]) {
          Interval dur = c.get("lighthouse.dropout_duration");
          e.dropout_until[b] = ev.t + d.uniform(dur.lo, dur.hi);
          continue;
        }
        float ang[4][2];
        bool vis[4];
        cfLighthouseModel(&e.true_stations[b], p, R, ang, vis);
        for (int s = 0; s < 4; s++)
          for (int j = 0; j < 2; j++) ang[s][j] += (float)d.normal(e.lh_angle_noise);
        cfEstimatorLighthouse(rt.est, b, ang, vis, ms);
      }
    }
    if (k == TICKS_PER_STEP - 1) {  // the step's IMU average, one tick before the prediction (k = 10)
      float p[3], v[3], R[3][3], w[3];
      x.at(0.5f, p, v, R, w);
      float aw[3], acc[3], gyro[3];
      for (int i = 0; i < 3; i++) aw[i] = (x.v1[i] - x.v0[i]) / dt;
      aw[2] += G;
      for (int i = 0; i < 3; i++) {
        acc[i] = (R[0][i] * aw[0] + R[1][i] * aw[1] + R[2][i] * aw[2] + e.acc_bias[i] + (float)d.normal(e.acc_noise_avg)) / G;
        const float g = 0.5f * (x.w0[i] + x.w1[i]) + e.gyro_bias[i] + (float)d.normal(e.gyro_noise_avg);
        gyro[i] = clipf(g, clip) * 180 / PI;
      }
      cfEstimatorImu(rt.est, acc, gyro);
    }
    cfEstimatorTick(rt.est, ms, thrust_over_idle, &out);
  }
  // E4 switch (Q13): flow off below flow_off_below (estimated height), on again above it
  if (flow_off_below > 0) {
    const bool off = out.position[2] < flow_off_below;
    if (off != rt.flow_disabled) {
      rt.flow_disabled = off;
      cfEstimatorSetFlowDisabled(rt.est, off);
    }
  }
  for (int i = 0; i < 3; i++)
    gyro_actor[i] = clipf(x.w1[i] + e.gyro_bias[i] + (float)d.normal(e.gyro_noise_sample), clip);
  rt.out = out;
  rt.failed = cfEstimatorFailed(rt.est, nullptr);
  rt.now_ms += TICKS_PER_STEP;
  rt.t = t1;
}

}  // namespace learning_to_fly::sensor_sim
