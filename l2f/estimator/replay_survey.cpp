// Replay a sensor-survey log (noise_survey.py, logs/noise_survey_<mode>_<time>.csv) through the
// estimator library and compare the result with the estimate the drone logged (plan E1, check 2).
//
// The survey logs at 100 Hz (cflib's limit), so the replay is approximate by construction: per
// row, the logged gyro / accelerometer values stand in for the 10 ms IMU average; a range
// reading is a change of range.zrange; a flow reading is every row's motion.deltaX / deltaY;
// a Lighthouse rotation is a change of the logged raw angles, which the firmware logs for deck
// sensor 0 only (the other three sensors and the throttle's choices are not in the log). The
// thrust flag comes from the motor speeds. The filter starts at the first logged estimate.
// So the result is a tolerance (a few mm of position), not float equality (that needs D4).
//
// Usage: replay_survey <survey.csv> <lighthouse system file> [--from <ms>]
//        replay_survey --selftest   (writes a synthetic log from ideal sensors, replays it)
// Diagnosis: REPLAY_DEBUG=1 prints the height error over time; REPLAY_NO_RANGE=1,
// REPLAY_NO_FLOW=1, REPLAY_NO_LH=1 leave that sensor out of the replay.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../learning_to_fly/include/learning_to_fly/simulator/sensor_sim.h"
#include "cf_estimator.h"

namespace {
constexpr float PI = 3.14159265f;
const char* COLUMNS[] = {"drone_ms", "tag", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "zrange",
                         "flow_dx", "flow_dy", "flow_squal", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
                         "rpm1", "rpm2", "rpm3", "rpm4", "vbat", "lh_active", "lh_bs0_sweep0", "lh_bs0_sweep1",
                         "lh_bs1_sweep0", "lh_bs1_sweep1"};

struct Row {
  std::map<std::string, double> v;
  bool has(const std::string& k) const { return v.count(k) > 0; }
  double operator[](const std::string& k) const { auto it = v.find(k); return it == v.end() ? NAN : it->second; }
};

// run_log CSV: '#' metadata lines, a header, then rows (empty cells: not received yet)
std::vector<Row> read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::string line;
  std::vector<std::string> header;
  std::vector<Row> rows;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    if (header.empty()) {
      header = cells;
      continue;
    }
    Row r;
    for (size_t i = 0; i < cells.size() && i < header.size(); i++)
      if (!cells[i].empty() && header[i] != "tag") r.v[header[i]] = std::atof(cells[i].c_str());
    rows.push_back(r);
  }
  return rows;
}

// quaternion (w, x, y, z) from the logged roll, pitch (legacy sign: -pitch), yaw [deg]
void quat_from_log(double roll, double pitch_logged, double yaw, float q[4]) {
  const double r = roll * PI / 180, p = -pitch_logged * PI / 180, y = yaw * PI / 180;
  const double cr = std::cos(r / 2), sr = std::sin(r / 2), cp = std::cos(p / 2), sp = std::sin(p / 2), cy = std::cos(y / 2), sy = std::sin(y / 2);
  q[0] = (float)(cr * cp * cy + sr * sp * sy);
  q[1] = (float)(sr * cp * cy - cr * sp * sy);
  q[2] = (float)(cr * sp * cy + sr * cp * sy);
  q[3] = (float)(cr * cp * sy - sr * sp * cy);
}

struct Stats {
  double se_pos = 0, se_vel = 0, se_rp = 0, se_yaw = 0, max_pos = 0;
  int n = 0;
};

// Replay rows through a fresh estimator; returns the error statistics against the logged estimate.
Stats replay(const std::vector<Row>& rows, const std::vector<cfBaseStation_t>& stations, double from_ms) {
  cfEstimator_t* est = cfEstimatorCreate(nullptr);
  for (size_t b = 0; b < stations.size(); b++) cfEstimatorSetBaseStation(est, (int)b, &stations[b]);
  Stats st;
  bool started = false;
  uint32_t now = 0;
  double last_range = NAN, last_angles[4] = {NAN, NAN, NAN, NAN};
  cfEstimate_t out{};
  for (const Row& r : rows) {
    if (!r.has("drone_ms") || !r.has("x") || !r.has("gyro_x") || r["drone_ms"] < from_ms) continue;
    const uint32_t t = (uint32_t)r["drone_ms"];
    if (!started) {
      const float p[3] = {(float)r["x"], (float)r["y"], (float)r["z"]};
      float q[4];
      quat_from_log(r["roll"], r["pitch"], r["yaw"], q);
      float R[3][3];
      learning_to_fly::sensor_sim::quat_to_R(q, R);
      const float v[3] = {(float)r["vx"], (float)r["vy"], (float)r["vz"]};
      float vb[3];
      for (int i = 0; i < 3; i++) vb[i] = R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2];
      const float p_diag[9] = {1e-4f, 1e-4f, 1e-4f, 1e-4f, 1e-4f, 1e-4f, 1e-4f, 1e-4f, 1e-4f};
      cfEstimatorResetTo(est, t, p, vb, q, p_diag, 1);
      now = t;
      started = true;
      continue;
    }
    if (t <= now) continue;
    bool thrust = false;
    for (int m = 1; m <= 4; m++) thrust = thrust || r["rpm" + std::to_string(m)] > 1000;
    // the row's readings arrive at its time; ticks up to it (the last one handles them)
    for (uint32_t ms = now + 1; ms <= t; ms++) {
      if (ms == t - 1 || (t - now == 1 && ms == t)) {
        const float acc[3] = {(float)r["acc_x"], (float)r["acc_y"], (float)r["acc_z"]};
        const float gyro[3] = {(float)r["gyro_x"], (float)r["gyro_y"], (float)r["gyro_z"]};
        cfEstimatorImu(est, acc, gyro);
      }
      if (ms == t) {
        if (r.has("zrange") && r["zrange"] != last_range && !std::getenv("REPLAY_NO_RANGE")) {
          cfEstimatorRange(est, (uint16_t)r["zrange"], ms);
          last_range = r["zrange"];
        }
        if (r.has("flow_dx") && !std::getenv("REPLAY_NO_FLOW")) cfEstimatorFlow(est, (int16_t)r["flow_dx"], (int16_t)r["flow_dy"], 0xB0, (float)(t - now) * 1e-3f);
        const char* keys[4] = {"lh_bs0_sweep0", "lh_bs0_sweep1", "lh_bs1_sweep0", "lh_bs1_sweep1"};
        for (int b = 0; b < 2 && b < (int)stations.size() && !std::getenv("REPLAY_NO_LH"); b++) {
          const double a0 = r[keys[2 * b]], a1 = r[keys[2 * b + 1]];
          if (std::isnan(a0) || std::isnan(a1) || (a0 == last_angles[2 * b] && a1 == last_angles[2 * b + 1])) continue;
          if (a0 == 0 || a1 == 0) continue;  // not seen (the firmware skips zero angles)
          float ang[4][2] = {{(float)a0, (float)a1}, {0, 0}, {0, 0}, {0, 0}};
          bool vis[4] = {true, false, false, false};
          cfEstimatorLighthouse(est, b, ang, vis, ms);
          last_angles[2 * b] = a0;
          last_angles[2 * b + 1] = a1;
        }
      }
      cfEstimatorTick(est, ms, thrust, &out);
    }
    now = t;
    double e2 = 0, v2 = 0;
    const char* pk[3] = {"x", "y", "z"};
    const char* vk[3] = {"vx", "vy", "vz"};
    for (int i = 0; i < 3; i++) {
      e2 += std::pow(out.position[i] - r[pk[i]], 2);
      v2 += std::pow(out.velocity[i] - r[vk[i]], 2);
    }
    const double dr = out.attitude_deg[0] - r["roll"], dp = out.attitude_deg[1] - r["pitch"];
    double dy = out.attitude_deg[2] - r["yaw"];
    dy = std::fmod(dy + 540.0, 360.0) - 180.0;
    st.se_pos += e2;
    st.se_vel += v2;
    st.se_rp += dr * dr + dp * dp;
    st.se_yaw += dy * dy;
    st.max_pos = std::max(st.max_pos, std::sqrt(e2));
    if (std::getenv("REPLAY_DEBUG") && (st.n < 12 || st.n % 100 == 0))
      std::printf("  t %u: replayed z %.3f vz %.3f, logged z %.3f vz %.3f, error %.1f mm\n", t, out.position[2], out.velocity[2], r["z"], r["vz"], 1e3 * std::sqrt(e2));
    st.n++;
  }
  const char* why = nullptr;
  if (cfEstimatorFailed(est, &why)) std::printf("estimator failed: %s\n", why);
  cfEstimatorDestroy(est);
  return st;
}

void report(const Stats& s) {
  if (!s.n) {
    std::printf("no rows replayed\n");
    return;
  }
  std::printf("%d rows: replayed - logged estimate: position RMS %.2f mm (max %.2f), velocity RMS %.1f mm/s, "
              "roll/pitch RMS %.2f deg, yaw RMS %.2f deg\n",
              s.n, 1e3 * std::sqrt(s.se_pos / s.n), 1e3 * s.max_pos, 1e3 * std::sqrt(s.se_vel / s.n),
              std::sqrt(s.se_rp / (2 * s.n)), std::sqrt(s.se_yaw / s.n));
}

// Self-test: a "drone" estimator with all four Lighthouse sensors and ideal sensors at 1 kHz
// flies a slow circle; every 10 ms a row is written in the survey's format with what cflib
// would log. The replay of that file must stay within a few mm of the logged estimate.
int selftest() {
  std::vector<cfBaseStation_t> bs(2);
  auto make_bs = [](cfBaseStation_t& b, float ox, float oy, float oz) {
    std::memset(&b, 0, sizeof(b));
    float f[3] = {-ox, -oy, 0.5f - oz}, n = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (float& v : f) v /= n;
    float y[3] = {-f[1], f[0], 0};
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
    b.calibration_valid = true;
  };
  make_bs(bs[0], -1.8f, -1.8f, 2.4f);
  make_bs(bs[1], 1.8f, 1.8f, 2.4f);
  cfEstimator_t* drone = cfEstimatorCreate(nullptr);
  for (int b = 0; b < 2; b++) cfEstimatorSetBaseStation(drone, b, &bs[b]);
  const float r0 = 0.3f, w = 1.0f, h = 0.6f, G = 9.81f;
  auto attitude = [&](double t, float R[3][3]) {
    // tilt into the turn: body z along a + g z, heading along x
    const float ax = -r0 * w * w * (float)std::cos(w * t), ay = -r0 * w * w * (float)std::sin(w * t);
    float zb[3] = {ax / G, ay / G, 1};
    float n = std::sqrt(zb[0] * zb[0] + zb[1] * zb[1] + 1);
    for (float& c : zb) c /= n;
    float xb[3] = {zb[2], 0, -zb[0]};
    n = std::sqrt(xb[0] * xb[0] + xb[2] * xb[2]);
    for (float& c : xb) c /= n;
    float yb[3] = {zb[1] * xb[2] - zb[2] * xb[1], zb[2] * xb[0] - zb[0] * xb[2], zb[0] * xb[1] - zb[1] * xb[0]};
    for (int i = 0; i < 3; i++) {
      R[i][0] = xb[i];
      R[i][1] = yb[i];
      R[i][2] = zb[i];
    }
  };
  auto truth = [&](double t, float p[3], float v[3], float R[3][3], float om[3]) {
    p[0] = r0 * std::cos(w * t);
    p[1] = r0 * std::sin(w * t);
    p[2] = h;
    v[0] = -r0 * w * std::sin(w * t);
    v[1] = r0 * w * std::cos(w * t);
    v[2] = 0;
    attitude(t, R);
    // body rates: omega^ = R^T dR/dt (central difference)
    const double dh = 1e-4;
    float R0[3][3], R1[3][3], D[3][3];
    attitude(t - dh, R0);
    attitude(t + dh, R1);
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        D[a][b] = 0;
        for (int k = 0; k < 3; k++) D[a][b] += R[k][a] * (R1[k][b] - R0[k][b]) / (float)(2 * dh);
      }
    om[0] = 0.5f * (D[2][1] - D[1][2]);
    om[1] = 0.5f * (D[0][2] - D[2][0]);
    om[2] = 0.5f * (D[1][0] - D[0][1]);
  };
  float p[3], v[3], R[3][3], om[3];
  truth(0, p, v, R, om);
  const float q0[4] = {1, 0, 0, 0}, p_diag[9] = {4e-6f, 4e-6f, 4e-6f, 1e-4f, 1e-4f, 1e-4f, 1e-5f, 1e-5f, 1e-5f};
  float vb0[3] = {v[0], v[1], v[2]};
  cfEstimatorResetTo(drone, 1000, p, vb0, q0, p_diag, 3);
  const std::string path = "replay_selftest.csv";
  std::ofstream f(path);
  f << "# script: replay_survey selftest\n";
  for (size_t i = 0; i < sizeof(COLUMNS) / sizeof(COLUMNS[0]); i++) f << (i ? "," : "") << COLUMNS[i];
  f << "\n";
  cfEstimate_t out{};
  uint16_t last_range = (uint16_t)std::lround(h * 1000);  // the sensor has been running before the log starts
  int16_t flow[2] = {0, 0};
  float angles[2][2] = {{0, 0}, {0, 0}};
  for (uint32_t ms = 1001; ms <= 11000; ms++) {
    const double t = (ms - 1000) * 1e-3;
    truth(t, p, v, R, om);
    float aw[3] = {-r0 * w * w * (float)std::cos(w * t), -r0 * w * w * (float)std::sin(w * t), G}, acc[3], gyro[3];
    for (int i = 0; i < 3; i++) acc[i] = (R[0][i] * aw[0] + R[1][i] * aw[1] + R[2][i] * aw[2]) / G;
    for (int i = 0; i < 3; i++) gyro[i] = om[i] * 180 / PI;
    if (ms % 10 == 9) cfEstimatorImu(drone, acc, gyro);
    if (ms % 25 == 0) {
      const float cone = std::max(0.0f, std::acos(std::min(1.0f, R[2][2])) - 7.5f * PI / 180);
      last_range = (uint16_t)std::lround(p[2] / std::cos(cone) * 1000);
      cfEstimatorRange(drone, last_range, ms);
    }
    if (ms % 10 == 5) {
      float vbb[3];
      for (int i = 0; i < 3; i++) vbb[i] = R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2];
      const float k = 10.0f * 0.01f * 35.0f / 0.71674f;
      const long nx = std::lround(k * vbb[0] * R[2][2] / p[2]), ny = std::lround(k * vbb[1] * R[2][2] / p[2]);
      flow[0] = (int16_t)(-ny);
      flow[1] = (int16_t)(-nx);
      cfEstimatorFlow(drone, flow[0], flow[1], 0xB0, 0.01f);
    }
    for (int b = 0; b < 2; b++)
      if (ms % 20 == (uint32_t)(10 * b)) {
        float ang[4][2];
        bool vis[4];
        cfLighthouseModel(&bs[b], p, R, ang, vis);
        cfEstimatorLighthouse(drone, b, ang, vis, ms);
        angles[b][0] = ang[0][0];
        angles[b][1] = ang[0][1];
      }
    cfEstimatorTick(drone, ms, true, &out);
    if (ms % 10 == 0) {
      f << ms << ",selftest," << out.position[0] << "," << out.position[1] << "," << out.position[2] << ","
        << out.attitude_deg[0] << "," << out.attitude_deg[1] << "," << out.attitude_deg[2] << "," << out.velocity[0] << ","
        << out.velocity[1] << "," << out.velocity[2] << "," << last_range << "," << flow[0] << "," << flow[1] << ",100,"
        << gyro[0] << "," << gyro[1] << "," << gyro[2] << "," << acc[0] << "," << acc[1] << "," << acc[2]
        << ",12000,12000,12000,12000,4.0,3," << angles[0][0] << "," << angles[0][1] << "," << angles[1][0] << ","
        << angles[1][1] << "\n";
    }
  }
  f.close();
  cfEstimatorDestroy(drone);
  std::vector<Row> rows = read_csv(path);
  Stats s = replay(rows, bs, 0);
  report(s);
  std::remove(path.c_str());
  const bool ok = s.n > 900 && std::sqrt(s.se_pos / s.n) < 0.005 && std::sqrt(s.se_rp / (2 * s.n)) < 0.5;
  std::printf("self-test %s (limits: position RMS < 5 mm, roll/pitch RMS < 0.5 deg)\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--selftest") return selftest();
  if (argc < 3) {
    std::printf("usage: %s <survey.csv> <lighthouse system file> [--from <ms>] | --selftest\n", argv[0]);
    return 1;
  }
  double from = 0;
  for (int i = 3; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--from") from = std::atof(argv[i + 1]);
  learning_to_fly::sensor_sim::Config cfg;
  cfg.read_lighthouse(argv[2]);
  std::vector<Row> rows = read_csv(argv[1]);
  report(replay(rows, cfg.stations, from));
  return 0;
}
