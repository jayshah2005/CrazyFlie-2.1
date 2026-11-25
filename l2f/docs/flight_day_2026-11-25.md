# Flight day 2026-11-25: split-critic aggregators on the real drone

Goal: fly one policy per method from the 8-seed split-critic study and compare the real flights
with the simulator's numbers for the same policy and mission.

## The policies

One checkpoint per method, all picked by the same rule from 8 seeds x checkpoints 1.2-2.0M:
no crash in any sim test (1,200 mission episodes across plain/simulator options x 6/12 ms x
clean/uneven motors/glitches, plus 100 waypoint episodes), hover jitter <= 0.12, then the lowest
circle error with uneven motors. Folders: `l2f/policies/2026-11-25_*` (actor.h5, actor.h, run.txt,
sim.json). Firmware: `~/l2f_firmware/firmware-cf21bl-2026.08-l2f-<tag>-mac.zip` (cold boot:
`flash_cold.py`). Variant E inputs: HISTORY=0, RPM_OBS=1.

| Method | Tag | Crash-free checkpoints (of 72) | Sim circle clean / uneven | Sim jitter | Sim waypoints 10 cm, 0.5 m (time) / 0.8 m |
| --- | --- | --- | --- | --- | --- |
| Variant E (TD3) | `E_td3_s73_1900k` | 5 | 4.5 / 8.0 cm | 0.101 | 96% (1.55 s) / 91% |
| Sum | `split_sum_s76_1500k` | 2 | 4.1 / 9.1 cm | 0.059 | 78% (1.41 s) / 76% |
| DualProj | `split_dualproj_s77_1400k` | 1 | 4.9 / 10.1 cm | 0.046 | 75% (1.25 s) / 69% |
| UPGrad | `split_upgrad_s77_1700k` | 6 | 3.5 / 7.7 cm | 0.064 | 95% (1.36 s) / 90% |
| Aligned-MTL | `split_alignedmtl_s73_1500k` | 12 | 4.0 / 7.5 cm | 0.111 | 95% (1.38 s) / 83% |
| SVD-PGA | `split_svdpga_s70_1200k` | 7 | 3.7 / 8.4 cm | 0.104 | 95% (1.09 s) / 93% |
| OS-NMF-PGA | `split_osnmfpga_s77_2000k` | 10 | 3.2 / 7.9 cm | 0.066 | 94% (1.45 s) / 81% |

Sim columns: means over the 4 mission test conditions (plain/simulator options x 6/12 ms) and
the waypoint tests (50 episodes, 3.5 s legs). "Crash-free checkpoints" counts the checkpoints
with no crash in any test and jitter <= 0.12.

## Faster circles in simulation (before flying them)

Mission test (height 0.75 m, R 0.6 m, ground effect, sensor noise, 12 ms), 100 episodes; circle
error clean / with uneven motors; crashes noted:

| Method | P 10 s (0.38 m/s) | P 6 s (0.63 m/s) | P 4 s (0.94 m/s) | P 3 s (1.26 m/s) |
| --- | --- | --- | --- | --- |
| Variant E | 4.4 / 7.7 | 8.1 / 10.7 (1 crash) | 14.7 / 16.2 | 21.8 / 23.5 (3 crashes) |
| Sum | 4.0 / 8.6 | 5.8 / 10.2 | 10.1 / 13.2 | 16.6 / 19.2 |
| DualProj | 4.8 / 9.2 | 6.7 / 11.4 | 10.9 / 14.0 | 16.0 / 18.1 |
| UPGrad | 3.5 / 7.2 | 6.4 / 9.2 | 12.1 / 13.7 | 18.7 / 19.8 |
| Aligned-MTL | 3.9 / 7.0 | 6.1 / 9.1 | 10.2 / 12.4 | 15.6 / 17.3 |
| SVD-PGA | 3.6 / 7.5 | 5.9 / 11.0 (2) | 10.2 / 13.7 (2) | 17.8 / 22.7 (**28 crashes**) |
| OS-NMF-PGA | 3.1 / 7.3 | 6.3 / 9.3 | 11.4 / 13.2 | 16.8 / 18.1 |

Every MOO pick tracks fast circles better than variant E in simulation (P 4 s: 10.1-12.1 against
14.7 cm). Do not fly SVD-PGA at P 3 s.

## The flights (match the sim test)

Height and circle as in the sim's mission test; no outer integrator (variant E has the
position-error integral input).

| Flight | Command | Sim counterpart |
| --- | --- | --- |
| Circle | `python rl_trigger.py --mode mission --height 0.75 --radius 0.6 --period 10 --descent-rate 0.25` | mission test, circle |
| Fast circle | same with `--period 6`, then `--period 4` | table above |
| Waypoints 0.5 m | `python rl_trigger.py --mode mission --height 0.75 --waypoints "0.4,0;0.1,0" --dwell 3.5 --descent-rate 0.25` | wp05L (0.4 m jump, then 0.3 m back, then 0.1 m to the centre) |
| Waypoints 0.8 m | same with `--waypoints "0.7,0;0.2,0"` | wp08L |

Lighthouse and the Enter-key switch were checked by hand on the day. After each flight:
`python l2f/sim/flight_report.py --label <method> logs/rl_trigger_flight_<time>.csv` prints the
same metrics as the sim (tracking error per phase, hover jitter, waypoint arrival, overshoot).

Suggested order (batteries): variant E, SVD-PGA, sum, then UPGrad, Aligned-MTL, OS-NMF-PGA,
DualProj. Per method: circle x2, waypoints 0.5 m, fast circle P 6 s (P 4 s if steady).

## All flights, all metrics

28 flights (`results/flights_2026-11-25/flights.csv`, waypoint legs in `legs.csv`). Track error:
mean 3D distance to the target in the circle phase (circle missions) or the waypoint phase
(waypoint missions). Smoothness over the airborne phases: tilt, rotation rate, wobble (RMS distance
from the 0.5 s moving average of the position), acceleration. Jitter: hover phases, the sim's
metric. The 17:36 flight is an OS-NMF-PGA fast circle flown after its reflash (video), inferred
from the flash order. Aligned-MTL's numbers after the flip are meaningless.

### Per method (medians over completed flights; error in cm, tilt in deg, rotation in deg/s, accel in m/s2)

| Method | Flights | Circle P10 | Fast circle P6 | Wp 0.8 m track | Wp 0.8 m settled | Hover | Tilt RMS | Rotation RMS | Wobble | Accel RMS | Jitter | Landing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OS-NMF-PGA | 7 | **7.2** | 13.9 | 24.2 | 13.9 | **2.0** | **5.5** | **43** | **1.4** | **2.3** | 0.058 | 3.3 |
| UPGrad | 3 | 7.6 | 15.5 | 21.1 | 11.4 | 3.9 | 8.2 | 63 | 1.8 | 2.6 | 0.085 | **2.2** |
| SVD-PGA | 5 | 8.8 | **12.0** | **18.5** | **8.0** | 6.3 | 11.5 | 104 | 2.6 | 4.0 | 0.222 | 2.8 |
| DualProj | 3 | 9.0 | 14.9 | 18.7 | 9.0 | 9.0 | 7.1 | 53 | 1.6 | 2.6 | **0.049** | 7.9 |
| Variant E (TD3) | 6 | 10.6 | 21.1 | 28.2 | 17.4 | 7.3 | 7.9 | 66 | 2.1 | 3.6 | 0.253 | 5.0 |
| Sum | 3 | 11.6 | 16.7 | 26.0 | 12.9 | 8.8 | 18.4 | 125 | 4.7 | 7.0 | 0.134 | 10.2 |
| Aligned-MTL | 1 | crashed | - | - | - | - | rocked +-40 deg | - | - | - | 0.300 | - |

### Every flight

| Method | Mission | Track err (cm) | Hover (cm) | Tilt RMS / max (deg) | Rotation RMS (deg/s) | Wobble (cm) | Accel RMS (m/s2) | Jitter | Waypoints: reached 10 cm / median time / overshoot / settled | Landing from start (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Variant E | circle P10 | 10.6 | 7.5 | 6.6 / 18 | 59 | 1.6 | 2.4 | 0.256 |  | 11.4 |
| Variant E | wp 0.5 m (telemetry lost) | 34.5 | 7.7 | 7.3 / 11 | 58 | 1.1 | 1.6 | 0.240 | 0% /  s / 0.0 / 35.9 |  |
| Variant E | wp 0.5 m | 14.2 | 7.6 | 5.0 / 10 | 50 | 1.0 | 1.4 | 0.253 | 100% / 0.76 s / 6.6 / 7.7 | 0.7 |
| Variant E | wp 0.8 m (telemetry lost) |  | 11.5 | 11.6 / 18 | 95 | 4.2 | 6.1 | 0.216 |  |  |
| Variant E | wp 0.8 m | 28.2 | 6.6 | 14.4 / 37 | 101 | 4.0 | 5.5 | 0.252 | 67% / 0.44 s / 18.3 / 17.4 | 6.9 |
| Variant E | fast circle P6 | 21.1 | 7.2 | 9.2 / 23 | 72 | 2.7 | 4.8 | 0.244 |  | 3.1 |
| SVD-PGA | circle P10 | 8.8 | 3.6 | 12.2 / 34 | 111 | 2.9 | 4.5 | 0.241 |  | 2.4 |
| SVD-PGA | wp 0.5 m | 11.6 | 7.1 | 10.5 / 34 | 97 | 2.1 | 3.2 | 0.203 | 100% / 0.96 s / 9.4 / 3.7 | 3.1 |
| SVD-PGA | wp 0.8 m (telemetry lost) | 28.4 | 4.4 | 19.7 / 43 | 169 | 4.9 | 7.7 | 0.221 | 100% / 1.54 s / 13.7 / 11.4 |  |
| SVD-PGA | wp 0.8 m | 18.5 | 7.1 | 13.9 / 41 | 130 | 3.2 | 5.2 | 0.243 | 100% / 1.26 s / 14.2 / 8.0 | 8.7 |
| SVD-PGA | fast circle P6 | 12.0 | 5.6 | 10.8 / 26 | 96 | 2.3 | 3.5 | 0.193 |  | 1.3 |
| OS-NMF-PGA | circle P10 | 7.3 | 2.4 | 4.9 / 16 | 39 | 1.5 | 2.4 | 0.058 |  | 1.4 |
| OS-NMF-PGA | circle P10 (repeat) | 7.1 | 2.0 | 5.5 / 19 | 44 | 1.6 | 2.5 | 0.072 |  | 5.1 |
| OS-NMF-PGA | wp 0.5 m | 12.1 | 1.3 | 3.9 / 12 | 31 | 1.2 | 1.8 | 0.048 | 100% / 0.88 s / 8.1 / 6.5 | 5.0 |
| OS-NMF-PGA | wp 0.8 m | 24.2 | 2.9 | 5.0 / 14 | 35 | 1.4 | 2.0 | 0.054 | 100% / 1.26 s / 17.0 / 13.9 | 1.9 |
| OS-NMF-PGA | fast circle P6 | 18.9 | 2.0 | 12.2 / 35 | 87 | 3.3 | 4.7 | 0.057 |  | 2.8 |
| Sum | circle P10 | 11.6 | 7.3 | 18.4 / 48 | 125 | 4.7 | 7.0 | 0.134 |  | 10.2 |
| Sum | wp 0.8 m | 26.0 | 8.9 | 20.7 / 57 | 140 | 5.2 | 8.2 | 0.136 | 100% / 0.74 s / 25.6 / 12.9 | 9.0 |
| Sum | fast circle P6 | 16.7 | 8.8 | 18.0 / 39 | 118 | 4.7 | 6.8 | 0.087 |  | 12.5 |
| Aligned-MTL | circle P10 (CRASHED) | 12.0 | 6.4 | 50.8 / 180 | 249 | 6.5 | 10.3 | 0.300 |  |  |
| UPGrad | circle P10 | 7.6 | 3.9 | 6.0 / 19 | 52 | 1.3 | 2.0 | 0.085 |  | 1.1 |
| UPGrad | wp 0.8 m | 21.1 | 2.9 | 8.2 / 24 | 63 | 1.8 | 2.8 | 0.079 | 100% / 1.16 s / 15.8 / 11.4 | 11.0 |
| UPGrad | fast circle P6 | 15.5 | 4.4 | 9.2 / 25 | 71 | 1.8 | 2.6 | 0.136 |  | 2.2 |
| OS-NMF-PGA | fast circle P6 (repeat 1) | 13.3 | 1.8 | 6.3 / 23 | 43 | 1.3 | 2.0 | 0.065 |  | 3.7 |
| OS-NMF-PGA | fast circle P6 (repeat 2) | 13.9 | 3.0 | 6.6 / 24 | 47 | 1.4 | 2.3 | 0.067 |  | 3.3 |
| DualProj | circle P10 | 9.0 | 8.0 | 6.7 / 24 | 53 | 1.7 | 2.6 | 0.046 |  | 8.2 |
| DualProj | wp 0.8 m | 18.7 | 9.6 | 7.1 / 23 | 59 | 1.6 | 2.6 | 0.057 | 100% / 1.1 s / 12.3 / 9.0 | 6.1 |
| DualProj | fast circle P6 | 14.9 | 9.0 | 7.3 / 24 | 52 | 1.4 | 2.2 | 0.049 |  | 7.9 |

## Final summary (all 7 methods; 28 flights)

One flight per mission per method unless noted; real run-to-run spread is about +-2.5 cm
(OS-NMF-PGA fast circle 18.9, 13.3, 13.9 cm), so close error differences are not rankings.

| Method | Circle P10 | Fast circle P6 | Waypoints 0.8 m: settled / oscillation | Hover | Tilt RMS | Wobble | Real jitter (sim) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OS-NMF-PGA | **7.2** (2 flights) | 18.9 / 13.3 / 13.9 (3) | 14-16 cm / none | **1.3-3.0** | **5-7 deg** | **1.4-1.5** | 0.05-0.07 (0.066) |
| UPGrad | 7.6 | 15.5 | 11-15 cm / none | 2.9-4.4 | 6-9 | 1.3-1.8 | 0.08-0.14 (0.064) |
| SVD-PGA | 8.8 | **12.0** | **8 cm** / +-5 cm | 3.6-7.1 | 11-14 | 2.3-3.2 | 0.19-0.24 (0.104) |
| DualProj | 9.0 | 14.9 | 7-14 cm / none | 8.0-9.6 | 7 | 1.4-1.7 | **0.05-0.06 (0.046)** |
| Variant E (TD3) | 10.6 | 21.1 | 17-26 cm / +-15 cm | 6.6-7.6 | 7-14 | 1.6-4.0 | 0.24-0.26 (0.101) |
| Sum | 11.6 | 16.7 | 13 cm / +-10-20 cm | 7.3-8.9 | **18-21** | **4.7-5.2** | 0.09-0.14 (0.059) |
| Aligned-MTL | crashed | - | - | 6.4 | rocked +-40 deg | - | 0.30 (0.111) |

- Every aggregator except Aligned-MTL beats variant E on the circle and the fast circle; PGA
  variants win their regimes: OS-NMF-PGA the calmest and most precise at low speed, SVD-PGA the
  most accurate at speed and on big jumps.
- On the same split critic, plain summing is the most violent policy (tilt RMS 18-21 deg) and
  Aligned-MTL crashed, while OS-NMF-PGA, UPGrad and DualProj fly smoothly: the aggregator, not the
  critic, decides how smoothly the policy flies.
- Sim jitter at checkpoint selection predicts real smoothness: the low-sim-jitter picks (OS-NMF,
  UPGrad, DualProj) keep it on the drone; the jerkier ones (sum, Aligned-MTL, SVD-PGA, E) get worse
  (E and SVD ~2.5x).
- Average tracking error alone misleads (sum looked mid-table); report smoothness next to it.

## Summary so far (16 flights; sum, UPGrad, Aligned-MTL, DualProj not flown yet)

| Mission | OS-NMF-PGA | SVD-PGA | Variant E (TD3) |
| --- | --- | --- | --- |
| Circle P 10 s | **7.3 / 7.1 cm** (2 flights) | 8.8 cm | 10.6 cm |
| Hover (typical) | **1.3-2.9 cm** | 3.6-7.1 cm | 6.6-7.6 cm (flies ~6 cm high) |
| Waypoints 0.5 m: time to 10 cm / settled | 1.32, 0.88 s / 6-7 cm | 1.08, 0.96 s / **4-5 cm** | 2.54, 0.76 s / 6-10 cm |
| Waypoints 0.8 m: settled error | 14-16 cm, no oscillation | **8 cm**, +-5 cm | 17-26 cm, +-15 cm oscillation at ~1 Hz |
| Fast circle P 6 s | 18.9 cm | **12.0 cm** | 21.1 cm |
| Hover jitter (real; sim) | **0.05-0.07; 0.066** | 0.19-0.24; 0.104 | 0.24-0.26; 0.101 |

- Both PGA variants beat variant E on every mission; the gap grows with speed (as in simulation).
- OS-NMF-PGA: smoothest and most precise at low speed; its real chatter matches the simulator
  (every other policy chatters at ~2.5x its sim jitter). Gentle: lags fast targets and settles
  10-15 cm off after 0.7 m jumps. Caveat: its checkpoint also had the lowest sim jitter of the
  picks, so part of the smoothness may be the checkpoint, not only the aggregator.
- SVD-PGA: best at speed (fast circle, big jumps), holds targets tightest.
- One flight per mission per method (OS-NMF circle twice): a consistent pattern, not yet a
  statistical result.
- Telemetry drop-outs: 3 of 6 waypoint flights before the fixes, 0 of 3 circles; the radio link
  stayed up (console messages kept arriving) while the log stream stopped, once after a corrupted
  log packet (`no LogEntry id=255`). Fixes in `rl_trigger.py`: `--slow-telemetry` (about 40% fewer
  log packets) and an automatic log-block restart before any blind landing. After the fixes one
  more drop (SVD-PGA 0.8 m, before the restart existed); none since. Waypoint missions set
  `rlt.pdlp=2.0` (circles 0.5): a possible link, not tested.
- Blind landing descends straight down from the current target (not back to the start).
- Logs and the combined table: `l2f/docs/results/flights_2026-11-25/` (16 flight CSVs,
  `flights.csv`, `legs.csv`).

## Smoothness: what average error misses

Sum's flights looked by far the worst, yet its average error was mid-table. Over the airborne
phases (now printed by `flight_report.py`): tilt RMS / max, rotation-rate RMS, wobble (RMS distance
from the 0.5 s moving average of the position), acceleration RMS.

| Flight | Avg error | Tilt RMS | Tilt max | Rotation RMS | Wobble | Accel RMS |
| --- | --- | --- | --- | --- | --- | --- |
| Circle P10: OS-NMF / E / SVD / sum | 6.5 / 9.3 / 7.5 / 11.7 cm | 4.9 / 6.6 / 12.2 / **18.4** deg | 16 / 19 / 34 / **48** | 39 / 59 / 111 / **125** deg/s | 1.5 / 1.6 / 2.9 / **4.7** cm | 2.4 / 2.4 / 4.6 / **7.0** m/s2 |
| Fast circle P6: E / SVD / OS-NMF / sum | 15.1 / 9.1 / 11.9 / 13.1 | 9.2 / 10.8 / 12.2 / **18.0** | 23 / 26 / 35 / **39** | 72 / 96 / 87 / **118** | 2.7 / 2.3 / 3.3 / **4.7** | 4.8 / 3.6 / 4.7 / **6.8** |
| Waypoints 0.8 m: OS-NMF / SVD / E / sum | 14.4 / 12.7 / 19.6 / 18.5 | 5.0 / 13.9 / 14.4 / **20.7** | 14 / 41 / 37 / **57** | 35 / 130 / 101 / **140** | 1.4 / 3.2 / 4.0 / **5.2** | 2.0 / 5.2 / 5.5 / **8.2** |

Sum rocks hardest on every mission (a 0.63 m/s circle needs about 4 deg of tilt; sum used 18);
OS-NMF-PGA is the calmest; SVD-PGA tracks well but tilts 11-14 deg. Report smoothness next to error.

## Real results

| Method | Flight log | Circle (cm) | Hover (cm) | Jitter | Waypoints 10 cm (time) | Overshoot | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Variant E | `rl_trigger_flight_20261125_144633.csv` (circle, P 10 s) | 10.6 (xy 9.7) | 7.5 (xy 3.3) | 0.256 | | | full mission; flew ~6 cm high in hover; landed 11.4 cm from start; battery 3.94 V, dipped to 3.36 V under load |
| Variant E | `rl_trigger_flight_20261125_145002.csv` (waypoints 0.5 m) | | | 0.24 | | | incomplete: telemetry lost 0.5 s into the first leg, blind auto-landing at the waypoint (battery started at 3.78 V); not counted |
| Variant E | `rl_trigger_flight_20261125_145616.csv` (waypoints 0.5 m) | | 7.6 | 0.253 | 2/2 legs; 0.4 m leg in 2.56 s, 0.3 m leg in 0.78 s (sim 96%, 1.55 s) | 6.9 / 12.0 cm | full mission; landed 0.7 cm from start; battery 3.89 V, min 3.28 V |
| Variant E | `rl_trigger_flight_20261125_150345.csv` (waypoints 0.8 m) | | | | | | incomplete: telemetry stopped in hover after a corrupted log packet (`no LogEntry id=255`); radio link stayed up; blind landing; not counted |
| Variant E | `rl_trigger_flight_20261125_150844.csv` (waypoints 0.8 m, `--slow-telemetry`) | | 6.6 | 0.252 | 0.7 m leg: never within 10 cm (overshot to 0.86 m, settled 12-17 cm past); 0.5 m leg back: 10 cm in 0.8 s, then drifted 20-30 cm off; sim 91% | 18 / 46 cm | full mission, no telemetry drop; after the big jumps an x oscillation of about +-15 cm at ~1 Hz (pitch +-20 deg) for several seconds; landed 6.9 cm from start |
| Variant E | `rl_trigger_flight_20261125_151353.csv` (fast circle, P 6 s, 0.63 m/s) | 21.1 (xy 20.5; sim 8.1 / 10.7) | 7.2 | 0.244 | | | full mission; flown radius 0.39-0.81 m (mean 0.66); gyro peaks 732 deg/s; landed 3.1 cm from start; battery 3.88 V, min 3.34 V |
| SVD-PGA | `rl_trigger_flight_20261125_151947.csv` (circle, P 10 s) | 8.8 (xy 8.4; sim 3.6 / 7.5) | 3.6 (xy 2.5) | 0.241 | | | full mission; height offset only ~2 cm (variant E ~6 cm); landed 2.4 cm from start; battery 3.88 V, min 3.24 V |
| SVD-PGA | `rl_trigger_flight_20261125_152426.csv` (waypoints 0.5 m) | | 7.1 | 0.203 | 2/2 legs; 0.4 m leg in 1.08 s, 0.3 m leg in 0.96 s (sim 95%, 1.09 s) | 9.4 / 10.0 cm | full mission; final errors 5.1 / 3.7 cm, steady after arrival; landed 3.1 cm from start; battery 4.11 V |
| SVD-PGA | `rl_trigger_flight_20261125_153304.csv` (waypoints 0.8 m, `--slow-telemetry`) | | 4.4 | 0.221 | partial: 0.7 m leg within 10 cm in 1.54 s (variant E never), overshoot 13.7 cm; telemetry stopped 2.2 s into the leg | 13.7 cm | incomplete: telemetry lost at 0.76 m, blind landing; battery 4.09 V |
| SVD-PGA | `rl_trigger_flight_20261125_154652.csv` (waypoints 0.8 m, slow telemetry + log restart) | | 7.1 | 0.243 | 2/2 big legs; 0.8 m leg in 1.26 s, 0.55 m leg in 1.32 s (sim 93%) | 14.4 / 14.2 cm | full mission, no telemetry drop; settled 0.73-0.79 m on the 0.7 m leg and 0.05-0.15 m on the 0.2 m leg; no sustained oscillation (variant E: +-15 cm); landed 8.7 cm from start; battery 4.11 V |
| SVD-PGA | `rl_trigger_flight_20261125_155328.csv` (fast circle, P 6 s, 0.63 m/s) | 12.0 (xy 11.4, max 15; sim 5.9 / 11.0) | 5.6 | 0.193 | | | full mission; flown radius 0.60-0.69 m (variant E 0.39-0.81); landed 1.3 cm from start; battery 4.10 V, min 3.58 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_160052.csv` (circle, P 10 s) | **7.3** (xy 7.2; sim 3.1 / 7.9) | **2.4** | **0.058** (sim 0.066) | | | full mission; **real jitter matches sim** (every other policy today 0.19-0.26); no height offset; gyro peaks 333 deg/s (others 730-960); landed 1.4 cm from start; battery 4.09 V, min 3.67 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_160408.csv` (circle, P 10 s; repeat, video) | **7.1** | **2.0** | **0.072** | | | full mission; repeat matches the first flight (7.3 / 2.4 / 0.058); landed 5.1 cm from start; battery 4.12 V, min 3.71 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_160742.csv` (waypoints 0.5 m) | | **1.3** | **0.048** | 2/2 legs; 0.4 m leg in 1.32 s, 0.35 m leg in 0.88 s (sim 94%, 1.45 s) | 10.2 / 8.1 cm | full mission; settled 0.41-0.48 m on the 0.4 m leg, then 0.00-0.03 m on the 0.1 m leg (7 cm short); smooth, no oscillation; landed 5.0 cm from start; battery 4.09 V, min 3.68 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_161219.csv` (waypoints 0.8 m) | | 2.9 | **0.054** | 2/2 big legs; 0.7 m leg in 1.52 s, 0.65 m leg in 1.26 s (sim 81%) | 21.6 / 17.0 cm | full mission, no telemetry drop; overshot to 0.90 m and settled 0.82-0.88 m (12-18 cm long) on the 0.7 m leg, then 0.02-0.09 m on the 0.2 m leg (11-18 cm short); smooth, no oscillation; landed 1.9 cm from start; battery 4.08 V, min 3.75 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_161639.csv` (fast circle, P 6 s, 0.63 m/s) | 18.9 (xy 17.8, max 38.5; sim 6.3 / 9.3) | 2.0 | **0.057** | | | full mission; flown radius 0.46-0.80 m; lags the moving target (SVD-PGA 12.0, variant E 21.1); landed 2.8 cm from start; battery 4.06 V, min 3.58 V |
| Sum | `rl_trigger_flight_20261125_162350.csv` (circle, P 10 s) | 11.6 (xy 11.0; sim 4.0 / 8.6) | 7.3 | 0.134 (sim 0.059) | | | full mission; flown radius 0.49-0.91 m; hover2 16.8 cm, landed 10.2 cm from start; battery 4.06 V, min 3.48 V |
| Sum | `rl_trigger_flight_20261125_163328.csv` (waypoints 0.8 m) | | 8.9 | 0.136 | 2/2 big legs; 0.7 m leg in 1.38 s, 0.5 m leg in 0.74 s (sim 76%) | 25.6 / 36.0 cm | full mission; swung 0.74-0.99 m around the 0.7 m target, overshot the return to -0.10 m, then oscillated +-10-20 cm (~1 Hz) to the end; landed 9.0 cm from start; battery 4.06 V, min 3.46 V |
| Sum | `rl_trigger_flight_20261125_164026.csv` (fast circle, P 6 s, 0.63 m/s) | 16.7 (xy 16.1, max 29.4; sim 5.8 / 10.2) | 8.8 | 0.087 | | | full mission; flown radius 0.42-0.80 m; landed 12.5 cm from start; battery 4.05 V, min 3.53 V |
| Aligned-MTL | `rl_trigger_flight_20261125_170321.csv` (circle, P 10 s) | 12.0 before the crash (sim 4.0 / 7.5) | 6.4 | 0.300 | | | **CRASHED** at the end of the circle / start of 'back' (~19.6 s): rocked +-40 deg with motors slamming 0 <-> full through the circle (battery dipping to 3.35 V), then tilted 45 deg, Lighthouse dropped twice (estimate jumped ~20 cm), motors saturated and it flipped (roll -110 deg) |
| UPGrad | `rl_trigger_flight_20261125_171303.csv` (circle, P 10 s) | 7.6 (sim 3.5 / 7.7) | 3.9 | 0.085 (sim 0.064) | | | full mission; calm: tilt RMS 6.0 deg (max 19), rotation 52 deg/s, wobble 1.3 cm, accel 2.0 m/s2; radius 0.55-0.71 m; landed 1.1 cm from start; battery 4.12 V, min 3.72 V |
| UPGrad | `rl_trigger_flight_20261125_172713.csv` (waypoints 0.8 m) | | 2.9 | 0.079 | 2/2 big legs; 0.7 m leg in 1.36 s, 0.6 m leg in 1.16 s (sim 90%) | 16.9 / 15.8 cm | full mission; settled 0.78-0.85 m (8-15 cm long), then 0.02-0.11 m (9-18 cm short); smooth: tilt RMS 8.2 deg, wobble 1.8 cm, accel 2.75 m/s2; landed 11.0 cm from start; battery 4.12 V, min 3.65 V |
| UPGrad | `rl_trigger_flight_20261125_173036.csv` (fast circle, P 6 s, 0.63 m/s) | 15.5 (xy 14.5, max 23.6; sim 6.4 / 9.2) | 4.4 | 0.136 | | | full mission; radius 0.54-0.79 m; tilt RMS 9.2 deg, wobble 1.8 cm, accel 2.6 m/s2; landed 2.2 cm from start; battery 4.09 V, min 3.46 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_173806.csv` (fast circle, P 6 s; repeat, video) | 13.9 (xy 13.7, max 22.8) | 3.0 | 0.067 | | | full mission; much better than its first fast circle (18.9): radius 0.54-0.71 m; calmest fast circle of the day: tilt RMS 6.6 deg, rotation 47 deg/s, wobble 1.4 cm, accel 2.3 m/s2; landed 3.3 cm from start; battery 3.97 V, min 3.48 V |
| DualProj | `rl_trigger_flight_20261125_174329.csv` (circle, P 10 s) | 9.0 (sim 4.9 / 10.1) | 8.0 | **0.046** (sim 0.046) | | | full mission; calm: tilt RMS 6.7 deg, rotation 53 deg/s, wobble 1.7 cm, accel 2.6 m/s2; radius 0.49-0.79 m; hover looser (8 cm); landed 8.2 cm from start; battery 4.11 V, min 3.74 V |
| DualProj | `rl_trigger_flight_20261125_174633.csv` (waypoints 0.8 m) | | 9.6 | 0.057 | 2/2 big legs; 0.7 m leg in 1.36 s, 0.5 m leg in 1.10 s (sim 69%) | 12.3 / 17.4 cm | full mission; settled 0.72-0.80 m (6.6 cm), then 0.00-0.05 m (14 cm short), ended ~10 cm behind the start; smooth: tilt RMS 7.1 deg, wobble 1.6 cm; landed 6.1 cm from start; battery 4.09 V, min 3.68 V |
| DualProj | `rl_trigger_flight_20261125_175209.csv` (fast circle, P 6 s, 0.63 m/s) | 14.9 (xy 14.7, max 24.1; sim 6.7 / 11.4) | 9.0 | 0.049 | | | full mission; radius 0.45-0.74 m; smooth: tilt RMS 7.3 deg, wobble 1.4 cm, accel 2.2 m/s2; landed 7.9 cm from start; battery 4.07 V, min 3.72 V |
| OS-NMF-PGA | `rl_trigger_flight_20261125_173650.csv` (fast circle, P 6 s; repeat 1, video) | 13.3 | 1.8 | 0.065 | | | full mission; tilt RMS 6.3 deg, wobble 1.3 cm; landed 3.7 cm from start |
