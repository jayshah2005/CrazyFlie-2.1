#!/usr/bin/env bash
# Build the learning_to_fly out-of-tree controller on crazyflie-firmware 2026.08
# for the Crazyflie 2.1 Brushless (default) and package nothing else; see
# ../README.md. Run inside WSL Ubuntu.
#
#   ACTOR=<checkpoint .h>  policy to compile in (default: the controller repo's actor.h)
#   DEFCONFIG=cf21bl_defconfig | cf2_defconfig
#   L2F=~/learning_to_fly  FW=~/crazyflie-firmware-2026.08  DST=~/l2f_2026
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L2F=${L2F:-$HOME/learning_to_fly}
export FW=${FW:-$HOME/crazyflie-firmware-2026.08}
SRC=$L2F/controller
DST=${DST:-$HOME/l2f_2026}
DEFCONFIG=${DEFCONFIG:-cf21bl_defconfig}

if [ ! -d "$FW" ]; then
  git clone -q --branch 2026.08 --depth 1 https://github.com/bitcraze/crazyflie-firmware.git "$FW"
  git -C "$FW" submodule update -q --init --recursive --depth 1
fi
git -C "$FW" checkout -q -- src/modules/src/controller/controller.c   src/modules/src/crtp_commander.c src/modules/src/stabilizer.c
python3 "$HERE/patch_firmware.py"

mkdir -p "$DST/external" "$DST/data"
cp "$SRC"/rl_tools_controller.c "$SRC"/rl_tools_controller.h "$SRC"/rl_tools_adapter.cpp "$SRC"/rl_tools_adapter.h "$DST/"
cp "${ACTOR:-$SRC/data/actor.h}" "$DST/data/actor.h"
ln -sfn "$FW" "$DST/external/crazyflie_firmware"
ln -sfn "$SRC/external/rl_tools" "$DST/external/rl_tools"

cat > "$DST/config" <<'EOF'
CONFIG_CONTROLLER_OOT=y
EOF

# -DRL_TOOLS_CONTROLLER and the include paths are needed by the patched firmware
# files too; C++-only flags go on the adapter object alone.
cat > "$DST/Makefile" <<'EOF'
CRAZYFLIE_BASE := external/crazyflie_firmware

OOT_CONFIG := $(PWD)/config
OOT_USES_CXX := 1
EXTRA_CFLAGS += -I$(PWD) -I$(PWD)/external/rl_tools/include -DRL_TOOLS_CONTROLLER

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
EOF

cat > "$DST/Kbuild" <<'EOF'
obj-y += rl_tools_controller.o
obj-y += rl_tools_adapter.o
CFLAGS_rl_tools_controller.o += -Wno-error
# rl_tools needs C++17 (GCC 10 defaults to C++14). ccflags-y reaches the C++
# object; the C file only warns about it, which -Wno-error above tolerates.
ccflags-y += -std=gnu++17
EOF

# Per-object CFLAGS do not reach C++ objects in this build system, so the
# adapter silences the third-party (rl_tools / generated actor.h) warnings and
# asks for -O3 itself.
python3 - "$DST/rl_tools_adapter.cpp" <<'EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
head = ('#pragma GCC optimize("O3")\n'
        '#pragma GCC diagnostic ignored "-Wmissing-braces"\n'
        '#pragma GCC diagnostic ignored "-Wsign-compare"\n'
        '#pragma GCC diagnostic ignored "-Wunused-local-typedefs"\n'
        '#pragma GCC diagnostic ignored "-Wdouble-promotion"\n')
if not t.startswith(head):
    p.write_text(head + t)
print('adapter pragmas set')
EOF

# 2026.08 passes a const setpoint and a stabilizerStep_t; the controller edits
# the setpoint when an original controller is selected, so work on a copy.
python3 - "$DST/rl_tools_controller.c" <<'EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
old = ('void controllerOutOfTree(control_t *control, setpoint_t *setpoint, '
       'const sensorData_t *sensors, const state_t *state, const uint32_t tick) {\n')
new = ('void controllerOutOfTree(control_t *control, const setpoint_t *setpoint_in, '
       'const sensorData_t *sensors, const state_t *state, const stabilizerStep_t tick) {\n'
       '  setpoint_t setpoint_copy = *setpoint_in;  // edited below when an original controller flies\n'
       '  setpoint_t *setpoint = &setpoint_copy;\n')
if new not in t:
    assert t.count(old) == 1, 'controllerOutOfTree signature not found'
    p.write_text(t.replace(old, new))
print('controller adapted')
EOF

# Crazyflie 2.1 Brushless: the policy outputs RPM in [0, 24000] (the simulator's
# action limit) and the ESC command is derived from the identified map
#   rpm = 5400 * (ratio * vbat) + 3750
# (sysid flights 2026-11-01), so the command compensates battery voltage.
python3 - "$DST/rl_tools_controller.c" <<'EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
edits = [
    ('#define MAX_RPM 21702.1\n',
     '#ifdef CONFIG_PLATFORM_CF21BL\n'
     '#define MAX_RPM 24000.0f     // must match action_limit.max in crazyflie_brushless.h\n'
     '#define RPM_PER_VOLT 5400.0f // identified: rpm = RPM_PER_VOLT * (ratio * vbat) + RPM_OFFSET\n'
     '#define RPM_OFFSET 3750.0f\n'
     '#else\n'
     '#define MAX_RPM 21702.1\n'
     '#endif\n'),
    ('#include "pm.h"\n', '#include "pm.h"\n#include "autoconf.h"\n'),
    ('static inline void update_state(',
     'static inline float rpm_to_ratio(float rpm){\n'
     '#ifdef CONFIG_PLATFORM_CF21BL\n'
     '  float vbat = pmGetBatteryVoltage();\n'
     '  if(vbat < 3.0f){\n'
     '    vbat = 3.0f;\n'
     '  }\n'
     '  return clip((rpm - RPM_OFFSET) / (RPM_PER_VOLT * vbat), 0, 1);\n'
     '#else\n'
     '  return rpm / MAX_RPM;\n'
     '#endif\n'
     '}\n'
     '\n'
     'static inline void update_state('),
    ('      float des_percentage = des_rpm / MAX_RPM;\n',
     '      float des_percentage = rpm_to_ratio(des_rpm);\n'),
]
for old, new in edits:
    if new in t:
        continue
    assert t.count(old) == 1, f'not found exactly once: {old!r}'
    t = t.replace(old, new)
p.write_text(t)
print('brushless rpm mapping set')
EOF

# Run the policy at 100 Hz, the simulator's step (dt = 0.01 s), instead of
# 500 Hz with 5-step averaging into the action history. At ~0.7 ms per forward
# pass, 500 Hz took about a third of the CPU inside the highest-priority task
# and starved the radio task (radiolink.c queue-overflow assert). Also drop the
# 1 Hz "Last setpoint" console spam.
python3 - "$DST/rl_tools_controller.c" "$DST/rl_tools_adapter.cpp" <<'EOF'
import sys, pathlib
ctrl, adapter = map(pathlib.Path, sys.argv[1:3])
for path, edits in [
    (ctrl, [('#define CONTROL_INTERVAL_MS 2\n', '#define CONTROL_INTERVAL_MS 10\n'),
            ('  DEBUG_PRINT("Last setpoint: x disposition/mode %f/%f/%d\\n", last_setpoint.position.x, last_setpoint.velocity.x, last_setpoint.mode.x);\n'
             '  DEBUG_PRINT("Last setpoint: y disposition/mode %f/%f/%d\\n", last_setpoint.position.y, last_setpoint.velocity.y, last_setpoint.mode.y);\n'
             '  DEBUG_PRINT("Last setpoint: z disposition/mode %f/%f/%d\\n", last_setpoint.position.z, last_setpoint.velocity.z, last_setpoint.mode.z);\n',
             '')]),
    (adapter, [('constexpr TI CONTROL_FREQUENCY_MULTIPLE = 5;', 'constexpr TI CONTROL_FREQUENCY_MULTIPLE = 1;')]),
]:
    t = path.read_text()
    for old, new in edits:
        if old not in t and new and new in t:
            continue
        assert t.count(old) == 1, f'{path.name}: not found exactly once: {old[:60]!r}'
        t = t.replace(old, new)
    path.write_text(t)
print('policy rate 100 Hz')
EOF

# 1) DShot: motorsSetRatio() stops and re-arms the motor's DMA (and M1/M2 share
#    a DMA request), so every motor must get a ratio before every
#    motorsBurstDshot() at 1 kHz, as the PID path does. Writing only on the
#    100 Hz policy ticks killed M1/M2 in flight (eRPM 65535 / frozen).
# 2) Handover: start the policy like a training episode, with the action
#    history set to the motors' current speeds (measured eRPM, else derived
#    from the commanded ratio), and evaluate it on the activation tick.
python3 - "$DST/rl_tools_controller.c" "$DST/rl_tools_adapter.cpp" "$DST/rl_tools_adapter.h" <<'EOF'
import sys, pathlib
ctrl, adapter, header = map(pathlib.Path, sys.argv[1:4])
edits = {
    ctrl: [
        ('      motor_cmd[i] = des_percentage * UINT16_MAX;\n'
         '      if(set_motors && use_orig_controller == 0){\n'
         '        motorsSetRatio(motors[i], clip((float)motor_cmd[i] / motor_cmd_divider, 0, UINT16_MAX));\n'
         '      }\n'
         '    }\n',
         '      motor_cmd[i] = des_percentage * UINT16_MAX;\n'
         '    }\n'),
        ('  if(!set_motors){\n    if(pre_set_motors){\n',
         '  // DShot: motorsSetRatio() re-arms each motor\'s DMA, so every motor needs a\n'
         '  // ratio before every motorsBurstDshot() (each 1 kHz tick), like the PID path.\n'
         '  if(set_motors && use_orig_controller == 0){\n'
         '    for(uint8_t i=0; i<4; i++){\n'
         '      motorsSetRatio(motors[i], clip((float)motor_cmd[i] / motor_cmd_divider, 0, UINT16_MAX));\n'
         '    }\n'
         '  }\n'
         '  if(!set_motors){\n    if(pre_set_motors){\n'),
        ('  if(tick % CONTROL_INTERVAL_MS == 0){\n    update_state(sensors, state);\n',
         '  if(tick % CONTROL_INTERVAL_MS == 0 || force_control_step){\n'
         '    force_control_step = false;\n'
         '    update_state(sensors, state);\n'),
        ('    timestamp_controller_activation = now;\n',
         '    timestamp_controller_activation = now;\n'
         '    {\n'
         '      // Start like a training episode: action history = current motor speeds.\n'
         '      float current[4];\n'
         '      for(uint8_t i=0; i<4; i++){\n'
         '        current[i] = clip(2 * motor_rpm_now(i) / MAX_RPM - 1, -1, 1);\n'
         '      }\n'
         '      rl_tools_reset_action_history(current);\n'
         '      force_control_step = true;\n'
         '    }\n'),
        ('static inline void update_state(',
         'static bool force_control_step = false;\n'
         '\n'
         'static inline float motor_rpm_now(uint8_t i){\n'
         '#ifdef CONFIG_MOTORS_ESC_PROTOCOL_DSHOT_BIDIRECTIONAL\n'
         '  uint16_t rpm = motorsGetRPM(motors[i]);\n'
         '  if(rpm < 40000){\n'
         '    return rpm;\n'
         '  }\n'
         '#endif\n'
         '#ifdef CONFIG_PLATFORM_CF21BL\n'
         '  return RPM_PER_VOLT * ((float)motorsGetRatio(motors[i]) / UINT16_MAX) * pmGetBatteryVoltage() + RPM_OFFSET;\n'
         '#else\n'
         '  return (float)motorsGetRatio(motors[i]) / UINT16_MAX * MAX_RPM;\n'
         '#endif\n'
         '}\n'
         '\n'
         'static inline void update_state('),
    ],
    adapter: [
        ('char* rl_tools_get_checkpoint_name(){',
         'void rl_tools_reset_action_history(const float* actions){\n'
         '#ifdef RL_TOOLS_ACTION_HISTORY\n'
         '    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){\n'
         '        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){\n'
         '            action_history[step_i][action_i] = actions[action_i];\n'
         '        }\n'
         '    }\n'
         '#endif\n'
         '    controller_tick = 0;\n'
         '}\n'
         '\n'
         'char* rl_tools_get_checkpoint_name(){'),
    ],
}
for path, changes in edits.items():
    t = path.read_text()
    for old, new in changes:
        if new in t:
            continue
        assert t.count(old) == 1, f'{path.name}: not found exactly once: {old[:70]!r}'
        t = t.replace(old, new)
    path.write_text(t)
h = header.read_text()
decl = '#ifdef __cplusplus\nextern "C"\n#endif\nvoid rl_tools_reset_action_history(const float* actions);\n'
if decl not in h:
    header.write_text(h + decl)
print('dshot every tick + handover history reset')
EOF

# Target offset (m) from the activation point, sent in the learned-controller
# packet; POSITION mode flies to origin + offset (used for scripted landing).
# The offset starts at zero on every activation.
python3 - "$DST/rl_tools_controller.c" "$DST/rl_tools_controller.h" <<'EOF'
import sys, pathlib
ctrl, header = map(pathlib.Path, sys.argv[1:3])
t = ctrl.read_text()
for old, new in [
    ('void rl_tools_controller_packet_received(){\n',
     'static float target_offset[3] = {0, 0, 0};\n'
     '\n'
     'void rl_tools_controller_set_target_offset(float x, float y, float z){\n'
     '  target_offset[0] = x;\n'
     '  target_offset[1] = y;\n'
     '  target_offset[2] = z;\n'
     '}\n'
     '\n'
     'void rl_tools_controller_packet_received(){\n'),
    ('    case POSITION:\n'
     '      target_pos[0] = origin[0];\n'
     '      target_pos[1] = origin[1];\n'
     '      target_pos[2] = origin[2];\n',
     '    case POSITION:\n'
     '      target_pos[0] = origin[0] + target_offset[0];\n'
     '      target_pos[1] = origin[1] + target_offset[1];\n'
     '      target_pos[2] = origin[2] + target_offset[2];\n'),
    ('    timestamp_controller_activation = now;\n',
     '    timestamp_controller_activation = now;\n'
     '    target_offset[0] = 0;\n'
     '    target_offset[1] = 0;\n'
     '    target_offset[2] = 0;\n'),
]:
    if new in t:
        continue
    assert t.count(old) == 1, f'not found exactly once: {old[:60]!r}'
    t = t.replace(old, new)
ctrl.write_text(t)
h = header.read_text()
decl = 'void rl_tools_controller_set_target_offset(float x, float y, float z);\n'
if decl not in h:
    header.write_text(h.replace('#endif', decl + '\n#endif', 1))
print('target offset')
EOF

cd "$DST"
rm -rf build
make "$DEFCONFIG" >/dev/null
make -j6 2>&1 | grep -E " error|Error |Flash|RAM|CCM" | head -40
ls -la build/*.bin
