"""Port the learning_to_fly firmware hooks onto crazyflie-firmware 2026.08.

Each edit must match exactly once, otherwise the script aborts.
"""
import os
import pathlib
import sys

FW = pathlib.Path(os.environ.get('FW', pathlib.Path.home() / 'crazyflie-firmware-2026.08'))


def edit(rel, old, new):
    path = FW / rel
    text = path.read_text()
    if new in text:
        print(f'already patched: {rel}')
        return
    if text.count(old) != 1:
        sys.exit(f'ERROR: expected exactly one match in {rel} for:\n{old}')
    path.write_text(text.replace(old, new))
    print(f'patched: {rel}')


# 1. Boot on PID even though the out-of-tree controller is compiled in (Kconfig
#    hides the default-controller choice when CONTROLLER_OOT=y, and auto-select
#    would pick it). The host selects it at runtime with stabilizer.controller.
edit('src/modules/src/controller/controller.c',
     '    #elif defined(CONFIG_CONTROLLER_OOT)\n',
     '    #elif defined(CONFIG_CONTROLLER_OOT) && !defined(RL_TOOLS_CONTROLLER)\n')

# 2. Meta-command 1 = "learned controller may fly" (the dead-man packet).
edit('src/modules/src/crtp_commander.c',
     '#include "crtp.h"\n',
     '#include "crtp.h"\n'
     '\n'
     '#ifdef RL_TOOLS_CONTROLLER\n'
     '#include <string.h>\n'
     '#include "rl_tools_controller.h"\n'
     '#endif\n')
edit('src/modules/src/crtp_commander.c',
     '  metaNotifySetpointsStop = 0,\n  nMetaCommands,\n',
     '  metaNotifySetpointsStop = 0,\n  metaLearnedController = 1,\n  nMetaCommands,\n')
edit('src/modules/src/crtp_commander.c',
     ' /* ---===== packetDecoders array =====--- */\n',
     '/* learnedController meta-command: the learned policy may drive\n'
     ' * the motors for the next 200 ms. It also feeds the commander watchdog with a\n'
     ' * zero-thrust setpoint so the supervisor does not cut the motors while the\n'
     ' * policy flies; the out-of-tree controller ignores that setpoint while it is\n'
     ' * active, and once these packets stop the fallback PID gets zero thrust.\n'
     ' */\n'
     'void learnedControllerDecoder(const void *data, size_t datalen)\n'
     '{\n'
     '#ifdef RL_TOOLS_CONTROLLER\n'
     '  // Optional payload: 3 floats, target offset (m) from the activation point,\n'
     '  // or 6 floats: offset and target velocity (m/s) for a moving target.\n'
     '  if (datalen != 0 && datalen != 3 * sizeof(float) && datalen != 6 * sizeof(float)) {\n'
     '    return;\n'
     '  }\n'
     '  if (datalen != 0) {\n'
     '    float target[6] = {0, 0, 0, 0, 0, 0};\n'
     '    memcpy(target, data, datalen);\n'
     '    rl_tools_controller_set_target_offset(target[0], target[1], target[2]);\n'
     '    rl_tools_controller_set_target_velocity(target[3], target[4], target[5]);\n'
     '  }\n'
     '  static setpoint_t keepAlive;\n'
     '  memset(&keepAlive, 0, sizeof(keepAlive));\n'
     '  commanderSetSetpoint(&keepAlive, COMMANDER_PRIORITY_CRTP);\n'
     '  rl_tools_controller_packet_received();\n'
     '#endif\n'
     '}\n'
     '\n'
     ' /* ---===== packetDecoders array =====--- */\n')
edit('src/modules/src/crtp_commander.c',
     '  [metaNotifySetpointsStop] = notifySetpointsStopDecoder,\n};\n',
     '  [metaNotifySetpointsStop] = notifySetpointsStopDecoder,\n'
     '  [metaLearnedController] = learnedControllerDecoder,\n};\n')

# 3. While the out-of-tree controller is selected it writes the motors itself;
#    do not overwrite them. Other controllers keep the normal motor path, and
#    motors are still stopped whenever the supervisor disallows flight.
edit('src/modules/src/stabilizer.c',
     '      if (supervisorAreMotorsAllowedToRun()) {\n        controlMotors(&control);\n',
     '      if (supervisorAreMotorsAllowedToRun()) {\n'
     '#ifdef RL_TOOLS_CONTROLLER\n'
     '        if (controllerGetType() != ControllerTypeOot) {\n'
     '          controlMotors(&control);\n'
     '        }\n'
     '#else\n'
     '        controlMotors(&control);\n'
     '#endif\n')
