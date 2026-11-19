
#include "../../multirotor.h"

// Crazyflie 2.1 Brushless (41.3 g with battery, Lighthouse and Flow deck).
// Sources:
//   - mass: scale
//   - thrust constant, motor time constant: sysid flights with the stock PID
//     (eRPM telemetry; hover 15,460 RPM, tau 0.055-0.065 s)
//   - inertia, arm length, torque/thrust ratio: Busetto et al., arXiv 2512.14450
//     (45 g configuration; inertia scaled by mass)
// Layout and rotor/torque directions follow crazy_flie.h.
#define L2F_CF21BL_ARM 0.03535

namespace rl_tools::rl::environments::multirotor::parameters::dynamics{
    template<typename T, typename TI, typename REWARD_FUNCTION>
    constexpr typename ParametersBase <T, TI, TI(4), REWARD_FUNCTION>::Dynamics crazyflie_brushless = {
            // Rotor positions
            {
                    { L2F_CF21BL_ARM, -L2F_CF21BL_ARM, 0},
                    {-L2F_CF21BL_ARM, -L2F_CF21BL_ARM, 0},
                    {-L2F_CF21BL_ARM,  L2F_CF21BL_ARM, 0},
                    { L2F_CF21BL_ARM,  L2F_CF21BL_ARM, 0},
            },
            // Rotor thrust directions
            {
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0, 1},
            },
            // Rotor torque directions
            {
                    {0, 0, -1},
                    {0, 0, +1},
                    {0, 0, -1},
                    {0, 0, +1},
            },
            // thrust constants: thrust [N] = c2 * rpm^2 (3.86e-8 N/(rad/s)^2)
            {
                    0,
                    0,
                    4.233e-10
            },
            // torque constant (rotor torque / thrust)
            0.0021,
            // mass vehicle
            0.0413,
            // gravity
            {0, 0, -9.81},
            // J
            {
                    {2.198e-5, 0, 0},
                    {0, 2.198e-5, 0},
                    {0, 0, 2.969e-5}
            },
            // J_inv
            {
                    {45495.9, 0, 0},
                    {0, 45495.9, 0},
                    {0, 0, 33681.4}
            },
            // T, RPM time constant
            0.06,
            // action limit (RPM)
            {0, 24000},
    };
}
