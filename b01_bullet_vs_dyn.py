
import time

import numpy as np
import pybullet as p

from blt_env.wind_visualizer import WindVisualizer
from blt_env.PatternGenerator import PatternGenerator

from util.data_definition import DroneType, PhysicsType
####up
from blt_env.drone import DroneBltEnv

# Logger class to store drone status (optional).
from util.data_logger import DroneDataLogger


if __name__ == "__main__":

    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X

    ## Select a pysical model.
    phy_mode = PhysicsType.PYB  # Dynamics computing by pybullet.
    # phy_mode = PhysicsType.DYN  # Dynamics computing by explicit method.

    init_xyzx = np.array([[0, 0, 1.5]])

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        phy_mode=phy_mode,
        init_xyzs=init_xyzx,
    )

    wind_visualizer = WindVisualizer()
    sim_freq = env.get_sim_freq()
    dp = env.get_drone_properties()
    max_rpm = dp.max_rpm
    hover_rpm = dp.hover_rpm
    patter_generator = PatternGenerator(dp,env)

    rpm = hover_rpm * np.ones(4)
    wind = 0

    sld_r0 = p.addUserDebugParameter(f"rotor 0 rpm", 0, max_rpm, hover_rpm)
    sld_r1 = p.addUserDebugParameter(f"rotor 1 rpm", 0, max_rpm, hover_rpm)
    sld_r2 = p.addUserDebugParameter(f"rotor 2 rpm", 0, max_rpm, hover_rpm)
    sld_r3 = p.addUserDebugParameter(f"rotor 3 rpm", 0, max_rpm, hover_rpm)
    wind_power = p.addUserDebugParameter(f"wind", -3, 3, 0)


    def get_gui_values():
        r0 = p.readUserDebugParameter(int(sld_r0))
        r1 = p.readUserDebugParameter(int(sld_r1))
        r2 = p.readUserDebugParameter(int(sld_r2))
        r3 = p.readUserDebugParameter(int(sld_r3))
        return r0, r1, r2, r3

    def get_gui_wind():
        wind_P = p.readUserDebugParameter(int(wind_power))
        return wind_P


    # # Logger to store drone status (optional).
    # d_log = DroneDataLogger(num_drones=1, logging_freq=sim_freq, logging_duration=0, )

    step_num = 240 * 30
    for i in range(step_num):
        start_time = time.time()
        wind_visualizer.update_wind_direction(env.get_wind_direction())
        direction = "forward"  # Replace this with desired direction (e.g., "backward", "left", "right", "up", "down")
        rpms = patter_generator.pattern_generator(direction)
        new_rpm = patter_generator.adjust_orientation(rpms)
        ki = env.step(np.array(new_rpm), wind)
        #print(f"simulated gps = {env.getSimulated_Gps()[0]}")
        #print(f"pos = {env.get_drones_kinematic_info()[0].pos}")
        #print(f"simulated_IMU: = {env.get_simulated_imu()[0]}")
        rpm = np.array(get_gui_values())
        wind = get_gui_wind()
        # # Logger to store drone status (optional).
        # d_log.log(drone_id=0, time_stamp=(i / sim_freq), kin_state=ki[0], rpm_values=rpm)
        time_step = 1 / sim_freq
        elapsed_time = time.time() - start_time
        sleep_time = max(0, time_step - elapsed_time)
        time.sleep(sleep_time)

    env.close()

    # # Logger to store drone status (optional).
    # d_log.plot()

