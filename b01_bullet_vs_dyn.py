
import time
from typing import Optional, List, Tuple, Union
import numpy as np
import pybullet as p
import pybullet_data
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

    rpm = np.array([hover_rpm,hover_rpm,hover_rpm,hover_rpm])
    wind = 0

    forward = p.addUserDebugParameter(f"forward", 0, 1, 0)
    backward = p.addUserDebugParameter(f"backward", 0, 1, 0)
    left = p.addUserDebugParameter(f"left", 0, 1, 0)
    right = p.addUserDebugParameter(f"right", 0, 1, 0)
    up = p.addUserDebugParameter(f"up", 0, 1, 0)
    down = p.addUserDebugParameter(f"down", 0, 1, 0)
    wind_power = p.addUserDebugParameter(f"wind", -3, 3, 0)

    def detect_direction(direction:List):
        if direction[0] >=0.5:
            return "forward"
        elif direction[1] >=0.5:
            return "backward"
        elif direction[2] >=0.5:
            return "left"
        elif direction[3] >=0.5:
            return "right"
        elif direction[4] >=0.5:
            return "up"
        elif direction[5] >=0.5:
            return "down"
        return "hover"



    def get_gui_values():
        r0 = p.readUserDebugParameter(forward)
        r1 = p.readUserDebugParameter(backward)
        r2 = p.readUserDebugParameter(left)
        r3 = p.readUserDebugParameter(right)
        r4 = p.readUserDebugParameter(up)
        r5 = p.readUserDebugParameter(down)
        return r0, r1, r2, r3 ,r4 , r5

    def get_gui_wind():
        wind_P = p.readUserDebugParameter(int(wind_power))
        return wind_P


    # # Logger to store drone status (optional).
    # d_log = DroneDataLogger(num_drones=1, logging_freq=sim_freq, logging_duration=0, )

    step_num = 240 * 30
    for i in range(step_num):
        start_time = time.time()
        wind_visualizer.update_wind_direction(env.get_wind_direction())
        gui_values = np.array(get_gui_values())
        direction = detect_direction(gui_values)
        #direction = "left"  # Replace this with desired direction (e.g., "backward", "left", "right", "up", "down")
        rpms = patter_generator.pattern_generator(direction)
        ki = env.step(np.array(rpms), wind)
        #print(f"simulated gps = {env.getSimulated_Gps()[0]}")
        #print(f"pos = {env.get_drones_kinematic_info()[0].pos}")
        #print(f"simulated_IMU: = {env.get_simulated_imu()[0]}")
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

