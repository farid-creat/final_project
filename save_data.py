import math
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pybullet as p
from RL.Memory import Memory
from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneForcePIDCoefficients, DroneControlTarget
from blt_env.drone import DroneBltEnv
from RL.Normalization import NormalizedEnv
from control.drone_ctrl import DSLPIDControl
import random

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# # Logger class to store drone status (optional).
# from util.data_logger import DroneDataLogger

def make_random_pos():
    x_init = random.uniform(-10, 10)
    y_init = random.uniform(-10, 10)
    z_init = random.uniform(0, 10)

    x_target = random.uniform(-10, 10)
    y_target = random.uniform(-10, 10)
    z_target = random.uniform(0, 10)

    point1 = np.array([x_init,y_init,z_init])
    point2 = np.array([x_target,y_target,z_target])
    distance = np.linalg.norm(point1 - point2)

    while(distance<1):
        x_target = random.uniform(-10, 10)
        y_target = random.uniform(-10, 10)
        z_target = random.uniform(0, 10)
        point2 = np.array([x_target, y_target, z_target])
        distance = np.linalg.norm(point1 - point2)
    point1 = np.array([[x_init,y_init,z_init]])
    return point1, point2

if __name__ == "__main__":

    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB

    init_xyzs , target= make_random_pos()

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        is_real_time_sim=True,
        init_xyzs = init_xyzs,
        init_target= target
    )

    # controller
    pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25]),
        I_for=np.array([.05, .05, .05]),
        D_for=np.array([.2, .2, .5]),
        P_tor=np.array([70000., 70000., 60000.]),
        I_tor=np.array([.0, .0, 500.]),
        D_tor=np.array([20000., 20000., 12000.]),
    )
    number_of_saved_data=1
    ctrl = DSLPIDControl(env, pid_coeff=pid)
    sample_number = 100000
    rpms = np.array([14300, 14300, 14300, 14300])
    memory = Memory(sample_number)
    normalizedEnv = NormalizedEnv(0,env.max_action)
    eposides = 1000
    for j in range(eposides):
        if number_of_saved_data==51:
            break
        state = env.reset()

        if j%100 ==0:
            print(f"episode {j}")
        if memory.size==sample_number:
            memory.save(f"replay_buffer_data{number_of_saved_data}.pkl")
            memory.size=0;
            print(f"saved replay_buffer_data{number_of_saved_data}.pkl")
            number_of_saved_data+=1
        for i in range(200000):
            new_state , reward , done ,kis = env.step(rpms)
            rpms_normalized = normalizedEnv.Normalized_action(rpms)
            memory.push(state,rpms_normalized,reward , new_state , done)
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=kis[0],
                ctrl_target=DroneControlTarget(
                    pos=env.get_target(),
                ),
            )
            state = new_state
            if done:
                break

    # Close the environment
env.close()

