import math
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pybullet as p

from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneForcePIDCoefficients, DroneControlTarget
from blt_env.drone import DroneBltEnv

from control.drone_ctrl import DSLPIDControl


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# # Logger class to store drone status (optional).
# from util.data_logger import DroneDataLogger

if __name__ == "__main__":

    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB

    init_xyzs = np.array([[0, 0, 1.5]])

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        is_real_time_sim=True,
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

    ctrl = DSLPIDControl(env, pid_coeff=pid)

    rpms = np.array([14300, 14300, 14300, 14300])

    # Initial target position
    pos = np.array([0, 0, 1.0])

    s_target_x = p.addUserDebugParameter("target_x", -2, 2, pos[0])
    s_target_y = p.addUserDebugParameter("target_y", -2, 2, pos[1])
    s_target_z = p.addUserDebugParameter("target_z", 0, 4, pos[2])

    def get_gui_values():
        tg_x = p.readUserDebugParameter(int(s_target_x))
        tg_y = p.readUserDebugParameter(int(s_target_y))
        tg_z = p.readUserDebugParameter(int(s_target_z))
        return tg_x, tg_y, tg_z


    def make_target(path1,j,r):
        return r*math.cos(path1[j]) ,r*math.sin(path1[j]) ,r
    path1  = np.linspace(0, 2 * np.pi, 15)
    j=0

    x=[]
    y=[]
    z=[]

    step_num = 2_000000
    for i in range(step_num):
        kis = env.step(rpms)
        x.append(kis[0].pos[0])
        y.append(kis[0].pos[1])
        z.append(kis[0].pos[2])
        tg_x, tg_y, tg_z = make_target(path1,j,1)

        rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
            control_timestep=env.get_sim_time_step(),
            kin_state=kis[0],
            ctrl_target=DroneControlTarget(
                pos=np.array([tg_x, tg_y, tg_z]),
            ),
        )
        print(rpms)
        if np.linalg.norm(pos_e)<=0.09:
            j+=1
        if j>=15:
            break

    # Close the environment
    env.close()



# Example data
x = np.array(x)
y = np.array(y)
z = np.array(z)

print(f"{x}\n{y}\n{z}")
# Create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data
ax.scatter(x, y, z, c='r', marker='o')  # For points
# ax.plot(x, y, z)  # For lines

# Customize plot
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Plot of Points')

# Save plot to file
plt.savefig('3d_plot.png')