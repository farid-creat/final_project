import torch
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
from RL.DDPG import DDPGagent
from RL.OUNoise import OUNoise
from control.drone_ctrl import DSLPIDControl
import random

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or another interactive backend
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
        is_gui=False,
        phy_mode=phy_mode,
        is_real_time_sim=False,
        init_xyzs = init_xyzs,
        init_target= target
    )
    state = env.reset()
    normalizedEnv = NormalizedEnv(0,env.max_action)
    oUNoise = OUNoise(4,0,env.max_action)
    agent =  DDPGagent(num_states=state.shape[0] , num_actions=4)
    batch_size = 128
    rewards = []
    avg_rewards = []
    sample_number = 100000
    omegas =None
    agent = torch.load('ddpg_agent2.pth', map_location=torch.device('cuda:0'))
    for episode in range(101):
        state = env.reset()
        episode_reward = 0
        for step in range(env.get_sim_freq() * 31):#31 second while 30 second is the whole episode
            action_normalized = agent.get_action(state)
            action_normalized_noised = oUNoise.get_action(action_normalized , step)
            action = normalizedEnv.Normalized_to_realspace(action=action_normalized_noised)
            print(action)
            new_state , reward , done , _ = env.step(action)
            agent.memory.push(state,action_normalized_noised,reward,new_state,done)
            state = new_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.figure()  # Create a new figure for each plot
    plt.plot(range(len(rewards)), rewards, label='Episode Rewards')
    plt.plot(range(len(rewards)), avg_rewards, label='Average 10 episode Rewards')
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.legend(loc='lower left', fontsize='small')
    plt.show()
    #plt.close()
