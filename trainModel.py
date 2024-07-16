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
        is_gui=True,
        phy_mode=phy_mode,
        is_real_time_sim=True,
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
    memory = Memory(sample_number)
    percent_outMemory = 30

    for episode in range(501):
        random_number = random.randint(1, 50)
        memory.load(f"replay_buffer_data{random_number}.pkl")
        state = env.reset()
        oUNoise.reset()
        episode_reward = 0
        if (episode+1)%20==0:
            torch.save(agent, f'ddpg_agent{(episode+1)//20}.pth')
            print(f'save ddpg_agent{(episode+1)//20}.pth')
            plt.figure()  # Create a new figure for each plot
            if len(rewards) >= 20:
                plt.plot(range(len(rewards) - 20, len(rewards)), rewards[-20:], label='Episode Rewards')
                plt.plot(range(len(rewards) - 20, len(rewards)), avg_rewards[-20:], label='Average 10 episode Rewards')
            else:
                plt.plot(range(len(rewards)), rewards, label='Episode Rewards')
                plt.plot(range(len(rewards)), avg_rewards, label='Average 10 episode Rewards')
            plt.xlabel('episode')
            plt.ylabel('rewards')
            #plt.xlim( (len(rewards)-20, len(rewards) ))  # Set the x-axis range from 20 to 40
            plt.legend(loc='lower left', fontsize='small')
            plt.savefig(f'ddpg_agent{(episode + 1) // 20}.png')
            plt.close()
        if (episode+1)%100==0:
            plt.figure()  # Create a new figure for each plot
            if len(rewards) >= 100:
                plt.plot(range(len(rewards) - 100, len(rewards)), rewards[-100:], label='Episode Rewards')
                plt.plot(range(len(rewards) - 100, len(rewards)), avg_rewards[-100:], label='Average 10 episode Rewards')
            else:
                plt.plot(range(len(rewards)), rewards, label='Episode Rewards')
                plt.plot(range(len(rewards)), avg_rewards, label='Average 10 episode Rewards')
            plt.xlabel('episode')
            plt.ylabel('rewards')
            #plt.xlim( (len(rewards)-20, len(rewards) ))  # Set the x-axis range from 20 to 40
            plt.legend(loc='lower left', fontsize='small')
            plt.savefig(f'total_100_ddpg_agent{(episode + 1) // 20}.png')
            plt.close()
        for step in range(env.get_sim_freq() * 31):#31 second while 30 second is the whole episode
            action_normalized = agent.get_action(state)
            action_normalized_noised = oUNoise.get_action(action_normalized , step)
            action = normalizedEnv.Normalized_to_realspace(action=action_normalized_noised)
            new_state , reward , done , _ = env.step(action)
            agent.memory.push(state,action_normalized_noised,reward,new_state,done)
            if agent.memory.size > batch_size:
                agent.update(batch_size , memory , percent_outMemory)
            state = new_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    torch.save(agent, 'ddpg_agent_final.pth')
    plt.figure()
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.savefig('rewards_30%_500e.png')
    plt.show()


"""
    agent = torch.load('ddpg_agent.pth', map_location=torch.device('cuda:0'))
    state = np.load('state.npy')
    print(f'state:\n {state}')
    action2 =agent.get_action(state)
    print(f'action2: \n{action2}')
    action = np.load('action.npy')
    print(f'action: \n{action}')
"""

"""
    eposides = 1000
    for j in range(eposides):
        state = env.reset()

        if j%100 ==0:
            print(f"episode {j}")
        if memory.size==sample_number:
            memory.save(f"replay_buffer_data{j}.pkl")
            memory.size=0;
            print(f"saved replay_buffer_data{j}.pkl")
        for i in range(200000):
            new_state , reward , done ,kis = env.step(rpms)
            memory.push(state,rpms,reward , new_state , done)
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


"""
