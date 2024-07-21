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
import os
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



def save_plots(name,rewards , policy_losss , critic_losss , n):
    agent.save_models(name)
    print(f'saved {name[:-1]}.pth')
    plt.figure()  # Create a new figure for each plot
    if len(rewards) >= n:
        plt.plot(range(len(rewards) - n, len(rewards)), rewards[-n:], label='Episode Rewards')
    else:
        plt.plot(range(len(rewards)), rewards, label='Episode Rewards')
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.legend(loc='lower left', fontsize='small')
    save_path = os.path.join(name, f'{name[-1]}_reward.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure()  # Create a new figure for each plot
    if len(policy_losss) >= n:
        plt.plot(range(len(policy_losss) - n, len(policy_losss)), policy_losss[-n:], label='Episode policy_losss')
    else:
        plt.plot(range(len(policy_losss)), policy_losss, label='Episode policy_losss')
    plt.xlabel('episode')
    plt.ylabel('policy_losss')
    plt.legend(loc='lower left', fontsize='small')
    save_path = os.path.join(name, f'{name}_policy_losss.png')
    plt.savefig(save_path)
    plt.close()

    plt.figure()  # Create a new figure for each plot
    if len(critic_losss) >= n:
        plt.plot(range(len(critic_losss) - n, len(critic_losss)), critic_losss[-n:], label='Episode critic_losss')
    else:
        plt.plot(range(len(critic_losss)), critic_losss, label='Episode critic_losss')
    plt.xlabel('episode')
    plt.ylabel('critic_losss')
    plt.legend(loc='lower left', fontsize='small')
    save_path = os.path.join(name, f'{name}_critic_losss.png')
    plt.savefig(save_path)
    plt.close()


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
    agent =  DDPGagent(num_states=state.shape[0] , num_actions=4 , max_memmory_size=500000)
    #agent.load_models("new_ddpg_20_agent8")
    print(f"load new_ddpg_20_agent8")
    batch_size = 512
    rewards = []
    sample_number = 500000
    omegas =None
    memory = Memory(sample_number)
    percent_outMemory = 80
    memory.load(f"replay_buffer_data{sample_number}.pkl")
    critic_losss = []
    policy_losss = []
    for episode in range(10001):
        print(episode)
        state = env.reset()
        oUNoise.reset()
        episode_reward = 0
        episode_critic_loss = 0
        episode_policy_loss = 0
        if (episode+1)%200==0:
            print(f'new save ddpg_20_agent{(episode+1)//200}.pth')
            save_plots(f'new_ddpg_20_agent{(episode+1)//200}',rewards,policy_losss,critic_losss,200)

        if (episode+1)%1000==0:
            save_plots(f'new_ddpg_100_agent{(episode + 1) // 1000}', rewards, policy_losss, critic_losss, 1000)

        for step in range(env.get_sim_freq() * 31):#31 second while 30 second is the whole episode
            action = agent.get_action(state)
            action_noised = oUNoise.get_action(action , step)
            #action = normalizedEnv.Normalized_to_realspace(action=action_normalized_noised)
            new_state , reward , done , _ = env.step(action_noised)
            agent.memory.push(state,action_noised,reward,new_state,done)
            if agent.memory.size > batch_size:
                critic_loss , policy_loss = agent.update(batch_size , memory , percent_outMemory)
                episode_critic_loss += critic_loss
                episode_policy_loss += policy_loss
            state = new_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward / (step + 1))
        critic_losss.append(episode_critic_loss / (step + 1))
        policy_losss.append(episode_policy_loss / (step + 1))
    save_plots(f'new_ddpg_total_agent{(episode + 1) // 20}', rewards, policy_losss, critic_losss, 600)


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
