
import pickle
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
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or another interactive backend
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
from DQN.dqn_agent import DQNAgent

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class paterGenerator:
    def __init__(self):
        self.pos_to_reach = None

    def resrt(self):
        self.pos_to_reach = None

    def patern_generator(self,action_num, ctrl, info):  # 0 no op 1 forward 2 back 3 left 4 right
        if self.pos_to_reach is None:
            self.pos_to_reach = info[0].pos
        distance = np.linalg.norm(info[0].pos - self.pos_to_reach)

        if action_num == 0: #0 no op
            if distance>1:
                self.pos_to_reach = [info[0].pos[0] ,info[0].pos[1],self.pos_to_reach[2] ]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                pos=np.array([self.pos_to_reach[0],self.pos_to_reach[1],self.pos_to_reach[2]]),
                ),
                )
            return rpms
        elif action_num == 1:#1 forward
            if distance<1:
                self.pos_to_reach = [info[0].pos[0] ,info[0].pos[1],self.pos_to_reach[2] ]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                pos=np.array([self.pos_to_reach[0],self.pos_to_reach[1] + 2,self.pos_to_reach[2]]),
                ),
                )
            return rpms
        elif action_num == 2:#2 back
            if distance<1:
                self.pos_to_reach = [info[0].pos[0] ,info[0].pos[1],self.pos_to_reach[2] ]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                pos=np.array([self.pos_to_reach[0],self.pos_to_reach[1] - 2,self.pos_to_reach[2]]),
                ),
                )
            return rpms
        elif action_num == 3:#3 left
            if distance<1:
                self.pos_to_reach = [info[0].pos[0] ,info[0].pos[1],self.pos_to_reach[2] ]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                pos=np.array([self.pos_to_reach[0] - 2,self.pos_to_reach[1],self.pos_to_reach[2]]),
                ),
                )
            return rpms
        elif action_num == 4:#4 right
            if distance<1:
                self.pos_to_reach = [info[0].pos[0] ,info[0].pos[1],self.pos_to_reach[2] ]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                pos=np.array([self.pos_to_reach[0] + 2,self.pos_to_reach[1],self.pos_to_reach[2]]),
                ),
                )
            return rpms

def fill_memory(env, dqn_agent , action_number):
    state = env.reset()
    patergenerator.resrt();
    omegas = np.array([14300, 14300, 14300, 14300])
    next_state, reward, done, info = env.step(omegas)
    while dqn_agent.memory.len() < dqn_agent.memory.capacity:
        action  = random.randint(0, action_number-1)
        ##########test
        omegas = patergenerator.patern_generator(action, ctrl, info)
        ##########
        next_state, reward, done, info = env.step(omegas)
        dqn_agent.memory.store(state=state,
                               action=action,
                               next_state=next_state,
                               reward=reward,
                               done=done)
        if done:
            state = env.reset()




# # Logger class to store drone status (optional).
# from util.data_logger import DroneDataLogger

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
    point2 = np.array([x_target,y_target,z_init])
    distance = np.linalg.norm(point1 - point2)

    while(distance<1):
        x_target = random.uniform(-10, 10)
        y_target = random.uniform(-10, 10)
        z_target = random.uniform(0, 10)
        point2 = np.array([x_target, y_target, z_init])
        distance = np.linalg.norm(point1 - point2)
    point1 = np.array([[x_init,y_init,z_init]])
    return point1, point2

def compute_moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')




def save_plots(agent,name,rewards , Q_Learning_Loss , n ,window_size=100):
    agent.save_models(name)
    print(f'saved {name[:-1]}.pth')
    plt.figure()  # Create a new figure for each plot
    if len(rewards) >= n:
        plt.plot(range(len(rewards) - n, len(rewards)), rewards[-n:], label='Episode Rewards')
        if len(rewards) >= window_size:
            moving_avg = compute_moving_average(rewards[-(n+window_size-1):], window_size)
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='orange')
    else:
        plt.plot(range(len(rewards)), rewards, label='Episode Rewards')
        if len(rewards) >= window_size:
            moving_avg = compute_moving_average(rewards[-(n+window_size-1):], window_size)
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='orange')
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.legend(loc='lower left', fontsize='small')
    save_path = os.path.join(name, f'{name[-1]}_reward.png')
    plt.savefig(save_path)
    plt.close()
    plt.figure()  # Create a new figure for each plot
    if len(Q_Learning_Loss) >= n:
        plt.plot(range(len(Q_Learning_Loss) - n, len(Q_Learning_Loss)), Q_Learning_Loss[-n:], label='Episode Q_Learning_Loss')
        if len(Q_Learning_Loss) >= window_size:
            moving_avg = compute_moving_average(Q_Learning_Loss[-(n+window_size-1):], window_size)
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='orange')
    else:
        plt.plot(range(len(Q_Learning_Loss)), Q_Learning_Loss, label='Episode Q_Learning_Loss')
        if len(Q_Learning_Loss) >= window_size:
            moving_avg = compute_moving_average(Q_Learning_Loss[-(n+window_size-1):], window_size)
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='orange')
    plt.xlabel('episode')
    plt.ylabel('Q_Learning_Loss')
    plt.legend(loc='lower left', fontsize='small')
    save_path = os.path.join(name, f'{name[-1]}_Q_Learning_Loss.png')
    plt.savefig(save_path)
    plt.close()

    save_path = os.path.join(name, f'{name[-1]}_Q_Learning_Loss_array.npy')
    Q_Learning_Loss_array = np.array(Q_Learning_Loss)
    np.save(save_path, Q_Learning_Loss_array)

    save_path = os.path.join(name, f'{name[-1]}_rewards_array.npy')
    rewards_array = np.array(rewards)
    np.save(save_path, rewards_array)

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
        is_real_time_sim=False,
        init_xyzs = init_xyzs,
        init_target= target
    )
    state = env.reset()

    pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25]),
        I_for=np.array([.05, .05, .05]),
        D_for=np.array([.2, .2, .5]),
        P_tor=np.array([70000., 70000., 60000.]),
        I_tor=np.array([.0, .0, 500.]),
        D_tor=np.array([20000., 20000., 12000.]),
    )
    ctrl = DSLPIDControl(env, pid_coeff=pid)
    patergenerator = paterGenerator()
    action_num=5
    dqn_agent = DQNAgent(device,state.shape[0], action_num)

    #agent.load_models("/content/drive/MyDrive/new_ddpg_400_agent42")
    batch_size = 128
    rewards = []
    Q_Learning_Loss = []
    best_score = -np.inf
    #rewards = np.load('/content/drive/MyDrive/new_ddpg_400_agent42/2_rewards_array.npy').tolist()
    #critic_losss = np.load('/content/drive/MyDrive/new_ddpg_400_agent42/2_critic_losss_array.npy').tolist()
    #policy_losss = np.load('/content/drive/MyDrive/new_ddpg_400_agent42/2_policy_losss_array.npy').tolist()
    sample_number = 10000
    update_frequency=100
    omegas = np.array([14300, 14300, 14300, 14300])
    #fill_memory(env, dqn_agent , action_num)
    #print('Memory filled. Current capacity: ', dqn_agent.memory.len())
    if True:
        for episode in range(100001):
            print(episode)
            state = env.reset()
            next_state, reward, done, info = env.step(omegas)
            episode_reward = 0
            episode_Q_Learning_Loss = 0
            if (episode+1)%10==0:
                print(f'./new save dqn_10_agent{(episode+1)//10}.pth')
                save_plots(f'./new_dqn_10_agent{(episode+1)//10}',rewards,Q_Learning_Loss,10)

            if (episode+1)%100==0:
                save_plots(f'./new_dqn_100_agent{(episode + 1) // 100}', rewards, Q_Learning_Loss, 100)

            for step in range(env.get_sim_freq() * 31):#31 second while 30 second is the whole episode
                action = dqn_agent.select_action(state)
                ####################test
                omegas = patergenerator.patern_generator(action,ctrl,info)
                #######################


                next_state, reward, done, info = env.step(omegas)

                dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward
                episode_Q_Learning_Loss = dqn_agent.learn(batchsize=batch_size)
                if episode % update_frequency == 0:
                    dqn_agent.update_target_net()
                if done or step == env.get_sim_freq() * 31 -1:
                    break
            dqn_agent.update_epsilon()
            rewards.append(episode_reward/(step+1))
            Q_Learning_Loss.append(episode_Q_Learning_Loss/(step+1))
            current_avg_score = np.mean(rewards[-100:])  # moving average of last 100 episodes

            if current_avg_score >= best_score:
                dqn_agent.save_model('{}/dqn_model'.format("best_model"))
                best_score = current_avg_score



        save_plots(f'new_ddpg_total_agent{100000}', rewards, Q_Learning_Loss, 100000)

    else:
        dqn_agent = DQNAgent(device,
                             state.shape[0],
                             action_num)
        dqn_agent.load_model('./new_dqn_2000_agent5')
        for episode in range(100001):
            print(episode)
            state = env.reset()

            patergenerator.resrt()
            next_state, reward, done, info = env.step(omegas)
            episode_reward = 0
            episode_Q_Learning_Loss = 0
            for step in range(env.get_sim_freq() * 31):#31 second while 30 second is the whole episode
                action = dqn_agent.select_action(state)
                ####################test
                omegas = patergenerator.patern_generator(action,ctrl,info)
                #######################


                next_state, reward, done, info = env.step(omegas)
                print(f'distance in episode {episode} : {next_state[0]}')
                print(f'reward:  {reward}')


"""

def test(env, dqn_agent, num_test_eps, seed, results_basepath, render=True):
    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.reset()
        while not done:

            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))


            dqn_agent = DQNAgent(device,
                                 env.observation_space.shape[0],
                                 env.action_space.n,
                                 discount=args.discount,
                                 eps_max=0.0,  # epsilon values should be zero to ensure no exploration in testing mode
                                 eps_min=0.0,
                                 eps_decay=0.0,
                                 train_mode=False)
            dqn_agent.load_model('{}/dqn_model'.format(args.results_folder))

            test(env=env, dqn_agent=dqn_agent, num_test_eps=args.num_test_eps, seed=seed,
                 results_basepath=args.results_folder, render=args.render)

            env.close()"""

