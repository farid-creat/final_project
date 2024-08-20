import matplotlib

matplotlib.use('Agg')

from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneForcePIDCoefficients, DroneControlTarget
from blt_env.drone import DroneBltEnv

from dueling_ddqn.DuelingDeepQNetwork import Agent
from control.drone_ctrl import DSLPIDControl
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # or another interactive backend
import numpy as np
import os
import random


class PaterGenerator:
    def __init__(self):
        self.pos_to_reach = None
        self.prev_command = -1

    def reset(self):
        self.pos_to_reach = None
        self.prev_command = -1

    def patern_generator(self, action_num, ctrl, info):  # 0 no op 1 forward 2 back 3 left 4 right
        rpms = None
        if self.pos_to_reach is None:
            self.pos_to_reach = info[0].pos
        distance = np.linalg.norm(info[0].pos - self.pos_to_reach)
        if action_num == 0:  # 0 no op
            if distance > 1 and self.prev_command!=0:
                self.pos_to_reach = [info[0].pos[0], info[0].pos[1], self.pos_to_reach[2]]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                    pos=np.array([self.pos_to_reach[0], self.pos_to_reach[1], self.pos_to_reach[2]]),
                ),
            )
        elif action_num == 1:  # 1 forward
            if distance < 1:
                self.pos_to_reach = [self.pos_to_reach[0], self.pos_to_reach[1] +1, self.pos_to_reach[2]]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                    pos=np.array([self.pos_to_reach[0], self.pos_to_reach[1], self.pos_to_reach[2]]),
                ),
            )
        elif action_num == 2:  # 2 back
            if distance < 1:
                self.pos_to_reach = [info[0].pos[0], info[0].pos[1] -1, self.pos_to_reach[2]]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                    pos=np.array([self.pos_to_reach[0], self.pos_to_reach[1], self.pos_to_reach[2]]),
                ),
            )
        elif action_num == 3:  # 3 left
            if distance < 1:
                self.pos_to_reach = [self.pos_to_reach[0] -1, self.pos_to_reach[1], self.pos_to_reach[2]]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                    pos=np.array([self.pos_to_reach[0], self.pos_to_reach[1], self.pos_to_reach[2]]),
                ),
            )
        elif action_num == 4:  # 4 right
            if distance < 1:
                self.pos_to_reach = [self.pos_to_reach[0]+1, self.pos_to_reach[1], self.pos_to_reach[2]]
            rpms, pos_e, _ = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=info[0],
                ctrl_target=DroneControlTarget(
                    pos=np.array([self.pos_to_reach[0], self.pos_to_reach[1], self.pos_to_reach[2]]),
                ),
            )
        self.prev_command = action_num
        return rpms


def fill_memory(env, dqn_agent, action_number , patergenerator):
    state = env.reset()
    patergenerator.reset();
    omegas = np.array([14300, 14300, 14300, 14300])
    next_state, reward, done, info = env.step(omegas)
    while not dqn_agent.memory.is_full():
 #       print(f'target : {env.init_target}\n'
 #             f'start : {env.start_pos}')
        x_error = state[1]
        y_error = state[2]
        if abs(x_error) > abs(y_error):
            if x_error>0:
                action=4
            else:
                action=3
        else:
            if y_error>0:
                action=1
            else:
                action=2
        #action = random.randint(0, action_number - 1)
        ##########test
        omegas = patergenerator.patern_generator(action, ctrl, info)
        ##########
        next_state, reward, done, info = env.step(omegas)


        dqn_agent.store_transition(state=state,
                               action=action,
                               reward=reward,
                               state_=next_state,
                               done=done)
        state = next_state
        if done:
            state = env.reset()
            patergenerator.reset()
            omegas = np.array([14300, 14300, 14300, 14300])
            next_state, reward, done, info = env.step(omegas)

def make_random_pos():
    x_init = random.uniform(-10, 10)
    y_init = random.uniform(-10, 10)
    z_init = random.uniform(0, 10)

    x_target = random.uniform(-10, 10)
    y_target = random.uniform(-10, 10)
    #z_target = random.uniform(0, 10)

    point1 = np.array([x_init, y_init, z_init])
    point2 = np.array([x_target, y_target, z_init])
    distance = np.linalg.norm(point1 - point2)

    while (distance < 2):
        x_target = random.uniform(-10, 10)
        y_target = random.uniform(-10, 10)
        z_target = random.uniform(0, 10)
        point2 = np.array([x_target, y_target, z_init])
        distance = np.linalg.norm(point1 - point2)
    point1 = np.array([[x_init, y_init, z_init]])
    return point1, point2

def save_plots(agent, name, rewards, Q_Learning_Loss, q_values, n):
    agent.save_models(name)
    print(f'saved {name[:-1]}.pth')
    plt.figure()  # Create a new figure for each plot
    if len(Q_Learning_Loss) >= n:
        plt.plot(range(len(q_values) - n, len(q_values)), q_values[-n:],
                 label='Episode Q_Learning_Loss')
    else:
        plt.plot(range(len(q_values)), q_values, label='Episode Q_Learning_Loss')
    plt.xlabel('episode')
    plt.ylabel('q_values')
    plt.legend(loc='lower left', fontsize='small')
    save_path = os.path.join(name, f'{name[-1]}_q_values.png')
    plt.savefig(save_path)
    plt.close()



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
    if len(Q_Learning_Loss) >= n:
        plt.plot(range(len(Q_Learning_Loss) - n, len(Q_Learning_Loss)), Q_Learning_Loss[-n:],
                 label='Episode Q_Learning_Loss')
    else:
        plt.plot(range(len(Q_Learning_Loss)), Q_Learning_Loss, label='Episode Q_Learning_Loss')
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


    save_path = os.path.join(name, f'{name[-1]}_q_values_array.npy')
    q_values_array = np.array(q_values)
    np.save(save_path, q_values_array)


if __name__ == "__main__":

    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB

    init_xyzs, target = make_random_pos()

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=False,
        phy_mode=phy_mode,
        is_real_time_sim=False,
        init_xyzs=init_xyzs,
        init_target=target
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
    patergenerator = PaterGenerator()
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=1.0, lr=1e-4,
                  input_dims=[state.shape[0]], n_actions=5, mem_size=50000,
                  eps_min=0.1, batch_size=64, eps_dec=1e-5, replace=1000)
    if load_checkpoint:
        agent.load_models()
    #agent.load_models("new_dqn_30_agent10")
    #agent.load_models("best_model/dqn_model")
    rewards = []
    q_values = []
    episode_mean_reward = []
    Q_Learning_Loss = []
    best_score = -np.inf
    #rewards = np.load('new_dqn_30_agent10/0_rewards_array.npy').tolist()
    #Q_Learning_Loss = np.load('new_dqn_30_agent10/0_Q_Learning_Loss_array.npy').tolist()
    #q_values = np.load('new_dqn_30_agent10/0_q_values_array.npy').tolist()
    omegas = np.array([14300, 14300, 14300, 14300])
    fill_memory(env, agent , 5 , patergenerator)
    print('Memory filled.')
    if True:
        for episode in range(5001):

            patergenerator.reset()
            state = env.reset()
            next_state, reward, done, info = env.step(omegas)
            episode_reward = 0
            q_value = 0
            episode_Q_Learning_Loss = 0
            step=0
            for step in range(env.get_sim_freq() * 11):
                action = agent.choose_action(state)
                ####################test
                q_value = agent.predict_q_value(state , action)
                omegas = patergenerator.patern_generator(action, ctrl, info)
                #######################

                state_, reward, done, info = env.step(omegas)
                episode_reward += reward
                agent.store_transition(state, action, reward, state_, done)
                episode_Q_Learning_Loss = agent.learn()
                #print(state[0])
                state = state_
                if done or step == env.get_sim_freq() * 11 - 1:
                    break
            Q_Learning_Loss.append(episode_Q_Learning_Loss)
            q_values.append(q_value)
            rewards.append(reward)
            episode_mean_reward.append(episode_reward/step)
            current_avg_score = np.mean(rewards[-30:])  # moving average of last 100 episodes
            print(f'episode : {episode} \n'
                  f'Q_Learning_Loss : {episode_Q_Learning_Loss}\n'
                  f'q_values : {q_value}\n'
                  f'reward : {reward}\n'
                  f'mean reward : {current_avg_score}\n'
                  f'---------------------------------------------')

            if (episode + 1) % 30 == 0:
                print(f'./new save dqn_30_agent{(episode + 1) // 30}.pth')
                save_plots(agent, f'./new_dqn_30_agent{(episode + 1) // 30}', rewards, Q_Learning_Loss ,q_values , 31)
                agent.reset_memory()
                fill_memory(env, agent, 5, patergenerator)
            if (episode + 1) % 100 == 0:
                save_plots(agent , f'./new_dqn_100_agent{(episode + 1) // 100}', rewards, Q_Learning_Loss ,q_values, 101)


            if current_avg_score >= best_score:
                agent.save_models('{}/dqn_model'.format("best_model"))
                best_score = current_avg_score

        save_plots(f'new_ddpg_total_agent{1000}', rewards, Q_Learning_Loss, 1000)

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
            for step in range(env.get_sim_freq() * 31):  # 31 second while 30 second is the whole episode
                action = dqn_agent.select_action(state)
                ####################test
                omegas = patergenerator.patern_generator(action, ctrl, info)
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
