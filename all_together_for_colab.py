import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import os
import torch.nn.functional as F
import random
from collections import deque
import pickle
import numpy as np
from abc import ABCMeta, abstractmethod

from typing import Optional, List
import re
import codecs

import xml.etree.ElementTree as XmlEt
import sys
import os

from typing import Optional, List, Tuple, Union
import time
import random

from logging import getLogger, NullHandler

import numpy as np
import pybullet as p
import pybullet_data


import gym
from gym import Env
from gym.spaces import Box, Tuple

from logging import getLogger, NullHandler, StreamHandler, INFO, DEBUG
import sys
import pybullet as p
import numpy as np



from typing import List, Optional
from dataclasses import dataclass, field
###up
from enum import IntEnum, Enum

import numpy as np


@dataclass(init=False, frozen=True)
class SharedConstants(object):
    AGGR_PHY_STEPS: int = 5
    # for default setting
    SUCCESS_MODEL_FILE_NAME: str = 'success_model.zip'
    DEFAULT_OUTPUT_DIR_PATH: str = './result'
    DEFAULT_DRONE_FILE_PATH: str = './assets/drone_x_01.urdf'
    DEFAULT_DRONE_TYPE_NAME: str = 'x'


class DroneType(IntEnum):
    OTHER = 0
    QUAD_PLUS = 1
    QUAD_X = 2


class PhysicsType(Enum):
    """Physics implementations enumeration class."""
    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Update with an explicit model of the dynamics
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash


class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"  # RPMS
    FORCE = "for"  # Desired thrust and torques (force)
    PID = "pid"  # PID control
    VEL = "vel"  # Velocity input (using PID control)
    TUN = "tun"  # Tune the coefficients of a PID controller
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs
    ONE_D_FORCE = "one_d_for"  # 1D (identical input to all motors) with desired thrust and torques
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control


class RlAlgorithmType(Enum):
    """Reinforcement Learning type enumeration class."""
    A2C = 'a2c'
    PPO = 'ppo'
    SAC = 'sac'
    TD3 = 'td3'
    DDPG = 'ddpg'


class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"  # Kinematics information (pose, linear and angular velocities)
    RGB = "rgb"  # RGB camera capture in each drone's POV


@dataclass(frozen=True)
class DroneForcePIDCoefficients(object):
    P_for: np.ndarray = None  # force
    I_for: np.ndarray = None
    D_for: np.ndarray = None
    P_tor: np.ndarray = None  # torque
    I_tor: np.ndarray = None
    D_tor: np.ndarray = None


@dataclass
class DroneKinematicsInfo(object):
    pos: np.ndarray = np.zeros(3)  # position
    quat: np.ndarray = np.zeros(4)  # quaternion
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    vel: np.ndarray = np.zeros(3)  # linear velocity
    ang_vel: np.ndarray = np.zeros(3)  # angular velocity


@dataclass
class DroneControlTarget(object):
    pos: np.ndarray = np.zeros(3)  # position
    vel: np.ndarray = np.zeros(3)  # linear velocity
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    rpy_rates: np.ndarray = np.zeros(3)  # roll, pitch, and yaw rates


@dataclass
class DroneProperties(object):
    """
    The drone parameters.

    kf : It is the proportionality constant for thrust, and thrust is proportional to the square of rotation speed.
    km : It is the proportionality constant for torque, and torque is proportional to the square of rotation speed.

    """
    type: int = 1  # The drone type 0:OTHER 1:QUAD_PLUS 2:QUAD_X
    g: float = 9.8  # gravity acceleration
    m: Optional[float] = None  # Mass of the drone.
    l: Optional[float] = None  # Length of the arm of the drone's rotor mount.
    thrust2weight_ratio: Optional[float] = None
    ixx: float = 0
    iyy: float = 0
    izz: float = 0
    J: np.ndarray = np.array([])
    J_inv: np.ndarray = np.array([])
    kf: Optional[float] = None  # The proportionality constant for thrust.
    km: Optional[float] = None  # The proportionality constant for torque.
    collision_h: Optional[float] = None
    collision_r: Optional[float] = None
    collision_shape_offsets: List[float] = field(default_factory=list)
    collision_z_offset: float = None
    max_speed_kmh: Optional[float] = None
    gnd_eff_coeff: Optional[float] = None
    prop_radius: Optional[float] = None
    drag_coeff_xy: float = 0
    drag_coeff_z: float = 0
    drag_coeff: np.ndarray = None
    dw_coeff_1: Optional[float] = None
    dw_coeff_2: Optional[float] = None
    dw_coeff_3: Optional[float] = None
    # compute after determining the drone type
    gf: float = 0  # gravity force
    hover_rpm: float = 0
    max_rpm: float = 0
    max_thrust: float = 0
    max_xy_torque = 0
    max_z_torque = 0
    grand_eff_h_clip = 0  # The threshold height for ground effects.
    A: np.ndarray = np.array([])
    inv_A: np.ndarray = np.array([])
    B_coeff: np.ndarray = np.array([])
    Mixer: np.ndarray = np.ndarray([])  # use for PID control

    def __post_init__(self):
        self.J = np.diag([self.ixx, self.iyy, self.izz])
        self.J_inv = np.linalg.inv(self.J)
        self.collision_z_offset = self.collision_shape_offsets[2]
        self.drag_coeff = np.array([self.drag_coeff_xy, self.drag_coeff_xy, self.drag_coeff_z])
        self.gf = self.g * self.m
        self.hover_rpm = np.sqrt(self.gf / (4 * self.kf))
        self.max_rpm = np.sqrt((self.thrust2weight_ratio * self.gf) / (4 * self.kf))
        self.max_thrust = (4 * self.kf * self.max_rpm ** 2)
        if self.type == 2:  # QUAD_X
            self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm ** 2) / np.sqrt(2)
            self.A = np.array([[1, 1, 1, 1], [1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2), -1 / np.sqrt(2)],
                               [-1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2)], [-1, 1, -1, 1]])
            self.Mixer = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])
        elif self.type in [0, 1]:  # QUAD_PLUS, OTHER
            self.max_xy_torque = (self.l * self.kf * self.max_rpm ** 2)
            self.A = np.array([[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
            self.Mixer = np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])
        self.max_z_torque = 2 * self.km * self.max_rpm ** 2
        self.grand_eff_h_clip = 0.25 * self.prop_radius * np.sqrt(
            (15 * self.max_rpm ** 2 * self.kf * self.gnd_eff_coeff) / self.max_thrust)
        self.inv_A = np.linalg.inv(self.A)
        self.B_coeff = np.array([1 / self.kf, 1 / (self.kf * self.l), 1 / (self.kf * self.l), 1 / self.km])



class FileHandler(metaclass=ABCMeta):
    def __init__(self, encoding: str = 'utf-8'):
        self._codec = encoding

    @staticmethod
    def suffix_check(suffix_list: List[str], file_name: str) -> bool:
        suffix_list = [s.strip('.') for s in suffix_list]
        suffix = '|'.join(suffix_list).upper()
        pattern_str = r'\.(' + suffix + r')$'
        if not re.search(pattern_str, file_name.upper()):
            mes = f"""
            Suffix of the file name should be '{suffix}'.
            Current file name ： {file_name}
            """
            print(mes)
            return False
        return True

    def read(self, file_name: str):
        try:
            with codecs.open(file_name, 'r', self._codec) as srw:
                return self.read_handling(srw)
        except FileNotFoundError:
            print(f"{file_name} can not be found ...")
        except OSError as e:
            print(f"OS error occurred trying to read {file_name}")
            print(e)

    def write(self, data, file_name: str):
        try:
            with codecs.open(file_name, 'w', self._codec) as srw:
                self.write_handling(data, srw)
        except OSError as e:
            print(f"OS error occurred trying to write {file_name}")
            print(e)

    @abstractmethod
    def read_handling(self, srw: codecs.StreamReaderWriter):
        raise NotImplementedError

    @abstractmethod
    def write_handling(self, data, srw: codecs.StreamReaderWriter):
        raise NotImplementedError


class DroneUrdfAnalyzer(FileHandler):
    def __init__(self, codec: str = 'utf-8'):
        super().__init__(codec)

    def read_handling(self, srw: codecs.StreamReaderWriter):
        return XmlEt.parse(srw)

    def write_handling(self, data, srw: codecs.StreamReaderWriter):
        pass

    def parse(self, urdf_file: str, drone_type: int = 0, g: float = 9.8) -> Optional[DroneProperties]:
        # check the file suffix
        if not self.suffix_check(['.urdf'], urdf_file):
            return None

        et = self.read(urdf_file)

        if et is None:
            return None

        root = et.getroot()
        prop = root[0]
        link = root[1]  # first link -> link name="base_link"

        dataset = DroneProperties(
            type=drone_type,
            g=g,
            m=float(link[0][1].attrib['value']),
            l=float(prop.attrib['arm']),
            thrust2weight_ratio=float(prop.attrib['thrust2weight']),
            ixx=float(link[0][2].attrib['ixx']),
            iyy=float(link[0][2].attrib['iyy']),
            izz=float(link[0][2].attrib['izz']),
            kf=float(prop.attrib['kf']),
            km=float(prop.attrib['km']),
            collision_h=float(link[2][1][0].attrib['length']),
            collision_r=float(link[2][1][0].attrib['radius']),
            collision_shape_offsets=[float(s) for s in link[2][0].attrib['xyz'].split(' ')],
            max_speed_kmh=float(prop.attrib['max_speed_kmh']),
            gnd_eff_coeff=float(prop.attrib['gnd_eff_coeff']),
            prop_radius=float(prop.attrib['prop_radius']),
            drag_coeff_xy=float(prop.attrib['drag_coeff_xy']),
            dw_coeff_1=float(prop.attrib['dw_coeff_1']),
            dw_coeff_2=float(prop.attrib['dw_coeff_2']),
            dw_coeff_3=float(prop.attrib['dw_coeff_3']),
        )

        return dataset




class Memory:
    def __init__(self , max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.size = 0
    def push(self , state , action , reward , next_state , done):
        experience = (state , action , reward , next_state , done)
        self.buffer.append(experience)
        if self.size<self.max_size:
            self.size+=1
    def sample (self, batch_size):
        actual_size = min(batch_size ,self.size)
        batch = random.sample(self.buffer, actual_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    def len(self):
        return self.size

    def save(self, file_path):
        """Save the buffer and size to a file."""
        with open(file_path, 'wb') as f:
            # Save the buffer and size to the file
            pickle.dump((self.buffer, self.size), f)

    def load(self, file_path):
        """Load the buffer and size from a file."""
        with open(file_path, 'rb') as f:
            # Load the buffer and size from the file
            self.buffer, self.size = pickle.load(f)




class OUNoise():
    def __init__(self , action_dim , action_low, action_high , mu=0.0 , theta = 0.15 , max_sigma = 0.3 , min_sigma = 0.3 , decay_period = 100_000):
        self.state = None
        self.mu = mu
        self.theta = theta
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.sigma = self.max_sigma
        self.reset()
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self , action , t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1,t/self.decay_period)
        return np.clip(action + ou_state , self.action_low , self.action_high)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_number=4):
        super(Actor, self).__init__()

        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_number):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization

        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)  # Linear layer
            if x.size(0) > 1:  # Check if batch size > 1
                x = self.hidden_layers[i + 1](x)  # BatchNorm layer
            x = F.relu(x)  # ReLU activation
        x = self.output_layer(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_number=4):
        super(Critic, self).__init__()

        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_number):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x = F.relu(self.input_layer(x))

        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)  # Linear layer
            x = self.hidden_layers[i + 1](x)  # BatchNorm layer
            x = F.relu(x)  # ReLU activation
        x = self.output_layer(x)
        return x




class DDPGagent:
    def __init__(self , num_states , num_actions , hidden_size = 512 , actor_learning = 1e-2 , critic_learning = 1e-2
                 , gamma = 0.95 , tau = 1e-2 , max_memmory_size = 100000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f'running on {device}')
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(self.num_states , hidden_size , self.num_actions).to(device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(device)

        self.critic = Critic(self.num_states + num_actions, hidden_size, 1).to(device)
        self.critic_target = Critic(self.num_states + num_actions, hidden_size, 1).to(device)

        for target_params , params in zip(self.actor_target.parameters() , self.actor.parameters()):
            target_params.data.copy_(params.data)
        for target_params , params in zip(self.critic_target.parameters() , self.critic.parameters()):
            target_params.data.copy_(params.data)

        self.memory = Memory(max_memmory_size)

        self.critic_criterion = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters() , lr=actor_learning)
        self.critic_optimizer = optim.Adam(self.critic.parameters() , lr= critic_learning)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).to(self.device).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0,:]
        return action

    def update(self , batch_size , out_memory = None , percent_outMemory = 0):
        amount_from_out = batch_size*percent_outMemory//100
        states , actions , rewards , next_states , dones = self.memory.sample(batch_size - amount_from_out)
        if out_memory is not None:
            states_out, actions_out, rewards_out, next_states_out, dones_out = out_memory.sample(amount_from_out)
            states = np.concatenate((states, states_out), axis=0)
            actions = np.concatenate((actions, actions_out), axis=0)
            rewards = np.concatenate((rewards, rewards_out), axis=0)
            next_states = np.concatenate((next_states, next_states_out), axis=0)
            dones = np.concatenate((dones, dones_out), axis=0)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        Qvals = self.critic.forward(states,actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states,next_actions.detach())
#        rewards = rewards.view(next_Q.size())
#        dones = dones.view(next_Q.size())

        Qprime = []
        for i in range(batch_size):
            Qprime.append(rewards[i] + self.gamma*next_Q[i] * (1-dones[i]))
        Qprime = torch.tensor(Qprime).to(self.device)
        Qprime = Qprime.view(Qvals.size())
        #Qprime = rewards + self.gamma * next_Q * (1 - dones)
        critic_loss = self.critic_criterion(Qvals , Qprime)
        policy_loss = -self.critic.forward(states , self.actor.forward(states)).mean()

        #print(f'Critic Loss: {critic_loss}, Policy Loss: {policy_loss}')
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for target_params , params in zip(self.actor_target.parameters() , self.actor.parameters()):
            target_params.data.copy_(params.data*self.tau + target_params.data * (1-self.tau))
        for target_params , params in zip(self.critic_target.parameters() , self.critic.parameters()):
            target_params.data.copy_(params.data * self.tau + target_params.data * (1 - self.tau))

        return critic_loss.detach().cpu() , policy_loss.detach().cpu()


    def save_models(self, path):
        # Create directories if they don't exist
        os.makedirs(path, exist_ok=True)

        # Save actor and critic models
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(path, 'actor_target.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.critic_target.state_dict(), os.path.join(path, 'critic_target.pth'))

    def load_models(self, path):
        # Load actor and critic models
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target.pth'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), map_location=self.device))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, 'critic_target.pth'), map_location=self.device))




def real_time_step_synchronization(sim_counts, start_time, time_step):
    """Syncs the stepped simulation with the wall-clock.

    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/utils/utils.py

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    sim_counts : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    time_step : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if time_step > .04 or sim_counts % (int(1 / (24 * time_step))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (sim_counts * time_step):
            time.sleep(time_step * sim_counts - elapsed)


def load_drone_properties(file_path: str, d_type: DroneType) -> DroneProperties:
    file_analyzer = DroneUrdfAnalyzer()
    return file_analyzer.parse(file_path, int(d_type))


class DroneBltEnv(Env):

    def __init__(
            self,
            urdf_path: str,
            d_type: DroneType = DroneType.QUAD_PLUS,
            phy_mode: PhysicsType = PhysicsType.PYB,
            sim_freq: int = 240,
            aggr_phy_steps: int = 1,
            num_drones: int = 1,
            is_gui: bool = True,
            is_real_time_sim: bool = False,
            init_xyzs: Optional[Union[List, np.ndarray]] = None,
            init_target=None,
            init_rpys: Optional[Union[List, np.ndarray]] = None,
    ):
        """
        Parameters
        ----------
        urdf_path : The drone *.URDF file path.
        d_type : Specifies the type of drone to be loaded from the *.URDF file.
        phy_mode : Specifies the type of physics simulation for PyBullet.
        sim_freq : Specifies the frequency of the PyBullet step simulations.
        aggr_phy_steps : The number of physics steps within one call to `self.step()`.
                        The frequency of the control action is changed by the aggr_phy_steps.
        num_drones : Number of drones to be loaded.
        is_gui : Whether to start PyBullet in GUI mode.
        """
        # super().__init__(is_gui=is_gui)
        self._is_gui = is_gui
        self._drone_type = d_type
        self._urdf_path = urdf_path
        self._physics_mode = phy_mode
        self._wind_direction = np.random.rand(3)
        self._previous_linear_velocity = [0, 0, 0]
        self._dp = load_drone_properties(self._urdf_path, self._drone_type)
        ###############################RL###########################
        self.init_target = init_target
        self.near_to_target_time = 0
        self.near_to_target_threshold_time = 5
        self.distance_threshold = 1
        self.episode_threshold = 30
        self.near_ground_time = 0
        self.near_ground_time_threshold = 0
        self.max_action = self._dp.max_rpm
        self.prev_distance = 100000;
        ################################RL###########################

        # print("--------------------------------------------------")
        # self.printout_drone_properties()
        # print("--------------------------------------------------")
        # PyBullet simulation settings.
        self._num_drones = num_drones
        self._aggr_phy_steps = aggr_phy_steps
        self._g = self._dp.g
        self._sim_freq = sim_freq
        self._sim_time_step = 1. / self._sim_freq
        self._is_realtime_sim = is_real_time_sim  # add wait time in step().

        # Initialization position of the drones.
        if init_xyzs is None:
            self._init_xyzs = np.vstack([
                np.array([x * 4 * self._dp.l for x in range(self._num_drones)]),
                np.array([y * 4 * self._dp.l for y in range(self._num_drones)]),
                np.ones(self._num_drones) * (self._dp.collision_h / 2 - self._dp.collision_z_offset + 0.1),
            ]).transpose().reshape(self._num_drones, 3)
        else:
            assert init_xyzs.ndim == 2, f"'init_xyzs' should has 2 dimension. current dims are {init_xyzs.ndim}."
            self._init_xyzs = np.array(init_xyzs)
        assert self._init_xyzs.shape[0] == self._num_drones, f""" Initialize position error.
        Number of init pos {self._init_xyzs.shape[0]} vs number of drones {self._num_drones}."""

        if init_rpys is None:
            self._init_rpys = np.zeros((self._num_drones, 3))
        else:
            assert init_rpys.ndim == 2, f"'init_rpys' should has 2 dimension. current dims are {init_rpys.ndim}."
            self._init_rpys = np.array(init_rpys)
        assert self._init_rpys.shape[0] == self._num_drones, f""" Initialize roll, pitch and yaw error.
        Number of init rpy {self._init_rpys.shape[0]} vs number of drones {self._num_drones}."""

        # Simulation status.
        self._sim_counts = 0
        self._last_rpm_values = np.zeros((self._num_drones, 4))
        ''' 
        The 'DroneKinematicInfo' class is simply a placeholder for the following information.
            pos : position
            quat : quaternion
            rpy : roll, pitch and yaw
            vel : linear velocity
            ang_vel : angular velocity
        '''
        self._kis = [DroneKinematicsInfo() for _ in range(self._num_drones)]

        if self._physics_mode == PhysicsType.DYN:
            self._rpy_rates = np.zeros((self._num_drones, 3))

        # PyBullet environment.
        self._client = p.connect(p.GUI) if self._is_gui else p.connect(p.DIRECT)
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF('plane.urdf')

        # Load drones.
        self._drone_ids = np.array([
            p.loadURDF(
                self._urdf_path,
                self._init_xyzs[i, :],
                p.getQuaternionFromEuler(self._init_rpys[i, :]),
            ) for i in range(self._num_drones)])
        print(len(self._drone_ids))
        # Update the information before running the simulations.
        self.update_drones_kinematic_info()
        ###################################RL######################################

        ###################################RL######################################
        # Start measuring time.
        self._start_time = time.time()
        print("---------------------------------------------------\n init")

    def get_target(self):
        return self.init_target

    def get_sim_time_step(self) -> float:
        return self._sim_time_step

    def get_sim_counts(self) -> int:
        return self._sim_counts

    def get_drone_properties(self) -> DroneProperties:
        return self._dp

    def get_drones_kinematic_info(self) -> List[DroneKinematicsInfo]:
        return self._kis

    def get_aggr_phy_steps(self) -> int:
        return self._aggr_phy_steps

    def get_sim_freq(self) -> int:
        return self._sim_freq

    def get_num_drones(self) -> int:
        return self._num_drones

    def get_wind_direction(self) -> int:
        return self._wind_direction

    def get_new_wind(self, wind_power) -> List[int]:
        wind_change = np.random.uniform(-0.1, 0.1, size=3)
        self._wind_direction += wind_change
        self._wind_direction /= np.linalg.norm(self._wind_direction)
        return self._wind_direction * wind_power

    def get_last_rpm_values(self) -> np.ndarray:
        return self._last_rpm_values

    def make_random_pos(self):
        x_init = random.uniform(-10, 10)
        y_init = random.uniform(-10, 10)
        z_init = random.uniform(0, 10)

        x_target = random.uniform(-10, 10)
        y_target = random.uniform(-10, 10)
        z_target = random.uniform(0, 10)

        point1 = np.array([x_init, y_init, z_init])
        point2 = np.array([x_target, y_target, z_target])
        distance = np.linalg.norm(point1 - point2)

        while (distance < 1):
            x_target = random.uniform(-10, 10)
            y_target = random.uniform(-10, 10)
            z_target = random.uniform(0, 10)
            point2 = np.array([x_target, y_target, z_target])
            distance = np.linalg.norm(point1 - point2)
        point1 = np.array([[x_init, y_init, z_init]])
        return point1, point2

    def refresh_bullet_env(self):
        start_pos, target = self.make_random_pos()
        """
        Refresh the PyBullet simulation environment.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `self.reset()` function.

        """
        self._sim_counts = 0
        self._last_rpm_values = np.zeros((self._num_drones, 4))
        self._kis = [DroneKinematicsInfo() for _ in range(self._num_drones)]
        if self._physics_mode == PhysicsType.DYN:
            self._rpy_rates = np.zeros((self._num_drones, 3))

        # Set PyBullet's parameters.
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        self._plane_id = p.loadURDF('plane.urdf')

        # Load drones.
        self._drone_ids = np.array([
            p.loadURDF(
                self._urdf_path,
                start_pos[0, :],
                p.getQuaternionFromEuler(self._init_rpys[i, :]),
            ) for i in range(self._num_drones)])

        self.update_drones_kinematic_info()

        # Reset measuring time.
        self._start_time = time.time()
        #########################RL#######################
        self.init_target = target
        self.near_to_target_time = 0
        self.near_ground_time = 0
        self.prev_distance = np.linalg.norm(self.init_target - self._kis[0].pos)
        new_state = np.concatenate(
            [np.array([self.prev_distance]), self.init_target - self._kis[0].pos, self._kis[0].rpy, self._kis[0].vel,
             self._kis[0].ang_vel,
             np.array([self.near_to_target_time]), np.array([self.near_ground_time])])
        #################################################
        return new_state

    def get_drone(self):
        return self._drone_ids[0]

    def update_drones_kinematic_info(self):
        for i in range(self._num_drones):
            pos, quat = p.getBasePositionAndOrientation(
                bodyUniqueId=self._drone_ids[i],
                physicsClientId=self._client,
            )
            rpy = p.getEulerFromQuaternion(quat)
            vel, ang_vel = p.getBaseVelocity(
                bodyUniqueId=self._drone_ids[i],
                physicsClientId=self._client,
            )
            # after test make it to work with multi drones
            self._previous_linear_velocity = self._kis[i].vel
            self._kis[i] = DroneKinematicsInfo(
                pos=np.array(pos),
                quat=np.array(quat),
                rpy=np.array(rpy),
                vel=np.array(vel),
                ang_vel=np.array(ang_vel),
            )

    def close(self) -> None:
        if p.isConnected() != 0:
            p.disconnect(physicsClientId=self._client)

    def reset(self):
        if p.isConnected() != 0:
            p.resetSimulation(physicsClientId=self._client)
            return self.refresh_bullet_env()

    def step(self, rpm_values, wind=0):
        """
        Parameters
        ----------
        wind : the power of wind
        rpm_values : Multiple arrays with 4 values as a pair of element.
                    Specify the rotational speed of the four rotors of each drone.
        """
        rpm_values = self.check_values_for_rotors(rpm_values)

        for _ in range(self._aggr_phy_steps):
            '''
            Update and store the drones kinematic info the same action value of "rpm_values" 
            for the number of times specified by "self._aggr_phy_steps".
            '''
            if self._aggr_phy_steps > 1 and self._physics_mode in [
                PhysicsType.DYN,
                PhysicsType.PYB_GND,
                PhysicsType.PYB_DRAG,
                PhysicsType.PYB_DW,
                PhysicsType.PYB_GND_DRAG_DW
            ]:
                self.update_drones_kinematic_info()

            # step the simulation
            for i in range(self._num_drones):
                self.physics(
                    rpm_values[i, :],
                    i,
                    self._last_rpm_values[i, :],
                    wind,
                )

            # In the case of the explicit solution technique, 'p.stepSimulation()' is not used.
            if self._physics_mode != PhysicsType.DYN:
                p.stepSimulation(physicsClientId=self._client)

            # Save the last applied action (for compute e.g. drag)
            self._last_rpm_values = rpm_values
        ####----------------------------------------
        self.prev_distance = np.linalg.norm(self.init_target - self._kis[0].pos)
        ###------------------------------------------
        # Update and store the drones kinematic information
        self.update_drones_kinematic_info()

        # Advance the step counter
        self._sim_counts = self._sim_counts + (1 * self._aggr_phy_steps)

        # Synchronize the step interval with real time.
        if self._is_realtime_sim:
            real_time_step_synchronization(self._sim_counts, self._start_time, self._sim_time_step)
        time_step = 1 / self._sim_freq

        #########################RL#######################
        new_state = np.concatenate(
            [np.array([self.prev_distance]), self.init_target - self._kis[0].pos, self._kis[0].rpy, self._kis[0].vel,
             self._kis[0].ang_vel,
             np.array([self.near_to_target_time]), np.array([self.near_ground_time])])
        distance = np.linalg.norm(self.init_target - self._kis[0].pos)
        reward = -(distance ** 2) + (distance - self.prev_distance)
        done = False
        # Check if the roll angle is approximately ±180 degrees (±π radians)
        if (np.abs(self._kis[0].rpy[0]) > np.pi / 2 or np.abs(self._kis[0].rpy[1]) > np.pi / 2) and self._kis[0].pos[2] < 0.1:  ##upside down on the land
            reward = -(distance + 2) ** 5
            done = True

        if distance <= self.distance_threshold:
            self.near_to_target_time += time_step
            reward = 1
        else:
            self.near_to_target_time = 0  # should stay near to target constantly

        if self.near_to_target_time > self.near_to_target_threshold_time:
            reward = 1
            done = True

        if self._sim_counts * time_step > self.episode_threshold:
            done = True

        if self._kis[0].pos[2] < 0.1 and self.init_target[2] > 0.2:
            self.near_ground_time += time_step
            reward = -(distance + 2) ** 3
        else:
            self.near_ground_time = 0

        if self.near_ground_time > self.near_ground_time_threshold:
            done = True;
            reward = -(distance + 2) ** 5

        #################################################

        return new_state, reward, done, self._kis

    def check_values_for_rotors(self, rpm_values: np.ndarray) -> np.ndarray:
        """
        Check that 'rpm_values', which specifies the rotation speed of the 4-rotors, are in the proper form.
        Also, if possible, modify 'rpm_values' to the appropriate form.


        Parameters and Returns
        ----------
        rpm_values : Multiple arrays with 4 values as a pair of element.
                    Specify the rotational speed of the four rotors of each drone.
        """
        cls_name = self.__class__.__name__
        assert isinstance(rpm_values, np.ndarray), f"Invalid rpm_values type is used on {cls_name}."
        assert rpm_values.ndim == 1 or rpm_values.ndim == 2, f"Invalid dimension of rpm_values is used on {cls_name}."
        if rpm_values.ndim == 1:
            assert len(rpm_values) == 4, f"Invalid number of elements were used for rpm_values on {cls_name}."
            ''' e.g.
            while, a = [100, 200, 300, 400]
            then, np.tile(a, (3, 1)) -> [[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]]
            '''
            rpm_values = np.tile(rpm_values, (self._num_drones, 1))
        elif rpm_values.ndim == 2:
            assert rpm_values.shape[1] == 4, f"Invalid number of elements were used for rpm_values on {cls_name}."
            rpm_values = np.reshape(rpm_values, (self._num_drones, 4))
        return rpm_values

    def physics(
            self,
            rpm: np.ndarray,
            nth_drone: int,
            last_rpm: Optional[np.ndarray],
            wind: int,
    ) -> None:
        """
        The type of physics simulation will be selected according to 'self._physics_mode'.

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        last_rpm : Previous specified value.
        """

        def pyb(rpm, nth_drone: int, last_rpm=None, wind: int = 0):
            self.apply_rotor_physics(rpm, nth_drone, wind)

        def dyn(rpm, nth_drone: int, last_rpm=None):
            self.apply_dynamics(rpm, nth_drone)

        def pyb_gnd(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_ground_effect(rpm, nth_drone)

        def pyb_drag(rpm, nth_drone: int, last_rpm):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_drag(last_rpm, nth_drone)  # apply last data

        def pyb_dw(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_downwash(nth_drone)

        def pyb_gnd_drag_dw(rpm, nth_drone: int, last_rpm):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_ground_effect(rpm, nth_drone)
            self.apply_drag(last_rpm, nth_drone)  # apply last data
            self.apply_downwash(nth_drone)

        def other(rpm, nth_drone: int, last_rpm):
            print(f"In {self.__class__.__name__}, invalid physic mode key.")

        phy_key = self._physics_mode.value

        key_dict = {
            'pyb': pyb,
            'dyn': dyn,
            'pyb_gnd': pyb_gnd,
            'pyb_drag': pyb_drag,
            'pyb_dw': pyb_dw,
            'pyb_gnd_drag_dw': pyb_gnd_drag_dw,
        }
        return key_dict.get(phy_key, other)(rpm, nth_drone, last_rpm, wind)

    def apply_rotor_physics(self, rpm: np.ndarray, nth_drone: int, wind: int):
        """
        Apply the individual thrusts and torques generated by the motion of the four rotors.
        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."
        wind_direction_power = self.get_new_wind(wind)

        forces = (np.array(rpm) ** 2) * self._dp.kf
        torques = (np.array(rpm) ** 2) * self._dp.km
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(
                objectUniqueId=self._drone_ids[nth_drone],
                linkIndex=i,  # link id of the rotors.
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self._client,
            )
        p.applyExternalForce(
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass.
            forceObj=wind_direction_power,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )
        p.applyExternalTorque(
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass.
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def apply_ground_effect(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply ground effect.

            Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."

        ''' getLinkState()
        computeLinkVelocity : 
            If set to 1, the Cartesian world velocity will be computed and returned.
        computeForwardKinematics : 
            If set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
        '''
        link_states = np.array(
            p.getLinkStates(
                bodyUniqueId=self._drone_ids[nth_drone],
                linkIndices=[0, 1, 2, 3, 4],
                computeLinkVelocity=1,
                computeForwardKinematics=1,
                physicsClientId=self._client,
            ),
            dtype=object,
        )

        # Simple, per-propeller ground effects.
        prop_heights = np.array(
            [link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2], link_states[3, 0][2]])
        prop_heights = np.clip(prop_heights, self._dp.grand_eff_h_clip, np.inf)
        gnd_effects = np.array(rpm) ** 2 * self._dp.kf * self._dp.gnd_eff_coeff * (
                self._dp.prop_radius / (4 * prop_heights)) ** 2

        ki = self._kis[nth_drone]
        if np.abs(ki.rpy[0]) < np.pi / 2 and np.abs(ki.rpy[1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(
                    objectUniqueId=self._drone_ids[nth_drone],
                    linkIndex=i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self._client,
                )

    def apply_drag(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply drag force.
        抗力を適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on the the system identification in (Forster, 2015).

            Chapter 4 Drag Coefficients
            http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """

        # Rotation matrix of the base.
        ki = self._kis[nth_drone]
        base_rot = np.array(p.getMatrixFromQuaternion(ki.quat)).reshape(3, 3)
        # Simple draft model applied to the center of mass
        drag_factors = -1 * self._dp.drag_coeff * np.sum(2 * np.pi * np.array(rpm) / 60)
        drag = np.dot(base_rot, drag_factors * np.array(ki.vel))
        p.applyExternalForce(
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass.
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def apply_downwash(self, nth_drone: int):
        """
        Apply downwash.
        ダウンウオッシュ（吹き下ろし）を適用

        The aerodynamic caused by the motion of the rotor blade's airfoil during the process of generating lift.
        Interactions between multiple drones.

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        ki_d = self._kis[nth_drone]
        for i in range(self._num_drones):
            ki_i = self._kis[i]
            delta_z = ki_i.pos[2] - ki_d.pos[2]
            delta_xy = np.linalg.norm(np.array(ki_i.pos[0:2]) - np.array(ki_d.pos[0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self._dp.dw_coeff_1 * (self._dp.prop_radius / (4 * delta_z)) ** 2
                beta = self._dp.dw_coeff_2 * delta_z + self._dp.dw_coeff_3
                downwash = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(
                    objectUniqueId=self._drone_ids[nth_drone],
                    linkIndex=4,  # link id of the center of mass.
                    forceObj=downwash,
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self._client,
                )

    def apply_dynamics(self, rpm: np.ndarray, nth_drone: int):
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."

        # Current state.
        ki = self._kis[nth_drone]
        pos = ki.pos
        quat = ki.quat
        rpy = ki.rpy
        vel = ki.vel
        rpy_rates = self._rpy_rates[nth_drone]  # angular velocity
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Compute thrust and torques.
        thrust, x_torque, y_torque, z_torque = self.rpm2forces(rpm)
        thrust = np.array([0, 0, thrust])

        thrust_world_frame = np.dot(rotation, thrust)
        forces_world_frame = thrust_world_frame - np.array([0, 0, self._dp.gf])

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self._dp.J, rpy_rates))
        rpy_rates_deriv = np.dot(self._dp.J_inv, torques)  # angular acceleration
        no_pybullet_dyn_accs = forces_world_frame / self._dp.m

        # Update state.
        vel = vel + self._sim_time_step * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self._sim_time_step * rpy_rates_deriv
        pos = pos + self._sim_time_step * vel
        rpy = rpy + self._sim_time_step * rpy_rates

        # Set PyBullet state
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self._drone_ids[nth_drone],
            posObj=pos,
            ornObj=p.getQuaternionFromEuler(rpy),
            physicsClientId=self._client,
        )

        # Note: the base's velocity only stored and not used.
        p.resetBaseVelocity(
            objectUniqueId=self._drone_ids[nth_drone],
            linearVelocity=vel,
            angularVelocity=[-1, -1, -1],  # ang_vel not computed by DYN
            physicsClientId=self._client,
        )

        # Store the roll, pitch, yaw rates for the next step
        # ki.rpy_rates = rpy_rates
        self._rpy_rates[nth_drone] = rpy_rates

    def rpm2forces(self, rpm: np.ndarray) -> Tuple:
        forces = np.array(rpm) ** 2 * self._dp.kf
        thrust = np.sum(forces)
        z_torques = np.array(rpm) ** 2 * self._dp.km
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self._drone_type == DroneType.QUAD_X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
        elif self._drone_type in [DroneType.QUAD_PLUS, DroneType.OTHER]:
            x_torque = (forces[1] - forces[3]) * self._dp.l
            y_torque = (-forces[0] + forces[2]) * self._dp.l
        return thrust, x_torque, y_torque, z_torque

    def printout_drone_properties(self) -> None:
        mes = f"""
        {self.__class__.__name__} loaded parameters from the .urdf :
        {self._urdf_path}
        m: {self._dp.m}
        l: {self._dp.l}
        ixx: {self._dp.ixx}
        iyy: {self._dp.iyy}
        izz: {self._dp.izz}
        kf: {self._dp.kf}
        km: {self._dp.km}
        J: {self._dp.J}
        thrust2weight_ratio: {self._dp.thrust2weight_ratio}
        max_speed_kmh: {self._dp.max_speed_kmh}
        gnd_eff_coeff: {self._dp.gnd_eff_coeff}
        prop_radius: {self._dp.prop_radius}
        drag_coeff_xy: {self._dp.drag_coeff_xy}
        drag_z_coeff: {self._dp.drag_coeff_z}
        dw_coeff_1: {self._dp.dw_coeff_1}
        dw_coeff_2: {self._dp.dw_coeff_2}
        dw_coeff_3: {self._dp.dw_coeff_3}
        gf: {self._dp.gf}
        hover_rpm: {self._dp.hover_rpm}
        max_rpm: {self._dp.max_rpm}
        max_thrust: {self._dp.max_thrust}
        max_xy_torque: {self._dp.max_xy_torque}
        max_z_torque: {self._dp.max_z_torque}
        grand_eff_h_clip: {self._dp.grand_eff_h_clip}
        grand_eff_h_clip: {self._dp.grand_eff_h_clip}
        A: {self._dp.A}
        B_coeff: {self._dp.B_coeff}
        Mixer: {self._dp.Mixer}
        """
        print(mes)

    def getSimulated_Gps(self, noise_std_dev: int = 0.1):
        # Simulate GPS reading with Gaussian noise
        noise = np.random.normal(scale=noise_std_dev, size=3)
        gps_reading = []
        for i in range(self._num_drones):
            pos = self._kis[i].pos
            pos = pos + noise
            gps_reading.append(pos)
        return gps_reading

    def get_simulated_imu(self, noise_std_dev=0.01):
        imu_reading = []
        time_step = 1 / self.get_sim_freq()
        for i in range(self._num_drones):
            orientation = self._kis[i].quat + np.random.normal(scale=noise_std_dev, size=4)
            angular_velocity = self._kis[i].ang_vel + np.random.normal(scale=noise_std_dev, size=3)
            #            linear_acceleration = [(self._kis[i].vel[j] - self._previous_linear_velocity[j]) / time_step for j in range(3)] + np.random.normal(scale=noise_std_dev, size=3)
            linear_acceleration = self._kis[i].vel + np.random.normal(scale=noise_std_dev, size=3)

            imu_reading.append((orientation, angular_velocity, linear_acceleration))
        return imu_reading




class WindVisualizer:
    def __init__(self, scale=0.5):
        # Create a cylinder representing the wind direction
        self.cylinder_id = p.createVisualShape(p.GEOM_CYLINDER,
                                               radius=0.05 * scale,
                                               length=1.0 * scale,
                                               rgbaColor=[1, 0, 0, 1],  # Red color
                                               visualFramePosition=[0, 0, 0.5 * scale])

        # Create a cone representing the wind direction
        self.cone_id = p.createVisualShape(p.GEOM_MESH,
                                            fileName="cone.obj",  # Cone mesh file
                                            rgbaColor=[1, 0, 0, 1],  # Red color
                                            meshScale=[0.1 * scale, 0.1 * scale, 0.3 * scale],  # Scale of the cone
                                            visualFramePosition=[0.5 * scale, 0, 0])  # Position of the cone

        # Combine the cylinder and cone into a single compound shape
        self.compound_id = p.createMultiBody(baseVisualShapeIndex=self.cylinder_id,
                                              basePosition=[0, 0, 0],
                                              baseOrientation=[0, 0, 0, 1])

    def update_wind_direction(self, wind_direction):
        # Convert wind direction vector to quaternion
        orientation = p.getQuaternionFromEuler([wind_direction[0], wind_direction[1], wind_direction[2]])

        # Update the position and orientation of the compound shape
        p.resetBasePositionAndOrientation(self.compound_id, [0, 0, 0], orientation)








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

    urdf_file = '/content/drive/MyDrive/drone_x_01.urdf'
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
    oUNoise = OUNoise(4,0,env.max_action)
    agent =  DDPGagent(num_states=state.shape[0] , num_actions=4 , max_memmory_size=500000)
    #agent.load_models("new_ddpg_20_agent8")
    batch_size = 1024
    rewards = []
    sample_number = 500000
    omegas =None
    memory = Memory(sample_number)
    percent_outMemory = 80
    memory.load("/content/drive/MyDrive/replay_buffer_data500000.pkl")
    critic_losss = []
    policy_losss = []
    for episode in range(10001):
        print(episode)
        state = env.reset()
        oUNoise.reset()
        episode_reward = 0
        episode_critic_loss = 0
        episode_policy_loss = 0
        if (episode+1)%100==0:
            print(f'/content/drive/MyDrive/new save ddpg_100_agent{(episode+1)//100}.pth')
            save_plots(f'/content/drive/MyDrive/new_ddpg_100_agent{(episode+1)//100}',rewards,policy_losss,critic_losss,100)

        if (episode+1)%1000==0:
            save_plots(f'/content/drive/MyDrive/new_ddpg_100_agent{(episode + 1) // 1000}', rewards, policy_losss, critic_losss, 1000)

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
    save_plots(f'new_ddpg_total_agent{(episode + 1) // 20}', rewards, policy_losss, critic_losss, 10000)