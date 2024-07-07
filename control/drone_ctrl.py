import sys
import os

# Get the current script's directory path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Check if the parent directory is not already in the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from scipy.optimize import nnls


import math
from typing import Tuple
from scipy.spatial.transform import Rotation
import numpy as np
import pybullet as p

from blt_env.drone import DroneBltEnv
from util.data_definition import DroneForcePIDCoefficients
from util.data_definition import DroneKinematicsInfo, DroneControlTarget

class DSLPIDControl():
    """
    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/control/DSLPIDControl.py

    """

    def __init__(
            self,
            env: DroneBltEnv,
            pid_coeff: DroneForcePIDCoefficients,
    ):

        # PID constant parameters
        self._PID = pid_coeff
        self._time_step = env.get_sim_time_step()
        self._dp = env.get_drone_properties()
        self._g = self._dp.g
        self._mass = self._dp.m
        ########use for sustainability##########
        self.max_rpms = self._dp.max_rpm
        #############################


        self._kf = self._dp.kf
        self._km = self._dp.km
        self._Mixer = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])#np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])
        self._gf = self._dp.gf

        # Initialized PID control variables
        self._last_rpy = np.zeros(3)
        self._last_pos_e = np.zeros(3)
        self._integral_pos_e = np.zeros(3)
        self._last_rpy_e = np.zeros(3)
        self._integral_rpy_e = np.zeros(3)
    def compute_control_from_kinematics(
            self,
            control_timestep: float,
            kin_state: DroneKinematicsInfo,
            ctrl_target: DroneControlTarget,
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone.
        Parameters
        ----------
        control_timestep: The time step at which control is computed.
        kin_state
        ctrl_target
        """
        return self.compute_control(
            control_timestep=control_timestep,
            current_position=kin_state.pos,
            current_quaternion=kin_state.quat,
            current_velocity=kin_state.vel,
            current_ang_velocity=kin_state.ang_vel,
            target_position=ctrl_target.pos,
            target_velocity=ctrl_target.vel,
            target_rpy=ctrl_target.rpy,
            target_rpy_rates=ctrl_target.rpy_rates,
        )

    def compute_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            current_ang_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray = np.zeros(3),
            target_rpy: np.ndarray = np.zeros(3),
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep: The time step at which control is computed.
        current_position: (3,1)-shaped array of floats containing the current position.
        current_quaternion: (4,1)-shaped array of floats containing the current orientation as a quaternion.
        current_velocity: (3,1)-shaped array of floats containing the current velocity.
        current_ang_velocity: (3,1)-shaped array of floats containing the current angular velocity.
        target_position: (3,1)-shaped array of floats containing the desired position.
        < The following are optionals. >
        target_velocity: (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_rpy: (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates: (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.
        """

        thrust, computed_target_rpy, pos_e = self.dsl_pid_position_control(
            control_timestep,
            current_position,
            current_quaternion,
            current_velocity,
            target_position,
            target_velocity,
            target_rpy,
        )
        rpm = self.dsl_pid_attitude_control(
            control_timestep,
            thrust,
            current_quaternion,
            computed_target_rpy,
            target_rpy_rates,
        )
        cur_rpy = p.getEulerFromQuaternion(current_quaternion)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def dsl_pid_position_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray,
            target_rpy: np.ndarray,
    ) -> Tuple:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3)
        pos_e = target_position - current_position
        vel_e = target_velocity - current_velocity
        pos_e = np.clip(pos_e, -2., 2.)
        self._integral_pos_e = self._integral_pos_e + pos_e * control_timestep
        self._integral_pos_e = np.clip(self._integral_pos_e, -2., 2.)
        self._integral_pos_e[2] = np.clip(self._integral_pos_e[2], -0.15, 0.15)

        # PID target thrust
        target_thrust = np.multiply(self._PID.P_for, pos_e) \
                        + np.multiply(self._PID.I_for, self._integral_pos_e) \
                        + np.multiply(self._PID.D_for, vel_e) \
                        + np.array([0, 0, self._gf])
        scalar_thrust = max(0, np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = math.sqrt(scalar_thrust / (4 * self._kf))
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        #print(f"------------------------------\ntarget_thrust\n : {target_thrust}")
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()

        # Target rotation
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        #print(f"---------------------------------------\ntarget_euler\n : {target_euler}")
        #print(f"\ntarget_thrust\n : {target_thrust}")
        if np.any(np.abs(target_euler) > math.pi):
            print(f"ctrl it {self._env.get_sim_counts()} in {self.__class__.__name__}, range [-pi, pi]")

        return thrust, target_euler, pos_e

    def dsl_pid_attitude_control(
            self,
            control_timestep: float,
            thrust: float,
            current_quaternion: np.ndarray,
            target_euler: np.ndarray,
            target_rpy_rates: np.ndarray,
    ) -> np.ndarray:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(current_quaternion))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self._last_rpy) / control_timestep
        self._last_rpy = cur_rpy
        self._integral_rpy_e = self._integral_rpy_e + rot_e * control_timestep
        self._integral_rpy_e = np.clip(self._integral_rpy_e, -1500., 1500.)
        self._integral_rpy_e[0:2] = np.clip(self._integral_rpy_e[0:2], -1., 1.)
        # PID target torques
        target_torques = - np.multiply(self._PID.P_tor, rot_e) \
                         + np.multiply(self._PID.D_tor, rpy_rates_e) \
                         + np.multiply(self._PID.I_tor, self._integral_rpy_e)


        #target_torques = np.clip(target_torques, -3200, 3200)
        #print(f"\ntarget_torques\n : {target_torques}")
        omegas = thrust + np.dot(self._Mixer, target_torques)#self._Mixer = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])
        omegas = np.clip(omegas, 0 , self.max_rpms)
        return  omegas
