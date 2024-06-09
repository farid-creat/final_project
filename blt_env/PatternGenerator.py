import numpy as np
import time

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    def reset_integral(self):
        self.integral = 0
        self.prev_error = 0
class PatternGenerator:
    def __init__(self, drone_properties, env):
        self.dp = drone_properties
        self.env = env
        self.hover_rpm = drone_properties.hover_rpm
        self.max_rpm = drone_properties.max_rpm
        # Constants
        self.mass = self.dp.m  # kg
        self.gravity = 9.81  # m/s^2
        self.L = self.dp.l  # m (distance from the center of the drone to each motor)
        self.k_f = self.dp.kf  # N/(rad/s)^2 (thrust coefficient)
        self.k_m = self.dp.km  # Nm/(rad/s)^2 (torque coefficient)

        # P controller gains
        self.k_p = 0.1  # Proportional gain for position
        self.k_p_phi = 0.05  # Proportional gain for roll
        self.k_p_theta = 0.05  # Proportional gain for pitch
        self.k_p_psi = 0.05  # Proportional gain for yaw

    def controller(self, target_pos):
        # Desired state
        x_d, y_d, z_d = target_pos
        phi_d, theta_d, psi_d = 0.0, 0.0, 0.0

        # Current state
        current_state = self.env.get_drones_kinematic_info()[0]
        x, y, z = current_state.pos
        phi, theta, psi = current_state.rpy

        # Position errors
        e_x = x_d - x
        e_y = y_d - y
        e_z = z_d - z
        print(f"------------------------------------\nPosition errors: e_x = {e_x}, e_y = {e_y}, e_z = {e_z}")

        # Desired forces
        F_x = self.k_p * e_x
        F_y = self.k_p * e_y
        F_z = self.k_p * e_z
        print(f"Desired forces: F_x = {F_x}, F_y = {F_y}, F_z = {F_z}")

        # Calculate desired roll and pitch angles using small-angle approximations
        phi_d = np.arctan2(min(F_y,self.mass*self.gravity), self.mass*self.gravity)
        theta_d = np.arctan2(min(F_x,self.mass*self.gravity), self.mass*self.gravity)
        print(f"Desired angles: phi_d = {phi_d}, theta_d = {theta_d}")

        # Orientation errors
        e_phi = phi_d - phi
        e_theta = theta_d - theta
        e_psi = psi_d - psi
        print(f"Orientation errors: e_phi = {e_phi}, e_theta = {e_theta}, e_psi = {e_psi}")

        # Desired torques
        tau_phi = self.k_p_phi * e_phi
        tau_theta = self.k_p_theta * e_theta
        tau_psi = self.k_p_psi * e_psi
        print(f"Desired torques: tau_phi = {tau_phi}, tau_theta = {tau_theta}, tau_psi = {tau_psi}")

        # Calculate individual thrusts
        T1 = F_z  - tau_theta / (2 * self.L) - tau_psi / (4 * self.k_m)
        T2 = F_z  - tau_phi / (2 * self.L) + tau_psi / (4 * self.k_m)
        T3 = F_z  + tau_theta / (2 * self.L) - tau_psi / (4 * self.k_m)
        T4 = F_z  + tau_phi / (2 * self.L) + tau_psi / (4 * self.k_m)
        print(f"T1: {T1} , T2: {T2} ,T3: {T3} ,T4: {T4} ")

        # Ensure thrusts are positive and within limits
        T1 = max(T1, 0)
        T2 = max(T2, 0)
        T3 = max(T3, 0)
        T4 = max(T4, 0)

        # Convert thrust to angular velocities using k_f
        omega_1 =self.hover_rpm+ np.sqrt(T1 / self.k_f)
        omega_2 =self.hover_rpm+ np.sqrt(T2 / self.k_f)
        omega_3 =self.hover_rpm+ np.sqrt(T3 / self.k_f)
        omega_4 =self.hover_rpm+ np.sqrt(T4 / self.k_f)

        # Convert angular velocities to RPM
        RPM_1 = omega_1 #* 60 / (2 * np.pi)
        RPM_2 = omega_2 #* 60 / (2 * np.pi)
        RPM_3 = omega_3 #* 60 / (2 * np.pi)
        RPM_4 = omega_4 #* 60 / (2 * np.pi)

        rpms = np.array([
            RPM_2,  # Front Left (Rotor 1 / Link 0)
            RPM_3,  # Rear Left (Rotor 2 / Link 1)
            RPM_4,  # Rear Right (Rotor 3 / Link 2)
            RPM_1   # Front Right (Rotor 4 / Link 3)
        ])
        print(f"RPMs: \n{rpms}")
        return rpms