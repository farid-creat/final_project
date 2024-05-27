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
        self.yaw_pid = PIDController(Kp=400, Ki=200, Kd=400)
        self.roll_pid = PIDController(Kp=400, Ki=200, Kd=400)
        self.pitch_pid = PIDController(Kp=400, Ki=200, Kd=400)
        self.height_pid = PIDController(Kp=40000, Ki=2000, Kd=40000)
        self.prev_direction="hover"

    def pattern_generator(self, direction):
        if direction == "forward":
            return self.forward()
        elif direction == "backward":
            return self.backward()
        elif direction == "left":
            return self.left()
        elif direction == "right":
            return self.right()
        elif direction == "up":
            return self.up()
        elif direction == "down":
            return self.down()
        else:
            return self.hover()

    def hover(self):
        # Reset PID controllers' integrals
        if self.prev_direction !="hover":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
            self.height_pid.reset_integral()
        self.prev_direction = "hover"
        return self.adjust_orientation(self.hover_rpm,0,0,0,1.5)

    def forward(self):
        if self.prev_direction != "forward":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
        self.prev_direction = "forward"
        return self.adjust_orientation(self.hover_rpm,-0.5,0,0,1.5)

    def backward(self):
        if self.prev_direction != "backward":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
        self.prev_direction = "backward"
        return self.adjust_orientation(self.hover_rpm,0.5,0,0,1.5)

    def left(self):
        if self.prev_direction != "left":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
        self.prev_direction = "left"
        return self.adjust_orientation(self.hover_rpm,0,-0.5,0,1.5)

    def right(self):
        if self.prev_direction != "right":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
        self.prev_direction = "right"
        return self.adjust_orientation(self.hover_rpm,0,0.5,0,1.5)

    def up(self):
        if self.prev_direction != "up":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
        self.prev_direction = "up"
        return self.adjust_orientation(self.hover_rpm,0,0,0,1.5)

    def down(self):
        if self.prev_direction != "down":
            self.roll_pid.reset_integral()
            self.pitch_pid.reset_integral()
            self.yaw_pid.reset_integral()
        self.prev_direction = "down"
        return self.adjust_orientation(self.hover_rpm,0,0,0,1.5)

    def adjust_orientation(self, base_rpms , desired_roll , desired_pitch , desired_yaw , desired_height):
        dt = 1 / self.env.get_sim_freq()

        # Get current orientations
        current_roll, current_pitch, current_yaw = self.env.get_drones_kinematic_info()[0].rpy
        current_height = self.env.get_drones_kinematic_info()[0].pos[2]

        print(f"current_roll = {current_roll} \ncurrent_pitch = {current_pitch}\ncurrent_yaw = {current_yaw}")

        # Compute PID corrections
        roll_adjustment = self.roll_pid.update(desired_roll, current_roll, dt)
        pitch_adjustment = self.pitch_pid.update(desired_pitch, current_pitch, dt)
        yaw_adjustment = self.yaw_pid.update(desired_yaw, current_yaw, dt)
        height_adjustment = self.height_pid.update(desired_height, current_height, dt)
        print(f"----------------------\nheight = {current_height}\nheight_adjustment = {height_adjustment}")

        rpms = (base_rpms +height_adjustment) + np.array([
            +roll_adjustment - pitch_adjustment + yaw_adjustment,  # Front Left (Rotor 1 / Link 0)
            +roll_adjustment + pitch_adjustment - yaw_adjustment,  # Rear Left (Rotor 2 / Link 1)
            -roll_adjustment + pitch_adjustment + yaw_adjustment,  # Rear Right (Rotor 3 / Link 2)
            -roll_adjustment - pitch_adjustment - yaw_adjustment  # Front Right (Rotor 4 / Link 3)
        ])


        # Log the corrections for debugging
        #print(f"Roll Correction: {roll_adjustment}\n, Pitch Correction: {pitch_adjustment}\n, Yaw Correction: {yaw_adjustment}\n")

        return np.clip(rpms, 0, self.max_rpm)