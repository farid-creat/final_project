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

class PatternGenerator:
    def __init__(self, drone_properties, env):
        self.dp = drone_properties
        self.env = env
        self.hover_rpm = drone_properties.hover_rpm
        self.max_rpm = drone_properties.max_rpm
        self.yaw_pid = PIDController(Kp=40, Ki=0, Kd=0)
        self.roll_pid = PIDController(Kp=40, Ki=0, Kd=0)
        self.pitch_pid = PIDController(Kp=40, Ki=0, Kd=0)

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
        return self.hover_rpm * np.ones(4)

    def forward(self):
        return self.hover_rpm + np.array([10, 10, -10, -10])

    def backward(self):
        return self.hover_rpm + np.array([-20, -20, 20, 20])

    def left(self):
        return self.hover_rpm + np.array([-20, 20, -20, 20])

    def right(self):
        return self.hover_rpm + np.array([20, -20, 20, -20])

    def up(self):
        return self.hover_rpm * 1.1 * np.ones(4)

    def down(self):
        return self.hover_rpm * 0.9 * np.ones(4)

    def adjust_orientation(self, rpms):
        dt = 1 / self.env.get_sim_freq()

        # Desired orientations (assuming we want to keep the drone level)
        desired_roll = 0
        desired_pitch = 0
        desired_yaw = 0

        # Get current orientations
        current_roll, current_pitch, current_yaw = self.env.get_drones_kinematic_info()[0].rpy

        # Compute PID corrections
        roll_correction = self.roll_pid.update(desired_roll, current_roll, dt)
        print(current_roll,current_pitch,current_yaw)
        pitch_correction = self.pitch_pid.update(desired_pitch, current_pitch, dt)
        yaw_correction = self.yaw_pid.update(desired_yaw, current_yaw, dt)

        # Apply corrections to RPM values
        # Rotor layout for a quadcopter (X configuration):
        # Front left (0), Front right (1), Rear left (2), Rear right (3)
        # Adjustments: Roll (left-right tilt), Pitch (front-back tilt), Yaw (rotation)
        rpms[0] = rpms[0] + roll_correction - pitch_correction + yaw_correction
        rpms[1] =rpms[1]+ roll_correction - pitch_correction - yaw_correction
        rpms[2] =rpms[2]+ roll_correction + pitch_correction - yaw_correction
        rpms[3] =rpms[3]+ roll_correction + pitch_correction + yaw_correction

        # Log the corrections for debugging
        print(f"Roll Correction: {roll_correction}, Pitch Correction: {pitch_correction}, Yaw Correction: {yaw_correction}")

        return np.clip(rpms, 0, self.max_rpm)