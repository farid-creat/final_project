import math
import time
from controller import Robot, Keyboard

def clamp(value, low, high):
    return max(low, min(value, high))

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Get and enable devices.
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    front_left_led = robot.getDevice("front left led")
    front_right_led = robot.getDevice("front right led")
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    compass = robot.getDevice("compass")
    compass.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Get propeller motors and set them to velocity mode.
    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(1.0)

    # Display the welcome message.
    print("Start the drone...")

    # Wait one second.
    while robot.step(timestep) != -1:
        if robot.getTime() > 1.0:
            break

    # Display manual control message.
    print("You can control the drone with your computer keyboard:")
    print("- 'up': move forward.")
    print("- 'down': move backward.")
    print("- 'right': turn right.")
    print("- 'left': turn left.")
    print("- 'shift + up': increase the target altitude.")
    print("- 'shift + down': decrease the target altitude.")
    print("- 'shift + right': strafe right.")
    print("- 'shift + left': strafe left.")

    # Constants, empirically found.
    k_vertical_thrust = 68.5  # with this thrust, the drone lifts.
    k_vertical_offset = 0.6  # Vertical offset where the robot actually targets to stabilize itself.
    k_vertical_p = 3.0  # P constant of the vertical PID.
    k_roll_p = 50.0  # P constant of the roll PID.
    k_pitch_p = 30.0  # P constant of the pitch PID.

    # Variables.
    target_altitude = 1.0  # The target altitude. Can be changed by the user.

    # Main loop
    while robot.step(timestep) != -1:
        time = robot.getTime()  # in seconds.

        # Retrieve robot position using the sensors.
        roll, pitch, _ = imu.getRollPitchYaw()
        _, _, altitude = gps.getValues()
        roll_velocity, pitch_velocity, _ = gyro.getValues()

        # Blink the front LEDs alternatively with a 1 second rate.
        led_state = int(time) % 2
        front_left_led.set(led_state)
        front_right_led.set(not led_state)

        # Transform the keyboard input to disturbances on the stabilization algorithm.
        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                pitch_disturbance = -2.0
            elif key == Keyboard.DOWN:
                pitch_disturbance = 2.0
            elif key == Keyboard.RIGHT:
                yaw_disturbance = -1.3
            elif key == Keyboard.LEFT:
                yaw_disturbance = 1.3
            elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
                roll_disturbance = -1.0
            elif key == (Keyboard.SHIFT + Keyboard.LEFT):
                roll_disturbance = 1.0
            elif key == (Keyboard.SHIFT + Keyboard.UP):
                target_altitude += 0.05
                print("target altitude: %.2f [m]" % target_altitude)
            elif key == (Keyboard.SHIFT + Keyboard.DOWN):
                target_altitude -= 0.05
                print("target altitude: %.2f [m]" % target_altitude)
            key = keyboard.getKey()

        # Compute the roll, pitch, yaw and vertical inputs.
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * math.pow(clamped_difference_altitude, 3.0)

        # Actuate the motors taking into consideration all the computed inputs.
        front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)

    print("Exited the main loop.")
    robot.resetPhysics()
    return

if __name__ == "__main__":
    main()
