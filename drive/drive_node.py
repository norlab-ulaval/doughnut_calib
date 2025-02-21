import rclpy
import threading
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, Imu
from std_msgs.msg import Bool, String, Float64, Int32
# from warthog_msgs.msg import Status

import os
import sys
import numpy as np
import pandas as pd

class DriveNode(Node):
    """
    Class that sends out commands to calibrate mobile ground robots
    """
    # def __init__(self, max_lin_speed, min_lin_speed, lin_speed_step, max_ang_speed, ang_steps,
    #              step_len, dead_man_button, dead_man_index, dead_man_threshold, ramp_trigger_button, ramp_trigger_index,
    #              calib_trigger_button, calib_trigger_index, response_model_window, steady_state_std_dev_threshold,
    #              cmd_rate_param, encoder_rate_param):
    def __init__(self):
        super().__init__('drive_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('command_model', 'differential_drive'),
                ('wheel_radius', 1.0),
                ('wheel_baseline', 1.0),
                ('auto_max_speed_characterization', True),
                ('auto_cmd_model_characterization', True),
                ('max_lin_speed', 2.0),
                ('max_ang_speed', 2.0),
                ('step_len', 6.0),
                ('n_calib_steps', 20),
                ('dead_man_button', True),
                ('dead_man_index', 0),
                ('dead_man_threshold', 0.5),
                ('calib_trigger_button', False),
                ('calib_trigger_index', 0),
                ('calib_threshold', 0.5),
                ('cmd_rate_param', 20),
                ('encoder_rate_param', 4),
                ('path_to_input_space_calib_data','none'),
            ]
        )
        self.cmd_model = self.get_parameter('command_model').get_parameter_value().string_value
        self.define_speed_limits = self.get_parameter('auto_max_speed_characterization').get_parameter_value().bool_value
        self.define_cmd_model = self.get_parameter('auto_cmd_model_characterization').get_parameter_value().bool_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheel_baseline = self.get_parameter('wheel_baseline').get_parameter_value().double_value
        self.max_lin_speed = self.get_parameter('max_lin_speed').get_parameter_value().double_value
        self.max_ang_speed = self.get_parameter('max_ang_speed').get_parameter_value().double_value
        self.step_len = self.get_parameter('step_len').get_parameter_value().double_value
        self.n_calib_steps = self.get_parameter('n_calib_steps').get_parameter_value().integer_value
        self.dead_man_button = self.get_parameter('dead_man_button').get_parameter_value().bool_value
        self.dead_man_index = self.get_parameter('dead_man_index').get_parameter_value().integer_value
        self.dead_man_threshold = self.get_parameter('dead_man_threshold').get_parameter_value().double_value
        self.calib_trigger_button = self.get_parameter('calib_trigger_button').get_parameter_value().bool_value
        self.calib_trigger_index = self.get_parameter('calib_trigger_index').get_parameter_value().integer_value
        self.calib_threshold = self.get_parameter('calib_threshold').get_parameter_value().double_value
        self.cmd_rate_param = self.get_parameter('cmd_rate_param').get_parameter_value().integer_value
        self.cmd_rate = self.create_rate(self.cmd_rate_param)
        self.encoder_rate = self.get_parameter('encoder_rate_param').get_parameter_value().integer_value
        
        # self.path_to_input_space_calib_data
        self.path_to_input_space_calib_data = self.get_parameter('path_to_input_space_calib_data').get_parameter_value().string_value

        self.cmd_msg = Twist()
        self.joy_bool = Bool()
        self.good_calib_step = Bool()
        self.good_calib_step.data = False
        self.calib_step_msg = Int32()
        self.calib_step_msg.data = 0
        
        self.left_wheel_msg = Float64()
        self.right_wheel_msg = Float64()
        self.state_msg = String()
        self.state_msg.data = "idle"  # 4 possible states : idle, ramp_up, ramp_down, calib
        
        
        self.left_wheel_current_msg = Float64()
        self.right_wheel_current_msg = Float64()

        self.left_wheel_voltage_msg = Float64()
        self.right_wheel_voltage_msg = Float64()

        self.joy_listener = self.create_subscription(
            Joy,
            'joy_in',
            self.joy_callback,
            1000)
        
        self.left_wheel_listener = self.create_subscription(
            Float64,
            'left_wheel_in',
            self.left_wheel_callback,
            1000)
        self.right_wheel_listener = self.create_subscription(
            Float64,
            'right_wheel_in',
            self.right_wheel_callback,
            1000)
        
            

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_out', 10)
        self.joy_pub = self.create_publisher(Bool, 'drive/joy_switch', 10)
        self.good_calib_step_pub = self.create_publisher(Bool, 'good_calib_step', 10)
        self.calib_step_pub = self.create_publisher(Int32, 'calib_step', 10)
        self.state_pub = self.create_publisher(String, 'calib_state', 10)

        self.dead_man = False
        self.ramp_trigger = False
        self.calib_trigger = False
        self.skip_calib_step_trigger = False
        self.prev_calib_step_trigger = False
        self.steady_state = False
        self.calibration_end = False
        self.first_order_calib = False
        self.step_skip_bool = False
        self.step_prev_bool = False
        # self.estop_bool = False
        

    def joy_callback(self, joy_data):
        global dead_man
        global dead_man_index
        if self.dead_man_button == False:
            if np.abs(joy_data.axes[self.dead_man_index]) >= np.abs(self.dead_man_threshold) \
                    and joy_data.axes[self.calib_trigger_index] == 0 \
                    and joy_data.buttons[self.calib_trigger_index] == 0 :

                self.dead_man = True
            else:
                self.dead_man = False
        else:
            if joy_data.buttons[self.dead_man_index] >= self.dead_man_threshold \
                    and joy_data.buttons[self.calib_trigger_index] == 0 \
                    and joy_data.buttons[self.calib_trigger_index] == 0:

                self.dead_man = True
            else:
                self.dead_man = False

        if self.calib_trigger_button == False:
            if np.abs(joy_data.axes[self.calib_trigger_index]) >= np.abs(self.calib_threshold):
                self.calib_trigger = True
            else:
                self.calib_trigger = False
        else:
            if joy_data.buttons[self.calib_trigger_index] >= 0.8:
                self.calib_trigger = True
            else:
                self.calib_trigger = False

    def left_wheel_callback(self, left_wheel_data):
        self.left_wheel_msg = left_wheel_data

    def right_wheel_callback(self, right_wheel_data):
        self.right_wheel_msg = right_wheel_data

    def powertrain_vel(self, cmd, last_vel, tau_c):
        return last_vel + (1 / tau_c) * (cmd - last_vel) * (1 / self.encoder_rate)

    def publish_cmd(self):
        self.cmd_vel_pub.publish(self.cmd_msg)
        self.cmd_rate.sleep()

    def publish_joy_switch(self):
        self.joy_pub.publish(self.joy_bool)

    def publish_state(self):
        self.state_pub.publish(self.state_msg)

    def reverse_engineer_command_model(self):
        left_encoder_vels_list = []
        right_encoder_vels_list = []
        command_linear_calibration = self.max_lin_speed / 4
        self.calib_lin_speed = command_linear_calibration
        self.calib_ang_speed = 0.0
        self.lin_speed = 0.0
        self.ang_speed = 0.0
        linear_vel_time_window = self.step_len
        linear_vel_elapsed_time = 0.0
        waiting_for_user_input = False
        self.get_logger().info('Press characterization trigger when ready, prepare for the robot to go in a straight line : ')
        while not waiting_for_user_input:
            #self.get_logger().info(str(self.calib_trigger))
            if self.calib_trigger:
                waiting_for_user_input = True
        previous_time = self.get_clock().now().nanoseconds * 1e-9
        while linear_vel_elapsed_time < linear_vel_time_window:
            self.cmd_msg.linear.x = command_linear_calibration
            self.cmd_msg.angular.z = 0.0
            self.publish_cmd()
            linear_vel_elapsed_time += (self.get_clock().now().nanoseconds * 1e-9 - previous_time)
            previous_time = self.get_clock().now().nanoseconds * 1e-9
            if linear_vel_elapsed_time >= self.step_len/3:
                left_encoder_vels_list.append(self.left_wheel_msg.data)
                right_encoder_vels_list.append(self.right_wheel_msg.data)
        self.cmd_msg.linear.x = 0.0
        self.publish_cmd()
        self.calib_lin_speed = command_linear_calibration
        self.calib_ang_speed = 0.0
        self.lin_speed = command_linear_calibration
        self.ang_speed = 0.0
        left_encoder_vels_array = np.array(left_encoder_vels_list)
        left_encoder_vels_mean = np.mean(left_encoder_vels_array)
        right_encoder_vels_array = np.array(right_encoder_vels_list)
        right_encoder_vels_mean = np.mean(right_encoder_vels_array)
        ## TODO: validate if wheel velocities are symmetrical
        self.calibrated_wheel_radius = command_linear_calibration / left_encoder_vels_mean
        # self.get_logger().info("calibrated wheel radius :" + str(self.calibrated_wheel_radius))

        left_encoder_vels_list = []
        right_encoder_vels_list = []
        command_angular_calibration = self.max_ang_speed / 2
        self.calib_lin_speed = 0.0
        self.calib_ang_speed = command_angular_calibration
        self.lin_speed = 0.0
        self.ang_speed = 0.0
        angular_vel_time_window = self.step_len
        angular_vel_elapsed_time = 0.0
        waiting_for_user_input = False
        self.get_logger().info(
            'Press characterization trigger when ready, prepare for the robot to turn on the spot : ')
        while not waiting_for_user_input:
            #self.get_logger().info(str(self.calib_trigger))
            if self.calib_trigger:
                waiting_for_user_input = True
        previous_time = self.get_clock().now().nanoseconds * 1e-9
        while angular_vel_elapsed_time < angular_vel_time_window:
            self.cmd_msg.linear.x = 0.0
            self.cmd_msg.angular.z = command_angular_calibration
            self.publish_cmd()
            angular_vel_elapsed_time += (self.get_clock().now().nanoseconds * 1e-9 - previous_time)
            previous_time = self.get_clock().now().nanoseconds * 1e-9
            if angular_vel_elapsed_time >= self.step_len:
                left_encoder_vels_list.append(self.left_wheel_msg.data)
                right_encoder_vels_list.append(self.right_wheel_msg.data)
        self.cmd_msg.angular.z = 0.0
        self.publish_cmd()
        self.calib_lin_speed = 0.0
        self.calib_ang_speed = command_angular_calibration
        self.lin_speed = 0.0
        self.ang_speed = command_angular_calibration
        left_encoder_vels_array = np.array(left_encoder_vels_list)
        left_encoder_vels_mean = np.mean(left_encoder_vels_array)
        right_encoder_vels_array = np.array(right_encoder_vels_list)
        right_encoder_vels_mean = np.mean(right_encoder_vels_array)
        ## TODO: validate if wheel velocities are symmetrical
        self.calibrated_baseline = -2 * self.calibrated_wheel_radius * left_encoder_vels_mean / command_angular_calibration
        # self.get_logger().info("calibrated baseline: " + str(self.calibrated_baseline))

        self.command_diff_drive_jacobian = self.calibrated_wheel_radius * np.array([[0.5, 0.5],
                                                                                    [-1/self.calibrated_baseline, 1/self.calibrated_baseline]])
        self.command_diff_drive_jacobian_inverse = np.linalg.inv(self.command_diff_drive_jacobian)
    def command_to_input_vector(self, command_linear_vel, command_angular_vel):
        command_vector = np.array([command_linear_vel, command_angular_vel])
        encoder_vels_vector = self.command_diff_drive_jacobian_inverse @ command_vector
        # reverse_body_command_vector = self.command_diff_drive_jacobian @ encoder_vels_vector
        # self.get_logger().info("reverse_compute_angular_vel :" + str(reverse_body_command_vector[1]))
        return encoder_vels_vector

    def input_to_command_vector(self, left_wheel_vel, right_wheel_vel):
        input_vector = np.array([left_wheel_vel, right_wheel_vel])
        body_vels_vector = self.command_diff_drive_jacobian @ input_vector
        # reverse_body_command_vector = self.command_diff_drive_jacobian @ encoder_vels_vector
        # self.get_logger().info("reverse_compute_angular_vel :" + str(reverse_body_command_vector[1]))
        return body_vels_vector

    def calibrate_maximum_linear_limits(self):
        encoders_saturated = False
        command_linear_maximum_limit = self.max_lin_speed * 10
        linear_vel_elapsed_time = 0.0
        max_vel_step = 0.25
        left_encoder_vels_sum = 0
        left_encoder_vels_num = 0
        left_encoder_vels_list = []
        right_encoder_vels_list = []
        waiting_for_user_input = False
        self.get_logger().info(
            'Press characterization trigger when ready, to go top speed forward : ')
        while not waiting_for_user_input:
            
            if self.calib_trigger:
                waiting_for_user_input = True
        previous_time = self.get_clock().now().nanoseconds * 1e-9
        while not linear_vel_elapsed_time >= 6.0:
            self.cmd_msg.linear.x = command_linear_maximum_limit
            self.cmd_msg.angular.z = 0.0
            self.publish_cmd()
            # self.get_logger().info("cmd_linear_x :" + str(self.cmd_msg.linear.x))
            # self.get_logger().info("left_encoder_measure :" + str(self.left_wheel_msg.data))
            self.encoder_command_vector = self.command_to_input_vector(self.cmd_msg.linear.x, self.cmd_msg.angular.z)
            # self.get_logger().info("left_encoder_command :" + str(self.encoder_command_vector[0]))
            linear_vel_elapsed_time += (self.get_clock().now().nanoseconds * 1e-9 - previous_time)
            previous_time = self.get_clock().now().nanoseconds * 1e-9
            if linear_vel_elapsed_time >= 2.0:
                left_encoder_vels_list.append(self.left_wheel_msg.data)
                right_encoder_vels_list.append(self.right_wheel_msg.data)
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.0
        self.publish_cmd()

        self.maximum_wheel_vel = np.mean(np.array([np.abs(np.median(np.asarray(left_encoder_vels_list))),
                                                   np.abs(np.median(np.asarray(right_encoder_vels_list)))]))
        self.minimum_wheel_vel = -self.maximum_wheel_vel


    def calibrate_input_space(self):
        """ Run the automatic calibration to find the Wheel radius and the basewidth if the 
        auto_cmd_model_characterization == True. If false, then use the value in the config (warthog.config.yaml) 

        Calculate the maximum speed if auto_max_speed_characterization == True. 

        Save the parameters to the path_to_input_space_calib_data. 

        """
        # 1. Reverse engineer the basewidth and wheel radius
        if self.define_cmd_model:
            self.reverse_engineer_command_model()
        else:
            self.get_logger().info('Using user-defined vehicle parameters : ')
            self.calibrated_wheel_radius = self.wheel_radius
            self.calibrated_baseline = self.wheel_baseline
            self.command_diff_drive_jacobian = self.calibrated_wheel_radius * np.array([[0.5, 0.5],
                                                                                        [-1 / self.calibrated_baseline,
                                                                                         1 / self.calibrated_baseline]])
            self.command_diff_drive_jacobian_inverse = np.linalg.inv(self.command_diff_drive_jacobian)
        self.get_logger().info('Characterized wheel radius [m]: ' + str(self.calibrated_wheel_radius))
        self.get_logger().info('Characterized baseline [m]: ' + str(self.calibrated_baseline))

        # 2. Calculate the maximum linear speed
        if self.define_speed_limits:
            self.calibrate_maximum_linear_limits()
        else:
            self.maximum_wheel_vel = self.command_to_input_vector(self.max_lin_speed, 0)[0]
            self.minimum_wheel_vel = -self.maximum_wheel_vel
        self.get_logger().info('Maximum wheel velocity [rad/s]: ' + str(self.maximum_wheel_vel))
        self.get_logger().info('Minimum wheel velocity [rad/s]: ' + str(self.minimum_wheel_vel))
        # self.step_len = self.transitory_time_max * 3
        self.maximum_linear_vel_positive = self.input_to_command_vector(self.maximum_wheel_vel, self.maximum_wheel_vel)[1]
        self.maximum_linear_vel_negative = self.input_to_command_vector(-self.maximum_wheel_vel, -self.maximum_wheel_vel)[1]

        ## 3. For saving input-space data
        self.input_space_array = np.array([self.calibrated_wheel_radius,
                                           self.calibrated_baseline,
                                           self.maximum_wheel_vel,
                                           self.minimum_wheel_vel])
        cols = ['calibrated_radius [m]',
                'calibrated_baseline [m]',
                'maximum_wheel_vel_positive [rad/s]',
                'maximum_wheel_vel_negative [rad/s]',
                ]
        self.input_space_array_dataframe = pd.DataFrame(self.input_space_array.reshape((1, len(cols))), columns=cols)
        self.input_space_array_dataframe.to_pickle(self.path_to_input_space_calib_data)
        
        return None
    def random_uniform_sampler_within_low_lvl_limits(self):
        
        #self.get_logger().info("Start sampling")
        sample_respect_low_lvl_controller = False

        while sample_respect_low_lvl_controller==False:
            wheel_vels = np.random.uniform(self.minimum_wheel_vel, self.maximum_wheel_vel, size=2)
            body_vels = self.input_to_command_vector(wheel_vels[0], wheel_vels[1])
            #self.get_logger().info("stuck")

            if (np.abs(body_vels[0])< self.max_lin_speed) and (np.abs(body_vels[1]) < self.max_ang_speed):
                sample_respect_low_lvl_controller = True 

        #self.get_logger().info("Finish sampling")
        self.get_logger().info(f"______________Current step : {self.calib_step_msg.data+1}/{self.n_calib_steps}____________")
        self.get_logger().info(f"Linear speed [m/s] : {np.round(body_vels[0],2)}")
        self.get_logger().info(f"Angular speed [rad/s] :  {np.round(body_vels[1],2)}")

        return body_vels
    def uniform_calibration_input_space_sampling(self):
        """ Random sampling of the input space (min wheel vel and max wheel vel)
        folowing the control limit of the low-level.         
        """
        self.get_logger().info("Press characterization trigger to start the uniform sampling")

        body_vels = self.random_uniform_sampler_within_low_lvl_limits()
        self.lin_speed = 0.0
        self.ang_speed = 0.0
        self.calib_lin_speed = body_vels[0]
        self.calib_ang_speed = body_vels[1]
        self.step_t = 0
        
        #self.get_logger().info(f"{self.calibration_end}") 
        #self.get_logger().info(f"self.state_msg.data {self.state_msg.data}")
        while self.calibration_end == False:
            self.publish_state()
            self.calib_step_pub.publish(self.calib_step_msg)
            #self.get_logger().info(f"{self.calib_trigger}") on se rend la 
            if self.state_msg.data == "idle":
                self.step_t = 0
                if self.calib_trigger == True:
                    # self.ramp_up()
                    self.state_msg.data = 'calib'
                else:
                    self.cmd_msg.linear.x = 0.0
                    self.cmd_msg.angular.z = 0.0
                    self.publish_cmd()
                    
            elif self.state_msg.data == "calib":

                if self.ramp_trigger == True: # Jamais mis à true
                    self.step_t = 0
                    continue

                if self.dead_man == False:
                    self.cmd_msg.linear.x = body_vels[0]
                    self.cmd_msg.angular.z = body_vels[1]
                    self.joy_bool.data = False
                    self.publish_cmd()
                    self.publish_joy_switch()

                    self.step_t += 0.05
                    if self.step_t >= self.step_len or self.skip_calib_step_trigger: #skip_calib_step_trigger toujours a false
                        # Sampling here 
                        self.good_calib_step.data = True
                        self.good_calib_step_pub.publish(self.good_calib_step)
                        self.good_calib_step.data = False
                        self.step_t = 0
                        self.calib_step_msg.data += 1
                        
                        if self.calib_step_msg.data == self.n_calib_steps:
                            self.calibration_end = True
                        else:
                            body_vels = self.random_uniform_sampler_within_low_lvl_limits()      
                            


                else:
                    self.get_logger().info("Incoming command from controller, calibration suspended.")
                    self.get_logger().info("Press characterization trigger to resume the uniform sampling")
                    self.lin_speed = 0.0
                    self.state_msg.data = "idle"
                    self.joy_switch = Bool()
                    self.joy_switch.data = True
                    self.publish_joy_switch()
        # self.calibration_end = False
        # self.state_msg.data = "idle"

    def run_calibration(self):
        self.get_logger().info(self.cmd_model)
        if self.cmd_model == 'differential_drive':
            self.calibrate_input_space()
        else:
            self.get_logger().info("Undefined command model, shutting down.")
            sys.exit()
        self.get_logger().info("Test")
        self.uniform_calibration_input_space_sampling()

def main(args=None):
    rclpy.init(args=args)
    drive_node = DriveNode()
    thread = threading.Thread(target=rclpy.spin, args=(drive_node, ), daemon=True)
    thread.start()     
    drive_node.run_calibration()  
    drive_node.get_logger().info("Calibration done.")
    
    #rclpy.spin(drive_node)
    drive_node.get_logger().info("Ctrl+A, D to detach from the screen. Don't forget to save in the run_drive.bash script already openned")
    
    drive_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
