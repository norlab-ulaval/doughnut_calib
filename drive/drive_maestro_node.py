import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float64, UInt32,Float32
from drive_custom_srv.srv import BashToPath,DriveMetaData,TrainMotionModel,ExecuteAllTrajectories,LoadTraj,LoadController
from drive_custom_srv.msg import PathTree
from std_srvs.srv import Empty,Trigger
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup,ReentrantCallbackGroup
import pathlib
from ament_index_python.packages import get_package_share_directory
import datetime
import shutil
import os 
from rclpy.qos import qos_profile_action_status_default
import yaml
from rclpy.executors import MultiThreadedExecutor
from threading import Event
import time
from norlab_controllers_msgs.srv import ExportData,ChangeController
from nav_msgs.msg import Odometry
from  drive.trajectory_creator.eight_trajectory import EightTrajectoryGenerator,RectangleTrajectoryGenerator,TurnAround

import numpy as np 
import matplotlib.pyplot as plt
from norlab_controllers_msgs.action import FollowPath
from norlab_controllers_msgs.msg import PathSequence,DirectionalPath,FollowerOptions
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped,Pose,Quaternion,Point 
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Path
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter,ParameterValue
from rclpy.action import ActionClient
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

class DriveMaestroNode(Node):
    """
    Class that sends out commands to the nodes for a full step by step drive experiment
    """
    # def __init__(self, max_lin_speed, min_lin_speed, lin_speed_step, max_ang_speed, ang_steps,
    #              step_len, dead_man_button, dead_man_index, dead_man_threshold, ramp_trigger_button, ramp_trigger_index,
    #              calib_trigger_button, calib_trigger_index, response_model_window, steady_state_std_dev_threshold,
    #              cmd_rate_param, encoder_rate_param):
    def __init__(self):
        super().__init__('drive_maestro_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('maestro_mode', 'Not defined'),
                ('experiment_folder_path', 'Not defined'),
                ('controller_basic_param','Not defined')
            ]
        )
        # Load the mode 
        #self.maestro_mode = self.get_parameter('maestro_mode').get_parameter_value().string_value
        #self.get_logger().info(str(self.maestro_mode))

        
        # Load the gui message 
        self.path_to_share_directory = pathlib.Path(get_package_share_directory('drive'))
        path_to_gui_message = self.path_to_share_directory.parent.parent.parent.parent/'src'/'DRIVE'/'drive'/'gui_message.yaml'
        with open(str(path_to_gui_message),'r') as f:
            self.gui_message = yaml.safe_load(f)["gui_message"]
        
        self.get_logger().info(str(self.gui_message))
        # Add on 
        self.service_done_event = Event()

        self.callback_group = ReentrantCallbackGroup()
        #######
        self.sub_node = rclpy.create_node('sub_node')

        # Inform the user to open the GUI aka foxglove interface.
        self.get_logger().info("Please open foxglove, connect to the robot and import the foxglove template located at ../DRIVE/gui/drive_gui")

        # Initialize the defautl value of msg to publish
        self.drive_maestro_operator_action_msg = String() #Create Topic for operator action
        self.drive_maestro_status_msg = String()

        # Create publisher 
        self.visualise_path_topic_name = "visualize_path_topic"
        self.path_to_drive_experiment_folder_pub = self.create_publisher(PathTree, 'experiment_data_paths', qos_profile_action_status_default) # Makes durability transient_local
        self.drive_maestro_operator_action_pub = self.create_publisher(String, 'operator_action', 10)
        self.drive_maestro_status_pub = self.create_publisher(String, 'maestro_status', 10)
        self.path_loaded_pub = self.create_publisher(Path,"path_to_reapeat",qos_profile_action_status_default)
        
        #Create subscriber
        self.calib_state_listener = self.create_subscription(String,'calib_state', self.calib_state_callback, 10)
        self.calib_state_listener_msg = 'not yet defined'
        self.odom_in = self.create_subscription(Odometry,'odom_in', self.odom_in_callback, 10)
        self.pose = Odometry()


        self.drive_finish_published = False

        timer_period = 0.5  # seconds #TIMER
        self.timer_callgroup = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(timer_period, self.timer_callback,callback_group=self.timer_callgroup) #TIMER execute callback

        # Services creations
        self.srv_call_group = MutuallyExclusiveCallbackGroup()  
        self.srv_send_metadata = self.create_service(DriveMetaData, 'send_metadata_form', self.send_metadata_form_callback) #service for starting drive

        self.srv_save_logger_dataset = self.create_service(Trigger, 'maestro_export_data', self.save_logger_dataset_service_client,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        self.srv_stop_mapping = self.create_service(Trigger, 'done_mapping', self.stop_mapping_service,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        
        self.srv_train_motion_model = self.create_service(TrainMotionModel, 'maestro_train_motion_model', self.model_training_service_client,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        
        self.srv_create_and_execute_trajectories = self.create_service(Empty, 'execute_all_trajectories', self.execute_all_trajectories_call_back,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        self.srv_load_a_trajectorie = self.create_service(LoadTraj, 'load_trajectory', self.load_trajectory_callback,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        self.srv_select_controller = self.create_service(LoadController, 'load_controller', self.load_controller_callback,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        self.srv_srv_load_max_speed = self.create_service(ExecuteAllTrajectories, 'load_maximum_speed', self.load_max_speed_callback,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        self.srv_trajectory_confirm = self.create_service(Empty, 'confirm_trajectory', self.confirm_traj_callback,callback_group=MutuallyExclusiveCallbackGroup() ) # service wrapper to call the service saving the calibration dataset
        
        
        # Creation of service client
        
        self.stop_mapping_client = self.sub_node.create_client(Empty, '/mapping/disable_mapping', callback_group=MutuallyExclusiveCallbackGroup() )
        self.save_calibration_dataset_client = self.sub_node.create_client(ExportData, '/drive/export_data', callback_group=MutuallyExclusiveCallbackGroup() )
        self.train_motion_model_client = self.sub_node.create_client(TrainMotionModel, '/drive/train_motion_model', callback_group=MutuallyExclusiveCallbackGroup() )
        self.change_controller_client = self.sub_node.create_client(ChangeController, '/controller/change_controller', callback_group=MutuallyExclusiveCallbackGroup() )
        self.maximum_speed_client = self.sub_node.create_client(SetParameters, '/controller/controller_node/set_parameters', callback_group=MutuallyExclusiveCallbackGroup() )
        
        # Creation of action client in the subnode so that the server of the maestro can call the action client and react in cnsequence of the feedback
        # 

        self._action_client = ActionClient(self, FollowPath, '/follow_path')

        # Self variable initialization 
        
        
        # Publish the run by master topic
        #self.path_to_drive_experiment_folder_pub = self.create_publisher(Bool, 'run_by_maestro', qos_profile_action_status_default) # Makes durability transient_local
        #self.run_by_master_msg = Bool()
        #self.run_by_master_msg.data = True

        
        # Path initialization
        self.path_to_drive_experiment_folder_msg = PathTree()
        self.path_to_drive_experiment_folder_msg.path_to_experiment_folder = "Not define yet" 
        self.path_to_drive_experiment_folder_msg.path_config_folder = "Not define yet" 
        self.path_to_drive_experiment_folder_msg.path_model_training_datasets = "Not define yet" 
        self.path_to_drive_experiment_folder_msg.path_model_training_results = "Not define yet" 
        self.drive_maestro_operator_action_msg.data = self.gui_message["metadata_form"]["operator_action_message"]
        self.drive_maestro_status_msg.data = self.gui_message["metadata_form"]["status_message"] #ini

    def timer_callback(self):
        self.publish_drive_maestro_operator_action()
        self.publish_drive_maestro_status()
        
    
    #TOPIC SUBSCRIBED
    
    def odom_in_callback(self,msg):
        #self.get_logger().info(str(msg))
        self.pose = msg
        
    def calib_state_callback(self, msg): #operator action FROM drive node
        self.calib_state_listener_msg = msg.data
        
        if self.calib_state_listener_msg == 'drive_finished' and self.drive_finish_published == False:
            self.drive_maestro_status_msg.data = self.gui_message["drive_save"]["status_message"]
            self.drive_maestro_operator_action_msg.data = self.gui_message["drive_save"]["operator_action_message"]
            self.publish_drive_maestro_status()
            self.publish_drive_maestro_operator_action()
            self.drive_finish_published =True
    #TOPIC PUBLISH
    
    def publish_drive_maestro_operator_action(self): # Operator action
        self.drive_maestro_operator_action_pub.publish(self.drive_maestro_operator_action_msg)

    def publish_drive_maestro_status(self): # Status
        self.drive_maestro_status_pub.publish(self.drive_maestro_status_msg)

    def publish_drive_maestro_path_to_drive_folder(self): # Path
        self.path_to_drive_experiment_folder_pub.publish(self.path_to_drive_experiment_folder_msg)

    # Usefull fonction 
    def create_folder(self,robot_arg,traction_arg,terrain_arg):
        """Create all the folder-tree that is used to save the data during a drive-to TNR experiements. 

        fill the self.path_dict with the path to save the info. 

        """
        path_to_directory_results = self.path_to_share_directory.parent.parent.parent.parent/'src'/'DRIVE'/'calib_data'
        
        basic_name_of_folder = robot_arg+"_"+traction_arg+"_"+terrain_arg
        path_to_experiment_folder = path_to_directory_results/basic_name_of_folder
        print(path_to_experiment_folder,2)

        if os.path.isdir(path_to_experiment_folder):
            now = datetime.datetime.now()
            basic_name_of_folder = basic_name_of_folder + f"_{now.year}_{now.month}_{now.day}_{now.hour}h{now.minute}s{now.second}"
            path_to_experiment_folder = path_to_directory_results/basic_name_of_folder
            print(path_to_experiment_folder,3)

        path_to_experiment_folder = path_to_experiment_folder
        path_config_folder = path_to_experiment_folder/"config_file_used"
        path_model_training_datasets = path_to_experiment_folder/"model_training_datasets"
        path_model_training_results = path_to_experiment_folder/"model_training_results"

        self.path_dict = {"path_experiment_folder":path_to_experiment_folder,
                    "path_config_folder":path_config_folder,
                    "path_model_training_datasets":path_model_training_datasets,
                    "path_model_training_results":path_model_training_results}
                    
        
        # Create directory  
        for keyys, _path in self.path_dict.items():
            _path.mkdir()
            self.path_dict[keyys] = str(_path)

        if self.only_execute_trajectories: # replace the parameters folder by the one included in the opinted experiment folder 

            folder_to_copy = ["model_training_datasets","model_training_results"]
            list_of_dest_path = [path_model_training_datasets,path_model_training_results]

            for i,folder in enumerate(folder_to_copy):
                self.get_logger().info("TEST_IN_CREATE_CONFIG  "+str(self.absolute_path_to_experiment_folder))
                folder_path_src = pathlib.Path(self.absolute_path_to_experiment_folder)/folder
                folder_path_dest = list_of_dest_path[i] # alredy path object
                folder_path_dest.rmdir()
                shutil.copytree(folder_path_src,folder_path_dest)

                #or file in list_files:
                #    file_src = folder_path_src/file
                #    file_dest = folder_path_dest/file
                #   shutil(file_src,file_dest)
            
            # Extract the path of all existing controller and load them as parameters
            
            list_controllers_learned = os.listdir(self.path_dict["path_model_training_results"])

            for training in list_controllers_learned:
                path_to_training_folder = pathlib.Path(self.path_dict["path_model_training_results"])/training
                
                list_file_and_folder_in_training = os.listdir(path_to_training_folder)

                for file_or_folder in list_file_and_folder_in_training:

                    path_to_file_or_folder = path_to_training_folder/file_or_folder

                    if path_to_file_or_folder.is_file() and str(path_to_file_or_folder)[-5:] == ".yaml":
                        path_to_config_file_controller = path_to_file_or_folder
                        self.get_logger().info("_____________"+folder)
                        self.get_logger().info("_____________"+file_or_folder)
                        self.declare_parameter(name="/controller_available/"+training+"/"+file_or_folder[:-5],
                                                value=str(path_to_config_file_controller))

        
    def copy_config_files_used_in_this_experiments(self,robot_arg):
        """Copy the config files used in this experiments to be able to validate if everything is alright.
        Eventually add a validation on the config file before continuing. 

        Args:
            robot_arg (_type_): _description_
        """

        if self.only_execute_trajectories:
            path_to_config_filed = pathlib.Path(self.absolute_path_to_experiment_folder)/"config_file_used"

            list_files = os.listdir(path_to_config_filed)
            for file in list_files:
                
                path_file = path_to_config_filed / file
                path_to_destination = str(pathlib.Path(self.path_dict["path_config_folder"])/path_file)
                
                with open(str(path_file),'r') as f:
                    metadata_file =  yaml.load(f, Loader=yaml.SafeLoader)
            
                metadata_file["source_of_this_file"] = self.absolute_path_to_experiment_folder

                with open(str(path_to_destination),'w') as f:
                    yaml.dump(metadata_file,f, sort_keys=False, default_flow_style=False)
                
                self.path_dict["path_to_calibration_node_config"] = str(path_to_config_filed/ f"_{robot_arg}.config.yaml")

        
        else:
            driver_node_config_specific = f"_{robot_arg}.config.yaml"
            logger_node_config_specific = f"_{robot_arg}_logger.config.yaml"
            
            path_driver_node_config = self.path_to_share_directory / driver_node_config_specific
            path_logger_config = self.path_to_share_directory /logger_node_config_specific
            
            self.path_dict["path_to_calibration_node_config"] = str(path_driver_node_config)
            config_file_driver_node = str(path_driver_node_config)
            config_file_logger = str(path_logger_config)

            shutil.copy(path_driver_node_config,str(pathlib.Path(self.path_dict["path_config_folder"])/driver_node_config_specific))
            shutil.copy(path_logger_config,str(pathlib.Path(self.path_dict["path_config_folder"])/logger_node_config_specific))
            

        

    #SEVICES callback and action client
    
    def send_metadata_form_callback(self, request, response):
        """
        
        1. TODO: log the information provided in a yaml file and save 
            it metadata.yaml in path_experiment_folder/metadata.yaml. 
        2. Create the path to save the data. 
        3. Copy the config file used in the protocole
        4. TODO: Compare the config file with a validation yaml file 
            to be sure that important parameters have not changed.
        5. Publish the path  
        6. 
        
        """  
        self.robot_arg = request.robot
        self.traction_arg = request.traction_geometry
        self.terrain_arg = request.terrain
        experiment_description = request.experiment_description
        self.only_execute_trajectories = request.only_execute_trajectories
        self.absolute_path_to_experiment_folder = request.absolute_path_to_experiment_folder
        self.get_logger().info("test" +request.absolute_path_to_experiment_folder)
        
        tire_pressure_psi = request.tire_pressure_psi

        weather_arg = request.weather

        
        

        #2. Create the path to save the data. 
        self.create_folder(self.robot_arg,self.traction_arg,self.terrain_arg)
        
        ## 1.0 TODO: log the information provided in a yaml file and save 
        #    it metadata.yaml in path_experiment_folder/metadata.yaml.

        metadata = {"metadata":{"robot":self.robot_arg,
                    "traction":self.traction_arg,
                    "terrain":self.terrain_arg,
                    "weather":weather_arg,
                    "tire_pressure_psi":tire_pressure_psi},
                    "only_execute_trajectories":self.only_execute_trajectories,
                    "descriptions of the experiments":experiment_description,
                    "path_to_the_experiment_folder_that_contains_the_stolen_parameters":self.absolute_path_to_experiment_folder,
                    }
        pathlib_to_object = pathlib.Path(self.path_dict["path_experiment_folder"])/"metadata.yaml"
        pathlib_to_object.touch() # Create dump
        #self.get_logger().info("\n"*3+yaml.__version__+"\n"*3)
        
        with open(str(pathlib_to_object),'w') as f:
                metadata_file = yaml.dump(metadata,f, sort_keys=False, default_flow_style=False)
        

        #3. Copy the config file used in the protocole
        self.copy_config_files_used_in_this_experiments(self.robot_arg)

        #4. TODO: Compare the config file with a validation yaml file 
            #to be sure that important parameters have not changed.

        #5 Publish the path. ----> Transient local so should be always accessible evn if published once. 
        
        self.path_to_drive_experiment_folder_msg.path_to_experiment_folder = self.path_dict["path_experiment_folder"]
        self.path_to_drive_experiment_folder_msg.path_config_folder = self.path_dict["path_config_folder"]
        self.path_to_drive_experiment_folder_msg.path_model_training_datasets = self.path_dict["path_model_training_datasets"]
        self.path_to_drive_experiment_folder_msg.path_model_training_results = self.path_dict["path_model_training_results"]
        self.path_to_drive_experiment_folder_msg.path_to_calibration_node_config = self.path_dict["path_to_calibration_node_config"]
        
        self.publish_drive_maestro_path_to_drive_folder() 
        
        response.status_messages = "All path have been created and published, back of file has been done too. and meta data has been saved."
        #6 Changing the drive status to start mapping 
        self.drive_maestro_status_msg.data = self.gui_message["mapping"]["status_message"] 
        self.drive_maestro_operator_action_msg.data = self.gui_message["mapping"]["operator_action_message"]

        response.status_messages = f"The results of the experiments are log in {self.path_to_drive_experiment_folder_msg.path_to_experiment_folder}"   
        return response
    
    def init_execute_traj(self):
        """To delete
        """

        if self.maestro_mode == "ExecuteTraj":
            self.drive_maestro_status_msg
            # Path initialization
            # self.maestro_mode = 
            self.path_to_drive_experiment_folder_msg = PathTree()
            pathlib_experiment_folder=  pathlib.Path(self.get_parameter('experiment_folder_path').get_parameter_value().string_value)
            self.path_to_drive_experiment_folder_msg.path_to_experiment_folder =str(pathlib_experiment_folder)
            self.path_to_drive_experiment_folder_msg.path_config_folder = str(pathlib_experiment_folder /"config_file_used")
            self.path_to_drive_experiment_folder_msg.path_model_training_datasets = str(pathlib_experiment_folder/ "model_training_datasets")
            self.path_to_drive_experiment_folder_msg.path_model_training_results =  str(pathlib_experiment_folder/"model_training_results")
            # Load the metadat_yaml to extract the param robot, traction and wheels
            pathlib_to_object = pathlib.Path(self.path_to_drive_experiment_folder_msg.path_to_experiment_folder)/"metadata.yaml"
        
            with open(str(pathlib_to_object),'w') as f:
                metadata_file =  yaml.load(f, Loader=yaml.SafeLoader)
            
            self.robot_arg = metadata_file["metadata"]["robot"]
            self.traction_arg = metadata_file["metadata"]["traction"]
            self.terrain_arg = metadata_file["metadata"]["terrain"]
            
            # Publish the state
            self.drive_maestro_operator_action_msg.data = self.gui_message["load_trajectory"]["operator_action_message"]
            self.drive_maestro_status_msg.data = self.gui_message["load_trajectory"]["status_message"] #init
        
    def stop_mapping_service(self, request, response):
        """Call the service training  motion model in the motion model training dataset. 
        2. Adjust the operator message and action. Shout out  ce og:
          https://answers.ros.org/question/373169/mistakes-using-service-and-client-in-same-node-ros2-python/.
        Args:
            request (_type_): _description_
            response (_type_): _description_

        Returns:
            _type_: _description_
        """

        if (self.drive_maestro_status_msg.data == self.gui_message["mapping"]["status_message"]) or (self.drive_maestro_status_msg.data == self.gui_message["load_trajectory"]["status_message"]):
            
            self.stop_mapping_client.wait_for_service()
            self.stop_mapping_req = Empty.Request() 
            self.future = self.stop_mapping_client.call_async(self.stop_mapping_req)
            rclpy.spin_until_future_complete(self.sub_node, self.future)
            answer = self.future.result()
            
            if self.future.result() is not None:

                if self.only_execute_trajectories:
                    self.drive_maestro_status_msg.data = self.gui_message["load_trajectory"]["status_message"]
                    self.drive_maestro_operator_action_msg.data = self.gui_message["load_trajectory"]["operator_action_message"]
                else:
                    self.drive_maestro_status_msg.data = self.gui_message["drive_ready"]["status_message"]
                    self.drive_maestro_operator_action_msg.data = self.gui_message["drive_ready"]["operator_action_message"]
            
                response.success = True
                response.message = "The mapping has been stopped and drive is ready to start."
                self.get_logger().info("The mapping has been stopped")
            else:
                response.success = False
                response.message = "The map has not been stopped."
                self.get_logger().info("There has been a problem stopping the mapping")
                
        else:
            response.success = False
            response.message = f"Not in the correct status, you should be in the {self.gui_message['mapping']['status_message']},but you are in the {self.drive_maestro_status_msg.data}"
        
        return response


    def save_logger_dataset_service_client(self,request,response):
        """
        
        1. Call the service  "save_data_callback(self, req, res)" in the logger node 
        2. Adjust the operator message and action. 

        Args:
            request (_type_): _description_
            response (_type_): _description_
        """

        
        if self.drive_maestro_status_msg.data == self.gui_message["drive_save"]["status_message"]:
            
            self.save_calibration_dataset_client.wait_for_service()
            req = ExportData.Request()

            path_to_dataset = pathlib.Path(self.path_to_drive_experiment_folder_msg.path_model_training_datasets) /"data_raw.pkl"

            path = String()
            path.data = str(path_to_dataset)
            req.export_path = path

            future = self.save_calibration_dataset_client.call_async(req)
            rclpy.spin_until_future_complete(self.sub_node, future)
            answer = future.result()
            self.get_logger().info("\n"*4+str(answer)+"\n"*4)

            if future.result() is not None:
                self.drive_maestro_status_msg.data = self.gui_message["model_training"]["status_message"]
                self.drive_maestro_operator_action_msg.data = self.gui_message["model_training"]["operator_action_message"]
            
                response.success = True
                response.message = f"The dataset has been saved to the following path {path_to_dataset}."
            else:
                response.success = False
                response.message = "The dataset was not saved has not been stopped."
        else:
            response.success = False
            response.message = f"Not in the correct status, you should be in the {self.gui_message['drive_save']['status_message']},but you are in the {self.drive_maestro_status_msg.data}"
        

            


        return response
        
        
        


    def model_training_service_client(self, request, response):
        """Call the service training  motion model in the motion model training dataset. 
        2. Adjust the operator message and action. 
        Args:
            request (_type_): _description_
            response (_type_): _description_

        Returns:
            _type_: _description_
        """

        if self.drive_maestro_status_msg.data == self.gui_message["model_training"]["status_message"] or self.drive_maestro_status_msg.data  ==  self.gui_message["load_trajectory"]["status_message"]:
            

            self.train_motion_model_client.wait_for_service()
            req = TrainMotionModel.Request()
            req.motion_model = request.motion_model
            req.number_of_seconds_2_train_on = request.number_of_seconds_2_train_on
            future = self.train_motion_model_client.call_async(req)
            rclpy.spin_until_future_complete(self.sub_node, future)
            answer = future.result()
            #self.get_logger().info("\n"*4+str(answer)+"\n"*4)

            if future.result() is not None:
                response.training_results = answer.training_results

                if req.motion_model == "all":
                    self.drive_maestro_status_msg.data = self.gui_message["load_trajectory"]["status_message"]
                    self.drive_maestro_operator_action_msg.data = self.gui_message["load_trajectory"]["operator_action_message"]
                    
                    self.create_controller_config_file("ideal-diff-drive-mpc", answer.path_to_trained_param_folder)
                    self.create_controller_config_file("pwrtrn-diff-drive-mpc", answer.path_to_trained_param_folder)
                    self.create_controller_config_file("slip-blr-pwrtrn-diff-drive-mpc", answer.path_to_trained_param_folder)
            else:
                
                response.training_results = "The training did not work"
        else:
            response.training_results = f"Not in the correct status '{self.drive_maestro_status_msg.data}', you should be in the {self.gui_message['model_training']['status_message']},but you are in the {self.drive_maestro_status_msg.data}"

        return response
        
    def load_trajectory_callback(self,request,response):


        # Extract position 
        pose_extracted =self.pose.pose.pose
        orientation = pose_extracted.orientation
        translation_2d = np.array([pose_extracted.position.x, pose_extracted.position.y,1])
        rotation = Rotation.from_quat(np.array([orientation.x,orientation.y,orientation.z,orientation.w]))
        yaw_rotation = rotation.as_euler("zyx")[0]

        self.get_logger().info(str(translation_2d))

        transform_2d = np.zeros((3,3))
        transform_2d[:,2] = translation_2d
        transform_2d[:2,:2] = np.array([[np.cos(yaw_rotation),-np.sin(yaw_rotation)],[np.sin(yaw_rotation),np.cos(yaw_rotation)]])

        # Pass the transform 

        
        #self.get_logger().info("\n"*3 +str(transform_2d)+"\n"*3 )
        #self.get_logger().info()

        # Type of controller posible []
        list_possible_trajectory = ["eight","rectangle","spin2win"]

        trajectory_type = request.trajectory_type
        trajectory_args = request.trajectory_args
        nb_repetition = request.nbr_repetition_of_each_speed
        frame_id = request.frame_id

        if trajectory_type in list_possible_trajectory:
            
            if trajectory_type == "eight":
                radius, entre_axe,horizon = trajectory_args
                if entre_axe >= 2*radius:
                    trajectory_generator = EightTrajectoryGenerator(radius,entre_axe,horizon)
                    trajectory_generator.compute_trajectory(number_of_laps=nb_repetition)
                    time_stamp = self.get_clock().now().to_msg()
                    self._path_to_execute,visualize_path_ros = trajectory_generator.export_2_norlab_controller(time_stamp,
                                                                                            frame_id,transform_2d)
                    self.path_loaded_pub.publish(visualize_path_ros)
                    response.success = True
                    response.message = f"The trajectory has been load. To visualize it, open the topic {self.visualise_path_topic_name} "
                else:
                    response.success = False
                    response.message = f"The trajectory has not been load the 'entreaxe' needs to be at least two times bigger than the radius."

                
            elif trajectory_type =="spin2win":
                
                n_tour,discretisation,sens_horaire = trajectory_args
                trajectory_generator = TurnAround(n_tour,np.deg2rad(discretisation),sens_horaire,transform_2d)
                trajectory_generator.compute_trajectory(number_of_laps=nb_repetition)

                time_stamp = self.get_clock().now().to_msg()
                self._path_to_execute ,visualize_path_ros = trajectory_generator.export_2_norlab_controller(time_stamp,
                                                                                        frame_id,transform_2d)
                self.get_logger().info(str(visualize_path_ros))
                self.path_loaded_pub.publish(visualize_path_ros)
                

                response.success = True
                response.message = f"The trajectory has been load. To visualize it, open the topic {self.visualise_path_topic_name} "
            
            elif trajectory_type =="rectangle":
                
                width, lenght,horizon = trajectory_args
                trajectory_generator = RectangleTrajectoryGenerator(width,lenght,horizon)
                trajectory_generator.compute_trajectory(number_of_laps=nb_repetition)
                time_stamp = self.get_clock().now().to_msg()
                self._path_to_execute ,visualize_path_ros = trajectory_generator.export_2_norlab_controller(time_stamp,
                                                                                        frame_id,transform_2d)
                self.get_logger().info(str(visualize_path_ros))
                self.path_loaded_pub.publish(visualize_path_ros)
                

                response.success = True
                response.message = f"The trajectory has been load. To visualize it, open the topic {self.visualise_path_topic_name} "

        else:
            response.success = False
            response.message = f"WRONG TRAJECTORY TYPE: please write one of the folowing trajectory type {list_possible_trajectory}"
        
        return response

    def confirm_traj_callback(self,request,answer):
        """
        Change the maestro_status from load_traj to load controller
        """
        self.drive_maestro_status_msg.data = self.gui_message["load_controller"]["status_message"]
        self.drive_maestro_operator_action_msg.data = self.gui_message["load_controller"]["operator_action_message"]
                    

        return answer


    def send_follow_path_goal(self):
        goal_msg = FollowPath.Goal()
        goal_msg.follower_options = self.follower_option
        goal_msg.path = self._path_to_execute
        self._action_client.wait_for_server()
        
        self._send_follow_path_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_follow_path_goal_future.add_done_callback(self.goal_response_followback_callback)

    def goal_response_followback_callback(self,future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('"Goal was rejected by server"')
            return

        self.get_logger().info('"Goal accepted by server, waiting for result"')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_followback_callback)

    def get_result_followback_callback(self, future):
        result = future.result().result
        if result.result_status.data == 1:
            self.get_logger().info(f'The trajectory has been completed, going back to load_traj_status')

        self.drive_maestro_status_msg.data = self.gui_message["load_trajectory"]["status_message"]
        self.drive_maestro_operator_action_msg.data = self.gui_message["load_trajectory"]["operator_action_message"]


    def create_controller_config_file(self,controller_name,path_to_training_folder):
        # controller_name can either be ideal-diff-drive-mpc, pwrtrn-diff-drive-mpc, slip-blr-pwrtrn-diff-drive-mpc

        # Controller param file
        self.path_to_share_directory = pathlib.Path(get_package_share_directory('drive'))
        
        path_to_controller_config = self.path_to_share_directory.parent.parent.parent.parent/'src'/'DRIVE'/'default_param'/'controller_params.yaml'
        
        with open(path_to_controller_config, 'r') as f:
            controller_params = yaml.load(f, Loader=yaml.SafeLoader)
        self.get_logger().info(str(controller_params))

        
        x_steps_training_folder = pathlib.Path(path_to_training_folder)

        x_steps_name = x_steps_training_folder.parts[-1]
        
        # Load the param of the motion model 
        if controller_name == "ideal-diff-drive-mpc":
            self.get_logger().info(f"Creating the ideal_diff_drive_config_file for {x_steps_name}")
            controller_params["controller_name"] = 'IdealDiffDriveMPC'

        elif controller_name == "pwrtrn-diff-drive-mpc": #GET LE NAME ET LE param manquant self.angular gain.
            path_to_pwrtrain_param = x_steps_training_folder/"powertrain"
            controller_params["pwrtrn_param_path"] = str(path_to_pwrtrain_param)
            self.get_logger().info("Creating the pwtrain config_file")
            controller_params["controller_name"] = 'PwrtrnDiffDriveMPC'

        elif controller_name == "slip-blr-pwrtrn-diff-drive-mpc":
            path_to_pwrtrain_param = x_steps_training_folder/"powertrain"
            controller_params["pwrtrn_param_path"] = str(path_to_pwrtrain_param)
            path_to_slip_blr_pwrtrain = x_steps_training_folder/"slip_blr"
            controller_params["slip_blr_param_path"] = str(path_to_slip_blr_pwrtrain)
            self.get_logger().info("Creating the slip_blr_pwrtrain")
            controller_params["controller_name"] = 'SlipBLRPwrtrnDiffDriveMPC'
        else:
            self.get_logger().warning("The controller name passed in the service is not alright")
            
        path_to_config_file_saved = x_steps_training_folder/(controller_name+".yaml")

        if path_to_config_file_saved.exists() == False:
            with open(path_to_config_file_saved, 'w') as f:
                yaml.dump(controller_params,f, sort_keys=False, default_flow_style=False)
        
        self.get_logger().info("_____________"+x_steps_name)
        self.get_logger().info("_____________"+controller_name)
        training_name = path_to_training_folder
        self.declare_parameter(name="/controller_available/"+x_steps_name+"/"+controller_name,value=str(path_to_config_file_saved))

    def load_controller_callback(self,request,response):

        
        path_to_controller_config_file = String(data = request.path_to_controller_config_file)
        
        req = ChangeController.Request()
        req.controller_config_path = path_to_controller_config_file
        future = self.change_controller_client.call_async(req)
        rclpy.spin_until_future_complete(self.sub_node, future)
        answer = future.result()

        ### TODO: add exception management and failing exception.
        response.success = True
        response.response = "YOU NEED TO VERIFY THAT THE PARAMETER HAVE CHANGED IN THE PARAMETER TAB"

        return response
    
    def load_max_speed_callback(self,request,response):
        if self.drive_maestro_status_msg.data == self.gui_message["play_traj"]["status_message"]:
            
            response.success = False
            response.message = "A trajectory is already being played based on the status"
        else:

            self.get_logger().warning("trying to set the maximum speed")
            
            # Create a Parameter message
            maximum_speed_linear_param = Parameter()
            maximum_speed_linear_param.name = "maximum_linear_velocity"
            # Set the parameter value type to integer
            maximum_speed_linear_param.value.type = ParameterType.PARAMETER_DOUBLE
            maximum_speed_linear_param.value.double_value = float(request.maximum_linear_speed)
            
            # Create a Parameter message
            maximum_speed_angular_param = Parameter()
            maximum_speed_angular_param.name = "maximum_angular_velocity"
            # Set the parameter value type to integer
            maximum_speed_angular_param.value.type = ParameterType.PARAMETER_DOUBLE
            maximum_speed_angular_param.value.double_value = float(request.maximum_angular_speed)

            
            # Set maximum linear speed
            req = SetParameters.Request()
            req.parameters =  [maximum_speed_linear_param,maximum_speed_angular_param]      
            future = self.maximum_speed_client.call_async(req)
            rclpy.spin_until_future_complete(self.sub_node, future)
            answer = future.result()
            
            response.success = True
            msg = ""
            for set_parameter_results in answer.results:

                if set_parameter_results.successful == False:
                    response.success = False
                    msg += f"at least one parameter have not been successfully set because :" +response.reason
            
            response.message = msg
            
            
            return response
    def execute_all_trajectories_call_back(self,request,response):
    
        #if self.drive_maestro_status_msg.data == self.gui_message["play_traj"]["status_message"]:
        #    
        #    response.success = False
        #    response.message = "A trajectory is already being played based on the status"
        #else:
            
        ## Send goal 
        list_possible_controller = ["ideal-diff-drive-mpc","pwrtrn-diff-drive-mpc","slip-blr-pwrtrn-diff-drive-mpc"]

        
        init_mode_ = UInt32()
        init_mode_.data = 1
        max_speed = Float32()
        max_speed.data = 1.0
        self.follower_option = FollowerOptions()
        self.follower_option.init_mode = init_mode_
        self.follower_option.velocity = max_speed

        future = self.send_follow_path_goal()

            # Used the precedent_traj and call the action client 
        #    self.drive_maestro_status_msg.data = self.gui_message["play_traj"]["status_message"]
        #    self.drive_maestro_operator_action_msg.data = self.gui_message["play_traj"]["operator_action_message"]
        #    
        #    response.success = False
        #    response.message = f"The controller name is bad, it should be one of the following: {list_possible_controller}"

        return response

def main():
    rclpy.init()
    drive_maestro = DriveMaestroNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(drive_maestro)
    try:
        drive_maestro.get_logger().info('Beginning drive_maestro_node')
        executor.spin()
    except KeyboardInterrupt:
        drive_maestro.get_logger().info('Keyboard interrupt, shutting down.\n')
    drive_maestro.destroy_node()
    rclpy.shutdown()

    #service_from_service = DriveMaestroNode()

    #executor = MultiThreadedExecutor()
    #rclpy.spin(service_from_service, executor)

    #rclpy.shutdown()

if __name__ == '__main__':
    main()