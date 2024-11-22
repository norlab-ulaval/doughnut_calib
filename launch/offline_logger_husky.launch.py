import os
import yaml
import errno

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch_ros.actions import Node
import pathlib


def launch_drive_orechestra(context, *args, **kwargs):


    # select the config file of the robot
    path_to_share_directory = pathlib.Path(get_package_share_directory('drive'))
    
    logger_node_config_specific = f"_husky_logger.config.yaml"
    config_file_logger = str(path_to_share_directory /logger_node_config_specific)

    # Logger node node
    logger_node = Node(
    package='drive',
    executable='husky_pose_cmds_logger_node',
    name="logger_node",
    output='screen',
    parameters=[config_file_logger],
    remappings=[                    
        ("odometry_in","/mapping/icp_odom"),
        ("imu_in","/mti100/data"), # valider l'IMU 
        ('/doughnut_cmd_vel', "/controller/cmd_vel" )
        ],
    namespace="drive"
    )

    return [logger_node
            ]


def generate_launch_description():

    
    return LaunchDescription([
        #robot_argument,
        #traction_mechanism,
        #terrain_argument,
        OpaqueFunction(function=launch_drive_orechestra)
    ])