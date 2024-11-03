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

    #robot_arg = context.perform_substitution(LaunchConfiguration('robot'))
    #terrain_arg = context.perform_substitution(LaunchConfiguration('terrain'))
    #traction_arg = context.perform_substitution(LaunchConfiguration('traction_gemoetry'))
    
    # select the config file of the robot
    path_to_share_directory = pathlib.Path(get_package_share_directory('drive'))
    
    driver_node_config_specific = f"_husky.config.yaml"
    logger_node_config_specific = f"_husky_logger.config.yaml"
    config_file_driver_node = str(path_to_share_directory / driver_node_config_specific)
    config_file_logger = str(path_to_share_directory /logger_node_config_specific)
    
    # Calibration node
    calibration_node = Node(
    package='drive',
    executable='calibration_node',
    name="calibration_node",
    output='screen',
    parameters=[
        config_file_driver_node],
    remappings=[
        ("odom_in", "/mapping/icp_odom"),
        ("joy_in","/teleop/joy"), # remap from="joy_in" to="hri_joy"  -->
        ("cmd_vel_out","/controller/cmd_vel"), #done
        ],
    namespace="drive"
    )
    
    
    
    
    # Logger node node
    logger_node = Node(
    package='drive',
    executable='pose_cmds_logger_node',
    name="logger_node",
    output='screen',
    parameters=[config_file_logger],
    remappings=[
        ("odometry_in","/mapping/icp_odom"),
        ("imu_in","mti100/data"), # valider l'IMU  # To do
        ("left_wheel_current_in","/status"),
        ("left_wheel_voltage_in","/status"),
        ("right_wheel_voltage_in","/status"),
        ("right_wheel_current_in","/status"), 
        ('/doughnut_cmd_vel', "/controller/cmd_vel" )
        ],
    namespace="drive"
    )
    ### Problème de ce noeud est que le remapping dépend des robots. donc 
    # même si launchfile bien fait on est fucked.
    return [
            calibration_node,
            logger_node,
            ]


def generate_launch_description():

    ## Start of One launchfile to rule them all but did not work because of the problem to pass
    # the info to the node. 
    #
    # Declare command line arguments
    #robot_argument = DeclareLaunchArgument(
    #    'robot',
    #    default_value="none",
    #    description='robot_config: [husky, marmotte, warthog_wheels, warthog_tracks]'
    #)
    #traction_mechanism = DeclareLaunchArgument(
    #    'traction_gemoetry',
    #    default_value="none",
    #    description='traction_gemoetry : [wheels, legs, tracks]'
    #)
#
    #terrain_argument = DeclareLaunchArgument(
    #    'terrain',
    #    default_value="none",
    #    description='Terrain type: [grass, gravel, ice_rink, snow]'
    #)

    
    return LaunchDescription([
        #robot_argument,
        #traction_mechanism,
        #terrain_argument,
        OpaqueFunction(function=launch_drive_orechestra)
    ])