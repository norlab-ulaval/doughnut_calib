#!/bin/bash

# kill leftover screens from previous bag
killall screen

# launch a record in a screen?
# screen -dmS record ros2 bag record -a -o /home/william/data/DRIVE/ice/bag2

# launch the logger in a screen
screen -dmS drive_logger ros2 launch drive offline_logger_warthog.launch.py

# launch the mapping (loc mode) in a screen
screen -dmS mapping ros2 launch drive offline_mapping.launch.py

# Play the rosbag in the current terminal so we can pipe the service call after it.
ros2 bag play -r 0.05 /media/william/6757-F443/NIC/asphalt/rosbags/rosbag2_2024_09_20-08_22_47_without_mapping/rosbag2_2024_09_20-08_22_47_to_remap --clock

# call the service to save the logger data
ros2 service call /drive/export_data norlab_controllers_msgs/srv/ExportData "export_path:
       data: '/media/william/6757-F443/NIC/asphalt/rosbags/rosbag2_2024_09_20-08_22_47_without_mapping/models_training_datasets/raw_dataframe.pkl'"