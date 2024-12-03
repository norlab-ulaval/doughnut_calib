#!/bin/bash

# kill leftover screens from previous bag
killall screen

# Check if a path argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_rosbag>"
    exit 1
fi

# Define the path to the rosbag and the result folder
INPUT_FOLDER="$1"
echo $INPUT_FOLDER

BASENAME="${INPUT_FOLDER:55:-16}"  # Get the base name without extension
echo $BASENAME

ROSBAG_PATH="$INPUT_FOLDER/${BASENAME}_to_remap"
echo $ROSBAG_PATH

RESULT_FOLDER="$INPUT_FOLDER/models_training_datasets"
echo $RESULT_FOLDER

MAP_FILE="$INPUT_FOLDER/map.csv"
echo $MAP_FILE

ROSBAG_RECORD="$INPUT_FOLDER/odom_bag"
echo $ROSBAG_RECORD

mkdir -p "$RESULT_FOLDER"

PKL_FILE="$RESULT_FOLDER/raw_dataframe.pkl"
echo $PKL_FILE

# Define the file path where the YAML content will be saved
config_file="/home/william/workspaces/drive_ws/src/norlab_robot/config/_drive_icp_mapper.yaml"

init_pose_file="$INPUT_FOLDER/init_pose.txt"
INIT_POSE=$(<"$init_pose_file")
echo $INIT_POSE

# Write the content to the file
cat <<EOL > "$config_file"
/**:
  icp_mapper:
    ros__parameters:
      robot_frame: "base_link"
      odom_frame: "odom"
      map_tf_publish_rate: 100.0
      initial_map_file_name: "$MAP_FILE"
      initial_robot_pose: "$INIT_POSE"
      is_mapping: false
      use_sim_time: true
EOL

# launch the record in a screen
screen -dmS record ros2 bag record /mapping/icp_odom -o "$ROSBAG_RECORD" -s mcap --use-sim-time

# launch the logger in a screen
screen -dmS drive_logger ros2 launch drive offline_logger_warthog.launch.py

# launch the mapping (loc mode) in a screen
screen -dmS mapping ros2 launch drive offline_mapping.launch.py

# Play the rosbag in the current terminal so we can pipe the service call after it.
ros2 bag play -r 0.1 "$ROSBAG_PATH" --clock

# call the service to save the logger data
ros2 service call /drive/export_data norlab_controllers_msgs/srv/ExportData "export_path:
       data: '"$PKL_FILE"'"

# stop the record screen
screen -S record -X stuff $'\003'

# stop all other screens
screen -S drive_logger -X stuff $'\003'
screen -S mapping -X stuff $'\003'