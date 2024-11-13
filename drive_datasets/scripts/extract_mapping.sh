#!/bin/bash

# Check if a path argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_rosbag>"
    exit 1
fi

# Define the path to the rosbag and the result folder
ROSBAG_PATH="$1"
BASENAME=$(basename "$ROSBAG_PATH")  # Get the base name without extension
RESULT_FOLDER="$(dirname "$ROSBAG_PATH")/${BASENAME}_without_mapping"
YAML_FILE="$RESULT_FOLDER/out.yaml"

# Create the result folder if it doesn't exist
mkdir -p "$RESULT_FOLDER"

# Write to the YAML file
cat <<EOL > "$YAML_FILE"
output_bags:
- uri: "$RESULT_FOLDER/${BASENAME}_to_remap"
  topics: [/vn100/vectornav/raw/imu, /left_drive/status/battery_current_corrected, /controller/ref_path, /drive/good_calib_step, /drive/imediate_path, /drive/operator_action_calibration, /vn100/data_unbiased, /vn100/vectornav/raw/time, /drive/calib_step, /left_drive/velocity, /left_drive/status/battery_current, /warthog_velocity_controller/odom, /tf_static, /mti30/velocity, /mti30/data_raw, /parameter_events, /vn100/data, /vn100/bias, /mti30/data, /mti30/bias, /mti30/imu_data_str, /rosout, /diagnostics, /right_drive/status/battery_voltage,/left_drive/status/speed, /right_drive/status/speed, /robot_description, /vn100/data_raw, /left_drive/status/battery_voltage, /right_drive/status/battery_current_corrected, /rslidar128/points, /vn100/velocity_body, /vn100/time_syncin, /warthog_velocity_controller/cmd_vel, /drive/model_trainer_node_status, /doughnut_cmd_vel, /drive/calib_state, /vn100/pose, /drive/maestro_status, /drive/path_to_reapeat, /controller/target_path]
  storage_id: "mcap"
EOL

echo "YAML file '$YAML_FILE' created successfully."


### Executing the filtering

source ~/workspaces/drive_ws/install/setup.bash 

ros2 bag convert -i $ROSBAG_PATH -o $YAML_FILE
