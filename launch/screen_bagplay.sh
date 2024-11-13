ROSBAG_PATH="$1"
PKL_FILE="$2"

echo "Rosbag path: $ROSBAG_PATH"
echo "Pkl file: $PKL_FILE"

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