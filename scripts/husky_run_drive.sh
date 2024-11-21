#!/bin/bash
#test=2
#echo -n "Enter experiment name, if empty the name will be the current date and time: "
#read experiment_name
#if [[ -z "$experiment_name" ]]; then
#   experiment_name="$(date '+%Y-%m-%d-%s')"
#fi


echo "Killing old screens"
screen -S "sensors" -X quit
screen -S "mapping" -X quit
screen -S "drive" -X quit
screen -S "records" -X quit
screen -S "controllers" -X quit
#screen -S "theodolite" -X quit
#screen -S "icp_visualization" -X quit

#echo "kill then run elie screens (rqt, joy_node, fox_glove_bridge)"
#screen -S "joyscreen" -X quit
#screen -S "rqt" -X quit
#screen -S "foxbridge" -X quit

#screen -dmS joyscreen ros2 run joy joy_node
#screen -dmS rqt rqt_graph
#screen -dmS foxbridge ros2 launch foxglove_bridge foxglove_bridge_launch.xml

echo "Starting sensors"
screen -dmS sensors ros2 launch norlab_robot sensors.launch.py
#screen -dmS records ros2 launch warthog_mapping sensors.launch.xml 
echo "Sensors started, access it with screen -r sensors"
sleep 2

echo "Starting mapping"
screen -dmS mapping ros2 launch norlab_robot mapping.launch.py
#screen -dmS mapping ros2 launch warthog_mapping realtime_mapping.launch.xml 

echo "Mapping started, access it with screen -r mapping"
sleep 2

echo "Launch drive in a screen name drive"
screen -dmS drive ros2 launch drive maestro_husky.launch.py #launch drive drive instead of drive wathog
sleep 2 


#echo "Launch the controller"
#screen -dmS controllers ros2 launch norlab_robot controller.launch.py traction:=wheels terrain:=ice controller:=ideal-diff-drive-mpc
#sleep 2
#echo "Starting theodolite"
#screen -dmS theodolite ros2 launch theodolite_pose theodolite_pose.launch.py
#echo "Theodolite started, access it with screen -r theodolite"

#echo "Starting icp_visualization"
#screen -dmS visualization ros2 launch theodolite_pose icp_pose.launch.py
#echo "Visualization started, access it with screen -r icp_visualization"

cd /home/robot/data
echo "Starting the record in the screen records"
screen -dmS records ros2 bag record -a 
sleep 2


# Define the service name and type
SERVICE_NAME="/drive/send_metadata_form"
SERVICE_TYPE="drive_custom_srv/srv/DriveMetaData"

# Prompt for each parameter
read -p "Enter terrain: " TERRAIN
#read -p "Enter robot: " ROBOT
ROBOT="husky"

#read -p "Enter traction geometry: " TRACTION_GEOMETRY
TRACTION_GEOMETRY="wheels"

read -p "Enter weather: " WEATHER
read -p "Enter experiment description: " EXPERIMENT_DESCRIPTION
#read -p "Only execute trajectories (true/false): " ONLY_EXECUTE_TRAJECTORIES
ONLY_EXECUTE_TRAJECTORIES=false

#read -p "Enter absolute path to experiment folder: " ABSOLUTE_PATH_TO_EXPERIMENT_FOLDER
ABSOLUTE_PATH_TO_EXPERIMENT_FOLDER=""
read -p "Enter tire pressure (PSI): " TIRE_PRESSURE_PSI

# Call the ROS 2 service
ros2 service call "$SERVICE_NAME" "$SERVICE_TYPE" "terrain: '$TERRAIN'
robot: '$ROBOT'
traction_geometry: '$TRACTION_GEOMETRY'
weather: '$WEATHER'
experiment_description: '$EXPERIMENT_DESCRIPTION'
only_execute_trajectories: $ONLY_EXECUTE_TRAJECTORIES
absolute_path_to_experiment_folder: '$ABSOLUTE_PATH_TO_EXPERIMENT_FOLDER'
tire_pressure_psi: $TIRE_PRESSURE_PSI"

screen -r drive
#
#
#
#
#
##echo "All screen have been started, open another terminal and enter the screen 'drive' to monitor DRIVE.
#    #Once you have entered the screen, press enter HERE."
##read continue
#
##screen -r drive
##screen -r records
#
#drive_done=0
#while [ "$drive_done" -eq 0 ]
#do
#   
#   echo "When you have finished DRIVE, BEFORE killing the screen 'drive', write 'finish' HERE and click enter to save the resulting dataset"
#   read start_drive
#   if [ $start_drive = "finish" ]; then drive_done=1; echo "$drive_done";fi
#done
#
#
#current_directory=$PWD
#export_path=""$PWD"/../calib_data/"$experiment_name"/"
#echo -n $export_path
#
#ros2 service call /path_to_folder drive_custom_srv/srv/BashToPath input:\ \'$export_path\'\
#
#if [ ! -d $export_path ]; then
#  mkdir -p $export_path
#fi
#ros2 service call /export_data norlab_controllers_msgs/srv/ExportData "export_path:
#  data: '"$export_path"/raw_dataframe.pkl'"
#
#sleep 5
#
#
#
#killall drive_node
#killall logger_node
##sleep 2
##read -p "When dataset gathering is done, press Enter to save"
#