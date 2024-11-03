#!/bin/bash

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