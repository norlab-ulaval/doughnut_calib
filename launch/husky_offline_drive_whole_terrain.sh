#!/bin/bash

# Check if a folder path argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_folder>"
    exit 1
fi

INPUT_FOLDER="$1"

# Check if the input path is a directory
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: '$INPUT_FOLDER' is not a valid directory."
    exit 1
fi

# Check if the extract_mapping script exists in the current directory
if [ ! -f "./offline_drive.sh" ]; then
    echo "Error: 'offline_drive' script not found in the current directory."
    exit 1
fi

# Loop through each subdirectory in the provided folder
for dir in "$INPUT_FOLDER"/*/; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Extract the full path
        FULL_PATH="$(realpath "$dir")"
        
        # Call the extract_mapping script with the directory path as an argument
        echo "Calling husky_offline_drive with path: $FULL_PATH"
        ./husky_offline_drive.sh "$FULL_PATH"
    fi
done