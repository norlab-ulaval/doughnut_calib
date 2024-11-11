#!/bin/bash
produce_video="false"



if [ "$#" -eq 1 ]; then
# Extract the value of produce_video from the argument
    if [[ $1 == produce_video=* ]]; then
        produce_video="${1#*=}"  # Get the value after '='
    else
        echo "Error: Argument must be in the format 'produce_video=true' or 'produce_video=false'."
        exit 1
    fi
fi
cd ../.. #drive/model_training/data_utils || { echo "Directory not found"; exit 1; }


# Activate the Pyenv environment named 'data_analysis'
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


if ! pyenv versions | grep -q "data_analysis"; then
    echo "Virtual environment 'data_analysis' does not exist."
    exit 1
fi

pyenv activate data_analysis
pyenv local data_analysis

pwd
cd ../../..
pwd 

python3 drive/model_training/data_utils/update_drive_inventory.py --produce_video=$produce_video