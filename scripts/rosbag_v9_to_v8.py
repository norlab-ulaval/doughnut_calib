import argparse
import os
from rosbags.rosbag2 import Writer, Reader
from tqdm import tqdm
import shutil

from typing import cast

from rosbags.interfaces import ConnectionExtRosbag2
from rosbags.typesys import get_types_from_msg, get_typestore, Stores

AUDIO_DATA_MSG = """
uint8[] data
"""
DRIVE_DATAPATH_MSG = """
string path_to_experiment_folder
string path_config_folder
string path_model_training_datasets
string path_model_training_results
string path_to_calibration_node_config
"""
HUSKY_STATUS_MSG = """
std_msgs/Header header

# MCU Uptime, in ms
uint32 uptime

# ROS Control loop frequency (PC-side)
float64 ros_control_loop_freq

# Current draw of platform components, in amps
float64 mcu_and_user_port_current
float64 left_driver_current
float64 right_driver_current

# Voltage of platform components, in volts
float64 battery_voltage
float64 left_driver_voltage
float64 right_driver_voltage

# Component temperatures, in C
float64 left_driver_temp
float64 right_driver_temp
float64 left_motor_temp
float64 right_motor_temp

# Battery capacity (Wh) and charge (%) estimate
uint16 capacity_estimate
float64 charge_estimate

# Husky error/stop conditions
bool timeout
bool lockout
bool e_stop
bool ros_pause
bool no_battery
bool current_limit
"""

TOPICS_INGORE = ["/lslidar16/packet",
                 "/audio/audio",
                 "/audio/audio_stamped",
                 "/lslidar16/sweep",
                 "/lslidar16/scan"]


def get_drive_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(
        get_types_from_msg(AUDIO_DATA_MSG, "audio_common_msgs/msg/AudioData")
    )
    typestore.register(
        get_types_from_msg(DRIVE_DATAPATH_MSG, "drive_custom_srv/msg/PathTree")
    )
    typestore.register(
        get_types_from_msg(HUSKY_STATUS_MSG, "husky_msgs/msg/HuskyStatus")
    )
    return typestore


def main(path: str, overwrite: bool):
    print(path)
    typestore = get_drive_typestore()
    input_split = path.split("/")
    output_path = "/".join(input_split[:-1] + ["v8_" + input_split[-1]])

    if os.path.exists(output_path) and overwrite:
        print(f"Overwriting existing output bag {output_path}.")
        shutil.rmtree(output_path)

    with Reader(path) as reader, Writer(output_path, version=8) as writer:
        conn_map = {}
        total_messages = reader.message_count

        for conn in reader.connections:
            ext = cast(ConnectionExtRosbag2, conn.ext)
            if conn.topic in TOPICS_INGORE:
                continue
            conn_map[conn.id] = writer.add_connection(
                conn.topic,
                conn.msgtype,
                serialization_format=ext.serialization_format,
                offered_qos_profiles=ext.offered_qos_profiles,
                typestore=typestore,
            )

        for conn, timestamp, rawdata in tqdm(
            reader.messages(connections=reader.connections),
            total=total_messages,
            desc="Processing input data",
        ):
            if conn.topic in TOPICS_INGORE:
                continue

            writer.write(
                conn_map[conn.id],
                timestamp,
                rawdata,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a version 9 rosbag to version 8."
    )
    parser.add_argument(
        "-p", "--rosbag_path", type=str, help="Path pointing to a ROS 2 bag file."
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing output bag.",
        action="store_true",
    )
    args = parser.parse_args()

    main(args.rosbag_path, args.overwrite)
