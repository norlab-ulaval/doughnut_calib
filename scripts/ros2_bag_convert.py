import threading
import os
import time
import shutil
import argparse
import subprocess
import yaml
from tqdm import tqdm


def print_file_size(input_size, output_path):
    time.sleep(2)
    with tqdm(total=input_size, desc="Processing", unit="B", unit_scale=True) as pbar:
        previous_output_size = 0
        while True:
            try:
                output_size = sum(
                    os.path.getsize(os.path.join(output_path, f))
                    for f in os.listdir(output_path)
                    if os.path.isfile(os.path.join(output_path, f))
                )
                # Update progress only if output size has increased
                pbar.update(output_size - previous_output_size)
                previous_output_size = output_size

                # Stop if output size meets or exceeds input size
                if output_size >= input_size:
                    print("Processing complete.")
                    break
            except Exception:
                pass
            time.sleep(0.2)
    print("Indexing...")


def main(input_path, output_path, overwrite):
    if os.path.exists(output_path) and overwrite:
        shutil.rmtree(output_path)
    if len(input_path) > 1:
        print("Merging multiple rosbag files")
    else:
        print("Converting db3 to mcap")

    input_size = 0
    for path in input_path:
        path = path[0]
        for root, _, files in os.walk(path):
            for file in files:
                try:
                    input_size += os.path.getsize(os.path.join(root, file))
                except PermissionError:
                    pass
    # Start a thread that checks file size
    thread = threading.Thread(
        target=print_file_size,
        args=(input_size, output_path),
    )
    thread.daemon = True  # This makes sure the thread exits when the main program exits
    thread.start()

    # Execute ros2 bag convert
    output_yaml = "/tmp/output_options.yaml"
    with open("/tmp/output_options.yaml", "w") as file:
        yaml.dump(
            {
                "output_bags": [
                    {
                        "uri": output_path,
                        "storage_id": "mcap",
                        "all_topics": True,
                        "all_services": True,
                    }
                ]
            },
            file,
            default_flow_style=False,
        )

    subprocess_cmd = ["ros2", "bag", "convert"]
    for path in input_path:
        subprocess_cmd += ["-i", path[0]]
    subprocess_cmd += ["-o", output_yaml]
    # print(subprocess_cmd)
    subprocess.run(subprocess_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO.")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Input file name.",
        action="append",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Output file name.",
        default=False,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing output bag.",
        action="store_true",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.overwrite)
