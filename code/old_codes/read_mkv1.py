import open3d as o3d
import numpy as np
from pathlib import Path
import cv2
import copy

PROJECT_PATH = Path(__file__).parents[1]
RAW_DATA_PATH = Path(PROJECT_PATH, "data/raw_data")
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")
COLOR_DATA_PATH = Path(PROC_DATA_PATH, "rgb")
DEPTH_DATA_PATH = Path(PROC_DATA_PATH, "depth")

# Iterate over all .mkv files in the raw_data directory
for mkv_file in RAW_DATA_PATH.glob("*.mkv"):
    # Extract the participant ID and task name from the file name
    file_name = mkv_file.stem
    participant_id, task_name = file_name.split("_")
    
    # Open the .mkv file and read all color and depth frames
    reader = o3d.io.AzureKinectMKVReader()
    reader.open(str(mkv_file))

    color_data = None
    depth_data = None

    while not reader.is_eof():
        frame = reader.next_frame()

        if frame is not None:
            color_image = copy.deepcopy(np.asarray(frame.color))
            depth_image = copy.deepcopy(np.asarray(frame.depth))

            if color_data is None:
                color_data = color_image[np.newaxis, ...]
            else:
                color_data = np.concatenate((color_data, color_image[np.newaxis, ...]), axis=0)

            if depth_data is None:
                depth_data = depth_image[np.newaxis, ...]
            else:
                depth_data = np.concatenate((depth_data, depth_image[np.newaxis, ...]), axis=0)

    # Save the color and depth data as .npy files
    color_path = Path(COLOR_DATA_PATH, f"{participant_id}_{task_name}_color.npy")
    depth_path = Path(DEPTH_DATA_PATH, f"{participant_id}_{task_name}_depth.npy")

    np.save(str(color_path), color_data)
    np.save(str(depth_path), depth_data)
