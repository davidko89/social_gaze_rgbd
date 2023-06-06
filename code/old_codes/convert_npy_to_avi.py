import cv2
import numpy as np
from pathlib import Path

ROOT_PATH = Path("D:/data")
RAW_DATA_PATH = ROOT_PATH.joinpath("raw_data")
PROC_DATA_PATH = ROOT_PATH.joinpath("proc_data")
COLOR_DATA_PATH = PROC_DATA_PATH.joinpath("color")

def create_avi_from_rgb(rgb_data, output_filename, fps=15):
    num_frames, height, width, _ = rgb_data.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = cv2.cvtColor(rgb_data[i], cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

# Iterate over all .npy files in the color_data directory
for npy_file in COLOR_DATA_PATH.glob("*.npy"):
    # Extract the participant ID from the file name
    participant_id = npy_file.stem

    # Construct the output path for the .avi file
    avi_path = Path(RAW_DATA_PATH, f"{participant_id}.avi")

    # If the output file already exist, skip processing this .npy file
    if avi_path.exists():
        print(f"Skipping {participant_id}.npy as the output .avi file already exist.")
        continue

    # Load the numpy array
    rgb_data = np.load(npy_file)

    # Create the .avi file
    create_avi_from_rgb(rgb_data, str(avi_path))


