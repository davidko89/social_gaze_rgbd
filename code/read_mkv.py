#%%
import pickle
import open3d as o3d
import numpy as np

from pathlib import Path
import copy
import cv2
from tqdm import tqdm

PROJECT_PATH = Path(__file__).parents[1]
RAW_DATA_PATH = Path(PROJECT_PATH, "data/raw_data")
PROC_DATA_PATH = Path(PROJECT_PATH, "data/proc_data")
COLOR_DATA_PATH = Path(PROC_DATA_PATH, "rgb")
DEPTH_DATA_PATH = Path(PROC_DATA_PATH, "depth")

#%%


# Iterate over all .mkv files in the raw_data directory
for mkv_file in RAW_DATA_PATH.glob("*.mkv"):
    # Extract the participant ID from the file name
    participant_id = mkv_file.stem

    # Construct the paths for the color and depth data
    color_path = Path(COLOR_DATA_PATH, f"{participant_id}_color.npy")
    depth_path = Path(DEPTH_DATA_PATH, f"{participant_id}_depth.npy")

    # If the output files already exist, skip processing this .mkv file
    if color_path.exists() and depth_path.exists():
        print(f"Skipping {participant_id}.mkv as the output files already exist.")
        continue

    # Open the .mkv file and read all color and depth frames
    reader = o3d.io.AzureKinectMKVReader()
    reader.open(str(mkv_file))

    color_data = []
    depth_data = []
    pbar = tqdm(total = round((reader.get_metadata().stream_length_usec)*(1e-6)*30))
    i = 0
    while not reader.is_eof():
        pbar.update(1)
        frame = reader.next_frame()
        if (i % 3) != 2:
            i += 1
            continue
        if frame is not None:
            color_data.append(copy.deepcopy(np.asarray(frame.color)))
            depth_data.append(copy.deepcopy(np.asarray(frame.depth)))
        
        i += 1

    pbar.close()
    reader.close()

    print("Loading Finish.")
    color_data = np.array(color_data)
    depth_data = np.array(depth_data)
    
    # Save the color and depth data as .npy files
    np.save(str(color_path), color_data)
    del color_data
    np.save(str(depth_path), depth_data)
    del depth_data
    
    #pickle.dump(depth_data, open(str(depth_path), 'wb'))

    # Print some diagnostic information
    #print(f"Processed {len(color_data)} frames from {participant_id}.mkv")


#%% 
# # Load the saved color data from the .npy file
# color_path = "../data/proc_data/rgb/td001_color.npy"
# color_data = np.load(str(color_path))
# # Choose a frame index to visualize (e.g., the first frame)
# frame_index = 0
# cv2.imwrite(str(Path(COLOR_DATA_PATH, 'td001_frame0.png')), color_data[frame_index])

# scp -P 10022 -r "C:\2023_asd_gaze\data\proc_data\rgb" cko4@103.22.220.153:/home/cko4/2023_asd_gaze/data/proc_data/
# scp -P 10022 -r "C:\2023_asd_gaze\data\proc_data\depth" cko4@103.22.220.153:/home/cko4/2023_asd_gaze/data/proc_data/