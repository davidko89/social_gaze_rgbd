import cv2
from pathlib import Path

ROOT_PATH = Path("D:/data")
RAW_DATA_PATH = ROOT_PATH.joinpath("raw_data_visual")
OUTPUT_PATH = ROOT_PATH.joinpath("raw_data")

def mkv_to_avi(input_filename, output_filename, fps=15):
    # Open the input file
    cap = cv2.VideoCapture(str(input_filename))

    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(output_filename), fourcc, fps, (width, height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # Write the frame to the output file
            out.write(frame)
        else:
            break

    # Release everything when job is finished
    cap.release()
    out.release()

# Iterate over all .mkv files in the raw_data directory
for mkv_file in RAW_DATA_PATH.glob("*.mkv"):
    # Extract the participant ID from the file name
    participant_id = mkv_file.stem

    # Construct the output path for the .avi file
    avi_path = Path(OUTPUT_PATH, f"{participant_id}.avi")

    # If the output file already exist, skip processing this .mkv file
    if avi_path.exists():
        print(f"Skipping {participant_id}.mkv as the output .avi file already exist.")
        continue

    # Create the .avi file
    mkv_to_avi(mkv_file, avi_path)

