import numpy as np
import pyaudio
import wave
import cv2
import open3d as o3d
import time

# Initialize the Azure Kinect sensor
o3d.logging.set_verbosity(o3d.logging.ERROR)
device = o3d.azure_kinect.device.AzureKinectDevice()
device.start()
color_stream = device.create_color_stream()
depth_stream = device.create_depth_stream()

# Set the color and depth capture settings
color_stream.start()
color_stream.set_capture_option(http://o3d.io.AzureKinectSensorConfig.COLOR_RESOLUTION, http://o3d.io.AzureKinectSensorConfig.ColorResolution.RESOLUTION_720P)
color_stream.set_capture_option(http://o3d.io.AzureKinectSensorConfig.FPS, http://o3d.io.AzureKinectSensorConfig.FPS_30)
depth_stream.start()
depth_stream.set_capture_option(http://o3d.io.AzureKinectSensorConfig.DEPTH_MODE, http://o3d.io.AzureKinectSensorConfig.DepthMode.NFOV_UNBINNED)
depth_stream.set_capture_option(http://o3d.io.AzureKinectSensorConfig.FPS, http://o3d.io.AzureKinectSensorConfig.FPS_30)

# Initialize the audio stream
audio_format = pyaudio.paInt16
sample_rate = 16000
chunk_size = 1024
audio_channels = 1

audio_stream = pyaudio.PyAudio().open(format=audio_format,
                                       channels=audio_channels,
                                       rate=sample_rate,
                                       input=True,
                                       frames_per_buffer=chunk_size)

# Record RGB and audio data and save to a synchronized video file
fourcc = cv2.VideoWriter_fourcc(*‘mp4v’)
out = cv2.VideoWriter(‘output.mp4’, fourcc, 30.0, (1280, 720))

frames_rgb = []
frames_depth = []
frames_audio = []

for i in range(0, int(sample_rate / chunk_size * 5)):
    # Record RGB and depth data
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_stream.capture_frame().color,
        depth_stream.capture_frame().depth,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False
    )

    color_array = np.asarray(rgbd.color)
    depth_array = np.asarray(rgbd.depth)

    frames_rgb.append(color_array)
    frames_depth.append(depth_array)

    # Record audio data
    audio_data = audio_stream.read(chunk_size)
    frames_audio.append(audio_data)

    # Save synchronized RGB and audio frames to video file
    timestamp = int(time.time() * 1000000)
    if len(frames_audio) > 1 and len(frames_rgb) > 1:
        audio_timestamp = (i - 1) * chunk_size / sample_rate
        while abs(audio_timestamp - (timestamp / 1000000)) > 0.005:
            if audio_timestamp < timestamp / 1000000:
                audio_data = frames_audio.pop(0)
                audio_timestamp += chunk_size / sample_rate
            else:
                rgb_data = frames_rgb.pop(0)
                depth_data = frames_depth.pop(0)
                out.write(cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB))

# Stop the Azure Kinect sensor and release resources
color_stream.stop()
depth_stream.stop()
device.stop()

audio_stream.stop_stream()
audio_stream.close()

out.release()
cv2.destroyAllWindows()