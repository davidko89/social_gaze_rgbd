import argparse
import os
import wave
import pyaudio
import open3d as o3d
import time

class RecorderWithCallback:

    def __init__(self, config, device, filename, align_depth_to_color):
        # Global flags
        self.flag_exit = False
        self.flag_record = False
        self.filename = filename

        self.align_depth_to_color = align_depth_to_color
        self.recorder = o3d.io.AzureKinectRecorder(config, device)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')

        # Audio recording settings
        self.audio_format = pyaudio.paInt16
        self.sample_rate = 16000  # Hertz
        self.chunk_size = 1024  # Each record will have 1024 frames
        self.audio_channels = 2

        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            start=True  # Start the audio stream immediately
        )

        self.audio_frames = []
        self.start_time = None  # Record the start time of the recording

    def escape_callback(self, vis):
        self.flag_exit = True
        if self.recorder.is_record_created():
            print('Recording finished.')
        else:
            print('Nothing has been recorded.')
        return False

    def space_callback(self, vis):
        if self.flag_record:
            print('Recording paused. Press [Space] to continue. Press [ESC] to save and exit.')
            self.flag_record = False
            self.audio_stream.stop_stream()  # Stop the audio stream

        elif not self.recorder.is_record_created():
            if self.recorder.open_record(f'D:/data/raw_data_visual/{self.filename}.mkv'):
                self.audio_frames = []  # Reset audio frames
                self.start_time = time.time()  # Record the start time
                self.audio_stream.start_stream()  # Start the audio stream
                print('Recording started. Press [SPACE] to pause. Press [ESC] to save and exit.')
                self.flag_record = True

        else:
            print('Recording resumed, video may be discontinuous. Press [SPACE] to pause. Press [ESC] to save and exit.')
            self.flag_record = True
            if not self.audio_stream.is_active():
                self.audio_stream.start_stream()  # Start the audio stream

        return False


    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis.create_window('recorder', 1920, 540)
        print("Recorder initialized. Press [SPACE] to start. Press [ESC] to save and exit.")

        vis_geometry_added = False
        while not self.flag_exit:
            rgbd = self.recorder.record_frame(self.flag_record, self.align_depth_to_color)

            if rgbd is None:
                continue

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

            if self.flag_record and self.audio_stream.is_active():
                data = self.audio_stream.read(self.chunk_size)
                self.audio_frames.append(data)

        self.recorder.close_record()
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()

        # Save original audio file to raw_data_audio directory
        wavefile = wave.open(f'D:/data/raw_data_audio/{self.filename}.wav', 'wb')
        wavefile.setnchannels(self.audio_channels)
        wavefile.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wavefile.setframerate(self.sample_rate)
        wavefile.writeframes(b''.join(self.audio_frames))
        wavefile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--participant', type=str, help='participant_id')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    if args.participant is not None:
        if not os.path.exists(f'D:/data/raw_data_visual'):
            os.makedirs(f'D:/data/raw_data_visual', exist_ok=True)
        if not os.path.exists(f'D:/data/raw_data_audio'):
            os.makedirs(f'D:/data/raw_data_audio', exist_ok=True)
        filename = args.participant
    else:
        assert args.participant, "Please input participant_id."
    print('Prepare writing to {}'.format(filename))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(config, device, filename, args.align_depth_to_color)
    try:
        r.run()
    except KeyboardInterrupt:
        print('Recording stopped by user')
    finally:
        r.audio_stream.stop_stream()
        r.audio_stream.close()
        r.audio.terminate()
        if r.recorder.is_record_created():
            r.recorder.close_record()
        print('Recording stopped')
