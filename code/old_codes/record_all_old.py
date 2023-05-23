import argparse
import os
import cv2
import copy
import wave
import pyaudio
import open3d as o3d
import numpy as np
import subprocess

class RecorderWithCallback:

    def __init__(self, config, device, file_path, align_depth_to_color):
        # Global flags
        self.flag_exit = False
        self.flag_record = False
        self.file_path = file_path

        self.align_depth_to_color = align_depth_to_color
        self.recorder = o3d.io.AzureKinectSensor(config)
        if not self.recorder.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        
        return False

    def space_callback(self, vis):
        if self.flag_record:
            print('Recording paused.'
                  'Press [Space] to continue.'
                  'Press [ESC] to save and exit.')
            self.flag_record = False
        else:
            print('Recording started.'
                    'Press [SPACE] to pause.'
                    'Press [ESC] to save and exit.')
            self.flag_record = True

        return False

    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis.create_window('recorder', 1920, 540)
        vis_geometry_added = False

        print("Recorder initialized. Press [SPACE] to start. "
              "Press [ESC] to save and exit.")
        
        # Audio recording setting
        audio_format = pyaudio.paInt16
        sample_rate = 48000
        chunk_size = 4096
        audio_channels = 2

        audio = pyaudio.PyAudio()

        # Audio device selection
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        
        # device_index = int(input("Please select the input device id for Azure Kinect: "))
        device_index = 2

        audio_stream = audio.open(format = audio_format,
                                  channels = audio_channels,
                                  rate = sample_rate,
                                  input = True,
                                  frames_per_buffer = chunk_size)
        
        # Video recording setting
        fourcc = cv2.VideoWriter_fourcc(*'XVID')        
        video = cv2.VideoWriter(f'D:/data/raw_data/tmp_video.avi', fourcc, 30, (1280, 720)) #(1920, 1080) (1280, 720)
        color_video = []
        depth_video = []
        
        audio_frames = []
        audio_stream.start_stream()
        
        # Recording start
        i = 0
        print('Start Recording...')
        
        while not self.flag_exit:
            rgbd = self.recorder.capture_frame(self.align_depth_to_color)
            
            '''Add audio frame to audio_frames(List)'''
            aud = audio_stream.read(chunk_size)
            audio_frames.append(aud)
            
            if (rgbd is not None) and (self.flag_record):                
                # print(f'frame recorded: {self.flag_record}')

                '''write video frame to video file'''
                video.write(cv2.cvtColor(np.asarray(rgbd.color), cv2.COLOR_RGB2BGR))

                if (i%3)==2:
                    color_video.append(copy.deepcopy(np.asarray(rgbd.color)))
                    depth_video.append(copy.deepcopy(np.asarray(rgbd.depth)))
                
                
            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

            i += 1
        
        self.recorder.disconnect()
        video.release()

        audio_stream.stop_stream()
        audio_stream.close()
        audio.terminate()
        
        # Save original audio file to tmp_audio.wav
        wavefile = wave.open(f'D:/data/raw_data/tmp_audio.wav', 'wb')
        wavefile.setnchannels(audio_channels)
        wavefile.setsampwidth(audio.get_sample_size(audio_format))
        wavefile.setframerate(sample_rate)
        wavefile.writeframes(b''.join(audio_frames))
        wavefile.close()
        del audio_frames

        # Make final video using ffmpeg
        try:
            cmd = f"ffmpeg -y -analyzeduration 2147483647 -probesize 2147483647 -channel_layout stereo -ac 2 -i D:/data/raw_data/tmp_audio.wav -i D:/data/raw_data/tmp_video.avi -s 1280x720 -r 30 -pix_fmt yuv420p D:/data/raw_data/{self.file_path}_raw_video.avi"
            p = subprocess.run(cmd, shell=True, check=True)
            print("FFmpeg has finished.")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg command failed with error: {str(e)}")

        print(len(color_video))

        #print(np.array(color_video).shape)
        color_video = np.array(color_video)
        np.save(f'D:/data/proc_data/color/{self.file_path}_color.npy', color_video)
        del color_video
        print('Color saving complete.')

        depth_video = np.array(depth_video)
        np.save(f'D:/data/proc_data/depth/{self.file_path}_depth.npy', depth_video)
        print('Depth saving complete.')
 
                  
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
        
        if not os.path.exists(f'D:/data/proc_data/color'):
            os.makedirs(f'D:/data/proc_data/color/', exist_ok=True)
            os.makedirs(f'D:/data/proc_data/depth/', exist_ok= True)
            os.makedirs(f'D:/data/raw_data/', exist_ok= True)
        #assert f'{args.participant}_color.npy' not in os.listdir('D:/data/proc_data/color'), 'participant number is duplicated.'
        file_path = args.participant
    else:
        assert args.participant, "Please input participant_id."
    print('Prepare writing to {}'.format(file_path))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(config, device, file_path,
                             args.align_depth_to_color)
    r.run()