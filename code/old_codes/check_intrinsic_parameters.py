import k4a

device = k4a.Device.open()

calibration = device.get_calibration()

color_camera_calibration = calibration.color_camera_calibration 
intrinsics = color_camera_calibration.intrinsic.parameters

print("fx = {:.2f}, fy = {:.2f}, cx = {:.2f}, cy = {:.2f}".format(
    intrinsics.param.fx, intrinsics.param.fy, intrinsics.param.cx, intrinsics.param.cy))

# this code doesn't seem to run in python. run using C# code
# fx: 961.8638, fy: 550.6279, cx: 912.718, cy: 912.5225