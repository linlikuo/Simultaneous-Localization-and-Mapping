import numpy as np

def load_Encoders(dataset):
    with np.load("Encoders%d.npz"%dataset) as data:###
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps
    return encoder_counts, encoder_stamps

def load_lidar(dataset):
    '''angle_min, angle_max, angle_increment, range_min, range_max, ranges, stamsp'''
    with np.load("Hokuyo%d.npz"%dataset) as data:
        angle_min = data["angle_min"] # start angle of the scan [rad]
        angle_max = data["angle_max"] # end angle of the scan [rad]
        angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        range_min = data["range_min"] # minimum range value [m]
        range_max = data["range_max"] # maximum range value [m]
        ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        stamsp = data["time_stamps"]  # acquisition times of the lidar scans
    return ranges, stamsp

def load_imu(dataset):
    '''angle_velocity, acceleration, stamps'''
    with np.load("Imu%d.npz"%dataset) as data:
        angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        stamps = data["time_stamps"]  # acquisition times of the imu measurements
    return angular_velocity, stamps

def load_rgbd(dataset):
    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    return disp_stamps, rgb_stamps
    
if __name__ == '__main__':
  dataset = 20
  
  with np.load("Encoders%d.npz"%dataset) as data:###
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load("Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load("Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

