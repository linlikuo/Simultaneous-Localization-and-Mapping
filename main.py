# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:13:01 2019

@author: linli
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:34:31 2019

@author: linli
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import imageio

from load_data import load_Encoders, load_lidar, load_imu, load_rgbd
from map_utils import mapCorrelation
from utils import softmax, stratified_resample, t_align, mapping, texture_mapping
#from t_mapping import texture_mapping

dataset = 21
L = (476.25+311.15)/2

#def main():
### load data

encoder_counts, encoder_stamps = load_Encoders(dataset)
lidar_ranges, lidar_stamps = load_lidar(dataset)
angular_velocity, imu_stamps = load_imu(dataset)
disp_stamps, rgb_stamps = load_rgbd(dataset)

### calculate location

num_data = np.shape(encoder_counts)[1]
# 2 x n data {right wheel, left wheel}
distance_data = np.zeros((3, num_data), np.float64)
distance_data[0,:] = np.copy(encoder_stamps)
distance_data[1,:] = (encoder_counts[0,:]+encoder_counts[2,:])/2*0.0022
distance_data[2,:] = (encoder_counts[1,:]+encoder_counts[3,:])/2*0.0022

# 2 x n wheel velocity {right, left}
wheel_velocity = np.zeros((2, num_data))
for idx in range(1, num_data):
    wheel_velocity[0,idx] = distance_data[1,idx]/(distance_data[0,idx]-distance_data[0,idx-1])
    wheel_velocity[1,idx] = distance_data[2,idx]/(distance_data[0,idx]-distance_data[0,idx-1])

# angular velocity
omega = np.zeros((1, num_data))
for idx in range(num_data):
    closed_index = t_align(distance_data[0,idx], imu_stamps)
    omega[0,idx] = angular_velocity[2,closed_index]
    
# angle(theta)
angle = np.zeros((1,num_data))
for idx in range(1, num_data):
    t = distance_data[0,idx]-distance_data[0,idx-1]
    angle[0,idx] = angle[0,idx-1] + omega[0,idx]*t

 
# velocity
velocity = np.array((wheel_velocity[0,:]+wheel_velocity[1,:])/2).reshape(1,num_data)

# 2 x n robot location {x,y}    
location = np.zeros((2,num_data), np.float64)
for idx in range(1, num_data):
    t = distance_data[0,idx]-distance_data[0,idx-1]
    sinc = math.sin(omega[0,idx]*t/2)/(omega[0,idx]*t/2)
    location[0,idx] = location[0,idx-1] + velocity[0,idx]*t*sinc*math.cos(angle[0,idx]+omega[0,idx]*t/2)
    location[1,idx] = location[1,idx-1] + velocity[0,idx]*t*sinc*math.sin(angle[0,idx]+omega[0,idx]*t/2)


plt.figure()   
plt.scatter(location[0,:],location[1,:])
pose = np.vstack((location, angle)).T




# mapping
cor_depth = np.zeros(lidar_stamps.size, np.int)
cor_rgb = np.zeros(lidar_stamps.size,np.int)
new_pose = np.zeros((lidar_stamps.size, 3))
for idx in range(lidar_stamps.size):
    closed_index = t_align(lidar_stamps[idx],encoder_stamps)
    new_pose[idx,:] = pose[closed_index,:]
    
    # corrspond lidar time to rgb and depth time and have the correspoonding pic name
    closed_depth = t_align(lidar_stamps[idx], disp_stamps)
    cor_depth[idx] = closed_depth+1
    closed_rgb = t_align(lidar_stamps[idx], rgb_stamps)
    cor_rgb[idx] = closed_rgb+1


robot_pos = np.zeros((lidar_ranges.shape[1],3))
for idx, t_lidar in enumerate(lidar_stamps):
    idx_closed = t_align(t_lidar, encoder_stamps)
    robot_pos[idx,:] = np.hstack((location[:,idx_closed].T, angle[0,idx_closed]))


    
res = 0.1
grid_size = int(80/res)
lidar_angles = np.arange(-135,135.25,0.25)*np.pi/180
grid = np.zeros((grid_size,grid_size))

now = np.array([0, 0, 0])
texture_map = np.zeros((grid_size, grid_size, 3))

walk_map = np.zeros((grid_size, grid_size))
slam_map = np.zeros((grid_size, grid_size))

# particle amount
N = 100
# particle
X = np.zeros((N,3))
# particle weight
W = np.ones(N)/N
# particle noise (you can fine tune)
noise = np.array([0.02, 0.02, 0.1*np.pi/180])

step = 10
slam_location = np.zeros((int(lidar_ranges.shape[1]/step)+1, 2))
walk_location = np.zeros((int(lidar_ranges.shape[1]/step)+1, 2))
pre_position = new_pose[0,:]
images = []
plt.figure()
img = np.zeros((grid_size, grid_size, 3))
for t, lidar_index in enumerate(range(0, lidar_ranges.shape[1], step)):
    print(t)
    #if t >= 50:
    #    break
    lidar = lidar_ranges[:,lidar_index]
    noises = np.random.randn(N,3)*noise
    displacement = new_pose[lidar_index,:] - pre_position
    X += (displacement + noises)
    X[:,2] %= 2*np.pi
    
    cors = []
    
    x_im, y_im = np.arange(grid.shape[0]), np.arange(grid.shape[1])
    l = 1
    xs, ys = np.arange(-res*l, res*l+res, res), np.arange(-res*l, res*l+res, res)
    temp = np.zeros_like(grid)
    temp[grid>0] = 1
    temp[grid<0] = -1
    
    for i in range(len(X)):
        world_angles = lidar_angles + X[i][2]
        x, y = lidar * np.cos(world_angles), lidar * np.sin(world_angles)
        x, y = x/res + grid.shape[0]//2, y/res + grid.shape[1]//2
        cor = mapCorrelation(temp, x_im, y_im, np.vstack((x, y)), (X[i][0]+xs)/res, (X[i][1]+ys)/res)
        cors.append(np.max(cor))
        
    cors = W * np.array(cors)
    W = softmax(cors)
    
    best = np.where(W == np.max(W))[0][0]
    now = X[best].copy()
    now[0] /= res
    now[1] /= res
    img_name = 'rgb%d_%d.png' %(dataset, cor_rgb[lidar_index])
    depth_name = 'disparity%d_%d.png' %(dataset, cor_depth[lidar_index])
    
    grid = mapping(grid, lidar, now, res, lidar_angles)
    now_texture = now.copy()
    
    texture_map = texture_mapping(texture_map, now_texture, res, img_name, depth_name)
    n_eff = 1 / (W**2).sum()
    if n_eff < 0.85 * N:
        idx = stratified_resample(W)
        X[:] = X[idx]
        W.fill(1/N)

    
    slam_map[int(now[0]) + slam_map.shape[0]//2, int(now[1]) + slam_map.shape[1]//2] = 1
    walk_map[int(robot_pos[lidar_index,0] / res) + walk_map.shape[0]//2, int(robot_pos[lidar_index,1]/ res) + walk_map.shape[1]//2] = 1
    
    slam_location[t,:] = np.array([int(now[0]) + slam_map.shape[0]//2, int(now[1]) + slam_map.shape[1]//2]).T
    walk_location[t,:] = np.array([int(robot_pos[lidar_index,0] / res) + walk_map.shape[0]//2, int(robot_pos[lidar_index,1]/ res) + walk_map.shape[1]//2]).T
    pre_position = new_pose[lidar_index,:]
    
    if t % 30 == 0:
        print('save')
        grid_pic = np.zeros((grid_size,grid_size))
        grid_pic[grid>0] = 2
        grid_pic[np.logical_and(0<=grid, grid <=0)] = 0
        grid_pic[grid<0] = 0.5        
        plt.imshow(grid_pic.astype(np.uint8), cmap = 'gray')
        plt.plot(slam_location[0:t,1], slam_location[0:t,0], 'r')
        plt.plot(walk_location[0:t,1], walk_location[0:t,0], 'b')
        plt.savefig('mapping_21_%d' %(t))
    
    images.append(grid.astype(np.uint8))
# Plot
    
imageio.mimsave('21.gif', images)
grid[grid>0] = 2
grid[np.logical_and(0<=grid, grid <=0)] = 0
grid[grid<0] = 0.5
#img = np.zeros((grid_size, grid_size, 3))
img[:,:,0] = grid * 70
img[:,:,1] = walk_map * 10
img[:,:,2] = slam_map * 127
plt.figure()
plt.plot(slam_location[:,1], slam_location[:,0])
plt.plot(walk_location[:,1], walk_location[:,0])
plt.imshow(img.astype(np.uint8))
plt.savefig('mapping_21')
plt.figure()
plt.imshow(texture_map.astype(np.uint8))
plt.savefig('tmapping_21')
plt.show()
    
if __name__ == '__main__':
    #main()
    pass