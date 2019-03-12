# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:08:58 2019

@author: linli
"""
import numpy as np
import imageio
import os
from map_utils import bresenham2D

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def stratified_resample(W):
    N = len(W)
    random_number = (np.random.rand(N) + range(N)) / N
    
    cum_W = np.cumsum(W)
    ret = []
    i, j = 0, 0
    while i < N:
        if random_number[i] < cum_W[j]:
            ret.append(j)
            i += 1
        else:
            j += 1
    return np.array(ret)

def t_align(target_time,cor_time):
    index = np.argmin(np.abs(cor_time-target_time))
    return index

def mapping(grid, scan, now, res, map_angles):
    free_odds = np.log(0.9/0.1)/4
    occup_odds = np.log(0.9/0.1)
    saturated = 127
    #valid = valid_scan(scan, map_angles)
    distance = scan
    theta = map_angles + now[2]
    x,  y = distance * np.cos(theta), distance * np.sin(theta)
    
    xi = (x/res).astype(int)
    yi = (y/res).astype(int)
    
    free_set = {}
    wall_set = {}
    
    for (a,b) in zip(xi,yi):
        line = np.array(bresenham2D(0, 0, a, b)).astype(int)
        xx = a + int(now[0]) + grid.shape[0]//2
        yy = b + int(now[1]) + grid.shape[1]//2
        wall_set[xx, yy] = True
        for j in range(len(line[0])-1):
            free_set[(line[0][j], line[1][j])] = True
       
    for k, _ in wall_set.items():
        xx, yy = k[0], k[1]
        if 0 <= xx < grid.shape[0] and 0 <= yy < grid.shape[1]:
            grid[xx,yy] += occup_odds
            #print('xx,yy', xx,yy)
    for k, _ in free_set.items():
        xx = k[0] + int(now[0]) + grid.shape[0]//2
        yy = k[1] + int(now[1]) + grid.shape[1]//2
        if 0 <= xx < grid.shape[0] and 0 <= yy < grid.shape[1]:
            grid[xx,yy] -= free_odds
            
    grid[grid > saturated] = saturated
    grid[grid < -saturated] = -saturated
    
    return grid


def texture_mapping(grid, now, res, img_name, depth_name):

    new_now = now.copy()
    new_now[0] = new_now[0] *res
    new_now[1] = new_now[1] *res
    roll = 0
    pitch = 0.36
    yaw = 0.021
    
    
    d = imageio.imread(os.path.join('..\\dataRGBD\\Disparity21\\',depth_name))
    rgb_img = imageio.imread(os.path.join('..\\dataRGBD\\RGB21\\',img_name))
    
    #valid = np.where(d < threshold)
    dd = -0.00304 * d+3.31
    depth = 1.03/dd
    valid = np.where(np.logical_and(depth<5, depth>0.05))
    #print(dd.shape)
    depth_i = valid[0]
    depth_j = valid[1]
    #print(depth.shape)
    
    # Intrinsic matrix
    intri_matrix = np.zeros((3,3))
    intri_matrix[0,0], intri_matrix[1,1] = 585.05108211, 585.05108211
    intri_matrix[0,2] = 242.94140713
    intri_matrix[1,2] = 315.83800193
    intri_matrix[2,2] = 1
    
    # Roc
    Roc = np.zeros((3,3))
    Roc[0,1] = -1
    Roc[1,2] = -1
    Roc[2,0] = 1
    
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                   [np.sin(yaw), np.cos(yaw),0],
                   [0          ,           0,1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0            , 1,             0],
                   [-np.sin(pitch),0, np.cos(pitch)]])
    Rx = np.array([[1,             0,             0],
                   [0,  np.cos(roll), -np.sin(roll)],
                   [0,  np.sin(roll),  np.cos(roll)]])
    
    # Rcb{rotation camera to body}
    R = (Rz.dot(Ry)).dot(Rx)
    
    # Pcb{position camera to body}
    Pcb = np.array([0.18,0.005,0.36]).T
    
    # Tcb{Transform from camera to body}
    Tcb = np.zeros((4,4))
    Tcb[0:3,0:3] = R
    Tcb[0:3,3] = Pcb
    Tcb[3,3] = 1
    
    # Rbw{rotation body to world}
    theta = new_now[2]
    Rbw = np.zeros((3,3))
    Rbw[0,0] = np.cos(theta)
    Rbw[0,1] = -np.sin(theta)
    Rbw[1,0] = np.sin(theta)
    Rbw[1,1] = np.cos(theta)
    Rbw[2,2] = 1
    
    # Tbw{Transform from body to world}
    Tbw = np.zeros((4,4))
    Tbw[0:3,0:3] = Rbw
    Tbw[0:3,3] = new_now
    Tbw[3,3] = 1
    
    Tcw = Tcb.dot(Tbw)

    #Twc = np.linalg.inv(Tcw)

    
    Rwc = Tcw[0:3,0:3]
    Pwc = Tcw[0:3,3]
    
    # extrinsic matrix 4 x 4
    extri_matrix = np.zeros((4,4))
    extri_matrix[0:3,0:3] = Roc.dot(Rwc.T)
    extri_matrix[0:3,3] = -Roc.dot(Rwc.T).dot(Pwc)
    extri_matrix[3,3] = 1

    for idx in range(depth_i.size):

        # Pixels
        rgb_i = (depth_i[idx]*526.37+dd[depth_i[idx],depth_j[idx]]*(-4.5*1750.36)+19276)/585.051
        rgb_j = (depth_j[idx]*526.37+16662)/585.051

        pixel = np.array([rgb_i, rgb_j, 1]).T
        Z0 = depth[depth_i[idx],depth_j[idx]]

        # Canonical projection
        cano = np.zeros((3,3))
        cano[0,0] = 1/Z0
        cano[1,1] = 1/Z0
        cano[2,2] = 1/Z0
    
        # Optical coordinate
        Optic_coor = np.linalg.inv(cano).dot(np.linalg.inv(intri_matrix)).dot(pixel)
        Optic_coor = np.concatenate((Optic_coor,np.array([1])))
        
        # World Coordinate
        World_coor = np.linalg.inv(extri_matrix).dot(Optic_coor)

        grid[int(World_coor[0]/res)+grid.shape[0]//2, int(World_coor[1]/res)+grid.shape[1]//2,:] = rgb_img[int(rgb_i), int(rgb_j),:]
    
    return grid
    
    