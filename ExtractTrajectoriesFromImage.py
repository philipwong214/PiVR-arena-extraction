# -*- coding: utf-8 -*-
"""
Created on Mon Feb 8 2021

@author: Philip Wong

This script extracts arena geometry and odor source locations from behavioral data collected on PiVR. 

"""

#%% Import required packages

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import savemat 

#%% Set path directory

root_dir = os.getcwd()
data_dir = os.path.join(root_dir,'EtA1000_away') #set directory of sample data set

#%% Main function 

#The function extractArena iterates through each subfolder contained in the data directory, extracting the arena and odor source locations
#for each behavioral experiment. 

def extractArena(data_path):
    
    #Loop through each subfolder in data_path
    for dirs in os.listdir(data_path):     
        
        #Return subfolder directory
        test = dirs
        file_dir = os.path.join(data_path,test) 
        
        #Import and preprocess csv data recorded from behavioral experiment
        temp = pandas.read_csv(os.path.join(file_dir,'final_head_position.csv'))
        temp = temp[temp['X-Head'].notna()]     
        
        #Extract time stamps
        time = temp[['Time [s]']]
        time = time.to_numpy()
        
        #Extract head position of animal
        heads = temp[['X-Head','Y-Head']]
        heads = heads.to_numpy()
        heads = np.int32(heads)
        
        #Plot trajectory of animal on top of background image file
        grayimg = np.load(os.path.join(file_dir,'Background.npy'))
        grayimg = grayimg.astype(np.uint8)
        cv2.polylines(grayimg,[heads],False,(255, 0, 0),2)
        img = grayimg.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        #Calculate scaling constant for adjusting to different image resolutions
        standard_length = 640
        [x_dim, y_dim] = grayimg.shape
        img_length = max([x_dim, y_dim])
        scale = img_length/standard_length
        
        #Apply Hough circles algorithms with minimum radius minR and maximum raidus maxR to find odor sources
        minR = int(15*scale)
        maxR = int(20*scale)
        circles = cv2.HoughCircles(grayimg, cv2.HOUGH_GRADIENT, 0.8, 300,
                                    param1=1, param2=20,
                                    minRadius=minR, maxRadius=maxR)
        
        
        #Plot detected odor sources, provided they exist
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = circles[0:2,:]
            for (x, y, r) in circles:
              cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        
        #Calculate distance between odor sources 
        circles_dist = np.linalg.norm(circles[0,0:1] - circles[1,0:1])
        
        #Apply Hough circles algorithms with minimum radius dish_minRadius and maximum raidus dish_maxRadius to find arena
        dish_minRadius = int(0.625*circles_dist)
        dish_maxRadius = int(0.775*circles_dist)
    
        dish = cv2.HoughCircles(image=grayimg, 
                                method=cv2.HOUGH_GRADIENT, 
                                dp=1.2, 
                                minDist=100*scale, 
                                param1=50,
                                param2=50,
                                minRadius=dish_minRadius, 
                                maxRadius=dish_maxRadius)
        
        #Plot detected arena
        if dish is not None:
            dish = np.round(dish[0, :]).astype("int")
            for (x, y, r) in dish:
              cv2.circle(img, (x, y), r, (0, 255, 0), 2)       
        else:
            dish = [0,0,0]
    
        
        #Store arena and odor source locations in dictionary
        mdic = {"position":heads,"cup":circles,"arena":dish,"time":time}
        
        #Save dictionary in a .MAT file
        save_dir = os.path.join(data_path,test)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)         
        savemat(os.path.join(save_dir,'trajectories.mat'),mdic)
        
        #Show recovered arena, odor sources, and trajectory
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #Save test image for debugging
        filename = os.path.join(save_dir,'CupPositions.jpg')
        cv2.imwrite(filename,img) 

#%% Run Test

extractArena(data_dir)