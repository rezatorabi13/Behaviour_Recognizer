"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This function extracts frame for each video file. It reads the video files from
Video folder, convert it to a desire number of frames, and saves them in Frames
folder. 
You can change the number of frames extracted from videos. This number is the 
sequence lenght which is used in part2 of network (RNN network).
By default it extracts one every 10 frames from video files. You can modify it in here.
"""

from parameters import Param
import cv2     # for capturing videos
import pandas as pd
from glob import glob
import os

th_frame = 0 #Threshold frame for trimming the video files.
Every_frame = 10 #It extract one frame per each frameRate number.
excel_path = Param.Excel_Path
Frames_path = Param.Frames_Path 
Videos_path = Param.Videos_Path
Video_format = Param.Video_Format
n_frames = Param.Number_of_Frames #Number of frames that is extracted from a video file

#Loading data 
data=pd.read_excel(excel_path) 

def extract_frames(i):    
    count = 0
    fnum = 0
    videoFile = data['Video'][i]
    cap = cv2.VideoCapture(Videos_path + '/'+videoFile + Video_format)   # capturing the video from the given path
    #Every_frame = 10 #It extract one frame per each frameRate number.
    # removing any other pre-existing files from the Frame folder
    files = glob(Frames_path +'/*')  
    for f in files:
        os.remove(f)
    while(cap.isOpened()):
        fnum +=1
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if fnum>th_frame:     #This is a thereshold for Triming the data. We set is zero by default 
            if (frameId % Every_frame == 0):
                if count<n_frames: 
                    # storing the frames to the destination folder 
                    filename = Frames_path + '//'+ videoFile +"_frame%3d.jpg" % count;count+=1
                    cv2.imwrite(filename, frame)
    cap.release()
    