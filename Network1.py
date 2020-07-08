"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This script converts a video file to a 2D matrix of features. It uses modules in
Part1 folder to convert a video clip to a set of frames and extract features for 
each frame in a convolutional network architecture. By default it extract 150 frame
from each video file and uses InceptionV3 to extract 2048 features for each frame. 
The output is a mtrix with shape 150*2048 for each video file that is saved as a
.csv file in the "Features" folder. If you would like to extract other number of frames 
than 150 from a video file you can modify it in "parameters" script. If you would like to
use another Convolutionalnetwork architecture than InceptionV3 you can modify it
in "Feature_extractor" madules in "Part1" folder.

Remember that, you need to define the path of your excel data in "parameters" script. Your
excel file contains the information of video names as well as the corresponding class.
"""
from parameters import Param
from Part1.Convertor import extract_frames
from Part1.Feature_extractor import extract_features
import pandas as pd
import os
from glob import glob
from tqdm import tqdm


#loading data 
data=pd.read_excel(Param.Excel_Path)
# removing all old csv files from the Features folder (cleaning the folder)
Num_frames = []
seq = glob(Param.Features_Path+'/*.csv') 
for s in seq:
    os.remove(s)
for i in tqdm(range(data.shape[0])): 
    extract_frames(i)
    videoFile = data['Video'][i]
    extract_features(videoFile)