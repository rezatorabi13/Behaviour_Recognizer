"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This script generates extracted features for each video which the part2 of network
(RNN network) use it as input.
You can change your pre-trained convolutional architecture in here. You also can 
change the size of your image you feed to base model in here.
"""
from parameters import Param
from tensorflow.keras.preprocessing import image   # for image pre-preprocessing
import numpy as np    
from glob import glob
from keras.applications.inception_v3 import InceptionV3, preprocess_input


Frames_path = Param.Frames_Path 
Videos_path = Param.Videos_Path
Feature_path = Param.Features_Path

#Create the base model Inception V3
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

#Extract features for videos
def extract_features(videoFile):
    movie_images = []
    video_frames = glob(Frames_path +'\*.jpg') 
    # for loop to read and store frames
    for fr in range(len(video_frames)):   
        # loading the image and keeping the arbitrary target size as (W,H,C) for Inception model
        img = image.load_img(video_frames[fr], target_size=(400,350,3)) # You can change the size of the image in here
        # converting it to array
        img = image.img_to_array(img)
        #Preprocess input
        img = preprocess_input(img)
        # appending the image to the list
        movie_images.append(img)

    # converting the list to numpy array
    movie_train = np.array(movie_images)
    # extracting features for video frames
    Features = base_model.predict(movie_train)
    #Save the extracted features for each video file in a destination file
    np.savetxt(Feature_path + '\\'+videoFile+'.csv', Features, delimiter=',')
    return Features   





 