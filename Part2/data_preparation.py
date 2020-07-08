"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This function prepare the data in an approperiate form for feedinginto the RNN 
network. It reads features, that is extracted in part1, from Featuresfolder and
convert them to approperiat numpy arrays. It also provides one hot codes for video 
lables.
"""
#import neccessary libraries
from parameters import Param
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm


#Creating X and y
def prepare_data(data):
    X = []
    y = []
    Video_names = data['Video']
    for i in tqdm(range(len(Video_names))):
        File_name = Video_names[i] + '.csv'
        Video = Video_names[i]
        X0 = genfromtxt(Param.Features_Path + '\\'+File_name, delimiter=',')
        X.append(X0)
        Class = list(data[data['Video']==Video]['Class'])[0]
        y.append(Class)

    X = np.array(X)
    y = np.array(y)
    return X,y
