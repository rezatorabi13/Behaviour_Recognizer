"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This script is for making prediction for any desire validation set. In k-fold validation,
data is devided into k portion. For each validation set the network has used k-1 portion 
for training the network and have saved the corresponding weights. Here we use the one 
remaned portion (validation data for kth_Validation) to predict behaviour class for them.

It read test data (extracted feature for test data in convolutional network) from "Feature" folder,
provides prediction for them, evaluates the validation accuracy (accuracy for unseen data)
and reports it. It also provides the activations before and after the last layer
(before and after the activation function of the last layer) for investigating the network 
performance in more details. After that, It organizes all the information and saves them in 
an excel file in the destination path.
"""
from parameters import Param
from Part2.Model import RNN_model
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import genfromtxt
from keras import backend as K
from sklearn.metrics import accuracy_score


#Define which validation set you would like to predict
kth_Validation=3

# The name of the weight for the kth_Validation
weight_path = Param.Destination_path + 'weights_' + str(kth_Validation)+'.hdf5'

#loading the model
model = RNN_model()

# loading the trained weights
model.load_weights(weight_path)

data=pd.read_excel(Param.Excel_Path)

Indexes = np.load(Param.Destination_path + 'Validation_Indexes.npy')

#preparing test data 
Video_N = []
Video_names = data['Video']
for i in Indexes[kth_Validation-1]:                  
    video_name = Video_names[i]
    Video_N.append(video_name)


X = []
y = []
for i in Indexes[kth_Validation-1]:                   
    File_name = Video_names[i] + '.csv'
    Video = Video_names[i]
    X0 = genfromtxt(Param.Features_Path + '\\' +File_name, delimiter=',')
    X.append(X0)
    Class = list(data[data['Video']==Video]['Class'])[0]
    y.append(Class)

X = np.array(X)

#Prediction
predict = []
actual = []
After_soft = []
Before_soft = []

for i in tqdm(range(X.shape[0])):

# Reading the input (feature) and converting it in the desire input_shape for lstm  
    prediction_movie = X[i]         
    prediction_movie = prediction_movie.reshape(1,prediction_movie.shape[0],prediction_movie.shape[1])
    
# predicting tags for each array
    prediction = model.predict_classes(prediction_movie)
    get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
    After_softmax = get_3rd_layer_output([prediction_movie])[0]
    get_2rd_layer_output = K.function([model.layers[0].input],[model.layers[2].output])
    Before_softmax = get_2rd_layer_output([prediction_movie])[0]
    
# appending the model prediction in predict list to assign the tag to the video
    predict.append(prediction[0])
    After_soft.append(After_softmax)
    Before_soft.append(Before_softmax)
    

# After softmax
Aft1 = []
for i in range(len(After_soft)):
    Aft1.append(After_soft[i][0][0])
    
Aft2 = []
for i in range(len(After_soft)):
    Aft2.append(After_soft[i][0][1])   


# Before softmax
Bef1 = []
for i in range(len(Before_soft)):
    Bef1.append(Before_soft[i][0][0])
    
Bef2 = []
for i in range(len(Before_soft)):
    Bef2.append(Before_soft[i][0][1])   


Pre_class = [x+1 for x in predict]

#Organizing the validation information to save in excel file
Validation= pd.DataFrame(Indexes[kth_Validation-1], columns = ['Index'])
Validation['Video'] = Video_N
Validation['Class'] = y
Validation['Predicted class'] = Pre_class
Validation['Network Output'] = predict
Validation['N1_After'] = Aft1
Validation['N2_After'] = Aft2
Validation['N1_Before'] = Bef1
Validation['N2_Before'] = Bef2

#Saving datafram to excel
Validation.to_excel(Param.Destination_path + 'Validation_{}.xlsx'.format(kth_Validation), index = False)

#Calculating accuracy and reporting it
A = Validation['Class']
A.tolist()

P = Validation['Predicted class']
P.tolist()

Accuracy = accuracy_score(P, A)*100
print('The accuracy for this validation set is: ',round(Accuracy), '%')

