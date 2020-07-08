"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This script prepare the data in an approperiate form for feeding into the RNN network,
and then train the network for behaiviour recognition. The code perform kfold cross validation
and will save k weight, one for each validation set, at the destination path. It also plot the
model accuracy and loss function versus epoch for each validation set. In addition, it saves
the indexes for test data for each validation to investigate model performance in more details
using prediction script.
"""
#import neccessary libraries
from parameters import Param
from Part2.Model import RNN_model
from Part2.data_preparation import prepare_data
from Part2.plot import Plot_Acurracy, Plot_Loss
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#epoch for training the model
epoch = 300

Destination_path = Param.Destination_path

#loding information about the data
data = pd.read_excel(Param.Excel_Path)  

#Prepare the data 
X,y = prepare_data(data)

groups = list(range(y.shape[0]))

k=Param.kfold
fold = 1
results = []
all_histories = []
Indexes = []
group_kfold = GroupKFold(n_splits=k)
for train_index, test_index in group_kfold.split(X, y, groups):

    Indexes.append(test_index)
     
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    #defining the mode
    model = RNN_model()
    
    file_name = Param.Destination_path + 'weights_{}.hdf5'.format(fold)
    mcp_save = ModelCheckpoint(file_name, save_best_only=True, monitor='val_loss', mode='min')   
    history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=100)
    all_histories.append(history)
        
    eval_result = model.evaluate(X_test, y_test)
    print(eval_result)
    results.append(eval_result)   
    fold += 1

#Saving the indexes of validation data
np.save(Param.Destination_path + 'Validation_Indexes.npy',Indexes)

#Plotting the model accuracy for all k validation sets and saving them
Plot_Acurracy(k,all_histories)

#Plotting the loss function versus epoch for all k validation sets and saving them    
Plot_Loss(k,all_histories)