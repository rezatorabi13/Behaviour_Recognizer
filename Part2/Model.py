"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi


"""
from parameters import Param
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras import optimizers
from keras import backend as K


def RNN_model():
    
    K.clear_session()
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=False, input_shape=(Param.Number_of_Frames,2048)))
    model.add(Dropout(0.2))
    model.add(Dense(2)) # Remember that we have two classes in here
    model.add(Activation('softmax'))  # Should be separated for further usage in knowledge extraction

    optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False) 
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
    
    return model