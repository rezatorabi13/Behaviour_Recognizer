"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This function plots model accuracy (accuracy versus epoch) as well as loss function
versus epoch for all k validation set and save them in the destination path.
"""
from parameters import Param
import matplotlib.pyplot as plt

def Plot_Acurracy(k,all_histories):
    for i in range(k):
        history=all_histories[i] 
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train','Test'], loc='upper left')
        plt.legend(['Train','Test'], loc='upper left')
        #Saving the graph
        plt.savefig(Param.Destination_path + 'model_accuracy_{}.jpg'.format(i+1), dpi=500)

def Plot_Loss(k,all_histories):  
    for i in range(k): 
        history=all_histories[i]
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.legend(['train', 'test'], loc='upper left')
        #Saving graph
        plt.savefig(Param.Destination_path+ 'loss_{}.jpg'.format(i+1), dpi=500) 