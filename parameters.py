"""
Behaiviour_Recognizer Toolbox 
© A. Luczak Lab
@author: Reza Torabi

This script contains information that is needed to be modified for your own data.
Remember that, you need to prepare an excel file containing the video names and 
their coresponding classes (see example folder). The Excel_Path is the path that 
the excel file is located. You also need to create 3 folders. A folder containning 
your video files, a folder that the extracted frames for a video is saved there, and
a folder that the extracted features is saved there. The path for these folders are
Videos_Path, Frames_Path, and Features_Path, respectively.
"""
        
class Param:
    #Paths
    Excel_Path = 'F:\\OpenField10.xlsx' # Modify this path for your own data (path for your own excel file)
    Videos_Path = 'F:\\Videos' # Modify this path for your own data (path for your own Videos folder)
    Frames_Path = 'F:\\Frames' # Modify this path for your own data (path for your own Frames folder)
    Features_Path = 'F:\\Features' # Modify this path for your own data (path for your own Features folder) 
    #Parameters
    Video_Format = '.mpg' #  modify this path for your own data (format of the video files)
    Number_of_Frames = 150 #Number of frames that is extracted from a video file. This is the sequence lenght that is fed to the part2 of the network (RNN network).By default it extracts one every 10 frames from video files. You can modify it in Convertor script in Part1 folder.
    kfold = 5 #Number of kfold cross validation.
    Destination_path = 'E:\\' # The path for saving model weights as well as model accuracy plots