
## Behaviour_Recognizer
_____________________________________________________________________________________________________
### A deep learning toolbox for data driven analysis of behavioural video data.

Behaivior_Recognizer is a deep learning based toolbox for analyzing the videos of behaviour and behaviour experiments. It provides a framework for state-of-the-art behaviour recognition and can be used by researchers and practitioners for automated behavior analyses. 

### Why use Behaviour_Recognizer?
Behavior is a sensitive indicator of brain function, in addition to being relatively non-invasive, is central to documenting normal and altered brain activity. In this regard, the Toolbox represent a state-of the-art method for behaviour classification. In addition, the method automatically identifies most relevant behavior for predictions. In short, it offers a one-step solution for feature selection and group classification. After behaviour recognition, by analyzing the network’s decision-making process using knowledge extraction methods, new insights into behavioral differences can be obtained.

### Demo:
As an example, we implemented the toolbox for data-driven analyses of infant rat behavior in an open field task to investigating brain alterations in development. The toolbox was applied to study the effect of maternal nicotine consumption prior to conception on rat pup’s motor development. It distinguished between the behaviour of control vs. nicotine group animals with a state-of-the-art accuracy. For mor information, please refer to our paper.

<p>
    <img src="Figure1.gif" alt="Figure1" width="280"/>
    <br>
    <em>Figure1: Open field task. The network was trained to distinguish if the rat pup is a nicotine or control animal. In nicotine animals the rat's mum consumed nicotine prior to conception.</em>
</p>


<p>
    <img src="Figure2.jpg" alt="Figure2" width="700"/>
    <br>
    <em>Fig. 2: Right) Model accuracy (accuracy versus epoch for train and test data). Left) Network attention after training the model.</em>
</p>

# Deep Neural Network training and architecture

The network architecture has two parts. In part1, we use a convolutional network (ConvNet) to convert each video frame to a set of features starting by pre-trained weights of models like Inception-V3. Features from video frames from a single video clip were than combined and passed to a recurrent neural network (RNN) in the second part. This allowed an analysis of behaviour and movements. The network is trained to assign a correct group category to each video clip. The code use kfold validation to train and validate the model.

<figure>
  <img src="Figure3.jpg" alt="Figure3" width="700"/>
  <figcaption>Figure3: Network architecture and training.</figcaption>
</figure>

### Requirements
This code requires you have the following libraries installed. 
- TensorFlow
- Keras
- OpenCV
- pandas
- numpy
- matplotlib
- tqdm
- glob
- os

Please see the `requirements.txt` for more details. 

## Instructions for using the toolbox for your own data

1. Download the repository to your local drive and unzip it.

2. Provide an excel file containing the name of you behavioural data as well as their corresponding classes. See example folder to find a sample excel file.

3. Create a folder and place your behavioural video data in that folder. Create 2 other folders, one for video frames and the other for extracted features. The code needs the path for these three folders as well as your own excel file. You can modify the paths in <code>parameters.py</code> file to personalize the toolbox for your own data.
 
4. Run <code>Network1.py</code> to **extract features** from the video frames using the CNN (*Part1 of the network*). Once you run this python script, the features will be saved in your feature folder. You can find related modules for part1 of the network in Part1 folder. 

5. Run <code>Network2.py</code> to **train the model** using RNN (*Part2 of the network*). You can find related modules for part2 of the network in Part2 folder. To modify the models, you need to edit <code>Model.py</code> in Part2 folder. To modify the number of validation sets in kfold validation, you just need to change kfold parameter in <code>parameters.py</code> file. The weights for trained model (for all validation sets) will be saved in your destination path (see <code>parameters.py</code> file). 

6. Finally, to make prediction based on the trained model and validate the model run <code>prediction.py</code>.

 For more information, refer to the explanation in each script and modules.

### Author
Reza Torabi

### License

This project is licensed under the MIT License - see the LICENSE file for details

### Citation

If you use the code, please cite us!
