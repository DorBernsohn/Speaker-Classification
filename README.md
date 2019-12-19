# Speaker_classification

In this notebook im going to comprae two architectures for speaker identification with MFCC as features on two datasets.

the architectures:
- LSTM model
- CNN model

the datasets:
- Digit spoken dataset (https://www.kaggle.com/divyanshu99/spoken-digit-dataset)
- Voxceleb dataset (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

The main pupose of the research is to see the difference between "easy" data to a "messy" one. 
The spoken numbers dataset has 15 speakers while all the content is numbers that peaple say (classical for voice recognition and STT).
The Voxceleb dataset has ~250 speakers while all the content is free speech of celebs from youtube videos. 

To test this complexity question, i thought that the best approach is to take the basic features. I calculated the 20 mfcc features for every sample and feed it to a NN.

Notebook Table of Content:


![Alt text](tableofcontent.png?raw=true)
