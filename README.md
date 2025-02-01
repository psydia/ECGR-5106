The repository contains code for problem 1- multi-layer perceptron with three hidden layers for the CIFAR-10 dataset and problem 2- multi-perceptron network that regresses the housing price.
Directory Structuce for problem 1-
   'data_loader.py'
   'model.py'
   'train.py'
The model checkpoint link - https://drive.google.com/drive/folders/1T1vX3GRFm-NtZxSrAC4s-G9dWzewkgxP?usp=drive_link
These three contains the code for solving 1a. where the model can be trained and the results (training loss and accuracy, validation accuracy) can be plotted after 20 epochs. 
    'inference.py' contains code to report precision, recall, F1 score, and confusion matrix.
    'ablation.py' contains the solution for 1b, where the complexity of the network has been increased by its width and depth.
The model checkpoint link - https://drive.google.com/drive/folders/1xjXis6qv48Qd6aBMCNG0ZebJnGPTrQ5g?usp=drive_link
    

Directory Structure for problem 2-
'main2a.py' contains code for multi-perceptron network that regresses the housing price without one hot encoding.
'main2b.py' contains code for the same problem but adding one hot encoding.
'ablation.py' includes code for the same problem where we increased the complexity of the problem and observed the results.
