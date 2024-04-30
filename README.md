Time series predictions are an important category of applications, 
where the task is to predict the onset of events, and to predict events before they occur, 
to ensure timely intervention. Sepsis is an important category of cariovascular diseases and shocks, 
with unspecific symptoms and markers, making it hard to predict and manage clinically. 


<br>
We implemented an LSTM and used <a href=https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/#data>SepsisExp</a> dataset to predict the onset of Sepsis 2 hours, 
4 hours and 6 hours in advance. The data is a set of laboratory measurements, per patient, per time interval. 
Please read the linked papers for more information. 

<br>

We tested the accuracy of the model for each time interval, as well as the minimal set of features needed to predict. 
We defined our own architecture, and we used optimizers, dropout and early stopping and cross fold validation. 
<br>
<br> <br> <br> <br>
Citation:
<br><br>
Shigehiko Schamoni, Michael Hagmann and Stefan Riezler
Ensembling Neural Networks for Improved Prediction and Privacy in Early Diagnosis of Sepsis
Proceedings of Machine Learning Research, 182, PMLR, Durham, NC, USA, 2022
<br>
<br>
H. A. Lindner, S. Schamoni, T. Kirschning, C. Worm, B. Hahn, F. S. Centner, J. J. Schoettler, M. Hagmann, J. Krebs, D. Mangold, S. Nitsch, S. Riezler, M. Thiel and V. Schneider-Lindner
Ground truth labels challenge the validity of sepsis consensus definitions in critical illness
Journal of Translational Medicine, 20(6), 27, 2022
