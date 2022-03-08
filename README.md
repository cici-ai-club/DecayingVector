# decayingVector_las
A repo where you could train events to cluster
## Description
- **savedata.py** is the file for split train and test data
- **rtraindecay.py** is the python file for train the model and will save the best model
- **rtestdecay_loov.py** is the python file for test the model and will output a csv file and print out accuracy reports
- **mynet.py** is the model file contains the neural network we used in the training and testing
- **model_clssave** is the directory contains all currently trained model, for example **bestmodel4_0.5.pt** is the model trained using all the users without user 4 under decaying rate 0.5
- **clssave** is the directory contains and the train, test data
- **mappings.npy** is the mappings which connects word and id and is built in rtraindecay.py
- **saveres_loov** is the directory stores the all the saving csv after running the the test file. For example, **savepredic_res_confidence0.5all_0.75_1.csv** is the test esult of user 1 under confidence 0.5 from a model traiend using dacaying rate 0.75.
## Split data of newevents.csv into train and test
```
python savedata.py
```
## Train a model
```
python rtraindecay.py 0 0.5 # train a model with all users except user 0 with decaying rate 0.5
python rtraindecay.py 1 0.75 # train a model with all users except user 1 with decaying rate 0.75

```
## Test a model
```
python rtestdecay_loov.py 0 0.5  # Test the trained model on user 0 data with decaying rate 0.5 (model trained with the data using other users but not user 0)
python rtestdecay_loov.py 1 0.75 # Test the trained model on user10 data with decaying rate 0.75 (model trained with the data using other users but not user 1)
```
