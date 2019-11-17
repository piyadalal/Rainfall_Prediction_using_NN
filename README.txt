This is a repo for the project "Local rainfall prediction for improving crop yield" for the course "Artificial Intelligence for Embedded Systems" at TUM.

It contains description of a model for predicting rainfall amounts for each hour up to 24 hours ahead.
The model is built from 3 fully connected neural networks for predicting independently rainfall amount for each hour.
There are programs for training and testing of each type of NNs and a program for making predictions based on the models.
The model is trained from data gathered from DWD.

Whole project is done in Python using Tensorflow. Other Python modules needed are: scikit-learn, matplotlib and numpy.

Repo contents:
	- README.txt : this file 
	- STILL_TO_DO : a list of problem which still exist in the code or either are still needed to implement
	- UPDATE_INFO : a detailed list of things which were changed since the last commit - Please update regularly!
	- SETUP_GUIDE : a guide what to setup for running the prediction model and its training
	- dataset/ : various sizes of datasets in form of: dataset_Xv_Yk.csv where X depicts number of weather variables used and Y number of datapoints in thousands
	- model_NNtype/ : folder for containing temporary data for each type of NN (bin, multi, reg)
	    - training_save_files/ : contains files of saved weights during training of the model
	- whole_model/ : folder contains all the trained NN models' weights and the data needed for the training, validation and testing - 3 for each hour
	    - hour_X_model/ : folder contains trained weights of each of 3 NNs needed for predicting rainfall amount for hour X and the needed data for the training (X = 1-24)
	        - NNtype_model : contains trained weights for each NN type in Tensorflow form (checkpoint, .index and .data)
	        - NNtype_nn_weights : contains weights for each NN type in written in CSV format
	        - NNtype_nn_data : contains data needed for the training, validation and testing of the NN saved in Python format .pk - TO LARGE TO COMMIT
	        - DONE_BY_NAME.txt : file containing who has done / is doing the training for the hour X and the results of the training - UPDATE TO REPO BEFORE THE START OF THE TRAINING (so that 2 people don't do it the same time)!!!
        - nn_data.pk : file in Python format .pk containing data for making prediction on them from 'prediction.py' - TO LARGE TO COMMIT
    - NNtype_nn.py : program for producing the needed data form for the training of the NN. It also does the training, validation and the testing of the model.
    - prediction.py : program for making rainfall predictions for X hours ahead - all the hour models have to be trained in advance !