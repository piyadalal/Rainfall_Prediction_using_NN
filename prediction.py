# Program makes prediction of rainfall for the next
# HOURS_AHEAD. This is done based on three NN for
# each hour.
#
# The first NN predicts if it will
# rain or not (1 or 0). If it should rain, it uses
# second NN to estimate the interval of the rainfall.
# With the third NN (regression) predicts the amount
# of rainfall in the last interval (largest). For
# other intervals, the output rainfall amount is
# considered the average amount in the interval
# (as the MAE of the regression even in those
# intervals would be similar to this decision).
#
# Program reads the datapoints from CSV file, modifies
# the data to a form appropriate as an input to the NN
# - expending the data to include besides current values
# of input variables also DAYS_BACK previous values of
# the inputs.
#
# From that data it produces an evaluation set.
# Its size is determined by EVAL_PER.
#
# The models for each layer are loaded from the respective
# folders containing the trained models.
#
# ATTENTION: Models have to be trained in advance as well
# as the newest version specified!
#
# Author: Uros Hudomalj
# Last revision: 28.11.2018

#################      IMPORTS      #################
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
import os
import pickle

#################     CONSTANTS     #################
ORIGINAL_DATASET = 'dataset/dataset_4v_nov_2018.csv'    # path to the input dataset
INPUTS = 4                                              # number of different types of variables to the NN (4: bar, temp, RH and rainfall; or 6: + wind speed and wind direction )

EVAL_PER = 0.20                                         # percentage of the whole data for the evaluation of the whole model

DAYS_BACK = 3                                           # number of days looking back for prediction
HOURS_AHEAD = 1                                         # number of hours ahead for which rainfall amounts are predicted

HOURS_IN_A_DAY = 24
NN_OUTPUTS = HOURS_AHEAD                                # number of outputs of NN
NN_INPUTS = DAYS_BACK * HOURS_IN_A_DAY * INPUTS         # number of inputs of NN (DAYS_BACK*24-1 past values and current value)

HIDDEN_NODES = int(NN_INPUTS * 2)                       # number of nodes in hidden layer (multiple of NN_INPUTS)

MAX_RAINFALL_AMOUNT = 62.4                              # representing normalized value of 1 in rainfall amount

### FOR PROGRAM CONTROL

STORE_DATA_PK = False                                   # if True, stores data read from CSV and augmented to LOAD_DATA_PK_PATH
STORE_DATA_PK_PATH = 'whole_model/nn_data.pk'           # path to save .pk file with data for nn_data_in and nn_data_out
LOAD_DATA_PK = True                                     # if True, skips reading data from CSV and loads data from LOAD_DATA_PK_PATH
LOAD_DATA_PK_PATH = STORE_DATA_PK_PATH                  # path to load .pk file with data for nn_data_in and nn_data_out

GRAPHICAL_REPRESENTATION_OF_ACCURACY = True             # if True, show graphs presenting true vs predicted values and error distribution for each hour ahead

BIN_MODELS_PATH = "whole_model/hour_{hour}_model/bin_model"        # path to the models of the binary NNs
MULTI_MODELS_PATH = "whole_model/hour_{hour}_model/multi_model"    # path to the models of the multi NNs
REG_MODELS_PATH = "whole_model/hour_{hour}_model/reg_model"        # path to the models of the regression NNs

#################      PROGRAM      #################

print("Starting program.")

if LOAD_DATA_PK == False:
    # DATA GATHERING
    nn_data_in = []         # list containing data for input of NN
    nn_data_out = []        # list containing data for output of NN

    # Open input CSV file
    with open(ORIGINAL_DATASET, 'r') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=';'))
        print("Read from CSV finished.\n")

        # Expand original data to a form for NN
        for i, itemi in enumerate(csv_reader):                                              # For each row
            if (i >= DAYS_BACK * 24 - 1) and (len(csv_reader) - HOURS_AHEAD > i):           # which has at least DAYS_BACK*24-1 previous values and has at least mi+1 values
                nn_data_in.append([])                                                       # make a new row in nn_data.
                nn_data_out.append([])

                for k in range(0, INPUTS):                                                  # For each measured input variable
                    for j in range(0, DAYS_BACK * 24):                                      # append DAYS_BACK*24-1 past and current variables to i. row of nn_data_in.                                                     # Add all other data to nn_data_in
                        tmp = csv_reader[i - j][k]
                        tmp = float(tmp)
                        nn_data_in[i - (DAYS_BACK * 24 - 1)].append(tmp)

                for j in range(1, HOURS_AHEAD + 1):                                         # Add next HOURS_AHEAD rainfall data to nn_data_out
                    if i + j < len(csv_reader):
                        tmp = csv_reader[i + j][3]                                          # 4. element ([3]) = RAINFALL AMOUNT
                        tmp = float(tmp)
                        nn_data_out[i - (DAYS_BACK * 24 - 1)].append(tmp)

    # Covert data lists to numpy arrays
    nn_data_in = np.array([ np.asarray(xi, dtype=float) for xi in nn_data_in ])
    nn_data_out = np.array([ np.asarray(xi, dtype=float) for xi in nn_data_out ])
    print("Data gathering finished.\n")

    # Store the nn_data for later use
    if STORE_DATA_PK==True:
        print("Storing nn_data for later use.")
        with open(STORE_DATA_PK_PATH, 'wb') as fi:
            # dump data into the file
            pickle.dump(nn_data_in, fi)
            pickle.dump(nn_data_out, fi)
else:
    print("Loading nn_data from saved values.")
    with open(LOAD_DATA_PK_PATH, 'rb') as fi:
        # load data from the file
        nn_data_in = pickle.load(fi)
        nn_data_out = pickle.load(fi)
    print("Finished loading nn_data.")

# Generate evaluation data
print("Generating evaluation set.")
not_needed_1, nn_data_in_eval, not_needed_2, nn_data_out_eval = train_test_split(nn_data_in, nn_data_out, test_size=EVAL_PER, random_state=42)

# Make rainfall prediction for each hour up to HOURS_AHEAD
for cur_hour in range(1,HOURS_AHEAD+1):
    # Load all the NN models for the current hour (cur_hour)
    print("Loading all NN models for hour{hour}".format(hour=cur_hour))

    # for the BIN NN
    # Setup layers of the BIN model
    bin_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(NN_INPUTS,)),
        keras.layers.Dense(HIDDEN_NODES, activation=tf.nn.leaky_relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    # Compiling model
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    bin_model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  sample_weight_mode=None)
    # Load saved weights of the model
    bin_model.load_weights(BIN_MODELS_PATH.format(hour=cur_hour))

    # for the MULTI NN
    # Setup layers of the MULTI model
    multi_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(NN_INPUTS,)),
        keras.layers.Dense(HIDDEN_NODES, activation=tf.nn.leaky_relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    # Compiling model
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    multi_model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  sample_weight_mode=None)
    # Load saved weights of the model
    multi_model.load_weights(MULTI_MODELS_PATH.format(hour=cur_hour))

    # for the REG NN
    # Setup layers of the REG model
    reg_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(NN_INPUTS,)),  # number of inputs to model is NN_INPUTS
        keras.layers.Dense(HIDDEN_NODES, activation=tf.nn.leaky_relu),
        keras.layers.Dense(1, activation=tf.math.exp)
    ])
    # Compiling model
    learning_rate = 0.0001
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)  # optimizer ('sgd' = stochastic gradient descent, or 'adam' or RMSPropOptimizer - a special version of 'sgd')
    reg_model.compile(optimizer=optimizer,
                  loss='mae',  # measurement of correctness of model [Mean Absolute Error or Mean Square Error]
                  metrics=['mse', 'mae'],  # other metrics for evaluation of model
                  sample_weight_mode=None)  # provide each input datapoint with a different weight
    # Load saved weights of the model
    reg_model.load_weights(REG_MODELS_PATH.format(hour=cur_hour))


    print("Starting prediction for hour {hour}.".format(hour=cur_hour))
    # Predict using bin NN model
    bin_pred = bin_model.predict(nn_data_in_eval)

    # turn predicted probabilities into classes
    bin_pred = np.argmax(bin_pred, axis=1)

    # do predictions on multi NN
    multi_pred = multi_model.predict(nn_data_in_eval)

    # again turn probabilities into labels
    multi_pred = np.argmax(multi_pred, axis=1)

    # do predictions on reg NN
    reg_pred = reg_model.predict(nn_data_in_eval)

    nn_data_out_pred = []
    # combine the results = non zero rainfalls change with corresponding regression NN amounts
    for i, xi in enumerate(bin_pred):
        nn_data_out_pred.append([])
        # if rain is predicted
        if xi>0:
            # if the rainfall is from the last interval
            if multi_pred[i]==3:
                # and set the appropriate regression value to the output
                tmp = reg_pred[i][0]
            else:
                # turn class label into average value for the interval
                tmp = (multi_pred[i]+0.5)/MAX_RAINFALL_AMOUNT
            nn_data_out_pred[i].append(tmp)
        else:
            nn_data_out_pred[i].append(0)

    # Covert data lists to numpy arrays
    nn_data_out_pred = np.array([ np.asarray(xi, dtype=float) for xi in nn_data_out_pred ])

    # Plotting the difference between predicted and real values on testing data
    print("Graphical representation of predicted vs true values on evaluation data for hour {hour}.".format(hour=cur_hour))
    plt.scatter(nn_data_out_eval[:,cur_hour-1]*MAX_RAINFALL_AMOUNT, nn_data_out_pred[:,0]*MAX_RAINFALL_AMOUNT)
    plt.xlabel('True Values [mm]')
    plt.ylabel('Predictions [mm]')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([0, MAX_RAINFALL_AMOUNT], [0, MAX_RAINFALL_AMOUNT])
    plt.show(block=True)

    # Plotting of the errors on testing data
    error = (nn_data_out_pred[:,0] - nn_data_out_eval[:,cur_hour-1])*MAX_RAINFALL_AMOUNT
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error [mm]")
    _ = plt.ylabel("Count [mm]")

    print("Max absolute error: {} mm".format(np.max(np.abs(error))))
    print("Average absolute error: {} mm".format(np.average(np.abs(error))))

    plt.show(block=True)