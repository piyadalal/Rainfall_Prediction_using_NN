# Program trains, validates and tests a fully connected
# neural network (NN) with one hidden layer for multi
# classification for one hour ahead. The hour is specified in
# HOUR_AHEAD.
#
# The NN's output are in range from 0 to 3, each representing
# an interval, namely 0.1-1, 1.1-2, 2.1-3, 3.1-inf respectively.
#
# It reads the datapoints from a CSV file, modifies the data
# to a form appropriate as an input to the NN - expending
# the data to include besides current values of input
# variables also DAYS_BACK previous values of the inputs.
#
# The input data are augmented to make the distribution
# of occurrences of each class approximately equal.
# New samples are added with adding small random noise
# with Guassian distribution to existing measurements.
#
# From that data it produces a training and a test set.
# Their ratio is determined by TRAIN_PER.
#
# After the model is build and trained, it can write the 
# weights and biases of the model to a CSV file.
#
# It also tests the model and includes an example of data
# prediction and its evaluation.
#
# ATTENTION: Number of input variables of the input data
# file has to be set (INPUTS)! 
#
# Author: Uros Hudomalj
# Last revision: 29.11.2018

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


HOUR_AHEAD = 9                                      # number of hour for which rainfall amounts are predicted

ORIGINAL_DATASET = 'dataset/dataset_4v_all.csv'     # path to the input dataset
INPUTS = 4                                          # number of different types of variables to the NN (4: bar, temp, RH and rainfall; or 6: + wind speed and wind direction )

NUM_CLASS = 4                                       # number of labels/classes to classify the data into

TEST_PER = 0.20                                     # percentage of the whole data for the testing of the model

DAYS_BACK = 3                                       # number of days looking back for prediction
HOURS_IN_A_DAY = 24
NN_INPUTS = DAYS_BACK * HOURS_IN_A_DAY * INPUTS     # number of inputs of NN (DAYS_BACK*24-1 past values and current value)

HIDDEN_NODES = int(NN_INPUTS * 2)                   # number of nodes in hidden layer (multiple of NN_INPUTS)

EPOCHS = 1000                                         # epochs = number of times the model goes through the training
TRAINING_VERBOSE = 2                                # for level of output during the training process (0-nothing, 1-progress bar, 2-full)

MAX_RAINFALL_AMOUNT = 62.4                          # representing normalized value of 1 in rainfall amount

### FOR PROGRAM CONTROL

# For model control/output summary

STORE_DATA_PK = True                                                          # if True, stores augmented to STORE_DATA_PK_PATH
STORE_DATA_PK_PATH = 'whole_model/hour_{hour}_model/multi_nn_data.pk'         # path to save .pk file with data for nn_data_in and nn_data_out
STORE_DATA_PK_PATH = STORE_DATA_PK_PATH.format(hour=HOUR_AHEAD)               # to add current hour ahead to the string
LOAD_DATA_PK = False                                    # if True, skips reading data from CSV and its augmentation and loads data from LOAD_DATA_PK_PATH
LOAD_DATA_PK_PATH = STORE_DATA_PK_PATH                  # path to load .pk file with data for nn_data_in and nn_data_out

LOAD_WEIGHTS = False                                     # if True, loads the last saved weights to the model (need to build and compile it prior to weights load)
TRAIN_MODEL = True                                       # if True, the model will train - either form beginning or from the loaded weights if LOAD_WEIGHTS is True

HOUR_MODEL_WEIGHTS_PATH = 'whole_model/hour_{hour}_model/multi_model'      # path where the weights of the completely trained model are saved - used later in rainfall prediction program

WRITE_MODEL_WEIGHTS = True                                                 # if True, writes the model weights to a CSV file specified in WRITE_WEIGHTS_PATH
WRITE_WEIGHTS_PATH = 'whole_model/hour_{hour}_model/multi_nn_weights.csv'  # path to writing weights of the model

PRINT_TRUE_AND_PRED = False                             # if True, print the predicted and true labels of the training data

### COMMENT - To disable saving of model weights during training add a comment to line 'callbacks' in model.fit
checkpoint_path = "model_multi/training_save_files/cp-{epoch:04d}.ckpt"    # path for saving the temporary model weights during training - file name includes epoch number
checkpoint_dir = os.path.dirname(checkpoint_path)

#################     FUNCTIONS     #################

# write_weights(model)
#  writes the weights of the model
def write_weights(model):
    print("Printing weights of the model.")
    text_print_model_weights = ["Weights from input to hidden layer.", "Bias from input to hidden layer.",
                                "Weights from hidden to output layer.", "Bias from hidden to output layer."]
    weights = model.get_weights()

    # Open output CSV file
    with open(WRITE_WEIGHTS_PATH.format(hour=HOUR_AHEAD), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        for i, w in enumerate(weights):
            # print(text_print_model_weights[i])
            writer.writerows([[text_print_model_weights[i]]])

            if i == 0 or i == 2:
                for wj in w:
                    tmp = list(wj)
                    writer.writerows([tmp])
            else:
                writer.writerows([list(w)])
    print("Finished printing model weights to CSV.\n")

#################      PROGRAM      #################

print("Starting program.")

if LOAD_DATA_PK==False:
    # make new inputs

    # DATA GATHERING
    nn_data_in = []         # list containing data for input of NN
    nn_data_out = []        # list containing data for output of NN classified into classes representing intervals

    nn_data_out_mm = []     # list containing data for output of NN in mm

    # Open input CSV file
    with open(ORIGINAL_DATASET, 'r') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=';'))
        print("Read from CSV finished.\n")

        # Expand original data to a form for NN
        for i, itemi in enumerate(csv_reader):                                              # For each row
            if (i >= DAYS_BACK * 24 - 1) and (len(csv_reader) - HOUR_AHEAD > i):            # which has at least DAYS_BACK*24-1 previous values and has at least HOUR_AHEAD values
                nn_data_in.append([])                                                       # make a new row in nn_data.
                nn_data_out.append([])
                nn_data_out_mm.append([])

                for k in range(0, INPUTS):                                                  # For each measured input variable
                    for j in range(0, DAYS_BACK * 24):                                      # append DAYS_BACK*24-1 past and current variables to i. row of nn_data_in.                                                     # Add all other data to nn_data_in
                        nn_data_in[i - (DAYS_BACK * 24 - 1)].append(csv_reader[i - j][k])

                for j in range(HOUR_AHEAD, HOUR_AHEAD + 1):                                 # Add the rainfall amount at HOUR_AHEAD to nn_data_out
                    if i + j < len(csv_reader):
                        tmp = csv_reader[i + j][3]                                          # 4. element ([3]) = RAINFALL AMOUNT
                        tmp = float(tmp)
                        nn_data_out_mm[i - (DAYS_BACK * 24 - 1)].append(tmp)                # write the rainfall amount to nn_data_out_mm
                        if tmp < 0.000001:                                                  # check if the rainfall amount is 0
                            nn_data_out[i - (DAYS_BACK * 24 - 1)].append(0)                 # then write a 0 to nn_data_out
                        else:
                            nn_data_out[i - (DAYS_BACK * 24 - 1)].append(1)                 # in case there is rain, write 1 to nn_data_out

    print("Data expansion finished.\n")

    # Covert data lists to numpy arrays - the classification into intervals works not correctly if done on lists
    nn_data_in = np.array([np.asarray(xi, dtype=float) for xi in nn_data_in])
    nn_data_out = np.array([np.asarray(xi, dtype=float) for xi in nn_data_out])
    nn_data_out_mm = np.array([np.asarray(xi, dtype=float) for xi in nn_data_out_mm])

    # From the nn_data get rid of the 0
    keep_idx = [row[0] != 0 for row in nn_data_out]
    nn_data_in = nn_data_in[keep_idx]
    nn_data_out = nn_data_out[keep_idx]
    nn_data_out_mm = nn_data_out_mm[keep_idx]

    # From the rest classify them in intervals { 0.1-1 = 0, 1.1-2 = 1, 2.1-3 = 2, 3.1-inf = 3}
    for i,x in enumerate(nn_data_out_mm):
        if x[0]>=0.001/MAX_RAINFALL_AMOUNT and x[0]<=1.001/MAX_RAINFALL_AMOUNT:
            nn_data_out[i,0] = 0
        elif x[0]>=1.001/MAX_RAINFALL_AMOUNT and x[0]<=2.001/MAX_RAINFALL_AMOUNT:
            nn_data_out[i,0] = 1
        elif x[0]>=2.001/MAX_RAINFALL_AMOUNT and x[0]<=3.001/MAX_RAINFALL_AMOUNT:
            nn_data_out[i,0] = 2
        elif x[0]>=3.001/MAX_RAINFALL_AMOUNT:
            nn_data_out[i,0] = 3

    # Convert np.arrays to lists to speed up the appending during data augmentation
    nn_data_in = nn_data_in.tolist()
    nn_data_out = nn_data_out.tolist()
    nn_data_out_mm = nn_data_out_mm.tolist()

    print("Starting with data augmentation.")
    # Generate additional measurements for each sample according to the data distribution (original label 0 = 77.5%, 1 = 13.3%, 2 = 4.5%, 3 = 4.8%)
    # Goal is to have approximately the same number of samples for each label. This is done by not augmenting the label with the highest
    # number of samples (e.g. label 0), and adding appropriate amount of new samples for each other label - for each sample of class 1,
    # 2 and 3 add 6, 17 and 16 samples respectively.
    N = [-1, 6, 17, 16]                 # number of new samples to generate per samples for each label

    tmp_size = len(nn_data_out)         # to set how long should data be augmented (as the array is being continuously modified)
    for i in range(0,tmp_size):
        cur_val = nn_data_out[i][0]     # get the sample label
        cur_N = N[int(cur_val)]         # see how many new samples have to be generated

        if cur_N > 0:                   # if the current label is to be augmented, additional samples are produced
            mean = 0                                 # mean value of the Guassian noise
            std_dev = 0.0005                         # standard deviation of the Guassian noise = equal to half of the rounding value of inputs in CSV
            old_sample = nn_data_in[i]
            old_sample = [float(k) for k in old_sample]
            old_sample = np.array(old_sample)
            old_sample_out_mm = nn_data_out_mm[i]
            old_sample_out_mm = [float(k) for k in old_sample_out_mm]
            old_sample_out_mm = np.array(old_sample_out_mm)
#            print("Adding {} new sample for label {} at row={}.".format(int(cur_N),cur_val,i))    # print, which row produces new samples (to know how long the process will take)
            # generate cur_N new samples with Guassian distribution
            for j in range(0,cur_N):
                # add noise to the input values
                noise = np.random.normal(mean, std_dev, old_sample.shape)
                new_sample = old_sample + noise
                new_sample = np.clip(new_sample, 0, 1)    # set any possible values outside of the interval [0,1] due to noise inside it

                new_sample = new_sample.tolist()
                nn_data_in.append(new_sample)             # append the new sample to the existing inputs

                # add noise to the output values in mm
                noise = np.random.normal(mean, std_dev, old_sample_out_mm.shape)
                new_sample_out_mm = old_sample_out_mm + noise
                new_sample_out_mm = np.clip(new_sample_out_mm, 0, 1)                    # set any possible values outside of the interval [0,1] due to noise inside it

                new_sample_out_mm = new_sample_out_mm.tolist()
                nn_data_out_mm.append(new_sample_out_mm) # append the corresponding output in mm to the existing outputs in mm

                # add a new label
                nn_data_out.append([[cur_val]])               # append the corresponding output label to the existing output labels

    # Covert data lists to numpy arrays
    nn_data_in = np.array([np.asarray(xi, dtype=float) for xi in nn_data_in])
    nn_data_out = np.array([np.asarray(xi, dtype=float) for xi in nn_data_out])
    nn_data_out_mm = np.array([np.asarray(xi, dtype=float) for xi in nn_data_out_mm])

    # Store the augmented data for later use
    if STORE_DATA_PK==True:
        print("Storing nn_data for later use.")
        with open(STORE_DATA_PK_PATH, 'wb') as fi:
            # dump data into the file
            pickle.dump(nn_data_in, fi)
            pickle.dump(nn_data_out, fi)
            pickle.dump(nn_data_out_mm, fi)
    print("Data augmentation finished.\n")
else:
    print("Loading nn_data from saved values.")
    with open(LOAD_DATA_PK_PATH, 'rb') as fi:
        # load data from the file
        nn_data_in = pickle.load(fi)
        nn_data_out = pickle.load(fi)
        nn_data_out_mm = pickle.load(fi)
    print("Finished loading nn_data.")

print("Dividing data to training and testing sets.")
# Generate training and testing data
nn_data_in_train, nn_data_in_test, nn_data_out_train, nn_data_out_test = train_test_split(nn_data_in, nn_data_out, test_size=TEST_PER, random_state=42)

# NN MODEL

# Setup model
# Setup layers of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(NN_INPUTS,)),                                         # number of inputs to model is NN_INPUTS
    keras.layers.Dense(HIDDEN_NODES, activation=tf.nn.leaky_relu),                          # hidden layer has HIDDEN_NODES and uses leaky relu function for activation
    keras.layers.Dense(NUM_CLASS, activation=tf.nn.softmax)                                 # output layer has NUM_CLASS outouts and uses softmax as activation to present the probabilites of the output to present a class
    ])

# Compiling model
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)                                 # optimizer ('sgd' = stochastic gradient descent, or 'adam' or RMSPropOptimizer - a special version of 'sgd')
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',                                           # measurement of correctness of model [Mean Absolute Error or Mean Square Error]
              metrics=['accuracy'],                                                             # other metrics for evaluation of model
              sample_weight_mode=None)                                                          # provide each input datapoint with a different weight

# For automatic saving of weights during training
# How to include the saved model see https://www.tensorflow.org/tutorials/keras/save_and_restore_models
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_weights_only=True,
    period=int(EPOCHS / 10))  # After how many epochs the weights are saved (currently each 10%)
#####

# Load saved weights of the model
if LOAD_WEIGHTS == True:
    print("Loading model weights form last saved values.")
    model.load_weights(HOUR_MODEL_WEIGHTS_PATH.format(hour=HOUR_AHEAD))

# Train the model - either form beginning or from the loaded weights if LOAD_WEIGHTS was true
if TRAIN_MODEL == True:
    print("Starting training of the model.")
    # Training the model
    model.fit(nn_data_in_train, nn_data_out_train,
              epochs=EPOCHS, validation_split=0.1,      # 10% of the training data is used for validation of the model during training
              callbacks=[cp_callback],                  # After each iteration of training execute a method - for saving model weights
              verbose=TRAINING_VERBOSE
              )

# (After training) save the model weights
model.save_weights(HOUR_MODEL_WEIGHTS_PATH.format(hour=HOUR_AHEAD))

# Write weights of the model (to a CSV)
if WRITE_MODEL_WEIGHTS == True: write_weights(model)

print("Starting evaluation of the model.")
# Evaluate model with test datapoints
test_loss, test_acc = model.evaluate(nn_data_in_test, nn_data_out_test)
print('Test accuracy: {}'.format(test_acc))

# Example of making predictions
nn_data_out_pred = model.predict(nn_data_in_test)

# Evaluate the correctness of the classifications
nn_data_out_pred_class = []                                                                 # list containing predicted labels/classes
for x in nn_data_out_pred:                                                                  # convert the probabilities into labels
    lab = np.argmax(x)                                                                      # find the highest probability - its position is the label
    nn_data_out_pred_class.append(int(lab))

dol = len(nn_data_out_pred)                                                                 # length = number of all the samples
tmp = [ int(x[0]) for x in nn_data_out_test]                                                # convert test data from np array into list of ints

same_idx = [nn_data_out_pred_class[i]==tmp[i] for i,xi in enumerate(nn_data_out_pred_class)]    # a list with correct predicted labels
same = [x for i,x in enumerate(nn_data_out_pred_class) if same_idx[i] == True]
dif_pred = [x for i,x in enumerate(nn_data_out_pred_class) if same_idx[i] == False]             # false ones in predicted
dif_test = [x for i,x in enumerate(tmp) if same_idx[i] == False]                                # the right ones in test
dif_pred_prob = [x for i,x in enumerate(nn_data_out_pred) if same_idx[i] == False]                                # probabilites of the false ones in pred

if PRINT_TRUE_AND_PRED == True:
    print(nn_data_out_pred_class)                                                               # print the predicted labels
    print(tmp)                                                                                  # print the true labels
    print("Differences - predicted vs true")
    print("Predicted:")
    print(dif_pred)
    print("True values:")
    print(dif_test)
    print("Predicted probabilities:")
    print(dif_pred_prob)

# Evaluation for each variable
cor_lab = [ 0 for x in range(0,NUM_CLASS)]                                                                                    # contains number of correct classification for each possible label
fal_lab = [ 0 for x in range(0,NUM_CLASS)]                                                                                    # contains number of false classification for each possible label
for i,x in enumerate(nn_data_out_pred_class):
    if x == tmp[i]:
        cor_lab[x] = cor_lab[x] + 1
    else:
        fal_lab[x] = fal_lab[x] + 1

print("Correct classifications in percentage: {:.2f}%". format(len(same)/dol*100))
for i in range(NUM_CLASS):
    print("Correctness of label {}: {:.2f}%".format(i, cor_lab[i]*100/(cor_lab[i]+fal_lab[i])))
