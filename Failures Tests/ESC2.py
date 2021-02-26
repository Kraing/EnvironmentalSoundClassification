### Common Functions used for HDA - Environmental Sould Classification Project
import numpy as np
import pandas as pd
import librosa

import os
import time
import re
from tqdm import tqdm
import h5py

import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import IPython.display
import librosa.display


########################
### COMMON FUNCTIONS ###
########################

# Load raw data
def Load_RAW(path):
    '''
        Input:
            path: folder of the dataset
        
        Output:
            raw_data:  list that contains the raw data
            cvs:       list that contains the cross-fold number
            labels:    list that contains the category information
    '''
    
    # Container for the dataset
    raw_data = []
    cvs = []
    labels = []
    # Load every file inside the folder
    for file_name in tqdm(os.listdir(path)):

        try:
            # Get audio data and sampling rate
            audio, sampling_rate = librosa.load(os.path.join(path, file_name), res_type='kaiser_fast')
            # Split the file name
            name_splitted = re.split('[-.]', file_name)
            
            # Append a row of 3 elements
            raw_data.append(audio)
            cvs.append(name_splitted[0])
            labels.append(name_splitted[3])
        except Exception as e:
            pass
    
    # Convert to numpy array
    raw_audio = np.asarray(raw_data)
    cvs = np.asarray(cvs, dtype=int)
    labels = np.asarray(labels, dtype=int)
    
    # onehot encode the labels in 50 classes
    onehot_labels = to_categorical(labels, num_classes=50)
    
    return raw_audio, cvs, onehot_labels


# Split loaded raw_data into folds
def Split_Folds(raw_audio, cvs, labels, verbose=False):
    '''
        Input:
            raw_audio: list that contains the raw data
            cvs:       list that contains the cross-fold number
            labels:    list that contains the category information
            verbose:   flag used to print produced folds information
        
        Output:
            f{1,2,3,4,5}:      folds that contains the raw data and labels
    '''
    
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    
    # Loop over each file audio
    for num, audio in enumerate(tqdm(raw_audio)):
        
        if cvs[num] == 1:
            f1.append((audio, labels[num]))
        elif cvs[num] == 2:
            f2.append([audio, labels[num]])
        elif cvs[num] == 3:
            f3.append([audio, labels[num]])
        elif cvs[num] == 4:
            f4.append([audio, labels[num]])
        elif cvs[num] == 5:
            f5.append([audio, labels[num]])
    
    # Convert to numpy array
    f1 = np.asarray(f1, dtype=object)
    f2 = np.asarray(f2, dtype=object)
    f3 = np.asarray(f3, dtype=object)
    f4 = np.asarray(f4, dtype=object)
    f5 = np.asarray(f5, dtype=object)
    
    if verbose:
        print("Folds size: %2d - %2d - %2d - %2d - %2d" % (len(f1), len(f2), len(f3), len(f4), len(f5)))

        print("Folds sample shape: ", len(f1[0]))

        print("Folds sample data shape: ", f1[0][0].shape)
        
        print("Folds sample label type: ", f1[0][1].shape)
    
    return f1, f2, f3, f4, f5


# Split dataset into data and labels
def Split_Data_Label(dataset):
    
    
    data = []
    label = []
    
    for i in range (len(dataset)):
        data.append(dataset[i][0])
        label.append(dataset[i][1])

    
    data = np.asarray(data)
    label = np.asarray(label)
    
    return data, label




# Load saved segments
def Load_Segments(dataset, fold):
    
    if dataset=='ESC10':
        if fold==1:
            hf = h5py.File('ESC10/F1.h5', 'r')    
        if fold==2:
            hf = h5py.File('ESC10/F2.h5', 'r')
        if fold==3:
            hf = h5py.File('ESC10/F3.h5', 'r')
        if fold==4:
            hf = h5py.File('ESC10/F4.h5', 'r')
        if fold==5:
            hf = h5py.File('ESC10/F5.h5', 'r')

        
        
    if dataset=='ESC50':
        if fold==1:
            hf = h5py.File('ESC50/F1.h5', 'r')    
        if fold==2:
            hf = h5py.File('ESC50/F2.h5', 'r')
        if fold==3:
            hf = h5py.File('ESC50/F3.h5', 'r')
        if fold==4:
            hf = h5py.File('ESC50/F4.h5', 'r')
        if fold==5:
            hf = h5py.File('ESC50/F5.h5', 'r')
    
    # Get training
    train_d = np.array(hf.get('train_data'))
    train_l = np.array(hf.get('train_label'))


    # Get validation
    val_d = np.array(hf.get('validation_data'))
    val_l = np.array(hf.get('validation_label'))

    # Get test
    test_d = np.array(hf.get('test_label'))
    test_l = np.array(hf.get('test_label'))

    hf.close()
    
    # Cast to float32
    train_d = np.asarray(train_d, dtype=np.float32)
    train_l = np.asarray(train_l, dtype=np.float32)
    
    val_d = np.asarray(val_d, dtype=np.float32)
    val_l = np.asarray(val_l, dtype=np.float32)
    
    test_d = np.asarray(test_d, dtype=np.float32)
    test_l = np.asarray(test_l, dtype=np.float32)
    
    
    
    return train_d, train_l, val_d, val_l, test_d, test_l


def Compute_MelSpec(dataset, bands=60):

    features = []
    for segment in dataset:
        features.append(librosa.core.amplitude_to_db(librosa.feature.melspectrogram(segment, n_mels=bands)))
    
    log_specgrams = np.asarray(features).reshape(len(features), bands, 41, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    
    # compute delta_1
    for i in range(len(log_specgrams)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
                              
    features = features.astype(np.float32)    
    return features

                              
def Compute_MelSpec3(dataset, bands=60):

    features = []
    for segment in dataset:
        features.append(librosa.core.amplitude_to_db(librosa.feature.melspectrogram(segment, n_mels=bands)))
    
    log_specgrams = np.asarray(features).reshape(len(features), bands, 41, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams)), np.zeros(np.shape(log_specgrams))), axis=3)
    
    # compute delta_1
    for i in range(len(log_specgrams)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
                              #compute delta_2
    for i in range(len(log_specgrams)):
        features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 1])
                              
    features = features.astype(np.float32)    
    return features
                              


def CreateTrainingSet10(data, label, name='', batch_size=32):
    
    # Shuffle the folds
    rnd_indices = np.arange(0, len(data))
    rnd_indices = np.random.shuffle(rnd_indices)
    
    data = data[rnd_indices].reshape((len(data), len(data[0])))
    label = label[rnd_indices].reshape((len(label), 10))
    
    
    data = data.astype(np.float32)
    label = label.astype(np.float32)

    # Create dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    
    # Cache the dataset
    #training_dataset = training_dataset.cache(name)
    
    # Shuffle all elements at every iteration
    training_dataset = training_dataset.shuffle(len(training_dataset))
    
    # Define batch_size and prefetch size
    training_dataset = training_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    return training_dataset


def CreateTrainingSet50(data, label, name='', batch_size=32):
    
    # Shuffle the folds
    rnd_indices = np.arange(0, len(data))
    rnd_indices = np.random.shuffle(rnd_indices)
    
    data = data[rnd_indices].reshape((len(data), len(data[0])))
    label = label[rnd_indices].reshape((len(label), 50))
    
    
    data = data.astype(np.float32)
    label = label.astype(np.float32)

    # Create dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    
    # Cache the dataset
    #training_dataset = training_dataset.cache(name)
    
    # Shuffle all elements at every iteration
    training_dataset = training_dataset.shuffle(len(training_dataset))
    
    # Define batch_size and prefetch size
    training_dataset = training_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    return training_dataset


def CreateValidationSet(data, label, name='', batch_size=32):
    
    data = data.astype(dtype=np.float32)
    label = label.astype(dtype=np.float32)
    
    # Create and cache training
    validation_dataset = tf.data.Dataset.from_tensor_slices((data, label))
    
    # Cache dataset
    #validation_dataset = validation_dataset.cache(name)
    
    # Define batch_size and prefetch size
    validation_dataset = validation_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    return validation_dataset




def Data_Augmentation(data_batch, ratio=0.2):
    
    
    pitch_factor = np.random.uniform(-12, 12, len(data_batch))
    n_r = np.random.uniform(0, 1, len(data_batch))
    p_r = np.random.uniform(0, 1, len(data_batch))
    i_r = np.random.uniform(0, 1, len(data_batch))
    
    aug = np.zeros(np.shape(data_batch))
    # Loop over samples
    for i in range(len(data_batch)):
        
        if(p_r[i]<ratio):
            aug[i] = librosa.effects.pitch_shift(np.asarray(data_batch[i]), 22050, pitch_factor[i])
            
        if(n_r[i]<ratio):
            aug[i] = data_batch[i] + np.random.uniform(0.001, 0.05, size=len(data_batch[i]))
        
        if(i_r[i]<ratio):
            aug[i] = np.flip(data_batch[i])
    
    return aug



def train(net, max_epochs, training_dataset, validation_dataset, aug_rate=0.2, verbose=True):
    
    epoch_loss= []
    epoch_acc = []

    epoch_vl = []
    epoch_va = []

    # Loop over the epochs
    for epoch in range(max_epochs):


        step_loss = []
        step_acc = []

        step_vl = []
        step_va = []

        start = time.time()
        # train over mini-batches
        for x_batch, y_batch in training_dataset:

            # randomly add noise-pitch-reverse
            #x_batch = Data_Augmentation(x_batch, ratio=aug_rate)

            # convert to melspectrogram
            x_batch = Compute_MelSpec(np.asarray(x_batch))

            # scale to 0 1
            x_batch = np.interp(x_batch, (-100., 150.), (0, 1))

            # train on batch
            step_stats = net.train_on_batch(x_batch, y_batch)

            # save loss and accuracy
            step_loss.append(step_stats[0])
            step_acc.append(step_stats[1])

        # compute validation stats

        for x_batch, y_batch in validation_dataset:

            # convert to melspectrogram
            x_batch = Compute_MelSpec(np.array(x_batch))

            # scale to 0 1
            x_batch = np.interp(x_batch, (-100., 150.), (0, 1))

            # compute validation stats
            val_stats = net.test_on_batch(x_batch, y_batch)

            # save loss and accuracy
            step_vl.append(val_stats[0])
            step_va.append(val_stats[1])

        end = time.time()

        # Save the mean loss and accuracy of the entire epoch
        epoch_loss.append(np.mean(step_loss))
        epoch_acc.append(np.mean(step_acc))
        epoch_vl.append(np.mean(step_vl))
        epoch_va.append(np.mean(step_va))

        # Print epoch training stats
        if verbose:
            print("Epoch %2d: \t t-loss: %3.6f \t t-acc: %.6f \t v-loss: %3.6f \t v-acc: %.6f \t time: %3.3f" % (epoch + 1, epoch_loss[-1], epoch_acc[-1], epoch_vl[-1], epoch_va[-1], (end - start)))
    
    return  epoch_loss, epoch_acc,  epoch_vl, epoch_va


######################
### NETWORK MODELS ###
######################


def PiczakNet10(input_shape):
    
    X_input = tf.keras.Input(input_shape)
    
    # First convolution block
    model = tf.keras.layers.Conv2D(80, kernel_size=(57, 6), strides=1, padding='same', name='conv0')(X_input)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(4, 3), strides=(1, 3), padding='same')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second convolution block
    model = tf.keras.layers.Conv2D(80, kernel_size=(1, 3), strides=1, padding='same', name='conv1')(model)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding='same')(model)
    
    # Flatten
    model = tf.keras.layers.Flatten()(model)
    
    # First fully-connected block
    model = tf.keras.layers.Dense(5000, activation='relu', name='fc0')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second fully-connected block
    model = tf.keras.layers.Dense(5000, activation='relu', name='fc1')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Output layer
    model = tf.keras.layers.Dense(10, activation=None, name='out')(model)
    
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = model, name='PiczakNet10')
    
    return model



def PiczakNet50(input_shape):
    
    X_input = tf.keras.Input(input_shape)
    
    # First convolution block
    model = tf.keras.layers.Conv2D(80, kernel_size=(57, 6), strides=1, padding='same', name='conv0')(X_input)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(4, 3), strides=(1, 3), padding='same')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second convolution block
    model = tf.keras.layers.Conv2D(80, kernel_size=(1, 3), strides=1, padding='same', name='conv1')(model)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding='same')(model)
    
    # Flatten
    model = tf.keras.layers.Flatten()(model)
    
    # First fully-connected block
    model = tf.keras.layers.Dense(5000, activation='relu', name='fc0')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second fully-connected block
    model = tf.keras.layers.Dense(5000, activation='relu', name='fc1')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Output layer
    model = tf.keras.layers.Dense(50, activation=None, name='out')(model)
    
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = model, name='PiczakNet50')
    
    return model

# Layer regularized
def PiczakNet_Reg(input_shape):
    
    X_input = tf.keras.Input(input_shape)
    
    # First convolution block
    model = tf.keras.layers.Conv2D(80, kernel_size=(57, 6), strides=1, padding='same', name='conv0')(X_input)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(4, 3), strides=(1, 3), padding='same')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second convolution block
    model = tf.keras.layers.Conv2D(80, kernel_size=(1, 3), strides=1, padding='same', name='conv1')(model)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(1, 3), strides=(1, 3), padding='same')(model)
    
    # Flatten
    model = tf.keras.layers.Flatten()(model)
    
    # First fully-connected block
    model = tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=l2(0.0001), name='fc0')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second fully-connected block
    model = tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=l2(0.0001), name='fc1')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Output layer
    model = tf.keras.layers.Dense(50, activation='softmax', kernel_regularizer=l2(0.0001), name='out')(model)
    
    # bias_regularizer=l2(0.01)
    
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = model, name='PiczakNet_Reg')
    
    return model



def DFFNet(input_shape):
    
    X_input = tf.keras.Input(input_shape)

    # Flatten
    model = tf.keras.layers.Flatten()(X_input)
    
    # First fully-connected block
    model = tf.keras.layers.Dense(256, activation='relu', name='fc0')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Second fully-connected block
    model = tf.keras.layers.Dense(256, activation='relu', name='fc1')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Output layer
    model = tf.keras.layers.Dense(50, activation='softmax', name='out')(model)
    
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = model, name='PiczakNet')
    
    return model


def ResBlock(X_input, in_f, out_f):
    
    # First block
    X = tf.keras.layers.Conv2D(in_f, kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer='glorot_uniform')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Second block
    X = tf.keras.layers.Conv2D(in_f, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='glorot_uniform')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Third block
    X = tf.keras.layers.Conv2D(out_f, kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer='glorot_uniform')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    
    # Shortcut path
    X_shortcut = tf.keras.layers.Conv2D(out_f, kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer='glorot_uniform')(X_input)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)
    
    # Merge input and the new branch
    X = tf.keras.layers.Add()([X_shortcut, X])
    X = tf.keras.layers.Activation('relu')(X)

    return X

def ESResNet(input_shape):
    
    X_input = tf.keras.Input(input_shape)
    
    X = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer='glorot_uniform')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = ResBlock(X, 64, 256)
    X = ResBlock(X, 256, 512)
    X = ResBlock(X, 512, 1024)
    X = ResBlock(X, 1024, 2048)

    X = tf.keras.layers.AveragePooling2D(pool_size=2)(X)
    
    X = tf.keras.layers.Flatten()(X)
    
    X = tf.keras.layers.Dense(1024, activation='relu')(X)
    
    out = tf.keras.layers.Dense(50, activation='softmax')(X)
    
    model = tf.keras.Model(inputs=X_input, outputs=out, name='ESResNet')
    
    return model


def BackBoneNet(input_shape):
    
    X_input = tf.keras.Input(input_shape)
    
    # First branch with basic 3x3 kernel
    model = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', name='b1_conv0')(X_input)
    model = tf.keras.layers.BatchNormalization(axis=3)(model)
    model = tf.keras.layers.Activation('relu')(model)
    
    # Second branch with two stacked 3x3 kernel should equal to a single 9x9 from the original input
    branch_2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='b2_conv0')(X_input)
    branch_2 = tf.keras.layers.BatchNormalization(axis=3)(branch_2)
    branch_2 = tf.keras.layers.Activation('relu')(branch_2)
    branch_2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', name='b2_conv1')(branch_2)
    branch_2 = tf.keras.layers.BatchNormalization(axis=3)(branch_2)
    branch_2 = tf.keras.layers.Activation('relu')(branch_2)

    
    # Third branch with stacked dilated kernel
    branch_3 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), dilation_rate=(2, 2), padding='same', name='b3_conv0')(X_input)
    branch_3 = tf.keras.layers.BatchNormalization(axis=3)(branch_3)
    branch_3 = tf.keras.layers.Activation('relu')(branch_3)

    # Fifth branch with stacked dilated kernel
    branch_4 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), dilation_rate=(4, 4), padding='same', name='b5_conv0')(X_input)
    branch_4 = tf.keras.layers.BatchNormalization(axis=3)(branch_4)
    branch_4 = tf.keras.layers.Activation('relu')(branch_4)
    
    sdc = tf.concat(values=[branch_3, branch_4], axis=3, name='stacked_dilated_conv')
    sdc = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', name='sdc_conv')(sdc)
    
    
    # Contanetare all the parallel branches
    X = tf.concat(values=[model, branch_2, sdc], axis=3, name='test')
    

    ##### MAIN PATH ##### 
    # First component of main path (3 lines)
    main_path = tf.keras.layers.Conv2D(128, kernel_size=1, strides=2, padding='valid', kernel_initializer='glorot_uniform', name='main_path_' + '1st')(X)
    main_path = tf.keras.layers.BatchNormalization(axis=3)(main_path)
    main_path = tf.keras.layers.Activation('relu')(main_path)
    
    # Second component of main path (3 lines)
    main_path = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform', name='main_path_' + '2nd')(main_path)
    main_path = tf.keras.layers.BatchNormalization(axis=3)(main_path)
    main_path = tf.keras.layers.Activation('relu')(main_path)

    # Third component of main path (2 lines)
    main_path = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', kernel_initializer='glorot_uniform', name='main_path_' + '3rd')(main_path)
    main_path = tf.keras.layers.BatchNormalization(axis=3)(main_path)

    ##### SHORTCUT PATH #### (2 lines)
    X_shortcut = tf.keras.layers.Conv2D(256, kernel_size=1, strides=2, padding='valid', kernel_initializer='glorot_uniform', name='shortcut')(X)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X_final = tf.keras.layers.Add()([X_shortcut, main_path])
    X_final = tf.keras.layers.BatchNormalization(axis=3)(X_final)
    X_final = tf.keras.layers.Activation('relu')(X_final)
    
    # Flatten and fully connect the features
    model = tf.keras.layers.Flatten()(X_final)
    model = tf.keras.layers.Dense(5000, activation='relu', name='fc0')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    model = tf.keras.layers.Dense(1000, activation='relu', name='fc1')(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=model, name='DevNet')
    
    return model


def ESC50Classifier(input_shape=1000):
    
    X_input = tf.keras.Input(input_shape)
    
    # Feature reduction
    model = tf.keras.layers.Dense(500, activation='relu', name='c_fc1')(X_input)
    model = tf.keras.layers.Dropout(0.5)(model)
    
    # Output classifier layer
    model = tf.keras.layers.Dense(50, activation='softmax', name='out')(model)
    
    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=model, name='ESC50_Classifier')
    
    return model


def Docking(back_bone, classifier):
    
    model = tf.keras.Model(inputs=back_bone.input, outputs=classifier(back_bone.output), name='Merged')

    return model
