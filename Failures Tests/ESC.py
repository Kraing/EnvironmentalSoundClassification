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


# Perform data augmentation on the original dataset
def Data_Augmentation(dataset, number, name='', path='Augmented_Data/', save=False, noise=False, pitch=False):
    '''
        Input:
            dataset:           dataset to augment
            number:            number of augmentations for each sample per type
            name:              name of the file if save
            path:              path in which save the data
            save:              flag for saving augmented data
            noise:             flag to enable noise augmentation
            pitch:             flag to enable pitching augmentation
            
        
        Output:
            new_dataset:       augmented data
    '''
    new_dataset = []
    
    # Loop over each sample and augment according to the input flags
    for sample in tqdm(dataset):
        
        # Append the original sample
        new_dataset.append([sample[0], sample[1]])
        
        # Generate noisy samples
        if(noise):
            noise_samples = Noise_Augmentation(sample[0], number)
            
            # Append the generated samples
            for gen in noise_samples:
                new_dataset.append([gen, sample[1]])
        
        # Generate pitched samples
        if(pitch):
            pitch_samples = Pitch_Augmentation(sample[0], number)
            
            # Append the generated samples
            for gen in pitch_samples:
                new_dataset.append([gen, sample[1]])
    
    new_dataset = np.asarray(new_dataset, dtype=object)
    
    # Get splitted version
    d2s, l2s = Split_Data_Label(new_dataset)
    
    d2s = np.asarray(d2s, dtype=np.float32)
    l2s = np.asarray(l2s, dtype=np.float32)
    
    if(save):
        hf = h5py.File(path + name + '.h5', 'w')
        hf.create_dataset('data', data=d2s)
        hf.create_dataset('label', data=l2s)
        hf.close()
        
        
    return new_dataset


# Load saved data
def Load_Augmented(name='', path='Augmented_Data/'):
    '''
        Input:
            name:      name of the file
            path:      path of the file
        
        Output:
            dataset:   loaded dataset with data and labels
    '''
    hf = h5py.File(path + name + '.h5', 'r')
    data =  np.array(hf.get('data'))
    labels = np.array(hf.get('label'))
    hf.close()
    
    data = np.asarray(data, dtype=np.float32)
    label = np.asarray(labels, dtype=np.float32)
    return data, labels


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




def Preprocessing(raw_audio, labels, bands=60, frames=41):
    '''
        Input:
            raw_audio:     list that contains the raw/augmented data
            labels:        list that contains the category information
            bands:         number of mel band to use
            frames:        number of frames to use
        
        Output:
            features:      numpy array that contains processed audio data with log-melspec and delta
            new_labels:    new labels for each augmented segment
    '''    
    
    new_labels = []
    augmented_spec = []
    
    # Normalize the raw data
    norm_factor = np.percentile(raw_audio, 99) - np.percentile(raw_audio, 5)
    raw_audio = raw_audio / norm_factor
    
    # Loop over each file audio
    for num, audio in enumerate(tqdm(raw_audio)):
    
        # Convert audio to melspectogram
        '''
            With default n_fft=2048 we have the filter size of 2048/2+1=1025 [Nyquist Frequency]
        '''
        melspec = librosa.feature.melspectrogram(audio, n_mels=bands, hop_length=512)
        
        # Convert melspec to log melspec
        logspec = librosa.core.amplitude_to_db(melspec)
        
        counter = 0
        # Spectrogram splitting with 50% overlap and adapt cv-fold and labels info
        for idx in range(0, len(logspec[0]) - frames, int(frames/2)):
            augmented_spec.append(logspec[:, idx:idx+frames])
            new_labels.append(labels[num])
            counter = counter +1
            
    # Reshape the outputs
    log_specgrams = np.asarray(augmented_spec).reshape(len(augmented_spec), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    new_labels = np.asarray(new_labels)
    
    # Fill the delta features
    for i in range(len(log_specgrams)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    features = features.astype(np.float32)
    labels = labels.astype(np.int)
    return features, new_labels

# Preprocessing
def Filtered_Preprocessing(raw_audio, labels, threshold=0.0001, bands=60, frames=41):
    '''
        Input:
            raw_audio:     list that contains the raw/augmented data
            labels:        list that contains the category information
            bands:         number of mel band to use
            frames:        number of frames to use
        
        Output:
            features:      numpy array that contains processed audio data with log-melspec and delta
            new_labels:    new labels for each augmented segment
    '''    

    
    segments = []
    segment_labels = []
    
    augmented_spec = []
    new_labels = []
    log_specgrams = []
    
    # Normalize the raw data
    #norm_factor = np.percentile(raw_audio, 99) - np.percentile(raw_audio, 5)
    #raw_audio = raw_audio / norm_factor
    
    # Loop over each file audio and divide into segments
    for num, audio in enumerate(tqdm(raw_audio)):
    
        # Convert audio to melspectogram
        '''
            With default n_fft=2048 we have the filter size of 2048/2+1=1025 [Nyquist Frequency]
        '''
        #melspec = librosa.feature.melspectrogram(audio, n_mels=bands, hop_length=512)

        # Spectrogram splitting with 50% overlap and adapt cv-fold and labels info
        for idx in range(0, len(raw_audio[0]), int(20480/2)):
            
            melspec = librosa.feature.melspectrogram(audio[idx:idx+20480], n_mels=bands)
            logspec = librosa.core.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
    
            
            #segments.append(melspec[:, idx:idx+frames])
            segment_labels.append(labels[num])
        
    # Check and ignore silent segments
    '''
    for i, segment in enumerate(tqdm(segments)):
        
        # Append only non silent segments and convert into db
        if(np.mean(segment) >= threshold):
            augmented_spec.append(segment)
            new_labels.append(segment_labels[i])
    '''       
            
    
    #augmented_spec = np.asarray(augmented_spec)
    #logspec = librosa.core.amplitude_to_db(augmented_spec)
    
    # Reshape the outputs
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    new_labels = np.asarray(new_labels, dtype=int)
    
    # Fill the delta features
    for i in range(len(log_specgrams)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    features = features.astype(np.float32)
    labels = labels.astype(np.int)
    
    return features, new_labels



def CreateTrainingSet(f1, f2, f3, lf1, lf2, lf3, batch_size=32):
    
    # Create training set
    merged_training_data = np.concatenate((f1, f2, f3))
    merged_training_label = np.concatenate((lf1, lf2, lf3))
    
    # Shuffle the folds
    rnd_indices = np.arange(0, len(merged_training_data))
    rnd_indices = np.random.shuffle(rnd_indices)
    
    merged_training_data = merged_training_data[rnd_indices].reshape((len(f1) + len(f2) + len(f3), 60, 41, 2))
    merged_training_label = merged_training_label[rnd_indices].reshape((len(f1) + len(f2) + len(f3), 50))
    
    
    merged_training_data = merged_training_data.astype(np.float32)
    merged_training_label = merged_training_label.astype(np.float32)

    # Create dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((merged_training_data, merged_training_label))
    
    # Cache the dataset
    training_dataset = training_dataset.cache("training_cache")
    
    # Shuffle all elements at every iteration
    training_dataset = training_dataset.shuffle(len(training_dataset))
    
    # Define batch_size and prefetch size
    training_dataset = training_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    return training_dataset


def CreateTrainingSet_Single(f1, lf1, batch_size=32):
    
    f1 = f1.astype(dtype=np.float32)
    lf1 = lf1.astype(dtype=np.float32)
    
    # Create and cache training
    training_dataset = tf.data.Dataset.from_tensor_slices((f1, lf1))
    
    # Cache dataset
    training_dataset = training_dataset.cache("training_cache_single")
    
    # Shuffle all elements at every iteration
    training_dataset = training_dataset.shuffle(len(training_dataset))
    
    # Define batch_size and prefetch size
    training_dataset = training_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    return training_dataset


def CreateValidationSet(f1, lf1, batch_size=32):
    
    f1 = f1.astype(dtype=np.float32)
    lf1 = lf1.astype(dtype=np.float32)
    
    # Create and cache training
    validation_dataset = tf.data.Dataset.from_tensor_slices((f1, lf1))
    
    # Cache dataset
    validation_dataset = validation_dataset.cache("validation_cache")
    
    # Shuffle all elements at every iteration
    #validation_dataset = validation_dataset.shuffle(len(validation_dataset))
    
    # Define batch_size and prefetch size
    validation_dataset = validation_dataset.batch(batch_size=batch_size).prefetch(buffer_size=1)
    
    return validation_dataset




######################
### NETWORK MODELS ###
######################


def PiczakNet(input_shape):
    
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
    model = tf.keras.Model(inputs = X_input, outputs = model, name='PiczakNet')
    
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