#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 07:48:40 2020

@author: dgatti
"""

# In[1]:

import numpy as np
import tensorflow as tf
from MODULES_2.Constants import _Params, _Paths, _Seeds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, Sequence

from pywt import wavedec2, coeffs_to_array
import json
import cv2
import h5py
from IPython.display import Image
from MODULES_2.Utils import standardize, normalize


# In[2]: 

# ### Generator functions for images and masks.

# CLASS_THRESHOLD      
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
    MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
    SCALE_BY_INPUT, SCALE_THRESHOLD = _Params()

step = 255/(NUM_CLASS-1)
class_threshold = []
for i in range(NUM_CLASS):
    jump = round(step*i)
    class_threshold.append(jump)     

# This function converts a thresholded categorical gray scale mask into a standard
# categorical mask with consecutive indices [0,1,2,3,...]
    
def to_train_indices(x,threshold=class_threshold):
    x = np.floor(x)
    unique_values = np.unique(x)
    delta = np.ceil(255/(2*NUM_CLASS-2))
    x_mod = np.ones_like(x)*NUM_CLASS
    for i, val in enumerate(threshold):
        ind = (x > (val-delta)) & (x <= (val+delta))
        x_mod[ind] = i
    assert(np.max(x_mod)<NUM_CLASS)
    return x_mod 

def to_val_indices(x):
    x = np.floor(x)
    unique_values = np.unique(x)
    x_mod = np.zeros_like(x)
    for i, val in enumerate(unique_values):
        ind = x == val
        x_mod[ind] = i
    return x_mod 

def to_one_hot_train(x,b,h,w,num_class):
    unique_values = np.unique(x)
    x_out = np.zeros((b,h,w,num_class))
    for i, val in enumerate(unique_values):
        x_mod = np.zeros_like(x)
        ind = x == val
        x_mod[ind] = 1
        x_out[:,:,:,i] = x_mod[:,:,:,0]
        
    # The following pooling is used to smooth out the ragged edges of 
    # the mask after the preprocessing operations (rotation, shift, etc.).
    x_out = tf.round(tf.keras.backend.pool2d(x_out, pool_size=(3,3), \
                            strides=(1,1), padding='same', pool_mode='avg'))    
    return x_out
    
def to_one_hot_val(x,b,h,w,num_class):
    unique_values = np.unique(x)
    x_out = np.zeros((b,h,w,num_class))
    for i, val in enumerate(unique_values):
        x_mod = np.zeros_like(x)
        ind = x == val
        x_mod[ind] = 1
        x_out[:,:,:,i] = x_mod[:,:,:,0]
        
    return x_out
                   
               
# In[3]:

# ### TRAINING SET            

def train_generator_1():
    
    # ### CONSTANTS       
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
        MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
        SCALE_BY_INPUT, SCALE_THRESHOLD = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
        
    TRAIN_SEED, VAL_SEED = _Seeds()
        
    train_data_gen_img_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,
                                 fill_mode='reflect')
                                 
    if MSK_COLOR_MODE == 'rgb':
        train_data_gen_msk_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,      
                                 fill_mode='reflect')
                                 
    elif MSK_COLOR_MODE == 'grayscale':
        train_data_gen_msk_args = dict(preprocessing_function=to_train_indices,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,      
                                 fill_mode='reflect')
    
    train_image_datagen = ImageDataGenerator(**train_data_gen_img_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_msk_args)
    
            
    train_image_generator = train_image_datagen.flow_from_directory(TRAIN_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                         
                                           class_mode=None,
                                           batch_size=TRAIN_SIZE,
                                           shuffle=False,                     
                                           seed=TRAIN_SEED)
    
    train_mask_generator = train_mask_datagen.flow_from_directory(TRAIN_MSK_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=CLASSES,
                                           color_mode=MSK_COLOR_MODE,
                                           class_mode=None,
                                           batch_size=TRAIN_SIZE,
                                           shuffle=False,                    
                                           seed=TRAIN_SEED)

    while True:
        if MSK_COLOR_MODE == 'rgb':
            yield(train_image_generator.next(), train_mask_generator.next())
        elif MSK_COLOR_MODE == 'grayscale':
            yield(train_image_generator.next(), to_one_hot_train(train_mask_generator.next(), \
                TRAIN_SIZE,HEIGHT,WIDTH,NUM_CLASS))

        
# In[4]:

# ### VALIDATION SET

def val_generator_1():

    # ### CONSTANTS     
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
        MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
        SCALE_BY_INPUT, SCALE_THRESHOLD = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
        
    TRAIN_SEED, VAL_SEED = _Seeds()    
    
    val_data_gen_img_args = dict(rescale=1./255)
    
    if MSK_COLOR_MODE == 'rgb':
        val_data_gen_msk_args = dict(rescale=1./255)
        
    elif MSK_COLOR_MODE == 'grayscale':
        val_data_gen_msk_args = dict(preprocessing_function=to_val_indices)
    
    val_image_datagen = ImageDataGenerator(**val_data_gen_img_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_msk_args)
    
    
    val_image_generator = val_image_datagen.flow_from_directory(VAL_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                     
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED)
    
    val_mask_generator = val_mask_datagen.flow_from_directory(VAL_MSK_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=CLASSES,
                                           color_mode=MSK_COLOR_MODE,
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED)

    while True:
        if MSK_COLOR_MODE == 'rgb':
            yield(val_image_generator.next(), val_mask_generator.next())
        elif MSK_COLOR_MODE == 'grayscale':
            yield(val_image_generator.next(), to_one_hot_val(val_mask_generator.next(), \
                VAL_SIZE,HEIGHT,WIDTH,NUM_CLASS))
                
# In[5]

# ### TRAINING SET for multiple masks            

def train_generator_2():
    
    global mask
    
    # ### CONSTANTS       
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
        MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
        SCALE_BY_INPUT, SCALE_THRESHOLD = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
    
    TRAIN_SEED, VAL_SEED = _Seeds()
        
    train_data_gen_img_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,
                                 fill_mode='reflect')
                                 

    train_data_gen_msk_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,      
                                 fill_mode='reflect')
                                 
    
    train_image_datagen = ImageDataGenerator(**train_data_gen_img_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_msk_args)
    
            
    train_image_generator = train_image_datagen.flow_from_directory(TRAIN_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                         
                                           class_mode=None,
                                           batch_size=TRAIN_SIZE,
                                           shuffle=False,                     
                                           seed=TRAIN_SEED,
                                           # save_to_dir='dataset/train_local/images/save',
                                           )

    for i in range(NUM_CLASS):
        globals()['train_mask_gen_{}'.format(i)] = train_mask_datagen.flow_from_directory(TRAIN_MSK_PATH,
                                            target_size=(HEIGHT,WIDTH),
                                            classes=[CLASSES[i]],
                                            color_mode=MSK_COLOR_MODE,
                                            class_mode=None,
                                            batch_size=TRAIN_SIZE,
                                            shuffle=False,                    
                                            seed=TRAIN_SEED,
                                            # save_to_dir='dataset/train_local/masks/save',
                                            )
        
    while True:
        yield(train_image_generator.next(), \
              np.round(np.squeeze(np.stack([globals()['train_mask_gen_{}'.format(i)].next() for i in range(NUM_CLASS)],axis=3))))        
             

# In[6]:

# ### VALIDATION SET for multiple masks

def val_generator_2():

    # ### CONSTANTS     
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
        MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
        SCALE_BY_INPUT, SCALE_THRESHOLD = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
    
    TRAIN_SEED, VAL_SEED = _Seeds()    
    
    val_data_gen_img_args = dict(rescale=1./255)
    val_data_gen_msk_args = dict(rescale=1./255)
            
    val_image_datagen = ImageDataGenerator(**val_data_gen_img_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_msk_args)
    
    
    val_image_generator = val_image_datagen.flow_from_directory(VAL_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                     
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED,
                                           # save_to_dir='dataset/val_local/images/save',
                                           )
    
    for i in range(NUM_CLASS):
        globals()['val_mask_gen_{}'.format(i)] = val_mask_datagen.flow_from_directory(VAL_MSK_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[CLASSES[i]],
                                           color_mode=MSK_COLOR_MODE,
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED,
                                           # save_to_dir='dataset/val_local/masks/save',
                                           )
    
    while True:
        yield(val_image_generator.next(), \
              np.round(np.squeeze(np.stack([globals()['val_mask_gen_{}'.format(i)].next() for i in range(NUM_CLASS)],axis=3))))
              

# In[7]:

# ### TRAIN GENERATOR FOR XRAY IMAGES

def get_train_generator(df, image_dir, x_col, y_cols, weight_col, 
                        shuffle=True,
                        augment=True,
                        batch_size=TRAIN_SIZE, 
                        seed=1, 
                        target_w = HEIGHT, 
                        target_h = WIDTH):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...")
    if augment:
        # normalize images
        image_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization= True,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=5,
            zoom_range=0.05,
            fill_mode='constant',
            cval=0)
    else:
        image_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization= True)            
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            weight_col=weight_col,
            color_mode="grayscale",
            class_mode="raw",
            # class_mode="binary",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator

# In[8]:
    
# VALID AND TEST GENERATOR FOR XRAY IMAGES

def get_test_and_valid_generator(valid_df, 
                                 test_df, 
                                 train_df, 
                                 image_dir, 
                                 x_col, y_cols, weight_col, 
                                 sample_size=100, 
                                 batch_size=VAL_SIZE, 
                                 seed=1, 
                                 target_w = HEIGHT, 
                                 target_h = WIDTH):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir, 
        x_col=x_col, 
        y_col=y_cols,
        weight_col=weight_col,
        color_mode="grayscale",        
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            weight_col=weight_col,
            color_mode="grayscale",             
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            weight_col=weight_col,
            color_mode="grayscale",             
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    
    return valid_generator, test_generator

# In[8]:

# ### TRAIN GENERATOR FOR XRAY IMAGES WITH/WITHOUT STANDARDIZATION

def get_generator(df,image_dir, 
                  x_col,y_cols,weight_col, 
                  shuffle=True,
                  augment=True,
                  batch_size=TRAIN_SIZE, 
                  seed=1, 
                  target_w = HEIGHT, 
                  target_h = WIDTH):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...")
    if augment:
        image_generator = ImageDataGenerator(samplewise_center=True,
                                             samplewise_std_normalization= True,
                                            # rescale=1./255,
                                             rotation_range=5,
                                             width_shift_range=0.05,
                                             height_shift_range=0.05,
                                             shear_range=5,
                                             zoom_range=0.05,
                                             horizontal_flip=True,
                                             vertical_flip=True,
#                                              brightness_range=[0.9,1.1],                                           
#                                              fill_mode='constant',
#                                              cval=0,
                                             fill_mode='reflect')
    else:
        # image_generator = ImageDataGenerator(rescale=1./255)
        image_generator = ImageDataGenerator(samplewise_center=True,
                                             samplewise_std_normalization= True)        
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(dataframe=df,
                                                    directory=image_dir,
                                                    x_col=x_col,
                                                    y_col=y_cols,
                                                    weight_col=weight_col,
                                                    color_mode="grayscale",
                                                    class_mode="raw",
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    seed=seed,
                                                    target_size=(target_w,target_h))
    
    return generator


def mlDWT(x,wave,wavemode,mralevel):
#     print(f'x_shape =  {x.shape}')
    x = x[:,:,0]
#     print(f'x_shape =  {x.shape}')
    dim = (x.shape[1],x.shape[0])
    
    coeffs = wavedec2(x, wave, mode=wavemode, level=mralevel)
    
    if mralevel == 3:
    
        # Level 3
        cA3 = cv2.resize(coeffs[0], dim, interpolation=cv2.INTER_LINEAR)
        cH3 = cv2.resize(coeffs[1][0], dim, interpolation=cv2.INTER_LINEAR)
        cV3 = cv2.resize(coeffs[1][1], dim, interpolation=cv2.INTER_LINEAR)
        cD3 = cv2.resize(coeffs[1][2], dim, interpolation=cv2.INTER_LINEAR)

        # Level 2
        cH2 = cv2.resize(coeffs[2][0], dim, interpolation=cv2.INTER_LINEAR)
        cV2 = cv2.resize(coeffs[2][1], dim, interpolation=cv2.INTER_LINEAR)
        cD2 = cv2.resize(coeffs[2][2], dim, interpolation=cv2.INTER_LINEAR)

        # Level 1
        cH1 = cv2.resize(coeffs[3][0], dim, interpolation=cv2.INTER_LINEAR)
        cV1 = cv2.resize(coeffs[3][1], dim, interpolation=cv2.INTER_LINEAR)
        cD1 = cv2.resize(coeffs[3][2], dim, interpolation=cv2.INTER_LINEAR)

        nchannels = 11
        w = np.zeros((dim[1],dim[0],nchannels))

        w[:,:,0] = x    
        w[:,:,1] = cA3
        w[:,:,2] = cH3
        w[:,:,3] = cV3
        w[:,:,4] = cD3
        w[:,:,5] = cH2
        w[:,:,6] = cV2
        w[:,:,7] = cD2
        w[:,:,8] = cH1
        w[:,:,9] = cV1
        w[:,:,10] = cD1
    
    elif mralevel == 2:
    
        # Level 2
        cA2 = cv2.resize(coeffs[0], dim, interpolation=cv2.INTER_LINEAR)
        cH2 = cv2.resize(coeffs[1][0], dim, interpolation=cv2.INTER_LINEAR)
        cV2 = cv2.resize(coeffs[1][1], dim, interpolation=cv2.INTER_LINEAR)
        cD2 = cv2.resize(coeffs[1][2], dim, interpolation=cv2.INTER_LINEAR)

        # Level 1
        cH1 = cv2.resize(coeffs[2][0], dim, interpolation=cv2.INTER_LINEAR)
        cV1 = cv2.resize(coeffs[2][1], dim, interpolation=cv2.INTER_LINEAR)
        cD1 = cv2.resize(coeffs[2][2], dim, interpolation=cv2.INTER_LINEAR)

        nchannels = 8
        w = np.zeros((dim[1],dim[0],nchannels))

        w[:,:,0] = x    
        w[:,:,1] = cA2
        w[:,:,2] = cH2
        w[:,:,3] = cV2
        w[:,:,4] = cD2
        w[:,:,5] = cH1
        w[:,:,6] = cV1
        w[:,:,7] = cD1    
    
    elif mralevel == 1:
    
        # Level 1
        cA1 = cv2.resize(coeffs[0], dim, interpolation=cv2.INTER_LINEAR)
        cH1 = cv2.resize(coeffs[1][0], dim, interpolation=cv2.INTER_LINEAR)
        cV1 = cv2.resize(coeffs[1][1], dim, interpolation=cv2.INTER_LINEAR)
        cD1 = cv2.resize(coeffs[1][2], dim, interpolation=cv2.INTER_LINEAR)

        nchannels = 5
        w = np.zeros((dim[1],dim[0],nchannels))

        w[:,:,0] = x    
        w[:,:,1] = cA1
        w[:,:,2] = cH1
        w[:,:,3] = cV1
        w[:,:,4] = cD1      

    return w

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.05, 0.05),
                           scale_limit=(-0.05, 0.05),
                           rotate_limit=(-5, 5), 
                           aspect_limit=(0.05, 0.05),
                           u=0.6,
                           borderMode=cv2.BORDER_REFLECT_101):
    n_aug_1 = 0
    if np.random.random() < u:
        n_aug_1 += 1
        height, width, channels = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
                        
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))                
        
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)                        

    return image, mask, n_aug_1

def randomHorizontalFlip(image, mask, u=0.3):
    n_aug_2 = 0
    if np.random.random() < u:        
        n_aug_2 += 1    
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask, n_aug_2

def randomVerticalFlip(image, mask, u=0.3):
    n_aug_3 = 0    
    if np.random.random() < u:        
        n_aug_3 += 1    
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask, n_aug_3

def randomDoubleFlip(image, mask, u=0.3):
    n_aug_4 = 0    
    if np.random.random() < u:        
        n_aug_4 += 1    
        image = cv2.flip(image, -1)
        mask = cv2.flip(mask, -1)

    return image, mask, n_aug_4
   

class DataGenerator(Sequence): 
        
    def __init__(self,
                 sample_list,
                 base_dir,
                 batch_size=1,
                 shuffle=True,
                 dim=(HEIGHT, WIDTH),
                 mask_dim=(HEIGHT, WIDTH),
                 num_channels=CHANNELS,
                 num_classes=NUM_CLASS,
                 num_outputs=1,
                 augment=True,
                 shift_limit=SHIFT_LIMIT,
                 scale_limit=SCALE_LIMIT,
                 rotate_limit=ROTATE_LIMIT,
                 aspect_limit=ASPECT_LIMIT,
                 u_aug=U_AUG,
                 standard=True,
                 mlDWT=False,
                 wave='haar',
                 wavemode='reflect',
                 mralevel=3,
                 verbose=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.mask_dim = mask_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.verbose = verbose
        self.sample_list = sample_list
        self.augment = augment
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit
        self.u_aug = u_aug
        self.standard = standard
        self.mlDWT = mlDWT
        self.wave = wave
        self.wavemode = wavemode
        self.mralevel = mralevel
        self.on_epoch_end()        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sample_list) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        sample_list_temp = [self.sample_list[k] for k in indexes]
           
        # Generate data
        x, m, y, w = self.__data_generation(sample_list_temp) 
 
        # for single output
        if self.num_outputs == 1:
            return [x, m], y, w    
        # for dual output
        elif self.num_outputs == 2:
            return [x, m], [y, y], [w, w]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)            

    def __data_generation(self, sample_list_temp):
        'Generates data containing batch_size samples' 

        X_batch = np.zeros((self.batch_size, *self.dim),
                           dtype=np.float32)
        M_batch = np.zeros((self.batch_size, *self.mask_dim),
                           dtype=np.float32)
        y_batch = np.zeros((self.batch_size, 2),
                           dtype=np.float32)
        w_batch = np.zeros(self.batch_size,
                           dtype=np.float32)            
        
        for i, ID in enumerate(sample_list_temp):
            
            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
    
            with h5py.File(self.base_dir + ID, 'r') as f:
                X_slice = np.array(f.get("X"))
                M_slice = np.array(f.get("M"))
                y_label = np.array(f.get("y"))
                w_weight = np.array(f.get("w"))                
                
            x = X_slice + 0
            
            # mlDWT
            if self.mlDWT == True:
                x = mlDWT(x,wave=self.wave,wavemode=self.wavemode,mralevel=self.mralevel)              
                # print(f'x_shape =  {x.shape}')
            
            m = M_slice + 0                 

            if self.augment == True:
                x, m, n_aug_1 = randomShiftScaleRotate(x, m,
                                              shift_limit=self.shift_limit,
                                              scale_limit=self.scale_limit,
                                              rotate_limit=self.rotate_limit,
                                              aspect_limit=self.aspect_limit,
                                              u=self.u_aug[0])                     

                x, m, n_aug_2 = randomHorizontalFlip(x, m, u=self.u_aug[1])

                x, m, n_aug_3 = randomVerticalFlip(x, m, u=self.u_aug[2])

                n_aug = n_aug_1+n_aug_2+n_aug_3
                
                n_aug_4 = 0
                if n_aug == 0:
                    x, m, n_aug_4 = randomDoubleFlip(x, m, u=self.u_aug[3])
                    n_aug = n_aug_1+n_aug_2+n_aug_3+n_aug_4                   

                # print(f"n_aug = {n_aug_1}_{n_aug_2}_{n_aug_3}_{n_aug_4}") 
            
                x = x.astype(np.float32)
                m = m.astype(np.float32)

            if self.augment == False:
                x = x.astype(np.float32)
                m = m.astype(np.float32)
                
            if len(x.shape) == 2:
                x = np.expand_dims(x,axis=-1)            
            if len(m.shape) == 2:
                m = np.expand_dims(m,axis=-1)
                
            # print(x.shape, m.shape)
            
            X_batch[i] = x
            M_batch[i] = m
            y_batch[i] = y_label
            w_batch[i] = w_weight                
        
        return X_batch, M_batch, y_batch, w_batch

    
