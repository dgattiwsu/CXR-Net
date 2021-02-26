#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 07:55:50 2020

@author: dgatti
"""

# ### CONSTANTS

def _Params():

    # Image features
    # HEIGHT = 262
    # WIDTH = 400
    ### 3 GPUs ####
    # HEIGHT = 437
    # WIDTH = 667
    ###############
    HEIGHT = 300
    WIDTH = 340
    
    # IMG_COLOR_MODE ='rgb'
    IMG_COLOR_MODE = 'grayscale'
    
    if IMG_COLOR_MODE == 'rgb':
        CHANNELS = 3
    elif IMG_COLOR_MODE == 'grayscale':
        CHANNELS = 1
    
    IMG_CLASS = 'img_gs'
    
    # Mask features
    # MSK_COLOR_MODE = 'rgb'
    # MSK_COLOR_MODE = 'rgba'
    MSK_COLOR_MODE = 'grayscale'
    NUM_CLASS = 2
    CLASSES = ['msk0','msk1','msk2','msk3']
    
    # Kernels 
    KS1 = (3,3) # default (3,3)
    KS2 = (3,3) # default (5,5)
    KS3 = (3,3) # default (7,7)
    # KS1 = (3,3) 
    # KS2 = (7,7)
    # KS3 = (11,11)    
    DL1 = (1,1) # default (1,1)
    DL2 = (2,2) # default (3,3)
    DL3 = (3,3) # default (5,5)
    
    # Convolutional residual blocks
    # NR1 = 8
    NR1 = 1 # Default 5
    # Convolutional block Filters
    # NF = 24
    # NF = 16 Default 16
    NF = 17
    
    
    # LSTM residual blocks
    # NR2 = 1    
    NR2 = 0    
    # LSTM block Filters
    NFL = 2
    # NFL = 6    
   
    # Dropout rates: 
    # DR1, conv blocks (recommended 0.05) 
    # DR2, LSTM blocks (recommended 0.1)
    DR1 = 0.05
    # DR1 = 0.1
    DR2 = 0.1
    
    # Residual block mode: add or concatenate different dilations
    DIL_MODE = "conc"
    # DIL_MODE = "add"
    
    # Weight mode: contour (recommended) or volume
    W_MODE = "contour"
    # W_MODE = "both"
    # W_MODE = "volume"
    
    # Loss_smoothing (recommended values: 1.0 for 'contour' mode, 1e-5 for 
    # 'volume' mode)
    if W_MODE == "contour":
        LS = 1
    elif W_MODE == "both":
        LS = 1        
    elif W_MODE == "volume":
        LS = 1e-5
    
    # Generators parameters
    SHIFT_LIMIT = (-0.1, 0.1) # default (-0.05, 0.05)
    SCALE_LIMIT = (-0.1, 0.1) # default (-0.05, 0.05)
    ROTATE_LIMIT = (-15, 15) # default (-5, 5) 
    ASPECT_LIMIT = (0.1, 0.1) # default (-0.05, 0.05)
    U_AUG = (0.7, 0.3, 0.2, 0.2) # Probabilities for Rot+Shift+Scale, Hor flip, Vert flip, Hor+Vert flip: 
                                 # default (0.6, 0.3, 0.3, 0.3)
    
    # Batch size for training and validation set
    TRAIN_SIZE = 15
    VAL_SIZE = 10
    
    # multilevel Discrete Wavelet Transform (DWT)
    MRA = False
    MRALEVEL = 2
    
    if MRALEVEL == 1:
        MRACHANNELS = 5
    elif MRALEVEL == 2:
        MRACHANNELS = 8
    elif MRALEVEL == 3:
        MRACHANNELS = 11
        
    WAVELET = 'db2'
    WAVEMODE = 'symmetric'
    
    # multilevel Wavelet Scattering Transform (WST)
    WST = True
    WST_J = 2 # default 2
    WST_L = 6 # default 8
    WST_FIRST_IMG = True # Default True if attention on the first image
    
    if WST:
        MRA = False
        if WST_J > 0:
            # WST as a convnet layer
            MRACHANNELS = 1
        else:
            # WST preprocessed
            MRACHANNELS = 81 # 81 for J=2,L=8 ; 49 for J2,L=6
            
    MSK_FLOAT = False
    MSK_THRESHOLD = 0.3
    SCALE_BY_INPUT = True
    SCALE_THRESHOLD = 0.6
    
    return HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
        MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
        SCALE_BY_INPUT, SCALE_THRESHOLD

def _Paths():
    TRAIN_IMG_PATH = "dataset/COVID_standardized_pos4_neg5_image_expand_float_6395_1_threshold_H5_CLASSWEIGHT/"
    TRAIN_MSK_PATH = ''
    VAL_IMG_PATH = "dataset/COVID_standardized_pos4_neg5_image_expand_float_6395_1_threshold_H5_CLASSWEIGHT/"
    VAL_MSK_PATH = ''
    TRAIN_MSK_CLASS = ['']
    VAL_MSK_CLASS = ['']
    return TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS

def _Seeds():
    TRAIN_SEED = 1
    VAL_SEED = 2
    return TRAIN_SEED, VAL_SEED
    
