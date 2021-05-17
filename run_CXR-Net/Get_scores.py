#!/usr/bin/env python3
# coding: utf-8

# ### Predict lung masks and Covid vs non-Covid classification for new patient CXR using Module 1 trained on the V7 lung segmentation database, and Module 2 trained on the HFHS dataset   
# 
# usage:  python Get_scores.py --dirs new_patient_cxr image_dcm 
#                                  image_resized_equalized_from_dcm mask_binary 
#                                  mask_float image_mask H5 grad_cam 
#                                  

# In[1]:

import os, sys, shutil, getopt
import argparse
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import cv2
import pandas as pd
import json
import datetime
import csv, h5py
import pydicom
from pydicom.data import get_testdata_files

# In[2]:

import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, model_from_json, load_model, clone_model 
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import datetime

# In[3]:
# Initialize the Parser
parser = argparse.ArgumentParser(description ='Sources and targets.') 
   
parser.add_argument('--dirs', metavar = 'D', dest = 'dirs',
                    type = str, nargs = 8,
                    help ='directories')

parser.add_argument('--imgs', metavar = 'I', dest = 'imgs',
                    type = str, nargs = '+',
                    help ='img names')

args = parser.parse_args()

# In[4]

# ### SOURCE and TARGET DIRECTORIES. These are derived from the command line arguments.

# Root directory
root_dir = args.dirs[0] # 'new_patient_cxr/

# Source directory containing COVID patients lung CXR's
dcm_source_img_path = os.path.join(root_dir, args.dirs[1]) # 'new_patient_cxr/image_dcm/'

# Source/target directory containing COVID patients lung DCM CXR's converted to PNG
source_resized_img_path = os.path.join(root_dir, args.dirs[2]) # 'new_patient_cxr/image_resized_equalized_from_dcm/'

# Target directories for predicted masks
target_resized_msk_path_binary = os.path.join(root_dir, args.dirs[3]) # 'new_patient_cxr/mask_binary/'
target_resized_msk_path_float = os.path.join(root_dir, args.dirs[4]) # 'new_patient_cxr/mask_float/'
target_img_mask_path = os.path.join(root_dir, args.dirs[5]) # 'new_patient_cxr/image_mask/'

# Target directory for H5 file containing both images and masks
h5_img_dir = os.path.join(root_dir, args.dirs[6])

# Target directory for gradcams
grad_cam_dir = os.path.join(root_dir, args.dirs[7]) # 'new_patient_cxr/grad_cam/'

# In[5]:

# PREDICTION and HEAT MAP

from MODULES_2.Generators import get_generator, DataGenerator
from MODULES_2.Networks import WaveletScatteringTransform, ResNet 
from MODULES_2.Networks import SelectChannel, TransposeChannel, ScaleByInput, Threshold 
from MODULES_2.Losses import other_metrics_binary_class
from MODULES_2.Constants import _Params, _Paths
from MODULES_2.Utils import get_class_threshold, standardize, commonelem_set
from MODULES_2.Utils import _HEAT_MAP_DIFF
from MODULES_2.Utils import get_roc_curve, compute_gradcam, get_roc_curve_sequence, plot_confusion_matrix
from MODULES_2.Utils import get_mean_roc_curve_sequence, get_multi_roc_curve_sequence

from tensorflow.keras.layers import Input, Average, Lambda, Multiply, Add, GlobalAveragePooling2D, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.compat.v1.logging import INFO, set_verbosity


# In[6]:

# ### SOURCE AND TARGET DIRECTORIES

# Source directories for images and masks
IMAGE_DIR = source_resized_img_path
MASK_DIR = target_resized_msk_path_float

# # Target directory for H5 file containing both images and masks
H5_IMAGE_DIR = h5_img_dir
   
# In[7]:   
    
# ### Read in json dictionary for the new patient names
with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle:
    new_patient_h5_dict = json.load(filehandle)     

# In[8]:

# ### MODEL AND RUN SELECTION
HEIGHT,WIDTH,CHANNELS,IMG_COLOR_MODE,MSK_COLOR_MODE,NUM_CLASS,KS1,KS2,KS3,\
DL1,DL2,DL3,NF,NFL,NR1,NR2,DIL_MODE,W_MODE,LS,SHIFT_LIMIT,SCALE_LIMIT,\
ROTATE_LIMIT,ASPECT_LIMIT,U_AUG,TRAIN_SIZE,VAL_SIZE,DR1,DR2,CLASSES,\
IMG_CLASS,MSK_FLOAT,MSK_THRESHOLD,MRA,MRALEVEL,MRACHANNELS,WAVELET,\
WAVEMODE,WST,WST_J,WST_L,WST_FIRST_IMG,SCALE_BY_INPUT,SCALE_THRESHOLD = _Params() 
    
TRAIN_IMG_PATH,TRAIN_MSK_PATH,TRAIN_MSK_CLASS,VAL_IMG_PATH,VAL_MSK_PATH,\
VAL_MSK_CLASS = _Paths()

# In[9]:

# ### Additional or modified network or fit parameters

NEW_RUN = False
NEW_MODEL_NUMBER = False

UPSAMPLE = False
UPSAMPLE_KERNEL = (2,2)

KS1=(3, 3)
KS2=(3, 3)
KS3=(3, 3)

WSTCHANNELS = 50

RESNET_DIM_1 = 75
RESNET_DIM_2 = 85

SCALE_BY_INPUT = False
SCALE_THRESHOLD = 0.6
SCALE_TO_SPAN = False
SPAN = 1.0

ATT = 'mh'
HEAD_SIZE = 64
NUM_HEAD = 2 
VALUE_ATT = True

BLUR_ATT = False
BLUR_ATT_STD = 0.1
BLUR_SBI = False
BLUR_SBI_STD = 0.1

NR1 = 2

PREP = True
STEM = True

KFOLD = 'Simple' # 'Simple','Strati','Group'

VAL_SIZE = 15

OPTIMIZER = Adam(learning_rate=0.002,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True)

# In[10]:

model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'

if NEW_MODEL_NUMBER:
    model_number = str(datetime.datetime.now())[0:10] + '_' + \
    str(datetime.datetime.now())[11:13] + '_' + \
    str(datetime.datetime.now())[14:16]
else:
    model_number = '2021-02-16_11_28'

# In[11]:

# ### ENSEMBLE MODEL
K.clear_session()

if SCALE_BY_INPUT:
    loi = 'multiply_2'
else:
    loi = 'multiply_1'

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  
    # MODELS

    wst_model = WaveletScatteringTransform(input_shape=(HEIGHT, WIDTH, CHANNELS),
                                            upsample=UPSAMPLE,
                                            upsample_kernel=UPSAMPLE_KERNEL)
    
    # wst_model.save('models/wst_model')
    
    # Alternatively, load the saved wst_model.
    # wst_model = load_model('models/wst_model')    

    resnet_model = ResNet(input_shape_1=(RESNET_DIM_1, RESNET_DIM_2, WSTCHANNELS),
                            input_shape_2=(RESNET_DIM_1, RESNET_DIM_2, 1),
                            num_class=NUM_CLASS,
                            ks1=KS1, ks2=KS2, ks3=KS3, 
                            dl1=DL1, dl2=DL2, dl3=DL3,
                            filters=NF,resblock1=NR1,
                            r_filters=NFL, resblock2=NR2,
                            dil_mode=DIL_MODE, 
                            sp_dropout=DR1,re_dropout=DR2,
                            prep=PREP,
                            stem=STEM,
                            mask_float=MSK_FLOAT,
                            mask_threshold=MSK_THRESHOLD,
                            att=ATT,
                            head_size=HEAD_SIZE,
                            num_heads=NUM_HEAD,
                            value_att=VALUE_ATT,
                            scale_by_input=SCALE_BY_INPUT,
                            scale_threshold=SCALE_THRESHOLD,
                            scale_to_span=SCALE_TO_SPAN,
                            span=SPAN,                          
                            blur_sbi=BLUR_SBI,
                            blur_sbi_std=BLUR_SBI_STD,                                                                 
                            return_seq=True)

    # recover individual resnet models
    resnet_model_0 = clone_model(resnet_model)
    resnet_model_0.load_weights('models/' + model_selection + '_' + model_number + '_M0' + '_resnet_weights.h5')

    
    for layer in resnet_model_0.layers:
        layer.trainable = False                
    resnet_model__0 = Model(inputs=[resnet_model_0.inputs], 
                            outputs=[resnet_model_0.get_layer(loi).output])  

    
    resnet_model_1 = clone_model(resnet_model)
    resnet_model_1.load_weights('models/' + model_selection + '_' + model_number + '_M1' + '_resnet_weights.h5')
    
    for layer in resnet_model_1.layers:
        layer.trainable = False                
    resnet_model__1 = Model(inputs=[resnet_model_1.inputs], 
                            outputs=[resnet_model_1.get_layer(loi).output]) 

    
    resnet_model_2 = clone_model(resnet_model)
    resnet_model_2.load_weights('models/' + model_selection + '_' + model_number + '_M2' + '_resnet_weights.h5')
    
    for layer in resnet_model_2.layers:
        layer.trainable = False                
    resnet_model__2 = Model(inputs=[resnet_model_2.inputs], 
                            outputs=[resnet_model_2.get_layer(loi).output])

    
    resnet_model_3 = clone_model(resnet_model)
    resnet_model_3.load_weights('models/' + model_selection + '_' + model_number + '_M3' + '_resnet_weights.h5')
    
    for layer in resnet_model_3.layers:
        layer.trainable = False                
    resnet_model__3 = Model(inputs=[resnet_model_3.inputs], 
                            outputs=[resnet_model_3.get_layer(loi).output]) 

    
    resnet_model_4 = clone_model(resnet_model)
    resnet_model_4.load_weights('models/' + model_selection + '_' + model_number + '_M4' + '_resnet_weights.h5')
    
    for layer in resnet_model_4.layers:
        layer.trainable = False                
    resnet_model__4 = Model(inputs=[resnet_model_4.inputs], 
                            outputs=[resnet_model_4.get_layer(loi).output])     

    
    resnet_model_5 = clone_model(resnet_model)
    resnet_model_5.load_weights('models/' + model_selection + '_' + model_number + '_M5' + '_resnet_weights.h5')
    
    for layer in resnet_model_5.layers:
        layer.trainable = False                
    resnet_model__5 = Model(inputs=[resnet_model_5.inputs], 
                            outputs=[resnet_model_5.get_layer(loi).output]) 
    

    # GRAPH 1
    
    wst_input_1 = Input(shape=(HEIGHT, WIDTH, CHANNELS))
    wst_input_2 = Input(shape=(HEIGHT, WIDTH, CHANNELS)) 
    
    wst_output_1 = wst_model([wst_input_1,wst_input_2])    
    
    y0 = resnet_model__0(wst_output_1)
    y1 = resnet_model__1(wst_output_1)    
    y2 = resnet_model__2(wst_output_1)
    y3 = resnet_model__3(wst_output_1)
    y4 = resnet_model__4(wst_output_1)
    y5 = resnet_model__5(wst_output_1)    
    
    d3 = Average()([y0,y1,y2,y3,y4,y5])     
    
    d3 = GlobalAveragePooling2D()(d3)
    
    resnet_output = Activation("softmax", name = 'softmax')(d3)     
    

    ensemble_model = Model([wst_input_1,wst_input_2], resnet_output,name='ensemble_wst_resnet')

    # Save ensemble model in TF2 ModelSave format
    # ensemble_model.save('models/ensemble_model')    
    # ensemble_model = load_model('models/ensemble_model')   
    # Save ensemble model in json format
    # config = ensemble_model.to_json()    

    ensemble_model.compile(optimizer=Adam(), 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy()]) 

    # Save ensemble model in TF2 ModelSave format
    # ensemble_model.save('models/ensemble_model')
    # Save ensemble model in h5 format    
    # ensemble_model.save('models/ensemble_model.h5')
 

# In[12]:

# ### GENERATOR for 1 IMAGE at a time

datadir = os.path.join(H5_IMAGE_DIR, '')

dataset = new_patient_h5_dict
                                      
valid_1_generator = DataGenerator(dataset["new_patient"], datadir, augment=False, shuffle=False, standard=False,                                      batch_size=1, dim=(HEIGHT, WIDTH, MRACHANNELS), mask_dim=(HEIGHT, WIDTH, 1),                                       mlDWT=False, mralevel=MRALEVEL, wave=WAVELET, wavemode=WAVEMODE, verbose=0)

# In[13]:

# ### PREDICT

valid_y_pred = []

for i in range(len(dataset["new_patient"])):
    x_m, y, w = valid_1_generator.__getitem__(i)
    y_pred = ensemble_model(x_m).numpy().tolist()
    valid_y_pred.append(y_pred[0])
    
# save predictions as dictionary
new_patient_pred_dict = {"new_patient_scores":valid_y_pred}
with open(os.path.join(H5_IMAGE_DIR, 'new_patient_scores.json'), 'w') as filehandle:
    json.dump(new_patient_pred_dict, filehandle)     
    
valid_y_pred = np.array(valid_y_pred)
    
# In[14]:
patient_no = 0
for idx,patient in enumerate(new_patient_h5_dict["new_patient"]):
    print(f'{patient[:-3]} scores: {valid_y_pred[idx]}')

new_patient_list = np.array(dataset["new_patient"]).tolist()


# In[15]:

# ### HEAT MAPS

print('Calculating heat maps')


OUT_IMAGE_DIR = os.path.join(grad_cam_dir,'')


FIG_SIZE = (16,20)

_HEAT_MAP_DIFF(ensemble_model,
               generator=valid_1_generator,
               layer='average',          
               labels=['Positive score','Negative score'],
               header='',
               figsize=FIG_SIZE,          
               image_dir=H5_IMAGE_DIR,out_image_dir=OUT_IMAGE_DIR,          
               img_list=new_patient_list,first_img=0,last_img=len(new_patient_list),          
               img_width=WIDTH,img_height=HEIGHT,display=True)

# In[]:
    
# def main():

# if __name__ == '__main__':
#     main()
    





