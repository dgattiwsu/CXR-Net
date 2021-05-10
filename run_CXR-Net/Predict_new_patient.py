#!/usr/bin/env python
# coding: utf-8

# ### Predict lung masks and Covid vs non-Covid classification for new patient CXR using Module 1 trained on the V7 lung segmentation database, and Module 2 trained on the HFHS dataset   
# 
# usage:  python Predict_new_patient.py --dirs new_patient_cxr image_dcm 
#                                  image_resized_equalized_from_dcm mask_binary 
#                                  mask_float image_mask H5 grad_cam 
#                                  [--imgs name1.dcm name2.dcm ... nameN.dcm]

# Choice of directories is mandatory. If no iamges are selected using the --imgs 
# flag, all dcm images in the source directory are processed. 

# In[1]:


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

from MODULES_1.Generators import train_generator_1, val_generator_1, test_generator_1
from MODULES_1.Generators import train_generator_2, val_generator_2, test_generator_2
from MODULES_1.Networks import ResNet_Atrous, Dense_ResNet_Atrous
from MODULES_1.Losses import dice_coeff
from MODULES_1.Losses import tani_loss, tani_coeff, weighted_tani_coeff
from MODULES_1.Losses import weighted_tani_loss, other_metrics
from MODULES_1.Constants import _Params, _Paths
from MODULES_1.Utils import get_class_threshold, get_model_memory_usage
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

# In[3]:

# ### CONSTANTS

HEIGHT,WIDTH,CHANNELS,IMG_COLOR_MODE,MSK_COLOR_MODE,NUM_CLASS,KS1,KS2,KS3,DL1,DL2,DL3,NF,NFL,NR1,NR2,DIL_MODE,W_MODE,LS,TRAIN_SIZE,VAL_SIZE,TEST_SIZE,DR1,DR2,CLASSES,IMG_CLASS = _Params()

TRAIN_IMG_PATH,TRAIN_MSK_PATH,TRAIN_MSK_CLASS,VAL_IMG_PATH,VAL_MSK_PATH,VAL_MSK_CLASS,TEST_IMG_PATH,TEST_MSK_PATH,TEST_MSK_CLASS = _Paths()

# In[4]: 

# ### LOAD LUNG SEGMENTATION MODEL FROM PREVIOUS RUN AND COMPILE

model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'
model_number = '2020-10-16_21_26' # model number from an earlier run
filepath = 'models/' + model_selection + '_' + model_number + '_all' + '.h5'

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = load_model(filepath, compile=False)     
    model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff]) 

print(f'Model selection: {model_selection}')
print(f'Model number: {model_number}')

# In[5]

# ### SOURCE and TARGET DIRECTORIES. These are derived from the command line arguments.

# print(CLASSES)

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

# Remove existing target directories and all their content if already present
pwd = os.getcwd()

if root_dir == pwd:
    for root, dirs, files in os.walk(source_resized_img_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))    
    for root, dirs, files in os.walk(target_resized_msk_path_binary):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    for root, dirs, files in os.walk(target_resized_msk_path_float):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d)) 
    for root, dirs, files in os.walk(target_img_mask_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    for root, dirs, files in os.walk(h5_img_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    for root, dirs, files in os.walk(grad_cam_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))            

# Create directory that will store the DCM derived png CXR
if not os.path.exists(source_resized_img_path):
    os.makedirs(source_resized_img_path)            
            
# Create directories that will store the masks on which to train the classification network
if not os.path.exists(target_resized_msk_path_binary):
    os.makedirs(target_resized_msk_path_binary)
    
if not os.path.exists(target_resized_msk_path_float):
    os.makedirs(target_resized_msk_path_float) 
    
if not os.path.exists(target_img_mask_path):
    os.makedirs(target_img_mask_path)

# Create directory that will store the H5 files
if not os.path.exists(h5_img_dir):
    os.makedirs(h5_img_dir)

# Create directory that will store the heat maps
if not os.path.exists(grad_cam_dir):
    os.makedirs(grad_cam_dir)
    
# In[7]:

# get CXR DCM image names from source directory and convert to png

PNG = True
print(f'DCM source: {dcm_source_img_path}')
print(f'PNG source: {source_resized_img_path}')

if args.imgs:
    print(f'Processing only selected imgs in the dcm directory')
    source_img_names = args.imgs
else:
    print(f'Processing all imgs in the dcm directory')
    source_img_names = [f for f in listdir(dcm_source_img_path) if isfile(join(dcm_source_img_path, f))]

for name in source_img_names:
           
    print(f'DCM image: {name}')   
    filename = os.path.join(dcm_source_img_path, name)
    dataset = pydicom.dcmread(filename)    
                        
    # Write PNG image    
    img = dataset.pixel_array.astype(float)    
    minval = np.min(img)
    maxval = np.max(img)
    scaled_img = (img - minval)/(maxval-minval) * 255.0
    
    WIDTH = 340
    HEIGHT = 300

    resized_img = cv2.resize(scaled_img, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
    resized_img_8bit = cv2.convertScaleAbs(resized_img, alpha=1.0)
    equalized_img = cv2.equalizeHist(resized_img_8bit)    
    
    if PNG == False:
        new_name = name.replace('.dcm', '.jpg')
    else:
        new_name = name.replace('.dcm', '.png')
        
    cv2.imwrite(os.path.join(source_resized_img_path, new_name), equalized_img) 
    
    print(f'PNG image: {new_name}')

# In[9]:
    
# if args.imgs:
#     print(f'Processing only selected imgs in the png directory')
#     source_img_names = args.imgs
# else:
#     print(f'Processing all imgs in the png directory')
#     source_img_names = [f for f in listdir(source_resized_img_path) if isfile(join(source_resized_img_path, f))]    

# get CXR image names from source directory
source_img_names = [f for f in listdir(source_resized_img_path) if isfile(join(source_resized_img_path, f))]

for name in source_img_names:
    # print(f'Image name: {name}')
    if name == '.DS_Store': 
        continue
    if name[:-4] + '.dcm' not in args.imgs:
        continue
    
    input_img = cv2.imread(os.path.join(source_resized_img_path, name), cv2.IMREAD_GRAYSCALE)
    scaled_img = input_img/255
    scaled_img = np.expand_dims(scaled_img,axis = [0,-1])
    mask = model(scaled_img).numpy()
    mask_float = np.squeeze(mask[0,:,:,0])    
    mask_binary = (mask_float > 0.5)*1
    
    mask_float *=255    
    mask_binary *=255
    cv2.imwrite(os.path.join(target_resized_msk_path_float, name), mask_float)
    cv2.imwrite(os.path.join(target_resized_msk_path_binary, name), mask_binary)
    
    fig = plt.figure(figsize=(20,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.squeeze(input_img), cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.squeeze(mask_binary), cmap="gray")       

    plt.savefig(os.path.join(target_img_mask_path, name + '_img_and_pred_mask.png')) 
    plt.close()
    

# In[10]:

# print(input_img.shape,mask.shape,mask_float.shape,mask_binary.shape)

# In[13]:

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

# In[10]:

# ### RECOVER STANDARDIZATION PARAMETERS FROM MODULE 1

with open('standardization_parameters_V7.json') as json_file:
    standardization_parameters = json.load(json_file) 
    
train_image_mean = standardization_parameters['mean']
train_image_std = standardization_parameters['std']

# print(train_image_mean, train_image_std)

# In[11]:

# ### PREPARE STANDARDIZED IMAGES as H5 FILES

# Source directories for images and masks
IMAGE_DIR = source_resized_img_path
MASK_DIR = target_resized_msk_path_float

# Target directory for H5 file containing both images and masks
H5_IMAGE_DIR = h5_img_dir

# print(IMAGE_DIR)
# print(H5_IMAGE_DIR)

pwd = os.getcwd()
if not os.path.isdir(H5_IMAGE_DIR):
    os.mkdir(H5_IMAGE_DIR)

    
# Loop over the set of images to predict

# get CXR image names from source directory
source_img_names = [f for f in listdir(source_resized_img_path) if isfile(join(source_resized_img_path, f))]

for valid_image_name in source_img_names:
    # print(f'Image name: {name}')
    if valid_image_name == '.DS_Store': 
        continue
    if valid_image_name[:-4] + '.dcm' not in args.imgs:
        continue    
    
    # Radiologist labels and class weights are not known
    valid_pos_label = 0
    valid_neg_label = 0
    valid_weight = 1.0
#     print(f'{i},index={new_patient_df.index[i]}')
#     valid_image_name, valid_pos_label, valid_neg_label, valid_weight =     new_patient_df.iloc[i]['Image'],    new_patient_df.iloc[i]['Positive'],    new_patient_df.iloc[i]['Negative'],    new_patient_df.iloc[i]['ClassWeight']

    valid_image = cv2.imread(os.path.join(IMAGE_DIR, valid_image_name), cv2.IMREAD_GRAYSCALE)
    
# Resize or equalize if this was not already done during datasets preparation    
    # valid_image = cv2.resize(valid_image, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
    # valid_image = cv2.equalizeHist(valid_image)

    valid_image = np.expand_dims(valid_image,axis=-1)
        
    # External learned mask of segmented lungs
    valid_learned_mask = cv2.imread(os.path.join(MASK_DIR, valid_image_name), cv2.IMREAD_GRAYSCALE).astype('float64')
    valid_learned_mask /= 255
    valid_learned_mask = np.expand_dims(valid_learned_mask,axis=-1)
    
    # Internal thresholded mask    
    low_ind = valid_image < 6
    high_ind = valid_image > 225    
    valid_thresholded_mask = np.ones_like(valid_image)
    valid_thresholded_mask[low_ind] = 0
    valid_thresholded_mask[high_ind] = 0

    # Combine the two masks
    valid_mask = np.multiply(valid_thresholded_mask,valid_learned_mask)
    
    # Standardization with training mean and std 
    valid_image = valid_image.astype(np.float64)
    valid_image -= train_image_mean
    valid_image /= train_image_std        
    
    with h5py.File(os.path.join(H5_IMAGE_DIR, valid_image_name[:-4] + '.h5'), 'w') as hf: 
        # Images
        Xset = hf.create_dataset(
            name='X',
            data=valid_image,
            shape=(HEIGHT, WIDTH, 1),
            maxshape=(HEIGHT, WIDTH, 1),
            compression="gzip",
            compression_opts=9)
        
        # Masks
        Mset = hf.create_dataset(
            name='M',
            data=valid_mask,
            shape=(HEIGHT, WIDTH, 1),
            maxshape=(HEIGHT, WIDTH, 1),
            compression="gzip",
            compression_opts=9)
        
        # Labels
        yset = hf.create_dataset(
            name='y',
            data=[valid_pos_label,valid_neg_label])
        
        # Class weights
        wset = hf.create_dataset(
            name='w',
            data=valid_weight)             

# In[12]:

# ### Generate json dictionary for the new patient names

new_patient_h5_name_list = []

for valid_image_name in source_img_names:
    if valid_image_name == '.DS_Store': 
        continue
    if valid_image_name[:-4] + '.dcm' not in args.imgs:
        continue    
# for i in range(n_new_patient):
#     # print(f'{i},index={new_patient_df.index[i]}')
#     new_patient_image_name = new_patient_df.iloc[i]['Image']
    new_patient_h5_name_list.append(valid_image_name[:-4] + '.h5') 

new_patient_h5_dict = {"new_patient":new_patient_h5_name_list}

# data set
with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'), 'w') as filehandle:
    json.dump(new_patient_h5_dict, filehandle)     

print(f'H5 dataset: {new_patient_h5_dict["new_patient"]}')

# In[14]:

# ### MODEL AND RUN SELECTION
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS,     KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS,     SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG,     TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD,     MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG,     SCALE_BY_INPUT, SCALE_THRESHOLD = _Params() 
    
TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH,         VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()


# In[15]:

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

# In[16]:

model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'

if NEW_MODEL_NUMBER:
    model_number = str(datetime.datetime.now())[0:10] + '_' +                    str(datetime.datetime.now())[11:13] + '_' +                    str(datetime.datetime.now())[14:16]
else:
    model_number = '2021-02-16_11_28'

# In[17]:

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

    # resnet_model_0.save('models/resnet_model_0')
    # resnet_model_0 = load_model('models/resnet_model_0')
    
    for layer in resnet_model_0.layers:
        layer.trainable = False                
    resnet_model__0 = Model(inputs=[resnet_model_0.inputs], 
                            outputs=[resnet_model_0.get_layer(loi).output])

    # resnet_model__0.save('models/resnet_model__0.h5')
    # resnet_model__0 = load_model('models/resnet_model__0.h5')    

    
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

# print(wst_model.name + ' model selected')
# print(ensemble_model.name + ' model selected')    


# In[25]:

# ensemble_model.summary()


# In[26]:

# plot_model(wst_model, show_shapes=True,\
#            show_layer_names=False,\
#            to_file='saved_images/' + model_selection + '_' + model_number + '_wst_architecture.png') 


# In[27]:

# plot_model(resnet_model, show_shapes=True,\
#            show_layer_names=False,\
#            to_file='saved_images/' + model_selection + '_' + model_number + '_resnet_architecture.png')


# In[28]:

# plot_model(ensemble_model, show_shapes=True,\
#            show_layer_names=False,\
#            to_file='saved_images/' + model_selection + '_' + model_number + '_ensemble_wst_resnet_architecture.png')


# In[18]:

# ### GENERATOR for 1 IMAGE at a time

datadir = os.path.join(H5_IMAGE_DIR, '')

dataset = new_patient_h5_dict
                                      
valid_1_generator = DataGenerator(dataset["new_patient"], datadir, augment=False, shuffle=False, standard=False,                                      batch_size=1, dim=(HEIGHT, WIDTH, MRACHANNELS), mask_dim=(HEIGHT, WIDTH, 1),                                       mlDWT=False, mralevel=MRALEVEL, wave=WAVELET, wavemode=WAVEMODE, verbose=0)

# In[19]:

# ### PREDICT

# valid_y_true = []
valid_y_pred = []

for i in range(len(dataset["new_patient"])):
    x_m, y, w = valid_1_generator.__getitem__(i)
    # valid_y_true.append(y[0].tolist())
    y_pred = ensemble_model(x_m).numpy().tolist()
    valid_y_pred.append(y_pred[0])
    
# valid_y_true = np.array(valid_y_true)
valid_y_pred = np.array(valid_y_pred)


# In[32]:
patient_no = 0
for idx,patient in enumerate(new_patient_h5_dict["new_patient"]):
    print(f'{patient[:-3]} scores: {valid_y_pred[idx]}')
# print(f'New patients dataset: {new_patient_h5_dict["new_patient"]}')
# print(valid_y_pred)

# In[33]:

# valid_pos_list = np.array(dataset["new_patient"])[valid_y_true[:,0]==1].tolist()
# valid_neg_list = np.array(dataset["new_patient"])[valid_y_true[:,1]==1].tolist()
new_patient_list = np.array(dataset["new_patient"]).tolist()
# print(new_patient_list)

# In[20]:

# ### HEAT MAPS

print('Calculating heat maps')

# pwd = os.getcwd()
# os.system('mkdir new_patient_gradcam')
# gradcam_path = os.path.join(pwd,'new_patient_gradcam')
# os.system('mkdir large_set_gradcam_valid_WST_RESNET/negative')
# OUT_IMAGE_DIR = os.path.join(gradcam_path,'negative/')
OUT_IMAGE_DIR = os.path.join(grad_cam_dir,'')
# print(H5_IMAGE_DIR)
# print(OUT_IMAGE_DIR)

# LABEL = "NaN"

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


# # In[21]:

# # ### RUN CXR WITH ALL MODELS IN MEMORY


# # ### GENERATE LUNG MASKS

# # get CXR image names from source directory                
# source_img_names = [f for f in listdir(source_resized_img_path) if isfile(join(source_resized_img_path, f))]

# for name in source_img_names:
#     print(f"CXR: {name}")
#     if name == '.DS_Store': 
#         continue    
#     input_img = cv2.imread(source_resized_img_path + name, cv2.IMREAD_GRAYSCALE)
#     scaled_img = input_img/255
#     scaled_img = np.expand_dims(scaled_img,axis = [0,-1])
#     mask = model(scaled_img).numpy()
#     mask_float = np.squeeze(mask[0,:,:,0])    
#     mask_binary = (mask_float > 0.5)*1
    
#     mask_float *=255    
#     mask_binary *=255
#     cv2.imwrite(target_resized_msk_path_float + name, mask_float)
#     cv2.imwrite(target_resized_msk_path_binary + name, mask_binary)    


# # ### READ PATIENT DATA AND STANDARDIZATION PARAMETERS
# new_patient_df = pd.read_csv("new_patient.csv",index_col = 0)

# n_new_patient = len(new_patient_df)
# # print(n_new_patient)

# with open('standardization_parameters_V7.json') as json_file:
#     standardization_parameters = json.load(json_file) 
    
# train_image_mean = standardization_parameters['mean']
# train_image_std = standardization_parameters['std']

# # print(train_image_mean, train_image_std)


# # ### PREPARE H5 FILES

# # Loop over the set of images to predict. In this case 'valid_pos_label' and 'valid_neg_label' are the tentative 
# # radiologist assignments already in the dataframe. 

# for i in range(n_new_patient):
#     # print(f'{i},index={new_patient_df.index[i]}')
#     valid_image_name, valid_pos_label, valid_neg_label, valid_weight =     new_patient_df.iloc[i]['Image'],    new_patient_df.iloc[i]['Positive'],    new_patient_df.iloc[i]['Negative'],    new_patient_df.iloc[i]['ClassWeight']

#     valid_image = cv2.imread(IMAGE_DIR + valid_image_name, cv2.IMREAD_GRAYSCALE)
    
# # Resize or equalize if this was not already done during datasets preparation    
#     # valid_image = cv2.resize(valid_image, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
#     # valid_image = cv2.equalizeHist(valid_image)

#     valid_image = np.expand_dims(valid_image,axis=-1)
        
#     # External learned mask of segmented lungs
#     valid_learned_mask = cv2.imread(MASK_DIR + valid_image_name, cv2.IMREAD_GRAYSCALE).astype('float64')
#     valid_learned_mask /= 255
#     valid_learned_mask = np.expand_dims(valid_learned_mask,axis=-1)
    
#     # Internal thresholded mask    
#     low_ind = valid_image < 6
#     high_ind = valid_image > 225    
#     valid_thresholded_mask = np.ones_like(valid_image)
#     valid_thresholded_mask[low_ind] = 0
#     valid_thresholded_mask[high_ind] = 0

#     # Combine the two masks
#     valid_mask = np.multiply(valid_thresholded_mask,valid_learned_mask)
    
#     # Standardization with training mean and std 
#     valid_image = valid_image.astype(np.float64)
#     valid_image -= train_image_mean
#     valid_image /= train_image_std        
    
#     with h5py.File(H5_IMAGE_DIR + valid_image_name[:-4] + '.h5', 'w') as hf: 
#         # Images
#         Xset = hf.create_dataset(
#             name='X',
#             data=valid_image,
#             shape=(HEIGHT, WIDTH, 1),
#             maxshape=(HEIGHT, WIDTH, 1),
#             compression="gzip",
#             compression_opts=9)
        
#         # Masks
#         Mset = hf.create_dataset(
#             name='M',
#             data=valid_mask,
#             shape=(HEIGHT, WIDTH, 1),
#             maxshape=(HEIGHT, WIDTH, 1),
#             compression="gzip",
#             compression_opts=9)
        
#         # Labels
#         yset = hf.create_dataset(
#             name='y',
#             data=[valid_pos_label,valid_neg_label])
        
#         # Class weights
#         wset = hf.create_dataset(
#             name='w',
#             data=valid_weight)             


# # ### GENERATE JSON DICTIONARY FOR NEW PATIENT NAMES

# new_patient_h5_name_list = []

# for i in range(n_new_patient):
#     new_patient_image_name = new_patient_df.iloc[i]['Image']
#     new_patient_h5_name_list.append(valid_image_name[:-4] + '.h5') 

# new_patient_h5_dict = {"new_patient":new_patient_h5_name_list}

# with open(H5_IMAGE_DIR + 'new_patient_dataset.json', 'w') as filehandle:
#     json.dump(new_patient_h5_dict, filehandle)     

# # print(new_patient_h5_dict["new_patient"])


# # ### PREDICT

# valid_y_true = []
# valid_y_pred = []

# for i in range(len(dataset["new_patient"])):
#     x_m, y, w = valid_1_generator.__getitem__(i)
#     valid_y_true.append(y[0].tolist())
#     y_pred = ensemble_model(x_m).numpy().tolist()
#     valid_y_pred.append(y_pred[0])
    
# valid_y_true = np.array(valid_y_true)
# valid_y_pred = np.array(valid_y_pred)

# new_patient_list = np.array(dataset["new_patient"]).tolist()
# # print(new_patient_list)


# # ### HEAT MAPS
# pwd = os.getcwd()
# os.system('mkdir new_patient_gradcam_WST_RESNET')
# gradcam_path = os.path.join(pwd,'new_patient_gradcam_WST_RESNET/')
# OUT_IMAGE_DIR = gradcam_path
# # print(H5_IMAGE_DIR)
# # print(OUT_IMAGE_DIR)

# if valid_y_true[:,0]==1:
#     LABEL = "POSITIVE"
# else:
#     LABEL = "NEGATIVE"
    
# print(f"Tentative radiologist assignment: {LABEL}")

# FIG_SIZE = (16,20)

# _HEAT_MAP_DIFF(ensemble_model,generator=valid_1_generator,layer='average',          labels=['Positive score','Negative score'],header='LABELED: ' + LABEL,figsize=FIG_SIZE,          image_dir=H5_IMAGE_DIR,out_image_dir=OUT_IMAGE_DIR,          img_list=new_patient_list,first_img=0,last_img=len(new_patient_list),          img_width=WIDTH,img_height=HEIGHT,display=True)


# # In[ ]:




