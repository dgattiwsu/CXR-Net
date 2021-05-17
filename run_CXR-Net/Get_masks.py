#!/usr/bin/env python3
# coding: utf-8

# ### Predict lung masks and Covid vs non-Covid classification for new patient CXR using Module 1 trained on the V7 lung segmentation database, and Module 2 trained on the HFHS dataset   
# 
# usage:  python Get_masks.py --dirs new_patient_cxr image_dcm 
#                                  image_resized_equalized_from_dcm mask_binary 
#                                  mask_float image_mask H5 grad_cam 
#                                  [--imgs name1.dcm name2.dcm ... nameN.dcm]

# Choice of directories is mandatory. If no images are selected using the --imgs 
# flag, all dcm images in the source directory are processed. 

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

# In[4]:

# ### CONSTANTS

HEIGHT,WIDTH,CHANNELS,IMG_COLOR_MODE,MSK_COLOR_MODE,NUM_CLASS,KS1,KS2,KS3,DL1,DL2,DL3,NF,NFL,NR1,NR2,DIL_MODE,W_MODE,LS,TRAIN_SIZE,VAL_SIZE,TEST_SIZE,DR1,DR2,CLASSES,IMG_CLASS = _Params()

TRAIN_IMG_PATH,TRAIN_MSK_PATH,TRAIN_MSK_CLASS,VAL_IMG_PATH,VAL_MSK_PATH,VAL_MSK_CLASS,TEST_IMG_PATH,TEST_MSK_PATH,TEST_MSK_CLASS = _Paths()

# In[5]: 

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

# In[6]

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

# In[8]:       

# get CXR image names from source directory
source_img_names = [f for f in listdir(source_resized_img_path) if isfile(join(source_resized_img_path, f))]

for name in source_img_names:
    # print(f'Image name: {name}')
    if name == '.DS_Store': 
        continue
    if (args.imgs) and (name[:-4] + '.dcm' not in args.imgs):
        continue
    
    input_img = cv2.imread(os.path.join(source_resized_img_path, name), cv2.IMREAD_GRAYSCALE)
    scaled_img = input_img/255
    # scaled_img2 = scaled_img*1
    
    scaled_img = np.expand_dims(scaled_img,axis = [0,-1])
    mask = model(scaled_img).numpy()
    mask_float = np.squeeze(mask[0,:,:,0])    
    mask_binary = (mask_float > 0.5)*1
              
    mask_float *=255    
    mask_binary *=255
    cv2.imwrite(os.path.join(target_resized_msk_path_float, name), mask_float)
    cv2.imwrite(os.path.join(target_resized_msk_path_binary, name), mask_binary)

    mask_float2 = cv2.imread(os.path.join(target_resized_msk_path_float, name), cv2.IMREAD_GRAYSCALE)    
    mask_binary2 = cv2.imread(os.path.join(target_resized_msk_path_binary, name), cv2.IMREAD_GRAYSCALE)      
    img_mask = cv2.hconcat([input_img,mask_float2,mask_binary2])
    cv2.imwrite(os.path.join(target_img_mask_path, name + '_img_and_pred_mask.png'), img_mask)    
        
# In[9]:

# ### RECOVER STANDARDIZATION PARAMETERS FROM MODULE 1

with open('standardization_parameters_V7.json') as json_file:
    standardization_parameters = json.load(json_file) 
    
train_image_mean = standardization_parameters['mean']
train_image_std = standardization_parameters['std']

# In[10]:

# ### PREPARE STANDARDIZED IMAGES as H5 FILES

# Source directories for images and masks
IMAGE_DIR = source_resized_img_path
MASK_DIR = target_resized_msk_path_float

# Target directory for H5 file containing both images and masks
H5_IMAGE_DIR = h5_img_dir

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
    if (args.imgs) and (valid_image_name[:-4] + '.dcm' not in args.imgs):
        continue    
    
    # Radiologist labels and class weights are not known
    valid_pos_label = 0
    valid_neg_label = 0
    valid_weight = 1.0

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

# In[11]:

# ### Generate json dictionary for the new patient names

new_patient_h5_name_list = []

for valid_image_name in source_img_names:
    if valid_image_name == '.DS_Store': 
        continue
    if (args.imgs) and (valid_image_name[:-4] + '.dcm' not in args.imgs):
        continue    

    new_patient_h5_name_list.append(valid_image_name[:-4] + '.h5') 

new_patient_h5_dict = {"new_patient":new_patient_h5_name_list}

# data set
with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'), 'w') as filehandle:
    json.dump(new_patient_h5_dict, filehandle)     

print(f'H5 dataset: {new_patient_h5_dict["new_patient"]}')

# In[12]:
    
# def main():

# if __name__ == '__main__':
#     main()
    
