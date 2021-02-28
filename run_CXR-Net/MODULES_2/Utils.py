#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:51:48 2020

@author: dgatti
"""

# In[1]:

import numpy as np
import copy
import random
import cv2, h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
import itertools

from copy import copy
from matplotlib.colors import Normalize
from matplotlib import cm


from sklearn.metrics import confusion_matrix, roc_curve

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
from MODULES_2.Constants import _Params, _Paths

# ### MODEL AND RUN SELECTION
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
    MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
    SCALE_BY_INPUT, SCALE_THRESHOLD = _Params() 
    
TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()

# In[1]
def overlay_mask(image_layer,mask_layer,channel,fraction):
    image_layer = copy.deepcopy(image_layer)
    
    mask_layer = copy.deepcopy(mask_layer)
    mask_layer = mask_layer[:,:,channel]
    ind = mask_layer.astype(bool)
    
    if image_layer.shape[2] == 1:
        image_layer = np.squeeze(image_layer)        
        image_layer[ind] = image_layer[ind]*fraction 
        mask_layer = mask_layer*(1-np.max(image_layer[ind]))
        g_layer = image_layer + mask_layer
        rgb_layer = np.dstack((image_layer,g_layer,g_layer))
        
    elif image_layer.shape[2] == 3:
        r_layer = image_layer[:,:,0]
        g_layer = image_layer[:,:,1]
        b_layer = image_layer[:,:,2]

        g_layer[ind] = g_layer[ind]*fraction
        b_layer[ind] = b_layer[ind]*fraction
        mask_g_layer = mask_layer*(1-np.max(g_layer))
        mask_b_layer = mask_layer*(1-np.max(b_layer))

        g_layer = g_layer + mask_g_layer
        b_layer = b_layer + mask_b_layer

        rgb_layer = np.dstack((r_layer,g_layer,b_layer))
        
    return rgb_layer
    
def overlay_mask_2(image_layer,mask_layer,channel,fraction,mask_color):
    image_layer = copy.deepcopy(image_layer)    
    mask_layer = copy.deepcopy(mask_layer)
    ind = mask_layer.astype(bool)
    
    if image_layer.shape[2] == 1:
        image_layer = np.squeeze(image_layer)        
        image_layer[ind] = image_layer[ind]*fraction 
        mask_layer = mask_layer*(1-np.max(image_layer[ind]))
        g_layer = (image_layer + mask_layer)
        
        if mask_color == 'cyan':
            rgb_layer = np.dstack((image_layer,g_layer,g_layer))
        elif mask_color == 'yellow':
            rgb_layer = np.dstack((g_layer,g_layer,image_layer))
        elif mask_color ==  'violet':
            rgb_layer = np.dstack((g_layer,image_layer,g_layer))

        
    elif image_layer.shape[2] == 3:

        r_layer = np.squeeze(np.expand_dim(image_layer[:,:,0],axis=-1)) 
        g_layer = np.squeeze(np.expand_dim(image_layer[:,:,1],axis=-1)) 
        b_layer = np.squeeze(np.expand_dim(image_layer[:,:,2],axis=-1))            
                          
        if mask_color == 'cyan':
                   
            g_layer[ind] = g_layer[ind]*fraction 
            b_layer[ind] = b_layer[ind]*fraction
            mask_layer = mask_layer*(1-np.max(image_layer[ind]))
            g_layer = g_layer + mask_layer            
            b_layer = b_layer + mask_layer            
           
        elif mask_color == 'yellow':
                   
            r_layer[ind] = r_layer[ind]*fraction 
            g_layer[ind] = g_layer[ind]*fraction
            mask_layer = mask_layer*(1-np.max(image_layer[ind]))
            r_layer = r_layer + mask_layer            
            g_layer = g_layer + mask_layer
            
        elif mask_color ==  'violet':
                    
            g_layer[ind] = g_layer[ind]*fraction 
            b_layer[ind] = g_layer[ind]*fraction 
            mask_layer = mask_layer*(1-np.max(image_layer[ind]))
            r_layer = g_layer + mask_layer            
            b_layer = b_layer + mask_layer                                    
            
        rgb_layer = np.dstack((r_layer,g_layer,b_layer))
        
    return rgb_layer
    
def get_class_threshold(NUM_CLASS):
    step = 255/(NUM_CLASS-1)
    class_threshold = []
    for i in range(NUM_CLASS):
        jump = round(step*i)
        class_threshold.append(jump)
    return class_threshold


# In[2]:

# ### PREPROCESSING


def standardize(image, by_layer=True):
    """
    Standardize mean and standard deviation 
        of each channel and z_dimension.

    Args:
        image (np.array): input image, 
            shape (num_channels, dim_x, dim_y, dim_z)

    Returns:
        standardized_image (np.array): standardized version of input image
    """
    
    # initialize to array of zeros, with same shape as the image
    standardized_image = np.zeros_like(image)

    # iterate over channels
    # for channels first
    # for c in range(image.shape[0]):
    # for channels last
    for c in range(image.shape[3]):        
        # iterate over the `z` dimension
        # for z in range(image.shape[3]):
        if by_layer == True:
            # print(f"Standardizing by layer")
            for z in range(image.shape[2]):            
                # get a slice of the image 
                # at channel c and z-th dimension `z`
                image_slice = image[:,:,z,c]

                slice_mean = np.mean(image_slice)
                # subtract the mean from image_slice
                # centered = image_slice - np.mean(image_slice)
                centered = image_slice - slice_mean

                slice_std = np.std(image_slice)
                centered_scaled = centered / (slice_std + 1e-17)

                # update  the slice of standardized image
                # with the scaled centered and scaled image
                # for channels first
                # standardized_image[c, :, :, z] = centered_scaled
                # for channels last
                standardized_image[:, :, z, c] = centered_scaled
                
            return standardized_image
                
        elif by_layer == False:
            # print(f"Standardizing by channel")
            image_channel = image[:,:,:,c]
            image_mean = np.mean(image_channel)
            centered = image_channel - image_mean
            image_std = np.std(image_channel)
            centered_scaled = centered / (image_std + 1e-17)
            standardized_image[:, :, :, c] = centered_scaled

            return standardized_image, image_mean, image_std


def normalize(image):
    
    # initialize to array of zeros, with same shape as the image
    normalized_image = np.zeros_like(image)

    # iterate over channels
    # for channels first
    # for c in range(image.shape[0]):
    # for channels last
    for c in range(image.shape[3]):
        minval = np.min(image[:,:,:,c])
        maxval = np.max(image[:,:,:,c])
        # iterate over the `z` dimension
        # for z in range(image.shape[3]):
        for z in range(image.shape[2]):            
            # get a slice of the image 
            # at channel c and z-th dimension `z`
            image_slice = image[:,:,z,c]

            # subtract the mean from image_slice
            normalized_slice = (image_slice - minval)/(maxval-minval)


            # update the image slice with the normalized slice
            # for channels first
            # normalized_image[c, :, :, z] = normalized_slice
            # for channels last
            normalized_image[:, :, z, c] = normalized_slice            

    return normalized_image


# In[3]:

# ### HEAT MAPS

def get_mean_std_per_batch(image_path, df, H, W):
    sample_data = []
    for idx, img in enumerate(df.sample(len(df))["Image"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def load_image(img, image_dir, df, H, W, preprocess=True):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H, W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
        x = x[:,:,:,0]
        x = np.expand_dims(x, axis=-1)        
    return x


def grad_cam(model, img, cls, layer_name, H, W):
    
    img = img[0]
    
    tf.keras.backend.set_floatx('float64')
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, cls]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam, loss


def compute_gradcam(model, img, image_dir, out_image_dir, model_selection, model_number,df, labels, selected_labels,
                    layer_name='bn', figsize=(15, 10), H=300, W=340, header='UNKNOWN'):
    
    preprocessed_input = load_image(img, image_dir, df, H, W)    
    
    print("Loading original image")
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.title("Original: " + header)
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, H, W, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            
            gradcam, predictions = grad_cam(model, preprocessed_input, i, layer_name, H, W)

            plt.subplot(131 + j)
            plt.title(f"{labels[i]}: p={predictions[0].numpy():.3f}")
            plt.axis('off')
            # plt.colorbar()
            plt.imshow(load_image(img, image_dir, df, H, W, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0].numpy()))
            j += 1   
      
    plt.savefig('gradcam_' + img,bbox_inches='tight')
    

def grad_cam_sequence(model, img, mask, cls, layer_name):
    
    
    tf.keras.backend.set_floatx('float64')
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img,mask]))       
        loss = predictions[:, cls]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam, loss


def compute_gradcam_sequence(model, img, image_dir, out_image_dir, model_selection, model_number, labels, selected_labels,
                    layer_name='bn', figsize=(15, 10), header='UNKNOWN'):
     
    with h5py.File(image_dir + img, 'r') as f:
        X_h5 = np.array(f.get("X"))
        M_h5 = np.array(f.get("M"))
        y_h5 = np.array(f.get("y"))
        w_h5 = np.array(f.get("w"))
        
    print("Loading original image")
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.title("Original: " + header)
    plt.axis('off')
    plt.imshow(X_h5[:,:,0], cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            
            gradcam, predictions = grad_cam_sequence(model, X_h5, M_h5, i, layer_name)

            plt.subplot(131 + j)
            plt.title(f"{labels[i]}: p={predictions[0].numpy():.3f}")
            plt.axis('off')
            # plt.colorbar()
            plt.imshow(X_h5[:,:,0],cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0].numpy()))
            j += 1   
        
    plt.savefig('gradcam_' + img,bbox_inches='tight')


# ### ROC

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals

def get_roc_curve_sequence(labels, predicted_vals, true_vals):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = true_vals[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)            
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals

def get_mean_roc_curve_sequence(labels, predicted_vals, true_vals):
    auc_roc_vals = []
    fpr_rf_array = []
    tpr_rf_array = []
    fpr_rf_array = np.empty(0)
    tpr_rf_array = np.empty(0)
    
    for i in range(len(labels)):
        try:
            gt = true_vals[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            fpr_rf_array = np.concatenate((fpr_rf_array,fpr_rf))
            tpr_rf_array = np.concatenate((tpr_rf_array,tpr_rf))            

        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )

    plt.figure(1, figsize=(10, 10))
    fpr_rf_array = np.sort(fpr_rf_array)
    tpr_rf_array = np.sort(tpr_rf_array)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf_array,tpr_rf_array,'-',label="AUC = " + str(round(np.mean(auc_roc_vals), 3)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='upper left')    
    plt.show()                 

    return auc_roc_vals

def get_multi_roc_curve_sequence(labels, predicted_vals, true_vals):
    auc_roc_vals = []
    fpr_rf_array = []
    tpr_rf_array = []
    fpr_rf_array = np.empty(0)
    tpr_rf_array = np.empty(0)
    
    for i in range(len(labels)):
        try:
            gt = true_vals[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            fpr_rf_array = np.concatenate((fpr_rf_array,fpr_rf))
            tpr_rf_array = np.concatenate((tpr_rf_array,tpr_rf))            

        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )

    fpr_rf_array = np.sort(fpr_rf_array)
    tpr_rf_array = np.sort(tpr_rf_array)
    plt.plot(fpr_rf_array,tpr_rf_array,'.')    

    return round(np.mean(auc_roc_vals), 3),fpr_rf_array,tpr_rf_array


def commonelem_set(a,b):
    one = set(a)
    two = set(b)
    if (one & two):
        return ("There are common elements in both lists:", one & two)
    else:
        return ("There are no common elements between these two sets.") 

    
def _HEAT_MAP(model,generator,layer,\
               labels=['Positive','Negative'],header='POSITIVE',figsize=(20,30),\
               image_dir='IMAGE_DIR',out_image_dir='OUT_IMAGE_DIR',\
               img_list=['img'],first_img=0,last_img=-1,\
               img_width=200,img_height=200,display=False):       

    print(f"Labels: {labels}, Image directory: {image_dir}")

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])

    img_ind = first_img
    for image in img_list[first_img:last_img]:
        x_m, y, w = generator.__getitem__(img_ind)

        img_ind += 1    

        print(f"Loading image {image}")
        plt.figure(figsize=figsize)
        plt.subplot(141)
        plt.title(f"{image[:-3]}: {header}")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')    

        print(f"Generating heatmap for class: {labels[0]}")      
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_m)       
            pos_loss = predictions[:, 0]

        output = conv_outputs[0]      
        grads = tape.gradient(pos_loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)
        cam = cv2.resize(cam, (img_width,img_height), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        plt.subplot(142)
        plt.title(f"{labels[0]}: p={pos_loss[0].numpy():.3f}")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=min(0.5, pos_loss[0].numpy()))

        print(f"Generating heatmap for class: {labels[1]}")    
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_m)       
            neg_loss = predictions[:, 1]

        output = conv_outputs[0]    
        grads = tape.gradient(neg_loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)
        cam = cv2.resize(cam, (img_width,img_height), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        plt.subplot(143)
        plt.title(f"{labels[1]}: p={neg_loss[0].numpy():.3f}")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=min(0.5, neg_loss[0].numpy()))

        plt.subplot(144)
        plt.title(f"Mask")
        plt.axis('off')
        
        if MSK_FLOAT:
            mask = x_m[1][0,:,:,0]
        else:
            mask = tf.where(tf.greater(x_m[1][0,:,:,0],MSK_THRESHOLD),1.0,0.0)
 
        plt.imshow(mask, cmap='gray')  

        plt.savefig(out_image_dir + 'heatmap_' + image[:-3] + '.png', bbox_inches='tight')

        if not display:        
            plt.close()


def _HEAT_MAP_DIFF(model,generator,layer,\
               labels=['Positive','Negative'],header='POSITIVE',figsize=(20,30),\
               image_dir='IMAGE_DIR',out_image_dir='OUT_IMAGE_DIR',\
               img_list=['img'],first_img=0,last_img=-1,\
               img_width=200,img_height=200,display=False):       

    print(f"Labels: {labels}, Image directory: {image_dir}")
       

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])

    img_ind = first_img
    for image in img_list[first_img:last_img]:
        x_m, y, w = generator.__getitem__(img_ind)
        
        if MSK_FLOAT:
            mask = x_m[1][0,:,:,0]
        else:
            mask = tf.where(tf.greater(x_m[1][0,:,:,0],MSK_THRESHOLD),1.0,0.0)         

        img_ind += 1    

        print(f"Image {image}")
        plt.figure(figsize=figsize)
        plt.subplot(141)        
        plt.title(f"{image[:-3]}: {header}")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')    
      
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_m)       
            pos_loss = predictions[:, 0]

        output = conv_outputs[0]      
        grads = tape.gradient(pos_loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)
        cam = cv2.resize(cam, (img_width,img_height), cv2.INTER_LINEAR)
        cam_pos = np.ma.masked_where(mask == 0.0, cam) 
        cam_pos = (cam_pos - cam_pos.min()) / (cam_pos.max()-cam_pos.min())    

        plt.subplot(142)    
        plt.title(f"{labels[0]}: {pos_loss[0].numpy():.3f}")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')
        plt.imshow(cam_pos, cmap='jet', alpha=min(0.5, pos_loss[0].numpy()))
  
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_m)       
            neg_loss = predictions[:, 1]

        output = conv_outputs[0]    
        grads = tape.gradient(neg_loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)
        cam = cv2.resize(cam, (img_width,img_height), cv2.INTER_LINEAR)
        cam_neg = np.ma.masked_where(mask == 0.0, cam)
        cam_neg = (cam_neg - cam_neg.min()) / (cam_neg.max()-cam_neg.min())          
              
        plt.subplot(143)
        plt.title(f"{labels[1]}: {neg_loss[0].numpy():.3f}")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')
        plt.imshow(cam_neg, cmap='jet', alpha=min(0.5, neg_loss[0].numpy()))

        # Difference map        
        cam_diff = cam_pos - cam_neg        
        
        plt.subplot(144)
        plt.title(f"DIFFERENCE MAP")
        plt.axis('off')
        plt.imshow(x_m[0][0,:,:,0], cmap='gray')
        norm=colors.Normalize(vmin=-1.0, vmax=1.0)
        cam_masked = np.ma.masked_where(mask == 0.0, cam_diff)        
        plt.imshow(cam_masked, cmap='jet', norm=norm, alpha=min(0.5, np.abs(pos_loss[0].numpy() - neg_loss[0].numpy())))

        plt.savefig(out_image_dir + 'heatmap_' + image[:-3] + '.png', bbox_inches='tight')

        if not display:        
            plt.close()                        
                        
            
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm,interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):    
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
#     plt.axis('off')
    plt.grid(b=None)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
