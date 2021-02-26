#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:51:48 2020

@author: dgatti
"""


# In[1]
from MODULES.Blocks import * 
from MODULES.Constants import _Params
from MODULES.Attention import *

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, SeparableConv2D, Multiply
from tensorflow.keras.layers import Conv2D, Concatenate, Add, LeakyReLU, AveragePooling2D
from tensorflow.keras.layers import Dropout, SpatialDropout2D, GlobalAveragePooling2D
from tensorflow.keras.layers import LayerNormalization, Lambda, UpSampling2D, GaussianNoise
import tensorflow.keras.initializers
import tensorflow.keras.regularizers
import tensorflow.keras.constraints
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Layer

import kymatio
from kymatio.keras import Scattering2D


# In[2]

# ### CONSTANTS
 
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    SHIFT_LIMIT, SCALE_LIMIT, ROTATE_LIMIT, ASPECT_LIMIT, U_AUG, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS, MSK_FLOAT, MSK_THRESHOLD, \
    MRA, MRALEVEL, MRACHANNELS, WAVELET, WAVEMODE, WST, WST_J, WST_L, WST_FIRST_IMG, \
    SCALE_BY_INPUT, SCALE_THRESHOLD = _Params()


# In[3]:    

class TransposeChannel(Layer):
    # add here additional parameters
    def __init__(self, perm_list=None, name=None, **kwargs):
        super(TransposeChannel, self).__init__(name=name, **kwargs)
        self.perm_list = perm_list

    def call(self, inputs):
        return tf.transpose(inputs,perm=self.perm_list)
    
    def get_config(self):  
        config = super(TransposeChannel, self).get_config()
        config.update({"perm_list": self.perm_list})
        return config    

    
class ScaleByInput(Layer):
    # add here additional parameters
    def __init__(self, scale_to_span=False, span=1.0, threshold=1.0, name=None, **kwargs):
        super(ScaleByInput, self).__init__(name=name, **kwargs)
        self.scale_to_span = scale_to_span
        self.span = span
        self.threshold = threshold

    def call(self, inputs):
        if self.scale_to_span:
            inputs_min = tf.math.reduce_min(inputs)
            inputs_max = tf.math.reduce_max(inputs)
            inputs_span = inputs_max - inputs_min
            output = (inputs - inputs_min)*self.span/inputs_span
        else:
            output = tf.where(tf.greater(inputs,tf.math.reduce_min(inputs)*self.threshold),1.0,0.0)
        return output
    
    def get_config(self):
        config = super(ScaleByInput, self).get_config()
        config.update({"scale_to_span": self.scale_to_span,"span": self.span, "threshold": self.threshold})
        return config              

    
class Threshold(Layer):
    # add here additional parameters
    def __init__(self, threshold=1.0, name=None, **kwargs):
        super(Threshold, self).__init__(name=name, **kwargs)
        self.threshold = threshold

    def call(self, inputs):    
        return tf.where(tf.greater(inputs,self.threshold),1.0,0.0)
    
    def get_config(self):
        config = super(Threshold, self).get_config()
        config.update({"threshold": self.threshold})
        return config            

    
class SelectChannel(Layer):
    # add here additional parameters
    def __init__(self, channel=0, add_dim=True, name=None, **kwargs):
        super(SelectChannel, self).__init__(name=name, **kwargs)
        self.channel = channel
        self.add_dim = add_dim

    def call(self, inputs):
        if self.add_dim:
            output = inputs[:,:,:,self.channel]
            output = output[:,:,:,tf.newaxis]
        else:
            output = inputs[:,:,:,self.channel]
        return output

    def get_config(self):
        config = super(SelectChannel, self).get_config()
        config.update({"channel": self.channel,"add_dim": self.add_dim})
        return config       

        
def WaveletScatteringTransform(input_shape=(HEIGHT, WIDTH, CHANNELS),
                               upsample=False,
                               upsample_kernel=(2,2)):

    input_1 = Input(shape=input_shape)    
    input_2 = Input(shape=input_shape)
    
    if upsample:
        input_u1 = UpSampling2D(size=upsample_kernel, interpolation="bilinear")(input_1)
        input_u2 = UpSampling2D(size=upsample_kernel, interpolation="bilinear")(input_2)
    else:
        input_u1 = input_1
        input_u2 = input_2  
        
    d0 = SelectChannel(channel=0,add_dim=False)(input_u1)
    d0 = Scattering2D(J=WST_J,L=WST_L)(d0)      
    d0 = TransposeChannel(perm_list=[0, 2, 3, 1])(d0)           

    m0 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding="valid")(input_u2)
        
    output_1 = Concatenate()([d0, m0])    

    model = Model([input_1,input_2], output_1, name='WST')

    return model


def ResNet(input_shape_1=(HEIGHT, WIDTH, CHANNELS),
           input_shape_2=(HEIGHT, WIDTH, 1),
           num_class=NUM_CLASS,
           ks1=KS1, ks2=KS2, ks3=KS3, 
           dl1=DL1, dl2=DL2, dl3=DL3,
           filters=NF,resblock1=NR1,
           r_filters=NFL, resblock2=NR2,
           dil_mode=DIL_MODE, 
           sp_dropout=DR1,re_dropout=DR2,
           prep=False,
           stem=False,
           mask_float=MSK_FLOAT,
           mask_threshold=MSK_THRESHOLD,
           att='mh',
           head_size=128,
           num_heads=12,
           value_att=True,
           scale_by_input=False,
           scale_threshold=1.0,
           scale_to_span=False,
           span=1.0,           
           blur_sbi=False,
           blur_sbi_std=0.1,                                  
           return_seq=True):

    input_1 = Input(shape=input_shape_1)    
    input_2 = Input(shape=input_shape_2)
    
    a0 = SelectChannel(channel=0)(input_1) 
    m0 = SelectChannel(channel=-1)(input_1)
    m1 = Threshold(threshold=mask_threshold)(m0)
                
    if scale_by_input:           
        s0 = ScaleByInput(scale_to_span=scale_to_span,threshold=scale_threshold)(a0)             

    if att == 'mh':
            if value_att:
                att_rows = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0,m1,m0])

                a0t = TransposeChannel(perm_list=[0, 2, 1, 3])(a0)
                m0t = TransposeChannel(perm_list=[0, 2, 1, 3])(m0)
                m1t = TransposeChannel(perm_list=[0, 2, 1, 3])(m1)            
                att_cols = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0t,m1t,m0t])            
                att_cols = TransposeChannel(perm_list=[0, 2, 1, 3])(att_cols)
            else:            
                att_rows = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0,m1])

                a0t = TransposeChannel(perm_list=[0, 2, 1, 3])(a0)                
                m0t = TransposeChannel(perm_list=[0, 2, 1, 3])(m0)
                m1t = TransposeChannel(perm_list=[0, 2, 1, 3])(m1)                
                att_cols = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0t,m1t])            
                att_cols = TransposeChannel(perm_list=[0, 2, 1, 3])(att_cols)

            a0m0 = Multiply()([a0, att_rows, att_cols])
            
    elif att == 'Multiply':
            a0m0 = Multiply()([a0, m0])        
                
    d0 = Concatenate()([input_1,a0m0])

    # Preparation for scattering transform output into conv blocks       
    if prep:        
        d0 = SpatialDropout2D(sp_dropout)(d0)
        d0 = BatchNormalization()(d0)
            
    for cycle in range(resblock1):
        if cycle == 0:
            if stem:
                d1 = stem_split_3k(d0, filters, mode=dil_mode, 
                                    kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                    dilation_1=dl1, dilation_2=dl2, dilation_3=dl3,
                                    kernel_regularizer=None)
            else:
                d1 = residual_block_split_3k(d0, filters, mode=dil_mode, 
                                             kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                             dilation_1=dl1, dilation_2=dl2, dilation_3=dl3,
                                             kernel_regularizer=None)                 
        else:
            d1 = residual_block_split_3k(d1, filters, mode=dil_mode, 
                                         kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                         dilation_1=dl1, dilation_2=dl2, dilation_3=dl3,
                                         kernel_regularizer=None) 
            
        d1 = SpatialDropout2D(sp_dropout)(d1)            
                 
    if resblock2 > 0:
        for cycle in range(resblock2):
            if cycle == 0:
                d2 = residual_convLSTM2D_block(d1,r_filters,num_class,rd=re_dropout,rs=return_seq)
            else:
                d2 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout,rs=return_seq) 
    else:
        d2 = residual_block_split_3k(d1, num_class, mode="add", 
                                    kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                    dilation_1=dl1, dilation_2=dl2,dilation_3=dl3,
                                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))
        
    d3 = LeakyReLU(alpha=0.1)(d2)
    
    if scale_by_input:
        if blur_sbi:
            # blurring also during evaluation
            # s0 = GaussianNoise(blur_sbi_std)(s0,training=True)
            s0 = GaussianNoise(blur_sbi_std)(s0)
        d3 = Multiply()([d3, s0])    
                  
    if mask_float:
        d3 = Multiply()([d3, m0])
    else:
        d3 = Multiply()([d3, m1])        
   
    d3 = GlobalAveragePooling2D()(d3)    
    
    output_1 = Activation("softmax", name = 'softmax')(d3)    
    
    model = Model(input_1, output_1, name='ResNet')    

    return model


def WaveletScatteringTransform_0(input_shape=(HEIGHT, WIDTH, CHANNELS),
                                 upsample=False,
                                 upsample_kernel=(2,2)):

    input_1 = Input(shape=input_shape)    
    input_2 = Input(shape=input_shape)
    
    if upsample:
        input_u1 = UpSampling2D(size=upsample_kernel, interpolation="bilinear")(input_1)
        input_u2 = UpSampling2D(size=upsample_kernel, interpolation="bilinear")(input_2)
    else:
        input_u1 = input_1
        input_u2 = input_2       
        
    d0 = SelectChannel(channel=0,add_dim=False)(input_u1)
    d0 = Scattering2D(J=WST_J,L=WST_L)(d0)      
    d0 = TransposeChannel(perm_list=[0, 2, 3, 1])(d0)            

    m0 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding="valid")(input_u2)
        
    output_1 = d0
    output_2 = m0
    
    model = Model([input_1,input_2], [output_1,output_2], name='WST')    

    return model


def ResNet_0(input_shape_1=(HEIGHT, WIDTH, CHANNELS),
           input_shape_2=(HEIGHT, WIDTH, 1),
           num_class=NUM_CLASS,
           ks1=KS1, ks2=KS2, ks3=KS3, 
           dl1=DL1, dl2=DL2, dl3=DL3,
           filters=NF,resblock1=NR1,
           r_filters=NFL, resblock2=NR2,
           dil_mode=DIL_MODE, 
           sp_dropout=DR1,re_dropout=DR2,
           prep=False,
           stem=False,
           mask_float=MSK_FLOAT,
           mask_threshold=MSK_THRESHOLD,
           head_size=64,
           num_heads=2,
           value_att=False,             
           scale_by_input=False,
           scale_threshold=1.0,
           scale_to_span=False,
           span=1.0,                 
           blur_sbi=False,
           blur_sbi_std=0.1,                                  
           return_seq=True):

    input_1 = Input(shape=input_shape_1)    
    input_2 = Input(shape=input_shape_2)
    
    a0 = SelectChannel(channel=0)(input_1)
    m0 = input_2
    m1 = Threshold(threshold=mask_threshold)(input_2)    
    
    if scale_by_input:           
        s0 = ScaleByInput(scale_to_span=scale_to_span,threshold=scale_threshold)(a0)
        
    if value_att:
        att_rows = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0,m1,m0])

        a0t = TransposeChannel(perm_list=[0, 2, 1, 3])(a0)
        m0t = TransposeChannel(perm_list=[0, 2, 1, 3])(m0)
        m1t = TransposeChannel(perm_list=[0, 2, 1, 3])(m1)            
        att_cols = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0t,m1t,m0t])            
        att_cols = TransposeChannel(perm_list=[0, 2, 1, 3])(att_cols)
    else:            
        att_rows = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0,m1])

        a0t = TransposeChannel(perm_list=[0, 2, 1, 3])(a0)                
        m0t = TransposeChannel(perm_list=[0, 2, 1, 3])(m0)
        m1t = TransposeChannel(perm_list=[0, 2, 1, 3])(m1)                
        att_cols = MultiHeadAttention(head_size=head_size,num_heads=num_heads)([a0t,m1t])            
        att_cols = TransposeChannel(perm_list=[0, 2, 1, 3])(att_cols)

    att_map = Multiply()([att_rows, att_cols])            
        
    a0_m0 = Multiply()([a0, m0])
    a0_m1 = Multiply()([a0, m1])    
    a0_att_map = Multiply()([a0, att_map])
     
    d0 = Concatenate()([input_1,m0,att_map,a0_m0,a0_m1,a0_att_map])
    
    # Preparation for scattering transform output into conv blocks    
    if prep:
        d0 = SpatialDropout2D(sp_dropout)(d0)
        d0 = BatchNormalization()(d0)
            
    for cycle in range(resblock1):
        if cycle == 0:
            if stem:
                d1 = stem_split_3k(d0, filters, mode=dil_mode, 
                                    kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                    dilation_1=dl1, dilation_2=dl2, dilation_3=dl3,
                                    kernel_regularizer=None)
            else:
                d1 = residual_block_split_3k(d0, filters, mode=dil_mode, 
                                             kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                             dilation_1=dl1, dilation_2=dl2, dilation_3=dl3,
                                             kernel_regularizer=None)                 
        else:
            d1 = residual_block_split_3k(d1, filters, mode=dil_mode, 
                                         kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                         dilation_1=dl1, dilation_2=dl2, dilation_3=dl3,
                                         kernel_regularizer=None) 
            
        d1 = SpatialDropout2D(sp_dropout)(d1)            
                 
    if resblock2 > 0:
        for cycle in range(resblock2):
            if cycle == 0:
                d2 = residual_convLSTM2D_block(d1,r_filters,num_class,rd=re_dropout,rs=return_seq)
            else:
                d2 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout,rs=return_seq) 
    else:
        d2 = residual_block_split_3k(d1, num_class, mode="add", 
                                    kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                    dilation_1=dl1, dilation_2=dl2,dilation_3=dl3,
                                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))
        
    d3 = LeakyReLU(alpha=0.1)(d2)
    
    if scale_by_input:
        if blur_sbi:
            # blurring also during evaluation
            # s0 = GaussianNoise(blur_sbi_std)(s0,training=True)
            s0 = GaussianNoise(blur_sbi_std)(s0)
        d3 = Multiply()([d3, s0])    
                       
    if mask_float:
        mask = m0
        d3 = Multiply()([d3, mask])
    else:
        mask = m1
        d3 = Multiply()([d3, mask])        
   
    d3 = GlobalAveragePooling2D()(d3)    
    
    output_1 = Activation("softmax", name = 'softmax')(d3)    
    
    model = Model([input_1,input_2], output_1, name='ResNet')    

    return model
