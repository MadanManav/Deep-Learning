from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc 

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}

      
    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
       conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
       print("Conv1 shape") #?,300,300,32
       print(conv1.get_shape())
    #(input, up_sample_rate, channels=nooffilter, subsample, normalizer, bn_params, iter)
    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    print("Conv 1-----manav")
    print(x.get_shape())

    #180x180x24
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)
    
    print("Block One dim ")
    print(x)

    DB2_skip_connection = x    
    #90x90x32
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)
    
    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)
    
    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)
    
    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)
    
    print("Block Four dim ")
    print(x)
    

    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'
        # TODO(1.1) - incorporate a upsample function which takes the features of x --320
        # and produces 120 output feature maps, which are 16x bigger in resolution than 
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5

        #t_out = TransitionUp_elu(x, 320, 3, 'First_conv')
        #print('the shape of the first upsampling',t_out.shape)
        #t_out = TransitionUp_elu(t_out, 320, 3, 'First_conv')
        #print('the shape of the second upsampling',t_out.shape)
        '''
        # for upsampling once
        print('the shape of x',x.get_shape())
        f1_transition_up = TransitionUp_elu(x,960,1,'First_transition_decoder')#19,19,960
        print('the shape after first upsampling',f1_transition_up.get_shape())
        f2_transition_up = TransitionUp_elu(f1_transition_up,96,2,'Second_transition_decoder')#38,38,96
        print('the shape after second upsampling',f2_transition_up.get_shape())
        f3_transition_up = TransitionUp_elu(f2_transition_up,32,2,'Third_transition_decoder')
        print('the shape after Third upsampling',f3_transition_up.get_shape())#76,76,32
        crop_f3 = crop(f3_transition_up,DB3_skip_connection)
        print('shape after cropping the third layer',crop_f3.get_shape())#75,75,32
        f4_transition_up = TransitionUp_elu(crop_f3,24,2,'Fourth_transition_decoder')
        print('the shape after fourth upsampling',f4_transition_up.get_shape())
        final_transition_up = TransitionUp_elu(f4_transition_up,32,2,'Final_transition_decoder')
        print('the shape after fourth upsampling',final_transition_up.get_shape())

        End_maps_decoder1 = slim.conv2d(final_transition_up, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)        
        
        '''
        f1_transition_up = TransitionUp_elu(x,120,16,'First_transition_decoder')
        #t_out = tf.contrib.layers.conv2d_transpose(inputs = x, num_outputs =120,kernel_size=Kernel_size,stride=Stride,padding='SAME')
        print('the shape of t_out',f1_transition_up.get_shape())
        t_out_croped = crop(f1_transition_up,self.tgt_image)
        print('the shape of t_out',f1_transition_up.get_shape())
        t_out_croped = Concat_layers(t_out_croped,conv1)
        
        End_maps_decoder1 = slim.conv2d(t_out_croped, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map.shape)
        
    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        #input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps 
        config2_transitionup1 = TransitionUp_elu(x,120,2,'config2_transitionup1')
        config2_transitionup1 = crop(config2_transitionup1,DB4_skip_connection)
        config2_transitionup1 = Concat_layers(config2_transitionup1,DB4_skip_connection)
        config2_transitionup1 = Convolution(current=config2_transitionup1,out_features=256, kernel_size = 3,name = 'config2_transitionup12')
        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1) 
        # and produces 120 output feature maps, which are 8x bigger in resolution than
        config2_transitionup1 = TransitionUp_elu(config2_transitionup1,120,8,'config2_transitionup13')
        config2_transitionup1 = crop(config2_transitionup1,self.tgt_image)
        
        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3

        End_maps_decoder1 = slim.conv2d(config2_transitionup1, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:
        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        current_up4 = TransitionUp_elu(input=x, filters=120, strideN=2, name='current_up4')
        print("\nshape after Transition", current_up4.shape)

        current_up4 = crop(current_up4, DB4_skip_connection)
        print("\nshape after cropping", current_up4.shape)

        current_up4 = Concat_layers(current_up4, DB4_skip_connection)
        print("\nshape after concat", current_up4.shape)

        current_up4 = Convolution(current=current_up4, out_features=256, kernel_size=3, name='current_up41')
        print("\nshape after convolution", current_up4.shape)

        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        current_up4 = TransitionUp_elu(input=current_up4, filters=160, strideN=2, name='current_up42')
        print("\nshape after second Transition", current_up4.shape)

        current_up4 = crop(current_up4, DB3_skip_connection)
        print("\nshape after second cropping", current_up4.shape)

        current_up4 = Concat_layers(current_up4, DB3_skip_connection)
        print("\nshape after second concat", current_up4.shape)

        #current_up4 = Convolution(current=current_up4, out_features=256, kernel_size=3, name='current_up31')
        #print("\nshape after second convolution", current_up4.shape)



        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)
        # and produces 120 output feature maps which are 4x bigger in resolution than
        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4

        current_up4 = TransitionUp_elu(input=current_up4, filters=120, strideN=4, name='current_up43')
        print("\nshape after third transition", current_up4.shape)

        current_up4 = crop(current_up4, self.tgt_image)
        print("\nshape after third cropping", current_up4.shape)

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration 
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################
        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        current_up5 = TransitionUp_elu(input=x, filters=120, strideN=2, name='current_up50')
        print("\nshape after Transition", current_up5.shape)
        current_up5 = crop(current_up5, DB4_skip_connection)
        print("\nshape after cropping", current_up5.shape)
        current_up5 = Concat_layers(current_up5, DB4_skip_connection)
        print("\nshape after concat", current_up5.shape)
        current_up5 = Convolution(current=current_up5, out_features=256, kernel_size=3, name='current_up51')
        print("\nshape after convolution", current_up5.shape)
        
       
        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        current_up5 = TransitionUp_elu(input=current_up5, filters=160, strideN=2, name='current_up52')
        print("\nshape after second Transition", current_up5.shape)
        current_up5 = crop(current_up5, DB3_skip_connection)
        print("\nshape after second cropping", current_up5.shape)
        current_up5 = Concat_layers(current_up5, DB3_skip_connection)
        print("\nshape after second concat", current_up5.shape)

        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.

        current_up5 = TransitionUp_elu(input=current_up5, filters=96, strideN=2, name='current_up53')
        print("\nshape after second Transition", current_up5.shape)
        current_up5 = crop(current_up5, DB2_skip_connection)
        print("\nshape after second cropping", current_up5.shape)
        current_up5 = Concat_layers(current_up5, DB2_skip_connection)
        print("\nshape after second concat", current_up5.shape)

        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3) 
        # and produce 120 output feature maps which are 2x bigger in resolution than 
        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4
        current_up5 = TransitionUp_elu(input=current_up5, filters=120, strideN=2, name='current_up54')
        print("\nshape after third transition", current_up5.shape)
        current_up5 = crop(current_up5, self.tgt_image)
        print("\nshape after third cropping", current_up5.shape)
        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder5') #(batchsize, width, height, N_classes)
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    
    return Reshaped_map
