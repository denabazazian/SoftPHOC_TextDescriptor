# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:32:48 2018

@author: dena
"""
# cd ~/path/to/softPHOC_synthText/word_spotting/codes/save_hm
# python save_char_pred.py /path/to/datasets/context_images/JPEGImages/*.jpg


from __future__ import division

import scipy
import scipy.io
import scipy.io as sio
import sys
import numpy as np
import cv2
from collections import defaultdict
from commands import getoutput as go
import glob, os
import re
from pylab import *
import tensorflow as tf
import skimage.io as io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import time

from numpy import array, zeros, argmin, inf, ndim
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from nltk.metrics.distance import edit_distance


if __name__=='__main__':

    '''---------------------------------------------------------------------------------------------------------------------
    defining classes and char2int dictionary
    ---------------------------------------------------------------------------------------------------------------------'''

    # char2int = {'a': 1, 'A': 1, 'b': 2, 'B': 2, 'c': 3, 'C': 3, 'd': 4, 'D': 4, 'e': 5, 'E': 5, 'f': 6, 'F': 6, 'g': 7,
    #             'G': 7, 'h': 8, 'H': 8, 'i': 9, 'I': 9, 'j': 10, 'J': 10, 'k': 11, 'K': 11, 'l': 12, 'L': 12, 'm': 13,
    #             'M': 13, 'n': 14, 'N': 14, 'o': 15, 'O': 15, 'p': 16, 'P': 16, 'q': 17, 'Q': 17, 'r': 18, 'R': 18,
    #             's': 19, 'S': 19, 't': 20, 'T': 20, 'u': 21, 'U': 21, 'v': 22, 'V': 22, 'w': 23, 'W': 23, 'x': 24,
    #             'X': 24, 'y': 25, 'Y': 25, 'z': 26, 'Z': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32,
    #             '7': 33, '8': 34, '9': 35, '0': 36, '!': 37, '$': 37, '>': 37, '<': 37, '.': 37, ':': 37, '-': 37,
    #             '_': 37, '(': 37, ')': 37, '[': 37, ']': 37, '{': 37, '}': 37, ',': 37, ';': 37, '#': 37, '?': 37,
    #             '%': 37, '*': 37, '/': 37, '@': 37, '^': 37, '&': 37, '=': 37, '+': 37, '€': 37, "'": 37, '`': 37,
    #             '"': 37, '\\': 37, '\xc2': 37, '\xb4': 37, ' ': 37, '\xc3': 37, '\x89': 37}  # '\':37

    number_of_classes = 38

    '''---------------------------------------------------------------------------------------------------------------------
    Set paths of img and gt
    ---------------------------------------------------------------------------------------------------------------------'''

    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
    
    sys.path.append("/path/to/tensorflow-FCN-gitHub/warmspringwinds/tf-image-segmentation/")
    sys.path.append("/path/to/tensorflow-FCN-gitHub/warmspringwinds/models/slim")

    restor_last_chackpoint = "/path/to/softPHOC_synthText/tf_model/tf_model_3_backup_118000/model_fcn32s_118000.ckpt"

    img_path = "/path/to/datasets/context_images/JPEGImages/"
    save_mat_path = "/path/to/char_pred_hm/"

    #indx = 0

    # imgnames_train = open(index_training_img).read()
    # imgnames_train = imgnames_train.split('\n')
    # print len(imgnames_train)
    # imgnames_train = [x for x in imgnames_train if x]
    # print len(imgnames_train)

    '''---------------------------------------------------------------------------------------------------------------------
    Set up the net
    ---------------------------------------------------------------------------------------------------------------------'''

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #os.environ["CUDA_VISIBLE_DEVICES"] = ''

    slim = tf.contrib.slim

    # from fcn_32s import FCN_32s
    # from inference_prediction import adapt_network_for_any_size_input

    from tf_image_segmentation.models.fcn_32s import FCN_32s
    from tf_image_segmentation.utils.inference_prediction import adapt_network_for_any_size_input



    #image_filename = '/home/dbazazia/nfs/dataset/ICDAR2015/ch4_train_img/img_43.jpg' #'/home/dbazazia/nfs/dataset/ICDAR2015/ch4_test/img_108.jpg' 

    image_filename_placeholder = tf.placeholder(tf.string)
    #feed_dict_to_use = {image_filename_placeholder: image_filename}
    image_tensor = tf.read_file(image_filename_placeholder)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    # Fake batch for image and annotation by adding leading empty axis.
    image_batch_tensor = tf.expand_dims(image_tensor, axis=0)
    # Be careful: after adaptation, network returns final labels and not logits
    FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)

    upsampled_logit, fcn_16s_variables_mapping_train = FCN_32s(image_batch_tensor=image_batch_tensor,
                                                            number_of_classes=number_of_classes,
                                                            is_training=False, reuse=None)
                                            
    pred = tf.argmax(upsampled_logit, dimension=3)                                         
    probabilities = tf.nn.softmax(upsampled_logit)

    initializer = tf.local_variables_initializer()

    saver = tf.train.Saver()

    cmap = plt.get_cmap('bwr')
    '''---------------------------------------------------------------------------------------------------------------------
    run the session for each image
    ---------------------------------------------------------------------------------------------------------------------'''
    with tf.Session() as sess:
        
        sess.run(initializer)
        #saver.restore(sess,"/home/dbazazia/workspace/softPHOC_synthText/tf_model/model_fcn32s_118000.ckpt")
        saver.restore(sess,restor_last_chackpoint)

        img_list = sorted(glob.glob(img_path+"*.jpg"))

        #for image_filename in sys.argv[1:]:
        #for image_filename in img_list: 
        for indx in range(17275, len(img_list)):    #(saved till 17281 or 17280)  #(0, len(img_list)): 
            #indx += 1
            #if indx > 20:
		  	#   break;
            image_filename = img_list[indx]
            #image_filename = glob(img_path+img_name+ ".*")[0]
            #image_filename = img_path+img_name+ ".jpg"

            #print image_filename
            img_name = image_filename.split('/')[-1].split('.')[0]
            print "image_name is : {}".format(img_name)

            img = cv2.imread(image_filename)
            feed_dict_to_use = {image_filename_placeholder: image_filename}


            image_np, pred_np , probabilities_np = sess.run([image_tensor, pred, probabilities], feed_dict=feed_dict_to_use)

            sio.savemat(save_mat_path+img_name+ '.mat',
                       {'character_prediction':probabilities_np[0,:,:,:]},
                       do_compression=True)

            print "index of completed image is : {}".format(indx)

                    


