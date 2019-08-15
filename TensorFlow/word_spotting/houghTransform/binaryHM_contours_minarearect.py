# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:54:48 2017

@author: dena
"""

from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
import cv2


alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
char2int = {'a': 1, 'A': 1, 'b': 2, 'B': 2, 'c': 3, 'C': 3, 'd': 4, 'D': 4, 'e': 5, 'E': 5, 'f': 6, 'F': 6, 'g': 7,
            'G': 7, 'h': 8, 'H': 8, 'i': 9, 'I': 9, 'j': 10, 'J': 10, 'k': 11, 'K': 11, 'l': 12, 'L': 12, 'm': 13,
            'M': 13, 'n': 14, 'N': 14, 'o': 15, 'O': 15, 'p': 16, 'P': 16, 'q': 17, 'Q': 17, 'r': 18, 'R': 18,
            's': 19, 'S': 19, 't': 20, 'T': 20, 'u': 21, 'U': 21, 'v': 22, 'V': 22, 'w': 23, 'W': 23, 'x': 24,
            'X': 24, 'y': 25, 'Y': 25, 'z': 26, 'Z': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32,
            '7': 33, '8': 34, '9': 35, '0': 36, '!': 37, '$': 37, '>': 37, '<': 37, '.': 37, ':': 37, '-': 37,
            '_': 37, '(': 37, ')': 37, '[': 37, ']': 37, '{': 37, '}': 37, ',': 37, ';': 37, '#': 37, '?': 37,
            '%': 37, '*': 37, '/': 37, '@': 37, '^': 37, '&': 37, '=': 37, '+': 37, '€': 37, "'": 37, '`': 37,
            '"': 37, '\\': 37, '\xc2': 37, '\xb4': 37, ' ': 37, '\xc3': 37, '\x89': 37} 

sys.path.append("/softPHOC_synthText/slim/")
sys.path.append("/softPHOC_synthText/models/")
sys.path.append("/softPHOC_synthText/softPHOC_utils/")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = ''

slim = tf.contrib.slim

# from tf_image_segmentation.models.fcn_32s import FCN_32s
from fcn_32s import FCN_32s

from tf_image_segmentation.utils.inference_prediction import adapt_network_for_any_size_input

number_of_classes = 38
image_folder = '/softPHOC_synthText/data/test_images/'
#image_filename = 'img_108.jpg' # less color contrast
image_filename = image_folder + 'img_341.jpg' #soup soop
#image_filename = image_folder + 'img_440.jpg'   # bad heatmap response
#image_filename = image_folder + 'img_485.jpg'   #bad heatmap
#image_filename = image_folder + 'img_65.jpg'
#image_filename = image_folder + 'img_14.jpg'
#image_filename = image_folder + 'img_418.jpg'  #owndays
# image_filename = image_folder + 'img_119.jpg' #RObert Tissm
# image_filename = image_folder + 'img_112.jpg' #Gold

image_filename_placeholder = tf.placeholder(tf.string)

feed_dict_to_use = {image_filename_placeholder: image_filename}

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

#sess=tf.Session()
with tf.Session() as sess:
    
    sess.run(initializer)

    saver.restore(sess, tf.train.latest_checkpoint(
        '/softPHOC_synthText/tf_model/model_fcn32s_118000.ckpt'))

    image_np, pred_np , probabilities_np = sess.run([image_tensor, pred, probabilities], feed_dict=feed_dict_to_use)
    #print sess.run(tf.shape(image_np))
    #print sess.run(tf.shape(pred_np))
    img_height = sess.run(tf.shape(image_np))[1]
    img_width = sess.run(tf.shape(image_np))[0]

query = 'n'
#query = 'communications'
#query = 'future'
# query = 'owndays'
# query = 'gold'
# query = 'soup'

chars = [char2int[x] for x in query]
binary_query = np.zeros((img_width, img_height))
prob_sum_query = np.zeros((img_width, img_height))
for i in chars:
    C_c_20 = probabilities_np.squeeze()[:, :, i] > 0.20
    binary_query += C_c_20
    prob_sum_query += probabilities_np.squeeze()[:, :, i]

binary_binary_query = binary_query != 0
binary_im = binary_binary_query.astype(np.uint8) * 255
im = cv2.imread(image_filename)

_, contours, _ = cv2.findContours(binary_im, 1, 1)
for c in contours:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
plt.imshow(im)
plt.show()