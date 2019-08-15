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
import time

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)#(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def hough_line(img, probabilities, use_probabilities):
  # Rho and Theta ranges
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * int(diag_len), int(num_thetas)), dtype=np.float32)
  y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      if use_probabilities:
            accumulator[int(rho), t_idx] += probabilities[y,x]
      else:
            accumulator[int(rho), t_idx] += 1

  return accumulator, thetas, rhos


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

from inference_prediction import adapt_network_for_any_size_input

number_of_classes = 38

image_folder = '/softPHOC_synthText/data/test_images/'
#image_filename = 'img_108.jpg' # less color contrast
#image_filename = image_folder + 'img_341.jpg' #soup soop
#image_filename = image_folder + 'img_440.jpg'   # bad heatmap response
#image_filename = image_folder + 'img_485.jpg'   #bad heatmap
#image_filename = image_folder + 'img_65.jpg'
#image_filename = image_folder + 'img_14.jpg'
#image_filename = image_folder + 'img_418.jpg'  #owndays
# image_filename = image_folder + 'img_119.jpg' #RObert Tissm
image_filename = image_folder + 'img_112.jpg' #Gold

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

    saver.restore(sess,"/softPHOC_synthText/tf_model/model_fcn32s_118000.ckpt")

    image_np, pred_np , probabilities_np = sess.run([image_tensor, pred, probabilities], feed_dict=feed_dict_to_use)
    #print sess.run(tf.shape(image_np))
    #print sess.run(tf.shape(pred_np))
    img_height = sess.run(tf.shape(image_np))[1]
    img_width = sess.run(tf.shape(image_np))[0]

#query = 'Robert'
#query = 'communications'
#query = 'future'
# query = 'owndays'
query = 'gold'

st = time.time()

chars = [char2int[x] for x in query]
binary_query = np.zeros((img_width,img_height))
prob_sum_query = np.zeros((img_width,img_height))
for i in chars:
    C_c_20 = probabilities_np.squeeze()[:,:,i] > 0.20
    binary_query += C_c_20
    prob_sum_query += probabilities_np.squeeze()[:,:,i]

binary_binary_query = binary_query != 0

USE_PROBABILITIES = True #False

accumulator, thetas, rhos = hough_line(binary_binary_query, prob_sum_query, USE_PROBABILITIES)
r,c = np.unravel_index(np.argmax(accumulator), accumulator.shape)

max_lines = 100
indexes = [np.unravel_index(x, accumulator.shape) for x in np.argsort(accumulator.ravel())[::-1]]
indexes = indexes[:max_lines]

im = cv2.imread(image_filename)
for rr, cc in indexes:
    rho = rhos[rr]
    theta = thetas[cc]

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 5000 * (-b))
    y1 = int(y0 + 5000 * (a))
    x2 = int(x0 - 5000 * (-b))
    y2 = int(y0 - 5000 * (a))

    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 5)

print "took " + str(time.time() - st)

plt.imshow(im)
plt.show()