# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 2018 18:38:21

@author: dena
"""

# cd ~/path/to/softPHOC_synthText/word_spotting/codes/save_hm
# python visualize_mat.py /path/to/char_pred_hm/n02746978_10031.mat

from __future__ import division
import scipy
import scipy.io
import cv2
import sys
import numpy as np
from glob import glob
from PIL import Image
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.misc import imresize, imsave, toimage


if __name__=='__main__':

    #save_hm = '/path/to/hm/soft_phoc/hm_vis/'
    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
    img_path = "/path/to/datasets/context_images/JPEGImages/"

    for character_map in sys.argv[1:]:

        #character_map = "/path/to/char_pred_hm/n02746978_11500.mat"

        map_mat = scipy.io.loadmat(character_map)

        #map_mat.keys()
        #['character_prediction', '__version__', '__header__', '__globals__']

        img_name = character_map.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path+img_name+'.jpg')

        for i in range (0,38):
           #plt.imshow(map_mat['character_prediction'][:,:,i]) 
           plt.imshow(map_mat['character_prediction'][:,:,i], vmin=0, vmax=1)
           plt.colorbar()
           plt.imshow(img,alpha=.5)
           plt.title(('%d,%s')%(i,alphabet[i]))
           plt.show()


        # for i in range(0,38):
        #     #fig = plt.figure()
        #     fig, axs = plt.subplots(2,1)
        #     #plt.imshow(softPHOC[:,:,i], vmin = 0 , vmax = 1)
        #     #plt.imshow(map_mat['softphoc_query'][:,:,i])
        #     axs[0].imshow(map_mat['softphoc_query'][:,:,i], vmin=0, vmax=1)
        #     axs[0].set_title(('annotation,%d,%s')%(i,alphabet[i])) 
        #     axs[0].axis('off')
        #     axs[1].imshow(map_mat['prediction'][:,:,i], vmin=0, vmax=1)
        #     #plt.imshow(img,alpha=.5)
        #     #plt.colorbar()   
        #     axs[1].set_title(('prediction,%d,%s')%(i,alphabet[i]))
        #     axs[1].axis('off')
        #     fig.savefig(save_hm+"hm_{}_{}.png".format(i, alphabet[i]))
        #     plt.close(fig)



