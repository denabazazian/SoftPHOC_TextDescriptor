# -*- coding: utf-8 -*-
"""
Created on Apr 24 2018 9:53:07

@author: Dena Bazazian
"""

from __future__ import division
import scipy.io as sio
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from collections import defaultdict
from commands import getoutput as go
import glob, os
import re
import math
import multiprocessing
import time
from os import listdir
from glob import glob
from tqdm import tqdm

Synth_dir_main = '/path/to/datasets/synthtext/'

main_img_dir = Synth_dir_main + 'SynthText/'
main_gt_dir = Synth_dir_main + 'gt/gt_word_polygon/'
img_name_IoU = Synth_dir_main + 'gt/img_name_max_IoU_new.txt'

img_name_IoU_file = open(img_name_IoU, 'w')

gt_dirs = glob(main_gt_dir+'*')
counter = 0
padding = 1
ft = time.time()
flush_iters = 1000

for f in range(0,len(gt_dirs)):  #### To read all the Folders
    for gt_file in listdir(gt_dirs[f]):  ##### To read all the GTs

        # print gt_dirs[f]+'/'+gt_file
        s = time.time()
        gt = open(gt_dirs[f]+'/'+gt_file).read()
        numbWords = gt.count('\n')

        max_iou = -1.0

        if numbWords == 0:
            max_iou = -1.0
        elif numbWords == 1:
            max_iou = 0.0
        else:
            gt = gt.split('\n')
            crds = np.zeros((numbWords, 4, 2))
            for GTword in xrange(numbWords):
                if len(gt[GTword]) > 0:
                    # print transcription
                    crds[GTword, :, :] = np.array([[[int(gt[GTword].split(',')[0]), int(gt[GTword].split(',')[1])],
                                     [int(gt[GTword].split(',')[2]), int(gt[GTword].split(',')[3])],
                                     [int(gt[GTword].split(',')[4]), int(gt[GTword].split(',')[5])],
                                     [int(gt[GTword].split(',')[6]), int(gt[GTword].split(',')[7])]]], dtype=np.int32)

            max_x = int(np.max(crds[:, :, 0]))
            max_y = int(np.max(crds[:, :, 1]))
            min_x = np.maximum(int(np.min(crds[:, :, 0])), 0)
            min_y = np.maximum(int(np.min(crds[:, :, 1])), 0)
            width_crop = max_x - min_x + padding*2
            height_crop = max_y - min_y + padding*2
            word_mask = np.zeros((height_crop, width_crop, numbWords), dtype=np.uint8)

            crds = crds - np.array([min_x - padding, min_y - padding])

            for GTword in xrange(numbWords):
                    crd = crds[GTword, ...]
                    channel_word = np.array(word_mask[:, :, GTword])
                    crdInt = np.asarray([crd]).astype(np.int32)
                    cv2.fillPoly(channel_word, crdInt, 1)
                    word_mask[:, :, GTword] = channel_word

            bool_word_masks = word_mask.astype(np.bool)

            # bool_intersection = np.all(bool_word_masks, axis=-1)
            bool_intersection = np.sum(word_mask, axis=-1) > 1
            intersection = np.count_nonzero(bool_intersection)

            if intersection > 0.0:
                # Compute IoUs
                bool_union = np.any(bool_word_masks, axis=-1)

                for word_id in range(numbWords):
                    cur_intersection = np.count_nonzero(bool_intersection[bool_word_masks[:, :, word_id]])
                    cur_union = np.count_nonzero(bool_union[bool_word_masks[:, :, word_id]])
                    if cur_union == 0:
                        print 'skipping image ' + gt_dirs[f]+'/'+gt_file
                        max_iou = 1000
                        continue
                    cur_iou = cur_intersection/cur_union
                    if cur_iou > max_iou:
                        max_iou = cur_iou
            else:
                max_iou = 0.0

        counter += 1

        if max_iou != 1000:
            img_name_IoU_file.write(str(gt_dirs[f].split('/')[-1])+'/'+gt_file.split('.')[0]+','+str(max_iou) + '\n')

        if counter % flush_iters == 0:
            time_per_image = (time.time() - ft)/counter
            print str(counter) + ': flushing \t img per second: ' + str(1/time_per_image) + '\t remining time: ' + str((900000-counter)*time_per_image/60.0) + ' minutes'
            img_name_IoU_file.flush()

img_name_IoU_file.close()