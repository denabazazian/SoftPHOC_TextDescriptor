from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
#import torch
#from torch.utils.data import Dataset
from mypath import Path

base_dir=Path.db_root_dir('synthText')
print('base_dir is:{}'.format(base_dir))
split='train'
image_dir = os.path.join(base_dir, 'SynthText')
print('image_dir is:{}'.format(image_dir))
cat_coord_bbx_dir = os.path.join(base_dir, 'gt_word_polygon')
print('cat_coord_bbx_dir:{}'.format(cat_coord_bbx_dir))

splits_dir = base_dir
with open(os.path.join(os.path.join(splits_dir, split + '.txt')), "r") as f:
    #lines = f.read().splitlines()
    lines = [l for l in (line.strip() for line in f) if l]
for ii, line in enumerate(lines):
    try:
        #print('line is: {}'.format(line))
        image = os.path.join(image_dir, line+'.jpg')
        cat = os.path.join(cat_coord_bbx_dir, line+'.txt') 
        #print('_image is : {}'.format(image))
        #print('_cat is: {}'.format(cat))
        assert os.path.isfile(image)
        assert os.path.isfile(cat)
        img = cv2.imread(image)
        imcvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv_size = lambda img: tuple(img.shape[1::-1])
        width_main, height_main = cv_size(img)
    except:
        print('error on image {}: {}'.format(ii, image))