# coding: utf-8
import scipy
import scipy.io as sio
import re
from itertools import chain
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

# print 'reading gt file...'
# gt = sio.loadmat('gt.mat')
# print 'done'

def read_image(gt,img_index):
    imgFile = str(gt['imnames'][0][img_index][0])
    img = cv2.cvtColor(cv2.imread(imgFile), cv2.COLOR_BGR2RGB)
    return img


def get_words(gt,img_index):
    cur_txt = gt['txt'][0][img_index]
    all_words = list(chain.from_iterable([re.sub(r'\s+','\n', w).strip().split('\n') for w in cur_txt]))
    return all_words


def draw_boxes(img, boxes, c=(255,0,0)):
    for i in range(boxes.shape[-1]):
        pts = boxes[:,:,i].transpose().reshape(-1,1,2).astype(np.int32)
        cv2.polylines(img, [pts], True, c, 2)
    return img


def get_char_in_word(gt,img_id, word_id):
    cur_words = get_words(gt,img_id)
    lengths = [len(x) for x in cur_words]
    cumulative_lengths = np.cumsum([0] + lengths)
    start_word = cumulative_lengths[word_id]
    end_word = cumulative_lengths[word_id + 1]
    charBBs = gt['charBB'][0][img_index]
    return charBBs[:,:,start_word:end_word], cur_words[word_id]


def get_box_from_chars(gt,img_index, word_id):
    cur_chars, cur_word = get_char_in_word(gt,img_index, word_id)
    original_chars = cur_chars.copy()
    char_heights = np.array([cur_chars[1,:,x][2] - cur_chars[1,:,x][1] for x in range(cur_chars.shape[-1])])
    to_use_inds = np.where(char_heights>np.median(char_heights)*0.75)[0]
    cur_chars = cur_chars[:,:,to_use_inds]

    # obtain box from the first and last character
    first_char = cur_chars[:,:,0]
    last_char = cur_chars[:,:,-1]
    top_left = first_char[:,0]
    bottom_left = first_char[:,-1]
    top_right = last_char[:,1]
    bottom_right = last_char[:,-2]
    new_box = np.stack([top_left, top_right, bottom_right, bottom_left], 1)[:,:,np.newaxis]

    # check if points are outside the box
    contour_box = new_box.transpose().reshape(-1,1,2).astype(np.int32)

    m_up = (new_box[1,1] - new_box[1,0]) / (new_box[0,1] - new_box[0,0]) # m = (y2-y1)/(x2-x1)
    m_left = (new_box[1,0] - new_box[1,3]) / (new_box[0,0] - new_box[0,3]) # m = (y2-y1)/(x2-x1)
    m_right = (new_box[1,2] - new_box[1,1]) / (new_box[0,2] - new_box[0,1]) # m = (y2-y1)/(x2-x1)
    m_down = (new_box[1,3] - new_box[1,2]) / (new_box[0,3] - new_box[0,2]) # m = (y2-y1)/(x2-x1)

    q_up = new_box[1,0] - m_up*new_box[0,0] # q = y-mx
    q_left = new_box[1,3] - m_left*new_box[0,3] # q = y-mx
    q_right = new_box[1,1] - m_right*new_box[0,1] # q = y-mx
    q_down = new_box[1,2] - m_down*new_box[0,2] # q = y-mx

    # check all points
    all_points = cur_chars.transpose().reshape(-1,2)
    dists = [cv2.pointPolygonTest(contour_box,tuple(x),True) for x in all_points]
    #Test upper points
    all_points = cur_chars[:,0:2,:].transpose().reshape(-1,2)
    dists = [cv2.pointPolygonTest(contour_box,tuple(x),True) for x in all_points]
    if min(dists) <= 0:
        max_dist_ind = np.argmin(dists)
        m_parallel = m_up
        q_parallel = -m_parallel*all_points[max_dist_ind, 0] + all_points[max_dist_ind, 1]

        y_box = m_up * all_points[max_dist_ind, 0] + q_up
        if y_box > all_points[max_dist_ind, 1]:

            # top left
            intersect_left_x = (q_left - q_parallel) / (m_parallel - m_left)
            intersect_left_y = m_parallel * intersect_left_x + q_parallel
            new_box[0,0,0] = intersect_left_x
            new_box[1,0,0] = intersect_left_y

            # top right
            intersect_right_x = (q_right - q_parallel) / (m_parallel - m_right)
            intersect_right_y = m_parallel * intersect_right_x + q_parallel
            new_box[0,1,0] = intersect_right_x
            new_box[1,1,0] = intersect_right_y

    #Test lower points
    all_points = cur_chars[:,2:,:].transpose().reshape(-1,2)
    dists = [cv2.pointPolygonTest(contour_box,tuple(x),True) for x in all_points]
    if min(dists) <= 0:
        max_dist_ind = np.argmin(dists)
        m_parallel = m_down
        q_parallel = -m_parallel*all_points[max_dist_ind, 0] + all_points[max_dist_ind, 1]

        y_box = m_down * all_points[max_dist_ind, 0] + q_down
        if y_box < all_points[max_dist_ind, 1]:

            # bottom left
            intersect_left_x = (q_left - q_parallel) / (m_parallel - m_left)
            intersect_left_y = m_parallel * intersect_left_x + q_parallel
            new_box[0,3,0] = intersect_left_x
            new_box[1,3,0] = intersect_left_y

            # bottom right
            intersect_right_x = (q_right - q_parallel) / (m_parallel - m_right)
            intersect_right_y = m_parallel * intersect_right_x + q_parallel
            new_box[0,2,0] = intersect_right_x
            new_box[1,2,0] = intersect_right_y

    # check all points
    all_points = cur_chars.transpose().reshape(-1,2)
    dists = [cv2.pointPolygonTest(contour_box,tuple(x),True) for x in all_points]

    # add back removed small characters
    m_up = (new_box[1,1] - new_box[1,0]) / (new_box[0,1] - new_box[0,0]) # m = (y2-y1)/(x2-x1)
    m_down = (new_box[1,3] - new_box[1,2]) / (new_box[0,3] - new_box[0,2]) # m = (y2-y1)/(x2-x1)
    q_up = new_box[1,0] - m_up*new_box[0,0] # q = y-mx
    q_down = new_box[1,2] - m_down*new_box[0,2] # q = y-mx

    if 0 not in to_use_inds:
        # print 'missing first'
        first_char = original_chars[:,:,0]
        m_left_char = (first_char[1, 3] - first_char[1, 0]) / (first_char[0, 3] - first_char[0, 0]) # m = (y2-y1)/(x2-x1)
        q_left_char = first_char[1, 3] - m_left_char*first_char[0, 3] # q = y-mx
        # top left
        intersect_top_left_x = (q_left_char - q_up) / (m_up - m_left_char)
        intersect_top_left_y = m_up * intersect_top_left_x + q_up
        new_box[0,0,0] = intersect_top_left_x
        new_box[1,0,0] = intersect_top_left_y
        # bottom left
        intersect_bottom_left_x = (q_left_char - q_down) / (m_down - m_left_char)
        intersect_bottom_left_y = m_down * intersect_top_left_x + q_down
        new_box[0,3,0] = intersect_bottom_left_x
        new_box[1,3,0] = intersect_bottom_left_y

    if len(cur_word)-1 not in to_use_inds:
        # print 'missing last'
        last_char = original_chars[:,:,-1]
        m_right_char = (last_char[1, 1] - last_char[1, 2]) / (last_char[0, 1] - last_char[0, 2]) # m = (y2-y1)/(x2-x1)
        q_right_char = last_char[1, 2] - m_right_char*last_char[0, 2] # q = y-mx
        # top right
        intersect_top_right_x = (q_right_char - q_up) / (m_up - m_right_char)
        intersect_top_right_y = m_up * intersect_top_right_x + q_up
        new_box[0,1,0] = intersect_top_right_x
        new_box[1,1,0] = intersect_top_right_y
        # bottom left
        intersect_bottom_right_x = (q_right_char - q_down) / (m_down - m_right_char)
        intersect_bottom_right_y = m_down * intersect_bottom_right_x + q_down
        new_box[0,2,0] = intersect_bottom_right_x
        new_box[1,2,0] = intersect_bottom_right_y
        if last_char[1,2] > intersect_bottom_right_y:
            new_box[1,2,0] = last_char[1,2]
    return new_box


if __name__=='__main__':

    #gt_dir = '/path/to/datasets/synthtext/gt/'
    gt_dir = '/path/to/dataset/SynthText/gt'
    #pathToGtMat = '/path/to/datasets/synthtext/gt.mat'
    pathToGtMat = '/path/to/dataset/SynthText/SynthText/gt.mat'
    gtTXTPath_word = gt_dir + 'gt_word_polygon/'
    gtTXTPath_char = gt_dir + 'gt_char/'
    error_list_file = gt_dir + 'error_list.txt'
    gtINDX_list_file = gt_dir + 'gtINDX_list.txt'
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)

    error_list_fid = open(error_list_file, 'w')
    gtINDX_list_fid = open(gtINDX_list_file, 'w')
    # gtTXTPath_word = '/path/to/datasets/synthtext/gt/gt_word_polygon/'
    # gtTXTPath_char = '/path/to/datasets/synthtext/gt/gt_char/'

    gt = scipy.io.matlab.loadmat(pathToGtMat)
    files, txt, wordBB, charBB = gt['imnames'],gt['txt'],gt['wordBB'],gt['charBB']

    for img_index in tqdm(range(0, len(files[0]))):
        try:
            imgFolder = str(files[0][img_index]).split('/')[0].split("'")[-1]
            imgName = str(files[0][img_index]).split('/')[1].split("'")[0]
            gtName = imgName.split('.')[0]+'.txt'

            # for the word level detection
            if not os.path.exists(gtTXTPath_word+imgFolder):
                os.makedirs(gtTXTPath_word+imgFolder)

            word_gt = open(gtTXTPath_word+imgFolder+'/'+gtName, "w")
            words_list = []
            for w in txt[0][img_index]:
                words_list += re.sub(r'\s+','\n',w).strip().split('\n')

            for word_id in range(0, len(words_list)):
                bb = get_box_from_chars(gt,img_index, word_id)
                word_gt.write(str(int(bb[0][0]))+','+str(int(bb[1][0]))+','+
                      str(int(bb[0][1]))+','+str(int(bb[1][1]))+','+
                      str(int(bb[0][2]))+','+str(int(bb[1][2]))+','+
                      str(int(bb[0][3]))+','+str(int(bb[1][3]))+','+
                      str(words_list[word_id])+'\n')
            word_gt.close()

            gtINDX_list_fid.write(str(img_index) + '\n')
            print 'Saved img: ' + str(img_index)
            print files[0][img_index]
        except:
            error_list_fid.write(str(img_index) + '\n')
            print 'Skipping img: ' + str(img_index)

    error_list_fid.close()
    gtINDX_list_fid.close()