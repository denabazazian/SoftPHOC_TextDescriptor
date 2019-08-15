# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 2018 15:07:21

@author: dena
"""

# python dtw_houghTransform_lines_bigram_CE_evaluation.py /data/icdar_ch4_testSet/img_*.jpg

import scipy
import scipy.io
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
import matplotlib.pyplot as plt
from PIL import Image
import time

from numpy import array, zeros, argmin, inf, ndim
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from nltk.metrics.distance import edit_distance
from dtw import fastdtw

threshold_bigram_heatmap = 0.005

def perspectiveTransfer(img, crd):
    width = np.max([(abs(crd[1, 0] - crd[0, 0])), (abs(crd[2, 0] - crd[3, 0]))], axis=0) # (top_left_X, top_right_X), (bottom_left_X, bottom_right_X)
    height = np.max([(abs(crd[0, 1] - crd[3, 1])), (abs(crd[1, 1] - crd[2, 1]))], axis=0) # (top_left_Y, top_right_Y), (bottom_left_X, bottom_right_Y)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    rect = cv2.getPerspectiveTransform(crd, dst)
    # print crd
    warped = cv2.warpPerspective(img, rect, (width, height))

    return warped, width, height


def visualize_phoc(word, phoc):
    fig = plt.figure()
    for (i, c) in enumerate(word):
        ax = fig.add_subplot(len(word), 1, i + 1)
        ax.plot(phoc[0, :, char2int[c]])
        ax.set_title('Letter: ' + c)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def synthesize(numClasses, word, width, height, vis=False):
    # Convert  string to list of integers.
    # chars = [(ord(x) - ord('a')+1) for x in word]
    chars = [char2int[x] for x in word]
    # And create the base 1D image over which we will histogram. This
    # image has the character value covering the *approximate* width
    # we expect it to occupy in the image (width / len(word)).
    base = np.zeros((width,))
    # for (c, x) in zip(chars, np.linspace(0, width, len(chars)+1)[:-1]):
    #    base[int(x):int(np.floor(x+(float(width)/len(chars))))] = c
    splits_char = np.linspace(0, width, len(chars) + 1)
    for (c, l, u) in zip(chars, splits_char[:-1], splits_char[1:]):
        base[int(l):int(u)] = c

    # Create the 1D PHOC.
    phoc = np.zeros((1, width, numClasses), dtype=np.float32)
    levels = len(word)
    # Loop over the desired subdivisions.
    for level in range(0, levels + 1):
        # We iterate over (low, hi) endpoint pairs in the original image.
        splits = np.linspace(0, width, level + 1)
        for (l, u) in zip(splits[:-1], splits[1:]):
            # And histogram values from the created image.
            hist = np.histogram(base[int(l):int(u)],
                                bins=numClasses,
                                range=(0, numClasses),
                                normed=True)[0]

            # Which we then *add* back into the PHOC we are
            # constructing into the x-range from which we computed the
            # histogram from.
            phoc[:, int(l):int(u), :] += hist

    # Conditionally visualize the non-zero PHOC channels.
    if vis:
        visualize_phoc(word, phoc)

    norms = np.abs(phoc).sum(axis=-1, keepdims=True)
    np.place(norms, norms == 0, 1)
    phoc /= norms

    if vis:
        visualize_phoc(word, phoc)

    phoc_new = np.zeros((height, width, numClasses))
    phoc_new[:, :, :] = phoc[None, :, :]

    return phoc_new

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

def get_houghTransform_points(img,query,probabilities_np,img_width,img_height ,max_lines = 5,USE_PROBABILITIES = False):
    chars = [char2int[x] for x in query]
    binary_query = np.zeros((img_width,img_height))
    prob_sum_query = np.zeros((img_width,img_height))
    for i in chars:
        C_c_20 = probabilities_np.squeeze()[:,:,i] > 0.20
        binary_query += C_c_20
        prob_sum_query += probabilities_np.squeeze()[:,:,i]

    partial_prods = [[] for x in range(len(chars) - 1)]
    for i_ in range(1, len(chars)):
        partial_prods[i_-1] = np.prod(probabilities_np.squeeze()[:, :, [chars[i_-1], chars[i_]]], axis=-1, keepdims=True)
    prob_sum_query = np.squeeze(np.mean(np.array(partial_prods), axis=0))
    binary_query = prob_sum_query > threshold_bigram_heatmap

    binary_binary_query = binary_query != 0

    ###USE_PROBABILITIES = False
    # accumulator, thetas, rhos = hough_line(binary_binary_query, prob_sum_query, USE_PROBABILITIES)
    # r,c = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    # ###max_lines = 100
    # indexes = [np.unravel_index(x, accumulator.shape) for x in np.argsort(accumulator.ravel())[::-1]]
    # indexes = indexes[:max_lines]
    # ###im = cv2.imread(image_filename)

    binary_im = binary_binary_query.astype(np.uint8) * 255

    # plt.imshow(binary_im)
    # plt.imshow(img,alpha=0.5)
    # plt.title(('%s')%(query))
    # plt.show()

    #lines = cv2.HoughLinesP(binary_im, 1, np.pi / 180, max_lines, 100, 10)
    #lines = cv2.HoughLinesP(image = binary_im,  rho = 1, theta =np.pi / 180, threshold=5, minLineLength=10, maxLineGap=10)
    lines = cv2.HoughLinesP(image = binary_im,  rho = 1, theta =np.pi / 180, threshold=5, minLineLength=10, maxLineGap=10)

    houghTransform_points = []

    if lines is not None: 
        for l in lines:
            eachPoints=[]
            for x1, y1, x2, y2 in l:
                eachPoints.append(x1)
                eachPoints.append(y1)
                eachPoints.append(x2)
                eachPoints.append(y2)

            houghTransform_points.append(eachPoints)

    return houghTransform_points

def get_points_lines(houghLines_points,probabilities_np):
    x0 = houghLines_points[0]
    y0 = houghLines_points[1]
    x1 = houghLines_points[2]
    y1 = houghLines_points[3]

    if x0 != x1:
        m = float((y1 - y0)) / (x1 - x0)
    else:
        m = 0
    q = y0 - (m * x0)

    # find the biggest dimention an loop over it to find all the pixels
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    # find all the points in the line based on the biggest dimension
    line_points = []
    if width >= height:
        # big_dim = width
        for xx in range(min(x0,x1), max(x0,x1)):
            each_point = []
            each_point.append(xx)
            yy = int(q + (m * xx))
            each_point.append(yy)
            line_points.append(each_point)

    else:
        # big_dim = height
        for yy in range(min(y0,y1), max(y0,y1)):
            each_point = []
            # each_point.append(xx)
            if m == 0:
                xx = x0
            else:
                xx = int((y0 - q) / (m))
            each_point.append(xx)
            each_point.append(yy)
            line_points.append(each_point)

    pred_points = []
    for point in line_points:
        #   pred_points_eachChannel = []
        #   pred_points_eachChannel.append(pred[point[0],point[1],:])
        #   pred_points.append(pred_points_eachChannel)
        pred_points.append(probabilities_np.squeeze()[point[1], point[0], :])

    # print  pred_points
    pred_points_array = np.array(pred_points)

    return pred_points_array,line_points

def get_axis_oriented_proposals(polygone_proposals,logits):

    numProposals = len(polygone_proposals)
    bboxes = np.empty([numProposals, 4]).astype(np.float).astype(np.int)  # TODO: this is ugly, had some problems casting to int

    logitsbb = []
    # top_left(x,y), top_right(x,y), bottom_right(x,y), bottom_left(x,y),
    for l in range(0, len(polygone_proposals)):

        bboxes[l, 0] = np.min([int(polygone_proposals[l][0]), int(polygone_proposals[l][6])], axis=0)  # left
        bboxes[l, 1] = np.min([int(polygone_proposals[l][3]), int(polygone_proposals[l][1])], axis=0)  # top
        bboxes[l, 2] = np.max([int(polygone_proposals[l][4]), int(polygone_proposals[l][2])], axis=0)  # right
        bboxes[l, 3] = np.max([int(polygone_proposals[l][5]), int(polygone_proposals[l][7])], axis=0)  # bottom

        crd = np.array([[int(polygone_proposals[l][0]), int(polygone_proposals[l][1])],
                        [int(polygone_proposals[l][2]), int(polygone_proposals[l][3])],
                        [int(polygone_proposals[l][4]), int(polygone_proposals[l][5])],
                        [int(polygone_proposals[l][6]), int(polygone_proposals[l][7])]], dtype="float32")

        warped_logits, width_gt, height_gt = perspectiveTransfer(logits[0, ...], crd)

        if warped_logits.shape == (720, 1280, 38):
            raise Exception("Error! Shape is " + str(warped_logits.shape))

        logitsbb.append(warped_logits)

    bbox_widths = bboxes[:, 2] - bboxes[:, 0]
    bbox_heights = bboxes[:, 3] - bboxes[:, 1]

    return bboxes, logitsbb, bbox_widths, bbox_heights

def box_line_overlap(bb, line_points):
    top_left = [bb[0], bb[1]]
    bottom_left = [bb[6], bb[7]]
    top_right = [bb[2], bb[3]]
    bottom_right = [bb[4], bb[5]]
    bb_arr = np.stack([top_left, top_right, bottom_right, bottom_left], 1)[:,:,np.newaxis]  
    bb_arr = bb_arr.transpose().reshape(-1,1,2).astype(np.int32)
  
    overlapping_pixels = len(np.where(np.array([cv2.pointPolygonTest(bb_arr, tuple(x), False) for x in line_points])==1)[0])
    line_length = float(len(line_points))
  
    return overlapping_pixels / line_length

if __name__ == '__main__':
    print '----------------'
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

    sys.path.append("/tf-image-segmentation/")
    sys.path.append("/models/slim")

    print sys.path

    #img_bbx_path = "/home/dena/Projects/softPHOC_synthText/word_spotting/results/dtw/dtw_houghTransform_img/"
    res_path = "/softPHOC_synthText/word_spotting/results/dtw/dtw_houghTransform_CE_res/"

    # if not os.path.exists(img_bbx_path):
    #     os.makedirs(img_bbx_path)
    # if not os.path.exists(res_path):
    #     os.makedirs(res_path)

    imgNum = 0
    all_words = 0
    #box_line_overlap_res_totall = []
    box_line_overlap_res_total = 0.0

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #os.environ["CUDA_VISIBLE_DEVICES"] = ''

    slim = tf.contrib.slim

    from tf_image_segmentation.models.fcn_32s import FCN_32s
    #from fcn_32s import FCN_32s

    from tf_image_segmentation.utils.inference_prediction import adapt_network_for_any_size_input

    number_of_classes = 38

    '''---------------------------------------------------------------------------------------------------------------------
    Cross entropy model, used for evaluating similarity_type 2 in gpu
    we take the negative cross-entropy to have a similarity instead of an error and make it compatible with other metric
    ---------------------------------------------------------------------------------------------------------------------'''
    i1 = tf.placeholder(tf.float32)
    i2 = tf.placeholder(tf.float32)
    ce = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=i1, logits=i2), axis=-1)
    '''---------------------------------------------------------------------------------------------------------------------
    Choose a similarity type for matching proposals and query
    0: cosine similarity
    1: histogram intersection
    2: cross-entropy
    3: histogram intersection over union
    4: DTW
    5: DTW+CE
    ---------------------------------------------------------------------------------------------------------------------'''
    # similarity_type = 0 #cosine
    # similarity_type = 1 #HistogramIntersection
    #similarity_type = 3  # CrossEntropy
    #similarity_type = 4  # DTW
    similarity_type = 5  # DTW+CrossEntropy
    take_always_argmax = False  # True #False
    min_proposal_area = 50

    image_filename_placeholder = tf.placeholder(tf.string)
    # feed_dict_to_use = {image_filename_placeholder: image_filename}
    image_tensor = tf.read_file(image_filename_placeholder)

    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

    # Fake batch for image and annotation by adding leading empty axis.
    image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

    #after adaptation, network returns final labels and not logits (adapt_netwrk)
    FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)

    upsampled_logit, fcn_16s_variables_mapping_train = FCN_32s(image_batch_tensor=image_batch_tensor,
                                                               number_of_classes=number_of_classes,
                                                               is_training=False, reuse=False)
    imgNum +=1

    upsampled_logit_resized = tf.image.resize_images(upsampled_logit, (720, 1280))

    pred = tf.argmax(upsampled_logit, dimension=3)
    probabilities = tf.nn.softmax(upsampled_logit)

    initializer = tf.local_variables_initializer()

    saver = tf.train.Saver()

    cmap = plt.get_cmap('bwr')

    save_heatmaps_path = './heatmaps_icdar/'
    if not os.path.exists(save_heatmaps_path):
        os.makedirs(save_heatmaps_path)

    #sess=tf.Session()
    with tf.Session() as sess:

        sess.run(initializer)

        saver.restore(sess,"/softPHOC_synthText/tf_model/tf_model_3_backup_118000/model_fcn32s_118000.ckpt")

        for image_filename in sys.argv[1:]:
            print image_filename

            img = cv2.imread(image_filename)
            feed_dict_to_use = {image_filename_placeholder: image_filename}

            image_np, pred_np , probabilities_np, logits = sess.run([image_tensor, pred, probabilities, upsampled_logit_resized], feed_dict=feed_dict_to_use)

            #print sess.run(tf.shape(image_np))
            #print sess.run(tf.shape(pred_np))
            img_height = sess.run(tf.shape(image_np))[1]
            img_width = sess.run(tf.shape(image_np))[0]


            fileName = image_filename.split('.')[0].split('/')[-1]

            res_file = open(res_path + 'res_' + fileName + '.txt', 'w')

            voc = open(image_filename.split('.')[0] + '.txt').read()  # for the GT ch4 testing
            #voc = open(img_name.split('img')[0] + 'voc_' + (img_name.split('/')[-1].split('.')[0]) + '.txt').read()  # for the dictionary and distractor

            numQWords = voc.count('\n')
            voc = re.sub(r'[^\x00-\x7f]', r'', voc)
            voc = voc.split('\r\n')

            query_dict = {}
            used_idx = []
            idnqw = 0
            for nqw in range(0, numQWords):
            #for nqw in range(1, 2): # just for word FUTURE
                if (len(voc[nqw]) > 0 and voc[nqw].split(',')[-1].strip() != '###'):
                    idnqw += 1
                    qword = voc[nqw].split(',')[-1]
                    if qword == '':
                        continue
                    print 'query word is %s' % qword

                    if query_dict.has_key(qword):
                        query_dict[qword] += 1
                    else:
                        query_dict[qword] = 1

                    bb = [int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1]),int(voc[nqw].split(',')[2]),
                          int(voc[nqw].split(',')[3]),int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5]),
                          int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])]

                    st = time.time()
                    houghTransform_proposals = get_houghTransform_points(img,qword,probabilities_np,img_width,img_height ,max_lines =5,USE_PROBABILITIES = False)
                    if len(houghTransform_proposals) == 0:
                        continue

                    # Build the soft-phoc representation of the query
                    target_size = (int(len(qword))*10, 1)  # target size is LEN_QUERY_WORD x 1 x 38
                    # target_size = (int(len(qword)))
                    #softphoc_query = synthesize(number_of_classes, qword, width=target_size[0], height=target_size[1], vis=False)
                    softphoc_query = synthesize(number_of_classes, qword, width=target_size[0], height=target_size[1], vis=False)
                    # resize all the proposals to target_size and compute the similarity with the query soft-phoc
                    if similarity_type == 0:  # cosine distance
                        softphoc_query = np.ravel(softphoc_query)
                        resized_preds = [np.ravel(cv2.resize(x, (target_size[0], target_size[1]))) for x in
                                         softphoc_proposal_preds]
                        similarities = [np.dot((softphoc_query / norm(softphoc_query)), (x / norm(x))) for x in
                                        resized_preds]
                        simi_thresh = 0.65
                    elif similarity_type == 1:  # histogram intersection
                        resized_preds = [(cv2.resize(x, (target_size[0], target_size[1]))) for x in
                                         softphoc_proposal_preds]
                        similarities = ([np.sum(np.minimum(x, softphoc_query)) / len(qword) for x in resized_preds])
                    elif similarity_type == 2:  # cross-entropy
                        # print "bboxes is:"
                        # print bboxes
                        # logitsbb = [logits[0, b[1]:b[3], b[0]:b[2], :] for b in bboxes.astype(np.int)]
                        resized_logits = [(cv2.resize(x, (target_size[0], target_size[1]))) for x in logitsbb]
                        logits_ce = np.squeeze(np.array(resized_logits))
                        labels_ce = np.tile(softphoc_query, (num_proposals, 1, 1))
                        # print "logit_ce is:"
                        # print logits_ce
                        # print " label_ce is:"
                        # print labels_ce
                        similarities = np.squeeze(sess.run([ce], feed_dict={i1: labels_ce, i2: logits_ce}))
                        simi_thresh = -2.00
                    elif similarity_type == 3:  # histogram intersection over union
                        resized_preds = [(cv2.resize(x, (target_size[0], target_size[1]))) for x in
                                         softphoc_proposal_preds]
                        similarities = ([np.sum(np.minimum(x, softphoc_query) / np.maximum(x, softphoc_query)) / len(qword) for x in
                         resized_preds])

                    elif similarity_type == 4: #dtw
                            similarities = []
                            costs = []
                            accs = []
                            paths = []
                            im = np.copy(img)
                            for line in houghTransform_proposals:
                                cv2.line(im, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
                                pred_points_array,_ = get_points_lines(line, probabilities_np)
                                # resized_preds = [(cv2.resize(x, (target_size[0]))) for x in pred_points_array]
                                # similarities = -np.mean([fastdtw(softphoc_query[:, :, x].transpose(), resized_preds[0][:, :, x].transpose(), 'euclidean')[0] for x in range(38)])
                                dist, cost, acc, path = fastdtw(np.squeeze(softphoc_query), pred_points_array,'euclidean')
                                similarities.append(-dist)
                                costs.append(cost)
                                accs.append(acc)
                                paths.append(path)
                            #print 'dtw'

                            # plt.imshow(im)
                            # plt.title(qword)
                            # plt.show()

                    elif similarity_type == 5: #dtw + ce
                            similarities = []
                            costs = []
                            accs = []
                            paths = []
                            im = np.copy(img)
                            for line in houghTransform_proposals:
                                cv2.line(im, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
                                pred_points_array,_ = get_points_lines(line, probabilities_np)
                                # resized_preds = [(cv2.resize(x, (target_size[0]))) for x in pred_points_array]
                                # similarities = -np.mean([fastdtw(softphoc_query[:, :, x].transpose(), resized_preds[0][:, :, x].transpose(), 'euclidean')[0] for x in range(38)])
                                dist, cost, acc, path = fastdtw(np.squeeze(softphoc_query), pred_points_array,'euclidean')
                                labels_ce = softphoc_query[:, path[0], :]
                                logits_ce = pred_points_array[np.newaxis, path[1], :]
                                ce_sim = np.squeeze(sess.run([ce], feed_dict={i1: labels_ce, i2: logits_ce}))
                                similarities.append(ce_sim)
                                costs.append(cost)
                                accs.append(acc)
                                paths.append(path)
                            #print 'dtw_CE'

                            # plt.imshow(im)
                            # plt.title(qword)
                            # plt.show()

                    else:
                        raise Exception('Unknown similarity type')

                    sorted_idx = np.argsort(similarities)
                    sorted_paths = [paths[x] for x in sorted_idx]

                    if take_always_argmax:
                        idx = sorted_idx[-1]
                    else:
                        # idx = sorted_idx[-query_dict[qword]]
                        idx = sorted_idx[np.maximum(-query_dict[qword], -len(sorted_idx))]

                    # print "similarity is:"
                    # print similarities
                    # print "similarity[idx] is:"
                    # print similarities[idx]

                    try:
                        sim = similarities[idx]
                    except:
                        sim = similarities

                    res = houghTransform_proposals[idx]

                    res_file.write("%d,%d,%d,%d,%s\r\n" % (res[0], res[1], res[2], res[3], qword)) 
                    
                    res_line = [res[0], res[1], res[2], res[3]]
                    _,res_line_points =  get_points_lines(res_line, probabilities_np) 

                    box_line_overlap_res = box_line_overlap(bb,res_line_points)

                    print "box_line_overlap:"
                    print box_line_overlap_res 
                    #box_line_overlap_res_totall.append(box_line_overlap_res)
                    #box_line_overlap_res_totall_average = np.average(box_line_overlap_res_totall)

                    box_line_overlap_res_total += box_line_overlap_res
                    all_words +=1
                    box_line_overlap_res_totall_average = box_line_overlap_res_total/all_words
                    print "total average:"
                    print box_line_overlap_res_totall_average

                    im = np.copy(img)

                    # pts = np.array([[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])],[int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[3])],[int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5])],[int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])],[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])]], np.int32)
                    # pts = pts.reshape((-1,1,2))
                    # cv2.polylines(img,[pts],True,(0,0,255),2)

                    cv2.line(im, (res[0], res[1]), (res[2], res[3]), (0, 255, 0), 2)
                    # #cv2.putText(im, qword, (res[0], res[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0, 255), 2)
                    
                    plt.imshow(im)
                    plt.title(qword)
                    plt.show()
            # cv2.imwrite(img_bbx_path + fileName + '.png', im)
                    print "took: " + str(time.time() - st)
        print "done!"