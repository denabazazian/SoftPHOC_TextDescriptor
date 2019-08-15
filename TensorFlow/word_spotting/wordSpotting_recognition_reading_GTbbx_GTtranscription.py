# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 2018 19:07:21

@author: dena
"""

# python wordSpotting_recognition_reading_GTbbx_GTtranscription.py /path/tos/softPHOC_synthText/word_spotting/data/icdar_ch4_testSet/img_*.jpg

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

def visualize_phoc(word, phoc):
    fig = plt.figure()
    for (i, c) in enumerate(word):
        ax = fig.add_subplot(len(word), 1, i+1)
        ax.plot(phoc[0, :, char2int[c]])
        ax.set_title('Letter: ' + c)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def synthesize(word, width, height, vis=False):
    # Convert  string to list of integers.
    #chars = [(ord(x) - ord('a')+1) for x in word]
    chars = [char2int[x] for x in word]
    # And create the base 1D image over which we will histogram. This
    # image has the character value covering the *approximate* width
    # we expect it to occupy in the image (width / len(word)).
    base = np.zeros((width,))
    #for (c, x) in zip(chars, np.linspace(0, width, len(chars)+1)[:-1]):
    #    base[int(x):int(np.floor(x+(float(width)/len(chars))))] = c
    splits_char = np.linspace(0, width, len(chars)+1)
    for(c,l,u) in zip(chars,splits_char[:-1],splits_char[1:]):
        base[int(l):int(u)] = c

    # Create the 1D PHOC.
    phoc = np.zeros((1,width, numClasses), dtype=np.float32)
    levels = len(word)
    # Loop over the desired subdivisions.
    for level in range(0,levels+1):
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
    np.place(norms,norms==0,1)
    phoc /= norms

    if vis:
        visualize_phoc(word, phoc)

    phoc_new = np.zeros((height,width,numClasses))
    phoc_new[:,:,:] = phoc[None,:,:]

    return phoc_new


def perspectiveTransfer(img, crd):
    width = np.max([(abs(crd[1, 0] - crd[0, 0])), (abs(crd[2, 0] - crd[3, 0]))], axis=0)
    height = np.max([(abs(crd[0, 1] - crd[3, 1])), (abs(crd[1, 1] - crd[2, 1]))], axis=0)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    rect = cv2.getPerspectiveTransform(crd, dst)
    # print crd
    warped = cv2.warpPerspective(img, rect, (width, height))

    return warped, width, height




if __name__=='__main__':

    '''---------------------------------------------------------------------------------------------------------------------
    defining classes and path to GTS
    ---------------------------------------------------------------------------------------------------------------------'''

    char2int = {'a': 1, 'A': 1, 'b': 2, 'B': 2, 'c': 3, 'C': 3, 'd': 4, 'D': 4, 'e': 5, 'E': 5, 'f': 6, 'F': 6, 'g': 7,
                'G': 7, 'h': 8, 'H': 8, 'i': 9, 'I': 9, 'j': 10, 'J': 10, 'k': 11, 'K': 11, 'l': 12, 'L': 12, 'm': 13,
                'M': 13, 'n': 14, 'N': 14, 'o': 15, 'O': 15, 'p': 16, 'P': 16, 'q': 17, 'Q': 17, 'r': 18, 'R': 18,
                's': 19, 'S': 19, 't': 20, 'T': 20, 'u': 21, 'U': 21, 'v': 22, 'V': 22, 'w': 23, 'W': 23, 'x': 24,
                'X': 24, 'y': 25, 'Y': 25, 'z': 26, 'Z': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32,
                '7': 33, '8': 34, '9': 35, '0': 36, '!': 37, '$': 37, '>': 37, '<': 37, '.': 37, ':': 37, '-': 37,
                '_': 37, '(': 37, ')': 37, '[': 37, ']': 37, '{': 37, '}': 37, ',': 37, ';': 37, '#': 37, '?': 37,
                '%': 37, '*': 37, '/': 37, '@': 37, '^': 37, '&': 37, '=': 37, '+': 37, '€': 37, "'": 37, '`': 37,
                '"': 37, '\\': 37, '\xc2': 37, '\xb4': 37, ' ': 37, '\xc3': 37, '\x89': 37}  # '\':37

    numClasses = 38

    '''---------------------------------------------------------------------------------------------------------------------
    defining batches
    ---------------------------------------------------------------------------------------------------------------------'''

    test_image_id = tf.placeholder(tf.int32)

    img_batch = tf.placeholder(tf.uint8, (1, 720, 1280, 3))
    width_img = tf.placeholder(tf.int32)
    height_img = tf.placeholder(tf.int32)

    '''---------------------------------------------------------------------------------------------------------------------
    GPU stuff
    ---------------------------------------------------------------------------------------------------------------------'''

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True

    '''---------------------------------------------------------------------------------------------------------------------
    Set paths and create log folder
    ---------------------------------------------------------------------------------------------------------------------'''

    sys.path.append("/softPHOC_tensorflow/slim/")  # for reading the net
    sys.path.append("/softPHOC_tensorflow/models/")
    sys.path.append("/home/dena/Projects/softPHOC_tensorflow/softPHOC_utils/")
    
    checkpoints_dir = '/tensorflow-FCN-textNonText/checkpoints'

    number_of_classes = 38

    slim = tf.contrib.slim
    checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

    from fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8

    '''---------------------------------------------------------------------------------------------------------------------
    Process batch with FCN_32s
    ---------------------------------------------------------------------------------------------------------------------'''
    is_training = tf.placeholder(tf.bool)

    upsampled_logits_batch, vgg_16_variables_mapping = FCN_32s(image_batch_tensor=img_batch,
                                                               number_of_classes=number_of_classes,
                                                               is_training=is_training)

    '''---------------------------------------------------------------------------------------------------------------------
    Give a range to logits
    ---------------------------------------------------------------------------------------------------------------------'''

    margin_logits = tf.constant(10.0)
    upsampled_logits_batch = tf.maximum(tf.minimum(upsampled_logits_batch, margin_logits), -margin_logits)

    '''---------------------------------------------------------------------------------------------------------------------
    Mask the output of the network and compute loss
    ---------------------------------------------------------------------------------------------------------------------'''
    # The FCN_32s output is a multiplication of 32. So, it should be resized as the img_batch before computing loss
    upsampled_logits_batch_resized = tf.image.resize_images(upsampled_logits_batch, (720, 1280))
    # upsampled_logits_batch_resized = upsampled_logits_batch

    with tf.name_scope("losses") as scope:
        logit_TB_ch0 = upsampled_logits_batch_resized[:, :, :, 0]
        logit_TB_ch1 = tf.reduce_sum(upsampled_logits_batch_resized[:, :, :, 1:], axis=-1)
        logit_TB = tf.stack((logit_TB_ch0, logit_TB_ch1), axis=3)

    # get predictions
    with tf.name_scope("predictions") as scope:
        char_probabilities = tf.nn.softmax(upsampled_logits_batch_resized)
        char_class_predictions = tf.argmax(char_probabilities, dimension=3)
        text_probabilities = tf.nn.softmax(logit_TB)
        text_class_predictions = tf.argmax(text_probabilities, dimension=3)

    '''---------------------------------------------------------------------------------------------------------------------
                                            GRAPH DEFINITION ENDS HERE
    ---------------------------------------------------------------------------------------------------------------------'''

    '''---------------------------------------------------------------------------------------------------------------------
    Init the network and load previous checkpoint
    ---------------------------------------------------------------------------------------------------------------------'''

    vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)
    init_fn = slim.assign_from_checkpoint_fn(model_path=checkpoint_path, var_list=vgg_16_without_fc8_variables_mapping)

    '''---------------------------------------------------------------------------------------------------------------------
    Test the network!
    ---------------------------------------------------------------------------------------------------------------------'''

    sess = tf.Session(config=config)

    model_variables = slim.get_model_variables()
    saver = tf.train.Saver(model_variables)

    # init variables
    init_fn(sess)
    global_vars_init_op = tf.global_variables_initializer()
    local_vars_init_op = tf.local_variables_initializer()
    combined_op = tf.group(global_vars_init_op, local_vars_init_op)
    sess.run(combined_op)

    saver.restore(sess, tf.train.latest_checkpoint(
        '/softPHOC_synthText/tf_model/tf_model_3_backup_118000/'))

    res_path = "/softPHOC_synthText/word_spotting/results/recognition_GTbbx/Reading_gtbbx_CE2_GTtranscription/"

    imgNum = 0

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
    ---------------------------------------------------------------------------------------------------------------------'''
    #similarity_type = 0 #cosine
    #similarity_type = 1 #HistogramIntersection
    similarity_type = 2 #CrossEntropy
    take_always_argmax = False #True #False
    min_proposal_area = 1200

    correct_words = 0
    num_words = 0
    accuracy_at_N = 0
    N = 1 #3

    for img_name in sys.argv[1:]:
        print img_name

        img = cv2.imread(img_name)
        cv_size = lambda img: tuple(img.shape[1::-1])
        width_main, height_main = cv_size(img)

        [test_text_prob, test_text_cls_prob, test_char_prob, test_char_pred, logits] = sess.run(
            [text_probabilities, text_class_predictions, char_probabilities, char_class_predictions, upsampled_logits_batch_resized],
            feed_dict={img_batch: np.expand_dims(img, 0), is_training: False})

        # make the result file
        fileName = img_name.split('.')[0].split('/')[-1]
        res_file = open(res_path + 'res_' + fileName + '.txt', 'w')

        # Read GT bounding boxes
        gt_file = open(img_name.split('.')[0] + '.txt').read()  # for the GT ch4 testing

        numGTs = gt_file.count('\n')
        gt_file = re.sub(r'[^\x00-\x7f]', r'', gt_file)
        gt_file = gt_file.split('\r\n')

        for l in range(0, numGTs):
            if (len(gt_file[l])>0 and gt_file[l].split(',')[-1].strip()!='###'):
                transcription = gt_file[l].split(',')[-1]
                # print transcription

                gt_left = np.min([int(gt_file[l].split(',')[0]),int(gt_file[l].split(',')[6])],axis=0) #left
                gt_top = np.min([int(gt_file[l].split(',')[1]),int(gt_file[l].split(',')[3])],axis=0) #top
                gt_right = np.max([int(gt_file[l].split(',')[2]),int(gt_file[l].split(',')[4])],axis=0) #right
                gt_bottom = np.max([int(gt_file[l].split(',')[5]),int(gt_file[l].split(',')[7])],axis=0) #bottom



                crd = np.array([[int(gt_file[l].split(',')[0]), int(gt_file[l].split(',')[1])],
                            [int(gt_file[l].split(',')[2]), int(gt_file[l].split(',')[3])],
                            [int(gt_file[l].split(',')[4]), int(gt_file[l].split(',')[5])],
                            [int(gt_file[l].split(',')[6]), int(gt_file[l].split(',')[7])]], dtype="float32")

                warped_logits, width_gt, height_gt = perspectiveTransfer(logits[0,...], crd)
                #sub_softPHOC = synthesize(transcription, int(width), int(height), vis=False)

                #voc = open(img_name.split('img')[0]+'voc_'+(img_name.split('/')[-1].split('.')[0])+'.txt').read()   # for the dictionary and distractor

                # numQWords = voc.count('\n')
                # voc = re.sub(r'[^\x00-\x7f]', r'', gt_file)
                # voc = voc.split('\r\n')

                voc = gt_file
                numQWords = numGTs

                # Read the text proposals of the correspond image
                csvName = img_name.split('.')[0] + '.csv'
                fileName = img_name.split('.')[0].split('/')[-1]

                query_dict = {}
                used_idx = []
                idnqw = 0

                queries = []
                sims = []

                for nqw in range(0, numQWords):
                    if (len(voc[nqw]) > 0 and voc[nqw].split(',')[-1].strip() != '###'):
                        idnqw += 1
                        #qword = voc[nqw].split(',')[-1]
                        qword = voc[nqw].split(',')[8]
                        if qword == '':
                            continue
                        #print 'query word is %s' % qword

                        if query_dict.has_key(qword):
                            query_dict[qword] += 1
                        else:
                            query_dict[qword] = 1

                        # Build the soft-phoc representation of the query
                        target_size = (int(width_gt), int(height_gt))   #(int(len(qword)), 1)  # target size is LEN_QUERY_WORD x 1 x 38
                        softphoc_query = synthesize(qword, width=target_size[0], height=target_size[1], vis=False)
                        # softphoc_proposal_preds = [test_char_prob[0, gt_top:gt_bottom, gt_left:gt_right, :] ]
                        # softphoc_proposal_preds = warped_logits

                        # resize all the proposals to target_size and compute the similarity with the query soft-phoc
                        if similarity_type == 0:  # cosine distance
                            softphoc_query = np.ravel(softphoc_query)
                            resized_preds = [np.ravel(cv2.resize(x, (target_size[0], target_size[1]))) for x in softphoc_proposal_preds]
                            similarities = [np.dot((softphoc_query/norm(softphoc_query)), (x/norm(x))) for x in resized_preds]
                            simi_thresh = 0.65
                        elif similarity_type == 1:  # histogram intersection
                            resized_preds = [(cv2.resize(x, (target_size[0], target_size[1]))) for x in softphoc_proposal_preds]
                            similarities = ([np.sum(np.minimum(x, softphoc_query))/len(qword) for x in resized_preds])
                        elif similarity_type == 2:  # cross-entropy
                            # print "bboxes is:"
                            # print bboxes
                            # logitsbb = [logits[0, b[1]:b[3], b[0]:b[2], :] for b in bboxes.astype(np.int)]
                            # resized_logits = [(cv2.resize(x, (target_size[0], target_size[1]))) for x in logitsbb]
                            # logits_ce = np.squeeze(np.array(resized_logits))
                            logits_ce = np.reshape(np.expand_dims(warped_logits, axis=0), (1, target_size[0] * target_size[1], 38))
                            labels_ce = np.reshape(np.expand_dims(softphoc_query, axis=0), (1, target_size[0] * target_size[1], 38))

                            similarities = np.squeeze(sess.run([ce], feed_dict={i1: labels_ce, i2: logits_ce}))
                            simi_thresh = -2.00
                        elif similarity_type == 3: # histogram intersection over union
                            resized_preds = [(cv2.resize(x, (target_size[0], target_size[1]))) for x in softphoc_proposal_preds]
                            similarities = ([np.sum(np.minimum(x, softphoc_query)/np.maximum(x, softphoc_query))/len(qword) for x in resized_preds])
                        else:
                            raise Exception('Unknown similarity type')
                        sims.append(similarities)
                        queries.append(qword)

                # print sims
                # print queries

                # allsims = [x[0] for x in sims]
                # print allsims


                # sorted_idx = np.argsort(allsims)
                # print queries[sorted_idx]

                sorted_idx = np.argsort(sims)
                # # print [(queries[x], sims[x]) for x in sorted_idx][::-1]

                if take_always_argmax:
                    idx = sorted_idx[-1]
                else:
                    idx = sorted_idx[-query_dict[qword]]
                # print similarities
                #print "idx is : %d"%idx
                # #print "length of similarity is:"
                # print np.shape(similarities)
                #print "similarity is:"
                #print sims[idx]

                topN = [queries[x].lower() for x in sorted_idx[-N:]]
                print topN
                trnscp = topN[0]

                if sims[idx] > simi_thresh:
                    # res = bboxes[idx, :]
                    # res[4] = similarities[idx]

                    # # resLeft, resTop, resRight, resBottom = [int(x) for x in res[:4]]
                    # [l, t, r, b] = res[:4]

                    # resLeft = int(l)
                    # resTop = int(t)
                    # resRight = int(r)
                    # resBottom = int(b)

                    res_file.write("%d,%d,%d,%d,%d,%d,%d,%d,%s\r\n" % (int(gt_file[l].split(',')[0]), int(gt_file[l].split(',')[1]), int(gt_file[l].split(',')[2]), 
                                                                       int(gt_file[l].split(',')[3]), int(gt_file[l].split(',')[4]), int(gt_file[l].split(',')[5]), 
                                                                       int(gt_file[l].split(',')[6]), int(gt_file[l].split(',')[7]), str(trnscp)))
                    res_file.flush()

                    #cv2.rectangle(img, (resLeft, resTop), (resRight, resBottom), (0, 255, 0), 2)
                    #cv2.putText(img, qword, (resLeft, resTop), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    # cv2.imwrite(img_bbx_path + fileName + '.png', img)
        print "done!"



                # sorted_idx = np.argsort(sims)
                # # print [(queries[x], sims[x]) for x in sorted_idx][::-1]

                # topN = [queries[x].lower() for x in sorted_idx[-N:]]

                # if transcription.lower() in topN:
                #     correct_words += 1
                # num_words +=1
                # accuracy_at_N = correct_words / float(num_words)
                # print (correct_words, num_words, accuracy_at_N)
