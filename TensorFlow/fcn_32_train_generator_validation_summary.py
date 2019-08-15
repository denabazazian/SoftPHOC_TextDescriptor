# -*- coding: utf-8 -*-


from __future__ import division
import re
import time
import numpy as np
import tensorflow as tf
from random import shuffle
import cv2
from matplotlib import pyplot as plt
import os.path as osp
from PIL import Image
import os, sys
import scipy.io as sio
from glob import glob
from tqdm import tqdm
import math

# Function to visualize the non-zero channels of PHOC.
def visualize_phoc(word, phoc):
    fig = plt.figure()
    for (i, c) in enumerate(word):
        ax = fig.add_subplot(len(word), 1, i + 1)
        ax.plot(phoc[0, :, char2int[c]])
        ax.set_title('Letter: ' + c)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


# Synthesize a 1-dimensional phoc of size (1, width, NUMLETTERS).
#    word: string of word to synthesize a PHOC for
#   width: width of the image to create.
#  levels: list of subdivisions to consider when computing PHOC.
#     vis: whether to visualize the PHOC or not.
#
# Example:
#
#  foo = synthesize('fubar', 100, levels=[1, 2, 3, 4], vis=True)

def synthesize(word, width, height, vis=False):
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

    # phoc = np.trunc(phoc*1000)/1000
    norms = np.abs(phoc).sum(axis=-1, keepdims=True)
    np.place(norms, norms == 0, 1)
    phoc /= norms
    phoc = 1 - (1 - phoc)  # round to the closest power of 2 to try to avoid rounding errors

    if vis:
        visualize_phoc(word, phoc)

    phoc_new = np.zeros((height, width, numClasses))
    phoc_new[:, :, :] = phoc[None, :, :]

    return phoc_new


def perspectiveTransfer(img, crd):
    width = np.max([(abs(crd[1, 0] - crd[0, 0])), (abs(crd[2, 0] - crd[3, 0]))], axis=0)
    height = np.max([(abs(crd[0, 1] - crd[3, 1])), (abs(crd[1, 1] - crd[2, 1]))], axis=0)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    rect = cv2.getPerspectiveTransform(crd, dst)
    # print crd
    warped = cv2.warpPerspective(img, rect, (width, height))

    return warped, width, height


def embed(warpedImg, crd, width_main, height_main, softPhoc_main, sub_softPHOC, binary_mask, enlarged_binary_mask):
    # s = time.time()
    # print 'embed'
    wrpcrd = np.array([[0, 0],
                       [warpedImg.shape[1], 0],
                       [warpedImg.shape[1], warpedImg.shape[0]],
                       [0, warpedImg.shape[0]]], dtype="float32")
    rect_inv = cv2.getPerspectiveTransform(wrpcrd, crd)

    height_warp = wrpcrd[2, 0]
    width_warp = wrpcrd[2, 1]
    offset_warpd = [[-width_warp / 2, -height_warp / 2],
                    [width_warp / 2, -height_warp / 2],
                    [width_warp / 2, height_warp / 2],
                    [-width_warp / 2, height_warp / 2]]
    enlarged_wrp = wrpcrd + offset_warpd
    enlarged_crd = cv2.perspectiveTransform(np.asarray([enlarged_wrp]), rect_inv)[0]  # coordinates in the image plane

    crdInt = np.asarray([crd]).astype(np.int32)
    cv2.fillPoly(binary_mask, crdInt, 1)

    enlarged_crdInt = np.asarray([enlarged_crd]).astype(np.int32)
    cv2.fillPoly(enlarged_binary_mask, enlarged_crdInt, 1)

    mywarp = lambda x: cv2.warpPerspective(x, rect_inv, (int(width_main), int(height_main)), flags=cv2.INTER_NEAREST)

    wrpd = map(mywarp, [sub_softPHOC])[0]
    warped_nonzero_inds = np.where(np.sum(wrpd, axis=-1, keepdims=True) != 0)
    softPhoc_main[warped_nonzero_inds[0], warped_nonzero_inds[1], :] = 0
    softPhoc_main[warped_nonzero_inds[0], warped_nonzero_inds[1], :] = 1 - (1 - wrpd[warped_nonzero_inds[0], warped_nonzero_inds[1],:])  # round to the closest power of 2 to try to avoid rounding errors

    return softPhoc_main, binary_mask, enlarged_binary_mask

def process_image(imgFile):
    # print 'process'
    downsampling_factor = 1 #1 #2
    img = cv2.imread(imgFile)
    # print imgFile
    # print img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(img.shape[1] / downsampling_factor), int(img.shape[0] / downsampling_factor)))
    cv_size = lambda img: tuple(img.shape[1::-1])
    width_main, height_main = cv_size(img)

    # define the initial softPHOC zero-array
    softPhoc_main = np.zeros((height_main, width_main, 38))

    binary_mask = np.zeros((height_main, width_main))
    enlarged_binary_mask = np.zeros((height_main, width_main))

    # find the line of transcription for that image
    imgName = imgFile.split('/')[-2]+'/'+imgFile.split('/')[-1].split('.')[0]
    #gtName = 'gt_' + imgName + '.txt' # for ICDAR
    gtName = imgName + '.txt'   #for SynthText
    gt = open(gtPath + gtName).read()
    numbWords = gt.count('\n')
    gt = re.sub(r'[^\x00-\x7f]', r'', gt)
    #gt = gt.split('\r\n')   #for ICDAR
    gt = gt.split('\n')   #for synthetic

    for GTword in xrange(numbWords):
        if (len(gt[GTword]) > 0 and gt[GTword].split(',')[-1].strip() != '###'):
            transcription = gt[GTword].split(',')[-1]
            # print transcription
            crd = np.array([[int(gt[GTword].split(',')[0]), int(gt[GTword].split(',')[1])],
                            [int(gt[GTword].split(',')[2]), int(gt[GTword].split(',')[3])],
                            [int(gt[GTword].split(',')[4]), int(gt[GTword].split(',')[5])],
                            [int(gt[GTword].split(',')[6]), int(gt[GTword].split(',')[7])]], dtype="float32")
            crd = crd / downsampling_factor
            warpedImg, width, height = perspectiveTransfer(img, crd)
            sub_softPHOC = synthesize(transcription, int(width), int(height), vis=False)

            #### embed this sub_softPHOC in softPHOC_main, in the exact location of the transcription
            t = time.time()
            softPhoc_main, binary_mask, enlarged_binary_mask = embed(warpedImg, crd, width_main, height_main,
                                                                     softPhoc_main, sub_softPHOC, binary_mask,
                                                                     enlarged_binary_mask)
    softPhoc_main[:, :, 0] = np.sum(softPhoc_main[:, :, 1:], axis=-1) == 0
    # print imgName + ' done '
    return (img, softPhoc_main, binary_mask, enlarged_binary_mask, width_main, height_main)

'''---------------------------------------------------------------------------------------------------------------------
Generator Function
---------------------------------------------------------------------------------------------------------------------'''
def gen(imgnames, is_training_gen):
    while True:
        if is_training_gen:
            shuffle(imgnames)
        for img_name in imgnames:
            try:
                img_file = osp.join(img_path, img_name+'.jpg')
                # if is_training_gen:
                #     rot_angle = np.random.randint(-30, 30) * math.pi / 180
                #     M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),rot_angle,1)
                #     img = cv2.warpAffine(img,M,(img.shape[1], img.shape[0]))
                img, softphoc, b_mask, enlarged_b_mask, width_img, height_img = process_image(img_file)

                # try to resize everything to a known shape for debugging
                img = cv2.resize(img, (512, 512))
                softphoc = cv2.resize(softphoc, (512, 512))
                b_mask = cv2.resize(b_mask, (512, 512))
                enlarged_b_mask = cv2.resize(enlarged_b_mask, (512, 512))
                # print img_name
                yield img, softphoc, b_mask, enlarged_b_mask, width_img, height_img
            except:
                print 'skipping ' + img_name

def gen_train():
    return gen(imgnames_train, True)

def gen_test():
    return gen(imgnames_test, False)


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

gtPath = '/path/to/datasets/synthtext/gt/gt_word_polygon/'

numClasses = 38
alphabet = "#abcdefghijklmnopqrstuvwxyz1234567890@"

'''---------------------------------------------------------------------------------------------------------------------
defining batches
---------------------------------------------------------------------------------------------------------------------'''

img_path = '/path/to/synthtext/input_images/'
index_training_img = '/path/to/datasets/synthtext/gt/img_idx_shuffle_maxIoUZero_training.txt'
index_testing_img = '/path/to/datasets/synthtext/gt/img_idx_shuffle_maxIoUZero_testing.txt'

imgnames_train = open(index_training_img).read()
imgnames_train = imgnames_train.split('\n')
print len(imgnames_train)
imgnames_train = [x for x in imgnames_train if x]
print len(imgnames_train)

imgnames_test = open(index_testing_img).read()
imgnames_test = imgnames_test.split('\n')
print len(imgnames_test)
imgnames_test = [x for x in imgnames_test if x]
print len(imgnames_test)
# imgnames_test = ['143/rajasthan_99_82']

training_dataset = tf.data.Dataset.from_generator(gen_train, (tf.uint8, tf.float32, tf.bool, tf.bool, tf.int32, tf.int32),(tf.TensorShape((512, 512, 3)), tf.TensorShape((512, 512, 38)), tf.TensorShape((512, 512)), tf.TensorShape((512, 512)), None, None))
training_dataset = training_dataset.batch(6)
training_dataset = training_dataset.prefetch(8)

test_dataset = tf.data.Dataset.from_generator(gen_test, (tf.uint8, tf.float32, tf.bool, tf.bool, tf.int32, tf.int32),(tf.TensorShape((512, 512, 3)), tf.TensorShape((512, 512, 38)), tf.TensorShape((512, 512)), tf.TensorShape((512, 512)), None, None))
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.prefetch(5)

iterator = training_dataset.make_one_shot_iterator()

training_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)

img_batch, annotation_batch, binary_mask_batch, enlarged_binary_mask_batch, width_img, height_img = iterator.get_next()

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

sys.path.append("/path/to/softPHOC_tensorflow/slim/")  # for reading the net
sys.path.append("/path/to/softPHOC_tensorflow/models/")
sys.path.append("/path/to/softPHOC_tensorflow/softPHOC_utils/")

checkpoints_dir = '/path/to/tensorflow-FCN-textNonText/checkpoints'


w1 = 100
w2 = 1000
w3 = 2500
lr = 0.00005
number_of_classes = 38
#batch_size = tf.placeholder(tf.int64)

# log_folder = './training_generator/' + str(time.strftime("%Y:%m:%d:%H:%M:%S")) + '_(' + str(w1) + ',' + str(
#     w2) + ',' + str(w3) + ')' + '_lr_' + str(lr)

# if not os.path.exists(log_folder):
#     os.makedirs(log_folder)

log_folder = '/path/to/softPHOC_synthText/log/log_2'   #$ python -m tensorflow.tensorboard --logdir=.            #$ tensorboard --logdir=.

slim = tf.contrib.slim
# checkpoint_path = os.path.join(checkpoints_dir, 'model_fcn32s.ckpt')
checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

from fcn_32s import FCN_32s, extract_vgg_16_mapping_without_fc8

'''---------------------------------------------------------------------------------------------------------------------
Process batch with FCN_32s
---------------------------------------------------------------------------------------------------------------------'''
# dropout_keep_prob=tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

upsampled_logits_batch, vgg_16_variables_mapping = FCN_32s(image_batch_tensor=img_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=is_training)
                                                            # dropout_keep_prob=dropout_keep_prob

'''---------------------------------------------------------------------------------------------------------------------
Give a range to logits
---------------------------------------------------------------------------------------------------------------------'''

margin_logits = tf.constant(10.0)
upsampled_logits_batch = tf.maximum(tf.minimum(upsampled_logits_batch, margin_logits), -margin_logits)

'''---------------------------------------------------------------------------------------------------------------------
Mask the output of the network and compute loss
---------------------------------------------------------------------------------------------------------------------'''
# The FCN_32s output is a multiplication of 32. So, it should be resized as the img_batch before computing loss
upsampled_logits_batch_resized = tf.image.resize_images(upsampled_logits_batch, (512, 512))


with tf.name_scope("losses") as scope:
    annotation_TB_ch0 = annotation_batch[:, :, :, 0]
    annotation_TB_ch1 = tf.reduce_sum(annotation_batch[:, :, :, 1:], axis=-1)
    annotation_TB = tf.stack((annotation_TB_ch0, annotation_TB_ch1), axis=3)

    logit_TB_ch0 = upsampled_logits_batch_resized[:, :, :, 0]
    logit_TB_ch1 = tf.reduce_sum(upsampled_logits_batch_resized[:, :, :, 1:], axis=-1)
    logit_TB = tf.stack((logit_TB_ch0, logit_TB_ch1), axis=3)

    annot1 = tf.boolean_mask(annotation_TB, binary_mask_batch)
    logits1 = tf.boolean_mask(logit_TB, binary_mask_batch)
    softmax_ce_loss1 = tf.cond(tf.size(logits1) > 0,
                               lambda: tf.reduce_mean(
                                   tf.nn.softmax_cross_entropy_with_logits(labels=annot1, logits=logits1)),
                               lambda: tf.constant(0.0))

    annot2 = tf.boolean_mask(annotation_TB, tf.logical_not(binary_mask_batch))
    logits2 = tf.boolean_mask(logit_TB, tf.logical_not(binary_mask_batch))
    softmax_ce_loss2 = tf.cond(tf.size(logits2) > 0,
                               lambda: tf.reduce_mean(
                                   tf.nn.softmax_cross_entropy_with_logits(labels=annot2, logits=logits2)),
                               lambda: tf.constant(0.0))

    annot3 = tf.boolean_mask(annotation_batch, enlarged_binary_mask_batch)
    logits3 = tf.boolean_mask(upsampled_logits_batch_resized, enlarged_binary_mask_batch)

    softmax_ce_loss3 = tf.cond(tf.size(logits3) > 0,
                               lambda: tf.reduce_mean(
                                   tf.nn.softmax_cross_entropy_with_logits(labels=annot3, logits=logits3)),
                               lambda: tf.constant(0.0))

    total_loss = (w1 * softmax_ce_loss1) + (w2 * softmax_ce_loss2) + (w3 * softmax_ce_loss3)

# get predictions
with tf.name_scope("predictions") as scope:
    char_probabilities = tf.nn.softmax(upsampled_logits_batch_resized)
    char_class_predictions = tf.argmax(char_probabilities, dimension=3)
    text_probabilities = tf.nn.softmax(logit_TB)
    text_class_predictions = tf.argmax(text_probabilities, dimension=3)

'''---------------------------------------------------------------------------------------------------------------------
                                        GRAPH DEFINITION ENDS HERE
                                            TRAINING STARTS HERE
---------------------------------------------------------------------------------------------------------------------'''

'''---------------------------------------------------------------------------------------------------------------------
Set the optimizer
---------------------------------------------------------------------------------------------------------------------'''

with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

'''---------------------------------------------------------------------------------------------------------------------
Init the network and load previous checkpoint
---------------------------------------------------------------------------------------------------------------------'''

vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)
init_fn = slim.assign_from_checkpoint_fn(model_path=checkpoint_path, var_list=vgg_16_without_fc8_variables_mapping)

'''---------------------------------------------------------------------------------------------------------------------
Define summaries for tensorboard
---------------------------------------------------------------------------------------------------------------------'''

# Losses
tf.summary.scalar('softmax_ce_loss1', softmax_ce_loss1)
tf.summary.scalar('softmax_ce_loss2', softmax_ce_loss2)
tf.summary.scalar('softmax_ce_loss3', softmax_ce_loss3)
tf.summary.scalar('total_loss', total_loss)

# Merge all train summaries
merged_summary_op = tf.summary.merge_all()

# Test summaries
test_mean_total_loss_ph = tf.placeholder(tf.float32)
test_loss1_ph = tf.placeholder(tf.float32)
test_loss2_ph = tf.placeholder(tf.float32)
test_loss3_ph = tf.placeholder(tf.float32)
test_loss_total_summary = tf.summary.scalar('test_total_loss', test_mean_total_loss_ph)
test_loss1_summary = tf.summary.scalar('test_loss1', test_loss1_ph)
test_loss2_summary = tf.summary.scalar('test_loss2', test_loss2_ph)
test_loss3_summary = tf.summary.scalar('test_loss3', test_loss3_ph)

# Merge test summaries
merged_test_summaries = tf.summary.merge([test_loss1_summary, test_loss2_summary, test_loss3_summary, test_loss_total_summary])


'''---------------------------------------------------------------------------------------------------------------------
Train the network!
---------------------------------------------------------------------------------------------------------------------'''

with tf.Session(config=config) as sess:
    # to see the graph in Tensorboard
    summary_string_writer = tf.summary.FileWriter(log_folder, sess.graph)

    model_variables = slim.get_model_variables()
    #saver = tf.train.Saver(model_variables)
    saver = tf.train.Saver(model_variables, max_to_keep=50, keep_checkpoint_every_n_hours=5)

    # init variables
    init_fn(sess)
    global_vars_init_op = tf.global_variables_initializer()
    local_vars_init_op = tf.local_variables_initializer()
    combined_op = tf.group(global_vars_init_op, local_vars_init_op)
    sess.run(combined_op)

    checkpoint_interval = 500
    dump_data_interval = 1500
    train_iters = 700002
    test_interval = 1500
    summaryWrite_interval = 5

    sess.run(training_init_op)

    # start training
    for i in xrange(train_iters):
        # Test every test_interval iterations, including iteration 0
        if i % test_interval == 0:# and i>0:
            sess.run(test_init_op)
            test_total_loss = 0
            test_total_loss1 = 0
            test_total_loss2 = 0
            test_total_loss3 = 0
            print 'testing'
            N_test = len(imgnames_test)
            for test_img_name in tqdm(imgnames_test[:N_test]):  # just try the first N_test images to debug
                [test_loss, test_loss1, test_loss2, test_loss3] = sess.run([total_loss, softmax_ce_loss1, softmax_ce_loss2, softmax_ce_loss3],
                                    feed_dict={is_training: False})
                test_total_loss += test_loss
                test_total_loss1 += test_loss1
                test_total_loss2 += test_loss2
                test_total_loss3 += test_loss3
            mean_tot_loss = test_total_loss / N_test
            mean_tot_loss1 = test_total_loss1 / N_test
            mean_tot_loss2 = test_total_loss2 / N_test
            mean_tot_loss3 = test_total_loss3 / N_test
            print 'TEST LOSS: ' + str(mean_tot_loss)
            test_summary_string = sess.run(merged_test_summaries, feed_dict={test_mean_total_loss_ph: mean_tot_loss,
                                                                             test_loss1_ph: mean_tot_loss1,
                                                                             test_loss2_ph: mean_tot_loss2,
                                                                             test_loss3_ph: mean_tot_loss3})
            summary_string_writer.add_summary(test_summary_string, i)
            sess.run(training_init_op)

        # Do a train step
        st = time.time()
        print("starting iteration# " + str(i))
        #sess.run(training_init_op)
        if i % summaryWrite_interval == 0:
            total_Loss_, summary_string, _ = sess.run([total_loss, merged_summary_op, train_step], feed_dict={is_training: True})
            print "took " + str(time.time() - st)
            print 'current_loss: ' + str(total_Loss_)
            summary_string_writer.add_summary(summary_string, i)
        else:
            sess.run(train_step, feed_dict={is_training: True})

        # save a checkpoint every checkpoint_interval iterations
        if i % checkpoint_interval == 0:
            save_path = saver.save(sess, "/path/to/softPHOC_synthText/tf_model/tf_model_2/model_fcn32s.ckpt")
            print("Model saved in file: %s" % save_path)


        # dump data for debug every dump_data_interval iterations
        if i % dump_data_interval == 0:
            # forward pass for one batch
            image_batch__, annotation_batch__, upsampled_logits_batch__, char_probabilities__, char_class_predictions__, text_probabilities__, text_class_predictions__ = sess.run(
                [img_batch, annotation_batch, upsampled_logits_batch_resized,
                 char_probabilities, char_class_predictions, text_probabilities, text_class_predictions],
                feed_dict={is_training: False})

            # save a .mat file with the results
            sio.savemat('/path/to/softPHOC_synthText/batch/batch_2/batch' + str(i) + '.mat',
                        {
                            'image_batch__': image_batch__,
                            'annotation_batch__': annotation_batch__,
                            'upsampled_logits_batch__': upsampled_logits_batch__,
                            'char_probabilities__': char_probabilities__,
                            'char_class_predictions__': char_class_predictions__,
                            'text_probabilities__': text_probabilities__,
                            'text_class_predictions__': text_class_predictions__
                        })

#save the final model
save_path = saver.save(sess, "/path/to/softPHOC_synthText/tf_model/tf_model_2/model_fcn32s.ckpt")
print("Model saved in file: %s" % save_path)

summary_string_writer.close()
