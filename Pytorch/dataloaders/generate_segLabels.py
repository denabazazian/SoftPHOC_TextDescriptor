import numpy as np
from glob import glob
from tqdm import tqdm
import math
from skimage.draw import polygon
import os.path as osp
import scipy.io as sio
import os
import sys
import re
import cv2


img_id = 0 
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
alphabet = "#abcdefghijklmnopqrstuvwxyz1234567890@"


# Function to visualize the non-zero channels of PHOC.
def visualize_phoc(word, phoc):
    fig = plt.figure()
    for (i, c) in enumerate(word):
        ax = fig.add_subplot(len(word), 1, i + 1)
        ax.plot(phoc[0, :, char2int[c]])
        ax.set_title('Letter: ' + c)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def synthesize(word, original_width, height, vis=False):
    # Convert  string to list of integers.
    # chars = [(ord(x) - ord('a')+1) for x in word]
    chars = [char2int[x] for x in word]
    # And create the base 1D image over which we will histogram. This
    # image has the character value covering the *approximate* width
    # we expect it to occupy in the image (width / len(word)).
    # width = int32(np.ceil((width/(4*len(word)))*(4*len(word))))
    #width = 4*len(word)
    # print (word, original_width, width)
    base = np.zeros((width,))
    # for (c, x) in zip(chars, np.linspace(0, width, len(chars)+1)[:-1]):
    #    base[int(x):int(np.floor(x+(float(width)/len(chars))))] = c
    splits_char = np.linspace(0, width, len(chars) + 1)
    for (c, l, u) in zip(chars, splits_char[:-1], splits_char[1:]):
        base[int(l):int(u)] = c

    # Create the 1D PHOC.
    phoc = np.zeros((1, width, numClasses), dtype=np.float32)
    #levels = [1, 2, 4]
    levels = len(word)
    # Loop over the desired subdivisions.
    for level in range(0,levels+1):
    #for level in levels:
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

    #phoc_new = cv2.resize(phoc, (original_width, height), interpolation=cv2.INTER_NEAREST)
    phoc_new = np.zeros((height,width,numClasses))
    phoc_new[:,:,:] = phoc[None,:,:]

    if vis:
        visualize_phoc(word, phoc_new)

    return phoc_new


def perspectiveTransfer(img, crd):
    width = np.max([(abs(crd[1, 0] - crd[0, 0])), (abs(crd[2, 0] - crd[3, 0]))], axis=0)
    height = np.max([(abs(crd[0, 1] - crd[3, 1])), (abs(crd[1, 1] - crd[2, 1]))], axis=0)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    rect = cv2.getPerspectiveTransform(crd, dst)
    warped = cv2.warpPerspective(img, rect, (width, height))

    return warped, width, height


def embed(warpedImg, crd, width_main, height_main, softPhoc_main, sub_softPHOC, binary_mask, enlarged_binary_mask):
    crd[:, 0] = np.maximum(np.minimum(crd[:, 0], width_main -1), 0)
    crd[:, 1] = np.maximum(np.minimum(crd[:, 1], height_main -1), 0)
    warped_nonzero_inds = polygon(crd[:, 1], crd[:, 0])
    ### warped_nonzero_inds = list(np.maximum(np.minimum(warped_nonzero_inds, 511), 0))  # avoid going outside from the image
    # warped_nonzero_inds = list(warped_nonzero_inds)
    # warped_nonzero_inds[0] = np.minimum(warped_nonzero_inds[0], height_main)
    # warped_nonzero_inds[1] = np.minimum(warped_nonzero_inds[1], width_main)
    # warped_nonzero_inds = np.maximum(warped_nonzero_inds, 0)
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
    # binary_mask[warped_nonzero_inds[0], warped_nonzero_inds[1]] = 1
    cv2.fillPoly(binary_mask, crdInt, 1)

    enlarged_crdInt = np.asarray([enlarged_crd]).astype(np.int32)
    cv2.fillPoly(enlarged_binary_mask, enlarged_crdInt, 1)

    wrpd = cv2.warpPerspective(sub_softPHOC, rect_inv, (int(width_main), int(height_main)), flags=cv2.INTER_NEAREST)

    softPhoc_main[warped_nonzero_inds[0], warped_nonzero_inds[1], :] = 0
    softPhoc_main[warped_nonzero_inds[0], warped_nonzero_inds[1], :] = 1 - (1-wrpd[warped_nonzero_inds[0], warped_nonzero_inds[1], :])

    return softPhoc_main, binary_mask, enlarged_binary_mask

#def process_image(imgFile,gtPath,imgName,resizing=False):
def process_image(gtFile,imgFile,resizing=False):

    # tt = time.time()
    img = cv2.imread(imgFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv_size = lambda img: tuple(img.shape[1::-1])
    width_main, height_main = cv_size(img)
    if resizing == True:
        resize_factor_x = width_main/512
        resize_factor_y = height_main/512
        width_main = 512
        height_main = 512
        img = cv2.resize(img, (512, 512))
    # define the initial softPHOC zero-array
    softPhoc_main = np.zeros((height_main, width_main, 38))

    binary_mask = np.zeros((height_main, width_main))
    enlarged_binary_mask = np.zeros((height_main, width_main))

    # find the line of transcription for that image
    #imgName = imgFile.split('/')[-2]+'/'+imgFile.split('/')[-1].split('.')[0]
    #gtName = 'gt_' + imgName + '.txt' # for ICDAR
    #gtName = imgName + '.txt'   #for SynthText
    #gt = open(os.path.join(gtPath,imgName+'.txt')).read()
    gt = open(gtFile).read()

    numbWords = gt.count('\n')
    gt = re.sub(r'[^\x00-\x7f]', r'', gt)
    #gt = gt.split('\r\n')   #for ICDAR
    gt = gt.split('\n')   #for synthetic

    for GTword in range(numbWords):
        if (len(gt[GTword]) > 0 and gt[GTword].split(',')[-1].strip() != '###'):
            transcription = gt[GTword].split(',')[-1]
            if len(transcription) == 0:
                continue
            # print transcription
            crd = np.array([[int(gt[GTword].split(',')[0]), int(gt[GTword].split(',')[1])],
                            [int(gt[GTword].split(',')[2]), int(gt[GTword].split(',')[3])],
                            [int(gt[GTword].split(',')[4]), int(gt[GTword].split(',')[5])],
                            [int(gt[GTword].split(',')[6]), int(gt[GTword].split(',')[7])]], dtype="float32")
            if resizing == True:
                crd[:, 0] /= resize_factor_x
                crd[:, 1] /= resize_factor_y
                
            warpedImg, width, height = perspectiveTransfer(img, crd)
            sub_softPHOC = synthesize(transcription, int(width), int(height), vis=False)

            #### embed this sub_softPHOC in softPHOC_main, in the exact location of the transcription
            softPhoc_main, binary_mask, enlarged_binary_mask = embed(warpedImg, crd, width_main, height_main,
                                                                     softPhoc_main, sub_softPHOC, binary_mask,
                                                                     enlarged_binary_mask)
    softPhoc_main[:, :, 0] = np.sum(softPhoc_main[:, :, 1:], axis=-1) == 0

    #print('Image {} is done!'.format(imgFile))
    #print('gt {} is made for softPHOC'.format(gtFile))

    return (img, softPhoc_main, binary_mask, enlarged_binary_mask, width_main, height_main)



#def generate_segmentation_softPHOC_labels(gt_coord_bbx_path, img_path, img_ID):
def generate_segmentation_softPHOC_labels(gt_file,img_file):

    # img_file = os.path.join(img_path,img_ID+'.jpg')
    # gtPath = gt_coord_bbx_path
    # imgName = img_ID
    #img, softphoc, b_mask, enlarged_b_mask, width_img, height_img = process_image(img_file,gt_coord_bbx_path,imgName,resizing=False)
    img, softphoc, b_mask, enlarged_b_mask, width_img, height_img = process_image(gt_file,img_file,resizing=False)

    return softphoc, b_mask, enlarged_b_mask


def generate_segmentation_labels(gt_coord_bbx_path, img_path, img_ID):
# for img_file in sys.argv[1:]:
#     print(img_file)
#     img_id +=1

#     img_name = img_file.split('.')[0].split('/')[-1]

    #img = cv2.imread(img_file)
    img = cv2.imread(os.path.join(img_path,img_ID+'.jpg'))
    #print(os.path.join(img_path,img_ID+'.jpg'))
    cv_size = lambda img: tuple(img.shape[1::-1])
    width_main, height_main = cv_size(img)

    #gt_png = img = np.zeros(width_main, height_main)
    gt_png = np.zeros([height_main,width_main],dtype=np.uint8)

    #read ground truth 
    #gt = open(img_name.split('img')[0]+'voc_'+(img_name.split('/')[-1].split('.')[0])+'.txt').read()   # for the dictionary and distractor
    gt = open(os.path.join(gt_coord_bbx_path,img_ID+'.txt')).read()
    
    numGTs = gt.count('\n')
    #print('numGT is {}'.format(numGTs))
    gt = re.sub(r'[^\x00-\x7f]', r'', gt)
    #print("gt is befor split :{}".format(gt))
    #gt = gt.split('\r\n')
    gt = gt.split('\n')
    #print("gt is after split :{}".format(gt))
    idgt = 0

    # Read the text proposals of the correspond image
    

    for words in range(0, numGTs):
        #if (len(gt[words]) > 0 and gt[words].split(',')[-1].strip() != '###'):
        #print("len GT[word] is: {}".format(len(gt[words])))
        if (len(gt[words]) > 0):
            idgt += 1
            gt_points = np.array( [[[gt[words].split(',')[0],gt[words].split(',')[1]],
                                  [gt[words].split(',')[2],gt[words].split(',')[3]],
                                  [gt[words].split(',')[4],gt[words].split(',')[5]],
                                  [gt[words].split(',')[6],gt[words].split(',')[7]]]],
                                  dtype=np.int32 )
            
            cv2.fillPoly(gt_png, gt_points, 1)
            #print(idgt)
    # write image
    # cv2.imwrite(gt_png_path + img_name + '.png', gt_png)
    # print("img_id {} done!".format(img_id))

    return gt_png


#######################################################################################################
#gt_png_path = '/dataset/ICDAR2015/ch4_gt_img_train_1/'
#gt_txt_path = '/dataset/ICDAR2015/ch4_train_gt/'
# gt_coord_bbx_path = '/dataset/ICDAR2015/ch4_train_gt/'
# img_path = '/dataset/ICDAR2015/ch4_train_img/'
# img_ID = 'img_49'

#######################################################################################################
# python generate_labels.py /dataset/ICDAR2015/ch4_train_img/img_49.jpg

#import matplotlib.pyplot as plt
#from skimage.draw import line, polygon, circle, ellipse

# def load_annotation(gt_path):
#     with gt_path.open(mode='r') as f:
#         label = dict()
#         label["coor"] = list()
#         label["ignore"] = list()
#         for line in f:
#             text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
#             x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
#             if text[8] == "###" or text[8] == "*":
#                 label["ignore"].append(True)
#             else:
#                 label["ignore"].append(False)
#             bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

