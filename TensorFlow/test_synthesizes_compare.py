# -*- coding: utf-8 -*-

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
# import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
import time


def visualize_phoc(word, phoc):
    fig = plt.figure()
    for (i, c) in enumerate(word):
        ax = fig.add_subplot(len(word), 1, i + 1)
        ax.plot(phoc[0, :, char2int[c]])
        ax.set_title('Letter: ' + c)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


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

    norms = np.abs(phoc).sum(axis=-1, keepdims=True)
    np.place(norms, norms == 0, 1)
    phoc /= norms

    if vis:
        visualize_phoc(word, phoc)

    phoc_new = np.zeros((height, width, numClasses))
    phoc_new[:, :, :] = phoc[None, :, :]

    return phoc_new


def synthesize_new(word, width, height, vis=False):
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
    levels = [1, 2, 4]
    # Loop over the desired subdivisions.
    for level in levels:
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

def synthesize_new_symmetry(word, original_width, height, vis=False):
    # Convert  string to list of integers.
    # chars = [(ord(x) - ord('a')+1) for x in word]
    chars = [char2int[x] for x in word]
    # And create the base 1D image over which we will histogram. This
    # image has the character value covering the *approximate* width
    # we expect it to occupy in the image (width / len(word)).
    # width = int32(np.ceil((width/(4*len(word)))*(4*len(word))))
    width = 4*len(word)
    # print (word, original_width, width)
    base = np.zeros((width,))
    # for (c, x) in zip(chars, np.linspace(0, width, len(chars)+1)[:-1]):
    #    base[int(x):int(np.floor(x+(float(width)/len(chars))))] = c
    splits_char = np.linspace(0, width, len(chars) + 1)
    for (c, l, u) in zip(chars, splits_char[:-1], splits_char[1:]):
        base[int(l):int(u)] = c

    # Create the 1D PHOC.
    phoc = np.zeros((1, width, numClasses), dtype=np.float32)
    levels = [1, 2, 4]
    # Loop over the desired subdivisions.
    for level in levels:
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

    phoc_new = cv2.resize(phoc, (original_width, height), interpolation=cv2.INTER_NEAREST)

    if vis:
        visualize_phoc(word, phoc_new)

    return phoc_new


def compare_phoc(word, phoc1, phoc2):
    fig = plt.figure()
    for (i, c) in enumerate(word):
        # ax = fig.add_subplot(len(word), 1, i + 1)
        ax = plt.subplot(len(word), 2, i*2+1)
        ax.plot(phoc1[0, :, char2int[c]])
        plt.ylim([0,1])
        if i==0:
            ax.set_title('Word-based\nLetter: ' + c)
        else:
            ax.set_title('Letter: ' + c)
    for (i, c) in enumerate(word):
        # ax = fig.add_subplot(len(word), 1, i + 1)
        ax = plt.subplot(len(word), 2, i*2+2)
        ax.plot(phoc2[0, :, char2int[c]])
        plt.ylim([0,1])
        if i==0:
            ax.set_title('Level-based\nLetter: ' + c)
        else:
            ax.set_title('Letter: ' + c)
    fig.subplots_adjust(hspace=0.5)
    # plt.suptitle('word-based    \n    level-based')
    plt.show()


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
word = 'apple'
t = time.time()
softphoc_old = synthesize(word, 400, 50, vis=False)
softphoc_new = synthesize_new_symmetry(word, 400, 50, vis=False)
compare_phoc(word, softphoc_old, softphoc_new)
print time.time() - t

# for w in word:
#     c = char2int[w]
#     plt.imshow(softphoc[:, :, c])
#     plt.colorbar()
#     plt.title(w)
#     plt.show()
