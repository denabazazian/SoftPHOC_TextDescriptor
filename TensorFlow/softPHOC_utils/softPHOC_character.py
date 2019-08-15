import skimage.io as io
import numpy as np
import os


def softPHOC_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

    #class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
    #               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #               'dog', 'horse', 'motorbike', 'person', 'potted-plant',
    #               'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    
    #class_names=['NonText', 'text']
    class_names=['background', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '@']

    #enumerated_array = enumerate(class_names[:-1])
    enumerated_array = enumerate(class_names[:])
    
    classes_lut = list(enumerated_array)

    # Add a special class representing ambigious regions
    # which has index 255.
    #classes_lut.append((255, class_names[-1]))
    
    classes_lut = dict(classes_lut)

    return classes_lut





    # #enumerated_array = enumerate(class_names[:-1])
    # enumerated_array = enumerate(class_names[:])
    
    # classes_lut = list(enumerated_array)
    
    # # Add a special class representing ambigious regions
    # # which has index 255.
    # #classes_lut.append((255, class_names[-1]))
    
    # classes_lut = dict(classes_lut)

    # #return dict({1: 'text', 0: 'nontext'}) #classes_lut
    # return classes_lut

