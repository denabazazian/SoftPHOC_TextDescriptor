import numpy as np
import os
import re
from matplotlib import pyplot as plt


pathToGtMat='/path/to/datasets/synthtext/gt.mat'
m=scipy.io.matlab.loadmat(pathToGtMat)
files,txt,wordBB,charBB= m['imnames'],m['txt'],m['wordBB'],m['charBB']

i = 6461
imgName = files[0][i]
imgName =  str(imgName).split("'")[1]
imgPath = '/path/to/datasets/synthtext/input_images/'+imgName+'.jpg'

img=cv2.imread(imgPath)

char_list = []
char_list = ''.join([str(w)for w in txt[0,i]])
char_list = ''.join(char_list.split())

char_BB = charBB[0][i]

top_left_x = np.asarray(char_BB[0][0])
top_right_x = np.asarray(char_BB[0][1])
bottom_right_x = np.asarray(char_BB[0][2])
bottom_left_x = np.asarray(char_BB[0][3])
top_left_y = np.asarray(char_BB[1][0])
top_right_y = np.asarray(char_BB[1][1])
bottom_right_y = np.asarray(char_BB[1][2])
bottom_left_y = np.asarray(char_BB[1][3])


for j in range(0, len(char_list)):
       pts = np.array([[int(top_left_x[j]),int(top_left_y[j])],[int(top_right_x[j]),int(top_right_y[j])],[int(bottom_right_x[j]),int(bottom_right_y[j])],[int(bottom_left_x[j]),int(bottom_left_y[j])],[int(top_left_x[j]),int(top_left_y[j])]], np.int32)
       pts = pts.reshape((-1,1,2))
       cv2.polylines(img,[pts],True,(0,0,255),2)
       cv2.putText(img,char_list[j],(int(top_left_x[j]),int(top_left_y[j])),cv2.FONT_HERSHEY_PLAIN,1.25,(0,0,255),2) #BGR-Red

cv2.imwrite('/path/to/test_gts_not_polygon3_char.png',img) 