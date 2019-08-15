import cv2
import numpy as np
import re


#imgName = '164/steel_75_99'
#imgName = '8/ballet_106_72'
#imgName = '8/ballet_117_25'
#imgName = '182/turtles_142_103'
#imgName = '117/night_13_93'
#imgName = '182/turtles_28_45'
#imgName = '199/zebra_70_59'
#imgName = '158/silk_132_44'
imgName = '143/rajasthan_99_82'

gt_file  = open ('/path/to/datasets/synthtext/gt/gt_word_polygon/'+ imgName +'.txt').read()
#gt_file  = open ('/path/to/datasets/synthtext/gt/gt_word/'+ imgName +'.txt').read()
#gt_file  = open ('/path/to/datasets/synthtext/gt/gt_char/'+ imgName +'.txt').read()


imgPath = '/path/to/datasets/synthtext/input_images/'+imgName+'.jpg'

img=cv2.imread(imgPath)
numQWords =  gt_file.count('\n')
voc = re.sub(r'[^\x00-\x7f]',r'',gt_file)
voc = gt_file.split('\n')
idnqw = 0
for nqw in range(0,numQWords):
    if (len(voc[nqw])>0 and voc[nqw].split(',')[-1].strip()!='###'):
       idnqw +=1
       qword = voc[nqw].split(',')[8] #voc[nqw].split(',')[-1]
       print qword
       pts = np.array([[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])],[int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[3])],[int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5])],[int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])],[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])]], np.int32)
       pts = pts.reshape((-1,1,2))
       cv2.polylines(img,[pts],True,(0,255,0),2) #BGR
       cv2.putText(img,qword,(int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])),cv2.FONT_HERSHEY_PLAIN,1.25,(0,255,0),2) #BGR

cv2.imwrite('/path/to/test_gts_not_polygon3.png',img) 
#cv2.imwrite('/path/to/test_gts_polygon3.png',img) 