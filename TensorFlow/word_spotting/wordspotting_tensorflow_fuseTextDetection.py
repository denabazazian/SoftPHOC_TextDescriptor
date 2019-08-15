# -*- coding: utf-8 -*-
"""
Created on Mon May 07 2018 19:07:21 

@author: dena
"""

# python wordspotting_tensorflow.py /home/fcn/wordSpotting/icdar_ch4_testSet/img_*.jpg

import scipy
import scipy.io
import sys
import numpy as np
from matplotlib import pyplot as plt
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
import matplotlib.pyplot as plt


def getIimg(image_filename,net,number_of_classes):
    
    from tf_image_segmentation.models.fcn_32s import FCN_32s

    im = Image.open(image_filename)
    heatMap = np.empty((im.size[1], im.size[0], number_of_classes))

    image_filename_placeholder = tf.placeholder(tf.string)
    feed_dict_to_use = {image_filename_placeholder: image_filename}
    image_tensor = tf.read_file(image_filename_placeholder)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    # Fake batch for image and annotation by adding leading empty axis.
    image_batch_tensor = tf.expand_dims(image_tensor, axis=0)
    # Be careful: after adaptation, network returns final labels and not logits
    FCN_32s = adapt_network_for_any_size_input(FCN_32s, 32)
    if imgNum == 0 and net_reuse==0:
            upsampled_logit, fcn_16s_variables_mapping_train = FCN_32s(image_batch_tensor=image_batch_tensor,
                                                           number_of_classes=number_of_classes,
                                                           is_training=False, reuse=None)  
    else:
            upsampled_logit, fcn_16s_variables_mapping_train = FCN_32s(image_batch_tensor=image_batch_tensor,
                                                           number_of_classes=number_of_classes,
                                                           is_training=False, reuse=True)


    pred = tf.argmax(upsampled_logit, dimension=3)                                         
    probabilities = tf.nn.softmax(upsampled_logit)
    initializer = tf.local_variables_initializer()
    saver = tf.train.Saver()
    cmap = plt.get_cmap('bwr')

    with tf.Session() as sess:    
        sess.run(initializer)
        saver.restore(sess,net)
        image_np, pred_np , probabilities_np = sess.run([image_tensor, pred, probabilities], feed_dict=feed_dict_to_use)
 
    for i in range (0,number_of_classes):
        heatMap[:,:,i] = probabilities_np.squeeze()[:,:,i]

    iimg=heatMap.cumsum(axis=0).cumsum(axis=1)
    
    return iimg
    
    
def getQueryHistHOC2(qword,alphabet,minusOneForMissing=False):
    qHist=np.zeros([len(alphabet)*3])
    m=defaultdict(lambda: len(alphabet)-1)
    m.update({alphabet[k]:k for k in range(1,len(alphabet))})
    for c in qword.lower():
        qHist[m[c]]+=1
    for c in qword[:len(qword)/2].lower():
        qHist[38+m[c]]+=1
    for c in qword[len(qword)/2:].lower():
        qHist[2*38+m[c]]+=1
    if (len(qword))%2:
       qHist[38+m[qword[len(qword)/2]]]+=.5
       qHist[2*38+m[qword[len(qword)/2]]]-=.5

    if minusOneForMissing:
       qHist[res==0]=-1
    return qHist[:]
    
    
def textProposals(csvName,iimg_char,iimg_text,qHist):
    idxTP=0

    lines=[l.strip().split(',') for l in open(csvName).read().split('\n') if len(l)>0]       
       
    BB = np.empty([len(lines),5], dtype='f')
    for lineNum in range(0,len(lines)):
        for colNum in range(0,4):
            BB[lineNum,colNum] = lines[lineNum][colNum] #converting the list of list to a numpy array
     

       
    BBsrt = BB[np.argsort(BB[:,-1]),:] #sorting proposals by the scores
    #BB[np.argsort(BB[:,-1])[:500],:] 
    resLTRB=np.empty([len(BBsrt),5])
    BBEnergy=np.empty([len(BBsrt),38*3])
    BBEnergyNorm=np.empty([len(BBsrt),38*3])
    qHistW=np.zeros(38*3)
    qHistNorm=np.zeros(38*3)
    
    for rr in range(0,len(BBsrt)):
        left = int(BBsrt[rr,0])
        top = int(BBsrt[rr,1])
        right = int(left + BBsrt[rr,2]) #right = left + width
        bottom = int(top + BBsrt[rr,3]) #bottom = top + height
        surface =  BBsrt[rr,2] * BBsrt[rr,3]       
        #print 'left: %s, top:%s, right:%s, bottom:%s' %(left,top,right,bottom)
        if right>= iimg_char.shape[1]:
            right = int(iimg_char.shape[1]-1)
        if top>= iimg_char.shape[0]:
            top = int(iimg_char.shape[0]-1)    
            

      
        #### HOC2 #######
        cw=(left+right)/2
        energyVect=np.empty([1,38*3])
        energyVect[0,:38]=iimg_char[bottom,right,:]+iimg_char[top,left,:]-(iimg_char[bottom,left,:]+iimg_char[top,right,:])
        energyVect[0,38:2*38]=iimg_char[bottom,cw,:]+iimg_char[top,left,:]-(iimg_char[bottom,left,:]+iimg_char[top,cw,:])      
        energyVect[0,38*2:]=iimg_char[bottom,right,:]+iimg_char[top,cw,:]-(iimg_char[bottom,cw,:]+iimg_char[top,right,:])
        
        energyVect/=surface

        ##### add information of text non-text #########
        cw=(left+right)/2
        energyVect_text=np.empty([1,2*3])
        energyVect_text[0,0:2]=iimg_text[bottom,right,:]+iimg_text[top,left,:]-(iimg_text[bottom,left,:]+iimg_text[top,right,:])
        energyVect_text[0,2:4]=iimg_text[bottom,cw,:]+iimg_text[top,left,:]-(iimg_text[bottom,left,:]+iimg_text[top,cw,:])      
        energyVect_text[0,4:6]=iimg_text[bottom,right,:]+iimg_text[top,cw,:]-(iimg_text[bottom,cw,:]+iimg_text[top,right,:])
        
        energyVect_text/=surface

        energyVect[0,0] *= energyVect_text[0,0]
        for envct in range(1,38):
            energyVect[0,envct] *= energyVect_text[0,1]
            
        energyVect[0,38] *= energyVect_text[0,2]
        for envct in range(39,38*2):
            energyVect[0,envct] *= energyVect_text[0,3]
            
        energyVect[0,38*2] *= energyVect_text[0,4]
        for envct in range(77,38*3):
            energyVect[0,envct] *= energyVect_text[0,5]
        ########    
       
        BBEnergy[idxTP,:] = energyVect 
        idxTP+=1
        
    #L1Normalization    
    for rr in range(0,len(BBsrt)):
        BBW = np.sum(BBEnergy[rr,:])
        BBEnergyNorm[rr,:] = BBEnergy[rr,:]/BBW
              
    qHistW = np.sum(qHist)
    qHistNorm = qHist/qHistW
    
    ### Histogram Intersection   
    HistIntersection = np.zeros((len(BBEnergyNorm),1))
         
    qHistNorma = np.zeros((1,len(qHistNorm)))
    for ll in range(0, len(qHistNorm)):
        qHistNorma[0,ll] = qHistNorm[ll]
     
    BBEnergyNorma = np.zeros((1,len(qHistNorm)))
    for rr in range(0,len(BBEnergyNorm)):
        for ll in range(0, len(qHistNorm)):
            BBEnergyNorma[0,ll] = BBEnergyNorm[rr][ll]

        dConcatenate = np.concatenate((BBEnergyNorma,qHistNorma), axis =0)
        HistIntersection[rr] = np.sum(np.min(dConcatenate, axis =0))/np.sum(dConcatenate)
    
    resLTRBInter=np.empty([BBsrt.shape[0],5])
    resLTRBInter[:,0]= BBsrt[:,0]
    resLTRBInter[:,1]= BBsrt[:,1]
    resLTRBInter[:,2]= BBsrt[:,2]+BBsrt[:,0]
    resLTRBInter[:,3]= BBsrt[:,3]+BBsrt[:,1] 
    resLTRBInter[:,4]= HistIntersection[:,0]

    return resLTRBInter
    

if __name__=='__main__':
    
    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"
    sys.path.append("/path/to/tensorflow-FCN-gitHub/warmspringwinds/tf-image-segmentation/")
    sys.path.append("/path/to/tensorflow-FCN-gitHub/warmspringwinds/models/slim")
    softPhoc_net_char= "/path/to/softPHOC_synthText/tf_model/tf_model_3_backup_118000/model_fcn32s_118000.ckpt"
    softPhoc_net_text= "/path/to/tensorflow-FCN-textNonText/tf_models_fcn32s_uint8/model_fcn32s.ckpt"

    img_bbx_path = "/path/to/softPHOC_synthText/word_spotting/img_bbx_res/"
    iou_res_path = "/path/to/softPHOC_synthText/word_spotting/iou_res/"
    #loc_res = "/path/to/softPHOC_synthText/word_spotting/loc_res/"

    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    slim = tf.contrib.slim
    from tf_image_segmentation.models.fcn_32s import FCN_32s
    from tf_image_segmentation.utils.inference_prediction import adapt_network_for_any_size_input
    number_of_classes = 38

    imgNum = 0 
    mainIOU = np.zeros((1,(len(sys.argv[1:])))) 
    net_reuse = 0

    for img_name in sys.argv[1:]:       
        print img_name 

        iimg_char = getIimg(img_name,softPhoc_net_char,number_of_classes)
        net_reuse +=1
        iimg_text = getIimg(img_name,softPhoc_net_text,number_of_classes)

        #Read the vocabulary from the gt
        voc  = open (img_name.split('.')[0]+'.txt').read()  # for the ch4 testing

        numQWords =  voc.count('\n')
        voc=re.sub(r'[^\x00-\x7f]',r'',voc)
        voc = voc.split('\r\n')

        #Read the text proposals of the correspond image
        csvName = img_name.split('.')[0]+'.csv'

        fileName = img_name.split('.')[0].split('/')[-1]

        #locRes = open (loc_res+fileName+'.txt', 'w')
        iou_res = open (iou_res_path+fileName+'.txt', 'w')
        
        subIOU = np.zeros((1,numQWords))
 
        img=cv2.imread(img_name.split('.')[0]+'.jpg')    
        cv_size = lambda img: tuple(img.shape[1::-1])
        width, height = cv_size(img)

        idnqw = 0                              
        for nqw in range(0,numQWords): 
            #qword = voc[nqw]
            if (len(voc[nqw])>0 and voc[nqw].split(',')[-1].strip()!='###'):
               idnqw +=1
               qword = voc[nqw].split(',')[-1]
               print 'query word is %s'%qword
               qHist = getQueryHistHOC2(qword,alphabet)
               
               res=textProposals(csvName,iimg_char,iimg_text,qHist)

               surf=(res[:,0]-res[:,2])*(res[:,1]-res[:,3])
               #res=res[surf>1200,:]
               idx=np.argsort(res[:,4])
               
               [l,t,r,b]=res[idx[-1],:4] 
                                  
               gtLeft=np.min([int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[6])],axis=0)
               gtTop=np.min([int(voc[nqw].split(',')[1]),int(voc[nqw].split(',')[3])],axis=0)
               gtRight=np.max([int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[4])],axis=0)
               gtBottom=np.max([int(voc[nqw].split(',')[5]),int(voc[nqw].split(',')[7])],axis=0)
               gtWidth=np.absolute(gtRight-gtLeft)
               gtHeight=np.absolute(gtTop-gtBottom)
               resLeft=int(l)
               resTop=int(t)
               resRight=int(r)
               resBottom=int(b)
               resWidth=np.absolute(resRight-resLeft)
               resHeight=np.absolute(resTop-resBottom)
               intL=np.max([resLeft,gtLeft],axis=0)
               intT=np.max([resTop,gtTop],axis=0)
               intR=np.min([resRight,gtRight],axis=0)
               intB=np.min([resBottom,gtBottom],axis=0)
               intW=(intR-intL)+1
               if intW<0:
                   intW = 0
               else:
                   intW=intW
               #intW[intW<0]=0 #'numpy.int64' object does not support item assignment
               intH=(intB-intT)+1
               if intH<0:
                   intH = 0
               else:
                   intH=intH
               #intH[intH<0]=0 #'numpy.int64' object does not support item assignment
               I=intH*intW
               U=resWidth*resHeight+gtWidth*gtHeight-I
               IoU=I/(U+.0000000001)
               #print "the subIoU is %f"%IoU
               subIOU[0][nqw] = IoU
                  
               iou_res.write("%d,%d,%d,%d,%d,%d,%d,%d,%s,%f\n" % (l,t,r,t,r,b,l,b,qword,IoU))
               iou_res.flush()

               #locRes.write("%d,%d,%d,%d,%d,%d,%d,%d\r\n" % (l,t,r,t,r,b,l,b))
                               
               pts = np.array([[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])],[int(voc[nqw].split(',')[2]),int(voc[nqw].split(',')[3])],[int(voc[nqw].split(',')[4]),int(voc[nqw].split(',')[5])],[int(voc[nqw].split(',')[6]),int(voc[nqw].split(',')[7])],[int(voc[nqw].split(',')[0]),int(voc[nqw].split(',')[1])]], np.int32)
               pts = pts.reshape((-1,1,2))
               cv2.polylines(img,[pts],True,(0,0,255),2)       
               cv2.putText(img,qword,(gtLeft,gtTop),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
               cv2.rectangle(img,(resLeft,resTop),(resRight,resBottom),(0,255,0),2)
               cv2.putText(img,qword,(resLeft,resTop),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
               
               cv2.imwrite(img_bbx_path+fileName+'.png',img)
               #print subIOU
                           
        if (idnqw!=0):       
            mainIOU[0][imgNum] = np.divide(np.sum(subIOU),idnqw)
            print "the IoU of is %f"%mainIOU[0][imgNum]
        else:
            mainIOU[0][imgNum] = 1.0000000
            cv2.imwrite(img_bbx_path+fileName+'.png',img)
        imgNum +=1                      
        iou_res.close()           
                   
    print "The total IoU is %f"%np.divide(np.sum(mainIOU),(imgNum))          #imgNum-1 or imgNum+1        
    resTot = open(iou_res_path+'totalIoU.txt','w')
    resTot.write ("sum of IoU: %f \nlength: %f \nmean IoU: %f\n" %(np.sum(mainIOU),imgNum, np.divide(np.sum(mainIOU),(imgNum))))
    resTot.close()  
