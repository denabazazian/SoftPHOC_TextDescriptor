#cd /softPHOC_scenImages
#python visualize_mat_gt.py /path/to/deeplabV3Plus/icdar_models_softPHOC/mat_files/img_126_resized.mat
#python visualize_mat_gt_resized.py /path/to/deeplabV3Plus/icdar_models_softPHOC/mat_files/hiking_0_88_resized.mat
from __future__ import division
import scipy
import scipy.io
import cv2
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.misc import imresize, imsave, toimage


if __name__=='__main__':    

    alphabet="#abcdefghijklmnopqrstuvwxyz1234567890@"

    save_softPHOC_hm_path = '/path/to/deeplabV3Plus/icdar_models_softPHOC/softPHOC_test/softPHOC_png/resized/'

    for softPHOCpath in sys.argv[1:]:
    #for test in range(0,1):  
        #softPHOCpath='/path/to/deeplabV3Plus/icdar_models_softPHOC/img_49.mat'

        softPHOC = scipy.io.loadmat(softPHOCpath)['softPHOC']
        b_mask = scipy.io.loadmat(softPHOCpath)['b_mask']
        b_enlarged_mask = scipy.io.loadmat(softPHOCpath)['enlarged_mask']

        imgName = ((softPHOCpath.split('/')[-1]).split('.'))[0]+'.jpg'
        #img_path = '/path/to/dataset/ICDAR2015/ch4_train_img/'+'img_126.jpg'   #imgName
        img_path = '/path/to/dataset/SynthText/synthText_deepLabV3Plus/SynthText/70/hiking_0_88.jpg'
        print(img_path)
        img = cv2.imread(img_path)
        cv_size = lambda img: tuple(img.shape[1::-1]) 
        width_main, height_main = cv_size(img)
        print(width_main)
        print(height_main)
        img = cv2.resize(img, (512, 512))
        cv_size = lambda img: tuple(img.shape[1::-1]) 
        width_main, height_main = cv_size(img)
        print(width_main)
        print(height_main)



        fig = plt.figure()
        plt.imshow(b_mask)
        plt.imshow(img,alpha=0.5)
        plt.colorbar()
        plt.title('b_mask')
        fig.savefig(save_softPHOC_hm_path+"b_mask.png")
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(b_enlarged_mask)
        plt.imshow(img,alpha=0.5)
        plt.colorbar()
        plt.title('b_enlarged_mask')
        fig.savefig(save_softPHOC_hm_path+"b_enlarged_mask.png")
        plt.close(fig)

        for i in range(0,38):
            fig = plt.figure()
            #plt.imshow(softPHOC[:,:,i], vmin = 0 , vmax = 1)
            plt.imshow(softPHOC[:,:,i])
            plt.imshow(img,alpha=0.5)
            plt.colorbar()
            plt.title(('%d,%s')%(i,alphabet[i]))
            fig.savefig(save_softPHOC_hm_path+"hm_{}_{}.png".format(i, alphabet[i]))
            plt.close(fig)





        # # fig = plt.figure()
        # # #fig.suptitle(transcription, fontsize=14)  
        # # fig.subplots_adjust(hspace=.1)                    
        # for i in range (0,38):
        #     #fig.add_subplot(8,5,(i+1)).imshow(softPhoc_main[:,:,i], vmin = 0 , vmax = 1)
        #     #plt.colorbar(fig.add_subplot(8,5,(i+1)).imshow(softPhoc_main[:,:,i], vmin = 0 , vmax = 1))
        #     #plt.imshow(softPHOC[:,:,i])
        #     plt.imshow(softPHOC[:,:,i], vmin = 0 , vmax = 1)
        #     plt.imshow(img, alpha=.5)
        #     #fig.add_subplot(8,5,(i+1)).set_title(('%d,%s')%(i,alphabet[i]))
        #     plt.title(('%d,%s')%(i,alphabet[i]))
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.show()
