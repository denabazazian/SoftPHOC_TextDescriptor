
import numpy as np
import scipy.io
import re
import os


def get_char_cor(char_BB):

    top_left_x = np.asarray(char_BB[0][0])
    top_right_x = np.asarray(char_BB[0][1])
    bottom_right_x = np.asarray(char_BB[0][2])
    bottom_left_x = np.asarray(char_BB[0][3])
    top_left_y = np.asarray(char_BB[1][0])
    top_right_y = np.asarray(char_BB[1][1])
    bottom_right_y = np.asarray(char_BB[1][2])
    bottom_left_y = np.asarray(char_BB[1][3])

    return top_left_x,top_right_x,bottom_right_x,bottom_left_x,top_left_y,top_right_y,bottom_right_y,bottom_left_y

def get_bbx_polygon(c,c_end,top_left_x,top_right_x,bottom_right_x,bottom_left_x,top_left_y,top_right_y,bottom_right_y,bottom_left_y):
    char_top_left_x = []
    char_top_right_x = []
    char_bottom_right_x = []
    char_bottom_left_x = []
    char_top_left_y = []
    char_top_right_y = []
    char_bottom_right_y = []
    char_bottom_left_y = []

    for w in range (c,c_end):
        char_top_left_x.append(top_left_x[w])
        char_top_right_x.append(top_right_x[w])
        char_bottom_right_x.append(bottom_right_x[w])
        char_bottom_left_x.append(bottom_left_x[w])
        char_top_left_y.append(top_left_y[w])
        char_top_right_y.append(top_right_y[w])
        char_bottom_right_y.append(bottom_right_y[w])
        char_bottom_left_y.append(bottom_left_y[w])

    word_top_left_x = min(char_top_left_x)
    word_top_right_x = max(char_top_right_x)
    word_bottom_right_x = max(char_bottom_right_x)
    word_bottom_left_x = min(char_bottom_left_x)
    word_top_left_y = min(char_top_left_y)
    word_top_right_y = min(char_top_right_y)
    word_bottom_right_y = max(char_bottom_right_y)
    word_bottom_left_y = max(char_bottom_left_y)

    return word_top_left_x,word_top_right_x,word_bottom_right_x,word_bottom_left_x,word_top_left_y,word_top_right_y,word_bottom_right_y,word_bottom_left_y


if __name__=='__main__':
   pathToGtMat='/path/to/datasets/synthtext/gt.mat'
   gtTXTPath_word = '/path/to/datasets/synthtext/gt/gt_word_polygon/'
   gtTXTPath_char = '/path/to/datasets/synthtext/gt/gt_char/'

   m=scipy.io.matlab.loadmat(pathToGtMat)
   files,txt,wordBB,charBB= m['imnames'],m['txt'],m['wordBB'],m['charBB']

   #for imgPath in sys.argv[1:]:
   for i in range(6459, 6465):
   #for i in range(789, 795):
   #for i in range(0, len(files[0])):
        imgFolder = str(files[0][i]).split('/')[0].split("'")[-1]
        imgName = str(files[0][i]).split('/')[1].split("'")[0]
        gtName = imgName.split('.')[0]+'.txt'

        # for the word level detection
        if not os.path.exists(gtTXTPath_word+imgFolder):
            os.makedirs(gtTXTPath_word+imgFolder)

        word_gt = open(gtTXTPath_word+imgFolder+'/'+gtName, "w")
        words_list = []
        for w in txt[0][i]:
            words_list += re.sub(r'\s+','\n',w).strip().split('\n')

        char_list = []
        char_list = ''.join([str(w)for w in txt[0,i]])
        char_list = ''.join(char_list.split())
        
        char_BB = charBB[0][i]
        top_left_x,top_right_x,bottom_right_x,bottom_left_x,top_left_y,top_right_y,bottom_right_y,bottom_left_y = get_char_cor(char_BB)

        c = 0 # counting all the characters
        for j in range(0, len(words_list)):
            len_word = len(words_list[j])
            c_end = c+len_word
            word_top_left_x,word_top_right_x,word_bottom_right_x,word_bottom_left_x,word_top_left_y,word_top_right_y,word_bottom_right_y,word_bottom_left_y = get_bbx_polygon(
                c,c_end,top_left_x,top_right_x,bottom_right_x,bottom_left_x,top_left_y,top_right_y,bottom_right_y,bottom_left_y)

            word_gt.write(str(int(word_top_left_x))+','+str(int(word_top_left_y))+','+
                  str(int(word_top_right_x))+','+str(int(word_top_right_y))+','+
                  str(int(word_bottom_right_x))+','+str(int(word_bottom_right_y))+','+
                  str(int(word_bottom_left_x))+','+str(int(word_bottom_left_y))+','+
                  str(words_list[j])+'\n')
            c=c+len_word

        word_gt.close()
        print i 
        print files[0][i]






        # #write the char in txt
        # if not os.path.exists(gtTXTPath_char+imgFolder):
        #     os.makedirs(gtTXTPath_char+imgFolder)
        # char_gt = open(gtTXTPath_char+imgFolder+'/'+gtName, "w")
        # for c in range(0,len(char_list)):