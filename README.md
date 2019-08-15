# SoftPHOC_TextDescriptor
Soft-PHOC is an intermediate representation of images based on character probability maps.

[this work](arxiv.org/pdf/1809.00854.pdf) has two implementations based on Pytorch and TensorFlow.


# Pytorch #

The pytorch implementation of SoftPHOC training.

## Installation ##

Find the environmet at: environment.yml
```
conda install python=3.6 ipython pytorch=0.4 torchvision opencv=3.4.4 tensorboardx mkl=2019 tensorboard tensorflow tqdm scikit-image
```
* Required packages:
    * Pytorch 0.4
    * OpenCV 3.4.4
    * mkl 2019
    * tqm
    * scikit-image
    * tensorboardX

### train ###

* For training ICDAR:
``` 
bash train_icdar.sh
```

* For training SynthText:
``` 
bash train_synthText.sh
```

# TensorFlow #

The TensorFlow implementation of Soft-PHOC. 


#Citation
Please cite [this work](arxiv.org/pdf/1809.00854.pdf) in your publications if it helps your research: 

@article{Bazazian18-softPHOC,
	author = {D.~Bazazian and D.~Karatzas and A.~Bagdanov},
	title = {Soft-PHOC Descriptor for End-to-End Word Spotting in Egocentric Scene Images},
	journal = {EPIC workshop at European Conference on Computer Vision Workshop},
	year = {2018},
        ee = {arxiv.org/pdf/1809.00854.pdf}
}
