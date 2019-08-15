from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
#from torchvision import transforms
#from dataloaders import custom_transforms as tr
#from datasets import generate_segLabels
from dataloaders.generate_segLabels_synthText import generate_segmentation_softPHOC_labels


class synthTextSegmentation(Dataset):
    """
    SynthText dataset
    """
    NUM_CLASSES = 38

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('synthText'),
                 split='train',
                 ):
        """
        :param base_dir: path to synthText dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'SynthText')
        #self._cat_dir = os.path.join(self._base_dir, 'ch4_train_gt_png')
        self._cat_coord_bbx_dir = os.path.join(self._base_dir, 'gt_word_polygon')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        #_splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                #lines = f.read().splitlines()
                lines = [l for l in (line.strip() for line in f) if l]

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                #_cat = os.path.join(self._cat_dir, line + ".png")
                #_cat = generate_segmentation_labels(self._cat_coord_bbx_dir,self._image_dir,line)
                _cat = os.path.join(self._cat_coord_bbx_dir, line + ".txt")

                print('_image is : {}'.format(_image))
                print('_cat is: {}'.format(_cat))
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                #self.categories.append(generate_segmentation_softPHOC_labels(self._cat_coord_bbx_dir,self._image_dir,line))

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target, b_mask, enlarged_b_mask = self._make_img_gt_point_pair(index)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        _img = np.array(_img).astype(np.float32).transpose((2, 0, 1))
        _target = np.array(_target).astype(np.float32).transpose((2, 0, 1))
        #b_mask = np.array(b_mask).astype(np.uint8).transpose((2, 0, 1))
        #enlarged_b_mask = np.array(enlarged_b_mask).astype(np.uint8).transpose((2, 0, 1))
        b_mask = np.array(b_mask).astype(np.uint8)  #.repeat(38,1,1)
        enlarged_b_mask = np.array(enlarged_b_mask).astype(np.uint8)  #.repeat(38,1,1)

        _img = torch.from_numpy(_img).float()
        _target = torch.from_numpy(_target).float()
        b_mask = (torch.from_numpy(b_mask).byte()).repeat(2,1,1)
        enlarged_b_mask = (torch.from_numpy(enlarged_b_mask).byte()).repeat(38,1,1)

        # N, C, H, W = _target.shape
        # for n in range(N):
        #     for w in range(W):
        #         for h in range(H):
        #             assert np.all(enlarged_b_mask[n, :, W, H].cpu().numpy() == enlarged_b_mask[n, 0, W, H].cpu().numpy())

        # import pdb; pdb.set_trace()

        #sample = {'image': _img, 'label': _target}
        sample = {'image': _img, 'label': _target, 'b_mask': b_mask, 'enlarged_b_mask':enlarged_b_mask, 'index': index, 'path': self.images[index]}

        # discard the data augmentation for now!
        # for split in self.split:
        #     if split == "train":
        #         return self.transform_tr(sample)
        #     elif split == 'val':
        #         return self.transform_val(sample)
        # for split in self.split:
        #     if split == "train":
        #         return sample
        #     elif split == 'val':
        #         return sample
        return sample

    def _make_img_gt_point_pair(self, index):
        #_img = Image.open(self.images[index]).convert('RGB')
        #_target = Image.open(self.categories[index])
        #print('img_indx is: {}'.format(self.images[index]))
        #print('gt_indx is: {}'.format(self.categories[index]))
        # _target = Image.fromarray(generate_segmentation_softPHOC_labels(self.categories[index],self.images[index]))
        #_target = np.array(generate_segmentation_softPHOC_labels(self.categories[index],self.images[index])).astype(np.float32)
        img, softphoc_label, b_mask, enlarged_b_mask = generate_segmentation_softPHOC_labels(self.categories[index],self.images[index])
        #_img = Image.open(img).convert('RGB')
        _img = Image.fromarray(img).convert('RGB')
        _target = softphoc_label
        # sio.savemat('/tmp-network/user/dbazazian/deeplabV3Plus/icdar_models_softPHOC/' + 'logits_target' + '.mat',     
        #         {'logit':logit, 'target':target}
        #         ) 
        # print('logits and target has been saved!')

        return _img, _target, b_mask, enlarged_b_mask

    # def transform_tr(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip(),
    #         tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
    #         tr.RandomGaussianBlur(),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # def transform_val(self, sample):

    #     composed_transforms = transforms.Compose([
    #         tr.FixScaleCrop(crop_size=self.args.crop_size),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # not sure is I can comment it?!?!
    def __str__(self):
        return 'synthText(split=' + str(self.split) + ')'


if __name__ == '__main__':
    #from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.base_size = 513
    # args.crop_size = 513

    synthText_train = synthTextSegmentation(args, split='train')
    #print('synthText_train:')
    #print(synthText_train)

    #dataloader = DataLoader(synthText_train, batch_size=2, shuffle=True, num_workers=0)
    dataloader = DataLoader(synthText_train, batch_size=2, shuffle=True, num_workers=4)

    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["image"].size()[0]):
    #         img = sample['image'].numpy()
    #         gt = sample['label'].numpy()
    #         tmp = np.array(gt[jj]).astype(np.uint8)
    #         segmap = decode_segmap(tmp, dataset='icdar')
    #         img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #         img_tmp *= (0.229, 0.224, 0.225)
    #         img_tmp += (0.485, 0.456, 0.406)
    #         img_tmp *= 255.0
    #         img_tmp = img_tmp.astype(np.uint8)
    #         plt.figure()
    #         plt.title('display')
    #         plt.subplot(211)
    #         plt.imshow(img_tmp)
    #         plt.subplot(212)
    #         plt.imshow(segmap)

    #     if ii == 1:
    #         break

    # plt.show(block=True)


