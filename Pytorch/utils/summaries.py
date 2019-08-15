import os
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, b_mask, b_enlarged_mask, global_step):
        # softmax = nn.Softmax(dim=1)
        # sigmoid = nn.Sigmoid()        
        # import pdb; pdb.set_trace()        
        # background = sigmoid(output[:,:1,:,:])
        # characters = softmax(output[:,1:,:,:])
        # output = torch.cat([background, characters], dim=1)

        # target_tb_ch0 = target[:,0,:,:]
        # target_tb_ch1 = torch.sum(target[:,1:,:,:], dim=1)
        # target_tb = torch.stack((target_tb_ch0,target_tb_ch1), dim=1)
    #def visualize_image(self, writer, dataset, image, target, output, global_step):
        #import pdb; pdb.set_trace()
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        #grid_image = make_grid(image.clone().cpu().data, 5, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
        #grid_image = make_grid(decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(),
        #                                               dataset=dataset), 5, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(target[:3], 1)[1].detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False, range=(0, 255))   
        #grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
        #                                                dataset=dataset), 3, normalize=False, range=(0, 255))
        #grid_image = make_grid(decode_seg_map_sequence(torch.max(target, 1)[1].detach().cpu().numpy(),
        #                                               dataset=dataset), 5, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
        #import pdb; pdb.set_trace()
        grid_image = make_grid(b_mask[:,0,:,:][:3].detach().cpu().data.unsqueeze(1)*255,3)
        writer.add_image('b_maskage', grid_image, global_step)
        grid_image = make_grid(b_enlarged_mask[:,0,:,:][:3].detach().cpu().data.unsqueeze(1)*255,3)
        writer.add_image('enlarged_b_mask', grid_image, global_step)