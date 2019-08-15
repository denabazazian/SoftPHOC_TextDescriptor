import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, cuda=False):
    #def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
    #def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=-100, cuda=False):
        #self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='tripleLoss_CE'):
        """Choices: ['ce' or 'focal' or 'BCE' or 'tripleLoss']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'BCE':
            return self.BCELoss
        elif mode == 'tripleLoss':
            return self.tripleLoss
        elif mode == 'tripleLoss_CE':
            return self.tripleLoss_CE    
        else:
            raise NotImplementedError

    def tripleLoss_CE (self, logit, target, b_mask, enlarged_b_mask):
        n, c, h, w = logit.size()
        ### to define the range of logits and discard outliers for normalizing
        # logits = torch.clamp(logits, min = -10, max=10)
        w1 = 0.01 #0.1  # 0.1
        w2 = 0.15 #1.0  # 1.0
        w3 = 0.35 #3.5  # 2.5        
        
        #norm_logsoftmax = nn.LogSoftmax(dim=1)
        norm_sigmoid = nn.Sigmoid()
        norm_softmax = nn.Softmax(dim=1)
        logit = norm_softmax(logit)
        #target = norm_softmax(target)

        criterion = nn.BCELoss()
        #criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.KLDivLoss()
        #criterion = nn.MSELoss()
        

        if self.cuda:
            criterion = criterion.cuda()

        target_tb_ch0 = target[:,0,:,:]
        target_tb_ch1 = torch.sum(target[:,1:,:,:], dim=1)
        target_tb = torch.stack((target_tb_ch0,target_tb_ch1), dim=1)

        logits_tb_ch0 = logit[:,0,:,:]
        logits_tb_ch1 = torch.sum(logit[:,1:,:,:], dim=1)
        logits_tb = torch.stack((logits_tb_ch0,logits_tb_ch1), dim=1)
        
        # normalize the two classes of text/non-text
        logits_tb = norm_sigmoid(logits_tb)
        target_tb = norm_sigmoid(target_tb)

        target1 = target_tb[b_mask==0]
        logits1 = logits_tb[b_mask==0]
        loss1 = criterion(logits1,target1)

        target2 = target_tb[b_mask]
        logits2 = logits_tb[b_mask]
        loss2 = criterion(logits2,target2)

        #import pdb; pdb.set_trace()
        target3 = target[enlarged_b_mask]
        logits3 = logit[enlarged_b_mask]
        loss3 = criterion(logits3,target3)

        #loss = criterion(logit,target)
        loss_total = (w1*loss1) + (w2*loss2) + (w3*loss3)

        if self.batch_average:
            loss1 /= n
            loss2 /= n
            loss3 /= n
            loss_total /= n
        
        #import pdb; pdb.set_trace()
        return loss1, loss2, loss3, loss_total
        #return loss_total


    def tripleLoss_CE_CESA (self, output, target, b_mask, enlarged_b_mask):
        n, c, h, w = output.size()
        use_mask = True
        ### to define the range of logits and discard outliers for normalizing
        # logits = torch.clamp(logits, min = -10, max=10)
        w1 = 0.1 #0.1  # 0.1
        w2 = 1.0 # 0.15 #1.0  # 1.0
        w3 = 0 #0.35 #3.5  # 2.5
        weights = [9.45366764e-01, 4.30531608e-03, 7.34439184e-04, 9.06505308e-04,
                   2.18408241e-03, 7.01268209e-03, 9.36291559e-04, 9.48284799e-04,
                   4.12308222e-03, 2.67910835e-03, 2.02433069e-04, 2.75845658e-04,
                   1.69573472e-03, 1.63786614e-03, 2.72039714e-03, 2.89382842e-03,
                   9.37674725e-04, 0.00000000e+00, 3.37702372e-03, 2.24371801e-03,
                   5.16348998e-03, 1.75837138e-03, 1.21487279e-03, 9.78910520e-04,
                   2.32306596e-04, 1.39811648e-03, 6.45206805e-05, 4.43325056e-04,
                   1.57445546e-04, 3.59270097e-04, 7.23875419e-06, 1.12298236e-05,
                   9.42513681e-05, 3.99242752e-05, 1.57124625e-04, 5.95923572e-04,
                   1.10866174e-04, 2.03173469e-03]
        # weights = [w2] * 38
        # weights[0] = w1

        all_loss = nn.BCEWithLogitsLoss(weight=torch.FloatTensor((1e-6 + np.array(weights)) ** -0.25))
        # bg_loss = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(np.array([w1])))
        # char_loss = nn.BCEWithLogitsLoss()
        #criterion = nn.KLDivLoss()
        #criterion = nn.MSELoss()

        # softmax = nn.Softmax(dim=1)
        if self.cuda:
            all_loss = all_loss.cuda()
            # bg_loss = bg_loss.cuda()
            # char_loss = char_loss.cuda()

        # target_is_background = target[:,0,:,:].view((n,1,h*w)) # needs to be between 0 and 1
        # output_is_background = output[:,0,:,:].view((n,1,h*w)) # needs to be -inf +inf
        # loss_is_background = bg_loss(output_is_background, target_is_background)

        # # import pdb; pdb.set_trace()

        # if use_mask:
        #     # n, w, h, 38
        #     enlarged_b_mask = enlarged_b_mask.view(n, 38, h*w)
        #     enlarged_b_mask = enlarged_b_mask.transpose(1, 2)

        target = target.view(n, 38, h*w)
        target = target.transpose(1, 2)
        target = target.reshape(-1, 38)

        output = output.view(n, 38, h*w)
        output = output.transpose(1, 2)
        output = output.reshape(-1, 38)

        #     target = target[enlarged_b_mask].view((-1, 38))
        #     output = output[enlarged_b_mask].view((-1, 38))

        # target_characters = softmax(target[:,1:].view((-1,37)))
        # output_characters = softmax(output[:,1:].view((-1,37)))
        # loss_characters = char_loss(output_characters, target_characters)

        # import pdb; pdb.set_trace()
        loss_total = all_loss(output, target)

        # loss_total = loss_is_background * w2 + loss_characters * w3


        # target_tb_ch0 = target[:,0,:,:]
        # target_tb_ch1 = torch.sum(target[:,1:,:,:], dim=1)
        # target_tb = torch.stack((target_tb_ch0,target_tb_ch1), dim=1)

        # logits_tb_ch0 = logit[:,0,:,:]
        # logits_tb_ch1 = torch.sum(logit[:,1:,:,:], dim=1)
        # logits_tb = torch.stack((logits_tb_ch0,logits_tb_ch1), dim=1)


        # target1 = normal(target_tb[b_mask==0])
        # logits1 = normal(logits_tb[b_mask==0])
        # #loss1 = criterion(logits1,target1)

        # target2 = normal(target_tb[b_mask])
        # logits2 = normal(logits_tb[b_mask])
        # #loss2 = criterion(logits2,target2)

        # #import pdb; pdb.set_trace()
        # target3 = target[enlarged_b_mask]
        # logits3 = logit[enlarged_b_mask]
        # loss3 = criterion(logits3,target3)

        # #loss = criterion(logit,target)
        # loss_total = (w1*loss1) + (w2*loss2) + (w3*loss3)

        # if self.batch_average:
        #     loss1 /= n
        #     loss2 /= n
        #     loss3 /= n
        #     loss_total /= n
        
        #import pdb; pdb.set_trace()
        return loss_total, loss_total, loss_total, loss_total
        # return loss_is_background, loss_characters, loss_characters, loss_total
        #return loss_total



    def tripleLoss (self, logit, target, b_mask, enlarged_b_mask):
        n, c, h, w = logit.size()
        ### to define the range of logits and discard outliers for normalizing
        # logits = torch.clamp(logits, min = -10, max=10)
        w1 = 0.01 #0.1  # 0.1
        w2 = 0.15 #1.0  # 1.0
        w3 = 0.35 #3.5  # 2.5        
        #criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.KLDivLoss()
        criterion = nn.MSELoss()
        if self.cuda:
            criterion = criterion.cuda()

        target_tb_ch0 = target[:,0,:,:]
        target_tb_ch1 = torch.sum(target[:,1:,:,:], dim=1)
        target_tb = torch.stack((target_tb_ch0,target_tb_ch1), dim=1)

        logits_tb_ch0 = logit[:,0,:,:]
        logits_tb_ch1 = torch.sum(logit[:,1:,:,:], dim=1)
        logits_tb = torch.stack((logits_tb_ch0,logits_tb_ch1), dim=1)

        target1 = target_tb[b_mask==0]
        logits1 = logits_tb[b_mask==0]
        loss1 = criterion(logits1,target1)

        target2 = target_tb[b_mask]
        logits2 = logits_tb[b_mask]
        loss2 = criterion(logits2,target2)

        #import pdb; pdb.set_trace()
        target3 = target[enlarged_b_mask]
        logits3 = logit[enlarged_b_mask]
        loss3 = criterion(logits3,target3)

        #loss = criterion(logit,target)
        loss_total = (w1*loss1) + (w2*loss2) + (w3*loss3)

        if self.batch_average:
            loss1 /= n
            loss2 /= n
            loss3 /= n
            loss_total /= n
        
        #import pdb; pdb.set_trace()
        return loss1, loss2, loss3, loss_total
        #return loss_total
        
    def BCELoss (self, logit, target):
        n, c, h, w = logit.size()
        # print('logit size n: {}, c:{}, h:{}, w:{}'.format(n,c,h,w))
        # n_t, c_t, h_t, w_t = target.size()
        # print('target size n: {}, c:{}, h:{}, w:{}'.format(n_t,c_t,h_t,w_t))
        
        #import pdb; pdb.set_trace()
       
        ### values should be between 0 and 1 
        #max_value = torch.max(logit[1,1,:,:],1)
        #min_value = torch.min(logit[1,1,:,:],1)
        
        # print('max value is:{}'.format(max_value))
        # print('min value is:{}'.format(min_value))

        #pred = F.softmax(logit, dim=1)
        #criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit,target)

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
        #                                 size_average=self.size_average)
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        size_average=self.size_average)
        # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
        #                                 size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




