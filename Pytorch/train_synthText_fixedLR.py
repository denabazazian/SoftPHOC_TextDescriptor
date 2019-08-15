import argparse
import os
import numpy as np
from tqdm import tqdm
import gc

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


# def label_stats(loader, nimg=100 ):
#     labels = np.zeros(38, dtype='float64')
#     n = 0
#     for sample in tqdm(loader):
#         target = sample['label']
#         #labels += np.bincount(target.cpu().numpy().ravel(),minlength=38)
#         # import pdb; pdb.set_trace()
#         labels += np.mean(target.cpu().numpy(), axis=(0,2,3))
#         n += 1
#         if n >= nimg: break
#     return labels / labels.sum()



class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        #### if initializer
        if args.init is not None:
            model = DeepLab(num_classes=21,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)
        else:
            model = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr}]


        # Define Optimizer
        # optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)

        optimizer = torch.optim.Adam(train_params,lr=args.lr,weight_decay=args.weight_decay,amsgrad=True)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        #self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                                    args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
            
        #initializing network
        if args.init is not None:
            if not os.path.isfile(args.init):
                raise RuntimeError("=> no initializer checkpoint found at '{}'" .format(args.init))
            checkpoint = torch.load(args.init)
            #args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            # del state_dict["decoder.last_conv.8.weight"]
            # del state_dict["decoder.last_conv.8.bias"]
            if args.cuda:
                self.model.module.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)
            # if not args.ft:
            #     self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.best_pred = checkpoint['best_pred']
            self.model.module.decoder.last_layer = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1).cuda()
            print("=> loaded initializer '{}' (epoch {})"
                  .format(args.init, checkpoint['epoch']))

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            ##state_dict = checkpoint['state_dict']
            ## del state_dict["decoder.last_conv.8.weight"]
            ## del state_dict["decoder.last_conv.8.bias"]
            if args.cuda:
                #self.model.module.load_state_dict(state_dict, strict=False)
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                #self.model.load_state_dict(state_dict, strict=False)
                self.model.load_state_dict(checkpoint['state_dict'])
            # if not args.ft:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

            #self.model.module.decoder.last_layer = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1) 

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        self.model.train()
        
        # trying to save a checkpoint and check if it exists...
        # import os
        # cur_path = os.path.dirname(os.path.abspath('.'))
        # print('saving mycheckpoint in:' + cur_path )
        # checkpoint_name = 'mycheckpoint.pth.tar'
        # save_path = cur_path + '/' + checkpoint_name
        # torch.save(self.model.module.state_dict(), save_path)
        # assert(os.path.isfile(save_path))
        # # torch.save(self.model.module.state_dict(), checkpoint_name)
        # # assert(os.path.isfile(cur_path + '/' + checkpoint_name))
        # print('checkpoint saved ok')
        # # checkpoint saved

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        # import pdb; pdb.set_trace()
        # label_w = label_stats(self.train_loader, nimg=70) #** -args.norm_loss if args.norm_loss != 0 else None
        # import pdb; pdb.set_trace()

        for i, sample in enumerate(tbar):
            #print("i is:{}, index is:{}".format(i,sample['index']))
            #print("path is:{}".format(sample['path']))
            #image, target = sample['image'], sample['label']
            image, target, index, path, b_mask, enlarged_b_mask = sample['image'], sample['label'], sample['index'], sample['path'], sample['b_mask'], sample['enlarged_b_mask']
            #print('sample for training index is :{} and path is:{}'.format(index,path))
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            #not using learning rate scheduler and apply a fixed learning rate    
            #self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            #import pdb; pdb.set_trace()
            #loss = self.criterion(output, target, b_mask, enlarged_b_mask)
            loss1, loss2, loss3, loss = self.criterion(output, target, b_mask, enlarged_b_mask)
            # criterion = nn.BCELoss()
            # loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            #import pdb; pdb.set_trace()
            tbar.set_description('Train loss_total: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            # tbar.set_description('Train loss1: %.3f' % (train_loss1 / (i + 1)))
            self.writer.add_scalar('train/total_loss1_iter', loss1.item(), i + num_img_tr * epoch)
            # tbar.set_description('Train loss2: %.3f' % (train_loss2 / (i + 1)))
            self.writer.add_scalar('train/total_loss2_iter', loss2.item(), i + num_img_tr * epoch)
            # tbar.set_description('Train loss3: %.3f' % (train_loss3 / (i + 1)))
            self.writer.add_scalar('train/total_loss3_iter', loss3.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10000) == 0:   #for the whole dataset
            #if i % (num_img_tr // 10) == 0:    # for debugging
                global_step = i + num_img_tr * epoch
                #self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, b_mask, enlarged_b_mask, global_step)
            # Save the model after each 500 iterations
            if i % 500 == 0:  #for the whole dataset
            #if i % 5 == 0:    # for debugging
                is_best = False
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)

            # perform the validation after each 1000 iterations
            if i % 1000 == 0 :  #for the whole dataset
            #if i % 15 == 0 :    # for debugging
                self.validation(i)
                #self.validation(i + num_img_tr * epoch)

            ## garbage collection pass
            #del image, target, index, path, b_mask, enlarged_b_mask, output
            #gc.collect()    

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/total_loss1_epoch', train_loss1, epoch)
        self.writer.add_scalar('train/total_loss2_epoch', train_loss2, epoch)
        self.writer.add_scalar('train/total_loss3_epoch', train_loss3, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            #print('saving checkpoint')
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        test_loss1 = 0.0
        test_loss2 = 0.0
        test_loss3 = 0.0
        for i, sample in enumerate(tbar):
            #image, target = sample['image'], sample['label']
            #image, target, index, path = sample['image'], sample['label'], sample['index'], sample['path']
            image, target, index, path, b_mask, enlarged_b_mask = sample['image'], sample['label'], sample['index'], sample['path'], sample['b_mask'],sample['enlarged_b_mask']            
            #print('sample for testing index is :{} and path is:{}'.format(index,path))
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            #loss = self.criterion(output, target)
            #loss = self.criterion(output, target, b_mask, enlarged_b_mask)
            loss1, loss2, loss3, loss = self.criterion(output, target, b_mask, enlarged_b_mask)
            test_loss += loss.item()
            test_loss1 += loss1.item()
            test_loss2 += loss2.item()
            test_loss3 += loss3.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = np.argmax(target, axis=1)
            #pred = np.argmax(pred, axis=1)
            #import pdb; pdb.set_trace()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/total_loss1_epoch', test_loss1, epoch)
        self.writer.add_scalar('val/total_loss2_epoch', test_loss2, epoch)
        self.writer.add_scalar('val/total_loss3_epoch', test_loss3, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        ## garbage collection pass
        #del image, target, index, path, b_mask, enlarged_b_mask, output
        #gc.collect()    

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='icdar',
                        choices=['icdar','synthText'],
                        help='dataset name (default: icdar)')
    parser.add_argument('--saveRoot', type=str, default='tmp-network',
                        choices=['tmp-network','nfs' , 'home'],
                        help='root to save experiments (default: tmp-network)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=True,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='tripleLoss_CE',
                        choices=['ce', 'focal' , 'BCE', 'tripleLoss', 'tripleLoss_CE'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #initializer from pascal
    parser.add_argument('--init', type=str, default=None,
                        help='put the path to initialize the training')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'icdar': 100,
            'synthText': 100,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'icdar': 0.003,
            'synthText': 0.0001,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
