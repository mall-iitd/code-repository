import argparse
import os
import numpy as np
from torchvision import transforms

from tqdm import tqdm
from utils.helpers import gen_train_dirs, plot_confusion_matrix, get_train_trans, get_test_trans
from utils.routines import train_epoch, evaluate
from datasets.cityscapes_ext import CityscapesExt
from torch.utils.data import DataLoader, ConcatDataset

from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from models.refinenet import RefineNet
from datasets.foggy_driving import foggyDrivingDataset
from datasets.foggy_driving_full import foggyDrivingFullDataset
from datasets.foggy_zurich import foggyZurichDataset
from datasets.overcast import overcastDataset

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from datasets.cityscapes_ext import CityscapesExt
import network
import matplotlib.pyplot as plt
from datasets.foggy_driving import foggyDrivingDataset
from datasets.foggy_driving_full import foggyDrivingFullDataset
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from DenseNCLoss import DenseNCLoss

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
torch.cuda.empty_cache()

import shutil, time, random
import matplotlib.pyplot as plt
import numpy as np

miou_nd=[]
iteration_list=[]


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
        model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
        # Define network
        
        #model = DeepLab(num_classes=self.nclass,
        #                backbone=args.backbone,
        #                output_stride=args.out_stride,
        #                sync_bn=args.sync_bn,
        #                freeze_bn=args.freeze_bn)
        #model = RefineNet(num_classes=self.nclass+1, pretrained=False)
        if args.base_model=='deeplabv3+_mobilenet':
            model = network.deeplabv3plus_mobilenet(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3+_resnet101':
            model = network.deeplabv3plus_resnet101(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3plus_resnet50':
            model = network.deeplabv3plus_resnet50(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3_resnet50':
            model = network.deeplabv3_resnet50(num_classes=self.nclass, output_stride=args.out_stride)
        #print(model)
        #checkpoint = torch.load('/home/anonymous6295/scratch/CIConv_zero_shot/experiments/3_segmentation/runs/20211014-212540/weights/checkpoint.pth')

        #if torch.cuda.is_available():
        #    model = torch.nn.DataParallel(model).cuda()
        #    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        #model.load_state_dict(checkpoint['model_state'], strict=True)
        #train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        #optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
            #                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        #print(model.backbone)
        
        optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
        ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
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
        
        if args.densencloss >0:
            self.densenclosslayer = DenseNCLoss(weight=args.densencloss, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)
            print(self.densenclosslayer)
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 108
            #args.start_epoch = checkpoint['epoch']
            if args.cuda and  args.base_model!='deeplabv3plus_resnet50':
                self.model.module.load_state_dict(checkpoint['model_state'])
                #self.model.load_state_dict(checkpoint['state_dict'])
                #print(checkpoint['state_dict'])
                #self.model.module.load_state_dict(checkpoint['state_dict'],strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            #if not args.ft:
            ##    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
             #   self.best_pred = checkpoint['best_score']
            #print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(args.resume, checkpoint['cur_itrs']))


        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        train_celoss = 0.0
        train_ncloss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        softmax = nn.Softmax(dim=1)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            croppings = (target!=254).float()
            target[target==254]=255
            # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
            # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            
            celoss = self.criterion(output, target)
            
            if self.args.densencloss ==0:
                loss = celoss
            else:
                probs = softmax(output)
                denormalized_image = denormalizeimage(sample['image'], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
                if self.args.cuda:
                    densencloss = densencloss.cuda()
                #loss = celoss + densencloss
                loss = 0.01*densencloss
                train_ncloss += densencloss.item()
            loss.backward()
        
            self.optimizer.step()
            train_loss += loss.item()
            train_celoss += celoss.item()
            
            tbar.set_description('Train loss: %.3f = CE loss %.3f + CRF loss: %.3f' 
                             % (train_loss / (i + 1),train_celoss / (i + 1),train_ncloss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        #if self.args.no_val:
        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))

    def test_time_training(self,trainer,dataloaders, epoch,loader):
        train_loss = 0.0
        train_celoss = 0.0
        train_ncloss = 0.0
        self.model.train()
        #tbar = tqdm(self.train_loader)
        tbar = tqdm(loader)
        #print(loader)
        iteration=0
        #num_img_tr = len(self.train_loader)
        num_img_tr = len(loader)
        #criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
        softmax = nn.Softmax(dim=1)
        #for i, sample in enumerate(tbar):
        #for i, (inputs, labels) in enumerate(loader):
        for i, (inputs, labels,filepath) in enumerate(loader):
            #image, target = sample['image'], sample['label']
            image = inputs
            target = labels
            image = image.float()
            target = target.long()
            #print(image)
            iteration = iteration+1
            #print(labels)
            iteration_list.append(iteration)
            croppings = (target!=254).float()
            target[target==254]=255
            # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
            # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            #corrects = torch.sum(output ==target)
            #print(corrects)
            #print(output.size())
            #print(target.size())
            #celoss = criterion(output, target)
            if self.args.densencloss ==0:
                loss = celoss
            else:
                print("nikil")
                probs = softmax(output)
                denormalized_image = denormalizeimage(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
                if self.args.cuda:
                    densencloss = densencloss.cuda()
                #loss = celoss + densencloss
                loss = 0.01*densencloss
                train_ncloss += densencloss.item()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            #train_celoss += celoss.item()
            #output = self.model(image)
            #corrects = torch.sum(output[0] ==target)
            #print(corrects)
            print('\n')
            trainer.validation(epoch)
            print("iteration")
            print('Night Driving mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
            miou_nd.append(miou*100)
            print(miou)
            print('Dark Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
            print(miou)
            print('Foggy Driving Dense mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
            print(miou)
            print('Foggy Driving Full mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
            print(miou)
            print('Foggy Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
            print(miou)
    
            
            tbar.set_description('Train loss: %.3f = CE loss %.3f + NC loss: %.3f' 
                             % (train_loss / (i + 1),train_celoss / (i + 1),train_ncloss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            #if i % (num_img_tr // 10) == 0:
            #    global_step = i + num_img_tr * epoch
            #    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        #if self.args.no_val:
        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))

    def test_time_training_source(self,trainer,dataloaders, epoch):
        train_loss = 0.0
        train_celoss = 0.0
        train_ncloss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        #tbar = tqdm(loader)
        #print(loader)
        iteration=0
        num_img_tr = len(self.train_loader)
        #num_img_tr = len(loader)
        criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
        softmax = nn.Softmax(dim=1)
        for i, sample in enumerate(tbar):
        #for i, (inputs, labels) in enumerate(self.train_loader):
        #for i, (inputs, labels,filepath) in enumerate(loader):
            image, target = sample['image'], sample['label']
            #image = inputs
            #target = labels
            image = image.float()
            target = target.long()
            #print(image)
            iteration = iteration+1
            #print(labels)
            iteration_list.append(iteration)
            croppings = (target!=254).float()
            target[target==254]=255
            # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
            # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            #print(output.size())
            #print(target.size())
            #celoss = criterion(output, target)
            if self.args.densencloss ==0:
                loss = celoss
            else:
                print("anonymous6295")
                probs = softmax(output)
                denormalized_image = denormalizeimage(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
                if self.args.cuda:
                    densencloss = densencloss.cuda()
                #loss = celoss + densencloss
                loss = 0.01*densencloss
                train_ncloss += densencloss.item()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            #train_celoss += celoss.item()
            print('\n')
            trainer.validation(epoch)
            print("iteration")
            print('Night Driving mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
            miou_nd.append(miou*100)
            print(miou)
            print('Dark Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
            print(miou)
            print('Foggy Driving Dense mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
            print(miou)
            print('Foggy Driving Full mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
            print(miou)
            print('Foggy Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
            print(miou)
    
            
            tbar.set_description('Train loss: %.3f = CE loss %.3f + NC loss: %.3f' 
                             % (train_loss / (i + 1),train_celoss / (i + 1),train_ncloss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            #if i % (num_img_tr // 10) == 0:
            #    global_step = i + num_img_tr * epoch
            #    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        #if self.args.no_val:
        if self.args.save_interval:
            # save checkpoint every interval epoch
            is_best = False
            if (epoch + 1) % self.args.save_interval == 0:
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best, filename='checkpoint_epoch_{}.pth.tar'.format(str(epoch+1)))

    


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target[target==254]=255
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
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

    def validation_nd(self,epoch,data):
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225) 
        target_size = (512,1024)
        crop_size = (384,768)
        #print(CityscapesExt.voidClass)
        test_acc_nd, test_loss_nd, miou_nd, confmat_nd, iousum_nd = evaluate(data,
            self.model, self.criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses,void=CityscapesExt.voidClass,optimizer=self.optimizer,
             maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        return miou_nd
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
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
    parser.add_argument('--test', type=int, default=1,required=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
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
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--base_model', type=str, default='deeplabv3+_mobilenet',
                        help='Base model')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # model saving option
    parser.add_argument('--save-interval', type=int, default=None,
                        help='save model interval in epochs')


    # rloss options
    parser.add_argument('--densencloss', type=float, default=0,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale',type=float,default=1.0,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()])
    cs_path = '/home/anonymous6295/scratch/datasets/cityscapes/'
    nd_path = '/home/anonymous6295/scratch/datasets/NighttimeDrivingTest/'
    dz_path = '/home/anonymous6295/scratch/datasets/Dark_Zurich_val_anon/'
    fd_path= '/home/anonymous6295/scratch/datasets/Foggy_Driving/'
    fz_path = '/home/anonymous6295/scratch/datasets/Foggy_Driving_Full/'
    fz_actual_path = '/home/anonymous6295/scratch/datasets/Foggy_Zurich/'
    oc_actual_path = '/home/anonymous6295/scratch/datasets/'
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    target_size = (512,1024)
    crop_size = (384,768)
    args = parser.parse_args()
    train_trans = get_train_trans(mean, std, target_size)
    test_trans = get_test_trans(mean, std, target_size)
    trainset = CityscapesExt(cs_path, split='train', target_type='semantic', transforms=train_trans)
    #trainset = CityscapesExt(cs_path, split='train', target_type='semantic')
    valset = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
    testset_day = CityscapesExt(cs_path, split='test', target_type='semantic', transforms=test_trans)
    testset_nd = NighttimeDrivingDataset(nd_path, transforms=test_trans)
    testset_dz = DarkZurichDataset(dz_path, transforms=test_trans)
    testset_fd = foggyDrivingDataset(fd_path, transforms=test_trans)
    testset_fz = foggyDrivingFullDataset(fz_path, transforms=test_trans)
    testset_fz_actual = foggyZurichDataset(fz_actual_path, transforms=test_trans)
    test_sets = torch.utils.data.ConcatDataset([testset_nd, testset_dz,testset_fd,testset_fz,testset_fz_actual])
    train_dev_loader = DataLoader(dataset=test_sets,batch_size=args.batch_size,shuffle=True)
    testset_dz_full = DarkZurichDataset(dz_path,split='test', transforms=test_trans)
    testset_oc = overcastDataset(oc_actual_path, transforms=test_trans)
    #print("done")
    #exit()
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_dz'] = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fdd'] = torch.utils.data.DataLoader(testset_fd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fd'] = torch.utils.data.DataLoader(testset_fz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_dz_full'] = torch.utils.data.DataLoader(testset_dz_full, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fz'] = torch.utils.data.DataLoader(testset_fz_actual, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    #dataloaders['testset_oc'] = torch.utils.data.DataLoader(testset_oc, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    #print(len(dataloaders))
    num_classes = len(CityscapesExt.validClasses)
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
            'cityscapes': 120,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.001,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    epoch=0
    #trainer.test_time_training(epoch,loader=dataloaders['test_fd'])    
    print("Validation Day time")
    trainer.validation(epoch)
    print('Night Driving mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
    print(miou)
    print('Dark Zurich mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
    print(miou)
    print('Foggy Driving Dense mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
    print(miou)
    print('Foggy Driving Full mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
    print(miou)
    print('Foggy Zurich mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
    print(miou)
    #trainer.validation_nd(100,dataloaders['test_nd'])
    print('Starting Epoch:', trainer.args.start_epoch)
    #print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        if args.test==0:
             miou=trainer.test_time_training_source(trainer,dataloaders,epoch)
        if args.test==1:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
        if args.test==2:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_dz'])
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
             #miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=train_dev_loader)
             #miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_dz_full'])
        if args.test==3:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_fdd'])
        if args.test==4:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_fd'])
        if args.test==5:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_fz'])
        if args.test==6:
            print("whole data")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=train_dev_loader)
        #trainer.test_time_training(trainer,dataloaders,epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
            #trainer.validation_nd(epoch,dataloaders['test_nd'])
            #trainer.validation_nd(epoch,dataloaders['test_dz'])
            #print('Foggy Driving Dense mIOU')
            #miou=trainer.test_time_training(epoch,loader=dataloaders['test_fd'])
            print("epoch")
            print('Night Driving mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
            print(miou)
            print('Dark Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
            print(miou)
            print('Foggy Driving Dense mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
            print(miou)
            print('Foggy Driving Full mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
            print(miou)
            print('Foggy Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
            print(miou)
    #plt.xlabel('Number of Iterations')
        # naming the y axis
    #plt.ylabel('mIOU Night Driving')
        # giving a title to my graph
    #plt.title('Night Driving')
    #plt.plot(iteration_list, miou_nd)
    # function to show the plot
    #plt.savefig('miou_nd'+str(epoch)+'.png')
    trainer.writer.close()








if __name__ == "__main__":
   main()
