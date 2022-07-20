from __future__ import absolute_import
from __future__ import division
import argparse
import os
import numpy as np
from torchvision import transforms
from os.path import join, isdir
from PIL import Image
import argparse
import logging
import os
import shutil, time, random

import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random

from tqdm import tqdm
from utils.helpers import gen_train_dirs, plot_confusion_matrix, get_train_trans, get_test_trans
from utils.routines import train_epoch, evaluate
from utils.routines import train_epoch, evaluate_test
from datasets.cityscapes_ext import CityscapesExt
from torch.utils.data import DataLoader, ConcatDataset
from dataloaders.utils import decode_segmap
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichDataset
from models.refinenet import RefineNet
from datasets.foggy_driving import foggyDrivingDataset
from datasets.foggy_driving_full import foggyDrivingFullDataset
from datasets.foggy_zurich import foggyZurichDataset
from datasets.overcast import overcastDataset
from datasets.acdc_fog import ACDCFogDataset
from datasets.acdc_night import ACDCNightDataset
from datasets.acdc_rain import ACDCRainDataset
from datasets.acdc_snow import ACDCSnowDataset
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
from torch.autograd import Variable
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

#@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    loss= -(x.cpu().softmax(1) * x.cpu().log_softmax(1)).sum(1).mean(0)
    #print(loss.size())
    loss=torch.sum(loss)
    return Variable(torch.tensor([loss]), requires_grad=True)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        #self.saver = Saver(args)
        #self.saver.save_experiment_config()
        # Define Tensorboard Summary
        #self.summary = TensorboardSummary(self.saver.experiment_dir)
        #self.writer = self.summary.create_summary()
        
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
        '''
        if args.base_model=='deeplabv3+_mobilenet':
            model = network.deeplabv3plus_mobilenet(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3+_resnet101':
            model = network.deeplabv3plus_resnet101(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3plus_resnet50':
            model = network.deeplabv3plus_resnet50(num_classes=self.nclass, output_stride=args.out_stride)
        if args.base_model=='deeplabv3_resnet50':
            model = network.deeplabv3_resnet50(num_classes=self.nclass, output_stride=args.out_stride)
        #print(model)
        num_classes = len(CityscapesExt.validClasses)
        '''
        random_seed = cfg.RANDOM_SEED  #304
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

        args.world_size = 1

        # Test Mode run two epochs with a few iterations of training and val
        if args.test_mode:
            args.max_epoch = 2

        if 'WORLD_SIZE' in os.environ:
            # args.apex = int(os.environ['WORLD_SIZE']) > 1
            args.world_size = int(os.environ['WORLD_SIZE'])
            print("Total world size: ", int(os.environ['WORLD_SIZE']))

        torch.cuda.set_device(args.local_rank)
        print('My Rank:', args.local_rank)
        # Initialize distributed communication
        args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

        torch.distributed.init_process_group(backend='nccl',
                                            init_method=args.dist_url,
                                            world_size=args.world_size,
                                            rank=args.local_rank)
        num_classes = len(CityscapesExt.validClasses)
        criterion, criterion_val = loss.get_loss(args)
        criterion_aux = loss.get_loss_aux(args)
        net = network.get_net(args, criterion, criterion_aux)
        optim, scheduler = optimizer.get_optimizer(args, net)

        #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        #net = network.warp_network_in_dataparallel(net, args.local_rank)
        if torch.cuda.is_available():
            net = torch.nn.DataParallel(net).cuda()
            print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        #if torch.cuda.is_available():
        #    model = torch.nn.DataParallel(model).cuda()
        #    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        #print(model)
        if args.snapshot:
            epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
         # Load weights from checkpoint
        #checkpoint = torch.load('/home/anonymous6295/scratch/CIConv_zero_shot/experiments/rloss/pytorch/pytorch-deeplab_v3_plus/ciconv-master/experiments/3_segmentation/cityscapes_w.pth.tar')
        #model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model = net

        #checkpoint = torch.load('/home/anonymous6295/scratch/CIConv_zero_shot/experiments/3_segmentation/runs/20211014-212540/weights/checkpoint.pth')

        #if torch.cuda.is_available():
        #    model = torch.nn.DataParallel(model).cuda()
        #    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        #model.load_state_dict(checkpoint['model_state'], strict=True)
        #train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        #optimizer1 = torch.optim.SGD(train_params, momentum=args.momentum,
        #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
        #print(model.backbone)
        optimizer1 = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        #optimizer = torch.optim.SGD(params=[
        #{'params': model.parameters(), 'lr': args.lr},
        #], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
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
        
        #self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion = criterion

        self.model, self.optimizer = model, optimizer1
        
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
        '''
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
        '''

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
                probs = Variable(probs, requires_grad=True)
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
                anonymous6295=1
                #self.nclass
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    for i in range(anonymous6295):
                        fig=plt.figure()
                        plt.imshow(grad_seg[0,i,:,:], cmap="hot") #vmin=0, vmax=1)
                        plt.colorbar()
                        plt.axis('off')
                        #if args.output_directory is not None:
                        plt.savefig(
                            join('./images/',str(filename)+'_'+str(iteration)+'grad_seg_class_' + str(i) +'.png')
                        )
                        
                            #plt.show(block=False)
                        plt.close(fig)
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

    def test_time_training_test(self,trainer,dataloaders, epoch,loader):
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
        criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
        softmax = nn.Softmax(dim=1)
        #for i, sample in enumerate(tbar):
        #for i, (inputs, labels) in enumerate(loader):
        for i, (inputs,filepath) in enumerate(loader):
            #image, target = sample['image'], sample['label']
            image = inputs
            #target = labels
            image = image.float()
            #target = target.long()
            #print(image)
            iteration = iteration+1
            #print(labels)
            iteration_list.append(iteration)
            #croppings = (target!=254).float()
            #target[target==254]=255
            # Pixels labeled 255 are those unlabeled pixels. Padded region are labeled 254.
            # see function RandomScaleCrop in dataloaders/custom_transforms.py for the detail in data preprocessing
            if self.args.cuda:
                image = image.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            #pred = output.cpu().numpy()
            #pred = np.argmax(output, axis=1)
            #softmax_entropy
            #corrects = torch.sum(output ==target)
            #print(corrects)
            #print(output.size())
            #print(target.size())
            #celoss = criterion(output, target)
            celoss = softmax_entropy(output).cuda()
            if self.args.densencloss ==0:
                loss = celoss
            else:
                print("anonymous6295")
                probs = softmax(output)
                probs = Variable(probs, requires_grad=True)
                new=torch.argmax(image)
                #print(image.shape)
                #print(output.shape)
                croppings = torch.ones([inputs.size(0),1080,1920]).float()
                print(croppings.shape)
                denormalized_image = denormalizeimage(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
                if self.args.cuda:
                    densencloss = densencloss.cuda()
                #loss = 0.000001*celoss.cuda() + densencloss
                loss = 10000*densencloss
                print(loss)
                train_ncloss += densencloss.item()
                loss.backward()
                grad_seg = probs.grad.cpu().numpy()
                self.optimizer.step()
                anonymous6295=1
                #self.nclass
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    for i in range(anonymous6295):
                        fig=plt.figure()
                        plt.imshow(grad_seg[0,i,:,:], cmap="hot") #vmin=0, vmax=1)
                        plt.colorbar()
                        plt.axis('off')
                        #if args.output_directory is not None:
                        plt.savefig(
                            join('./images/',str(filename)+'_'+str(iteration)+'grad_seg_class_' + str(i) +'.png')
                        )
                        
                            #plt.show(block=False)
                        plt.close(fig)
                #self.optimizer.zero_grad()
                #segmap = decode_segmap(new,'cityscapes')*255
                #segmap = segmap.astype(np.uint8)
                #segimg = Image.fromarray(image, 'RGB')
                #fig=plt.figure()
                #plt.imshow(segimg) #vmin=0, vmax=1)
                #plt.savefig(
                #        join('./images/',str(iteration)+'image' + str(i) +'.png')
                #    )
                #plt.close(fig)
                anonymous6295=1
                #self.nclass
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    for i in range(anonymous6295):
                        fig=plt.figure()
                        plt.imshow(grad_seg[0,i,:,:], cmap="hot") #vmin=0, vmax=1)
                        plt.colorbar()
                        plt.axis('off')
                        #if args.output_directory is not None:
                        plt.savefig(
                            join('./images/',str(filename)+'_'+str(iteration)+'grad_seg_class_' + str(i) +'.png')
                        )
                        
                            #plt.show(block=False)
                        plt.close(fig)
                #if args.output_directory is None:
                        #plt.show(block=True)
                #exit()
            train_loss += loss.item()
            #train_celoss += celoss.item()
            #output = self.model(image)
            #corrects = torch.sum(output[0] ==target)
            #print(corrects)
            print('\n')
            '''
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
            '''
            print("ACDC val Night dataset")
            miou=trainer.validation_nd(epoch,dataloaders['valset_acdc_night'])
            print(miou)
            print("ACDC Night dataset")
            trainer.validation_nd_test(epoch,dataloaders['testset_acdc_night'])
            #print(miou)
            '''
            print("ACDC Rain dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_rain'])
            print(miou)
            print("ACDC Night dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_night'])
            print(miou)
            print("ACDC Snow dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_snow'])
            print(miou)
            '''
        
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
        criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
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
            #self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            #softmax_entropy
            #corrects = torch.sum(output ==target)
            #print(corrects)
            #print(output.size())
            #print(target.size())
            #celoss = criterion(output, target)
            #celoss = softmax_entropy(output).cuda()
            if self.args.densencloss ==0:
                loss = celoss
            else:
                start_time = time.perf_counter ()
                print("nikil")
                probs = softmax(output)
                #print(croppings.shape)
                #exit()
                probs = Variable(probs, requires_grad=True)
                denormalized_image = denormalizeimage(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                densencloss = self.densenclosslayer(denormalized_image,probs,croppings)
                if self.args.cuda:
                    densencloss = densencloss.cuda()
                #loss = 0.000001*celoss.cuda() + densencloss
                loss = 0.01*densencloss
                train_ncloss += densencloss.item()
                loss.backward()
                grad_seg = probs.grad.cpu().numpy()
                self.optimizer.step()
                self.optimizer.zero_grad()
                end_time = time.perf_counter ()
                print(end_time - start_time, "seconds")
                exit()
                train_loss += loss.item()
                anonymous6295=0
                #self.nclass
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    for k in range(anonymous6295):
                        fig=plt.figure()
                        plt.imshow(grad_seg[i,k,:,:], cmap="hot") #vmin=0, vmax=1)
                        plt.colorbar()
                        plt.axis('off')
                        #if args.output_directory is not None:
                        plt.savefig(
                            join('./images/',str(filename)+'_'+str(epoch)+'grad_seg_class_' + str(k) +'.png')
                        )
                        
                            #plt.show(block=False)
                        plt.close(fig)
                #if args.output_directory is None:
            #train_celoss += celoss.item()
            #output = self.model(image)
            #corrects = torch.sum(output[0] ==target)
            #print(corrects)
            print('\n')
            #trainer.validation(epoch)
            #print("iteration")
            #print('Night Driving mIOU')
            #miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
            #miou_nd.append(miou*100)
            #print(miou)
            print('Dark Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
            miou_nd.append(miou*100)
            #print(miou)
            #print(miou)
            #print('Dark Zurich mIOU')
            #miou=trainer.validation_nd_test(epoch,dataloaders['test_dz_full'])
            #print(miou)
            '''
            '''
            '''
            print('Foggy Driving Dense mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
            print(miou)
            print('Foggy Driving Full mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
            print(miou)
            print('Foggy Zurich mIOU')
            miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
            print(miou)
            '''
            '''
            print("ACDC Fog dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_fog'])
            print(miou)
    
            print("ACDC Rain dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_rain'])
            print(miou)
            print("ACDC Night dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_night'])
            print(miou)
            print("ACDC Snow dataset")
            miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_snow'])
            print(miou)
            '''
            tbar.set_description('Train loss: %.3f = CE loss %.3f + NC loss: %.3f' 
                             % (train_loss / (i + 1),train_celoss / (i + 1),train_ncloss / (i + 1)))
            #self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            #if i % (num_img_tr // 10) == 0:
            #    global_step = i + num_img_tr * epoch
            #    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        #self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        #if self.args.no_val:
        '''
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
            '''

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
                self.optimizer.zero_grad()
            train_loss += loss.item()
            #train_celoss += celoss.item()
            print('\n')
            '''
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
            '''
            trainer.validation_nd_test(epoch,dataloaders['testset_acdc_fog'])
            
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

    def validation_nd_test(self,epoch,data):
        mean = (0.485, 0.456, 0.406) 
        std = (0.229, 0.224, 0.225) 
        target_size = (512,1024)
        crop_size = (384,768)
        #print(CityscapesExt.voidClass)
        evaluate_test(data,
            self.model, self.criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses,void=CityscapesExt.voidClass,optimizer=self.optimizer,
             maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        return miou_nd

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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--arch', type=str, default='network.deepv3.DeepV3Plus',
                        help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                        and deepWV3Plus (backbone: WideResNet38).')
    parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                        help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
    parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                        help='uniformly sample images across the multiple source domains')
    parser.add_argument('--val_dataset', nargs='*', type=str, default=['bdd100k'],
                        help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=['cityscapes'],
                        help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    parser.add_argument('--cv', type=int, default=0,
                        help='cross-validation split id to use. Default # of splits set to 3 in config')
    parser.add_argument('--class_uniform_pct', type=float, default=0,
                        help='What fraction of images is uniformly sampled')
    parser.add_argument('--class_uniform_tile', type=int, default=1024,
                        help='tile size for class uniform sampling')
    parser.add_argument('--coarse_boost_classes', type=str, default=None,
                        help='use coarse annotations to boost fine data with specific classes')

    parser.add_argument('--img_wt_loss', action='store_true', default=False,
                        help='per-image class-weighted loss')
    parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                        help='class-weighted loss')
    parser.add_argument('--batch_weighting', action='store_true', default=False,
                        help='Batch weighting for class (use nll class weighting using batch stats')

    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')
    parser.add_argument('--strict_bdr_cls', type=str, default='',
                        help='Enable boundary label relaxation for specific classes')
    parser.add_argument('--rlx_off_iter', type=int, default=-1,
                        help='Turn off border relaxation after specific epoch count')
    parser.add_argument('--rescale', type=float, default=1.0,
                        help='Warm Restarts new learning rate ratio compared to original lr')
    parser.add_argument('--repoly', type=float, default=1.5,
                        help='Warm Restart new poly exp')

    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use Nvidia Apex AMP')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='parameter used by apex library')

    parser.add_argument('--sgd', action='store_true', default=True)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--amsgrad', action='store_true', default=False)

    parser.add_argument('--freeze_trunk', action='store_true', default=False)
    parser.add_argument('--hardnm', default=0, type=int,
                        help='0 means no aug, 1 means hard negative mining iter 1,' +
                        '2 means hard negative mining iter 2')

    parser.add_argument('--trunk', type=str, default='resnet101',
                        help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('--max_epoch', type=int, default=180)
    parser.add_argument('--max_iter', type=int, default=30000)
    parser.add_argument('--max_cu_epoch', type=int, default=100000,
                        help='Class Uniform Max Epochs')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--rrotate', type=int,
                        default=0, help='degree of random roate')
    parser.add_argument('--color_aug', type=float,
                        default=0.0, help='level of color augmentation')
    parser.add_argument('--gblur', action='store_true', default=False,
                        help='Use Guassian Blur Augmentation')
    parser.add_argument('--bblur', action='store_true', default=False,
                        help='Use Bilateral Blur Augmentation')
    parser.add_argument('--lr_schedule', type=str, default='poly',
                        help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='polynomial LR exponent')
    parser.add_argument('--bs_mult', type=int, default=2,
                        help='Batch size for training per gpu')
    parser.add_argument('--bs_mult_val', type=int, default=1,
                        help='Batch size for Validation per gpu')
    parser.add_argument('--crop_size', type=int, default=720,
                        help='training crop size')
    parser.add_argument('--pre_size', type=int, default=None,
                        help='resize image shorter edge to this before augmentation')
    parser.add_argument('--scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--restore_optimizer', action='store_true', default=False)

    parser.add_argument('--city_mode', type=str, default='train',
                        help='experiment directory date name')
    parser.add_argument('--date', type=str, default='default',
                        help='experiment directory date name')
    parser.add_argument('--exp', type=str, default='default',
                        help='experiment directory name')
    parser.add_argument('--tb_tag', type=str, default='',
                        help='add tag to tb dir')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='logs/tb',
                        help='Save Tensorboard Path')
    parser.add_argument('--syncbn', action='store_true', default=True,
                        help='Use Synchronized BN')
    parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                        help='Dump Augmentated Images for sanity check')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Minimum testing to verify nothing failed, ' +
                        'Runs code for 1 epoch of train and val')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                        help='Weight Scaling for the losses')
    parser.add_argument('--maxSkip', type=int, default=0,
                        help='Skip x number of  frames of video augmented dataset')
    parser.add_argument('--scf', action='store_true', default=False,
                        help='scale correction factor')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                        help='url used to set up distributed training')

    parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,0,0,0,0,0],
                        help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
    parser.add_argument('--wt_reg_weight', type=float, default=0.0)
    parser.add_argument('--relax_denom', type=float, default=2.0)
    parser.add_argument('--clusters', type=int, default=50)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--dynamic', action='store_true', default=False)

    parser.add_argument('--image_in', action='store_true', default=False,
                        help='Input Image Instance Norm')
    parser.add_argument('--cov_stat_epoch', type=int, default=5,
                        help='cov_stat_epoch')
    parser.add_argument('--visualize_feature', action='store_true', default=False,
                        help='Visualize intermediate feature')
    parser.add_argument('--use_wtloss', action='store_true', default=False,
                        help='Automatic setting from wt_layer')
    parser.add_argument('--use_isw', action='store_true', default=False,
                        help='Automatic setting from wt_layer')    
    
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset1', type=str, default='cityscapes',
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
    parser.add_argument('--start_epoch1', type=int, default=110,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr1', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum1', type=float, default=0.9,
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
    acdc_path='/home/anonymous6295/scratch/datasets/acdc_dataset/'
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
    testset_acdc_fog = ACDCFogDataset(acdc_path,split='test', transforms=test_trans)
    testset_acdc_rain = ACDCRainDataset(acdc_path,split='test', transforms=test_trans)
    testset_acdc_snow = ACDCSnowDataset(acdc_path,split='test', transforms=test_trans)
    testset_acdc_night = ACDCNightDataset(acdc_path,split='test', transforms=test_trans)
    valset_acdc_fog = ACDCFogDataset(acdc_path,split='val', transforms=test_trans)
    valset_acdc_rain = ACDCRainDataset(acdc_path,split='val', transforms=test_trans)
    valset_acdc_snow = ACDCSnowDataset(acdc_path,split='val', transforms=test_trans)
    valset_acdc_night = ACDCNightDataset(acdc_path,split='val', transforms=test_trans)
    val_sets_acdc = torch.utils.data.ConcatDataset([testset_acdc_night, testset_acdc_snow,testset_acdc_rain,testset_acdc_fog])
    train_dev_loader_acdc = DataLoader(dataset=test_sets,batch_size=args.batch_size,shuffle=True)


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
    dataloaders['testset_acdc_fog'] = torch.utils.data.DataLoader(testset_acdc_fog, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['testset_acdc_rain'] = torch.utils.data.DataLoader(testset_acdc_rain, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['testset_acdc_snow'] = torch.utils.data.DataLoader(testset_acdc_snow, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['testset_acdc_night'] = torch.utils.data.DataLoader(testset_acdc_night, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_acdc_fog'] = torch.utils.data.DataLoader(valset_acdc_fog, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_acdc_rain'] = torch.utils.data.DataLoader(valset_acdc_rain, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_acdc_snow'] = torch.utils.data.DataLoader(valset_acdc_snow, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['valset_acdc_night'] = torch.utils.data.DataLoader(valset_acdc_night, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

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
            'cityscapes': 130,
            'pascal': 50,
        }
        args.epochs = 30

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
    #print("Validation Day time")
    #trainer.validation(epoch)
    
    print('Night Driving mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_nd'])
    print(miou)

    print('Dark Zurich mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_dz'])
    print(miou)
    '''
    print('Foggy Driving Dense mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_fdd'])
    print(miou)
    print('Foggy Driving Full mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_fd'])
    print(miou)
    print('Foggy Zurich mIOU')
    miou=trainer.validation_nd(epoch,dataloaders['test_fz'])
    print(miou)
    '''
    #print("ACDC val Night dataset")
    #miou=trainer.validation_nd(epoch,dataloaders['valset_acdc_night'])
    #print(miou)
    #print("ACDC Night dataset")
    #trainer.validation_nd_test(epoch,dataloaders['testset_acdc_night'])
    #print(miou)
    #exit()
    '''
    print("ACDC Rain dataset")
    miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_rain'])
    print(miou)
    print("ACDC Night dataset")
    miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_night'])
    print(miou)
    print("ACDC Snow dataset")
    miou=trainer.validation_nd(epoch,dataloaders['testset_acdc_snow'])
    print(miou)
    '''
    #exit()
    #trainer.validation_nd(100,dataloaders['test_nd'])
    print('Starting Epoch:', trainer.args.start_epoch)
    #print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        if args.test==0:
             miou=trainer.test_time_training_source(trainer,dataloaders,epoch)
        if args.test==1:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_dz'])
        if args.test==2:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_dz'])
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
             #miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
             #miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['test_nd'])
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
        if args.test==7:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['valset_acdc_fog'])
             miou=trainer.test_time_training_test(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_fog'])
        if args.test==8:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_rain'])
        if args.test==9:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_snow'])
        if args.test==10:
             miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=dataloaders['valset_acdc_night'])
             miou=trainer.test_time_training_test(trainer,dataloaders,epoch,loader=dataloaders['testset_acdc_night'])
        #trainer.test_time_training(trainer,dataloaders,epoch)
        if args.test==11:
            print("whole data acdc")
            miou=trainer.test_time_training(trainer,dataloaders,epoch,loader=train_dev_loader_acdc)
        #if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        #    trainer.validation(epoch)
            #trainer.validation_nd(epoch,dataloaders['test_nd'])
            #trainer.validation_nd(epoch,dataloaders['test_dz'])
            #print('Foggy Driving Dense mIOU')
            #miou=trainer.test_time_training(epoch,loader=dataloaders['test_fd'])
            '''
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
            '''
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
