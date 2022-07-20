# -*- coding: utf-8 -*-
import torch
import time
import os
import cv2
import torch.nn as nn

from DenseNCLoss import DenseNCLoss
from utils.helpers import AverageMeter, ProgressMeter, visim, vislbl,vislbl_test
from utils.get_iou import iouCalc

def train_epoch(dataloader, model, criterion, optimizer, epoch, void=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Epoch: [{}]".format(epoch))
    
    # set model in training mode
    model.train()
    
    end = time.time()
    
    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time()-end)

            # input resolution
            res = inputs.shape[2]*inputs.shape[3]
            croppings = (labels!=254).float()
            inputs = inputs.float().cuda()
            inputs = inputs.float()
            labels = labels.long().cuda()
            labels = labels.long()
            #labels[labels == 255 ] = 19
            print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            outputs = model(inputs)
            #probs = softmax(outputs)
            preds = torch.argmax(outputs, 1)
            
            celoss = criterion(outputs, labels)
            denseNClosslayer = DensencLoss(weight=1e-3, sigma_rgb=1, sigma_xy=2, scale_factor=1).cuda()
            #outputs1 =outputs.cpu().numpy()
            #inputs = inputs.detach().to("cpu").numpy()
            #outputs = outputs.detach().to("cpu").numpy()
            nc_loss = denseNClosslayer(inputs,outputs,croppings)
            loss = celoss
            #loss = nc_loss.cuda()
            print(nc_loss)
            
            # backward
            loss.backward()
            optimizer.step()
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            print(preds)
            print(labels.data)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            
            # Output training info
            progress.display(epoch_step)
            # Append current stats to csv
            with open('logs/log_batch.csv', 'a') as log_batch:
                log_batch.write('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(epoch,
                                epoch_step, loss/bs, loss_running.avg,
                                acc, acc_running.avg))
                            
            batch_time.update(time.time() - end)
            end = time.time()
            
    print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,loss_running.avg,acc_running.avg))
    
    return loss_running.avg, acc_running.avg

def evaluate(dataloader, model, criterion, epoch, classLabels, validClasses, void,optimizer, maskColors=None, mean=None, std=None):
    iou = iouCalc(classLabels, validClasses, voidClass = void)
    #print(iou)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, loss_running, acc_running],
        prefix='Test: ')
    # set model in training mode
    model.eval()
    #print(len(dataloader))
    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time()-end)
            # input resolution
            croppings = (labels!=254).float()
            res = inputs.shape[2]*inputs.shape[3]
            #optimizer.zero_grad()
            inputs = inputs.float()
            labels = labels.long()
            #labels = (labels!=255).long()
            #labels = labels-1
            #print(labels.size())
            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            labels = labels.cuda()
            #labels=labels>-1
    
            criterion = nn.CrossEntropyLoss(ignore_index=void)
            #print(outputs.size())
            celoss = criterion(outputs, labels)
            # Statistics
            #denseNClosslayer = DenseNCLoss(weight=1e-3, sigma_rgb=1, sigma_xy=2, scale_factor=1).cuda()
            #print(preds)
            #print(labels.data)
            #outputs1 =outputs.cpu().numpy()
            #inputs = inputs.detach().to("cpu").numpy()
            #outputs = outputs.detach().to("cpu").numpy()
            #nc_loss = denseNClosslayer(inputs,outputs,croppings)
            loss = celoss.cuda()
            #print(nc_loss)
            # backward propogation
            #loss.backward()
            #optimizer.step()
            bs = inputs.size(0) # current batch size
            #print(loss)
            loss = loss.item()
            loss_running.update(loss, bs)
            
            corrects = torch.sum(preds == labels.data)
            #print(corrects)
            bs = inputs.size(0)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)
            
            # Save visualizations of first batch
            #if epoch_step == 0 and maskColors is not None:
            if maskColors is not None:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    #print(filepath)
                    # Only save inputs and labels once
                    if True:
                        img = visim(inputs[i,:,:,:], mean, std)
                        label = vislbl(labels[i,:,:], maskColors)
                        if len(img.shape) == 3:
                            cv2.imwrite('images/{}.png'.format(filename),img[:,:,::-1])
                        else:
                            cv2.imwrite('images/{}.png'.format(filename),img)
                        cv2.imwrite('images/{}_gt.png'.format(filename),label[:,:,::-1])
                    # Save predictions
                    pred = vislbl(preds[i,:,:], maskColors)
                    cv2.imwrite('images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            #progress.display(epoch_step)
        
        miou, iou_summary, confMatrix = iou.outputScores(epoch=epoch)
        #print(' * Acc {:.3f}'.format(acc_running.avg))
        #print(iou_summary)

    return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary

def evaluate_test(dataloader, model, criterion, epoch, classLabels, validClasses, void,optimizer, maskColors=None, mean=None, std=None):
    #iou = iouCalc(classLabels, validClasses, voidClass = void)
    #print(iou)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, loss_running, acc_running],
        prefix='Test: ')
    # set model in training mode
    model.eval()
    #print(len(dataloader))
    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, filepath) in enumerate(dataloader):
            data_time.update(time.time()-end)
            # input resolution
            #croppings = (labels!=254).float()
            res = inputs.shape[2]*inputs.shape[3]
            #optimizer.zero_grad()
            inputs = inputs.float()
            #labels = labels.long()
            #labels = (labels!=255).long()
            #labels = labels-1
            #print(labels.size())
            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            #labels = labels.cuda()
            #labels=labels>-1
    
            #criterion = nn.CrossEntropyLoss(ignore_index=void)
            #print(outputs.size())
            #celoss = criterion(outputs, labels)
            # Statistics
            #denseNClosslayer = DenseNCLoss(weight=1e-3, sigma_rgb=1, sigma_xy=2, scale_factor=1).cuda()
            #print(preds)
            #print(labels.data)
            #outputs1 =outputs.cpu().numpy()
            #inputs = inputs.detach().to("cpu").numpy()
            #outputs = outputs.detach().to("cpu").numpy()
            #nc_loss = denseNClosslayer(inputs,outputs,croppings)
            #loss = celoss.cuda()
            #print(nc_loss)
            # backward propogation
            #loss.backward()
            #optimizer.step()
            bs = inputs.size(0) # current batch size
            #print(loss)
            #loss = loss.item()
            #loss_running.update(loss, bs)
            
            #corrects = torch.sum(preds == labels.data)
            #print(corrects)
            bs = inputs.size(0)
            #nvoid = int((labels==void).sum())
            #acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            #acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            #iou.evaluateBatch(preds, labels)
            
            # Save visualizations of first batch
            #if epoch_step == 0 and maskColors is not None:
            if maskColors is not None:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    #print(filepath)
                    #filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    #print(filepath)
                    filename = filename.split("_rgb_anon")
                    # Only save inputs and labels once
                    '''
                    if True:
                        img = visim(inputs[i,:,:,:], mean, std)
                        #label = vislbl(labels[i,:,:], maskColors)
                        if len(img.shape) == 3:
                            cv2.imwrite('images/{}.png'.format(filename),img[:,:,::-1])
                        else:
                            cv2.imwrite('images/{}.png'.format(filename),img)
                        #cv2.imwrite('images/{}_gt.png'.format(filename),label[:,:,::-1])
                    # Save predictions
                    '''
                    pred = vislbl_test(preds[i,:,:], maskColors)
                    #cv2.imwrite('images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])
                    #cv2.imwrite('images/{}.png'.format(filename[0]),pred)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            #progress.display(epoch_step)
    