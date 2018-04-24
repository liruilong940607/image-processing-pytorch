import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable

import numpy as np
import argparse
import time
import cv2
import os
import h5py
import json
import random

from datalayer import DatasetCocoKpt
from model_origin import get_model
from utils import to_varabile, AverageMeter, adjust_learning_rate, save_checkpoint

def train(dataLoader, netmodel, optimizer, epoch, iteration):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    lossesList_paf = [AverageMeter('PAF_S%d'%i) for i in range(1,7)]
    lossesList_heatmap = [AverageMeter('HM_S%d'%i) for i in range(1,7)]
        
    # switch to train mode
    netmodel.train()

    lossfunc = nn.MSELoss(size_average=False).cuda()
    
    end = time.time()
    for i, data in enumerate(dataLoader):  
        data_time.update(time.time() - end)
        input, heatmap_gt, paf_gt, ignoremask = data
        bz, c, h, w = input.size()
        
        input_var = to_varabile(input, requires_grad=True, is_cuda=True)
        heatmap_gt_var = to_varabile(heatmap_gt, requires_grad=False, is_cuda=True)
        paf_gt_var = to_varabile(paf_gt, requires_grad=False, is_cuda=True)
        ignoremask_var = to_varabile(ignoremask, requires_grad=False, is_cuda=True)
        
        outs_stages = netmodel(input_var)
        loss = 0
        for stage, outs in enumerate(outs_stages):
            out_paf, out_heatmap = outs
            loss_paf = lossfunc(out_paf*ignoremask_var, paf_gt_var)
            loss_heatmap = lossfunc(out_heatmap*ignoremask_var, heatmap_gt_var)
            loss += loss_paf + loss_heatmap
            lossesList_paf[stage].update(loss_paf.data[0], bz)
            lossesList_heatmap[stage].update(loss_heatmap.data[0], bz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.printfreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr: [{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                   epoch, i, len(dataLoader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                   data_time=data_time))
            print('          \t'
                  '{loss_stage1.name:<8s}:{loss_stage1.val:>8.2f}({loss_stage1.avg:>8.2f})\t'
                  '{loss_stage2.name:<8s}:{loss_stage2.val:>8.2f}({loss_stage2.avg:>8.2f})\t'
                  '{loss_stage3.name:<8s}:{loss_stage3.val:>8.2f}({loss_stage3.avg:>8.2f})\t'
                  '{loss_stage4.name:<8s}:{loss_stage4.val:>8.2f}({loss_stage4.avg:>8.2f})\t'
                  '{loss_stage5.name:<8s}:{loss_stage5.val:>8.2f}({loss_stage5.avg:>8.2f})\t'
                  '{loss_stage6.name:<8s}:{loss_stage6.val:>8.2f}({loss_stage6.avg:>8.2f})'.format(
                   loss_stage1=lossesList_paf[0], loss_stage2=lossesList_paf[1], loss_stage3=lossesList_paf[2],
                   loss_stage4=lossesList_paf[3], loss_stage5=lossesList_paf[4], loss_stage6=lossesList_paf[5]))
            print('          \t'
                  '{loss_stage1.name:<8s}:{loss_stage1.val:>8.2f}({loss_stage1.avg:>8.2f})\t'
                  '{loss_stage2.name:<8s}:{loss_stage2.val:>8.2f}({loss_stage2.avg:>8.2f})\t'
                  '{loss_stage3.name:<8s}:{loss_stage3.val:>8.2f}({loss_stage3.avg:>8.2f})\t'
                  '{loss_stage4.name:<8s}:{loss_stage4.val:>8.2f}({loss_stage4.avg:>8.2f})\t'
                  '{loss_stage5.name:<8s}:{loss_stage5.val:>8.2f}({loss_stage5.avg:>8.2f})\t'
                  '{loss_stage6.name:<8s}:{loss_stage6.val:>8.2f}({loss_stage6.avg:>8.2f})'.format(
                   loss_stage1=lossesList_heatmap[0], loss_stage2=lossesList_heatmap[1], loss_stage3=lossesList_heatmap[2],
                   loss_stage4=lossesList_heatmap[3], loss_stage5=lossesList_heatmap[4], loss_stage6=lossesList_heatmap[5]))
                    
        if i % args.savefreq == 0:  
            torch.save(netmodel.state_dict(), 'snapshot/epoch%d_%d.pkl'%(epoch,i))
        
    return iteration+1+i

def main(args):    
    datasetTrain = DatasetCocoKpt(ImageRoot='/home/dalong/data/coco2017/train2017', 
                                 AnnoFile='/home/dalong/data/coco2017/annotations/person_keypoints_train2017.json', 
                                 istrain=True)
    # datasetVal = DatasetCocoKpt(ImageRoot='/home/dalong/data/coco2017/val2017', 
    #                              AnnoFile='/home/dalong/data/coco2017/annotations/person_keypoints_val2017.json', 
    #                              istrain=False)
    
    dataLoader_train = torch.utils.data.DataLoader(datasetTrain, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)
    # dataLoader_val = torch.utils.data.DataLoader(datasetVal, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    
    print ('===========> loading model <===========')
    model = get_model().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weightdecay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weightdecay) 
    
    iteration = 0
    epoches = 20
    for epoch in range(epoches):
        print ('===========>   training    <===========')
        learning_rate = adjust_learning_rate(optimizer, iteration, args.lr, policy='step', policy_parameter={'gamma': 0.333, 'step_size': 13275}, multiple=[1., 2., 4., 8.])
        iteration = train(dataLoader_train, model, optimizer, epoch, iteration)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--workers', default=4, type=int, 
                        help='number of data loading workers')
    parser.add_argument('--batchsize', default=8, type=int, 
                        help='mini-batch size') # 50: 12/gpu
    parser.add_argument('--lr', default=1e-5, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weightdecay', default=5e-4, type=float, 
                        help='weight decay')
    parser.add_argument('--printfreq', default=10, type=int, 
                        help='print frequency')
    parser.add_argument('--savefreq', default=1000, type=int, 
                        help='save frequency')
    args = parser.parse_args()
    
    main(args)
