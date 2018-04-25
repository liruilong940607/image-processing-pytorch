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
from model_origin import get_model, set_lr_groups
from utils import to_varabile, AverageMeter, adjust_learning_rate, save_checkpoint
from logger import Logger

def train(dataLoader, netmodel, optimizer, epoch, iteration, logger, args):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    lossesList_paf = [AverageMeter('PAF_S%d'%i) for i in range(1,7)]
    lossesList_heatmap = [AverageMeter('HM_S%d'%i) for i in range(1,7)]
        
    # switch to train mode
    netmodel.train()
    lossfunc = nn.MSELoss(size_average=False).cuda()
    
    end = time.time()
    for i, data in enumerate(dataLoader):  
        iteration += 1
        learning_rate = adjust_learning_rate(optimizer, iteration, args.lr, 
                                             policy=args.lr_policy, 
                                             policy_parameter={'gamma': args.gamma, 'step_size': args.stepsize}, 
                                             multiple=[1., 2., 4., 8.])
        
        data_time.update(time.time() - end)
        input, heatmap_gt, paf_gt, ignoremask = data
        bz, c, h, w = input.size()
        
        input_var = to_varabile(input, requires_grad=True, is_cuda=True)
        heatmap_gt_var = to_varabile(heatmap_gt, requires_grad=False, is_cuda=True)
        paf_gt_var = to_varabile(paf_gt, requires_grad=False, is_cuda=True)
        ignoremask_var = to_varabile(ignoremask, requires_grad=False, is_cuda=True)
        
        outs_stages = netmodel(input_var)
        loss = 0
        heat_weight = 1.0 / 2.0 / bz # for convenient to compare with origin code
        vec_weight = 1.0 / 2.0 / bz
        for stage, outs in enumerate(outs_stages):
            out_paf, out_heatmap = outs
            loss_paf = lossfunc(out_paf*ignoremask_var, paf_gt_var) * vec_weight
            loss_heatmap = lossfunc(out_heatmap*ignoremask_var, heatmap_gt_var) * heat_weight
            loss += (loss_paf + loss_heatmap)
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
                  '{loss_stage1.name:<8s}:{loss_stage1.val:>8.4f}({loss_stage1.avg:>8.4f})\t'
                  '{loss_stage2.name:<8s}:{loss_stage2.val:>8.4f}({loss_stage2.avg:>8.4f})\t'
                  '{loss_stage3.name:<8s}:{loss_stage3.val:>8.4f}({loss_stage3.avg:>8.4f})\t'
                  '{loss_stage4.name:<8s}:{loss_stage4.val:>8.4f}({loss_stage4.avg:>8.4f})\t'
                  '{loss_stage5.name:<8s}:{loss_stage5.val:>8.4f}({loss_stage5.avg:>8.4f})\t'
                  '{loss_stage6.name:<8s}:{loss_stage6.val:>8.4f}({loss_stage6.avg:>8.4f})'.format(
                   loss_stage1=lossesList_paf[0], loss_stage2=lossesList_paf[1], loss_stage3=lossesList_paf[2],
                   loss_stage4=lossesList_paf[3], loss_stage5=lossesList_paf[4], loss_stage6=lossesList_paf[5]))
            print('          \t'
                  '{loss_stage1.name:<8s}:{loss_stage1.val:>8.4f}({loss_stage1.avg:>8.4f})\t'
                  '{loss_stage2.name:<8s}:{loss_stage2.val:>8.4f}({loss_stage2.avg:>8.4f})\t'
                  '{loss_stage3.name:<8s}:{loss_stage3.val:>8.4f}({loss_stage3.avg:>8.4f})\t'
                  '{loss_stage4.name:<8s}:{loss_stage4.val:>8.4f}({loss_stage4.avg:>8.4f})\t'
                  '{loss_stage5.name:<8s}:{loss_stage5.val:>8.4f}({loss_stage5.avg:>8.4f})\t'
                  '{loss_stage6.name:<8s}:{loss_stage6.val:>8.4f}({loss_stage6.avg:>8.4f})'.format(
                   loss_stage1=lossesList_heatmap[0], loss_stage2=lossesList_heatmap[1], loss_stage3=lossesList_heatmap[2],
                   loss_stage4=lossesList_heatmap[3], loss_stage5=lossesList_heatmap[4], loss_stage6=lossesList_heatmap[5]))
            
        if i % (args.printfreq*10) == 0:
            print ('===========> logger <===========')
            # (1) Log the scalar values
            info = {
                'lossesList_heatmap': lossesList_heatmap[5].val,
                'lossesList_paf': lossesList_paf[5].val
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, iteration)

            # (3) Log the images
            info = {
                'images': input[0:2].numpy(),
                'pafs_gt': torch.sum(torch.abs(paf_gt[0:2]), dim=1).numpy(),
                'pafs_pred': torch.sum(torch.abs(out_paf.cpu()[0:2]), dim=1).detach().numpy(),
                'hms_gt': torch.sum((heatmap_gt[0:2, :-1]), dim=1).numpy(),
                'hms_pred': torch.sum((out_heatmap.cpu()[0:2, :-1]), dim=1).detach().numpy()
            }
            for tag, images in info.items():
                print tag, images.shape
                logger.image_summary(tag, images, iteration)
        
        if i % args.savefreq == 0:  
            torch.save(netmodel.state_dict(), 'snapshot/lr_%d_%d.pkl'%(epoch,i))
        
    return iteration

def main(args): 
    logger = Logger('./logs')
    print ('===========> loading data <===========')
    datasetTrain = DatasetCocoKpt(ImageRoot='/home/dalong/data/coco2017/train2017', 
                                 AnnoFile='/home/dalong/data/coco2017/annotations/person_keypoints_train2017.json', 
                                 istrain=True)
    dataLoader_train = torch.utils.data.DataLoader(datasetTrain, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)
    print ('===========> loading model <===========')
    model = get_model(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(set_lr_groups(model.module, args.lr), args.lr, momentum=args.momentum,
                                weight_decay=args.weightdecay)
    iteration = 0
    epoch = 0
    while iteration < args.max_iter:
        print ('===========>   training    <===========')
        iteration = train(dataLoader_train, model, optimizer, epoch, iteration, logger, args)
        epoch += 1
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--workers', default=6, type=int, 
                        help='number of data loading workers')
    #################################################################
    ## https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/training/example_proto/pose_solver.prototxt
    parser.add_argument('--batchsize', default=48, type=int, 
                        help='mini-batch size') # 50: 12/gpu
    parser.add_argument('--lr', default=2e-6, type=float, 
                        help='initial learning rate')
    parser.add_argument('--lr_policy', default='step', type=str, 
                        help='learning rate policy')
    parser.add_argument('--gamma', default=0.333, type=float, 
                        help='used by step learning rate policy')
    parser.add_argument('--stepsize', default=15000, type=int, 
                        help='used by step learning rate policy')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weightdecay', default=5e-4, type=float, 
                        help='weight decay')
    parser.add_argument('--max_iter', default=60000, type=int, 
                        help='weight decay')
    ##################################################################
    parser.add_argument('--printfreq', default=10, type=int, 
                        help='print frequency')
    parser.add_argument('--savefreq', default=2000, type=int, 
                        help='save frequency')
    args = parser.parse_args()
    
    main(args)
