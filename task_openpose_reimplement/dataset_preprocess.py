import os
import sys
import math
import json
import numpy as np
import cv2
sys.path.insert(0,'/home/dalong/data/coco2017/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
import argparse
from tqdm import tqdm  

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IgnoremasksDir', type=str, default='./ignoremasks',
                        help='the save_dir for the generated ignoremasks files')
    parser.add_argument('--AllmasksDir', type=str, default='./allmasks',
                        help='the save_dir for the generated allmasks files')
    parser.add_argument('--AnnofileTrain', type=str,
                        default='/home/dalong/data/coco2017/annotations/person_keypoints_train2017.json',
                        help='the trainset annofile')
    parser.add_argument('--AnnofileVal', type=str, 
                        default='/home/dalong/data/coco2017/annotations/person_keypoints_val2017.json',
                        help='the valset annofile')
    parser.add_argument('--ImageDirTrain', type=str,
                        default='/home/dalong/data/coco2017/train2017/',
                        help='the trainset image dir')
    parser.add_argument('--ImageDirVal', type=str, 
                        default='/home/dalong/data/coco2017/val2017/',
                        help='the valset image dir')
    return parser.parse_args()

def gen_ignoremask(args):
    outdir_ignore = args.IgnoremasksDir
    outdir_all = args.AllmasksDir
    if not os.path.exists(outdir_ignore):
        os.makedirs(outdir_ignore)
    if not os.path.exists(outdir_all):
        os.makedirs(outdir_all)
    print ('[Proprocess] generate ignoremasks to %s'%outdir_ignore)
    print ('[Proprocess] generate allmasks to %s'%outdir_all)
    for split in [args.AnnofileTrain, args.AnnofileVal]:
        coco = COCO(split)
        catIds = coco.getCatIds(catNms=['person']);
        imgIds = sorted(coco.getImgIds(catIds=catIds))
        for i, img_id in tqdm(enumerate(imgIds)):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)
            numPeople = len(img_anns)
            name = coco.imgs[img_id]['file_name']
            height = coco.imgs[img_id]['height']
            width = coco.imgs[img_id]['width']

            mask_all = np.zeros((height, width), dtype=np.uint8)
            mask_miss = np.zeros((height, width), dtype=np.uint8)
            flag = 0
            for p in img_anns:
                if p['iscrowd'] == 1:
                    mask_crowd = coco.annToMask(p)
                    temp = np.bitwise_and(mask_all, mask_crowd)
                    mask_crowd = mask_crowd - temp
                    flag += 1
                    continue
                else:
                    mask = coco.annToMask(p)
                mask_all = np.bitwise_or(mask, mask_all)
                if p['num_keypoints'] <= 0:
                    mask_miss = np.bitwise_or(mask, mask_miss)
            if flag < 1:
                mask_miss = np.logical_not(mask_miss)
            elif flag == 1:
                mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
                mask_all = np.bitwise_or(mask_all, mask_crowd)
            else:
                raise Exception('crowd segments > 1')
            mask_all[mask_all>1] = 1
            mask_miss[mask_miss>1] = 1
            cv2.imwrite(os.path.join(outdir_all, name.split('.')[0] + '.png'), np.uint8(mask_all))
            cv2.imwrite(os.path.join(outdir_ignore, name.split('.')[0] + '.png'), np.uint8(mask_miss))
            
    
if __name__ == '__main__':
    args = parse()
    gen_ignoremask(args)