import random
import numpy as np
import os
import cv2
from tqdm import tqdm
import math
import pylab
import time

import torch
from torch.utils import data
import torchvision
from html import MYHTML


def aug_matrix(w1, h1, w2, h2, angle_range=(-45, 45), scale_range=(0.5, 1.5), trans_range=(-0.3, 0.3)):
    ''' 
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''
    dx = (w2-w1)/2.0
    dy = (h2-h1)/2.0
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])

    angle = random.random()*(angle_range[1]-angle_range[0])+angle_range[0]
    scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
    scale *= np.min([float(w2)/w1, float(h2)/h1])
    alpha = scale * math.cos(angle/180.0*math.pi)
    beta = scale * math.sin(angle/180.0*math.pi)

    trans = random.random()*(trans_range[1]-trans_range[0])+trans_range[0]
    centerx = w2/2.0 + w2*trans
    centery = h2/2.0 + h2*trans
    H = np.array([[alpha, beta, -beta*centery+(1-alpha)*centerx], 
                  [-beta, alpha, beta*centerx+(1-alpha)*centery],
                  [0,         0,                            1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H 

def visualize(canvas_inp, keypoints_inp, group=True):
    '''
    canvas_inp: (H, W, 3) [0, 255]
    keypoints_inp: (N, np, 3), [0, 1]
    return : (N, H, W, 3)
    '''
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    canvas = np.float32(canvas_inp)
    H, W, C = canvas.shape
    keypoints = np.int32(np.array(keypoints_inp)*[W, H, 1.0])
    N, npart, _ = keypoints.shape
    
    if group:
        canvas_t = canvas.copy()
        for keypoint in keypoints:
            for i, part in enumerate(keypoint):
                if part[0]==0 and part[1]==0:
                    continue
                cv2.circle(canvas_t, tuple(part[0:2]), 11, colors[i%len(colors)], thickness=-1)
        to_plot = cv2.addWeighted(canvas, 0.3, canvas_t, 0.7, 0)
        return np.uint8(to_plot[np.newaxis, ...]) # (1, H, W, 3)
    else:
        to_plots = np.zeros((N, H, W, C), dtype = np.float32)
        for ni, keypoint in enumerate(keypoints):
            canvas_t = canvas.copy()
            for i, part in enumerate(keypoint):
                if part[0]==0 and part[1]==0:
                    continue
                cv2.circle(canvas_t, tuple(part[0:2]), 11, colors[i%len(colors)], thickness=-1)
            to_plots[ni] = cv2.addWeighted(canvas, 0.3, canvas_t, 0.7, 0)
        return np.uint8(to_plots) # (N, H, W, 3)

def generate_heatmap(heatmap, kpt, stride, sigma):

    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] == 0:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    heatmap[h][w][j] += math.exp(-dis)
                    if heatmap[h][w][j] > 1:
                        heatmap[h][w][j] = 1

    return heatmap

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[i][0]
            b = vec_pair[i][1]
            if kpts[j][a][2] == 0 or kpts[j][b][2] == 0:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1

    return vector

class DatasetCocoKpt(object):
    def __init__(self, ImageRoot, AnnoFile, istrain=True):
        import sys
        sys.path.insert(0,'/home/dalong/data/coco2017/cocoapi/PythonAPI/')
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from pycocotools import mask as maskUtils
        self.COCO = COCO
        self.COCOeval = COCOeval
        self.maskUtils = maskUtils
        
        self.root = ImageRoot
        self.coco = COCO(AnnoFile)
        self.catIds = self.coco.getCatIds(catNms=['person']);
        self.imgIds = sorted(self.coco.getImgIds(catIds=self.catIds))
        self.itemIds = []
        for img_id in self.imgIds:
            annos = self.coco.getAnnIds(imgIds=img_id)
            for idx in range(len(annos)):
                self.itemIds.append([img_id, idx])
                
        self.istrain = istrain
        self.MEAN = [128.0, 128.0, 128.0]
        self.ours_atrs =  ["nose", "neck", 
                        "right_shoulder", "right_elbow", "right_wrist",
                        "left_shoulder","left_elbow","left_wrist",
                        "right_hip","right_knee","right_ankle",
                        "left_hip","left_knee","left_ankle",
                        "left_eye","right_eye","left_ear","right_ear", "BG"]
        self.coco_atrs = ["nose","left_eye","right_eye","left_ear","right_ear",
                        "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
                        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        self.flipRef = np.array([1, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14, 17,16]) - 1
        self.vec_pair = np.array([[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                           [1,16], [16,18], [3,17], [6,18]]) - 1
        self.theta = 1.0
        self.sigma = 7.0
        
        # == params ==
        self.stride = 8
        self.max_rotate_degree = 45
        self.crop_size_w = 368
        self.crop_size_h = 368
        self.scale_prob = 1
        self.scale_min = 0.5
        self.scale_max = 1.1
        self.target_dist = 0.6
        self.center_perterb_max = 40

        print ('total item in dataset is : %d'%self.__len__())
        
    def __len__(self):
        return len(self.itemIds)

    def __getitem__(self, idx):
        img_id, image, keypoints_gt, scale, center = self.loadItem(idx) 
        ignorepath = 'ignoremasks/'+self.coco.loadImgs(img_id)[0]['file_name'].replace('jpg', 'npy')
        if os.path.exists(ignorepath):
            ignoremask = np.load(ignorepath).astype(np.float32)
        else:
            ignoremask = np.zeros(image.shape[0:2], dtype = np.float32)
        input, heatmap, paf, ignoremask = self.inputProcess(img_id, image, keypoints_gt, scale, center, ignoremask)
        return input, heatmap, paf, ignoremask
    
    def loadItem(self, idx):
        # basic info
        img_id, anno_idx = self.itemIds[idx]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(anno_ids)
        
        # load image
        image = np.float32(cv2.imread(os.path.join(self.root, path)))
        height, width, channel = image.shape
        
        # load && normalized keypoints && normalized bbox
        keypoints_gt = []
        for idx, anno in enumerate(annos):
            keypoints_gt.append( np.array(anno['keypoints']).reshape([-1,3]) ) 
            if idx == anno_idx:
                bbox_gt = anno['bbox'] # (x1,y1,w,h) 
                scale = bbox_gt[3] / float(self.crop_size_h)
                center = np.array([bbox_gt[0] + bbox_gt[2]/2.0, bbox_gt[1] + bbox_gt[3]/2.0])
        keypoints_gt = np.array(keypoints_gt) / [float(width), float(height), 1.0] # (gtN, 17, 3) normalized
        
        return img_id, image, keypoints_gt, scale, center
    
    '''
    # TODO : loadImage and random select bz (OR) loadItem. which is best?
    def loadImage(self, idx):
        # basic info
        img_id = self.imgIds[idx]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(anno_ids)
        # load image
        inp = np.float32(cv2.imread(os.path.join(self.root, path)))
        height, width, channel = inp.shape
        # load && normalized keypoints && normalized bbox
        gtN = len(annos)
        keypoints_gt = [] #  (gtN, self.num_parts, 3)
        bbox_gt = [] # (gtN, (x1,y1,x2,y2))
        for i, anno in enumerate(annos):
            keypoints_gt.append( np.array(anno['keypoints']).reshape([-1,3]) )
            keypoints_gt[i] = keypoints_gt[i]/[float(width), float(height), 1.0] # normalize
            bbox = anno['bbox']/[float(width), float(height), float(width), float(height)] # (x1,y1,w,h) normalize
            bbox_gt.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
        keypoints_gt = np.array(keypoints_gt)   
        bbox_gt = np.array(bbox_gt)
        # return 
        results_dict = {
            'img_id': img_id,
            'path': os.path.join(self.root, path), 
            'image': inp, # (H, W, 3)
            'bbox_gt': bbox_gt # (gtN, (x1,y1,x2,y2)), normalized
            'keypoints_gt': keypoints_gt # (gtN, self.num_parts, 3), normalized
        }
        
        return results_dict
    '''
    
    def inputProcess(self, img_id, image, keypoints_gt, scale, center, ignoremask):
        '''
        keypoints_gt: (gtN, 17, 3)
        '''
        def _TransformJoints(keypoints_gt):
            assert keypoints_gt.shape[1]==17
            COCO_to_ours_1 = np.array([1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4]) - 1
            COCO_to_ours_2 = np.array([1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4]) - 1
            keypoints_ours = (keypoints_gt[:, COCO_to_ours_1, :] + keypoints_gt[:, COCO_to_ours_2, :]) * 0.5
            # neck visible? (visiable is 2, unvisiable is 1 and not labeled is 0)
            for i in range(len(keypoints_gt)):
                if keypoints_gt[i,6-1,2] == keypoints_gt[i,7-1,2]:
                    pass
                elif keypoints_gt[i,6-1,2]==0 or keypoints_gt[i,7-1,2]==0:
                    keypoints_ours[i,1,2] = 0
                else:
                    keypoints_ours[i,1,2] = 1
            return keypoints_ours
          
        def _augmentation_scale(image, ignoremask, keypoints_gt, center):
            if random.random() > self.scale_prob:
                scale_multiplier = 1.0
            else:
                scale_multiplier = random.random() * (self.scale_max - self.scale_min) + self.scale_min
            s = self.target_dist / scale * scale_multiplier
            _image = cv2.resize(image, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            _ignoremask = cv2.resize(ignoremask, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            _keypoints_gt = keypoints_gt.copy() # because of normalized
            _center = center * s
            return _image, _ignoremask, _keypoints_gt, _center
        
        def _pointAffine(points, mat):
            points = np.array(points)
            shape = points.shape
            points = points.reshape(-1, 2)
            return np.dot( np.concatenate((points, points[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)

        def _augmentation_rotate(image, ignoremask, keypoints_gt, center):
            height, width = image.shape[0:2]
            Haug = aug_matrix(width, height, width, height, 
                               angle_range=(-self.max_rotate_degree, self.max_rotate_degree), 
                               scale_range=(1.0, 1.0), 
                               trans_range=(-0.0, 0.0))
            Rect = _pointAffine(np.array([[0,0], [width,0], [0,height], [width, height]]), Haug)
            widthRect = int(np.max(Rect[:,0]) - np.min(Rect[:,0]))
            heightRect = int(np.max(Rect[:,1]) - np.min(Rect[:,1]))
            Haug[0,2] += widthRect/2.0 - width/2.0
            Haug[1,2] += heightRect/2.0 - height/2.0
            _image = cv2.warpAffine(image, Haug, (widthRect, heightRect), 
                                    borderValue=(128,128,128), flags=cv2.INTER_CUBIC)
            _ignoremask = cv2.warpAffine(ignoremask, Haug, (widthRect, heightRect), 
                                         borderValue=255, flags=cv2.INTER_CUBIC)
            _keypoints_gt = keypoints_gt.copy()
            _keypoints_gt[:, :, 0:2] = _pointAffine(keypoints_gt[:, :, 0:2]*[width, height], Haug) / [widthRect, heightRect]
            _keypoints_gt[keypoints_gt[:,:,2]==0] = 0
            _center = _pointAffine(np.array(center), Haug)
            return _image, _ignoremask, _keypoints_gt, _center
        
        def _augmentation_crop(image, ignoremask, keypoints_gt, center):
            height, width = image.shape[0:2]
            x_offset = int((random.random() - 0.5) * 2 * self.center_perterb_max)
            y_offset = int((random.random() - 0.5) * 2 * self.center_perterb_max)
            c = np.array(center) + [x_offset, y_offset]
            offset_left = int(-(c[0] - (self.crop_size_w/2.0)))
            offset_up = int(-(c[1] - (self.crop_size_h/2.0)))
            Haug = np.array([[1.0,  0.0, offset_left], 
                             [-0.0, 1.0,  offset_up]])
            _image = cv2.warpAffine(image, Haug, (self.crop_size_w, self.crop_size_h), 
                                    borderValue=(128,128,128), flags=cv2.INTER_CUBIC)
            _ignoremask = cv2.warpAffine(ignoremask, Haug, (self.crop_size_w, self.crop_size_h), 
                                         borderValue=255, flags=cv2.INTER_CUBIC)
            _keypoints_gt = keypoints_gt.copy()
            _keypoints_gt[:, :, 0:2] = _pointAffine(keypoints_gt[:, :, 0:2]*[width, height], Haug) / [self.crop_size_w, self.crop_size_h]
            _keypoints_gt[keypoints_gt[:,:,2]==0] = 0
            _center = _pointAffine(np.array(center), Haug)
            
            return _image, _ignoremask, _keypoints_gt, _center
        
        def _augmentation_flip(image, ignoremask, keypoints_gt, center):
            if random.random()<0.5:
                return image, ignoremask, keypoints_gt, center
            else:
                height, width = image.shape[0:2]
                _image = image[:,::-1,:].copy()
                _ignoremask = ignoremask[:,::-1].copy()
                _keypoints_gt = keypoints_gt[:,self.flipRef,:].copy()
                _keypoints_gt[:, :, 0] = 1.0 - _keypoints_gt[:, :, 0]
                _keypoints_gt[_keypoints_gt[:,:,2]==0] = 0
                _center = np.array([width-center[0], center[1]])
                return _image, _ignoremask, _keypoints_gt, _center
        
        a = time.time()
        Visualize = False 
        if Visualize:
            html = MYHTML('web/', 'train_visulaize')
            html.new_line()
            html.add_image(np.uint8(image), 'origin image: %d*%d'%(image.shape[0],image.shape[1]))
        # aug
        image, ignoremask, keypoints_gt, center = _augmentation_scale(image, ignoremask, keypoints_gt, center)
        if Visualize:
            vis = visualize(image, keypoints_gt, group=True)[0]
            cv2.circle(vis, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
            html.add_image(np.uint8(vis), 'after aug_scale: %d*%d'%(image.shape[0],image.shape[1]))
        image, ignoremask, keypoints_gt, center = _augmentation_rotate(image, ignoremask, keypoints_gt, center)
        if Visualize:
            vis = visualize(image, keypoints_gt, group=True)[0]
            cv2.circle(vis, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
            html.add_image(np.uint8(vis), 'after aug_rotate: %d*%d'%(image.shape[0],image.shape[1]))
        image, ignoremask, keypoints_gt, center = _augmentation_crop(image, ignoremask, keypoints_gt, center)
        if Visualize:
            vis = visualize(image, keypoints_gt, group=True)[0]
            cv2.circle(vis, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
            html.add_image(np.uint8(vis), 'after aug_crop: %d*%d'%(image.shape[0],image.shape[1]))
        image, ignoremask, keypoints_gt, center = _augmentation_flip(image, ignoremask, keypoints_gt, center)
        if Visualize:
            vis = visualize(image, keypoints_gt, group=True)[0]
            cv2.circle(vis, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
            html.add_image(np.uint8(vis), 'after aug_flip: %d*%d'%(image.shape[0],image.shape[1]))
        keypoints_gt = _TransformJoints(keypoints_gt) # 17 -> ours 18
        if Visualize:
            vis = visualize(image, keypoints_gt, group=True)[0]
            cv2.circle(vis, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
            html.add_image(np.uint8(vis), '17 kpt -> 18 kpt: %d*%d'%(image.shape[0],image.shape[1]))
        print ('[AUG]', (time.time()-a) , 's')
        a = time.time()
        
        # some simple process
        input = (image - 128.0)/256.0
        input = input.transpose((2,1,0)).astype(np.float32)
        label_size_h = self.crop_size_h/self.stride
        label_size_w = self.crop_size_w/self.stride
        ignoremask = cv2.resize(ignoremask, (label_size_w, label_size_h), interpolation=cv2.INTER_CUBIC)
        ignoremask[ignoremask>0.5] = 1
        ignoremask[ignoremask<=0.5] = 0
        ignoremask = ignoremask.astype(np.float32).reshape((label_size_h, label_size_w, 1))
        if Visualize:
            html.add_image(np.uint8(ignoremask[:,:,0]*255), 'ignoremask: %d*%d'%(ignoremask.shape[0],ignoremask.shape[1]))
        ignoremask = ignoremask.transpose((2,1,0))
        print ('[simple process]', (time.time()-a) , 's')
        a = time.time()
        
        # generate heatmap and paf
        heatmap = np.zeros((label_size_h, label_size_w, keypoints_gt.shape[1] + 1), dtype=np.float32)
        heatmap = generate_heatmap(heatmap, keypoints_gt*[self.crop_size_h, self.crop_size_w, 1], self.stride, self.sigma)
        heatmap[:,:,-1] = 1.0 - np.max(heatmap[:,:,:-1], axis=2) # for background
        if Visualize:
            html.new_line()
            for i in range(heatmap.shape[2]):
                html.add_image(np.uint8(pylab.cm.hsv(cv2.resize(heatmap[:,:,i],image.shape[0:2]))[:,:,0:3]*255*0.5+image*0.5), self.ours_atrs[i])
        heatmap = heatmap.transpose((2,1,0))
        heatmap = (heatmap * ignoremask).astype(np.float32)
        print ('[heatmap]', (time.time()-a) , 's')
        a = time.time()
        
        paf = np.zeros((label_size_h, label_size_w, len(self.vec_pair) * 2), dtype=np.float32)
        cnt = np.zeros((label_size_h, label_size_w, len(self.vec_pair)), dtype=np.int32)
        paf = generate_vector(paf, cnt, keypoints_gt*[self.crop_size_h, self.crop_size_w, 1], self.vec_pair, self.stride, self.theta)
        if Visualize:
            html.new_line()
            for i in range(paf.shape[2]):
                html.add_image(np.uint8(pylab.cm.hsv(cv2.resize(np.abs(paf[:,:,i]),image.shape[0:2]))[:,:,0:3]*255*0.5+image*0.5), 
                               '%s-%s'%(self.ours_atrs[self.vec_pair[int(i/2)][0]], self.ours_atrs[self.vec_pair[int(i/2)][1]])
                              )
        
            html.save()
        paf = paf.transpose((2,1,0))  
        paf = (paf * ignoremask).astype(np.float32)
        print ('[paf]', (time.time()-a) , 's')
        a = time.time()
        
        '''
        height, width = image.shape[0:2]
        vis1 = visualize(image, keypoints_gt, group=True)[0]
        cv2.circle(vis1, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
        image, ignoremask, keypoints_gt, center = _augmentation_scale(image, ignoremask, keypoints_gt, center)
        vis2 = visualize(image, keypoints_gt, group=True)[0]
        cv2.circle(vis2, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
        image, ignoremask, keypoints_gt, center = _augmentation_rotate(image, ignoremask, keypoints_gt, center)
        vis3 = visualize(image, keypoints_gt, group=True)[0]
        cv2.circle(vis3, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
        image, ignoremask, keypoints_gt, center = _augmentation_crop(image, ignoremask, keypoints_gt, center)
        vis4 = visualize(image, keypoints_gt, group=True)[0]
        cv2.circle(vis4, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
        image, ignoremask, keypoints_gt, center = _augmentation_flip(image, ignoremask, keypoints_gt, center)
        vis5 = visualize(image, keypoints_gt, group=True)[0]
        cv2.circle(vis5, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
        cv2.imwrite('test.jpg', np.hstack((vis1, 
                                           cv2.resize(vis2, (int(float(vis2.shape[1])*height/vis2.shape[0]), height)),
                                           cv2.resize(vis3, (int(float(vis3.shape[1])*height/vis3.shape[0]), height)),
                                           cv2.resize(vis4, (int(float(vis4.shape[1])*height/vis4.shape[0]), height)),
                                           cv2.resize(vis5, (int(float(vis5.shape[1])*height/vis5.shape[0]), height))
                                          )))
        '''
        
        return input, heatmap, paf, ignoremask


if __name__ == '__main__':
    dataset = DatasetCocoKpt(ImageRoot='/home/dalong/data/coco2017/train2017', 
                             AnnoFile='/home/dalong/data/coco2017/annotations/person_keypoints_train2017.json', 
                             istrain=True)
    data = dataset[1]