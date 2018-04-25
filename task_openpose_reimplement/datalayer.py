# Timing:(@profile) kernprof -l -v XXXX.py

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

    #H = H.dot(matrix_trans)[0:2, :]
    H = H.dot(matrix_trans)
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

# load heatmap and paf
def putGaussianMaps(entry, center, stride, grid_x, grid_y, sigma):
    start = stride / 2.0 - 0.5  # 0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    threshold = 4.6025 * sigma ** 2 * 2
    sqrt_threshold = math.sqrt(threshold)
    #(start + g_x * stride - center[0]) ** 2 + (start + g_y * stride - center[1]) ** 2 <= threshold ** 2
    min_y = max(0, int((center[1] - sqrt_threshold - start) / stride))
    max_y = min(grid_y-1, int((center[1] + sqrt_threshold - start) / stride))
    min_x = max(0, int((center[0] - sqrt_threshold - start) / stride))
    max_x = min(grid_x-1, int((center[0] + sqrt_threshold - start) / stride))
    g_y = np.arange(min_y,max_y+1)[:, None]
    g_x = np.arange(min_x,max_x+1)
    y = start + g_y * stride
    x = start + g_x * stride
    d2 = ((x - center[0]) ** 2 + (y - center[1]) ** 2) / 2 / sigma ** 2
    idx = np.where(d2<4.6025)
    circle = entry[min_y:max_y+1,min_x:max_x+1][idx]
    circle += np.exp(-d2[idx])  ## circle += np.exp(-d2[idx]) ?? 
    circle[circle > 1] = 1
    entry[min_y:max_y + 1, min_x:max_x + 1][idx] = circle

def putVecMaps(entryX, entryY, centerA_ori, centerB_ori, grid_x, grid_y, stride, thre):
    centerA = centerA_ori * (1.0 / stride)
    centerB = centerB_ori * (1.0 / stride)
    line = centerB - centerA

    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)

    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    norm_line = math.sqrt(line[0] * line[0] + line[1] * line[1]) + 1e-9
    lastX = entryX[min_y:max_y, min_x:max_x]
    lastY = entryY[min_y:max_y, min_x:max_x]
    line = 1.0 * line / norm_line
    g_y = np.arange(min_y, max_y)[:, None]
    g_x = np.arange(min_x, max_x)
    v0 = g_x - centerA[0]
    v1 = g_y - centerA[1]
    dist = abs(v0 * line[1] - v1 * line[0])
    idx = dist <= thre
    lastX[idx] = line[0]
    lastY[idx] = line[1]
    entryX[min_y:max_y, min_x:max_x] = lastX
    entryY[min_y:max_y, min_x:max_x] = lastY
    
def generate_heatmap(heatmap, kpt, stride, sigma):
    height, width, num_point = heatmap.shape
    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] == 0:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            putGaussianMaps(heatmap[:,:,j], [x,y], stride, width, height, sigma)

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
        ignorepath = 'ignoremasks/'+self.coco.loadImgs(img_id)[0]['file_name'].replace('jpg', 'png')
        ignoremask = cv2.imread(ignorepath, 0).astype(np.float32)
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
                if scale <= 0: # anno bug, return next
                    return self.loadItem(idx+1)
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
    #@profile
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
        
        def _augmentation_scale_fast(width, height, scale):
            if random.random() > self.scale_prob:
                scale_multiplier = 1.0
            else:
                scale_multiplier = random.random() * (self.scale_max - self.scale_min) + self.scale_min
            s = self.target_dist / scale * scale_multiplier
            Haug = np.array([[s,  0., 0.], 
                             [-0., s, 0.],
                             [0., 0., 1,]])
            return Haug, width*s, height*s

        
        def _pointAffine(points, mat):
            points = np.array(points)
            shape = points.shape
            points = points.reshape(-1, 2)
            return np.dot( np.concatenate((points, points[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)
        
        def _augmentation_rotate_fast(width, height):
            Haug = aug_matrix(width, height, width, height, 
                               angle_range=(-self.max_rotate_degree, self.max_rotate_degree), 
                               scale_range=(1.0, 1.0), 
                               trans_range=(-0.0, 0.0))
            Rect = _pointAffine(np.array([[0,0], [width,0], [0,height], [width, height]]), Haug[0:2, :])
            widthRect = int(np.max(Rect[:,0]) - np.min(Rect[:,0]))
            heightRect = int(np.max(Rect[:,1]) - np.min(Rect[:,1]))
            Haug[0,2] += widthRect/2.0 - width/2.0
            Haug[1,2] += heightRect/2.0 - height/2.0
            return Haug, widthRect, heightRect

        
        def _augmentation_crop_fast(width, height, center):
            x_offset = int((random.random() - 0.5) * 2 * self.center_perterb_max)
            y_offset = int((random.random() - 0.5) * 2 * self.center_perterb_max)
            c = np.array(center) + [x_offset, y_offset]
            offset_left = int(-(c[0] - (self.crop_size_w/2.0)))
            offset_up = int(-(c[1] - (self.crop_size_h/2.0)))
            Haug = np.array([[1.0,  0.0, offset_left], 
                             [-0.0, 1.0,   offset_up],
                             [0.0,  0.0,         1.0]])
            return Haug, self.crop_size_w, self.crop_size_h

        
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

        Visualize = True 
        if Visualize:
            html = MYHTML('web/', 'train_visulaize')
            html.new_line()
            html.add_image(np.uint8(image), 'origin image: %d*%d'%(image.shape[0],image.shape[1]))

        # aug fast
        height, width = image.shape[0:2]
        Haug1, tmpW, tmpH = _augmentation_scale_fast(width, height, scale)
        Haug2, tmpW, tmpH = _augmentation_rotate_fast(int(tmpW), int(tmpH))
        Haug3, tmpW, tmpH = _augmentation_crop_fast(int(tmpW), int(tmpH), 
                                                   _pointAffine(np.array(center), Haug2.dot(Haug1)[0:2, :])) 
        Haug_all = Haug3.dot(Haug2).dot(Haug1)[0:2, :]
        image = cv2.warpAffine(image, Haug_all, (int(tmpW), int(tmpH)), 
                                    borderValue=(128,128,128), flags=cv2.INTER_CUBIC)
        ignoremask = cv2.warpAffine(ignoremask, Haug_all, (int(tmpW), int(tmpH)), 
                                    borderValue=255, flags=cv2.INTER_CUBIC)
        keypoints_gt[:, :, 0:2] = _pointAffine(keypoints_gt[:, :, 0:2]*[width, height], Haug_all) / [tmpW, tmpH]
        keypoints_gt[keypoints_gt[:,:,2]==0] = 0
        center = _pointAffine(np.array(center), Haug_all)
        image, ignoremask, keypoints_gt, center = _augmentation_flip(image, ignoremask, keypoints_gt, center)
        keypoints_gt = _TransformJoints(keypoints_gt) # 17 -> ours 18
        
        if Visualize:
            vis = visualize(image, keypoints_gt, group=True)[0]
            cv2.circle(vis, tuple(center.astype(np.int32)), 11, [255,255,255], thickness=-1)
            html.add_image(np.uint8(vis), 'after aug: %d*%d'%(image.shape[0],image.shape[1]))
        
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
        
        paf = np.zeros((label_size_h, label_size_w, len(self.vec_pair) * 2), dtype=np.float32)
        cnt = np.zeros((label_size_h, label_size_w, len(self.vec_pair)), dtype=np.int32)
        paf = generate_vector(paf, cnt, keypoints_gt*[self.crop_size_h, self.crop_size_w, 1], self.vec_pair, self.stride, self.theta)
        if Visualize:
            html.new_line()
            for i in range(paf.shape[2]):
                html.add_image(np.uint8(pylab.cm.hsv(cv2.resize(np.abs(paf[:,:,i]),image.shape[0:2]))[:,:,0:3]*255*0.5+image*0.5), '%s-%s'%(self.ours_atrs[self.vec_pair[int(i/2)][0]], self.ours_atrs[self.vec_pair[int(i/2)][1]])
                              )
        
            html.save()
        paf = paf.transpose((2,1,0))  
        paf = (paf * ignoremask).astype(np.float32)
        
        
        return input, heatmap, paf, ignoremask


if __name__ == '__main__':
    dataset = DatasetCocoKpt(ImageRoot='/home/dalong/data/coco2017/train2017', 
                             AnnoFile='/home/dalong/data/coco2017/annotations/person_keypoints_train2017.json', 
                             istrain=True)
    #for i in range(20):
    #    data = dataset[i]
    data = dataset[1]