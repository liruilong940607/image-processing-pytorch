import cv2
import sys
import os
import numpy as np

class CocoBase(object):
    def __init__(self, ImageRoot, AnnoFile):
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
        
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        return self.loadImage(idx)
    
    def loadImage(self, idx):
        # basic info
        img_id = self.imgIds[idx]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(anno_ids)
        # load image
        inp = np.float32(cv2.imread(os.path.join(self.root, path)))
        height, width, channel = inp.shape
        # load && normalized keypoints && mask && bbox
        gtN = len(annos)
        keypoints_gt = [] #  (gtN, self.num_parts, 3)
        masks_gt = [] # (gtN, height, width)
        for i, anno in enumerate(annos):
            if 'keypoints' in anno:
                keypoints_gt.append( np.array(anno['keypoints']).reshape([-1,3]) )
                keypoints_gt[i] = keypoints_gt[i]/[float(width), float(height), 1.0] # normalize
            if 'segmentation' in anno:
                masks_gt.append( self.annToMask(anno, height, width).astype(np.float32) )
        keypoints_gt = np.array(keypoints_gt)   
        masks_gt = np.array(masks_gt)
        # return 
        results_dict = {
            'img_id': img_id,
            'path': os.path.join(self.root, path), 
            'image': inp, # (H, W, 3)
            'masks_gt': masks_gt, # (gtN, height, width)
            'keypoints_gt': keypoints_gt # (gtN, self.num_parts, 3), normalized
        }
        
        return results_dict
    
    def annToRLE(self, anno, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = anno['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = self.maskUtils.frPyObjects(segm, height, width)
            rle = self.maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = self.maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = anno['segmentation']
        return rle

    def annToMask(self, anno, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(anno, height, width)
        mask = self.maskUtils.decode(rle)
        return mask
    