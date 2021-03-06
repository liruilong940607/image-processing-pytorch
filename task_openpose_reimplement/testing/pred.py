import cv2 as cv 
import cv2
import numpy as np
import scipy
import PIL.Image
import math
import json
import sys
sys.path.insert(0, '/mnt/nvme0n1/dalong/Realtime_Multi-Person_Pose_Estimation/caffe/python/')
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
import pylab as plt
import os
import random
import scipy
from scipy.ndimage.filters import gaussian_filter

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

param, model = config_reader()
if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

def pred_single(oriImg):
    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])

        net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        #net.forward() # dry run
        net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        start_time = time.time()
        output_blobs = net.forward()

        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
        paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    #[2] 
    all_peaks = []
    peak_counter = 0

    for part in range(19-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    
    #[3]
    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    
                    if norm==0:
                        continue

                    vec = np.divide(vec, norm)
                    
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    #[4]       
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    
    def to_jsonformat(subset, candidate):
        '''
        default: ["nose", "neck", 
            "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder",,"left_elbow",,"left_wrist",
            "right_hip","right_knee","right_ankle",
            "left_hip","left_knee","left_ankle",
            "left_eye","right_eye","left_ear","right_ear"]
        coco: ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
            "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        '''
        keypoint_item = {}
        score_item = {}
        submit_points_index = []
        for n in xrange(len(subset)):
            key = 'human%d'%(n+1)
            keypoint_item[key] = np.zeros(shape=(18*3), dtype = np.int64).tolist()
            score_item[key] = subset[n][-2]
            for i in xrange(18):
                index = int(subset[n][i])
                if index == -1:
                    continue
                else:
                    submit_points_index.append(index)
                    X = int(round(candidate[index][0]))
                    Y = int(round(candidate[index][1]))
    #                 real_index = partidx_convert_to_submit(i)
                    real_index = i
                    keypoint_item[key][real_index*3+0] = X
                    keypoint_item[key][real_index*3+1] = Y
                    keypoint_item[key][real_index*3+2] = 2

        return keypoint_item, submit_points_index, score_item

    keypoint_item, submit_points_index, score_item = to_jsonformat(subset, candidate)
    
    return keypoint_item, score_item

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/dalong/data/coco2017/cocoapi/PythonAPI/')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    AnnoFile = '/home/dalong/data/coco2017/annotations/person_keypoints_val2017.json'
    coco = COCO(AnnoFile)
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = sorted(coco.getImgIds(catIds=catIds))
    
    image_dir = '/home/dalong/data/coco2017/val2017/'
    image_list = [coco.loadImgs(image_id)[0]['file_name'][:-4] for image_id in imgIds]
        

#     image_dir = '/home/dalong/data/ATR/train/images/'
#     image_list = os.listdir(image_dir)
#     image_list = [item[:-4] for item in image_list if item[-4:]=='.jpg']
    
    json_all = []
    s = time.time()
    for i in range(len(image_list)):
        if i%20==0:
            time_cost = (time.time()-s) / 20.0
            s = time.time()
            print i, '/', len(image_list), 'aver time cost:', time_cost, 's / frame'

        image_id = image_list[i]
        oriImg = cv.imread(os.path.join(image_dir, image_id+'.jpg')) # B,G,R order
        keypoint_item, score_item = pred_single(oriImg)
        
        # gt
        # print image_id
        # anno_ids = coco.getAnnIds(imgIds=int(image_id))
        # annos = coco.loadAnns(anno_ids)
        # for i in range(len(annos)):
        #     keypoint_item[keypoint_item.keys()[i]] = annos[i]['keypoints']
        # print annos
        if len(keypoint_item)==0:
            keypoint_item = {}
        json_item = {'image_id':image_id, 'keypoint_annotations':keypoint_item, 'scores': score_item}
        json_all.append(json_item)
        # break
        
    # coco submit format
    convert_ids = [0, 15,14,17,16, 5,2, 6,3, 7,4, 11,8, 12,9, 13,10]
    submits = []
    cnt = 0
    for item in json_all:
        for humankey, kpts18 in item['keypoint_annotations'].items():
            kpts17 = np.array(kpts18).reshape(-1, 3)[convert_ids]
            score = item['scores'][humankey]
            cnt += 1
            submits.append({
                "image_id": int(item['image_id']),
                "category_id": 1,
                "score": score,
                "keypoints": kpts17.reshape(-1).tolist(),
                "id": cnt 
            })
#     gt = annos[1]['keypoints']
#     pt = kpts17.reshape(-1).tolist()
    
#     colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255]]
    
#     for i in range(17):
#         canvas_gt = oriImg.copy()
#         x,y,v = gt[i*3+0:i*3+3]
#         if x==0 and y==0:
#             pass
#         else:
#             cv2.circle(canvas_gt, (x,y), 11, colors[i%len(colors)], thickness=-1)
    
#         canvas_pt = oriImg.copy()
#         x,y,v = pt[i*3+0:i*3+3]
#         if x==0 and y==0:
#             pass
#         else:
#             cv2.circle(canvas_pt, (x,y), 11, colors[i%len(colors)], thickness=-1)
        
#         cv2.imwrite('test_%d.jpg'%i, np.hstack((canvas_gt, canvas_pt)))
#     print submits
    print 'start write json'
    with open('results.json', 'w') as json_file:
        json_file.write(json.dumps(submits))
    print 'write json done.'
    
    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes('results.json')
    # Evaluate
    cocoEval = COCOeval(coco, coco_results, 'keypoints')
    #cocoEval.params.imgIds = [imgIds[0]]
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

