import os
import numpy as np
import cv2

class ATR(object):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.mean = np.array([136, 145, 151])
        self.atr_labels = ['background', 'hat', 'hair', 'sunglass', 'upper-clothes',
                           'skirt', 'pants', 'dress', 'belt', 'left-shoe', 'right-shoe',
                           'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm', 'bag',
                           'scarf'
                          ]

        file_list = os.listdir(os.path.join(root, self.split, 'images'))
        self.imgIds = [item.split('.')[0] for item in file_list if item.split('.')[-1] in ['jpg', 'png']]
        
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        image_id = self.imgIds[idx]
        image, label = self.load_data(image_id)
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32), label.astype(np.int64)
    
    def load_data(self, image_id):
        image_path = self.root +'/'+ self.split + '/images/' + image_id + '.jpg'
        label_path = self.root +'/'+ self.split + '/parsing/' + image_id + '.png'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        return image, label
    
    def flip_label(self, label):
        pairs = [[9,10], [12,13], [14,15]]
        for pair in pairs:
            tmpIds = (label==pair[0])
            label[label==pair[1]] = pair[0]
            label[tmpIds] = pair[1]
        return label
    
        
        