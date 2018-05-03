import cv2
import numpy as np
import scipy.io as sio

class CocoStuff10k(object):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.mean = np.array([104.008, 116.669, 122.675]

        # Load all path to images
        file_list = tuple(open(root + '/imageLists/' + split + '.txt', 'r'))
        self.imgIds = [id_.rstrip() for id_ in file_list]

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        image_id = self.imgIds[idx]
        image, label = self.load_data(image_id)
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32), label.astype(np.int64)

    def load_data(self, image_id):
        image_path = self.root + '/images/' + image_id + '.jpg'
        label_path = self.root + '/annotations/' + image_id + '.mat'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = sio.loadmat(label_path)['S'].astype(np.int64)
        return image, label
