import dominate
from dominate.tags import *
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import argparse
import random
import cv2

'''
## Simple Usage
from html import *
html = MYHTML('./web', 'simple')
lists = get_filenames('/home/dalong/data/coco2017/train2017/')
html.add_line(lists, tag='images')
lists = ['./ignoremasks/'+f.split('/')[-1].replace('jpg', 'png') for f in lists]
html.add_line(lists, tag='ignoremasks', maxid=1)
html.save()
'''

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ImagesDir', type=str, default=None,
                        help='the dir stores images')
    parser.add_argument('--LabelsDir', type=str, default=None,
                        help='the dir stores labels')
    parser.add_argument('--PredsDir', type=str, default=None,
                        help='the dir stores preds')
    parser.add_argument('--HtmlDir', type=str, default='./web/',
                        help='the dir stores outputdefaulthtml file')
    parser.add_argument('--HtmlName', type=str, default='index',
                        help='the stored html file name')
    parser.add_argument('--Num', type=int, default=10,
                        help='the max number to show')
    parser.add_argument('--MaxLabelId', type=int, default=None,
                        help='the Classes number')
    return parser.parse_args()

# lists
def get_filenames(imagesdir, sample_N=20):
    EXTs = ['jpg', 'png', 'jpeg', 'bmp']
    lists = [os.path.join(imagesdir,f) for f in os.listdir(imagesdir) if f.split('.')[-1] in EXTs]
    if sample_N:
        sample_N = min(sample_N, len(lists))
        randIdxs = random.sample(range(len(lists)), sample_N)
        lists = [lists[idx] for idx in randIdxs]
    return lists

# read images
def get_images_txts(lists, tag='image', maxid=None):
    images = [cv2.imread(file) for file in lists]
    txts = []
    for i, data in enumerate(zip(images, lists)):
        image, file = data
        if image is None:
            images[i] = np.zeros((10,10,3), dtype = np.uint8)
            txts += ['%s [ID]%s not found!'%(tag,file.split('/')[-1])]
            continue
        txts += ['%s [ID]%s  [Size]%d*%d  [ValueRange]%d-%d'%(tag, 
                                                              file.split('/')[-1], 
                                                              image.shape[0], image.shape[1], 
                                                              np.min(image), np.max(image)
                                                             )]
    if maxid:
        scale = int(255.0/maxid)
        images = [image*scale for image in images]
    return images, txts   


class MYHTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

        self.t = None
        self.images = []
        self.txts = []
        
    def add_header(self, str):
        with self.doc:
            h3(str)
            
    def add_line(self, lists, tag='image', maxid=None):
        images, txts = get_images_txts(lists, tag, maxid)
        self.new_line()
        for image, txt in zip(images, txts): self.add_image(image, txt)
            
    def new_line(self, height=400):
        if self.t is not None and len(self.images)>0:
            with self.t:
                with tr():
                    for im, txt in zip(self.images, self.txts):
                        if len(im.shape)==3:
                            pil_image = Image.fromarray(im[:,:,::-1])
                        else:
                            pil_image = Image.fromarray(im)
                        buff = BytesIO()
                        pil_image.save(buff, format="JPEG")
                        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                img(style="height:%dpx" % self.height, src="data:image/jpg;base64,%s"%new_image_string)
                                br()
                                p(txt)    
        self.add_table()
        self.height = height
        self.images = []
        self.txts = []
        
    def add_image(self, im, txt):
        self.images.append(im)
        self.txts.append(txt)
        
    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)
        
    def add_images(self, ims, txts, height=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt in zip(ims, txts):
                    if len(im.shape)==3:
                        pil_image = Image.fromarray(im[:,:,::-1])
                    else:
                        pil_image = Image.fromarray(im)
                    buff = BytesIO()
                    pil_image.save(buff, format="JPEG")
                    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            img(style="height:%dpx" % height, src="data:image/jpg;base64,%s"%new_image_string)
                            br()
                            p(txt)
                            
    def save(self):
        if self.t is not None and len(self.images)>0:
            with self.t:
                with tr():
                    for im, txt in zip(self.images, self.txts):
                        if len(im.shape)==3:
                            pil_image = Image.fromarray(im[:,:,::-1])
                        else:
                            pil_image = Image.fromarray(im)
                        buff = BytesIO()
                        pil_image.save(buff, format="JPEG")
                        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                img(style="height:%dpx" % self.height, src="data:image/jpg;base64,%s"%new_image_string)
                                br()
                                p(txt)  
                                
        html_file = '%s/%s.html' % (self.web_dir, self.title)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

def demo():
    html = MYHTML('web/', 'test_html')
    html.add_header('hello world')
    #html.add_images([np.zeros((100,100), dtype = np.uint8)+128 for _ in range(20)], ['yeah' for _ in range(20)])
    #html.add_images([np.zeros((100,100), dtype = np.uint8)+128 for _ in range(20)], ['yeah' for _ in range(20)])
    html.new_line()
    for _ in range(20):
        html.add_image(np.zeros((100,100), dtype = np.uint8)+128, 'yeah')
    html.new_line()
    for _ in range(20):
        html.add_image(np.zeros((100,100), dtype = np.uint8)+128, 'yeah')
    html.save()

    
if __name__ == '__main__':
    args = parse()
    
    html = MYHTML(args.HtmlDir, args.HtmlName)
    
    lists = None
    if args.ImagesDir:
        lists = imagelists = get_filenames(args.ImagesDir)
    if args.LabelsDir:
        lists = labellists = get_filenames(args.LabelsDir)
    if args.PredsDir:
        lists = predlists = get_filenames(args.PredsDir)
    assert lists
    
    # select    
    sample_N = min(args.Num, len(lists))
    randIdxs = random.sample(range(len(lists)), sample_N)
    
    if args.ImagesDir:
        images, txts = get_images_txts([imagelists[idx] for idx in randIdxs], split='image', maxid=None)
        html.new_line()
        for image, txt in zip(images, txts): html.add_image(image, txt) 
    if args.LabelsDir:
        images, txts = get_images_txts([labellists[idx] for idx in randIdxs], split='label', maxid=args.MaxLabelId)
        html.new_line()
        for image, txt in zip(images, txts): html.add_image(image, txt) 
    if args.PredsDir:
        images, txts = get_images_txts([predlists[idx] for idx in randIdxs], split='pred', maxid=args.MaxLabelId)
        html.new_line()
        for image, txt in zip(images, txts): html.add_image(image, txt) 
    html.save()