import dominate
from dominate.tags import *
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image

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
    
    def new_line(self, height=400):
        if self.t is not None and len(self.images)>0:
            with self.t:
                with tr():
                    for im, txt in zip(self.images, self.txts):
                        if len(im.shape)==3:
                            pil_img = Image.fromarray(im[:,:,::-1])
                        else:
                            pil_img = Image.fromarray(im)
                        buff = BytesIO()
                        pil_img.save(buff, format="JPEG")
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
                        pil_img = Image.fromarray(im[:,:,::-1])
                    else:
                        pil_img = Image.fromarray(im)
                    buff = BytesIO()
                    pil_img.save(buff, format="JPEG")
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
                            pil_img = Image.fromarray(im[:,:,::-1])
                        else:
                            pil_img = Image.fromarray(im)
                        buff = BytesIO()
                        pil_img.save(buff, format="JPEG")
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


if __name__ == '__main__':
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
