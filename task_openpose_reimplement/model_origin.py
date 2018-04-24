import torch
import torch.nn as nn

import model_backbone_resnet
import model_backbone_vgg

from model_backbone_vgg import model_urls
import torch.utils.model_zoo as model_zoo

def init_with_pretrain(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    for k, v in pretrained_dict.items():
        print 'init layer: ', k
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def get_model():
    
    cfg_vgg = {
        # 'M' means MaxPooling
        'downsample8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 256, 128]
    }
    backbone_net = model_backbone_vgg.VGG(model_backbone_vgg.make_layers(cfg_vgg['downsample8']), outplanes=cfg_vgg['downsample8'][-1])
    # init vgg with vgg19
    print '============= init with VGG19 ================'
    init_with_pretrain(backbone_net, model_zoo.load_url(model_urls['vgg19']))

    # [128,3,1,1]: channels, kernal_size, pad, stride.
    cfg_stage_1 = {    
        'origin': [ [128,3,1,1], [128,3,1,1], [128,3,1,1], [512,1,0,1] ]
    }
    cfg_stage_n = {
        'origin': [ [128,7,3,1], [128,7,3,1], [128,7,3,1], [128,7,3,1], [128,7,3,1], [128,1,0,1] ]
    }
    
    # 1x1 Convolution
    def conv1x1(in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, padding=0)
    # 3x3 Convolution
    def conv3x3(in_channels, out_channels, stride=1, groups=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, groups=groups)
    # 7x7 Convolution
    def conv3x3(in_channels, out_channels, stride=1, groups=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=7,
                         stride=stride, padding=3, groups=groups)
   
    class Branch(nn.Module):
        def __init__(self, in_channels, cfg):
            '''
            cfg should like this: 
                cfg_stage_1 = {    
                    'origin': [ [128,3,1,1], [128,3,1,1], [128,3,1,1], [512,1,0,1] ]
                }
            note: [128,3,1,1] means channels, kernal_size, pad, stride.
            '''
            super(Branch, self).__init__()
            self.in_channels = in_channels
            self.cfg = cfg
            self.outplanes = cfg[-1][0]
            self.sequence = self._make_layers()
            
        def _make_layers(self):
            layers = []
            for v in self.cfg:
                conv2d = nn.Conv2d(self.in_channels, out_channels=v[0], kernel_size=v[1], padding=v[2], stride=v[3])
                layers += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v[0]
            return nn.Sequential(*layers)
        
        def forward(self, x):
            return self.sequence(x)
            
    class Stage(nn.Module):
        def __init__(self, in_channels, feature_channels, cfg, is_final=False):
            super(Stage, self).__init__()
            self.in_channels = in_channels
            self.cfg = cfg
            self.is_final = is_final
            self.branch_paf = Branch(in_channels, cfg)
            self.conv_paf = conv1x1(self.branch_paf.outplanes, 38)
            
            self.branch_heatmap = Branch(in_channels, cfg)
            self.conv_heatmap = conv1x1(self.branch_heatmap.outplanes, 19)
            
            if not is_final:
                self.outplanes = feature_channels + 38 + 19

        def forward(self, x, features):
            feature_paf = self.branch_paf(x)
            out_paf = self.conv_paf(feature_paf)
            
            feature_heatmap = self.branch_heatmap(x)
            out_heatmap = self.conv_heatmap(feature_heatmap)
            
            if self.is_final:
                return out_paf, out_heatmap
            else:
                out = torch.cat([out_paf,out_heatmap,features],1)
                return out, [out_paf, out_heatmap]
      
    class ModelOrigin(nn.Module):
        def __init__(self, backbone_net):
            super(ModelOrigin, self).__init__()
            self.backbone_net = backbone_net
            feature_channels = self.backbone_net.outplanes
            self.stage1 = Stage(feature_channels, feature_channels, cfg_stage_1['origin'])
            self.stage2 = Stage(self.stage1.outplanes, feature_channels, cfg_stage_n['origin'])
            self.stage3 = Stage(self.stage2.outplanes, feature_channels, cfg_stage_n['origin'])
            self.stage4 = Stage(self.stage3.outplanes, feature_channels, cfg_stage_n['origin'])
            self.stage5 = Stage(self.stage4.outplanes, feature_channels, cfg_stage_n['origin'])
            self.stage6 = Stage(self.stage5.outplanes, feature_channels, cfg_stage_n['origin'], is_final=True)
        
        def forward(self, x):
            features = self.backbone_net(x)
            out, outs1 = self.stage1(features, features)
            out, outs2 = self.stage2(out, features)
            out, outs3 = self.stage3(out, features)
            out, outs4 = self.stage4(out, features)
            out, outs5 = self.stage5(out, features)
            # final
            outs6 = self.stage6(out, features)
            return outs1,outs2,outs3,outs4,outs5,outs6
    
    def convert_official_model():
        import sys
        sys.path.insert(0, '/home/dalong/Workspace0/caffe103/python/')
        import caffe
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        import cv2
        import numpy as np

        init_net = caffe.Net('pose_deploy.prototxt', 'pose_iter_440000.caffemodel', caffe.TEST)
        init_dict_items = init_net.params.items()
        model = get_model()
        model_dict = model.state_dict()
        model_dict_items = model_dict.items()
        
        pretrained_dict = {}
        for i in range(0, 24):
            print model_dict_items[i][0], model_dict_items[i][1].shape
            print init_dict_items[int(i/2)][0], init_dict_items[int(i/2)][1][i%2].data.shape
            pretrained_dict[model_dict_items[i][0]] = torch.from_numpy(init_dict_items[int(i/2)][1][i%2].data)
        for i in range(24, 44):
            idxs = 12+np.array([0,0, 2,2, 4,4, 6,6, 8,8, 1,1, 3,3, 5,5, 7,7, 9,9])
            j = idxs[i-24]
            print model_dict_items[i][0], model_dict_items[i][1].shape 
            print init_dict_items[j][0], init_dict_items[j][1][i%2].data.shape
            pretrained_dict[model_dict_items[i][0]] = torch.from_numpy(init_dict_items[j][1][i%2].data)
        for k in range(0,5):
            for i in range(44+28*k, 44+28*(k+1)):
                idxs = 22+14*k+np.array([0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12,
                                    1,1, 3,3, 5,5, 7,7, 9,9, 11,11, 13,13])
                j = idxs[i-(44+28*k)]
                print model_dict_items[i][0], model_dict_items[i][1].shape 
                print init_dict_items[j][0], init_dict_items[j][1][i%2].data.shape
                pretrained_dict[model_dict_items[i][0]] = torch.from_numpy(init_dict_items[j][1][i%2].data)
        model.load_state_dict(pretrained_dict)
        torch.save(model.state_dict(), './convert/pose_iter_440000_pytorch.pkl')
    
    model = ModelOrigin(backbone_net)
    print '============= init with Openpose Official Model ================'
    model.load_state_dict(torch.load('./convert/pose_iter_440000_pytorch.pkl'))
    return model

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import cv2
    import numpy as np
    

    # init model
    model = get_model().cuda()

    # test simple
    # print 'forward'
    # input = Variable(torch.randn(1, 3, 368, 368)).cuda()
    # outputs = model(input)
    # print outputs[0].data.cpu().size(), outputs[1].data.cpu().size()

    
    # test img
    print 'forward'
    img = cv2.imread('./sample_image/upper.jpg')
    img = cv2.resize(img, (368,368))
    
    in_ = np.array(img, dtype=np.float32)/256.0
    in_ -= 0.5
    in_ = in_.transpose((2,0,1))
    in_ = in_[np.newaxis, :, :, :]
    
    input = Variable(torch.from_numpy(in_)).cuda()
    
    out_paf, out_heatmap = model(input)
    
    out_paf = out_paf.data.cpu().numpy()
    out_heatmap = out_heatmap.data.cpu().numpy()
    
    print np.max(out_paf), np.max(out_heatmap), np.min(out_paf), np.min(out_heatmap)
    
    paf = np.max(abs(out_paf[0]), axis = 0)
    heatmap = np.max(abs(out_heatmap[0]), axis = 0)
    cv2.imwrite('./sample_image/paf.jpg', np.uint8(paf*255))  
    cv2.imwrite('./sample_image/heatmap.jpg', np.uint8(heatmap*255))  

