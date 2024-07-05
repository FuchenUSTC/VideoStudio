import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image

from .data_loader import RescaleClean
from .data_loader import ToTensorClean

from .u2net import U2NET # full size version 173.6 MB



def convert_pred_to_fg_bg(image, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    
    fg_imo_np = (np.array(imo) >= 128).astype(np.float32)
    bg_imo_np = (np.array(imo) < 128).astype(np.float32)
    
    fg_image = (image/255).astype(np.float32) * fg_imo_np
    bg_image = (image/255).astype(np.float32) * bg_imo_np
    
    
    fg_image_rgb = Image.fromarray((fg_image*255).astype(np.uint8)).convert('RGB')
    bg_image_rgb = Image.fromarray((bg_image*255).astype(np.uint8)).convert('RGB')
    
    return fg_image_rgb, bg_image_rgb
     

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


class U2NetSegmentationProcessor(object):
    def __init__(self, u2net_pth, device):
        self.image_transform=transforms.Compose([RescaleClean(320), ToTensorClean(flag=0)])
        self.u2net = U2NET(3,1)
        self.u2net.load_state_dict(torch.load(u2net_pth))
        self.u2net.to(device)
        self.u2net.eval()
        self.device = device
    
    def obtain_fg_bg_pil(self, img):
        img_trans = self.image_transform(img)
        inputs_test = img_trans.type(torch.FloatTensor)
        inputs_test = inputs_test.unsqueeze(0)
        inputs_test = Variable(inputs_test.to(self.device))
        with torch.no_grad():
            d1,_,_,_,_,_,_= self.u2net(inputs_test)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        fg_image_rgb, bg_image_rgb = convert_pred_to_fg_bg(np.array(img), pred)
        return fg_image_rgb, bg_image_rgb
        
        

        