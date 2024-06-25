import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image, ImageOps
from models.svit import svit_small
import torchvision.utils as vutils
import torch.nn.functional as F
import random
from random_color import get_random_colors
import os
from pathlib import Path
# from colourmap import colourmap
import numpy as np
import cv2


model = svit_small()
checkpoint = torch.load('ckpt/best.pth', map_location='cpu')    
model.load_state_dict(checkpoint['model'], strict=True)
# torch.save(model, 'svit-small-83.6')
# exit()

model = model.cuda()
model.eval()

spixel_maps = torch.Tensor(get_random_colors(49)).t().float().div(255).reshape(1, 3, 7, 7).cuda() * 0.7 + 0.2
# spixel_maps = torch.Tensor([(165, 184, 213), (209, 131, 6), (209, 166, 168), (2, 220, 224), (127, 163, 117), (111, 2, 154), (237, 94, 231), (105, 252, 90), (66, 87, 167), (163, 85, 65), (56, 158, 13), (229, 184, 238), (116, 249, 248), (21, 15, 249), (246, 11, 161), (157, 149, 116), (246, 216, 82), (135, 171, 159), (158, 19, 46), (50, 171, 101), (155, 97, 147), (222, 242, 175), (152, 127, 244), (10, 44, 157), (250, 92, 112), (172, 242, 22), (225, 180, 127), (89, 164, 143), (164, 185, 89), (172, 40, 159), (179, 253, 231), (36, 143, 158), (139, 234, 145), (7, 176, 69), (174, 150, 40), (129, 59, 211), (113, 80, 160), (140, 119, 159), (85, 173, 63), (36, 240, 25), (250, 175, 216), (164, 113, 104), (170, 17, 224), (169, 156, 220), (149, 199, 254), (48, 115, 222), (114, 142, 154), (165, 130, 148), (180, 230, 179)]).t().float().div(255).reshape(1, 3, 7, 7).cuda()

# image_paths = os.listdir('data')
image_paths = ['11.jpg']
for path in image_paths:
    image = Image.open('data/'+path)
    save_dir =  f'results/{path[:-4]}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    

    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform0 = transforms.Compose([
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),])

    image0 = transform0(image)
              
    vutils.save_image(image0.unsqueeze(0).data.cpu(), f'{save_dir}/img.png')
    # spixel_maps = torch.rand(1, 3, 7, 7).mul(255).round().float().div(255).cuda()
    # spixel_maps = F.adaptive_avg_pool2d(image.unsqueeze(0), (7, 7)).cuda()
    
    vutils.save_image(F.interpolate(spixel_maps, (56, 56), mode='nearest').data.cpu(), f'{save_dir}/map.png')                    
                    
    transform1 = transforms.Compose([
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])                
                    
    image = transform1(image0).cuda()
    image0 = image0.cuda()

    image = image.unsqueeze(0)



    with torch.no_grad():
        y, atts, affs  = model(image)
        
    print(f'Class: {y.argmax(dim=1).flatten(0).cpu().item()}')

    spixel_num = 49

    spixel_features = F.unfold(spixel_maps, kernel_size=3, stride=1, padding=1) # (1, 3*9, 7*7)
    spixel_features = spixel_features.reshape(3, 9, 49).permute(2, 1, 0) # (49, 9, 3)
    for idx, a in enumerate(affs):
        if a is None:
            continue
        #  a: 1, 9, 56, 56
        _, _, h, w = a.shape
        
                
        # a = F.gumbel_softmax(a, dim=1, tau=0.001, hard=True)
        # a = F.softmax(a, dim=1)
        
        # print(a.max(dim=1)[0])
        
        a = F.one_hot(torch.argmax(a, dim=1), 9).permute(0, 3, 1, 2).float()
                
        a = a.reshape(1, 9, 7, h//7, 7, w//7).transpose(3, 4).reshape(9, 49, h*w//49).permute(1, 2, 0)#(49, 64, 9)
        y = a @ spixel_features # (49, 64, 3)
        y = y.permute(2, 0, 1).reshape(1, 3, 7, 7, h//7, w//7).transpose(3, 4).reshape(1, 3, h, w)
        
        # y = F.interpolate(y, image0.unsqueeze(0).shape[2:])  + image0.unsqueeze(0) 
        
        # vutils.save_image(y.data.cpu(), f'{save_dir}/f{idx}.png', normalize=True)
        
        org_img = ((image0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        
        y = F.interpolate(y, image0.unsqueeze(0).shape[2:])
        # y = (y[:,:,:-1,:-1] - y[:,:,1:,:-1]).abs() + (y[:,:,:-1,:-1] - y[:,:,:-1,1:]).abs()
        # y = (y.gt(0)).float()
        
        stoken_img = (y.squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        stoken_img = cv2.cvtColor(stoken_img, cv2.COLOR_BGR2RGB)
        
        add_img = cv2.addWeighted(org_img, 0.7, stoken_img, 0.3, 0)
        cv2.imwrite(f'{save_dir}/stoken{idx}.jpg', add_img)
        
        map_img = (F.interpolate(spixel_maps, image0.unsqueeze(0).shape[2:]).squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        add_img = cv2.addWeighted(org_img, 0.7, map_img, 0.3, 0)
        cv2.imwrite(f'{save_dir}/map{idx}.jpg', add_img)
        
    
    

                   

