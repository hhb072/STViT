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
import cv2
import numpy as np

@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)        

def naive_sparse_bmm(sparse_mat, dense_mat, transpose=False):
    if transpose:
        return torch.stack([torch.sparse.mm(s_mat, d_mat.t()) for s_mat, d_mat in zip(sparse_mat, dense_mat)], 0)
    else:
        return torch.stack([torch.sparse.mm(s_mat, d_mat) for s_mat, d_mat in zip(sparse_mat, dense_mat)], 0)
        
def sparse_permute(sparse_mat, order):
    values = sparse_mat.coalesce().values()
    indices = sparse_mat.coalesce().indices()
    indices = torch.stack([indices[o] for o in order], 0).contiguous()
    return torch.sparse_coo_tensor(indices, values)

def make_abs_indices(affinity_matrix, num_spixels_height=7, num_spixels_width=7):
    with torch.no_grad():
        batchsize, _, height, width = affinity_matrix.shape
        device = affinity_matrix.device
        # num_spixels_height, num_spixels_width = height//kernel_size[0], width//kernel_size[1]
        num_spixels =  num_spixels_height*num_spixels_width
    
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, num_spixels_height, num_spixels_width).float()
        init_label_map = F.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)
        init_label_map = init_label_map.reshape(batchsize, -1)
        
        abs_indices = get_abs_indices(init_label_map, num_spixels_width)
        mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
        abs_indices = abs_indices[:, mask]
        return abs_indices, mask

test_image='data/5.jpg'

model = svit_small()
checkpoint = torch.load('ckpt/best.pth', map_location='cpu')    
model.load_state_dict(checkpoint['model'], strict=False)
# torch.save(model, 'svit-small-83.6')
# exit()

model = model.cuda()
model.eval()

# spixel_maps = torch.Tensor(get_random_colors(49)).t().float().div(255).reshape(1, 3, 7, 7).cuda()
spixel_maps = torch.ones(1, 1, 7, 7).cuda()

# image_paths = os.listdir('data')
image_paths = ['10.jpg']
for path in image_paths:
    image = Image.open('data/'+path)
    save_dir =  f'results/{path[:-4]}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # org_img = cv2.imread('data/'+path, flags=1) 

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

    spixel_features = F.unfold(spixel_maps, kernel_size=3, stride=1, padding=1) # (1, 1*9, 7*7)
    # spixel_features = spixel_features.reshape(3, 9, 49).permute(2, 1, 0) # (49, 9, 3)
    spixel_features = spixel_features.reshape(9, 49)
    for idx, (affinity_matrix, at) in enumerate(zip(affs, atts)):
        if affinity_matrix is None:
            continue
        # af: 1, 9, 56, 56
        # at: 1, heads, 49, 49
        _, _, h, w = affinity_matrix.shape
               
        abs_indices, mask = make_abs_indices(affinity_matrix)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)
        sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices, reshaped_affinity_matrix[mask]) # (B, m, N)
        permuted_assignment = sparse_permute(sparse_abs_affinity, (0, 2, 1)) # (B, N, m)
        
        output = []
        for index in range(at.shape[1]):
            y = naive_sparse_bmm(permuted_assignment, at[:, index]) # (N, m)
            y = naive_sparse_bmm(permuted_assignment, y.transpose(-1, -2))
            # y = y.transpose(-1, -2) # (1, N, N)
            y = y/(y.max(dim=-1, keepdim=True)[0]+1e-12)
            y = y.reshape(h*w, 1, h, w)
            vutils.save_image(y.data.cpu(), f'{save_dir}/layer{idx}_head{index}.png', nrow=w)
            
            # save per image
            save_dir2 =  f'results/{path[:-4]}/layer{idx}_head{index}'
            Path(save_dir2).mkdir(parents=True, exist_ok=True)
            y = y.reshape(h, w, h, w)
            
            
            i = 10
            j = 10
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    gray_img = (y[i,j] * 255).cpu().numpy().astype(np.uint8)
                    heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
                    heat_img = cv2.resize(heat_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                    # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)                    
                    
                    org_img = (image0.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB) 
                    
                    add_img = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)
                    
                    ratio = 224//h
                    # cv2.circle(add_img,(j*(224//w),i*(224//h)),224//h,(255,0,255))
                    cv2.rectangle(add_img,(j*ratio,i*ratio),(j*ratio+ratio,i*ratio+ratio),(0,255,0),1)
                    # cv2.rectangle(org_img,(j*ratio,i*ratio),(j*ratio+ratio,i*ratio+ratio),(0,255,0),2)
                    
                    cv2.imwrite(f'{save_dir2}/heatmap{i}_{j}.png', heat_img)
                    cv2.imwrite(f'{save_dir2}/add_img{i}_{j}.png', add_img)
                    # cv2.imwrite(f'{save_dir2}/org_img{i}_{j}.png', org_img)   
                
                
            
            
            
    

                   

