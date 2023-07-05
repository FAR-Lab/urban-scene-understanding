import sys, os 


import glob 
import cv2 
import torch 
import torch.backends.cudnn as cudnn 
from numpy import random 
import copy 

sys.path.append('/share/ju/xcpedestrian/urban-scene-understanding/SAMAdapter/models')
import models as samadapter 
print(samadapter)
import yaml 
import matplotlib.pyplot as plt 
import numpy as np 

from audio_processing import * 
from corrections_filters import * 

from torchvision import transforms 
import subprocess 



def load_sam_config(config_path): 
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
        return config 

def init_sam_model(weights_path, config): 
    model = samadapter.make(config['model']).cuda()
    sam_checkpoint = torch.load(weights_path, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    model.eval()
    
    return model 

def generate_shadow_mask(model, frame): 
    BATCH_SIZE = 1
    IMG_SIZE = (1024, 1024)
    TRANSFORM_IMG = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             #std=[0.229, 0.224, 0.225] )
        ])

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():

            frame_copy = copy.deepcopy(frame) 
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            #frame_copy = shadow_correction(frame_copy, 1, 1, 20, 0.001,0.001,1, 0)
            image_tensor = TRANSFORM_IMG(frame_copy)
            image_tensor = image_tensor.unsqueeze(0) 
            test_input = image_tensor.to(device)

            pred_test = torch.sigmoid(model.infer(test_input))[0]
            kernel = np.ones((10,10),np.uint8)
            denoised_mask = cv2.morphologyEx(pred_test[0].cpu().numpy(), cv2.MORPH_OPEN, kernel)
            denoised_mask = cv2.resize(denoised_mask, (1280,1280))
            
            num_images = len(glob.glob("./denoised_mask*"))
            #(denoised_mask)

            denoised_mask_img  = (denoised_mask * 255).astype(np.uint8)
            denoised_mask_img = cv2.cvtColor(denoised_mask_img, cv2.COLOR_GRAY2BGR)
            #print(denoised_mask_img)

            #cv2.imwrite(f"denoised_mask_{num_images}.png", denoised_mask_img)

            return denoised_mask


def shadow_deweighting(xyxy, shadow_mask, xyxy2xywh, gn):
     # SHADOW DEWEIGHTING 
        #print((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
        coords = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
        coords = list(map(lambda x: x * 1280, coords))
        #print(coords)
        
        
        shadow_mask_weights = shadow_mask[int(coords[1]):int(coords[1]+coords[3]), int(coords[0]):int(coords[0]+coords[2])]
        #denoised_mask_img  = (shadow_mask * 255).astype(np.uint8)

        #denoised_mask_img[int(coords[1]):int(coords[1]+coords[3]), int(coords[0]):int(coords[0]+coords[2])] = 100
        #denoised_mask_img = cv2.cvtColor(denoised_mask_img, cv2.COLOR_GRAY2BGR)
       
        

        #num_images = len(glob.glob("./denoised_mask*"))
        #cv2.imwrite(f"denoised_mask_{num_images}.png", denoised_mask_img)

        # take the average of every value in shadow_mask_weights 
        shadow_mask_avg = np.mean(shadow_mask_weights)
        #print(shadow_mask_avg)

        return shadow_mask_avg 

        