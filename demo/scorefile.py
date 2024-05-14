import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.ASpanFormer.aspanformer import ASpanFormer 
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import demo_utils 

import cv2
import torch
import numpy as np

config_path = '/ml-aspanformer/configs/aspan/outdoor/aspan_test.py'
weights_path = './weights/indoor.ckpt'
long_dim0 = 512 
long_dim1 = 512 


channels = 2
rows = 256 # experimentation 
cols = 512

def outputOfModel(model , classifier , img0_path , img1_path ):
    
    img0_g,img1_g=cv2.imread(img0_path,0),cv2.imread(img1_path,0)
    img0_g,img1_g=demo_utils.resize(img0_g,long_dim0),demo_utils.resize(img1_g,long_dim1)
    
    data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
          'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()} 
    with torch.no_grad():   
        model(data,online_resize=True)
        corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()
    
    input_tensor = torch.zeros(channels, rows, cols)

    
    for i in range(len(corr0)):
        try:
            input_tensor[0 ,int(corr0[i][0]) , int(corr0[i][1])] = corr1[i][0]+1 
            input_tensor[1 ,int(corr0[i][0]) , int(corr0[i][1])] = corr1[i][1]+1 
        except: 
            pass 
     
    # input_to_Mymodule.append((input_tensor, label))
    input_tensor = input_tensor.unsqueeze(0)
    out = classifier(input_tensor) 
    _, predicted = torch.max(out.data, 1)
    predicted_label = predicted.tolist()[0]
    return predicted_label



config = get_cfg_defaults()
config.merge_from_file(config_path)
_config = lower_config(config)
matcher = ASpanFormer(config=_config['aspan'])
state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
matcher.load_state_dict(state_dict,strict=False)
matcher.cuda(),matcher.eval()
classifier = torch.load("./model_13_05_2024.pth")


# to get the output of the model just call this function 

# result = outputOfModel(model , classifier , img0_path , img1_path )  # i.e reult gives 0/1 