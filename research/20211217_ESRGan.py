# %%

import torch
import argparse
import cv2
import glob
import numpy as np
import os
import torch

from model.ESRGan.models import RRDBNet

model_path = 'weights/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'  # noqa: E501
input_path = 'data/SR_testing_datasets/BSDS100/'  # input test image folder
output_path = 'results/ESRGAN'  # output folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# set up model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32)
model.load_state_dict(torch.load(model_path)['params'], strict=True)
model.eval()
model = model.to(device)

os.makedirs(output_path, exist_ok=True)
for idx, path in enumerate(sorted(glob.glob(os.path.join(input_path, '*')))):
    imgname = os.path.splitext(os.path.basename(path))[0]
    print('Testing', idx, imgname)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(
        img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)
    # inference
    try:
        with torch.no_grad():
            output = model(img)
    except Exception as error:
        print('Error', error, imgname)
    else:
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f'{imgname}_ESRGAN.png'), output)

# %%
output
# %%
