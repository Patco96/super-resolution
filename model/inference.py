from model.SRGan.utils import *
from PIL import Image
import numpy as np
import cv2


def load_image(hr_img, halve=False):
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)

    return hr_img, lr_img


def srresnet_predict(srresnet, lr_img):
    '''Super-resolution (SR) with SRResNet'''
    sr_img_srresnet = srresnet(convert_image(
        lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(
        sr_img_srresnet, source='[-1, 1]', target='pil')
    return sr_img_srresnet


def esrgan_predict(esrgan_generator, lr_img):
    '''Super-resolution (SR) with ESRGAN'''
    sr_img_esrgan = np.asarray(lr_img).astype(np.float32) / 255.
    sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_RGB2BGR)
    sr_img_esrgan = torch.from_numpy(np.transpose(
        sr_img_esrgan[:, :, [2, 1, 0]], (2, 0, 1))).float()
    sr_img_esrgan = sr_img_esrgan.unsqueeze(0).to(device)
    # inference
    with torch.no_grad():
        sr_img_esrgan = esrgan_generator(sr_img_esrgan)

    sr_img_esrgan = sr_img_esrgan.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    sr_img_esrgan = np.transpose(sr_img_esrgan[[2, 1, 0], :, :], (1, 2, 0))
    sr_img_esrgan = (sr_img_esrgan * 255.0).round().astype(np.uint8)
    sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_BGR2RGB)
    sr_img_esrgan = Image.fromarray(sr_img_esrgan)
    return sr_img_esrgan


def srgan_predict(srgan_generator, lr_img):
    '''Super-resolution (SR) with SRGAN'''
    sr_img_srgan = srgan_generator(convert_image(
        lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    return sr_img_srgan
