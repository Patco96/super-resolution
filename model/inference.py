import cv2
import torch
import numpy as np
from model.SRGan.utils import convert_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(hr_img, halve=False):
    hr_img = hr_img.convert("RGB")
    if halve:
        hr_img = hr_img.resize(
            (int(hr_img.width / 2), int(hr_img.height / 2)), Image.LANCZOS
        )
    lr_img = hr_img.resize(
        (int(hr_img.width / 4), int(hr_img.height / 4)), Image.BICUBIC
    )

    return hr_img, lr_img


def srresnet_predict(srresnet, lr_img):
    """
    Super-resolution (SR) with SRResNet
    PIL -> ImageNet -> SRResNet -> [-1, 1] -> PIL
    """
    # PIL to ImageNet
    lr_img = (
        convert_image(lr_img, source="pil", target="imagenet-norm")
        .unsqueeze(0)
        .to(device)
    )
    # Inference
    sr_img_srresnet = srresnet(lr_img)
    # Squeeze
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    # To PIL
    sr_img_srresnet = convert_image(sr_img_srresnet, source="[-1, 1]", target="pil")
    return sr_img_srresnet


def esrgan_predict(esrgan_generator, lr_img):
    """
    Super-resolution (SR) with ESRGAN
    PIL -> Numpy -> [0,1] -> BGR -> Transpose -> Tensor ->  -> ESRGAN -> BGR[0, 1] ->  -> PIL
    """

    # PIL to OpenCV (RGB yet)
    sr_img_esrgan = np.asarray(lr_img).astype(np.float32) / 255.0
    # To BGR
    sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_RGB2BGR)
    # To Tensor
    # sr_img_esrgan = torch.from_numpy(sr_img_esrgan).float()
    sr_img_esrgan = torch.from_numpy(
        np.transpose(sr_img_esrgan[:, :, [2, 1, 0]], (2, 0, 1))
    ).float()
    sr_img_esrgan = sr_img_esrgan.unsqueeze(0).to(device)
    # Inference
    with torch.no_grad():
        sr_img_esrgan = esrgan_generator(sr_img_esrgan)
    # Squeeze and clamp
    sr_img_esrgan = sr_img_esrgan.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # IDK
    sr_img_esrgan = np.transpose(sr_img_esrgan[[2, 1, 0], :, :], (1, 2, 0))
    # To [0, 255]
    sr_img_esrgan = (sr_img_esrgan * 255.0).round().astype(np.uint8)
    # To RGB
    sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_BGR2RGB)
    # To PIL
    sr_img_esrgan = Image.fromarray(sr_img_esrgan)
    return sr_img_esrgan


def srgan_predict(srgan_generator, lr_img):
    """
    Super-resolution (SR) with SRGAN
    PIL -> ImageNet -> SRGAN -> [-1, 1] -> PIL
    """

    # PIL to ImageNet
    lr_img = (
        convert_image(lr_img, source="pil", target="imagenet-norm")
        .unsqueeze(0)
        .to(device)
    )
    # Inference
    sr_img_srgan = srgan_generator(lr_img)
    # Squeeze
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    # To PIL
    sr_img_srgan = convert_image(sr_img_srgan, source="[-1, 1]", target="pil")
    return sr_img_srgan
