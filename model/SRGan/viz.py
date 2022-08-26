from model.SRGan.utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import cv2


def visualize_sr(img, srresnet, srgan_generator, esrgan_generator, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.
    :param img: filepath of the HR iamge
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """
    srresnet = False

    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert("RGB")
    if halve:
        hr_img = hr_img.resize(
            (int(hr_img.width / 2), int(hr_img.height / 2)), Image.LANCZOS
        )
    lr_img = hr_img.resize(
        (int(hr_img.width / 4), int(hr_img.height / 4)), Image.BICUBIC
    )

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRResNet
    if False:
        sr_img_srresnet = srresnet(
            convert_image(lr_img, source="pil", target="imagenet-norm")
            .unsqueeze(0)
            .to(device)
        )
        sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
        sr_img_srresnet = convert_image(sr_img_srresnet, source="[-1, 1]", target="pil")
    else:
        sr_img_esrgan = np.asarray(lr_img).astype(np.float32) / 255.0
        sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_RGB2BGR)
        sr_img_esrgan = torch.from_numpy(
            np.transpose(sr_img_esrgan[:, :, [2, 1, 0]], (2, 0, 1))
        ).float()
        sr_img_esrgan = sr_img_esrgan.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            sr_img_esrgan = esrgan_generator(sr_img_esrgan)

        sr_img_esrgan = sr_img_esrgan.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        sr_img_esrgan = np.transpose(sr_img_esrgan[[2, 1, 0], :, :], (1, 2, 0))
        sr_img_esrgan = (sr_img_esrgan * 255.0).round().astype(np.uint8)
        sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_BGR2RGB)
        sr_img_esrgan = Image.fromarray(sr_img_esrgan)

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(
        convert_image(lr_img, source="pil", target="imagenet-norm")
        .unsqueeze(0)
        .to(device)
    )
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    print(sr_img_srgan.shape)
    sr_img_srgan = convert_image(sr_img_srgan, source="[-1, 1]", target="pil")

    # Create grid
    margin = 40
    grid_img = Image.new(
        "RGB",
        (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin),
        (255, 255, 255),
    )

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function."
        )
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(
        xy=[
            margin + bicubic_img.width / 2 - text_size[0] / 2,
            margin - text_size[1] - 5,
        ],
        text="Bicubic",
        font=font,
        fill="black",
    )

    # Place SRResNet/ESRGAN image
    sr_img_srresnet = sr_img_esrgan
    text = "SRResNet" if srresnet else "ESRGAN"
    grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
    text_size = font.getsize(text)
    draw.text(
        xy=[
            2 * margin
            + bicubic_img.width
            + sr_img_srresnet.width / 2
            - text_size[0] / 2,
            margin - text_size[1] - 5,
        ],
        text=text,
        font=font,
        fill="black",
    )

    # Place SRGAN image
    grid_img.paste(sr_img_srgan, (margin, 2 * margin + sr_img_srresnet.height))
    text_size = font.getsize("SRGAN")
    draw.text(
        xy=[
            margin + bicubic_img.width / 2 - text_size[0] / 2,
            2 * margin + sr_img_srresnet.height - text_size[1] - 5,
        ],
        text="SRGAN",
        font=font,
        fill="black",
    )

    # Place original HR image
    grid_img.paste(
        hr_img, (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height)
    )
    text_size = font.getsize("Original HR")
    draw.text(
        xy=[
            2 * margin
            + bicubic_img.width
            + sr_img_srresnet.width / 2
            - text_size[0] / 2,
            2 * margin + sr_img_srresnet.height - text_size[1] - 1,
        ],
        text="Original HR",
        font=font,
        fill="black",
    )

    # Display grid
    grid_img.show()

    return grid_img


def visualize_6(img, srresnet, srgan_generator, esrgan_generator, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.
    :param img: filepath of the HR iamge
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """

    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert("RGB")
    if halve:
        hr_img = hr_img.resize(
            (int(hr_img.width / 2), int(hr_img.height / 2)), Image.LANCZOS
        )
    lr_img = hr_img.resize(
        (int(hr_img.width / 4), int(hr_img.height / 4)), Image.BICUBIC
    )

    # Nearest neighbour Upsampling
    nearest_img = lr_img.resize((hr_img.width, hr_img.height), Image.NEAREST)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(
        convert_image(lr_img, source="pil", target="imagenet-norm")
        .unsqueeze(0)
        .to(device)
    )
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source="[-1, 1]", target="pil")

    # Super-resolution (SR) with ESRGAN
    sr_img_esrgan = np.asarray(lr_img).astype(np.float32) / 255.0
    sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_RGB2BGR)
    sr_img_esrgan = torch.from_numpy(
        np.transpose(sr_img_esrgan[:, :, [2, 1, 0]], (2, 0, 1))
    ).float()
    sr_img_esrgan = sr_img_esrgan.unsqueeze(0).to(device)
    # inference
    with torch.no_grad():
        sr_img_esrgan = esrgan_generator(sr_img_esrgan)

    sr_img_esrgan = sr_img_esrgan.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    sr_img_esrgan = np.transpose(sr_img_esrgan[[2, 1, 0], :, :], (1, 2, 0))
    sr_img_esrgan = (sr_img_esrgan * 255.0).round().astype(np.uint8)
    sr_img_esrgan = cv2.cvtColor(sr_img_esrgan, cv2.COLOR_BGR2RGB)
    sr_img_esrgan = Image.fromarray(sr_img_esrgan)

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(
        convert_image(lr_img, source="pil", target="imagenet-norm")
        .unsqueeze(0)
        .to(device)
    )
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    print(sr_img_srgan.shape)
    sr_img_srgan = convert_image(sr_img_srgan, source="[-1, 1]", target="pil")

    # Create grid
    margin = 40
    N_cols = 3
    N_rows = 2
    grid_img = Image.new(
        "RGB",
        (
            N_cols * hr_img.width + (N_cols + 1) * margin,
            N_rows * hr_img.height + (N_rows + 1) * margin,
        ),
        (255, 255, 255),
    )

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("Keyboard.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function."
        )
        font = ImageFont.load_default()

    # Place low resolution image
    x0 = margin
    y0 = margin
    if False:
        grid_img.paste(lr_img, (x0, y0))
        text_size = font.getsize("Low resolution")
        draw.text(
            xy=[x0 + lr_img.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
            text="Low resolution",
            font=font,
            fill="black",
        )
    else:
        grid_img.paste(nearest_img, (x0, y0))
        text_size = font.getsize("Nearest neighbour upsampling resolution")
        draw.text(
            xy=[x0 + nearest_img.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
            text="Nearest neighbour upsampling",
            font=font,
            fill="black",
        )

    # Place SRResNet image
    x0 = 2 * margin + sr_img_srresnet.width
    y0 = margin
    grid_img.paste(sr_img_srresnet, (x0, y0))
    text_size = font.getsize("SRResNet")
    draw.text(
        xy=[x0 + sr_img_srresnet.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
        text="SRResNet",
        font=font,
        fill="black",
    )

    # Place SRGAN image
    x0 = 3 * margin + sr_img_srresnet.width + sr_img_srresnet.width
    y0 = margin
    # 2 * margin + sr_img_srresnet.height))
    grid_img.paste(sr_img_srgan, (x0, y0))
    text_size = font.getsize("SRGAN")
    draw.text(
        xy=[x0 + bicubic_img.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
        text="SRGAN",
        font=font,
        fill="black",
    )

    # Place bicubic-upsampled image
    x0 = margin
    y0 = 2 * margin + sr_img_srresnet.height
    grid_img.paste(bicubic_img, (x0, y0))
    text_size = font.getsize("Bicubic upsampling")
    draw.text(
        xy=[x0 + bicubic_img.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
        text="Bicubic upsampling",
        font=font,
        fill="black",
    )

    # Place ESRGAN image
    x0 = 2 * margin + sr_img_srresnet.width
    y0 = 2 * margin + sr_img_srresnet.height
    grid_img.paste(sr_img_esrgan, (x0, y0))
    text_size = font.getsize("ESRGAN")
    draw.text(
        xy=[x0 + sr_img_esrgan.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
        text="ESRGAN",
        font=font,
        fill="black",
    )

    # Place original HR image
    x0 = 3 * margin + sr_img_srresnet.width + sr_img_srresnet.width
    y0 = 2 * margin + sr_img_srresnet.height
    grid_img.paste(hr_img, (x0, y0))
    text_size = font.getsize("Original HR")
    draw.text(
        xy=[x0 + sr_img_srresnet.width / 2 - text_size[0] / 2, y0 - text_size[1] - 1],
        text="Original HR",
        font=font,
        fill="black",
    )

    # Display grid
    grid_img.show()

    return grid_img
