import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from model.ESRGan.models import RRDBNet
from model.inference import load_image, srresnet_predict, esrgan_predict, srgan_predict


def grid_figure(img, srresnet, srgan_generator, esrgan_generator):
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
    hr_img, lr_img = load_image(img)

    # Nearest neighbour Upsampling
    nearest_img = lr_img.resize((hr_img.width, hr_img.height), Image.NEAREST)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet_predict(srresnet, lr_img)

    # Super-resolution (SR) with ESRGAN
    sr_img_esrgan = esrgan_predict(esrgan_generator, lr_img)

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_predict(srgan_generator, lr_img)

    # Create grid
    margin = 40
    N_cols = 3
    N_rows = 2
    grid_img = Image.new('RGB', (N_cols * hr_img.width + (N_cols+1) * margin,
                         N_rows * hr_img.height + (N_rows+1) * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("Keyboard.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place low resolution image
    x0 = margin
    y0 = margin
    grid_img.paste(nearest_img, (x0, y0))
    text_size = font.getsize("Nearest neighbour upsampling")
    draw.text(xy=[x0 + nearest_img.width / 2 - text_size[0] / 2, y0 -
                  text_size[1] - 5], text="Nearest neighbour upsampling", font=font, fill='black')

    # Place SRResNet image
    x0 = 2 * margin + sr_img_srresnet.width
    y0 = margin
    grid_img.paste(sr_img_srresnet, (x0, y0))
    text_size = font.getsize("SRResNet")
    draw.text(
        xy=[x0 + sr_img_srresnet.width / 2 -
            text_size[0] / 2, y0 - text_size[1] - 5],
        text="SRResNet", font=font, fill='black')

    # Place SRGAN image
    x0 = 3 * margin + sr_img_srresnet.width + sr_img_srresnet.width
    y0 = margin
    # 2 * margin + sr_img_srresnet.height))
    grid_img.paste(sr_img_srgan, (x0, y0))
    text_size = font.getsize("SRGAN")
    draw.text(
        xy=[x0 + bicubic_img.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5],
        text="SRGAN", font=font, fill='black')

    # Place bicubic-upsampled image
    x0 = margin
    y0 = 2 * margin + sr_img_srresnet.height
    grid_img.paste(bicubic_img, (x0, y0))
    text_size = font.getsize("Bicubic upsampling")
    draw.text(xy=[x0 + bicubic_img.width / 2 - text_size[0] / 2, y0 - text_size[1] - 5], text="Bicubic upsampling",
              font=font,
              fill='black')

    # Place ESRGAN image
    x0 = 2 * margin + sr_img_srresnet.width
    y0 = 2 * margin + sr_img_srresnet.height
    grid_img.paste(sr_img_esrgan, (x0, y0))
    text_size = font.getsize("ESRGAN")
    draw.text(
        xy=[x0 + sr_img_esrgan.width /
            2 - text_size[0] / 2, y0 - text_size[1] - 5],
        text="ESRGAN", font=font, fill='black')

    # Place original HR image
    x0 = 3*margin+sr_img_srresnet.width+sr_img_srresnet.width
    y0 = 2*margin+sr_img_srresnet.height
    grid_img.paste(hr_img, (x0, y0))
    text_size = font.getsize("Original HR")
    draw.text(xy=[x0 + sr_img_srresnet.width / 2 - text_size[0] / 2,
              y0 - text_size[1] - 1], text="Original HR", font=font, fill='black')

    # Display grid
    grid_img.show()

    return grid_img


@st.cache()
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model checkpoints
    srgan_checkpoint = "weights/checkpoint_srgan.pth.tar"
    srresnet_checkpoint = "weights/checkpoint_srresnet.pth.tar"
    esrgan_checkpoint = 'weights/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'

    # Load models
    srresnet = torch.load(srresnet_checkpoint, map_location=device)[
        'model'].to(device)
    srresnet.eval()

    srgan_generator = torch.load(srgan_checkpoint, map_location=device)[
        'generator'].to(device)
    srgan_generator.eval()

    esrgan_generator = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32)
    esrgan_generator.load_state_dict(torch.load(
        esrgan_checkpoint)['params'], strict=True)
    esrgan_generator.eval()
    esrgan_generator = esrgan_generator.to(device)

    return srresnet, srgan_generator, esrgan_generator
