# %%
import start  # noqa
from model.SRGan.viz import visualize_sr, visualize_6
from model.viz import grid_figure

# %%
from PIL import Image
import torch
from model.ESRGan.models import RRDBNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "weights/checkpoint_srgan.pth.tar"
srresnet_checkpoint = "weights/checkpoint_srresnet.pth.tar"
esrgan_checkpoint = "weights/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"

# Load models
srresnet = torch.load(srresnet_checkpoint, map_location=device)["model"].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint, map_location=device)["generator"].to(
    device
)
srgan_generator.eval()

esrgan_generator = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
)
esrgan_generator.load_state_dict(torch.load(esrgan_checkpoint)["params"], strict=True)
esrgan_generator.eval()
esrgan_generator = esrgan_generator.to(device)

# %%

IMG_PATH = "data/val/wild/flickr_wild_001119.jpg"

img = visualize_6(IMG_PATH, srresnet, srgan_generator, esrgan_generator)
img
# %%
img = visualize_sr(IMG_PATH, srresnet, srgan_generator, esrgan_generator)
img
# %%
# 295087, 300091
img = grid_figure(
    Image.open(IMG_PATH, mode="r"), srresnet, srgan_generator, esrgan_generator
)


# %%
