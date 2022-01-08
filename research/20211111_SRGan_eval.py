# %%
from matplotlib import cm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from model.SRGan.utils import convert_image, AverageMeter, create_data_lists
from model.SRGan.datasets import SRDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# import model.SRGan.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = "data"
train_folders = os.path.join(data_folder, "SR_training_datasets/BSDS200")
test_folders = os.path.join(data_folder, "SR_testing_datasets/BSDS100")

create_data_lists(
    [train_folders], [test_folders], min_size=100, output_folder=data_folder
)

# %%


# Data
data_folder = "data/"
test_data_names = ["Set5", "Set14", "BSDS100"]
test_data_names = ["BSDS100"]

# Model checkpoints
srgan_checkpoint = "weights/checkpoint_srgan.pth.tar"
srresnet_checkpoint = "weights/checkpoint_srresnet.pth.tar"

# Load model, either the SRResNet or the SRGAN
# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet
# %%
# srgan_generator =
srgan_generator = torch.load(srgan_checkpoint, map_location=device)["generator"].to(
    device
)
srgan_generator.eval()
model = srgan_generator

# %%
# Evaluate

for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader
    test_dataset = SRDataset(
        data_folder,
        split="test",
        crop_size=0,
        scaling_factor=4,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
        test_data_name=test_data_name,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            # Move to default device
            lr_imgs = lr_imgs.to(
                device
            )  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            # (batch_size (1), 3, w, h), in [-1, 1]
            hr_imgs = hr_imgs.to(device)

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(
                sr_imgs, source="[-1, 1]", target="y-channel"
            ).squeeze(
                0
            )  # (w, h), in y-channel
            hr_imgs_y = convert_image(
                hr_imgs, source="[-1, 1]", target="y-channel"
            ).squeeze(
                0
            )  # (w, h), in y-channel
            psnr = peak_signal_noise_ratio(
                hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.0
            )
            ssim = structural_similarity(
                hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.0
            )
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

    # Print average PSNR and SSIM
    print("PSNR - {psnrs.avg:.3f}".format(psnrs=PSNRs))
    print("SSIM - {ssims.avg:.3f}".format(ssims=SSIMs))

print("\n")
# %%

imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

for lr_img, sr_img, hr_img in zip(lr_imgs, sr_imgs, hr_imgs):
    lr_img = (lr_img * imagenet_std.to(device)) + imagenet_mean.to(device)
    lr_img = lr_img.squeeze(0).cpu().numpy()
    sr_img = sr_img.squeeze(0).cpu().numpy()
    hr_img = hr_img.squeeze(0).cpu().numpy()
    sr_img = convert_image(
        sr_img.reshape(sr_img.shape[1], sr_img.shape[2], sr_img.shape[0]),
        source="[-1, 1]",
        target="pil",
    )
    hr_img = convert_image(hr_img, source="[-1, 1]", target="pil")
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(
        lr_img.reshape(lr_img.shape[1], lr_img.shape[2], lr_img.shape[0]), cmap="gray"
    )
    ax[0].set_title("LR")
    ax[1].imshow(
        sr_img.reshape(sr_img.shape[1], sr_img.shape[2], sr_img.shape[0]), cmap="gray"
    )
    ax[1].set_title("SR")
    ax[2].imshow(
        hr_img.reshape(hr_img.shape[1], hr_img.shape[2], hr_img.shape[0]), cmap="gray"
    )
    ax[2].set_title("HR")

# %%

name = []
for image, img_name in zip([lr_img, sr_img, hr_img], ["LR", "SR", "HR"]):
    img = image.reshape(image.shape[1], image.shape[2], image.shape[0]) * 255
    pil_img = Image.fromarray(np.uint8(img))
    # pil_img.save(f"{img_name}.png")
# %%
pil_img
# %%
image = hr_img
img = image.reshape(image.shape[1], image.shape[2], image.shape[0]) * 255
Image.fromarray(np.uint8(img))

# .save("LR.png")
# %%

imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader
    test_dataset = SRDataset(
        data_folder,
        split="test",
        crop_size=0,
        scaling_factor=4,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
        test_data_name=test_data_name,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            if i != 0:
                continue
            # Move to default device
            lr_imgs = lr_imgs.to(
                device
            )  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            # (batch_size (1), 3, w, h), in [-1, 1]
            hr_imgs = hr_imgs.to(device)

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]
            for lr_img, hr_img, sr_img in zip(lr_imgs, hr_imgs, sr_imgs):
                lr_img = lr_img * imagenet_std + imagenet_mean
                lr_img = convert_image(lr_img, source="[0, 1]", target="pil")
                hr_img = convert_image(hr_img, source="[-1, 1]", target="pil")
                sr_img = convert_image(sr_img, source="[-1, 1]", target="pil")
                lr_img.save(f"{test_data_name}_LR.png")
                hr_img.save(f"{test_data_name}_HR.png")
                sr_img.save(f"{test_data_name}_SR.png")


# %%
