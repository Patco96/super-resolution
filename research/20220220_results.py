# %%
import os
import time
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model.inference import load_image, srresnet_predict, esrgan_predict, srgan_predict
from model.viz import load_models, grid_figure
from PIL import Image


def open_image(img_path: str):
    img = Image.open(img_path, mode="r")
    size = img.size
    if size[0] % 4 != 0:
        print("Cropping image to " +
              str((0, 0, size[0]-size[0] % 4, size[1])))
        img = img.crop((0, 0, size[0]-size[0] % 4, size[1]))
    size = img.size
    if size[1] % 4 != 0:
        print("Cropping image to " +
              str((0, 0, size[0], size[1]-size[1] % 4)))
        img = img.crop((0, 0, size[0], size[1]-size[1] % 4))

    return img


srresnet, srgan_generator, esrgan_generator = load_models()


# %%
folder_path = "./test/random"
folder_path = "./data/SR_testing_datasets/BSDS100"
images = [folder_path + "/" +
          image for image in os.listdir(folder_path) if not image.startswith(".")]
original_imgs = [open_image(image) for image in images]
hr_imgs, lr_imgs = [], []
for original_img in original_imgs:
    hr_img, lr_img = load_image(original_img)
    hr_imgs.append(hr_img)
    lr_imgs.append(lr_img)

# %%
results = []

# Nearest neighbor
print("Nearest neighbor")
t0 = time.time()
nearest_imgs = [lr_img.resize(
    (hr_img.width, hr_img.height), Image.NEAREST) for lr_img, hr_img in zip(lr_imgs, hr_imgs)]
t1 = time.time()
results.append(
    {"Model": "Nearest neighbour upsampling", "Image": nearest_imgs, "Time": t1-t0})

# Bicubic
print("Bicubic")
t0 = time.time()
nearest_imgs = [lr_img.resize(
    (hr_img.width, hr_img.height), Image.BICUBIC) for lr_img, hr_img in zip(lr_imgs, hr_imgs)]
t1 = time.time()
results.append({"Model": "Bicubic upsampling",
                "Image": nearest_imgs, "Time": t1-t0})

# SRResNet
print("SRResNet")
t0 = time.time()
srresnet_imgs = [srresnet_predict(
    srresnet, lr_img) for lr_img in lr_imgs]
t1 = time.time()
results.append(
    {"Model": "SRResNet", "Image": srresnet_imgs, "Time": t1-t0})

# SRGAN
print("SRGAN")
t0 = time.time()
sr_imgs_srgan = [srgan_predict(srgan_generator, lr_img)
                 for lr_img in lr_imgs]
t1 = time.time()
results.append(
    {"Model": "SRGAN", "Image": sr_imgs_srgan, "Time": t1-t0})

# ESRGAN
print("ESRGAN")
t0 = time.time()
sr_imgs_esrgan = [esrgan_predict(
    esrgan_generator, lr_img) for lr_img in lr_imgs]
t1 = time.time()
results.append(
    {"Model": "ESRGAN", "Image": sr_imgs_esrgan, "Time": t1-t0})

df = pd.DataFrame(results)
df.drop(columns=["Image"], inplace=True)

print(df)


# %%


def get_ssim(sr_img, hr_img):
    return structural_similarity(
        np.array(hr_img), np.array(sr_img), data_range=255.0
    )


np_hr_img = np.array(hr_img)
for idx, result in enumerate(results):
    results[idx]["PSNR"] = []
    results[idx]["SSIM"] = []
    for img_idx, img in enumerate(result["Image"]):
        np_hr_img = np.array(hr_imgs[img_idx])
        np_sr_img = np.array(img)
        try:
            results[idx]["PSNR"].append(peak_signal_noise_ratio(
                np_hr_img, np_sr_img, data_range=255.0))
            results[idx]["SSIM"].append(structural_similarity(
                np_hr_img, np_sr_img, data_range=255.0, multichannel=True))
        except ValueError as e:
            print(results[idx]["Model"] + " has np_hr_img shape: " + str(
                np_hr_img.shape) + " and np_sr_img shape: " + str(np_sr_img.shape))

df = pd.DataFrame(results)

df
# %%
print(folder_path)
df['Mean PSNR'] = df['PSNR'].apply(lambda x: np.mean(x))
df['Mean SSIM'] = df['SSIM'].apply(lambda x: np.mean(x))
df
# %%

for img in images[2:4]:
    out = grid_figure(Image.open(img, mode='r'), srresnet,
                      srgan_generator, esrgan_generator)
    # out.save("results/" + img.replace(".jpg", "_out.png"))
out
# %%
