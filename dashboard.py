from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model.SRGan.utils import convert_image
from model.inference import load_image, srresnet_predict, esrgan_predict, srgan_predict
from model.viz import load_models
from pathlib import Path
from PIL import Image
import os
import time
# import torch
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")


@st.experimental_singleton()
def get_models():
    return load_models()


def format_func(s: str):
    return s.split(".")[0]


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


srresnet, srgan_generator, esrgan_generator = get_models()


# srgan_checkpoint = "checkpoint_srgan_animals_81.pth.tar"
# device = "cpu"
# srgan_generator_animals = torch.load(srgan_checkpoint, map_location=device)[
#     'generator'].to(device)
# srgan_generator_animals.eval()

st.title("Super-Resolution")

show_comparisson = st.checkbox("Show Comparisson", False)

if show_comparisson:

    folders = os.listdir(Path(__file__).parents[0] / "test/")
    selected_folder = st.selectbox("Select a dataset", folders)
    folder_path = Path(__file__).parents[0] / "test/" / selected_folder
    images = [str(folder_path / image)
              for image in os.listdir(folder_path) if not image.startswith(".")]
    with st.expander("See images"):
        st.image(images, width=100)

    original_imgs = [Image.open(image) for image in images]
    hr_imgs, lr_imgs = [], []
    for original_img in original_imgs:
        hr_img, lr_img = load_image(original_img)
        hr_imgs.append(hr_img)
        lr_imgs.append(lr_img)

    clicked = st.button("Predict")

    if clicked:
        results = []

        # Nearest neighbor
        t0 = time.time()
        nearest_imgs = [lr_img.resize(
            (hr_img.width, hr_img.height), Image.NEAREST) for lr_img in lr_imgs]
        t1 = time.time()
        results.append(
            {"Model": "Nearest neighbour upsampling", "Image": nearest_imgs, "Time": t1-t0})

        # Bicubic
        t0 = time.time()
        nearest_imgs = [lr_img.resize(
            (hr_img.width, hr_img.height), Image.BICUBIC) for lr_img in lr_imgs]
        t1 = time.time()
        results.append({"Model": "Bicubic upsampling",
                        "Image": nearest_imgs, "Time": t1-t0})

        # SRResNet
        t0 = time.time()
        srresnet_imgs = [srresnet_predict(
            srresnet, lr_img) for lr_img in lr_imgs]
        t1 = time.time()
        results.append(
            {"Model": "SRResNet", "Image": srresnet_imgs, "Time": t1-t0})

        # SRGAN
        t0 = time.time()
        sr_imgs_srgan = [srgan_predict(srgan_generator, lr_img)
                         for lr_img in lr_imgs]
        t1 = time.time()
        results.append(
            {"Model": "SRGAN", "Image": sr_imgs_srgan, "Time": t1-t0})

        # ESRGAN
        t0 = time.time()
        sr_imgs_esrgan = [esrgan_predict(
            esrgan_generator, lr_img) for lr_img in lr_imgs]
        t1 = time.time()
        results.append(
            {"Model": "ESRGAN", "Image": sr_imgs_esrgan, "Time": t1-t0})

        df = pd.DataFrame(results)
        df.drop(columns=["Image"], inplace=True)

        st.table(df)


else:
    select_sample = st.checkbox("Select a sample", True)

    results = []

    if select_sample:
        samples = os.listdir(Path(__file__).parents[0] / "test/samples/")

        options = [f.split("/")[-1] for f in samples if not f.startswith(".")]

        # Show samples
        selected_image = st.selectbox("Select a sample", options,
                                      format_func=format_func)
        original_img = f"test/samples/{selected_image}"
    else:
        original_img = st.file_uploader(
            "Upload image", type=["jpg", "png", "jpeg"])

    if original_img:

        original_img = open_image(original_img)

        # st.image(original_img, caption="Original image", use_column_width=True)

        col1, col2 = st.columns(2)

        hr_img, lr_img = load_image(original_img)
        with col1:
            st.write("Nearest neighbour upsampling")
            t0 = time.time()
            nearest_img = lr_img.resize(
                (hr_img.width, hr_img.height), Image.NEAREST)
            t1 = time.time()
            st.image(nearest_img, use_column_width=True)
            results.append(
                {"Model": "Nearest neighbour upsampling", "Image": nearest_img, "Time": t1-t0})

            st.write("Bicubic upsampling")
            t0 = time.time()
            nearest_img = lr_img.resize(
                (hr_img.width, hr_img.height), Image.BICUBIC)
            t1 = time.time()
            st.image(nearest_img, use_column_width=True)
            results.append(
                {"Model": "Bicubic upsampling", "Image": nearest_img, "Time": t1-t0})

            st.write("SRResNet")
            t0 = time.time()
            srresnet_img = srresnet_predict(srresnet, lr_img)
            t1 = time.time()
            st.image(srresnet_img, use_column_width=True)
            results.append(
                {"Model": "SRResNet", "Image": srresnet_img, "Time": t1-t0})

        with col2:

            st.write("High resolution")
            st.image(hr_img, use_column_width=True)
            results.append({"Model": "High resolution",
                            "Image": hr_img, "Time": 0})

            st.write("SRGAN")
            with st.spinner("Generating SRGAN image..."):
                t0 = time.time()
                sr_img_srgan = srgan_predict(srgan_generator, lr_img)
                t1 = time.time()
            st.image(sr_img_srgan, use_column_width=True)
            results.append(
                {"Model": "SRGAN", "Image": sr_img_srgan, "Time": t1-t0})

            # st.write("SRGAN animals")
            # with st.spinner("Generating SRGAN animals image..."):
            #     sr_img_srgan = srgan_predict(srgan_generator_animals, lr_img)
            # st.image(sr_img_srgan, use_column_width=True)

            st.write("ESRGAN")
            with st.spinner("Generating ESRGAN image..."):
                t0 = time.time()
                sr_img_esrgan = esrgan_predict(esrgan_generator, lr_img)
                t1 = time.time()
            st.image(sr_img_esrgan, use_column_width=True)
            results.append(
                {"Model": "ESRGAN", "Image": sr_img_esrgan, "Time": t1-t0})

        st.header("Metrics")

        def get_ssim(sr_img, hr_img):
            return structural_similarity(
                np.array(hr_img), np.array(sr_img), data_range=255.0
            )

        np_hr_img = np.array(hr_img)
        for idx, result in enumerate(results):
            np_sr_img = np.array(result["Image"])
            try:
                results[idx]["PSNR"] = peak_signal_noise_ratio(
                    np_hr_img, np_sr_img, data_range=255.0)
                results[idx]["SSIM"] = structural_similarity(
                    np_hr_img, np_sr_img, data_range=255.0, multichannel=True)
            except ValueError as e:
                st.write(results[idx]["Model"] + " has np_hr_img shape: " + str(
                    np_hr_img.shape) + " and np_sr_img shape: " + str(np_sr_img.shape))

        df = pd.DataFrame(results)
        df.drop(columns=["Image"], inplace=True)

        st.table(df)
