import os
import time
# import torch
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from pathlib import Path

from model.viz import load_models
from model.inference import load_image, srresnet_predict, esrgan_predict, srgan_predict
from model.SRGan.utils import convert_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@st.experimental_singleton()
def get_models():
    return load_models()


def format_func(s: str):
    return s.split(".")[0]


srresnet, srgan_generator, esrgan_generator = get_models()


# srgan_checkpoint = "checkpoint_srgan_animals_81.pth.tar"
# device = "cpu"
# srgan_generator_animals = torch.load(srgan_checkpoint, map_location=device)[
#     'generator'].to(device)
# srgan_generator_animals.eval()

st.title("Super-Resolution")


select_sample = False  # st.checkbox("Select a sample", False)

results = []

if select_sample:
    samples = os.listdir(Path(__file__).parents[0] / "images/samples/")

    options = [f.split("/")[-1] for f in samples if not f.startswith(".")]

    # Show samples
    selected_image = st.selectbox("Select a sample", options,
                                  format_func=format_func)
    original_img = f"images/samples/{selected_image}"
else:
    original_img = st.file_uploader(
        "Upload image", type=["jpg", "png", "jpeg"])

if original_img:

    original_img = Image.open(original_img, mode="r")

    st.image(original_img, caption="Original image", use_column_width=True)

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
            (hr_img.width, hr_img.height), Image.NEAREST)
        t1 = time.time()
        st.image(nearest_img, use_column_width=True)
        results.append({"Model": "Bicubic upsampling", "Image": nearest_img})

        st.write("SRResNet")
        t0 = time.time()
        srresnet_img = srresnet_predict(srresnet, lr_img)
        t1 = time.time()
        st.image(srresnet_img, use_column_width=True)
        results.append(
            {"Model": "SRResNet", "Image": srresnet_img, "Time": t1-t0})

    with col2:

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

        st.write("High resolution")
        st.image(hr_img, use_column_width=True)
        results.append({"Model": "High resolution",
                       "Image": hr_img, "Time": 0})

    st.header("Metrics")

    def get_ssim(sr_img, hr_img):
        return structural_similarity(
            np.array(hr_img), np.array(sr_img), data_range=255.0
        )

    np_hr_img = np.array(hr_img)
    for idx, result in enumerate(results):
        np_sr_img = np.array(result["Image"])

        results[idx]["PSNR"] = peak_signal_noise_ratio(
            np_hr_img, np_sr_img, data_range=255.0)
        results[idx]["SSIM"] = structural_similarity(
            np_hr_img, np_sr_img, data_range=255.0, multichannel=True)

    df = pd.DataFrame(results)
    df.drop(columns=["Image"], inplace=True)

    st.table(df)
