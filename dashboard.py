import streamlit as st
from PIL import Image
from model.viz import load_models
from model.inference import load_image, srresnet_predict, esrgan_predict, srgan_predict

srresnet, srgan_generator, esrgan_generator = load_models()


st.title("Super-Resolution")

original_img = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if original_img:

    original_img = Image.open(original_img, mode="r")
    st.image(original_img, caption="Original image", use_column_width=True)

    col1, col2 = st.columns(2)

    hr_img, lr_img = load_image(original_img)
    with col1:
        st.write("Nearest neighbour upsampling")
        nearest_img = lr_img.resize(
            (hr_img.width, hr_img.height), Image.NEAREST)
        st.image(nearest_img, use_column_width=True)

        st.write("Bicubic upsampling")
        nearest_img = lr_img.resize(
            (hr_img.width, hr_img.height), Image.NEAREST)
        st.image(nearest_img, use_column_width=True)

        st.write("SRResNet")
        srresnet_img = srresnet_predict(srresnet, lr_img)
        st.image(srresnet_img, use_column_width=True)

    with col2:

        st.write("SRGAN")
        with st.spinner("Generating SRGAN image..."):
            sr_img_srgan = srgan_predict(srgan_generator, lr_img)
        st.image(sr_img_srgan, use_column_width=True)

        st.write("ESRGAN")
        with st.spinner("Generating ESRGAN image..."):
            sr_img_esrgan = esrgan_predict(esrgan_generator, lr_img)
        st.image(sr_img_esrgan, use_column_width=True)

        st.write("High resolution")
        st.image(hr_img, use_column_width=True)
