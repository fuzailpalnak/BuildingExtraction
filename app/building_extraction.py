import urllib
import numpy as np
import streamlit as st
import cv2
import torch
from PIL import Image
import os
from torch.utils import model_zoo

from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    convert_tensor_to_numpy,
)
from building_footprint_segmentation.utils.operations import handle_image_size

MAX_SIZE = 384
MODEL_URL = "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip"


st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache(allow_output_mutation=True)
def cached_model():
    refine_net = ReFineNet()
    state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
    refine_net.load_state_dict(state_dict)
    return refine_net


model = cached_model()


def main():
    st.sidebar.title("Building Extraction")

    choice = st.sidebar.selectbox(
        "Choose what to do",
        ["Demo", "About"],
    )
    if choice == "Demo":
        extract()
    elif choice == "About":
        st.markdown(get_file_content_as_string("about.md"))


def extract():
    st.title("Building Extraction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "tif", "tiff"])

    if uploaded_file is not None:
        st.header("Image")
        original_image = np.array(Image.open(uploaded_file))

        st.image(original_image, caption="Input Image", use_column_width=True)
        original_height, original_width = original_image.shape[:2]

        if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
            original_image = handle_image_size(original_image, (MAX_SIZE, MAX_SIZE))

        # Apply Normalization
        normalized_image = min_max_image_net(img=original_image)

        tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))

        with torch.no_grad():
            # Perform prediction
            prediction = model(tensor_image)
            prediction = prediction.sigmoid()

        prediction_binary = convert_tensor_to_numpy(prediction[0]).reshape(
            (MAX_SIZE, MAX_SIZE)
        )

        prediction_3_channels = cv2.cvtColor(prediction_binary, cv2.COLOR_GRAY2RGB)

        dst = cv2.addWeighted(
            original_image,
            1,
            (prediction_3_channels * (0, 255, 0)).astype(np.uint8),
            0.4,
            0,
        )

        st.header("Prediction")
        st.image(prediction_binary, caption="Mask", use_column_width=True)

        st.header("Prediction Overlay on Image")
        st.image(dst, caption="Overlay", use_column_width=True)


# Download a single file and make its content available as a string.
@st.cache(show_spinner=True)
def get_file_content_as_string(path):
    url = os.path.join(
        "https://raw.githubusercontent.com/fuzailpalnak/BuildingExtraction/master/",
        path,
    )
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
