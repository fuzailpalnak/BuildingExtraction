import urllib
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import os


def main():
    # Render the readme as markdown using st.markdown.
    # readme_text = st.markdown(get_file_content_as_string("information.md"))
    # Download external dependencies.
    # for filename in EXTERNAL_DEPENDENCIES.keys():
    #     download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Building Extraction")

    choice = st.sidebar.selectbox(
        "Choose what to do",
        ["Information about the Project", "DEMO", "Show me the code", "About"],
    )
    if choice == "Information about the Project":
        st.markdown(get_file_content_as_string("information.md"))
    elif choice == "Show the source code":
        st.code(get_file_content_as_string("building_extraction.py"))
    elif choice == "DEMO":
        run_the_app()
    elif choice == "About":
        st.markdown(get_file_content_as_string("about.md"))


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    img_file_buffer = input_image_ui()
    if img_file_buffer is not None:
        image = load_image(img_file_buffer)
    else:
        # DEFAULT IMAGE
        with urllib.request.urlopen(
            "https://user-images.githubusercontent.com/24665570/91715533-713d5000-ebab-11ea-9057-20bb17687b8b.png"
        ) as response:
            image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]  # BGR -> RGB
    st.image(image, use_column_width=True)
    threshold = confidence_ui()

    st.subheader("BUILDING EXTRACTION")
    st.markdown("Prediction with confidence `%3.1f`" % (threshold,))
    if threshold < 0.50:
        st.image(np.array(image).astype(np.uint8), use_column_width=True)
    else:
        st.image(
            np.array(np.zeros((256, 256, 3))).astype(np.uint8), use_column_width=True
        )


def input_image_ui():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    img_file_buffer = st.sidebar.file_uploader("Upload an image")
    return img_file_buffer


@st.cache(show_spinner=False)
def load_image(img_file_buffer):
    return Image.open(img_file_buffer)


def confidence_ui():
    st.sidebar.markdown("# Model")
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.01)
    return confidence


def predict(image):
    @st.cache(allow_output_mutation=True)
    def load_network(config_path, weights_path):
        # load_network
        pass

    w, h, _ = image.shape

    # Add normalization and convert to tensor

    model = load_network(None, None)
    prediction = model(image)

    # convert prediction to numpy
    prediction = prediction.reshape((w, h))
    return prediction


# Download a single file and make its content available as a string.
@st.cache(show_spinner=True)
def get_file_content_as_string(path):
    master = "https://raw.githubusercontent.com/fuzailpalnak/BuildingExtraction/master/"
    branch = "https://raw.githubusercontent.com/fuzailpalnak/BuildingExtraction/extraction/modelv1/"
    url = os.path.join(branch, path)

    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
