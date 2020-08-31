import urllib
import numpy as np
import streamlit as st
import cv2
from PIL import Image


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies.
    # for filename in EXTERNAL_DEPENDENCIES.keys():
    #     download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Building Extraction DEMO")
    app_mode = st.sidebar.selectbox("Choose what to do",
        ["Information about the Project", "DEMO", "Show the source code"])
    if app_mode == "Information about the Project":
        st.sidebar.success('To continue select "DEMO".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))
    elif app_mode == "DEMO":
        readme_text.empty()
        run_the_app()


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    img_file_buffer = input_image_ui()
    if img_file_buffer is not None:
        image = load_image(img_file_buffer)
    else:
        # DEFAULT IMAGE
        with urllib.request.urlopen(
                "https://user-images.githubusercontent.com/24665570/91715533-713d5000-ebab-11ea-9057-20bb17687b8b.png") as response:
            image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]  # BGR -> RGB

    st.image(image, use_column_width=True)
    threshold = object_detector_ui()

    st.subheader("BUILDING EXTRACTION")
    st.markdown("Prediction with confidence `%3.1f`" % (threshold,))
    if threshold < 0.50:
        st.image(np.array(image).astype(np.uint8), use_column_width=True)
    else:
        st.image(np.array(np.zeros((256, 256, 3))).astype(np.uint8), use_column_width=True)


def input_image_ui():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    img_file_buffer = st.sidebar.file_uploader("Upload an image")
    return img_file_buffer


@st.cache(show_spinner=False)
def load_image(img_file_buffer):
    return Image.open(img_file_buffer)


def object_detector_ui():
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
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
