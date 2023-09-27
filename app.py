# -------------------- Imports -------------------- #
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, 
    BatchNormalization, Dropout, concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image


# -------------------- Streamlit Configuration -------------------- #
st.set_page_config(
    page_title="Visualization Memorability Prediction Model",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# -------------------- Custom Styling -------------------- #

st.markdown(
            """
            <style>
                body {
                    font-family: 'Helvetica Neue', sans-serif;
                    background-color: #f0f2f6;
                }
                h1, h2 {
                    color: #333;
                }
                .stButton > button {
                    background-color: #007AFF;
                    color: white;
                    font-weight: bold;
                }
                .stSlider > div > div > div > div {
                    background-color: #007AFF !important;
                }
                .stRadio > div > div > label > div > div {
                    background-color: #007AFF;
                    color: white;
                }
                .stSelectbox > div > div > select {
                    background-color: #f0f2f6;
                    border: 1px solid #007AFF;
                }
            </style>
            """, unsafe_allow_html=True
        )


# -------------- Function Definitions -------------- #
def calculate_visual_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    density = min(3, int(variance // 5000))
    return density

def calculate_distinct_colors(image):
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0).shape[0]
    colors = min(3, int(unique_colors // 30000))
    return colors

def calculate_data_ink_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    data_ink = np.sum(edges > 0)
    ratio = min(3, int((data_ink / edges.size) * 10))
    return ratio

@st.cache_data()
def load_model():
    input_image = Input(shape=(256, 256, 3), name="input_image")
    x = Conv2D(64, (3, 3), activation='relu')(input_image)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    flatten_layer = Flatten()(x)

    input_meta = Input(shape=(9,), name="input_meta")
    dense1 = Dense(16, activation='relu')(input_meta)
    batch_norm1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.5)(batch_norm1)
    dense2 = Dense(8, activation='relu')(dropout1)
    batch_norm2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.5)(batch_norm2)

    merged = concatenate([flatten_layer, dropout2])
    merged_dense1 = Dense(16, activation='relu')(merged)
    merged_batch_norm1 = BatchNormalization()(merged_dense1)
    merged_dropout1 = Dropout(0.5)(merged_batch_norm1)
    merged_dense2 = Dense(16, activation='relu')(merged_dropout1)
    merged_batch_norm2 = BatchNormalization()(merged_dense2)
    merged_dropout2 = Dropout(0.5)(merged_batch_norm2)

    output_layer = Dense(1, activation='sigmoid')(merged_dropout2)
    min_dprime = 0.00
    max_dprime = 3.00
    range_dprime = max_dprime - min_dprime
    output = tf.keras.layers.Lambda(lambda x: x * range_dprime + min_dprime)(output_layer)

    model = Model(inputs=[input_image, input_meta], outputs=output)
    model.load_weights("visualization_prediction_model_weights.h5")


    return model
model = load_model()

# -------------------- Streamlit Configuration -------------------- #


# -------------------- Main App -------------------- #
def main():
    st.title("Input an Image for the Model to Predict its Memorability")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and preprocess image for model prediction
        pil_image = Image.open(uploaded_file).resize((256, 256)).convert("RGB")
        image = np.array(pil_image) / 255.0  # Normalize the image
        image_cv = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2BGR)  # Convert to format suitable for metadata calculation

        # Automatically calculate metadata
        visual_density = calculate_visual_density(image_cv)
        distinct_colors = calculate_distinct_colors(image_cv)
        data_ink_ratio = calculate_data_ink_ratio(image_cv)

        st.image(Image.open(uploaded_file).resize((100, 100)), caption='Uploaded Visualization.', use_column_width=False)
        st.write("""
        ### Automatically Detected Metadata
        - **Calculated Visual Density**: {}
        - **Calculated Distinct Colors**: {}
        - **Calculated Data-Ink Ratio**: {}
        """.format(visual_density, distinct_colors, data_ink_ratio))
        # Additional Metadata
        human_recognizable_object = int(st.radio('Does it contain a Human Recognizable Object?', ['Yes', 'No']) == 'Yes')
        human_depiction = int(st.radio('Does it depict humans?', ['Yes', 'No']) == 'Yes')
        vistype = st.selectbox('Type of Visualization', ['Lines', 'Bars', 'None'])

        # Allow users to manually adjust automatically calculated metadata
        visual_density = st.slider('Visual Density (0: Low, 3: High)', 0, 3, visual_density)
        distinct_colors = st.slider('Number of Distinct Colors (0: Low, 3: High)', 0, 3, distinct_colors)
        data_ink_ratio = st.slider('Data-Ink Ratio (0: Low, 3: High)', 0, 3, data_ink_ratio)

        # One-hot encoding for 'vistype'
        vistype_lines = int(vistype == 'Lines')
        vistype_bars = int(vistype == 'Bars')

        # Prepare image and metadata input for the model
        image_input = image[np.newaxis, ...]

        metadata = np.array([
            1 - human_recognizable_object,  # Not Human Recognizable
            1 - human_depiction,  # Does not depict humans
            vistype_lines, 
            vistype_bars, 
            visual_density, 
            human_depiction,  # Depicts humans
            distinct_colors, 
            data_ink_ratio, 
            human_recognizable_object  # Human Recognizable
        ]).reshape(1, -1)

        if st.button('Predict'):
            # Prediction
            d_prime_prediction = model.predict([image_input, metadata])[0][0]
            st.write(f"### Predicted Memorability Score (d-prime value): {d_prime_prediction}")
            st.info("The d-prime value indicates how memorable your visualization is. The value ranges from 0.0-3.0. A higher value means it's more likely to be remembered.")


            # Plotting the normal distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.linspace(0, 3, 1000)
            y = norm.pdf(x, 1.5, 0.5)  # Assuming mean=1.5 and std_dev=0.5 for the normal distribution

            ax.plot(x, y, color='blue')
            ax.fill_between(x, y, where=(x <= d_prime_prediction), color='blue', alpha=0.5)
            ax.set_title(f"Memorability Score Distribution")
            ax.set_xlabel("d-prime value")
            ax.set_ylabel("Probability Density")
            ax.axvline(d_prime_prediction, color='red', linestyle='--')
            ax.annotate(f'Predicted: {d_prime_prediction}', xy=(d_prime_prediction, 0), xytext=(d_prime_prediction, 0.2),
                        arrowprops=dict(facecolor='red', arrowstyle='->'))
            ax.grid(True)

            st.pyplot(fig)




# -------------------- Initialize -------------------- #
if __name__ == "__main__":
    model = load_model()
    main()
