# Importing libraries
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from pickle import load
import json
import os
from extract_features import extract_features
import librosa

# Constants
MODEL_DIR = "src/models"
FACTORIZE_DIR = "factorize"
MODEL_FILENAME = "Neuronal_Network.pkl"
SCALER_FILENAME = "scaler_without_outliers.pkl"
GENRE_DICT_FILENAME = "factorized_genre_top.json"


# Styles
def load_css(file_name):
    """Loads the contents of a CSS file and injects the styles into Streamlit."""

    current_dir = os.path.dirname(__file__)
    css_path = os.path.join(current_dir, file_name)

    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Warning: No CSS file founded '{file_name}'.")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")


load_css("styles.css")

# Web app
st.title("BeatFinder")
st.header("Automatic Classification of Musical Genres")
st.write(
    "Upload your track and BeatFinder will use advanced AI to instantly determine its genre"
)

# Charging model and scaler
try:
    # first we download the data from'src/models'
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)

    model = load(open(model_path, "rb"))
    scaler = load(open(scaler_path, "rb"))

except Exception as e:
    # If it fails, we'll try to download the data from '/models'
    try:
        model_path = os.path.join("models", MODEL_FILENAME)
        scaler_path = os.path.join("models", SCALER_FILENAME)

        model = load(open(model_path, "rb"))
        scaler = load(open(scaler_path, "rb"))
    except Exception as e_alt:
        st.error(f"Unexpected Error. Please, try again later. {e_alt}")
        st.stop()


# Factorization
try:
    dict_path = os.path.join(FACTORIZE_DIR, GENRE_DICT_FILENAME)
    with open(dict_path) as f:
        genre_dict = json.load(f)
except FileNotFoundError:
    st.error(f"Error: Unexpected error. Please, try again later {dict_path}")
    st.stop()


if "notification_permission_requested" not in st.session_state:
    st.session_state.notification_permission_requested = False

## audio
uploaded_file = st.file_uploader("Choose a file")
status_placeholder = st.empty()

if uploaded_file is not None and not st.session_state.notification_permission_requested:

    # request permission to user
    js_request_permission = """
    <script>
        if (!("Notification" in window)) {
            // El navegador no soporta notificaciones, no hacemos nada.
        } else if (Notification.permission !== "denied" && Notification.permission !== "granted") {
            Notification.requestPermission();
        }
    </script>
    """

    components.html(js_request_permission, height=0, width=0)

# Prediction
if st.button("Predict"):
    if uploaded_file is None:
        status_placeholder.error(
            "Please, upload a music file before start a prediction."
        )
    else:
        with st.spinner("🎧 Analyzing track... It may take some seconds"):
            temp_filename = ""
            try:
                # 1. Save the temporary file
                file_extension = os.path.splitext(uploaded_file.name)[1]
                temp_filename = f"temp_audio{file_extension}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                y, sr = librosa.load(temp_filename, sr=None)

                # 2. Feature Extraction
                row = extract_features(y, sr).reshape(1, -1)

                # 3. Prediction
                row_scaled = scaler.transform(row)
                prediction = model.predict(row_scaled)[0]
                predicted_index = int(np.argmax(prediction))
                genre_prediction = genre_dict[str(predicted_index)]

                js_send_notification = f"""
                <script>
                    if ("Notification" in window && Notification.permission === "granted") {{
                        new Notification("Gender Classified!", {{
                            body: "The result is: {genre_prediction}",
                            icon: "https://i.imgur.com/your-app-icon.png"
                        }});
                    }}
                </script>
                """

                # 4. Notification
                components.html(js_send_notification, height=0, width=0)
                status_placeholder.success(
                    f"🎉 **Gender Classified! The result is: {genre_prediction}**"
                )

            except Exception as e:
                status_placeholder.error(
                    f"An error occurred during processing. Make sure the file is a valid audio format. Details: {e}"
                )

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

st.caption(
    "Gender Classification results are generated using an Artificial Intelligence (AI) model and are provided for informational purposes only; accuracy is not 100% guaranteed. Regarding privacy, your audio files are temporarily processed on our servers for classification purposes only and are deleted immediately after the result is obtained. We do not store, share, or redistribute your content."
)
