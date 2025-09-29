# Importing libraries
import streamlit as st
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
MODEL_FILENAME = "neuronal_network.pkl" 
SCALER_FILENAME = "scaler_without_outliers.pkl"
GENRE_DICT_FILENAME = "factorized_genre_top.json"

# Styles
def load_css(file_name):
    """Loads the contents of a CSS file and injects the styles into Streamlit."""
    
    current_dir = os.path.dirname(__file__)
    css_path = os.path.join(current_dir, file_name) 

    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Warning: No CSS file founded '{file_name}'.")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

load_css('styles.css')


# Web app
st.title('BeatFinder')
st.subheader('Music Genre Classification')

# Charging model and scaler
try:
    # first we download the data from'src/models'
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)
    
    model = load(open(model_path, 'rb'))
    scaler = load(open(scaler_path, 'rb'))
    st.sidebar.success("Models and Scaler PKL loaded from the 'src/models' path.")
    
except Exception as e:
    # If it fails, we'll try to download the data from '/models'
    try:
        model_path = os.path.join("models", MODEL_FILENAME)
        scaler_path = os.path.join("models", SCALER_FILENAME)
        
        model = load(open(model_path, 'rb'))
        scaler = load(open(scaler_path, 'rb'))
        st.sidebar.success("Modelos y Scaler PKL cargados desde la ruta 'models/'.")
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



## audio
uploaded_file = st.file_uploader("Choose a file")
row = pd.DataFrame([])
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_filename = f"temp_audio{file_extension}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    y, sr = librosa.load(temp_filename, sr=None)
    # Extraer features
    row = extract_features(y, sr).reshape(1, -1)

print(row)


# all rows needed: ['chroma_cens_kurtosis_01', 'chroma_cens_kurtosis_02', 'chroma_cens_kurtosis_03', 'chroma_cens_kurtosis_04', 'chroma_cens_kurtosis_05', 'chroma_cens_kurtosis_06', 'chroma_cens_kurtosis_07', 'chroma_cens_kurtosis_08', 'chroma_cens_kurtosis_09', 'chroma_cens_kurtosis_10', 'chroma_cens_kurtosis_11', 'chroma_cens_kurtosis_12', 'chroma_cens_max_01', 'chroma_cens_max_02', 'chroma_cens_max_03', 'chroma_cens_max_04', 'chroma_cens_max_05', 'chroma_cens_max_06', 'chroma_cens_max_07', 'chroma_cens_max_08', 'chroma_cens_max_09', 'chroma_cens_max_10', 'chroma_cens_max_11', 'chroma_cens_max_12', 'chroma_cens_mean_01', 'chroma_cens_mean_02', 'chroma_cens_mean_03', 'chroma_cens_mean_04', 'chroma_cens_mean_05', 'chroma_cens_mean_06', 'chroma_cens_mean_07', 'chroma_cens_mean_08', 'chroma_cens_mean_09', 'chroma_cens_mean_10', 'chroma_cens_mean_11', 'chroma_cens_mean_12', 'chroma_cens_median_01', 'chroma_cens_median_02', 'chroma_cens_median_03', 'chroma_cens_median_04', 'chroma_cens_median_05', 'chroma_cens_median_06', 'chroma_cens_median_07', 'chroma_cens_median_08', 'chroma_cens_median_09', 'chroma_cens_median_10', 'chroma_cens_median_11', 'chroma_cens_median_12', 'chroma_cens_min_01', 'chroma_cens_min_02', 'chroma_cens_min_03', 'chroma_cens_min_04', 'chroma_cens_min_05', 'chroma_cens_min_06', 'chroma_cens_min_07', 'chroma_cens_min_08', 'chroma_cens_min_09', 'chroma_cens_min_10', 'chroma_cens_min_11', 'chroma_cens_min_12', 'chroma_cens_skew_01', 'chroma_cens_skew_02', 'chroma_cens_skew_03', 'chroma_cens_skew_04', 'chroma_cens_skew_05', 'chroma_cens_skew_06', 'chroma_cens_skew_07', 'chroma_cens_skew_08', 'chroma_cens_skew_09', 'chroma_cens_skew_10', 'chroma_cens_skew_11', 'chroma_cens_skew_12', 'chroma_cens_std_01', 'chroma_cens_std_02', 'chroma_cens_std_03', 'chroma_cens_std_04', 'chroma_cens_std_05', 'chroma_cens_std_06', 'chroma_cens_std_07', 'chroma_cens_std_08', 'chroma_cens_std_09', 'chroma_cens_std_10', 'chroma_cens_std_11', 'chroma_cens_std_12', 'chroma_cqt_kurtosis_01', 'chroma_cqt_kurtosis_02', 'chroma_cqt_kurtosis_03', 'chroma_cqt_kurtosis_04', 'chroma_cqt_kurtosis_05', 'chroma_cqt_kurtosis_06', 'chroma_cqt_kurtosis_07', 'chroma_cqt_kurtosis_08', 'chroma_cqt_kurtosis_09', 'chroma_cqt_kurtosis_10', 'chroma_cqt_kurtosis_11', 'chroma_cqt_kurtosis_12', 'chroma_cqt_max_01', 'chroma_cqt_max_02', 'chroma_cqt_max_03', 'chroma_cqt_max_04', 'chroma_cqt_max_05', 'chroma_cqt_max_06', 'chroma_cqt_max_07', 'chroma_cqt_max_08', 'chroma_cqt_max_09', 'chroma_cqt_max_10', 'chroma_cqt_max_11', 'chroma_cqt_max_12', 'chroma_cqt_mean_01', 'chroma_cqt_mean_02', 'chroma_cqt_mean_03', 'chroma_cqt_mean_04', 'chroma_cqt_mean_05', 'chroma_cqt_mean_06', 'chroma_cqt_mean_07', 'chroma_cqt_mean_08', 'chroma_cqt_mean_09', 'chroma_cqt_mean_10', 'chroma_cqt_mean_11', 'chroma_cqt_mean_12', 'chroma_cqt_median_01', 'chroma_cqt_median_02', 'chroma_cqt_median_03', 'chroma_cqt_median_04', 'chroma_cqt_median_05', 'chroma_cqt_median_06', 'chroma_cqt_median_07', 'chroma_cqt_median_08', 'chroma_cqt_median_09', 'chroma_cqt_median_10', 'chroma_cqt_median_11', 'chroma_cqt_median_12', 'chroma_cqt_min_01', 'chroma_cqt_min_02', 'chroma_cqt_min_03', 'chroma_cqt_min_04', 'chroma_cqt_min_05', 'chroma_cqt_min_06', 'chroma_cqt_min_07', 'chroma_cqt_min_08', 'chroma_cqt_min_09', 'chroma_cqt_min_10', 'chroma_cqt_min_11', 'chroma_cqt_min_12', 'chroma_cqt_skew_01', 'chroma_cqt_skew_02', 'chroma_cqt_skew_03', 'chroma_cqt_skew_04', 'chroma_cqt_skew_05', 'chroma_cqt_skew_06', 'chroma_cqt_skew_07', 'chroma_cqt_skew_08', 'chroma_cqt_skew_09', 'chroma_cqt_skew_10', 'chroma_cqt_skew_11', 'chroma_cqt_skew_12', 'chroma_cqt_std_01', 'chroma_cqt_std_02', 'chroma_cqt_std_03', 'chroma_cqt_std_04', 'chroma_cqt_std_05', 'chroma_cqt_std_06', 'chroma_cqt_std_07', 'chroma_cqt_std_08', 'chroma_cqt_std_09', 'chroma_cqt_std_10', 'chroma_cqt_std_11', 'chroma_cqt_std_12', 'chroma_stft_kurtosis_01', 'chroma_stft_kurtosis_02', 'chroma_stft_kurtosis_03', 'chroma_stft_kurtosis_04', 'chroma_stft_kurtosis_05', 'chroma_stft_kurtosis_06', 'chroma_stft_kurtosis_07', 'chroma_stft_kurtosis_08', 'chroma_stft_kurtosis_09', 'chroma_stft_kurtosis_10', 'chroma_stft_kurtosis_11', 'chroma_stft_kurtosis_12', 'chroma_stft_max_01', 'chroma_stft_max_02', 'chroma_stft_max_03', 'chroma_stft_max_04', 'chroma_stft_max_05', 'chroma_stft_max_06', 'chroma_stft_max_07', 'chroma_stft_max_08', 'chroma_stft_max_09', 'chroma_stft_max_10', 'chroma_stft_max_11', 'chroma_stft_max_12', 'chroma_stft_mean_01', 'chroma_stft_mean_02', 'chroma_stft_mean_03', 'chroma_stft_mean_04', 'chroma_stft_mean_05', 'chroma_stft_mean_06', 'chroma_stft_mean_07', 'chroma_stft_mean_08', 'chroma_stft_mean_09', 'chroma_stft_mean_10', 'chroma_stft_mean_11', 'chroma_stft_mean_12', 'chroma_stft_median_01', 'chroma_stft_median_02', 'chroma_stft_median_03', 'chroma_stft_median_04', 'chroma_stft_median_05', 'chroma_stft_median_06', 'chroma_stft_median_07', 'chroma_stft_median_08', 'chroma_stft_median_09', 'chroma_stft_median_10', 'chroma_stft_median_11', 'chroma_stft_median_12', 'chroma_stft_min_01', 'chroma_stft_min_02', 'chroma_stft_min_03', 'chroma_stft_min_04', 'chroma_stft_min_05', 'chroma_stft_min_06', 'chroma_stft_min_07', 'chroma_stft_min_08', 'chroma_stft_min_09', 'chroma_stft_min_10', 'chroma_stft_min_11', 'chroma_stft_min_12', 'chroma_stft_skew_01', 'chroma_stft_skew_02', 'chroma_stft_skew_03', 'chroma_stft_skew_04', 'chroma_stft_skew_05', 'chroma_stft_skew_06', 'chroma_stft_skew_07', 'chroma_stft_skew_08', 'chroma_stft_skew_09', 'chroma_stft_skew_10', 'chroma_stft_skew_11', 'chroma_stft_skew_12', 'chroma_stft_std_01', 'chroma_stft_std_02', 'chroma_stft_std_03', 'chroma_stft_std_04', 'chroma_stft_std_05', 'chroma_stft_std_06', 'chroma_stft_std_07', 'chroma_stft_std_08', 'chroma_stft_std_09', 'chroma_stft_std_10', 'chroma_stft_std_11', 'chroma_stft_std_12', 'mfcc_kurtosis_01', 'mfcc_kurtosis_02', 'mfcc_kurtosis_03', 'mfcc_kurtosis_04', 'mfcc_kurtosis_05', 'mfcc_kurtosis_06', 'mfcc_kurtosis_07', 'mfcc_kurtosis_08', 'mfcc_kurtosis_09', 'mfcc_kurtosis_10', 'mfcc_kurtosis_11', 'mfcc_kurtosis_12', 'mfcc_kurtosis_13', 'mfcc_kurtosis_14', 'mfcc_kurtosis_15', 'mfcc_kurtosis_16', 'mfcc_kurtosis_17', 'mfcc_kurtosis_18', 'mfcc_kurtosis_19', 'mfcc_kurtosis_20', 'mfcc_max_01', 'mfcc_max_02', 'mfcc_max_03', 'mfcc_max_04', 'mfcc_max_05', 'mfcc_max_06', 'mfcc_max_07', 'mfcc_max_08', 'mfcc_max_09', 'mfcc_max_10', 'mfcc_max_11', 'mfcc_max_12', 'mfcc_max_13', 'mfcc_max_14', 'mfcc_max_15', 'mfcc_max_16', 'mfcc_max_17', 'mfcc_max_18', 'mfcc_max_19', 'mfcc_max_20', 'mfcc_mean_01', 'mfcc_mean_02', 'mfcc_mean_03', 'mfcc_mean_04', 'mfcc_mean_05', 'mfcc_mean_06', 'mfcc_mean_07', 'mfcc_mean_08', 'mfcc_mean_09', 'mfcc_mean_10', 'mfcc_mean_11', 'mfcc_mean_12', 'mfcc_mean_13', 'mfcc_mean_14', 'mfcc_mean_15', 'mfcc_mean_16', 'mfcc_mean_17', 'mfcc_mean_18', 'mfcc_mean_19', 'mfcc_mean_20', 'mfcc_median_01', 'mfcc_median_02', 'mfcc_median_03', 'mfcc_median_04', 'mfcc_median_05', 'mfcc_median_06', 'mfcc_median_07', 'mfcc_median_08', 'mfcc_median_09', 'mfcc_median_10', 'mfcc_median_11', 'mfcc_median_12', 'mfcc_median_13', 'mfcc_median_14', 'mfcc_median_15', 'mfcc_median_16', 'mfcc_median_17', 'mfcc_median_18', 'mfcc_median_19', 'mfcc_median_20', 'mfcc_min_01', 'mfcc_min_02', 'mfcc_min_03', 'mfcc_min_04', 'mfcc_min_05', 'mfcc_min_06', 'mfcc_min_07', 'mfcc_min_08', 'mfcc_min_09', 'mfcc_min_10', 'mfcc_min_11', 'mfcc_min_12', 'mfcc_min_13', 'mfcc_min_14', 'mfcc_min_15', 'mfcc_min_16', 'mfcc_min_17', 'mfcc_min_18', 'mfcc_min_19', 'mfcc_min_20', 'mfcc_skew_01', 'mfcc_skew_02', 'mfcc_skew_03', 'mfcc_skew_04', 'mfcc_skew_05', 'mfcc_skew_06', 'mfcc_skew_07', 'mfcc_skew_08', 'mfcc_skew_09', 'mfcc_skew_10', 'mfcc_skew_11', 'mfcc_skew_12', 'mfcc_skew_13', 'mfcc_skew_14', 'mfcc_skew_15', 'mfcc_skew_16', 'mfcc_skew_17', 'mfcc_skew_18', 'mfcc_skew_19', 'mfcc_skew_20', 'mfcc_std_01', 'mfcc_std_02', 'mfcc_std_03', 'mfcc_std_04', 'mfcc_std_05', 'mfcc_std_06', 'mfcc_std_07', 'mfcc_std_08', 'mfcc_std_09', 'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12', 'mfcc_std_13', 'mfcc_std_14', 'mfcc_std_15', 'mfcc_std_16', 'mfcc_std_17', 'mfcc_std_18', 'mfcc_std_19', 'mfcc_std_20', 'rmse_kurtosis_01', 'rmse_max_01', 'rmse_mean_01', 'rmse_median_01', 'rmse_min_01', 'rmse_skew_01', 'rmse_std_01', 'spectral_bandwidth_kurtosis_01', 'spectral_bandwidth_max_01', 'spectral_bandwidth_mean_01', 'spectral_bandwidth_median_01', 'spectral_bandwidth_min_01', 'spectral_bandwidth_skew_01', 'spectral_bandwidth_std_01', 'spectral_centroid_kurtosis_01', 'spectral_centroid_max_01', 'spectral_centroid_mean_01', 'spectral_centroid_median_01', 'spectral_centroid_min_01', 'spectral_centroid_skew_01', 'spectral_centroid_std_01', 'spectral_contrast_kurtosis_01', 'spectral_contrast_kurtosis_02', 'spectral_contrast_kurtosis_03', 'spectral_contrast_kurtosis_04', 'spectral_contrast_kurtosis_05', 'spectral_contrast_kurtosis_06', 'spectral_contrast_kurtosis_07', 'spectral_contrast_max_01', 'spectral_contrast_max_02', 'spectral_contrast_max_03', 'spectral_contrast_max_04', 'spectral_contrast_max_05', 'spectral_contrast_max_06', 'spectral_contrast_max_07', 'spectral_contrast_mean_01', 'spectral_contrast_mean_02', 'spectral_contrast_mean_03', 'spectral_contrast_mean_04', 'spectral_contrast_mean_05', 'spectral_contrast_mean_06', 'spectral_contrast_mean_07', 'spectral_contrast_median_01', 'spectral_contrast_median_02', 'spectral_contrast_median_03', 'spectral_contrast_median_04', 'spectral_contrast_median_05', 'spectral_contrast_median_06', 'spectral_contrast_median_07', 'spectral_contrast_min_01', 'spectral_contrast_min_02', 'spectral_contrast_min_03', 'spectral_contrast_min_04', 'spectral_contrast_min_05', 'spectral_contrast_min_06', 'spectral_contrast_min_07', 'spectral_contrast_skew_01', 'spectral_contrast_skew_02', 'spectral_contrast_skew_03', 'spectral_contrast_skew_04', 'spectral_contrast_skew_05', 'spectral_contrast_skew_06', 'spectral_contrast_skew_07', 'spectral_contrast_std_01', 'spectral_contrast_std_02', 'spectral_contrast_std_03', 'spectral_contrast_std_04', 'spectral_contrast_std_05', 'spectral_contrast_std_06', 'spectral_contrast_std_07', 'spectral_rolloff_kurtosis_01', 'spectral_rolloff_max_01', 'spectral_rolloff_mean_01', 'spectral_rolloff_median_01', 'spectral_rolloff_min_01', 'spectral_rolloff_skew_01', 'spectral_rolloff_std_01', 'tonnetz_kurtosis_01', 'tonnetz_kurtosis_02', 'tonnetz_kurtosis_03', 'tonnetz_kurtosis_04', 'tonnetz_kurtosis_05', 'tonnetz_kurtosis_06', 'tonnetz_max_01', 'tonnetz_max_02', 'tonnetz_max_03', 'tonnetz_max_04', 'tonnetz_max_05', 'tonnetz_max_06', 'tonnetz_mean_01', 'tonnetz_mean_02', 'tonnetz_mean_03', 'tonnetz_mean_04', 'tonnetz_mean_05', 'tonnetz_mean_06', 'tonnetz_median_01', 'tonnetz_median_02', 'tonnetz_median_03', 'tonnetz_median_04', 'tonnetz_median_05', 'tonnetz_median_06', 'tonnetz_min_01', 'tonnetz_min_02', 'tonnetz_min_03', 'tonnetz_min_04', 'tonnetz_min_05', 'tonnetz_min_06', 'tonnetz_skew_01', 'tonnetz_skew_02', 'tonnetz_skew_03', 'tonnetz_skew_04', 'tonnetz_skew_05', 'tonnetz_skew_06', 'tonnetz_std_01', 'tonnetz_std_02', 'tonnetz_std_03', 'tonnetz_std_04', 'tonnetz_std_05', 'tonnetz_std_06', 'zcr_kurtosis_01', 'zcr_max_01', 'zcr_mean_01', 'zcr_median_01', 'zcr_min_01', 'zcr_skew_01', 'zcr_std_01']

# Prediction
if st.button('Predict'):
    if row.shape[1] == 0:
        st.error("Please, upload a music file before start a prediction.")
    else:
        row_scaled = scaler.transform(row)
        prediction = model.predict(row_scaled)[0]
        predicted_index = int(np.argmax(prediction))
        genre_prediction = genre_dict[str(predicted_index)]
        st.success(f"Predicted genre: {genre_prediction}") 