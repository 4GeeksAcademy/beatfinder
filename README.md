[Español](README-es.md) | [English](README.md)

# BeatFinder 🎧: Music Genre Classification with Neural Networks
**BeatFinder** is the final project for the Data Science & Machine Learning Bootcamp at 4Geeks Academy. Its main goal is to automatically classify audio recordings into one of 16 main genres using the signal's acoustic features and a Deep Neural Network (DNN) model.

## 📄 Table of Contents
- [Project Goal](#-project-goal) 🎯
- [Technology and Tools](#-technology-and-tools) 🧠
- [Data Exploration and Preprocessing (EDA)](#-data-exploration-and-preprocessing-eda) 📊
- [Model Results](#-model-results) 🏆
- [Repository Structure](#-repository-structure) ⚙️
- [Future Ideas and Project Expansion](#-future-ideas-and-project-expansion) 🚀
- [Co-creators](#-co-creators) 🧑‍💻

---

## 🎯 Project Goal
The main objective of this project is to develop a classification model capable of identifying the musical genre of a track based solely on its acoustic properties.

The ability to perform this task automatically is valuable in contexts such as:
1.  **Optimizing** music recommendation systems,
2.  The efficient **organization** of large digital libraries,
3.  **Analyzing** trends in the music industry.

To achieve this, we have done the following:
-   **Feature Extraction**: Using Librosa, we transformed the audio signals (represented by the [FMA dataset](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium)) into key numerical features (MFCCs, Spectral Centroid, Tonality, etc.).

-   **Preprocessing and Exploration (EDA)**: We cleaned, analyzed, and standardized the data, comparing the impact of outlier detection and capping.

-   **Modeling**: We designed and optimized a Dense Neural Network (DNN) to maximize accuracy in genre identification after exploring other models such as Random Forest, SVM, or Logistic Regression.

*Finally, the model has 49,133 rows and 519 variables, and can predict up to 16 different genres.*

## 🧠 Technology and Tools
| Category | Key Tools |
| :--- | :--- |
| **Modeling & ML** | **TensorFlow/Keras** (Neural Networks), **Scikit-learn** (Random Forest, SVM), **Keras Tuner** (Hyperparameter Optimization). |
| **Audio Analysis** | **Librosa** (Extraction of MFCCs, ZCR, Tonnetz, etc.) |
| **Language** | **Python, Streamlit** |
| **Data Management** | **Pandas, NumPy** |
| **Visualization** | **Seaborn, Matplotlib** |
| **ML Engineering** | Saving **scalers** (`.pkl`) and **models**. |

## 📊 Data Exploration and Preprocessing (EDA)
The project was based on the FMA (Free Music Archive) dataset, which contains metadata and pre-extracted features.

Challenges Overcome:

-   **Column Cleaning**: The complex three-level hierarchy in the feature columns was handled, and genre information was converted from strings to lists.

-   **Winning Dataset Selection**: 6 data combinations were evaluated (With/Without Outliers, Raw/Normalized/MinMax) to determine which offered the best initial accuracy, finding that [Mention the winning dataset here, e.g., X_train_without_outliers_norm] was the most effective for classification.

-   **Imbalance Management**: Univariate analysis revealed a marked class imbalance (predominance of Electronic, Rock, Experimental), a critical factor to consider in performance evaluation. For this reason, and due to a lack of data, we decided to keep the most representative music genres.

-   **RAM Management**: To carry out our work, we used Google Colab and GitHub Codespaces and soon ran out of RAM due to the large amount of data the computer had to process. To address this, we adjusted the datatypes and removed all the variables we were no longer using.

## 🏆 Model Results
Four main models were compared. The best performance was achieved after hyperparameter optimization of the Neural Network using Keras Tuner with the best-selected dataset.

| Model | Optimization | Accuracy |
| :--- | :--- | :--- |
| **Neural Network (DNN)** | Keras Tuner | 69% |
| Random Forest | RandomizedSearchCV | 55.34% |
| SVM (RBF/Linear) | RandomizedSearchCV (Subset) | % |
| Logistic Regression | Base | % |

🥇 The winning model was the Neural Network with a final accuracy of over 69%.

## ⚙️ Repository Structure

```
├── music_genre_identifier/
│   ├── data/
│   │   ├── processed/ (Contains .parquet of normalized/minmaxed X_train/test)
│   └── src/
│       ├── factorized_data/ (Contains factorized_genre_top.json for decoding)
│       └── models/ (Contains the scaler and .pkl models for Streamlit deployment)
├── models/
│   └── neural_network.pkl (The final winning model object)
└── README.md
```

## 🚀 Future Ideas and Project Expansion

### 1. Product and UX/UI Ideas (Left Side)
These ideas focus on how the end-user would interact with the product and how value would be generated.

-   **Enter a verse and automatically get detailed information about the song**: This implies an advanced audio search engine (similar to Shazam or SoundHound), which would require a Machine Learning model for audio hashing search.

-   **Integrate a recommendation system**: Use the genre classification model and the extracted features to recommend similar songs, which requires an additional recommendation model (content-based or collaborative filtering).

-   **Visualization of the audio and the recommended playlist**: You could listen to a preview of the selected song, both the one you upload and those in the playlist.

-   **Monetization**: A banner with advertising or even a paid subscription to avoid these ads.

### 2. Technical Development and ML Ideas (Right Side)
These ideas are direct improvements for the database and the Machine Learning model.

-   **Explore advanced techniques for handling class imbalance**: This would improve the model's accuracy and allow it to identify more music genres that are not currently available.

-   **Perform deeper feature engineering or more sophisticated feature selection**: such as BPM, segmentation, etc.

-   **Integration with external APIs (Spotify, Apple Music, YouTube Music)**: This is the practical implementation of the song preview we mentioned earlier.

Here's a mockup of what the web application could look like:

<img width="631" height="367" alt="image" src="https://github.com/user-attachments/assets/6cc471df-e92e-46cc-9a72-533a6adde78f" />


## 🧑‍💻 Co-creators
This project was developed in collaboration by:

[Daniel Páez](https://github.com/danielpaez-dev) | [Ivan Díaz](https://github.com/ivandla96) | [Tulio Giménez](https://github.com/TulioGimenez)