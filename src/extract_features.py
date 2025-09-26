import numpy as np
import scipy.stats as stats

def extract_features(y, sr):
    import librosa
    features = []
    # Chroma CENS
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(chroma_cens, axis=1)
        features.extend(stat)
    # Chroma CQT
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(chroma_cqt, axis=1)
        features.extend(stat)
    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(chroma_stft, axis=1)
        features.extend(stat)
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(mfcc, axis=1)
        features.extend(stat)
    # RMSE
    rmse = librosa.feature.rms(y=y)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(rmse, axis=1)
        features.extend(stat)
    # Spectral Bandwidth
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(sb, axis=1)
        features.extend(stat)
    # Spectral Centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(sc, axis=1)
        features.extend(stat)
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(contrast, axis=1)
        features.extend(stat)
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(rolloff, axis=1)
        features.extend(stat)
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(tonnetz, axis=1)
        features.extend(stat)
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    for stat_func in [stats.kurtosis, np.max, np.mean, np.median, np.min, stats.skew, np.std]:
        stat = stat_func(zcr, axis=1)
        features.extend(stat)
    return np.array(features)
