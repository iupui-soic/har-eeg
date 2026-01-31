"""
HAR Feature Extraction Functions
Extracts time-domain, frequency-domain, and 3D features from IMU signals.
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq


def extract_time_domain_features(signal):
    """Extract time-domain features from a 1D signal"""
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['median'] = np.median(signal)
    features['range'] = np.max(signal) - np.min(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['skewness'] = stats.skew(signal)
    features['kurtosis'] = stats.kurtosis(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    features['zcr'] = len(zero_crossings) / len(signal)
    features['mad'] = np.mean(np.abs(signal - np.mean(signal)))
    return features


def extract_frequency_domain_features(signal, sampling_rate):
    """Extract frequency-domain features from a 1D signal"""
    features = {}
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/sampling_rate)
    
    pos_mask = xf > 0
    freqs = xf[pos_mask]
    magnitude = np.abs(yf[pos_mask])
    power = magnitude**2
    
    dominant_idx = np.argmax(magnitude)
    features['dominant_freq'] = freqs[dominant_idx]
    features['spectral_energy'] = np.sum(power)
    
    psd_norm = power / np.sum(power)
    psd_norm = psd_norm[psd_norm > 0]
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
    
    features['mean_freq'] = np.sum(freqs * magnitude) / np.sum(magnitude)
    
    cumsum_power = np.cumsum(power)
    median_idx = np.where(cumsum_power >= cumsum_power[-1] / 2)[0][0]
    features['median_freq'] = freqs[median_idx]
    
    band1_mask = (freqs >= 0) & (freqs < 5)
    features['power_0_5hz'] = np.sum(power[band1_mask])
    
    band2_mask = (freqs >= 5) & (freqs < 10)
    features['power_5_10hz'] = np.sum(power[band2_mask])
    
    band3_mask = (freqs >= 10) & (freqs < 12.5)
    features['power_10_12_5hz'] = np.sum(power[band3_mask])
    
    return features


def extract_3d_features(window):
    """Extract 3D features from a multi-axis window"""
    axis1, axis2, axis3 = window[:, 0], window[:, 1], window[:, 2]
    features = {}
    
    magnitude = np.sqrt(axis1**2 + axis2**2 + axis3**2)
    features['mag_mean'] = np.mean(magnitude)
    features['mag_std'] = np.std(magnitude)
    features['mag_max'] = np.max(magnitude)
    features['mag_min'] = np.min(magnitude)
    
    jerk = np.diff(magnitude)
    features['jerk_mean'] = np.mean(np.abs(jerk))
    features['jerk_std'] = np.std(jerk)
    features['jerk_max'] = np.max(np.abs(jerk))
    
    pitch = np.arctan2(axis2, np.sqrt(axis1**2 + axis3**2))
    roll = np.arctan2(axis1, np.sqrt(axis2**2 + axis3**2))
    
    features['pitch_mean'] = np.mean(pitch)
    features['pitch_std'] = np.std(pitch)
    features['roll_mean'] = np.mean(roll)
    features['roll_std'] = np.std(roll)
    
    features['sma'] = np.sum(np.abs(axis1) + np.abs(axis2) + np.abs(axis3)) / len(axis1)
    
    features['corr_12'] = np.corrcoef(axis1, axis2)[0, 1]
    features['corr_13'] = np.corrcoef(axis1, axis3)[0, 1]
    features['corr_23'] = np.corrcoef(axis2, axis3)[0, 1]
    
    features['energy'] = np.sum(axis1**2 + axis2**2 + axis3**2) / len(axis1)
    
    return features


def extract_all_har_features(acg_window, gyro_window, sampling_rate):
    """
    Extract all 152 features for HAR in the correct order.
    
    Args:
        acg_window: Accelerometer window (N, 3)
        gyro_window: Gyroscope window (N, 3)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        numpy array of 152 features
    """
    all_features = []
    
    # ACG 3D features (16)
    acg_3d = extract_3d_features(acg_window)
    for val in acg_3d.values():
        all_features.append(val)
    
    # ACG time-domain per axis (11 × 3 = 33)
    for axis in range(3):
        time_feats = extract_time_domain_features(acg_window[:, axis])
        for val in time_feats.values():
            all_features.append(val)
    
    # ACG frequency-domain per axis (8 × 3 = 24)
    for axis in range(3):
        freq_feats = extract_frequency_domain_features(acg_window[:, axis], sampling_rate)
        for val in freq_feats.values():
            all_features.append(val)
    
    # GYRO 3D features (16)
    gyro_3d = extract_3d_features(gyro_window)
    for val in gyro_3d.values():
        all_features.append(val)
    
    # GYRO time-domain per axis (11 × 3 = 33)
    for axis in range(3):
        time_feats = extract_time_domain_features(gyro_window[:, axis])
        for val in time_feats.values():
            all_features.append(val)
    
    # GYRO frequency-domain per axis (8 × 3 = 24)
    for axis in range(3):
        freq_feats = extract_frequency_domain_features(gyro_window[:, axis], sampling_rate)
        for val in freq_feats.values():
            all_features.append(val)
    
    return np.array(all_features)
