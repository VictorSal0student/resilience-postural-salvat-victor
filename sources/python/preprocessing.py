"""
Preprocessing Module for Gait Data
===================================

This module implements preprocessing steps following Antoine's methodology:
1. Compute barycenter of 4 Dos markers
2. Apply EMD (Empirical Mode Decomposition)
3. Butterworth low-pass filtering
4. Downsampling (400 Hz → 100 Hz)

Functions
---------
compute_barycenter : Calculate center of 4 markers
apply_emd_decomposition : EMD with IMF selection
butterworth_filter : Low-pass filtering
downsample_signal : Decimation
preprocess_pipeline : Complete preprocessing chain

Author: Victor SALVAT
Date: 2026-03-30
Reference: Antoine's EMD.m and real_main.m scripts
License: MIT
"""

import numpy as np
from scipy.signal import butter, filtfilt, decimate
from typing import Dict, Tuple, Optional
import warnings

# Try to import EMD library
try:
    from PyEMD import EMD
    EMD_AVAILABLE = 'PyEMD'
except ImportError:
    try:
        from emd import sift
        EMD_AVAILABLE = 'EMD-signal'
    except ImportError:
        EMD_AVAILABLE = None
        warnings.warn(
            "No EMD library found. Install with: pip install EMD-signal\n"
            "EMD decomposition will be skipped (analysis will use filtered signal directly)."
        )


def compute_barycenter(dos01: np.ndarray, dos02: np.ndarray, 
                       dos03: np.ndarray, dos04: np.ndarray) -> np.ndarray:
    """
    Compute barycenter (center of mass) of 4 Dos markers.
    
    Parameters
    ----------
    dos01, dos02, dos03, dos04 : np.ndarray
        Each shape (3, n_samples) representing [X, Y, Z] coordinates
    
    Returns
    -------
    np.ndarray
        Shape (n_samples, 3) - Barycenter coordinates [X, Y, Z]
    
    Examples
    --------
    >>> barycenter = compute_barycenter(dos01, dos02, dos03, dos04)
    >>> print(barycenter.shape)  # (n_samples, 3)
    
    Notes
    -----
    Reference: Antoine's EMD.m lines 49-56
    Computes mean across 4 markers for each axis (X, Y, Z).
    """
    # Stack all markers: shape (4, 3, n_samples)
    all_markers = np.stack([dos01, dos02, dos03, dos04], axis=0)
    
    # Mean across markers (axis=0): shape (3, n_samples)
    mean_coords = np.mean(all_markers, axis=0)
    
    # Transpose to (n_samples, 3) for easier handling
    barycenter = mean_coords.T
    
    # Subtract mean (center around zero)
    barycenter = barycenter - np.mean(barycenter, axis=0)
    
    print(f"  Barycenter computed: {barycenter.shape}")
    print(f"    X range: [{barycenter[:, 0].min():.2f}, {barycenter[:, 0].max():.2f}] mm")
    print(f"    Y range: [{barycenter[:, 1].min():.2f}, {barycenter[:, 1].max():.2f}] mm")
    print(f"    Z range: [{barycenter[:, 2].min():.2f}, {barycenter[:, 2].max():.2f}] mm")
    
    return barycenter


def apply_emd_decomposition(signal: np.ndarray, imf_indices: list = [3, 4, 5, 6, 7, 8]) -> np.ndarray:
    """
    Apply Empirical Mode Decomposition and reconstruct signal from selected IMFs.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal (e.g., sacrum Z-axis)
    imf_indices : list, default=[3, 4, 5, 6, 7, 8]
        IMF indices to combine (0-indexed)
        Antoine uses IMFs 4-9 (1-indexed) = indices 3-8 (0-indexed)
    
    Returns
    -------
    np.ndarray
        Reconstructed signal from selected IMFs
    
    Notes
    -----
    Reference: Antoine's EMD.m lines 36-46
    Antoine combines IMF 4-9 for MFilter data to remove high-frequency noise
    and low-frequency drift while preserving gait-related oscillations.
    
    If EMD library not available, returns original signal with warning.
    """
    if EMD_AVAILABLE is None:
        warnings.warn("EMD not available - returning original signal")
        return signal
    
    print(f"  Applying EMD decomposition...")
    
    if EMD_AVAILABLE == 'PyEMD':
        # PyEMD library
        emd = EMD()
        IMFs = emd(signal)
        
    elif EMD_AVAILABLE == 'EMD-signal':
        # EMD-signal library
        imf = sift.sift(signal)
        IMFs = imf.T  # Transpose to match PyEMD format
    
    n_imfs = IMFs.shape[0]
    print(f"    Decomposed into {n_imfs} IMFs")
    
    # Select IMFs (adjust indices if not enough IMFs)
    valid_indices = [i for i in imf_indices if i < n_imfs]
    if len(valid_indices) < len(imf_indices):
        warnings.warn(f"Only {n_imfs} IMFs available, using indices {valid_indices}")
    
    # Combine selected IMFs
    reconstructed = np.sum(IMFs[valid_indices, :], axis=0)
    
    print(f"    Combined IMFs {valid_indices}")
    print(f"    Reconstructed signal: mean={reconstructed.mean():.2f}, std={reconstructed.std():.2f}")
    
    return reconstructed


def butterworth_filter(signal: np.ndarray, fs: float = 100.0, 
                       cutoff: float = 5.0, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal to filter
    fs : float, default=100.0
        Sampling frequency (Hz)
    cutoff : float, default=5.0
        Cutoff frequency (Hz)
    order : int, default=4
        Filter order
    
    Returns
    -------
    np.ndarray
        Filtered signal
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 78-109
    Uses bidirectional filtering (filtfilt) to avoid phase shift.
    """
    # Design filter
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Handle NaN/Inf values
    signal_clean = signal.copy()
    signal_clean[~np.isfinite(signal_clean)] = 0
    
    # Apply zero-phase filtering
    filtered = filtfilt(b, a, signal_clean)
    
    print(f"  Butterworth filter applied: {order}th order, fc={cutoff} Hz")
    
    return filtered


def downsample_signal(signal: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Downsample signal by integer factor.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal to downsample
    factor : int, default=4
        Downsampling factor (e.g., 4 for 400 Hz → 100 Hz)
    
    Returns
    -------
    np.ndarray
        Downsampled signal
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 136-158
    Uses scipy.signal.decimate which applies anti-aliasing filter.
    """
    downsampled = decimate(signal, factor, zero_phase=True)
    
    print(f"  Downsampled by factor {factor}: {len(signal)} → {len(downsampled)} samples")
    
    return downsampled


def preprocess_pipeline(sacrum_data: Dict, apply_emd: bool = True, 
                        downsample_factor: int = 4) -> Dict:
    """
    Complete preprocessing pipeline following Antoine's methodology.
    
    Parameters
    ----------
    sacrum_data : dict
        Dictionary from load_data.extract_sacrum_markers()
    apply_emd : bool, default=True
        Whether to apply EMD decomposition
    downsample_factor : int, default=4
        Downsampling factor (400 Hz → 100 Hz)
    
    Returns
    -------
    dict
        Preprocessed data with keys:
        - 'barycenter': (n_samples, 3) - raw barycenter
        - 'sacrum_z': 1D array - Z-axis after all preprocessing
        - 'sacrum_z_filtered': 1D array - Z-axis filtered (before downsampling)
        - 'fs_original': original sampling frequency
        - 'fs_final': final sampling frequency after downsampling
        - 'time': time vector (seconds)
    
    Examples
    --------
    >>> from load_data import load_pmocap_file
    >>> data = load_pmocap_file('004_Cued.mat')
    >>> preprocessed = preprocess_pipeline(data['sacrum'])
    >>> sacrum_z = preprocessed['sacrum_z']
    """
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Compute barycenter
    print("\n[1/5] Computing barycenter...")
    barycenter = compute_barycenter(
        sacrum_data['Dos01']['value'],
        sacrum_data['Dos02']['value'],
        sacrum_data['Dos03']['value'],
        sacrum_data['Dos04']['value']
    )
    
    # Extract Z-axis (vertical, most relevant for gait)
    sacrum_z_raw = barycenter[:, 2]
    
    # Step 2: EMD decomposition (optional)
    print("\n[2/5] EMD decomposition...")
    if apply_emd and EMD_AVAILABLE is not None:
        sacrum_z_emd = apply_emd_decomposition(sacrum_z_raw)
    else:
        print("  Skipping EMD (not available or disabled)")
        sacrum_z_emd = sacrum_z_raw
    
    # Step 3: Butterworth filtering
    print("\n[3/5] Butterworth filtering...")
    fs_original = sacrum_data['fs']
    sacrum_z_filtered = butterworth_filter(
        sacrum_z_emd, 
        fs=fs_original, 
        cutoff=5.0, 
        order=4
    )
    
    # Step 4: Downsampling
    print("\n[4/5] Downsampling...")
    sacrum_z_downsampled = downsample_signal(sacrum_z_filtered, factor=downsample_factor)
    fs_final = fs_original / downsample_factor
    
    # Step 5: Create time vector
    print("\n[5/5] Creating time vector...")
    time = np.arange(len(sacrum_z_downsampled)) / fs_final
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  Original: {len(sacrum_z_raw)} samples @ {fs_original} Hz")
    print(f"  Final: {len(sacrum_z_downsampled)} samples @ {fs_final} Hz")
    print(f"  Duration: {time[-1]:.2f} seconds")
    print(f"{'='*70}\n")
    
    return {
        'barycenter': barycenter,
        'sacrum_z': sacrum_z_downsampled,
        'sacrum_z_filtered': sacrum_z_filtered,
        'fs_original': fs_original,
        'fs_final': fs_final,
        'time': time,
        'emd_applied': apply_emd and EMD_AVAILABLE is not None
    }


# Example usage
if __name__ == "__main__":
    print("Preprocessing module loaded.")
    print(f"EMD library: {EMD_AVAILABLE if EMD_AVAILABLE else 'Not available'}")
    print("\nExample usage:")
    print(">>> from load_data import load_pmocap_file")
    print(">>> data = load_pmocap_file('004_Cued.mat')")
    print(">>> preprocessed = preprocess_pipeline(data['sacrum'])")
