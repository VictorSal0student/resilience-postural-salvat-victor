"""
Data Loading Module for pMOCAP Files (CODAMOTION)
==================================================

This module handles loading and parsing of pMOCAP .mat files (MATLAB v7.3 HDF5 format)
containing 3D motion capture data for gait analysis.

Functions
---------
load_pmocap_file : Load complete pMOCAP structure
extract_sacrum_markers : Extract 4 Dos markers (Dos01-04)
extract_analog_signal : Extract audio beep signal

Author: Victor SALVAT
Date: 2026-03-30
Project: Locomotor Resilience Analysis - Time Delay Embedding Pipeline
License: MIT
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings


def load_pmocap_file(filepath: str) -> Dict:
    """
    Load complete pMOCAP .mat file structure.
    
    Parameters
    ----------
    filepath : str
        Path to .mat file (e.g., '004_Cued.mat')
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'sacrum': dict with 4 Dos markers (Dos01-04)
        - 'analog': dict with audio beep signal
        - 'metadata': dict with sampling rates, participant info
    
    Examples
    --------
    >>> data = load_pmocap_file('data/004_Cued.mat')
    >>> print(data['metadata']['participant_id'])
    '004'
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Loading pMOCAP file: {filepath.name}")
    
    with h5py.File(filepath, 'r') as f:
        # Extract participant ID from filename
        participant_id = filepath.stem.split('_')[0]
        
        # Extract sacrum markers (Dos01-04)
        sacrum_data = extract_sacrum_markers(f)
        
        # Extract analog signal (audio beeps)
        analog_data = extract_analog_signal(f)
        
        # Metadata
        metadata = {
            'participant_id': participant_id,
            'condition': 'Cued',
            'filename': filepath.name,
            'sacrum_fs': sacrum_data['fs'],
            'analog_fs': analog_data['fs'],
            'duration': sacrum_data['duration']
        }
    
    print(f"  ✓ Loaded successfully")
    print(f"    Participant: {participant_id}")
    print(f"    Duration: {metadata['duration']:.2f} s")
    print(f"    Sacrum fs: {metadata['sacrum_fs']} Hz")
    print(f"    Analog fs: {metadata['analog_fs']} Hz")
    
    return {
        'sacrum': sacrum_data,
        'analog': analog_data,
        'metadata': metadata
    }


def extract_sacrum_markers(f: h5py.File) -> Dict:
    """
    Extract 4 Dos markers from MFilter group.
    
    Parameters
    ----------
    f : h5py.File
        Opened HDF5 file handle
    
    Returns
    -------
    dict
        Dictionary with:
        - 'Dos01', 'Dos02', 'Dos03', 'Dos04': arrays (3, n_samples) [X, Y, Z]
        - 'fs': sampling frequency (Hz)
        - 'occluded': occlusion masks for each marker
        - 'duration': recording duration (seconds)
    """
    base_path = 'pMOCAP/MOCAP/MFilter'
    marker_names = ['Dos01', 'Dos02', 'Dos03', 'Dos04']
    
    markers = {}
    fs = None
    
    for marker_name in marker_names:
        marker_path = f'{base_path}/{marker_name}'
        
        # Value (coordinates)
        value = f[f'{marker_path}/value'][()]  # Shape: (3, n_samples)
        
        # Sampling rate
        if fs is None:
            fs = float(f[f'{marker_path}/Rate'][0, 0])
        
        # Occlusion mask
        occluded = f[f'{marker_path}/occluded'][()].flatten()
        
        markers[marker_name] = {
            'value': value,  # (3, n_samples): X, Y, Z
            'occluded': occluded,
            'n_occluded': np.sum(occluded > 0)
        }
    
    # Calculate duration
    n_samples = markers['Dos01']['value'].shape[1]
    duration = n_samples / fs
    
    # Summary
    total_occluded = sum(m['n_occluded'] for m in markers.values())
    print(f"    Sacrum markers: {len(markers)} × {n_samples} samples")
    print(f"    Occluded samples: {total_occluded} ({100*total_occluded/(n_samples*4):.2f}%)")
    
    return {
        **markers,
        'fs': fs,
        'duration': duration,
        'n_samples': n_samples
    }


def extract_analog_signal(f: h5py.File) -> Dict:
    """
    Extract analog signal (audio beeps for perturbation detection).
    
    Parameters
    ----------
    f : h5py.File
        Opened HDF5 file handle
    
    Returns
    -------
    dict
        Dictionary with:
        - 'value': array (n_samples,) - audio signal
        - 'fs': sampling frequency (Hz)
        - 'occluded': occlusion mask
    """
    analog_path = 'pMOCAP/MOCAP/Analog/Analog01'
    
    # Value
    value = f[f'{analog_path}/value'][()].flatten()
    
    # Sampling rate
    fs = float(f[f'{analog_path}/Rate'][0, 0])
    
    # Occlusion mask
    occluded = f[f'{analog_path}/occluded'][()].flatten()
    
    print(f"    Analog signal: {len(value)} samples at {fs} Hz")
    
    return {
        'value': value,
        'fs': fs,
        'occluded': occluded
    }


def load_multiple_participants(data_dir: str, participant_ids: list) -> Dict:
    """
    Load multiple participant files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing .mat files
    participant_ids : list
        List of participant IDs (e.g., ['001', '004', '005'])
    
    Returns
    -------
    dict
        Dictionary mapping participant_id -> loaded data
    
    Examples
    --------
    >>> data = load_multiple_participants('data/', ['001', '004', '005'])
    >>> print(data['004']['metadata']['duration'])
    """
    data_dir = Path(data_dir)
    all_data = {}
    
    print(f"\n{'='*70}")
    print(f"LOADING MULTIPLE PARTICIPANTS")
    print(f"{'='*70}\n")
    
    for pid in participant_ids:
        filepath = data_dir / f'{pid}_Cued.mat'
        
        try:
            data = load_pmocap_file(filepath)
            all_data[pid] = data
            print()
        except Exception as e:
            warnings.warn(f"Failed to load {pid}: {e}")
            continue
    
    print(f"{'='*70}")
    print(f"✓ Loaded {len(all_data)}/{len(participant_ids)} participants")
    print(f"{'='*70}\n")
    
    return all_data


# Example usage
if __name__ == "__main__":
    # Test with single file
    test_file = Path('data/004_Cued.mat')
    
    if test_file.exists():
        data = load_pmocap_file(test_file)
        
        print("\n" + "="*70)
        print("DATA STRUCTURE")
        print("="*70)
        print(f"\nKeys: {list(data.keys())}")
        print(f"\nSacrum markers: {list(data['sacrum'].keys())}")
        print(f"Dos01 shape: {data['sacrum']['Dos01']['value'].shape}")
        print(f"\nAnalog signal shape: {data['analog']['value'].shape}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please provide path to a valid pMOCAP file.")
