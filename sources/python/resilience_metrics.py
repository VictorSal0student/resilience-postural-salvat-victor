"""
Resilience Metrics Module
==========================

Compute locomotor resilience metrics following Antoine's methodology.

Key Metrics:
- Recovery Time: time to return to stable zone
- Peak Deviation: maximum distance from reference
- Stabilization: number of consecutive stable steps

Reference: Antoine's real_main.m (lines 772-1056)

Author: Victor SALVAT
Date: 2026-03-30
License: MIT
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Tuple, Optional
import warnings


def compute_recovery_time(distances: np.ndarray,
                           thresholds: Dict,
                           perturbation_idx: int,
                           step_indices: np.ndarray,
                           fs: float = 100.0,
                           n_steps_required: int = 10,
                           tolerance_nan: int = 5) -> Dict:
    """
    Compute recovery time after perturbation.
    
    Recovery is defined as returning to stable zone (T1 or T2) and staying
    there for n consecutive steps.
    
    Parameters
    ----------
    distances : np.ndarray
        Euclidean distances to reference trajectory
    thresholds : dict
        Torus thresholds from state_space.compute_torus_thresholds()
    perturbation_idx : int
        Index of perturbation onset (samples)
    step_indices : np.ndarray
        Indices of detected steps/peaks (samples)
    fs : float, default=100.0
        Sampling frequency (Hz)
    n_steps_required : int, default=10
        Number of consecutive stable steps required
    tolerance_nan : int, default=5
        Maximum tolerance for NaN values in stability check
    
    Returns
    -------
    dict
        Recovery metrics:
        - 'recovery_time': time from peak to stability (seconds)
        - 'recovery_idx': index of recovery point (samples)
        - 'peak_idx': index of peak deviation (samples)
        - 'peak_value': peak distance (mm)
        - 'n_steps_to_recovery': number of steps to recover
    
    Examples
    --------
    >>> recovery = compute_recovery_time(distances, thresholds, pert_idx, steps)
    >>> print(f"Recovery time: {recovery['recovery_time']:.2f} s")
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 876-998
    
    Algorithm:
    1. Find peak deviation after perturbation
    2. Search forward from peak for stability
    3. Stability = T1 or T2 zone for n consecutive steps
    4. Recovery time = time from peak to stability
    """
    # Extract thresholds
    T1 = thresholds['T1']
    T2 = thresholds['T2']
    zones = thresholds['zones']
    
    # Define search window for peak (5 seconds after perturbation)
    search_end = min(len(distances), perturbation_idx + int(5 * fs))
    search_range = range(perturbation_idx, search_end)
    
    # Find all peaks in search window
    search_distances = distances[perturbation_idx:search_end]
    peaks, _ = find_peaks(search_distances)
    
    if len(peaks) == 0:
        # No peak found - use maximum
        peak_idx_rel = np.argmax(search_distances)
        peak_idx = perturbation_idx + peak_idx_rel
    else:
        # Find first peak that is higher than all subsequent points
        peak_idx = None
        for peak_rel in peaks:
            candidate_idx = perturbation_idx + peak_rel
            window_end = min(len(distances), candidate_idx + 700)
            
            post_window = distances[candidate_idx:window_end]
            if len(post_window) > 1 and post_window[0] >= np.max(post_window[1:]):
                peak_idx = candidate_idx
                break
        
        if peak_idx is None:
            # Fallback: use global maximum
            peak_idx_rel = np.argmax(search_distances)
            peak_idx = perturbation_idx + peak_idx_rel
    
    peak_value = distances[peak_idx]
    
    # Search for recovery starting from peak
    recovery_idx = None
    
    # Filter steps that come after peak
    steps_after_peak = step_indices[step_indices > peak_idx]
    
    if len(steps_after_peak) < n_steps_required:
        warnings.warn(
            f"Not enough steps after peak ({len(steps_after_peak)} < {n_steps_required}). "
            "Cannot compute recovery time."
        )
        return {
            'recovery_time': np.nan,
            'recovery_idx': np.nan,
            'peak_idx': peak_idx,
            'peak_value': peak_value,
            'n_steps_to_recovery': np.nan
        }
    
    # Search for stability
    for k in range(peak_idx, len(distances)):
        # Must start in stable zone (T1 or T2)
        if zones[k] not in [1, 2]:
            continue
        
        # Find next n steps
        next_steps = steps_after_peak[steps_after_peak > k]
        if len(next_steps) < n_steps_required:
            break  # Not enough steps remaining
        
        target_step = next_steps[n_steps_required - 1]
        
        # Check stability in range [k, target_step]
        check_range = range(k, target_step + 1)
        unstable_count = np.sum(
            (zones[check_range] != 1) & (zones[check_range] != 2)
        )
        
        if unstable_count <= tolerance_nan:
            recovery_idx = k
            break
    
    # Compute metrics
    if recovery_idx is not None:
        recovery_time = (recovery_idx - peak_idx) / fs
        
        # Count steps between peak and recovery
        steps_in_range = step_indices[(step_indices >= peak_idx) & 
                                       (step_indices <= recovery_idx)]
        n_steps = len(steps_in_range)
    else:
        recovery_time = np.nan
        n_steps = np.nan
    
    return {
        'recovery_time': recovery_time,
        'recovery_idx': recovery_idx,
        'peak_idx': peak_idx,
        'peak_value': peak_value,
        'n_steps_to_recovery': n_steps
    }


def compute_all_perturbation_metrics(distances: np.ndarray,
                                      thresholds: Dict,
                                      perturbations: Dict,
                                      step_indices: np.ndarray,
                                      fs: float = 100.0) -> Dict:
    """
    Compute resilience metrics for both perturbations (Slow and Fast).
    
    Parameters
    ----------
    distances : np.ndarray
        Euclidean distances to reference
    thresholds : dict
        Torus thresholds
    perturbations : dict
        Perturbation timings from perturbation_detection
    step_indices : np.ndarray
        Step/peak indices
    fs : float, default=100.0
        Sampling frequency
    
    Returns
    -------
    dict
        Metrics for both perturbations
    
    Examples
    --------
    >>> metrics = compute_all_perturbation_metrics(
    ...     distances, thresholds, perturbations, steps, fs=100
    ... )
    >>> print(metrics['slow']['recovery_time'])
    >>> print(metrics['fast']['recovery_time'])
    """
    results = {}
    
    print("\n" + "="*70)
    print("COMPUTING RESILIENCE METRICS")
    print("="*70)
    
    for pert_name, pert_time in zip(['slow', 'fast'], perturbations['perturbation_times']):
        if np.isnan(pert_time):
            print(f"\n[{pert_name.upper()}] Skipped (not detected)")
            results[pert_name] = {
                'recovery_time': np.nan,
                'peak_value': np.nan,
                'n_steps_to_recovery': np.nan
            }
            continue
        
        print(f"\n[{pert_name.upper()}] Perturbation at t={pert_time:.2f}s")
        
        # Convert time to index
        pert_idx = int(pert_time * fs)
        
        # Compute metrics
        metrics = compute_recovery_time(
            distances, thresholds, pert_idx, step_indices, fs
        )
        
        # Print results
        print(f"  Peak deviation: {metrics['peak_value']:.2f} mm at t={(metrics['peak_idx']/fs):.2f}s")
        if not np.isnan(metrics['recovery_time']):
            print(f"  Recovery time: {metrics['recovery_time']:.2f} s")
            print(f"  Steps to recovery: {metrics['n_steps_to_recovery']:.0f}")
        else:
            print(f"  Recovery time: NOT ACHIEVED")
        
        results[pert_name] = metrics
    
    print(f"\n{'='*70}\n")
    
    return results


def detect_steps(signal: np.ndarray, fs: float = 100.0,
                 min_prominence: float = 5.0, min_distance: int = 50) -> np.ndarray:
    """
    Detect step events from sacrum vertical oscillations.
    
    Parameters
    ----------
    signal : np.ndarray
        Sacrum Z-axis signal
    fs : float, default=100.0
        Sampling frequency
    min_prominence : float, default=5.0
        Minimum peak prominence
    min_distance : int, default=50
        Minimum distance between steps (samples)
    
    Returns
    -------
    np.ndarray
        Indices of detected steps
    
    Notes
    -----
    Steps are detected as negative peaks (valleys) in sacrum Z signal,
    corresponding to foot contact events.
    """
    # Detect negative peaks (valleys = foot contacts)
    peaks, _ = find_peaks(-signal, prominence=min_prominence, distance=min_distance)
    
    print(f"  Detected {len(peaks)} steps")
    print(f"    Mean cadence: {len(peaks) / (len(signal)/fs) * 60:.1f} steps/min")
    
    return peaks


def create_summary_table(metrics: Dict, participant_id: str, 
                          group: str = 'Unknown') -> Dict:
    """
    Create summary table for export to CSV.
    
    Parameters
    ----------
    metrics : dict
        Metrics from compute_all_perturbation_metrics()
    participant_id : str
        Participant ID (e.g., '004')
    group : str
        'Young' or 'Aging'
    
    Returns
    -------
    dict
        Flattened metrics table
    """
    summary = {
        'Participant': participant_id,
        'Group': group,
        'Slow_RecoveryTime': metrics['slow']['recovery_time'],
        'Slow_PeakDeviation': metrics['slow']['peak_value'],
        'Slow_StepsToRecovery': metrics['slow']['n_steps_to_recovery'],
        'Fast_RecoveryTime': metrics['fast']['recovery_time'],
        'Fast_PeakDeviation': metrics['fast']['peak_value'],
        'Fast_StepsToRecovery': metrics['fast']['n_steps_to_recovery']
    }
    
    return summary


# Example usage
if __name__ == "__main__":
    print("Resilience Metrics module loaded.")
    print("\nExample usage:")
    print(">>> steps = detect_steps(sacrum_z)")
    print(">>> metrics = compute_all_perturbation_metrics(distances, thresholds, perturbations, steps)")
