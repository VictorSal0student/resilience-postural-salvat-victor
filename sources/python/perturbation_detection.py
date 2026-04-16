"""
Perturbation Detection Module
==============================

Detects auditory perturbations (Slow and Fast) from analog beep signal.

Key Functions:
- Detect beeps in analog audio signal
- Identify Slow perturbations (cadence -25%)
- Identify Fast perturbations (cadence +25%)
- Validate perturbation timing

Reference: Antoine's real_main.m (lines 192-277)

Author: Victor SALVAT
Date: 2026-03-30
License: MIT
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt


def detect_beeps(analog_signal: np.ndarray, fs: float = 500.0,
                 min_height: float = 35.0, min_distance: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect beep events in analog audio signal.
    
    Parameters
    ----------
    analog_signal : np.ndarray
        Analog signal containing audio beeps
    fs : float, default=500.0
        Sampling frequency of analog signal (Hz)
    min_height : float, default=35.0
        Minimum peak height to detect beeps
    min_distance : int, default=100
        Minimum distance between beeps (samples)
    
    Returns
    -------
    beep_times : np.ndarray
        Times of detected beeps (seconds)
    beep_indices : np.ndarray
        Indices of detected beeps (samples)
    
    Examples
    --------
    >>> beep_times, beep_idx = detect_beeps(analog_data['value'])
    >>> print(f"Detected {len(beep_times)} beeps")
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 192-203
    """
    # Take absolute value to handle negative signals
    analog_abs = np.abs(analog_signal)
    
    # Find peaks (beeps)
    peaks, properties = find_peaks(
        analog_abs,
        height=min_height,
        distance=min_distance
    )
    
    # Convert to time
    beep_times = peaks / fs
    
    print(f"  Detected {len(beep_times)} beeps")
    print(f"    First beep: {beep_times[0]:.2f} s")
    print(f"    Last beep: {beep_times[-1]:.2f} s")
    
    return beep_times, peaks


def detect_perturbations(beep_times: np.ndarray, 
                         tolerance: float = 0.15,
                         n_consecutive: int = 3) -> Dict:
    """
    Detect Slow and Fast perturbations from beep intervals.
    
    Perturbations are identified when inter-beep intervals deviate significantly
    from baseline (first 10 beeps):
    - Slow: intervals > baseline * (1 + tolerance)
    - Fast: intervals < baseline * (1 - tolerance)
    
    Parameters
    ----------
    beep_times : np.ndarray
        Times of beeps (seconds)
    tolerance : float, default=0.15
        Tolerance threshold (15% = ±25% cadence change detectable)
    n_consecutive : int, default=3
        Minimum number of consecutive anomalous intervals
    
    Returns
    -------
    dict
        Contains:
        - 'slow_time': timing of Slow perturbation (seconds)
        - 'fast_time': timing of Fast perturbation (seconds)
        - 'slow_idx': index of Slow perturbation in beep array
        - 'fast_idx': index of Fast perturbation in beep array
        - 'baseline_interval': baseline inter-beep interval
        - 'intervals': all inter-beep intervals
    
    Examples
    --------
    >>> perturbations = detect_perturbations(beep_times)
    >>> print(f"Slow at t={perturbations['slow_time']:.2f}s")
    >>> print(f"Fast at t={perturbations['fast_time']:.2f}s")
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 205-277
    
    Cadence changes:
    - Slow: 75% of baseline (interval 25% longer)
    - Fast: 125% of baseline (interval 20% shorter)
    """
    # Compute inter-beep intervals
    intervals = np.diff(beep_times)
    
    # Baseline interval (average of first 10 beeps)
    baseline_interval = np.mean(intervals[:10])
    baseline_cadence = 60 / baseline_interval  # beats per minute
    
    # Tolerance bounds
    lower_bound = baseline_interval * (1 - tolerance)
    upper_bound = baseline_interval * (1 + tolerance)
    
    print(f"\n  Baseline cadence: {baseline_cadence:.1f} bpm")
    print(f"  Baseline interval: {baseline_interval:.3f} s")
    print(f"  Detection bounds: [{lower_bound:.3f}, {upper_bound:.3f}] s")
    
    # Detect Slow perturbation (long intervals)
    slow_idx = _find_perturbation_onset(
        intervals, upper_bound, n_consecutive, direction='above'
    )
    
    # Detect Fast perturbation (short intervals)
    fast_idx = _find_perturbation_onset(
        intervals, lower_bound, n_consecutive, direction='below'
    )
    
    # Convert to times
    slow_time = beep_times[slow_idx] if slow_idx is not None else np.nan
    fast_time = beep_times[fast_idx] if fast_idx is not None else np.nan
    
    # Sort chronologically
    if not np.isnan(slow_time) and not np.isnan(fast_time):
        if slow_time > fast_time:
            # Swap if Fast comes first
            slow_time, fast_time = fast_time, slow_time
            slow_idx, fast_idx = fast_idx, slow_idx
            slow_label, fast_label = 'Fast', 'Slow'
        else:
            slow_label, fast_label = 'Slow', 'Fast'
    else:
        slow_label, fast_label = 'Perturbation 1', 'Perturbation 2'
    
    print(f"\n  Perturbations detected:")
    if not np.isnan(slow_time):
        print(f"    {slow_label}: t = {slow_time:.2f} s (beep #{slow_idx})")
    else:
        print(f"    {slow_label}: NOT DETECTED")
    
    if not np.isnan(fast_time):
        print(f"    {fast_label}: t = {fast_time:.2f} s (beep #{fast_idx})")
    else:
        print(f"    {fast_label}: NOT DETECTED")
    
    return {
        'slow_time': slow_time,
        'fast_time': fast_time,
        'slow_idx': slow_idx,
        'fast_idx': fast_idx,
        'baseline_interval': baseline_interval,
        'baseline_cadence': baseline_cadence,
        'intervals': intervals,
        'perturbation_times': np.array([slow_time, fast_time]),
        'perturbation_labels': [slow_label, fast_label]
    }


def _find_perturbation_onset(intervals: np.ndarray, threshold: float,
                              n_consecutive: int, direction: str = 'above') -> Optional[int]:
    """
    Find first occurrence of n consecutive anomalous intervals.
    
    Parameters
    ----------
    intervals : np.ndarray
        Inter-beep intervals
    threshold : float
        Threshold value
    n_consecutive : int
        Required number of consecutive anomalies
    direction : str
        'above' or 'below' threshold
    
    Returns
    -------
    int or None
        Index of perturbation onset, or None if not found
    """
    if direction == 'above':
        anomalies = intervals > threshold
    elif direction == 'below':
        anomalies = intervals < threshold
    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    # Find sequences of n consecutive True values
    for i in range(len(intervals) - n_consecutive + 1):
        sequence = anomalies[i:i + n_consecutive]
        if np.all(sequence):
            # Look back up to 5 intervals to find true onset
            search_start = max(0, i - 5)
            local_anomalies = anomalies[search_start:i + 1]
            
            # First anomaly in local window
            first_anomaly = np.where(local_anomalies)[0]
            if len(first_anomaly) > 0:
                return search_start + first_anomaly[0]
            else:
                return i
    
    return None


def validate_perturbations(perturbations: Dict, 
                            expected_slow: Optional[float] = None,
                            expected_fast: Optional[float] = None,
                            tolerance: float = 10.0) -> Dict:
    """
    Validate detected perturbations against expected timings.
    
    Parameters
    ----------
    perturbations : dict
        Output from detect_perturbations()
    expected_slow : float, optional
        Expected timing for Slow perturbation (seconds)
    expected_fast : float, optional
        Expected timing for Fast perturbation (seconds)
    tolerance : float, default=10.0
        Acceptable timing error (seconds)
    
    Returns
    -------
    dict
        Validation results
    """
    results = {
        'slow_valid': False,
        'fast_valid': False,
        'slow_error': np.nan,
        'fast_error': np.nan
    }
    
    if expected_slow is not None and not np.isnan(perturbations['slow_time']):
        error = abs(perturbations['slow_time'] - expected_slow)
        results['slow_error'] = error
        results['slow_valid'] = error < tolerance
    
    if expected_fast is not None and not np.isnan(perturbations['fast_time']):
        error = abs(perturbations['fast_time'] - expected_fast)
        results['fast_error'] = error
        results['fast_valid'] = error < tolerance
    
    return results

def plot_perturbation_detection(beep_times: np.ndarray, perturbations: Dict,
                                 save_path: Optional[str] = None):
    """
    Plot perturbation detection verification (like Antoine's MATLAB).
    
    Shows inter-beep intervals with baseline, thresholds, and detected perturbations.
    
    Parameters
    ----------
    beep_times : np.ndarray
        Times of beeps (seconds)
    perturbations : dict
        Output from detect_perturbations()
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    intervals = perturbations['intervals']
    baseline = perturbations['baseline_interval']
    
    # Compute bounds
    tolerance = 0.15
    lower_bound = baseline * (1 - tolerance)
    upper_bound = baseline * (1 + tolerance)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot intervals
    ax.plot(beep_times[1:], intervals, '-o', linewidth=1.5, markersize=4,
            color='gray', alpha=0.7, label='Inter-beep intervals')
    
    # Baseline and thresholds
    ax.axhline(baseline, linestyle='--', color='black', linewidth=2,
               label=f'Baseline ({baseline:.3f}s)')
    ax.axhline(lower_bound, linestyle=':', color='red', linewidth=2,
               alpha=0.7, label='Thresholds (±15%)')
    ax.axhline(upper_bound, linestyle=':', color='red', linewidth=2, alpha=0.7)
    
    # Mark perturbations
    slow_idx = perturbations['slow_idx']
    fast_idx = perturbations['fast_idx']
    
    if slow_idx is not None and slow_idx < len(intervals):
        ax.plot(beep_times[slow_idx], intervals[slow_idx],
                'bo', markersize=12, markeredgewidth=2, markerfacecolor='blue',
                label=f'{perturbations["perturbation_labels"][0]} start')
    
    if fast_idx is not None and fast_idx < len(intervals):
        ax.plot(beep_times[fast_idx], intervals[fast_idx],
                'ro', markersize=12, markeredgewidth=2, markerfacecolor='red',
                label=f'{perturbations["perturbation_labels"][1]} start')
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Inter-beep interval (s)', fontweight='bold', fontsize=12)
    ax.set_title('Detection of Perturbations', fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


# Example usage
if __name__ == "__main__":
    print("Perturbation Detection module loaded.")
    print("\nExample usage:")
    print(">>> beep_times, _ = detect_beeps(analog_data['value'], fs=500)")
    print(">>> perturbations = detect_perturbations(beep_times)")