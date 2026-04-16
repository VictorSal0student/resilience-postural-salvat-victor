"""
Time Delay Embedding Module
============================

This module implements Time Delay Embedding (TDE) methods for state-space reconstruction
of dynamical systems from 1D time series.

Key Functions:
- Average Mutual Information (AMI) for optimal time delay (tau)
- False Nearest Neighbors (FNN) for optimal embedding dimension
- Phase space reconstruction

References:
- Wurdeman, S. R. (2016). State-space reconstruction. In Nonlinear analysis 
  for human movement variability (pp. 55-82). CRC Press.
- Antoine's real_main.m (lines 117-186)

Author: Victor SALVAT
Date: 2026-03-30
License: MIT
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple
import warnings
from sklearn.neighbors import NearestNeighbors


def compute_ami(signal: np.ndarray, max_lag: int = 100, 
                num_bins: int = 20) -> Tuple[int, np.ndarray]:
    """
    Compute Average Mutual Information to determine optimal time delay (tau).
    
    The optimal tau is the first minimum of the AMI curve, indicating the time
    delay at which the signal becomes sufficiently decorrelated from itself.
    
    Parameters
    ----------
    signal : np.ndarray
        1D time series (e.g., sacrum Z-axis)
    max_lag : int, default=100
        Maximum time delay to test (in samples)
    num_bins : int, default=20
        Number of bins for histogram estimation
    
    Returns
    -------
    tau_optimal : int
        Optimal time delay (samples)
    ami_values : np.ndarray
        AMI values for all tested lags
    
    Examples
    --------
    >>> tau, ami = compute_ami(sacrum_z, max_lag=100)
    >>> print(f"Optimal tau: {tau} samples")
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 121-152
    
    Mathematical formulation:
    AMI(τ) = Σ p(x(t), x(t+τ)) * log[p(x(t), x(t+τ)) / (p(x(t)) * p(x(t+τ)))]
    
    where p() are probability distributions estimated from histograms.
    """
    ami_values = np.zeros(max_lag)
    
    # Compute histogram edges once
    edges = np.linspace(signal.min(), signal.max(), num_bins + 1)
    
    for lag in range(1, max_lag + 1):
        # Create shifted versions
        original = signal[lag:]
        shifted = signal[:-lag]
        
        # 2D histogram (joint probability)
        joint_counts, _, _ = np.histogram2d(original, shifted, bins=edges)
        joint_prob = joint_counts / joint_counts.sum()
        
        # Marginal probabilities
        px = joint_prob.sum(axis=1)  # P(x(t))
        py = joint_prob.sum(axis=0)  # P(x(t+τ))
        
        # Mutual Information
        mi = 0.0
        for i in range(num_bins):
            for j in range(num_bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (px[i] * py[j] + 1e-10)
                    )
        
        ami_values[lag - 1] = mi
    
    # Find first local minimum (optimal tau)
    # Negative AMI because find_peaks finds maxima
    peaks, _ = find_peaks(-ami_values, prominence=0.01)
    
    if len(peaks) > 0:
        tau_optimal = peaks[0] + 1  # +1 because lag indexing starts at 1
    else:
        # Fallback: use lag at which AMI drops to 1/e of initial value
        threshold = ami_values[0] / np.e
        candidates = np.where(ami_values < threshold)[0]
        tau_optimal = candidates[0] + 1 if len(candidates) > 0 else 10
        warnings.warn(f"No clear AMI minimum found. Using tau={tau_optimal} (fallback)")
    
    return tau_optimal, ami_values


def compute_fnn(signal: np.ndarray, max_dim: int = 10, tau: int = 10,
                threshold: float = 1.0, r_threshold: float = 15.0) -> Tuple[int, np.ndarray]:
    """
    Compute False Nearest Neighbors to determine optimal embedding dimension.
    
    The optimal dimension is the first dimension at which the FNN percentage
    falls below a threshold, indicating that the attractor is fully unfolded.
    
    Parameters
    ----------
    signal : np.ndarray
        1D time series
    max_dim : int, default=10
        Maximum embedding dimension to test
    tau : int, default=10
        Time delay (from AMI)
    threshold : float, default=1.0
        FNN percentage threshold (e.g., 1.0 = 1%)
    r_threshold : float, default=15.0
        Distance threshold ratio for FNN criterion
    
    Returns
    -------
    dim_optimal : int
        Optimal embedding dimension
    fnn_rates : np.ndarray
        FNN percentage for each dimension
    
    Examples
    --------
    >>> dim, fnn = compute_fnn(sacrum_z, max_dim=10, tau=tau_optimal)
    >>> print(f"Optimal dimension: {dim}")
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 163-181
    
    FNN criterion: A neighbor in d dimensions is "false" if it becomes
    far away when embedding in d+1 dimensions, indicating insufficient
    dimension for proper attractor reconstruction.
    """
    fnn_rates = np.zeros(max_dim)
    
    for dim in range(1, max_dim + 1):
        fnn_count = _compute_fnn_for_dimension(signal, dim, tau, r_threshold)
        
        # Total possible neighbors
        n_points = len(signal) - dim * tau
        
        # FNN percentage
        if n_points > 0:
            fnn_rates[dim - 1] = 100.0 * fnn_count / max(n_points, 1)
        else:
            fnn_rates[dim - 1] = 100.0
    
    # Find first dimension below threshold
    below_threshold = np.where(fnn_rates < threshold)[0]
    
    if len(below_threshold) > 0:
        dim_optimal = below_threshold[0] + 1  # +1 for 1-indexed dimension
    else:
        # Fallback: use dimension with minimum FNN rate
        dim_optimal = np.argmin(fnn_rates) + 1
        warnings.warn(
            f"FNN never dropped below {threshold}%. "
            f"Using dim={dim_optimal} (minimum FNN={fnn_rates[dim_optimal-1]:.2f}%)"
        )
    
    return dim_optimal, fnn_rates


def _compute_fnn_for_dimension(signal: np.ndarray, dim: int, tau: int,
                                r_threshold: float = 10.0) -> int:
    """
    Compute FNN count for a specific dimension.
    Uses real k-nearest neighbor search (Antoine's method).
    """
    from sklearn.neighbors import NearestNeighbors
    
    N = len(signal)
    max_idx = N - (dim - 1) * tau
    
    if max_idx <= 0:
        return N  # All points are "false" if we can't embed
    
    # Construct embedded data matrix
    embedded_data = np.zeros((max_idx, dim))
    for i in range(dim):
        start_idx = i * tau
        end_idx = start_idx + max_idx
        embedded_data[:, i] = signal[start_idx:end_idx]
    
    # Remove NaN rows
    valid_rows = ~np.isnan(embedded_data).any(axis=1)
    embedded_data_clean = embedded_data[valid_rows]
    base_indices = np.where(valid_rows)[0]
    
    if len(embedded_data_clean) < 2:
        return max_idx
    
    # Build k-NN tree
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(embedded_data_clean)
    distances, indices = nbrs.kneighbors(embedded_data_clean)
    
    # Count false nearest neighbors
    fnn_count = 0
    valid_count = 0
    
    for i in range(len(embedded_data_clean)):
        neighbor_idx = indices[i, 1]
        nearest_dist = distances[i, 1]
        
        idx_i = base_indices[i]
        idx_n = base_indices[neighbor_idx]
        
        if (idx_i + dim * tau >= N) or (idx_n + dim * tau >= N):
            continue
        if np.isnan(signal[idx_i + dim * tau]) or np.isnan(signal[idx_n + dim * tau]):
            continue
        
        valid_count += 1
        
        dist_increased = abs(signal[idx_i + dim * tau] - signal[idx_n + dim * tau])
        
        if nearest_dist > 0 and (dist_increased / nearest_dist) > r_threshold:
            fnn_count += 1
    
    return fnn_count if valid_count > 0 else max_idx


def phase_space_reconstruction(signal: np.ndarray, tau: int, 
                                dim: int) -> np.ndarray:
    """
    Reconstruct phase space from 1D time series using time delay embedding.
    
    Creates a multi-dimensional representation of the dynamical system by
    embedding the signal: [x(t), x(t+τ), x(t+2τ), ..., x(t+(dim-1)τ)]
    
    Parameters
    ----------
    signal : np.ndarray
        1D time series
    tau : int
        Time delay (samples)
    dim : int
        Embedding dimension
    
    Returns
    -------
    np.ndarray
        Phase space matrix, shape (n_points, dim)
        where n_points = len(signal) - (dim-1)*tau
    
    Examples
    --------
    >>> state_space = phase_space_reconstruction(sacrum_z, tau=10, dim=3)
    >>> print(state_space.shape)  # (n_points, 3)
    
    Notes
    -----
    Reference: Antoine's real_main.m line 185
    
    For dim=3, tau=10:
    state_space[i, :] = [signal[i], signal[i+10], signal[i+20]]
    """
    # Calculate number of points in reconstructed space
    n_points = len(signal) - (dim - 1) * tau
    
    if n_points <= 0:
        raise ValueError(
            f"Signal too short for reconstruction. "
            f"Need at least {(dim-1)*tau + 1} samples, got {len(signal)}"
        )
    
    # Initialize phase space matrix
    phase_space = np.zeros((n_points, dim))
    
    # Fill each dimension with time-delayed signal
    for d in range(dim):
        start_idx = d * tau
        end_idx = start_idx + n_points
        phase_space[:, d] = signal[start_idx:end_idx]
    
    # VERIFY OUTPUT DIMENSION
    assert phase_space.shape[1] == dim, f"Expected {dim} dimensions, got {phase_space.shape[1]}"
    
    return phase_space


def auto_tde_parameters(signal: np.ndarray, max_lag: int = 100,
                        max_dim: int = 10, verbose: bool = True) -> Tuple[int, int]:
    """
    Automatically determine optimal TDE parameters (tau and dim).
    
    Convenience function that runs both AMI and FNN analysis.
    
    Parameters
    ----------
    signal : np.ndarray
        1D time series
    max_lag : int, default=100
        Maximum lag for AMI
    max_dim : int, default=10
        Maximum dimension for FNN
    verbose : bool, default=True
        Print results
    
    Returns
    -------
    tau_optimal : int
        Optimal time delay
    dim_optimal : int
        Optimal embedding dimension
    
    Examples
    --------
    >>> tau, dim = auto_tde_parameters(sacrum_z)
    >>> state_space = phase_space_reconstruction(sacrum_z, tau, dim)
    """
    if verbose:
        print("\n" + "="*70)
        print("TIME DELAY EMBEDDING - PARAMETER SELECTION")
        print("="*70)
    
    # Step 1: AMI for tau
    if verbose:
        print("\n[1/2] Computing Average Mutual Information (AMI)...")
    
    tau_optimal, ami_values = compute_ami(signal, max_lag=max_lag)
    
    if verbose:
        print(f"  ✓ Optimal tau: {tau_optimal} samples")
        print(f"    AMI at tau: {ami_values[tau_optimal-1]:.4f}")
    
    # Step 2: FNN for dim
    if verbose:
        print("\n[2/2] Computing False Nearest Neighbors (FNN)...")
    
    dim_optimal, fnn_rates = compute_fnn(signal, max_dim=max_dim, tau=tau_optimal)
    
    if verbose:
        print(f"  ✓ Optimal dimension: {dim_optimal}")
        print(f"    FNN at dim: {fnn_rates[dim_optimal-1]:.2f}%")
        print(f"\n{'='*70}")
        print(f"OPTIMAL PARAMETERS: tau={tau_optimal}, dim={dim_optimal}")
        print(f"{'='*70}\n")
    
    return tau_optimal, dim_optimal


# Example usage
if __name__ == "__main__":
    print("Time Delay Embedding module loaded.")
    print("\nExample usage:")
    print(">>> tau, dim = auto_tde_parameters(sacrum_z)")
    print(">>> state_space = phase_space_reconstruction(sacrum_z, tau, dim)")
