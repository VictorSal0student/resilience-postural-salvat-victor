"""
State Space Reconstruction Module
==================================

This module handles creation and manipulation of state-space representations
for locomotor resilience analysis.

Key Functions:
- Create reference trajectory from baseline period
- Compute rotation matrices for trajectory alignment
- Calculate Euclidean distances to reference
- Generate torus stability thresholds

Reference: Antoine's real_main.m (lines 279-453)

Author: Victor SALVAT
Date: 2026-03-30
License: MIT
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
import warnings


def create_reference_trajectory(state_space: np.ndarray, 
                                 baseline_end_idx: int,
                                 smooth_window: int = 50) -> Dict:
    """
    Create reference trajectory from baseline (pre-perturbation) period.
    
    The reference trajectory represents the "normal" gait pattern before
    perturbation occurs. It's extracted from baseline period and smoothed.
    
    Parameters
    ----------
    state_space : np.ndarray
        Full state-space matrix (n_points, 3)
    baseline_end_idx : int
        Index marking end of baseline period
    smooth_window : int, default=50
        Window size for smoothing (samples)
    
    Returns
    -------
    dict
        Contains:
        - 'trajectory': smoothed reference trajectory (m, 3)
        - 'centroid': center point (3,)
        - 'rotation_matrix': rotation to align with XY plane (3, 3)
        - 'projected': baseline projected onto plane (m, 3)
    
    Examples
    --------
    >>> ref = create_reference_trajectory(state_space, baseline_end_idx=6000)
    >>> ref_traj = ref['trajectory']
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 279-310
    
    Steps:
    1. Extract baseline portion of state-space
    2. Compute centroid (mean position)
    3. Fit plane to baseline points
    4. Rotate to align plane with XY
    5. Smooth the trajectory
    """
    # Extract baseline - FORCE TO 3D
    baseline = state_space[:baseline_end_idx, :3]  # Take only first 3 columns
    
    # Compute centroid
    centroid = np.mean(baseline, axis=0)
    
    # Center baseline around centroid
    baseline_centered = baseline - centroid
    
    # Fit plane using SVD (simple approach)
    # Normal vector is the eigenvector with smallest eigenvalue
    U, S, Vt = np.linalg.svd(baseline_centered)
    normal = Vt[-1, :]  # Last row (already 3D since baseline_centered is (n, 3))
    
    # Ensure normal is unit vector
    normal = normal / np.linalg.norm(normal)
    
    # Create rotation matrix to align normal with Z-axis
    z_axis = np.array([0, 0, 1])
    rotation_matrix = _rotation_matrix_from_vectors(normal, z_axis)
    
    # Apply rotation
    baseline_rotated = baseline_centered @ rotation_matrix.T
    
    # Smooth trajectory (moving average)
    if smooth_window > 1:
        trajectory_smooth = _smooth_trajectory(baseline_rotated, smooth_window)
    else:
        trajectory_smooth = baseline_rotated
    
    print(f"  Reference trajectory created from baseline:")
    print(f"    Baseline points: {len(baseline)}")
    print(f"    Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
    print(f"    Smoothing window: {smooth_window}")
    
    return {
        'trajectory': trajectory_smooth,
        'centroid': centroid,
        'rotation_matrix': rotation_matrix,
        'projected': baseline_rotated,
        'normal': normal
    }


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates vec1 to align with vec2.
    
    Uses Rodrigues' rotation formula.
    """
    # Normalize vectors and ensure they are 1D with exactly 3 components
    a = np.asarray(vec1).flatten()[:3]  # Force 3D
    b = np.asarray(vec2).flatten()[:3]  # Force 3D
    
    # Normalize
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # Check if vectors are already aligned
    if np.allclose(a, b):
        return np.eye(3)
    
    # Check if vectors are opposite
    if np.allclose(a, -b):
        # Find an orthogonal vector
        if abs(a[0]) < 0.9:
            ortho = np.array([1, 0, 0])
        else:
            ortho = np.array([0, 1, 0])
        v = np.cross(a, ortho)
        v = v / np.linalg.norm(v)
        # 180 degree rotation
        kmat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = 2 * np.outer(v, v) - np.eye(3)
        return R
    
    # Rotation axis
    v = np.cross(a, b)
    
    # Rotation angle
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # Handle near-parallel vectors
    if s < 1e-10:
        return np.eye(3)
    
    # Normalize rotation axis
    v = v / s
    
    # Skew-symmetric cross-product matrix
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    # Rodrigues formula
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    
    return R


def _smooth_trajectory(trajectory: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth trajectory using moving average.
    """
    from scipy.ndimage import uniform_filter1d
    
    smoothed = np.zeros_like(trajectory)
    for i in range(trajectory.shape[1]):
        smoothed[:, i] = uniform_filter1d(trajectory[:, i], size=window, mode='nearest')
    
    return smoothed


def compute_euclidean_distances(state_space: np.ndarray,
                                 reference: Dict,
                                 phase_matching: bool = False) -> np.ndarray:
    """
    Compute Euclidean distance from each point to reference trajectory.
    
    Parameters
    ----------
    state_space : np.ndarray
        Full state-space (n_points, 3)
    reference : dict
        Reference trajectory dict from create_reference_trajectory()
    phase_matching : bool, default=True
        Match points by phase angle (recommended for cyclic motion)
    
    Returns
    -------
    np.ndarray
        Distances (n_points,)
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 772-791
    
    For gait (cyclic motion), we match each point to the reference
    based on its phase angle, not just temporal proximity.
    """
    # Center and rotate state-space - FORCE TO 3D
    state_centered = state_space[:, :3] - reference['centroid']
    state_rotated = state_centered @ reference['rotation_matrix'].T
    
    ref_traj = reference['trajectory']
    
    distances = np.zeros(len(state_space))
    
    if phase_matching:
        # Match by phase angle (for cyclic trajectories)
        # Compute phase angle for reference
        ref_angles = np.arctan2(ref_traj[:, 1], ref_traj[:, 0])
        ref_angles_deg = np.degrees(ref_angles) + 180  # [0, 360]
        
        # Compute phase angle for all points
        state_angles = np.arctan2(state_rotated[:, 1], state_rotated[:, 0])
        state_angles_deg = np.degrees(state_angles) + 180
        
        # For each point, find nearest reference point by phase
        for i in range(len(state_space)):
            phase_diff = np.abs(ref_angles_deg - state_angles_deg[i])
            phase_diff = np.minimum(phase_diff, 360 - phase_diff)  # Wrap around
            
            nearest_idx = np.argmin(phase_diff)
            distances[i] = np.linalg.norm(state_rotated[i] - ref_traj[nearest_idx])
    
    else:
        # Simple nearest neighbor matching
        for i in range(len(state_space)):
            dists_to_ref = np.linalg.norm(ref_traj - state_rotated[i], axis=1)
            distances[i] = np.min(dists_to_ref)
    
    return distances


def compute_torus_thresholds(distances: np.ndarray,
                              baseline_mask: np.ndarray,
                              quantile: float = 0.975) -> Dict:
    """
    Compute stability thresholds (torus radii) from baseline distribution.
    
    Creates 3 zones:
    - T1 (1σ): Very stable
    - T2 (2σ): Stable
    - T3 (3σ): Unstable
    - >T3: Very unstable
    
    Parameters
    ----------
    distances : np.ndarray
        Euclidean distances to reference
    baseline_mask : np.ndarray
        Boolean mask for baseline period
    quantile : float, default=0.975
        Quantile for threshold (e.g., 0.975 = 97.5%)
    
    Returns
    -------
    dict
        Thresholds and classifications
    
    Notes
    -----
    Reference: Antoine's real_main.m lines 805-851
    """
    # Baseline distances only
    baseline_distances = distances[baseline_mask]
    baseline_distances = baseline_distances[np.isfinite(baseline_distances)]
    
    # Compute statistics
    mean_dist = np.mean(baseline_distances)
    std_dist = np.std(baseline_distances)
    
    # Quantile-based base threshold
    threshold_base = np.quantile(baseline_distances, quantile)
    
    # Define thresholds based on standard deviations from mean
    # This is more robust than multiplying by resolution
    T1 = mean_dist + 1.0 * std_dist  # 1 sigma
    T2 = mean_dist + 2.0 * std_dist  # 2 sigma
    T3 = mean_dist + 3.0 * std_dist  # 3 sigma
    
    # Classify all points
    zones = np.zeros(len(distances), dtype=int)
    zones[distances <= T1] = 1  # Very stable
    zones[(distances > T1) & (distances <= T2)] = 2  # Stable
    zones[(distances > T2) & (distances <= T3)] = 3  # Unstable
    zones[distances > T3] = 4  # Very unstable
    
    # Baseline statistics
    n_baseline = len(baseline_distances)
    pct_T1 = 100 * np.sum(zones[:n_baseline] == 1) / n_baseline
    pct_T2 = 100 * np.sum(zones[:n_baseline] == 2) / n_baseline
    pct_T3 = 100 * np.sum(zones[:n_baseline] == 3) / n_baseline
    pct_out = 100 * np.sum(zones[:n_baseline] == 4) / n_baseline
    
    print(f"\n  Torus thresholds computed:")
    print(f"    T1 (1σ): {T1:.2f} mm - Very stable: {pct_T1:.1f}%")
    print(f"    T2 (2σ): {T2:.2f} mm - Stable: {pct_T2:.1f}%")
    print(f"    T3 (3σ): {T3:.2f} mm - Unstable: {pct_T3:.1f}%")
    print(f"    >T3:     Very unstable: {pct_out:.1f}%")
    
    return {
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'zones': zones,
        'baseline_mean': mean_dist,
        'baseline_std': std_dist,
        'percentages': {
            'T1': pct_T1,
            'T2': pct_T2,
            'T3': pct_T3,
            'out': pct_out
        }
    }


# Example usage
if __name__ == "__main__":
    print("State Space Reconstruction module loaded.")
    print("\nExample usage:")
    print(">>> ref = create_reference_trajectory(state_space, baseline_end_idx=6000)")
    print(">>> distances = compute_euclidean_distances(state_space, ref)")
    print(">>> thresholds = compute_torus_thresholds(distances, baseline_mask)")
