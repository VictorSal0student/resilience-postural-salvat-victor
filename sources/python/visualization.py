"""
Visualization Module
====================

Generate publication-quality figures for locomotor resilience analysis.

Figure Types:
- AMI and FNN plots (TDE parameter selection)
- 3D state-space trajectories
- Recovery curves with zones
- Group comparisons (Young vs Aging)

Author: Victor SALVAT
Date: 2026-03-30
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def plot_ami_fnn(ami_values: np.ndarray, fnn_rates: np.ndarray,
                 tau_optimal: int, dim_optimal: int,
                 save_path: Optional[str] = None):
    """
    Plot AMI and FNN curves with optimal parameters marked.
    
    Parameters
    ----------
    ami_values : np.ndarray
        AMI values for each lag
    fnn_rates : np.ndarray
        FNN percentages for each dimension
    tau_optimal : int
        Optimal tau (marked on AMI plot)
    dim_optimal : int
        Optimal dimension (marked on FNN plot)
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # AMI plot
    lags = np.arange(1, len(ami_values) + 1)
    ax1.plot(lags, ami_values, 'b-', linewidth=1.5)
    ax1.plot(tau_optimal, ami_values[tau_optimal - 1], 'ro', 
             markersize=10, label=f'Optimal τ = {tau_optimal}')
    ax1.set_xlabel('Time Delay (samples)', fontweight='bold')
    ax1.set_ylabel('Average Mutual Information', fontweight='bold')
    ax1.set_title('AMI - Time Delay Selection', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FNN plot
    dims = np.arange(1, len(fnn_rates) + 1)
    ax2.plot(dims, fnn_rates * 100, 's-', linewidth=1.5, markersize=6)
    ax2.plot(dim_optimal, fnn_rates[dim_optimal - 1] * 100, 'ro',
             markersize=10, label=f'Optimal dim = {dim_optimal}')
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='Threshold (10%)')
    ax2.set_xlabel('Embedding Dimension', fontweight='bold')
    ax2.set_ylabel('False Nearest Neighbors (%)', fontweight='bold')
    ax2.set_title('FNN - Dimension Selection', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_state_space_3d(state_space: np.ndarray, reference: Dict,
                        perturbations: Dict, title: str = "State Space Reconstruction",
                        save_path: Optional[str] = None):
    """
    Plot 3D state-space trajectory with reference and perturbations marked.
    
    Parameters
    ----------
    state_space : np.ndarray
        Full state-space (n, 3)
    reference : dict
        Reference trajectory dict
    perturbations : dict
        Perturbation timings
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Force to 3D
    state_space_3d = state_space[:, :3]
    
    # Reference trajectory
    ref_traj = reference['trajectory']
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
            'purple', linewidth=2.5, label='Reference Trajectory', alpha=0.8)
    
    # Centroid
    centroid = reference['centroid']
    ax.scatter(centroid[0], centroid[1], centroid[2],
               c='red', s=80, marker='o', label='Centroid', zorder=5)
    
    # Full trajectory (lighter)
    ax.plot(state_space_3d[:, 0], state_space_3d[:, 1], state_space_3d[:, 2],
            'gray', linewidth=0.5, alpha=0.3, label='Full Trajectory')
    
    # Mark perturbations if available
    if 'perturbation_times' in perturbations:
        for i, (label, t) in enumerate(zip(perturbations['perturbation_labels'], 
                                             perturbations['perturbation_times'])):
            if not np.isnan(t):
                idx = int(t * 100)  # Assuming 100 Hz
                if idx < len(state_space_3d):
                    color = 'orange' if i == 0 else 'green'
                    ax.scatter(state_space_3d[idx, 0], state_space_3d[idx, 1], state_space_3d[idx, 2],
                              c=color, s=100, marker='*', label=f'{label}', zorder=5)
    
    ax.set_xlabel('X(t)', fontweight='bold', fontsize=12)
    ax.set_ylabel('X(t + τ)', fontweight='bold', fontsize=12)
    ax.set_zlabel('X(t + 2τ)', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close after saving to avoid display issues
        print(f"  Saved: {save_path}")
    
    return fig


def plot_recovery_curves(time: np.ndarray, distances: np.ndarray,
                          thresholds: Dict, perturbations: Dict, metrics: Dict,
                          save_path: Optional[str] = None):
    """
    Plot recovery curves showing distance evolution and stability zones.
    
    Parameters
    ----------
    time : np.ndarray
        Time vector
    distances : np.ndarray
        Euclidean distances
    thresholds : dict
        Torus thresholds
    perturbations : dict
        Perturbation timings
    metrics : dict
        Recovery metrics
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    T1, T2, T3 = thresholds['T1'], thresholds['T2'], thresholds['T3']
    zones = thresholds['zones']
    
    for i, (ax, pert_name) in enumerate(zip(axes, ['slow', 'fast'])):
        # Distance curve
        ax.plot(time, distances, 'k-', linewidth=1, alpha=0.7, label='Distance to Reference')
        
        # Thresholds
        ax.axhline(T1, color='green', linestyle='--', linewidth=2, 
                   alpha=0.7, label='T1 (Very Stable)')
        ax.axhline(T2, color='orange', linestyle='--', linewidth=2,
                   alpha=0.7, label='T2 (Stable)')
        ax.axhline(T3, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='T3 (Unstable)')
        
        # Perturbation timing
        pert_time = perturbations['perturbation_times'][i]
        if not np.isnan(pert_time):
            ax.axvline(pert_time, color='purple', linestyle=':', linewidth=2,
                      label=f'Perturbation ({perturbations["perturbation_labels"][i]})')
            
            # Peak and recovery markers
            if pert_name in metrics and not np.isnan(metrics[pert_name]['peak_value']):
                peak_idx = metrics[pert_name]['peak_idx']
                peak_time = time[peak_idx]
                peak_val = metrics[pert_name]['peak_value']
                
                ax.plot(peak_time, peak_val, 'r*', markersize=15,
                       label=f'Peak ({peak_val:.1f} mm)')
                
                if not np.isnan(metrics[pert_name]['recovery_time']):
                    recovery_idx = metrics[pert_name]['recovery_idx']
                    recovery_time = time[recovery_idx]
                    recovery_val = distances[recovery_idx]
                    
                    ax.plot(recovery_time, recovery_val, 'g*', markersize=15,
                           label=f'Recovery ({metrics[pert_name]["recovery_time"]:.2f}s)')
        
        ax.set_ylabel('Distance (mm)', fontweight='bold')
        ax.set_title(f'{perturbations["perturbation_labels"][i]} Perturbation Recovery',
                     fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time (s)', fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_group_comparison(summary_table: list, metric: str = 'recovery_time',
                          save_path: Optional[str] = None):
    """
    Plot boxplot comparison between Young and Aging groups.
    
    Parameters
    ----------
    summary_table : list of dict
        Summary data for all participants
    metric : str
        Metric to plot ('recovery_time' or 'peak_deviation')
    save_path : str, optional
        Path to save figure
    """
    import pandas as pd
    
    df = pd.DataFrame(summary_table)
    
    # Prepare data for plotting
    data_slow = df[['Group', 'Slow_RecoveryTime' if metric == 'recovery_time' 
                     else 'Slow_PeakDeviation']].copy()
    data_slow.columns = ['Group', 'Value']
    data_slow['Perturbation'] = 'Slow'
    
    data_fast = df[['Group', 'Fast_RecoveryTime' if metric == 'recovery_time'
                     else 'Fast_PeakDeviation']].copy()
    data_fast.columns = ['Group', 'Value']
    data_fast['Perturbation'] = 'Fast'
    
    data_combined = pd.concat([data_slow, data_fast], ignore_index=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=data_combined, x='Perturbation', y='Value', hue='Group',
                palette={'Young': 'skyblue', 'Aging': 'salmon'}, ax=ax)
    
    ylabel = 'Recovery Time (s)' if metric == 'recovery_time' else 'Peak Deviation (mm)'
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_xlabel('Perturbation Type', fontweight='bold', fontsize=12)
    ax.set_title(f'{ylabel} - Group Comparison', fontweight='bold', fontsize=14)
    ax.legend(title='Group', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def generate_all_figures(tau: int, dim: int, ami_values: np.ndarray, fnn_rates: np.ndarray,
                          time: np.ndarray, state_space: np.ndarray, distances: np.ndarray,
                          reference: Dict, thresholds: Dict, perturbations: Dict,
                          metrics: Dict, participant_id: str,
                          output_dir: str = 'results/figures'):
    """
    Generate complete set of figures for one participant.
    
    Parameters
    ----------
    [All parameters from previous functions]
    output_dir : str
        Directory to save figures
    
    Returns
    -------
    dict
        Paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating figures for participant {participant_id}...")
    
    saved_files = {}
    
    # 1. TDE parameters
    path = output_dir / f'{participant_id}_tde_parameters.png'
    plot_ami_fnn(ami_values, fnn_rates, tau, dim, save_path=path)
    saved_files['tde_params'] = path
    plt.close()
    
    # 2. 3D state-space
    path = output_dir / f'{participant_id}_state_space_3d.png'
    plot_state_space_3d(state_space, reference, perturbations, 
                        title=f'Participant {participant_id} - State Space',
                        save_path=path)
    saved_files['state_space'] = path
    plt.close()
    
    # 3. Recovery curves
    path = output_dir / f'{participant_id}_recovery_curves.png'
    plot_recovery_curves(time, distances, thresholds, perturbations, metrics,
                         save_path=path)
    saved_files['recovery'] = path
    plt.close()
    
    print(f"✓ Generated {len(saved_files)} figures")
    
    return saved_files


# Example usage
if __name__ == "__main__":
    print("Visualization module loaded.")
    print("\nExample usage:")
    print(">>> figures = generate_all_figures(tau, dim, ami, fnn, time, ...)")
