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
    ax2.axhline(y=1, color='orange', linestyle='--', linewidth=2,  # MODIFIÉ : 10 → 1
                alpha=0.7, label='Threshold (1%)')                  # MODIFIÉ : 10% → 1%
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
    from scipy.signal import savgol_filter
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Recovery Dynamics Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    perturbations_list = [
        ('slow_time', 'slow', 'Slow Perturbation Recovery', 0),
        ('fast_time', 'fast', 'Fast Perturbation Recovery', 1)
    ]
    
    for pert_key, metrics_key, title, idx in perturbations_list:
        ax = axes[idx]
        pert_time = perturbations[pert_key]
        
        if np.isnan(pert_time):
            ax.text(0.5, 0.5, f'No {pert_key.split("_")[0]} perturbation detected',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            continue
        
        window_before = 30
        window_after = 70
        start_idx = max(0, int((pert_time - window_before) * 100))
        end_idx = min(len(distances), int((pert_time + window_after) * 100))
        
        distances_window = distances[start_idx:end_idx]
        time_window = time[start_idx:end_idx]
        
        window_length = min(51, len(distances_window) if len(distances_window) % 2 == 1 else len(distances_window) - 1)
        if window_length < 5:
            window_length = 5
        distances_smooth = savgol_filter(distances_window, window_length=window_length, polyorder=3)
        
        ax.plot(time_window, distances_smooth, 'k-', linewidth=2, label='Distance to Reference', zorder=3)
        
        ax.axhline(thresholds['T1'], color='green', linestyle='--', linewidth=2.5, 
                   label='T1 (Very Stable)', alpha=0.8, zorder=2)
        ax.axhline(thresholds['T2'], color='orange', linestyle='--', linewidth=2.5, 
                   label='T2 (Stable)', alpha=0.8, zorder=2)
        ax.axhline(thresholds['T3'], color='red', linestyle='--', linewidth=2.5, 
                   label='T3 (Unstable)', alpha=0.8, zorder=2)
        
        ax.axvline(pert_time, color='purple', linestyle=':', linewidth=2, alpha=0.6, zorder=1)
        
        m = metrics[metrics_key]
        recovery_time = m['recovery_time']
        
        peak_idx = np.argmax(distances_smooth)
        max_distance = distances_smooth[peak_idx]
        max_time = time_window[peak_idx]
        
        ax.plot(max_time, max_distance, 'r*', markersize=20, markeredgewidth=2, 
                markeredgecolor='darkred', label=f'Perturbation ({metrics_key.capitalize()})', zorder=5)
        ax.text(max_time, max_distance + 4, f'Peak\n({max_distance:.1f}mm)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.9))
        
        if recovery_time == 0:
            ax.text(pert_time + 10, thresholds['T1'] + 2, 'Instant Recovery', 
                    ha='left', va='bottom', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', 
                             alpha=0.9, linewidth=2))
        else:
            recovery_idx_rel = np.where((time_window > max_time) & (distances_smooth <= thresholds['T1']))[0]
            if len(recovery_idx_rel) > 0:
                recovery_idx = recovery_idx_rel[0]
                recovery_point_time = time_window[recovery_idx]
                recovery_point_dist = distances_smooth[recovery_idx]
                
                arrow_y = max_distance * 0.65
                ax.annotate('', xy=(recovery_point_time, arrow_y),
                           xytext=(max_time, arrow_y),
                           arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=3, alpha=0.9),
                           zorder=4)
                
                mid_time = (max_time + recovery_point_time) / 2
                ax.text(mid_time, arrow_y - 3, f'Recovery {recovery_time:.2f}s', 
                       ha='center', va='top', fontsize=11, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', 
                                edgecolor='green', alpha=0.95, linewidth=2),
                       zorder=4)
                
                ax.plot(recovery_point_time, recovery_point_dist, 'g*', markersize=20, 
                        markeredgewidth=2, markeredgecolor='darkgreen', 
                        label=f'Recovery ({recovery_time:.2f}s)', zorder=6)
                
                ax.axvline(recovery_point_time, color='green', linestyle=':', 
                          linewidth=1.5, alpha=0.5, zorder=1)
        
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Distance (mm)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black', 
                  markerscale=0.8, handletextpad=0.8, labelspacing=0.6)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#f0f0f0')
        
        y_max = max(max_distance, thresholds['T3']) * 1.15
        ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
