"""
export_phase_space.py
=====================
Export phase space data to CSV for torus_3d.R visualization.
Generates two independent CSVs: one per perturbation (SLOW, FAST),
each with its own baseline and torus thresholds.

Usage:
    python sources/python/export_phase_space.py

Author: Victor SALVAT
Date: 2026-03-30
"""

import numpy as np
import pandas as pd
import h5py
import warnings
from pathlib import Path
from scipy.signal import decimate
import sys

# ── USER SETTINGS ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
MAT_FILE    = ROOT / 'data' / '004_Cued.mat'
PERT_SLOW_S = 174.0
PERT_FAST_S = 355.0
PERT_DUR_S  = 5.0
FS          = 400
FS_DOWN     = 100
DIM         = 3
N_SUBSAMPLE = 500
OUTPUT_DIR  = ROOT / 'results' / 'figures'
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / 'sources' / 'python'))
from time_delay_embedding import auto_tde_parameters, phase_space_reconstruction
from state_space import create_reference_trajectory, compute_euclidean_distances, compute_torus_thresholds

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Chargement sacrum Z ────────────────────────────────────────────────────────
print('Loading .mat file...')
with h5py.File(MAT_FILE, 'r') as f:
    signals = []
    for name in ['Dos01', 'Dos02', 'Dos03', 'Dos04']:
        value    = f['pMOCAP/MOCAP/MFilter'][name]['value'][:]
        occluded = f['pMOCAP/MOCAP/MFilter'][name]['occluded'][:].squeeze().astype(bool)
        z = value[2, :].copy()
        z[occluded] = np.nan
        signals.append(z)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    sacrum_z = np.nanmean(signals, axis=0)

nan_mask = ~np.isfinite(sacrum_z)
if nan_mask.sum() > 0:
    sacrum_z[nan_mask] = np.interp(
        np.where(nan_mask)[0],
        np.where(~nan_mask)[0],
        sacrum_z[~nan_mask]
    )

# ── Décimation 400 → 100 Hz ────────────────────────────────────────────────────
sacrum_clean = decimate(sacrum_z, FS // FS_DOWN, zero_phase=True)
n_samples    = len(sacrum_clean)
duration_s   = n_samples / FS_DOWN
print(f'  Signal: {n_samples} samples @ {FS_DOWN} Hz = {duration_s:.1f}s')

# ── TDE (calculé une seule fois) ───────────────────────────────────────────────
tau, _ = auto_tde_parameters(sacrum_clean, max_lag=100, max_dim=10)
print(f'  tau={tau}, dim={DIM} (forced)')

state_space = phase_space_reconstruction(sacrum_clean, tau=tau, dim=DIM)
t_ss        = np.arange(len(state_space)) / FS_DOWN

# ── Helpers ────────────────────────────────────────────────────────────────────
def subsample(arr, n_max):
    if len(arr) <= n_max:
        return arr
    idx = np.round(np.linspace(0, len(arr) - 1, n_max)).astype(int)
    return arr[idx]

def append_segment(rows, pts, label):
    for p in pts:
        rows.append({'x': p[0], 'y': p[1], 'z': p[2], 'type': label})

def make_torus_ellipse(ref_traj, scale, n_theta=60):
    rows  = []
    theta = np.linspace(0, 2 * np.pi, n_theta)
    step  = max(1, len(ref_traj) // 600)
    for i in range(step, len(ref_traj) - step, step):
        center = ref_traj[i]
        T = ref_traj[i + step] - ref_traj[i - step]
        norm = np.linalg.norm(T)
        if norm < 1e-10:
            continue
        T /= norm
        perp = np.array([0, 0, 1]) if abs(T[2]) < 0.9 else np.array([1, 0, 0])
        N = np.cross(T, perp);  N /= np.linalg.norm(N)
        B = np.cross(T, N)
        for t in theta:
            rows.append(center + scale * (np.cos(t) * N + np.sin(t) * B))
    return np.array(rows) if rows else np.empty((0, 3))

# ── Fonction principale : export pour une perturbation ────────────────────────
def export_perturbation(pert_label, t_baseline_start, t_baseline_end,
                        t_pert_start, pert_color_label, output_csv):
    """
    Génère un CSV pour une perturbation avec sa propre baseline et ses seuils.

    Args:
        pert_label        : 'SLOW' ou 'FAST'
        t_baseline_start  : début baseline en secondes
        t_baseline_end    : fin baseline en secondes (= début perturbation)
        t_pert_start      : début perturbation en secondes
        pert_color_label  : label dans le CSV ('XR_slow' ou 'XR_fast')
        output_csv        : Path du fichier CSV de sortie
    """
    print(f'\n── Exporting {pert_label} ──')

    baseline_mask = (t_ss >= t_baseline_start) & (t_ss < t_baseline_end)
    pre_pert_idx  = int(t_baseline_end * FS_DOWN) - (DIM - 1) * tau

    print(f'  Baseline: {t_baseline_start:.0f}s → {t_baseline_end:.0f}s '
          f'({baseline_mask.sum()} points)')

    # Trajectoire de référence et seuils propres à cette baseline
    ref        = create_reference_trajectory(state_space, baseline_end_idx=pre_pert_idx)
    distances  = compute_euclidean_distances(state_space, ref, phase_matching=False)
    thresholds = compute_torus_thresholds(distances, baseline_mask)

    print(f'  T1={thresholds["T1"]:.1f}mm  T2={thresholds["T2"]:.1f}mm  '
          f'T3={thresholds["T3"]:.1f}mm')

    ss_rot = (state_space[:, :3] - ref['centroid']) @ ref['rotation_matrix'].T

    rows = []

    # Trajectoire de référence
    from scipy.ndimage import uniform_filter1d
    bl_smooth = ss_rot[baseline_mask]
    # Lissage fort pour obtenir la trajectoire "moyenne"
    window = max(1, len(bl_smooth) // 20)
    for axis in range(3):
        bl_smooth[:, axis] = uniform_filter1d(bl_smooth[:, axis], size=window)
    append_segment(rows, subsample(bl_smooth, N_SUBSAMPLE), 'RefTraj')

    # Ellipses (T1 visible, T2/T3 discrètes)
    for label, T in [('Ellipse1', thresholds['T1']),
                     ('Ellipse2', thresholds['T2']),
                     ('Ellipse3', thresholds['T3'])]:
        pts = make_torus_ellipse(ref['trajectory'], T)
        append_segment(rows, subsample(pts, N_SUBSAMPLE), label)

    # Baseline
    append_segment(rows, subsample(ss_rot[baseline_mask], N_SUBSAMPLE), 'XR_baseline')

    # Perturbation
    pert_mask = (t_ss >= t_pert_start) & (t_ss < t_pert_start + PERT_DUR_S)
    if pert_mask.sum() > 0:
        append_segment(rows, subsample(ss_rot[pert_mask], N_SUBSAMPLE), pert_color_label)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, sep=';', index=False)
    print(f'  ✓ Saved: {output_csv.name} ({len(df_out)} rows)')
    print(f'  Types: {df_out["type"].value_counts().to_dict()}')

# ── Export SLOW : baseline [0s → 174s] ────────────────────────────────────────
export_perturbation(
    pert_label       = 'SLOW',
    t_baseline_start = 0.0,
    t_baseline_end   = PERT_SLOW_S,
    t_pert_start     = PERT_SLOW_S,
    pert_color_label = 'XR_slow',
    output_csv       = OUTPUT_DIR / 'phase_space_SLOW.csv'
)

# ── Export FAST : baseline [174s → 355s] ──────────────────────────────────────
export_perturbation(
    pert_label       = 'FAST',
    t_baseline_start = PERT_SLOW_S,
    t_baseline_end   = PERT_FAST_S,
    t_pert_start     = PERT_FAST_S,
    pert_color_label = 'XR_fast',
    output_csv       = OUTPUT_DIR / 'phase_space_FAST.csv'
)

print('\n✓ Done.')