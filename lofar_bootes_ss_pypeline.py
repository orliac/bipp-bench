# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Simulated LOFAR imaging with pypeline (StandardSynthesis).
"""

import os
import sys
import time
import astropy.units as u
import astropy.coordinates as coord
import astropy.time as atime
import imot_tools.io.s2image as s2image
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import bluebild
from imot_tools.io.plot import cmap
import pypeline.phased_array.beamforming as beamforming
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.instrument as instrument
import imot_tools.math.sphere.transform as transform
import pypeline.phased_array.data_gen.statistics as statistics
from pypeline.util import frame
from mpl_toolkits.mplot3d import Axes3D
import imot_tools.io.s2image as im
import imot_tools.io.plot as implt

import bipptb

# Setting up the benchmark
args = bipptb.check_args(sys.argv)
print("-I- args =", args)
proc_unit = args.processing_unit
precision = args.precision
N_station = args.nsta
N_pix     = args.pixw
N_levels  = args.nlev
out_dir   = args.outdir 
print("-I- Command line input -----------------------------")
print("-I- precision =", precision)
print("-I- N_station =", N_station)
print("-I- proc unit =", proc_unit)
print("-I- N_pix     =", N_pix)
print("-I- N_levels  =", N_levels)
print("-I- outdir    =", out_dir)
print("-I- ------------------------------------------------")

N_bits   = 32 if args.precision == 'single' else 64
dtype_f  = np.float32 if N_bits == 32 else np.float64

# For reproducible results
np.random.seed(0)

# Create context with selected processing unit.
# Options are "NONE", "AUTO", "CPU" and "GPU".
bb_proc_unit = None
if proc_unit == 'cpu':
    bb_proc_unit = bluebild.ProcessingUnit.CPU
elif proc_unit == 'gpu':
    bb_proc_unit = bluebild.ProcessingUnit.GPU
ctx = None if proc_unit == 'none' else bluebild.Context(bb_proc_unit)

jkt0_s = time.time()

# Observation
obs_start = atime.Time(56879.54171302732, scale="utc", format="mjd")
field_center = coord.SkyCoord(ra=218 * u.deg, dec=34.5 * u.deg, frame="icrs")
FoV_deg = 8.0
FoV, frequency = np.deg2rad(FoV_deg), 145e6
wl = constants.speed_of_light / frequency

# Instrument
dev = instrument.LofarBlock(N_station)
mb_cfg = [(_, _, field_center) for _ in range(N_station)]
mb = beamforming.MatchedBeamformerBlock(mb_cfg)
gram = bb_gr.GramBlock(ctx)

# Data generation
T_integration = 8
N_src         = 40
fs            = 196000
SNR           = 30
sky_model = source.from_tgss_catalog(field_center, FoV, N_src=N_src)
vis       = statistics.VisibilityGeneratorBlock(
    sky_model, T_integration, fs=fs, SNR=SNR
)
times     = obs_start + (T_integration * u.s) * np.arange(3595)
N_antenna = dev(times[0]).data.shape[0]

# Imaging parameters
time_slice = 2000
times = times[::time_slice]
SIGMA = 1.0

# Grids
lim = np.sin(FoV / 2)
grid_slice = np.linspace(-lim, lim, N_pix)
l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
uvw_frame = frame.uvw_basis(field_center)
px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
px_w = px_grid.shape[1]
px_h = px_grid.shape[2]

print(f"-I- sigma = {SIGMA:.3f}")
print(f"-I- N_bits {N_bits}")
print(f"-I- px_w = {px_w:d}, px_h = {px_h:d}")
print(f"-I- N_antenna = {N_antenna:d}")
print(f"-I- T_integration =", T_integration)
print(f"-I- Field center  =", field_center)
print(f"-I- Field of view =", FoV_deg, "deg")
print(f"-I- frequency =", frequency)
print(f"-I- SNR =", SNR)
print(f"-I- fs =", fs)
print(f"-I- time_slice =", time_slice)
print(f"-I- OMP_NUM_THREADS =", os.getenv('OMP_NUM_THREADS'))


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
ifpe_vis = 0
I_est = bb_pe.IntensityFieldParameterEstimator(N_levels, sigma=SIGMA)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    t_ss = time.time()
    S = vis(XYZ, W, wl)
    ifpe_vis += (time.time() - t_ss)
    G = gram(XYZ, W, wl)
    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
print("-I- IFPE N_eig =", N_eig)
ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

# Imaging
ifim_s = time.time()
ifim_vis = 0
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx) #EO: bug in C++ version???
#I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx=None)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, N_levels, N_bits, ctx)
i_it = 0
for t in times:
    t_it = time.time()
    d2h = True if t == times[-1] else False
    XYZ = dev(t)
    W = mb(XYZ, wl)
    t_ss = time.time()
    S = vis(XYZ, W, wl)
    ifim_vis += (time.time() - t_ss)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    I_mfs(D, V, XYZ.data, W.data, c_idx, d2h)
    t_it = time.time() - t_it
    print(f" ... ifim t_it {i_it} {t_it:.3f} sec")
    i_it += 1
I_std, I_lsq = I_mfs.as_image()
print("I_lsq.shape =", I_lsq.shape)
print("I_std.shape =", I_std.shape)
ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec")


### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=SIGMA)
for t in times:
    XYZ = dev(t)
    W = mb(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
sfpe_e = time.time()
print(f"#@#SFPE {sfpe_e - sfpe_s:.3f} sec")

# Imaging
sfim_s = time.time()
S_dp  = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits, ctx)
i_it = 0
for t in times:
    t_it = time.time()
    XYZ = dev(t)
    W = mb(XYZ, wl)
    D, V = S_dp(XYZ, W, wl)
    UVW_baselines_t = dev.baselines(t, uvw=True, field_center=field_center)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    t_it = time.time() - t_it
    print(f" ... sfim t_it {i_it} {t_it:.3f} sec")
    i_it += 1
_, S_ss = S_mfs.as_image()
I_std_eq = s2image.Image(I_std.data / S_ss.data, I_lsq.grid)
I_lsq_eq = s2image.Image(I_lsq.data / S_ss.data, I_lsq.grid)
sfim_e = time.time()
print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")


#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)

print("######################################################################")
print("-I- c_centroid =\n", c_centroid, "\n")
print("-I- px_grid:", px_grid.shape, "\n", px_grid, "\n")
print("-I- I_lsq:\n", I_lsq.data, "\n")
print("-I- I_lsq_eq:\n", I_lsq_eq.data, "\n")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

bipptb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.outdir)
bipptb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.outdir)

bipptb.dump_json((ifpe_e - ifpe_s), ifpe_vis, (ifim_e - ifim_s), ifim_vis,
                 (sfpe_e - sfpe_s), (sfim_e - sfim_s),
                 'stats.json', out_dir)

### Plotting section
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild least-squares, sensitivity-corrected image (SS)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.')
fp = "I_lsq.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)


plt.figure()
ax = plt.gca()
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'Bluebild std, sensitivity-corrected image (SS)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {np.round(FoV * 180 / np.pi)} degrees.')
fp = "I_std.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)

