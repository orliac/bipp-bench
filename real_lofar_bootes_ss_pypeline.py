# #############################################################################
# lofar_bootes_ss.py
# ==================
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Real-data LOFAR imaging with Bluebild (StandardSynthesis).
"""

import os
import sys
import time
from tqdm import tqdm as ProgressBar
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import bluebild
import pypeline.phased_array.bluebild.data_processor as bb_dp
import pypeline.phased_array.bluebild.gram as bb_gr
import pypeline.phased_array.bluebild.imager.spatial_domain as bb_sd
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
from pypeline.util import frame
import bipptb



hdul = fits.open('test02-dirty.fits')
hdul.info()
data = hdul[0].data
print("data.shape =", data.shape)
print("data =", data)
#sys.exit(0)

# Setting up the benchmark
args = bipptb.check_args(sys.argv)
print("-I- args =", args)
proc_unit = args.processing_unit
precision = args.precision
N_levels  = args.nlev
out_dir   = args.outdir
print("-I- Command line input -----------------------------")
print("-I- precision     =", precision)
print("-I- N. stations   =", args.nsta)
print("-I- proc unit     =", proc_unit)
print("-I- N. pix        =", args.pixw)
print("-I- N_levels      =", N_levels)
print("-I- outdir        =", out_dir)
print("-I- WSClean scale =", args.wsc_scale)
print("-I- Time slice PE =", args.time_slice_pe)
print("-I- Time slice IM =", args.time_slice_im)
print("-I- sigma         =", args.sigma)
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


# Compute field of view from args
FoV = Angle(args.pixw * args.wsc_scale * u.arcsec)
print("Fov.rad =", FoV.rad)
print("Fov.deg =", FoV.deg)

# Instrument
ms_file = "gauss4_t201806301100_SBL180.MS"
ms = measurement_set.LofarMeasurementSet(ms_file, args.nsta)
print(f"ms.field_center = {ms.field_center}")
gram = bb_gr.GramBlock(ctx)

# Observation
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
print(f"frequency = {frequency}")

wl = constants.speed_of_light / frequency.to_value(u.Hz)
sky_model = source.from_tgss_catalog(ms.field_center, FoV.rad, N_src=20)

# Imaging
#_, _, px_colat, px_lon = grid.equal_angle(
#    N=ms.instrument.nyquist_rate(wl), direction=ms.field_center.cartesian.xyz.value, FoV=FoV.rad
#)
#px_grid = transform.pol2cart(1, px_colat, px_lon)#.reshape(3, -1)
lim = np.sin(FoV.rad / 2)
grid_slice = np.linspace(-lim, lim, args.pixw)
l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
uvw_frame = frame.uvw_basis(ms.field_center)
px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
px_w = px_grid.shape[1]
px_h = px_grid.shape[2]
print("-I- px_grid.shape =", px_grid.shape)


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma)
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_pe), column="DATA"):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)
    print("S.shape =", S.shape)
    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
print("-I- IFPE N_eig =", N_eig)
print("-I- c_centroid =\n", c_centroid)
ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

# Imaging
ifim_s = time.time()
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, args.nlev, N_bits, ctx)
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="DATA"):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)
    print("S.shape =", S.shape)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
I_std, I_lsq = I_mfs.as_image()
print("I_lsq.shape =", I_lsq.shape)
print("I_std.shape =", I_std.shape)
ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec")

I_est = None
I_dp  = None
I_mfs = None


### Sensitivity Field =========================================================

# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=args.sigma)
for t in ms.time["TIME"][::args.time_slice_im]:
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    G = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
print("-I- SFPE N_eig =", N_eig)
sfpe_e = time.time()
print(f"#@#SFPE {sfpe_e - sfpe_s:.3f} sec")

# Imaging
sfim_s = time.time()
S_dp  = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx)
S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, N_bits, ctx)
for t, f, S in  ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="DATA"):
    wl = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)
    D, V = S_dp(XYZ, W, wl)
    _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
_, S = S_mfs.as_image()
sfim_e = time.time()
print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")

#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)

I_std_eq = s2image.Image(I_std.data / S.data, I_std.grid)
I_lsq_eq = s2image.Image(I_lsq.data / S.data, I_lsq.grid)

I_std_eq.to_fits('bluebild_ss_gauss4_std_eq.fits')
I_lsq_eq.to_fits('bluebild_ss_gauss4_lsq_eq.fits')

print("######################################################################")
print("-I- c_centroid =\n", c_centroid, "\n")
print("-I- px_grid:", px_grid.shape, "\n", px_grid, "\n")
print("-I- I_lsq:\n", I_lsq.data, "\n")
print("-I- I_lsq_eq:\n", I_lsq_eq.data, "\n")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

bipptb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.outdir)
bipptb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.outdir)

#bipptb.dump_json((ifpe_e - ifpe_s), ifpe_vis, (ifim_e - ifim_s), ifim_vis,
#                 (sfpe_e - sfpe_s), (sfim_e - sfim_s),
#                 'stats.json', out_dir)

# Plot Results ================================================================
fig, ax = plt.subplots(ncols=2)

#I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax[0])
#ax[0].set_title("Bluebild Standardized Image")

print("I_lsq_eq.data.shape =", I_lsq_eq.data.shape)
for i in range(0, 1):#I_lsq_eq.data.shape[0]):
    print(np.min(I_lsq_eq.data[i,:,:]), np.max(I_lsq_eq.data[i,:,:]))
    im0 = ax[0].imshow(I_lsq_eq.data[i,:,:] / np.max(I_lsq_eq.data[i,:,:]))
    plt.colorbar(im0, ax=ax[0])
    ax[0].set_title("Bluebild Least-Squares Image")

    print(np.min(data[0,0,:,:]), np.max(data[0,0,:,:]))
    im1 = ax[1].imshow(data[0,0,:,:] / np.max(data[0,0,:,:]))
    ax[1].set_title("WSClean dirty image")
    plt.colorbar(im1, ax=ax[1])

    fig.savefig('bb_gauss4_'+ str(i) + '.png')
