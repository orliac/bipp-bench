"""
Real-data LOFAR imaging with Bluebild (NUFFT).
"""

import os
import sys
import time
import re
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
import pypeline.phased_array.bluebild.parameter_estimator as bb_pe
import pypeline.phased_array.data_gen.source as source
import pypeline.phased_array.measurement_set as measurement_set
import pypeline.phased_array.bluebild.imager.fourier_domain as bb_im
from pypeline.util import frame
import bipptb


# Setting up the benchmark
args = bipptb.check_args(sys.argv)

# For reproducible results
np.random.seed(0)

# Create context with selected processing unit.
# Options are "NONE", "AUTO", "CPU" and "GPU".
bb_proc_unit = None
if args.processing_unit == 'cpu':
    bb_proc_unit = bluebild.ProcessingUnit.CPU
elif args.processing_unit == 'gpu':
    bb_proc_unit = bluebild.ProcessingUnit.GPU
ctx = None if args.processing_unit == 'none' else bluebild.Context(bb_proc_unit)

jkt0_s = time.time()

# Compute field of view from args
FoV = Angle(args.pixw * args.wsc_scale * u.arcsec)
print("-I- Fov.rad =", FoV.rad)
print("-I- Fov.deg =", FoV.deg)

# Instrument
if args.ms_file == None:
    raise Exception("args.ms_file not set.")
if not os.path.exists(args.ms_file):
    raise Exception(f"Could not open {args.ms_file}")
ms = measurement_set.LofarMeasurementSet(args.ms_file, args.nsta, station_only=True)
print(f"-I- ms.field_center = {ms.field_center}")
gram = bb_gr.GramBlock(ctx)

# Observation (old fashion, single channel)
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
print(f"-I- frequency = {frequency}")
wl = constants.speed_of_light / frequency.to_value(u.Hz)

# Grids
lim = np.sin(FoV.rad / 2)
grid_slice = np.linspace(-lim, lim, args.pixw)
l_grid, m_grid = np.meshgrid(grid_slice, grid_slice)
n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
uvw_frame = frame.uvw_basis(ms.field_center)
px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
print("-I- lmn_grid.shape =", lmn_grid.shape)
print("-I- px_grid.shape  =", px_grid.shape)


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma)
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_pe), column="DATA"):
    wl_   = constants.speed_of_light / f.to_value(u.Hz)
    if wl_ != wl: sys.exit(1)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)
    G    = gram(XYZ, W, wl)
    I_est.collect(S, G)
N_eig, c_centroid = I_est.infer_parameters()
print("-I- N_eig =", N_eig)
print("-I- c_centroid =\n", c_centroid)
print("-I- XYZ.shape =", XYZ.shape)

ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

N_antenna, N_station = W.shape
print("-I- N_antenna =", N_antenna)
print("-I- N_station =", N_station)

# Imaging
n_vis_ifim = 0
ifim_s = time.time()
ifim_vis = 0
#EO: noise added by C++ I_dp. Setting ctx=None makes results consistent with Python solution
I_dp  = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx=ctx)
IV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq', 'sqrt'))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=args.pixw, FoV=FoV.rad,
                                      field_center=ms.field_center, eps=args.nufft_eps,
                                      n_trans=1, precision=args.precision, ctx=ctx)

i_it = 0
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="DATA"):
    t_it = time.time()
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
    W    = ms.beamformer(XYZ, wl)
    S, W = measurement_set.filter_data(S, W)
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    S_corrected = IV_dp(D, V, W, c_idx)
    nufft_imager.collect(UVW_baselines_t, S_corrected)
    n_vis = np.count_nonzero(S.data)
    n_vis_ifim += n_vis
    t_it = time.time() - t_it
    if i_it < 10: print(f" ... ifim t_it {i_it} {t_it:.3f} sec")
    i_it += 1
I_lsq, I_sqrt = nufft_imager.get_statistic()
ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec  NVIS {n_vis_ifim}")


### Sensitivity Field =========================================================

# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=args.sigma)
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_pe), column="DATA"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    _, W = measurement_set.filter_data(S, W)
    G    = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
print("-I- SFPE N_eig =", N_eig)
sfpe_e = time.time()
print(f"#@#SFPE {sfpe_e - sfpe_s:.3f} sec")

# Imaging
sfim_s = time.time()
S_dp  = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx)
SV_dp = bb_dp.VirtualVisibilitiesDataProcessingBlock(N_eig, filters=('lsq',))
nufft_imager = bb_im.NUFFT_IMFS_Block(wl=wl, grid_size=args.pixw, FoV=FoV.rad,
                                      field_center=ms.field_center, eps=args.nufft_eps,
                                      n_trans=1, precision=args.precision, ctx=ctx)
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="DATA"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    _, W = measurement_set.filter_data(S, W)
    D, V = S_dp(XYZ, W, wl)
    UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
    S_sensitivity = SV_dp(D, V, W, cluster_idx=np.zeros(N_eig, dtype=int))
    nufft_imager.collect(UVW_baselines_t, S_sensitivity)
sensitivity_image = nufft_imager.get_statistic()[0]
I_lsq_eq  = s2image.Image(I_lsq  / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
I_sqrt_eq = s2image.Image(I_sqrt / sensitivity_image, nufft_imager._synthesizer.xyz_grid)
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
print("-I- I_lsq:\n", I_lsq, "\n")
print("-I- I_lsq_eq:\n", I_lsq_eq.data, "\n")

print("-I- args.output_directory:", args.output_directory)

bipptb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.output_directory)
bipptb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.output_directory)

bipptb.dump_json({'ifpe_s': ifpe_s, 'ifpe_e': ifpe_e,
                  'ifim_s': ifim_s, 'ifim_e': ifim_e,
                  'sfpe_s': sfpe_s, 'sfpe_e': sfpe_e,
                  'sfim_s': sfim_s, 'sfim_e': sfim_e,
                  'n_vis_ifim': n_vis_ifim,
                  'filename': 'stats.json',
                  'out_dir': args.output_directory})
