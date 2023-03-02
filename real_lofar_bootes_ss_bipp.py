
"""
LOFAR imaging with BIPP SS from MS dataset
"""

import os
import sys
import time
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import bipp
import bipp.gram as bb_gr
import bipp.parameter_estimator as bb_pe
import bipp.measurement_set as measurement_set
import bipp.frame as frame
import bipp.imot_tools.io.s2image as s2image
import bipptb


# Setting up the benchmark
args = bipptb.check_args(sys.argv)

# For reproducible results
np.random.seed(0)

# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context(args.processing_unit)

jkt0_s = time.time()

# Compute field of view from args
FoV = Angle(args.pixw * args.wsc_scale * u.arcsec)
print("Fov.rad =", FoV.rad)
print("Fov.deg =", FoV.deg)

# Instrument
if args.ms_file == None:
    raise Exception("args.ms_file not set.")
if not os.path.exists(args.ms_file):
    raise Exception(f"Could not open {args.ms_file}")
ms = measurement_set.LofarMeasurementSet(args.ms_file, args.nsta, station_only=False)
print(f"-I- ms.field_center = {ms.field_center}")
gram = bb_gr.GramBlock(ctx)

# Observation
channel_ids = [0]

# Grids
lmn_grid, xyz_grid = frame.make_grids(args.pixw, FoV.rad, ms.field_center)
px_w = xyz_grid.shape[1]
px_h = xyz_grid.shape[2]
xyz_grid = xyz_grid.reshape(3, -1)
print("-I- lmd_grid.shape =", lmn_grid.shape)
print("-I- xyz_grid.shape =", xyz_grid.shape, "(after reshaping in SS)")

### Intensity Field ===========================================================

# Parameter Estimation
ifpe_vis = 0
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma, ctx=ctx)
for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=slice(None, None, args.time_slice_pe), column="DATA"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    G    = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)
    I_est.collect(S, G)
N_eig, intensity_intervals = I_est.infer_parameters()
print("-I- N_eig =", N_eig)
print("-I- intensity intervals =\n", intensity_intervals)
print("-I- XYZ.shape =", XYZ.shape)

ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

# EO
I_est = None
N_antenna, N_station = W.shape
print("-I- N_antenna =", N_antenna)
print("-I- N_station =", N_station)

# Imaging
ifim_s = time.time()
ifim_vis = 0
imager = bipp.StandardSynthesis(
    ctx,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    ["LSQ", "STD"],
    xyz_grid[0],
    xyz_grid[1],
    xyz_grid[2],
    args.precision)

i_it = 0
for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=slice(None, None, args.time_slice_im), column="DATA"):
    t_it = time.time()
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    S, _ = measurement_set.filter_data(S, W)
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, S.data)
    t_it = time.time() - t_it
    print(f" ... ifim t_it {i_it} {t_it:.3f} sec")
    i_it += 1
I_lsq = imager.get("LSQ").reshape((-1, px_w, px_h))
I_std = imager.get("STD").reshape((-1, px_w, px_h))
ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec")

#print("I_lsq", I_lsq.shape, "\n", I_lsq)
#print("I_std", I_std.shape, "\n", I_std)

#sys.exit(0)

### Sensitivity Field =========================================================
# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=args.sigma, ctx=ctx)
for t, f, _ in ms.visibilities(channel_id=channel_ids, time_id=slice(None, None, args.time_slice_pe), column="NONE"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W   = ms.beamformer(XYZ, wl)
    G   = gram(XYZ, W, wl)
    S_est.collect(G)
N_eig = S_est.infer_parameters()
print("-I- SFPE N_eig =", N_eig)
sfpe_e = time.time()
print(f"#@#SFPE {sfpe_e - sfpe_s:.3f} sec")

# Imaging
sfim_s = time.time()
sensitivity_intervals = np.array([[0, np.finfo("f").max]])
imager = None  # release previous imager first to some additional memory
imager = bipp.StandardSynthesis(
    ctx,
    N_antenna,
    N_station,
    sensitivity_intervals.shape[0],
    ["INV_SQ"],
    xyz_grid[0],
    xyz_grid[1],
    xyz_grid[2],
    args.precision,
)
for t, f, _ in ms.visibilities(channel_id=channel_ids, time_id=slice(None, None, args.time_slice_im), column="NONE"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data)

sensitivity_image = imager.get("INV_SQ").reshape((-1, px_w, px_h))

print("-I- sensitivity_image.shape =", sensitivity_image.shape)

I_std_eq = s2image.Image(I_std / sensitivity_image, xyz_grid.reshape(3, px_w, px_h))
I_lsq_eq = s2image.Image(I_lsq / sensitivity_image, xyz_grid.reshape(3, px_w, px_h))

sfim_e = time.time()
print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")

#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)


print("######################################################################")
print("-I- intensity_intervals =\n", intensity_intervals, "\n")
print("-I- xyz_grid:", xyz_grid.shape, "\n", xyz_grid, "\n")
print("-I- I_lsq:\n", I_lsq, "\n")
print("-I- I_lsq_eq:\n", I_lsq_eq.data, "\n")

bipptb.dump_data(I_lsq_eq.data, 'I_lsq_eq_data', args.output_directory)
bipptb.dump_data(I_lsq_eq.grid, 'I_lsq_eq_grid', args.output_directory)

bipptb.dump_json((ifpe_e - ifpe_s), ifpe_vis, (ifim_e - ifim_s), ifim_vis,
                 (sfpe_e - sfpe_s), (sfim_e - sfim_s),
                 'stats.json', args.output_directory)

"""
### Plot results
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'BIPP LSQ, sensitivity-corrected image (NEW SS)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_lsq.png"
if out_dir: fp = os.path.join(out_dir, fp)
plt.savefig(fp)


plt.figure()
ax = plt.gca()
I_std_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'BIPP STD, sensitivity-corrected image (NEW SS)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_std.png"
if out_dir: fp = os.path.join(out_dir, fp)
plt.savefig(fp)
"""
