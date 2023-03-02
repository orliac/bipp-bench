
"""
LOFAR imaging with BIPP NUFFT from MS dataset
"""

import os
import sys
import time
import re
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
ms = measurement_set.LofarMeasurementSet(args.ms_file, args.nsta, station_only=True)
print(f"-I- ms.field_center = {ms.field_center}")
gram = bb_gr.GramBlock(ctx)

# Observation
channel_id = 0
frequency = ms.channels["FREQUENCY"][channel_id]
print(f"-I- frequency = {frequency}")
wl = constants.speed_of_light / frequency.to_value(u.Hz)

# Grids
lmn_grid, xyz_grid = frame.make_grids(args.pixw, FoV.rad, ms.field_center)
px_w = xyz_grid.shape[1]
px_h = xyz_grid.shape[2]
print("-I- lmd_grid.shape =", lmn_grid.shape)
print("-I- xyz_grid.shape =", xyz_grid.shape)


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma, ctx=ctx)
for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_pe), column="DATA"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    W    = ms.beamformer(XYZ, wl)
    G    = gram(XYZ, W, wl)
    S, _ = measurement_set.filter_data(S, W)
    I_est.collect(S, G)
N_eig, intensity_intervals = I_est.infer_parameters()
print("-I- IFPE N_eig =", N_eig)
print("-I- intensity intervals =", intensity_intervals)
print("-I- XYZ.shape =", XYZ.shape)

ifpe_e = time.time()
print(f"#@#IFPE {ifpe_e - ifpe_s:.3f} sec")

# EO
I_est = None
N_antenna, N_station = W.shape
print("-I- N_antenna =", N_antenna)
print("-I- N_station =", N_station)

# Imaging
n_vis_ifim = 0
ifim_s = time.time()
imager = bipp.NufftSynthesis(
    ctx,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    ["LSQ", "SQRT"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    args.precision,
    args.nufft_eps)

for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="DATA"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
    W    = ms.beamformer(XYZ, wl)
    S, _ = measurement_set.filter_data(S, W)
    n_vis = np.count_nonzero(S.data)
    n_vis_ifim += n_vis
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)

I_lsq  = imager.get("LSQ").reshape((-1, args.pixw, args.pixw))
I_sqrt = imager.get("SQRT").reshape((-1, args.pixw, args.pixw))
ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec  NVIS {n_vis_ifim}")


### Sensitivity Field =========================================================

# Parameter Estimation
sfpe_s = time.time()
S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=args.sigma, ctx=ctx)
for t, f, _ in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="NONE"):
    wl  = constants.speed_of_light / f.to_value(u.Hz)
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
imager = bipp.NufftSynthesis(
    ctx,
    N_antenna,
    N_station,
    sensitivity_intervals.shape[0],
    ["INV_SQ"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    args.precision,
    args.nufft_eps)

for t, f, _ in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="NONE"):
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ = ms.instrument(t)
    W   = ms.beamformer(XYZ, wl)
    UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data, uvw, None)

sensitivity_image = imager.get("INV_SQ").reshape((-1, args.pixw, args.pixw))

I_lsq_eq  = s2image.Image(I_lsq  / sensitivity_image, xyz_grid)
I_sqrt_eq = s2image.Image(I_sqrt / sensitivity_image, xyz_grid)

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

bipptb.dump_json((ifpe_e - ifpe_s), 0.0, (ifim_e - ifim_s), 0.0,
                 (sfpe_e - sfpe_s), (sfim_e - sfim_s),
                 'stats.json', args.output_directory)

sys.exit(0)

# Plot Results ================================================================


if args.wsc_log:
    print("-I- got wsc_log ", args.wsc_log)

    hdul = fits.open('test02-dirty.fits')
    hdul.info()
    data = hdul[0].data
    print("data.shape =", data.shape)
    print("data =", data)


    lines = open(args.wsc_log, "r").readlines()
    for line in lines:
        line = line.strip()
        patt = "Total nr. of visibilities to be gridded:"
        if re.search(patt, line):
            wsc_totvis = line.split(patt)[-1]
        patt = "effective count after weighting:"
        if re.search(patt, line):
            wsc_gridvis = line.split(patt)[-1]

        if re.search("Inversion:", line):
            wsc_t_inv, wsc_t_pred, wsc_t_deconv = re.split("\s*Inversion:\s*|\s*,\s*prediction:\s*|\s*,\s*deconvolution:\s*", line)[-3:]
            print("wsc_times =", wsc_t_inv, wsc_t_pred, wsc_t_deconv)

from mpl_toolkits.axes_grid1 import make_axes_locatable


# Produce different images including additional energy levels
#
for nlev in range(1, I_lsq_eq.data.shape[0] + 1):

    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    plt.suptitle(f"bipp energy levels from 0 to {nlev - 1}, {n_vis_ifim} visibilities\n" +
                 f"WSClean visibilities: total {wsc_totvis}, effective after weighting {wsc_gridvis}\n" +
                 f"WSClean times: inv {wsc_t_inv}, pred {wsc_t_pred}, deconv {wsc_t_deconv}", fontsize=22)

    bb_eq_cum = np.zeros([I_lsq_eq.data.shape[1], I_lsq_eq.data.shape[2]])
    for i in range(0, nlev):
        print(" ... adding level", i)
        bb_eq_cum += I_lsq_eq.data[i,:,:]
    print(f"min, max bb lev {i}: {np.min(bb_eq_cum)}, {np.max(bb_eq_cum)}")

    # Align min to 0.0 and normalize
    bb_eq_cum  = np.fliplr(bb_eq_cum)
    bb_eq_cum -= np.min(bb_eq_cum)
    bb_eq_cum /= np.max(bb_eq_cum)
    print(f"min, max bb : {np.min(bb_eq_cum)}, {np.max(bb_eq_cum)}")
    
    im0 = ax[0].imshow(bb_eq_cum)
    ax[0].set_title("Shifted normalized bipp LSQ dirty", fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    print(f"min, max wsclean : {np.min(data[0,0,:,:])}, {np.max(data[0,0,:,:])}")
    normed_wsc  = data[0,0,:,:] - np.min(data[0,0,:,:])
    normed_wsc /= np.max(normed_wsc)
    print(f"min, max wsclean : {np.min(normed_wsc)}, {np.max(normed_wsc)}")
    im1 = ax[1].imshow(normed_wsc)
    ax[1].set_title("Shifted normalized WSClean dirty", fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    diff = bb_eq_cum - normed_wsc
    print(f"min, max diff    : {np.min(diff)}, {np.max(diff)}")
    im2 = ax[2].imshow(diff)
    ax[2].set_title("bipp minus WSClean", fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    plt.tight_layout()

    fig.savefig('bipp_nufft_gauss4_0-'+ str(nlev - 1) + '.png')






"""
### Plotting section
plt.figure()
ax = plt.gca()
I_lsq_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'BIPP LSQ, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_lsq.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)


plt.figure()
ax = plt.gca()
I_sqrt_eq.draw(catalog=sky_model.xyz.T, ax=ax, data_kwargs=dict(cmap='cubehelix'), show_gridlines=False, catalog_kwargs=dict(s=30, linewidths=0.5, alpha = 0.5))
ax.set_title(f'BIPP SQRT, sensitivity-corrected image (NUFFT)\n'
             f'Bootes Field: {sky_model.intensity.size} sources (simulated), LOFAR: {N_station} stations, FoV: {FoV_deg} degrees.')
fp = "I_sqrt.png"
if args.outdir:
    fp = os.path.join(args.outdir, fp)
plt.savefig(fp)
"""
