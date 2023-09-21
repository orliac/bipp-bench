"""
OSKAR imaging with Pypeline SS from MS dataset
"""

import os
import sys
import time
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
import imot_tools.io.s2image as s2image
import imot_tools.math.sphere.grid as grid
import imot_tools.math.sphere.transform as transform
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
import plots

print(f"-I- SLURM_CPUS_PER_TASK = {os.environ['SLURM_CPUS_PER_TASK']}")
print(f"-I- OMP_NUM_THREADS     = {os.environ['OMP_NUM_THREADS']}")

# Setting up the benchmark
args = bipptb.check_args(sys.argv)

# For reproducible results
np.random.seed(0)

np.set_printoptions(precision=6, linewidth=120)

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
if args.telescope == 'LOFAR':
    ms = measurement_set.LofarMeasurementSet(args.ms_file, args.nsta, station_only=True)
elif args.telescope == 'SKALOW':
    ms = measurement_set.SKALowMeasurementSet(args.ms_file)
elif args.telescope == 'MWA':
    ms = measurement_set.MwaMeasurementSet(args.ms_file)
else:
    raise Exception(f"Unknown telescope {args.telescope}!")

outname = f"{args.package}_{args.algo}_{args.processing_unit}"

print(f"-I- ms.field_center = {ms.field_center}")
gram = bb_gr.GramBlock(ctx)

# Observation (old fashion, single channel)
channel_id = args.channel_id
frequency = ms.channels["FREQUENCY"][channel_id]
print(f"-I- frequency = {frequency}")
wl = constants.speed_of_light / frequency.to_value(u.Hz)

# Grids
print(f"FoV = {FoV:.6f} rad")
lim = np.sin(FoV.rad / 2)
print(f"lim = {lim:.6f}")
offset = lim / (args.pixw / 2) * 0.5
print(f"half pixel offset = {offset:.6f}")
grid_slice1 = np.linspace(-lim - offset, lim - offset, args.pixw)
grid_slice2 = np.linspace(-lim + offset, lim + offset, args.pixw)
#grid_slice = np.linspace(-lim, lim, args.pixw)
l_grid, m_grid = np.meshgrid(grid_slice2, grid_slice1)
n_grid = np.sqrt(1 - l_grid ** 2 - m_grid ** 2)  # No -1 if r on the sphere !
lmn_grid = np.stack((l_grid, m_grid, n_grid), axis=0)
uvw_frame = frame.uvw_basis(ms.field_center)
px_grid = np.tensordot(uvw_frame, lmn_grid, axes=1)
print("-I- lmd_grid.shape =", lmn_grid.shape)
print("-I- px_grid.shape =", px_grid.shape)

time_id_pe = slice(args.time_start_idx, args.time_end_idx, args.time_slice_pe)
time_id_im = slice(args.time_start_idx, args.time_end_idx, args.time_slice_im)

### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma,
                                               filter_negative_eigenvalues=args.filter_negative_eigenvalues)
IFPE_NVIS  = 0
IFPE_TVIS  = 0
IFPE_TPLOT = 0
i_it = 0
t0 = time.time()
for t, f, S, _ in ms.visibilities(channel_id=[channel_id], time_id=time_id_pe, column="DATA"):
    t1 = time.time()
    t_vis = t1 - t0
    IFPE_TVIS += t_vis
    wl_   = constants.speed_of_light / f.to_value(u.Hz)
    if wl_ != wl: raise Exception("Mismatch on frequency")
    XYZ  = ms.instrument(t)
    t2   = time.time()
    W    = ms.beamformer(XYZ, wl)
    t3   = time.time()
    S, W = measurement_set.filter_data(S, W)
    n_vis = np.count_nonzero(S.data)
    IFPE_NVIS += n_vis
    t4 = time.time()
    G    = gram(XYZ, W, wl)
    t5 = time.time()
    I_est.collect(S, G)
    t6 = time.time()
    print(f"-T- it {i_it:4d}:  vis {t_vis:.3f},  xyz {t2-t1:.3f}, w {t3-t2:.3f}  fd = {t4-t3:.3f}  g = {t5-t4:.3f}, coll {t6-t5:.3f}, tot {t6-t0:.3f}")
    t0 = time.time()
    i_it += 1

N_eig, c_centroid = I_est.infer_parameters()
print("-I- IFPE N_eig =", N_eig)
print("-I- c_centroid =\n", c_centroid)

ifpe_e = time.time()
IFPE = ifpe_e - ifpe_s
IFPE_TPROC = IFPE - IFPE_TVIS - IFPE_TPLOT
print(f"#@#IFPE {IFPE:.3f} sec")
print(f"#@#IFPE_NVIS {IFPE_NVIS} vis")
print(f"#@#IFPE_TVIS {IFPE_TVIS:.3f} sec")
print(f"#@#IFPE_TPLOT {IFPE_TPLOT:.3f} sec")
print(f"#@#IFPE_TPROC {IFPE_TPROC:.3f} sec")

N_antenna, N_station = W.shape
print("-I- N_antenna =", N_antenna)
print("-I- N_station =", N_station)


# Imaging
ifim_s = time.time()
I_dp = bb_dp.IntensityFieldDataProcessorBlock(N_eig, c_centroid, ctx,
                                              filter_negative_eigenvalues=args.filter_negative_eigenvalues)
I_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, args.nlev, args.nbits, ctx,
                                 filter_negative_eigenvalues=args.filter_negative_eigenvalues)

IFIM_NVIS  = 0
IFIM_TVIS  = 0
IFIM_TPLOT = 0
i_it = 0
t0 = time.time()
for t, f, S, uvw in ms.visibilities(channel_id=[channel_id], time_id=time_id_im, column="DATA"):
    t1 = time.time()
    t_vis = t1 - t0
    IFIM_TVIS += t_vis
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    t2 = time.time()
    W    = ms.beamformer(XYZ, wl)
    t3 = time.time()
    S, W = measurement_set.filter_data(S, W)
    t4 = time.time()
    D, V, c_idx = I_dp(S, XYZ, W, wl)
    t5 = time.time()
    _ = I_mfs(D, V, XYZ.data, W.data, c_idx)
    t6 = time.time()
    n_vis = np.count_nonzero(S.data)
    IFIM_NVIS += n_vis

    if i_it == 0:
        t_plot = time.time()
        G = gram(XYZ, W, wl)
        plots.plot_gram_matrix(G.data, f"{args.outname}_G_it{i_it}", args.output_directory, "Gram matrix")
        plots.plot_visibility_matrix(S.data, f"{args.outname}_S_it{i_it}", args.output_directory, "Visibility matrix")
        plots.plot_beamweight_matrix(W.data, f"{args.outname}_W_it{i_it}", args.output_directory, "Beam-forming matrix")
        plots.plot_uvw(uvw, args.pixw, f"{args.outname}_uvw_it{i_it}", args.output_directory, "UV projected baselines")
        plots.plot_eigen_vectors(V,  f"{args.outname}_V_it{i_it}", args.output_directory, "Eigen vectors")
        t_plot = time.time() - t_plot
        print(f"-W- Plotting took {t_plot:.3f} sec.")
        IFIM_TPLOT += t_plot

    print(f"-T- it {i_it:4d}:  vis {t_vis:.3f},  xyz {t2-t1:.3f}, w {t3-t2:.3f}  fd = {t4-t3:.3f}  i_dp = {t5-t4:.3f}, i_mfs = {t6-t5:.3f}, tot {t6-t0:.3f}")
    t0 = time.time()
    i_it += 1

I_std, I_lsq = I_mfs.as_image()

ifim_e = time.time()
IFIM = ifim_e - ifim_s
IFIM_TPROC = IFIM - IFIM_TVIS - IFIM_TPLOT
print(f"#@#IFIM      {IFIM:.3f} sec")
print(f"#@#IFIM_NVIS {IFIM_NVIS} vis")
print(f"#@#IFIM_TVIS {IFIM_TVIS:.3f} sec")
print(f"#@#IFIM_PLOT {IFIM_TPLOT:.3f} sec")
print(f"#@#IFIM_PROC {IFIM_TPROC:.3f} sec")

#sys.exit(0)

if 1 == 0:
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
    n_vis_sfim = 0
    S_dp  = bb_dp.SensitivityFieldDataProcessorBlock(N_eig, ctx)
    S_mfs = bb_sd.Spatial_IMFS_Block(wl, px_grid, 1, args.nbits, ctx)
    for t, f, S in ms.visibilities(channel_id=[channel_id], time_id=slice(None, None, args.time_slice_im), column="DATA"):
        wl   = constants.speed_of_light / f.to_value(u.Hz)
        XYZ  = ms.instrument(t)
        W    = ms.beamformer(XYZ, wl)
        _, W = measurement_set.filter_data(S, W)
        D, V = S_dp(XYZ, W, wl)
        _ = S_mfs(D, V, XYZ.data, W.data, cluster_idx=np.zeros(N_eig, dtype=int))
    _, S = S_mfs.as_image()

    I_std_eq = s2image.Image(I_std.data / S.data / i_it, I_std.grid)
    I_lsq_eq = s2image.Image(I_lsq.data / S.data / i_it, I_lsq.grid)
    sfim_e = time.time()
else:
    I_std_eq = s2image.Image(I_std.data / i_it, I_std.grid)
    I_lsq_eq = s2image.Image(I_lsq.data / i_it, I_lsq.grid)
    sfpe_s = sfpe_e = sfim_s = sfim_e = 0

print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")


#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)


print("######################################################################")
#print("-I- c_centroid =\n", c_centroid, "\n")
#print("-I- px_grid:", px_grid.shape, "\n", px_grid, "\n")
#print("-I- I_lsq:\n", I_lsq.data, "\n")
print("-I- I_lsq_eq:\n", I_lsq_eq.data, "\n")

print("-I- args.output_directory:", args.output_directory)

#bipptb.dump_data(I_lsq.data,    f"{args.outname}_I_lsq_data",   args.output_directory)
bipptb.dump_data(I_lsq_eq.data, f"{args.outname}_I_lsq_eq_data", args.output_directory)
bipptb.dump_data(I_lsq_eq.grid, f"{args.outname}_I_lsq_eq_grid", args.output_directory)

bipptb.dump_json({'ifpe': ifpe_e - ifpe_s,
                  'ifim': ifim_e - ifim_s,
                  'sfpe': sfpe_e - sfpe_s,
                  'sfim': sfim_e - sfim_s,
                  'tot' : jkt0_e - jkt0_s,
                  'ifpe_tvis':  IFPE_TVIS,
                  'ifpe_tplot': IFPE_TPLOT,
                  'ifpe_tproc': IFPE_TPROC,
                  'ifim_tvis':  IFIM_TVIS,
                  'ifim_tplot': IFIM_TPLOT,
                  'ifim_tproc': IFIM_TPROC,
                  'ifpe_nvis': IFPE_NVIS,
                  'ifim_nvis': IFIM_NVIS,
                  'filename': f"{args.outname}_stats.json",
                  'out_dir': args.output_directory})
