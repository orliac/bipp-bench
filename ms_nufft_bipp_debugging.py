"""
Imaging with BIPP NUFFT from MS dataset
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
import bipp.source as source
import bipp.statistics as vis
import bipptb
import plots

print(f"-I- SLURM_CPUS_PER_TASK = {os.environ['SLURM_CPUS_PER_TASK']}")
print(f"-I- OMP_NUM_THREADS     = {os.environ['OMP_NUM_THREADS']}")

# Setting up the benchmark
args = bipptb.check_args(sys.argv)

# For reproducible results
np.random.seed(0)

np.set_printoptions(precision=3, linewidth=120)


# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context(args.processing_unit)

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

# Observation
channel_ids = [args.channel_id]

# Grids
lmn_grid, xyz_grid = frame.make_grids(args.pixw, FoV.rad, ms.field_center)
print("-I- lmd_grid.shape =", lmn_grid.shape)
print("-I- xyz_grid.shape =", xyz_grid.shape)

# Nufft Synthesis options
opt = bipp.NufftSynthesisOptions()
opt.set_tolerance(args.nufft_eps)
# Set the maximum number of data packages that are processed together after collection.
# A larger number increases memory usage, but usually improves performance.
# If set to "None", an internal heuristic will be used.
opt.set_collect_group_size(None)
#opt.set_collect_group_size(1)

# Set the domain splitting methods for image and uvw coordinates.
# Splitting decreases memory usage, but may lead to lower performance.
# Best used with a wide spread of image or uvw coordinates.
# Possible options are "grid", "none" or "auto"
gp = 4
opt.set_local_image_partition(bipp.Partition.grid([gp, gp, 1]))
#opt.set_local_image_partition(bipp.Partition.auto())
#opt.set_local_image_partition(bipp.Partition.none())
opt.set_local_uvw_partition(bipp.Partition.grid([gp, gp, 1]))
#opt.set_local_uvw_partition(bipp.Partition.auto())
#opt.set_local_uvw_partition(bipp.Partition.none())

time_id_pe = slice(args.time_start_idx, args.time_end_idx, args.time_slice_pe)
time_id_im = slice(args.time_start_idx, args.time_end_idx, args.time_slice_im)


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()
I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma, ctx=ctx,
                                               filter_negative_eigenvalues=args.filter_negative_eigenvalues)
if args.debug: # SINGLE EPOCH !!
    print("-W- debugging mode activated, reading from dumped data")
    args.time_slice_pe = 1000000000
    args.time_slice_im = 1000000000
    f = bipptb.load_data('FREQ', args.output_directory, verbose=True)
    wl   = constants.speed_of_light / f
    XYZ  = bipptb.load_pkl('XYZ', args.output_directory, verbose=True)
    S    = bipptb.load_pkl('S', args.output_directory, verbose=True)
    W    = bipptb.load_pkl('W', args.output_directory, verbose=True)
    G    = gram(XYZ, W, wl)
    I_est.collect(S, G)
else:
    first_ep = True
    it = 0
    t0 = time.time()
    for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_pe, column="DATA"):
        t1 = time.time()
        wl   = constants.speed_of_light / f.to_value(u.Hz)
        XYZ  = ms.instrument(t)
        t2 = time.time()
        W    = ms.beamformer(XYZ, wl)
        S, W = measurement_set.filter_data(S, W)
        t3 = time.time()
        if first_ep:
            bipptb.dump_data(np.array(f.to_value(u.Hz)), 'FREQ', args.output_directory, verbose=True)
            bipptb.dump_pkl(XYZ, 'XYZ', args.output_directory, verbose=True)
            bipptb.dump_pkl(S, 'S', args.output_directory, verbose=True)
            bipptb.dump_pkl(W, 'W', args.output_directory, verbose=True)
            UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
            bipptb.dump_pkl(UVW_baselines_t, 'UVW_bsl', args.output_directory, verbose=True)
            first_ep = False
        G = gram(XYZ, W, wl)
        t4 = time.time()
        I_est.collect(S, G)
        t5 = time.time()
        print(f" -- it {it:4d}  t1 = {t1-t0:.3f}  t2 = {t2-t1:.3f} t3 = {t3-t2:.3f}  t4 = {t4-t3:.3f}  t5 = {t5-t4:.3f}")
        t0 = t5 #reset epoch
        it += 1

N_eig, intensity_intervals = I_est.infer_parameters()
print("-I- IFPE N_eig =", N_eig)
print("-I- intensity intervals =\n", intensity_intervals)
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

imager = bipp.NufftSynthesis(
    ctx,
    opt,
    N_antenna,
    N_station,
    intensity_intervals.shape[0],
    ["LSQ"],# "SQRT"],
    lmn_grid[0],
    lmn_grid[1],
    lmn_grid[2],
    args.precision,
    args.filter_negative_eigenvalues) # set this one to False to keep all eigenvalues 

i_it = 0
if args.debug: # SINGLE EPOCH !!
    t_it = time.time()
    f = bipptb.load_data('FREQ', args.output_directory, verbose=True)
    wl   = constants.speed_of_light / f
    XYZ  = bipptb.load_pkl('XYZ', args.output_directory, verbose=True)
    S    = bipptb.load_pkl('S', args.output_directory, verbose=True)
    n_vis = np.count_nonzero(S.data)
    n_vis_ifim += n_vis
    W    = bipptb.load_pkl('W', args.output_directory, verbose=True)
    UVW_baselines_t = bipptb.load_pkl('UVW_bsl', args.output_directory, verbose=True)
    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
    imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)
    t_it = time.time() - t_it
    if i_it < 3: print(f" ... ifim t_it {i_it} {t_it:.3f} sec")
    i_it += 1
else:
    for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_im, column="DATA"):
        t_it = time.time()
        wl   = constants.speed_of_light / f.to_value(u.Hz)
        XYZ  = ms.instrument(t)
        UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)

        if 1 == 0:
            uvw = np.linalg.norm(UVW_baselines_t, axis=2)
            xyz = np.linalg.norm(XYZ.data[:, None, :] - XYZ.data[None, ...], axis=2)
            print(f"-I- max uvw baseline at {i_it}/{f:.3f} =", np.max(uvw))
            print(f"-I- max xyz baseline at {i_it}/{f:.3f} =", np.max(xyz))
            print(S.shape)
            print(UVW_baselines_t.shape)
            print(UVW_baselines_t)
            print(uvw)
            print(uvw.shape)
            print("-D- uvw 80th percentile =", np.percentile(uvw, 80))
            print(W.shape)
            rad2asec = 180 * 3600 / np.pi
            bsl_cutoff = wl / args.wsc_scale * rad2asec / 1.35
            print(f"-I- bsl_cutoff = {bsl_cutoff}")
            print("-D- Before", np.count_nonzero(S.data))
            idx = np.nonzero(xyz > bsl_cutoff)
            print(idx)
            S.data.setflags(write=1)
            print(S.data.flags)
            S.data[idx] = 0
            S = vis.VisibilityMatrix(data=S.data, beam_idx=S.index[0])
            print("-D- After", np.count_nonzero(S.data))
        
        ## The resolution in arcsec can be approximated as: FWHM (") = 76 / max_baseline (km) / frequency (GHz).
        ## => max_baseline (wl) = args.wsc_scale / 76
        ##max_uvw = 76 * f.to_value(u.MHz) / args.wsc_scale
        ##print(f"-I- max_uvw = {max_uvw:.3f} [m]")

        #sys.exit(1)

        W    = ms.beamformer(XYZ, wl)
        S, W = measurement_set.filter_data(S, W)
        plots.plot_gram_matrix(S.data, f"S_{outname}", args.output_directory, "Visibility matrix")
        plots.plot_gram_matrix(G.data, f"W_{outname}", args.output_directory, "Beam-forming matrix")
        uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
        imager.collect(N_eig, wl, intensity_intervals, W.data, XYZ.data, uvw, S.data)
        n_vis = np.count_nonzero(S.data)
        n_vis_ifim += n_vis
        t_it = time.time() - t_it
        #EO: remove that in production, just to plot G
        G = gram(XYZ, W, wl)
        plots.plot_gram_matrix(G.data, f"G_{outname}", args.output_directory, "Gram matrix")
        if i_it < 3: print(f" ... ifim t_it {i_it} {t_it:.3f} sec")
        i_it += 1

#EO: scale by the number of processed epochs
#I_lsq = np.divide(imager.get("LSQ").reshape((-1, args.pixw, args.pixw)), i_it)
I_lsq = imager.get("LSQ").reshape((-1, args.pixw, args.pixw))
print("-D- I_lsq.shape", I_lsq.shape)
print("-D- I_lsq =\n", I_lsq)

#I_sqrt = imager.get("SQRT").reshape((-1, args.pixw, args.pixw))

ifim_e = time.time()
print(f"#@#IFIM {ifim_e - ifim_s:.3f} sec  NVIS {n_vis_ifim}")



### Sensitivity Field =========================================================

if 1 == 0:
    
    print("\n @@@@@@@@@@ SENSITIVITY @@@@@@@@@@\n")

    # Parameter Estimation
    sfpe_s = time.time()
    S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=args.sigma, ctx=ctx,
                                                     filter_negative_eigenvalues=args.filter_negative_eigenvalues)
    for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_pe, column="DATA"):
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
    #sensitivity_intervals = np.array([[0, np.finfo("f").max]])
    sensitivity_intervals = np.array([[np.finfo("f").min, np.finfo("f").max]])
    imager = None  # release previous imager first to some additional memory
    imager = bipp.NufftSynthesis(
        ctx,
        opt,
        N_antenna,
        N_station,
        sensitivity_intervals.shape[0],
        ["INV_SQ"],
        lmn_grid[0],
        lmn_grid[1],
        lmn_grid[2],
        args.precision,
        args.filter_negative_eigenvalues)

    for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_im, column="DATA"):
        wl   = constants.speed_of_light / f.to_value(u.Hz)
        XYZ  = ms.instrument(t)
        W    = ms.beamformer(XYZ, wl)
        _, W = measurement_set.filter_data(S, W)
        UVW_baselines_t = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
        uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
        imager.collect(N_eig, wl, sensitivity_intervals, W.data, XYZ.data, uvw, None)

    sensitivity_image = imager.get("INV_SQ").reshape((-1, args.pixw, args.pixw)) / W.data.shape[0]
    bipptb.dump_data(sensitivity_image.reshape((args.pixw, args.pixw)), 'sensitivity_data', args.output_directory)
    plots.plot_2d_matrix(sensitivity_image.reshape((args.pixw, args.pixw)), f"sensitivity_{outname}", args.output_directory, "Sensitivity", 'Width [pix]', 'Height [pix]')
    #I_sqrt_eq = s2image.Image(I_sqrt / sensitivity_image, xyz_grid)
    I_lsq_eq  = s2image.Image(I_lsq  / sensitivity_image, xyz_grid)
    #print(I_lsq_eq.data)
    sfim_e = time.time()
    print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

else:
    #I_std_eq = s2image.Image(I_std, xyz_grid)
    I_lsq_eq = s2image.Image(I_lsq, xyz_grid)
    sfpe_s = sfpe_e = sfim_s = sfim_e = 0

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")


#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)

"""
print("######################################################################")
print("-I- intensity_intervals =\n", intensity_intervals, "\n")
print("-I- xyz_grid:", xyz_grid.shape, "\n", xyz_grid, "\n")
print("-I- I_lsq:\n", I_lsq, "\n")
print("-I- I_lsq_eq:\n", I_lsq_eq.data, "\n")
"""
print("-I- args.output_directory:", args.output_directory)

bipptb.dump_data(I_lsq.data, 'I_lsq_data', args.output_directory)
bipptb.dump_data(I_lsq_eq.data, f"I_lsq_eq_data_{outname}", args.output_directory)
bipptb.dump_data(I_lsq_eq.grid, f"I_lsq_eq_grid_{outname}", args.output_directory)

bipptb.dump_json({'ifpe_s': ifpe_s, 'ifpe_e': ifpe_e,
                  'ifim_s': ifim_s, 'ifim_e': ifim_e,
                  'sfpe_s': sfpe_s, 'sfpe_e': sfpe_e,
                  'sfim_s': sfim_s, 'sfim_e': sfim_e,
                  'tot_s' : jkt0_s, 'tot_e' : jkt0_e,
                  'n_vis_ifim': n_vis_ifim,
                  'filename': f"stats_{outname}.json",
                  'out_dir': args.output_directory})
