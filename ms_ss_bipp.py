"""
Imaging with BIPP SS from MS dataset
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
import bipptb
import plots
import json

print(f"-I- SLURM_CPUS_PER_TASK = {os.environ['SLURM_CPUS_PER_TASK']}")
print(f"-I- OMP_NUM_THREADS     = {os.environ['OMP_NUM_THREADS']}")

# Setting up the benchmark
args = bipptb.check_args(sys.argv)

# For reproducible results
np.random.seed(0)

np.set_printoptions(precision=6, linewidth=120)


# Create context with selected processing unit.
# Options are "AUTO", "CPU" and "GPU".
ctx = bipp.Context(args.processing_unit)

# Enable this to create an alternate context CPU <-> GPU to test Gram implemenations
if args.check_alt_gram:
    if args.processing_unit == 'cpu':
        ctx_ =  bipp.Context('gpu')
    elif args.processing_unit == 'gpu':
        ctx_ =  bipp.Context('cpu')
    else:
        print("-E- Case not handled")
        sys.exit(1)


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
channel_ids = np.arange(args.channel_id_start, args.channel_id_end)
print("-I channel_ids = ", channel_ids)

# Grids
lmn_grid, xyz_grid = frame.make_grids(args.pixw, FoV.rad, ms.field_center)
px_w = xyz_grid.shape[1]
px_h = xyz_grid.shape[2]
xyz_grid = xyz_grid.reshape(3, -1)
print("-I- lmd_grid.shape =", lmn_grid.shape)
print("-I- xyz_grid.shape =", xyz_grid.shape, "(after reshaping in SS)")

time_id_pe = slice(args.time_start_idx, args.time_end_idx, args.time_slice_pe)
time_id_im = slice(args.time_start_idx, args.time_end_idx, args.time_slice_im)


if args.dump_tests_json:
    data_all = []


### Intensity Field ===========================================================

# Parameter Estimation
ifpe_s = time.time()

I_est = bb_pe.IntensityFieldParameterEstimator(args.nlev, sigma=args.sigma, ctx=ctx)
    
IFPE_NVIS  = 0
IFPE_TVIS  = 0
IFPE_TPLOT = 0
i_it = 0
t0 = time.time()
for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_pe, column="DATA"):
    t1 = time.time()
    t_vis = t1 - t0
    IFPE_TVIS += t_vis
    wl   = constants.speed_of_light / f.to_value(u.Hz)
    XYZ  = ms.instrument(t)
    t2   = time.time()
    W    = ms.beamformer(XYZ, wl)
    t3   = time.time()
    S, W = measurement_set.filter_data(S, W)
    n_vis = np.count_nonzero(S.data)
    IFPE_NVIS += n_vis
    t4 = time.time()
    G = gram(XYZ, W, wl)

    if args.check_alt_gram:
        gram_  = bb_gr.GramBlock(ctx_)
        G_     = gram_(XYZ, W, wl)
        G_diff = G.data - G_.data
        TOL = 1E-8
        assert(np.absolute(np.min(G_diff)) < TOL)
        assert(np.absolute(np.max(G_diff)) < TOL)
        
    t5 = time.time()
    I_est.collect(S, G)
    t6 = time.time()
    print(f"-T- it {i_it:4d}:  vis {t_vis:.3f},  xyz {t2-t1:.3f}, w {t3-t2:.3f}  fd = {t4-t3:.3f}  g = {t5-t4:.3f}, coll {t6-t5:.3f}, tot {t6-t0:.3f}")
    t0 = time.time()
    i_it += 1
    
N_eig_int, intensity_intervals = I_est.infer_parameters(fne=args.filter_negative_eigenvalues)
print("-I- IFPE N_eig_int =", N_eig_int)
print("-I- intensity intervals =\n", intensity_intervals)
print("-I- XYZ.shape =", XYZ.shape)

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
n_vis_ifim = 0
ifim_s = time.time()
ifim_vis = 0

imager = bipp.StandardSynthesis(ctx,
                                N_antenna,
                                N_station,
                                intensity_intervals.shape[0],
                                args.int_filters,
                                xyz_grid[0],
                                xyz_grid[1],
                                xyz_grid[2],
                                args.precision)

IFIM_NVIS  = 0
IFIM_TVIS  = 0
IFIM_TPLOT = 0
i_it = 0
t0 = time.time()
for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_im, column="DATA"):
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
    imager.collect(N_eig_int, wl, intensity_intervals, W.data, XYZ.data, S.data)
    t5 = time.time()
    n_vis = np.count_nonzero(S.data)
    IFIM_NVIS += n_vis

    if i_it == 0:
        t_plot = time.time()
        G = gram(XYZ, W, wl)
        C = 299792458
        print(f"-I- Freqency = {f}, wl = {C/f:.3f}, 2/wl = {2/wl:.5f}, wl/2 = {wl/2:.5f}")
        plots.plot_gram_matrix(G.data, f"{args.outname}_G_it{i_it}_f{f.value/1000000:.3f}MHz", args.output_directory, "Gram matrix")
        plots.plot_gram_matrix(G.data - np.identity(W.shape[0], dtype=complex), f"{args.outname}_G_reduced_it{i_it}_f{f.value/1000000:.3f}MHz", args.output_directory, "Gram matrix minux identity")
        G_0d = np.copy(G.data)
        np.fill_diagonal(G_0d, 0)
        plots.plot_gram_matrix(G_0d, f"{args.outname}_G_0d_it{i_it}_f{f.value/1000000:.3f}MHz", args.output_directory, "Gram matrix with null diagonal")
        plots.plot_visibility_matrix(S.data, f"{args.outname}_S_it{i_it}", args.output_directory, "Visibility matrix")
        plots.plot_beamweight_matrix(W.data, f"{args.outname}_W_it{i_it}", args.output_directory, "Beam-forming matrix")
        t_plot = time.time() - t_plot
        print(f"-W- Plotting took {t_plot:.3f} sec.")
        IFIM_TPLOT += t_plot

    print(f"-T- it {i_it:4d} mjd {t.mjd:.7f}:  vis {t_vis:.3f},  xyz {t2-t1:.3f}, w {t3-t2:.3f}  fd = {t4-t3:.3f}  coll = {t5-t4:.3f}, tot {t5-t0:.3f} -- time = {t.mjd:.6f}")
    t0 = time.time()
    i_it += 1

    if args.dump_tests_json:
        data_all.append({"xyz": np.transpose(XYZ.data  - np.mean(XYZ.data, axis=0)).tolist(),
                         #"uvw": np.transpose(uvw).tolist(),
                         "w_real": np.transpose(W.data.real).tolist(),
                         "w_imag": np.transpose(W.data.imag).tolist(),
                         "s_real": np.transpose(S.data.real).tolist(),
                         "s_imag": np.transpose(S.data.imag).tolist()})

I_filtered = {}
for filt in args.int_filters:
    I_filtered[filt] = imager.get(filt).reshape((-1, args.pixw, args.pixw))
I_lsq  = s2image.Image(I_filtered.get("LSQ"), xyz_grid.reshape(3, px_w, px_h))


ifim_e = time.time()
IFIM = ifim_e - ifim_s
IFIM_TPROC = IFIM - IFIM_TVIS - IFIM_TPLOT
print(f"#@#IFIM      {IFIM:.3f} sec")
print(f"#@#IFIM_NVIS {IFIM_NVIS} vis")
print(f"#@#IFIM_TVIS {IFIM_TVIS:.3f} sec")
print(f"#@#IFIM_PLOT {IFIM_TPLOT:.3f} sec")
print(f"#@#IFIM_PROC {IFIM_TPROC:.3f} sec")


### Sensitivity Field =========================================================
N_eig_sen = 0
sensitivity_intervals = np.array([])
S_filtered = {}

if args.sen_filters:
    
    print("\n @@@@@@@@@@ SENSITIVITY @@@@@@@@@@\n")
    
    # Parameter Estimation
    sfpe_s = time.time()
    S_est = bb_pe.SensitivityFieldParameterEstimator(sigma=args.sigma, ctx=ctx)
    for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_pe, column="DATA"):
        wl   = constants.speed_of_light / f.to_value(u.Hz)
        XYZ  = ms.instrument(t)
        W    = ms.beamformer(XYZ, wl)
        _, W = measurement_set.filter_data(S, W)
        G    = gram(XYZ, W, wl)
        S_est.collect(G)
    N_eig_sen = S_est.infer_parameters()
    print("-I- SFPE N_eig_sen =", N_eig_sen)
    sfpe_e = time.time()
    print(f"#@#SFPE {sfpe_e - sfpe_s:.3f} sec")
    
    # Imaging
    sfim_s = time.time()
    sensitivity_intervals = np.array([[0, np.finfo("f").max]])
    #sensitivity_intervals = np.array([[np.finfo("f").min, np.finfo("f").max]])
    imager = None  # release previous imager first to some additional memory
    imager = bipp.StandardSynthesis(ctx,
                                    N_antenna,
                                    N_station,
                                    sensitivity_intervals.shape[0],
                                    args.sen_filters,
                                    xyz_grid[0],
                                    xyz_grid[1],
                                    xyz_grid[2],
                                    args.precision)

    for t, f, S in ms.visibilities(channel_id=channel_ids, time_id=time_id_im, column="DATA"):
        wl   = constants.speed_of_light / f.to_value(u.Hz)
        XYZ  = ms.instrument(t)
        W    = ms.beamformer(XYZ, wl)
        _, W = measurement_set.filter_data(S, W)
        imager.collect(N_eig_sen, wl, sensitivity_intervals, W.data, XYZ.data)

    for filt in args.sen_filters:
        S_filtered[filt] = imager.get(filt).reshape((-1, args.pixw, args.pixw))
        # Don't use sensitivity for now!
        #bipptb.dump_data(sensitivity_image.reshape((args.pixw, args.pixw)), 'sensitivity_data', args.output_directory)
        #plots.plot_2d_matrix(sensitivity_image.reshape((args.pixw, args.pixw)), f"sensitivity_{outname}", args.output_directory, "Sensitivity", 'Width [pix]', 'Height [pix]')
        #I_sqrt_eq = s2image.Image(I_sqrt / sensitivity_image, xyz_grid)
        #I_lsq_eq  = s2image.Image(I_lsq  / sensitivity_image, xyz_grid)

    sfim_e = time.time()
    print(f"#@#SFIM {sfim_e - sfim_s:.3f} sec")

else:
    #I_std_eq = s2image.Image(I_std, xyz_grid.reshape(3, px_w, px_h))
    #I_lsq_eq = s2image.Image(I_lsq, xyz_grid.reshape(3, px_w, px_h))
    sfpe_s = sfpe_e = sfim_s = sfim_e = 0

jkt0_e = time.time()
print(f"#@#TOT {jkt0_e - jkt0_s:.3f} sec\n")


if args.dump_tests_json:
    json_input = {'wl': wl,
                  'eps': args.nufft_eps,
                  'n_antenna': N_antenna,
                  'n_beam': N_station,
                  'n_eig_int': N_eig_int,
                  'n_eig_sen': N_eig_sen,
                  'intervals_int': intensity_intervals.tolist(),
                  'intervals_sen': sensitivity_intervals.tolist(),
                  'data': data_all
    }
    
    json_path = os.path.join(args.output_directory, f"{args.telescope.lower()}_ss_input_bipp_{args.precision}.json")
    with open(json_path, "w") as json_file:
        json.dump(json_input, json_file)
    print("-I- wrote JSON input file", json_path)

    json_output = {'pixel_x': xyz_grid[0,:].flatten(order='F').tolist(),
                   'pixel_y': xyz_grid[1,:].flatten(order='F').tolist(),
                   'pixel_z': xyz_grid[2,:].flatten(order='F').tolist(),
                   'lmn_x': lmn_grid[0,:].flatten(order='F').tolist(),
                   'lmn_y': lmn_grid[1,:].flatten(order='F').tolist(),
                   'lmn_z': lmn_grid[2,:].flatten(order='F').tolist()}
    
    for filt in I_filtered.keys():
        json_output['int_' + filt.lower()] = np.vstack(np.squeeze(np.vsplit(I_filtered[filt].data, intensity_intervals.shape[0]))).tolist()
        
    for filt in S_filtered.keys():
        json_output['sen_' + filt.lower()] = np.vstack(np.squeeze(np.vsplit(S_filtered[filt].data, sensitivity_intervals.shape[0]))).tolist()
    
    json_path = os.path.join(args.output_directory, f"{args.telescope.lower()}_ss_output_bipp_{args.precision}.json")
    with open(json_path, "w") as json_file:
        json.dump(json_output, json_file)
    print("-I- wrote JSON output file", json_path)


#EO: early exit when profiling
if os.getenv('BB_EARLY_EXIT') == "1":
    print("-I- early exit signal detected")
    sys.exit(0)


print("######################################################################")
#print("-I- intensity_intervals =\n", intensity_intervals, "\n")
#print("-I- xyz_grid:", xyz_grid.shape, "\n", xyz_grid, "\n")
print("-I- I_lsq:\n", I_lsq.data, "\n")
#for filt in I_filtered.keys():
#    print(f"-I- I {filt}:\n", I_filtered.get(filt),"\n")
print("-I- args.output_directory:", args.output_directory)

# ***For backward compatibility save data as eq_data***
bipptb.dump_data(I_lsq.data, f"{args.outname}_I_lsq_eq_data", args.output_directory)
bipptb.dump_data(I_lsq.grid, f"{args.outname}_I_lsq_eq_grid", args.output_directory)

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
