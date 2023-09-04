import sys
import os
import argparse
from pathlib import Path
import numpy
import json
from distutils.util import strtobool
import pickle
from astropy import units as u
from astropy.coordinates import Angle


def dump_json(info):
    #print(info)
    stats = { 
        "timings": {
            'ifpe': info['ifpe'],
            'ifim': info['ifim'],
            'sfpe': info['sfpe'],
            'sfim': info['sfim'],
            'tot' : info['tot'],
            'ifpe_vis': info['ifpe_tvis'],
            'ifim_vis': info['ifim_tvis'],
            'ifpe_plot': info['ifpe_tplot'],
            'ifim_plot': info['ifim_tplot'],
            'ifpe_proc': info['ifpe_tproc'],
            'ifim_proc': info['ifim_tproc'],

        },
        "visibilities": {'ifpe': info['ifpe_nvis'],
                         'ifim': info['ifim_nvis']}
    }

    if info['out_dir']:
        with open(os.path.join(info['out_dir'], info['filename']), "w") as outfile:
            outfile.write(json.dumps(stats, indent=4))


def compare_solutions(ref, sol):
    print("-R- reference:", ref, "\n-R- solution :", sol)
    with open(ref, 'r') as openfile: ref_stats = json.load(openfile)
    with open(sol, 'r') as openfile: sol_stats = json.load(openfile)
    speedup_ifim = ref_stats['timings']['ifim'] / sol_stats['timings']['ifim']
    speedup_ivis = ref_stats['timings']['ivis'] / sol_stats['timings']['ivis']
    speedup_idp  = ref_stats['timings']['idp']  / sol_stats['timings']['idp']
    speedup_imfs = ref_stats['timings']['imfs'] / sol_stats['timings']['imfs']
    print(f"-R- speedups : ifim = {speedup_ifim:5.1f}, ivis = {speedup_ivis:3.1f}, ",
          f"idp = {speedup_idp:4.1f}, imfs = {speedup_imfs:5.1f}")

    # Compare imfs npy
    ref_std = numpy.load(ref.rsplit('.', 1)[0] + '_imfs_std.npy', allow_pickle=True)
    ref_lsq = numpy.load(ref.rsplit('.', 1)[0] + '_imfs_lsq.npy', allow_pickle=True)
    sol_std = numpy.load(sol.rsplit('.', 1)[0] + '_imfs_std.npy', allow_pickle=True)
    sol_lsq = numpy.load(sol.rsplit('.', 1)[0] + '_imfs_lsq.npy', allow_pickle=True)
    rmse_lsq, max_abs_err_lsq = stats_image_diff(ref_lsq, sol_lsq)
    rmse_std, max_abs_err_std = stats_image_diff(ref_std, sol_std)
    print(f"-R- LSQ stats: rmse = {rmse_lsq:.2E}, max abs err = {max_abs_err_lsq:.2E}")
    print(f"-R- STD stats: rmse = {rmse_std:.2E}, max abs err = {max_abs_err_std:.2E}")


def dump_pkl(stats, filename, outdir, verbose=False):
    fp = os.path.join(outdir, filename + '.pkl')
    with open(fp, 'wb') as f:
        pickle.dump(stats, f)
    if verbose:
        print(f"-I- Pickle dump to {fp}")

def load_pkl(filename, outdir, verbose=False):
    fp = os.path.join(outdir, filename + '.pkl')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(f"-I- Pickle load from {fp}")
    return data

def dump_data(stats, filename, outdir, verbose=False):
    if outdir:
        fp = os.path.join(outdir, filename + '.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, stats)
        if verbose:
            print(f"-I- Dumped {fp}")

def load_data(filename, outdir, verbose=False):
    fp = os.path.join(outdir, filename + '.npy')
    data = numpy.load(fp, allow_pickle=True)
    if verbose:
        print(f"-I- Loaded {fp}")
    return data

def dump_stats(stats, filename, outdir):
    if outdir:
        I_std, I_lsq = stats.as_image()
        fp = os.path.join(outdir, filename + '_lsq.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, I_lsq.data)
            print("-I- wrote", fp)
        fp = os.path.join(outdir, filename + '_std.npy')
        with open(fp, 'wb') as f:
            numpy.save(f, I_std.data)
            print("-I- wrote", fp)


def check_args(args_in):
    print("-I- command line arguments =", args_in)
    parser = argparse.ArgumentParser(args_in)
    parser.add_argument("--cluster", help="Cluster on which to run the benchmark",
                        choices=['izar', 'jed'], required=True)
    parser.add_argument("--compiler", help="Compiler to use",
                        choices=['cuda', 'rocm', 'gcc', 'intel'], required=True)
    parser.add_argument("--output_directory", help="Path to dumping location (no dumps if not set)",
                        required=True)
    parser.add_argument("--processing_unit",  help="Bluebild processing unit (for ctx definition)",
                        choices=['auto', 'cpu', 'gpu', 'none'], required=True)
    parser.add_argument("--precision", help="Floating point calculation precision",
                        choices=['single', 'double'], default='double')
    parser.add_argument("--package", help="Package to run, either bipp or pypeline",
                        choices=['bipp', 'pypeline'], required=True)
    parser.add_argument("--algo", help="Algorithm to use in Bluebild", choices=['ss', 'nufft'], required=True)
    parser.add_argument("--nsta", help="Number of stations in simulation",  #EO: should defaults to None == all stations
                        required=False, type=int) 
    parser.add_argument("--nlev", help="Number of energy levels in simulation",
                        required=True, type=int)
    parser.add_argument("--pixw", help="Image width/height in pixels", required=True, type=int)
    parser.add_argument("--wsc_scale", help="WSClean scale [arcsec]",  required=False, type=float)
    parser.add_argument("--fov_deg",   help="Field of view [degree]",  required=False, type=float)

    parser.add_argument("--wsc_log", help="WSClean log file")

    parser.add_argument("--time_start_idx", help="Index of first epoch to process",
                        type=int, default=0)
    parser.add_argument("--time_end_idx",  help="Index of last epoch to process",
                        type=int, default=1)
    parser.add_argument("--time_slice_pe",  help="Time slice for parameter estimation (PE)",
                        type=int, default=1)
    parser.add_argument("--time_slice_im", help="Time slice for imaging (IM)",
                        type=int, default=1)
    parser.add_argument("--sigma", help="Sigma in intensity field parameter estimation",
                        type=float, default=0.95)
    parser.add_argument("--nufft_eps", help="NUFFT convergence epsilon",
                        type=float, default=0.001)
    parser.add_argument("--ms_file", help="Path to MS file to process")
    parser.add_argument("--telescope", help="Observing instrument in use",
                        choices=['LOFAR', 'MWA', 'SKALOW', ], required=True)
    parser.add_argument("--filter_negative_eigenvalues", help="Switch to filter or not negative eigenvalues",
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--debug", help="Switch to debugging mode (read S from dumped .npy)",
                        default=False, required=False, action='store_true')
    parser.add_argument("--channel_id", help="ID of channel to process",
                        required=True, type=int)
    parser.add_argument("--outname", help="Basename for output files", required=True, type=str)
    parser.add_argument("--maxuvw_m", help="Maximum uvw baseline length to consider [m]", required=False, type=float)

    args = parser.parse_args()
    """
    if args.outdir:
        if not os.path.exists(args.outdir):
            print('-E- --outdir ('+args.outdir+') must exist if set')
            sys.exit(1)
        print("-I- dumping directory: ", args.outdir)
    else:
        print("-W- will not dump anything since --outdir was not set")
    """

    if args.wsc_scale == None and args.fov_deg == None:
        raise Exception("either --wsc_scale or --fov_deg must be set")

    assert args.time_start_idx >= 0
    assert args.time_end_idx >= 0
    assert args.time_start_idx <= args.time_end_idx
    assert args.time_slice_pe >= 1
    assert args.time_slice_im >= 1

    if args.package == "bipp" and args.processing_unit == 'none':
        print('-E- bipp processing unit cannot be none.')
        sys.exit(1)

    if args.package == "pypeline" and args.sigma == 1.0:
        print('-W- pypeline reset of args.sigma from 1.0 to 0.99999')
        args.sigma = 0.99999

    args.nbits = 32 if args.precision == 'single' else 64

    # 0 station means all stations
    if args.nsta == 0: args.nsta = None

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)


    print("-I- Command line input -----------------------------")
    print("-I- MS file        =", args.ms_file)
    print("-I- Telescope      =", args.telescope)
    print("-I- precision      =", args.precision)
    print("-I- N. bits        =", args.nbits)
    print("-I- N. stations    =", args.nsta)
    print("-I- proc unit      =", args.processing_unit)
    print("-I- N. pix         =", args.pixw)
    print("-I- N. levels      =", args.nlev)
    print("-I- WSClean scale  =", args.wsc_scale)
    print("-I- Time start idx =", args.time_start_idx)
    print("-I- Time end idx   =", args.time_end_idx)
    print("-I- Time slice PE  =", args.time_slice_pe)
    print("-I- Time slice IM  =", args.time_slice_im)
    print("-I- sigma          =", args.sigma)
    print("-I- NUFFT epsilon  =", args.nufft_eps)
    print("-I- Output dir.    =", args.output_directory)
    print("-I- Filter neg eig =", args.filter_negative_eigenvalues)
    print("-I- ------------------------------------------------")

    return args


# Compute the RMSE between two image
def stats_image_diff(image1, image2):
    assert image1.shape == image2.shape, \
        f"-E- shapes of images to compare do not match {image1.data.shape} vs {image2.data.shape}"
    print("-I- comparing images with shape ", image1.shape)
    diff = image2 - image1
    rmse = numpy.sqrt(numpy.sum(diff**2)/numpy.size(diff))
    max_abs = numpy.max(numpy.abs(diff))
    return rmse, max_abs



