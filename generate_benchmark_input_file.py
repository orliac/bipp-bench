# Generate Slurm job array input file

import os
import sys
import numpy as np
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--pipeline",       help="Prefix of pipeline to run", required=True)
parser.add_argument("--bench_name",     help="Benchmark name", required=True)
parser.add_argument("--cluster",        help="Cluster to run on", required=True, choices=['izar', 'jed'])
parser.add_argument("--proc_unit",      help="Processing unit", required=True,   choices=['auto', 'cpu', 'gpu', 'none'])
parser.add_argument("--compiler",       help="Compiler to use", required=True,   choices=['cuda', 'rocm', 'gcc', 'intel'])
parser.add_argument("--precision",      help="FP precision", required=True,      choices=['single', 'double'])
parser.add_argument("--package",        help="Package to use", required=True,    choices=['bipp', 'pypeline'])
parser.add_argument("--out_dir",        help="Output directory", required=True)
parser.add_argument("--in_file",        help="Path to benchmark input file", required=True)
#parser.add_argument("--ms_file",        help="Path to MS dataset to process", required=False)
parser.add_argument("--time_start_idx", help="Start time index", required=True)
parser.add_argument("--time_end_idx",   help="End time index", required=True)
parser.add_argument("--time_slice_pe",  help="Time slice in parameter estimation", required=False, default=1)
parser.add_argument("--time_slice_im",  help="Time slice in imaging", required=False, default=1)
parser.add_argument("--sigma",          help="Fraction of power spectrum to be considered", required=False)
parser.add_argument("--wsc_scale",      help="WSClean pixel scale in arcsec", required=True)
parser.add_argument("--fov_deg",        help="Field of view in degree", required=False)
parser.add_argument("--algo",           help="Algorithm to use (ss, nufft)", required=True)
parser.add_argument("--telescope",      help="Telescope", required=True)
parser.add_argument("--filter_negative_eigenvalues", help="Filter negative eigenvalues? 0/1", required=True, choices=["0", "1"])
parser.add_argument("--channel_id",     help="Channel ID to process", required=True, type=int)
parser.add_argument("--outname",        help="Prefix of output files", required=True, type=str)

args = parser.parse_args()

# No reference Python in BIPP
if args.package == "bipp" and args.proc_unit == "none":
    print("-E- bipp cannot run none")
    sys.exit(1)

# Set up jobs to run in the benchmark
nLevels   = [1, 2, 4, 8]
pixWidths = [256, 512, 1024, 2048] ### equivalent of wsc_size
nStations = [0] # 0 == all stations

import subprocess
#--fov_deg {args.fov_deg} 
with open(args.in_file, 'w') as f:
    for nsta in np.sort(nStations):
        for nlev in np.sort(nLevels):
            cli_  = f"--cluster {args.cluster} --processing_unit {args.proc_unit} --compiler {args.compiler}"
            cli_ += f" --precision {args.precision} --package {args.package} --nlev {nlev}"
            cli_ += f" --wsc_scale {args.wsc_scale}"
            #cli_ += f" --ms_file {args.ms_file}"
            cli_ += f" --time_start_idx {args.time_start_idx} --time_end_idx {args.time_end_idx}"
            cli_ += f" --time_slice_pe {args.time_slice_pe} --time_slice_im {args.time_slice_im}"
            cli_ += f" --sigma {args.sigma} --algo {args.algo} --telescope {args.telescope}"
            cli_ += f" --channel_id {args.channel_id} --outname {args.outname} --nufft_eps 0.00001"
            cli_ += f" --filter_negative_eigenvalues {args.filter_negative_eigenvalues}"
            cli_ += f" " #keep space at the end!
            
            if nsta == 0: # case all stations
                for pixw in np.sort(pixWidths):
                    #cmd = ["python3", "get_scale.py", "--pixw", str(pixw)]
                    #p = subprocess.run(cmd, capture_output=True)
                    #wsc_scale = p.stdout.decode("utf-8").strip()
                    outdir = os.path.join(args.out_dir, str(nsta), str(nlev), str(pixw))
                    #cli = cli_ + f"--nsta 0 --pixw {pixw} --wsc_scale {args.wsc_scale} --output_directory {outdir}\n"
                    cli = cli_ + f"--nsta 0 --pixw {pixw} --output_directory {outdir}\n"
                    f.write(cli)
            else:
                if nlev <= nsta:
                    for pixw in np.sort(pixWidths):
                        cmd = ["python3", "get_scale.py", "--pixw", str(pixw)]
                        #p = subprocess.run(cmd, capture_output=True)
                        #wsc_scale = p.stdout.decode("utf-8").strip()
                        outdir = os.path.join(args.out_dir, str(nsta), str(nlev), str(pixw))
                        #cli = cli_ + f"--nsta {nsta} --pixw {pixw} --wsc_scale {args.wsc_scale} --output_directory {outdir}\n"
                        cli = cli_ + f"--nsta {nsta} --pixw {pixw} --output_directory {outdir}\n"
                        f.write(cli)
