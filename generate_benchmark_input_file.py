# Generate Slurm job array input file

import os
import sys
import numpy as np
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--pipeline",      help="Prefix of pipeline to run", required=True)
parser.add_argument("--bench_name",    help="Benchmark name", required=True)
parser.add_argument("--cluster",       help="Cluster to run on", required=True, choices=['izar', 'jed'])
parser.add_argument("--proc_unit",     help="Processing unit", required=True,   choices=['auto', 'cpu', 'gpu', 'none'])
parser.add_argument("--compiler",      help="Compiler to use", required=True,   choices=['cuda', 'rocm', 'gcc', 'intel'])
parser.add_argument("--precision",     help="FP precision", required=True,      choices=['single', 'double'])
parser.add_argument("--package",       help="Package to use", required=True,    choices=['bipp', 'pypeline'])
parser.add_argument("--out_dir",       help="Output directory", required=True)
parser.add_argument("--in_file",       help="Path to benchmark input file", required=True)
parser.add_argument("--ms_file",       help="Path to MS dataset to process", required=False)
parser.add_argument("--time_slice_pe", help="Time slice in parameter estimation", required=False)
parser.add_argument("--time_slice_im", help="Time slice in imaging", required=False)
parser.add_argument("--sigma",         help="Fraction of power spectrum to be considered", required=False)
parser.add_argument("--wsc_scale",     help="WSClean pixel scale in arcsec", required=False)


args = parser.parse_args()

# No reference Python in BIPP
if args.package == "bipp" and args.proc_unit == "none":
    print("-E- bipp cannot run none")
    sys.exit(1)

# Set up jobs to run in the benchmark
#nLevels   = [1, 2, 4, 16, 32, 60]
#pixWidths = [256, 512, 1024, 2048, 4096]
#nStations = [15, 30, 60]

nLevels   = [4]
pixWidths = [1024]
nStations = [0]   # 0 == None == All

with open(args.in_file, 'w') as f:
    for nsta in np.sort(nStations):
        for nlev in np.sort(nLevels):
            cli  = f"--cluster {args.cluster} --processing_unit {args.proc_unit} --compiler {args.compiler} "
            cli += f"--precision {args.precision} --package {args.package} --nlev {nlev}  --wsc_scale {args.wsc_scale} "
            cli += f"--ms_file {args.ms_file} --time_slice_pe {args.time_slice_pe} --time_slice_im {args.time_slice_im} "
            cli += f"--sigma {args.sigma} "
            
            if nsta == 0: # case all stations
                for pixw in np.sort(pixWidths):
                    outdir = os.path.join(args.out_dir, str(nsta), str(nlev), str(pixw))
                    cli += f"--nsta 0 --pixw {pixw} --output_directory {outdir}\n"
                    f.write(cli)
            else:
                if nlev <= nsta:
                    for pixw in np.sort(pixWidths):
                        outdir = os.path.join(args.out_dir, str(nsta), str(nlev), str(pixw))
                        cli += f"--nsta {nsta} --pixw {pixw} --output_directory {outdir}\n"
                        f.write(cli)
f.close()
