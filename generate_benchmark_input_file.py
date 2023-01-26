# Generate Slurm job array input file

import os
import sys
import numpy as np
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--bench_name", help="Benchmark name", required=True)
parser.add_argument("--cluster",    help="Cluster to run on", required=True,
                    choices=['izar', 'jed'])
parser.add_argument("--proc_unit",  help="Processing unit", required=True,
                    choices=['auto', 'cpu', 'gpu', 'none'])
parser.add_argument("--compiler",   help="Compiler to use", required=True,
                    choices=['cuda', 'rocm', 'gcc', 'intel'])
parser.add_argument("--precision", help="FP precision", required=True,
                    choices=['single', 'double'])
parser.add_argument("--package", help="Package to use: bipp or pypeline", required=True,
                    choices=['bipp', 'pypeline'])
parser.add_argument("--out_dir", help="Output directory", required=True)
parser.add_argument("--in_file", help="Path to benchmark input file", required=True)

args = parser.parse_args()

benchName = args.bench_name
cluster   = args.cluster
procUnit  = args.proc_unit
compiler  = args.compiler
precision = args.precision
package   = args.package
outDir    = args.out_dir
inFile    = args.in_file

if package == "bipp" and procUnit == "none":
    print("-E- bipp cannot run none")
    sys.exit(1)

# Set up jobs to run in the benchmark
pixWidths = [128, 256] # 512, 1024, 2048, 4096]
nLevels   = [16, 32] #, 64]
nStations = [32, 64] #, 128, 256, 512]

with open(inFile, 'w') as f:
    for nsta in np.sort(nStations):
        for nlev in np.sort(nLevels):
            for pixw in np.sort(pixWidths):
                outdir = os.path.join(outDir, str(nsta), str(nlev), str(pixw))
                cli  = f"--outdir {outdir} --cluster {cluster} --processing_unit {procUnit} --compiler {compiler} "
                cli += f"--precision {precision} --package {package} --nsta {nsta} --nlev {nlev} --pixw {pixw}\n"
                f.write(cli)
f.close()
