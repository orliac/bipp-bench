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
parser.add_argument("--out_dir", help="Output directory", required=True)
parser.add_argument("--in_file", help="Path to benchmark input file", required=True)

args = parser.parse_args()

benchName = args.bench_name # 'bench00'
cluster   = args.cluster    # 'izar'
procUnit  = args.proc_unit  # 'cpu'
compiler  = args.compiler   # 'gcc'
precision = args.precision  # 'double'
outDir    = args.out_dir    #
inFile    = args.in_file

pixWidths = [128, 256] # 512, 1024, 2048, 4096]
nLevels   = [16, 32] #, 64]
nStations = [32, 64] #, 128, 256, 512]

with open(inFile, 'w') as f:
    for nsta in np.sort(nStations):
        for nlev in np.sort(nLevels):
            for pixw in np.sort(pixWidths):
                outdir = os.path.join(outDir, benchName, cluster, procUnit, compiler, precision,
                                      str(nsta), str(nlev), str(pixw))
                cli  = f"--outdir {outdir} --cluster {cluster} --processing_unit {procUnit} --compiler {compiler} "
                cli += f"--precision {precision} --nsta {nsta} --nlev {nlev} --pixw {pixw}\n"
                f.write(cli)
f.close()
