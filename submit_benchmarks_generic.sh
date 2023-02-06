#!/bin/bash

set -e

# options may be followed by one colon to indicate they have a required argument
if ! options=$(getopt -u -o "" -l bench-name:,package:,proc-unit:,compiler:,cluster:,slurm-opts:,algo: -- "$@")
then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

set -- $options

while [ $# -gt 0 ]
do
    case "$1" in
        --bench-name) bench_name="$2"; shift;;
        --package)    package="$2";    shift;;
        --proc-unit)  proc_unit="$2";  shift;;
        --compiler)   compiler="$2";   shift;;
        --cluster)    cluster="$2";    shift;;
        --slurm-opts) slurm_opts="$2"; shift;;
        --algo)       algo="$2";       shift;;
        (--) shift; break;;
        (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
        (*) break;;
    esac
    shift
done

: ${bench_name:?Missing mandatory --bench-name option to specify which benchmark to run}
: ${package:?Missing mandatory --package option to specify which package to run [bipp, pypeline]}
: ${proc_unit:?Missing mandatory --proc-unit  option to specify the processing unit [none, cpu, gpu]}
: ${compiler:?Missing mandatory --compiler option to specify the compiler [gcc, cuda]}
: ${cluster:?Missing mandatory --cluster option to specify the compiler [izar, jed]}
: ${slurm_opts:?Missing mandatory --slurm-opts}
: ${algo:?Missing mandatory --algo [ss, nufft]}

if [ "$package" != "bipp" ] && [ "$package" != "pypeline" ]; then
    echo "-E- Only packages bipp and pypeline are allowed"
    exit 1
fi
if [ "$proc_unit" != "none" ] && [ "$proc_unit" != "gpu" ] && \
    [ "$proc_unit" != "cpu" ]; then
    echo "-E- Allowed proc units: none, gpu, cpu."
    exit 1
fi
if [ "$compiler" != "gcc" ] && [ "$compiler" != "cuda" ]; then
    echo "-E- Unknown compiler specified. Allowed are: gcc, cuda."
    exit 1
fi
if [ "$cluster" != "izar" ] && [ "$cluster" != "jed" ]; then
    echo "-E- Unknown cluster specified. Allowed are: izar, jed."
    exit 1
fi
if [ "$cluster" == "jed" ] && [ "$compiler" == "cuda" ]; then
    echo "-E- Incompatible options $cluster and $compiler."
    exit 1
fi
if [ "$proc_unit" == "gpu" ] && [ "$compiler" != "cuda" ]; then # or rocm later on
    echo "-E- Incompatible options $proc_unit and $compiler."
    exit 1
fi
if [ "$proc_unit" == "cpu" ] && [ "$compiler" != "gcc" ]; then  # or intel later on
    echo "-E- Incompatible options $proc_unit and $compiler."
    exit 1
fi

#
## Generate benchmark input file
#
module purge
module load gcc python
OUTDIR="/work/ska/orliac/benchmarks"
PRECISION="double"
out_dir="$OUTDIR/$bench_name/$package/$cluster/$proc_unit/$compiler/$PRECISION/$algo"
[ ! -d $out_dir ] && mkdir -pv $out_dir
in_file="$out_dir/benchmark.in"
python generate_benchmark_input_file.py --bench_name=$bench_name --proc_unit=$proc_unit \
    --compiler=$compiler --cluster=$cluster --precision=$PRECISION --package=$package \
    --in_file=$in_file --out_dir=$out_dir
[ $? -eq 0 ] || (echo "-E- $ python generate_benchmark_input_file.py ... failed" && exit 1)
echo "-I- Generated input file $in_file"
module purge

# Count the number of lines in $in_file
NJOBS=$(wc -l < $in_file)
echo NJOBS = $NJOBS
NJOBS=$(($NJOBS-1))

#
## Copy template files
# 
#SBATCH_SH=sbatch_${package}_${cluster}.sh
SBATCH_SH=sbatch_generic.sh

[ -f $SBATCH_SH ] || (echo "-E- Template submission file for $package on $cluster \
>>$SBATCH_SH<< not found" && exit 1)

cp -v lofar_bootes_${algo}_${package}.py $out_dir
cp -v $SBATCH_SH                         $out_dir
cp -v bipptb.py                          $out_dir

cd $out_dir

slurm_opts=$(sed "s/|/ /g" <<< $slurm_opts)
slurm_opts="$slurm_opts --array 0-${NJOBS}"
[ "$cluster" == "izar" ] && slurm_opts="$slurm_opts%4"
echo
echo "slurm_opts >>$slurm_opts<<"
echo

sbatch $slurm_opts $SBATCH_SH --cluster $cluster --package $package

cd -
echo "-I- $SBATCH_SH submitted from $in_dir"
