#!/bin/bash

set -e


# options may be followed by one colon to indicate they have a required argument
ARGUMENT_LIST=(
    "pipeline"
    "bench-name"
    "outdir"
    "package"
    "proc-unit"
    "compiler"
    "cluster"
    "slurm-opts"
    "algo"
    "telescope"
    "time-start-idx"
    "time-end-idx"
    "time-slice-pe"
    "time-slice-im"
    "sigma"
    "wsc_scale"
    "precision"
    "filter_negative_eigenvalues"
    "channel_id"
)

# debug line
getopt -u -o "" -l "$(printf "%s:," "${ARGUMENT_LIST[@]}")" -- "$@"

if ! options=$(getopt -u -o "" -l "$(printf "%s:," "${ARGUMENT_LIST[@]}")" -- "$@")
then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

set -- $options

while [ $# -gt 0 ]
do
    case "$1" in
        --pipeline)       pipeline="$2";       shift;;
        --bench-name)     bench_name="$2";     shift;;
        --outdir)         outdir="$2";         shift;;
        --package)        package="$2";        shift;;
        --proc-unit)      proc_unit="$2";      shift;;
        --compiler)       compiler="$2";       shift;;
        --cluster)        cluster="$2";        shift;;
        --slurm-opts)     slurm_opts="$2";     shift;;
        --algo)           algo="$2";           shift;;
        --telescope)      telescope="$2";      shift;;
        --time-start-idx) time_start_idx="$2"; shift;;
        --time-end-idx)   time_end_idx="$2";   shift;;
        --time-slice-pe)  time_slice_pe="$2";  shift;;
        --time-slice-im)  time_slice_im="$2";  shift;;
        --sigma)          sigma="$2";          shift;;
        --wsc_scale)      wsc_scale="$2";      shift;;
        --precision)      precision="$2";      shift;;
        --filter_negative_eigenvalues) fne="$2"; shift;;
        --channel_id)     channel_id="$2";        shift;;
        (--) shift; break;;
        (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
        (*) break;;
    esac
    shift
done

: ${pipeline:?Missing mandatory --pipeline option to specify the prefix of the pipeline to run}
: ${bench_name:?Missing mandatory --bench-name option to specify which benchmark to run}
: ${outdir:?Missing mandatory --outdir option}
: ${package:?Missing mandatory --package option to specify which package to run [bipp, pypeline]}
: ${proc_unit:?Missing mandatory --proc-unit  option to specify the processing unit [none, cpu, gpu]}
: ${compiler:?Missing mandatory --compiler option to specify the compiler [gcc, cuda]}
: ${cluster:?Missing mandatory --cluster option to specify the compiler [izar, jed]}
: ${slurm_opts:?Missing mandatory --slurm-opts}
: ${algo:?Missing mandatory --algo option [ss, nufft]}
: ${sigma:?Missing mandatory --sigma option}
: ${precision:?Missing mandatory --precision option [single, double]}
: ${telescope:?Missing mandatory --telescope option}
: ${fne:?Missing mandatory --filter_negative_eigenvalues option}
: ${channel_id:?Missing mandatory --channel_id option}
#: ${fov_deg:?Missing mandatory --fov_deg option}
: ${wsc_scale:?Missing mandatory --wsc_scale option}

if [[ $sigma < 0.0 ]] || [[ $sigma > 1.0 ]]; then
    echo "-E- Invalid value for sigma ($sigma). Must be > 0.0 and < 1.0"
    exit 1
fi
if [ "$pipeline" != "lofar_bootes" ] && [ "$pipeline" != "real_lofar_bootes" ] && \
    [ "$pipeline" != "slim_lofar_bootes" ] && [ "$pipeline" != "skalow_ms" ] && \
    [ "$pipeline" != "ms" ]; then
    echo "-E- Pipeline not reckognized"
    exit 1
fi
if [ "$algo" != "ss" ] && [ "$algo" != "nufft" ]; then
    echo "-E- Only algo ss and nufft are allowed"
    exit 1
fi
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
if [ "$fne" != "0" ] && [ "$fne" != "1" ]; then
    echo "-E- Incompatible option $fne for fne."
    exit 1
fi

outname=${telescope}_${package}_${algo}_${proc_unit}_${fne}

#
## Generate benchmark input file
#
module purge
module load gcc python
out_dir=${outdir}/${bench_name}/${package}/${cluster}/${proc_unit}/${compiler}/${precision}/${algo}
[ ! -d $out_dir ] && mkdir -pv $out_dir
in_file="$out_dir/benchmark.in"

python generate_benchmark_input_file.py \
    --pipeline=$pipeline \
    --bench_name=$bench_name --proc_unit=$proc_unit \
    --compiler=$compiler --cluster=$cluster --precision=$precision --package=$package \
    --in_file=$in_file --out_dir=${out_dir} \
    --time_start_idx=$time_start_idx --time_end_idx=$time_end_idx \
    --time_slice_pe=$time_slice_pe --time_slice_im=$time_slice_im \
    --sigma=$sigma \
    --wsc_scale=$wsc_scale \
    --algo=$algo --telescope=$telescope --filter_negative_eigenvalues=$fne \
    --channel_id=$channel_id --outname=$outname


    
[ $? -eq 0 ] || (echo "-E- $ python generate_benchmark_input_file.py ... failed" && exit 1)
echo "-I- Generated input file $in_file"

module purge

# Count the number of lines in $in_file
NJOBS=$(wc -l < $in_file)
[[ $NJOBS > 0 ]] || (echo "NJOBS ($NJOBS) must be strictly positive!" && exit 1)
NJOBS=$(($NJOBS-1))


# Copy template files
SBATCH_SH=sbatch_generic.sh
[ -f $SBATCH_SH ] || (echo "-E- Template submission file for $package on $cluster \
>>$SBATCH_SH<< not found" && exit 1)
cp -v ${pipeline}_${algo}_${package}.py  $out_dir
cp -v $SBATCH_SH                         $out_dir
cp -v bipptb.py                          $out_dir
cp -v wscleantb.py                       $out_dir
cp -v casatb.py                          $out_dir
cp -v casa_tclean.py                     $out_dir
cp -v plots.py                           $out_dir
cp -v wsclean_log_to_json.py             $out_dir
cp -v casa_log_to_json.py                $out_dir
cp -v add_time_stats_to_bluebild_json.py $out_dir
cp -v get_ms_timerange.py                $out_dir
cp -v get_scale.py                       $out_dir
#cp -r $ms_file                           $out_dir
cp -v oskar_sim.py                       $out_dir

cd $out_dir

slurm_opts=$(sed "s/|/ /g" <<< $slurm_opts)
slurm_opts="$slurm_opts --array 0-${NJOBS}"
[ "$cluster" == "izar" ] && slurm_opts="$slurm_opts%16" ### Adapt here for max simulatneous jobs
#[ "$cluster" == "izar" ] && slurm_opts="$slurm_opts%10" ### Adapt here for max simulatneous jobs
echo
echo "slurm_opts >>$slurm_opts<<"
echo

sbatch $slurm_opts $SBATCH_SH --cluster $cluster --package $package --pipeline $pipeline --algo $algo

cd -
echo "-I- $SBATCH_SH submitted from $out_dir"
