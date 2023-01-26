#!/bin/bash

set -e

module purge
module load gcc python

# Generate benchmark input file
outdir="/work/ska/orliac/benchmarks"
bench_name=bench02
package=pypeline
cluster=izar
proc_unit=gpu
compiler=cuda
precision=double

in_dir="$outdir/$bench_name/$package/$cluster/$proc_unit/$compiler/$precision"
[ ! -d $in_dir ] && mkdir -pv $in_dir

in_file="$in_dir/benchmark.in"
echo "-I- Will generate input file $in_file"

python generate_benchmark_input_file.py --bench_name=$bench_name --proc_unit=$proc_unit \
    --compiler=$compiler --cluster=$cluster --precision=$precision --out_dir=$outdir \
    --in_file=$in_file
[ $? -eq 0 ] || (echo "-E- $ python generate_benchmark_input_file.py ... failed" && exit 1)

echo
cat $in_file
echo

pwd
ls -l

module purge

SBATCH_SH=sbatch_pypeline_izar.sh

cp -v lofar_bootes_ss_pypeline.py $in_dir
cp -v $SBATCH_SH                  $in_dir
cp -v bipptb.py                   $in_dir

cd $in_dir
ls -l

#cat $SBATCH_SH
sbatch $SBATCH_SH
