import os
import sys
import json
import numpy
import benchtb

### Input: benchmark's root directory

bench_root = "/work/ska/orliac/benchmarks/paper_dirty-image"
if not os.path.isdir(bench_root):
    raise Exception("-E- Benchmark directory not found")

tree = benchtb.get_benchmark_hierarchy(bench_root)
#print(tree)

def analyze_triplet_nsta_nlev_pixw(bench_root, nsta, nlev, pixw):
        paths_sols = benchtb.get_list_of_solutions(bench_root=bench_root, nsta=nsta, nlev=nlev, pixw=pixw)
        print("=========================================================================")
        print(f"nsta = {nsta}, nlev = {nlev}, pixw = {pixw}")
        for path_sol in paths_sols:
            bipp_stats = benchtb.read_bipp_json_stat_file(path_sol)
            casa_stats = benchtb.read_casa_json_stat_file(path_sol)
            wsc_stats  = benchtb.read_wsclean_json_stat_file(path_sol)
            specs = benchtb.get_solution_specs_from_path(path_sol=path_sol, bench_root=bench_root)
            casa_report = f"casa: {casa_stats['timings']['t_tclean']:5.2f}s"
            wsc_report  = f"wsclean: {wsc_stats['timings']['t_inv']:5.2f} s,"
            if bipp_stats:
                bipp_report  = f"{specs['package']} {specs['proc_unit']} {specs['algo']:5s}: "
                bipp_report += f"{bipp_stats['timings']['ifpe']:5.1f} {bipp_stats['timings']['ifim']:5.1f} "
                bipp_report += f"{bipp_stats['timings']['sfpe']:5.1f} {bipp_stats['timings']['sfim']:5.1f} "
                bipp_report += f"{bipp_stats['time']['real']:7.2f} s,"
            else:
                bipp_report = f"{specs['package']} {specs['proc_unit']} {specs['algo']:5s}:  {'(failed)':32s},"
            print(bipp_report, wsc_report, casa_report)
        print("=========================================================================")

analyze_triplet_nsta_nlev_pixw(bench_root=bench_root, nsta=0, nlev=4, pixw=512)
analyze_triplet_nsta_nlev_pixw(bench_root=bench_root, nsta=0, nlev=4, pixw=1024)
analyze_triplet_nsta_nlev_pixw(bench_root=bench_root, nsta=0, nlev=4, pixw=2048)
