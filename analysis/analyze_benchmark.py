import os
import sys
import json
import numpy as np
import benchtb
import argparse
import matplotlib.pyplot as plt


def check_args(args_in):
    print("-I- command line arguments =", args_in)
    parser = argparse.ArgumentParser(args_in)
    parser.add_argument("--bench_root", help="Root directory of benchmark to analyis", required=True)
    args = parser.parse_args()
    if not os.path.isdir(args.bench_root):
        raise Exception("-E- Benchmark directory not found")
    return args


def plot_wsclean_vs_bipp(nsta, nlev, cluster, pixws):
    print(f"@@@ plot_wsclean_vs_bipp nsta={nsta}, nlev={nlev} @@@")
    paths_sols = benchtb.get_list_of_solutions(bench_root=args.bench_root, nsta=nsta, nlev=nlev, pixw=2048)
    print("paths_sols =\n", paths_sols)

    marker_size = 8
    line_width  = 0.5

    sols = {
        "pcs" : {
            'path': f"pypeline/{cluster}/none/gcc/double/ss",
            'legend': 'Bluebild CPU SS',
            'color': 'black',
            'marker': 'x', 'markersize': marker_size, 'linestyle': '-', 'linewidth': line_width
        },
        "bgs" : {
            'path': f"bipp/{cluster}/gpu/cuda/double/ss",
            'legend': 'BIPP GPU SS',
            'color': '#76B900',
            'marker': 'p', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        },
        "bcs" : {
            'path': f"bipp/{cluster}/cpu/gcc/double/ss",
            'legend': 'BIPP CPU SS',
            'color': 'red',
            'marker': 'p', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        },
        "bgn" : {
            'path': f"bipp/{cluster}/gpu/cuda/double/nufft",
            'legend': 'BIPP GPU NUFFT',
            'color': '#76B900',
            'marker': '*', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        },
        "bcn" : {
            'path': f"bipp/{cluster}/cpu/gcc/double/nufft",
            'legend': 'BIPP CPU NUFFT',
            'color': 'red',
            'marker': '*', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        }
    }

    print("sols =\n", sols)
    t_min = sys.float_info.max
    t_max = sys.float_info.min
    all_wsc  = {}
    all_casa = {}
    for pixw in sorted(pixws):
        all_wsc[pixw]  = []
        all_casa[pixw] = []

    for sol in sols:
        print(sol, "@@", sols[sol])
        path_lev = os.path.join(args.bench_root, sols[sol]['path'], str(nsta), str(nlev))
        if not os.path.isdir(path_lev): raise Exception(f"path {path_lev} does not exist.")
        sols[sol]['tts'] = {'bipp': [], 'casa': [], 'wsc': []}
        for pixw in sorted(pixws):
            path_pixw = os.path.join(path_lev, str(pixw))
            if not os.path.isdir(path_pixw): raise Exception(f"path {path_pixw} does not exist.")
            print(f" .. {path_pixw}")
            bipp_stats = benchtb.read_bipp_json_stat_file(path_pixw)
            casa_stats = benchtb.read_casa_json_stat_file(path_pixw)
            wsc_stats  = benchtb.read_dirty_wsclean_json_stat_file(path_pixw)
            print(f"   -> {bipp_stats}")
            print(f"   -> {casa_stats}")
            print(f"   -> {wsc_stats}")
            if bipp_stats is not None:
                sols[sol]['tts']['bipp'].append((pixw, bipp_stats['time']['real']))
            if casa_stats is not None:
                sols[sol]['tts']['casa'].append((pixw, casa_stats['timings']['t_tclean']))
                all_casa[pixw].append(casa_stats['timings']['t_tclean'])
            if wsc_stats is not None:
                sols[sol]['tts']['wsc'].append((pixw, wsc_stats['timings']['t_inv']))
                all_wsc[pixw].append(wsc_stats['timings']['t_inv'])
            for pkg in 'bipp', 'casa', 'wsc':
                if sols[sol]['tts'][pkg][-1][1] < t_min: t_min = sols[sol]['tts'][pkg][-1][1]
                if sols[sol]['tts'][pkg][-1][1] > t_max: t_max = sols[sol]['tts'][pkg][-1][1]


    print("---- plotting")
    print(sols)
    print(all_casa)
    x_min = np.power(2, np.floor(np.log2(np.min(pixws)))) / 1.2
    x_max = np.power(2, np.ceil(np.log2(np.max(pixws)))) * 1.2
    y_min = np.power(10, np.floor(np.log10(t_min)))
    y_max = np.power(10, np.ceil(np.log10(t_max)))

    fig, ax = plt.subplots()
    ax.set_yscale("log", basey=10)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log", basex=2)
    ax.set_xlim(x_min, x_max)

    for sol in sols:
        print(sol, sols[sol]['legend'])
        x, y = zip(*sols[sol]['tts']['bipp'])
        ax.plot(x, y, label=sols[sol]['legend'], color=sols[sol]['color'],
                marker=sols[sol]['marker'], markersize=sols[sol]['markersize'],
                linewidth=sols[sol]['linewidth'], linestyle=sols[sol]['linestyle'])

    # Add averaged CASA
    sol = 'casa'
    sols[sol] = {'tts': [],
                 'legend': 'CASA',
                 'color': 'pink',
                 'marker': 'H', 'markersize': marker_size, 'linestyle': '-', 'linewidth': line_width
    }
    for pixw in sorted(pixws):
        sols[sol]['tts'].append((pixw, np.average(all_casa[pixw])))
    x, y = zip(*sols[sol]['tts'])
    ax.plot(x, y, label=sols[sol]['legend'], color=sols[sol]['color'],
            marker=sols[sol]['marker'], markersize=sols[sol]['markersize'],
            linewidth=sols[sol]['linewidth'], linestyle=sols[sol]['linestyle'])

    # Add averaged WSClean
    sol = 'wsc'
    sols[sol] = {'tts': [],
                 'legend': 'WSClean',
                 'color': 'skyblue',
                 'marker': '+', 'markersize': marker_size, 'linestyle': '--', 'linewidth': line_width
    }
    for pixw in sorted(pixws):
        sols[sol]['tts'].append((pixw, np.average(all_casa[pixw])))
    x, y = zip(*sols[sol]['tts'])
    ax.plot(x, y, label=sols[sol]['legend'], color=sols[sol]['color'],
            marker=sols[sol]['marker'], markersize=sols[sol]['markersize'],
            linewidth=sols[sol]['linewidth'], linestyle=sols[sol]['linestyle'])

    ax.set(xlabel='Image width [pixel]', ylabel='Time to solution [s]')

    ax.legend(fontsize=8)

    msg = f"{nlev} energy level" if nlev == 1 else f"{nlev} energy levels"
    ax.text(x_min * 1.03, y_min * 1.1, msg, family='monospace')


    fig.savefig(f"tts_{nlev}_energy_levels.png")


if __name__ == "__main__":
    args = check_args(sys.argv)
    print(args)

    tree = benchtb.get_benchmark_hierarchy(args.bench_root)
    print("tree =\n", tree, "\n")

    plot_wsclean_vs_bipp(nlev=1, nsta=0, cluster='izar', pixws=tree['pixws'])
    plot_wsclean_vs_bipp(nlev=8, nsta=0, cluster='izar', pixws=tree['pixws'])

    sys.exit(0)


"""
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
"""
