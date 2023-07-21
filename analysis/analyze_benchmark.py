import os
import sys
import json
import numpy as np
import benchtb
import argparse
import matplotlib.pyplot as plt


def check_args(args_in):
    #print("-I- command line arguments =", args_in)
    parser = argparse.ArgumentParser(args_in)
    parser.add_argument("--bench_root", help="Root directory of benchmark to analyis", required=True)
    parser.add_argument("--telescope",  help="Instrument", required=True, choices=['LOFAR', 'MWA', 'SKALOW'])
    args = parser.parse_args()
    if not os.path.isdir(args.bench_root):
        raise Exception("-E- Benchmark directory not found")
    return args

            
def define_all_solutions():

    marker_size = 8
    line_width  = 0.5

    all_sols = {
        "pcs" : {
            'path': f"pypeline/{CLUSTER}/none/gcc/{PRECISION}/ss",
            'legend': 'Bluebild CPU SS',
            'name': 'BluebildSsCpu',
            'color': 'black',
            'marker': 'x', 'markersize': marker_size, 'linestyle': '-', 'linewidth': line_width
        },
        "bgs" : {
            'path': f"bipp/{CLUSTER}/gpu/cuda/{PRECISION}/ss",
            'legend': 'BIPP GPU SS',
            'name': 'BippSsGpu',
            'color': '#76B900',
            'marker': 'p', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        },
        "bcs" : {
            'path': f"bipp/{CLUSTER}/cpu/gcc/{PRECISION}/ss",
            'legend': 'BIPP CPU SS',
            'color': 'red',
            'name': 'BippSsCpu',
            'marker': 'p', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        },
        "bgn" : {
            'path': f"bipp/{CLUSTER}/gpu/cuda/{PRECISION}/nufft",
            'legend': 'BIPP GPU NUFFT',
            'name': 'BippNufftGpu',
            'color': '#76B900',
            'marker': '*', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        },
        "bcn" : {
            'path': f"bipp/{CLUSTER}/cpu/gcc/{PRECISION}/nufft",
            'legend': 'BIPP CPU NUFFT',
            'name': 'BippNufftCpu',
            'color': 'red',
            'marker': '*', 'markersize': marker_size, 'linestyle': 'dashed', 'linewidth': line_width
        }
    }

    return all_sols


def plot_wsclean_vs_bipp(nsta, nlev, pixws):
    print(f"@@@ plot_wsclean_vs_bipp nsta={nsta}, nlev={nlev} @@@")
    paths_sols = benchtb.get_list_of_solutions(bench_root=args.bench_root, nsta=nsta, nlev=nlev, pixw=2048)
    print("paths_sols =\n", paths_sols)
    
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
            print(f" path_pixw = {path_pixw}")
            specs = benchtb.get_solution_specs_from_path(path_pixw, args.bench_root)
            json_base = os.path.join(path_pixw, f"{args.telescope}_{specs['package']}_{specs['algo']}_{specs['proc_unit']}_0")
            bipp_json = json_base + "_stats.json"
            bipp_stats = benchtb.read_json_file(bipp_json)
            casa_stats = benchtb.read_dirty_casa_json_stat_file(path_pixw)
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

# Min, max, mean, std, rms
def check_solutions_consistency(nsta, nlev, pixw):
    print(f"\n@@@ check_solutions_consistency nsta={nsta}, nlev={nlev}, pixw={pixw} @@@")
    paths_sols = benchtb.get_list_of_solutions(bench_root=args.bench_root, nsta=nsta, nlev=nlev, pixw=pixw)
    #print("paths_sols =\n", paths_sols)
    nsol = len(paths_sols)
    out = {}
    for i in range(0, nsol):
        #print(i, paths_sols[i])
        specs = benchtb.get_solution_specs_from_path(paths_sols[i], args.bench_root)
        #print(specs)
        npyi = os.path.join(paths_sols[i], f"{args.telescope}_{specs['package']}_{specs['algo']}_{specs['proc_unit']}_{FNE}_I_lsq_eq_data.npy")
        dati = np.load(npyi)
        nli, nsi = benchtb.get_solution_nickname_from_path(path_sol=paths_sols[i], bench_root=args.bench_root)
        for j in range(i+1, nsol):
            specs = benchtb.get_solution_specs_from_path(paths_sols[j], args.bench_root)
            npyj = os.path.join(paths_sols[j], f"{args.telescope}_{specs['package']}_{specs['algo']}_{specs['proc_unit']}_0_I_lsq_eq_data.npy")
            datj = np.load(npyj)
            nlj, nsj = benchtb.get_solution_nickname_from_path(path_sol=paths_sols[j], bench_root=args.bench_root)
            #if nli not in out: out[nli] = {}
            if nsi not in out: out[nsi] = {}
            rmse, maxdiff = benchtb.stats_image_diff_compressing_levels(dati, datj)
            #print(f" ADDING  {nli:13s} {nlj:13s} {nlev:2d} {pixw:4d}: RMSE = {rmse:8.2f}, max diff = {maxdiff:8.2f}")
            #out[nli][nlj] = {'rmse': rmse}
            out[nsi][nsj] = {'rmse': rmse}

    return out


def plot_solutions_consistency(nlev, nsta, pixws):
    cons = {}
    for pixw in sorted(pixws):
        cons[pixw] = check_solutions_consistency(nlev=nlev, nsta=0, pixw=pixw)
    print("cons\n", cons)
    
    #sols = ['BluebildCpuSs', 'BippCpuSs',  'BippGpuSs', 'BippCpuNufft', 'BippGpuNufft']
    sols_names = ['pcs', 'bcs',  'bgs', 'bcn', 'bgn']
    pixws_markers = {256: 's', 512: 'o', 1024: '+', 2048: 'x'}
    pixws_colors  = {256: 'violet', 512: 'magenta', 1024: 'green', 2048: 'blue'}
    nsols = len(sols_names)

    #x = np.arange(0, nsols)

    fig, axes = plt.subplots(nsols, 1, sharex=True, sharey=True)
    fig.set_figwidth(7)
    fig.set_figheight(6)
    suptitle = fig.suptitle('RMS [Jy/beam] of differences between solutions', x=0.6, y=1.05, fontsize=16)

    max_rms = -1
    j = 0
    lo = []
    ln = []
    for soly in sols_names:
        ln.append(sols[soly]['name'])
        i_off = 0
        for pixw in sorted(pixws):
            x = []
            y = []
            i = 0
            x_off = -0.2 + i_off * 0.1
            for solx in sols_names:
                if rmse := cons.get(pixw, {}).get(solx, {}).get(soly, {}).get('rmse'):
                    y.append(rmse)
                elif rmse := cons.get(pixw, {}).get(soly, {}).get(solx, {}).get('rmse'):
                    y.append(rmse)
                else:
                    y.append(0.0)
                x.append(i + x_off)
                i += 1
            if np.amax(y) > max_rms: max_rms = np.amax(y)
            print(soly, pixw, x, y)
            l = axes[j].plot(x, y, color=pixws_colors[pixw], marker=pixws_markers[pixw], linestyle='-.',
                             linewidth=0.3, markersize=6)
            if j == 0:
                lo.append(l)
            i_off += 1

        j += 1
    axes[0].set_xticks(range(0, nsols))
    axes[nsols-1].set_xticklabels(ln, rotation=45, ha='right')

    # Hide x labels and tick labels for all but bottom plot.
    j = 0
    for axe, soly in zip(axes, sols):
        axe.set_ylim(-0.4, max_rms * 1.3)
        axe.label_outer()
        axe.set_ylabel(sols[soly]['name'], rotation=0, horizontalalignment='right')
        if j == 0:
            axe.legend(labels=pixws_colors.keys(), title="Image resolution [pixel]",
                       loc='upper center', bbox_to_anchor=(0.5, 2.2),
                       ncol=4, fancybox=True, shadow=True)

        j += 1
    
    # Adding a plot in the figure which will encapsulate all the subplots with axis showing only
    ax = fig.add_subplot(1, 1, 1, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax2 = ax.twinx()
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel('RMS [Jy/Beam]', rotation=-90)
    ax2.yaxis.set_label_coords(1.1, .5)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    #plt.savefig(f"consistency_{nlev}.png", bbox_extra_artists=(suptitle,), bbox_inches="tight")
    fig.savefig(f"consistency_{nlev}.png")


def plot_solutions_ranges(nsta, pixws):
    print(f"@@@ plot_solutions_ranges")

    print(sols)

    #for lev in sorted(tree['nlevs']):
    for nlev in 1,:
        for nsta in sorted(tree['nstas']):
            for pixw in sorted(tree['pixws']):
                for sol in sols.keys():
                    json_dir = os.path.join(args.bench_root, sols[sol]['path'], str(nsta), str(nlev), str(pixw))
                    specs    = benchtb.get_solution_specs_from_path(json_dir, args.bench_root)
                    json_bsn = f"{args.telescope}_{specs['package']}_{specs['algo']}_{specs['proc_unit']}_0_wsc_casa_bb.json"
                    bipp_json = os.path.join(json_dir, json_bsn)
                    #print("bipp_json:", bipp_json)
                    stats = benchtb.read_json_file(bipp_json)
                    #print(stats)
                    wsc_info  = f"wsc: [{stats['bb-wsc']['wsc']['min']:6.2f}, {stats['bb-wsc']['wsc']['max']:6.2f}]"
                    casa_info = f"casa: [{stats['casa-wsc']['casa']['min']:6.2f}, {stats['casa-wsc']['casa']['max']:6.2f}]"
                    bb_info   = f"bb: [{stats['bb-wsc']['bb']['min']:6.2f}, {stats['bb-wsc']['bb']['max']:6.2f}]"
                    print(f"{nlev} {pixw:4d} {wsc_info} {casa_info} {bb_info}")


# Check that solutions are the same regardless the number of clustered energy levels
#
def compare_levels():
    print(f"\n@@@ compare_levels\n")
 
    levels = sorted(tree['nlevs'])
    nlev   = len(levels)
    print(levels, nlev)
    
    to_plot = {}

    for sol in sols.keys():
        to_plot[sol] = {}
        x = []
        y = []
        max_rmse = -1
        min_rmse = 1E10
        for nsta in sorted(tree['nstas']):
            for pixw in sorted(tree['pixws']):
                for i in range(0, nlev):
                    json_dir = os.path.join(args.bench_root, sols[sol]['path'], str(nsta), str(i), str(pixw))
                    specs    = benchtb.get_solution_specs_from_path(json_dir, args.bench_root)
                    npyi_base = f"{args.telescope}_{specs['package']}_{specs['algo']}_{specs['proc_unit']}_{FNE}_I_lsq_eq_data.npy"
                    npyi = os.path.join(args.bench_root, sols[sol]['path'], str(nsta), str(levels[i]), str(pixw), npyi_base)
                    for j in range(i+1, nlev):
                        json_dir = os.path.join(args.bench_root, sols[sol]['path'], str(nsta), str(j), str(pixw))
                        specs    = benchtb.get_solution_specs_from_path(json_dir, args.bench_root)
                        npyi_base = f"{args.telescope}_{specs['package']}_{specs['algo']}_{specs['proc_unit']}_{FNE}_I_lsq_eq_data.npy"
                        npyj = os.path.join(args.bench_root, sols[sol]['path'], str(nsta), str(levels[j]), str(pixw), npyi_base)
                        #print("  ", npyi)
                        #print("  ", npyj)
                        dati = np.load(npyi)
                        datj = np.load(npyj)
                        dati = np.sum(dati, axis=0)
                        datj = np.sum(datj, axis=0)
                        diff = dati - datj                        
                        min_diff, max_diff = np.amin(diff), np.amax(diff)
                        x.append(abs(min_diff))
                        y.append(abs(max_diff))
                        rmse, maxdiff = benchtb.stats_image_diff(dati, datj)
                        print(f"{sol} {nsta} {pixw:4d} {levels[i]} {levels[j]}: [{min_diff:.3e}, {max_diff:.3e}], rmse = {rmse:.3e}, max abs diff = {maxdiff:.3e}")
                        if rmse > max_rmse: max_rmse = rmse
                        if rmse < min_rmse: min_rmse = rmse
            to_plot[sol]['x'] = x
            to_plot[sol]['y'] = y
        print(f" @@@ min, max rmse for {sols[sol]['name']} = {min_rmse:.3e}, {max_rmse:.3e}")
    fig, ax = plt.subplots()
    for sol in to_plot.keys():
        ax.loglog(to_plot[sol]['x'], to_plot[sol]['y'], 'o', label=sols[sol]['name'],
                  color=sols[sol]['color'], marker=sols[sol]['marker'])
    ax.set_xlabel('Solution absolute minimum difference [Jy/beam]')
    ax.set_ylabel('Solution maximum difference [Jy/beam]')
    ax.legend()
    plt.savefig('scatter.png')


if __name__ == "__main__":
    args = check_args(sys.argv)
    print(args)

    tree = benchtb.get_benchmark_hierarchy(args.bench_root)
    print("tree =\n", tree, "\n")

    CLUSTER='izar'
    PRECISION='single'
    FNE=0

    # Define all possible solutions
    all_sols = define_all_solutions()

    # Select solutions that were actually benchmarked 
    #sols = {"bgs": all_sols['bgs']}
    sols = all_sols
    print("sols =\n", sols)

    # Compare same solution over the various energy levels.
    #compare_levels()
    #sys.exit(0)

    #plot_solutions_ranges(nsta=0, pixws=tree['pixws'])
    
    #for nlev in 1, 2, 4, 8:
    #    plot_wsclean_vs_bipp(nlev=nlev, nsta=0, cluster='izar', pixws=tree['pixws'])

    # Show first solutions are equivalent regardless the number of energy levels considered
    # => only plot for a single clustering
    for nlev in 1,:
        plot_solutions_consistency(nlev=nlev, nsta=0, pixws=tree['pixws'])
        
    #sys.exit(0)



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
