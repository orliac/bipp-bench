import sys
import os
import json
import benchtb
import argparse
import numpy as np
import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def check_args(args_in):
    parser = argparse.ArgumentParser(args_in)
    parser.add_argument("--sky_file", help="Json file containing sky information", required=True)
    parser.add_argument("--outdir",   help="Output directory", required=True)
    parser.add_argument("--wsc_size", help="WSClean size paramater", required=True, type=int)
    parser.add_argument("--wsc_scale", help="WSClean scale paramater", required=True, type=int)
    args = parser.parse_args()
    if not os.path.isfile(args.sky_file):
        raise Exception("-E- Sky file to analyze not found")
    return args


if __name__ == "__main__":
    args = check_args(sys.argv)
    print(args)

    # dummy plots, just to get the Path objects
    a = plt.scatter([1,2],[3,4], marker='s')
    b = plt.scatter([1,2],[3,4], marker='^')
    square_mk,      = a.get_paths()
    triangle_up_mk, = b.get_paths()
    a.remove()
    b.remove()

    basename = os.path.basename(args.sky_file)
    basename = os.path.splitext(basename)[0]
    
    add_label = False
    if args.wsc_size == 256:
        add_label = True
        
    sky_data = benchtb.read_json_file(args.sky_file)
    #print(sky_data)
    #print(sky_data['CASA'])
    #print(sky_data['WSClean'])
    #print(sky_data['Bluebild'])
    ref_px = sky_data['ref_px']
    ref_py = sky_data['ref_py']
    print("ref_pix =", ref_px, ref_py)

    plot = {}
    sol_colors = {'CASA': 'pink', 'WSClean': 'skyblue', 'Bluebild': 'red'}
    #sol_colors = {'CASA': 'lightgray', 'WSClean': 'dimgray', 'Bluebild': 'black'}

    list_sols = ['CASA', 'WSClean', 'Bluebild']
    min_dist = -0.2
    max_recovery = max_dist = 0
    min_recovery = 1E10
    for sol in list_sols:
        data = sky_data[sol]
        #print(data)
        plot[sol] = {}
        
        for source in data:
            #print(source)
            dx = source['simulated']['px'] - ref_px
            dy = source['simulated']['py'] - ref_py
            dist_to_center = int(np.sqrt(dx*dx + dy*dy))
            loss = (source['recovered']['intensity'] - source['simulated']['intensity']) / source['simulated']['intensity'] * -100
            recovery = 100 - loss
            if recovery > max_recovery:
                max_recovery = recovery
            if recovery < min_recovery:
                min_recovery = recovery
            dist = source['recovered']['dist']
            if dist > max_dist:
                max_dist = dist
            #print(f"{dx} {dy}, {dist_to_center} => loss = {loss:.1f} %")
            if not dist_to_center in plot[sol]:
                plot[sol][dist_to_center] = {'loss': [], 'dist': [], 'recovery': []}
            plot[sol][dist_to_center]['loss'].append(loss)
            plot[sol][dist_to_center]['dist'].append(dist)
            plot[sol][dist_to_center]['recovery'].append(recovery)
        #print(plot)

    #print(plot['CASA'])

    max_dist = max_dist + 0.2
    max_recovery = math.ceil((max_recovery + 0.5) / 10) * 10
    min_recovery = math.floor((min_recovery - 0.5) / 10) * 10

    # Generate 2 plot types (pt), one so that all res can be combined horizontally
    # and a second type to be used independently

    basename_ = basename.replace('9_sources_', '9s_')
    
    for pt in 'comb', 'indep':
        
        basename = basename_ + f"_{pt}"

        fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [1.4, 1]})
        fig.set_figwidth(6)
        fig.set_figheight(8)

        dfs = []
        flats = []
        flats_keys = []
        colors = []
        for sol in plot:
            colors.append(sol_colors[sol])
            flat = {}
            for k in plot[sol]:
                #flat[k] = pd.Series(plot[sol][k]['loss'])
                flat[k] = pd.Series(plot[sol][k]['recovery'])
            df = pd.DataFrame(flat)
            flats.append(df)
            flats_keys.append(sol)
            df = pd.concat(flats, keys=flats_keys, axis=1, names=['package','cfov_dist'])#.stack(0)
            df = pd.melt(df)
            #print(df)

        my_palette = sns.set_palette(sns.color_palette(colors))
    
        sns.swarmplot(data=df, x='cfov_dist', y='value', hue='package',
                      size=9, palette=my_palette, ax=axes[0], dodge=True)

        N_hues = len(pd.unique(df.cfov_dist))
        swarm_coll = axes[0].collections
        ncol = len(swarm_coll)
        assert(ncol % N_hues == 0)
    
        for i in range(0, int(ncol/N_hues)):
            swarm_coll[i*N_hues + 0].set_paths([triangle_up_mk])
            swarm_coll[i*N_hues + 1].set_paths([square_mk])

        axes[0].legend(swarm_coll[-3:], list_sols)
        
        axes[0].legend(frameon=True, loc='lower left', fontsize=12, markerscale=1.2)
        axes[0].set(xlabel=None)
        axes[0].set_xticklabels([])
        axes[0].set_ylabel('Source intensity recovery\n[%]', fontsize=15)
        if not add_label and pt == 'comb':
            axes[0].set_ylabel('', fontsize=15)
        if pt == 'comb':
            axes[0].set_ylim(40, 160)
        else:
            axes[0].set_ylim(min_recovery, max_recovery)
        axes[0].axhline(100.0, color='lightgray', linestyle='dashed')
        axes[0].set_title(f"{args.wsc_size} x {args.wsc_size} x {args.wsc_scale}\"", fontsize=24)

        axes[0].tick_params(axis='x', labelsize=12, top=True)
        axes[0].tick_params(axis='y', labelsize=12, right=True)

        axes[0].axvline(x = 0.5, color = 'lightgray')
        axes[0].axvline(x = 1.5, color = 'lightgray')
    
        dfs = []
        flats = []
        flats_keys = []
        colors = []
        for sol in plot:
            colors.append(sol_colors[sol])
            flat = {}
            for k in plot[sol]:
                flat[k] = pd.Series(plot[sol][k]['dist'])
            df = pd.DataFrame(flat)
            flats.append(df)
            flats_keys.append(sol)
        df = pd.concat(flats, keys=flats_keys, axis=1, names=['package','cfov_dist'])
        df = pd.melt(df)
        #print(df)

        my_palette = sns.set_palette(sns.color_palette(colors))
    
        sns.swarmplot(data=df, x='cfov_dist', y='value', hue='package',
                      size=10, palette=my_palette, ax=axes[1], dodge=True)
    
        swarm_coll = axes[1].collections
        ncol = len(swarm_coll)
        assert(ncol % N_hues == 0)
    
        for i in range(0, int(ncol/N_hues)):
            swarm_coll[i*N_hues + 0].set_paths([triangle_up_mk])
            swarm_coll[i*N_hues + 1].set_paths([square_mk])

        axes[1].set_xlabel("Distance between simulated source and \ncenter of field of view [pixel]", fontsize=15)
        axes[1].set_ylabel('Recoverd position error\n[pixel]', fontsize=15)
        if not add_label and pt == 'comb':
            axes[1].set_ylabel('', fontsize=15)
        if pt == 'comb':
            axes[1].set_ylim(-0.2, 3.1)
        else:
            axes[1].set_ylim(-0.2, max_dist)
            
        axes[1].get_legend().remove()

        axes[1].tick_params(axis='x', labelsize=12, top=True)
        axes[1].tick_params(axis='y', labelsize=12, right=True)
        
        axes[1].axvline(x = 0.5, color = 'lightgray')
        axes[1].axvline(x = 1.5, color = 'lightgray')
        axes[1].axhline(0.0, color='lightgray', linestyle='dashed')
        
    
        for dpi in 100,:
            for fmt in '.png', '.pdf':
                plot_file = os.path.join(args.outdir, basename + '_rec_sky_dpi_'+ str(dpi) + fmt)        
                plt.savefig(plot_file, bbox_inches='tight', dpi=dpi)
                print("-I-", plot_file)
