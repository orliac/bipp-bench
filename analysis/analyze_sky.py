import sys
import os
import json
import benchtb
import argparse
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    
    for sol in 'CASA', 'WSClean', 'Bluebild':
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
            dist = source['recovered']['dist']
            #print(f"{dx} {dy}, {dist_to_center} => loss = {loss:.1f} %")
            if not dist_to_center in plot[sol]:
                plot[sol][dist_to_center] = {'loss': [], 'dist': [], 'recovery': []}
            plot[sol][dist_to_center]['loss'].append(loss)
            plot[sol][dist_to_center]['dist'].append(dist)
            plot[sol][dist_to_center]['recovery'].append(recovery)
        #print(plot)

    #print(plot['CASA'])

    
    fig, axes = plt.subplots(2,1)
    fig.set_figwidth(5.5)
    fig.set_figheight(9)

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
                  size=9, palette=my_palette, ax=axes[0])

    axes[0].legend(frameon=True, loc='lower left', fontsize=14, markerscale=2)
    axes[0].set(xlabel=None)
    axes[0].set_xticklabels([])
    #axes[0].set_ylabel('Intensity loss [%]', fontsize=12)
    axes[0].set_ylabel('Intensity recovery [%]', fontsize=18)
    if not add_label:
        axes[0].set_ylabel('', fontsize=18)
    axes[0].set_ylim(40, 160)
    axes[0].axhline(100.0, color='lightgray', linestyle='dashed')
    axes[0].set_title(f"{args.wsc_size} x {args.wsc_size} x {args.wsc_scale}\"", fontsize=24)

    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    
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
                  size=10, palette=my_palette, ax=axes[1])
  
    axes[1].set_xlabel("Distance from center of \nfield of view [pixel]", fontsize=18)
    axes[1].set_ylabel('Distance to truth [pixel]', fontsize=18)
    if not add_label:
        axes[1].set_ylabel('', fontsize=18)
        
    axes[1].set_ylim(-0.2, 3.1)
    axes[1].get_legend().remove()

    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    out_png = os.path.join(args.outdir, "recovered_vs_simulated.png")
    plt.savefig(out_png)
    print("-I- Figure saved under:", out_png)
    #plt.show()
    
