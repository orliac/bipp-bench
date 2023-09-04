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
    args = parser.parse_args()
    if not os.path.isfile(args.sky_file):
        raise Exception("-E- Sky file to analyze not found")
    return args


if __name__ == "__main__":
    args = check_args(sys.argv)
    print(args)

    sky_data = benchtb.read_json_file(args.sky_file)
    #print(sky_data['CASA'])
    #print(sky_data['WSClean'])
    #print(sky_data['Bluebild'])
    ref_pix = sky_data['ref_pix']
    print("ref_pix =", ref_pix)

    plot = {}
    sol_colors = {'CASA': 'pink', 'WSClean': 'skyblue', 'Bluebild': 'red'}
    
    for sol in 'CASA', 'WSClean', 'Bluebild':
        data = sky_data[sol]
        #print(data)
        plot[sol] = {}
        
        for source in data:
            print(source)
            dx = source['simulated']['px'] - ref_pix
            dy = source['simulated']['py'] - ref_pix
            dist_to_center = int(np.sqrt(dx*dx + dy*dy))
            loss = (source['recovered']['intensity'] - source['simulated']['intensity']) / source['simulated']['intensity'] * -100
            dist = source['recovered']['dist']
            print(f"{dx} {dy}, {dist_to_center} => loss = {loss:.1f} %")
            if not dist_to_center in plot[sol]:
                plot[sol][dist_to_center] = {'loss': [], 'dist': []}
            plot[sol][dist_to_center]['loss'].append(loss)
            plot[sol][dist_to_center]['dist'].append(dist)
        print(plot)

    print(plot['CASA'])

    
    fig, axes = plt.subplots(2,1)
    fig.set_figwidth(6)
    #fig.set_figheight(6)

    dfs = []
    flats = []
    flats_keys = []
    colors = []
    for sol in plot:
        colors.append(sol_colors[sol])
        flat = {}
        for k in plot[sol]:
            flat[k] = pd.Series(plot[sol][k]['loss'])
        df = pd.DataFrame(flat)
        flats.append(df)
        flats_keys.append(sol)
    df = pd.concat(flats, keys=flats_keys, axis=1, names=['package','cfov_dist'])#.stack(0)
    df = pd.melt(df)
    #print(df)

    my_palette = sns.set_palette(sns.color_palette(colors))
    
    sns.swarmplot(data=df, x='cfov_dist', y='value', hue='package',
                  size=10, palette=my_palette, ax=axes[0])

    axes[0].legend(frameon=False)
    axes[0].set(xlabel=None)
    axes[0].set_xticklabels([])
    axes[0].set_ylabel('Intensity loss [%]', fontsize=12)

    
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
  
    axes[1].set_xlabel('Distance from center of field of view [pixel]', fontsize=12)
    axes[1].set_ylabel('Distance to truth [pixel]', fontsize=12)
    
    axes[1].get_legend().remove()

    plt.tight_layout()
    plt.savefig("recovered_vs_simulated.png")
    #plt.show()
    
