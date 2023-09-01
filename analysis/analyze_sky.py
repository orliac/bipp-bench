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

    
    dfs = []
    flats = []
    flats_keys = []
    for sol in plot:
        flat = {}
        for k in plot[sol]:
            flat[k] = pd.Series(plot[sol][k]['loss'])
        df = pd.DataFrame(flat)
        print("flat =", flat)
        print("df =", df)
        flats.append(df)
        flats_keys.append(sol)
    print(flats)
    print(flats_keys)
    df = pd.concat(flats, keys=flats_keys, axis=1, names=['a','b'])#.stack(0)
    df = pd.melt(df)#df.reset_index(level=1, names=['a','b'])
    print(df)
    print(df.index.names)
    
    #df = pd.melt(df, var_name = 'x', value_name = 'y') 
    #sns.swarmplot(data=df, x='x', y='y')
    sns.swarmplot(data=df, x='b', y='value', hue='a')

    plt.gca().set(xlabel='Distance from center of field of view [pixel]',
                  ylabel='Recovered intensity loss [%]')
    
    plt.show()
    
