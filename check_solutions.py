import os
import sys
import json
import matplotlib.pyplot as plt

def list_directories(path):
    return [ name for name in os.listdir(path) 
             if os.path.isdir(os.path.join(path, name)) and name != '__pycache__' ]

# Benchmark's root directory
bench_root = "/work/ska/orliac/benchmarks/bench04_72"
if not os.path.isdir(bench_root):
    raise Exception("-E- Benchmark directory not found")

# Global dict of dicts 
ifims = {}

for package in list_directories(bench_root):                                               #package
    ifims[package] = {}
    p1 = os.path.join(bench_root, package)
    for cluster in list_directories(p1):                                                   #cluster
        ifims[package][cluster] = {}
        p2 = os.path.join(p1, cluster)
        for proc_unit in list_directories(p2):                                             #processing unit
            ifims[package][cluster][proc_unit] = {}
            p3 = os.path.join(p2, proc_unit)
            for compiler in list_directories(p3):                                          #compiler
                ifims[package][cluster][proc_unit][compiler] = {}
                p4 = os.path.join(p3, compiler)
                for precision in list_directories(p4):                                     #precision
                    ifims[package][cluster][proc_unit][compiler][precision] = {}
                    p5 = os.path.join(p4, precision)
                    for algo in list_directories(p5):                                      #algo
                        ifims[package][cluster][proc_unit][compiler][precision][algo] = {} 
                        p6 = os.path.join(p5, algo)
                        for nsta in list_directories(p6):                                  #nsta
                            ifims[package][cluster][proc_unit][compiler][precision][algo][int(nsta)] = {}
                            p7 = os.path.join(p6, nsta)
                            for nlev in list_directories(p7):                              #nlev
                                ifims[package][cluster][proc_unit][compiler][precision][algo][int(nsta)][int(nlev)] = {}
                                p8 = os.path.join(p7, nlev)
                                for pixw in list_directories(p8):                          #pixw
                                    #print(f"{package:8s} {cluster:4s} {proc_unit:4s} {compiler:4s} {precision:6s} {algo} {nsta:2s} {nlev:>2s} {pixw:>4s}")
                                    stats_json = os.path.join(p8, pixw, 'stats.json')
                                    if not os.path.isfile(stats_json):
                                        raise Exeception(f"-E- Missing expected json stats file {stats_json}")
                                    with open(stats_json) as stats_fh:
                                        stats = json.load(stats_fh)
                                        ifims[package][cluster][proc_unit][compiler][precision][algo][int(nsta)][int(nlev)][int(pixw)] = float(stats['timings']['ifim'])

#print(ifims['pypeline'])
#print(ifims['bipp'])

colors = {}
colors['pypeline'] = {'none': 'black', 'cpu': 'indianred', 'gpu': 'palegreen'}
colors['bipp']     = {                 'cpu': 'red', 'gpu': 'lawngreen'}
markers = {'pypeline': '-o', 'bipp': '-*'}

# Make plots
for cluster in 'izar',:
    for nsta in 15,30,60:
        for nlev in 1,2,4,16,32:
            if nlev > nsta:
                continue
            plt_name = f"pypeline_izar_nsta{nsta}_nlev{nlev}.png"
            print(f"-I- Plotting nsta={nsta} nlev={nlev}: {plt_name}")
            
            fig, ax = plt.subplots()
            
            ax.set(xlabel='Image width & height [pixel]',
                   ylabel='Time to solution [s]',
                   title=f"cluster {cluster}, {precision} precision, nsta = {nsta}, nlev = {nlev}")
            sols =[('pypeline', 'none', 'gcc'),
                   ('pypeline', 'cpu', 'gcc'), ('pypeline', 'gpu', 'cuda'),
                   ('bipp', 'cpu', 'gcc'), ('bipp', 'gpu', 'cuda')]

            for package, proc_unit, compiler in sols:
                ifims_sorted = dict(sorted(ifims[package][cluster][proc_unit][compiler]['double']['ss'][nsta][nlev].items()))
                x, y = ifims_sorted.keys(), ifims_sorted.values()
                print(x, "\n", y)
                ax.plot(x, y, markers[package], color=colors[package][proc_unit], label=f"{package} {proc_unit}")
                
            ax.set_yscale('log', base=10)
            ax.set_xscale('log', base=2)
            ax.legend()
            fig.savefig(plt_name)
            plt.show()
        
