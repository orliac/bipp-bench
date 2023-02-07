import os
import sys
import json

def list_directories(path):
    return [ name for name in os.listdir(path) 
             if os.path.isdir(os.path.join(path, name)) and name != '__pycache__' ]

# Benchmark's root directory
bench_root = "/work/ska/orliac/benchmarks/bench04_72"
if not os.path.isdir(bench_root):
    raise Exception("-E- Benchmark directory not found")

# Global dict of dicts 
ifims = {}

for package in list_directories(bench_root):
    ifims[package] = {}
    p1 = os.path.join(bench_root, package)
    for cluster in list_directories(p1):
        ifims[package][cluster] = {}
        p2 = os.path.join(p1, cluster)
        for proc_unit in list_directories(p2):
            ifims[package][cluster][proc_unit] = {}
            p3 = os.path.join(p2, proc_unit)
            for compiler in list_directories(p3):
                ifims[package][cluster][proc_unit][compiler] = {}
                p4 = os.path.join(p3, compiler)
                for precision in list_directories(p4):
                    ifims[package][cluster][proc_unit][compiler][precision] = {}
                    p5 = os.path.join(p4, precision)
                    for algo in list_directories(p5):
                        ifims[package][cluster][proc_unit][compiler][precision][algo] = {}
                        p6 = os.path.join(p5, algo)
                        for nsta in list_directories(p6):
                            ifims[package][cluster][proc_unit][compiler][precision][algo][nsta] = {}
                            p7 = os.path.join(p6, nsta)
                            ifims[nsta] = {}
                            for nlev in list_directories(p7):
                                ifims[package][cluster][proc_unit][compiler][precision][algo][nsta][nlev] = {}
                                p8 = os.path.join(p7, nlev)
                                for pixw in list_directories(p8):
                                    print(f"{package:8s} {cluster:4s} {proc_unit:4s} {compiler:4s} {precision:6s} {algo} {nsta:2s} {nlev:>2s} {pixw:>4s}")
                                    stats_json = os.path.join(p8, pixw, 'stats.json')
                                    if not os.path.isfile(stats_json):
                                        raise Exeception(f"-E- Missing expected json stats file {stats_json}")
                                    with open(stats_json) as stats_fh:
                                        stats = json.load(stats_fh)
                                        ifims[package][cluster][proc_unit][compiler][precision][algo][nsta][nlev][pixw] = float(stats['timings']['ifim'])

print(ifims['pypeline'])
print(ifims['bipp'])
sys.exit(0)
                                    
