import sys
import os
import re
import json
import warnings
import numpy

# Specs
def spec_list():
    return ['package', 'cluster', 'proc_unit', 'compiler', 'precision', 'algo', 'nsta', 'nlev', 'pixw']

def get_spec_dictionary_from_list(specs):
    ref_spec_list = spec_list()
    if len(specs) != len(ref_spec_list):
        raise Exception("Passed list is not of the same length than reference list")
    specs_dic = {}
    for i in range(0, len(specs)):
        #print(i, ref_spec_list[i])
        specs_dic[ref_spec_list[i]] = specs[i]
    return specs_dic

# List directories in given path, with some exclusions
#
def list_directories(path):
    return [ name for name in os.listdir(path) 
             if os.path.isdir(os.path.join(path, name)) and name != '__pycache__' and name[0] != '_' ]


# Compute the RMSE between two images
#
def stats_image_diff(image1, image2):
    assert image1.shape == image2.shape, \
        f"-E- shapes of images to compare do not match {image1.data.shape} vs {image2.data.shape}"
    #print("-I- comparing images with shape ", image1.shape)
    diff = image2 - image1
    rmse = numpy.sqrt(numpy.sum(diff**2)/numpy.size(diff))
    max_abs = numpy.max(numpy.abs(diff))
    return rmse, max_abs


# Compute the RMSE between two images with random number of energy
# levels (axis 0). Lengths of axis 1 and 2 must match.
# 
def stats_image_diff_compressing_levels(image1, image2):
    assert len(image1.shape) == 3
    assert len(image2.shape) == 3
    assert image1.shape[1:3] == image2.shape[1:3], \
        f"-E- shapes of images to compare do not match on 2nd and 3rd dimensions {image1.data.shape[1:3]} vs {image2.data.shape[1:3]}"
    img1 = numpy.sum(image1, axis=0)
    img2 = numpy.sum(image2, axis=0)
    return stats_image_diff(img1, img2)


def get_solution_nickname_from_path(bench_root, path_sol):

    specs = get_solution_specs_from_path(path_sol, bench_root)

    if specs['package'] == 'bipp':
        nick_long  = 'Bipp'
        nick_short = 'b' #'B'
    elif specs['package'] == 'pypeline':
        nick_long  = 'Bluebild'
        nick_short = 'p' #'Bb'
    else:
        raise Exception(f"Unknown package {specs['package']}")

    if specs['proc_unit'] == 'none':
        nick_long  += 'Cpu'
        nick_short += 'c' #'C'
    elif specs['proc_unit'] == 'cpu':
        nick_long  += 'Cpu'
        nick_short += 'c' #'C'
    elif specs['proc_unit'] == 'gpu':
        nick_long  += 'Gpu'
        nick_short += 'g' #'G'
    else:
        raise Exception(f"Unknown proc_unit {specs['proc_unit']}")

    if specs['algo'] == 'ss':
        nick_long  += 'Ss'
        nick_short += 's' #'S'
    elif specs['algo'] == 'nufft':
        nick_long  += 'Nufft'
        nick_short += 'n' #'N'
    else:
        raise Exception(f"Unknown algo {specs['algo']}")
    
    return nick_long, nick_short


def get_list_of_solutions(bench_root, nsta, nlev, pixw):
    paths = []
    for package_ in list_directories(bench_root):                                               #package
        p1 = os.path.join(bench_root, package_)
        for cluster_ in list_directories(p1):                                                   #cluster
            p2 = os.path.join(p1, cluster_)
            for proc_unit_ in list_directories(p2):                                             #processing unit
                p3 = os.path.join(p2, proc_unit_)
                for compiler_ in list_directories(p3):                                          #compiler
                    p4 = os.path.join(p3, compiler_)
                    for precision_ in list_directories(p4):                                     #precision
                        p5 = os.path.join(p4, precision_)
                        for algo_ in list_directories(p5):                                      #algo
                            p6 = os.path.join(p5, algo_)
                            for nsta_ in list_directories(p6):                                  #nsta
                                p7 = os.path.join(p6, nsta_)
                                for nlev_ in list_directories(p7):                              #nlev
                                    p8 = os.path.join(p7, nlev_)
                                    for pixw_ in list_directories(p8):                          #pixw
                                        if nsta == int(nsta_) and nlev == int(nlev_) and pixw == int(pixw_):
                                            path_sol = os.path.join(bench_root, package_, cluster_, proc_unit_, compiler_, precision_, algo_, nsta_, nlev_, pixw_)
                                            #print(" ...", path_sol)
                                            paths.append(path_sol)
    return paths


def get_solution_specs_from_path(path_sol, bench_root):
    rel_path = os.path.relpath(path_sol, bench_root)
    specs = re.split('/', rel_path)
    if len(specs) != 9:
        raise Exception("Exactly 9 components are expected")
    if specs[0] != 'bipp' and specs[0] != 'pypeline':
        raise Exception(f"First spec is expected to be eiter bipp or pypeline, got {specs[0]}")
    
    return get_spec_dictionary_from_list(specs)

def read_json_file(file):
    if os.path.isfile(file):
        fp = open(file)
        return json.load(fp)
    else:
        warnings.warn(f"file {file} not found! CHECK SOLUTION!!")
        return None

def read_bipp_json_stat_file(fp):
    bipp_json = os.path.join(fp)
    return read_json_file(fp)

def read_casa_json_stat_file(path_sol):
    casa_json = os.path.join(path_sol, 'casa.json')
    return read_json_file(casa_json)

def read_dirty_casa_json_stat_file(path_sol):
    casa_json = os.path.join(path_sol, 'dirty_casa.json')
    return read_json_file(casa_json)

def read_wsclean_json_stat_file(path_sol):
    wsclean_json = os.path.join(path_sol, 'wsclean.json')
    return read_json_file(wsclean_json)

def read_dirty_wsclean_json_stat_file(path_sol):
    wsclean_json = os.path.join(path_sol, 'dirty_wsclean.json')
    return read_json_file(wsclean_json)


def get_benchmark_hierarchy(bench_root):

    tree = {'packages':  [], 'clusters':   [], 'proc_units': [],
            'compilers': [], 'precisions': [], 'algos':      [],
            'nstas':     [], 'nlevs':      [], 'pixws':      []}

    for package in list_directories(bench_root):                                               #package
        if package not in tree['packages']: tree['packages'].append(package)

        p1 = os.path.join(bench_root, package)
        for cluster in list_directories(p1):                                                   #cluster
            if cluster not in tree['clusters']: tree['clusters'].append(cluster)

            p2 = os.path.join(p1, cluster)
            for proc_unit in list_directories(p2):                                             #processing unit
                if proc_unit not in tree['proc_units']: tree['proc_units'].append(proc_unit)

                p3 = os.path.join(p2, proc_unit)
                for compiler in list_directories(p3):                                          #compiler
                    if compiler not in tree['compilers']: tree['compilers'].append(compiler)

                    p4 = os.path.join(p3, compiler)
                    for precision in list_directories(p4):                                     #precision
                        if precision not in tree['precisions']: tree['precisions'].append(precision)

                        p5 = os.path.join(p4, precision)
                        for algo in list_directories(p5):                                      #algo
                            if algo not in tree['algos']: tree['algos'].append(algo)

                            p6 = os.path.join(p5, algo)
                            for nsta in list_directories(p6):                                  #nsta
                                if nsta.isdigit() and int(nsta) not in tree['nstas']:
                                    tree['nstas'].append(int(nsta))

                                p7 = os.path.join(p6, nsta)
                                for nlev in list_directories(p7):                              #nlev
                                    if nlev.isdigit() and int(nlev) not in tree['nlevs']:
                                        tree['nlevs'].append(int(nlev))

                                    p8 = os.path.join(p7, nlev)
                                    for pixw in list_directories(p8):                          #pixw
                                        if int(pixw) not in tree['pixws']: tree['pixws'].append(int(pixw))

    return tree
