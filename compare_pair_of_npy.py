import sys
import numpy as np
import os

"""
sol1 = "/work/ska/papers/bipp/sim_skalow/benchmarks-4/oskar_9_point_sources_0-50-1-1_single_0/bipp/izar/cpu/gcc/single/nufft/0/1/1024/SKALOW_bipp_nufft_cpu_0_I_lsq_eq_data.npy"
sol2 = "/work/ska/papers/bipp/sim_skalow/benchmarks-4/oskar_9_point_sources_0-50-1-1_single_0/pypeline/izar/none/gcc/single/ss/0/1/1024/SKALOW_pypeline_ss_none_0_I_lsq_eq_data.npy"

data1 = np.load(sol1, allow_pickle=True)
data2 = np.load(sol2, allow_pickle=True)

# Sum over energy levels
data1 = np.sum(data1, axis=0)
data2 = np.sum(data2, axis=0)

diff = data1 - data2
rmse = np.sqrt(np.sum(diff**2)/np.size(diff))
min, max  = np.min(diff), np.max(diff)
max_abs = np.max(np.abs(diff))
print(data1.shape, data2.shape)
print(f"range1 = [{np.min(data1):.7f}, {np.max(data1):.7f}]")
print(f"range2 = [{np.min(data2):.7f}, {np.max(data2):.7f}]")
print(f"diff range = [{min:.7f}, {max:.7f}], rmse = {rmse:.7f}")

sys.exit(0)
"""

def intra_solution_comparison(root):

    path_sol = os.path.join(root)
    if not os.path.exists(path_sol):
        raise Exception(f"Path {path_sol} not found")
    print(f"-I- path_sol: {path_sol}")

    outname = ''
    if args.outname:
        outname = "_" + args.outname
    
    file_pypeline  = os.path.join(path_sol, "pypeline_ss_none" + outname + "_I_lsq_eq_data.npy")
    file_bippSsCpu = os.path.join(path_sol, "bipp_ss_cpu"      + outname + "_I_lsq_eq_data.npy")
    file_bippSsGpu = os.path.join(path_sol, "bipp_ss_gpu"      + outname + "_I_lsq_eq_data.npy")
    file_bippNuCpu = os.path.join(path_sol, "bipp_nufft_cpu"   + outname + "_I_lsq_eq_data.npy")
    file_bippNuGpu = os.path.join(path_sol, "bipp_nufft_gpu"   + outname + "_I_lsq_eq_data.npy")
        
    data_pypeline  =  np.sum(np.load(file_pypeline,  allow_pickle=True), axis=0)
    data_bippSsCpu =  np.sum(np.load(file_bippSsCpu, allow_pickle=True), axis=0)
    data_bippSsGpu =  np.sum(np.load(file_bippSsGpu, allow_pickle=True), axis=0)
    data_bippNuCpu =  np.sum(np.load(file_bippNuCpu, allow_pickle=True), axis=0)
    data_bippNuGpu =  np.sum(np.load(file_bippNuGpu, allow_pickle=True), axis=0)
    
    data = {'pypeline': data_pypeline,
            'bippSsCpu': data_bippSsCpu,
            'bippSsGpu': data_bippSsGpu,
            'bippNuCpu': data_bippNuCpu,
            'bippNuGpu': data_bippNuGpu}
    
    ref = 'pypeline'
    data_ref = data[ref]
    print(f"{ref} = [{np.min(data_ref):10.7f}, {np.max(data_ref):10.7f}], {data_ref.shape}")
            
    for sol in data.keys():
        if sol == ref: continue
        data_sol = data[sol]
        assert(data_sol.shape == data_ref.shape)
        diff = data_sol - data_ref
        rmse = np.sqrt(np.sum(diff**2)/np.size(diff))
        min, max  = np.min(diff), np.max(diff)
        max_abs = np.max(np.abs(diff))
        print(f"{sol} = [{np.min(data_sol):10.7f}, {np.max(data_sol):10.7f}], diff = [{min:10.7f}, {max:10.7f}], RMSE = {rmse:.7f}")
        
        plt.matshow(diff, vmin=-max_abs, vmax=max_abs, cmap=plt.cm.seismic)
        plt.title(f"{sol} minus {ref}")
        plt.colorbar(label=f"Intensity difference [Jy/beam]", shrink=0.8)
        plt_name = f"{sol}_minus_{ref}.png"
        plt_file = os.path.join(root, plt_name)
        plt.savefig(plt_file, bbox_inches='tight')
        #print(f"-I- saved plot {plt_file}")
        plt.close()

    
        
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    help="Directory containing the .npy files", required=True)
    parser.add_argument("--outname", help="Extra naming", required=False, default=None)
    args = parser.parse_args()
    print(args)
    intra_solution_comparison(root=args.root)
    sys.exit(0)
                              
    #root = "/work/ska/orliac/debug/oskar_9s_BENCH/1024/4/gram_issue"
    #root = "/work/ska/orliac/debug/oskar_9s_SS_SP/1024/4"
    #root = "/work/ska/orliac/debug/oskar_9s_SS_SP/256/4"

    if args.precision == 'both':
        intra_solution_comparison(root=root, sol=None, precision='single')
        intra_solution_comparison(root=root, sol=None, precision='double')
    else:
        intra_solution_comparison(root=args.root, sol=None, precision=args.precision)
    
    #for subsol in 'bug', 'fix':
        
    sys.exit(0)

    
    root = "/work/ska/orliac/debug/oskar_9s_BENCH/1024/4/gram_issue"

    for precision in 'double', 'single':
        prec = 'DP'
        if precision == 'single':
            prec = 'SP'
        sol_ = 'pr19'
        ref_ = 'fin'
        sol = '0-1-1-1_'+ sol_ + "_" + precision
        ref = '0-1-1-1_'+ ref_ + "_" + precision

        path_sol = os.path.join(root, sol)
        if not os.path.exists(path_sol):
            raise Exception(f"Path {path_sol} not found")
        path_ref = os.path.join(root, ref)
        if not os.path.exists(path_ref):
            raise Exception(f"Path {path_ref} not found")
        
        file_sol_pypeline  = os.path.join(path_sol, "pypeline_ss_none_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_sol_bippSsCpu = os.path.join(path_sol, "bipp_ss_cpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_sol_bippSsGpu = os.path.join(path_sol, "bipp_ss_gpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_sol_bippNuCpu = os.path.join(path_sol, "bipp_nufft_cpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_sol_bippNuGpu = os.path.join(path_sol, "bipp_nufft_gpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        data_sol_pypeline  = np.sum(np.load(file_sol_pypeline,  allow_pickle=True), axis=0)
        data_sol_bippSsCpu = np.sum(np.load(file_sol_bippSsCpu, allow_pickle=True), axis=0)
        data_sol_bippSsGpu = np.sum(np.load(file_sol_bippSsGpu, allow_pickle=True), axis=0)
        data_sol_bippNuCpu = np.sum(np.load(file_sol_bippNuCpu, allow_pickle=True), axis=0)
        data_sol_bippNuGpu = np.sum(np.load(file_sol_bippNuGpu, allow_pickle=True), axis=0)

        file_ref_pypeline  = os.path.join(path_ref, "pypeline_ss_none_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_ref_bippSsCpu = os.path.join(path_ref, "bipp_ss_cpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_ref_bippSsGpu = os.path.join(path_ref, "bipp_ss_gpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_ref_bippNuCpu = os.path.join(path_ref, "bipp_nufft_cpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        file_ref_bippNuGpu = os.path.join(path_ref, "bipp_nufft_gpu_oskar_bipp_1source_I_lsq_eq_data.npy")
        data_ref_pypeline  = np.sum(np.load(file_ref_pypeline,  allow_pickle=True), axis=0)
        data_ref_bippSsCpu = np.sum(np.load(file_ref_bippSsCpu, allow_pickle=True), axis=0)
        data_ref_bippSsGpu = np.sum(np.load(file_ref_bippSsGpu, allow_pickle=True), axis=0)
        data_ref_bippNuCpu = np.sum(np.load(file_ref_bippNuCpu, allow_pickle=True), axis=0)
        data_ref_bippNuGpu = np.sum(np.load(file_ref_bippNuGpu, allow_pickle=True), axis=0)

        data_sol = {'pypeline':  data_sol_pypeline,
                    'bippSsCpu': data_sol_bippSsCpu,
                    'bippSsGpu': data_sol_bippSsGpu,
                    'bippNuCpu': data_sol_bippNuCpu,
                    'bippNuGpu': data_sol_bippNuGpu}
        data_ref = {'pypeline':  data_ref_pypeline,
                    'bippSsCpu': data_ref_bippSsCpu,
                    'bippSsGpu': data_ref_bippSsGpu,
                    'bippNuCpu': data_ref_bippNuCpu,
                    'bippNuGpu': data_ref_bippNuGpu}
        
        for sol in data_sol.keys():
            sol_data = data_sol[sol]
            ref_data  = data_ref[sol]
            assert(sol_data.shape == ref_data.shape)
            diff = sol_data - ref_data
            rmse = np.sqrt(np.sum(diff**2)/np.size(diff))
            min, max  = np.min(diff), np.max(diff)
            max_abs = np.max(np.abs(diff))
            print(f"{sol:9s} {prec} = [{np.min(sol_data):10.7f}, {np.max(sol_data):10.7f}], diff = [{min:10.7f}, {max:10.7f}], RMSE = {rmse:.7f}")
            plt.matshow(diff, vmin=-max_abs, vmax=max_abs, cmap=plt.cm.seismic)
            plt.title(f"{sol_} minus {ref_} {sol} {precision} precision")
            plt.colorbar(label=f"Intensity difference [Jy/beam]", shrink=0.8)
            plt_name = f"{sol_}_minus_{ref_}_{sol}_{precision}.png"
            plt_file = os.path.join(root, plt_name)
            plt.savefig(plt_file, bbox_inches='tight')
            plt.close()
            print(f"-I- saved plot {plt_file}")
            
    #sys.exit(0)

    
    for sol_ in 'old', 'new', 'pr19', 'fin':
        print(f"@@@@ {sol_}")
        for precision in 'double', 'single':
            prec = 'DP'
            if precision == 'single':
                prec = 'SP'
            sol = '0-1-1-1_' + sol_ + "_" + precision
            path_sol = os.path.join(root, sol)
            if not os.path.exists(path_sol):
                raise Exception(f"Path {path_sol} not found")
            file_pypeline  = os.path.join(path_sol, "pypeline_ss_none_oskar_bipp_1source_I_lsq_eq_data.npy")
            file_bippSsCpu = os.path.join(path_sol, "bipp_ss_cpu_oskar_bipp_1source_I_lsq_eq_data.npy")
            file_bippSsGpu = os.path.join(path_sol, "bipp_ss_gpu_oskar_bipp_1source_I_lsq_eq_data.npy")
            file_bippNuCpu = os.path.join(path_sol, "bipp_nufft_cpu_oskar_bipp_1source_I_lsq_eq_data.npy")
            file_bippNuGpu = os.path.join(path_sol, "bipp_nufft_gpu_oskar_bipp_1source_I_lsq_eq_data.npy")

            data_pypeline  =  np.sum(np.load(file_pypeline,  allow_pickle=True), axis=0)
            data_bippSsCpu =  np.sum(np.load(file_bippSsCpu, allow_pickle=True), axis=0)
            data_bippSsGpu =  np.sum(np.load(file_bippSsGpu, allow_pickle=True), axis=0)
            data_bippNuCpu =  np.sum(np.load(file_bippNuCpu, allow_pickle=True), axis=0)
            data_bippNuGpu =  np.sum(np.load(file_bippNuGpu, allow_pickle=True), axis=0)

            data = {'bippSsCpu': data_bippSsCpu,
                    'bippSsGpu': data_bippSsGpu,
                    'bippNuCpu': data_bippNuCpu,
                    'bippNuGpu': data_bippNuGpu}

            ref = 'pypeline'
            data_ref = data_pypeline
            print(f"{ref}  {prec} = [{np.min(data_ref):10.7f}, {np.max(data_ref):10.7f}], {data_ref.shape}")
            
            for sol in data.keys():
                data_sol = data[sol]
                assert(data_sol.shape == data_ref.shape)
                diff = data_sol - data_ref
                rmse = np.sqrt(np.sum(diff**2)/np.size(diff))
                min, max  = np.min(diff), np.max(diff)
                max_abs = np.max(np.abs(diff))
                print(f"{sol} {prec} = [{np.min(data_sol):10.7f}, {np.max(data_sol):10.7f}], diff = [{min:10.7f}, {max:10.7f}], RMSE = {rmse:.7f}")

                plt.matshow(diff, vmin=-max_abs, vmax=max_abs, cmap=plt.cm.seismic)
                plt.title(f"BIPP {sol_}: {sol} minus {ref} {sol} {precision} precision")
                plt.colorbar(label=f"Intensity difference [Jy/beam]", shrink=0.8)
                plt_name = f"{sol_}_{sol}_minus_{ref}_{precision}.png"
                plt_file = os.path.join(root, plt_name)
                plt.savefig(plt_file, bbox_inches='tight')
                print(f"-I- saved plot {plt_file}")
                plt.close()
