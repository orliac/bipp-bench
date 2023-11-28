import sys
import argparse

def ms_set_weight_to(ms_file, w):
    print("-I- ms_file:", ms_file)

    cols_to_del = ['WEIGHT_SPECTRUM', 'SIGMA_SPECTRUM']
    
    tb.open(ms_file, nomodify=False)
    cols = tb.colnames()
    #print("-D- Before:\n", cols)
    for col in cols_to_del:
        if col in cols:
            print(f"-D- found col to del {col}")
            tb.removecols([col])
        else:
            print(f"-D- column to delete {col} not found in MS")
    tb.close()
    
    #weight_spectrum not supported!
    ms.open(ms_file, nomodify=False)
    rec = ms.getdata(["weight", "sigma"])
    rec['weight'][:,:] = 1
    rec['sigma'][:,:] = 1
    ms.putdata(rec)
    ms.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Edit weights in MS datasets')
    parser.add_argument('--ms_file',   help='MS file to modify', required=True)
    args = parser.parse_args()

    ms_set_weight_to(args.ms_file, 1.0)
