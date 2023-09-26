import sys
import argparse

def ms_set_weight_to(ms_file, w):
    print("-I- ms_file:", ms_file)

    #weight_spectrum not supported!
    ms.open(ms_file, nomodify=False)
    rec = ms.getdata(["weight", "sigma"])
    rec['weight'][:,:] = 1
    rec['sigma'][:,:] = 1
    ms.putdata(rec)
    ms.close()
    
    tb.open(ms_file, nomodify=False)
    print("-D- Before:\n", tb.colnames())
    tb.removecols(['WEIGHT_SPECTRUM'])
    tb.removecols(['SIGMA_SPECTRUM'])
    print("-D- After:\n", tb.colnames())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Edit weights in MS datasets')
    parser.add_argument('--ms_file',   help='MS file to modify', required=True)
    args = parser.parse_args()

    ms_set_weight_to(args.ms_file, 1.0)
