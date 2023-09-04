## Do not print anything else than the result in arcsec as this is to be read
## by Bash!!
## ----------------------------------------------------------------------------
import numpy as np
import argparse
from astropy import units as u
from astropy.coordinates import Angle

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",  type=int,   required=True, help="Equivalent of WSC size parameter [pixel]")
    p.add_argument("--scale", type=float, required=True, help="Equivalent of WSC scale parameter [arcsec]")
    args = p.parse_args()
    return args

def main():
    args = get_args()
    res = args.size * args.scale / 3600.0
    print(f"{res:.6f}")

if __name__ == "__main__":
    main()
