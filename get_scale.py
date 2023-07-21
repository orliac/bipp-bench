## Do not print anything else than the result in arcsec as this is to be read
## by Bash!!
## ----------------------------------------------------------------------------
import numpy as np
import argparse
from astropy import units as u
from astropy.coordinates import Angle

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pixw", type=int,   required=True, help="Square image width in pixels")
    p.add_argument("--fov",  type=float, required=True, help="Field of view in --fov_unit")
    p.add_argument("--unit", type=str,   required=True, help="Unit of field of view", choices=['deg', 'rad'])
    args = p.parse_args()
    return args

def main():
    args = get_args()
    #print("-D- get_scale.py args =", args)
    if args.unit == 'deg':
        angle = Angle(args.fov, u.degree)
    elif args.unit == 'radian':
        angle = Angle(args.fov, u.radian)

    # Pixel resolution in arcsec
    res = f"{angle.arcsec / args.pixw:.6f}"
    print(res)

if __name__ == "__main__":
    main()
