import sys
import time
import argparse

parser = argparse.ArgumentParser(description='Run CASA tclean to produce a dirty image')

parser.add_argument('--ms_file',  help='path to MS dataset to process', required=True)
parser.add_argument('--out_name', help='Output name', required=True)
parser.add_argument('--imsize',   help='Image width in pixel', type=int, required=True)
parser.add_argument('--cell',     help='Cell size in asec', type=float, required=True)
parser.add_argument('--spw',      help='Select spectral window/channels', required=False)
parser.add_argument('--observation', help='Observation ID range', required=False)
parser.add_argument('--timerange', help='Range of time to select from data', required=False)
parser.add_argument('--gridder',  help='CASA gridder to use', choices=['wproject'], required=False)
parser.add_argument('--wprojplanes', help="CASA number of W projection planes", type=int, required=False)

args = parser.parse_args()

print("-D- casa_tclean.py args =", args)

# generate standard listobs listing
print(f"-I- Running CASA listobs on {args.ms_file}")
#listobs(vis=args.ms_file, verbose=True, listfile="casa_listobs.out", overwrite=True)
print("\n\n")

print("-D- args.timerange =", args.timerange)

ts = time.time()
tclean(vis=args.ms_file, imagename=args.out_name, imsize=args.imsize, cell=str(args.cell)+'arcsec', niter=0,
       selectdata=True, spw=args.spw, timerange=args.timerange, gridder=args.gridder, wprojplanes=args.wprojplanes)
te = time.time()
print(f"#@# dirty_casa {te - ts:.3f}")

dirty_stats=imstat(imagename=args.out_name + '.image')
print("-I- dirty_stats =\n", dirty_stats)

exportfits(args.out_name + '.image', args.out_name + '.image.fits', overwrite=True)
print(f"-I- CASA dirty image exported to {args.out_name + '.image.fits'}")

#imview('./casa_dirty.image')
