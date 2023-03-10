import sys
import time
import argparse

parser = argparse.ArgumentParser(description='Run CASA tclean to produce a dirty image')

parser.add_argument('--ms_file',  help='path to MS dataset to process', required=True)
parser.add_argument('--out_name', help='Output name', required=True)
parser.add_argument('--imsize',   help='Image width in pixel', type=int, required=True)
parser.add_argument('--cell',     help='Cell size in asec', type=int, required=True)
parser.add_argument('--spw',      help='Select spectral window/channels', required=False)
args = parser.parse_args()

ts = time.time()
tclean(vis=args.ms_file, imagename=args.out_name, imsize=args.imsize, cell=args.cell, niter=0, selectdata=True, spw=args.spw)
te = time.time()
print(f"#@# dirty_casa {te - ts:.3f}")

dirty_stats=imstat(imagename=args.out_name + '.image')
print("-I- dirty_stats =\n", dirty_stats)

exportfits(args.out_name + '.image', args.out_name + '.image.fits', overwrite=True)
print(f"-I- CASA dirty image exported to {args.out_name + '.image.fits'}")

#imview('./casa_dirty.image')
