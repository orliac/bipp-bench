import sys
import os
import argparse
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import plots

def check_args(args_in):
    #print("-I- command line arguments =", args_in)
    parser = argparse.ArgumentParser(args_in)
    parser.add_argument("--bench_root", help="Root directory of benchmark to analyis", required=True)
    parser.add_argument("--subpath_sol1", help="First solution to compare sub path", required=False, default=None)
    parser.add_argument("--subpath_sol2", help="Second solution to compare sub path", required=False, default=None)
    parser.add_argument("--sol1", help="First solution to compare", required=True)
    parser.add_argument("--sol2", help="Second solution to compare", required=True)
    parser.add_argument("--ms_file",   help="MS file", required=True)
    parser.add_argument("--outdir",   help="Directory for output file", required=True)
    parser.add_argument("--outname",   help="Basename for output file", required=True)

    args = parser.parse_args()
    if not os.path.isdir(args.bench_root):
        raise Exception("-E- Benchmark directory not found")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    return args


# cp basefits to fits, and replace data in fits1 from npy
def hack_fits(base_fits, dest_fits, npy):

    shutil.copyfile(base_fits, dest_fits)
    print("-I- dest_fits =", dest_fits)
    hdulist = fits.open(dest_fits)
    hdulist[0].data = np.sum(np.load(npy).transpose(1,2,0), axis=2)
    hdulist[0].data = np.fliplr(hdulist[0].data)
    hdulist.writeto(dest_fits, overwrite=True)
    hdulist.close()

    
def plot(bench_root, subpath_sol1, subpath_sol2, sol1, sol2):
    print("-I- bench_root =", bench_root)
    print("-I- sol1 =", subpath_sol1, sol1)
    print("-I- sol2 =", subpath_sol2, sol2)
    p1 = p2 = bench_root
    if subpath_sol1:
        p1 = os.path.join(bench_root, subpath_sol1)
    if subpath_sol2:
        p2 = os.path.join(bench_root, subpath_sol2)
    for p in p1, p2:
        if not os.path.exists(p):
            raise Exception(f"p {p} not found")
    
    mins = maxs = []

    ress = ('256', '512', '1024', '2048')
    nres = len(ress)
    data = {}; wcs  = {}; sky_files = {}
    
    for res in ress:
        pp1 = os.path.join(p1, res)
        pp2 = os.path.join(p2, res)
        pp1_ = pp1 + '/*I_lsq_eq_data.npy'
        pp2_ = pp2 + '/*I_lsq_eq_data.npy'
        sky  = pp1 + '/*.sky'
        npys1 = glob.glob(pp1_)
        npys2 = glob.glob(pp2_)
        skys  = glob.glob(sky)
        if len(npys1) != 1:
            raise Exception("-E- Should be exactly one, but is", len(npys1))
        if len(npys2) != 1:
            raise Exception("-E- Should be exactly one, but is", len(npys2))
        if len(skys) != 1:
            raise Exception("-E- Should be exactly one, but is", len(skys))
        npy1 = npys1[0]
        npy2 = npys2[0]
        sky  = skys[0]
        print("-I-", npy1)
        print("-I-", npy2)
        print("-I-", sky)
        wsc_fits = os.path.join(pp1, 'dirty_wsclean-image.fits')
        if not os.path.exists(wsc_fits):
            raise Exception(f"-E- WSClean dirty images fits file {wsc_fits} not found")
        print("-I-", wsc_fits)

        npy1_fits = '/tmp/npy1.fits'
        npy2_fits = '/tmp/npy2.fits'
        
        hack_fits(wsc_fits, npy1_fits, npy1)
        hack_fits(wsc_fits, npy2_fits, npy2)

        bb1_hdu = fits.open(npy1_fits)[0]
        bb2_hdu = fits.open(npy2_fits)[0]
        for hdu in bb1_hdu, bb2_hdu:
            print(hdu.data.shape)
            if hdu.data.shape[0] == 1 and hdu.data.shape[1] == 1:
                hdu.data = hdu.data.reshape(hdu.data.shape[2:4])
            print(hdu.data.shape)

        bb1_wcs = WCS(bb1_hdu.header)
        bb2_wcs = WCS(bb2_hdu.header)

        wcs_ = bb1_wcs;  #print("wcs =", wcs_)
        wcs[res] = wcs_
        
        bb1_data = bb1_hdu.data
        bb2_data = bb2_hdu.data
        diff     = bb1_data - bb2_data
        data[res] = diff
        #print(diff)
        sky_files[res] = sky
        mins.append(np.min(diff))
        maxs.append(np.max(diff))

    #print(data)
        
    glob_min = np.min(mins)
    glob_max = np.max(maxs)
    minmax = np.max(np.abs([glob_min, glob_max]))
    #print(glob_min, glob_max, "=>", minmax)

    fig = plt.figure(figsize=(20,18))

    plt.rc('axes',  titlesize=20) 
    plt.rc('axes',  labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    
    plot_grid    = True
    plot_circles = False

    i = 0
    for res in ress:
        pos = 220 + i + 1
        plt.subplot(pos, projection=wcs[res], slices=('x', 'y', 0, 0))
        casa_sky = plots.my_subplot(plt, wcs, data[res], -minmax, minmax,
                                    plot_grid, plot_circles, int(res), 'seismic',
                                    None, None, False, int(res), 4, solution=res)
        plt.title(f"{res} x {res}", pad=14)
        #plt.gca().coords[0].set_axislabel(" ")
        plt.gca().coords[0].set_ticklabel_visible(True)
        plt.gca().coords[1].set_ticklabel_visible(True)
        plt.gca().coords[1].set_ticklabel_position('lr')
        if i%2 == 0:
            plt.gca().coords[1].set_axislabel("Declination")
        else:
            plt.gca().coords[1].set_axislabel(" ")
        if i >= 2:
            plt.gca().coords[0].set_axislabel("Right ascension")
        else:
            plt.gca().coords[0].set_axislabel(" ")
        i += 1
        
    cb_ax = fig.add_axes([0.94, 0.35, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)
    cbar.set_label("Intensity difference [Jy/beam]", rotation=-90, va='bottom')

        
    basename = 'bluebild_differential_plots'
    for dpi in 200,:
        for fmt in '.png', '.pdf':
            plot_file = basename + '_dpi_'+ str(dpi) + fmt
            plt.savefig(plot_file, bbox_inches='tight', dpi=dpi)
            print("-I-", plot_file)


def plot_test(bench_root, subpath_sol1, subpath_sol2, sol1, sol2, ms_file):
    print("-I- bench_root =", bench_root)
    print("-I- sol1 =", sol1)
    print("-I- sol2 =", sol2)
    
    mins = maxs = []
    data = {}; wcs  = {}; sky_files = {}
    
    p1_ = os.path.join(bench_root, subpath_sol1, sol1 + '*I_lsq_eq_data.npy'); print(f"ls -rtl {p1_}")
    p2_ = os.path.join(bench_root, subpath_sol2, sol2 + '*I_lsq_eq_data.npy')
    sky = os.path.join(bench_root, sol1 + '*.sky'); #print(sky)
    npys1 = glob.glob(p1_)
    npys2 = glob.glob(p2_)
    skys  = glob.glob(sky)
    if len(npys1) != 1:
        raise Exception(f"-E- npys1 should be exactly one, but is {len(npys1)}")
    if len(npys2) != 1:
        raise Exception(f"-E- npys2 should be exactly one, but is {len(npys2)}")
    #if len(skys) != 1:
    #    raise Exception(f"-E- skys should be exactly one, but is {len(skys)}")
    npy1 = npys1[0]
    npy2 = npys2[0]
    #sky  = skys[0]
    print("-I-", npy1)
    print("-I-", npy2)
    #print("-I-", sky)
    wsc_fits = os.path.join(bench_root, subpath_sol1, ms_file + '-dirty_wsclean-image.fits')
    if not os.path.exists(wsc_fits):
        raise Exception(f"-E- WSClean dirty images fits file {wsc_fits} not found")
    print("-I-", wsc_fits)

    npy1_fits = '/tmp/npy1.fits'
    npy2_fits = '/tmp/npy2.fits'
        
    hack_fits(wsc_fits, npy1_fits, npy1)
    hack_fits(wsc_fits, npy2_fits, npy2)

    bb1_hdu = fits.open(npy1_fits)[0]
    bb2_hdu = fits.open(npy2_fits)[0]
    for hdu in bb1_hdu, bb2_hdu:
        print(hdu.data.shape)
        if hdu.data.shape[0] == 1 and hdu.data.shape[1] == 1:
            hdu.data = hdu.data.reshape(hdu.data.shape[2:4])
        print(hdu.data.shape)

    bb1_wcs = WCS(bb1_hdu.header)
    bb2_wcs = WCS(bb2_hdu.header)

    wcs = bb1_wcs;  #print("wcs =", wcs_)
        
    bb1_data = bb1_hdu.data
    bb2_data = bb2_hdu.data
    diff     = bb1_data - bb2_data
    mins.append(np.min(diff))
    maxs.append(np.max(diff))
    
    #print(data)
        
    glob_min = np.min(mins)
    glob_max = np.max(maxs)
    minmax = np.max(np.abs([glob_min, glob_max]))
    #print(glob_min, glob_max, "=>", minmax)

    fig = plt.figure(figsize=(20,18))

    plt.rc('axes',  titlesize=20) 
    plt.rc('axes',  labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    
    plot_grid    = True
    plot_circles = False

    res = '1024'
    pos = 111
    
    plt.subplot(pos, projection=wcs, slices=('x', 'y', 0, 0))
    casa_sky = plots.my_subplot(plt, wcs, diff, -minmax, minmax,
                                plot_grid, plot_circles, int(res), 'seismic',
                                None, None, False, int(res), 4, solution=res)
    plt.title(f"{res} x {res}", pad=14)
    #plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(True)
    plt.gca().coords[1].set_ticklabel_visible(True)
    plt.gca().coords[1].set_ticklabel_position('lr')
    plt.gca().coords[1].set_axislabel("Declination")
    plt.gca().coords[0].set_axislabel("Right ascension")
        
    cb_ax = fig.add_axes([0.94, 0.35, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)
    cbar.set_label("Intensity difference [Jy/beam]", rotation=-90, va='bottom')

        
    basename = 'bluebild_differential_plots'
    
    for dpi in 200, 400:
        for fmt in '.png', '.pdf':
            plot_file = args.outname + '_dpi_'+ str(dpi) + fmt
            plot_file = os.path.join(args.outdir, plot_file)
            plt.savefig(plot_file, bbox_inches='tight', dpi=dpi)
            print("-I-", plot_file)
        
if __name__ == "__main__":
    
    args = check_args(sys.argv)
    print(args)

    MARKER_SIZE = 8
    LINE_WIDTH  = 0.5

    """
    plot(bench_root = args.bench_root,
         subpath_sol1 = args.subpath_sol1,
         subpath_sol2 = args.subpath_sol2,
         sol1 = args.sol1,
         sol2 = args.sol2)
    """
    plot_test(bench_root = args.bench_root,
              subpath_sol1 = args.subpath_sol1,
              subpath_sol2 = args.subpath_sol2,
              sol1       = args.sol1,
              sol2       = args.sol2,
              ms_file    = args.ms_file)
